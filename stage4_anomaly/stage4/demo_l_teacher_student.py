"""★Demo L（本阶段通关模块）：50 行核心的手写 Teacher-Student。
这个文件是阶段 4 的毕业作品：从零手写一个完整的 Teacher-Student 异常检测器——造数据、训练 student 抄 teacher、算距离图、定标尺、评 AUC——做完这个，无监督线你就通了。


提纲原文要求：teacher 用预训练网络中间层冻结，student 同构小网络，
在数据上训练，画差值热图，用分位数归一化 + AUC 评估。

对应主仓库 unsupervised_training 的完整缩影：
  教材纯好件 -> student 抄 teacher(归一化后) -> 距离图 -> 分位标尺
  -> TopK100 分数 -> AUC。真实 MVTec 版见 README 作业（--data 开关）。

合成数据：条纹纹理背景（"正常"），随机位置贴一块乱纹（"异常"）——
teacher 的固定卷积核能同时看清两种纹理，student 只在好件上学会条纹，
遇到乱纹块抄不像 -> 距离暴涨。

运行：python -m stage4.demo_l_teacher_student [--steps 200] [--save 热图.png]
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .calibration_and_auc import map_quantiles, roc_auc, topk_mean


# ---------------------------------------------------------------------------
# 1. 数据：好件 = 条纹纹理；异常 = 好件 + 一块乱纹贴纸
# ---------------------------------------------------------------------------

def make_ok(n, size=64, seed=0):
    """好件 = 条纹纹理；相位从**固定小集合**里取（保证"正常"可学）。

    教学要点：如果每张好件相位都随机，student 永远学不完"正常"，
    好件自己也会被当成异常——这就是真实工程里"正常必须有稳定模式"。
    这里固定 2 个相位（i%2），保留多样性又不至于学不完。
    """
    rng = np.random.default_rng(seed)
    phases = [0.0, np.pi]                               # 固定相位集合
    imgs = np.zeros((n, size, size, 3), np.float32)
    for i in range(n):
        stripe = np.sin(np.arange(size) / 3.0 + phases[i % 2])[None, :, None]
        imgs[i] = 0.5 + 0.25 * np.tile(stripe, (size, 1, 3))
        imgs[i] += rng.normal(0, 0.03, imgs[i].shape)
    return torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()


def make_ng(ok_batch, seed=1):
    """在好件上贴 [12,20] 的乱纹块，返回(图, 异常掩码)。"""
    rng = np.random.default_rng(seed)
    imgs = ok_batch.clone(); masks = torch.zeros(len(imgs), 1,
                                                 *imgs.shape[2:])
    for i in range(len(imgs)):
        y = int(rng.integers(8, imgs.shape[2] - 28))
        x = int(rng.integers(8, imgs.shape[3] - 28))
        patch = torch.from_numpy(rng.uniform(0.1, 0.9,
                            (3, 20, 12)).astype(np.float32))
        imgs[i, :, y:y + 20, x:x + 12] = patch
        masks[i, :, y:y + 20, x:x + 12] = 1
    return imgs, masks


# ---------------------------------------------------------------------------
# 2. 双胞胎：teacher = 固定卷积"预训练特征器"；student 同构可学
# ---------------------------------------------------------------------------

class FeatureNet(nn.Module):
    """
    双胞胎共用特征网络：两层卷积（teacher 冻结/student 可学）。
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)

    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))))


def build_pair(seed=0):
    """
    构建 T-S 对：teacher 冻结、student 同构可学、conv1 继承。
    """
    torch.manual_seed(seed)
    teacher = FeatureNet()
    for p in teacher.parameters():
        p.requires_grad_(False)                       # 冻结：真老师
    teacher.eval()
    student = FeatureNet()
    with torch.no_grad():                             # conv1 继承(主仓库同款)
        student.conv1.load_state_dict(teacher.conv1.state_dict())
    return teacher, student


# ---------------------------------------------------------------------------
# 3. 训练（只喂好件）+ 评估（距离图 -> 分位标尺 -> TopK 分数 -> AUC）
# ---------------------------------------------------------------------------

def train_student(teacher, student, ok, steps=300, lr=2e-3, seed=0):
    """
    只在好件上训练 student（抄 teacher 归一化后的答卷）。
    """
    torch.manual_seed(seed)
    opt = torch.optim.AdamW([p for p in student.parameters()
                             if p.requires_grad], lr=lr)
    with torch.no_grad():
        t_all = teacher(ok)
        mean = t_all.mean(dim=(0, 2, 3), keepdim=True)         # teacher 归一化
        std = t_all.flatten(2).std(dim=2).mean(0, keepdim=True)[..., None, None]
        # 防除零下限：随机 teacher 对平滑图常有"死通道"(全零/全常数,
        # std=0 -> 归一化出 NaN)——主仓库处处 eps 钳制的同一课
        std = std.clamp_min(1e-3)
        t_norm_all = (t_all - mean) / std
    for it in range(steps):
        i = it % len(ok)
        s = student(ok[i:i + 1])
        loss = F.mse_loss(s, t_norm_all[i:i + 1])              # 抄作业
        opt.zero_grad(); loss.backward(); opt.step()
    return mean, std


@torch.no_grad()
def anomaly_map(teacher, student, imgs, mean, std):
    """
    算一批图的异常图：teacher(归一化) vs student 的逐像素距离²。
    """
    t = (teacher(imgs) - mean) / std
    s = student(imgs)
    return ((t - s) ** 2).mean(1, keepdim=True)                # [N,1,h,w]


def evaluate(teacher, student, ok_test, ng_imgs, mean, std):
    """
    完整评估：异常图→分位标尺→TopK100 打分→AUC。
    """
    amap_ok = anomaly_map(teacher, student, ok_test, mean, std)
    amap_ng = anomaly_map(teacher, student, ng_imgs, mean, std)
    qs, qe = map_quantiles(amap_ok)                            # val 池定标尺
    scores = np.array([topk_mean(m[0].numpy(), k=100) for m in amap_ok]
                      + [topk_mean(m[0].numpy(), k=100) for m in amap_ng])
    labels = np.concatenate([np.zeros(len(amap_ok)), np.ones(len(amap_ng))])
    return roc_auc(labels, scores), (amap_ok, amap_ng, qs, qe)


def run(steps=300, seed=0, save=None):
    """
    Demo L 端到端入口：造数据→训练→评估→(可选)存热图。
    """
    torch.manual_seed(seed)
    ok_train = make_ok(32, seed=0)                             # 教材：纯好件
    ok_test = make_ok(16, seed=7)                              # 参考：定标尺
    ng_batch, _ = make_ng(make_ok(16, seed=9), seed=3)         # 考卷：NG

    teacher, student = build_pair()
    mean, std = train_student(teacher, student, ok_train, steps=steps)
    auc, (amap_ok, amap_ng, qs, qe) = evaluate(
        teacher, student, ok_test, ng_batch, mean, std)
    print(f"AUC = {auc:.3f} | 标尺 q_start={float(qs):.5f} "
          f"q_end={float(qe):.5f}")
    print(f"好件平均异常分 {amap_ok.mean():.5f} << NG 平均 {amap_ng.mean():.5f}"
          " -> '抄不像的地方就是异常'")
    if save:
        import cv2
        h = (amap_ng[0, 0].numpy() * 255 / max(amap_ng.max().item(), 1e-9)
             ).clip(0, 255).astype(np.uint8)
        cv2.applyColorMap(h, cv2.COLORMAP_JET)
        cv2.imwrite(save, cv2.applyColorMap(h, cv2.COLORMAP_JET))
        print("热图已保存:", save)
    return auc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    run(steps=args.steps, save=args.save)
