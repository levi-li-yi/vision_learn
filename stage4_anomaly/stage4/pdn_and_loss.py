"""课时 2+3：PDN 双胞胎构建 + 分位数困难挖掘损失——modeling/trainer 复刻。
这个文件教你两件事：①怎么构建'双胞胎'网络(teacher 冻结、student 可学、conv1 继承)；②'只罚最难的 0.1% 像素'这种损失为什么比'全平均'更有效。


对应主仓库：
  - core/modeling.py build_teacher_student_models：teacher 冻结
    (requires_grad=False)、student 同构、conv1 权重从 teacher 整体继承；
  - trainer.py 118-124 行损失：distance=(t_norm - s)^2，取 0.999 分位
    （最难的前 0.1%）求均值——"只罚最难的像素"的全部 5 行。

运行：python -m stage4.pdn_and_loss
"""
import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# PDN 微缩版（层结构与 core/modeling.py PDN 同序，通道缩小 16 倍）
# ---------------------------------------------------------------------------

class PDNMini(nn.Sequential):
    """Conv->ReLU->AvgPool x2 -> Conv1x1 -> Conv3x3 -> Conv4x4 -> Conv1x1。"""

    def __init__(self, in_ch=3, out_ch=16):
        layers = [
            nn.Conv2d(in_ch, 16, 4, padding=0), nn.ReLU(),
            nn.AvgPool2d(2, 2, padding=1),
            nn.Conv2d(16, 32, 4, padding=0), nn.ReLU(),
            nn.AvgPool2d(2, 2, padding=1),
            nn.Conv2d(32, 32, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, out_ch, 4), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 1),
        ]
        super().__init__(*layers)


def build_teacher_student(out_ch=16, seed=0):
    """主仓库同款三动作：teacher 冻结 / student 同构 / conv1 权重继承。"""
    torch.manual_seed(seed)
    teacher = PDNMini(out_ch=out_ch)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    student = PDNMini(out_ch=out_ch)
    with torch.no_grad():                               # 对应 s_conv.weight.copy_
        student[0].weight.copy_(teacher[0].weight)
        student[0].bias.copy_(teacher[0].bias)
    return teacher, student


# ---------------------------------------------------------------------------
# teacher 归一化 + 分位数困难挖掘损失
# ---------------------------------------------------------------------------

@torch.no_grad()
def teacher_channel_stats(teacher, images):
    """主仓库 teacher_normalization 的单批版：通道均值/方差。"""
    out = teacher(images)
    mean = out.mean(dim=(0, 2, 3), keepdim=True)
    std = out.flatten(2).std(dim=2).mean(0, keepdim=True)[..., None, None]
    return mean, std


def quantile_hard_mining_loss(distance: torch.Tensor, q: float = 0.999):
    """主仓库 trainer.py 122-124 行：取最烫的前 (1-q) 像素求均值。

    distance: [B,C,H,W] 任意形状；kthvalue 在展平后取第 q 分位。
    """
    flat = distance.flatten()
    k = max(1, int(flat.numel() * q))
    d_hard = torch.kthvalue(flat, k).values
    return distance[distance >= d_hard].mean()


def plain_mean_loss(distance: torch.Tensor):
    """对照组：全像素平均（被 99.9% 的简单像素稀释）。"""
    return distance.mean()


def loss_gradient_concentration(distance: torch.Tensor, q: float = 0.999):
    """两种损失下各像素收到的梯度权重：困难像素在量化版中被放大多少倍。"""
    d = distance.detach().requires_grad_(True)
    quantile_hard_mining_loss(d, q).backward()
    g_q = d.grad.abs().clone(); d.grad = None
    plain_mean_loss(d).backward()
    g_m = d.grad.abs().clone()
    hard_mask = d >= torch.quantile(d.detach(), q)
    return {"quantile_on_hard": g_q[hard_mask].mean().item(),
            "mean_on_hard": g_m[hard_mask].mean().item(),
            "boost": (g_q[hard_mask] / g_m[hard_mask].clamp_min(1e-12)
                      ).mean().item()}


if __name__ == "__main__":
    teacher, student = build_teacher_student()
    imgs = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        t = teacher(imgs); mean, std = teacher_channel_stats(teacher, imgs)
        t_norm = (t - mean) / std
    s = student(imgs)
    distance = (t_norm - s) ** 2
    convs_t = [m.weight for m in teacher if isinstance(m, nn.Conv2d)]
    convs_s = [m.weight for m in student if isinstance(m, nn.Conv2d)]
    print("teacher 输出:", tuple(t.shape), "| 冻结:", all(
        not p.requires_grad for p in teacher.parameters()))
    print("conv1 继承: 一致 =", torch.equal(convs_t[0], convs_s[0]),
          "| 其余独立: 一致 =", all(torch.equal(a, b)
                                    for a, b in zip(convs_t[1:], convs_s[1:])))
    print(f"量化困难损失 = {quantile_hard_mining_loss(distance):.4f}"
          f" vs 全均值 = {plain_mean_loss(distance):.4f}")
    stats = loss_gradient_concentration(distance)
    print(f"困难像素梯度放大倍数 ≈ {stats['boost']:.0f}x"
          "（99.9% 简单像素几乎不稀释它）")
