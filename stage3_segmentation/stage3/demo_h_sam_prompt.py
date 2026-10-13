"""★Demo H（概念版）：SAM 的 prompt 范式 vs 本工程的纯二分类分割。

真实作业（README）：跑 facebookresearch/segment-anything，用点/框
prompt 玩几张图。本模块离线呈现两者的**范式差异**——

  SAM：输入 (图, prompt) -> 输出"prompt 指的东西"的掩码；
       同一张图换个 prompt，输出跟着变——模型学的是"指哪切哪"。
  本工程：输入 (图) -> 输出"是不是缺陷"的单通道分数图；
       没有 prompt，类别空间写死为一类（缺陷）——"闭集语义分割"。

为什么要学这个（本工程没用 prompt，但……）：
  本工程的模型骨干（EfficientViTBackbone + SamNeck）和预训练权重
  （efficientvit_sam_b3_r256.pt）都来自 SAM 生态——但**只取了
  "眼睛"（图像编码器/特征提取器），没取"大脑"（prompt 编码器+
  掩码解码器）**。EfficientViTSeg 的 forward 是 head(encoder(x))，
  没有 prompt 的位置；判断逻辑（闭集二分类）是自研的分割头。
  准确说法："用了和 SAM 同源的特征提取器"，不是"在用 SAM"。
  学 prompt 范式是为了认识这双"眼睛"的原厂，理解它为什么天然会看图
  （在 11 亿个掩码上预训练过）；另外上游标注工具的"框一下自动描轮廓"
  用的正是 SAM 的 box prompt——你每天都在间接使用它。

实现要点（本玩具踩出来的两个真坑，SAM 训练同样要面对）：
  1) prompt 信号要够"显眼"：单像素 one-hot 在小网里会被稀释，
     这里用 5x5 高幅块（amp=8）——SAM 用独立的 prompt encoder 是同一道理；
  2) 监督必须 per-prompt：每个训练样本只监督"它带的 prompt 对应的
     输出通道"；若对全部通道施压，3/4 的梯度互相冲突，
     网络只会塌缩成"见亮就报"（0.4/0.4 的并集解）。

运行：python -m stage3.demo_h_sam_prompt
"""
import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_LOCS = [(16, 16), (48, 20), (20, 48), (48, 48)]


class PromptableSeg(nn.Module):
    """玩具版"promptable 分割"：prompt 的 one-hot 通道拼进输入。"""

    def __init__(self, hidden=8):
        super().__init__()
        # 第二层必须是 3x3：两层感受野 5px，prompt 信号才能覆盖到 blob 边缘
        # （换成 1x1 会退回"见亮就报"的并集解——真实踩过的坑）
        self.net = nn.Sequential(nn.Conv2d(1 + 4, hidden, 3, padding=1),
                                 nn.ReLU(), nn.Conv2d(hidden, 4, 3, padding=1))

    def forward(self, image, prompt_maps):
        return self.net(torch.cat([image, prompt_maps], dim=1))


def make_prompt_map(size, y, x, k, n_prompts=4, amp=8.0, r=3):
    """第 k 个 prompt = (y,x) 处的 5x5 高幅提示块 -> 第 k 通道。"""
    m = torch.zeros(1, n_prompts, size, size)
    m[0, k, max(0, y - r):min(size, y + r),
      max(0, x - r):min(size, x + r)] = amp
    return m


def make_task(size=64, seed=0):
    """四个角落各一个目标块；masks[k] = 第 k 个目标的真值。"""
    torch.manual_seed(seed)
    img = torch.rand(1, 1, size, size) * 0.2
    masks = torch.zeros(4, 1, size, size)
    for k, (y, x) in enumerate(DEFAULT_LOCS):
        img[:, :, y - 6:y + 6, x - 6:x + 6] += 0.6
        masks[k, :, y - 6:y + 6, x - 6:x + 6] = 1.0
    return img, masks


def train_promptable(net, img, masks, steps=250, lr=1e-2, seed=0):
    """per-prompt 监督：样本 b 只监督通道 b（SAM 同款训练口径）。

    注意不在此处重置随机种子：小网的条件化学习对初始化敏感，
    由调用方在构造 net 前统一 manual_seed（测试里就是这么做的）。
    """
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    pw = torch.tensor(20.0)
    size = img.shape[-1]
    for _ in range(steps):
        opt.zero_grad()
        total = 0
        for b, (y, x) in enumerate(DEFAULT_LOCS):
            logits = net(img, make_prompt_map(size, y, x, b))[0, b]
            total = total + F.binary_cross_entropy_with_logits(
                logits, masks[b, 0], pos_weight=pw)
        (total / len(DEFAULT_LOCS)).backward()
        opt.step()
    return net


@torch.no_grad()
def prompt_dice(net, img, masks, locs):
    """逐 prompt：自家目标 Dice vs 别家目标 Dice（条件化的证据）。"""
    size = img.shape[-1]
    out = []
    for k, (y, x) in enumerate(locs):
        pred = (torch.sigmoid(net(img, make_prompt_map(size, y, x, k))[0, k])
                > 0.5).float()
        d = lambda a, b: (2 * (a * b).sum()
                          / (a.sum() + b.sum() + 1e-9)).item()
        out.append((d(pred, masks[k]), d(pred, masks[(k + 1) % 4])))
    return out


if __name__ == "__main__":
    img, masks = make_task()
    net = train_promptable(PromptableSeg(), img, masks)
    for k, (ds, do) in enumerate(prompt_dice(net, img, masks, DEFAULT_LOCS)):
        print(f"prompt {k} -> 自家目标 Dice={ds:.3f} | 别家目标 Dice={do:.3f}")
    print("同一模型，四个 prompt 四个输出——SAM 范式的本质；")
    print("本工程无 prompt、只判'是不是缺陷'——闭集语义分割（见 README 对比表）")
