"""★Demo F：手写 Focal 与 Dice——core/loss_utils.py 的公式级复刻。
这个文件教你手写两种'打分公式'(Focal 和 Dice)，并用实验证明为什么它们比普通 BCE 更适合'缺陷只占画面 1%'的工业场景。


对应主仓库 sigmoid_focal_loss（65-100 行）与 dice_loss（142-169 行），
以及"为什么用 Focal+Dice 组合而不用纯 BCE"的完整答案：
  - BCE 对 1% 前景的数据有"偷懒解"：全部预测背景，损失已很低；
  - Focal 乘 (1-p_t)^gamma：已经很准的像素罚分骤减 -> 火力集中难像素；
  - Dice 看整图面积重合：全预测背景时 Dice=0 -> 重罚偷懒解。
本模块在人造 1% 前景数据上定量复现三者差异。

运行：python -m stage3.losses_focal_dice
"""
import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# 1. 手写三件（与主仓库逐式对齐）
# ---------------------------------------------------------------------------

def bce_manual(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """标准二元交叉熵（逐像素均值）。"""
    return F.binary_cross_entropy_with_logits(logits, targets)


def sigmoid_focal_manual(logits, targets, alpha=0.4, gamma=2.0):
    """主仓库式：bce * p_t^gamma * alpha_t（逐像素均值）。

    p_t = 判对方向的置信度；alpha_t = 正类 alpha / 负类 (1-alpha)。
    """
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = targets * (1 - probs) + (1 - targets) * probs
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    return (bce * p_t ** gamma * alpha_t).mean()


def dice_loss_manual(logits, targets, smooth=1.0):
    """主仓库式：1 - (2交+平滑)/(面积和+平滑)，整图一起算。"""
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1); targets = targets.flatten(1)
    inter = (probs * targets).sum(-1)
    denom = probs.sum(-1) + targets.sum(-1)
    return (1 - (2 * inter + smooth) / (denom + smooth)).mean()


# ---------------------------------------------------------------------------
# 2. 1% 前景的不平衡实验（工业缺陷的真实占比）
# ---------------------------------------------------------------------------

def make_imbalance_batch(n=16, h=64, w=64, fg_ratio=0.01, seed=0):
    """目标掩码：约 1% 像素为前景（一小块）。"""
    g = torch.Generator().manual_seed(seed)
    masks = torch.zeros(n, 1, h, w)
    k = max(1, int(h * w * fg_ratio))
    for i in range(n):
        idx = torch.randperm(h * w, generator=g)[:k]
        masks.view(n, -1)[i, idx] = 1.0
    return masks


def make_learnable_imbalance(n=16, h=64, w=64, seed=0):
    """可学习的 1% 前景：图上放一个 6x6 亮块，答案=亮块位置。

    与 make_imbalance_batch 的区别：掩码由输入决定（网络学得到），
    用于收敛对比实验；后者是纯随机散点（只用于"偷懒解罚分"分析）。
    """
    g = torch.Generator().manual_seed(seed)
    imgs = torch.rand(n, 1, h, w, generator=g) * 0.15
    masks = torch.zeros(n, 1, h, w)
    for i in range(n):
        y = int(torch.randint(2, h - 8, (1,), generator=g))
        x = int(torch.randint(2, w - 8, (1,), generator=g))
        imgs[i, :, y:y + 6, x:x + 6] += 0.7
        masks[i, :, y:y + 6, x:x + 6] = 1.0
    return imgs, masks


def lazy_solution_penalty(loss_fn) -> float:
    """偷懒解（全预测背景 logit=-6）在该损失下的值——越低越容易被偷懒。"""
    masks = make_imbalance_batch()
    logits = torch.full_like(masks, -6.0)     # sigmoid(-6)≈0 -> 全背景
    return float(loss_fn(logits, masks))


# ---------------------------------------------------------------------------
# 3. 收敛对比：同一小网络，BCE vs Focal vs Focal+3Dice
# ---------------------------------------------------------------------------

class TinySegNet(nn.Module):
    """
    两层卷积的迷你分割网——用于对比不同损失的收敛差异。
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(8, 1, 1))

    def forward(self, x):
        return self.net(x)


def train_with(loss_name, epochs=60, seed=0):
    """
    用指定损失训练 TinySegNet 若干步，返回最终 P/R——不平衡对比。
    """
    torch.manual_seed(seed)
    imgs, masks = make_learnable_imbalance(seed=1)      # 可学任务：亮块=答案
    net = TinySegNet()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        logits = net(imgs)
        if loss_name == "bce":
            loss = bce_manual(logits, masks)
        elif loss_name == "focal":
            loss = sigmoid_focal_manual(logits, masks)
        else:                                   # focal + 3*dice（主仓库配方）
            loss = (sigmoid_focal_manual(logits, masks)
                    + 3.0 * dice_loss_manual(logits, masks))
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = (torch.sigmoid(net(imgs)) > 0.5).float()
        tp = (pred * masks).sum(); fp = (pred * (1 - masks)).sum()
        fn = ((1 - pred) * masks).sum()
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    return {"precision": prec.item(), "recall": rec.item()}


if __name__ == "__main__":
    print("== 偷懒解（全预测背景）在各损失下的代价 ==")
    for name, fn in [("bce", bce_manual), ("focal", sigmoid_focal_manual),
                     ("dice", dice_loss_manual)]:
        print(f"  {name:<6}: {lazy_solution_penalty(fn):.4f}"
              + ("   <- 几乎不罚,BCE 的偷懒动机" if name == "bce" else
                 "   <- 重罚偷懒" if name == "dice" else ""))
    print("\n== 同预算训练后 (P/R) ==")
    for name in ("bce", "focal", "focal+3dice"):
        print(f"  {name:<10}: {train_with(name)}")
