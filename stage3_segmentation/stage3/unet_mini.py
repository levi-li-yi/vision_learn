"""课时 2：迷你 U-Net 与跳连——为什么"跳连"能救回细节。

对应主仓库级联头之外的另一条主流路线（Demo E 的预演：精读
milesial/Pytorch-UNet 前先在这里理解跳连为什么有效）。

跳连（skip connection）：编码器每级下采样丢掉的细节，通过横向捷径
直接送到解码器同级——解码器不用"从压缩特征里凭空恢复纹理"。

运行：python -m stage3.unet_mini
"""
import torch
import torch.nn.functional as F
from torch import nn


class UNetTiny(nn.Module):
    """单级下采样 + 单级上采样 + 一条跳连的最小 U-Net。"""

    def __init__(self, ch=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(),
                                 nn.MaxPool2d(2))                    # 64->32
        self.mid = nn.Conv2d(ch, ch, 3, padding=1)
        self.dec = nn.ConvTranspose2d(ch, ch, 4, 2, 1)               # 32->64
        self.out = nn.Conv2d(ch + 1, 1, 1)      # 跳连拼接后 ch+输入1通道

    def forward(self, x):
        e = self.enc(x)                    # 编码：压缩+丢细节
        m = F.relu(self.mid(e))
        d = F.relu(self.dec(m))
        skip = x                            # 跳连：原分辨率细节直送
        return self.out(torch.cat([d, skip], dim=1))


class PlainEncDec(nn.Module):
    """对照组：同样的容量，没有跳连。"""

    def __init__(self, ch=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.mid = nn.Conv2d(ch, ch, 3, padding=1)
        self.dec = nn.ConvTranspose2d(ch, ch, 4, 2, 1)
        self.out = nn.Conv2d(ch * 2, 1, 1)   # 输入通道数对齐(cat 后同宽)

    def forward(self, x):
        x = F.relu(self.enc(x))
        x = F.relu(self.mid(x))
        x = F.relu(self.dec(x))
        # 无跳连时用零填充补齐通道（公平对照）
        pad = torch.zeros_like(x)
        return self.out(torch.cat([x, pad], dim=1))


def dice_score(pred, target, eps=1e-7):
    """
    Dice 分数：预测和答案的重合程度(1=完美)。
    """
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def make_shape_dataset(n=32, size=64, seed=0):
    """合成任务（偏难版）：低对比小方块——放大跳连的优势。

    块 6~12px、对比度 0.5、噪声底 0.15：无跳连的对照网在小步数内
    追不上（细节经下采样后难以恢复），跳连把原分辨率细节直送解码器。
    """
    g = torch.Generator().manual_seed(seed)
    imgs, masks = [], []
    for _ in range(n):
        img = torch.rand(1, size, size, generator=g) * 0.15
        m = torch.zeros(1, size, size)
        a = int(torch.randint(6, 12, (1,), generator=g))
        x = int(torch.randint(0, size - a, (1,), generator=g))
        y = int(torch.randint(0, size - a, (1,), generator=g))
        m[:, y:y + a, x:x + a] = 1
        img = img + m * 0.5
        imgs.append(img); masks.append(m)
    return torch.stack(imgs), torch.stack(masks)


def train_seg(net, imgs, masks, steps=80, lr=3e-3):
    """
    训练一个小分割网（BCE+Dice 组合损失），返回最终 Dice。
    """
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logits = net(imgs)
        loss = (F.binary_cross_entropy_with_logits(logits, masks)
                + dice_loss(logits, masks))
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = (torch.sigmoid(net(imgs)) > 0.5).float()
    return dice_score(pred, masks).item()


def dice_loss(logits, targets):
    """
    Dice 损失 = 1 - Dice 分数；逐样本展平算(整图面积视角)。
    """
    p = torch.sigmoid(logits).flatten(1); t = targets.flatten(1)
    return (1 - (2 * (p * t).sum(-1) + 1) / (p.sum(-1) + t.sum(-1) + 1)).mean()


if __name__ == "__main__":
    imgs, masks = make_shape_dataset()
    torch.manual_seed(0); d_unet = train_seg(UNetTiny(), imgs, masks)
    torch.manual_seed(0); d_plain = train_seg(PlainEncDec(), imgs, masks)
    print(f"U-Net(带跳连) Dice = {d_unet:.3f}")
    print(f"普通编解码(无跳连) Dice = {d_plain:.3f}")
    print("-> 跳连把原分辨率细节直送解码器，边界恢复明显更好")
