"""课时 1：编码器-解码器与上采样——core/seg.py 级联头的微缩复刻。
这个文件教你语义分割模型的'喇叭结构'：先把图压缩看大局，再逐级放大恢复细节——两种放大方式(反卷积 vs 双线性)的差别用手感体验。


对应主仓库 CascadedSegHead 的"喇叭结构"：64->128(反卷积)->256(反卷积)
->1024(双线性 x4)。本模块用 64->128->256 三级复刻同一形状链，
并实验"双线性插值的软边 vs 反卷积可学边"。

两个核心问题：
  1) 为什么要逐级放大而不是一步到位？——一步拉伸=糊，逐级+卷积=锐；
  2) 反卷积(ConvTranspose2d)和双线性插值差在哪？——前者带可学参数
     （边放大边学怎么修），后者零参数纯几何拉伸。

运行：python -m stage3.encoder_decoder
"""
import torch
import torch.nn.functional as F
from torch import nn


class CascadeHead(nn.Module):
    """主仓库 CascadedSegHead 的同构缩小版（通道减半、末级不放大到 1024）。

    形状链与主仓库逐级对齐：in(64x64) --up1--> 128x128 --up2--> 256x256。
    """

    def __init__(self, in_ch=32, mid_ch=16, out_ch=1):
        super().__init__()
        self.up1 = nn.Sequential(                       # 对应主仓库 up1
            nn.ConvTranspose2d(in_ch, mid_ch, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(),
        )
        self.up2 = nn.Sequential(                       # 对应主仓库 up2
            nn.ConvTranspose2d(mid_ch, mid_ch // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid_ch // 2), nn.ReLU(),
            nn.Conv2d(mid_ch // 2, out_ch, 1),
        )

    def forward(self, x):
        x = self.up1(x)                                 # 64->128
        x = self.up2(x)                                 # 128->256
        return x


class BilinearHead(nn.Module):
    """对照组：同样的卷积精修，但放大全部用零参数双线性。"""

    def __init__(self, in_ch=32, mid_ch=16, out_ch=1):
        super().__init__()
        self.refine1 = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 3, padding=1),
                                     nn.BatchNorm2d(mid_ch), nn.ReLU())
        self.refine2 = nn.Sequential(nn.Conv2d(mid_ch, out_ch, 3, padding=1))

    def forward(self, x):
        x = self.refine1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False)          # 64->128 纯拉伸
        x = self.refine2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False)          # 128->256 纯拉伸
        return x


def soft_edge_fraction(mask: torch.Tensor) -> float:
    """0/1 掩码被插值放大后，"既非 0 也非 1"的过渡像素占比——软边度量。

    双线性天然产生软边（主仓库"图像可以柔"的位级证据）；最近邻则保持硬边
    （主仓库"答案必须硬"、掩码永远 NEAREST 的原因）。
    """
    up = F.interpolate(mask.float(), scale_factor=4, mode="bilinear",
                       align_corners=False)
    soft = ((up > 0.05) & (up < 0.95)).float().mean().item()
    return soft


if __name__ == "__main__":
    print("== 级联头形状链（应与主仓库 64->128->256 一致）==")
    head = CascadeHead()
    y = head(torch.randn(1, 32, 64, 64))
    print(f"输入 [1,32,64,64] -> 输出 {list(y.shape)}")

    print("\n== 双线性 vs 最近邻：软边占比 ==")
    mask = (torch.rand(1, 1, 32, 32) > 0.5).float()
    near = F.interpolate(mask, scale_factor=4, mode="nearest")
    up_bil = F.interpolate(mask, scale_factor=4, mode="bilinear",
                           align_corners=False)
    soft_bil = ((up_bil > 0.05) & (up_bil < 0.95)).float().mean().item()
    soft_near = ((near > 0.05) & (near < 0.95)).float().mean().item()
    print(f"双线性软边占比 = {soft_bil:.2%}（图像用：过渡自然）")
    print(f"最近邻软边占比 = {soft_near:.2%}（答案用：非黑即白）")
