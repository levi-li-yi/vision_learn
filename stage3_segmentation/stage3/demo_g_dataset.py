"""★Demo G（离线骨架）：工业缺陷数据实战的本地预备课。

真实作业（README 有完整指引）：下载 NEU-DET（东北大学钢材表面缺陷，
含 pitted_surface 砂眼类）或 DAGM 2007，训一个分割模型。

本模块提供两样离线可测的东西：
  1) SynthNEU 生成器——"钢面纹理 + 缺陷斑块 + 像素答案"的合成替代，
     保证管线逻辑不下载也能跑通；
  2) 真实数据的目录契约 build_real_dataset()——按 NEU-DET 解包后的
     结构组织 imgs/masks，指明它怎样映射到本工程的数据拓扑。

运行：python -m stage3.demo_g_dataset
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SynthNEU(Dataset):
    """合成"钢面缺陷"：随机轧制纹理 + 1~3 个暗/亮斑块 = 缺陷答案。"""

    def __init__(self, n=128, size=96, seed=0):
        self.items = []
        rng = np.random.default_rng(seed)
        for _ in range(n):
            # 轧制纹理：竖向条纹 + 噪声
            img = np.tile(np.linspace(90, 160, size), (size, 1))
            img += rng.normal(0, 6, img.shape)
            mask = np.zeros((size, size), np.float32)
            for _ in range(rng.integers(1, 4)):
                cy, cx = rng.integers(10, size - 10, 2)
                r = rng.integers(4, 12)
                yy, xx = np.ogrid[:size, :size]
                blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
                dark = rng.random() < 0.5
                img[blob] += -50 if dark else 45
                mask[blob] = 1.0
            self.items.append((img.clip(0, 255).astype(np.float32) / 255.0,
                               mask))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, mask = self.items[i]
        return (torch.from_numpy(img).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0))

    def foreground_ratio(self) -> float:
        """平均前景占比——本实验里约 1~3%，与工业缺陷同级。"""
        return float(np.mean([m.sum() / m.size for _, m in self.items]))


def build_real_dataset(root: str):
    """NEU-DET 目录契约（作业下载后替换 SynthNEU 用）。

    期望结构（解包后自行归整）：
      root/images/*.bmp   原图
      root/masks/*.png    缺陷像素答案（没有像素答案时可用 Demo H 的
                          SAM 辅助生成，或退化用 bbox 中心圆近似）
    返回 (list[Path 图], list[Path 掩码])，供 DataLoader 消费。
    """
    root = Path(root)
    imgs = sorted((root / "images").glob("*.bmp"))
    masks = [p.with_name(p.stem + ".png").with_stem(p.stem)
             for p in imgs]
    masks = [root / "masks" / (p.stem + ".png") for p in imgs]
    missing = [m for m in masks if not m.exists()]
    if missing:
        raise FileNotFoundError(f"缺 {len(missing)} 张掩码，先补齐再训")
    return imgs, masks


if __name__ == "__main__":
    ds = SynthNEU(64)
    img, mask = ds[0]
    print("样本形状:", tuple(img.shape), tuple(mask.shape),
          "| 前景占比: %.2f%%" % (ds.foreground_ratio() * 100))
