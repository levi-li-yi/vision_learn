"""课时 2：nn.Module / Dataset / DataLoader——主仓库三件套的微缩版。

对应主仓库：
  - core/seg.py 的 EfficientViTSeg(nn.Module)——forward 输入输出张量；
  - data_utils/data.py 的 FixtureSupervisedDataset(Dataset)——__len__/__getitem__；
  - train_entry 的 DataLoader(num_workers=4, collate_fn=allocate)——搬运工。

合成任务：给 16x16 的单通道图分类"亮块在左上还是在右下"（数据即答案，
无需下载；真实数据集在 Demo C/D）。

运行：python -m stage2.nn_module_dataset_loader
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# 1. 模型：两段卷积 + 池化 + 全连接（麻雀虽小，五脏俱全）
# ---------------------------------------------------------------------------

class TinyCNN(nn.Module):
    """对应主仓库三段式的影子：层在 __init__ 声明，forward 只写数据流。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16->8
            nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 8->4
        )
        self.head = nn.Linear(8 * 4 * 4, 2)

    def forward(self, x):                # x: [B,1,16,16] -> logits: [B,2]
        return self.head(self.features(x).flatten(1))


# ---------------------------------------------------------------------------
# 2. 数据集：合成"亮块位置"二分类
# ---------------------------------------------------------------------------

class BrightCornerDataset(Dataset):
    """对应主仓库 Dataset 契约：__len__ 报课表行数，__getitem__ 出一份样本。"""

    def __init__(self, n=256, size=16, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, 2, (n,), generator=g)
        self.noise = torch.rand(n, 1, size, size, generator=g) * 0.3
        self.size = size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.noise[idx].clone()
        lab = int(self.labels[idx])
        q = 0 if lab == 0 else self.size // 2          # 0=左上 1=右下
        img[:, q:q + self.size // 2, q:q + self.size // 2] += 0.7
        return img, lab


def collate_stack(batch):
    """对应主仓库 allocate：把一批 (图,标签) 拼成两个批量张量。"""
    imgs = torch.stack([b[0] for b in batch], 0)
    labs = torch.tensor([b[1] for b in batch])
    return imgs, labs


if __name__ == "__main__":
    ds = BrightCornerDataset()
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_stack)
    model = TinyCNN()
    imgs, labs = next(iter(loader))
    print("batch:", imgs.shape, labs.shape, "| 前向输出:", model(imgs).shape)
    print("参数量:", sum(p.numel() for p in model.parameters()))
