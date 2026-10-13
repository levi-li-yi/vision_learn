"""★Demo C：CIFAR-10 完整训练脚本（不用任何高层框架）。
这个文件是阶段 2 的毕业作品：在 CIFAR-10 图片分类任务上写一个完整的训练脚本——热身调度、梯度裁剪、混合精度、断点续训全都有，不用任何高级框架，纯 PyTorch 从零写。


要求清单（提纲原文）：warmup+余弦调度、checkpoint 保存/续训、梯度裁剪——
再加本仓库同款：bf16 autocast、梯度累积、两组学习率（backbone 前缀可开关）。

两种运行模式：
  smoke（默认） : torchvision.FakeData 合成图，CPU 数秒跑完——验证骨架正确；
  real          : --data <目录> 真实 CIFAR-10（自动下载）。

运行：
  python -m stage2.demo_c_cifar                       # smoke 自检
  python -m stage2.demo_c_cifar --data ./cifar_data --epochs 30
通关练习：把 smoke 换成 real 跑到 test acc >= 85%（ResNet 结构或更长轮次）。
"""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .optim_scheduler import build_scheduler
from .train_loop import train_one_epoch, accuracy


# ---------------------------------------------------------------------------
# 模型：小 ResNet 风格 CNN（CIFAR 尺寸 32x32）
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    残差块：卷积+捷径(输入直加输出)——学"增量"比学"全部"容易。
    """
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(cout), nn.ReLU())
        self.short = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.short(x)


class SmallResNet(nn.Module):
    """backbone. 前缀留给差异化学习率实验（对应 backbone_lr_scale）。"""

    def __init__(self, nc=10):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 32), nn.MaxPool2d(2),      # 32->16
            ConvBlock(32, 64), nn.MaxPool2d(2),     # 16->8
            ConvBlock(64, 128), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, nc)

    def forward(self, x):
        return self.head(self.backbone(x).flatten(1))


# ---------------------------------------------------------------------------
# 数据：smoke 用 FakeData（无需下载、可复现）；real 用 CIFAR-10
# ---------------------------------------------------------------------------

def make_loaders(batch_size=64, smoke=True, data_dir=None):
    """
    构建 DataLoader；smoke 用合成数据(免下载)，real 用 CIFAR-10。
    """
    import torchvision
    import torchvision.transforms as T
    tf_train = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    tf_test = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    if smoke:
        train = torchvision.datasets.FakeData(512, (3, 32, 32), 10,
                                              transform=tf_train)
        test = torchvision.datasets.FakeData(128, (3, 32, 32), 10,
                                             transform=tf_test)
    else:
        train = torchvision.datasets.CIFAR10(data_dir, train=True,
                                             download=True, transform=tf_train)
        test = torchvision.datasets.CIFAR10(data_dir, train=False,
                                            download=True, transform=tf_test)
    kw = dict(num_workers=0, pin_memory=torch.cuda.is_available())
    return (DataLoader(train, batch_size, shuffle=True, **kw),
            DataLoader(test, batch_size, shuffle=False, **kw))


# ---------------------------------------------------------------------------
# 训练主流程：调度/裁剪/AMP/断点 全家桶
# ---------------------------------------------------------------------------

def run_training(epochs=2, smoke=True, data_dir=None, ckpt=None,
                 resume=False, accumulation_steps=1, lr=1e-3,
                 backbone_lr_scale=1.0, warmup_steps=50, device=None):
    """Demo C 完整训练：热身+余弦调度、梯度裁剪、bf16、断点保存/续训。

    对应主仓库 train_entry.run() 的单卡版——所有"高级"技巧全在这。
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders(smoke=smoke, data_dir=data_dir)
    model = SmallResNet().to(device)

    groups = [
        {"params": [p for n, p in model.named_parameters()
                    if n.startswith("backbone")], "lr": lr * backbone_lr_scale},
        {"params": [p for n, p in model.named_parameters()
                    if not n.startswith("backbone")], "lr": lr},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps = epochs * len(train_loader)
    sch = build_scheduler(opt, total_steps=steps // accumulation_steps,
                          warmup_steps=min(warmup_steps, steps // 4), min_lr=lr / 10)
    start_epoch, start_step = 0, 0

    ckpt_path = Path(ckpt) if ckpt else None
    if resume and ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["weight"])
        opt.load_state_dict(state["optimizer"])
        sch.load_state_dict(state["scheduler"])
        start_step = state["step"]
        start_epoch = start_step // max(1, len(train_loader))
        print(f"[resume] 从 step={start_step} (epoch {start_epoch}) 续训")

    step = start_step
    for ep in range(start_epoch, epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16
                                ) if device == "cuda" else torch.autocast("cpu", enabled=False):
                loss = nn.functional.cross_entropy(model(x), y) / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad()
                step += 1
        acc = accuracy(model, test_loader)
        print(f"epoch {ep+1}/{epochs} step={step} lr={opt.param_groups[1]['lr']:.2e} "
              f"test_acc={acc:.2%}")
        if ckpt_path:
            torch.save({"weight": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": sch.state_dict(),
                        "step": step}, ckpt_path)
            print(f"[ckpt] -> {ckpt_path}")
    return accuracy(model, test_loader)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="CIFAR-10 目录(给了即真实模式)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run_training(epochs=args.epochs, smoke=args.data is None,
                 data_dir=args.data, ckpt=args.ckpt, resume=args.resume)
