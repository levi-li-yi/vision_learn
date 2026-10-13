"""课时 4：AdamW 双组旋钮 + warmup→cosine 调度——trainer.__init__ 的复刻。

对应主仓库 trainer.py 39-88 行：
  - 分组：名字以 "backbone" 开头进骨干组（lr×backbone_lr_scale），其余进 head 组；
  - 断言：backbone_lr >= min_lr，否则当场报错（信息里写明三种修法）；
  - 调度：LinearLR(0.1→1.0, warmup_steps) --SequentialLR--> CosineAnnealingLR(min_lr)。
本模块把参数名前缀参数化（主仓库前缀是 image_encoder.）。

运行：python -m stage2.optim_scheduler
"""
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def split_param_groups(model: nn.Module, lr: float, backbone_lr_scale: float,
                       backbone_prefix: str = "backbone"):
    """按名字前缀把参数分成两组——差异化学习率的全部秘密。"""
    bb = [p for n, p in model.named_parameters() if n.startswith(backbone_prefix)]
    hd = [p for n, p in model.named_parameters() if not n.startswith(backbone_prefix)]
    return [
        {"params": bb, "lr": lr * backbone_lr_scale, "name": "backbone"},
        {"params": hd, "lr": lr, "name": "head"},
    ]


def assert_backbone_floor(lr: float, scale: float, min_lr: float):
    """对应主仓库 39-44 行断言：骨干组退火不能穿底。"""
    if lr * scale < min_lr:
        raise AssertionError(
            f"lr*scale ({lr*scale}) must stay >= min_lr ({min_lr}); "
            f"raise lr/scale or lower min_lr (min_lr==lr*scale => 骨干恒定)")


def build_scheduler(optimizer, total_steps: int, warmup_steps: int = 2000,
                    start_factor: float = 0.1, min_lr: float = 4e-5):
    """热身(0.1x→1x)接余弦退火(→min_lr)，里程碑在 warmup_steps。"""
    warm = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0,
                    total_iters=warmup_steps)
    decay = CosineAnnealingLR(optimizer,
                              T_max=max(1, total_steps - warmup_steps),
                              eta_min=min_lr)
    return SequentialLR(optimizer, [warm, decay], milestones=[warmup_steps])


class DemoNet(nn.Module):
    """带 backbone./head. 前缀的玩具模型。"""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 8)
        self.head = nn.Linear(8, 2)


def lr_at(scheduler, optimizer, step: int) -> float:
    """走到第 step 步时 head 组的学习率（作图/断言用）。"""
    for _ in range(step):
        scheduler.step()
    return optimizer.param_groups[1]["lr"]


if __name__ == "__main__":
    lr, scale, min_lr = 4e-4, 0.1, 4e-5
    assert_backbone_floor(lr, scale, min_lr)
    opt = torch.optim.AdamW(split_param_groups(DemoNet(), lr, scale))
    sch = build_scheduler(opt, total_steps=10000, warmup_steps=2000, min_lr=min_lr)
    for s in (1, 1000, 2000, 6000, 10000):
        print(f"step {s:>5}: head lr = {lr_at(sch, opt, s if s == 1 else s - (1 if s==1 else 0)):.2e}")
