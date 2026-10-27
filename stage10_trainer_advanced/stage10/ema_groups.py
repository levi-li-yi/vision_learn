"""课时 2：EMA 影子权重 + 四参数组 AdamW + autotune —— trainer.py 训练器核心三件。

  1. EMA（decay=0.999）：评估/考试/best 交付**全走 EMA**——评估口径必须与落盘口径
     同源，否则"日志高分、重载低分"。三处工程细节：
     · 热身：d = min(decay, (1+step)/(10+step))——前期 shadow 还拖着随机初始化，
       直接 0.999 会把模型拖死，让 decay 从 ~0.1 平滑升上来；
     · 浮点 buffer（BN running_mean/var）同样做滑动平均、整数 buffer（如
       num_batches_tracked）照抄——buffer 照抄最新值会让 EMA 的 BN 统计抖动；
     · `torch._foreach_mul_/_add_` 批量算子一次更新整串参数（避免 Python 循环
       逐参数两次内核调用）；只同步 requires_grad 参数（冻结参数 deepcopy 时
       已一致）。
  2. AdamW 四参数组：lora 1e-4 / bb(骨干非 LoRA 可训) 1e-4 / neck / head(主 lr
     4e-4×batch/8)——修复"两分组把骨干可训参数全挂主 lr 高火力训练"的名不副实。
  3. autotune 三参数：batch=显存GB÷3 夹 [8,16]、lr=4e-4×batch/8、
     workers=CPU核÷2 夹 [2,8]——零命令行参数开箱即跑。

对应主仓库：src/train/trainer.py::ema_update / __init__(groups、_ema_* 段)、
src/train/main.py::autotune。

运行：python -m stage10.ema_groups
"""
import copy
import os

import torch
import torch.nn as nn


class ModelEMA:
    """EMA 影子权重：shadow ← d·shadow + (1−d)·θ（含热身与 buffer 分类处理）。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.step = 0
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        src_p = dict(model.named_parameters())
        dst_p = dict(self.ema_model.named_parameters())
        train_names = [n for n, p in src_p.items() if p.requires_grad]
        self._src = [src_p[n] for n in train_names]
        self._dst = [dst_p[n] for n in train_names]
        bufs = list(zip(model.buffers(), self.ema_model.buffers()))
        self._fbuf = [(b, b2) for b, b2 in bufs if b.dtype.is_floating_point]
        self._ibuf = [(b, b2) for b, b2 in bufs if not b.dtype.is_floating_point]

    @torch.no_grad()
    def update(self):
        """每个 optimizer step 后调用。"""
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        if self._dst:
            torch._foreach_mul_(self._dst, d)
            torch._foreach_add_(self._dst, self._src, alpha=1.0 - d)
        for b, b2 in self._fbuf:                 # BN running_mean/var 滑动平均
            b2.mul_(d).add_(b, alpha=1.0 - d)
        for b, b2 in self._ibuf:                 # num_batches_tracked 等照抄
            b2.copy_(b)
        self.step += 1


def build_param_groups(model: nn.Module, lora_lr: float = 1e-4, main_lr: float = 4e-4):
    """四参数组：lora / bb(骨干非LoRA) / neck / head——trainer.py 同款分组规则。"""
    groups = {"lora": [], "bb": [], "neck": [], "head": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            groups["lora"].append(p)
        elif n.startswith("image_encoder.backbone"):
            groups["bb"].append(p)
        elif n.startswith("image_encoder"):
            groups["neck"].append(p)
        else:
            groups["head"].append(p)
    return [
        {"params": groups["lora"], "lr": lora_lr, "name": "lora"},
        {"params": groups["bb"], "lr": lora_lr, "name": "bb"},
        {"params": groups["neck"], "lr": main_lr, "name": "neck"},
        {"params": groups["head"], "lr": main_lr, "name": "head"},
    ]


def autotune(vram_gb: float = None, cpu_count: int = None):
    """batch=GB÷3 夹 [8,16]；lr=4e-4×batch/8；workers=核÷2 夹 [2,8]。"""
    if vram_gb is None:
        vram_gb = (torch.cuda.get_device_properties(0).total_memory / 2**30
                   if torch.cuda.is_available() else 8.0)
    if cpu_count is None:
        cpu_count = os.cpu_count() or 8
    batch = int(max(8, min(16, vram_gb // 3)))
    return {"batch": batch, "lr": 4e-4 * batch / 8,
            "workers": max(2, min(8, cpu_count // 2))}


def make_lora_model(dim: int = 64):
    """带 Conv-LoRA 与 BN 的迷你模型（复用 stage10.conv_lora）。"""
    from stage10.conv_lora import MiniBackbone, apply_lora
    model = MiniBackbone(dim)
    apply_lora(model)
    return model


class WrappedNet(nn.Module):
    """给 MiniBackbone 套 image_encoder.* 名字，让分组规则可对拍。"""

    def __init__(self, bb):
        super().__init__()
        self.image_encoder = nn.Module()
        self.image_encoder.backbone = bb
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        return self.head(self.image_encoder.backbone(x))


if __name__ == "__main__":
    torch.manual_seed(0)
    model = WrappedNet(make_lora_model())
    x = torch.randn(4, 64, 8, 8)

    # 1. autotune
    for gb in (8, 12, 24, 48):
        t = autotune(vram_gb=gb, cpu_count=16)
        print(f"显存 {gb:>2}GB → batch={t['batch']}, lr={t['lr']:.1e}, workers={t['workers']}")

    # 2. 四参数组
    groups = build_param_groups(model)
    desc = ", ".join(f"{g['name']} lr={g['lr']:.0e} "
                     f"({sum(p.numel() for p in g['params'])} 参数)" for g in groups)
    print(f"参数组: {desc}")

    # 3. EMA：热身 + 收敛跟踪（探针取可训练的 lora_a——冻结的 conv.weight 永远不动）
    ema = ModelEMA(model, decay=0.999)
    probe_w0 = model.image_encoder.backbone.stages[3].lora_a.weight.detach().clone()
    with torch.no_grad():                       # 手动模拟权重漂移
        for step in range(1, 201):
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * 0.02)
            ema.update()
            if step in (1, 10, 50, 200):
                shadow = ema.ema_model.image_encoder.backbone.stages[3].lora_a.weight
                cur = model.image_encoder.backbone.stages[3].lora_a.weight
                gap = float((shadow - probe_w0).abs().mean()) / \
                    float((cur - probe_w0).abs().mean())
                print(f"step {step:>3}: decay={min(0.999, step / (10.0 + step)):.3f} | "
                      f"shadow漂移/真权重漂移 = {gap:.2f} (EMA 滞后平滑, 前期热身不被拖死)")
    bn_int = [b for b, _ in ema._ibuf]
    print(f"整数 buffer 照抄个数 {len(bn_int)} (num_batches_tracked 类); "
          f"浮点 buffer 滑动平均个数 {len(ema._fbuf)} (BN running stats)")
