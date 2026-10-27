"""课时 1：Conv-LoRA 原位注入与 deepcopy 陷阱 —— seg.py::_attach_lora 复刻。

真工程给 EfficientViT-SAM 骨干 stages[3..4] 的 **1×1 Conv** 原位挂 LoRA 旁路：
  · 筛选条件：kernel==(1,1) 且 in/out ≥ 64（小于 64 通道的层旁路收益低于开销）；
  · 冻结原 conv，lora_a kaiming_uniform(a=√5)、lora_b 零初始化 → 起步恒等，
    **键名不变**（add_module 挂在 cl 上），官方权重可原样 strict 载入；
  · 前向替换：cl.forward = partial(_lora_fwd, cl, scale)。

本课的题眼是注释里那段**deepcopy 陷阱**：前向替换必须用 partial 捕获 cl——
deepcopy 会经 memo 把 partial 捕获的对象重绑到拷贝自身；而闭包/默认参数捕获的
函数对象在 deepcopy 中按引用共享，捕获不随拷贝走——EMA 拷贝的前向会执行**训练侧
原模块**：EMA 权重对这些层不生效，且训练态 SyncBN 在单 rank 前向中发 all_gather
与其他 rank 错位 → **多卡死锁**。本课用实验复现两种写法的差异。

再补两件配套设施：
  · train() 覆盖强制冻结段 BN eval（running stats 不更新）+ 构造后 BN 断言体检；
  · 起步权重自动识别（ckpt_utils.load_base）：键全为 image_encoder.* → 官方骨干
    逐键 copy + 白名单断言；否则全模型 strict=False 迁移。

对应主仓库：src/train/seg.py::_attach_lora / _apply_lora / OfficialEfficientViTSeg.train、
src/train/trainer.py（BN 断言段）、src/train/ckpt_utils.py::load_base。

运行：python -m stage10.conv_lora
"""
import copy
import functools

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """迷你 ConvLayer 替身：conv(1×1) + bn + act，与真工程 ConvLayer 同构。"""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MiniBackbone(nn.Module):
    """stages[0..2] 冻结 + stages[3..4] 挂 LoRA 的迷你骨干（1/2/4/8 通道分层）。"""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.stages = nn.ModuleList([
            ConvBlock(dim, dim), ConvBlock(dim, dim), ConvBlock(dim, dim),   # 冻结
            ConvBlock(dim, dim), ConvBlock(dim, dim),                        # 挂 LoRA
        ])
        for st in self.stages[:3]:
            for p in st.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        for st in self.stages:
            x = st(x)
        return x

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:                                   # 覆盖：冻结段 BN 永远 eval
            for st in self.stages[:3]:
                st.eval()
        return self


def _lora_fwd(cl_: nn.Module, s: float, x):
    """LoRA 前向：原前向 + α/r · B(A(x))。"""
    return type(cl_).forward(cl_, x) + s * cl_.lora_b(cl_.lora_drop(cl_.lora_a(x)))


def attach_lora_conv(cl: nn.Module, r: int = 16, alpha: float = 32.0,
                     dropout: float = 0.05, use_partial: bool = True) -> bool:
    """给 ConvBlock 原位挂 LoRA（键名不变）。use_partial=False 复刻"闭包陷阱"写法。"""
    conv = getattr(cl, "conv", None)
    if not isinstance(conv, nn.Conv2d) or conv.kernel_size != (1, 1) \
            or conv.in_channels < 64 or conv.out_channels < 64:
        return False
    for p in conv.parameters():
        p.requires_grad_(False)
    a = nn.Conv2d(conv.in_channels, r, 1, bias=False)
    b = nn.Conv2d(r, conv.out_channels, 1, bias=False)
    nn.init.kaiming_uniform_(a.weight, a=5 ** 0.5)
    nn.init.zeros_(b.weight)
    cl.add_module("lora_a", a)
    cl.add_module("lora_b", b)
    cl.add_module("lora_drop", nn.Dropout(dropout))
    scale = alpha / r
    if use_partial:
        # 正确写法：partial 捕获的 cl 在 deepcopy 中经 memo 重绑到拷贝自身
        cl.forward = functools.partial(_lora_fwd, cl, scale)
    else:
        # 错误写法（闭包捕获）：deepcopy 后 forward 里的 cl 仍指向训练侧原对象
        def _bad_fwd(x):
            return type(cl).forward(cl, x) + scale * cl.lora_b(cl.lora_drop(cl.lora_a(x)))
        cl.forward = _bad_fwd
    return True


def apply_lora(module: nn.Module, r: int = 16, alpha: float = 32.0,
               dropout: float = 0.05, use_partial: bool = True) -> int:
    """递归挂载：命中的层不再深入其子模块。"""
    n = 0
    for name, child in module.named_children():
        if attach_lora_conv(child, r, alpha, dropout, use_partial):
            n += 1
        else:
            n += apply_lora(child, r, alpha, dropout, use_partial)
    return n


def bn_live_count(model: nn.Module, stages_slice=slice(0, 3)) -> int:
    """体检：冻结 stages 里仍处于训练态的 BN 个数（必须为 0）。"""
    stages = getattr(model, "stages", None)
    if stages is None:
        return 0
    return sum(1 for st in stages[stages_slice] for m in st.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training)


def load_base(model: nn.Module, state_dict: dict) -> str:
    """起步权重自动识别：全为 image_encoder.* → 'encoder'（逐键 copy 白名单断言）；
    否则 → 'full'（load_state_dict strict=False 迁移）。"""
    if state_dict and all(k.startswith("image_encoder.") for k in state_dict):
        enc = model.image_encoder if hasattr(model, "image_encoder") else model
        ref = dict(enc.state_dict())
        unexpected = [k for k in state_dict
                      if k[len("image_encoder."):] not in ref]
        assert not unexpected, f"官方骨干含未知键: {unexpected[:3]}"
        with torch.no_grad():
            for k, v in state_dict.items():
                ref_key = k[len("image_encoder."):]
                assert ref_key in ref, f"骨干缺键: {ref_key}"
                ref[ref_key].copy_(v)
        return "encoder"
    missing = model.load_state_dict(state_dict, strict=False)
    return "full"


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 64, 8, 8)

    # 1. 恒等起步 + 键名不变
    bb = MiniBackbone()
    base_out = bb(x).detach().clone()
    n = apply_lora(bb)
    print(f"挂载 LoRA 层数: {n}（只挂 1×1 且通道 ≥64 的 conv）")
    print(f"起步恒等: max|Δ| = {float((bb(x) - base_out).abs().max()):.2e} (lora_b 零初始化)")
    names = [k for k, _ in bb.stages[3].named_parameters()]
    print(f"键名不变（lora 挂在原模块上）: {names}")

    # 2. deepcopy 陷阱对照
    good = MiniBackbone()
    apply_lora(good, use_partial=True)
    ema_good = copy.deepcopy(good)
    with torch.no_grad():                       # 把 EMA 拷贝的权重整体改掉
        for p in ema_good.parameters():
            p.mul_(1.5)
    diff_good = float((ema_good(x) - good(x)).abs().max())
    print(f"partial 写法: EMA 拷贝改权重后输出改变 max|Δ| = {diff_good:.3f} (>0 → 拷贝用自己的权重)")

    bad = MiniBackbone()
    apply_lora(bad, use_partial=False)          # 闭包陷阱
    ema_bad = copy.deepcopy(bad)
    with torch.no_grad():
        for p in ema_bad.parameters():
            p.mul_(1.5)
    diff_bad = float((ema_bad(x) - bad(x)).abs().max())
    print(f"闭包写法: EMA 拷贝改权重后输出不变 (max|Δ| = {diff_bad:.3f}) "
          f"—— 拷贝的前向仍执行训练侧原模块！多卡下原模块的训练态 SyncBN "
          f"还会在单 rank 前向中发 all_gather → 死锁")

    # 3. BN 断言体检
    bb.train()
    live = bn_live_count(bb)
    print(f"train() 覆盖后冻结段 BN 训练态个数 = {live} (必须 0，否则 running stats 被偷改)")
