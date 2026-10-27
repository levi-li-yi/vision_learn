"""课时 2：手写 LoRA 注入 nn.Linear —— PEFT LoraConfig(r=32, α=32) 的迷你复刻。

真工程给 FLUX.1-Fill transformer 挂 LoRA（diffusers add_adapter + peft.LoraConfig）：

    target_modules = ["to_k", "to_q", "to_v", "to_out.0",
                      "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out"]
    r=32, lora_alpha=32, init_lora_weights="gaussian"

原理：冻结原权重 W，旁路一条低秩通道  x → xW + (α/r)·(xA)B：
  · A 用 Kaiming/高斯初始化（有梯度、随机方向）；
  · B 零初始化——**起步恒等**（不破坏预训练分布，训练从零扰动开始）；
  · 训练完可以**合并** W' = W + (α/r)·B·A，推理零开销（fuse_lora + unload_lora_weights）。
本课在一个名字与真工程一致（to_q/to_k/to_v/to_out.0）的迷你注意力块上手工实现全套：
attach（冻结+旁路）/ 只导出 LoRA 参数 / merge（折叠回 W）/ 训练对照。

对应主仓库：src/train/lora_synth.py::_lora_step2_worker（lora_config 段、
get_peft_model_state_dict、save_lora_weights 原子落盘）与 _gpu_inpaint_worker
（load_lora_weights + fuse_lora + unload_lora_weights）。

运行：python -m stage8.lora_linear
"""
import math

import torch
import torch.nn as nn


class TinyAttention(nn.Module):
    """迷你单头自注意力：子模块名刻意与 FLUX transformer 同名（LoRA target 按名字匹配）。"""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(x.shape[-1]), dim=-1)
        return self.to_out(att @ v)


class LoRALinear(nn.Module):
    """冻结的 nn.Linear + 低秩旁路：y = xW + (α/r)·(xA)B；B=零 → 起步恒等。"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                      # 冻结预训练权重
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        self.lora_a = nn.Linear(base.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_normal_(self.lora_a.weight)      # A：随机方向
        nn.init.zeros_(self.lora_b.weight)               # B：零 → 恒等起步
        self.merged = False

    def forward(self, x):
        if self.merged:                                  # 已折叠：直接走合并后的 W'
            return self.base(x)
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scaling

    def merge(self):
        """把旁路折叠回 W：W' = W + (α/r)·B·A，之后与原模型逐位等价（fuse_lora 同义）。"""
        if not self.merged:
            with torch.no_grad():
                self.base.weight += (self.lora_b.weight @ self.lora_a.weight) * self.scaling
            self.merged = True
        return self


def attach_lora(model: nn.Module, r: int = 8, alpha: float = 8,
                targets=("to_q", "to_k", "to_v", "to_out.0")) -> nn.Module:
    """按模块名把 nn.Linear 替换成 LoRALinear（PEFT target_modules 按名字匹配同义）。"""
    replaced = {}

    def _replace(mod: nn.Module, prefix: str = ""):
        for child_name, child in mod.named_children():
            full = f"{prefix}{child_name}"
            if full in targets and isinstance(child, nn.Linear):
                replaced[full] = child
                setattr(mod, child_name, LoRALinear(child, r=r, alpha=alpha))
            else:
                _replace(child, prefix=f"{full}.")

    _replace(model)
    assert replaced, f"没有命中任何 LoRA target: {targets}"
    return model


def lora_state_dict(model: nn.Module) -> dict:
    """只导出 LoRA 参数（get_peft_model_state_dict 同义）——权重文件只有原模型的零头。"""
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if "lora_" in k}


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_task(seed: int = 0, n: int = 256, dim: int = 16, seq: int = 8):
    """玩具任务：把随机序列"去噪映射"到固定目标分布（模拟学一类缺陷风格）。"""
    g = torch.Generator().manual_seed(seed)
    src = torch.randn((n, seq, dim), generator=g)
    tgt = torch.tanh(src * 1.7 + 0.3)
    return src, tgt


def train_model(model: nn.Module, src, tgt, steps: int = 200, lr: float = 2e-3,
                seed: int = 0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    for _ in range(steps):
        pred = model(src)
        loss = ((pred - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def merge_all(model: nn.Module):
    """把所有 LoRALinear 旁路折叠回原权重（fuse_lora + unload 同义，推理零开销）。"""
    merged = []
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            module.merge()
            merged.append(name)
    return merged


if __name__ == "__main__":
    torch.manual_seed(0)
    src, tgt = make_task()

    # 基线：同一个冻结预训练注意力（attach 前后是同一个对象，恒等起步可对拍）
    model = TinyAttention()
    for p in model.parameters():
        p.requires_grad_(False)
    base_loss = float(((model(src) - tgt) ** 2).mean())
    frozen_out = model(src).detach().clone()

    # LoRA 微调：只训旁路（原地包裹，base 权重不变）
    attach_lora(model, r=8, alpha=8)
    total, trainable = count_params(model)
    print(f"LoRA 参数量: 可训练 {trainable} / 总 {total} "
          f"(16 维玩具比例失真；真工程 12 亿参数只训 ~1%)")
    before = model(src).detach()
    losses = train_model(model, src, tgt, steps=200)
    print(f"冻结基线 loss={base_loss:.4f} | LoRA 训练 loss {losses[0]:.4f} → {losses[-1]:.4f}")

    # 起步恒等（B=零初始化）与合并等价（W' = W + α/r·BA）
    delta0 = float((before - frozen_out).abs().max())
    out_before_merge = model(src).detach()
    merged_names = merge_all(model)
    out_after_merge = model(src).detach()
    print(f"零初始化恒等起步: 训练前附加输出与原模型最大偏差 {delta0:.2e}")
    print(f"合并 {len(merged_names)} 个旁路 {merged_names} 后输出最大偏差 "
          f"{float((out_before_merge - out_after_merge).abs().max()):.2e} (fuse_lora 等价)")

    probe = attach_lora(TinyAttention(), r=8, alpha=8)
    sd = lora_state_dict(probe)
    n_full = sum(v.numel() for v in probe.state_dict().values())
    n_lora = sum(v.numel() for v in sd.values())
    print(f"lora_state_dict 仅含旁路: {n_lora} 参数 (全模型 {n_full})——"
          "真工程 save_lora_weights 只落这个零头，交付体积 KB 级。")
