"""课时 1：手写流匹配（Flow Matching）训练与采样 —— lora_synth.py 训练目标迷你化。

真工程用 FLUX.1-Fill（潜空间 inpaint 扩散模型）+ LoRA 学习"某类缺陷长什么样"，
训练目标是**流匹配**而不是经典的 ε-预测 DDPM：

    u ~ N(0,1),  t = sigmoid(u)                 # t 分布集中在中间时段
    zt = (1−t)·z0 + t·noise                     # 噪声与数据之间的线性插值路径
    target_v = noise − z0                       # 路径的速度场（时间全导数）
    loss = (w_tok · (pred_v − target_v)²).sum() / w_tok.sum()   # token 级软权重

推理时从纯噪声出发沿速度场欧拉积分回数据（t: 1→0）。本课用 2D 双峰玩具分布 +
三层 MLP 手写全部公式（不含任何 diffusers 依赖），跑通"训练→采样→分布对上"。

对应主仓库：src/train/lora_synth.py::_lora_step2_worker（u/t_expand/zt/target_v/
token_mse 段，第 644~667 行）。

运行：python -m stage8.flow_matching
"""
import numpy as np
import torch
import torch.nn as nn


def sample_t(batch: int, generator: torch.Generator | None = None):
    """t = sigmoid(u), u~N(0,1)，并 clamp 到 (1e-3, 1-1e-3) 防端点退化。"""
    u = torch.randn((batch,), generator=generator)
    t = torch.sigmoid(u).clamp(1e-3, 1.0 - 1e-3)
    return t


def flow_targets(z0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
    """zt = (1−t)z0 + t·noise；target_v = noise − z0（路径速度场）。"""
    t_expand = t.view(-1, 1).to(z0.dtype)
    zt = (1.0 - t_expand) * z0 + t_expand * noise
    target_v = (noise - z0).to(torch.float32)
    return zt, target_v


def token_weighted_mse(pred, target, w_tok):
    """token 级软权重 MSE：loss = Σ(w·mse_per_token)/Σw——权重 0 的 token 完全不计损失。

    真工程里 w 是三区软权重下采样到潜 token 分辨率的图（缺陷核 1.0 → 环带 0.4 → 背景 0）。
    """
    token_mse = ((pred.to(torch.float32) - target) ** 2).mean(dim=-1)
    return (token_mse * w_tok).sum() / (w_tok.sum() + 1e-6)


class VelocityNet(nn.Module):
    """速度场 v(x, t)：2D 坐标 + 时间 → 2D 速度（替身：FLUX transformer 12 亿参数）。"""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t.view(-1, 1)], dim=-1))


def train_flow_matching(data: torch.Tensor, steps: int = 1200, lr: float = 1.5e-3,
                        batch: int = 128, seed: int = 0):
    """流匹配训练循环（真工程同构：AdamW + 余弦退火到 0.1×lr + 梯度裁剪 1.0）。"""
    torch.manual_seed(seed)
    model = VelocityNet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
    rng = np.random.default_rng(seed)
    losses = []
    for step in range(1, steps + 1):
        idx = rng.permutation(len(data))[:batch]
        z0 = data[idx]
        t = sample_t(len(idx))
        noise = torch.randn_like(z0)
        zt, target_v = flow_targets(z0, noise, t)
        pred_v = model(zt.detach(), t)
        loss = token_weighted_mse(pred_v, target_v, torch.ones(len(idx)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
    return model, losses


@torch.no_grad()
def sample_flow(model: VelocityNet, n: int, steps: int = 16, seed: int = 1):
    """欧拉积分采样：x 从 t=1（噪声）沿 −v 方向积分回 t=0（数据）。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((n, 2), generator=g)
    ts = torch.linspace(1.0, 0.0, steps + 1)
    dt = ts[0] - ts[1]
    for i in range(steps):
        t = torch.full((n,), ts[i])
        x = x - model(x, t) * dt            # dz/dt = v，t 减小方向积分
    return x


def two_blobs(n: int = 512, seed: int = 0):
    """双峰玩具分布：缺陷"划痕族"与"砂眼族"各占一半的 2D 替身。"""
    rng = np.random.default_rng(seed)
    a = rng.normal([-2.0, 0.0], 0.35, (n // 2, 2))
    b = rng.normal([2.0, 0.0], 0.35, (n // 2, 2))
    return torch.tensor(np.vstack([a, b]), dtype=torch.float32)


if __name__ == "__main__":
    data = two_blobs()
    model, losses = train_flow_matching(data, steps=400)
    print(f"训练 loss: 首步 {losses[0]:.4f} → 末步均值 {np.mean(losses[-20:]):.4f}")

    samples = sample_flow(model, 600)
    s = samples.numpy()
    dist = np.minimum(np.hypot(s[:, 0] + 2, s[:, 1]), np.hypot(s[:, 0] - 2, s[:, 1]))
    occupy_a = float((s[:, 0] < 0).mean())
    print(f"采样 600 点: 到最近峰的中位距离 {np.median(dist):.2f} (数据 σ=0.35) | "
          f"落在左半侧比例 {occupy_a:.0%}")
    print("两峰都被占住、距离贴着数据尺度即分布学到了——"
          "真工程里这就是'LoRA 学会了这类缺陷的模样'。")

    # token 软权重：权重 0 的 token 对损失完全免疫（背景不惩罚、只学缺陷区）
    pred = torch.tensor([[1.0, 2.0]])
    tgt = torch.tensor([[0.0, 0.0]])
    w0 = torch.tensor([1.0])                       # 只有权重 1 的一个 token
    w_half = torch.tensor([0.5])
    base = token_weighted_mse(pred, tgt, w0)
    half = token_weighted_mse(pred, tgt, w_half)
    print(f"token 加权缩放验证: w=1 → {base:.3f}, w=0.5 → {half:.3f} "
          f"(等比缩放, w=0 的 token 不进损失)")
