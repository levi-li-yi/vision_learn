"""课时 5：混合精度(autocast bf16) 与 GradScaler——"快的部分快，准的部分准"。
这个文件教你'怎么让 GPU 算得快还不出错'：用 bf16 低精度提速，但关键运算保持高精度——'快的部分快，准的部分准'。


对应主仓库：
  - trainer.py 120-121 行 `autocast(dtype=torch.bfloat16)` 前向；
  - GradScaler 配 bf16 近似恒等、配 fp16 才真正干活（放大损失防小梯度下溢）；
  - 4.5 章真实事故：推理侧 fp16 溢出产 NaN —— 本模块亲手复现"溢出"。

三个实验：
  1) autocast 下矩阵乘输出 dtype 变 bf16（算得快的证据）；
  2) fp16 存大数变 inf，bf16 有限（位分配的实验版）；
  3) GradScaler 基本动作：scale(loss) 放大 -> backward -> unscale 还原。

运行：python -m stage2.amp_scaler（CUDA 可用则用 GPU，否则 CPU 模拟 dtype 行为）
"""
import torch
from torch import nn


def matmul_dtype_under_autocast(device: str) -> torch.dtype:
    """autocast 区域内线性层输出是 bf16——低精度干重活。"""
    net = nn.Linear(16, 16).to(device)
    x = torch.randn(4, 16, device=device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16
                        if device != "cpu" or torch.backends.mps.is_available() else torch.bfloat16):
        # CPU 也支持 bf16 autocast（torch>=2.0）
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            pass
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        out = net(x)
    return out.dtype


def fp16_gradient_underflow(lr_scale_needed: bool = True) -> dict:
    """小梯度在 fp16 下会"四舍五入成 0"——GradScaler 存在的理由。"""
    g_fp32 = torch.tensor(1e-8, dtype=torch.float32)
    g_fp16 = g_fp32.to(torch.float16)          # 太小 -> 0
    scaled = (g_fp32 * 2 ** 16).to(torch.float16)  # 先放大 -> 保得住
    return {"raw_fp16": g_fp16.item(), "scaled_fp16": scaled.item()}


def scaler_round_trip(device: str = "cpu"):
    """GradScaler 三步曲演示：scale -> backward -> unscale（梯度还原）。"""
    w = torch.tensor([1.0], requires_grad=True)
    scaler = torch.amp.GradScaler(enabled=True)
    loss = (w * 2).sum()
    scaler.scale(loss).backward()
    scaler.unscale_(torch.optim.SGD([w]))
    return w.grad.item()      # 还原后应等于真实梯度 2.0


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备:", dev)
    print("autocast 输出 dtype:", matmul_dtype_under_autocast(dev))
    print("fp16 梯度下溢实验:", fp16_gradient_underflow())
    print("GradScaler 还原梯度:", scaler_round_trip())
