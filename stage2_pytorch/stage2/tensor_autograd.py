"""课时 1：Tensor 与自动求导——"旋钮往哪拧、拧多少"的机器实现。
这个文件教你两件事：①PyTorch 怎么自动算梯度(不用手推导数)；②三种浮点格式(fp32/fp16/bf16)的区别——为什么训练用 bf16 不会溢出。


对应主仓库：trainer.train() 里 loss.backward() 一行背后发生的事；
以及 4.5 章"bf16 混合精度"（fp16 会溢出、bf16 指数位与 fp32 同宽）。

三个实验：
  1) 手算二次函数梯度 vs autograd —— 机器算的和课本公式一致；
  2) 三种浮点格式的位分配 —— 解释"为什么 bf16 天生不溢出"；
  3) 数值范围演示 —— fp16 放不大数变 inf，bf16 原样放下。

运行：python -m stage2.tensor_autograd
"""
import torch


# ---------------------------------------------------------------------------
# 1. 手算梯度 vs autograd
# ---------------------------------------------------------------------------

def manual_quadratic_grad(x: float) -> float:
    """d(x^2)/dx = 2x —— 课本公式。"""
    return 2.0 * x


def autograd_quadratic_grad(x: float) -> float:
    """让 PyTorch 自己算：requires_grad=True 声明"要跟踪对它的求导"。"""
    t = torch.tensor(float(x), requires_grad=True)
    y = t ** 2          # 前向：记录计算图
    y.backward()        # 反向：从 y 一路算回 t 的梯度
    return t.grad.item()


# ---------------------------------------------------------------------------
# 2. 浮点格式的位分配（符号/指数/尾数）
# ---------------------------------------------------------------------------

def float_dtype_bits() -> dict:
    """三种格式的 (总位, 指数位, 尾数位)——bf16 与 fp32 指数位同为 8。"""
    return {"float32": (32, 8, 23), "float16": (16, 5, 10), "bfloat16": (16, 8, 7)}


def fp16_overflows(v: float) -> bool:
    """fp16 最大约 65504：超了就变 inf（无监督线推理 NaN 事故的微型标本）。"""
    return bool(torch.tensor(v, dtype=torch.float16) == float("inf"))


def bf16_holds(v: float) -> bool:
    """bf16 指数位与 fp32 同宽：同样的大数放得下（精度略粗而已）。"""
    return bool(torch.isfinite(torch.tensor(v, dtype=torch.bfloat16)))


# ---------------------------------------------------------------------------
# 3. 演示
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for x in (3.0, -1.5, 0.0):
        print(f"d(x^2)/dx @ x={x}: 手算={manual_quadratic_grad(x)}, "
              f"autograd={autograd_quadratic_grad(x)}")

    print("\n浮点位分配:", float_dtype_bits())
    big = 1e5
    print(f"fp16 存 {big}: 溢出={fp16_overflows(big)}（变 inf）")
    print(f"bf16 存 {big}: 有限={bf16_holds(big)}（指数位够宽）")
    a = torch.tensor(0.123456789, dtype=torch.bfloat16).float()
    print(f"bf16 精度代价: 0.123456789 -> {a.item():.7f}（尾数 7 位，够训练用）")
