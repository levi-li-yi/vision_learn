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
    """d(x^2)/dx = 2x —— 课本公式；作为标准答案，与 autograd 结果对拍。"""
    # 把 x 想成旋钮、y=x² 是效果：梯度的正负 = 往哪拧（为正就反向调小 x），
    # 绝对值 = 拧多少（离最低点 x=0 越远步子越大，越近自然收小）。
    return 2.0 * x


def autograd_quadratic_grad(x: float) -> float:
    """让 PyTorch 自己算（不给公式）：贴标签→前向记账→反向查表拼梯度，与手算版对拍。"""
    # requires_grad=True 相当于给数字贴"求导标签"：此后它参与的运算都会被记账。
    t = torch.tensor(float(x), requires_grad=True)
    y = t ** 2          # 前向：算出数值，同时记下"y 由 t 平方而来"（撒面包屑）
    y.backward()        # 反向：并非"自己推"公式——是查工程师预写好的求导规则表
    # 公式来源三层：①导数定理=数学家证的；②每个小操作的反向规则=PyTorch 工程师
    # 手写进 derivatives.yaml 求导表（pow 条目 exponent*self^(exponent-1)，代回即 2t）；
    # ③整个网络的复合公式没人写过——backward() 运行时查表+链式法则现场拼装。
    return t.grad.item()  # .grad 装着梯度张量，.item() 取出里面的普通小数


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
