import math

import pytest
import torch

from stage2.tensor_autograd import (
    autograd_quadratic_grad, bf16_holds, float_dtype_bits, fp16_overflows,
    manual_quadratic_grad,
)


class TestAutograd:
    @pytest.mark.parametrize("x", [3.0, -1.5, 0.0, 0.123])
    def test_matches_manual(self, x):
        assert autograd_quadratic_grad(x) == pytest.approx(manual_quadratic_grad(x))

    def test_chain_rule(self):
        """复合函数 y=(3x)^2，dy/dx=18x——backward 沿计算图逐级相乘。"""
        t = torch.tensor(2.0, requires_grad=True)
        y = (3 * t) ** 2
        y.backward()
        assert t.grad.item() == pytest.approx(36.0)


class TestDtypeBits:
    def test_table(self):
        b = float_dtype_bits()
        assert b["float32"] == (32, 8, 23)
        assert b["float16"] == (16, 5, 10)
        assert b["bfloat16"] == (16, 8, 7)
        # bf16 与 fp32 指数位同宽：这是"天然不溢出"的位级原因
        assert b["bfloat16"][1] == b["float32"][1]

    def test_fp16_overflow_vs_bf16(self):
        big = 1e5                      # fp16 上限约 65504
        assert fp16_overflows(big)
        assert bf16_holds(big)
        small = 3e4                    # fp16 放得下的范围
        assert not fp16_overflows(small)
