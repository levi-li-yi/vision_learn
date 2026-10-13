import pytest
import torch
from torch import nn

from stage2.amp_scaler import (
    fp16_gradient_underflow, matmul_dtype_under_autocast, scaler_round_trip,
)
from stage2.checkpoint_modes import (
    apply_mode, load_five, save_five,
)
from stage2.optim_scheduler import (
    DemoNet, assert_backbone_floor, build_scheduler, lr_at, split_param_groups,
)


class TestParamGroups:
    def test_split_by_prefix(self):
        groups = split_param_groups(DemoNet(), lr=1.0, backbone_lr_scale=0.1)
        assert groups[0]["name"] == "backbone" and groups[0]["lr"] == pytest.approx(0.1)
        assert groups[1]["name"] == "head" and groups[1]["lr"] == pytest.approx(1.0)
        n_bb = len(groups[0]["params"]); n_hd = len(groups[1]["params"])
        assert n_bb > 0 and n_hd > 0
        assert n_bb + n_hd == len(list(DemoNet().parameters()))   # 不重不漏

    def test_backbone_floor_assert(self):
        assert_backbone_floor(4e-4, 0.1, 4e-5)          # 主仓库默认值：合法
        with pytest.raises(AssertionError):
            assert_backbone_floor(4e-4, 0.05, 4e-5)     # 穿底：必须响亮失败


class TestScheduler:
    def test_warmup_then_cosine(self):
        lr, min_lr, warm, total = 1.0, 0.1, 10, 110
        opt = torch.optim.SGD(split_param_groups(DemoNet(), lr, 1.0))
        sch = build_scheduler(opt, total_steps=total, warmup_steps=warm,
                              start_factor=0.1, min_lr=min_lr)
        v0 = opt.param_groups[1]["lr"]
        assert v0 == pytest.approx(0.1 * lr, rel=1e-3)   # 起点 0.1x
        at_warm = lr_at(sch, opt, warm - 0)               # 走到热身结束
        assert at_warm == pytest.approx(lr, rel=0.05)     # 升到全速
        at_end = lr_at(sch, opt, total - warm)            # 余弦到底
        assert at_end <= min_lr * 1.2                     # 落地 min_lr 附近

    def test_cosine_monotone_decreasing_after_warmup(self):
        opt = torch.optim.SGD(split_param_groups(DemoNet(), 1.0, 1.0))
        sch = build_scheduler(opt, total_steps=200, warmup_steps=20, min_lr=0.05)
        for _ in range(20):                       # 先走完热身段（上升期，不计）
            sch.step()
        lrs = []
        for _ in range(160):                      # 余弦段应单调不升
            sch.step()
            lrs.append(opt.param_groups[1]["lr"])
        assert all(a >= b - 1e-9 for a, b in zip(lrs, lrs[1:]))   # 单调不升


class TestCheckpoint:
    def test_five_roundtrip(self, tmp_path):
        m = nn.Linear(4, 2)
        opt = torch.optim.AdamW(m.parameters())
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50)
        sc = torch.amp.GradScaler(enabled=True)
        for _ in range(7):
            sch.step()
        p = tmp_path / "st.pth.tar"
        save_five(m, opt, sch, sc, step=777, path=p)

        m2 = nn.Linear(4, 2)
        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            torch.optim.AdamW(m2.parameters()), 50)
        step = load_five(p, m2, scheduler=sch2)
        assert step == 777 and sch2.last_epoch == sch.last_epoch
        for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
            assert torch.equal(a, b)                     # 逐位恢复

    @pytest.mark.parametrize("mode,has,expect", [
        ("fresh", True, "delete_old_state"),
        ("resume", True, "restore_five"),
        ("finetune", False, "warn_from_scratch"),
    ])
    def test_modes(self, mode, has, expect):
        assert apply_mode(mode, has) == expect
        with pytest.raises(ValueError):
            apply_mode("unknown", True)


class TestAmp:
    def test_autocast_dtype(self):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        assert matmul_dtype_under_autocast(dev) == torch.bfloat16

    def test_fp16_underflow_and_scale_rescue(self):
        r = fp16_gradient_underflow()
        assert r["raw_fp16"] == 0.0                       # 1e-8 在 fp16 里消失
        assert abs(r["scaled_fp16"] - 1e-8 * 2 ** 16) / (1e-8 * 2 ** 16) < 1e-3

    def test_scaler_restores_true_gradient(self):
        assert scaler_round_trip() == pytest.approx(2.0)
