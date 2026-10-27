"""stage10 全模块测试：Conv-LoRA 与 deepcopy 陷阱 / EMA / 参数组 / DDP。"""
import copy

import pytest
import torch
import torch.nn as nn

from stage10.conv_lora import (
    MiniBackbone, apply_lora, attach_lora_conv, bn_live_count, load_base,
)
from stage10.ddp_spawn import find_free_port, run_ddp
from stage10.ema_groups import (
    ModelEMA, autotune, build_param_groups, make_lora_model,
)
from stage10.gpu_metrics import GPUMetricAccumulator, TinySeg, warmup_infer_shapes


class TestConvLora:
    def test_identity_start_and_keynames(self):
        torch.manual_seed(0)
        bb = MiniBackbone()
        x = torch.randn(2, 64, 8, 8)
        base_out = bb(x).detach().clone()
        n = apply_lora(bb)
        assert n > 0
        assert float((bb(x) - base_out).abs().max()) < 1e-5     # B=零 → 恒等
        assert any("lora_a" in k for k, _ in bb.stages[3].named_parameters())
        assert any("lora_b" in k for k, _ in bb.stages[3].named_parameters())

    def test_freeze_and_filter(self):
        bb = MiniBackbone()
        apply_lora(bb)
        for st_idx, st in enumerate(bb.stages):
            for name, p in st.named_parameters():
                if "lora_" in name:
                    assert p.requires_grad
                elif st_idx < 3:
                    assert not p.requires_grad            # 冻结段
        # 小通道 conv 不挂（筛 in/out ≥ 64）
        small = nn.Sequential(nn.Conv2d(8, 8, 1))
        wrapped = type("W", (nn.Module,), {})()
        wrapped.conv = small
        assert attach_lora_conv(wrapped) is False

    def test_partial_deepcopy_trap(self):
        # 正确写法（partial）：EMA 拷贝改权重后输出跟着变
        torch.manual_seed(0)
        x = torch.randn(1, 64, 8, 8)
        good = MiniBackbone()
        apply_lora(good, use_partial=True)
        ema = copy.deepcopy(good)
        with torch.no_grad():
            for p in ema.parameters():
                p.mul_(1.5)
        assert float((ema(x) - good(x)).abs().max()) > 0.1

        # 错误写法（闭包）：拷贝的前向仍执行原模块 → 改权重无效
        bad = MiniBackbone()
        apply_lora(bad, use_partial=False)
        ema_bad = copy.deepcopy(bad)
        with torch.no_grad():
            for p in ema_bad.parameters():
                p.mul_(1.5)
        assert float((ema_bad(x) - bad(x)).abs().max()) < 1e-6

    def test_bn_assert_after_train(self):
        bb = MiniBackbone()
        apply_lora(bb)
        bb.train()
        assert bn_live_count(bb) == 0                     # 冻结段 BN 全 eval
        bb.eval()
        bb.train()
        assert bn_live_count(bb) == 0                     # 反复切换仍守住

    def test_train_override_keeps_lora_training(self):
        bb = MiniBackbone()
        apply_lora(bb)
        bb.train()
        # 挂 LoRA 的 stages[3] BN 应保持训练态
        assert all(m.training for m in bb.stages[3].modules()
                   if isinstance(m, nn.BatchNorm2d))

    def test_load_base_recognizes_encoder(self):
        bb = MiniBackbone()
        sd = {f"image_encoder.{k}": v for k, v in bb.state_dict().items()}
        kind = load_base(bb, sd)
        assert kind == "encoder"
        partial_sd = {"stages.0.conv.weight": torch.zeros_like(
            bb.stages[0].conv.weight)}
        kind2 = load_base(bb, partial_sd)
        assert kind2 == "full"                            # 非全 encoder.* → 全模型迁移


class TestEMA:
    def test_warmup_decay(self):
        ema = ModelEMA(nn.Linear(4, 4), decay=0.999)
        assert ema.step == 0
        assert min(0.999, (1 + 0) / (10 + 0)) == pytest.approx(0.1)  # 起步 ≈0.1
        for _ in range(1000):
            ema.update()
        # 热身公式 (1+step)/(10+step)：1000 步时 1001/1010≈0.991，未到 0.999
        assert min(0.999, (1 + 1000) / (10 + 1000)) == pytest.approx(0.99109)
        for _ in range(9000):                            # 到 ~9000 步才升满 0.999
            ema.update()
        assert min(0.999, (1 + 10000) / (10 + 10000)) == pytest.approx(0.999)

    def test_shadow_tracks_settled_weights(self):
        torch.manual_seed(0)
        model = nn.Linear(8, 8)
        ema = ModelEMA(model, decay=0.9)
        with torch.no_grad():
            for _ in range(100):                         # 抖动阶段（零均值噪声）
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.05)
                ema.update()
            for _ in range(2000):                        # 权重停住 → shadow 收敛贴住
                ema.update()
        assert float((ema.ema_model.weight - model.weight).abs().max()) < 1e-3

    def test_buffers_handled_by_dtype(self):
        model = nn.BatchNorm1d(4)
        ema = ModelEMA(model, decay=0.5)
        assert len(ema._fbuf) == 2                        # running_mean/var 滑动平均
        assert len(ema._ibuf) == 1                        # num_batches_tracked 照抄
        x = torch.randn(8, 4)
        model(x)
        ema.update()
        assert ema.ema_model.num_batches_tracked == model.num_batches_tracked

    def test_ema_model_frozen(self):
        ema = ModelEMA(nn.Linear(4, 4))
        assert all(not p.requires_grad for p in ema.ema_model.parameters())
        assert not ema.ema_model.training


class TestParamGroups:
    def make_wrapped(self):
        from stage10.ema_groups import WrappedNet
        return WrappedNet(make_lora_model())

    def test_four_groups_with_names(self):
        model = self.make_wrapped()
        groups = build_param_groups(model, lora_lr=1e-4, main_lr=4e-4)
        names = [g["name"] for g in groups]
        assert names == ["lora", "bb", "neck", "head"]
        assert len(groups[0]["params"]) > 0               # LoRA 参数就位
        assert all(g["lr"] == 1e-4 for g in groups[:2])
        assert all(g["lr"] == 4e-4 for g in groups[2:])
        n_grouped = sum(len(g["params"]) for g in groups)
        n_train = sum(1 for p in model.parameters() if p.requires_grad)
        assert n_grouped == n_train                       # 无一遗漏

    def test_autotune_formula(self):
        assert autotune(vram_gb=24, cpu_count=32) == {"batch": 8, "lr": 4e-4,
                                                      "workers": 8}
        assert autotune(vram_gb=48, cpu_count=4)["batch"] == 16
        assert autotune(vram_gb=48, cpu_count=4)["lr"] == pytest.approx(8e-4)
        assert autotune(vram_gb=48, cpu_count=4)["workers"] == 2


class TestGPUMetrics:
    def test_accumulate_and_pop(self):
        torch.manual_seed(0)
        model = TinySeg()
        acc = GPUMetricAccumulator(device="cpu")
        masks = torch.zeros(8, 64, 64, dtype=torch.long)
        masks[:, 20:28, 20:28] = 1
        masks[:, 0, 0] = 255                              # 忽略像素
        probs = torch.sigmoid(model(torch.randn(8, 3, 64, 64)).squeeze(1))
        acc.update(torch.tensor(0.5), torch.tensor(0.3), probs, masks)
        acc.update(torch.tensor(0.5), torch.tensor(0.3), probs, masks)
        m = acc.pop_metrics(2)
        assert acc.sync_count == 1
        assert m["focal"] == pytest.approx(0.5) and m["dice"] == pytest.approx(0.3)
        assert 0.0 <= m["iou"] <= 1.0
        m2 = acc.pop_metrics(1)
        assert m2["iou"] == 0.0                           # 取回后清零

    def test_iou_ignores_255_and_empty(self):
        acc = GPUMetricAccumulator(device="cpu")
        probs = torch.zeros(2, 8, 8)                      # 全预测 0
        masks = torch.full((2, 8, 8), 255)                # 全忽略
        acc.update(torch.tensor(0.0), torch.tensor(0.0), probs, masks)
        m = acc.pop_metrics(1)
        assert m["iou"] == 0.0                            # 无有效样本不除零

    def test_perfect_prediction_iou_one(self):
        acc = GPUMetricAccumulator(device="cpu")
        probs = torch.zeros(1, 8, 8)
        probs[:, 2:5, 2:5] = 1.0                          # sigmoid(大 logit)≈1
        masks = torch.zeros(1, 8, 8, dtype=torch.long)
        masks[:, 2:5, 2:5] = 1
        acc.update(torch.tensor(0.0), torch.tensor(0.0), probs, masks)
        assert acc.pop_metrics(1)["iou"] == pytest.approx(1.0)

    def test_warmup_shapes(self):
        model = TinySeg()
        assert warmup_infer_shapes(model) == 2            # b16 + b8 两规格


class TestDDP:
    def test_find_free_port(self):
        p = find_free_port()
        assert 1024 < p < 65536

    def test_two_process_ddp_cpu(self):
        res = run_ddp(world_size=2)
        assert res["last_loss"] < res["first_loss"]       # 训练有效
        assert res["all_same"] is True                    # 各 rank 参数一致
        assert res["broadcast"] == pytest.approx(res["last_loss"], rel=1e-4)
