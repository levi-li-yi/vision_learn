"""stage8 全模块测试：流匹配 / LoRA 注入 / 三区软权重与缓存 / 差分提纯。"""
import cv2
import numpy as np
import pytest
import torch

from stage8.diff_extract import (
    MAX_REFILL_ROUNDS, REFILL_MARGIN, compute_diff_and_sigma,
    cosine_feather_blend, deform_and_scale_template,
    extract_defect_mask_hysteresis, make_synthetic_case, quality_gate,
)
from stage8.flow_matching import (
    VelocityNet, flow_targets, sample_flow, sample_t, token_weighted_mse,
    train_flow_matching, two_blobs,
)
from stage8.lora_linear import (
    LoRALinear, TinyAttention, attach_lora, count_params, lora_state_dict,
    make_task, merge_all, train_model,
)
from stage8.soft_weight_crop import (
    MAX_LORA_CROPS, crops_per_defect, downsample_tokens, preencode_cache,
    sample_crop_window, subsample_by_area, three_zone_weights,
)


class TestFlowMatching:
    def test_t_clamped_and_centered(self):
        t = sample_t(10000)
        assert (t > 0).all() and (t < 1).all()
        assert 0.4 < float(t.median()) < 0.6              # sigmoid(标准正态) 中位 ≈ 0.5

    def test_flow_targets_formula(self):
        z0 = torch.tensor([[1.0, 2.0]])
        noise = torch.tensor([[3.0, 0.0]])
        t = torch.tensor([0.25])
        zt, v = flow_targets(z0, noise, t)
        # zt = 0.75·[1,2] + 0.25·[3,0] = [1.5, 1.5]
        assert torch.allclose(zt, torch.tensor([[1.5, 1.5]]))
        assert torch.allclose(v, torch.tensor([[2.0, -2.0]]))

    def test_token_weighted_mse(self):
        pred = torch.tensor([[1.0, 3.0], [0.0, 0.0]])
        tgt = torch.zeros(2, 2)
        w = torch.tensor([1.0, 0.0])                       # 第二个 token 权重 0
        loss = token_weighted_mse(pred, tgt, w)
        assert loss.item() == pytest.approx((1.0 + 9.0) / 2)   # 只算第一个
        assert token_weighted_mse(pred, tgt, torch.zeros(2)).item() == 0.0

    def test_train_and_sample_recovers_modes(self):
        data = two_blobs(n=512, seed=0)
        model, losses = train_flow_matching(data, seed=0)
        assert np.mean(losses[-20:]) < losses[0] * 0.7     # 收敛
        s = sample_flow(model, 400, steps=24, seed=1).numpy()
        dist = np.minimum(np.hypot(s[:, 0] + 2, s[:, 1]),
                          np.hypot(s[:, 0] - 2, s[:, 1]))
        assert float(np.median(dist)) < 0.9                # 样本贴住某个峰 (数据 σ=0.35)
        frac_left = float((s[:, 0] < 0).mean())
        assert 0.25 < frac_left < 0.75                     # 两峰都被占住


class TestLoRALinear:
    def test_identity_at_start(self):
        torch.manual_seed(0)
        attn = TinyAttention()
        x = torch.randn(2, 8, 16)
        base_out = attn(x).detach().clone()
        attach_lora(attn)                                 # 原地包裹：base 权重不变
        assert float((attn(x) - base_out).abs().max()) < 1e-5   # B=零 → 恒等

    def test_freeze_and_params(self):
        model = attach_lora(TinyAttention())
        for name, p in model.named_parameters():
            if "lora_" not in name:
                assert not p.requires_grad
            else:
                assert p.requires_grad
        total, trainable = count_params(model)
        assert trainable < total * 0.5                     # 只训旁路

    def test_targets_hit_four_modules(self):
        model = attach_lora(TinyAttention())
        hits = [m for m in model.modules() if isinstance(m, LoRALinear)]
        assert len(hits) == 4                              # to_q/k/v + to_out.0

    def test_train_reduces_loss(self):
        src, tgt = make_task()
        model = attach_lora(TinyAttention())
        losses = train_model(model, src, tgt, steps=150, seed=0)
        assert losses[-1] < losses[0] * 0.6

    def test_merge_equivalence(self):
        src, tgt = make_task()
        model = attach_lora(TinyAttention())
        train_model(model, src, tgt, steps=80, seed=0)
        model.eval()
        with torch.no_grad():
            before = model(src)
            merged = merge_all(model)
            after = model(src)
        assert len(merged) == 4
        assert torch.allclose(before, after, atol=1e-4)    # 折叠后逐位等价

    def test_lora_state_dict_compact(self):
        model = attach_lora(TinyAttention())
        sd = lora_state_dict(model)
        assert sd and all("lora_" in k for k in sd)
        n_lora = sum(v.numel() for v in sd.values())
        n_all = sum(v.numel() for v in model.state_dict().values())
        assert n_lora < n_all * 0.5


class TestSoftWeightCrop:
    def build(self):
        h = w = 128
        scratch = np.array([[40, 64], [70, 64], [70, 71], [40, 71]], np.int32)
        other = np.array([[90, 20], [110, 20], [110, 34], [90, 34]], np.int32)
        weight, heal = three_zone_weights((h, w), [scratch], [other], dilate_px=8)
        return weight, heal

    def test_zone_values(self):
        weight, heal = self.build()
        assert weight[64:72, 40:71].min() == pytest.approx(1.0)     # 核心 1.0
        # 划痕左缘 x=40：距核 8px 的 x=32 → 线性衰减到底 0.4；距 2px 的 x=38 → ≈0.85
        assert weight[67, 32] == pytest.approx(0.4, abs=0.06)
        assert weight[67, 38] > 0.80
        assert weight[67, 30] == 0.0                                # 8px 环带之外
        assert weight[0:10, 0:10].max() == 0.0                      # 远背景 0
        assert weight[20:35, 90:111].max() == 0.0                   # 无关标注 0
        assert (heal[20:35, 90:111] > 0).any()                      # 修补蒙版盖住它

    def test_downsample_tokens(self):
        weight, _ = self.build()
        tokens = downsample_tokens(weight, factor=16)
        assert tokens.shape == (8, 8)
        assert float(tokens.max()) > 0.9                           # max_pool 保核

    def test_preencode_cache(self):
        import tempfile
        calls = {"n": 0}

        def enc(x):
            calls["n"] += 1
            return torch.tensor([x])
        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/encoded.pt"
            a = preencode_cache([1, 2, 3], enc, path)
            b = preencode_cache([1, 2, 3], enc, path)
        assert calls["n"] == 3 and torch.equal(torch.stack(a), torch.stack(b))

    def test_crop_window_contains_bbox(self):
        rng = np.random.default_rng(0)
        poly = np.array([[500, 500], [540, 505], [542, 515], [502, 512]], np.int32)
        for _ in range(5):
            x0, y0 = sample_crop_window(1024, 1024, poly, 512, 8, rng)
            assert x0 <= 500 - 24 and 542 + 24 <= x0 + 512
            assert y0 <= 500 - 24 and 515 + 24 <= y0 + 512

    def test_crops_per_defect(self):
        assert crops_per_defect(15) == 4                  # ceil(60/15)
        assert crops_per_defect(3) == 4                   # min(4, 20) 封顶
        assert crops_per_defect(60) == 1
        assert crops_per_defect(100) == 1

    def test_subsample_by_area(self):
        recs = [{"area": a} for a in range(10, 510)]
        picked = subsample_by_area(recs)
        assert len(picked) == MAX_LORA_CROPS
        assert picked[0]["area"] == 10                    # 最小面积保留


class TestDiffExtract:
    def test_deform_shape_and_scale_clip(self):
        rng = np.random.default_rng(0)
        tpl = np.zeros((40, 90), dtype=np.uint8)
        cv2.rectangle(tpl, (5, 12), (85, 28), 255, -1)
        outs = [deform_and_scale_template(tpl, 300, 600, np.random.default_rng(s))
                for s in range(5)]
        assert all(o is not None for o in outs)
        for o in outs:
            area = int((o > 0).sum())
            assert 60 <= area <= 2400                      # 缩放 clip(0.5,2)² 的宽容区间

    def test_deform_deterministic(self):
        tpl = np.zeros((40, 90), dtype=np.uint8)
        import cv2
        cv2.rectangle(tpl, (5, 12), (85, 28), 255, -1)
        a = deform_and_scale_template(tpl, 300, 600, np.random.default_rng(7))
        b = deform_and_scale_template(tpl, 300, 600, np.random.default_rng(7))
        assert np.array_equal(a, b)

    def test_feather_inside_outside(self):
        import cv2
        clean = np.full((40, 40), 100, dtype=np.uint8)
        gen = np.full((40, 40), 200, dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (29, 29), 255, -1)
        out = cosine_feather_blend(clean, gen, mask, feather_px=4)
        assert (out[:8, :] == 100).all()                   # 掩码外严格原样
        assert out[20, 20] == 200                          # 深处 α=1 取生成图
        edge = out[10, 20]                                 # 边缘 1px：α=0.5−0.5cos(π/4)
        assert 100 < edge < 200

    def test_sigma_clipped_range(self):
        clean, gen, m_core, m_local, zone = make_synthetic_case()
        delta, sigma = compute_diff_and_sigma(clean, clean.copy(), m_core, m_local, zone)
        assert sigma == pytest.approx(2.0)                 # 无差异 → 下界 clip 2.0

    def test_hysteresis_recovers_planted_defect(self):
        clean, gen, m_core, m_local, zone = make_synthetic_case()
        blended = cosine_feather_blend(clean, gen, m_local)
        delta, sigma = compute_diff_and_sigma(clean, blended, m_core, m_local, zone)
        m_gt, peak_snr = extract_defect_mask_hysteresis(delta, sigma, m_core,
                                                       m_local, zone)
        inter = ((m_gt > 0) & (m_core > 0)).sum()
        union = ((m_gt > 0) | (m_core > 0)).sum()
        assert inter / union > 0.6 and peak_snr > 3.5

    def test_hysteresis_empty_when_no_change(self):
        clean, gen, m_core, m_local, zone = make_synthetic_case()
        delta, sigma = compute_diff_and_sigma(clean, clean.copy(), m_core, m_local, zone)
        m_gt, peak_snr = extract_defect_mask_hysteresis(delta, sigma, m_core, m_local, zone)
        assert (m_gt > 0).sum() == 0 and peak_snr == 0.0

    def test_quality_gate(self):
        core_area = 400
        good = np.zeros((20, 20), np.uint8)
        good[:10, :10] = 255                               # 100px ∈ [60, 1000]
        assert quality_gate(good, 5.0, core_area)[0]
        tiny = np.zeros((20, 20), np.uint8)
        tiny[:2, :3] = 255                                 # 6px < max(6, 0.15*400=60)
        assert not quality_gate(tiny, 5.0, core_area)[0]
        weak = good.copy()
        assert not quality_gate(weak, 2.0, core_area)[0]   # snr < 3.5
        huge = np.zeros((60, 60), np.uint8)
        huge[:50, :50] = 255                               # 2500 > 2.5*400
        assert not quality_gate(huge, 5.0, core_area)[0]

    def test_refill_rounds_constant(self):
        assert MAX_REFILL_ROUNDS == 3
        assert REFILL_MARGIN[0] == 1.15 and REFILL_MARGIN[1] == 1.25
