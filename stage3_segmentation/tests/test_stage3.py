"""stage3 全模块测试：形状链 / 软边 / 跳连 / 损失公式 / 抽题 / 扫描。"""
import numpy as np
import pytest
import torch

from stage3.demo_g_dataset import SynthNEU
from stage3.demo_h_sam_prompt import (
    DEFAULT_LOCS, PromptableSeg, make_prompt_map, make_task,
    prompt_dice, train_promptable,
)
from stage3.encoder_decoder import BilinearHead, CascadeHead, soft_edge_fraction
from stage3.losses_focal_dice import (
    TinySegNet, bce_manual, dice_loss_manual, lazy_solution_penalty,
    make_imbalance_batch, sigmoid_focal_manual, train_with,
)
from stage3.pointrend_sampling import (
    sample_uncertain_points, uncertainty_neg_abs_logit,
)
from stage3.threshold_sweep import best_f1_threshold, micro_vs_per_sample, sweep_metrics
from stage3.unet_mini import PlainEncDec, UNetTiny, make_shape_dataset, train_seg


class TestEncoderDecoder:
    def test_cascade_shape_chain_matches_repo(self):
        """64->128->256——与主仓库 CascadedSegHead 前两级逐级一致。"""
        y = CascadeHead()(torch.randn(1, 32, 64, 64))
        assert list(y.shape) == [1, 1, 256, 256]

    def test_bilinear_head_same_output_size(self):
        y = BilinearHead()(torch.randn(1, 32, 64, 64))
        assert list(y.shape) == [1, 1, 256, 256]

    def test_bilinear_makes_soft_edges_nearest_does_not(self):
        mask = (torch.rand(1, 1, 32, 32) > 0.5).float()
        assert soft_edge_fraction(mask) > 0.05          # 双线性：软过渡带
        near = torch.nn.functional.interpolate(mask, scale_factor=4,
                                               mode="nearest")
        soft_near = ((near > 0.05) & (near < 0.95)).float().mean().item()
        assert soft_near == pytest.approx(0.0)          # 最近邻：保持硬边


class TestUNetSkip:
    def test_skip_connection_beats_plain(self):
        imgs, masks = make_shape_dataset(32, 64, seed=0)
        # 断崖式收敛：steps=100 时跳连已越过悬崖(≈0.91)、无跳连仍为 0
        torch.manual_seed(0); d_unet = train_seg(UNetTiny(), imgs, masks, steps=100)
        torch.manual_seed(0); d_plain = train_seg(PlainEncDec(), imgs, masks, steps=100)
        assert d_unet > 0.5
        assert d_plain < 0.3                            # 同预算远未爬上来


class TestLosses:
    def test_perfect_prediction_near_zero(self):
        masks = make_imbalance_batch(4, 32, 32, seed=1)
        logits = (masks * 2 - 1) * 8                    # 正确且自信
        assert dice_loss_manual(logits, masks).item() < 0.05
        assert sigmoid_focal_manual(logits, masks).item() < 0.01

    def test_focal_discounts_easy_pixels_vs_bce(self):
        """对已经很准的像素，focal 的逐像素罚 < bce 的逐像素罚。"""
        t = torch.tensor(1.0)
        logit_easy = torch.tensor(6.0)                  # p≈0.998 判对
        bce_px = torch.nn.functional.binary_cross_entropy_with_logits(
            logit_easy, t, reduction="none")
        focal_px = sigmoid_focal_manual(
            torch.tensor([[6.0]]), torch.tensor([[1.0]]))
        # 逐像素对比：focal 乘了 p_t^gamma < 1
        assert focal_px.item() < bce_px.item()

    def test_lazy_solution_dice_penalizes(self):
        """全预测背景：BCE 几乎不罚(偷懒动机)，Dice 重罚。"""
        b = lazy_solution_penalty(bce_manual)
        d = lazy_solution_penalty(dice_loss_manual)
        f = lazy_solution_penalty(sigmoid_focal_manual)
        assert d > b                                     # Dice 治偷懒
        assert d > f

    def test_repo_recipe_reaches_high_recall(self):
        """focal+3dice 在 1% 前景上收敛出高召回（直击工业痛点）。"""
        r = train_with("focal+3dice", epochs=80)
        assert r["recall"] > 0.5 and r["precision"] > 0.3


class TestPointRend:
    def test_point_count_and_shape(self):
        logits = torch.randn(2, 1, 64, 64)
        coords = sample_uncertain_points(logits, num_points=1000, seed=0)
        assert coords.shape == (2, 1000, 2)
        assert coords.max() < 64 and coords.min() >= 0

    def test_biased_to_uncertain_region(self):
        logits = (torch.rand(1, 1, 128, 128) - 0.5) * 10
        logits[:, :, 50:60, :] = (torch.rand(1, 1, 10, 128) - 0.5) * 0.5
        coords = sample_uncertain_points(logits, num_points=1024, seed=0)
        flat_unc = -logits.flatten().abs()
        chosen = flat_unc[coords[..., 0] * 128 + coords[..., 1]]
        baseline = flat_unc[torch.randperm(flat_unc.numel())[:1024]]
        # 不确定性=-|logit|：越接近 0 越拿不准 -> 被抽中者均值应更大(更靠 0)
        assert chosen.mean() > baseline.mean()

    def test_uncertainty_formula(self):
        assert uncertainty_neg_abs_logit(torch.tensor(0.0)).item() == pytest.approx(0.0)
        assert uncertainty_neg_abs_logit(torch.tensor(5.0)).item() == pytest.approx(-5.0)


class TestSweep:
    def test_perfect_separation_best_f1(self):
        # 背景分 0.12、前景分 0.34：只有 0.2 档完美分开（避免并列取首）
        scores = np.concatenate([np.full(50, 0.12), np.full(50, 0.34)])
        gts = np.concatenate([np.zeros(50), np.ones(50)])
        table = sweep_metrics(scores, gts)
        best = best_f1_threshold(table)
        assert best == 0.2
        assert table[best]["f1"] == pytest.approx(1.0)

    def test_eps_no_divzero_on_all_wrong(self):
        scores = np.full(10, 0.9); gts = np.zeros(10)
        table = sweep_metrics(scores, gts, thresholds=[0.5])
        assert table[0.5]["precision"] == pytest.approx(0.0, abs=1e-6)

    def test_micro_weights_large_defect_more(self):
        r = micro_vs_per_sample()
        assert r["micro_at_0.5"]["f1"] == pytest.approx(1.0)
        assert r["per_sample_mean_iou"] == 1.0           # 两尺都完美但口径不同


class TestDemoGH:
    def test_synth_neu_shapes_and_ratio(self):
        ds = SynthNEU(32, 96, seed=0)
        img, mask = ds[0]
        assert img.shape == (1, 96, 96) and mask.shape == (1, 96, 96)
        assert 0.005 < ds.foreground_ratio() < 0.10      # 工业级占比

    def test_prompt_changes_output(self):
        img, masks = make_task(seed=0)
        torch.manual_seed(0)                            # 构造 net 前定种子
        net = train_promptable(PromptableSeg(), img, masks, steps=250)
        for k, (d_self, d_other) in enumerate(
                prompt_dice(net, img, masks, DEFAULT_LOCS)):
            assert d_self > 0.5, f"prompt{k} 自家 Dice={d_self}"
            assert d_self > d_other * 1.5, (
                f"prompt{k} 条件化不足: self={d_self} other={d_other}")
