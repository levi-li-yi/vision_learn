"""stage4 全模块测试：范式地图 / PDN 继承 / 量化损失 / 标定阶梯 / Demo L。"""
import numpy as np
import pytest
import torch

from stage4.calibration_and_auc import (
    best_f1_threshold, bucket_key, build_bucket_table, map_quantiles,
    robust_quantiles, roc_auc, score_map, topk_mean,
)
from stage4.demo_l_teacher_student import (
    build_pair, evaluate, make_ng, make_ok, train_student,
)
from stage4.paradigm_map import (
    make_features, roc_auc as auc_t, score_autoencoder,
    score_cutpaste, score_memory_bank, score_teacher_student,
)
from stage4.pdn_and_loss import (
    build_teacher_student, loss_gradient_concentration,
    plain_mean_loss, quantile_hard_mining_loss, teacher_channel_stats,
)


class TestParadigms:
    def make_case(self, seed=0):
        ok, ng = make_features(seed=seed)
        x = torch.cat([ok, ng])
        y = torch.cat([torch.zeros(len(ok)), torch.ones(len(ng))])
        return ok, x, y

    def test_all_four_paradigms_detect(self):
        ok, x, y = self.make_case()
        for name, fn in [("AE", score_autoencoder),
                         ("TS", score_teacher_student),
                         ("CutPaste", score_cutpaste)]:
            auc = auc_t(y, fn(ok, x, seed=0))
            assert auc > 0.9, f"{name} AUC={auc}"
        _, x, y = self.make_case()
        bank = ok[:200]
        auc = auc_t(y, score_memory_bank(bank, x))
        assert auc > 0.9

    def test_auc_edges(self):
        y = torch.tensor([0., 0., 1., 1.])
        assert auc_t(y, torch.tensor([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
        assert auc_t(y, torch.tensor([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
        assert auc_t(y, torch.tensor([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


class TestPDN:
    def test_build_contract(self):
        teacher, student = build_teacher_student(seed=0)
        assert all(not p.requires_grad for p in teacher.parameters())
        convs_t = [m.weight for m in teacher if isinstance(m, torch.nn.Conv2d)]
        convs_s = [m.weight for m in student if isinstance(m, torch.nn.Conv2d)]
        assert torch.equal(convs_t[0], convs_s[0])        # conv1 继承
        assert not any(torch.equal(a, b)                  # 其余独立
                       for a, b in zip(convs_t[1:], convs_s[1:]))

    def test_channel_stats_shapes(self):
        teacher, _ = build_teacher_student()
        imgs = torch.rand(2, 3, 64, 64)
        mean, std = teacher_channel_stats(teacher, imgs)
        assert mean.shape[-2:] == (1, 1) and std.shape[-2:] == (1, 1)


class TestQuantileLoss:
    def test_equals_mean_of_top_tail(self):
        # 确定性构造：0..999 共 1000 个值；kthvalue(999) = 第 999 小 = 998，
        # 尾部 = {998, 999}（k 语义：k-th smallest，尾大小 = n-k+并列）
        d = torch.arange(1000, dtype=torch.float32).view(1, 1, 10, 100)
        assert quantile_hard_mining_loss(d, q=0.999).item() == pytest.approx(998.5)
        # 一般张量：与"阈值以上元素均值"逐位一致
        r = torch.rand(2, 4, 32, 32)
        flat = r.flatten()
        k = max(1, int(flat.numel() * 0.999))
        v = torch.kthvalue(flat, k).values
        assert quantile_hard_mining_loss(r).item() == pytest.approx(
            flat[flat >= v].mean().item(), rel=1e-5)

    def test_hard_pixel_gradient_boost(self):
        d = torch.rand(2, 1, 32, 32)
        stats = loss_gradient_concentration(d)
        assert stats["boost"] > 10          # 困难像素梯度放大 >10 倍


class TestCalibration:
    def test_normal_path_ungated(self):
        qs, qe, mode = robust_quantiles(torch.rand(4, 1, 64, 64))
        assert mode == "ungated" and qe > qs

    def test_degraded_ladder(self):
        const = torch.full((4, 1, 64, 64), 0.5)
        envs = torch.zeros_like(const); envs[..., :32] = 1
        qs, qe, mode = robust_quantiles(const, envs)
        assert mode == "eps" and qe > qs     # 恒保证 q_end > q_start
        assert qs.item() == pytest.approx(0.5)

    def test_gated_path_when_env_has_variation(self):
        # 全局退化、包络内不退化 -> gated：n=16384，k99.5 对应"顶部 82 个"；
        # 包络(每图 100px)内放 30 个随机值——全局 >0.5 的约 15 个 < 82 =>
        # 全局 q90=q99.5=0.5 退化；包络内随机占 30%，分位拉开。
        maps = torch.full((4, 1, 64, 64), 0.5)
        envs = torch.zeros_like(maps)
        envs[..., 10:20, 10:20] = 1                      # 包络：10x10 小窗
        maps[..., 10:20, 10:13] = torch.rand(4, 1, 10, 3)
        qs, qe, mode = robust_quantiles(maps, envs)
        assert mode == "gated" and qe > qs

    def test_score_map_order_preserving(self):
        qs, qe = torch.tensor(0.1), torch.tensor(0.5)
        m = torch.tensor([0.1, 0.3, 0.5, 0.9])
        s = score_map(m, qs, qe)
        assert s[0] == pytest.approx(0.0) and s[1] == pytest.approx(0.05)
        assert (s[1:] > s[:-1]).all()                    # 保序仿射

    def test_bucket_key_format(self):
        assert bucket_key(0, 0) == "x0_y0"
        assert bucket_key(800, 1024) == "x800_y1024"

    def test_bucket_table_min_samples_gate(self):
        thin = torch.rand(1, 1, 32, 32)                  # 只有 1 张 -> 薄桶
        fat = torch.rand(8, 1, 32, 32)
        table, (gqs, gqe), modes = build_bucket_table(
            {"x0_y0": thin, "x1_y0": fat})
        assert "x0_y0" not in table and "x1_y0" in table

    def test_topk_robust_to_single_spike(self):
        rng = np.random.default_rng(0)
        base = rng.normal(0.1, 0.02, (64, 64))
        spike = base.copy(); spike[10, 10] = 5.0
        assert topk_mean(spike) / topk_mean(base) < 1.5  # TopK 几乎不受影响
        assert spike.max() / base.max() > 20             # max 被劫持

    def test_best_f1(self):
        y = np.array([0] * 50 + [1] * 50)
        s = np.concatenate([np.random.default_rng(0).normal(0.3, 0.05, 50),
                            np.random.default_rng(1).normal(0.7, 0.05, 50)])
        thr, f1 = best_f1_threshold(y, s)
        assert f1 > 0.95 and 0.4 < thr < 0.6


class TestDemoL:
    def test_end_to_end_auc(self):
        torch.manual_seed(0)
        ok_train = make_ok(16, seed=0)
        ok_test = make_ok(8, seed=7)
        ng, _ = make_ng(make_ok(8, seed=9), seed=3)
        teacher, student = build_pair()
        mean, std = train_student(teacher, student, ok_train, steps=120)
        auc, (amap_ok, amap_ng, _, _) = evaluate(
            teacher, student, ok_test, ng, mean, std)
        assert auc > 0.85
        assert amap_ng.mean() > 2 * amap_ok.mean()      # 异常处距离暴涨
