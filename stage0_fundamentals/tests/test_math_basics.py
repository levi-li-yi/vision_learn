"""math_basics 的测试：矩阵乘/距离/方差公式/分位数(kthvalue 口径)/困难挖掘/AUC。

交叉验证原则：手写实现 vs NumPy 的"另一个算法"（quickselect、linalg.norm、
matmul），不测同一条代码路径。
"""

import numpy as np

from stage0.math_basics import (
    auc_by_hand, hard_mining_quantile, kth_smallest, l2_distance_by_hand,
    map_quantiles, matmul_by_hand, mean_std_by_sum_sq,
    squared_channel_distance, simulate_anomaly_scores
)


class TestMatmul:
    def test_matches_numpy_on_known(self):
        a = np.array([[1., 2.], [3., 4.]])
        b = np.array([[5., 6.], [7., 8.]])
        np.testing.assert_allclose(matmul_by_hand(a, b), a @ b)

    def test_matches_numpy_random_rect(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(3, 5))
        b = rng.normal(size=(5, 2))
        np.testing.assert_allclose(matmul_by_hand(a, b), np.matmul(a, b))

    def test_shape_mismatch_rejected(self):
        import pytest
        with pytest.raises(ValueError):
            matmul_by_hand(np.zeros((2, 3)), np.zeros((2, 3)))


class TestDistance:
    def test_l2_matches_numpy_norm(self):
        rng = np.random.default_rng(1)
        u, v = rng.normal(size=6), rng.normal(size=6)
        assert abs(l2_distance_by_hand(u, v) - np.linalg.norm(u - v)) < 1e-10

    def test_l2_known_value(self):
        assert abs(l2_distance_by_hand([0, 0], [3, 4]) - 5.0) < 1e-12

    def test_squared_channel_distance_matches_manual(self):
        rng = np.random.default_rng(2)
        t = rng.normal(size=(4, 5, 3)).astype(np.float32)
        s = rng.normal(size=(4, 5, 3)).astype(np.float32)
        amap = squared_channel_distance(t, s)
        assert amap.shape == (4, 5)
        manual = ((t.astype(np.float64) - s.astype(np.float64)) ** 2).mean(axis=-1)
        np.testing.assert_allclose(amap, manual, rtol=1e-5)

    def test_anomaly_pixel_stands_out(self):
        """埋一个异常位置，其距离应远大于正常位置。"""
        rng = np.random.default_rng(3)
        t = rng.normal(size=(8, 8, 4)).astype(np.float32)
        s = t + rng.normal(0, 0.02, size=(8, 8, 4)).astype(np.float32)
        s[5, 5] += 4.0
        amap = squared_channel_distance(t, s)
        assert amap[5, 5] > 10 * np.delete(amap, 5 * 8 + 5).mean()


class TestMeanStd:
    def test_matches_numpy_builtin(self):
        rng = np.random.default_rng(4)
        x = rng.normal(50, 7, size=5000)
        m, s = mean_std_by_sum_sq(x)
        assert abs(m - x.mean()) < 1e-9
        assert abs(s - x.std()) < 1e-6      # 浮点误差容限

    def test_near_constant_array_std_is_zero(self):
        """全同值数组：E[x^2]-E[x]^2 理论为 0，浮点算出 ~1e-7 量级的残渣，
        max(var, 0) 钳制保证不会因负方差报 NaN。"""
        m, s = mean_std_by_sum_sq(np.full(100, 3.14))
        assert s < 1e-6


class TestKthSmallest:
    """仓库 kthvalue 口径：k = clamp(int(n*q), 1, n)，第 k 小（1 起数）。

    独立参考实现：np.partition（quickselect，与排序不同算法）。
    """

    @staticmethod
    def _reference(x, q):
        n = x.size
        k = min(max(int(n * q), 1), n)
        return float(np.partition(np.asarray(x, np.float64).ravel(), k - 1)[k - 1])

    def test_matches_partition_reference(self):
        rng = np.random.default_rng(5)
        for q in (0.1, 0.5, 0.9, 0.995, 0.999):
            x = rng.normal(size=1234)
            assert abs(kth_smallest(x, q) - self._reference(x, q)) < 1e-12

    def test_known_small_array(self):
        x = np.arange(10)                     # 0..9
        assert kth_smallest(x, 0.9) == 8.0    # k = int(10*0.9) = 9 -> 第 9 小 = 8
        assert kth_smallest(x, 0.5) == 4.0    # k = 5 -> 第 5 小 = 4

    def test_clamping(self):
        x = np.arange(10)
        assert kth_smallest(x, 0.0) == 0.0        # k=0 钳到 1 -> 最小值
        assert kth_smallest(x, 2.0) == 9.0        # k 超界钳到 n -> 最大值

    def test_map_quantiles_order(self):
        rng = np.random.default_rng(6)
        amap = rng.normal(0, 1, size=(32, 32))
        q_start, q_end = map_quantiles(amap)
        assert q_start < q_end                    # 恒保证 q_end > q_start
        assert abs(q_start - self._reference(amap, 0.9)) < 1e-12
        assert abs(q_end - self._reference(amap, 0.995)) < 1e-12

    def test_hard_mining_threshold(self):
        rng = np.random.default_rng(7)
        d = rng.uniform(0, 1, size=1000)
        thr = hard_mining_quantile(d, q=0.999)
        n_above = int((d >= thr).sum())
        assert 1 <= n_above <= 2                  # 前 0.1% -> 约 1 个像素入选


class TestAuc:
    def test_perfect_separation(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert abs(auc_by_hand(labels, scores) - 1.0) < 1e-12

    def test_reversed_scores(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        assert abs(auc_by_hand(labels, scores) - 0.0) < 1e-12

    def test_hand_computable_case(self):
        # pos=[0.5], neg=[0.4, 0.6]: 赢 1 输 1 -> AUC = 0.5
        labels = np.array([0, 0, 1])
        scores = np.array([0.4, 0.6, 0.5])
        assert abs(auc_by_hand(labels, scores) - 0.5) < 1e-12

    def test_simulated_scores_reasonable_auc(self):
        """模拟的 OK/NG 分布，手写 AUC 应与 sklearn 一致（若可用）。"""
        ok, ng = simulate_anomaly_scores()
        labels = np.concatenate([np.zeros(len(ok)), np.ones(len(ng))])
        scores = np.concatenate([ok, ng])
        auc = auc_by_hand(labels, scores)
        assert 0.9 < auc <= 1.0                   # 两带基本分离
        try:
            from sklearn.metrics import roc_auc_score
            assert abs(auc - roc_auc_score(labels, scores)) < 1e-9
        except ImportError:
            pass                                  # 未装 sklearn 时跳过对照
