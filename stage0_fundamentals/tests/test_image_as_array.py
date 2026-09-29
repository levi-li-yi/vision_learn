"""image_as_array 的测试：RGB 形状/灰度化/手写滤波（对照 sliding_window_view）/去噪/离群检测。"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from stage0.image_as_array import (
    add_impulse_noise, defect_map_by_local_contrast, describe_shape,
    grayscale_mean, grayscale_weighted, make_rgb_image, mean_filter_3x3_by_hand,
    mean_filter_3x3_vectorized
)


class TestRgbAsArray:
    def test_shape_and_dtype(self):
        rgb = make_rgb_image(12, 16)
        assert rgb.shape == (12, 16, 3)
        assert rgb.dtype == np.uint8

    def test_color_bars_in_right_channels(self):
        rgb = make_rgb_image(12, 16)
        red_pixel = rgb[4, 8]        # 横条区
        blue_pixel = rgb[9, 1]       # 竖条区
        assert red_pixel[0] > red_pixel[1]          # R 通道主导 -> 偏红
        assert blue_pixel[2] > blue_pixel[0]        # B 通道主导 -> 偏蓝

    def test_describe_shape_rejects_2d(self):
        import pytest
        with pytest.raises(ValueError):
            describe_shape(np.zeros((4, 4)))


class TestGrayscale:
    def test_mean_grayscale_known_values(self):
        rgb = np.array([[[60, 120, 30]]], dtype=np.uint8)    # (1,1,3)
        assert abs(grayscale_mean(rgb)[0, 0] - 70.0) < 1e-9  # (60+120+30)/3

    def test_weighted_grayscale_matches_formula(self):
        rgb = np.array([[[60, 120, 30]]], dtype=np.uint8)
        expected = 0.299 * 60 + 0.587 * 120 + 0.114 * 30
        assert abs(grayscale_weighted(rgb)[0, 0] - expected) < 1e-4

    def test_output_is_2d(self):
        rgb = make_rgb_image(6, 8)
        assert grayscale_mean(rgb).shape == (6, 8)
        assert grayscale_weighted(rgb).shape == (6, 8)

    def test_weighted_darker_than_mean_on_saturated_red(self):
        """纯红像素：加权灰度(0.299*255=76) 应明显低于平均灰度(85)。"""
        rgb = np.full((1, 1, 3), 255, dtype=np.uint8)
        rgb[..., 1:] = 0
        assert grayscale_weighted(rgb)[0, 0] < grayscale_mean(rgb)[0, 0]


class TestMeanFilter:
    @staticmethod
    def _reference(gray):
        """独立参考实现：滑窗视图 + 窗口均值（与手写循环不同的代码路径）。"""
        padded = np.pad(gray.astype(np.float32), 1, mode="constant")
        return sliding_window_view(padded, (3, 3)).mean(axis=(-2, -1))

    def test_impulse_response(self):
        """全零图中心放 9：输出中心 = 9/9 = 1——卷积核归一性的直接检验。"""
        gray = np.zeros((5, 5), dtype=np.float32)
        gray[2, 2] = 9.0
        out = mean_filter_3x3_by_hand(gray)
        assert abs(out[2, 2] - 1.0) < 1e-6
        assert abs(out[1, 1] - 1.0) < 1e-6       # (1,1) 的窗口恰好擦到中心 (2,2)
        assert abs(out[0, 0] - 0.0) < 1e-6       # (0,0) 的窗口够不到中心 -> 0
        assert abs(out[0, 4] - 0.0) < 1e-6       # 对角角落不受影响

    def test_flat_image_unchanged_in_interior(self):
        """平坦图内部滤波后不变；最外圈受零填充压暗（边界效应单独验证）。"""
        gray = np.full((6, 7), 42.0, dtype=np.float32)
        out = mean_filter_3x3_by_hand(gray)
        np.testing.assert_allclose(out[1:-1, 1:-1], gray[1:-1, 1:-1], atol=1e-6)
        assert out[0, 0] < gray[0, 0]            # 角落窗口 4/9 是 0 -> 被压暗

    def test_matches_sliding_window_reference_random(self):
        rng = np.random.default_rng(11)
        gray = rng.uniform(0, 255, size=(15, 17)).astype(np.float32)
        np.testing.assert_allclose(
            mean_filter_3x3_by_hand(gray), self._reference(gray), atol=1e-4)

    def test_vectorized_matches_hand_written(self):
        rng = np.random.default_rng(12)
        gray = rng.uniform(0, 255, size=(9, 9)).astype(np.float32)
        np.testing.assert_allclose(
            mean_filter_3x3_by_hand(gray),
            mean_filter_3x3_vectorized(gray), atol=1e-4)

    def test_output_shape_preserved(self):
        gray = np.zeros((8, 8), dtype=np.float32)
        assert mean_filter_3x3_by_hand(gray).shape == (8, 8)

    def test_rejects_non_2d(self):
        import pytest
        with pytest.raises(ValueError):
            mean_filter_3x3_by_hand(np.zeros((4, 4, 3)))


class TestDenoise:
    def test_filter_reduces_impulse_noise_error(self):
        """平滑背景 + 椒盐噪声：均值滤波'摊薄'尖峰——最大误差大幅下降。

        实测认知点：均值误差只轻微下降（误差从噪点摊到 3x3 邻域），
        但最大误差被砍掉大半——这就是"稀释"的含义。
        高频随机纹理不适合当干净图（会被连着抹掉）；零填充影响边界，排除一圈。
        """
        clean = np.tile(np.linspace(50, 70, num=14), (12, 1)).astype(np.float32)
        noisy = add_impulse_noise(clean, ratio=0.05, seed=13)
        denoised = mean_filter_3x3_by_hand(noisy)

        inner = (slice(1, -1), slice(1, -1))
        abs_err = np.abs((noisy - clean)[inner]), np.abs((denoised - clean)[inner])
        assert abs_err[1].mean() < abs_err[0].mean()        # 均值误差下降
        assert abs_err[1].max() < 0.5 * abs_err[0].max()    # 最大误差砍半以上

    def test_noise_is_deterministic_with_seed(self):
        clean = np.full((10, 10), 60, dtype=np.float32)
        n1 = add_impulse_noise(clean, 0.1, seed=99)
        n2 = add_impulse_noise(clean, 0.1, seed=99)
        np.testing.assert_array_equal(n1, n2)


class TestDefectMap:
    def test_hot_spots_hit_real_noise(self):
        """局部对比度图的高响应位置应覆盖大部分真实噪声点（内部区域检出率>80%）。"""
        rng = np.random.default_rng(14)
        clean = rng.integers(50, 70, size=(16, 16)).astype(np.float32)
        noisy = add_impulse_noise(clean, ratio=0.04, seed=14)

        dmap = defect_map_by_local_contrast(noisy)
        inner_mask = np.zeros(noisy.shape, dtype=bool)
        inner_mask[1:-1, 1:-1] = True
        noise_pos = {tuple(p) for p in np.argwhere((noisy != clean) & inner_mask)}
        hot_pos = {tuple(p) for p in np.argwhere((dmap > 30) & inner_mask)}

        hit = len(noise_pos & hot_pos)
        assert hit / len(noise_pos) > 0.8, f"检出率过低: {hit}/{len(noise_pos)}"
