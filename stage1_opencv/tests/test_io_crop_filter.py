"""io_crop_filter 的测试：BGR/读写往返/裁剪/手写滤波 vs OpenCV 对拍。"""

import numpy as np
import pytest
import cv2

from stage1.io_crop_filter import (
    bgr_to_rgb, crop_roi, gaussian_kernel_1d, gaussian_kernel_2d,
    manual_gaussian_blur, manual_median_filter, save_and_load, to_gray
)


class TestIo:
    def test_roundtrip_lossless(self, tmp_path):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
        path = tmp_path / "round.png"
        back = save_and_load(img, path)
        assert np.array_equal(img, back)

    def test_bgr_to_rgb_swaps_channels(self):
        bgr = np.zeros((2, 2, 3), np.uint8)
        bgr[..., 0] = 200                                    # B 通道高
        rgb = bgr_to_rgb(bgr)
        assert rgb[..., 0].max() == 0 and rgb[..., 2].max() == 200

    def test_to_gray_uses_weighted_formula(self):
        bgr = np.array([[[30, 120, 60]]], np.uint8)          # B=30 G=120 R=60
        expect = 0.299 * 60 + 0.587 * 120 + 0.114 * 30       # BGR 顺序下 R=60
        assert abs(float(to_gray(bgr)[0, 0]) - round(expect)) <= 1

    def test_crop_roi_coordinates(self):
        img = np.arange(30).reshape(5, 6)
        patch = crop_roi(img, x0=2, y0=1, size=3)
        assert patch.shape == (3, 3)
        assert patch[0, 0] == img[1, 2]                      # 先 y 后 x


class TestGaussianKernel:
    def test_matches_cv2_getGaussianKernel(self):
        for ksize, sigma in [(3, 0.8), (5, 1.2), (7, 2.0)]:
            mine = gaussian_kernel_1d(ksize, sigma)
            cv = cv2.getGaussianKernel(ksize, sigma)
            np.testing.assert_allclose(mine, cv[:, 0], atol=1e-8)

    def test_normalized_and_symmetric(self):
        k = gaussian_kernel_1d(5, 1.0)
        assert abs(k.sum() - 1.0) < 1e-12
        np.testing.assert_allclose(k, k[::-1], atol=1e-12)

    def test_2d_is_outer_product(self):
        k1 = gaussian_kernel_1d(5, 1.0)
        np.testing.assert_allclose(gaussian_kernel_2d(5, 1.0), np.outer(k1, k1))

    def test_even_kernel_rejected(self):
        with pytest.raises(ValueError):
            gaussian_kernel_1d(4, 1.0)


class TestFilters:
    def test_manual_gaussian_matches_cv2_on_float32(self):
        """float32 输入下 cv2 用浮点核，应与手写逐位一致（uint8 是定点快速实现，
        允许 ~1 灰度级差——这个差异本身是教学点）。"""
        rng = np.random.default_rng(1)
        gray = rng.uniform(0, 255, size=(30, 40)).astype(np.float32)
        mine = manual_gaussian_blur(gray, ksize=5, sigma=1.2)
        cv = cv2.GaussianBlur(gray, (5, 5), 1.2)
        np.testing.assert_allclose(mine, cv, atol=1e-3)

    def test_gaussian_smooths_impulse(self):
        gray = np.full((20, 20), 100, np.float32)
        gray[10, 10] = 200.0
        out = manual_gaussian_blur(gray, ksize=5, sigma=1.2)
        assert out[10, 10] < 150                             # 尖峰被摊薄
        assert out[0, 0] == pytest.approx(100, abs=1e-3)     # 远处不动

    def test_median_removes_salt_pepper_exactly(self):
        noisy = np.full((20, 20), 128, np.uint8)
        noisy[3, 4] = 0
        noisy[15, 8] = 255
        noisy[7, 18] = 0
        out = manual_median_filter(noisy, ksize=3).astype(np.uint8)
        assert (out == 128).all()

    def test_median_vs_gaussian_on_salt_pepper(self):
        """中值完胜高斯（去椒盐噪声的教科书结论，量化钉住）。"""
        rng = np.random.default_rng(2)
        clean = rng.integers(100, 156, size=(40, 40)).astype(np.float32)
        noisy = clean.copy()
        mask = rng.random(clean.shape) < 0.05
        noisy[mask] = rng.choice([0.0, 255.0], size=mask.sum())

        med = manual_median_filter(noisy, 3)
        gau = cv2.GaussianBlur(noisy.astype(np.uint8), (3, 3), 0)
        err_med = np.abs(med - clean).mean()
        err_gau = np.abs(gau.astype(np.float32) - clean).mean()
        assert err_med < err_gau
