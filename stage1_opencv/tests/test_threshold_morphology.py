"""threshold_morphology 的测试：固定阈值/Otsu/腐蚀膨胀对拍/开闭行为。"""

import numpy as np
import pytest

from stage1.threshold_morphology import (
    binarize_fixed, binarize_otsu, dilate, dilate_manual, erode, erode_manual,
    morph_close, morph_open
)


def _square(size=30, offset=10, canvas=50):
    img = np.zeros((canvas, canvas), np.uint8)
    img[offset: offset + size, offset: offset + size] = 255
    return img


class TestThreshold:
    def test_fixed_threshold(self):
        gray = np.array([[10, 100], [200, 250]], np.uint8)
        binary = binarize_fixed(gray, 100)
        assert binary.tolist() == [[0, 0], [255, 255]]        # > 100 才白

    def test_otsu_lands_in_valley(self):
        rng = np.random.default_rng(0)
        img = np.full((80, 100), 120, np.uint8)
        img[10:30, 10:40] = 40                                 # 暗缺陷群
        img[50:70, 60:90] = 45
        img = np.clip(img.astype(np.int16)
                      + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
        thr, binary = binarize_otsu(img)
        assert 60 < thr < 110                                  # 落在两峰谷底
        # 暗区被完整捕获（非反相：亮=前景约定下暗区是 0）
        assert (binary[10:30, 10:40] == 0).mean() > 0.95
        assert (binary[50:70, 60:90] == 0).mean() > 0.95
        assert (binary[35:45, :] == 255).all()                 # 背景为白


class TestErodeDilate:
    def test_manual_erode_matches_cv2_interior(self):
        binary = _square()
        inner = (slice(2, -2), slice(2, -2))
        assert np.array_equal(erode_manual(binary)[inner], erode(binary)[inner])

    def test_manual_dilate_matches_cv2_interior(self):
        binary = _square()
        inner = (slice(2, -2), slice(2, -2))
        assert np.array_equal(dilate_manual(binary)[inner], dilate(binary)[inner])

    def test_erode_shrinks_and_dilate_expands(self):
        binary = _square(size=30)                              # 900 px
        assert int((erode(binary) > 0).sum()) < 900
        assert int((dilate(binary) > 0).sum()) > 900
        # 3x3 矩形核下精确值：腐蚀 28x28、膨胀 32x32
        assert int((erode(binary) > 0).sum()) == 28 * 28
        assert int((dilate(binary) > 0).sum()) == 32 * 32

    def test_dilate_is_erode_dual_on_inverse(self):
        """膨胀(白) == 反相后的腐蚀(黑) 再反相——形态学对偶性的直接验证。"""
        binary = _square(size=20, offset=5)
        lhs = dilate(binary)
        rhs = 255 - erode(255 - binary)
        assert np.array_equal(lhs, rhs)


class TestOpenClose:
    def test_open_removes_single_pixel_noise(self):
        noisy = _square()
        noisy[5, 5] = 255
        noisy[45, 45] = 255
        opened = morph_open(noisy, ksize=3)
        assert int((opened > 0).sum()) == 900                  # 只剩 30x30 大块

    def test_open_restores_square_exactly(self):
        """矩形块 开(矩形核) 后精确复原（腐蚀 28x28 -> 膨胀回 30x30）。"""
        opened = morph_open(_square(), ksize=3)
        assert int((opened > 0).sum()) == 900

    def test_close_fills_small_hole(self):
        holed = _square()
        holed[20:22, 20:22] = 0                               # 2x2 洞
        closed = morph_close(holed, ksize=3)
        assert int((closed > 0).sum()) == 900                  # 洞被填上

    def test_close_cannot_fill_big_hole_with_small_kernel(self):
        """结构元必须比洞大：5x5 洞用 3x3 核填不上（口径认知，钉住）。"""
        holed = _square()
        holed[20:25, 20:25] = 0                               # 5x5 洞
        closed = morph_close(holed, ksize=3)
        assert int((closed > 0).sum()) == 900 - 25             # 洞还在

    def test_close_fills_big_hole_with_big_kernel(self):
        holed = _square()
        holed[20:25, 20:25] = 0
        closed = morph_close(holed, ksize=7)
        assert int((closed > 0).sum()) == 900                  # 7x7 核填得上
