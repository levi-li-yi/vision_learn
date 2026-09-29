"""blending 的测试：羽化 alpha / Alpha 融合 / 贴纸变换 / Cut-Paste GT 收敛。"""

import numpy as np
import pytest

from stage1.blending import (
    alpha_blend, cut_paste, feather_mask, make_sticker, transform_sticker
)


class TestFeatherMask:
    def test_alpha_range_and_extremes(self):
        _, mask = make_sticker(64, seed=0)
        alpha = feather_mask(mask, ksize=5)
        assert alpha.min() >= 0.0 and alpha.max() <= 1.0
        assert abs(alpha[32, 32] - 1.0) < 1e-6                 # 核心区 = 1
        assert alpha[2, 2] == 0.0                              # 远离处 = 0

    def test_even_kernel_rejected(self):
        with pytest.raises(ValueError):
            feather_mask(np.zeros((10, 10), np.uint8), ksize=4)

    def test_larger_kernel_wider_feather(self):
        """羽化核越大，过渡带越宽（半和指标：alpha 在 (0,1) 开区间的像素更多）。"""
        _, mask = make_sticker(64, seed=0)
        a_small = feather_mask(mask, ksize=3)
        a_big = feather_mask(mask, ksize=15)
        band_small = int(((a_small > 0) & (a_small < 1)).sum())
        band_big = int(((a_big > 0) & (a_big < 1)).sum())
        assert band_big > band_small


class TestAlphaBlend:
    def _setup(self):
        _, mask = make_sticker(64, seed=1)
        alpha = feather_mask(mask, ksize=5)
        sticker = np.zeros((64, 64, 3), np.uint8)
        sticker[:] = (50, 100, 150)
        bg = np.full((120, 160, 3), 200, np.uint8)
        return bg, sticker, alpha

    def test_core_takes_sticker_far_untouched(self):
        bg, sticker, alpha = self._setup()
        out = alpha_blend(bg, sticker, alpha, 20, 20)
        cy, cx = 20 + 32, 20 + 32                              # 贴纸中心
        assert out[cy, cx].tolist() == [50, 100, 150]          # 核心区 = 贴纸原色
        assert out[5, 5].tolist() == [200, 200, 200]           # 远处 = 背景原值

    def test_background_not_mutated_in_place(self):
        bg, sticker, alpha = self._setup()
        original = bg.copy()
        _ = alpha_blend(bg, sticker, alpha, 20, 20)
        assert np.array_equal(bg, original)                    # 无副作用

    def test_out_of_bounds_raises(self):
        bg, sticker, alpha = self._setup()
        with pytest.raises(ValueError):
            alpha_blend(bg, sticker, alpha, 120, 60)           # x+w 越界


class TestTransformSticker:
    def test_brightness_saturates_not_wraps(self):
        """stage0 的 uint8 溢出红线在这里复测：250*1.15 必须饱和到 255。"""
        sticker = np.full((16, 16), 250, np.uint8)
        mask = np.full((16, 16), 255, np.uint8)
        st, _ = transform_sticker(sticker, mask, brightness=1.15)
        assert st.min() == 255

    def test_flip_mirrors(self):
        sticker = np.arange(16).reshape(4, 4).astype(np.uint8) * 10
        mask = np.full((4, 4), 255, np.uint8)
        st, _ = transform_sticker(sticker, mask, flip=True)
        assert np.array_equal(st, sticker[:, ::-1])

    def test_identity_transform_noop(self):
        sticker = np.arange(64).reshape(8, 8).astype(np.uint8)
        mask = np.full((8, 8), 255, np.uint8)
        st, mk = transform_sticker(sticker, mask)
        assert np.array_equal(st, sticker)
        assert np.array_equal(mk, mask)


class TestCutPaste:
    def test_gt_subset_of_allowed(self):
        """防御性 GT 收敛：GT 永远不出 allowed（仓库 gt &= allowed_binary 同口径）。"""
        sticker, mask = make_sticker(64, seed=2)
        bg = np.full((150, 200, 3), 180, np.uint8)
        allowed = np.zeros((150, 200), np.uint8)
        allowed[20:100, 100:200] = 255
        _, gt = cut_paste(bg, sticker, mask, x=70, y=30, angle=30, allowed=allowed)
        assert int(((gt > 0) & (allowed == 0)).sum()) == 0     # 零泄漏

    def test_gt_without_allowed_keeps_full_shape(self):
        sticker, mask = make_sticker(64, seed=2)
        bg = np.full((150, 200, 3), 180, np.uint8)
        _, gt = cut_paste(bg, sticker, mask, x=70, y=30, angle=30, allowed=None)
        assert int((gt > 0).sum()) > 500                       # 贴纸形状基本保留

    def test_background_far_away_unchanged(self):
        sticker, mask = make_sticker(64, seed=3)
        bg = np.full((150, 200, 3), 180, np.uint8)
        img, _ = cut_paste(bg, sticker, mask, x=100, y=60)
        assert img[0:40, :].max() == 180                       # 顶部一条完全没动
        assert (img != 180).any()                              # 贴纸区域确实变了

    def test_reproducible_with_same_inputs(self):
        sticker, mask = make_sticker(64, seed=4)
        bg = np.full((150, 200, 3), 180, np.uint8)
        kwargs = dict(x=90, y=50, angle=45, brightness=0.9, flip=True)
        img1, gt1 = cut_paste(bg, sticker, mask, **kwargs)
        img2, gt2 = cut_paste(bg, sticker, mask, **kwargs)
        assert np.array_equal(img1, img2) and np.array_equal(gt1, gt2)
