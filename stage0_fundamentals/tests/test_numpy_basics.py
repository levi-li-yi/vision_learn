"""numpy_basics 的测试：创建/切片/归约/归一化/dtype 溢出/视图拷贝。"""

import numpy as np

from stage0.numpy_basics import (
    channel_means, crop_patch, make_gradient_image, normalize, safe_brightness
)


class TestCreation:
    def test_gradient_shape_dtype_range(self):
        img = make_gradient_image(4, 8)
        assert img.shape == (4, 8)
        assert img.dtype == np.uint8
        assert 0 <= img.min() and img.max() <= 255

    def test_gradient_row_is_monotonic(self):
        img = make_gradient_image(2, 16)
        assert (np.diff(img[0].astype(int)) >= 0).all()


class TestCropPatch:
    """验证坐标系：image[y0 : y0+size, x0 : x0+size]。"""

    def test_crop_matches_manual_indexing(self):
        img = np.arange(30).reshape(5, 6)
        patch = crop_patch(img, x0=2, y0=1, patch_size=3)
        assert patch.shape == (3, 3)
        assert patch.tolist() == [[img[1, 2], img[1, 3], img[1, 4]],
                                  [img[2, 2], img[2, 3], img[2, 4]],
                                  [img[3, 2], img[3, 3], img[3, 4]]]

    def test_crop_at_origin(self):
        img = np.arange(16).reshape(4, 4)
        assert crop_patch(img, 0, 0, 2).tolist() == [[0, 1], [4, 5]]

    def test_rejects_3d(self):
        with __import__("pytest").raises(ValueError):
            crop_patch(np.zeros((4, 4, 3)), 0, 0, 2)


class TestReductions:
    def test_channel_means(self):
        rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = 10, 20, 30
        np.testing.assert_allclose(channel_means(rgb), [10.0, 20.0, 30.0])


class TestNormalize:
    def test_matches_manual_formula(self):
        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        out = normalize(rgb, mean, std)
        manual = (rgb.astype(np.float32) / 255.0
                  - np.asarray(mean, np.float32).reshape(1, 1, 3)) \
                 / np.asarray(std, np.float32).reshape(1, 1, 3)
        np.testing.assert_allclose(out, manual, rtol=1e-6)

    def test_broadcasting_per_channel(self):
        """(H,W,3) 与 (3,) 相减，每个通道用各自的 mean——广播语义。"""
        rgb = np.full((2, 2, 3), 255, dtype=np.uint8)      # 全白
        mean = [1.0, 0.5, 0.0]                             # 故意不对称
        out = normalize(rgb, mean, std=[1.0, 1.0, 1.0])
        np.testing.assert_allclose(out[0, 0], [0.0, 0.5, 1.0])


class TestDtypeTrap:
    def test_safe_brightness_saturates(self):
        patch = np.array([250, 100, 0], dtype=np.uint8)
        out = safe_brightness(patch, 1.15)
        assert out[0] == 255        # 287.5 正确饱和到 255，而不是回绕成 31

    def test_overflow_documented(self):
        """把陷阱钉在测试里：错误写法确实会回绕（287.5 -> 31）。"""
        patch = np.array([250], dtype=np.uint8)
        wrong = (patch * 1.15).astype(np.uint8)
        assert wrong[0] == 31

    def test_no_darkening_below_range(self):
        patch = np.array([10], dtype=np.uint8)
        assert safe_brightness(patch, 0.2)[0] == 2


class TestViewVsCopy:
    def test_slice_is_view(self):
        from stage0.numpy_basics import demo_view_vs_copy
        a = demo_view_vs_copy()
        assert a[0, 0] == 999        # 函数内部已断言 view 跟变 / copy 不变
