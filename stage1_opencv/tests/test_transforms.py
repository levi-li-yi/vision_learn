"""transforms 的测试：平移/旋转/warp 的 dsize 顺序/点位变换/透视。"""

import numpy as np
import pytest

from stage1.transforms import (
    apply_affine, apply_perspective, estimate_perspective, rotation_matrix,
    transform_points, translation_matrix
)


class TestTranslation:
    def test_moves_pixel_exactly(self):
        img = np.zeros((60, 80), np.uint8)
        img[10:20, 30:40] = 200
        moved = apply_affine(img, translation_matrix(10, 20), out_wh=(80, 60))
        assert moved[35, 45] == 200                            # 新位置拿到原值
        assert moved[15, 35] == 0                              # 老位置已空

    def test_dsize_is_width_height_order(self):
        img = np.zeros((60, 100), np.uint8)                    # 高60 宽100
        out = apply_affine(img, translation_matrix(0, 0), out_wh=(100, 60))
        assert out.shape == (60, 100)
        out_wrong = apply_affine(img, translation_matrix(0, 0), out_wh=(60, 100))
        assert out_wrong.shape == (100, 60)                    # 传反了就转置了

    def test_out_of_bounds_raises(self):
        from stage1.blending import alpha_blend
        bg = np.zeros((50, 50, 3), np.uint8)
        st = np.zeros((20, 20, 3), np.uint8)
        alpha = np.zeros((20, 20), np.float32)
        with pytest.raises(ValueError):
            alpha_blend(bg, st, alpha, 40, 40)                 # 40+20 > 50


class TestTransformPoints:
    def test_manual_matrix_multiply(self):
        M = np.float32([[2, 0, 1], [0, 2, 5]])                 # 放大2倍+平移
        pts = transform_points(M, [(3, 4), (0, 0)])
        np.testing.assert_allclose(pts, [[7, 13], [1, 5]])     # (2x+1, 2y+5)

    def test_translation_identity_on_points(self):
        M = translation_matrix(-5, 3)
        pts = transform_points(M, [(10, 10)])
        np.testing.assert_allclose(pts, [[5, 13]])


class TestRotation:
    def test_center_is_fixed_point(self):
        M = rotation_matrix((50, 30), 37.5, scale=1.1)
        out = transform_points(M, [(50, 30)])
        np.testing.assert_allclose(out, [[50, 30]], atol=1e-6)

    def test_90deg_maps_corners(self):
        M = rotation_matrix((50, 30), 90)
        pts = transform_points(M, [(60, 30)])                  # 中心右侧 10px
        np.testing.assert_allclose(pts, [[50, 20]], atol=1e-6)  # 转到上方 10px

    def test_nearest_rotation_preserves_pixel_count(self):
        img = np.zeros((60, 100), np.uint8)
        img[5:55, 45:55] = 255                                 # 500 px 竖条
        rot = apply_affine(img, rotation_matrix((50, 30), 90), out_wh=(100, 60))
        assert int((rot > 0).sum()) == 500


class TestPerspective:
    def test_maps_quad_to_corners(self):
        quad = [(60, 40), (220, 60), (200, 220), (40, 200)]
        corners = [(40, 40), (200, 40), (200, 200), (40, 200)]
        P = estimate_perspective(quad, corners)
        hom = np.concatenate([np.float32(quad), np.ones((4, 1))], axis=1)
        mapped = hom @ P.T                                     # (4, 3) 齐次
        mapped_xy = mapped[:, :2] / mapped[:, 2:]              # 除以 w 归一
        np.testing.assert_allclose(mapped_xy, corners, atol=1e-6)

    def test_needs_exactly_four_points(self):
        with pytest.raises(ValueError):
            estimate_perspective([(0, 0), (1, 1), (2, 2)], [(0, 0), (1, 1), (2, 2)])

    def test_rectifies_quad_content(self):
        board = np.zeros((240, 240), np.uint8)
        quad = [(60, 40), (220, 60), (200, 220), (40, 200)]
        cv2_poly = np.int32(quad)
        import cv2
        cv2.fillPoly(board, [cv2_poly], 128)
        P = estimate_perspective(quad, [(40, 40), (200, 40), (200, 200), (40, 200)])
        out = apply_perspective(board, P, out_wh=(240, 240))
        ys, xs = np.nonzero(out)
        assert 35 <= xs.min() <= 45 and 195 <= xs.max() <= 205
        assert 35 <= ys.min() <= 45 and 195 <= ys.max() <= 205
