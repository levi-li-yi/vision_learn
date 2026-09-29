"""registration 的测试（本阶段核心）：ORB/相位相关各自的能力边界 + 混合策略 + 蒙版搬运。

所有场景都用已知真值变换的合成图，误差全部可量化。
"""

import numpy as np

from stage1.registration import (
    estimate_orb_similarity, estimate_phase_translation, hybrid_register,
    make_part_template, make_scene, make_smooth_pair, point_map_error, warp_mask
)

W, H = 240, 180


class TestOrbSimilarity:
    def test_recovers_rotation_translation_scale(self):
        template = make_part_template(W, H, seed=0)
        scene, M_gt = make_scene(template, angle_deg=8.0, tx=10.0, ty=-6.0,
                                 scale=1.05, noise=2.0, seed=1)
        M, meta = estimate_orb_similarity(template, scene)
        assert M is not None, f"ORB 不应失败: {meta}"
        assert meta["inlier_ratio"] >= 0.3
        assert point_map_error(M, M_gt, (W, H)) < 1.5          # 亚像素~1px 级

    def test_fails_on_smooth_gradient(self):
        """无角点 -> ORB 无特征可匹配（失效区之一的直接复现）。"""
        template, scene = make_smooth_pair(W, H, tx=8.0, ty=6.0)
        M, meta = estimate_orb_similarity(template, scene)
        assert M is None
        assert meta["reason"] == "keypoints"


class TestPhaseCorrelation:
    def test_recovers_pure_translation_subpixel(self):
        template, scene = make_smooth_pair(W, H, tx=8.0, ty=6.0)
        M, meta = estimate_phase_translation(template, scene)
        dx, dy = meta["shift"]
        assert abs(dx - 8.0) < 0.5
        assert abs(dy - 6.0) < 0.5
        assert meta["response"] >= 0.35

    def test_unreliable_on_rotation(self):
        """旋转破坏平移定理：相位相关要么响应低、要么估出的平移误差巨大。"""
        template = make_part_template(W, H, seed=0)
        scene, M_gt = make_scene(template, angle_deg=8.0, tx=10.0, ty=-6.0,
                                 scale=1.0, noise=2.0, seed=1)
        M, meta = estimate_phase_translation(template, scene)
        err = point_map_error(M, M_gt, (W, H))
        assert meta["response"] < 0.35 or err > 20             # 两种失效表现至少其一


class TestHybrid:
    def test_picks_orb_for_rotation_scene(self):
        template = make_part_template(W, H, seed=0)
        scene, M_gt = make_scene(template, angle_deg=8.0, tx=10.0, ty=-6.0,
                                 scale=1.05, noise=2.0, seed=1)
        method, M, _ = hybrid_register(template, scene)
        assert method == "orb"
        assert point_map_error(M, M_gt, (W, H)) < 1.5

    def test_falls_back_to_phase_on_smooth(self):
        template, scene = make_smooth_pair(W, H, tx=8.0, ty=6.0)
        method, M, _ = hybrid_register(template, scene)
        assert method == "phase"
        assert abs(M[0, 2] - 8.0) < 0.5 and abs(M[1, 2] - 6.0) < 0.5

    def test_degrades_to_none_when_both_fail(self):
        """两张不同的纯色图：无特征（ORB 无点）、无结构（相位相关无响应）。
        对应仓库'配准失败退化原始 region'的触发条件。"""
        template = np.full((H, W), 100, np.uint8)
        scene = np.full((H, W), 150, np.uint8)
        method, M, _ = hybrid_register(template, scene)
        assert method == "none"
        assert M is None

    def test_rejects_absurd_shift(self):
        """max_shift 门控：估出越界位移的病态解必须被拒（align_max_shift 语义）。"""
        template, scene = make_smooth_pair(W, H, tx=8.0, ty=6.0)
        method, M, _ = hybrid_register(template, scene, max_shift=4)
        assert method == "none"                                # 真位移 8 > 上界 4


class TestWarpMask:
    """通关标准：把标件坐标系的蒙版搬到现场坐标系。"""

    def test_moves_square_exactly(self):
        region = np.zeros((H, W), np.uint8)
        region[40:120, 60:180] = 255
        moved = warp_mask(region, np.float32([[1, 0, 10], [0, 1, 20]]), (W, H))
        ys, xs = np.nonzero(moved)
        assert ys.min() == 60 and ys.max() == 139              # +20
        assert xs.min() == 70 and xs.max() == 189              # +10
        assert int((moved > 0).sum()) == 80 * 120              # 面积不变
        assert set(np.unique(moved)) <= {0, 255}               # NEAREST 保二值

    def test_rotation_with_content_preserved(self):
        region = np.zeros((H, W), np.uint8)
        region[70:110, 100:140] = 255                          # 居中 40x40
        M = np.float32([[np.cos(np.pi / 6), -np.sin(np.pi / 6), 0],
                        [np.sin(np.pi / 6), np.cos(np.pi / 6), 0]])
        moved = warp_mask(region, M, (W, H))
        assert 0.9 * 1600 < int((moved > 0).sum()) <= 1600     # 旋转 30° 面积基本守恒

    def test_end_to_end_register_then_warp(self):
        """端到端通关：配准估出 M -> 蒙版搬过去 -> 与真值 warp 的蒙版 IoU 高。"""
        template = make_part_template(W, H, seed=0)
        scene, M_gt = make_scene(template, angle_deg=5.0, tx=8.0, ty=5.0,
                                 scale=1.0, noise=2.0, seed=2)
        _, M_est, _ = hybrid_register(template, scene)

        region = np.zeros((H, W), np.uint8)
        region[30:150, 50:190] = 255
        gt_warp = warp_mask(region, M_gt, (W, H))
        est_warp = warp_mask(region, M_est, (W, H))
        inter = int(((gt_warp > 0) & (est_warp > 0)).sum())
        union = int(((gt_warp > 0) | (est_warp > 0)).sum())
        assert inter / union > 0.95                            # IoU > 0.95
