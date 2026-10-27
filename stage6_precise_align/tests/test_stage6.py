"""stage6 全模块测试：角点网格 / 粗尺度单应 / 位姿门控 / 法向吸附 / 圆滑架桥 / 缺陷实例。"""
import json
import os

import cv2
import numpy as np
import pytest

from stage6.corner_grid import (
    align_coarse, harris_points, pose_gate, quarantine_failed,
    random_homography, solve_homography, spatial_select, swap_rc_xy,
    make_textured,
)
from stage6.defect_instance import (
    export_real_patch, largest_intersection_zone, make_two_zone_scene,
    refine_core, ring_contrast, solve_zd_specs, zone_background_median,
)
from stage6.normal_snap import (
    make_part_and_zone, redraw_hierarchy, sample_profiles, snap_zone_train,
)
from stage6.smooth_bridge import (
    bridge_shifts, make_notched_part, smooth_keep_curvature, snap_zone_infer,
)


def iou(a, b):
    a, b = np.asarray(a) > 127, np.asarray(b) > 127
    return (a & b).sum() / max(1, (a | b).sum())


class TestSpatialSelect:
    def test_grid_coverage_and_cap(self):
        # 36 格每格 100 个点 → 每格最多保留 30 个，且 36 格全覆盖
        h = w = 600
        rows, cols = [], []
        for gy in range(6):
            for gx in range(6):
                for k in range(100):
                    rows.append(gy * 100 + 50)
                    cols.append(gx * 100 + 50)
        sr, sc = spatial_select(np.array(rows), np.array(cols), (h, w))
        assert len(sr) == 36 * 30
        # 每格恰好 30 个（同格点位相同，验证按格截断）
        cells = {(r // 100, c // 100) for r, c in zip(sr, sc)}
        assert len(cells) == 36

    def test_strongest_first_per_cell(self):
        # 每格给响应序在前(强)的 2 个与在后的 98 个 → 只保留前 2 个
        rows, cols = np.array([10] * 100), np.array([10] * 100)
        sr, sc = spatial_select(rows, cols, (600, 600), per_cell=2)
        assert len(sr) == 2 and len(sc) == 2

    def test_boundary_points_clipped(self):
        # 行/列落在图像边缘的点必须被 clip 进最后一格而不是越界
        sr, sc = spatial_select(np.array([599, 599]), np.array([599, 599]), (600, 600))
        assert len(sr) == 2


class TestSwapRCXY:
    def test_involution(self):
        H = np.array([[1.0, 0.01, 5], [-0.02, 0.99, -3], [1e-6, -2e-6, 1.0]])
        H_xy = swap_rc_xy(H)
        H_back = swap_rc_xy(H_xy)
        assert np.allclose(H / H[2, 2], H_back, atol=1e-9)

    def test_point_transform_equivalence(self):
        # H_rc 作用在 (row,col) 点上 == H_xy 作用在 (x,y) 点上
        H_rc = np.array([[1.0, 0.02, 4], [-0.01, 1.0, -6], [0.0, 0.0, 1.0]])
        H_xy = swap_rc_xy(H_rc)
        pts_rc = np.float32([[[3, 7]], [[10, 20]]])          # (row, col)
        pts_xy = np.float32([[[7, 3]], [[20, 10]]])          # (x, y) = (col, row)
        out_rc = cv2.perspectiveTransform(pts_rc, H_rc).reshape(-1, 2)
        out_xy = cv2.perspectiveTransform(pts_xy, H_xy).reshape(-1, 2)
        # 输出同一物理点，只是坐标序不同：(row,col) 反转即 (x,y)
        assert np.allclose(out_rc[:, ::-1], out_xy, atol=1e-6)


class TestAlignCoarse:
    def test_recovers_ground_truth(self):
        ref = make_textured(seed=0, h=768, w=1024)
        rng = np.random.default_rng(7)
        H_gt = random_homography(rng)
        qry = cv2.warpPerspective(ref, H_gt, (1024, 768),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        res = align_coarse(ref, qry)
        assert res["ok"], res["reason"]
        assert res["inliers"] >= 50
        corners = np.float32([[0, 0], [1024, 0], [0, 768], [1024, 768]]).reshape(-1, 1, 2)
        gt = cv2.perspectiveTransform(corners, H_gt).reshape(-1, 2)
        est = cv2.perspectiveTransform(corners, res["H"]).reshape(-1, 2)
        assert float(np.abs(gt - est).max()) < 2.0          # 金标准：四角复现 < 2px

    def test_coarse_restore_scale(self):
        # 粗尺度路径必须正确还原到原分辨率：H ≈ H_gt（而不仅是小图上的 Hs）
        ref = make_textured(seed=1, h=768, w=1024)
        H_gt = np.array([[1.0, 0, 25.0], [0, 1.0, -18.0], [0, 0, 1.0]])
        qry = cv2.warpPerspective(ref, H_gt, (1024, 768), borderMode=cv2.BORDER_REPLICATE)
        res = align_coarse(ref, qry)
        assert res["ok"] and float(np.abs(res["H"] - H_gt).max()) < 1.0

    def test_unrelated_image_fails(self):
        ref = make_textured(seed=0)
        other = make_textured(seed=99)
        res = align_coarse(ref, other)
        assert not res["ok"] and res["reason"]

    def test_size_mismatch_fails(self):
        ref = make_textured(seed=0)
        res = align_coarse(ref, ref[:100, :100])
        assert not res["ok"]

    def test_solve_homography_needs_four(self):
        assert solve_homography([]) == (None, 0)


class TestPoseGate:
    def test_rejects_bad_scale(self):
        H = np.diag([2.5, 2.5, 1.0])                        # det=6.25 超出 (0.5, 2.0)
        ok, shift, det = pose_gate(H, (768, 1024))
        assert not ok and det > 2.0

    def test_rejects_big_shift(self):
        H = np.array([[1.0, 0, 400.0], [0, 1.0, 300.0], [0, 0, 1.0]])
        ok, shift, det = pose_gate(H, (768, 1024))
        assert not ok and shift > 300

    def test_accepts_normal_pose(self):
        H = np.array([[1.0, 0.01, 15.0], [-0.01, 1.0, 10.0], [0, 0, 1.0]])
        ok, _, _ = pose_gate(H, (768, 1024))
        assert ok


class TestQuarantine:
    def test_moves_files_and_writes_jsonl(self, tmp_path):
        src_dir = tmp_path / "pointA"
        src_dir.mkdir()
        bmp = src_dir / "img_001.bmp"
        bmp.write_bytes(b"fake")
        (src_dir / "img_001.bmp.tmp").write_bytes(b"fake-ann")
        fail_root = str(tmp_path / "align_failed")

        lines = quarantine_failed(
            [{"point": "pointA", "stem": "img_001", "bmp_path": str(bmp),
              "reason": "内点不足 (12 < 50)", "inliers": 12, "is_val": True}],
            fail_root)
        assert len(lines) == 1
        dst = tmp_path / "align_failed" / "pointA" / "val" / "img_001.bmp"
        assert dst.is_file() and not bmp.exists()
        assert (tmp_path / "align_failed" / "pointA" / "val" / "img_001.bmp.tmp").is_file()
        jsonl = tmp_path / "align_failed" / "moved.jsonl"
        rec = json.loads(jsonl.read_text(encoding="utf-8").strip())
        assert rec["stem"] == "img_001" and rec["reason"].startswith("内点不足")
        assert rec["dst_bmp"] == str(dst)


class TestNormalSnap:
    def test_snap_improves_iou_and_keeps_clean(self):
        gray, part, proj, _ = make_part_and_zone()
        snapped = snap_zone_train(proj, gray)
        assert iou(snapped, part) > iou(proj, part)
        assert iou(snapped, part) > 0.90
        # 训练模式肩部内收：吸附后的区几乎全部落在零件上（区内干净）
        zone, part_b = snapped > 127, part > 127
        clean = (zone & part_b).sum() / zone.sum()
        assert clean > 0.97

    def test_profiles_shape(self):
        gray, _, proj, _ = make_part_and_zone()
        cnts, _ = cv2.findContours(proj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = cnts[0][:, 0, :].astype(np.float32)
        from stage6.normal_snap import contour_normals, orient_normals
        _, _, nx, ny = contour_normals(pts)
        nx, ny = orient_normals(proj, pts, nx, ny)
        prof, ts = sample_profiles(gray, pts, nx, ny, snap_px=10)
        assert prof.shape == (len(pts), 23) and len(ts) == 23

    def test_hole_preserved_in_redraw(self):
        # 带孔分区（圆环）经层级重绘后孔必须保留
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 80, 255, -1)
        cv2.circle(mask, (100, 100), 40, 0, -1)
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        out = redraw_hierarchy(cnts, hier, (200, 200))
        assert out[100, 100] == 0 and out[100, 130] == 0    # 孔内部保持 0 (r<40)
        assert out[100, 170] == 255 and out[100, 150] == 255  # 环带保持 255 (40<r<80)

    def test_short_contour_untouched(self):
        # 点数 < 24 的碎片轮廓不吸附，原样返回
        gray = np.full((120, 160), 100, dtype=np.uint8)
        proj = np.zeros((120, 160), dtype=np.uint8)
        cv2.circle(proj, (80, 60), 8, 255, -1)              # 小圆轮廓 ~40 点 > 24？半径 8 → ~45 点
        cv2.circle(proj, (130, 30), 3, 255, -1)             # 半径 3 → ~18 点 < 24
        out = snap_zone_train(proj, gray)
        assert (out[30, 130] > 0)                           # 小轮廓仍在


class TestSmoothBridge:
    def test_notch_covered_by_infer_zone(self):
        gray, notch, proj = make_notched_part()
        zone_infer = snap_zone_infer(proj, gray)
        zone_train = snap_zone_train(proj, gray)
        cover_i = ((zone_infer > 127) & (notch > 0)).sum() / (notch > 0).sum()
        cover_t = ((zone_train > 127) & (notch > 0)).sum() / (notch > 0).sum()
        assert cover_i > cover_t                            # 推理区比训练区包得多

        def boundary_radius_along(mask, center, angle_deg):
            a = np.deg2rad(angle_deg)
            h, w = mask.shape
            rmax = 0
            for r in range(1, max(h, w)):
                x = int(round(center[0] + r * np.cos(a)))
                y = int(round(center[1] + r * np.sin(a)))
                if not (0 <= x < w and 0 <= y < h) or mask[y, x] == 0:
                    break
                rmax = r
            return rmax

        center = (proj.shape[1] // 2, proj.shape[0] // 2)
        r_inf_notch = boundary_radius_along(zone_infer, center, 0)     # 缺口方向
        r_inf_back = boundary_radius_along(zone_infer, center, 180)    # 正常对边
        r_trn_notch = boundary_radius_along(zone_train, center, 0)
        assert abs(r_inf_notch - r_inf_back) <= 4           # 架桥：推理边界在缺口处不下凹
        assert r_inf_notch >= r_trn_notch + 4               # 训练边界在缺口处明显内凹

    def test_smooth_keeps_curvature(self):
        # 锯齿圆轮廓去锯齿后：半径波动显著减小，而面积（曲率/半径）基本不缩
        ang = np.linspace(0, 2 * np.pi, 300, endpoint=False)
        rng = np.random.default_rng(2)
        r0 = 100.0
        pts = np.stack([160 + (r0 + rng.uniform(-4, 4, len(ang))) * np.cos(ang),
                        160 + (r0 + rng.uniform(-4, 4, len(ang))) * np.sin(ang)], axis=1).astype(np.float32)
        smoothed = smooth_keep_curvature(pts)
        rad_before = np.hypot(*(pts - [160, 160]).T)
        rad_after = np.hypot(*(smoothed - [160, 160]).T)
        assert rad_after.std() < rad_before.std() * 0.5     # 锯齿被抹掉
        assert abs(rad_after.mean() - r0) < 1.5             # 圆弧不收缩不外鼓
        assert abs(rad_after.mean() - rad_before.mean()) < 0.5

    def test_bridge_only_lifts_dents(self):
        shifts = np.zeros(200, dtype=np.float32)
        shifts[90:110] = -6.0                               # 局部凹陷（崩边缺口向内）
        lifted = bridge_shifts(shifts)
        assert lifted[90:110].min() > -6.0                  # 凹陷被拉起
        assert float(np.abs(lifted[:80]).max()) == pytest.approx(0.0)  # 正常段不动
        flat = bridge_shifts(np.zeros(200, dtype=np.float32))
        assert float(np.abs(flat).max()) == pytest.approx(0.0)         # 无凹陷时零干预


class TestDefectInstance:
    def test_attribution_by_largest_intersection(self):
        gray, zones, smooth, defects, _ = make_two_zone_scene()
        for d in defects:
            z, inter = largest_intersection_zone(d["pts"], smooth, gray.shape, fallback=zones)
            assert z == d["zone_true"] and inter >= 4

    def test_attribution_rejects_outside(self):
        gray, zones, smooth, _, _ = make_two_zone_scene()
        far = np.array([[160, 5], [170, 5], [170, 15], [160, 15]], np.int32)  # 缝隙里
        z, inter = largest_intersection_zone(far, smooth, gray.shape, fallback=zones)
        assert z is None or inter < 4

    def test_ring_contrast_measures_planted_delta(self):
        gray, zones, smooth, defects, ann = make_two_zone_scene()
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ann_dil = cv2.dilate(ann, k5) > 0
        for d, expect in zip(defects, [90, 90, 45]):        # 60 vs 150 / 45 vs 90
            diff = ring_contrast(gray, d["pts"], smooth[d["zone_true"]], other_ann=ann_dil)
            assert diff is not None and diff > 30
            assert abs(diff - expect) < 12                  # p80 口径接近设计对比度

    def test_ring_contrast_none_for_flat(self):
        gray, zones, smooth, _, _ = make_two_zone_scene()
        # 贴一个对比度 < 6 的"伪缺陷"（亮区上 150 vs 152）
        poly = np.array([[50, 30], [70, 30], [70, 40], [50, 40]], np.int32)
        g2 = gray.copy()
        cv2.fillPoly(g2, [poly], 152)
        assert ring_contrast(g2, poly, smooth["加工面"]) is None

    def test_export_patch_pad_and_background_fill(self):
        gray, zones, smooth, defects, ann = make_two_zone_scene()
        d = defects[0]
        patch, mask = export_real_patch(gray, d["pts"], smooth[d["zone_true"]])
        pts = d["pts"]
        expect_h = int(pts[:, 1].max()) + 1 + 8 - (int(pts[:, 1].min()) - 8)
        expect_w = int(pts[:, 0].max()) + 1 + 8 - (int(pts[:, 0].min()) - 8)
        assert patch.shape == (expect_h, expect_w)          # 外扩 8px
        assert (mask > 0).sum() > 4
        outside = patch[mask == 0]
        assert outside.std() == pytest.approx(0.0)          # 掩码外全部填同一背景中值

    def test_zone_background_median_excludes_ann(self):
        gray, zones, _, defects, ann = make_two_zone_scene()
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ann_dil = cv2.dilate(ann, k5) > 0
        med = zone_background_median(gray, zones["加工面"], ann_dil.astype(np.uint8))
        assert med == pytest.approx(150, abs=2)             # 干净背景，不被缺陷(60)拉偏

    def test_specs_bounds(self):
        gray, zones, smooth, defects, ann = make_two_zone_scene()
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ann_dil = (cv2.dilate(ann, k5) > 0).astype(np.uint8)
        instances = []
        for d in defects:
            _, mask = export_real_patch(gray, d["pts"], smooth[d["zone_true"]],
                                        ann_mask=ann_dil)
            instances.append({"area": int((mask > 0).sum()),
                              "abs_diff": ring_contrast(gray, d["pts"],
                                                        smooth[d["zone_true"]],
                                                        other_ann=ann_dil > 0)})
        specs = solve_zd_specs(instances)
        assert specs["min"] >= 18.0 and specs["max"] >= specs["min"] * 1.5
        assert 16.0 <= specs["diff_lo"] < specs["diff_med"] < specs["diff_hi"] <= 195.0
        assert specs["b_prob"] == 0.50 and specs["num_instances"] == 3

    def test_refine_core_and_aspect(self):
        gray, zones, smooth, defects, _ = make_two_zone_scene()
        scratch, pore = defects[0], defects[2]
        p1, m1 = export_real_patch(gray, scratch["pts"], smooth[scratch["zone_true"]])
        p2, m2 = export_real_patch(gray, pore["pts"], smooth[pore["zone_true"]])
        core1, aspect1 = refine_core(p1, m1)
        core2, aspect2 = refine_core(p2, m2)
        assert ((core1 > 0) & (m1 > 0)).sum() == (core1 > 0).sum()   # 核心被掩码约束
        assert aspect1 > 2.0                                        # 划痕细长
        assert aspect2 < 2.0                                        # 砂眼团状
