"""defect_pipeline 的测试（Demo B）：经典流水线检出率 / 面积门限 / region 门控。"""

import numpy as np

from stage1.connected_components import label_components
from stage1.defect_pipeline import (
    classical_detect, hit, make_metal_background, make_region_mask, plant_defects,
    region_gate
)


def _scene(seed=1):
    bg = make_metal_background(200, 150, seed=0)
    region = make_region_mask(bg.shape)
    img, gt, facts = plant_defects(bg, seed=seed)
    return img, gt, facts, region


class TestClassicalDetect:
    def test_detects_all_planted_defects(self):
        img, gt, facts, _ = _scene()
        comps, _ = classical_detect(img, min_area=30)
        n_real = sum(1 for t, _ in facts if t != "stain_outside_region")
        n_hit = sum(1 for c in comps if hit(gt, c))
        assert n_hit == n_real == 6                            # 缺陷全检出

    def test_stain_is_false_positive_without_gate(self):
        img, gt, facts, _ = _scene()
        comps, _ = classical_detect(img, min_area=30)
        n_real = sum(1 for t, _ in facts if t != "stain_outside_region")
        assert len(comps) == n_real + 1                        # 污渍被当缺陷检出
        stain = next(f for t, f in facts if t == "stain_outside_region")
        stain_comps = [c for c in comps
                       if abs(c["cx"] - stain[0]) < 10 and abs(c["cy"] - stain[1]) < 10]
        assert len(stain_comps) == 1                           # 那个假阳确实是污渍

    def test_min_area_filters_noise_dots(self):
        img, gt, facts, _ = _scene()
        comps, _ = classical_detect(img, min_area=30)
        assert all(c["area"] >= 30 for c in comps)             # r=1 噪点全被剔除

    def test_clean_background_exposes_otsu_limitation(self):
        """规则方法的极限（钉在测试里）：纯背景是单峰直方图，Otsu 没有谷底
        可站，会把阈值切在噪声中间 -> 拉丝纹理整行误检。这正是产线上
        '光照一漂固定阈值就崩'、最终上深度学习的微观原因。"""
        bg = make_metal_background(200, 150, seed=7)
        comps, _ = classical_detect(bg, min_area=30)
        assert len(comps) > 0                                  # 病理行为：有假阳
        assert comps[0]["area"] > 30 * 10                      # 且连成大片


class TestRegionGate:
    def test_gate_suppresses_outside_region(self):
        img, gt, facts, region = _scene()
        comps, closed = classical_detect(img, min_area=30)
        gated = region_gate(closed, region)
        num, labels, stats, cents = label_components(gated)
        kept = [{"area": int(stats[i, 4]), "cx": float(cents[i][0]),
                 "cy": float(cents[i][1])} for i in range(1, num)
                if stats[i, 4] >= 30]
        assert all(hit(gt, c) for c in kept)                   # 门控后零假阳
        assert len(kept) == 6

    def test_gate_keeps_inside_region_untouched(self):
        img, gt, _, region = _scene()
        _, closed = classical_detect(img, min_area=30)
        gated = region_gate(closed, region)
        inside = closed.copy()
        inside[region == 0] = 0                                # 手工做同样的 AND
        assert np.array_equal(gated, inside)

    def test_gate_zeroes_outside_pixels(self):
        mask = np.full((10, 10), 255, np.uint8)
        region = np.zeros((10, 10), np.uint8)
        region[:, 5:] = 255
        gated = region_gate(mask, region)
        assert (gated[:, :5] == 0).all()
        assert (gated[:, 5:] == 255).all()
