"""connected_components 的测试：stats 精确性 / 面积过滤 / 外轮廓对照。"""

import cv2
import numpy as np

from stage1.connected_components import (
    components_report, draw_boxes, external_contours, filter_by_area,
    label_components
)


def _scene():
    binary = np.zeros((100, 120), np.uint8)
    cv2.rectangle(binary, (10, 10), (49, 49), 255, -1)     # 40x40 方块
    cv2.circle(binary, (90, 30), 15, 255, -1)              # r=15 圆
    cv2.rectangle(binary, (30, 70), (89, 89), 255, -1)     # 60x20 长条
    for y, x in [(55, 55), (60, 100), (5, 110)]:           # 三个孤立噪点
        binary[y, x] = 255
    return binary


class TestLabelComponents:
    def test_stats_exact_for_square(self):
        num, labels, stats, cents = label_components(_scene())
        # 编号按光栅扫描顺序（谁先被扫到谁是小号），别假设 stats[1] 是谁——
        # 按面积定位最大域（1600 的方块）再逐项校验
        sq_id = int(np.argmax(stats[1:, 4])) + 1
        assert stats[sq_id].tolist() == [10, 10, 40, 40, 1600]
        assert cents[sq_id][0] == 29.5 and cents[sq_id][1] == 29.5   # 质心在几何中心

    def test_label_count(self):
        num, _, stats, _ = label_components(_scene())
        # 3 大块 + 3 个互不相邻的噪点 = 6 个目标 + 1 个背景
        assert num == 7

    def test_labels_partition_pixels(self):
        _, labels, _, _ = label_components(_scene())
        binary = _scene()
        assert (labels > 0).sum() == int((binary > 0).sum())  # 每个白像素都有归属

    def test_report_sorted_by_area_with_filter(self):
        comps = components_report(_scene(), min_area=50)
        areas = [c["area"] for c in comps]
        assert areas == sorted(areas, reverse=True)
        assert len(comps) == 3                                # 三个大块
        assert areas[0] == 1600


class TestFilterByArea:
    def test_removes_small_keeps_large(self):
        binary = _scene()
        filtered = filter_by_area(binary, min_area=100)
        assert len(components_report(filtered)) == 3
        # 期望值用"总白像素 - 3 个噪点"从输入侧计算（圆的像素数不等于 pi*r^2）
        assert int((filtered > 0).sum()) == int((binary > 0).sum()) - 3

    def test_threshold_boundary_inclusive(self):
        """area == min_area 的域被保留（>= 语义，对应 eval min_area 口径）。"""
        binary = np.zeros((20, 20), np.uint8)
        binary[5:10, 5:15] = 255                              # 恰好 50 px
        assert int((filter_by_area(binary, min_area=50) > 0).sum()) == 50


class TestContours:
    def test_bbox_matches_components(self):
        binary = _scene()
        cc = components_report(binary)
        ct = external_contours(binary)
        assert len(cc) == len(ct) == 6                         # 6 个目标域
        # 每个连通域 bbox 与外轮廓 bbox 一致（两种统计口径互相印证）
        bboxes_cc = sorted((c["left"], c["top"], c["w"], c["h"]) for c in cc)
        bboxes_ct = sorted(c["bbox"] for c in ct)
        assert bboxes_cc == bboxes_ct

    def test_contour_area_leq_pixel_area(self):
        """折线面积 <= 像素面积（轮廓走像素中心，4x4 方块折线面积是 9）。"""
        binary = np.zeros((20, 20), np.uint8)
        binary[5:9, 5:9] = 255                                 # 4x4 = 16 px
        c = external_contours(binary)[0]
        assert c["area"] == 9.0
        assert c["area"] < 16

    def test_draw_boxes_smoke(self):
        img = np.full((100, 120, 3), 200, np.uint8)
        out = draw_boxes(img, components_report(_scene()))
        assert out.shape == img.shape
        assert (out != 200).any()                              # 确实画了框
