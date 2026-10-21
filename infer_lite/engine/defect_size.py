from __future__ import annotations

from typing import Any, List

import halcon as ha

from common.logger import setup_logger

logger = setup_logger("defect_size")


def calculate_defect_size(polygon_points: List[List[int]], camera_param: Any, pose: Any):
    """根据相机内外参，将像素多边形转换为世界坐标系下的真实物理尺寸。

    严格复刻 `缺陷计算/caluculate_defect_size.py`（docstring/返回一致，仅整理）。

    Args:
        polygon_points: 多边形顶点列表 [[x, y], ...]，像素坐标（全图绝对坐标）
        camera_param:   相机内参 (CameraParam)
        pose:           相机外参位姿 (Pose)

    Returns:
        diameter_circle: 最小包围圆直径 (mm)
        area_real_mm2:   多边形在世界坐标系下的真实面积 (mm²)
    """
    # 像素坐标提取（HALCON 中 row 对应 y，col 对应 x）
    rows = [max(0, int(point[1])) for point in polygon_points]  # y -> row
    cols = [max(0, int(point[0])) for point in polygon_points]  # x -> col

    # 直接由像素点构造 XLD 轮廓并投影到 Z=0 世界物理平面 -> 面积 + 最小包围圆。
    # 注：region->gen_contour_region_xld("border") 对细线/开放多边形会返回空轮廓（HALCON 24.11 实测），
    # 改用 gen_contour_polygon_xld 直接由顶点构造轮廓，鲁棒且与训练侧等价。
    defect_xld = ha.gen_contour_polygon_xld(rows, cols)
    defect_xld_world = ha.contour_to_world_plane_xld(defect_xld, camera_param, pose, "mm")

    area_mm2, _, _, _ = ha.area_center_xld(defect_xld_world)
    _, _, radius = ha.smallest_circle_xld(defect_xld_world)
    if not area_mm2 or not radius:
        # 投影后 XLD 退化（空元组）：返回 0，由调用方 min_defect_diameter_mm 过滤丢弃，避免误判 NG。
        logger.warning("defect_size: 投影后 XLD 退化(area/radius 空)，返回 0（将由直径过滤丢弃）")
        return 0.0, 0.0
    diameter_circle = 2.0 * radius[0]

    return diameter_circle, area_mm2[0]
