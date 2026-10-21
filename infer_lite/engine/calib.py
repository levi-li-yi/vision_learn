from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import halcon as ha

from common.exceptions import CalibNotFound
from common.logger import setup_logger

logger = setup_logger("calib")

# 现场示教产物固定文件名（与 缺陷计算/ 示例一致）
CAMERA_PARAM_FILE = "camera_param.dat"
POSE_FILE = "pose.dat"


class CalibRegistry:
    """相机内外参按 product_code（机种）缓存。

    语义：同一机种（一台设备终端）共用一套内外参，现场示教阶段固化相机与产品的
    相对距离后导出。布局：{calib_dir}/{product_code}/{camera_param.dat,pose.dat}
    （默认扁平，无中间子目录；如需中间层可经 sub_dir 启用）。

    首载 read_cam_par / read_pose 在锁外执行（double-check），避免阻塞并发查询；
    与 ModelRegistry 同构，但无淘汰（机种数有限且常驻内存代价极低）。
    """

    def __init__(self, calib_dir: str, sub_dir: str = ""):
        self.calib_dir = Path(calib_dir)
        self.sub_dir = sub_dir
        self._cache: "OrderedDict[str, tuple[Any, Any]]" = OrderedDict()
        self._lock = threading.RLock()

    def _paths(self, product_code: str) -> tuple[Path, Path]:
        base = self.calib_dir / product_code
        if self.sub_dir:
            base = base / self.sub_dir
        return base / CAMERA_PARAM_FILE, base / POSE_FILE

    def get(self, product_code: str) -> tuple[Any, Any]:
        """返回 (camera_param, pose)；缺失抛 CalibNotFound（映射 500）。"""
        with self._lock:
            hit = self._cache.get(product_code)
            if hit is not None:
                self._cache.move_to_end(product_code)
                return hit

        # ---- 注册锁外：磁盘 IO + HALCON 解析 ----
        cam_path, pose_path = self._paths(product_code)
        if not cam_path.exists() or not pose_path.exists():
            raise CalibNotFound(str(cam_path.parent))
        cam_par = ha.read_cam_par(str(cam_path))
        pose = ha.read_pose(str(pose_path))

        with self._lock:
            # double-check：并发首载时首者获胜
            existing = self._cache.get(product_code)
            if existing is not None:
                return existing
            self._cache[product_code] = (cam_par, pose)
            logger.info(f"calib loaded product={product_code} dir={cam_path.parent}")
            return cam_par, pose
