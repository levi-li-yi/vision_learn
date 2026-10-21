#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""budget.py —— 底图划分与切块网格(真工程 budget 的迷你同构)。

· 网格:stride = int(tile×0.75)(与考试/推理同款全图网格,含 25% 重叠带);
· 干净块:与任何标注多边形外框(外扩 4px)不相交的块 —— 有标注的图上跳过脏块,
  保证负样本/贴图底 100% 干净(真工程"跳过含标注块"同思路);
· 记账:每块预裁缓存(灰度 uint8) + 该块命中的分区列表(投影后的)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from io_utils import load_polygons


@dataclass
class TileRecord:
    img_path: str
    x: int
    y: int
    tile: np.ndarray          # uint8 灰度 [ts, ts]
    zone_union: np.ndarray    # bool [ts, ts] 该块的分区并集(投影后)
    zones: List[str]          # 命中的分区名


def grid_coords(h: int, w: int, ts: int) -> List[Tuple[int, int]]:
    """全图网格(stride=int(ts*0.75),边界 clamp)——与考试/推理逐字同款。"""
    stride = int(ts * 0.75)
    xs = [min(i * stride, max(0, w - ts)) for i in range(int(math.ceil(w / stride)))]
    ys = [min(j * stride, max(0, h - ts)) for j in range(int(math.ceil(h / stride)))]
    return [(x, y) for y in ys for x in xs]


def dirty_boxes(img_path: str, h: int, w: int, pad: int = 4) -> np.ndarray:
    """全部标注多边形(含"排除")的外框 [N,4](x0,y0,x1,y1,外扩 pad)。"""
    boxes = []
    for pts in load_polygons(img_path):
        x0 = max(0, int(pts[:, 0].min()) - pad); x1 = min(w, int(pts[:, 0].max()) + 1 + pad)
        y0 = max(0, int(pts[:, 1].min()) - pad); y1 = min(h, int(pts[:, 1].max()) + 1 + pad)
        boxes.append((x0, y0, x1, y1))
    return np.asarray(boxes, dtype=np.int32).reshape(-1, 4)


def build_tiles(img_path: str, gray: np.ndarray,
                eff_zones: Dict[str, np.ndarray], ts: int) -> List[TileRecord]:
    h, w = gray.shape[:2]
    boxes = dirty_boxes(img_path, h, w)
    union_full = np.zeros((h, w), dtype=bool)
    for zm in eff_zones.values():
        union_full |= zm
    out: List[TileRecord] = []
    for x, y in grid_coords(h, w, ts):
        if _overlap_mask(boxes, x, y, x + ts, y + ts):
            continue  # 脏块跳过(标注区不进干净底池)
        zone_sub = union_full[y:y + ts, x:x + ts]
        if not zone_sub.any():
            continue  # 纯区外块:训练无意义(标签全 255)
        zones_hit = [z for z, zm in eff_zones.items() if zm[y:y + ts, x:x + ts].any()]
        tile = gray[y:y + ts, x:x + ts]
        if tile.shape != (ts, ts):
            padded = np.zeros((ts, ts), dtype=np.uint8)
            padded[:tile.shape[0], :tile.shape[1]] = tile
            tile = padded
        out.append(TileRecord(img_path=img_path, x=x, y=y, tile=tile,
                              zone_union=zone_sub, zones=zones_hit))
    return out


def _overlap_mask(boxes: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> bool:
    """任一外框与 [x0,y0,x1,y1] 相交 → True。"""
    if boxes.size == 0:
        return False
    return bool(np.any((boxes[:, 0] < x1) & (boxes[:, 2] > x0) &
                       (boxes[:, 1] < y1) & (boxes[:, 3] > y0)))
