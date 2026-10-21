#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patches.py —— 真实缺陷小图抽取(真工程 real_patches 的迷你同构)。

在**原始 .bmp** 上(零插值失真)按单个缺陷完整标注外框外扩 8px 抠出小图+掩码;
掩码 = 标注多边形 ∩ 所属投影分区(按最大交集归属分区,与 zone_defect_map 同规则)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np

from io_utils import load_polygons_with_labels, read_gray

EXTRACT_PAD = 8  # 外扩边距 px(与真工程 export_pad_px=8 一致)


@dataclass
class DefectPatch:
    zone: str          # 所属检测分区
    label: str         # 缺陷名(= .tmp label)
    image: np.ndarray  # uint8 灰度小图
    mask: np.ndarray   # bool 掩码(缺陷本体)


def extract_patches(img_path: str, gray: np.ndarray,
                    eff_zones: Dict[str, np.ndarray]) -> List[DefectPatch]:
    out: List[DefectPatch] = []
    h, w = gray.shape[:2]
    for lbl, pts in load_polygons_with_labels(img_path):
        if lbl == "排除":
            continue
        poly = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly, [pts], 1)
        best, best_n, best_mask = None, 0, None
        for z, zm in eff_zones.items():
            inter = np.logical_and(poly > 0, zm)
            n = int(inter.sum())
            if n > best_n:
                best, best_n, best_mask = z, n, inter
        if best is None or best_n < 10:
            continue  # 不落在任何检测区(或过小):跳过
        ys, xs = np.nonzero(best_mask)
        y0 = max(0, int(ys.min()) - EXTRACT_PAD); y1 = min(h, int(ys.max()) + 1 + EXTRACT_PAD)
        x0 = max(0, int(xs.min()) - EXTRACT_PAD); x1 = min(w, int(xs.max()) + 1 + EXTRACT_PAD)
        if y1 - y0 < 4 or x1 - x0 < 4:
            continue
        out.append(DefectPatch(
            zone=best, label=lbl,
            image=gray[y0:y1, x0:x1].copy(),
            mask=best_mask[y0:y1, x0:x1].copy(),
        ))
    return out


def area_stats(patches: List[DefectPatch]) -> Dict[str, tuple]:
    """{(zone,label): (面积min, 面积max)} —— 贴图时目标尺寸对数均匀抽样的观测范围。"""
    acc: Dict[str, list] = {}
    for p in patches:
        acc.setdefault((p.zone, p.label), []).append(int(p.mask.sum()))
    return {k: (min(v), max(v)) for k, v in acc.items()}
