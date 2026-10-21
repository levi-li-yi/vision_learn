#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zones.py —— 点位素材索引 + "图不动,蒙版动"的分区逆向投射(真工程逆向对齐的迷你同构)。

每张原图配准一次(基准图↔原图),分区蒙版经 H 投到该图自己的坐标系——
与真工程 dev 版哲学一致:原图永不做几何变换;配准失败回退基准系朴素分区。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from align import OrbAligner, project_mask
from io_utils import (DOMAIN, VAL_DIRNAME, image_size, load_polygons_with_labels,
                      load_zones)


@dataclass
class PointMat:
    """单点位素材索引(真工程 checkup.PointMat 的迷你同构)。"""

    name: str
    dir: str
    ref_name: str                                   # 基准图主名(json 同名)
    ref_path: str
    zones: Dict[str, np.ndarray]                    # {分区名: bool 蒙版(基准系)}
    detect_zones: List[str] = field(default_factory=list)   # 除 domain 外的分区
    train_paths: List[str] = field(default_factory=list)    # 点位根目录训练池(OK+NG)
    val_paths: List[str] = field(default_factory=list)      # val/ 独立考卷
    img_size: tuple = (0, 0)                        # (H, W)


def scan_point(pdir: str) -> PointMat:
    """体检+索引一个点位:分区 json/基准图/同分辨率/考卷隔离,不合格大声报错。"""
    name = os.path.basename(pdir)
    jsons = [f for f in os.listdir(pdir) if f.endswith(".json")]
    if len(jsons) != 1:
        raise SystemExit(f"[体检失败] {name}/ 需恰好 1 份分区 json(现有 {len(jsons)})")
    ref_name = os.path.splitext(jsons[0])[0]
    ref_path = os.path.join(pdir, ref_name + ".bmp")
    if not os.path.isfile(ref_path):
        raise SystemExit(f"[体检失败] {name}/ 缺少与 json 同名的基准图 {ref_name}.bmp")
    zones = load_zones(os.path.join(pdir, jsons[0]))
    if DOMAIN not in zones:
        raise SystemExit(f"[体检失败] {name}/ 分区 json 必含 '{DOMAIN}'(对齐依据)")
    detect_zones = [z for z in zones if z != DOMAIN]
    if not detect_zones:
        raise SystemExit(f"[体检失败] {name}/ 至少 1 个待检分区")

    size = image_size(ref_path)
    train_paths, val_paths = [], []
    for f in sorted(os.listdir(pdir)):
        p = os.path.join(pdir, f)
        if not os.path.isfile(p) or not f.lower().endswith(".bmp"):
            continue
        if image_size(p) != size:
            raise SystemExit(f"[体检失败] {name}/{f} 分辨率与基准图不一致")
        train_paths.append(p)
    vdir = os.path.join(pdir, VAL_DIRNAME)
    if os.path.isdir(vdir):
        for f in sorted(os.listdir(vdir)):
            p = os.path.join(vdir, f)
            if os.path.isfile(p) and f.lower().endswith(".bmp"):
                if image_size(p) != size:
                    raise SystemExit(f"[体检失败] {name}/val/{f} 分辨率与基准图不一致")
                val_paths.append(p)
    if not train_paths:
        raise SystemExit(f"[体检失败] {name}/ 训练池为空")
    return PointMat(name=name, dir=pdir, ref_name=ref_name, ref_path=ref_path,
                    zones=zones, detect_zones=detect_zones,
                    train_paths=train_paths, val_paths=val_paths, img_size=size)


def effective_zones(mat: PointMat, gray: np.ndarray,
                    aligner: Optional[OrbAligner]) -> Dict[str, np.ndarray]:
    """该原图上的 {分区名: bool 有效区}。

    配准成功 → 投影分区(图不动,蒙版动);失败/无 aligner → 基准系朴素分区。
    (真工程此处还有 ±10px 吸附 + 推理圆滑化;迷你版从简,差异见 README。)
    """
    H = aligner.align(gray) if aligner is not None else None
    out: Dict[str, np.ndarray] = {}
    for z in mat.detect_zones:
        m = mat.zones[z]
        if H is not None:
            m = project_mask(m.astype(np.uint8) * 255, H) > 0
        out[z] = m
    return out


def zone_defect_map(mats: List[PointMat], eff_by_img: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, List[str]]:
    """汇总 {分区名: [缺陷名]}(manifest.zones 的来源):按缺陷多边形与投影分区的
    最大交集归属(真工程 auto_bind_zones 的迷你同构),顺序稳定去重。"""
    zone_defects: Dict[str, List[str]] = {}
    for path, eff in eff_by_img.items():
        if not eff:
            continue
        for lbl, pts in load_polygons_with_labels(path):
            if lbl == "排除":
                continue
            sub = np.zeros_like(next(iter(eff.values())), dtype=np.uint8)
            cv2.fillPoly(sub, [pts], 1)
            best, best_n = None, 0
            for z, zm in eff.items():
                n = int(np.logical_and(sub > 0, zm).sum())
                if n > best_n:
                    best, best_n = z, n
            if best is not None and lbl not in zone_defects.setdefault(best, []):
                zone_defects[best].append(lbl)
    return zone_defects
