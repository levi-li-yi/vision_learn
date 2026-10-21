#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dataset.py —— 在线贴图数据集(真工程 dataset"每轮在线造题"的迷你同构)。

每 epoch 正负 1:1:一半干净块泊松贴缺陷(标签:缺陷=1/区内背景=0/区外=255),
一半干净块不贴(区内=0/区外=255)。预处理与真工程/推理同款:灰度×3 + ImageNet 归一化。
验证集固定(同种子贴好的同一批题),val_loss 全像素稠密口径。
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from budget import TileRecord
from io_utils import load_polygons_with_labels
from patches import DefectPatch
from paste import paste_defect, sample_target_area

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_gray3(tile_u8: np.ndarray) -> np.ndarray:
    """uint8 [H,W] → float32 [3,H,W](灰度复制 3 通道 + ImageNet 归一化)。"""
    x = np.repeat(tile_u8[..., None].astype(np.float32) / 255.0, 3, axis=2)
    return ((x - _MEAN) / _STD).transpose(2, 0, 1).copy()


def make_label(zone_union: np.ndarray, defect: Optional[np.ndarray] = None,
               ts: int = None) -> np.ndarray:
    """标签 uint8 [H,W]:区内背景 0;缺陷 1;区外 255。"""
    label = np.full(zone_union.shape, 255, dtype=np.uint8)
    label[zone_union] = 0
    if defect is not None:
        label[defect & zone_union] = 1
    return label


def _pick_patch(patches: List[DefectPatch], zone_hint: Optional[str],
                rng: random.Random) -> Tuple[DefectPatch, int, int]:
    """按(优先同分区)抽一个小图;返回 (patch, 面积min, 面积max)——面积范围用全局同 label 观测。"""
    pool = [p for p in patches if p.zone == zone_hint] or patches
    p = rng.choice(pool)
    same = [q for q in patches if q.label == p.label]
    areas = [int(q.mask.sum()) for q in same]
    return p, min(areas), max(areas)


class OnlinePasteDataset(torch.utils.data.Dataset):
    """每 __getitem__ 现场掷骰子贴图(epoch 间天然换题,与真工程"每轮在线造题"同思路)。"""

    def __init__(self, tiles: List[TileRecord], patches: List[DefectPatch],
                 seed: int, positive_ratio: float = 0.5,
                 dilate_min: int = 5, dilate_max: int = 7):
        assert tiles and patches, "干净块池与缺陷小图池都不能为空"
        self.tiles = tiles
        self.patches = patches
        self.rng = random.Random(seed)
        self.positive_ratio = positive_ratio
        self.dilate_min, self.dilate_max = dilate_min, dilate_max

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        rec = self.tiles[idx]
        tile = rec.tile
        defect_label = None
        if self.rng.random() < self.positive_ratio:
            p, a_min, a_max = _pick_patch(self.patches, self.rng.choice(rec.zones), self.rng)
            target = sample_target_area(a_min, a_max, self.rng)
            res = paste_defect(tile, rec.zone_union, p.image, p.mask, target,
                               self.rng, self.dilate_min, self.dilate_max)
            if res is not None:
                tile, defect_label = res
        img = torch.from_numpy(normalize_gray3(tile))
        label = torch.from_numpy(make_label(rec.zone_union, defect_label))
        return img, label


def build_fixed_val(tiles: List[TileRecord], patches: List[DefectPatch], seed: int,
                    n: int = 32) -> List[Tuple[np.ndarray, np.ndarray]]:
    """固定验证题集:同种子贴好一遍,之后逐轮原题重考(val_loss 跨轮可比)。"""
    rng = random.Random(seed)
    picks = [tiles[i % len(tiles)] for i in range(min(n, len(tiles) * 2))]
    out = []
    for rec in picks:
        tile, defect_label = rec.tile, None
        if rng.random() < 0.5:
            p, a_min, a_max = _pick_patch(patches, rng.choice(rec.zones), rng)
            res = paste_defect(tile, rec.zone_union, p.image, p.mask,
                               sample_target_area(a_min, a_max, rng), rng)
            if res is not None:
                tile, defect_label = res
        out.append((normalize_gray3(tile), make_label(rec.zone_union, defect_label)))
    return out
