#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paste.py —— 泊松融合贴缺陷(真工程旧版泊松贴图方案的学习版回归,替代 FLUX 生成)。

与旧训练工程同要点:
  · 掩码外扩 5~7px 再 seamlessClone(NORMAL_CLONE)——盖过 OpenCV 内部 4px 腐蚀,
    保住缺陷边缘梯度(MIXED_CLONE 会让斜划痕发淡,弃用);
  · 目标尺寸在观测 [min,max] 内**对数均匀**抽样(普通均匀会让大缺陷霸屏);
  · 随机旋转;贴入后按贴图掩码 ∩ 分区 打标签(区内缺陷=1,区内背景=0,区外=255)。
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import cv2
import numpy as np

PLACE_TRIES = 10


def _transform_patch(patch_img: np.ndarray, patch_mask: np.ndarray,
                     target_area: int, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    src_area = int(patch_mask.sum())
    if src_area <= 0:
        return patch_img, patch_mask
    s = float(np.sqrt(target_area / src_area))
    h, w = patch_img.shape[:2]
    img = cv2.resize(patch_img, (max(2, int(round(w * s))), max(2, int(round(h * s)))),
                     interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
    msk = cv2.resize(patch_mask.astype(np.uint8), (img.shape[1], img.shape[0]),
                     interpolation=cv2.INTER_NEAREST) > 0
    mat = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1.0)
    img_r = cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    msk_r = cv2.warpAffine(msk.astype(np.uint8), mat, (img.shape[1], img.shape[0]),
                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0) > 0
    return img_r, msk_r


def paste_defect(tile: np.ndarray, zone: np.ndarray, patch_img: np.ndarray,
                 patch_mask: np.ndarray, target_area: int, rng: random.Random,
                 dilate_min: int = 5, dilate_max: int = 7
                 ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """把缺陷泊松贴进 tile 的分区附近。成功 → (贴后 tile, 贴图标签 bool 掩版);失败 → None。"""
    th, tw = tile.shape[:2]
    img, msk = _transform_patch(patch_img, patch_mask, target_area,
                                rng.uniform(0.0, 360.0))
    ph, pw = msk.shape[:2]
    if ph >= th or pw >= tw:
        return None
    k = rng.randint(dilate_min, dilate_max)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    msk_d = cv2.dilate(msk.astype(np.uint8), kernel) > 0

    for _ in range(PLACE_TRIES):
        cx = rng.randint(0, tw - pw)
        cy = rng.randint(0, th - ph)
        # 落点约束:缺陷本体 ≥90% 在分区内(贴边缺陷允许小幅骑跨)
        zone_sub = zone[cy:cy + ph, cx:cx + pw]
        if zone_sub.shape != msk.shape:
            continue
        inside = int(np.logical_and(msk, zone_sub).sum())
        if inside < 0.9 * max(1, int(msk.sum())):
            continue
        roi = tile[cy:cy + ph, cx:cx + pw]
        try:
            blended = cv2.seamlessClone(
                cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR),
                (msk_d * 255).astype(np.uint8),
                (pw // 2, ph // 2), cv2.NORMAL_CLONE,
            )[:, :, 0]  # 取 B 通道回灰度(BGR 三通道同源灰度,等价)
        except cv2.error:
            return None
        tile = tile.copy()
        tile[cy:cy + ph, cx:cx + pw] = blended
        label = np.zeros((th, tw), dtype=bool)
        label[cy:cy + ph, cx:cx + pw] = msk & zone_sub
        return tile, label
    return None


def sample_target_area(a_min: int, a_max: int, rng: random.Random) -> int:
    """对数均匀抽样(每个数量级等概率)。"""
    lo = np.log(max(4, a_min))
    hi = np.log(max(5, a_max))
    return int(np.exp(rng.uniform(lo, hi)))
