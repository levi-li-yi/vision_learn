#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""align.py —— OpenCV 逆向对齐(真工程 Halcon Point8DofAligner 的迷你同构)。

与真工程同思想、轻实现:
  · 同:基准图→目标图的 8 自由度单应 H;基准侧角点经 domain 过滤;结构尺度(宽 512)
        小图上估计,再 S⁻¹·H·S 还原全分辨率;内点数与位姿合理性双安检;
        失败返回 None,调用方回退朴素分区(图不动,蒙版动)。
  · 异:ORB+BFMatcher+findHomography 替代 Halcon Harris+RANSAC(免 license);
        本工程 H 直接作用于 (x,y) 坐标(真工程是 (row,col) + rc/xy 交换)——
        蒙版投影因此可直连 cv2.warpPerspective。
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

COARSE_WIDTH = 512       # 结构尺度:估计域最长边
MIN_INLIERS = 30         # 内点下限(真工程 50;迷你 ORB 点少放宽)
MATCH_RATIO = 0.75       # Lowe ratio
RANSAC_REPROJ = 3.0


def _to_small(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-9:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA)


class OrbAligner:
    """基准图(灰度 + domain 蒙版)预备一次;align(目标灰度图) → H(基准→目标, xy)或 None。"""

    def __init__(self, ref_gray: np.ndarray, domain_mask: Optional[np.ndarray],
                 coarse_width: int = COARSE_WIDTH):
        self.a_shape = ref_gray.shape[:2]  # (H, W)
        h, w = self.a_shape
        self.scale = 1.0
        if 0 < coarse_width < w:
            self.scale = coarse_width / float(w)
        ref_s = _to_small(ref_gray, self.scale)
        kp, des = self._orb(ref_s)
        if domain_mask is not None:
            dm = cv2.resize(domain_mask.astype(np.uint8), (ref_s.shape[1], ref_s.shape[0]),
                            interpolation=cv2.INTER_NEAREST) > 0
            keep = [i for i, k in enumerate(kp) if dm[int(k.pt[1]), int(k.pt[0])]]
            kp = [kp[i] for i in keep]
            des = des[keep] if des is not None and len(keep) else des
        self.ref_kp, self.ref_des = kp, des

    @staticmethod
    def _orb(img: np.ndarray):
        orb = cv2.ORB_create(nfeatures=2000)
        kp, des = orb.detectAndCompute(img, None)
        if des is None:
            return [], None
        return kp, des

    def align(self, query_gray: np.ndarray) -> Optional[np.ndarray]:
        h, w = self.a_shape
        if query_gray.shape[:2] != self.a_shape:
            return None
        qry_s = _to_small(query_gray, self.scale)
        kp2, des2 = self._orb(qry_s)
        if des2 is None or self.ref_des is None or len(self.ref_kp) < MIN_INLIERS or len(kp2) < MIN_INLIERS:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw = matcher.knnMatch(self.ref_des, des2, k=2)
        good = [m for m, n in (pair for pair in raw if len(pair) == 2)
                if m.distance < MATCH_RATIO * n.distance]
        if len(good) < MIN_INLIERS:
            return None
        src = np.float32([self.ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        Hs, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ)
        if Hs is None:
            return None
        inliers = int(mask.sum())
        if inliers < MIN_INLIERS:
            return None
        S = np.diag([self.scale, self.scale, 1.0])
        H = np.linalg.inv(S) @ Hs @ S
        H /= H[2, 2]
        if not self._sane(H, w, h):
            return None
        return H

    @staticmethod
    def _sane(H: np.ndarray, w: int, h: int) -> bool:
        corners = np.float32([[0, 0], [w, 0], [0, h], [w, h]]).reshape(-1, 1, 2)
        try:
            moved = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        except cv2.error:
            return False
        shift = float(np.abs(moved - corners.reshape(-1, 2)).max())
        det = float(np.linalg.det(H[:2, :2]))
        return shift <= 0.15 * max(h, w) and 0.5 < det < 2.0


def project_mask(mask_u8: np.ndarray, H: np.ndarray) -> np.ndarray:
    """基准系二值蒙版 (uint8 0/255) 经 H(基准→目标, xy) 逆向投射到目标图坐标系。"""
    h, w = mask_u8.shape[:2]
    warped = cv2.warpPerspective(mask_u8, H, (w, h),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped
