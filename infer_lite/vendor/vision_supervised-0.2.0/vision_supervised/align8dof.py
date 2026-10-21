#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""align8dof.py —— 训练/推理共享的逆向对齐与"有效检测区"原语(★ D5 同一套)。

逐字提取自训练仓 supervised_apps/train 的 src/train/align.py @ origin/dev(3cdfdcb),
严禁单方面修改任何参数或分支逻辑——两侧口径漂移即判定失效:
  · Point8DofAligner        Halcon 8-DOF 单应(基准图 → 目标图), warp_image=False 默认;
  · project_mask_to_query   基准系二值蒙版经 H 逆向投射到目标原图坐标系(rc/xy 交换已内置);
  · compute_effective_zone  ±SNAP_PX(10) 法向边缘吸附; smooth_for_infer=True 为推理/考卷
                            口径(曲率保持去锯齿 + 大窗口圆滑架桥, 包住贴边缺陷);
  · align_and_compute_effective_zones  组合入口: 投影 + 吸附(+圆滑) → 各分区 bool 有效检测区。
"""

from __future__ import annotations

import time
from typing import Dict

import cv2
import halcon
import numpy as np

GRID = 6          # 6×6 = 36 格
PER_CELL = 30     # 每格取最强 30 个
SNAP_PX = 10      # 分区边界法向吸附范围 (±10px 二次精对齐)


def spatial_select(rows, cols, shape, grid=GRID, per_cell=PER_CELL):
    """分网格采样: 图像分 grid×grid 格, 每格取最强角点, 保证空间均匀。"""
    if len(rows) == 0:
        return [], []
    h, w = shape
    gy = np.clip((np.array(rows) / h * grid).astype(int), 0, grid - 1)
    gx = np.clip((np.array(cols) / w * grid).astype(int), 0, grid - 1)
    cell = gy * grid + gx
    sel = []
    for c in range(grid * grid):
        idx = np.where(cell == c)[0]
        sel.extend(idx[:per_cell])
    return [rows[i] for i in sel], [cols[i] for i in sel]


def project_mask_to_query(ref_mask_u8: np.ndarray, H: np.ndarray) -> np.ndarray:
    """将基准图坐标系下的二值蒙版 (uint8 0/255) 经单应矩阵 H 逆向投射到目标原图坐标系, 并与目标图范围求交。"""
    H_rc = np.asarray(H, dtype=np.float64)
    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    H_xy = swap @ H_rc @ swap
    h, w = ref_mask_u8.shape[:2]
    warped = cv2.warpPerspective(
        ref_mask_u8, H_xy, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return (warped > 127).astype(np.uint8) * 255


def _remap_1d_profiles(gray_f: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """分块调用 cv2.remap, 防止超长轮廓点数 N >= 32767 触发 OpenCV SHRT_MAX 断言。"""
    n = map_x.shape[0]
    chunk = 16000
    if n <= chunk:
        return cv2.remap(gray_f, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    parts = [
        cv2.remap(
            gray_f, map_x[i:i + chunk], map_y[i:i + chunk],
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        for i in range(0, n, chunk)
    ]
    return np.concatenate(parts, axis=0)


def compute_effective_zone(
    proj_u8: np.ndarray,
    gray: np.ndarray,
    snap_px: int = SNAP_PX,
    smooth_for_infer: bool = False,
) -> np.ndarray:
    """计算目标图上的"有效检测区" = (精对齐分区 ∩ 目标图):
      · 在逆向投射后的分区边界法向 ±snap_px (±10px) 范围内搜索底图真实边缘梯度峰值做二次精对齐;
      · 训练模式 (smooth_for_infer=False): 直接紧贴底图局部真实边缘 (向内微收 1px 避开斜坡暗部), 确保区内干净;
      · 推理模式 (smooth_for_infer=True): 对轮廓做去锯齿曲率保持滤波 + 大窗口法向偏移圆滑处理,
        平滑架桥跨过崩边、塌边与贴边缺口, 将边缘缺陷包含进有效检测区内。
    """
    cnts, hier = cv2.findContours(proj_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cnts or hier is None:
        return (proj_u8 > 127).astype(np.uint8) * 255

    h, w = gray.shape[:2]
    gray_f = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0.8)
    ts = np.arange(-snap_px - 1, snap_px + 2, dtype=np.float32)
    t_vals = np.arange(-snap_px, snap_px + 1, dtype=np.float32)

    out_cnts = []
    for idx, c in enumerate(cnts):
        pts = c[:, 0, :].astype(np.float32)
        n = len(pts)
        if n < 24:
            out_cnts.append(np.round(pts).astype(np.int32).reshape(-1, 1, 2))
            continue

        # 推理模式: 先对基准图逆向投射轮廓做曲率保持去锯齿 (消除 define_masks 锯齿与漏掉的塌边, 且不收缩圆弧)
        if smooth_for_infer and n >= 60:
            win_g = min(41, (n // 4) | 1)
            pad_g = win_g // 2
            ker_g = cv2.getGaussianKernel(win_g, win_g / 5.0).reshape(-1)
            gx = np.convolve(np.pad(pts[:, 0], pad_g, mode="wrap"), ker_g, mode="valid")
            gy = np.convolve(np.pad(pts[:, 1], pad_g, mode="wrap"), ker_g, mode="valid")
            rx = pts[:, 0] - gx
            ry = pts[:, 1] - gy
            rx_s = np.convolve(np.pad(rx, pad_g, mode="wrap"), ker_g, mode="valid")
            ry_s = np.convolve(np.pad(ry, pad_g, mode="wrap"), ker_g, mode="valid")
            pts = np.stack([gx + rx_s, gy + ry_s], axis=-1).astype(np.float32)

        wt = min(12, n // 4)
        diffs = np.roll(pts, -wt, axis=0) - np.roll(pts, wt, axis=0)
        lengths = np.hypot(diffs[:, 0], diffs[:, 1]) + 1e-6
        tx = diffs[:, 0] / lengths
        ty = diffs[:, 1] / lengths
        nx, ny = -ty, tx

        probe_in_x = np.clip(np.round(pts[:, 0] - 3.0 * nx).astype(np.int32), 0, w - 1)
        probe_in_y = np.clip(np.round(pts[:, 1] - 3.0 * ny).astype(np.int32), 0, h - 1)
        probe_out_x = np.clip(np.round(pts[:, 0] + 3.0 * nx).astype(np.int32), 0, w - 1)
        probe_out_y = np.clip(np.round(pts[:, 1] + 3.0 * ny).astype(np.int32), 0, h - 1)
        if float(np.mean(proj_u8[probe_out_y, probe_out_x])) > float(np.mean(proj_u8[probe_in_y, probe_in_x])):
            nx, ny = -nx, -ny

        map_x = (pts[:, 0:1] + nx[:, None] * ts[None, :]).astype(np.float32)
        map_y = (pts[:, 1:2] + ny[:, None] * ts[None, :]).astype(np.float32)
        prof = _remap_1d_profiles(gray_f, map_x, map_y)
        prof = cv2.GaussianBlur(prof, (1, 5), 0.0)
        grad = 0.5 * np.abs(prof[:, 2:] - prof[:, :-2])
        weighted_grad = grad * (1.0 - 0.015 * np.abs(t_vals)[None, :])
        best_idx = np.argmax(weighted_grad, axis=1)
        peak_grad = grad[np.arange(n), best_idx]
        shifts = np.where(peak_grad >= 4.0, t_vals[best_idx], 0.0).astype(np.float32)

        if not smooth_for_infer:
            # 训练模式: 忠实吸附到底图局部边缘 (向内微收 1.0px 避开边缘斜坡暗过渡带)
            shifts = np.where(peak_grad >= 4.0, np.clip(shifts - 1.0, -snap_px, snap_px), 0.0).astype(np.float32)
            pad = min(5, n // 6)
            if pad >= 1:
                s_pad = np.pad(shifts, pad, mode="wrap")
                shifts = np.median(
                    np.lib.stride_tricks.sliding_window_view(s_pad, 2 * pad + 1), axis=-1
                ).astype(np.float32)
        else:
            # 推理模式: 大窗口圆滑滤波, 平滑架桥跨过崩边、塌边与贴边砂眼缺口
            pad = min(60, n // 4)
            if pad >= 3:
                s_pad = np.pad(shifts, pad, mode="wrap")
                win_view = np.lib.stride_tricks.sliding_window_view(s_pad, 2 * pad + 1)
                shifts = np.percentile(win_view, 65, axis=-1).astype(np.float32)
                ker_s = cv2.getGaussianKernel(2 * pad + 1, pad / 2.0).reshape(-1)
                shifts = np.convolve(np.pad(shifts, pad, mode="wrap"), ker_s, mode="valid").astype(np.float32)

        new_pts = pts + np.stack([nx * shifts, ny * shifts], axis=-1)
        new_pts[:, 0] = np.clip(new_pts[:, 0], 0, w - 1)
        new_pts[:, 1] = np.clip(new_pts[:, 1], 0, h - 1)
        out_cnts.append(np.round(new_pts).astype(np.int32).reshape(-1, 1, 2))

    out = np.zeros((h, w), dtype=np.uint8)
    for idx in range(len(out_cnts)):
        if hier[0, idx, 3] < 0:
            cv2.drawContours(out, out_cnts, idx, 255, thickness=cv2.FILLED)
    for idx in range(len(out_cnts)):
        if hier[0, idx, 3] >= 0:
            cv2.drawContours(out, out_cnts, idx, 0, thickness=cv2.FILLED)
    return out


def align_and_compute_effective_zones(
    ref_zone_u8: Dict[str, np.ndarray],
    H: np.ndarray,
    target_gray: np.ndarray,
    smooth_for_infer: bool = False,
    snap_px: int = SNAP_PX,
) -> Dict[str, np.ndarray]:
    """将基准图各检测分区逆向投射到目标原图并执行 ±snap_px 边缘吸附, 返回各分区的 bool 有效检测区。"""
    out: Dict[str, np.ndarray] = {}
    for z, rm_u8 in ref_zone_u8.items():
        proj_u8 = project_mask_to_query(rm_u8, H)
        eff_u8 = compute_effective_zone(
            proj_u8, target_gray, snap_px=snap_px, smooth_for_infer=smooth_for_infer
        )
        out[z] = eff_u8 > 127
    return out


class Point8DofAligner:
    """Halcon 管线: (结构尺度缩放) → points_harris(分区均匀) →
    proj_match_points_ransac(8-DOF 单应: 基准图 → 目标图)"""

    def __init__(self, ref_gray: np.ndarray, domain_mask: np.ndarray,
                 row_tolerance: int = 80, col_tolerance: int = 80,
                 match_threshold: float = 0.3, coarse_width: int = 512):
        self.a_shape = ref_gray.shape
        self.row_tol = int(row_tolerance)
        self.col_tol = int(col_tolerance)
        self.match_thresh = float(match_threshold)
        h, w = self.a_shape
        self.scale = 1.0
        if 0 < int(coarse_width) < w:
            self.scale = int(coarse_width) / float(w)
        small_shape = (max(1, int(round(h * self.scale))),
                       max(1, int(round(w * self.scale))))

        t0 = time.perf_counter()
        ref_s = self._to_struct_scale(ref_gray)
        self.h_ref = halcon.himage_from_numpy_array(np.ascontiguousarray(ref_s))
        rows, cols = halcon.points_harris(self.h_ref, 0.7, 2.0, 0.08, 1e-3)
        rows, cols = list(rows), list(cols)
        if domain_mask is not None and len(rows) > 0:
            dm_s = cv2.resize(domain_mask.astype(np.uint8),
                              (small_shape[1], small_shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
            ok = dm_s[np.array(rows, dtype=int), np.array(cols, dtype=int)]
            rows = [r for r, k in zip(rows, ok) if k]
            cols = [c for c, k in zip(cols, ok) if k]
        self.ref_rows, self.ref_cols = spatial_select(rows, cols, small_shape)
        self.ms_detect_ref = (time.perf_counter() - t0) * 1000

    def _to_struct_scale(self, gray: np.ndarray) -> np.ndarray:
        if abs(self.scale - 1.0) < 1e-9:
            return gray
        h, w = gray.shape
        return cv2.resize(gray,
                          (max(1, int(round(w * self.scale))),
                           max(1, int(round(h * self.scale)))),
                          interpolation=cv2.INTER_AREA)

    def align(self, query_gray: np.ndarray, warp_image: bool = False) -> dict:
        """求解基准图 → 目标图的单应矩阵 H (默认不透视变换底图)。"""
        t_total = time.perf_counter()
        h, w = self.a_shape
        if query_gray.shape != self.a_shape:
            return self._fail(f"待测图尺寸 {query_gray.shape[:2]} 与基准图 "
                              f"{self.a_shape} 不一致")

        qry_s = self._to_struct_scale(query_gray)
        h_test = halcon.himage_from_numpy_array(np.ascontiguousarray(qry_s))
        tr, tc = halcon.points_harris(h_test, 0.7, 2.0, 0.08, 1e-3)
        test_rows, test_cols = spatial_select(list(tr), list(tc), qry_s.shape)

        if len(test_rows) < 50 or len(self.ref_rows) < 50:
            return self._fail(f"角点不足 (基准 {len(self.ref_rows)} / 待测 {len(test_rows)})")

        tol_r = max(2, int(round(self.row_tol * self.scale)))
        tol_c = max(2, int(round(self.col_tol * self.scale)))
        hom_mat2d, pts1, pts2 = halcon.proj_match_points_ransac(
            self.h_ref, h_test,
            self.ref_rows, self.ref_cols,
            test_rows, test_cols,
            'ncc', 10, 0, 0, tol_r, tol_c, 0.0,
            self.match_thresh, 'gold_standard', 1.5, 42)

        Hs = np.array(hom_mat2d, dtype=np.float64).reshape(3, 3)
        Hs /= Hs[2, 2]
        n_inliers = len(pts1) if pts1 else 0
        if n_inliers < 50:
            return self._fail(f"内点不足 ({n_inliers} < 50)")

        S = np.diag([self.scale, self.scale, 1.0])
        H = np.linalg.inv(S) @ Hs @ S
        H /= H[2, 2]

        corners = np.float32([[0, 0], [0, w], [h, 0], [h, w]]).reshape(-1, 1, 2)
        try:
            moved = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
            shift = float(np.abs(moved - corners.reshape(-1, 2)).max())
            det = float(np.linalg.det(H[:2, :2]))
        except cv2.error:
            return self._fail("单应变换求解失败")
        if shift > 0.15 * max(self.a_shape) or not (0.5 < det < 2.0):
            return self._fail(f"位姿不合理 (角点最大位移 {shift:.0f}px, 尺度 det {det:.3f})")

        return {
            "ok": True, "reason": None, "aligned": None, "native": None,
            "H": H, "inliers": n_inliers, "max_shift_px": round(shift, 1),
            "det": round(det, 4),
            "ms_total": round((time.perf_counter() - t_total) * 1000, 1),
        }

    @staticmethod
    def _fail(reason: str) -> dict:
        return {"ok": False, "reason": reason, "aligned": None, "native": None,
                "H": None, "inliers": 0, "max_shift_px": 0.0, "det": 0.0}
