"""课时 3：推理模式圆滑化与架桥 —— compute_effective_zone 核心拆解（下）。

推理/考卷模式（smooth_for_infer=True）与训练模式目标相反：训练要区内干净（向内收），
推理要把**边缘缺肉包含进检测区**——贴边崩边、塌边缺口都是要检出的缺陷，边界若紧贴
零件边缘，缺口一半在区外就被漏检。两级处理：

  1. 曲率保持去锯齿（前置）：轮廓拆"高斯平滑主形 + 残差再平滑"叠加——
     直接高斯平滑整条轮廓会把圆弧往内收缩（弦短弓弯），而"平滑主形 + 平滑残差"
     保留低频曲率，只抹掉高频手画锯齿，圆弧半径基本不缩；
  2. 肩部定位 + 大窗架桥：正常边界与训练同源定位在肩部（−1.0px）并做小窗中值平滑；
     随后用超大窗口（pad=min(60, n//4)）取 **55 百分位** 偏移再高斯卷积得到"架桥线"，
     shifts = max(局部, 架桥)——只在局部凹陷（崩边/缺口，局部偏移显著为负）处被架桥拉起，
     正常边界完全不受影响。百分位取 55 而不是均值：均值会把正常段也整体外推。

对应主仓库：src/train/align.py::compute_effective_zone(smooth_for_infer 分支)。

运行：python -m stage6.smooth_bridge
"""
import cv2
import numpy as np

from stage6.normal_snap import (
    SNAP_PX, contour_normals, directed_peak, median_smooth,
    orient_normals, redraw_hierarchy, sample_profiles,
)

SHOULDER_IN_INFER = 1.0   # 推理模式肩部内收 1.0px（训练模式 1.5px）


def smooth_keep_curvature(pts: np.ndarray) -> np.ndarray:
    """曲率保持去锯齿：高斯主形 + 残差平滑叠加（不收缩圆弧）。

    主形 = 大高斯核平滑掉锯齿后的低频形状；残差 = 原轮廓 − 主形（锯齿高频）。
    残差再做高斯平滑（高频几乎归零），叠加回来 = 保留曲率的低频 + 抹平的高频。
    """
    n = len(pts)
    win_g = min(31, (n // 4) | 1)
    pad_g = win_g // 2
    ker_g = cv2.getGaussianKernel(win_g, win_g / 5.0).reshape(-1)
    gx = np.convolve(np.pad(pts[:, 0], pad_g, mode="wrap"), ker_g, mode="valid")
    gy = np.convolve(np.pad(pts[:, 1], pad_g, mode="wrap"), ker_g, mode="valid")
    rx, ry = pts[:, 0] - gx, pts[:, 1] - gy
    rx_s = np.convolve(np.pad(rx, pad_g, mode="wrap"), ker_g, mode="valid")
    ry_s = np.convolve(np.pad(ry, pad_g, mode="wrap"), ker_g, mode="valid")
    return np.stack([gx + rx_s, gy + ry_s], axis=-1).astype(np.float32)


def bridge_shifts(shifts_loc: np.ndarray) -> np.ndarray:
    """大窗架桥：55 百分位窗口偏移 + 高斯卷积 → 只把凹陷段拉回正常边界水平。"""
    n = len(shifts_loc)
    pad = min(60, n // 4)
    if pad < 3:
        return shifts_loc
    s_pad = np.pad(shifts_loc, pad, mode="wrap")
    win_view = np.lib.stride_tricks.sliding_window_view(s_pad, 2 * pad + 1)
    s_bridge = np.percentile(win_view, 55, axis=-1).astype(np.float32)
    ker_s = cv2.getGaussianKernel(2 * pad + 1, pad / 2.5).reshape(-1)
    s_bridge = np.convolve(np.pad(s_bridge, pad, mode="wrap"), ker_s, mode="valid").astype(np.float32)
    return np.maximum(shifts_loc, s_bridge).astype(np.float32)


def snap_zone_infer(proj_u8: np.ndarray, gray: np.ndarray, snap_px: int = SNAP_PX) -> np.ndarray:
    """推理模式有效检测区：曲率保持去锯齿 → 肩部定位 → 大窗架桥（复刻推理分支全流程）。"""
    cnts, hier = cv2.findContours(proj_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cnts or hier is None:
        return (proj_u8 > 127).astype(np.uint8) * 255
    h, w = gray.shape[:2]
    gray_f = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0.8)
    zone_med = float(np.median(gray_f[proj_u8 > 127])) if np.any(proj_u8 > 127) else 128.0

    out_cnts = []
    for c in cnts:
        pts = c[:, 0, :].astype(np.float32)
        n = len(pts)
        if n < 24:
            out_cnts.append(np.round(pts).astype(np.int32).reshape(-1, 1, 2))
            continue
        if n >= 60:
            pts = smooth_keep_curvature(pts)          # ① 前置去锯齿（长轮廓才做）
        _, _, nx, ny = contour_normals(pts)
        nx, ny = orient_normals(proj_u8, pts, nx, ny)
        prof, _ = sample_profiles(gray, pts, nx, ny, snap_px)
        best_t, peak_grad, _ = directed_peak(prof, zone_med, snap_px)
        shifts_base = np.where(peak_grad >= 4.0,
                               np.clip(best_t - SHOULDER_IN_INFER, -snap_px, snap_px),
                               0.0).astype(np.float32)
        shifts_loc = median_smooth(shifts_base, pad=min(5, n // 6))   # ② 肩部 + 小窗中值
        shifts = bridge_shifts(shifts_loc)                            # ③ 大窗架桥
        new_pts = pts + np.stack([nx * shifts, ny * shifts], axis=-1)
        new_pts[:, 0] = np.clip(new_pts[:, 0], 0, w - 1)
        new_pts[:, 1] = np.clip(new_pts[:, 1], 0, h - 1)
        out_cnts.append(np.round(new_pts).astype(np.int32).reshape(-1, 1, 2))
    return redraw_hierarchy(out_cnts, hier, gray.shape[:2])


def make_notched_part(seed: int = 5, h: int = 260, w: int = 340):
    """合成"边缘崩边缺口"零件：亮圆盘边缘被啃掉一块缺口（贴边塌边缺陷替身）。

    返回 (gray, notch_mask 缺口真值, proj_u8 略内缩的分区)。缺口长在真实边缘上，
    一半在零件内一半在零件外——推理分区必须把它包住，训练分区则会把它切开。
    """
    rng = np.random.default_rng(seed)
    gray = np.full((h, w), 35.0, dtype=np.float32)
    part = np.zeros((h, w), dtype=np.uint8)
    notch = np.zeros((h, w), dtype=np.uint8)
    cx, cy, rad = w // 2, h // 2, h // 3
    cv2.circle(gray, (cx, cy), rad, 205.0, -1)
    cv2.circle(part, (cx, cy), rad, 255, -1)
    # 缺口：以 (cx+rad, cy) 为中心的暗斑，一半在零件边缘外
    cv2.circle(gray, (cx + rad, cy), 9, 40.0, -1)
    cv2.circle(notch, (cx + rad, cy), 9, 255, -1)
    gray += rng.normal(0, 1.5, gray.shape).astype(np.float32)
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    ang = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    poly = np.stack([cx + (rad - 4.0) * np.cos(ang), cy + (rad - 4.0) * np.sin(ang)], axis=1)
    poly += rng.uniform(-2, 2, poly.shape)            # 手画抖动
    proj = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(proj, [poly.astype(np.int32)], 255)
    return gray, notch, proj


if __name__ == "__main__":
    from stage6.normal_snap import snap_zone_train

    gray, notch, proj = make_notched_part()
    zone_train = snap_zone_train(proj, gray)
    zone_infer = snap_zone_infer(proj, gray)

    cover_t = ((zone_train > 127) & (notch > 0)).sum() / (notch > 0).sum()
    cover_i = ((zone_infer > 127) & (notch > 0)).sum() / (notch > 0).sum()
    print(f"缺口被训练分区覆盖 {cover_t:.1%}（紧贴边缘，缺肉被切开——贴图训练要的正是区内干净）")
    print(f"缺口被推理分区覆盖 {cover_i:.1%}（架桥外推，把边缘缺肉包进检测区防漏检）")

    # 曲率保持验证：圆弧去锯齿后面积不缩（直接大核平滑会缩 >5%）
    a_proj = (proj > 127).sum()
    smooth_only = smooth_keep_curvature(
        cv2.findContours(proj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0][:, 0, :].astype(np.float32))
    hull = np.zeros_like(proj)
    cv2.fillPoly(hull, [np.round(smooth_only).astype(np.int32)], 255)
    print(f"去锯齿前后面积 {a_proj} → {(hull > 127).sum()} (圆弧不收缩；直接大核平滑会缩 >5%)")
