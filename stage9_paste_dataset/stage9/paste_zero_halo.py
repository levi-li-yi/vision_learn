"""课时 1：零光晕泄漏（Zero-Halo-Bleed）泊松融合 —— dataset.py::paste_poisson 全复刻。

cv2.seamlessClone 直接把缺陷小图贴进底图会产生"光晕"：泊松方程的解会顺手修改掩码
**外**一圈像素（亮缺陷带出亮晕、暗缺陷压出暗晕），而训练标签规定 label=0 的像素必须
是 100% 干净背景——光晕就是标签噪声。真工程的解法七件套：

  1. 对比度半压缩到 128 中枢：src/dst 都做 (I − 中值)×0.5 + 128，克隆后残差 ×2 还原——
     压缩动态范围防止 uint8 截断吃掉泊松解的细节；
  2. 写回掩码 eval_mask = (m_float > 0.30) ∧ zone_win：label=0 处权重恒 0（零光晕的根）；
  3. 距离变换 α 环：外圈 1px 权重 0.55，向内 ≥2px 满幅 1.0（边缘半融合无硬茬）；
  4. 极性强制 polarity：暗缺陷（−1）把残差翻成负值并截断在 +4；亮缺陷（+1）反之——
     防止泊松解在个别像素"翻极性"制造反色噪点；
  5. 目标幅值重标定 target_diff：p50/p75 混合度量当前幅值，按目标对比度缩放 clip(0.5, 6)；
  6. 暗区孔洞门控 hole_gate：背景亮度 ≤20 的暗区不许提亮（暗孔里不会长亮缺陷）；
  7. 区外 carve/built：贴边缺陷触及检测区边界时，在所有区并集**之外**按随机比例
     w_ext 自然过渡（挖暗/built 亮），但相邻区的 label=0 像素一个都不许动。

对应主仓库：src/train/dataset.py::paste_poisson（172~335 行逐段对应）。

运行：python -m stage9.paste_zero_halo
"""
from typing import Optional, Tuple

import cv2
import numpy as np

DILATE_MIN, DILATE_MAX = 5, 7   # 泊松求解外扩域像素数（rng.randint(5, 7)）


def paste_poisson(
    gray: np.ndarray,
    patch: np.ndarray,
    m_float: np.ndarray,
    rng,
    off: Tuple[int, int] = (0, 0),
    zone_win: Optional[np.ndarray] = None,
    union_win: Optional[np.ndarray] = None,
    target_diff: Optional[float] = None,
    polarity: int = 0,
):
    """零光晕泄漏泊松融合：返回 (blended, label) 或 None。逐行复刻 dataset.py 同名函数。"""
    k = rng.randint(DILATE_MIN, DILATE_MAX)
    label = m_float > 0.30
    m = (m_float > 0.35).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    src_bg = float(patch[0, 0])                          # 小图左上角 = 去背景填充值
    pad = max(0, k + 2 - ys.min(), k + 2 - xs.min(),
              k + 2 - (m.shape[0] - 1 - ys.max()),
              k + 2 - (m.shape[1] - 1 - xs.max()))
    patch_pad = patch
    if pad > 0:
        patch_pad = cv2.copyMakeBorder(patch, pad, pad, pad, pad,
                                       cv2.BORDER_REPLICATE)
        m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    dil = cv2.dilate(m, ker)                             # 求解域 = 掩码外扩 k px
    if dil.sum() < 10:
        return None
    bx, by, bw, bh = cv2.boundingRect(dil)
    sx0, sy0 = max(0, bx - 1), max(0, by - 1)
    sx1, sy1 = min(patch_pad.shape[1], bx + bw + 1), min(patch_pad.shape[0], by + bh + 1)
    gx0, gy0 = int(off[0]) - pad + bx, int(off[1]) - pad + by
    H, W = gray.shape[:2]
    rx0, ry0 = max(0, gx0 - 2), max(0, gy0 - 2)
    rx1, ry1 = min(W, gx0 + bw + 2), min(H, gy0 + bh + 2)
    if rx1 <= rx0 or ry1 <= ry0:
        return None

    dst_roi = gray[ry0:ry1, rx0:rx1]
    if zone_win is not None:
        z_roi = zone_win[ry0:ry1, rx0:rx1]
        if np.any(z_roi) and not np.all(z_roi):
            in_med = int(np.median(dst_roi[z_roi]))
            dst_clean_roi = np.where(z_roi, dst_roi, np.uint8(in_med))
        else:
            dst_clean_roi = dst_roi
    else:
        dst_clean_roi = dst_roi

    # ① 双方对比度半压缩到 128 中枢
    dst_med = float(np.median(dst_clean_roi))
    dst_c = np.clip(128.0 + (dst_clean_roi.astype(np.float32) - dst_med) * 0.5, 0, 255).astype(np.uint8)
    src_raw = patch_pad[sy0:sy1, sx0:sx1].astype(np.float32) - src_bg
    src_c = np.clip(128.0 + src_raw * 0.5, 0, 255).astype(np.uint8)

    src3 = cv2.cvtColor(src_c, cv2.COLOR_GRAY2BGR)
    dst3 = cv2.cvtColor(dst_c, cv2.COLOR_GRAY2BGR)
    mask_roi = (dil[sy0:sy1, sx0:sx1] * 255).astype(np.uint8)
    p = (gx0 - rx0 + bw // 2, gy0 - ry0 + bh // 2)
    try:
        out3 = cv2.seamlessClone(src3, dst3, mask_roi, p, cv2.NORMAL_CLONE)
    except cv2.error:
        return None

    oh, ow = m_float.shape
    ox, oy = int(off[0]), int(off[1])
    iy0, ix0 = max(0, oy, ry0), max(0, ox, rx0)
    iy1, ix1 = min(H, oy + oh, ry1), min(W, ox + ow, rx1)
    if iy1 <= iy0 or ix1 <= ix0:
        return None
    lbl_sub = label[iy0 - oy:iy1 - oy, ix0 - ox:ix1 - ox]
    if not lbl_sub.any():
        return None

    # ② 残差 = 泊松解(×2 还原) 与 小图原始差 各半
    delta_c = 2.0 * (out3[iy0 - ry0:iy1 - ry0, ix0 - rx0:ix1 - rx0, 0].astype(np.float32)
                     - dst_c[iy0 - ry0:iy1 - ry0, ix0 - rx0:ix1 - rx0].astype(np.float32))
    src_d = patch[iy0 - oy:iy1 - oy, ix0 - ox:ix1 - ox].astype(np.float32) - src_bg
    res = 0.50 * delta_c + 0.50 * src_d

    roi_g = gray[iy0:iy1, ix0:ix1].astype(np.float32)
    z_sub = zone_win[iy0:iy1, ix0:ix1] if zone_win is not None else None
    # ③ 写回掩码：label ∧ 有效检测区；label=0 处权重恒 0
    if z_sub is not None:
        eval_mask = lbl_sub & z_sub
        if not eval_mask.any():
            return None
    else:
        eval_mask = lbl_sub

    # ④ 极性强制
    if eval_mask.any():
        if polarity < 0:
            if float(np.median(res[eval_mask])) > 0.0:
                res = -res
            res = np.minimum(res, 4.0)
        elif polarity > 0:
            if float(np.median(res[eval_mask])) < 0.0:
                res = -res
            res = np.maximum(res, -4.0)

    # ⑤ 距离变换 α 环：外圈 1px 0.55，≥2px 满幅
    dt_lbl = cv2.distanceTransform(label.astype(np.uint8), cv2.DIST_L2, 3)
    dt_sub = dt_lbl[iy0 - oy:iy1 - oy, ix0 - ox:ix1 - ox]
    alpha_in = np.clip(0.55 + 0.45 * (dt_sub - 1.0), 0.0, 1.0)
    alpha_in[~eval_mask] = 0.0

    # ⑥ 目标幅值重标定
    if target_diff is not None and target_diff > 0 and eval_mask.any():
        eff_abs = np.abs((alpha_in * res)[eval_mask])
        cur_mag = 0.50 * float(np.percentile(eff_abs, 50)) + 0.50 * float(np.percentile(eff_abs, 75))
        if cur_mag >= 1.0:
            scale_f = float(np.clip(float(target_diff) / cur_mag, 0.50, 6.0))
            res = res * scale_f

    # ⑦ 暗区孔洞门控：背景 ≤20 的暗区不许提亮
    hole_gate = np.clip((roi_g - 20.0) / 35.0, 0.0, 1.0)
    res = np.where(res > 0.0, res * hole_gate, res)

    out_sub = np.clip(roi_g + alpha_in * res, 0.0, 255.0)

    # ⑧ 贴边缺陷的区外 carve/built 过渡（只动所有区并集之外，恒 label=255 区）
    u_sub = union_win[iy0:iy1, ix0:ix1] if union_win is not None else z_sub
    if z_sub is not None and u_sub is not None and eval_mask.any() and not np.all(u_sub):
        outside_all_zones = ~u_sub
        out_def = lbl_sub & outside_all_zones
        if not out_def.any():
            ker3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dil_eval = (cv2.dilate(eval_mask.astype(np.uint8), ker3) > 0) & outside_all_zones
            if dil_eval.any():
                out_def = dil_eval
        if out_def.any():
            r_mode = rng.random()
            if r_mode >= 0.30:
                w_ext = rng.uniform(0.35, 0.85) if r_mode < 0.70 else 1.0
                alpha_out = np.clip(0.55 + 0.45 * (dt_sub - 1.0), 0.0, 1.0)
                dst_clean_sub = dst_clean_roi[iy0 - ry0:iy1 - ry0,
                                              ix0 - rx0:ix1 - rx0].astype(np.float32)
                def_surf = np.clip(dst_clean_sub + alpha_out * res, 0.0, 255.0)
                def_med_val = float(np.median(out_sub[eval_mask]))
                if def_med_val < dst_med - 10.0:        # 暗缺陷：区外向下 carve
                    carved = np.minimum(roi_g, def_surf)
                    if r_mode >= 0.70:
                        out_vals = dst_roi[~z_roi] if (zone_win is not None and np.any(~z_roi)) \
                            else roi_g[~z_sub]
                        dark_out = out_vals[out_vals < dst_med - 25.0]
                        if len(dark_out) > 0:
                            ext_med = float(np.median(dark_out))
                            carved = np.minimum(carved, max(
                                ext_med, float(np.percentile(out_sub[eval_mask], 25))))
                    out_sub[out_def] = (1.0 - w_ext) * roi_g[out_def] + w_ext * carved[out_def]
                elif def_med_val > dst_med + 10.0:      # 亮缺陷：区外向上 built
                    built = np.maximum(roi_g, def_surf)
                    out_sub[out_def] = (1.0 - w_ext) * roi_g[out_def] + w_ext * built[out_def]

    blended = gray.copy()
    blended[iy0:iy1, ix0:ix1] = out_sub.astype(np.uint8)
    return blended, label


def naive_clone(gray, patch, m_float, off=(0, 0)):
    """朴素 seamlessClone 对照组：直接整图克隆——光晕泄漏的现场。"""
    H, W = gray.shape
    h, w = patch.shape[:2]
    ox, oy = off
    sy, sx = max(0, -oy), max(0, -ox)
    ey, ex = min(h, H - oy), min(w, W - ox)
    canvas = gray.copy()
    canvas[oy + sy:oy + ey, ox + sx:ox + ex] = \
        gray[oy + sy:oy + ey, ox + sx:ox + ex]
    src3 = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    dst3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask_roi = ((m_float > 0.35).astype(np.uint8) * 255)[sy:ey, sx:ex]
    center = (ox + sx + (ex - sx) // 2, oy + sy + (ey - sy) // 2)
    out3 = cv2.seamlessClone(src3, dst3, mask_roi, center, cv2.NORMAL_CLONE)
    return cv2.cvtColor(out3, cv2.COLOR_BGR2GRAY)


def make_scene(seed: int = 0, h: int = 160, w: int = 160):
    """亮底图 + 有效检测区 + 一块暗缺陷小图（左上角为去背景中值）。"""
    rng = np.random.default_rng(seed)
    gray = np.full((h, w), 150.0, dtype=np.float32)
    gray += rng.normal(0, 1.5, gray.shape)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    zone = np.zeros((h, w), dtype=bool)
    zone[30:130, 30:130] = True

    ph = pw = 36
    patch = np.full((ph, pw), 150.0, dtype=np.uint8)     # 背景与底图同值
    mask = np.zeros((ph, pw), dtype=np.float32)
    cv2.circle(mask, (pw // 2, ph // 2), 10, 1.0, -1)    # 圆形缺陷
    cv2.circle(patch, (pw // 2, ph // 2), 10, 60.0, -1)  # 暗缺陷 (Δ≈90)
    return gray, zone, patch, mask


def label_on_canvas(label: np.ndarray, off: tuple, shape) -> np.ndarray:
    """把小图局部 label 铺回整图画布（paste_poisson 返回的 label 是 patch 局部形状）。"""
    canvas = np.zeros(shape, dtype=bool)
    oy, ox = off
    h, w = label.shape
    sy, sx = max(0, -oy), max(0, -ox)
    ey, ex = min(h, shape[0] - oy), min(w, shape[1] - ox)
    canvas[max(0, oy):oy + ey, max(0, ox):ox + ex] = label[sy:ey, sx:ex]
    return canvas


if __name__ == "__main__":
    import random

    gray, zone, patch, mask = make_scene()
    off = (60, 60)                                       # 贴在区中心
    zone_win = zone.copy()

    naive = naive_clone(gray, patch, mask, off)
    changed_naive = (naive != gray)
    full_mask = label_on_canvas(mask > 0.30, off, gray.shape)
    halo = int((changed_naive & ~full_mask).sum())
    rng = random.Random(0)
    got = paste_poisson(gray.copy(), patch, mask, rng, off=off, zone_win=zone_win,
                        polarity=-1, target_diff=90.0)
    out, label = got
    full_label = label_on_canvas(label, off, gray.shape)
    diff = (out.astype(int) - gray.astype(int))

    leak = int(np.count_nonzero(diff[~full_label]))
    in_zone = full_label & zone
    visible = float(np.abs(diff[in_zone]).mean())
    center = (78, 78)
    print(f"朴素 seamlessClone: 掩码外泄漏像素 {halo} 个 (泊松光晕)；且平坦缺陷内部被解成背景 "
          f"(中心 {naive[center]} ≈ 背景值，而非缺陷值 60)")
    print(f"零光晕版: label 掩码外改动像素 = {leak} 个 (恒 0 —— 标签 0 的背景 100% 不动)")
    print(f"零光晕版缺陷中心 {out[center]} (变暗 {gray[center] - out[center]:.0f})；"
          f"区内可见对比度 |ΔI| 均值 = {visible:.1f} (目标 90, 残差各半混合保住缺陷内容)")
