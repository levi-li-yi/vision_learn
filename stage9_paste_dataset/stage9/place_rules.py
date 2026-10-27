"""课时 2：区内/区边缘五五分放置 + 可见性复检 + 上下文镜像窗口 —— dataset.py 放置规则。

缺陷贴在哪不是随便选的——贴图位置的分布直接决定模型学到什么：

  · 五五分（BOUNDARY_PLACE_PROB=0.5）：50% 贴在**有效检测区内部**（对准缺陷质心，
    d ≥ min(d_max/2, 0.75·r_eq) 的"深处"，保证缺陷完整落区内）；50% 贴在**区边缘带**
    （d ≤ 2.5px 的边界像素上，掩码随机像素对准），越出区的部分被 zone 自然截断——
    真实产线缺陷就是长在边缘上的，只学完整落区的缺陷会漏检边缘缺陷；
  · 最小区内像素门槛：边缘模式 ≥ max(8, 0.15·缺陷面积)，区内模式 ≥ max(8, 0.75·缺陷面积)
    ——截断后只剩 2px 的贴图没有训练价值；
  · 可见性复检：贴完后缺陷掩码内 |ΔI| 均值 ≥ 8 或 p75 ≥ 12 才收（VISIBLE_MIN_DIFF），
    区内占比门槛边缘模式 0.15 / 区内模式 0.70——泊松解偶发"贴了个寂寞"要扔掉重试；
  · 上下文镜像窗口：泊松方程需要边界条件，切块四周外扩 margin 镜像补齐（REFLECT_101），
    在扩展窗口里求解再裁回 512——否则块边缘的泊松解被硬边界扭曲出接缝。

对应主仓库：src/train/dataset.py::try_place / context_window / _paste_one_zone
（可见性复检段）/ rotate_patch / sample_target / solve_scale。

运行：python -m stage9.place_rules
"""
import math
import random
from typing import Optional, Tuple

import cv2
import numpy as np

BOUNDARY_PLACE_PROB = 0.50   # 区内与区边缘五五分
MIN_IN_ZONE_PX = 8           # 截断后留在区内的最小像素
VISIBLE_MIN_DIFF = 8.0       # 可见性复检下限
SCALE_MAX = 2.0


def solve_scale(cur: float, target: float, typ: str = "area") -> float:
    """按目标面积解缩放倍率：开根号（面积类）。"""
    if cur <= 0:
        return 0.0
    return (target / cur) ** 0.5


def sample_target(rng, lo: float, hi: float, med: Optional[float] = None) -> float:
    """规格内中位数锚定对数均匀抽样：50% 抽 [lo, med]、50% 抽 [med, hi]。"""
    if hi <= lo:
        return float(lo)
    if med is not None and lo < med < hi:
        if rng.random() < 0.5:
            return math.exp(rng.uniform(math.log(lo), math.log(med)))
        return math.exp(rng.uniform(math.log(med), math.log(hi)))
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def rotate_patch(rng, patch, mask):
    """随机旋转 0~360°（外接框随转扩边，图与掩码同转不缩放）。"""
    ang = rng.uniform(0, 360)
    h, w = patch.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos) + 1
    nh = int(h * cos + w * sin) + 1
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    p = cv2.warpAffine(patch, M, (nw, nh), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    m = cv2.warpAffine(mask.astype(np.float32), M, (nw, nh),
                       flags=cv2.INTER_LINEAR, borderValue=0)
    return p, m


def try_place(rng, mask_bool: np.ndarray, zone_tile: np.ndarray,
              zy: np.ndarray, zx: np.ndarray, d_vals: np.ndarray, d_max: float,
              boundary_prob: float = BOUNDARY_PLACE_PROB
              ) -> Optional[Tuple[int, int, bool]]:
    """五五分放置：返回 (oy, ox, is_b_mode) 或 None。逐行复刻 dataset.py::try_place。"""
    ph, pw = mask_bool.shape
    ts = zone_tile.shape[0]
    if len(zy) == 0 or d_max <= 0.0:
        return None

    mys, mxs = np.nonzero(mask_bool)
    n_mask_px = len(mys)
    if n_mask_px == 0:
        return None
    my_c = int(round(float(mys.mean())))
    mx_c = int(round(float(mxs.mean())))

    m_area = float(n_mask_px)
    r_eq = max(1.0, (m_area / math.pi) ** 0.5)           # 等效圆半径

    # 区比缺陷还窄（d_max < 0.7·r_eq）时强制边缘模式
    is_b_mode = (d_max < 0.7 * r_eq) or (rng.random() < boundary_prob)

    if is_b_mode:
        b_idx = np.flatnonzero(d_vals <= 2.5)            # 边缘带：距边界 ≤ 2.5px
        cand_ys, cand_xs = (zy[b_idx], zx[b_idx]) if len(b_idx) > 0 else (zy, zx)
        n_cand = len(cand_ys)
        if n_cand == 0:
            return None
        min_in_px = max(MIN_IN_ZONE_PX, int(0.15 * m_area))
        for _attempt in range(12):
            k = rng.randrange(n_cand)
            cy, cx = int(cand_ys[k]), int(cand_xs[k])
            p_idx = rng.randrange(n_mask_px)             # 掩码随机像素对准边缘点
            my_a, mx_a = int(mys[p_idx]), int(mxs[p_idx])
            oy, ox = cy - my_a, cx - mx_a
            sy, sx = max(0, -oy), max(0, -ox)
            ey, ex = min(ph, ts - oy), min(pw, ts - ox)
            if ey <= sy or ex <= sx:
                continue
            inside = mask_bool[sy:ey, sx:ex] & zone_tile[oy + sy:oy + ey, ox + sx:ox + ex]
            if int(inside.sum()) >= min_in_px:
                return oy, ox, True
        return None
    else:
        lo_d = min(d_max * 0.5, 0.75 * r_eq)             # 深处：离边足够远
        i_idx = np.flatnonzero(d_vals >= lo_d)
        cand_ys, cand_xs = (zy[i_idx], zx[i_idx]) if len(i_idx) > 0 else (zy, zx)
        n_cand = len(cand_ys)
        if n_cand == 0:
            return None
        min_in_px = max(MIN_IN_ZONE_PX, int(0.75 * m_area))
        for _attempt in range(12):
            k = rng.randrange(n_cand)
            cy, cx = int(cand_ys[k]), int(cand_xs[k])
            oy, ox = cy - my_c, cx - mx_c                # 质心对准
            sy, sx = max(0, -oy), max(0, -ox)
            ey, ex = min(ph, ts - oy), min(pw, ts - ox)
            if ey <= sy or ex <= sx:
                continue
            inside = mask_bool[sy:ey, sx:ex] & zone_tile[oy + sy:oy + ey, ox + sx:ox + ex]
            if int(inside.sum()) >= min_in_px:
                return oy, ox, False
        return None


def context_window(img: np.ndarray, x: int, y: int, ts: int, margin: int) -> np.ndarray:
    """切块 (x, y, ts) 四周外扩 margin 的窗口；越出整图部分镜像补齐（REFLECT_101）。"""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = x - margin, y - margin, x + ts + margin, y + ts + margin
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    win = img[cy0:cy1, cx0:cx1]
    if (cx0, cy0, cx1, cy1) == (x0, y0, x1, y1):
        return win
    return cv2.copyMakeBorder(win, cy0 - y0, y1 - cy1, cx0 - x0, x1 - cx1,
                              cv2.BORDER_REFLECT_101)


def visibility_check(result, gray, in_mask):
    """贴后可见性复检：|ΔI| 均值 ≥ 8 或 p75 ≥ 12；否则该次贴图作废。"""
    diff_sub = np.abs(result.astype(np.int16) - gray.astype(np.int16))[in_mask]
    if len(diff_sub) == 0:
        return False
    return not (float(diff_sub.mean()) < VISIBLE_MIN_DIFF
                and float(np.percentile(diff_sub, 75)) < VISIBLE_MIN_DIFF * 1.5)


def make_zone(seed: int = 0, h: int = 200, w: int = 200):
    """合成有效检测区（矩形区）与预计算的距离变换量。"""
    zone = np.zeros((h, w), dtype=bool)
    zone[40:160, 40:160] = True
    zone_u8 = zone.astype(np.uint8)
    dist_in = cv2.distanceTransform(zone_u8, cv2.DIST_L2, 3)
    zy, zx = np.nonzero(zone)
    d_vals = dist_in[zy, zx]
    return zone, zy, zx, d_vals, float(d_vals.max())


def make_defect_patch(ph: int = 24, pw: int = 40):
    """一块长条划痕小图（0/255 掩码 + 灰度）。"""
    patch = np.full((ph, pw), 150, dtype=np.uint8)
    mask = np.zeros((ph, pw), dtype=bool)
    mask[8:16, 4:36] = True
    patch[mask] = 60
    return patch, mask


if __name__ == "__main__":
    zone, zy, zx, d_vals, d_max = make_zone()
    patch, mask = make_defect_patch()
    rng = random.Random(3)

    n_b = n_i = 0
    placements = []
    for _ in range(200):
        got = try_place(rng, mask, zone, zy, zx, d_vals, d_max)
        if got is None:
            continue
        oy, ox, is_b = got
        n_b += is_b
        n_i += not is_b
        ph, pw = mask.shape
        region = zone[oy:oy + ph, ox:ox + pw]
        frac_in = (mask & region).sum() / mask.sum()
        placements.append((is_b, frac_in))
    print(f"200 次放置: 边缘模式 {n_b} / 区内模式 {n_i} (≈五五分)")
    edge_fracs = [f for b, f in placements if b]
    inner_fracs = [f for b, f in placements if not b]
    print(f"边缘模式区内占比: 中位 {sorted(edge_fracs)[len(edge_fracs) // 2]:.2f} (可截断)")
    print(f"区内模式区内占比: 最小 {min(inner_fracs):.2f} (≥0.75 门槛保证基本完整)")

    # 可见性复检：贴了个寂寞（Δ<8）会被拒收
    gray = np.full((60, 60), 150, dtype=np.uint8)
    faint = gray.copy()
    faint[20:30, 20:30] = 147                     # Δ=3 的"隐形"缺陷
    m = np.zeros((60, 60), dtype=bool)
    m[20:30, 20:30] = True
    print(f"隐形缺陷可见性复检 = {visibility_check(faint, gray, m)} (Δ=3 < 8 拒收)")
    print(f"可见缺陷可见性复检 = {visibility_check(gray.copy() - 60, gray, m)} (Δ=60 收)")

    # 上下文镜像窗口：块贴着整图左上角
    img = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)
    win = context_window(img, 0, 0, 50, 10)
    print(f"上下文窗口形状 {win.shape} = 50 + 2×10 (越界部分镜像补齐)")
