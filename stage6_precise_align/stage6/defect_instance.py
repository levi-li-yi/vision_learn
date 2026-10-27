"""课时 4：真实缺陷实例解析 —— 归属 / 环带对比度 / 去背景导出 / 规格解算 / 核心提纯。

逆向对齐之后，每张 NG 图上的每个手工标注多边形要变成可训练的"真实缺陷实例"。
本课覆盖 align.py / assets.py 中这一段的全部口径：

  1. 最大交集归属：缺陷与各"圆滑有效检测区"（smooth 版，防贴边缺陷被向内吸附出区）
     求交集，面积最大者为归属区；全部 < 4px 时回退用"未吸附投射分区"再试一次；
     外接框局部栅格化（子窗口 fillPoly）避免整图分配；
  2. 环形带对比度：缺陷掩码 dilate15 − dilate5 得环带 ∧ 无其他标注 ∧ 区内，
     环带背景中值 bg，对比度 = p80(|缺陷灰度 − bg|)（80 分位抗椒盐），≥ 6 才有效；
  3. 去背景导出：外扩 8px 裁剪，掩码外填"所属有效检测区干净背景中值"
     （排除全部标注膨胀区后取中值）——real_*.png 与 _mask.png 的诞生过程；
  4. 规格自动解算：面积 [max(18, p2×0.75), max(lo×1.5, p98×1.30)]、
     对比度 [max(16, p10×0.85), min(195, ...)]、b_prob=0.5（区内/区边缘五五分）；
  5. 核心提纯：|patch − bg| ≥ max(6, 0.28·p85) 的核心像素再 3×3 膨胀回约束掩码
     （去半透明边缘），minAreaRect 求长宽比。

对应主仓库：src/train/align.py::_extract_product_defect_instances /
auto_bind_zones(zd_specs 段)、src/train/assets.py::refine_asset_patch_and_mask。

运行：python -m stage6.defect_instance
"""
import cv2
import numpy as np

EXPORT_PAD_PX = 8      # 对应 export_pad_px=8：小缺陷图外扩 8px
RING_K_OUT = 15        # 环带外核 dilate15
RING_K_IN = 5          # 环带内核 dilate5（其他标注膨胀也用 k5）
MIN_RING_DIFF = 6.0    # 对应 diff_p80 >= 6.0 才计对比度
MIN_INTER_PX = 4       # 对应 best_area < 4px 跳过


def largest_intersection_zone(poly_pts: np.ndarray, zones: dict, shape,
                              fallback: dict = None):
    """最大交集归属：外接框局部栅格化（子窗口 fillPoly），返回 (zone 名, 交集面积)。

    真工程先对 smooth_zones（圆滑区）求归属，全部 < 4px 再对 proj_zones（未吸附投射区）
    兜底——贴边缺陷可能被训练吸附挤出区外，但圆滑区一定包住。
    """
    h, w = shape
    x0 = max(0, int(poly_pts[:, 0].min()) - 2)
    y0 = max(0, int(poly_pts[:, 1].min()) - 2)
    x1 = min(w, int(poly_pts[:, 0].max()) + 3)
    y1 = min(h, int(poly_pts[:, 1].max()) + 3)
    m_sub = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(m_sub, [poly_pts - [x0, y0]], 1)
    poly_bool = m_sub > 0
    if poly_bool.sum() < MIN_INTER_PX:
        return None, 0

    best_zone, best_area = None, 0
    for z, zm in zones.items():
        inter = int(np.count_nonzero(poly_bool & zm[y0:y1, x0:x1]))
        if inter > best_area:
            best_area, best_zone = inter, z
    if (best_zone is None or best_area < MIN_INTER_PX) and fallback:
        for z, zm in fallback.items():
            inter = int(np.count_nonzero(poly_bool & zm[y0:y1, x0:x1]))
            if inter > best_area:
                best_area, best_zone = inter, z
    return best_zone, best_area


def ring_contrast(gray: np.ndarray, poly_pts: np.ndarray, zone_mask: np.ndarray,
                  other_ann: np.ndarray = None):
    """环形带对比度：dilate15−dilate5 环带 ∧ 干净分区 → 背景中值 → p80(|ΔI|)。

    返回 None 表示测不出（环带全被标注占满 / 对比度 < 6）——真工程里该实例的
    abs_diff 记 None，规格解算时跳过它。
    """
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING_K_OUT, RING_K_OUT))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING_K_IN, RING_K_IN))
    h, w = gray.shape
    x0 = max(0, int(poly_pts[:, 0].min()) - RING_K_OUT)
    y0 = max(0, int(poly_pts[:, 1].min()) - RING_K_OUT)
    x1 = min(w, int(poly_pts[:, 0].max()) + RING_K_OUT + 1)
    y1 = min(h, int(poly_pts[:, 1].max()) + RING_K_OUT + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    mw = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(mw, [poly_pts - [x0, y0]], 1)
    z_win = zone_mask[y0:y1, x0:x1]
    ring = (cv2.dilate(mw, k15) > 0) & ~(cv2.dilate(mw, k5) > 0) & z_win
    if other_ann is not None:
        ring &= ~(other_ann[y0:y1, x0:x1] > 0)
    if mw.any() == 0 or not ring.any():
        return None
    sub_g = gray[y0:y1, x0:x1].astype(np.float32)
    bg_m = float(np.median(sub_g[ring]))
    diff_p80 = float(np.percentile(np.abs(sub_g[mw > 0] - bg_m), 80))
    return diff_p80 if diff_p80 >= MIN_RING_DIFF else None


def zone_background_median(gray, zone_mask, ann_mask=None):
    """有效检测区干净背景中值：排除全部标注膨胀区后取中值；全占满则退化为区内中值。"""
    clean = zone_mask & ((ann_mask == 0) if ann_mask is not None else True)
    if np.any(clean):
        return int(np.median(gray[clean]))
    if np.any(zone_mask):
        return int(np.median(gray[zone_mask]))
    return int(np.median(gray))


def export_real_patch(gray: np.ndarray, poly_pts: np.ndarray, zone_mask: np.ndarray,
                      ann_mask: np.ndarray = None, pad: int = EXPORT_PAD_PX):
    """去背景导出：外扩 pad 裁剪，掩码外填所属检测区干净背景中值 → (patch, mask)。"""
    h, w = gray.shape
    xs, ys = poly_pts[:, 0], poly_pts[:, 1]
    px0 = max(0, int(xs.min()) - pad)
    py0 = max(0, int(ys.min()) - pad)
    px1 = min(w, int(xs.max()) + 1 + pad)
    py1 = min(h, int(ys.max()) + 1 + pad)
    patch = gray[py0:py1, px0:px1].copy()
    z_crop = zone_mask[py0:py1, px0:px1]
    mask = np.zeros(patch.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [poly_pts - [px0, py0]], 255)
    ann_crop = ann_mask[py0:py1, px0:px1] if ann_mask is not None else None
    bg_pixels = z_crop & (mask == 0) & ((ann_crop == 0) if ann_crop is not None else True)
    if np.any(bg_pixels):
        bg_fill = int(np.median(patch[bg_pixels]))
    else:
        bg_fill = zone_background_median(gray, zone_mask, ann_mask)
    patch[mask == 0] = np.uint8(bg_fill)
    return patch, mask


def solve_zd_specs(instances: list) -> dict:
    """规格自动解算：由真实实例分位数解算面积/对比度区间与落位概率（auto_bind_zones 同款）。

    · 面积：lo = max(18, p2×0.75)、hi = max(lo×1.5, p98×1.30)、med 钳在区间内
    · 对比度：[max(16, p10×0.85), min(195, max(lo×1.35, p90×1.15))]
    · b_prob = 0.50：区内 / 区边缘五五分落位
    """
    areas = [float(r["area"]) for r in instances if r["area"] > 0]
    diffs = [float(r["abs_diff"]) for r in instances if r.get("abs_diff") is not None]
    p_lo = float(np.percentile(areas, 2)) * 0.75
    p_hi = float(np.percentile(areas, 98)) * 1.30
    lo = max(18.0, p_lo)
    hi = max(lo * 1.5, p_hi)
    med = float(np.clip(np.median(areas), lo, hi))
    if diffs:
        diff_lo = max(16.0, float(np.percentile(diffs, 10)) * 0.85)
        diff_hi = min(195.0, max(diff_lo * 1.35, float(np.percentile(diffs, 90)) * 1.15))
        diff_med = float(np.clip(np.median(diffs), diff_lo, diff_hi))
    else:
        diff_lo, diff_med, diff_hi = 25.0, 60.0, 135.0
    return {"min": round(lo, 1), "max": round(hi, 1), "med": round(med, 1),
            "diff_lo": round(diff_lo, 1), "diff_med": round(diff_med, 1),
            "diff_hi": round(diff_hi, 1), "b_prob": 0.50,
            "num_instances": len(instances)}


def refine_core(patch: np.ndarray, mask: np.ndarray):
    """掩码核心提纯（assets.refine_asset_patch_and_mask 迷你版）。

    |patch − 掩码内背景中值| ≥ max(6, 0.28·p85) 的像素为核心 → 3×3 膨胀回约束掩码；
    顺带返回 minAreaRect 长宽比（细长划痕 vs 团状砂眼的形态判别）。
    """
    bg = float(np.median(patch[mask > 0]))
    absd = np.abs(patch.astype(np.float32) - bg)
    thr = max(6.0, 0.28 * float(np.percentile(absd[mask > 0], 85)))
    core = ((absd >= thr) & (mask > 0)).astype(np.uint8)
    core = cv2.dilate(core, np.ones((3, 3), np.uint8)) & (mask > 0)
    if core.sum() < 4:
        core = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aspect = 1.0
    if cnts:
        (cx, cy), (lw, hh), ang = cv2.minAreaRect(cnts[0])
        aspect = float(max(lw, hh)) / max(1.0, float(min(lw, hh)))
    return core, aspect


def make_two_zone_scene(seed: int = 11, h: int = 240, w: int = 320):
    """合成"左加工面 / 右装配面"双区零件 + 3 个缺陷（划痕×2、砂眼×1）。

    返回 (gray, zones dict, smooth_zones dict, defects list, ann_mask)。
    defects: [{"label", "pts", "zone_true"}]——zone_true 用来对拍归属结果。
    """
    rng = np.random.default_rng(seed)
    gray = np.full((h, w), 30.0, dtype=np.float32)
    cv2.rectangle(gray, (20, 20), (150, h - 20), 150.0, -1)     # 左：加工面（亮）
    cv2.rectangle(gray, (170, 20), (w - 20, h - 20), 90.0, -1)  # 右：装配面（暗）
    zones, smooth = {}, {}
    zm_l = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(zm_l, (25, 25), (145, h - 25), 1, -1)          # 左区略内缩
    zones["加工面"] = zm_l.astype(bool)
    zm_r = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(zm_r, (175, 30), (w - 30, h - 30), 1, -1)      # 右区略内缩
    zones["装配面"] = zm_r.astype(bool)
    # 圆滑区 = 原区外扩 3px（模拟 smooth_for_infer 包边）
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for z, zm in zones.items():
        smooth[z] = cv2.dilate(zm.astype(np.uint8), k).astype(bool)
    gray += rng.normal(0, 1.2, gray.shape).astype(np.float32)

    defects, ann = [], np.zeros((h, w), dtype=np.uint8)
    scratch1 = np.array([[60, 80], [130, 90], [131, 97], [61, 88]], np.int32)   # 左区划痕
    scratch2 = np.array([[70, 160], [120, 168], [120, 175], [70, 167]], np.int32)  # 左区划痕2
    pore = np.array([[240, 120], [250, 120], [250, 130], [240, 130]], np.int32)  # 右区砂眼
    for lbl, pts in [("划痕", scratch1), ("划痕", scratch2), ("砂眼", pore)]:
        shade = 60.0 if lbl == "划痕" else 45.0
        cv2.fillPoly(gray, [pts], shade)
        cv2.fillPoly(ann, [pts], 1)
        defects.append({"label": lbl, "pts": pts,
                        "zone_true": "加工面" if pts[0, 0] < 160 else "装配面"})
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray, zones, smooth, defects, ann


if __name__ == "__main__":
    gray, zones, smooth, defects, ann = make_two_zone_scene()
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING_K_IN, RING_K_IN))
    ann_dil = cv2.dilate(ann, k5) > 0

    instances = []
    for d in defects:
        z, inter = largest_intersection_zone(d["pts"], smooth, gray.shape, fallback=zones)
        diff = ring_contrast(gray, d["pts"], smooth[z], other_ann=ann_dil)
        patch, mask = export_real_patch(gray, d["pts"], smooth[z], ann_mask=ann_dil)
        core, aspect = refine_core(patch, mask)
        area = int((mask > 0).sum())
        print(f"{d['label']}: 归属区={z} (交集 {inter}px, 真值 {d['zone_true']}) | "
              f"面积={area}px | 环带对比度 p80={diff if diff else 'None'} | "
              f"核心像素={int(core.sum())} | 长宽比={aspect:.2f} | 小图尺寸={patch.shape}")
        instances.append({"area": area, "abs_diff": diff})

    specs = solve_zd_specs(instances)
    print(f"\n规格自动解算: 面积∈[{specs['min']}, {specs['max']}] 中位={specs['med']} | "
          f"|ΔI|∈[{specs['diff_lo']}, {specs['diff_hi']}] | 区内/边缘={specs['b_prob']:.0%}")
