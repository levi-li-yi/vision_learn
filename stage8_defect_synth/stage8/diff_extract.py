"""课时 4：弹性模板形变 + 余弦羽化 + 前后差分滞后阈值提纯 —— 合成缺陷的最后一公里。

LoRA 在干净底图上"重绘"出缺陷后，怎么拿到**像素级掩码答案**？——对比重绘前后：

  1. 弹性形变模板（deform_and_scale_template）：真实缺陷掩码做形状种子——
     随机翻转/旋转/预缩放后，用高斯模糊随机位移场做弹性扭曲
     (amp=clip(短轴×0.15, 1.2, 5.5), σ=clip(长轴×0.20, 4, 18))，
     再按对数均匀采样目标面积定缩放 (s=clip(√(目标/当前), 0.5, 2.0))；
  2. 余弦羽化融合（cosine_feather_blend）：α = 0.5 − 0.5·cos(πt)，t = 距边距离/4px——
     边缘平滑过渡不带硬茬，区外严格原样；
  3. 差分与噪声尺（compute_diff_and_sigma）：delta = 0.75·|Δ灰度| + 0.25·|ΔSobel 梯度|；
     噪声 σ 用背景 MAD 估计 σ=1.4826·median(|x−中值|)（抠门但无偏），
     环带 σ 取 med+0.5·MAD，最终 σ = clip(max(σ_grain, 0.65·σ_ring), 2, 12)；
  4. 滞后阈值提纯（extract_defect_mask_hysteresis）：3.5σ 高阈值出种子、1.8σ 低阈值
     连通域生长，且必须同时"含种子 ∧ 碰到模板容差区"才收；大缺陷兜底整区填充；
  5. 质检门槛（Step4）：面积 ∈ [max(6, 0.15·core), 2.5·core]、peak_snr ≥ 3.5、
     面积×训练缩放² ≥ 规格下限；最多 3 轮差额补齐（首轮 1.15 / 续轮 1.25 超采样）。

对应主仓库：src/train/lora_synth.py::deform_and_scale_template / cosine_feather_blend /
compute_diff_and_sigma / extract_defect_mask_hysteresis /
_run_inpaint_and_extract_for_zone_defect（Step4 门槛与 MAX_REFILL_ROUNDS）。

运行：python -m stage8.diff_extract
"""
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. 弹性形变模板
# ---------------------------------------------------------------------------
def deform_and_scale_template(tpl_mask, area_lo, area_hi, rng,
                              tile_size=512, dilate_px=8):
    """掩码模板 → 翻转/旋转/弹性扭曲/按目标面积缩放（公式逐项对应真工程）。"""
    m = (tpl_mask > 127).astype(np.uint8) * 255
    if rng.random() < 0.5:
        m = cv2.flip(m, 1)
    if rng.random() < 0.5:
        m = cv2.flip(m, 0)
    pre_scale = float(rng.uniform(0.90, 1.10))
    angle = float(rng.uniform(0.0, 360.0))
    h, w = m.shape[:2]
    diag = int(np.ceil(np.hypot(h, w) * max(1.0, pre_scale))) + 24
    canvas = np.zeros((diag, diag), dtype=np.uint8)
    canvas[(diag - h) // 2:(diag - h) // 2 + h,
           (diag - w) // 2:(diag - w) // 2 + w] = m

    rot_mat = cv2.getRotationMatrix2D((diag / 2.0, diag / 2.0), angle, pre_scale)
    warped = cv2.warpAffine(canvas, rot_mat, (diag, diag),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    ys, xs = np.where(warped > 127)
    if len(xs) < 4:
        return None

    bw = max(1, int(xs.max() - xs.min() + 1))
    bh = max(1, int(ys.max() - ys.min() + 1))
    amp = float(np.clip(min(bw, bh) * 0.15, 1.2, 5.5))       # 弹性幅度 ∝ 短轴
    sigma = float(np.clip(max(bw, bh) * 0.20, 4.0, 18.0))    # 平滑尺度 ∝ 长轴
    ksize = int(sigma * 4) | 1

    dx = cv2.GaussianBlur(rng.standard_normal((diag, diag)).astype(np.float32),
                          (ksize, ksize), sigma)
    dy = cv2.GaussianBlur(rng.standard_normal((diag, diag)).astype(np.float32),
                          (ksize, ksize), sigma)
    dx = dx / (np.std(dx) + 1e-6) * amp
    dy = dy / (np.std(dy) + 1e-6) * amp

    grid_y, grid_x = np.meshgrid(np.arange(diag, dtype=np.float32),
                                 np.arange(diag, dtype=np.float32), indexing="ij")
    deformed = cv2.remap(warped, grid_x + dx, grid_y + dy,
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    deformed_bin = (deformed > 127).astype(np.uint8) * 255
    ys2, xs2 = np.where(deformed_bin > 0)
    if len(xs2) < 4:
        return None

    cropped = deformed_bin[ys2.min():ys2.max() + 1, xs2.min():xs2.max() + 1]
    target_area = float(np.exp(rng.uniform(np.log(area_lo), np.log(area_hi))))
    cur_area = float(np.count_nonzero(cropped))
    if cur_area <= 0:
        return None
    s = float(np.clip((target_area / cur_area) ** 0.5, 0.50, 2.0))
    max_side = tile_size - 2 * (dilate_px + 6)
    if max(cropped.shape) * s > max_side:
        s = float(max_side) / float(max(cropped.shape))
    nh = max(3, int(round(cropped.shape[0] * s)))
    nw = max(3, int(round(cropped.shape[1] * s)))
    if (nh, nw) != cropped.shape[:2]:
        cropped = (cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_LINEAR) > 127
                   ).astype(np.uint8) * 255
    if np.count_nonzero(cropped) < 4:
        return None
    return np.pad(cropped, 4, mode="constant")


# ---------------------------------------------------------------------------
# 2. 余弦羽化融合
# ---------------------------------------------------------------------------
def cosine_feather_blend(p_clean, p_gen, m_local, feather_px: int = 4):
    """α = 0.5 − 0.5·cos(π·t)，t = 掩码内距离变换/羽化宽度；区外严格原样。"""
    mask_bin = (m_local > 127).astype(np.uint8)
    if feather_px <= 0:
        out = p_clean.copy()
        out[mask_bin > 0] = p_gen[mask_bin > 0]
        return out
    dist_in = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
    t = np.clip(dist_in / float(feather_px), 0.0, 1.0)
    alpha = 0.5 - 0.5 * np.cos(np.pi * t)
    alpha[mask_bin == 0] = 0.0
    blended = alpha * p_gen.astype(np.float32) + (1.0 - alpha) * p_clean.astype(np.float32)
    out = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    out[mask_bin == 0] = p_clean[mask_bin == 0]
    return out


# ---------------------------------------------------------------------------
# 3. 差分与噪声尺
# ---------------------------------------------------------------------------
def compute_diff_and_sigma(p_clean, p_final, m_core, m_local, crop_zone=None):
    """delta = 0.75|Δgray| + 0.25|Δgrad|；σ 用 MAD 估计，环带折半封顶钳 [2, 12]。"""
    c_f = p_clean.astype(np.float32)
    f_f = p_final.astype(np.float32)

    d_gray = np.abs(f_f - c_f)
    gx_c = cv2.Sobel(c_f, cv2.CV_32F, 1, 0, ksize=3)
    gy_c = cv2.Sobel(c_f, cv2.CV_32F, 0, 1, ksize=3)
    gx_f = cv2.Sobel(f_f, cv2.CV_32F, 1, 0, ksize=3)
    gy_f = cv2.Sobel(f_f, cv2.CV_32F, 0, 1, ksize=3)
    d_grad = 0.25 * np.hypot(gx_f - gx_c, gy_f - gy_c)
    delta = 0.75 * d_gray + 0.25 * d_grad

    zone_bool = (crop_zone > 0) if crop_zone is not None else np.ones_like(m_local, dtype=bool)
    blur_c = cv2.GaussianBlur(c_f, (5, 5), 1.2)
    ext_mask = (m_local == 0) & zone_bool
    if not np.any(ext_mask):
        ext_mask = m_local == 0
    if np.any(ext_mask):
        res_ext = np.abs(c_f[ext_mask] - blur_c[ext_mask])
        sigma_grain = 1.4826 * float(np.median(res_ext))     # MAD 噪声估计
    else:
        sigma_grain = 3.0

    ring_mask = (m_local > 0) & (m_core == 0) & zone_bool
    if np.count_nonzero(ring_mask) >= 16:
        ring_vals = delta[ring_mask]
        ring_med = float(np.median(ring_vals))
        ring_mad = 1.4826 * float(np.median(np.abs(ring_vals - ring_med)))
        sigma_ring = ring_med + 0.5 * ring_mad
    else:
        sigma_ring = sigma_grain

    sigma = float(np.clip(max(sigma_grain, sigma_ring * 0.65), 2.0, 12.0))
    return delta, sigma


# ---------------------------------------------------------------------------
# 4. 滞后阈值提纯
# ---------------------------------------------------------------------------
def extract_defect_mask_hysteresis(delta, sigma, m_core, m_local, crop_zone=None,
                                   snr_high: float = 3.5, snr_low: float = 1.8):
    """3.5σ 种子 + 1.8σ 连通域生长 + 模板容差区约束；大缺陷整区填充兜底。"""
    zone_bool = (crop_zone > 0) if crop_zone is not None else np.ones_like(m_local, dtype=bool)
    ker_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    valid_region = (cv2.erode((m_local > 0).astype(np.uint8) * 255, ker_erode) > 0) & zone_bool
    if not np.any(valid_region):
        valid_region = (m_local > 0) & zone_bool

    t_high = snr_high * sigma
    t_low = snr_low * sigma
    high_seeds = (delta >= t_high) & valid_region
    low_cands = (delta >= t_low) & valid_region
    if not np.any(high_seeds):
        return np.zeros_like(m_local, dtype=np.uint8), 0.0

    ker_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    core_tol = (cv2.dilate((m_core > 0).astype(np.uint8) * 255, ker_core) > 0) & zone_bool

    num_labels, labels = cv2.connectedComponents(low_cands.astype(np.uint8), connectivity=8)
    m_gt = np.zeros_like(m_local, dtype=np.uint8)
    for lbl in range(1, num_labels):
        comp = labels == lbl
        if np.any(comp & high_seeds) and np.any(comp & core_tol):
            m_gt[comp] = 255

    m_gt[~zone_bool] = 0
    if not np.any(m_gt):
        return m_gt, 0.0
    m_gt = cv2.morphologyEx(m_gt, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    gt_vals = delta[m_gt > 0]
    peak_snr = float(np.percentile(gt_vals, 95) / max(1e-4, sigma)) if len(gt_vals) else 0.0

    # 大缺陷兜底：模板核心占检测区 70% 以上 → 整区填充（防漏成斑点）
    core_in_zone = (m_core > 0) & zone_bool
    core_in_area = int(np.count_nonzero(core_in_zone))
    if core_in_area >= 16 and peak_snr >= snr_high:
        num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
            zone_bool.astype(np.uint8), connectivity=8)
        overlap_comp_area = 0
        for cid in range(1, num_cc):
            if np.any((cc_labels == cid) & core_in_zone):
                overlap_comp_area += int(cc_stats[cid, cv2.CC_STAT_AREA])
        if overlap_comp_area > 0 and core_in_area >= 0.70 * overlap_comp_area:
            ker_hole = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            closed = cv2.morphologyEx(m_gt, cv2.MORPH_CLOSE, ker_hole)
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(m_gt)
            if cnts:
                cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
            m_gt = np.where(core_in_zone & (filled > 0), 255, m_gt).astype(np.uint8)

    m_gt[~zone_bool] = 0
    return m_gt, peak_snr


# ---------------------------------------------------------------------------
# 5. 质检门槛
# ---------------------------------------------------------------------------
def quality_gate(m_gt, peak_snr, area_core, snr_high: float = 3.5,
                 train_scale_max: float = 1.3, spec_min: float = 18.0):
    """Step4 三门：面积带、信噪比、面积×训练缩放² ≥ 规格下限。"""
    area_gt = int(np.count_nonzero(m_gt))
    if area_gt < max(6, int(area_core * 0.15)) or peak_snr < snr_high \
            or area_gt > int(area_core * 2.5):
        return False, area_gt
    if float(area_gt) * (train_scale_max ** 2) < spec_min:
        return False, area_gt
    return True, area_gt


MAX_REFILL_ROUNDS = 3        # 对应 MAX_REFILL_ROUNDS：最多 3 轮差额补齐
REFILL_MARGIN = {0: 1.15, 1: 1.25, 2: 1.25}   # 首轮 1.15，续轮 1.25 超采样


def make_synthetic_case(seed: int = 0, h: int = 200, w: int = 256):
    """合成一次"重绘前后"：干净底图 + 模板形状的暗缺陷（替身 FLUX 重绘结果）。"""
    rng = np.random.default_rng(seed)
    clean = np.full((h, w), 150.0, dtype=np.float32)
    clean += rng.normal(0, 2.0, clean.shape)               # 底图噪声（σ≈2）
    clean = np.clip(clean, 0, 255).astype(np.uint8)

    tpl = np.zeros((40, 90), dtype=np.uint8)
    cv2.rectangle(tpl, (5, 12), (85, 28), 255, -1)         # "划痕"长条模板
    deformed = deform_and_scale_template(tpl, 300, 600, rng)

    mask = np.zeros((h, w), dtype=np.uint8)
    mh, mw = deformed.shape
    y0, x0 = 80, 80
    mask[y0:y0 + mh, x0:x0 + mw] = deformed
    gen = clean.copy()
    gen[mask > 0] = (gen[mask > 0].astype(np.float32) * 0.45).astype(np.uint8)  # 暗缺陷
    gen = np.clip(gen + rng.normal(0, 2.0, gen.shape), 0, 255).astype(np.uint8)

    # 真实管线同构：m_core = 模板形状，m_local = 模板外扩 8px 的重绘区
    # （滞后阈值的 5px 腐蚀落在环带上，核心不被吃掉）
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * 8 + 1, 2 * 8 + 1))
    m_core = mask
    m_local = cv2.dilate(mask, ker)

    zone = np.zeros((h, w), dtype=np.uint8)
    zone[20:180, 20:230] = 255
    return clean, gen, m_core, m_local, zone


if __name__ == "__main__":
    clean, gen, m_core, m_local, zone = make_synthetic_case()
    blended = cosine_feather_blend(clean, gen, m_local, feather_px=4)
    delta, sigma = compute_diff_and_sigma(clean, blended, m_core, m_local, zone)
    m_gt, peak_snr = extract_defect_mask_hysteresis(delta, sigma, m_core, m_local, zone)
    ok, area = quality_gate(m_gt, peak_snr, int((m_core > 0).sum()))

    inter = ((m_gt > 0) & (m_core > 0)).sum()
    union = ((m_gt > 0) | (m_core > 0)).sum()
    print(f"噪声尺 σ={sigma:.2f} (底图真实噪声≈2, MAD 估计接近) | "
          f"peak_snr={peak_snr:.1f} (≥3.5 通过)")
    print(f"提纯掩码 vs 真值 IoU = {inter / max(1, union):.3f} | "
          f"面积 {area}px (真值 {int((m_core > 0).sum())}px) | 质检 {'通过' if ok else '拒绝'}")
    outside = (blended[(m_local == 0) & (zone > 0)] != clean[(m_local == 0) & (zone > 0)]).mean()
    print(f"羽化后重绘区外改动比例 {outside:.2%} (余弦羽化只动重绘区内)")
