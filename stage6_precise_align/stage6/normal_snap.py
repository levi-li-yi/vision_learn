"""课时 2：±10px 法向剖面吸附（训练模式）—— compute_effective_zone 核心拆解（上）。

"有效检测区"是训练与推理共用的核心几何资产：把逆向投射后的分区边界，沿**轮廓法向**
在 ±10px 范围内搜索底图真实边缘的梯度峰值，把边界"吸附"到零件真实边缘上。
训练模式（smooth_for_infer=False）忠实紧贴边缘并**向内微收 1.5px 停在斜坡肩部**，
确保区内干净（贴缺陷训练时背景 100% 无缺陷）；推理模式（下节课）改为圆滑架桥。

本课逐段拆解 align.py::compute_effective_zone 的训练分支：

  1. 轮廓切向 → 法向：前后各 wt=min(12, n//4) 个点的差向量归一化为切向，旋转 90° 得法向；
  2. 法向剖面采样：轮廓每个点沿法向 ±10px 生成一条采样线，全部拼成 (n, 23) 的 map，
     用 cv2.remap 一次取回整张灰度剖面（分块 16000 防超 OpenCV SHRT_MAX 断言）；
  3. 法向方向自动定向：比较轮廓内外各 3px 处的分区均值，保证法向指向"区外"——
     防止吸附到对面暗缝的反向梯度沿；
  4. 定向梯度 + 距离衰减：中心差分 × 方向符号 × (1 − 0.015·|t|)，离中心越远权重越低；
  5. 训练吸附：峰值梯度 ≥ 4 才吸，落点 = 峰值 − 1.5px（斜坡肩部），滑窗中值平滑去毛刺；
  6. 层级轮廓还原：RETR_CCOMP 父子关系 + 面积从大到小 drawContours(FILLED)，
     外环 255 内环 0——带孔分区（圆环零件）不能被抹平。

对应主仓库：src/train/align.py::compute_effective_zone(not smooth_for_infer 分支) /
_remap_1d_profiles / (层级轮廓还原段)。

运行：python -m stage6.normal_snap
"""
import cv2
import numpy as np

SNAP_PX = 10        # 对应 align.py SNAP_PX
PEAK_GRAD_MIN = 4.0  # 对应 peak_grad >= 4.0 才吸附
SHOULDER_IN = 1.5    # 对应训练模式向内微收 1.5px（推理模式是 1.0px）
GRAD_DECAY = 0.015   # 对应权重 (1 - 0.015*|t|)


def contour_normals(pts: np.ndarray):
    """轮廓点切向/法向：前后各 wt=min(12, n//4) 点的差向量 → 切向，旋转 90° → 法向。"""
    n = len(pts)
    wt = min(12, n // 4)
    diffs = np.roll(pts, -wt, axis=0) - np.roll(pts, wt, axis=0)
    lengths = np.hypot(diffs[:, 0], diffs[:, 1]) + 1e-6
    tx, ty = diffs[:, 0] / lengths, diffs[:, 1] / lengths
    return tx, ty, -ty, tx          # (切向, 法向)


def orient_normals(proj_u8: np.ndarray, pts: np.ndarray, nx, ny):
    """法向定向：法向默认指向"区外"——比较 ±3px 探针处分区均值，反了就整体翻转。

    定错方向会把边界吸到对面暗缝的反向梯度沿上（对着墙找门，找到了后门）。
    """
    h, w = proj_u8.shape
    probe_in_x = np.clip(np.round(pts[:, 0] - 3.0 * nx).astype(int), 0, w - 1)
    probe_in_y = np.clip(np.round(pts[:, 1] - 3.0 * ny).astype(int), 0, h - 1)
    probe_out_x = np.clip(np.round(pts[:, 0] + 3.0 * nx).astype(int), 0, w - 1)
    probe_out_y = np.clip(np.round(pts[:, 1] + 3.0 * ny).astype(int), 0, h - 1)
    if float(np.mean(proj_u8[probe_out_y, probe_out_x])) > float(np.mean(proj_u8[probe_in_y, probe_in_x])):
        return -nx, -ny
    return nx, ny


def remap_1d_profiles(gray_f: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """分块 cv2.remap：轮廓点数 N ≥ 32767 会触发 OpenCV SHRT_MAX 断言，16000 一块拼接。"""
    n = map_x.shape[0]
    chunk = 16000
    if n <= chunk:
        return cv2.remap(gray_f, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    parts = [cv2.remap(gray_f, map_x[i:i + chunk], map_y[i:i + chunk],
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
             for i in range(0, n, chunk)]
    return np.concatenate(parts, axis=0)


def sample_profiles(gray: np.ndarray, pts: np.ndarray, nx, ny, snap_px: int = SNAP_PX):
    """每个轮廓点沿法向 ±(snap_px+1) 采样一条灰度剖面 → (n, 2*snap_px+3) 矩阵。"""
    gray_f = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0.8)
    ts = np.arange(-snap_px - 1, snap_px + 2, dtype=np.float32)
    map_x = (pts[:, 0:1] + nx[:, None] * ts[None, :]).astype(np.float32)
    map_y = (pts[:, 1:2] + ny[:, None] * ts[None, :]).astype(np.float32)
    prof = remap_1d_profiles(gray_f, map_x, map_y)
    return cv2.GaussianBlur(prof, (1, 5), 0.0), ts


def directed_peak(prof: np.ndarray, zone_med: float, snap_px: int = SNAP_PX):
    """定向梯度峰值搜索：返回 (best_t, peak_grad, dir_sign)。

    dir_sign 由"区内部更亮还是更暗"决定（in_val−min 与 max−in_val 谁大），
    保证只认从区内到区外那个方向的灰度跃变。
    """
    t_vals = np.arange(-snap_px, snap_px + 1, dtype=np.float32)
    p_mid = prof[:, 1:-1]
    in_val = np.maximum(np.median(p_mid[:, :5], axis=1), zone_med * 0.85)
    min_val = p_mid.min(axis=1)
    max_val = p_mid.max(axis=1)
    dir_sign = np.where((in_val - min_val) >= (max_val - in_val), -1.0, 1.0).astype(np.float32)

    raw_grad = 0.5 * (prof[:, 2:] - prof[:, :-2]) * dir_sign[:, None]
    grad = np.maximum(0.0, raw_grad)
    weighted = grad * (1.0 - GRAD_DECAY * np.abs(t_vals)[None, :])
    best_idx = np.argmax(weighted, axis=1)
    peak_grad = grad[np.arange(len(best_idx)), best_idx]
    return t_vals[best_idx], peak_grad, dir_sign


def median_smooth(shifts: np.ndarray, pad: int) -> np.ndarray:
    """环形滑窗中值平滑（wrap 填充）：去掉逐点独立吸附产生的锯齿毛刺。"""
    if pad < 1:
        return shifts
    s_pad = np.pad(shifts, pad, mode="wrap")
    return np.median(
        np.lib.stride_tricks.sliding_window_view(s_pad, 2 * pad + 1), axis=-1
    ).astype(np.float32)


def redraw_hierarchy(cnts, hier, shape) -> np.ndarray:
    """层级轮廓还原：按面积从大到小 drawContours(FILLED)，外环(父<0)填 255、内环(孔)填 0。"""
    out = np.zeros(shape, dtype=np.uint8)
    order = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)
    for idx in order:
        val = 255 if hier[0, idx, 3] < 0 else 0
        cv2.drawContours(out, cnts, idx, val, thickness=cv2.FILLED)
    return out


def snap_zone_train(proj_u8: np.ndarray, gray: np.ndarray, snap_px: int = SNAP_PX) -> np.ndarray:
    """训练模式有效检测区：逐段组装上面 5 步（复刻 compute_effective_zone 训练分支）。"""
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
        if n < 24:                                   # 太短的轮廓（碎片）不值得吸附，原样保留
            out_cnts.append(np.round(pts).astype(np.int32).reshape(-1, 1, 2))
            continue
        _, _, nx, ny = contour_normals(pts)
        nx, ny = orient_normals(proj_u8, pts, nx, ny)
        prof, _ = sample_profiles(gray, pts, nx, ny, snap_px)
        best_t, peak_grad, _ = directed_peak(prof, zone_med, snap_px)
        shifts = np.where(peak_grad >= PEAK_GRAD_MIN,
                          np.clip(best_t - SHOULDER_IN, -snap_px, snap_px), 0.0).astype(np.float32)
        shifts = median_smooth(shifts, pad=min(5, n // 6))
        new_pts = pts + np.stack([nx * shifts, ny * shifts], axis=-1)
        new_pts[:, 0] = np.clip(new_pts[:, 0], 0, w - 1)
        new_pts[:, 1] = np.clip(new_pts[:, 1], 0, h - 1)
        out_cnts.append(np.round(new_pts).astype(np.int32).reshape(-1, 1, 2))
    return redraw_hierarchy(out_cnts, hier, gray.shape[:2])


def make_part_and_zone(seed: int = 3, h: int = 240, w: int = 320, jag: float = 2.0):
    """合成零件与"手画锯齿分区"：亮圆角零件在暗背景上，分区多边形故意画偏 + 抖出锯齿。

    返回 (gray, part_mask 真值, proj_u8 锯齿分区, 偏移量 offset)。分区内缘比真实边缘
    整体向内偏 offset px，且每 3 个顶点抖动 ±jag px——模拟 define_masks 手画误差，
    而机台拍摄时零件又装偏了几像素。
    """
    rng = np.random.default_rng(seed)
    gray = np.full((h, w), 35.0, dtype=np.float32)
    part_mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy, rad = w // 2, h // 2, h // 3
    cv2.circle(gray, (cx, cy), rad, 205.0, -1)          # 零件本体（亮）
    cv2.circle(gray, (cx, cy), rad, 22.0, 4)            # 边缘暗过渡带（斜坡）
    cv2.circle(part_mask, (cx, cy), rad, 255, -1)
    gray += rng.normal(0, 1.5, gray.shape).astype(np.float32)
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    offset = 5.0                                        # 手画分区比真实边缘整体内缩 5px
    ang = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    poly = np.stack([cx + (rad + offset + rng.uniform(-jag, jag, len(ang))) * np.cos(ang),
                     cy + (rad + offset + rng.uniform(-jag, jag, len(ang))) * np.sin(ang)],
                    axis=1).astype(np.int32)
    proj = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(proj, [poly], 255)
    return gray, part_mask, proj, offset


if __name__ == "__main__":
    gray, part, proj, offset = make_part_and_zone()
    snapped = snap_zone_train(proj, gray)

    def iou(a, b):
        a, b = a > 127, b > 127
        return (a & b).sum() / max(1, (a | b).sum())

    print(f"吸附前 分区∩零件 IoU = {iou(proj, part):.4f}  (手画偏 {offset}px + 锯齿)")
    print(f"吸附后 分区∩零件 IoU = {iou(snapped, part):.4f}  (±{SNAP_PX}px 法向吸附 + 肩部内收 {SHOULDER_IN}px)")
    area_in = int(((snapped > 127) & (part > 127)).sum())
    area_snapped = int((snapped > 127).sum())
    print(f"吸附后区面积 = {area_snapped}px, 其中落在零件上 {area_in}px "
          f"({area_in / area_snapped:.1%} 区内干净度——训练模式要求贴近 100%)")
