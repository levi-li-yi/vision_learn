"""课时 1：分网格角点采样 + 8-DOF 单应粗尺度求解 + 位姿门控 —— align.py 对齐管线前半段（免 Halcon 复刻）。

真工程用 Halcon points_harris + proj_match_points_ransac('ncc', ..., 'gold_standard', 1.5, 42) 求解
基准图 → 目标图的单应矩阵 H；本课用 cv2.cornerHarris + 邻域 NCC 模板匹配 + RANSAC cv2.findHomography
复刻同一条管线，并保留真工程的全部"安检"细节：

  · spatial_select：图分 6×6 = 36 格、每格取最强 30 个角点（GRID=6/PER_CELL=30，空间均匀，
    避免角点全挤在纹理丰富的角落、让单应在小区域内退化）；
  · 粗尺度求解：图宽 > 512 时先缩到宽 512 在小图上求 Hs，再 H = S⁻¹·Hs·S 还原到原始分辨率
    （S = diag(s, s, 1)，H /= H[2,2] 归一化）——NCC 匹配窗口大小与耗时都与分辨率解耦；
  · 位姿门控：四角 cv2.perspectiveTransform 最大位移 < 15%·max(H,W)、仿射子阵行列式 ∈ (0.5, 2.0)、
    内点数 ≥ 50 —— 任一不过即配准失败（宁可失败也不给下游一个病态 H）；
  · rc↔xy 坐标系交换：Halcon 返回 (row,col) 序的单应，用置换矩阵 swap@H_rc@swap 转成 (x,y) 序
    才能给 OpenCV 的 warpPerspective 用；
  · 失败隔离：配准失败的图连同胞 .tmp 标注一起移入 .tmp/{产品}/align_failed/，moved.jsonl 逐条
    审计（时间/原因/内点数/位移/源目路径），绝不静默丢弃。

对应主仓库：src/train/align.py::spatial_select / project_mask_to_query(swap 段) /
Point8DofAligner.align / align_all(failed_items 隔离段)。

运行：python -m stage6.corner_grid
"""
import json
import os
import shutil
from datetime import datetime

import cv2
import numpy as np

GRID = 6           # 对应 align.py GRID：6×6 = 36 格
PER_CELL = 30      # 对应 align.py PER_CELL：每格取最强 30 个
COARSE_W = 512     # 对应 Point8DofAligner(coarse_width=512)
MIN_INLIERS = 50   # 对应 align.py：内点 < 50 直接失败


def harris_points(gray: np.ndarray, thresh_rel: float = 1e-3):
    """Harris 角点（Halcon points_harris(0.7,2.0,0.08,1e-3) 的 OpenCV 替身）。

    返回 (rows, cols, resp)，全部按响应从强到弱排序——spatial_select 的
    "每格取前 30 个"依赖这个序（与 Halcon 返回序一致：强在前）。
    """
    resp = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    thr = resp.max() * thresh_rel
    rows, cols = np.nonzero(resp > thr)
    r = resp[rows, cols]
    order = np.argsort(-r)
    return rows[order], cols[order], r[order]


def spatial_select(rows, cols, shape, grid: int = GRID, per_cell: int = PER_CELL):
    """分网格采样：图像分 grid×grid 格，每格取最强角点，保证空间均匀。

    逐行复刻 align.py::spatial_select（含 np.clip 防边界点溢出到不存在的格）。
    """
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
    return [int(rows[i]) for i in sel], [int(cols[i]) for i in sel]


def swap_rc_xy(H_rc: np.ndarray) -> np.ndarray:
    """(row,col) 序单应 → (x,y) 序：swap @ H_rc @ swap（置换矩阵左右夹乘）。

    Halcon 的齐次坐标是 (row, col, 1)，OpenCV 是 (x, y, 1) = (col, row, 1)。
    对齐右下角必须再除以 H[2,2] 归一化，否则齐次尺度不一致。
    """
    H_rc = np.asarray(H_rc, dtype=np.float64)
    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    H_xy = swap @ H_rc @ swap
    return H_xy / H_xy[2, 2]


def ncc_match_pairs(ref_s: np.ndarray, qry_s: np.ndarray, ref_pts, qry_pts,
                    half: int = 7, tol: int = 40, thresh: float = 0.3):
    """邻域 NCC 模板匹配（Halcon proj_match_points_ransac('ncc', tol, tol, 0.0, 0.3) 替身）。

    对基准图每个角点取 (2*half+1)² 小块，在目标图同名角点 ±tol 窗口内做
    TM_CCOEFF_NORMED 模板匹配，得分 ≥ thresh 记为一对候选。真工程中两图角点
    已经各自做过 spatial_select，这里直接在候选点集间匹配。
    """
    h, w = qry_s.shape
    pairs = []
    for r, c in ref_pts:
        r0, r1 = r - half, r + half + 1
        c0, c1 = c - half, c + half + 1
        if r0 < 0 or c0 < 0 or r1 > ref_s.shape[0] or c1 > ref_s.shape[1]:
            continue
        patch = ref_s[r0:r1, c0:c1].astype(np.float32)
        # 在目标图角点附近 ±tol 搜索（真工程 row_tolerance/col_tolerance=80，按粗尺度比例缩放后传入）
        sy0, sy1 = max(0, r - tol - half), min(h, r + tol + half + 1)
        sx0, sx1 = max(0, c - tol - half), min(w, c + tol + half + 1)
        if sy1 - sy0 < patch.shape[0] or sx1 - sx0 < patch.shape[1]:
            continue
        region = qry_s[sy0:sy1, sx0:sx1].astype(np.float32)
        res = cv2.matchTemplate(region, patch, cv2.TM_CCOEFF_NORMED)
        _, best, _, loc = cv2.minMaxLoc(res)
        if best >= thresh:
            qr = sy0 + loc[1] + half
            qc = sx0 + loc[0] + half
            pairs.append(((c, r), (qc, qr), float(best)))   # (x, y) 序
    return pairs


def pose_gate(H: np.ndarray, shape, max_shift_frac: float = 0.15):
    """位姿合理性门控：四角最大位移 < 15%·max(H,W) 且仿射子阵行列式 ∈ (0.5, 2.0)。

    行列式近似面积缩放：det≈0.5/2.0 之外说明匹配把图缩放/翻转到了离谱尺度；
    四角位移门限拦住"求出来一个把图甩出画面"的病态解。
    """
    h, w = shape
    corners = np.float32([[0, 0], [w, 0], [0, h], [w, h]]).reshape(-1, 1, 2)
    try:
        moved = cv2.perspectiveTransform(corners, np.asarray(H, np.float64)).reshape(-1, 2)
    except cv2.error:
        return False, float("inf"), 0.0
    shift = float(np.abs(moved - corners.reshape(-1, 2)).max())
    det = float(np.linalg.det(H[:2, :2]))
    ok = shift <= max_shift_frac * max(h, w) and 0.5 < det < 2.0
    return ok, shift, det


def solve_homography(pairs, ransac_reproj: float = 1.5):
    """RANSAC 单应（真工程 gold_standard 重投影门限 1.5px）。返回 H(3,3) 与内点数。"""
    if len(pairs) < 4:
        return None, 0
    src = np.float32([p[0] for p in pairs]).reshape(-1, 1, 2)
    dst = np.float32([p[1] for p in pairs]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_reproj)
    if H is None:
        return None, 0
    return H / H[2, 2], int(mask.sum())


def align_coarse(ref_gray: np.ndarray, qry_gray: np.ndarray,
                 coarse_width: int = COARSE_W, tol: int = 80,
                 match_thresh: float = 0.3) -> dict:
    """完整求解管线：粗尺度 → Harris+网格采样 → NCC 匹配 → RANSAC → 尺度还原 → 位姿门控。

    返回与真工程 Point8DofAligner.align 同构的 dict（ok/reason/H/inliers/...）。
    """
    if qry_gray.shape != ref_gray.shape:
        return {"ok": False, "reason": f"待测图尺寸 {qry_gray.shape[:2]} 与基准图 "
                                       f"{ref_gray.shape} 不一致", "H": None, "inliers": 0}
    h, w = ref_gray.shape
    scale = 1.0
    if 0 < coarse_width < w:
        scale = coarse_width / float(w)
    small = (max(1, round(h * scale)), max(1, round(w * scale)))

    def to_struct(g):
        if abs(scale - 1.0) < 1e-9:
            return g
        return cv2.resize(g, (small[1], small[0]), interpolation=cv2.INTER_AREA)

    ref_s, qry_s = to_struct(ref_gray), to_struct(qry_gray)

    rr, rc_, _ = harris_points(ref_s)
    qr, qc, _ = harris_points(qry_s)
    ref_rows, ref_cols = spatial_select(rr, rc_, small)
    qry_rows, qry_cols = spatial_select(qr, qc, small)
    if len(ref_rows) < 50 or len(qry_rows) < 50:
        return {"ok": False, "reason": f"角点不足 (基准 {len(ref_rows)} / 待测 {len(qry_rows)})",
                "H": None, "inliers": 0}

    ref_pts = list(zip(ref_rows, ref_cols))
    qry_pts = list(zip(qry_rows, qry_cols))
    tol_s = max(2, int(round(tol * scale)))
    pairs = ncc_match_pairs(ref_s, qry_s, ref_pts, qry_pts, tol=tol_s, thresh=match_thresh)

    Hs, n_inliers = solve_homography(pairs)
    if Hs is None or n_inliers < MIN_INLIERS:
        return {"ok": False, "reason": f"内点不足 ({n_inliers} < {MIN_INLIERS})",
                "H": None, "inliers": n_inliers}

    S = np.diag([scale, scale, 1.0])                     # 粗尺度还原：H = S⁻¹·Hs·S
    H = np.linalg.inv(S) @ Hs @ S
    H /= H[2, 2]

    ok, shift, det = pose_gate(H, ref_gray.shape)
    if not ok:
        return {"ok": False, "reason": f"位姿不合理 (角点最大位移 {shift:.0f}px, 尺度 det {det:.3f})",
                "H": None, "inliers": n_inliers}
    return {"ok": True, "reason": None, "H": H, "inliers": n_inliers,
            "max_shift_px": round(shift, 1), "det": round(det, 4)}


def quarantine_failed(failed_items, fail_root: str) -> list:
    """配准失败隔离：bmp(+.tmp 标注) 移入 align_failed/{点位}/，moved.jsonl 逐条审计。

    简化自 align.py::align_all(failed_items 段)：真工程隔离后 print_red 红字汇总并
    sys.exit(1) 大声退出——学习版返回审计行，由调用方决定是否退出。
    """
    os.makedirs(fail_root, exist_ok=True)
    jsonl_path = os.path.join(fail_root, "moved.jsonl")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    for item in failed_items:
        stem = item["stem"]
        src_bmp = item["bmp_path"]
        src_tmp = src_bmp + ".tmp"
        sub_rel = os.path.join(item["point"], "val") if item.get("is_val") else item["point"]
        dst_dir = os.path.join(fail_root, sub_rel)
        os.makedirs(dst_dir, exist_ok=True)
        dst_bmp = os.path.join(dst_dir, os.path.basename(src_bmp))
        if os.path.isfile(src_bmp):
            shutil.move(src_bmp, dst_bmp)
        dst_tmp = None
        if os.path.isfile(src_tmp):
            dst_tmp = dst_bmp + ".tmp"
            shutil.move(src_tmp, dst_tmp)
        rec = {"time": now_str, "point": item["point"], "stem": stem,
               "reason": item["reason"], "inliers": item.get("inliers", 0),
               "src_bmp": src_bmp, "dst_bmp": dst_bmp, "dst_tmp": dst_tmp}
        lines.append(rec)
    with open(jsonl_path, "a", encoding="utf-8") as jf:
        for rec in lines:
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return lines


def make_textured(seed: int = 0, h: int = 768, w: int = 1024) -> np.ndarray:
    """合成一张角点丰富的机台纹理图（随机圆斑 + 短线段 + 噪声），供对齐练习。"""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 60.0, dtype=np.float32)
    for _ in range(900):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        rad = rng.integers(3, 14)
        shade = rng.uniform(90, 220)
        cv2.circle(img, (int(cx), int(cy)), int(rad), float(shade), -1)
    for _ in range(700):
        y0, x0 = rng.integers(0, h - 1), rng.integers(0, w - 1)
        y1 = int(np.clip(y0 + rng.integers(-40, 40), 0, h - 1))
        x1 = int(np.clip(x0 + rng.integers(-40, 40), 0, w - 1))
        cv2.line(img, (int(x0), int(y0)), (x1, y1), float(rng.uniform(70, 200)), 1)
    img += rng.normal(0, 3.0, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def random_homography(rng, shift=18.0, deg=1.5, scale_jitter=0.02) -> np.ndarray:
    """造一个"机台正常波动"量级的真值单应：平移 ±shift px、微旋转、微缩放。"""
    ang = np.deg2rad(rng.uniform(-deg, deg))
    sc = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
    tx, ty = rng.uniform(-shift, shift), rng.uniform(-shift, shift)
    return np.array([[sc * np.cos(ang), -sc * np.sin(ang), tx],
                     [sc * np.sin(ang),  sc * np.cos(ang), ty],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    ref = make_textured(seed=0)
    H_gt = random_homography(rng)
    qry = cv2.warpPerspective(ref, H_gt, (ref.shape[1], ref.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    res = align_coarse(ref, qry)
    print(f"真值 H (平移/微转/微缩):\n{np.round(H_gt, 4)}")
    if res["ok"]:
        corners = np.float32([[0, 0], [1024, 0], [0, 768], [1024, 768]]).reshape(-1, 1, 2)
        gt = cv2.perspectiveTransform(corners, H_gt).reshape(-1, 2)
        est = cv2.perspectiveTransform(corners, res["H"]).reshape(-1, 2)
        err = float(np.abs(gt - est).max())
        print(f"配准成功: 内点 {res['inliers']} | 四角位移 {res['max_shift_px']}px | "
              f"det {res['det']} | 四角复现误差 {err:.2f}px (< 1.5px 即金标准)")
    else:
        print(f"配准失败: {res['reason']}")

    # 反例：拿一张完全无关的图来配准 —— 门控应把它拦下（角点不足/内点不足/位姿不合理）
    other = make_textured(seed=99)
    res2 = align_coarse(ref, other)
    print(f"无关图配准: ok={res2['ok']}, 原因={res2['reason']}")
