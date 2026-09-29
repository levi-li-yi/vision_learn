"""Demo A（本阶段核心）：图像配准——ORB+RANSAC 相似变换 vs 相位相关，及混合策略。

对应主仓库 data_utils/data.py 的 JIT 标件混合配准：
  "ORB 相似变换优先（同时解旋转+平移+缩放，内点比阈值 0.3），
   不置信时回退纯平移相位相关（align_min_score / align_max_shift 门控），
   二者失效区互补；都失败则退化原始 region（region_margin 安全带）。"

为什么需要配准（业务）：region 蒙版画在标件基准图上，现场装夹每次有
平移/微旋转——地图和实地差几毫米，缺陷就可能贴错位置。配准 = 现场把
两者对齐，再把蒙版 warp 到现场坐标系。

两种方法的失效区（本模块用实验亲自复现）：
  - ORB 靠"角点"匹配：有旋转/缩放时依然稳；但图里没有角点（平滑渐变、
    近重复纹理）时无特征可匹配 -> 失败；
  - 相位相关靠傅里叶平移定理：纯平移亚像素级精准；但有旋转时定理被
    破坏 -> 结果不可信。

运行：python -m stage1.registration
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. 合成"标件图 / 现场图"（真值变换已知，便于量化误差）
# ---------------------------------------------------------------------------

def make_part_template(w=240, h=180, seed=0) -> np.ndarray:
    """标件基准图：随机纹理 + 亮斑 + 暗块，给 ORB 提供丰富角点。"""
    rng = np.random.default_rng(seed)
    img = rng.integers(90, 110, size=(h, w), dtype=np.uint8)      # 底噪纹理
    for _ in range(60):                                            # 亮斑
        cx, cy = int(rng.integers(10, w - 10)), int(rng.integers(10, h - 10))
        r = int(rng.integers(4, 14))
        cv2.circle(img, (cx, cy), r, int(rng.integers(150, 255)), -1)
    for _ in range(10):                                            # 暗块
        x, y = int(rng.integers(15, w - 15)), int(rng.integers(15, h - 15))
        a = int(rng.integers(8, 18))
        cv2.rectangle(img, (x - a, y - a), (x + a, y + a),
                      int(rng.integers(40, 80)), -1)
    return img


def make_scene(template: np.ndarray, angle_deg=0.0, tx=0.0, ty=0.0,
               scale=1.0, noise=2.0, seed=1):
    """按已知相似变换造"现场图"，返回 (scene, 真值矩阵 M_gt)。

    M_gt 把标件坐标系映射到现场坐标系——配准的目标就是把 M_gt 估出来。
    """
    h, w = template.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    scene = cv2.warpAffine(template, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=100)
    rng = np.random.default_rng(seed)
    if noise > 0:
        scene = np.clip(scene.astype(np.float32)
                        + rng.normal(0, noise, scene.shape), 0, 255).astype(np.uint8)
    return scene, M


def make_smooth_pair(w=200, h=150, tx=8.0, ty=6.0, seed=2):
    """平滑渐变+高斯鼓包的模板/现场对：解析构造的纯平移（无边界伪影）。

    平滑到没有任何角点 -> ORB 必失败；平移显著 -> 相位相关必成功。
    """
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.4, h * 0.6
    s2 = 2 * (w / 6) ** 2

    def f(x, y):
        return 120 + 0.35 * x + 0.15 * y + 40 * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / s2))

    template = np.clip(f(xs, ys), 0, 255).astype(np.uint8)
    scene = np.clip(f(xs - tx, ys - ty), 0, 255).astype(np.uint8)  # 内容平移 (+tx,+ty)
    return template, scene


# ---------------------------------------------------------------------------
# 2. 两个配准器
# ---------------------------------------------------------------------------

def estimate_orb_similarity(template, scene, min_matches=8, min_inlier_ratio=0.3,
                            ransac_thr=3.0):
    """ORB 特征 + 比率测试 + RANSAC 估计相似变换（旋转+缩放+平移，2x3）。

    链路（与仓库同构）：ORB 提角点描述子 -> BFMatcher knn -> Lowe 比率
    测试去歧义 -> estimateAffinePartial2D(RANSAC) 去外点 -> 内点比门控。
    返回 (M 或 None, meta)；None = 本方法对这对图失灵。
    """
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(scene, None)
    if des1 is None or des2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
        return None, {"reason": "keypoints", "n1": len(kp1), "n2": len(kp2)}

    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2)
            if m.distance < 0.75 * n.distance]
    if len(good) < min_matches:
        return None, {"reason": "ratio_test", "n_good": len(good)}

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=ransac_thr,
                                             maxIters=5000)
    if M is None:
        return None, {"reason": "ransac", "n_good": len(good)}
    meta = {"inlier_ratio": float(inliers.mean()), "n_good": len(good)}
    if meta["inlier_ratio"] < min_inlier_ratio:
        return None, {**meta, "reason": "inlier_ratio"}
    return M, meta


def estimate_phase_translation(template, scene):
    """相位相关估纯平移。返回 (M 2x3, meta 含 response)。

    原理一句话：图像在空间域平移 = 频域相位线性变化；两图互功率谱的
    相位反变换后在真位移处出一个尖峰。前提：只有平移，没有旋转/缩放。
    """
    shift, response = cv2.phaseCorrelate(template.astype(np.float32),
                                         scene.astype(np.float32))
    dx, dy = float(shift[0]), float(shift[1])
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return M, {"shift": (dx, dy), "response": float(response)}


# ---------------------------------------------------------------------------
# 3. 混合策略 + 蒙版搬运（仓库 data_utils 混合配准的教学版）
# ---------------------------------------------------------------------------

def hybrid_register(template, scene, min_inlier_ratio=0.3,
                    min_phase_response=0.35, max_shift=64):
    """ORB 优先、相位相关回退、都失败降级——返回 (方法名, M 或 None, meta)。

    门控对应仓库参数：min_inlier_ratio=内点比 0.3、min_phase_response=
    align_min_score 0.35、max_shift=align_max_shift 64（位移合理性上界，
    用来拒绝"估出天文数字位移"的病态解）。
    """
    M_orb, meta_orb = estimate_orb_similarity(template, scene,
                                              min_inlier_ratio=min_inlier_ratio)
    if M_orb is not None:
        if abs(M_orb[0, 2]) <= max_shift and abs(M_orb[1, 2]) <= max_shift:
            return "orb", M_orb, meta_orb
        meta_orb["reason"] = "shift_out_of_range"

    M_ph, meta_ph = estimate_phase_translation(template, scene)
    (dx, dy), resp = meta_ph["shift"], meta_ph["response"]
    if resp >= min_phase_response and max(abs(dx), abs(dy)) <= max_shift:
        return "phase", M_ph, meta_ph
    return "none", None, {**meta_ph, **meta_orb}


def warp_mask(mask: np.ndarray, M: np.ndarray, out_wh) -> np.ndarray:
    """通关函数：把标件坐标系下的 region 蒙版 warp 到现场坐标系。

    INTER_NEAREST 保持 0/255 二值语义（插值会产生中间灰度值）。
    """
    return cv2.warpAffine(mask, M, (int(out_wh[0]), int(out_wh[1])),
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


def point_map_error(M_est, M_gt, shape_wh):
    """配准误差评估：图内均匀采样点，分别用两个矩阵映射，平均欧氏距离 (px)。"""
    w, h = shape_wh
    xs = np.linspace(20, w - 20, 5)
    ys = np.linspace(20, h - 20, 4)
    pts = np.array([[x, y, 1.0] for y in ys for x in xs])
    a, b = pts @ M_est.T, pts @ M_gt.T
    return float(np.linalg.norm(a[:, :2] - b[:, :2], axis=1).mean())


# ---------------------------------------------------------------------------
# 演示入口：三个场景亲眼看"失效区互补"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    W, H = 240, 180

    print("=" * 64)
    print("场景 A：纹理丰富 + 旋转 8 度 + 平移 + 缩放 1.05 -> ORB 主场")
    print("-" * 64)
    template = make_part_template(W, H, seed=0)
    scene, M_gt = make_scene(template, angle_deg=8.0, tx=10.0, ty=-6.0,
                             scale=1.05, noise=2.0, seed=1)
    M_orb, meta = estimate_orb_similarity(template, scene)
    M_ph, meta_ph = estimate_phase_translation(template, scene)
    e_orb = point_map_error(M_orb, M_gt, (W, H))
    e_ph = point_map_error(M_ph, M_gt, (W, H))
    print(f"ORB: 内点比 {meta['inlier_ratio']:.2f}，点位误差 {e_orb:.2f}px  <- 准")
    print(f"相位相关: response {meta_ph['response']:.2f}，"
          f"按平移硬算误差 {e_ph:.2f}px  <- 旋转把平移定理破坏，不可信")
    method, M_h, _ = hybrid_register(template, scene)
    print(f"hybrid_register 选择: {method}")

    print("=" * 64)
    print("场景 B：平滑渐变（无角点）+ 纯平移 (8, 6) -> 相位相关主场")
    print("-" * 64)
    t2, s2 = make_smooth_pair(W, H, tx=8.0, ty=6.0)
    M_orb2, meta2 = estimate_orb_similarity(t2, s2)
    M_ph2, meta2_ph = estimate_phase_translation(t2, s2)
    print(f"ORB: {meta2.get('reason')}（关键点 {meta2.get('n1')} 个——没有角点可匹配）")
    print(f"相位相关: 估出平移 ({meta2_ph['shift'][0]:.2f}, {meta2_ph['shift'][1]:.2f})，"
          f"response {meta2_ph['response']:.2f}，真值 (8, 6)  <- 亚像素级")
    method2, M_h2, _ = hybrid_register(t2, s2)
    print(f"hybrid_register 选择: {method2}")

    print("=" * 64)
    print("场景 C：蒙版搬运（通关标准）")
    print("-" * 64)
    region = np.zeros((H, W), np.uint8)
    region[40:120, 60:180] = 255                     # 标件坐标系下的工艺蒙版
    M_move = np.float32([[1, 0, 10], [0, 1, 20]])    # 现场整体平移 (10, 20)
    moved = warp_mask(region, M_move, (W, H))
    ys, xs = np.nonzero(moved)
    print(f"蒙版原区域 y[40,120) x[60,180) -> warp 后 y[{ys.min()},{ys.max()}] "
          f"x[{xs.min()},{xs.max()}]（应 60..139 / 70..189）")
    print("这一步就是仓库'把工艺 region 蒙版搬到现场图实际坐标系后再约束缺陷放置'。")
