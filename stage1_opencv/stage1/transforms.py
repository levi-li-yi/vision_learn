"""仿射与透视变换：把"坐标换算"变成"图上重采样"。

对应主仓库：
- JIT 配准的最后一步 = 用估计出的 2x3 仿射矩阵 warpAffine 把 region 蒙版
  搬到现场图坐标系（registration.warp_mask 的底座）；
- Cut-Paste 的贴纸旋转/缩放（blending.transform_sticker 的底座）。

必须刻进肌肉记忆的坐标纪律：
  - cv2 的尺寸参数是 (宽, 高)：warpAffine 的 dsize=(w, h)，别写成 (h, w)；
  - 变换矩阵 M 是 2x3：[[a, b, tx], [c, d, ty]]，作用于齐次坐标 [x, y, 1]；
  - NumPy 下标是 [y, x]。三套约定在一个函数里碰头，是最常见的翻车点。

运行：python -m stage1.transforms
"""

import cv2
import numpy as np


def translation_matrix(tx: float, ty: float) -> np.ndarray:
    """纯平移的 2x3 仿射矩阵。"""
    return np.float32([[1, 0, tx], [0, 1, ty]])


def rotation_matrix(center_xy, angle_deg: float, scale: float = 1.0) -> np.ndarray:
    """绕 center 旋转（+缩放）的 2x3 仿射矩阵。

    注意 center 是 (x, y)；angle 单位是度，图像坐标系 y 向下，
    正角度在图上是逆时针（数学直觉）/屏幕上看似顺时针——不必纠结，
    测试里用像素对拍说话。
    """
    return cv2.getRotationMatrix2D((float(center_xy[0]), float(center_xy[1])),
                                   angle_deg, scale)


def apply_affine(image: np.ndarray, M: np.ndarray, out_wh, interp=cv2.INTER_NEAREST):
    """warpAffine：dsize 是 (宽, 高)！interp 默认 NEAREST 便于测试精确对拍。"""
    return cv2.warpAffine(image, M, (int(out_wh[0]), int(out_wh[1])),
                          flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def transform_points(M: np.ndarray, points_xy) -> np.ndarray:
    """用 2x3 矩阵变换一批 (x, y) 点——评估配准精度就是拿它算点位误差。"""
    pts = np.asarray(points_xy, dtype=np.float64)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)   # (N, 3)
    return hom @ M.T                                             # (N, 2)


def estimate_perspective(src_quad, dst_quad) -> np.ndarray:
    """由 4 对点求 3x3 透视矩阵（getPerspectiveTransform）。

    用途：把斜拍的"平面"拉正（文档/PCB/屏幕），或建模相机的透视畸变。
    仿射 = 透视的特例（平行线保持平行）；透视 8 自由度 vs 仿射 6 自由度。
    """
    src = np.float32(src_quad)
    dst = np.float32(dst_quad)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("需要恰好 4 对 (x, y) 点")
    return cv2.getPerspectiveTransform(src, dst)


def apply_perspective(image: np.ndarray, P: np.ndarray, out_wh,
                      interp=cv2.INTER_NEAREST):
    return cv2.warpPerspective(image, P, (int(out_wh[0]), int(out_wh[1])),
                               flags=interp, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)


# ---------------------------------------------------------------------------
# 演示入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 64)
    print("1) 平移：矩阵、warp、点位换算三种视角")
    print("-" * 64)
    img = np.zeros((60, 80), np.uint8)
    img[10:20, 30:40] = 200                                  # 特征小块
    M = translation_matrix(10, 20)
    moved = apply_affine(img, M, out_wh=(80, 60))            # dsize=(宽, 高)!
    print(f"原块区域 (y 10..20, x 30..40) -> 平移后 (y 30..40, x 40..50)")
    print(f"warp 后 [35, 45] 处的值 = {moved[35, 45]}（应等于原 200）")
    pts = transform_points(M, [(35, 15)])                    # 原块中心点
    print(f"transform_points 把原中心 (35,15) 映到 {pts[0].tolist()}（应 [45,35]）")

    print("=" * 64)
    print("2) 旋转 90 度：dsize 顺序陷阱 + 点位验证")
    print("-" * 64)
    rect = np.zeros((60, 100), np.uint8)                     # 高 60、宽 100
    rect[5:55, 45:55] = 255
    M90 = rotation_matrix((50, 30), 90)
    rot = apply_affine(rect, M90, out_wh=(100, 60))          # dsize=(宽,高)！
    print(f"旋转 90 度后白像素数 {int((rot > 0).sum())}（应 500，"
          f"NEAREST 不增不减；out_wh 若传成 (60,100) 会裁掉一块）")
    center_after = transform_points(M90, [(50, 30)])
    print(f"旋转中心 (50,30) 映射到自身: {center_after[0].tolist()}")

    print("=" * 64)
    print("3) 透视：四点拉正一个'斜拍四边形'")
    print("-" * 64)
    board = np.zeros((240, 240), np.uint8)
    quad = [(60, 40), (220, 60), (200, 220), (40, 200)]      # 输入图里的斜四边形
    corners = [(40, 40), (200, 40), (200, 200), (40, 200)]   # 目标：拉回正方形
    cv2.fillPoly(board, [np.int32(quad)], 128)               # 内容画在四边形处
    P = estimate_perspective(quad, corners)
    rectified = apply_perspective(board, P, out_wh=(240, 240))
    ys, xs = np.nonzero(rectified)
    print(f"拉正后内容 bbox: x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]"
          f"（应接近 40..200 的正方形）")
