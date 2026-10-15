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
    """纯平移的 2x3 仿射矩阵：把整张图往右挪 tx、往下挪 ty。

    人话规则就两条公式：新x = 老x + tx；新y = 老y + ty。返回的表就是把这两条
    公式抄成"系数表"（计算机只认识数）——规矩只有一条：一行 = 一条公式，一行
    三个数 = 三个位置（老x 乘几、老y 乘几、再加一个常数）：

        [[1, 0, tx],    第一行读作：新x = 1×老x + 0×老y + tx×1 = 老x + tx
         [0, 1, ty]]    第二行读作：新y = 0×老x + 1×老y + ty×1 = 老y + ty

    乘 1 原样、乘 0 消失，化简回来就是那两条人话公式。第三项是纯常数，约定写成
    "常数 × 1"——代码里那个 1 是实体：transform_points 的 np.ones((N, 1)) 就是
    给每个点补上这一列。手工算（tx=10, ty=20，点 (5,5)）：新x = 1×5+0+10 = 15、
    新y = 0+1×5+20 = 25 -> (15, 25)；demo 实测：白块 (y1..2, x1..2) 平移 (3,2)
    后落在 (y3..4, x4..5)。

    ★ 推广（什么叫仿射）：把表里 6 个数换成任意的——新x = a×老x + b×老y + e、
      新y = c×老x + d×老y + f——凡是能写成"各乘一个数再加一个数"的变形都叫仿射
      （旋转/缩放/斜切全在内：直线还是直线、平行线还平行；带除法、近大远小的叫
      透视，不算）。平移是最懒的一种：左边 4 个"变形配料"是 [[1,0],[0,1]] =
      形状一点不动，只有第三列常数在干活。

    定位：这是"写表"函数（np.float32 是 cv2 要的格式）——warpAffine（搬像素）
    与 transform_points（搬坐标点）都只认这张表，不关心你是平移还是旋转。
    """
    # 一行一条公式：第一行管新x、第二行管新y；第三列 = 平移量（常数项）
    return np.float32([[1, 0, tx], [0, 1, ty]])


def rotation_matrix(center_xy, angle_deg: float, scale: float = 1.0) -> np.ndarray:
    """绕 center 旋转（+缩放）的 2x3 仿射矩阵——本质是"填三个空、换一张表"：

        rotation_matrix(center,  angle,   scale)
                         ↑       ↑        ↑
                      绕哪转   转多少度   顺带放大/缩小（1.0 = 不动）

    表和平移表是同一种表（2x3、一行一条公式），只是数字换成三角函数算出来的
    （带小数，还有 6.12e-17 这种"浮点版的 0"）——不用读，那正是让 cv2 算的
    理由。用法也一样：M = rotation_matrix(...) 造表，apply_affine(img, M, ...)
    执行。

    center 就是"图钉钉在哪"：钉住的点不跟着动，其他部分围着它画圈。
    实测（5x5 小图、左上角一个 L、绕正中心转 90 度）：

        原图:      转完:
        ##...      .....
        #....  ->  .....
        .....      .....
        .....      #....
        .....      ##...

    三个要点：
      - center 是 (x, y)，angle 单位是度。
      - 方向：正角度在屏幕上看是逆时针（实测"中心右侧 10px -> 中心上方
        10px"，测试 test_90deg_maps_corners 钉的就是它；与 cv2 官方文档
        "positive values mean counter-clockwise" 一致）。
      - center 外层包 float()：cv2 5.0.0 实测塞 np.int32/np.int64 会报
        "Can't parse 'center'"，而 stats/argwhere 给的恰是 numpy 整数——
        这层是"把 numpy 标量翻译成 Python float"的保险。

    ★ scale 只改"一半"（实测最简例：不旋转、钉子钉在 x=3）：
        scale=1 -> [[1, 0, 0], [0, 1, 0]]
        scale=2 -> [[2, 0, -3], [0, 2, 0]]
        scale=3 -> [[3, 0, -6], [0, 3, 0]]
      左上 2×2 = "放大镜的倍数"：缩放时整体乘倍数（1 -> 2 -> 3）。
      第三列 = "按住钉子的那只手"：不跟着乘、每次重算——只按 2×2 算的话，
      钉子 3 会跑到 2×3=6，这一刀就补 -3 把它按回 3（scale=3 时补 -6）；
      所以还可能变号（转 90° 例：scale 1 -> 2 时第三列 20 -> -10）。
      实测分工：钉子 (3,0) 三档 scale 都不动（第三列功劳）；其右侧 1 格的
      (4,0) 离钉子变成 1 / 2 / 3 格（2×2 功劳）。
    """
    # float() 不是装饰：cv2 不认 numpy 整数标量（见 docstring 第三点）
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
