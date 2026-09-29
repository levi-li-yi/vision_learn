"""图像读写、BGR 通道顺序、切片裁剪与滤波（OpenCV 入门 + 与 stage0 手写版对拍）。

对应主仓库：
- 按 (x0, y0, patch_size) 把 4096x3000 大图切 1024 patch（读写+裁剪）；
- 无监督 eval 的固定后处理 GaussianBlur(33,33, sigma=4)（core/utils.py::test）。

本模块的学习方法：手写高斯/中值滤波 -> 与 OpenCV 结果对拍 -> 以后用库但懂原理。

运行：python -m stage1.io_crop_filter
"""

from pathlib import Path

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# 1. 读写图与 BGR 陷阱
# ---------------------------------------------------------------------------

def save_and_load(image: np.ndarray, path) -> np.ndarray:
    """imwrite + imread 往返。注意：cv2 读写都是 **BGR** 通道序（不是 RGB）。

    工程纪律：与任何 RGB 约定的系统交互前，先 cv2.cvtColor(BGR2RGB / RGB2BGR)，
    忘了换序是视觉工程第一大经典 bug（颜色发蓝/发红）。
    """
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"写图失败: {path}")
    loaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if loaded is None:
        raise IOError(f"读图失败: {path}")
    return loaded


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """BGR -> RGB：反转最后一维。"""
    return image[:, :, ::-1]


def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    """彩色转灰度（cv2 加权 0.299R+0.587G+0.114B，与 stage0 手写同公式）。"""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# 2. 切片裁剪（与 stage0.crop_patch 同一件事）
# ---------------------------------------------------------------------------

def crop_roi(image: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    """按仓库口径裁 patch：image[y0:y0+size, x0:x0+size]。

    关键坐标纪律（全工程通用）：
      NumPy 切片下标顺序是 [y, x]（行、列）；
      而 cv2 函数的尺寸/坐标参数几乎都是 (x, y)（宽、高）——warpAffine 的
      dsize=(w,h)、phaseCorrelate 返回 (dx,dy)。两套顺序并存，写代码前先想清楚。
    """
    return image[y0: y0 + size, x0: x0 + size]


# ---------------------------------------------------------------------------
# 3. 手写滤波 vs OpenCV
# ---------------------------------------------------------------------------

def gaussian_kernel_1d(ksize: int, sigma: float) -> np.ndarray:
    """一维高斯核：exp(-x^2 / 2*sigma^2) 归一化，长度 ksize（奇数），中心为 0。"""
    if ksize % 2 == 0:
        raise ValueError("ksize 必须是奇数")
    xs = np.arange(ksize) - ksize // 2
    kernel = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_kernel_2d(ksize: int, sigma: float) -> np.ndarray:
    """二维高斯核 = 两个一维核做外积（可分离性：高斯模糊可拆成两次一维卷积，
    这正是 cv2.GaussianBlur 快的原因之一）。"""
    k1 = gaussian_kernel_1d(ksize, sigma)
    return np.outer(k1, k1)


def manual_gaussian_blur(gray: np.ndarray, ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
    """手写二维高斯模糊：reflect 填充 + 滑窗视图 + einsum 加权求和。

    与 cv2.GaussianBlur 对拍：默认 border BORDER_REFLECT_101 == np.pad 'reflect'。
    """
    kernel = gaussian_kernel_2d(ksize, sigma)
    r = ksize // 2
    padded = np.pad(gray.astype(np.float32), r, mode="reflect")
    windows = sliding_window_view(padded, (ksize, ksize))     # (H, W, k, k)
    return np.einsum("ijkl,kl->ij", windows, kernel)


def manual_median_filter(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """手写中值滤波：每个窗口取中位数。

    直觉：中值对"个别极端值"（椒盐噪点 0/255）免疫，而均值会被拉偏——
    所以去椒盐噪声用中值，不用高斯。
    """
    r = ksize // 2
    padded = np.pad(gray.astype(np.float32), r, mode="reflect")
    windows = sliding_window_view(padded, (ksize, ksize))
    return np.median(windows, axis=(-2, -1))


# ---------------------------------------------------------------------------
# 演示入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 64)
    print("1) BGR 通道序陷阱")
    print("-" * 64)
    rgb_pixel = np.uint8([[[200, 30, 30]]])          # RGB 下的"红色"像素
    bgr_img = bgr_to_rgb(rgb_pixel)                  # 转成 BGR 存储
    print(f"RGB 红色像素 {rgb_pixel[0,0].tolist()} -> BGR 存储 {bgr_img[0,0].tolist()}"
          f"（cv2.imwrite 后用看图软件打开仍是红色，因为它们都按 BGR 约定读）")

    print("=" * 64)
    print("2) 读写往返 + 裁剪")
    print("-" * 64)
    rng = np.random.default_rng(0)
    big = rng.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)
    tmp = Path("build_io_demo.png")
    back = save_and_load(big, tmp)
    print(f"64x80x3 写读往返一致: {np.array_equal(big, back)}")
    patch = crop_roi(big, x0=32, y0=16, size=24)
    print(f"crop_roi(x0=32, y0=16, size=24) -> shape={patch.shape}"
          f"（先 y 后 x，shape=(24,24,3)）")
    tmp.unlink()

    print("=" * 64)
    print("3) 手写高斯 vs cv2.GaussianBlur")
    print("-" * 64)
    gray = rng.integers(0, 256, size=(40, 50)).astype(np.uint8)
    mine = manual_gaussian_blur(gray, ksize=5, sigma=1.2)
    cv = cv2.GaussianBlur(gray, (5, 5), 1.2)
    diff = float(np.abs(mine - cv).max())
    print(f"手写 vs OpenCV 最大像素差 = {diff:.2f} 灰度级（uint8 输入下 OpenCV "
          f"走定点快速实现，差在 1 个灰度级内属正常；float32 输入则逐位一致）")

    print("=" * 64)
    print("4) 高斯 vs 中值：椒盐噪声谁行谁不行")
    print("-" * 64)
    clean = np.full((30, 30), 128, np.uint8)
    noisy = clean.copy()
    noisy[5, 5] = 0; noisy[20, 8] = 255; noisy[12, 25] = 0     # 三粒椒盐
    g = cv2.GaussianBlur(noisy, (3, 3), 0)
    m = cv2.medianBlur(noisy, 3)
    print(f"噪声点原值 0 -> 高斯后 {g[5,5]:.1f}（被拉偏） | 中值后 {m[5,5]}（复原 128）")
    print(f"全图平均误差: 高斯 {np.abs(g.astype(int)-128).mean():.2f} | "
          f"中值 {np.abs(m.astype(int)-128).mean():.2f}（中值完胜）")
