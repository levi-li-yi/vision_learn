"""二值化（固定阈值 / 大津 Otsu）与形态学（腐蚀/膨胀/开/闭）。

对应主仓库：
- 二值化 + 连通域面积过滤 = 监督 eval 的"概率图 > threshold -> mask -> 组件统计"；
- 形态学闭运算 = 无监督 eval 输出二值 mask 后 MORPH_CLOSE(15,15) 的碎片粘合。

核心直觉（背下来）：
- 腐蚀：白区收缩（窗口内只要有一个黑，中心变黑）；膨胀：白区扩张。
- 开 = 先腐蚀再膨胀：去掉比结构元小的白点（椒噪声），大块基本还原。
- 闭 = 先膨胀再腐蚀：填掉比结构元小的黑洞，大块基本还原。

运行：python -m stage1.threshold_morphology
"""

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def binarize_fixed(gray: np.ndarray, threshold: int) -> np.ndarray:
    """固定阈值二值化：> threshold 为 255，否则 0。"""
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def binarize_otsu(gray: np.ndarray, inverse: bool = False) -> tuple:
    """大津法自动选阈值：把直方图分成两类、类间方差最大的那个分割点。

    为什么需要它：现场光照会漂，写死的阈值今天好用明天误报；
    Otsu 对"双峰直方图"（背景一群、缺陷一群）能自动站在谷底。
    返回 (threshold, binary)。
    """
    flag = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    thr, binary = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
    return thr, binary


def erode_manual(binary: np.ndarray, ksize: int = 3) -> np.ndarray:
    """手写腐蚀 = 最小值滤波（窗口全白才白）。仅教学用，内部区域与 cv2 一致。"""
    r = ksize // 2
    padded = np.pad(binary, r, mode="constant", constant_values=0)
    windows = sliding_window_view(padded, (ksize, ksize))
    return (windows.min(axis=(-2, -1))).astype(np.uint8)


def dilate_manual(binary: np.ndarray, ksize: int = 3) -> np.ndarray:
    """手写膨胀 = 最大值滤波（窗口有一个白就白）。"""
    r = ksize // 2
    padded = np.pad(binary, r, mode="constant", constant_values=0)
    windows = sliding_window_view(padded, (ksize, ksize))
    return (windows.max(axis=(-2, -1))).astype(np.uint8)


def erode(binary, ksize=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.erode(binary, kernel)


def dilate(binary, ksize=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.dilate(binary, kernel)


def morph_open(binary: np.ndarray, ksize: int = 3) -> np.ndarray:
    """开运算：先腐蚀再膨胀——去小白点、保大块。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def morph_close(binary: np.ndarray, ksize: int = 3) -> np.ndarray:
    """闭运算：先膨胀再腐蚀——填小黑洞、保大块。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def rect_struct_element(ksize: int = 3) -> np.ndarray:
    """结构元：决定"拿什么形状的窗口去腐蚀/膨胀"。矩形全 1 最常用；
    椭圆核（MORPH_ELLIPSE）边缘更圆滑，仓库无监督 eval 用的就是椭圆核(15,15)。"""
    return cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))


# ---------------------------------------------------------------------------
# 演示入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 64)
    print("1) 固定阈值 vs 大津自动阈值")
    print("-" * 64)
    rng = np.random.default_rng(1)
    # 双峰图：背景 ~120、缺陷 ~40
    img = np.full((60, 80), 120, np.uint8)
    img[20:30, 10:25] = 40
    img[40:50, 50:70] = 45
    img = np.clip(img.astype(np.int16) + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    thr, binary = binarize_otsu(img)
    print(f"Otsu 自动选的阈值 = {thr:.1f}（落在两峰之间的谷底，约 70~100）")
    print(f"固定阈值 100 的结果与 Otsu 白像素数: "
          f"{int((binarize_fixed(img, 100) > 0).sum())} vs {int((binary > 0).sum())}")

    print("=" * 64)
    print("2) 手写腐蚀/膨胀 vs OpenCV（内部区域）")
    print("-" * 64)
    binary = np.zeros((50, 50), np.uint8)
    binary[10:40, 10:40] = 255                       # 30x30 白方块
    inner = (slice(2, -2), slice(2, -2))             # cv2 边界策略不同，只对拍内部
    same_erode = np.array_equal(erode_manual(binary)[inner], erode(binary)[inner])
    same_dilate = np.array_equal(dilate_manual(binary)[inner], dilate(binary)[inner])
    print(f"腐蚀手写==cv2（内部）: {same_erode}，白像素 900 -> "
          f"{int((erode(binary) > 0).sum())}")
    print(f"膨胀手写==cv2（内部）: {same_dilate}，白像素 900 -> "
          f"{int((dilate(binary) > 0).sum())}（收缩/扩张一圈）")

    print("=" * 64)
    print("3) 开运算去椒噪声 / 闭运算填洞")
    print("-" * 64)
    noisy = np.zeros((50, 50), np.uint8)
    noisy[10:40, 10:40] = 255                        # 大白块
    for y, x in [(5, 5), (8, 45), (45, 20)]:         # 三粒孤立椒噪声
        noisy[y, x] = 255
    opened = morph_open(noisy, ksize=3)
    print(f"开运算前白像素 {(noisy > 0).sum()} -> 后 {(opened > 0).sum()} "
          f"（900 + 3 噪点 -> 900：噪点没了，大块尺寸基本不变）")

    holed = np.zeros((50, 50), np.uint8)
    holed[10:40, 10:40] = 255
    holed[20:22, 20:22] = 0                          # 中间挖个 2x2 洞
    closed = morph_close(holed, ksize=3)
    print(f"闭运算：背景像素 {(holed == 0).sum()}（含 4px 洞） -> "
          f"{(closed == 0).sum()}（洞被填上 = 少 4，外轮廓不变；"
          f"注意结构元要比洞大才填得上——5x5 的洞得用 7x7 核）")
