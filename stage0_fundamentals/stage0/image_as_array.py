"""图像即数组：RGB 形状 / 灰度化 / 手写 3x3 均值滤波（纯 NumPy，不用任何图像库）。

这是阶段 0 的毕业设计，对应学习路线的通关标准：
  "能一句话说清：一张 RGB 图在内存里就是一个 (H, W, 3) 的 uint8 数组，
   H 行 W 列，每个格子 3 个 0~255 的整数，分别叫 B/G/R 或 R/G/B（看约定）。"

滤波器为什么重要：均值滤波 = 最简单的卷积。看懂手写版，就为阶段 2/3 的
"卷积层"和仓库无监督 eval 里的 GaussianBlur(33,33) 打好了地基。

运行方式：
    python -m stage0.image_as_array
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. RGB 图的构造与形状
# ---------------------------------------------------------------------------

def make_rgb_image(height: int, width: int) -> np.ndarray:
    """构造一张 (H, W, 3) 的 uint8 演示图：红色横条 + 蓝色竖条 + 噪声。

    通道约定：本工程按 [R, G, B] 排最后一维（OpenCV 读图是 BGR，
    交换是经典 bug 来源——先记住"永远确认通道顺序"）。
    """
    rng = np.random.default_rng(7)
    img = rng.integers(40, 80, size=(height, width, 3), dtype=np.uint8)  # 底噪

    # 红色横条：占满宽，中间 1/4 高
    y0, y1 = height // 3, height // 3 + max(height // 4, 1)
    img[y0:y1, :, 0] = 200     # R 通道拉高
    img[y0:y1, :, 1:] = 30     # G/B 通道压低（注意：第 2 维是宽，第 3 维才是通道）

    # 蓝色竖条：占满高，左侧 1/5 宽
    img[:, : max(width // 5, 1), 0] = 30
    img[:, : max(width // 5, 1), 2] = 200
    return img


def describe_shape(rgb: np.ndarray) -> str:
    """通关标准的一句话版（程序化）。"""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"不是 RGB 图: shape={rgb.shape}")
    h, w, _ = rgb.shape
    total_bytes = rgb.nbytes
    return (f"高{h} x 宽{w} x 3通道，dtype={rgb.dtype}，"
            f"共 {h*w*3:,} 个数、占 {total_bytes:,} 字节（每数 1 字节）")


# ---------------------------------------------------------------------------
# 2. 灰度化：两种做法
# ---------------------------------------------------------------------------

def grayscale_mean(rgb: np.ndarray) -> np.ndarray:
    """朴素灰度：三通道取平均。 (H,W,3) -> (H,W)，axis=-1 压掉通道维。"""
    return rgb.mean(axis=-1)


def grayscale_weighted(rgb: np.ndarray) -> np.ndarray:
    """加权灰度：0.299R + 0.587G + 0.114B（亮度感知权重，工程标准做法）。

    这就是"卷积核"的雏形：一组固定权重对邻域（这里是通道维）做加权求和。
    """
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return rgb.astype(np.float32) @ weights      # (H,W,3) @ (3,) -> (H,W) 广播点积


# ---------------------------------------------------------------------------
# 3. 手写 3x3 均值滤波（毕业设计核心）
# ---------------------------------------------------------------------------

def mean_filter_3x3_by_hand(gray: np.ndarray) -> np.ndarray:
    """纯 NumPy 手写 3x3 均值滤波：零填充 + 双重循环滑窗。

    步骤（背下来，这就是卷积的定义）：
      1) 边界外补 1 圈 0（zero-padding），保证输出与输入同尺寸；
      2) 滑动 3x3 窗口扫过每个位置 (i, j)；
      3) 每个位置取窗口内 9 个像素的平均值作为输出。
    效果：邻域平均 -> 抹平单点噪声（模糊的来源也是它）。
    """
    if gray.ndim != 2:
        raise ValueError("请输入二维灰度图")

    padded = np.pad(gray.astype(np.float32), pad_width=1, mode="constant",
                    constant_values=0)
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            window = padded[i: i + 3, j: j + 3]      # 3x3 邻域
            out[i, j] = window.mean()                # 核为全 1/9 的卷积
    return out


def mean_filter_3x3_vectorized(gray: np.ndarray) -> np.ndarray:
    """同一滤波器的向量化实现：9 次切片平移相加，代替 9*h*w 次单点访问。

    对照仓库 eval 的高斯模糊 cv2.GaussianBlur——工业代码永远用向量化/
    库实现，但"它等价于滑窗加权求和"这件事必须懂。
    """
    h, w = gray.shape
    padded = np.pad(gray.astype(np.float32), 1, mode="constant", constant_values=0)
    acc = np.zeros((h, w), dtype=np.float32)
    for di in (0, 1, 2):
        for dj in (0, 1, 2):
            acc += padded[di: di + h, dj: dj + w]     # 9 个方向各平移一次相加
    return acc / 9.0


def add_impulse_noise(gray: np.ndarray, ratio: float = 0.02, seed: int = 3):
    """撒椒盐噪声（随机像素置 0 或 255）——均值滤波要对付的敌人。"""
    rng = np.random.default_rng(seed)
    noisy = gray.copy()
    mask = rng.random(gray.shape) < ratio
    noisy[mask] = rng.choice([0, 255], size=int(mask.sum()))
    return noisy


# ---------------------------------------------------------------------------
# 4. 把滤波当"缺陷检测"用：离群点 = 与邻域均值差异大的点
# ---------------------------------------------------------------------------

def defect_map_by_local_contrast(gray: np.ndarray) -> np.ndarray:
    """局部对比度 = |原图 - 均值滤波图|。

    一行代码的"伪异常检测"：孤立亮点（椒盐噪/小砂眼雏形）在平滑图上
    会被抹掉，于是差值图上凸显——这就是无监督线"与正常模板的差距图"
    的最原始形态。
    """
    return np.abs(gray.astype(np.float32) - mean_filter_3x3_by_hand(gray))


# ---------------------------------------------------------------------------
# 演示入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("1) RGB 图即数组")
    print("-" * 60)
    rgb = make_rgb_image(12, 16)
    print(f"rgb.shape = {rgb.shape}，dtype = {rgb.dtype}")
    print(describe_shape(rgb))
    print(f"红色横条某像素 [y=4, x=8] = {rgb[4, 8].tolist()} (R,G/B 低 -> 偏红)")
    print(f"蓝色竖条某像素 [y=9, x=1] = {rgb[9, 1].tolist()} (B 高 -> 偏蓝)")

    print("=" * 60)
    print("2) 灰度化")
    print("-" * 60)
    g_mean = grayscale_mean(rgb)
    g_weighted = grayscale_weighted(rgb)
    print(f"平均灰度   shape={g_mean.shape}，红条区值={g_mean[4, 8]:.1f}")
    print(f"加权灰度   shape={g_weighted.shape}，红条区值={g_weighted[4, 8]:.1f}")
    print("（加权版更暗：人眼对 G 敏感、对 R/B 迟钝，红蓝条权重被压低）")

    print("=" * 60)
    print("3) 手写 3x3 均值滤波")
    print("-" * 60)
    gray = np.full((6, 6), 10.0, dtype=np.float32)
    gray[3, 3] = 100.0                                  # 一个孤立亮点
    filtered = mean_filter_3x3_by_hand(gray)
    print("输入（(3,3) 处埋了 100 的亮点）:")
    print(gray.astype(int))
    print("手写滤波输出:")
    print(filtered.astype(int))
    vec = mean_filter_3x3_vectorized(gray)
    assert np.allclose(filtered, vec), "两种实现应一致"
    print(f"亮点处 {gray[3,3]:.0f} -> {filtered[3,3]:.1f}（被邻域稀释）；"
          f"向量化版输出一致 ✓")

    print("=" * 60)
    print("4) 均值滤波去椒盐噪声（平滑背景 + 只看内部）")
    print("-" * 60)
    clean = np.tile(np.linspace(50, 70, num=12), (10, 1)).astype(np.float32)
    noisy = add_impulse_noise(clean, ratio=0.05)
    denoised = mean_filter_3x3_by_hand(noisy)
    # 零填充会让最外圈像素的窗口混入 0（亮图被"压暗"），教学统计先排除边界一圈
    inner = (slice(1, -1), slice(1, -1))
    err_before = float(np.abs((noisy - clean)[inner]).mean())
    err_after = float(np.abs((denoised - clean)[inner]).mean())
    print(f"内部区域平均误差: 加噪 {err_before:.2f} -> 滤波后 {err_after:.2f} "
          f"（噪声被邻域稀释；代价是边缘轻微变糊、边界一圈受零填充影响）")

    print("=" * 60)
    print("5) 局部对比度 = 伪异常检测（同样只看内部）")
    print("-" * 60)
    dmap = defect_map_by_local_contrast(noisy)
    inner_mask = np.zeros(noisy.shape, dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    hot = {tuple(int(v) for v in p) for p in np.argwhere((dmap > 30) & inner_mask)}
    real_noise = {tuple(int(v) for v in p)
                  for p in np.argwhere((noisy != clean) & inner_mask)}
    hit = len(hot & real_noise)
    print(f"真实噪声点 {len(real_noise)} 个，高响应点 {len(hot)} 个，命中 {hit} 个"
          f" -> 孤立离群点被抓（工业质检里这就是'区域门控/边界排除'的雏形）")

    print("=" * 60)
    print("通关自测（回答不出就回去重看对应文件）:")
    print(" a. 一张 1024x1024 的 RGB 图占多少字节？shape 怎么写？")
    print(" b. image[y, x] 里 y 和 x 分别指哪个方向？")
    print(" c. 均值滤波为什么能去噪、又为什么会糊？")
    print(" d. (H,W,3) @ (3,) 为什么不用循环就能算出加权灰度？")
