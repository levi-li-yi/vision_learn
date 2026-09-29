"""NumPy 数组操作：创建 / 切片 / 广播 / 归约 / dtype 陷阱。

为什么学这些：整个仓库的图像在进入 PyTorch 之前都是 NumPy 数组；
DataLoader 里的切片裁图（patch 切 1024）、Cut-Paste 的贴纸融合、
eval 里的连通域统计，底层全是这几个操作。

运行方式：
    python -m stage0.numpy_basics
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. 创建与 dtype：uint8 是图像的世界
# ---------------------------------------------------------------------------

def make_gradient_image(height: int, width: int) -> np.ndarray:
    """构造一张横向渐变灰度图 (H, W)，值域 [0, 255]。

    np.arange + 广播生成，比双重循环快得多——这是 NumPy 的第一课：
    用"整块数组运算"代替"逐元素循环"（向量化）。
    """
    row = np.linspace(0, 255, num=width, dtype=np.float32)   # (W,)
    img = np.tile(row, (height, 1))                          # (H, W)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def make_checkerboard(size: int, cell: int = 16) -> np.ndarray:
    """棋盘格图：向量化思维示范（无循环）。

    规则：格子坐标 (行号+列号) 为偶数则白、奇数则黑——
    用两次广播比较一次性生成整图，代替双重 for 循环。
    """
    cell_index = np.arange(size) // cell                    # 每像素属于第几个格
    parity = (cell_index[:, None] + cell_index[None, :]) % 2  # 广播: (S,1)+(1,S)->(S,S)
    return np.where(parity == 0, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. 切片：裁图就是切片（对应仓库按 (x0, y0, patch_size) 裁 patch）
# ---------------------------------------------------------------------------

def crop_patch(image: np.ndarray, x0: int, y0: int, patch_size: int) -> np.ndarray:
    """从 (H, W) 图里裁一个 patch。

    坐标系约定（务必背下来）：
      - shape 顺序 (高, 宽)，即 (rows, cols)
      - 索引顺序 [y, x]——先行(高)后列(宽)
    仓库 meta.json 里的切片坐标 (x0, y0) 指"列起点, 行起点"，切片时写作
    image[y0 : y0+size, x0 : x0+size]。
    """
    if image.ndim != 2:
        raise ValueError("本函数只处理二维灰度图")
    return image[y0: y0 + patch_size, x0: x0 + patch_size]


# ---------------------------------------------------------------------------
# 3. 归约：沿哪个 axis 算，是新手第一大坑
# ---------------------------------------------------------------------------

def channel_means(rgb: np.ndarray) -> np.ndarray:
    """按通道求均值。

    记忆法：axis 就是要被"压掉"的维度。
    RGB 图形状 (H, W, 3)，压掉 H 和 W (axis=(0,1))，剩 (3,)。
    """
    return rgb.mean(axis=(0, 1))


def normalize(image: np.ndarray, mean, std) -> np.ndarray:
    """ImageNet 归一化：(x/255 - mean) / std——仓库训练/推理的固定第一步。

    本函数同时是"广播"的示范：image (H,W,3) 减去 mean (3,)，
    NumPy 自动把 (3,) 对齐到最后一维，逐通道相减。
    """
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return (image.astype(np.float32) / 255.0 - mean) / std


# ---------------------------------------------------------------------------
# 4. dtype 陷阱：uint8 会溢出回绕（真实踩过的坑：亮度扰动爆掉）
# ---------------------------------------------------------------------------

def safe_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """亮度乘法：必须先升 dtype 再裁剪再转回，否则 uint8 溢出回绕。

    错误写法: (image * factor).astype(np.uint8)
      250 * 1.15 = 287.5 -> uint8 只留 8 位 -> 31（本该是 255）
    """
    scaled = image.astype(np.float32) * factor
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 5. 视图 vs 拷贝：切片默认不复制内存
# ---------------------------------------------------------------------------

def demo_view_vs_copy():
    a = np.arange(12).reshape(3, 4)
    view = a[0:2, :]            # 视图：与 a 共享内存
    copy = a[0:2, :].copy()      # 拷贝：独立内存

    a[0, 0] = 999
    assert view[0, 0] == 999, "视图应看到修改"
    assert copy[0, 0] == 0, "拷贝不应看到修改"
    return a


# ---------------------------------------------------------------------------
# 演示入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("1) 创建：渐变图 / 棋盘格")
    print("-" * 60)
    grad = make_gradient_image(4, 8)
    print(f"渐变图 shape={grad.shape} dtype={grad.dtype}")
    print(grad)
    board = make_checkerboard(64)
    print(f"棋盘格 shape={board.shape} 唯一值={np.unique(board)} "
          f"亮块占比={(board == 255).mean():.2f}")

    print("=" * 60)
    print("2) 切片：crop_patch(grad, x0=4, y0=1, size=3)")
    print("-" * 60)
    print(f"全图 (高,宽)={grad.shape}，注意打印方向：每行是一'高'")
    print(crop_patch(grad, x0=4, y0=1, patch_size=3))

    print("=" * 60)
    print("3) 归约 + 广播：通道均值与 ImageNet 归一化")
    print("-" * 60)
    rgb = np.stack([np.full((2, 2), 10, np.uint8),
                    np.full((2, 2), 20, np.uint8),
                    np.full((2, 2), 30, np.uint8)], axis=-1)   # (2,2,3)
    print(f"rgb shape={rgb.shape}，通道均值={channel_means(rgb)}")
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    print(f"归一化后 [0,0] 位置三通道值 = {normalize(rgb, mean, std)[0, 0]}")

    print("=" * 60)
    print("4) dtype 陷阱：uint8 溢出")
    print("-" * 60)
    patch = np.array([250, 100, 0], dtype=np.uint8)
    wrong = (patch * 1.15).astype(np.uint8)          # 错误示范
    right = safe_brightness(patch, 1.15)
    print(f"原始        : {patch}")
    print(f"错误写法    : {wrong}   <- 250 变 31，溢出回绕！")
    print(f"安全写法    : {right}   <- 250 正确饱和到 255")

    print("=" * 60)
    print("5) 视图 vs 拷贝")
    print("-" * 60)
    a = demo_view_vs_copy()
    print(f"修改源数组后：\n{a}\n（view 跟着变，copy 不变——切片默认是视图）")
