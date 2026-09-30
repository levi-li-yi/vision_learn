"""NumPy 数组操作：创建 / 切片 / 广播 / 归约 / dtype 陷阱。

为什么学这些：整个仓库的图像在进入 PyTorch 之前都是 NumPy 数组；
DataLoader 里的切片裁图（patch 切 1024）、Cut-Paste 的贴纸融合、
eval 里的连通域统计，底层全是这几个操作。

两大前置认知（读全文件的钥匙）：
  1. np 数组 = 一张数字表格；np 的全部本事 = 不写循环、一次操作整张表格；
  2. 参数两种传法：位置参数按顺序对号（np.clip(x, 0, 255) 的 0 和 255），
     关键字参数指名道姓（dtype=np.uint8）。dtype = 装什么规格的数：
       uint8   0~255 整数，图像标准，1 字节（装不下会"绕圈"：300 变 44）
       int32   普通整数，不指定时的默认，4 字节
       float32 小数（单精度）；float64 小数（双精度，linspace 的默认）
     参数旋钮共四大类：类型(dtype=) / 形状(元组) / 方向(axis=) / 范围(两个数)。

完整 API 手册（每个方法的全部参数+可运行例子）见同目录 NUMPY_API_REFERENCE.md。

运行方式：
    python -m stage0.numpy_basics
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. 创建与 dtype：uint8 是图像的世界
# ---------------------------------------------------------------------------

def make_gradient_image(height: int, width: int) -> np.ndarray:
    """构造一张横向渐变灰度图 (H, W)，值域 [0, 255]。

    NumPy 第一课——向量化：全程无 for 循环，三行造一张图。
    """
    # np.linspace(起点, 终点, num=个数, dtype=类型)
    #   从起点到终点"均匀撒 num 个点"，两头都包含：linspace(0,255,8)
    #   -> [0, 36.4, 72.9, ..., 255]。num= 关键字传"撒几个"（不传默认 50！）；
    #   dtype=float32 造小数表（均匀点大多是 36.4 这种小数，且下一步要参与运算）。
    #   对比记忆：arange 管"步长"，linspace 管"个数"。
    row = np.linspace(0, 255, num=width, dtype=np.float32)   # (W,) 一条一维横线

    # np.tile(表格, (行份数, 列份数))
    #   把 row 当印章复印：行方向印 height 份、列方向 1 份——一条线 (W,)
    #   摞成一张表 (H, W)。"形状/布局类参数一律用元组装"是 np 的统一风格。
    img = np.tile(row, (height, 1))                          # (H, W)

    # 安全三连（顺序固定，详细机理见 safe_brightness）：
    #   np.round(表)          四舍五入，36.4 -> 36.
    #   np.clip(表, 0, 255)   超界拉回：>255 改 255、<0 改 0
    #   .astype(np.uint8)     转图像标准类型。注意：astype 是"砍小数"不是
    #                         四舍五入（1.7 -> 1），所以必须排在 round 之后
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def make_checkerboard(size: int, cell: int = 16) -> np.ndarray:
    """棋盘格图：向量化思维示范（无循环）。

    规则：格子坐标 (行号+列号) 为偶数则白、奇数则黑。
    """
    # np.arange(个数)：造 0,1,2,... 连续整数（arange(64) -> [0..63]）。
    # // 整除（7//2=3，只要商）：坐标除以格宽 -> "属于第几格"，
    # 得到 [0]*16+[1]*16+[2]*16+[3]*16 这种归格编号。
    cell_index = np.arange(size) // cell

    # [:, None] / [None, :]：切片写法"插一根长 1 的新轴"，把 (64,) 摆竖成
    #   (64,1)、摆横成 (1,64)——专给广播摆姿势。
    # 竖 (64,1) + 横 (1,64)：广播自动撑成 (64,64) 的"行列加法表"，
    #   位置 (i,j) 上 = 行格号[i] + 列格号[j]。
    # % 取余（7%2=1）：和为偶数 -> 0、奇数 -> 1，黑白判据就出来了。
    parity = (cell_index[:, None] + cell_index[None, :]) % 2  # (S,S) 的 0/1 表

    # np.where(对错表, 填A, 填B)：对(True)的位置填 255、错(False)的位置填 0
    #   ——"逐格 if-else 一次做完"。常与比较运算连招：先比较得对/错表、再上色。
    return np.where(parity == 0, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. 切片：裁图就是切片（对应仓库按 (x0, y0, patch_size) 裁 patch）
# ---------------------------------------------------------------------------

def crop_patch(image: np.ndarray, x0: int, y0: int, patch_size: int) -> np.ndarray:
    """从 (H, W) 图里裁一个 patch。

    切片语法 a[行起:行止, 列起:列止]，三条铁律（全工程通用）：
      - 含头不含尾：a[1:4] 只要第 1,2,3 个；
      - 逗号前管"行(y/高)"、逗号后管"列(x/宽)"——注意与参数 (x0, y0) 顺序相反；
      - 冒号单独出现 = 全要；冒号两边可写算式（y0 : y0+size = 从 y0 起要 size 个）。
    """
    # .ndim：表格自带的"维数"标签（属性，不加括号）。2 = 灰度图表、3 = 彩色图
    # ——入口安检：3 维的彩色图不许冒充 2 维灰度图（错误越早暴露越好定位）。
    if image.ndim != 2:
        raise ValueError("本函数只处理二维灰度图")
    return image[y0: y0 + patch_size, x0: x0 + patch_size]


# ---------------------------------------------------------------------------
# 3. 归约与广播
# ---------------------------------------------------------------------------

def channel_means(rgb: np.ndarray) -> np.ndarray:
    """按通道求均值：R/G/B 各自的平均亮度。

    .mean(axis=方向)：求平均。口诀：**axis 写谁，谁就消失**——
    (H,W,3) 写 axis=(0,1) 即压掉"高和宽"两根方向，只剩 (3,)。
    axis 可为单个数字或元组；不传则全部压成一个数。
    （sum/max/min 同理，同一家族。）属性同理三件套：shape 多大、
    ndim 几维、dtype 装什么——都是"铭牌"，不加括号。
    """
    return rgb.mean(axis=(0, 1))


def normalize(image: np.ndarray, mean, std) -> np.ndarray:
    """ImageNet 归一化：(x/255 - mean) / std——仓库训练/推理的固定第一步。

    返回行从左往右四件事：
      1. .astype(np.float32)  uint8 升级成小数（整数运算会丢小数/溢出）；
      2. / 255.0              0~255 压到 0~1（网络爱吃 0 附近的小数字）；
      3. - mean               广播：(H,W,3) 减 (3,)，NumPy 自动把 (3,) 对齐到
                              最后一维——R 减 R 的均值、G 减 G 的、B 减 B 的；
      4. / std                同理，除以各通道的波动范围。
    广播 = 免费的隐形复制 + 逐元素运算：形状不同的两表相运算，短的自动
    "沿缺的方向复制撑开"再逐格算，不用写循环。
    mean/std 的数值是 ImageNet 百万张图统计出来的"世界平均"。
    """
    # np.asarray(数据, dtype=类型)：把列表转成表格。参数与 np.array 相同，
    # 唯一区别：传进来的已是表格时不复制（能省则省版，省内存）。
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return (image.astype(np.float32) / 255.0 - mean) / std


# ---------------------------------------------------------------------------
# 4. dtype 陷阱：uint8 会溢出回绕（真实踩过的坑：亮度扰动爆掉）
# ---------------------------------------------------------------------------

def safe_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """亮度乘法：必须"float -> 乘 -> round -> clip -> 转 uint8"走完。

    错误写法 (image * factor).astype(np.uint8) 为什么炸：
    250 * 1.15 = 287.5；uint8 只有 8 位、最大装 255，装不下就"绕圈"：
    287 - 256 = 31——本想调亮，250 反而变成 31 的深灰色（溢出回绕）。
    """
    # 先升小数再乘：uint8 直接乘小数，结果类型和溢出都不可控
    scaled = image.astype(np.float32) * factor
    # 安全三连：round 四舍五入 -> clip 夹 0~255（287.5 饱和到 255，不绕圈）
    # -> astype 转回图像标准类型。主工程所有亮度扰动/贴纸亮度都是这套写法。
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 5. 视图 vs 拷贝：切片默认不复制内存
# ---------------------------------------------------------------------------

def demo_view_vs_copy():
    """切片默认是"视图"（看同一批数据的另一扇窗），.copy() 才是真复制。

    工程意义：Cut-Paste 改贴纸/背景时，若拿切片当草稿随便改，会污染
    原始训练数据——想隔离必须显式 .copy()。
    """
    # np.arange(12) 造 0..11；.reshape(3,4) 数字不变、换个摆法（总数必须相等）
    a = np.arange(12).reshape(3, 4)
    view = a[0:2, :]            # 切片 = 视图：与 a 共享同一批数（另一扇窗）
    copy = a[0:2, :].copy()     # .copy() = 复印件：独立内存，从此各过各的

    a[0, 0] = 999               # 从 a 这扇窗进去改数
    assert view[0, 0] == 999, "视图应看到修改"    # 同一间房，看得到
    assert copy[0, 0] == 0, "拷贝不应看到修改"     # 早搬走了，不受影响
    return a


# ---------------------------------------------------------------------------
# 演示入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("1) 创建：渐变图 / 棋盘格")
    print("-" * 60)
    grad = make_gradient_image(4, 8)
    # .shape/.dtype：铭牌属性——多大盘子 / 装什么类型
    print(f"渐变图 shape={grad.shape} dtype={grad.dtype}")
    print(grad)
    board = make_checkerboard(64)
    # np.unique(表)：表里有哪几种不同的值（验证棋盘格只有黑白 0 和 255）。
    # (board == 255)：逐格比较得对/错表；对/错表 .mean() = True 占比 = 白块占比
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
    # np.stack([表A,表B,表C], axis=-1)：三张 (2,2) 纯色片沿"最后一根方向"摞成
    # (2,2,3)——"三张透明胶片叠成彩色图"的动作本身。axis=0 则摞在最前(3,H,W)。
    # np.full(形状, 值, 类型)：造全填同一个数的表；第 3 个位置参数恰好是 dtype。
    rgb = np.stack([np.full((2, 2), 10, np.uint8),
                    np.full((2, 2), 20, np.uint8),
                    np.full((2, 2), 30, np.uint8)], axis=-1)   # (2,2,3)
    print(f"rgb shape={rgb.shape}，通道均值={channel_means(rgb)}")
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    print(f"归一化后 [0,0] 位置三通道值 = {normalize(rgb, mean, std)[0, 0]}")

    print("=" * 60)
    print("4) dtype 陷阱：uint8 溢出")
    print("-" * 60)
    # np.array(数据, dtype=类型)：不传 dtype 则 np 猜（这里明确用图像标准 uint8）
    patch = np.array([250, 100, 0], dtype=np.uint8)
    wrong = (patch * 1.15).astype(np.uint8)          # 错误示范：287.5 绕圈变 31
    right = safe_brightness(patch, 1.15)
    print(f"原始        : {patch}")
    print(f"错误写法    : {wrong}   <- 250 变 31，溢出回绕！")
    print(f"安全写法    : {right}   <- 250 正确饱和到 255")

    print("=" * 60)
    print("5) 视图 vs 拷贝")
    print("-" * 60)
    a = demo_view_vs_copy()
    print(f"修改源数组后：\n{a}\n（view 跟着变，copy 不变——切片默认是视图）")
