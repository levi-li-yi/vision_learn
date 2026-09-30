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

注释里每个 API 都带"例: 入参 -> 出参"，全部可在交互环境亲手验证
（.venv/Scripts/python.exe）。完整参数手册见同目录 NUMPY_API_REFERENCE.md。

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
    # np.linspace(起点, 终点, num=个数, dtype=类型)：从起点到终点均匀撒 num 个点，
    # 两头都包含；num= 不传默认 50 个。
    #   例: np.linspace(0, 10, num=3)              -> array([ 0.,  5., 10.])
    #   本行(width=8): np.linspace(0, 255, num=8)  -> [0., 36.4, 72.9, ..., 255.] shape (8,)
    #   dtype=float32 造小数表（均匀点大多是 36.4 这种小数，且下一步要参与运算）。
    #   对比记忆：arange 管"步长"，linspace 管"个数"。
    row = np.linspace(0, 255, num=width, dtype=np.float32)   # (W,) 一条一维横线

    # np.tile(表格, (行份数, 列份数))：把表格当印章复印铺贴。
    #   例: np.tile([1, 2, 3], (2, 1))  -> [[1,2,3], [1,2,3]]  shape (2,3)
    #   本行: row(8,) x (4,1)           -> 同一条线摞 4 行     shape (4,8)
    #   "形状/布局类参数一律用元组装"是 np 的统一风格。
    img = np.tile(row, (height, 1))                          # (H, W)

    # 安全三连（顺序固定，机理见 safe_brightness）：
    #   np.round(表) 四舍五入
    #     例: np.round([36.4, 72.9]) -> array([36., 73.])
    #   np.clip(表, 下限, 上限) 超界拉回
    #     例: np.clip([250, 287.5, -3], 0, 255) -> array([250., 255., 0.])
    #   .astype(类型) 换类型。注意：astype 是"砍小数"不是四舍五入，必须排 round 后
    #     例: np.array([1.7, 2.2]).astype(np.uint8) -> array([1, 2])（1.7 砍成 1！）
    #         np.array([250, 300, -5], dtype=np.uint8) -> [250, 44, 251]（绕圈）
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def make_checkerboard(size: int, cell: int = 16) -> np.ndarray:
    """棋盘格图：向量化思维示范（无循环）。

    规则：格子坐标 (行号+列号) 为偶数则白、奇数则黑。
    """
    # np.arange(个数)：造 0,1,2,... 连续整数。
    #   例: np.arange(5) -> array([0, 1, 2, 3, 4])
    # // 整除（7//2=3，只要商）：坐标除以格宽 -> "属于第几格"。
    #   例(size=64,cell=16): np.arange(64)//16
    #     -> [0,0,...,0, 1,1,...,1, 2,..., 3,...]（每 16 个一组）shape (64,)
    cell_index = np.arange(size) // cell

    # [:, None] / [None, :]：切片写法"插一根长 1 的新轴"（专给广播摆姿势）。
    #   例: np.arange(4)            -> [0,1,2,3]             shape (4,) 原始一维
    #       np.arange(4)[:, None]   -> [[0],[1],[2],[3]]     shape (4,1) 摆竖
    #       np.arange(4)[None, :]   -> [[0,1,2,3]]           shape (1,4) 摆横
    # 竖(64,1) + 横(1,64)：广播自动撑成 (64,64) 的"行列加法表"。% 取余（7%2=1）。
    #   迷你例: (竖 + 横)      = [[1],[2]] + [[10,20]] -> [[11,21],[12,22]] shape(2,2)
    #           (竖 + 横) % 2  = [[11,21],[12,22]] % 2 -> [[1,1],[0,0]]    （奇偶判据）
    #   本行: (64,1) + (1,64) -> (64,64)，位置(i,j) = 行格号[i] + 列格号[j]
    parity = (cell_index[:, None] + cell_index[None, :]) % 2  # (S,S) 的 0/1 表

    # np.where(对错表, 填A, 填B)：对(True)的位置填 A、错(False)的位置填 B——
    # "逐格 if-else 一次做完"。
    #   例: np.where([True, False, True], 255, 0) -> array([255,   0, 255])
    #   本行: 偶数格 -> 255（白），奇数格 -> 0（黑）
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
    # .ndim：表格自带的"维数"标签（属性，不加括号）——入口安检。
    #   例: np.zeros((2, 3)).ndim   -> 2   （灰度图表）
    #       np.zeros((2, 3, 3)).ndim -> 3  （彩色图，不许冒充 2 维灰度图）
    if image.ndim != 2:
        raise ValueError("本函数只处理二维灰度图")
    # 切片示例:
    #   a = [[10, 20, 30],
    #        [40, 50, 60]]
    #   a[0:2, 1:3] -> [[20, 30],   # 第 0~1 行、第 1~2 列
    #                   [50, 60]]
    #   本行: image[1:4, 4:7]（y0=1,x0=4,size=3 时）-> 3x3 小块
    return image[y0: y0 + patch_size, x0: x0 + patch_size]


# ---------------------------------------------------------------------------
# 3. 归约与广播
# ---------------------------------------------------------------------------

def channel_means(rgb: np.ndarray) -> np.ndarray:
    """按通道求均值：R/G/B 各自的平均亮度。

    .mean(axis=方向)——为什么 axis=0 是压"行"？axis 不是"行/列"的名字，而是
    **shape 元组里第几个位置的编号**（与下标从 0 数是同一套规矩）：

        m = [[10, 20],
             [30, 40]]                    # shape (2, 2)
        格子地址 m[行号, 列号]：行号 = 地址第 0 个数（axis=0），
                                 列号 = 地址第 1 个数（axis=1）

    axis 指"沿哪根方向走、把走过的数合并"，不是"对哪一堆做"：
      * mean(axis=0)：沿"从上到下"合并（列号相同的凑一堆）-> 每列一个平均
            例: m.mean(axis=0) -> array([20., 30.])   # (10+30)/2, (20+40)/2
      * mean(axis=1)：沿"从左到右"合并（行号相同的凑一堆）-> 每行一个平均
            例: m.mean(axis=1) -> array([15., 35.])   # (10+20)/2, (30+40)/2
      * 不传 -> 全压成一个数：m.mean() -> 25.0

    反直觉警告：axis=0 不是"对每一行求平均"，恰好相反。
    自查技巧：被压掉的轴会从 shape 里消失（用 2x3 例子看得出区别）——
        (2,3) --axis=0--> (3,)  第 0 位没了 = 压了行，剩 3 列各一个数
        (2,3) --axis=1--> (2,)  第 1 位没了 = 压了列，剩 2 行各一个数

    口诀：axis 写谁，谁就消失；元组 (0,1) = 一次压两根（sum/max/min 同家族）。
    本行: (2,2,3).mean(axis=(0,1)) -> (3,) = [R均值, G均值, B均值]
    属性三件套（都不加括号）：shape 多大、ndim 几维、dtype 装什么。
    """
    return rgb.mean(axis=(0, 1))


def normalize(image: np.ndarray, mean, std) -> np.ndarray:
    """ImageNet 归一化：(x/255 - mean) / std——仓库训练/推理的固定第一步。

    为什么：网络爱吃"0 附近、波动统一"的小数字；原始图是 0~255 的大数字，
    且三通道脾气不同（R 普遍偏亮、B 偏暗），直接喂训练又慢又不稳。

    三步流水线（比喻：考试分数标准化）：
      0. .astype(np.float32) 先转小数（整数除法丢小数，10/255=0.039 装不下）；
      1. / 255.0             换"百分制"：0~255 -> 0~1。例: 255 -> 1.0, 10 -> 0.039；
      2. - mean              减"全班平均分"：比平均亮记正数、暗记负数，中心挪到 0。
                             广播：(H,W,3) 减 (3,)，短的自动摊开对齐——
                             R 减 R 的均值、G 减 G 的、B 减 B 的（各用各的尺子）；
      3. / std               除"全班波动幅度"：三个通道的波动统一成 ±1 左右。
                             为什么除：同样"高出平均 5 分"，波动大的考试里不算
                             什么、波动小里就是学霸——除以波动，跨通道才可比。
    mean/std 是 ImageNet 百万张自然照片统计的"平均长相/平均波动"：
      mean = [0.485, 0.456, 0.406]，std = [0.229, 0.224, 0.225]

    真实数字链（演示图 rgb[0,0] = [10, 20, 30]）：
      [10, 20, 30]
      --/255-->    [ 0.039,  0.078,  0.118]
      --- -mean--> [-0.446, -0.378, -0.288]   # 比平均暗，全负
      --- /std --> [-1.95,  -1.69,  -1.28 ]   # "比平均暗 1~2 个波动单位"
    所有像素最终落在约 -2.x ~ +2.x 的小区间——网络训练稳定的标准开场。
    口径纪律：训练与推理必须用同一套 mean/std；尺子换了，网络看到的世界就变了。
    """
    # np.asarray(数据, dtype=类型)：把列表转表格。与 np.array 参数相同，唯一区别：
    # 传进来的已是表格时不复制（能省则省版）。
    #   例: np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    #     -> array([0.485, 0.456, 0.406], dtype=float32)   shape (3,)
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
    # 先升小数再乘（uint8 直接乘小数，结果类型和溢出都不可控）：
    #   例: np.array([250], np.uint8).astype(np.float32) * 1.15 -> array([287.5])
    scaled = image.astype(np.float32) * factor
    # 安全三连的完整数值链（以 250 x 1.15 为例）：
    #   287.5 --round--> 288. --clip(0,255)--> 255. --astype--> 255（正确饱和）
    #   对比错误写法: 287.5 --astype(uint8)--> 31（绕圈，287-256）
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 5. 视图 vs 拷贝：切片默认不复制内存
# ---------------------------------------------------------------------------

def demo_view_vs_copy():
    """切片默认是"视图"（看同一批数据的另一扇窗），.copy() 才是真复制。

    工程意义：Cut-Paste 改贴纸/背景时，若拿切片当草稿随便改，会污染
    原始训练数据——想隔离必须显式 .copy()。
    """
    # np.arange(12) 造 0..11；.reshape(行,列) 数字不变换摆法（总数必须相等）。
    #   例: np.arange(12).reshape(3, 4)
    #     -> [[ 0,  1,  2,  3],
    #         [ 4,  5,  6,  7],
    #         [ 8,  9, 10, 11]]
    a = np.arange(12).reshape(3, 4)
    view = a[0:2, :]            # 切片 = 视图：与 a 共享同一批数（另一扇窗）
    copy = a[0:2, :].copy()     # .copy() = 复印件：独立内存，从此各过各的

    a[0, 0] = 999               # 从 a 这扇窗进去改数
    # 之后的现场证据：
    #   a     -> [[999, 1, 2, 3], ...]
    #   view[0,0] -> 999（同一间房，看得到）
    #   copy[0,0] -> 0  （早搬走了，不受影响）
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
    # .shape/.dtype：铭牌属性——多大盘子 / 装什么类型
    #   例: grad.shape -> (4, 8)；grad.dtype -> dtype('uint8')
    print(f"渐变图 shape={grad.shape} dtype={grad.dtype}")
    print(grad)
    board = make_checkerboard(64)
    # np.unique(表)：表里有哪几种不同的值。
    #   例: np.unique([0, 255, 0, 255]) -> array([  0, 255])
    # (board == 255)：逐格比较得对/错表；对/错表 .mean() = True 占比 = 白块占比。
    #   例: np.array([True, False, True]).mean() -> 0.667
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
    # np.full(形状, 值, 类型)：造全填同一个数的表；第 3 个位置参数恰好是 dtype。
    #   例: np.full((2, 2), 10, np.uint8)
    #     -> [[10, 10],
    #         [10, 10]]   dtype=uint8
    # np.stack([表A,表B,表C], axis=-1)：沿"最后一根方向"摞，新增最后一维——
    # "三张透明胶片叠成彩色图"的动作本身。
    # （axis 编号规则见 channel_means 注释：0/1/2 = shape 里第 0/1/2 个数；
    #   -1 = 倒数第一根，即最后一维）
    #   例: np.stack([全10的(2,2), 全20的(2,2), 全30的(2,2)], axis=-1)
    #     -> [[[10,20,30],[10,20,30]],
    #         [[10,20,30],[10,20,30]]]   shape (2,2,3)
    #   （若 axis=0 则摞在最前变 (3,2,2)，通道就不在最后了）
    rgb = np.stack([np.full((2, 2), 10, np.uint8),
                    np.full((2, 2), 20, np.uint8),
                    np.full((2, 2), 30, np.uint8)], axis=-1)   # (2,2,3)
    print(f"rgb shape={rgb.shape}，通道均值={channel_means(rgb)}")
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    print(f"归一化后 [0,0] 位置三通道值 = {normalize(rgb, mean, std)[0, 0]}")

    print("=" * 60)
    print("4) dtype 陷阱：uint8 溢出")
    print("-" * 60)
    # np.array(数据, dtype=类型)：不传 dtype 则 np 猜（整数猜 int32）。
    #   例: np.array([250, 100, 0], dtype=np.uint8)
    #     -> array([250, 100, 0], dtype=uint8)
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
