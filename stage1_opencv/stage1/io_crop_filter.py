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
    """按仓库口径裁 patch：从 (x0, y0) 起抠一块 size×size 的方形小图。
    ROI = Region Of Interest（感兴趣区域），主仓库叫"切 patch"。

    比喻：像在 Excel 里框选一片单元格——报坐标的习惯是 (列, 行)，但真去取
    数据时得先定位行。本函数唯一的易错点就是这层错位：

      ★ 参数顺序 (x0, y0) 与切片顺序 [y, x] 正好相反
        图是 (H, W) 的行列表：切片第 1 个下标管"行(y)"、第 2 个管"列(x)"。

    小学算术（测试同款 img = np.arange(30).reshape(5, 6)，5 行 6 列）：
        crop_roi(img, x0=2, y0=1, size=3)
          -> image[1 : 1+3, 2 : 2+3] = image[1:4, 2:5]  取第 1,2,3 行 × 第 2,3,4 列
          -> [[ 8,  9, 10],
              [14, 15, 16],
              [20, 21, 22]]
        断言钉死: patch[0,0] == img[1,2] == 8            （先 y 后 x）

    ★ 写反了不报错：写成 image[x0:x0+size, y0:y0+size] 形状仍是 3×3，只是抠错
      了地方（上例会变成 13,14,15 / 19,20,21 / 25,26,27）。静默 bug，外在只
      表现为"框整体偏了"——所以测试专门用 patch[0,0] == img[1,2] 守着。

    坐标纪律（全工程通用）：NumPy 用 [y, x]，而 cv2 的尺寸/坐标参数几乎都是
      (x, y)——warpAffine 的 dsize=(w,h)、phaseCorrelate 返回 (dx,dy)。两套并存，
      交接点就是下面这行切片。

    另注：返回值是视图（零拷贝、共享原图内存），改 patch 会波及原图，要独立
      副本需 .copy()。切片三条铁律（含头不含尾等）见 stage0.numpy_basics.crop_patch。
    """
    # 逗号前 = 行(y/高)、逗号后 = 列(x/宽)：参数里的 y0/x0 在此处对调
    #   例: y0=1, x0=2, size=3 -> image[1:4, 2:5]
    return image[y0: y0 + size, x0: x0 + size]


# ---------------------------------------------------------------------------
# 3. 手写滤波 vs OpenCV
# ---------------------------------------------------------------------------

def gaussian_kernel_1d(ksize: int, sigma: float) -> np.ndarray:
    """造一条"钟形权重表"（一维高斯核）：模糊时给每个邻居按"离得多近"打分，
    正中间的自己权重最大、越远的邻居越小——像一台打分器，不是把邻居等权平均。

    手算造表三步（ksize=5, sigma=1.0）：
      1) xs = np.arange(5) - 5//2 = [0,1,2,3,4] - 2 = [-2,-1,0,1,2]
         减 ksize//2 是把坐标中心对齐到 0（公式里要用 x²，必须对称）
      2) exp(-x² / 2σ²)，此处 2σ² = 2：
           x²      = [4,    1,     0,   1,     4]
           -x²/2   = [-2.0, -0.5,  0.0, -0.5,  -2.0]
           exp(..) = [0.1353, 0.6065, 1.0, 0.6065, 0.1353]
                                              ↑ 中心 x=0 得 exp(0)=1（峰值），
                                                向两侧衰减 -> 自然长成钟形
      3) 除以总和 2.4837 归一化：
           kernel  = [0.0545, 0.2442, 0.4026, 0.2442, 0.0545]   权重和 = 1

    ★ 第 3 步归一化是灵魂：不除的话权重和 = 2.48，加权求和会把每个像素放大
      2.48 倍——整幅图变亮。除完和为 1，"加权求和"就成了"加权平均"，亮度不变。
      测试钉两条：sum==1；左右对称（k == k[::-1]）。

    σ 是"胖瘦旋钮"（ksize=7 时的中心权重）：
        σ=0.5 -> 0.787（瘦，几乎不糊） | σ=1 -> 0.399 | σ=2 -> 0.216（胖，糊）
      ksize 要盖得住钟形的胖度，经验值 ksize ≈ 6σ+1 取奇数——太小会砍掉钟形的
      尾巴，太大则两端权重≈0 白算（σ=0.5 配 ksize=7，两端正好是 0.0）。
      仓库口径：无监督 eval 的固定后处理 GaussianBlur(33,33,σ4) 就是大 σ 抹平。

    为什么先写 1D：高斯可分离——二维钟形 = 横一维 ⊗ 纵一维（外积），所以二维
      模糊能拆成两次一维，计算量 k² -> 2k，这正是 cv2.GaussianBlur 快的原因之一
      （见下方 gaussian_kernel_2d）。
    对拍：与 cv2.getGaussianKernel(ksize, sigma)[:, 0] 逐位一致（atol=1e-8）。
    """
    # 偶数核没有唯一的正中心（长度 4 的"中间"是条缝），直接拒绝
    if ksize % 2 == 0:
        raise ValueError("ksize 必须是奇数")
    # 造坐标轴并把中心对齐到 0：例 ksize=5 -> [-2,-1,0,1,2]
    xs = np.arange(ksize) - ksize // 2
    # 高斯公式 exp(-x²/2σ²)：x=0 处 =1（峰值），两侧按 σ 衰减，合起来是钟形
    kernel = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    # 归一化：权重和变 1，加权求和 -> 加权平均，图不变亮也不变暗
    return kernel / kernel.sum()


def gaussian_kernel_2d(ksize: int, sigma: float) -> np.ndarray:
    """二维高斯核 = 两个一维核做外积（可分离性：高斯模糊可拆成两次一维卷积，
    这正是 cv2.GaussianBlur 快的原因之一）。"""
    k1 = gaussian_kernel_1d(ksize, sigma)
    return np.outer(k1, k1)


def manual_gaussian_blur(gray: np.ndarray, ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
    """手写二维高斯模糊：把每个像素换成"它和邻居的加权平均"，权重即钟形核。

    四步流水线：
      1) kernel = gaussian_kernel_2d(ksize, sigma)   造 k×k 钟形权重表（见上）
      2) padded = np.pad(gray, r, mode="reflect")    边缘"照镜子"补 r 圈，r = ksize//2
         为何补 r：像素的邻域是 i-r..i+r；不补则边缘像素没有完整邻居（黑边/越界）。
         reflect = 以边缘为轴翻折、边缘不重复自己 == cv2 的 BORDER_REFLECT_101，
         也正是 cv2.GaussianBlur 的默认边界（对拍能对上就靠它）。
      3) windows = sliding_window_view(padded, (ksize, ksize))   给每个像素配邻域
      4) np.einsum("ijkl,kl->ij", windows, kernel)               邻域加权求和

    ★ 第 3 步最反直觉，换个说法（4×4 图、ksize=3）：
      结果 = "H×W 个格子、每格装一张 k×k 小表"，形状 (H, W, k, k)——
          前两维 = 第几个像素，后两维 = 它的那张小表。
      等价于这个双层循环（自己跑一遍就懂）：
          result = np.zeros((H, W, k, k))
          for i in range(H):
              for j in range(W):
                  result[i, j] = padded[i:i+k, j:j+k]
      -> 4×4 图得到 4×4 个格子、每格 3×3；相邻格子只差一格，就是"窗口在滑动"。
      格子数 = (H+2r) - k + 1 = H：正因为补了 r 圈，输出的格子数才刚好等于原图
      尺寸，且每格正中心对准一个原像素。
      与循环版的唯一区别：result[i,j]=... 是真拷贝；sliding_window_view 是视图、
      一字节不拷，只改"读法"(strides)——大图上内存差几十倍。

    einsum 只记一条规则：字母出现在 -> 右边 = 保留，没出现 = 求和消掉。
      "ijkl,kl->ij" 即 i,j 保留（每个像素各算）、k,l 求和（邻域内加权相加）。

    为什么开头 .astype(np.float32)：加权平均必出小数；且 cv2.GaussianBlur 对 uint8
      走定点快速实现、与手写浮点版差约 1 灰度级，只有 float32 才逐位一致。
    """
    # 1) k×k 钟形权重表（两个 1D 核外积，见 gaussian_kernel_2d）
    kernel = gaussian_kernel_2d(ksize, sigma)
    # r = ksize//2 是邻域半径；补 r 圈后窗口起点数正好等于原图边长，输出不失真
    r = ksize // 2
    # 2) 边缘"照镜子"补 r 圈（float32 才与 cv2 对得齐）
    padded = np.pad(gray.astype(np.float32), r, mode="reflect")
    # 3) 每像素的 k×k 邻域：视图，形状 (H, W, k, k)，零拷贝
    windows = sliding_window_view(padded, (ksize, ksize))
    # 4) i,j 保留、k,l 求和：每个像素的邻域加权平均 = 模糊后的值
    return np.einsum("ijkl,kl->ij", windows, kernel)


def manual_median_filter(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """手写中值滤波：每个像素换成"它 k×k 邻域里的中位数"。

    ★ 前三行与 manual_gaussian_blur 完全相同，只差最后一行——高斯把邻域加权平均，
      中值把邻域排序后取正中间那个。"H×W 个格子、每格装一张 k×k 小表"那套理解
      直接复用，区别只在怎么把小表压扁成一个数。

    手算：含 1 个椒盐点的 3x3 窗口（真实值应为 128）
        窗口 = [[128,128,128], [128,128,128], [128,128,255]]
        排序 = [128,128,128,128,128,128,128,128,255]
        中值 = 第 5 个 = 128     <- 255 被排到末尾、够不着中间，只有一票
        均值 = 142.11            <- 255 直接参与求和，被拉高 14
      （含 1 个 0 的窗口同理：中值 128、均值 113.78。）
      机理：中值只关心"排在第几位"，极值再离谱也只有一票；均值是求和，有多少吃多少。

    作用（工程上为什么要它）：
      1) 流水线降噪防误检：defect_pipeline.classical_detect 是"灰度 -> 高斯模糊 ->
         反相 Otsu -> 闭运算 -> 面积过滤"，滤波是第一道工序；不滤的话孤立噪点会被
         Otsu 一并划成缺陷，最后报出面积 1~2px 的假缺陷（就是"误检"）。
      2) 独家本事——去噪还保边界（实测台阶图，左 100 右 200 模拟缺陷锐边）：
             原图  第4行: 100,100,100,100,100 | 200,200,200,200,200
             中值后      100,100,100,100,100 | 200,200,200,200,200   边界原样
             高斯后      100,100,100,100,125 | 175,200,200,200,200   糊成斜坡
         加 8% 椒盐后跨边界落差（理想 100）：中值 100 / 高斯 56。原因是窗口跨边界时
         一侧像素占多数，中位数就取那一侧 -> 台阶不被摊开。为何要紧：缺陷的面积/
         尺寸/bbox 都从二值掩码算出来，边界一糊尺寸就偏，min_area 门限随之失真。
      选型：椒盐（孤立 0/255 坏点）用中值；小幅高斯噪声用高斯（中值慢在要排序）。

    定位：与 cv2.medianBlur 对拍，属"手写一遍 -> 以后用库但懂原理"；也在回答
      README 通关标准"解释中值去椒盐、高斯去高斯噪声的机理差异"。
    """
    # r = ksize//2：邻域半径（与高斯版同款）
    r = ksize // 2
    # 边缘"照镜子"补 r 圈（与高斯版同款）
    padded = np.pad(gray.astype(np.float32), r, mode="reflect")
    # 每像素的 k×k 邻域：视图，形状 (H, W, k, k)，零拷贝（与高斯版同款）
    windows = sliding_window_view(padded, (ksize, ksize))
    # 唯一区别：此处取中位数（高斯版此处是 einsum 加权平均）
    # axis=(-2,-1) 压掉最后两维（-1=小表的列、-2=小表的行）-> 回到 (H,W)，等价 axis=(2,3)
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
