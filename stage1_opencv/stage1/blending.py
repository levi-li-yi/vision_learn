"""Alpha 混合与边缘羽化：Cut-Paste 缺陷合成的"贴纸"全套。

对应主仓库的 DefectTransformer / DefectBlender：
- 贴纸变换：翻转 0.5 / 旋转 0-360 / 缩放 0.7-1.3 / 亮度 0.75-1.15；
- 高保真融合：核心区保持贴纸原始颜色，边缘高斯羽化（blend_kernel_size=5）；
- 防御性 GT 收敛：blend 后 GT mask &= allowed（越界的标注不算数）。

为什么边缘要羽化：硬边贴图有"贴纸感"（合成域差），模型学到的是
"有硬边的异物"而不是"缺陷外观"；羽化让边界自然过渡，缩小合成图与
真实缺陷图的差距——README 里"单个缺陷只检出一部分 = 合成域差"的对策。

运行：python -m stage1.blending
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. 贴纸制造：不规则缺陷形状 + 蒙版
# ---------------------------------------------------------------------------

def make_sticker(size=64, seed=0):
    """造一张缺陷贴纸：几个高斯斑叠加成不规则暗斑。

    工序三步：4 滴随机墨叠成浓度山峦 -> 35% 等高线圈出变形虫形状（蒙版
    =GT）-> 暗噪点当材质、按形状裁下。返回 (sticker uint8 图, mask
    uint8 0/255 蒙版)——合成 GT 直接由 mask 来，这就是"贴上什么形状，
    答案就是什么形状"（免标注红利）。可复现原理：随机只在摇参数时发生
    （每滴 4 个数），骰子是伪随机搅拌机（同 seed 吐同一数列，是重放录像
    非重掷运气），"化开"是静止的钟形公式、非模拟过程——同 seed 必出
    同一张贴纸。
    """
    rng = np.random.default_rng(seed)          # 伪随机搅拌机：seed=配方号，同 seed 同数列
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)   # 每像素的 (y,x) 门牌号
    blob = np.zeros((size, size), np.float32)  # 全零"墨水浓度地图"，等着往上叠墨
    for _ in range(4):                         # 叠 4 滴：单滴太规则（正圆），多滴才像缺陷
        bx = size / 2 + rng.normal(0, size / 8)    # 本滴落点 x：画布中心 + 正态抖动
        by = size / 2 + rng.normal(0, size / 8)    # 落点 y 同款抖动
        s2 = 2 * (size / 6 * rng.uniform(0.6, 1.4)) ** 2   # 本滴胖瘦（σ² 项，0.6~1.4 倍随机）
        # 一滴墨 = 静止的钟形公式（参数定则每像素浓度唯一，非"洇开"模拟）：
        # 中心~1、向外按 e^(-d²/σ²) 衰减，乘随机浓淡 0.5~1.0 后整滴叠加
        blob += rng.uniform(0.5, 1.0) * np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / s2)
    mask = (blob > blob.max() * 0.35).astype(np.uint8) * 255  # 35% 等高线圈形状：蒙版=GT
    texture = rng.integers(20, 90, (size, size), dtype=np.uint8)   # 暗噪点当缺陷材质（20~90）
    sticker = cv2.bitwise_and(texture, texture, mask=mask)  # 拿形状当剪刀裁材质，蒙版外清零
    return sticker, mask                       # 贴纸 + 形状成对出厂，蒙版身兼 GT


def feather_mask(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """边缘羽化：对二值蒙版做高斯模糊，得到 [0,1] 的 alpha 渐变。

    给剪纸般锋利的 0/255 边缘"抹柔和"（对应仓库 blend_kernel_size=5）：
    圈内 1（全贴纸）、圈外 0（不碰）、边缘渐变带（两者温柔混合），
    消掉硬边的"贴纸感"。注意 0~1 是按位置铺开的渐变斜坡——离边多远
    定数值多少（确定性加权平均），不是随机浮点数。kernel 越大越宽。
    为什么羽化蒙版而非贴纸：alpha 管"掺多少"=纯形状问题，蒙版是形状
    权威（干净 0/255 台阶，一模糊就是完美斜坡）；贴纸自己是暗纹理，
    当配方会把缺陷洗淡、纹理糊掉。类型 float32——比例需要小数，与
    下游 alpha_blend 的运算同型号。
    """
    if ksize % 2 == 0:                         # 门卫：窗口须奇数——偶数没有正中心像素，
        raise ValueError("ksize 必须是奇数")    # 结果会整体偏移半像素；响亮失败优于安静出错
    # 高斯模糊 = 邻域钟形加权平均：圈内邻居全 255 -> 仍 1；圈外全 0 -> 仍 0；
    # 跨边像素邻居半内半外 -> 平均出中间值，锋利边界变 ~ksize 宽渐变斜坡
    # （数值由离边远近唯一决定，非随机）。σ 传 0 = 按窗口自动推算。
    return cv2.GaussianBlur(mask, (ksize, ksize), 0).astype(np.float32) / 255.0  # /255 换 0~1 比例（float32）


# ---------------------------------------------------------------------------
# 2. 贴纸变换（mini DefectTransformer）
# ---------------------------------------------------------------------------

def transform_sticker(sticker, mask, angle=0.0, scale=1.0, flip=False,
                      brightness=1.0):
    """翻转/旋转/缩放/亮度——数据增广四件套。

    贴纸"变装流水线"（对应仓库随机范围：旋转 0-360 / 缩放 0.7-1.3 /
    翻转 0.5 / 亮度 0.75-1.15）：同一素材演化千姿百态，模型才不会背下
    "长这样才算缺陷"。铁律：贴纸与蒙版成对同步变，否则贴的样子和 GT
    对不上。亮度沿用 stage0 的安全写法：先 float -> clip -> uint8
    （250*1.15 要饱和到 255 而不是回绕成 31）。
    """
    h, w = sticker.shape                       # 量尺寸：旋转中心和画布都要用
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)  # 施工图：绕中心转 + 缩
    st = cv2.warpAffine(sticker, M, (w, h), flags=cv2.INTER_LINEAR,  # 照片类：线性插值
                        borderMode=cv2.BORDER_REPLICATE)  # 露边复制边缘像素，不造假黑框
    mk = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,    # 二值蒙版：最近邻
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)  # 出画布=圈外；与 st 同一 M
    if flip:
        st, mk = st[:, ::-1], mk[:, ::-1]      # 左右镜像：列倒序，两份成对翻
    # 亮度安全链：先 float 再乘（uint8 装 287 会回绕成 31）-> round -> 夹 255
    # （顶天花板=饱和非回绕）-> 回 uint8。蒙版不调亮度：形状不管明暗
    st = np.clip(np.round(st.astype(np.float32) * brightness), 0, 255).astype(np.uint8)
    return st, mk                              # 变装后的贴纸 + 同步蒙版，成对交货


# ---------------------------------------------------------------------------
# 3. Alpha 融合 + 完整 Cut-Paste（mini DefectBlender + 防御性 GT 收敛）
# ---------------------------------------------------------------------------

def alpha_blend(background, sticker, alpha, x, y):
    """out = bg*(1-a) + sticker*a：alpha=1 处完全是贴纸，=0 处完全是背景。

    Cut-Paste 里真正"贴"的那一下，Photoshop 图层不透明度的逐像素版：
    a = 贴纸掺入占比，1 纯贴纸 / 0 纯背景 / 中间渐变融合（羽化消"贴纸
    感"）。x,y 为粘贴左上角（x 列 y 行）。全程在 float 里算、round 后再
    回 uint8（防丢精度/回绕）；copy 后才落笔，不弄脏调用方的原图。
    """
    h, w = sticker.shape[:2]          # 贴纸高宽（[:2] 跳过通道维）
    # 门卫：粘贴矩形须完整在背景内——numpy 越界切片只会悄悄裁短（粘出错位
    # 或爆难懂报错），不如人话当场拒绝
    if y < 0 or x < 0 or y + h > background.shape[0] or x + w > background.shape[1]:
        raise ValueError("贴纸越界：请保证 x,y + 贴纸尺寸在背景图内")
    roi = background[y: y + h, x: x + w].astype(np.float32)   # 抠出背景补丁；转 float 防回绕
    a3 = alpha[..., None]   # (h,w)->(h,w,1)：广播自动拉成 3 通道，一份权重管 RGB
    # 核心公式：a=1 纯贴纸 / a=0 纯背景 / 中间按占比渐变融合（羽化带）
    blended = np.round(roi * (1 - a3) + sticker.astype(np.float32) * a3)
    out = background.copy()           # 复制底片再动手（切片是视图，别毁原图）
    out[y: y + h, x: x + w] = blended.astype(np.uint8)   # 掺好贴回同位置（只改这块）
    return out                        # 合成图交货


def cut_paste(background, sticker, mask, x, y, angle=0.0, scale=1.0, flip=False,
             brightness=1.0, feather_k=5, allowed=None):
    """一次完整合成：变换 -> 羽化 -> 融合 -> 生成 GT。

    总装车间：串起各工位（变装/转彩/羽化/粘贴）再誊答案。调用一次 =
    出厂一个免标注训练样本：(image=合成考题, gt=逐像素答案)。
    allowed（工艺允许区，调用方传入；默认 None=不做收敛）：
    - 语义：非零=允许缺陷标注算数，0=不许——"不许区"没有单独数据，
      就是 allowed 取 0 的那部分；仓库真实来源=工艺 region 蒙版经配准
      warp 到现场坐标（见 stage1.registration 的 warp_mask）；
    - 格式：与背景同尺寸的单通道数组，非零即允许（0/1、布尔、灰度皆可）；
    - 给了就做仓库的"防御性 GT 收敛"：GT &= allowed，越界标注不算数
      （只删答案不动合成图，兜底保险）。
    """
    st, mk = transform_sticker(sticker, mask, angle, scale, flip, brightness)
    # ↑ 第一站变装：交货成对 (st=变装贴纸, mk=同步蒙版)，图和答案锁死同步
    if st.ndim == 2:                                    # 灰度贴纸统一转 3 通道再融合
        st = cv2.cvtColor(st, cv2.COLOR_GRAY2BGR)       # （垫成 (h,w,3) 才能和背景逐像素相乘）
    # 第二站羽化，羽化"形状"而非"内容"：mk 与 st 同施工图、逐像素对齐；若拿
    # 贴纸自己当配方——暗纹理(0~89)除以255封顶0.35，缺陷被洗淡、纹理糊掉
    alpha = feather_mask(mk, feather_k)    # 交货 float32 的 0~1 掺色配方（核心1/圈外0/边缘渐变）
    image = alpha_blend(background, st, alpha, x, y)   # 第三站粘贴：交货合成图（背景副本，只动贴纸区）

    gt = np.zeros(background.shape[:2], np.uint8)   # 开答案纸：背景同尺寸、全零单通道
    gt[y: y + st.shape[0], x: x + st.shape[1]] = mk   # 誊答案：mk 原样贴同一位置，贴哪标哪
    if allowed is not None:
        # 防御性 GT 收敛 = 逐格审批：gt 标了缺陷(255) 且 allowed 同位非零才留，
        # allowed 的 0 一票否决（255&255=255，255&0=0）；(allowed>0)->*255 先把
        # allowed 归一成严格 0/255（防 128 这类中间值搅浑二值答案）
        gt = cv2.bitwise_and(gt, (allowed > 0).astype(np.uint8) * 255)  # GT &= allowed
    return image, gt   # 出厂训练样本对：image=喂模型的输入，gt=0/255 监督标签（已收敛）


# ---------------------------------------------------------------------------
# 演示入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 64)
    print("1) 贴纸与羽化 alpha")
    print("-" * 64)
    sticker, mask = make_sticker(64, seed=0)
    alpha = feather_mask(mask, ksize=5)
    cy = cx = 32
    print(f"贴纸形状 mask 白像素 {int((mask > 0).sum())}，"
          f"中心 alpha={alpha[cy, cx]:.2f}（核心=1），角落 alpha={alpha[2, 2]:.2f}（外部=0），"
          f"值域 [{alpha.min():.2f}, {alpha.max():.2f}]")

    print("=" * 64)
    print("2) Alpha 融合：核心像贴纸、边缘渐变、远处不碰")
    print("-" * 64)
    bg = np.full((120, 160, 3), 180, np.uint8)               # 均匀浅色背景
    st3 = cv2.cvtColor(sticker, cv2.COLOR_GRAY2BGR)
    out = alpha_blend(bg, st3, alpha, 40, 30)
    print(f"核心处 [{out[62, 72].tolist()}] ~ 贴纸原色 [{st3[32, 32].tolist()}]")
    print(f"远离处 [{out[5, 5].tolist()}] == 背景原值 [{bg[5, 5].tolist()}]（未被动过）")

    print("=" * 64)
    print("3) 完整 Cut-Paste + 防御性 GT 收敛")
    print("-" * 64)
    bg = np.full((150, 200, 3), 180, np.uint8)               # 背景放大以容纳贴纸
    allowed = np.zeros((150, 200), np.uint8)
    allowed[20:100, 100:200] = 255                            # 只允许贴右半区域
    image, gt = cut_paste(bg, sticker, mask, x=120, y=30,            # 全部贴进 allowed
                          angle=30, brightness=0.9, allowed=allowed)
    image2, gt2 = cut_paste(bg, sticker, mask, x=70, y=30,           # 故意半越出 allowed
                            angle=30, allowed=allowed)
    _, gt2_raw = cut_paste(bg, sticker, mask, x=70, y=30,            # 不收敛版作对照
                           angle=30, allowed=None)
    n_raw = int((gt2_raw > 0).sum())
    n_kept = int((gt2 > 0).sum())
    print(f"整张贴入: GT 白像素 {int((gt > 0).sum())}（= 变换后贴纸形状，全在 allowed 内）")
    print(f"半越界贴入: 贴纸共 {n_raw} px，GT &= allowed 后只剩 {n_kept} px"
          f"（收敛掉 {n_raw - n_kept} px——越界部分不算标注，与仓库防御性 GT 收敛同口径）")
