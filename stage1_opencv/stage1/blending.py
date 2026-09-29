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

    返回 (sticker uint8 图, mask uint8 0/255 蒙版)——贴纸 + 形状，
    合成 GT 直接由 mask 来，这就是"贴上什么形状，答案就是什么形状"。
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    blob = np.zeros((size, size), np.float32)
    for _ in range(4):
        bx = size / 2 + rng.normal(0, size / 8)
        by = size / 2 + rng.normal(0, size / 8)
        s2 = 2 * (size / 6 * rng.uniform(0.6, 1.4)) ** 2
        blob += rng.uniform(0.5, 1.0) * np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / s2)
    mask = (blob > blob.max() * 0.35).astype(np.uint8) * 255
    texture = rng.integers(20, 90, (size, size), dtype=np.uint8)   # 暗色缺陷纹理
    sticker = cv2.bitwise_and(texture, texture, mask=mask)
    return sticker, mask


def feather_mask(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """边缘羽化：对二值蒙版做高斯模糊，得到 [0,1] 的 alpha 渐变。

    中间还是 1（核心区原色），边缘变成 0~1 过渡（融合带），
    kernel 越大羽化越宽（仓库 blend_kernel_size=5）。
    """
    if ksize % 2 == 0:
        raise ValueError("ksize 必须是奇数")
    return cv2.GaussianBlur(mask, (ksize, ksize), 0).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# 2. 贴纸变换（mini DefectTransformer）
# ---------------------------------------------------------------------------

def transform_sticker(sticker, mask, angle=0.0, scale=1.0, flip=False,
                      brightness=1.0):
    """翻转/旋转/缩放/亮度——数据增广四件套。

    亮度沿用 stage0 的安全写法：先 float -> clip -> uint8（250*1.15 要饱和到
    255 而不是回绕成 31）。
    """
    h, w = sticker.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    st = cv2.warpAffine(sticker, M, (w, h), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    mk = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if flip:
        st, mk = st[:, ::-1], mk[:, ::-1]
    st = np.clip(np.round(st.astype(np.float32) * brightness), 0, 255).astype(np.uint8)
    return st, mk


# ---------------------------------------------------------------------------
# 3. Alpha 融合 + 完整 Cut-Paste（mini DefectBlender + 防御性 GT 收敛）
# ---------------------------------------------------------------------------

def alpha_blend(background, sticker, alpha, x, y):
    """out = bg*(1-a) + sticker*a：alpha=1 处完全是贴纸，=0 处完全是背景。"""
    h, w = sticker.shape[:2]
    if y < 0 or x < 0 or y + h > background.shape[0] or x + w > background.shape[1]:
        raise ValueError("贴纸越界：请保证 x,y + 贴纸尺寸在背景图内")
    roi = background[y: y + h, x: x + w].astype(np.float32)
    a3 = alpha[..., None]                                   # (h, w, 1) 广播到 3 通道
    blended = np.round(roi * (1 - a3) + sticker.astype(np.float32) * a3)
    out = background.copy()
    out[y: y + h, x: x + w] = blended.astype(np.uint8)
    return out


def cut_paste(background, sticker, mask, x, y, angle=0.0, scale=1.0, flip=False,
             brightness=1.0, feather_k=5, allowed=None):
    """一次完整合成：变换 -> 羽化 -> 融合 -> 生成 GT。

    allowed 不为 None 时执行仓库的"防御性 GT 收敛"：GT &= allowed，
    贴纸落在允许区外的部分不算标注（杜绝越界标注）。
    返回 (合成图, GT 蒙版 uint8 0/255)。
    """
    st, mk = transform_sticker(sticker, mask, angle, scale, flip, brightness)
    if st.ndim == 2:                                    # 灰度贴纸统一转 3 通道再融合
        st = cv2.cvtColor(st, cv2.COLOR_GRAY2BGR)
    alpha = feather_mask(mk, feather_k)
    image = alpha_blend(background, st, alpha, x, y)

    gt = np.zeros(background.shape[:2], np.uint8)
    gt[y: y + st.shape[0], x: x + st.shape[1]] = mk
    if allowed is not None:
        gt = cv2.bitwise_and(gt, (allowed > 0).astype(np.uint8) * 255)  # GT &= allowed
    return image, gt


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
