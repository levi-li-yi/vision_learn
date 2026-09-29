"""Demo B：经典质检流水线 + region 门控——规则方法的力和极限。

流水线（与提纲一致）：灰度 -> 高斯模糊 -> 二值化(Otsu) -> 形态学闭运算
-> 连通域统计 -> 面积过滤 -> 检出组件表。

对应主仓库：
- 无监督 eval 的后处理链 GaussianBlur -> 二值化 -> MORPH_CLOSE(15,15) ->
  findContours（core/utils.py::test）；
- region 门控：检出结果 AND region 蒙版，抑制"待检表面之外"的假阳——
  装夹区的正常结构、背景污渍不是产品缺陷，但灰度上长得像。

运行：python -m stage1.defect_pipeline
"""

import cv2
import numpy as np

from stage1.connected_components import components_report


# ---------------------------------------------------------------------------
# 1. 合成"金属加工面"检测场景
# ---------------------------------------------------------------------------

def make_metal_background(w=200, h=150, seed=0) -> np.ndarray:
    """拉丝金属质感：随机噪声 + 水平方向模糊（拉丝纹理）。"""
    rng = np.random.default_rng(seed)
    img = rng.normal(128, 5, size=(h, w)).astype(np.float32)
    img = cv2.blur(img, (9, 1))                               # 横向拖出丝状纹理
    return np.clip(img, 0, 255).astype(np.uint8)


def plant_defects(image: np.ndarray, seed=1):
    """种缺陷：划痕(线) + 砂眼(暗点) + 噪点(比砂眼小) + 装夹区污渍。

    位置固定（教学要的是稳定可复现），r>=4 保证砂眼面积 >= min_area。
    返回 (图, GT 蒙版, 缺陷描述列表)。污渍故意种在 region 外——
    不门控就是假阳，门控后应被抑制。
    """
    rng = np.random.default_rng(seed)
    img = image.copy()
    gt = np.zeros(image.shape, np.uint8)
    facts = []

    for p1, p2 in [((60, 30), (120, 45)),                    # 三条划痕，彼此分离
                   ((100, 80), (150, 70)),
                   ((70, 110), (95, 130))]:
        cv2.line(img, p1, p2, 45, 2, cv2.LINE_AA)
        cv2.line(gt, p1, p2, 255, 2, cv2.LINE_AA)
        facts.append(("scratch", (*p1, *p2)))

    for cx, cy, r in [(150, 30, 5), (55, 65, 4), (170, 110, 6)]:   # 三个砂眼
        cv2.circle(img, (cx, cy), r, 55, -1)
        cv2.circle(gt, (cx, cy), r, 255, -1)
        facts.append(("pinhole", (cx, cy, r)))

    for _ in range(6):                                        # 噪点：r=1，面积过滤应剔除
        cx, cy = int(rng.integers(5, image.shape[1] - 5)), int(rng.integers(5, image.shape[0] - 5))
        cv2.circle(img, (cx, cy), 1, 90, -1)

    stain = (25, 25, 8)                                       # 装夹区污渍：region 外
    cv2.circle(img, (stain[0], stain[1]), stain[2], 50, -1)
    facts.append(("stain_outside_region", stain))
    return img, gt, facts


def make_region_mask(shape) -> np.ndarray:
    """工艺蒙版：只检测中间的加工面（对应 template_regions 的 region.png）。"""
    h, w = shape
    region = np.zeros((h, w), np.uint8)
    region[15: h - 15, 40: w - 10] = 255
    return region


# ---------------------------------------------------------------------------
# 2. 经典检测流水线 + 门控
# ---------------------------------------------------------------------------

def classical_detect(image: np.ndarray, min_area: int = 30, blur_k: int = 5):
    """灰度 -> 高斯模糊 -> 反相 Otsu -> 闭运算 -> 面积过滤 -> 组件表。

    反相（THRESH_BINARY_INV）：缺陷比背景暗，反相后缺陷=白、背景=黑，
    后续形态学/连通域一律在"白=目标"的约定下做。
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    _, binary = cv2.threshold(smooth, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return components_report(closed, min_area=min_area), closed


def region_gate(binary_mask: np.ndarray, region: np.ndarray) -> np.ndarray:
    """region 门控：mask AND region（>0 二值化语义与仓库一致）。

    仓库原文：服务端裁 region 再 >0 得二值包络，与预测做 AND，
    抑制'待检表面外'假阳。
    """
    return ((binary_mask > 0) & (region > 0)).astype(np.uint8) * 255


def hit(gt_mask: np.ndarray, comp, tol: int = 3) -> bool:
    """组件质心是否落在某条 GT 缺陷上（tol 容差圈）。"""
    cy, cx = int(round(comp["cy"])), int(round(comp["cx"]))
    y0, y1 = max(0, cy - tol), min(gt_mask.shape[0], cy + tol + 1)
    x0, x1 = max(0, cx - tol), min(gt_mask.shape[1], cx + tol + 1)
    return bool((gt_mask[y0: y1, x0: x1] > 0).any())


# ---------------------------------------------------------------------------
# 演示入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bg = make_metal_background(200, 150, seed=0)
    region = make_region_mask(bg.shape)
    img, gt, facts = plant_defects(bg, seed=1)
    n_real = sum(1 for t, _ in facts if t != "stain_outside_region")

    print("=" * 64)
    print("1) 经典流水线检出（无门控）")
    print("-" * 64)
    comps, closed = classical_detect(img, min_area=30)
    print(f"{'left':>4} {'top':>4} {'w':>3} {'h':>3} {'area':>5}  命中GT?")
    for c in comps:
        print(f"{c['left']:>4} {c['top']:>4} {c['w']:>3} {c['h']:>3} {c['area']:>5}"
              f"  {'是' if hit(gt, c) else '否 <- 假阳(装夹区污渍)'}")
    detected = [c for c in comps if hit(gt, c)]
    print(f"真实缺陷 {n_real} 个，检出 {len(detected)} 个；"
          f"假阳 {len(comps) - len(detected)} 个（污渍长得像缺陷，规则分不清'在哪'）")

    print("=" * 64)
    print("2) region 门控后：位置先验上场")
    print("-" * 64)
    gated = region_gate(np.where(closed > 0, 255, 0).astype(np.uint8), region)
    from stage1.connected_components import label_components
    num, labels, stats, cents = label_components(gated)
    kept = []
    for i in range(1, num):
        left, top, w_, h_, area = stats[i]
        if area >= 30:
            kept.append({"left": int(left), "top": int(top), "w": int(w_),
                         "h": int(h_), "area": int(area),
                         "cx": float(cents[i][0]), "cy": float(cents[i][1])})
    hit_gated = [c for c in kept if hit(gt, c)]
    print(f"门控后组件 {len(kept)} 个，命中 {len(hit_gated)} 个，"
          f"假阳 {len(kept) - len(hit_gated)} 个（region 外的污渍被抹掉）")
    print("-> 这就是仓库'region AND 门控'的业务含义：先圈定'哪里算数'，再看'像不像缺陷'。")

    print("=" * 64)
    print("3) 规则方法的极限（诚实提醒）")
    print("-" * 64)
    print("本例缺陷对比度高、光照可控，规则就够了；真实产线里缺陷灰度多变、")
    print("纹理背景复杂，固定阈值的误检/漏检不可兼得——这正是主仓库上深度学习")
    print("（监督分割学'缺陷长相'、无监督学'正常长相'）的动机。")
