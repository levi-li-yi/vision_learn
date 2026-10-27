"""课时 4：0.75 步长网格切块 + 骑缝小检测区居中补块 —— budget.py::build_image_grid_for_zones。

在单张原图上决定"哪些 512×512 图块可以进训练池"，四条规则：

  1. 步长 0.75×tile：相邻块 25% 重叠，兼顾覆盖率与块数（边缘 min(i·stride, W−ts)
     钳位并去重，防止最后一块和倒数第二块位置相同）；
  2. 并集门槛：所有检测区并集在块内 ≥ 16px（纯背景块不要）；
  3. 单区门槛：至少一个区在块内 ≥ 8px（沾边块对任何区都没意义）；
  4. 脏块剔除：块内含任何人工标注（含"排除"，标注外扩安全膨胀 ANN_SAFE_PAD_PX=4）即跳过
     ——训练底图块必须 100% 干净。

再加**骑缝补块**：小检测区连通域的外接框若不被任何网格块完全包含，会被网格线切成
碎片——以连通域中心补一块 512×512，保证每个小检测区至少有一块"完整版"图块。

对应主仓库：src/train/budget.py::build_image_grid_for_zones
（dataset.py::TileDataset 与 exam.py::predict_heatmap_t 使用同款网格口径）。

运行：python -m stage7.grid_patch
"""
import math

import cv2
import numpy as np

ANN_SAFE_PAD_PX = 4    # 对应 budget.py ANN_SAFE_PAD_PX（与 dataset.py 严格一致）
MIN_IN_ZONE_PX = 8     # 对应 MIN_IN_ZONE_PX
MIN_UNION_ZONE_PX = 16  # 对应 MIN_UNION_ZONE_PX


def build_image_grid_for_zones(zone_masks: dict, img_size, tile_size: int = 512,
                               ann_mask: np.ndarray = None, stride_ratio: float = 0.75):
    """基于有效检测区的干净图块坐标列表 [(x, y), ...]。"""
    H, W = img_size
    ts = tile_size
    stride = int(ts * stride_ratio)

    zone_union = None
    for zm in zone_masks.values():
        zone_union = zm if zone_union is None else (zone_union | zm)
    if zone_union is None or not zone_union.any():
        return []

    xs = list(dict.fromkeys([min(i * stride, W - ts) for i in range(int(math.ceil(W / stride)))]))
    ys = list(dict.fromkeys([min(j * stride, H - ts) for j in range(int(math.ceil(H / stride)))]))

    coords, coord_set = [], set()
    for x in xs:
        for y in ys:
            if int(zone_union[y:y + ts, x:x + ts].sum()) < MIN_UNION_ZONE_PX:
                continue
            if ann_mask is not None and ann_mask[y:y + ts, x:x + ts].any():
                continue
            if not any(int(zm[y:y + ts, x:x + ts].sum()) >= MIN_IN_ZONE_PX
                       for zm in zone_masks.values()):
                continue
            if (x, y) not in coord_set:
                coords.append((x, y))
                coord_set.add((x, y))

    # 骑缝补块：小连通域外接框不被任何网格块完全包含 → 以中心补一块
    for zm in zone_masks.values():
        num_cc, _labels, stats, _ = cv2.connectedComponentsWithStats(
            zm.astype(np.uint8), connectivity=8)
        for cid in range(1, num_cc):
            bx = int(stats[cid, cv2.CC_STAT_LEFT])
            by = int(stats[cid, cv2.CC_STAT_TOP])
            bw = int(stats[cid, cv2.CC_STAT_WIDTH])
            bh = int(stats[cid, cv2.CC_STAT_HEIGHT])
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < MIN_IN_ZONE_PX or bw > ts or bh > ts:
                continue
            contained = any((x <= bx and bx + bw <= x + ts and y <= by and by + bh <= y + ts)
                            for (x, y) in coords)
            if not contained:
                cx = bx + bw // 2
                cy = by + bh // 2
                nx = int(np.clip(cx - ts // 2, 0, W - ts))
                ny = int(np.clip(cy - ts // 2, 0, H - ts))
                if int(zone_union[ny:ny + ts, nx:nx + ts].sum()) < MIN_UNION_ZONE_PX:
                    continue
                if ann_mask is not None and ann_mask[ny:ny + ts, nx:nx + ts].any():
                    continue
                if (nx, ny) not in coord_set:
                    coords.append((nx, ny))
                    coord_set.add((nx, ny))
    return coords


def make_scene(seed: int = 4, h: int = 1200, w: int = 1600, tile: int = 512):
    """合成一张双区大图 + 一块骑缝小检测区 + 两处标注。"""
    rng = np.random.default_rng(seed)
    zones = {}
    big = np.zeros((h, w), dtype=bool)
    big[100:1100, 100:1500] = True                        # 大检测区
    zones["加工面"] = big
    small = np.zeros((h, w), dtype=bool)                  # 骑缝小区：卡在网格缝上
    small[300:360, 760:820] = True
    zones["槽底"] = small
    ann = np.zeros((h, w), dtype=bool)
    yy, xx = np.mgrid[-20:20, -20:20]                     # 半径 18 的圆盘标注
    for cx, cy in [(400, 400), (1200, 700)]:
        ann[cy - 20:cy + 20, cx - 20:cx + 20] |= (yy ** 2 + xx ** 2 <= 18 ** 2)
    ker = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ANN_SAFE_PAD_PX + 1, 2 * ANN_SAFE_PAD_PX + 1))
    ann_dil = cv2.dilate(ann.astype(np.uint8), ker) > 0
    return zones, ann_dil, rng


if __name__ == "__main__":
    zones, ann_dil, _ = make_scene()
    coords = build_image_grid_for_zones(zones, (1200, 1600), tile_size=512,
                                        ann_mask=ann_dil)
    stride = int(512 * 0.75)
    print(f"网格步长 {stride}px（0.75×512），命中干净块 {len(coords)} 个：")
    for (x, y) in sorted(coords):
        tags = [z for z, zm in zones.items()
                if int(zm[y:y + 512, x:x + 512].sum()) >= MIN_IN_ZONE_PX]
        print(f"  ({x:>4}, {y:>4}) 覆盖区: {tags}")
    # 验证骑缝小区被某块完整包含
    contained = any(x <= 760 and 820 <= x + 512 and y <= 300 and 360 <= y + 512
                    for (x, y) in coords)
    print(f"骑缝小检测区(300,760)-(360,820) 被完整包含: {contained}")
