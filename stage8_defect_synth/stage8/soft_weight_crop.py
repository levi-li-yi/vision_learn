"""课时 3：三区软权重掩码 + 潜变量预编码缓存 + 训练裁块策略 —— LoRA 数据准备三件套。

LoRA 训练图块的"答案"不是 0/1 硬掩码，而是一张**软权重图**（决定每个潜 token 的
损失权重）与一套预编码缓存策略：

  1. 三区软权重（_build_three_zone_masks）：
     · 缺陷核心区 = 1.0；
     · 外扩 8px 过渡环带 = 1.0 线性衰减到 0.4（距离变换度量；边缘半像素也算缺陷）；
     · 其他标注区 = 0.0 且交 inpaint 修补蒙版（重绘底图前先抹掉无关标注）；
     · 其余背景 = 0.0（完全不学）。
  2. 权重下采样到 token 分辨率：0.5·max_pool + 0.5·avg_pool 各半——max 保峰值
     （小缺陷核不被均值稀释没），avg 保平滑（过渡带不突变）。
  3. 潜变量预编码缓存（_preencode_dataset）：VAE encode → (z−shift)×scaling →pack
     一次性算完落盘 CPU，训练时 text_encoder/VAE 全部卸载、只跑 transformer——
     本课用"惰性编码 + 落盘 + 读缓存"迷你复刻（编码次数 = 数据条数，与轮数无关）。
  4. 裁块策略（_sample_crop_window + run_step1）：每实例裁 k=ceil(60/N)≤4 张带抖动
     窗口的 512 块；实例 >240 张（MAX_LORA_CROPS）按面积升序等距抽 240。

对应主仓库：src/train/lora_synth.py::_build_three_zone_masks / _preencode_dataset /
_sample_crop_window / run_step1_for_zone_defect。

运行：python -m stage8.soft_weight_crop
"""
import math

import cv2
import numpy as np
import torch


def three_zone_weights(shape, target_polys, ignore_polys=None,
                       dilate_px: int = 8, edge_weight_min: float = 0.4):
    """三区软权重：返回 (weight_map, heal_mask)。逐行复刻 _build_three_zone_masks。"""
    h, w = shape
    raw_mask = np.zeros((h, w), dtype=np.uint8)
    for pts in target_polys:
        cv2.fillPoly(raw_mask, [np.asarray(pts, dtype=np.int32)], 255)

    if dilate_px > 0:
        ker = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        cond_mask = cv2.dilate(raw_mask, ker, iterations=1)
    else:
        cond_mask = raw_mask.copy()

    inv_raw = np.where(raw_mask > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv_raw, cv2.DIST_L2, 5)

    weight_map = np.zeros((h, w), dtype=np.float32)
    weight_map[raw_mask > 0] = 1.0

    ring = (cond_mask > 0) & (raw_mask == 0)
    if np.any(ring) and dilate_px > 0:
        ratio = np.clip(dist[ring] / float(dilate_px), 0.0, 1.0)
        weight_map[ring] = 1.0 - (1.0 - float(edge_weight_min)) * ratio

    heal_mask = np.zeros((h, w), dtype=np.uint8)
    if ignore_polys:
        ig_raw = np.zeros((h, w), dtype=np.uint8)
        for pts in ignore_polys:
            cv2.fillPoly(ig_raw, [np.asarray(pts, dtype=np.int32)], 255)
        ker_heal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ig_heal = cv2.dilate(ig_raw, ker_heal, iterations=1)
        heal_mask[(ig_heal > 0) & (raw_mask == 0)] = 255
        ig_dil = cv2.dilate(ig_raw, ker, iterations=1) if dilate_px > 0 else ig_raw
        weight_map[(ig_dil > 0) & (raw_mask == 0)] = 0.0
    return weight_map, heal_mask


def downsample_tokens(weight: np.ndarray, factor: int = 16) -> np.ndarray:
    """权重 → 潜 token 分辨率：0.5·max_pool + 0.5·avg_pool（真工程同款各半）。"""
    w_4d = torch.from_numpy(weight.astype(np.float32))[None, None]
    pooled = (0.5 * torch.nn.functional.max_pool2d(w_4d, kernel_size=factor, stride=factor)
              + 0.5 * torch.nn.functional.avg_pool2d(w_4d, kernel_size=factor, stride=factor))
    return pooled[0, 0].numpy()


def preencode_cache(items: list, encode_fn, cache_path: str):
    """潜变量预编码缓存：首访逐条编码并落盘，之后直接读缓存（编码次数 = 条数）。"""
    import os
    if os.path.isfile(cache_path):
        return torch.load(cache_path)
    encoded = [encode_fn(x) for x in items]
    tmp = cache_path + ".tmp"
    torch.save(encoded, tmp)
    os.replace(tmp, cache_path)
    return encoded


def sample_crop_window(img_h, img_w, poly: np.ndarray, tile_size: int,
                       dilate_px: int, rng: np.random.Generator, jitter: bool = True):
    """围绕缺陷多边形裁 512 窗口：margin=dilate+16 内随机抖动；放不下则居中钳位。"""
    px0, py0 = int(poly[:, 0].min()), int(poly[:, 1].min())
    px1, py1 = int(poly[:, 0].max()), int(poly[:, 1].max())
    cx = (px0 + px1) // 2
    cy = (py0 + py1) // 2
    margin = dilate_px + 16
    pw = px1 - px0 + 1
    ph = py1 - py0 + 1

    if jitter and pw + 2 * margin < tile_size:
        min_x0 = max(0, px1 + margin - tile_size)
        max_x0 = min(img_w - tile_size, px0 - margin)
        x0 = int(rng.integers(min_x0, max_x0 + 1)) if min_x0 <= max_x0 else \
            int(np.clip(cx - tile_size // 2, 0, max(0, img_w - tile_size)))
    else:
        x0 = int(np.clip(cx - tile_size // 2, 0, max(0, img_w - tile_size)))

    if jitter and ph + 2 * margin < tile_size:
        min_y0 = max(0, py1 + margin - tile_size)
        max_y0 = min(img_h - tile_size, py0 - margin)
        y0 = int(rng.integers(min_y0, max_y0 + 1)) if min_y0 <= max_y0 else \
            int(np.clip(cy - tile_size // 2, 0, max(0, img_h - tile_size)))
    else:
        y0 = int(np.clip(cy - tile_size // 2, 0, max(0, img_h - tile_size)))
    return x0, y0


MAX_LORA_CROPS = 240          # 对应 lora_synth.MAX_LORA_CROPS


def crops_per_defect(n_real: int, target_total: int = 60) -> int:
    """k = ceil(60/N) 且 ≤ 4：真工程保证每个 {区/缺陷} 维度至少 ~60 张训练块。"""
    return max(1, min(4, int(math.ceil(target_total / max(1, n_real)))))


def subsample_by_area(records: list, limit: int = MAX_LORA_CROPS):
    """实例 > limit 张时按面积升序排序后等距抽 limit 张（真工程 np.linspace 同款）。"""
    if len(records) <= limit:
        return list(records)
    order = np.argsort([r["area"] for r in records])
    pick = np.linspace(0, len(records) - 1, limit, dtype=int)
    return [records[int(order[i])] for i in pick]


if __name__ == "__main__":
    import tempfile

    # 1. 三区软权重：一个 30×14 的"划痕"多边形 + 旁边一块"无关标注"
    h = w = 128
    scratch = np.array([[40, 64], [70, 64], [70, 71], [40, 71]], np.int32)
    other = np.array([[90, 20], [110, 20], [110, 34], [90, 34]], np.int32)
    weight, heal = three_zone_weights((h, w), [scratch], [other], dilate_px=8)

    core = weight[64:72, 40:71]
    ring_far = weight[64:72, 30:39]                     # 距核 ~2~9px 的环带
    print(f"核心区权重: min={core.min():.2f} (应为 1.0)")
    print(f"环带权重范围: [{ring_far[ring_far > 0].min():.2f}, {ring_far.max():.2f}] "
          f"(1.0 线性衰减到 0.4)")
    print(f"无关标注处权重: {weight[20:35, 90:111].max():.2f} (置 0 不学) | "
          f"修补蒙版像素数: {int((heal > 0).sum())} (供 inpaint 抹掉)")

    # 2. token 下采样：128×128 / 16 → 8×8
    tokens = downsample_tokens(weight, factor=16)
    print(f"token 权重图: {tokens.shape}, 峰值 {tokens.max():.2f} "
          f"(max_pool 保住小缺陷核不被均值抹没)")

    # 3. 预编码缓存：编码函数只被调用"条数"次，二次访问零编码
    calls = {"n": 0}

    def slow_encode(item):
        calls["n"] += 1
        return torch.tensor([item], dtype=torch.float32)

    with tempfile.TemporaryDirectory() as td:
        cache = f"{td}/encoded.pt"
        a = preencode_cache([1, 2, 3], slow_encode, cache)
        b = preencode_cache([1, 2, 3], slow_encode, cache)
        print(f"预编码缓存: 首次编码调用 {calls['n']} 次, 二次 {calls['n'] - 3} 次 "
              f"(真工程训练循环每步零 VAE 调用)")

    # 4. 裁块策略
    rng = np.random.default_rng(0)
    big = np.array([[500, 500], [540, 505], [542, 515], [502, 512]], np.int32)
    x0, y0 = sample_crop_window(1024, 1024, big, 512, 8, rng)
    print(f"裁块窗口 ({x0}, {y0}): 缺陷外接框+margin 完整落入 = "
          f"{x0 <= 500 - 24 and 542 + 24 <= x0 + 512 and y0 <= 500 - 24 and 515 + 24 <= y0 + 512}")
    print(f"crops_per_defect: 15 个实例 → k={crops_per_defect(15)} (≈60/15)，"
          f"3 个实例 → k={crops_per_defect(3)} (封顶 4)")
    recs = [{"area": a} for a in range(10, 10 + 500)]
    picked = subsample_by_area(recs)
    print(f"500 实例 → 等距抽 {len(picked)} 张 (MAX_LORA_CROPS={MAX_LORA_CROPS})，"
          f"最小面积 {picked[0]['area']} 被保留 (面积升序排序保证小缺陷入选)")
