"""课时 4：原图 LRU 预算缓存 + 一块多区互斥贴图 + 切块抖动 —— dataset.py 内存与覆盖工程。

  1. LRU 预算缓存（_orig_gray）：原图整张解码进内存按"字节预算"LRU 淘汰——预算不是
     拍脑袋：cache_mb = (可用内存 − 8G) × 0.6 ÷ 持有者数（多 worker DataLoader 每个
     进程都是持有者，不除就是 OOM）；命中 move_to_end、超预算从最旧端弹出；
  2. 一块多区每区各贴（occupied 互斥）：一个 512 块里含多个有效检测区时，每区
     各贴一个属于该区的缺陷，且新缺陷掩码 ∧ ~occupied——两区缺陷绝不重叠，
     标签不会互相踩（一块最多同时产 2~3 个缺陷，模型学会"同图多缺陷"）；
  3. 切块抖动（jitter）：每 epoch 网格坐标 ±stride/2 随机抖动（最多 6 次尝试），
     避开含标注块、保证区并集 ≥ 16px——同一个网格位置每轮看到的窗口都不同，
     等价于免费的数据增强。

对应主仓库：src/train/dataset.py::TileDataset._orig_gray / __getitem__
（occupied 与 jitter 段）/ main.py::_worker（cache_mb 解算段）。

运行：python -m stage9.tile_cache
"""
import random
from collections import OrderedDict

import numpy as np


class LRUBudgetCache:
    """字节预算 LRU：put 超预算时从最旧端淘汰，直到回到预算内（至少留 1 张）。"""

    def __init__(self, budget_bytes: int):
        self.budget = int(budget_bytes)
        self.size = 0
        self._store: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        hit = self._store.get(key)
        if hit is not None:
            self.hits += 1
            self._store.move_to_end(key)
            return hit
        self.misses += 1
        return None

    def put(self, key, value) -> None:
        nbytes = getattr(value, "nbytes", len(value))
        if key in self._store:
            self.size -= getattr(self._store[key], "nbytes", len(self._store[key]))
        self._store[key] = value
        self.size += nbytes
        while self.size > self.budget and len(self._store) > 1:
            _, old = self._store.popitem(last=False)
            self.size -= getattr(old, "nbytes", len(old))

    def __len__(self):
        return len(self._store)

    def keys(self):
        return list(self._store.keys())


def solve_cache_budget(avail_mb: float, holders: int = 1,
                       reserve_gb: float = 8.0, frac: float = 0.6) -> float:
    """cache_mb = (可用内存 − 8G) × 0.6 ÷ 持有者数；负数归零（宁可不缓存也不换页）。"""
    usable = (avail_mb - reserve_gb * 1024) * frac
    return max(0.0, usable / max(1, holders))


def paste_multi_zone(zone_defs: dict, sampler, rng) -> tuple:
    """一块多区每区各贴一个缺陷（occupied 互斥）。

    zone_defs: {区名: {mask(块内 bool), pick: 无参函数返回本次缺陷掩码(绝对坐标)}}
    返回 (occupied 并集, 本次实际贴入的区列表, 逐区实际贴入掩码 dict)。
    """
    ts = next(iter(zone_defs.values()))["mask"].shape[0]
    occupied = np.zeros((ts, ts), dtype=bool)
    pasted, placed = [], {}
    for zname, z in zone_defs.items():
        zone_mask = z["mask"]
        if not (zone_mask & ~occupied).any():
            continue
        def_mask = sampler(zname, rng)
        if def_mask is None:
            continue
        def_mask = def_mask & zone_mask & ~occupied      # 互斥：不踩别的区
        if def_mask.sum() < 8:
            continue
        occupied |= def_mask
        pasted.append(zname)
        placed[zname] = def_mask
    return occupied, pasted, placed


def jitter_coord(x0, y0, stride, W, H, ts, rng: random.Random,
                 ann_mask=None, union_mask=None, tries: int = 6):
    """切块抖动：±stride/2 随机偏移，最多 6 次尝试避开标注块并保持区并集 ≥ 16px。"""
    jitter = stride // 2
    x, y = x0, y0
    for _ in range(tries):
        cand_x = min(max(0, x0 + rng.randint(-jitter, jitter)), W - ts)
        cand_y = min(max(0, y0 + rng.randint(-jitter, jitter)), H - ts)
        if ann_mask is not None and ann_mask[cand_y:cand_y + ts, cand_x:cand_x + ts].any():
            continue
        if union_mask is not None and \
                int(union_mask[cand_y:cand_y + ts, cand_x:cand_x + ts].sum()) < 16:
            continue
        x, y = cand_x, cand_y
        break
    return x, y


if __name__ == "__main__":
    try:
        import psutil
        avail = psutil.virtual_memory().available / 2**20
    except ImportError:                                   # psutil 非必需，演示用假值
        avail = 32768.0
    # 1. LRU 预算缓存：3 张 1MB"原图"塞进 2.5MB 预算
    imgs = {f"img_{i}": np.zeros((1024, 1024), dtype=np.uint8) + i for i in range(3)}
    cache = LRUBudgetCache(budget_bytes=int(2.5 * 2**20))
    for k, v in imgs.items():
        cache.put(k, v)
    print(f"预算 2.5MB 装 3 张 1MB 原图 → 常驻 {len(cache)} 张: {cache.keys()} (最旧被淘汰)")
    cache.get("img_0")
    cache.get("img_2")
    cache.put("img_1", imgs["img_1"])                    # img_2 最近使用，img_0 淘汰
    print(f"访问 img_0/img_2 后再放 img_1 → {cache.keys()} (img_0 最久未用被淘汰)")

    for holders in (1, 4):
        mb = solve_cache_budget(avail, holders=holders)
        print(f"可用 {avail / 1024:.0f}GB → cache_mb = {mb:.0f}MB (÷{holders} 持有者)")

    # 2. 一块多区互斥贴图：左加工面 + 右装配面各贴一个
    ts = 128
    zones = {
        "加工面": {"mask": np.zeros((ts, ts), bool)},
        "装配面": {"mask": np.zeros((ts, ts), bool)},
    }
    zones["加工面"]["mask"][10:60, 10:60] = True
    zones["装配面"]["mask"][10:60, 70:120] = True
    rng = random.Random(0)

    def sampler(zname, rng):
        m = np.zeros((ts, ts), bool)
        if zname == "加工面":                             # 故意画出界一点测互斥
            m[30:50, 50:75] = True
        else:
            m[30:50, 75:95] = True
        return m
    occupied, pasted, placed = paste_multi_zone(zones, sampler, rng)
    raw_a = sampler("加工面", rng)
    would_leak = int((raw_a & zones["装配面"]["mask"]).sum())
    print(f"两区各贴一个: 实际贴入 {pasted} | 占用像素 {int(occupied.sum())} | "
          f"加工面缺陷若无互斥会越界进右区 {would_leak}px，互斥后实际越界 0px "
          f"(右区自有缺陷 {int(placed['装配面'].sum())}px 完整保留)")

    # 3. 切块抖动
    H = W = 1000
    ann = np.zeros((H, W), dtype=bool)
    ann[300:320, 400:420] = True                          # 一处标注
    union = np.zeros((H, W), dtype=bool)
    union[:, :] = True
    rng = random.Random(1)
    moved = 0
    for _ in range(50):
        x, y = jitter_coord(384, 384, 96, W, H, 128, rng, ann, union)
        moved += (x, y) != (384, 384)
        assert not ann[y:y + 128, x:x + 128].any()
    print(f"50 次抖动: {moved} 次偏离原网格点 (±48px), 全部避开标注块")
