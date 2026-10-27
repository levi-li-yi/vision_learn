"""课时 1：8:2 训练/早停确定性分卷 + 零干净块置换 —— budget.py::split_train_earlystop。

训练池的每一张底图按 20%（上限 10 张）被划出去做"早停验证集"——它们只用来监控
val_loss 连续不降就早停，绝不参与贴图训练。两个容易忽视的设计：

  1. 逐点位独立种子：rng = default_rng(seed + md5(点位名)偏移)。分卷结果只由
     (全局种子, 本点位名, 本点位图列表) 决定——**新增/删除其他点位不会让本点位
     重新洗牌**（否则跨次运行的缓存指纹全部失效）；
  2. 零干净块置换：某个小检测区的唯一干净底图恰好被分进早停集时，自动把它换回
     训练池（从训练池随机顶一张出去）；全零则配额阻断大声退出——绝不让某个
     检测区在训练里"失踪"。

对应主仓库：src/train/budget.py::_point_seed_offset / split_train_earlystop /
_preslice_clean_tile_counts(置换段)。

运行：python -m stage7.split_seed
"""
import hashlib

import numpy as np

MAX_EARLYSTOP_PER_POINT = 10   # 对应 budget.py 同名常量
EARLYSTOP_RATIO = 0.2


def point_seed_offset(point_name: str) -> int:
    """md5(点位名) 前 4 字节小端整数 % 1000003 —— 与图序无关的确定性偏移。"""
    h = hashlib.md5(point_name.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") % 1000003


def split_train_earlystop(stems_by_point: dict, seed: int):
    """各点位独立 8:2 分卷（早停集上限 10 张；单点位只有 1 张时不分）。

    返回 (train_roster, val_stems)：均为 {点位: [图名]}。逐行复刻真工程语义，
    包括"池子空即抛错"。
    """
    train_roster, val_stems = {}, {}
    for pname in sorted(stems_by_point.keys()):
        pool = sorted(stems_by_point[pname])
        n_p = len(pool)
        if n_p == 0:
            raise RuntimeError(f"{pname}: 训练池可用图像为 0")
        rng = np.random.default_rng(seed + point_seed_offset(pname))
        order = rng.permutation(n_p)
        n_val = min(MAX_EARLYSTOP_PER_POINT,
                    max(1, int(round(n_p * EARLYSTOP_RATIO)))) if n_p >= 2 else 0
        va_ix = set(order[:n_val].tolist())
        val_stems[pname] = [pool[i] for i in range(n_p) if i in va_ix]
        train_roster[pname] = [pool[i] for i in range(n_p) if i not in va_ix]
    return train_roster, val_stems


def zone_tile_counts(stem_zone_tiles: dict) -> dict:
    """聚合每张图对各区干净块数的贡献：{图: {区: 块数}} → {区: 总块数}。"""
    counts: dict = {}
    for per_zone in stem_zone_tiles.values():
        for z, c in per_zone.items():
            counts[z] = counts.get(z, 0) + c
    return counts


def swap_zero_clean_zone(train_stems: list, val_stems: list, stem_zone_tiles: dict,
                         required_zones: list):
    """零干净块置换：required_zones 中某区在训练池干净块数为 0 而早停集里有 → 换回一张。

    stem_zone_tiles: {图名: {区: 干净块数}}，训练池与早停集共用同一张表。
    返回 (train_stems, val_stems, swapped)；早停集也没有干净块则 swapped=False
    ——真工程此时 SystemExit 配额阻断，要求补拍 OK 底图。
    """
    def counts(stems):
        return zone_tile_counts({s: stem_zone_tiles.get(s, {}) for s in stems})

    tr_counts = counts(train_stems)
    if all(tr_counts.get(z, 0) > 0 for z in required_zones):
        return list(train_stems), list(val_stems), False
    for v_idx, v_stem in enumerate(list(val_stems)):
        v_zones = stem_zone_tiles.get(v_stem, {})
        rescues = [z for z in required_zones
                   if v_zones.get(z, 0) > 0 and tr_counts.get(z, 0) == 0]
        if not rescues:
            continue
        if train_stems:
            swap_out = train_stems.pop(0)                 # 顶一张训练图进早停集
            val_stems[v_idx] = swap_out
        else:
            val_stems.pop(v_idx)
        train_stems.append(v_stem)                        # 救命图回训练池
        return list(train_stems), list(val_stems), True
    return list(train_stems), list(val_stems), False


if __name__ == "__main__":
    stems = {f"P{i}": [f"img_{i}_{k:03d}" for k in range(40)] for i in range(1, 4)}
    train, val = split_train_earlystop(stems, seed=20260927)
    for p in sorted(stems):
        print(f"{p}: 总 {len(stems[p])} → 训练 {len(train[p])} + 早停 {len(val[p])} "
              f"(比例 {len(val[p]) / len(stems[p]):.0%})")

    # 确定性 & 点位独立性：加一个新点位，老点位分卷纹丝不动
    train2, val2 = split_train_earlystop({**stems, "P9": [f"n_{k}" for k in range(11)]},
                                         seed=20260927)
    assert all(train2[p] == train[p] and val2[p] == val[p] for p in stems)
    print("新增点位 P9 后，P1~P3 分卷结果不变（逐点位独立种子）")

    # 零干净块置换：假设早停集第一张是唯一带"槽底"干净块的图（小检测区典型险情）
    stem_tiles = {s: {"加工面": 10} for s in train["P1"] + val["P1"]}
    rescue = val["P1"][0]
    stem_tiles[rescue] = {"加工面": 10, "槽底": 6}
    tr, va, swapped = swap_zero_clean_zone(list(train["P1"]), list(val["P1"]),
                                          stem_tiles, required_zones=["加工面", "槽底"])
    after = zone_tile_counts({s: stem_tiles[s] for s in tr})
    print(f"置换发生={swapped} | 救回图 {rescue} 回训练池={rescue in tr} | "
          f"置换后训练池槽底干净块={after.get('槽底', 0)} | 顶替进早停集={[s for s in va if s not in val['P1']]}")

    # 反例：早停集里也没有槽底干净块 → 置换失败（真工程 SystemExit 配额阻断）
    tr2, va2, swapped2 = swap_zero_clean_zone(list(train["P1"]), list(val["P1"]),
                                              stem_tiles, required_zones=["不存在的区"])
    print(f"早停集无该区干净块时置换发生={swapped2}（真工程此处 SystemExit：请补拍 OK 底图）")
