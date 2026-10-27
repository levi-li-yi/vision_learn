"""课时 5：单轮训练预算数学模型与同源合成配额导出 —— budget.py::compute_product_budget。

全流程共享一套由"检测区有效面积/图块数 + 缺陷种类数"驱动的数学模型：

  · 基准过图量：v_base = clip(200 / M_total, 1, 5)——点位图越多单图切块当量越低，
    但最少 1 倍、最多 5 倍（点位少也不许把同一张图复读 50 遍）；
  · 单轮总预算：B_epoch = clip(round(v_base · T_total), ΣL, min(16000, ΣU))；
    下界总和超预算时先按比例压低下界（每区保 1）；
  · 分片数：num_shards = max_{p,z} ceil(T_{p,z} / E_{p,z})（≤ epochs）——保证在
    最大轮数内 100% 原始底图块全部参训（大区的块分片跨轮轮转，而非一轮全吃）；
  · 同源导出合成配额：阶段 3 每轮在 (p,z) 贴 E_{p,z} 次缺陷，均分 D 种、50% 抽合成、
    单张合成图增强复用 5 次 → q_raw = E / (10·D)；跨点位同名区汇总钳位 [60, 600]
    再按权重最大余数法分回各点位。

"同源"是本课的题眼：合成张数不是拍脑袋定的，而是从贴图需求逐级除出来的——
改 epochs/点位数/缺陷种类数，合成配额自动跟着变。

对应主仓库：src/train/budget.py::compute_product_budget（去掉对齐/切片 IO 的纯数学内核）。

运行：python -m stage7.quota_model
"""
import math

import numpy as np

from stage7.largest_remainder import largest_remainder_round
from stage7.waterfill import waterfill_allocate, zone_bounds, zone_weights

TARGET_PASSES_PER_POINT = 200.0   # 对应 TARGET_PASSES_PER_POINT
MAX_VARIANTS = 5.0                # 对应 MAX_VARIANTS
MAX_EPOCH_POS_TILES = 16000       # 对应 MAX_EPOCH_POS_TILES（多点位单轮时长上限）
MIN_SYNTH_PER_ZD = 60             # 对应 MIN_SYNTH_PER_ZD
MAX_SYNTH_PER_ZD = 600            # 对应 MAX_SYNTH_PER_ZD
SYNTH_AUG_REUSE_FACTOR = 5.0      # 对应 SYNTH_AUG_REUSE_FACTOR（增强复用 5 次）


def compute_epoch_budget(pz_tile_counts: dict, pz_defect_counts: dict,
                         m_total: int, epochs: int = 20, seed: int = 20260927) -> dict:
    """单轮图块预算 E_{p,z} + 分片数 + 合成小缺陷图配额 Q_synth 的统一解算。

    pz_tile_counts: {"点位/检测区": 干净块数 T}
    pz_defect_counts: {"点位/检测区": 绑定缺陷种类数 D}
    m_total: 训练底图总张数 M_total。
    """
    # 1. 基准过图量与权重/上下界
    v_base = max(1.0, min(MAX_VARIANTS, TARGET_PASSES_PER_POINT / max(1.0, float(m_total))))
    weights = zone_weights(pz_tile_counts, pz_defect_counts)
    lows, highs = zone_bounds(pz_tile_counts, pz_defect_counts, epochs=epochs,
                              max_variants=MAX_VARIANTS)

    # 2. 总预算 B_epoch（下界总和超预算 → 先按比例压缩下界，每区保底 1）
    eff_cap = min(float(MAX_EPOCH_POS_TILES), sum(highs.values()))
    sum_lows = sum(lows.values())
    if sum_lows > eff_cap and sum_lows > 0.0:
        scale_low = eff_cap / sum_lows
        for k in lows:
            lows[k] = max(1.0, lows[k] * scale_low)
        sum_lows = sum(lows.values())

    t_total = sum(pz_tile_counts.values())
    raw_epoch_budget = round(v_base * float(t_total))
    b_epoch = float(np.clip(raw_epoch_budget, sum_lows, max(sum_lows, eff_cap)))
    pz_epoch_pos = waterfill_allocate(weights, lows, highs, b_epoch)

    # 3. 分片数与单块重复倍数
    num_shards = 1
    pz_v_mult = {}
    for k, t_pz in pz_tile_counts.items():
        e_pz = max(1.0, pz_epoch_pos[k])
        pz_v_mult[k] = e_pz / float(max(1, t_pz))
        req_shards = int(math.ceil(float(t_pz) / e_pz - 1e-9))
        if req_shards > num_shards:
            num_shards = min(int(epochs), req_shards)

    # 4. 同源导出合成配额：q_raw = E / D / 2 / 5 = E / (10·D)，按 {区/缺陷} 汇总钳位分回
    synth_raw: dict = {}            # {(zone, defect): {point: raw_q}}
    for key in sorted(pz_tile_counts):
        point, zone = key.split("/", 1)
        e_pz = float(pz_epoch_pos[key])
        d_pz = max(1, pz_defect_counts[key])
        per_defect_paste = e_pz / d_pz                 # 均分给 D 种缺陷
        raw_q = per_defect_paste / 2.0 / SYNTH_AUG_REUSE_FACTOR   # 50% 合成 × 复用 5 次
        for d in range(d_pz):
            synth_raw.setdefault((zone, f"缺陷{d + 1}"), {})[point] = raw_q

    synth_quotas = {}
    for zd_key, p_map in synth_raw.items():
        p_names = sorted(p_map.keys())
        raw_weights = [float(p_map[p]) for p in p_names]
        raw_sum = int(math.ceil(sum(raw_weights)))
        clamped_total = int(np.clip(raw_sum, MIN_SYNTH_PER_ZD, MAX_SYNTH_PER_ZD))
        allocs = largest_remainder_round(raw_weights, clamped_total, seed=seed + 37)
        synth_quotas[zd_key] = {p: max(1, int(q)) for p, q in zip(p_names, allocs)}

    zd_totals = {f"{z}/{d}": sum(v.values()) for (z, d), v in synth_quotas.items()}
    return {
        "v_base": round(v_base, 3), "t_total": int(t_total), "b_epoch": int(round(b_epoch)),
        "num_shards": num_shards,
        "pz_epoch_pos": {k: int(round(v)) for k, v in pz_epoch_pos.items()},
        "pz_v_mult": {k: round(v, 4) for k, v in pz_v_mult.items()},
        "synth_quotas": synth_quotas, "zd_totals": zd_totals,
    }


if __name__ == "__main__":
    # 两点位：P1 三个区（大/中/极小），P2 一个同名大区 —— 跨点位同名区合成配额合并
    tiles = {"P1/加工面": 1200, "P1/装配面": 300, "P1/槽底": 40, "P2/加工面": 500}
    defects = {"P1/加工面": 2, "P1/装配面": 1, "P1/槽底": 1, "P2/加工面": 2}
    m_total = 12 * 8                   # 8 点位 × 12 张训练底图

    b = compute_epoch_budget(tiles, defects, m_total, epochs=20)
    print(f"M_total={m_total} → v_base={b['v_base']} | T_total={b['t_total']} | "
          f"B_epoch={b['b_epoch']} | num_shards={b['num_shards']}")
    print(f"{'区':<12}{'T':>6}{'E 单轮':>8}{'重复倍数':>9}")
    for k in sorted(tiles):
        print(f"{k:<12}{tiles[k]:>6}{b['pz_epoch_pos'][k]:>8}{b['pz_v_mult'][k]:>9.2f}")
    print("合成小缺陷图配额（同源导出，钳位 [60,600]）:")
    for (z, d), p_map in sorted(b["synth_quotas"].items()):
        detail = ", ".join(f"{p}={q}" for p, q in sorted(p_map.items()))
        print(f"  {z}/{d}: 共 {b['zd_totals'][f'{z}/{d}']} 张 ({detail})")

    print("\n规律: 槽底 T=40 被 5T 封顶只贴 200 块但保底 100; 大区跨分片轮转"
          f" (num_shards={b['num_shards']}); 同名'加工面'两点的合成需求合并后钳位。")
