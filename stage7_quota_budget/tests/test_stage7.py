"""stage7 全模块测试：分卷种子 / 注水分配 / 最大余数 / 网格补块 / 配额模型。"""
import numpy as np
import pytest

from stage7.grid_patch import (
    ANN_SAFE_PAD_PX, build_image_grid_for_zones, make_scene,
)
from stage7.largest_remainder import largest_remainder_round
from stage7.quota_model import (
    MAX_SYNTH_PER_ZD, MIN_SYNTH_PER_ZD, compute_epoch_budget,
)
from stage7.split_seed import (
    MAX_EARLYSTOP_PER_POINT, point_seed_offset, split_train_earlystop,
    swap_zero_clean_zone, zone_tile_counts,
)
from stage7.waterfill import waterfill_allocate, zone_bounds, zone_weights


class TestSplitSeed:
    def make_pools(self):
        return {f"P{i}": [f"img_{k:03d}" for k in range(40)] for i in range(1, 4)}

    def test_ratio_and_cap(self):
        train, val = split_train_earlystop(self.make_pools(), seed=1)
        for p in train:
            n_val = len(val[p])
            assert n_val == max(1, int(round(40 * 0.2)))       # 20% = 8 张
            assert n_val <= MAX_EARLYSTOP_PER_POINT
            assert len(train[p]) + len(val[p]) == 40
            assert not set(train[p]) & set(val[p])              # 互斥

    def test_cap_ten_for_huge_pool(self):
        pool = {"PX": [f"i{k}" for k in range(200)]}
        _, val = split_train_earlystop(pool, seed=3)
        assert len(val["PX"]) == MAX_EARLYSTOP_PER_POINT        # 上限 10 而不是 40

    def test_deterministic_and_point_independent(self):
        pools = self.make_pools()
        t1, v1 = split_train_earlystop(pools, seed=9)
        t2, v2 = split_train_earlystop(pools, seed=9)
        assert t1 == t2 and v1 == v2                            # 同种子确定
        pools_more = {**pools, "P9": [f"n{k}" for k in range(11)]}
        t3, v3 = split_train_earlystop(pools_more, seed=9)
        assert all(t3[p] == t1[p] and v3[p] == v1[p] for p in pools)  # 点位独立

    def test_seed_offset_stable(self):
        assert point_seed_offset("P1") == point_seed_offset("P1")
        assert point_seed_offset("P1") != point_seed_offset("P2")
        assert 0 <= point_seed_offset("任意点位名") < 1000003

    def test_empty_pool_raises(self):
        with pytest.raises(RuntimeError):
            split_train_earlystop({"PE": []}, seed=1)

    def test_single_image_all_train(self):
        _, val = split_train_earlystop({"PS": ["only"]}, seed=1)
        assert val["PS"] == []                                  # 单张不分卷


class TestSwapZeroClean:
    def test_swap_back_rescue_image(self):
        train = ["a", "b", "c"]
        val = ["r1", "r2"]                                      # r1 是唯一带槽底的图
        tiles = {"a": {"加工面": 10}, "b": {"加工面": 8}, "c": {"加工面": 9},
                 "r1": {"加工面": 10, "槽底": 6}, "r2": {"加工面": 7}}
        tr, va, swapped = swap_zero_clean_zone(train, val, tiles,
                                              required_zones=["加工面", "槽底"])
        assert swapped and "r1" in tr
        assert len(tr) == 3 and len(va) == 2                    # 一换一
        assert zone_tile_counts({s: tiles[s] for s in tr})["槽底"] == 6

    def test_no_swap_when_all_clean(self):
        tiles = {"a": {"加工面": 10}, "b": {"加工面": 10}, "v": {"加工面": 10}}
        tr, va, swapped = swap_zero_clean_zone(["a", "b"], ["v"], tiles,
                                              required_zones=["加工面"])
        assert not swapped and tr == ["a", "b"] and va == ["v"]

    def test_fail_when_no_rescue_anywhere(self):
        tiles = {"a": {"加工面": 10}, "v": {"加工面": 10}}
        _, _, swapped = swap_zero_clean_zone(["a"], ["v"], tiles,
                                             required_zones=["神秘区"])
        assert not swapped                                       # 真工程 SystemExit


class TestWaterfill:
    def test_bounds_respected_random(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            keys = [f"k{i}" for i in range(6)]
            weights = {k: float(rng.uniform(1, 100)) for k in keys}
            lows = {k: float(rng.uniform(0, 50)) for k in keys}
            highs = {k: float(lows[k] + rng.uniform(10, 200)) for k in keys}
            budget = float(rng.uniform(sum(lows.values()) * 0.8, sum(highs.values()) * 1.2))
            alloc = waterfill_allocate(weights, lows, highs, budget)
            for k in keys:
                assert lows[k] - 1e-6 <= alloc[k] <= highs[k] + 1e-6
            if budget >= sum(lows.values()):
                assert sum(alloc.values()) <= budget + 1e-6   # 预算够得到 ΣL 时永不超发
            # 预算 < ΣL 分支按设计全按下界躺平（真工程会先按比例压低下界避开此分支）

    def test_exact_total_for_benign_bounds(self):
        # 常规素材配置（L ≪ 份额 ≪ U）：预算精确分完
        rng = np.random.default_rng(2)
        for _ in range(30):
            keys = [f"k{i}" for i in range(6)]
            weights = {k: float(rng.uniform(1, 100)) for k in keys}
            lows = {k: float(rng.uniform(0, 5)) for k in keys}
            highs = {k: float(lows[k] + rng.uniform(500, 2000)) for k in keys}
            budget = float(rng.uniform(sum(lows.values()) + 100, sum(highs.values()) - 100))
            alloc = waterfill_allocate(weights, lows, highs, budget)
            assert sum(alloc.values()) == pytest.approx(budget, rel=1e-9)

    def test_low_budget_all_lows(self):
        w = {"a": 5.0, "b": 1.0}
        lows = {"a": 10.0, "b": 20.0}
        highs = {"a": 100.0, "b": 100.0}
        alloc = waterfill_allocate(w, lows, highs, 5)           # 预算 5 < ΣL=30
        assert alloc == {"a": 10.0, "b": 20.0}

    def test_high_budget_all_highs(self):
        w = {"a": 5.0, "b": 1.0}
        lows = {"a": 10.0, "b": 20.0}
        highs = {"a": 100.0, "b": 50.0}
        alloc = waterfill_allocate(w, lows, highs, 1000)
        assert alloc == {"a": 100.0, "b": 50.0}

    def test_proportional_when_interior(self):
        w = {"a": 3.0, "b": 1.0}
        lows = {"a": 0.0, "b": 0.0}
        highs = {"a": 1e9, "b": 1e9}
        alloc = waterfill_allocate(w, lows, highs, 400)
        assert alloc["a"] == pytest.approx(300.0) and alloc["b"] == pytest.approx(100.0)

    def test_zone_weights_and_bounds_formulas(self):
        tiles = {"p/z": 100}
        defects = {"p/z": 2}
        assert zone_weights(tiles, defects)["p/z"] == pytest.approx(np.sqrt(200))
        lows, highs = zone_bounds(tiles, defects, epochs=20)
        assert highs["p/z"] == 500.0                            # U = 5T
        assert lows["p/z"] == 200.0                             # L = max(100·D=200, T/20=5)

    def test_low_capped_by_high(self):
        lows, highs = zone_bounds({"p/z": 10}, {"p/z": 3}, epochs=20)
        assert lows["p/z"] == 50.0 == highs["p/z"]              # L=min(5T=50, 300)=50


class TestLargestRemainder:
    def test_sum_exact(self):
        rng = np.random.default_rng(1)
        for target in (7, 60, 600, 1234):
            w = rng.uniform(0.1, 100, 17).tolist()
            alloc = largest_remainder_round(w, target, seed=0)
            assert sum(alloc) == target and all(a >= 0 for a in alloc)

    def test_fairness_within_one(self):
        w = [10.0, 20.0, 30.0, 40.0]
        alloc = largest_remainder_round(w, 100, seed=0)
        for a, wi in zip(alloc, w):
            assert abs(a - wi / 100 * 100) < 1.0                # 每项偏离理想 < 1

    def test_zero_weight_gets_zero(self):
        assert largest_remainder_round([0.0, 5.0, 0.0], 10) == [0, 10, 0]

    def test_empty_and_zero_target(self):
        assert largest_remainder_round([], 5) == []
        assert largest_remainder_round([1.0, 2.0], 0) == [0, 0]

    def test_tie_break_changes_with_seed(self):
        seen = {tuple(largest_remainder_round([1.5, 1.5, 1.5], 5, seed=s))
                for s in range(6)}
        assert len(seen) >= 2                                   # 平票扰动确实换人


class TestGridPatch:
    def test_scene_blocks_and_rules(self):
        zones, ann_dil, _ = make_scene()
        coords = build_image_grid_for_zones(zones, (1200, 1600), 512, ann_mask=ann_dil)
        assert len(coords) > 0
        for (x, y) in coords:
            assert 0 <= x <= 1600 - 512 and 0 <= y <= 1200 - 512
            assert not ann_dil[y:y + 512, x:x + 512].any()       # 干净
            union = (zones["加工面"] | zones["槽底"])[y:y + 512, x:x + 512]
            assert union.sum() >= 16                             # 并集门槛

    def test_stride_and_edge_dedup(self):
        zones, _, _ = make_scene()
        coords = build_image_grid_for_zones(zones, (1200, 1600), 512)
        stride = int(512 * 0.75)
        for seq, bound in ((sorted({x for x, _ in coords}), 1600 - 512),
                           (sorted({y for _, y in coords}), 1200 - 512)):
            assert len(seq) == len(set(seq))                   # 尾部钳位去重
            assert all(0 <= v <= bound for v in seq)
            gaps = [b - a for a, b in zip(seq, seq[1:])]
            assert all(0 < g <= stride for g in gaps)          # 间距 ≤ 一个步长

    def test_small_zone_gets_completeness_patch(self):
        zones, ann_dil, _ = make_scene()
        coords = build_image_grid_for_zones(zones, (1200, 1600), 512, ann_mask=ann_dil)
        contained = any(x <= 760 and 820 <= x + 512 and y <= 300 and 360 <= y + 512
                        for (x, y) in coords)
        assert contained                                          # 骑缝小区有完整版块

    def test_annotation_kills_block(self):
        zones, ann_dil, _ = make_scene()
        coords = build_image_grid_for_zones(zones, (1200, 1600), 512, ann_mask=ann_dil)
        for (x, y) in coords:                                     # 标注(400,400)±(20+pad) 所在块被剔
            inside = ann_dil[y:y + 512, x:x + 512].any()
            assert not inside

    def test_empty_zones(self):
        assert build_image_grid_for_zones({}, (100, 100), 512) == []
        assert build_image_grid_for_zones({"z": np.zeros((100, 100), bool)},
                                          (100, 100), 512) == []


class TestQuotaModel:
    def demo_inputs(self):
        tiles = {"P1/加工面": 1200, "P1/装配面": 300, "P1/槽底": 40, "P2/加工面": 500}
        defects = {"P1/加工面": 2, "P1/装配面": 1, "P1/槽底": 1, "P2/加工面": 2}
        return tiles, defects

    def test_budget_within_bounds(self):
        tiles, defects = self.demo_inputs()
        b = compute_epoch_budget(tiles, defects, m_total=96)
        lows_d, highs_d = zone_bounds(tiles, defects, epochs=20)
        for k in tiles:
            e = b["pz_epoch_pos"][k]
            assert lows_d[k] - 1.0 <= e <= highs_d[k] + 1.0      # 注水结果在界内（取整容差 1）
            assert b["pz_v_mult"][k] <= 5.0 + 1e-6               # 单块单轮重复 ≤ 5
        total = sum(b["pz_epoch_pos"].values())
        assert total == pytest.approx(b["b_epoch"], abs=len(tiles))  # 总和≈预算

    def test_shards_cover_all_tiles(self):
        tiles, defects = self.demo_inputs()
        b = compute_epoch_budget(tiles, defects, m_total=96, epochs=20)
        for k, t in tiles.items():
            e = max(1.0, b["pz_epoch_pos"][k])
            assert b["num_shards"] >= int(np.ceil(t / e)) - 1    # ceil 容差
        assert b["num_shards"] <= 20                             # 不超过 epochs

    def test_synth_quota_clamped(self):
        tiles, defects = self.demo_inputs()
        b = compute_epoch_budget(tiles, defects, m_total=96)
        assert b["zd_totals"]
        for zd, tot in b["zd_totals"].items():
            assert MIN_SYNTH_PER_ZD <= tot <= MAX_SYNTH_PER_ZD
            # 同名区跨点位合并：加工面的两类缺陷在 P1/P2 都有分配
        jmh = [v for (z, d), m in b["synth_quotas"].items() if z == "加工面"
               for v in m.values()]
        assert len(jmh) == 4                                      # 2 缺陷 × 2 点位

    def test_v_base_formula(self):
        b = compute_epoch_budget({"P/加工面": 100}, {"P/加工面": 1}, m_total=40)
        assert b["v_base"] == pytest.approx(5.0)                  # 200/40=5 → 封顶 5
        b2 = compute_epoch_budget({"P/加工面": 100}, {"P/加工面": 1}, m_total=5000)
        assert b2["v_base"] == pytest.approx(1.0)                 # 200/5000 → 保底 1

    def test_deterministic(self):
        tiles, defects = self.demo_inputs()
        b1 = compute_epoch_budget(tiles, defects, m_total=96)
        b2 = compute_epoch_budget(tiles, defects, m_total=96)
        assert b1 == b2
