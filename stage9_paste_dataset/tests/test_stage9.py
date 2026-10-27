"""stage9 全模块测试：零光晕泊松 / 五五分放置 / 牌靴与采样器 / LRU 与互斥贴图。"""
import random

import cv2
import numpy as np
import pytest

from stage9.deck_sampling import (
    EpochShardSampler, ShardDataset, deck_draw, deck_perm, decode_index, sample_rng,
)
from stage9.paste_zero_halo import (
    label_on_canvas, make_scene, naive_clone, paste_poisson,
)
from stage9.place_rules import (
    VISIBLE_MIN_DIFF, context_window, make_defect_patch, make_zone, rotate_patch,
    sample_target, solve_scale, try_place, visibility_check,
)
from stage9.tile_cache import (
    LRUBudgetCache, jitter_coord, paste_multi_zone, solve_cache_budget,
)


class TestPasteZeroHalo:
    def paste_default(self, seed=0, target_diff=90.0, polarity=-1):
        gray, zone, patch, mask = make_scene()
        rng = random.Random(seed)
        got = paste_poisson(gray.copy(), patch, mask, rng, off=(60, 60),
                            zone_win=zone.copy(), polarity=polarity,
                            target_diff=target_diff)
        assert got is not None
        out, label = got
        full = label_on_canvas(label, (60, 60), gray.shape)
        return gray, zone, out, full

    def test_zero_leak_outside_label(self):
        gray, zone, out, label = self.paste_default()
        diff = out.astype(int) - gray.astype(int)
        assert np.count_nonzero(diff[~label]) == 0        # 标签 0 的背景一个都不动

    def test_visible_inside_label(self):
        gray, zone, out, label = self.paste_default()
        diff = np.abs(out.astype(int) - gray.astype(int))
        in_zone = label & zone
        assert float(diff[in_zone].mean()) > 20           # 区内可见

    def test_dark_polarity_no_positive_burst(self):
        gray, zone, out, label = self.paste_default(polarity=-1)
        diff = out.astype(int) - gray.astype(int)
        in_zone = label & zone
        assert float(diff[in_zone].max()) <= 6.0          # 暗缺陷极性: 正残差截断在 +4×α
        assert float(diff[in_zone].min()) < -20           # 有实质变暗

    def test_target_diff_scales_amplitude(self):
        _, _, out30, label30 = self.paste_default(target_diff=30.0)
        _, _, out120, label120 = self.paste_default(target_diff=120.0)
        gray, zone, _, _ = self.paste_default()
        d30 = np.abs(out30.astype(int) - gray.astype(int))[label30 & zone].mean()
        d120 = np.abs(out120.astype(int) - gray.astype(int))[label120 & zone].mean()
        assert d120 > d30 * 1.5                           # 重标定放大可见对比度

    def test_hole_gate_blocks_brightening_in_dark(self):
        # 暗底图 (≤20) 上贴亮缺陷：孔洞门控把正残差全部压掉
        h = w = 160
        gray = np.full((h, w), 10, dtype=np.uint8)
        zone = np.zeros((h, w), dtype=bool)
        zone[30:130, 30:130] = True
        patch = np.full((36, 36), 10, dtype=np.uint8)
        mask = np.zeros((36, 36), dtype=np.float32)
        cv2.circle(mask, (18, 18), 10, 1.0, -1)
        cv2.circle(patch, (18, 18), 10, 220.0, -1)        # 亮缺陷
        got = paste_poisson(gray.copy(), patch, mask, random.Random(0), off=(60, 60),
                            zone_win=zone.copy(), polarity=1)
        if got is None:
            pytest.skip("seamlessClone 在极暗场景偶发无解")
        out, label = got
        full = label_on_canvas(label, (60, 60), gray.shape)
        diff = out.astype(int) - gray.astype(int)
        assert float(diff[full].max()) <= 2.0             # 门控: 暗区不许提亮

    def test_naive_clone_has_halo(self):
        gray, zone, patch, mask = make_scene()
        naive = naive_clone(gray, patch, mask, off=(60, 60))
        full_mask = label_on_canvas(mask > 0.30, (60, 60), gray.shape)
        halo = int(((naive != gray) & ~full_mask).sum())
        assert halo > 0                                   # 朴素克隆必然泄漏掩码外像素


class TestPlaceRules:
    def test_fifty_fifty_and_truncation(self):
        zone, zy, zx, d_vals, d_max = make_zone()
        _, mask = make_defect_patch()
        rng = random.Random(3)
        modes, fracs = [], []
        for _ in range(300):
            got = try_place(rng, mask, zone, zy, zx, d_vals, d_max)
            if got is None:
                continue
            oy, ox, is_b = got
            modes.append(is_b)
            ph, pw = mask.shape
            region = zone[oy:oy + ph, ox:ox + pw]
            fracs.append(((mask & region).sum() / mask.sum(), is_b))
        assert 0.30 < float(np.mean(modes)) < 0.70        # 五五分
        edge = [f for f, b in fracs if b]
        inner = [f for f, b in fracs if not b]
        assert min(inner) >= 0.75                         # 区内模式基本完整
        assert any(f < 0.98 for f in edge)                # 边缘模式有截断形态

    def test_interior_min_in_px(self):
        zone, zy, zx, d_vals, d_max = make_zone()
        _, mask = make_defect_patch()
        rng = random.Random(5)
        for _ in range(100):
            got = try_place(rng, mask, zone, zy, zx, d_vals, d_max,
                            boundary_prob=0.0)             # 强制区内模式
            if got is None:
                continue
            oy, ox, _ = got
            ph, pw = mask.shape
            inside = (mask & zone[oy:oy + ph, ox:ox + pw]).sum()
            assert inside >= max(8, int(0.75 * mask.sum()))

    def test_edge_min_in_px(self):
        zone, zy, zx, d_vals, d_max = make_zone()
        _, mask = make_defect_patch()
        rng = random.Random(6)
        for _ in range(100):
            got = try_place(rng, mask, zone, zy, zx, d_vals, d_max,
                            boundary_prob=1.0)             # 强制边缘模式
            if got is None:
                continue
            oy, ox, _ = got
            ph, pw = mask.shape
            inside = (mask & zone[oy:oy + ph, ox:ox + pw]).sum()
            assert inside >= max(8, int(0.15 * mask.sum()))

    def test_visibility_check(self):
        gray = np.full((60, 60), 150, dtype=np.uint8)
        m = np.zeros((60, 60), dtype=bool)
        m[20:30, 20:30] = True
        faint = gray.copy()
        faint[m] = 147                                    # Δ=3
        strong = gray.copy()
        strong[m] = 90                                    # Δ=60
        assert not visibility_check(faint, gray, m)       # 隐形缺陷拒收
        assert visibility_check(strong, gray, m)
        assert VISIBLE_MIN_DIFF == 8.0

    def test_context_window(self):
        img = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)
        win = context_window(img, 0, 0, 50, 10)
        assert win.shape == (70, 70)
        assert np.array_equal(win[10:60, 10:60], img[0:50, 0:50])   # 中心块原样
        inner = context_window(img, 20, 20, 50, 10)
        assert inner.shape == (70, 70)                    # 不贴边时不补镜像

    def test_sample_target_median_anchor(self):
        rng = random.Random(0)
        lo, med, hi = 20.0, 50.0, 500.0
        below = sum(sample_target(rng, lo, hi, med=med) < med for _ in range(1000))
        assert 0.40 < below / 1000 < 0.60                 # 中位锚定五五开
        assert solve_scale(100.0, 400.0) == pytest.approx(2.0)


class TestDeckSampling:
    def test_deck_full_coverage_first_round(self):
        n = 12
        first = [deck_draw("z", "d", "all", n, g) for g in range(n)]
        assert sorted(first) == list(range(n))            # 无放回全覆盖

    def test_deck_deterministic_across_processes(self):
        a = deck_perm("加工面", "划痕", "all", 10, 0)
        b = deck_perm("加工面", "划痕", "all", 10, 0)
        assert a == b
        c = deck_perm("装配面", "划痕", "all", 10, 0)      # 换区换牌
        assert a != c

    def test_deck_balanced_over_many_rounds(self):
        n = 10
        draws = [deck_draw("z", "d", "all", n, g) for g in range(10 * n)]
        counts = np.bincount(draws, minlength=n)
        assert counts.min() >= 8 and counts.max() <= 12   # 每张 10±2 次

    def test_sample_rng_formula(self):
        a = [sample_rng(42, 1, 0, 7).random() for _ in range(5)]
        b = [sample_rng(42, 1, 0, 7).random() for _ in range(5)]
        c = [sample_rng(42, 2, 0, 7).random() for _ in range(5)]
        d = [sample_rng(42, 1, 1, 7).random() for _ in range(5)]
        assert a == b and a != c and a != d

    def test_sampler_epoch_encoding_and_shards(self):
        ds = ShardDataset(n_pairs=8, num_shards=2)
        n_pool = len(ds.pool_items)
        for epoch in (1, 2):
            s = EpochShardSampler(ds, num_replicas=1, seed=0)
            s.set_epoch(epoch)
            idxs = list(iter(s))
            assert all(decode_index(i, n_pool)[0] == epoch for i in idxs)
        s1 = EpochShardSampler(ds, seed=0)
        s1.set_epoch(1)
        s2 = EpochShardSampler(ds, seed=0)
        s2.set_epoch(2)
        i1 = {decode_index(i, n_pool)[1] for i in s1}
        i2 = {decode_index(i, n_pool)[1] for i in s2}
        assert i1 | i2 == set(range(n_pool))              # 两分片合起来全覆盖
        assert len(i1 & i2) == 0

    def test_sampler_ddp_replica_split(self):
        ds = ShardDataset(n_pairs=10, num_shards=1)
        n_pool = len(ds.pool_items)
        parts = []
        for rank in (0, 1):
            s = EpochShardSampler(ds, num_replicas=2, rank=rank, seed=0)
            parts.append({decode_index(i, n_pool)[1] for i in s})
        assert len(parts[0] & parts[1]) == 0              # 条带互斥
        assert parts[0] | parts[1] == set(range(n_pool))  # 合并全覆盖


class TestTileCache:
    def test_lru_budget_eviction(self):
        img = np.zeros((1024, 1024), dtype=np.uint8)      # 1MB/张
        cache = LRUBudgetCache(budget_bytes=int(2.5 * 2**20))
        for k in ("i0", "i1", "i2"):
            cache.put(k, img)
        assert len(cache) == 2 and cache.size <= cache.budget + img.nbytes
        assert cache.get("i0") is None                    # 放 i2 时最旧的 i0 已被淘汰
        assert cache.get("i1") is not None and cache.get("i2") is not None
        cache.put("i3", img)                              # 此时 LRU 是 i1（刚被 get 刷新的是 i2）
        assert cache.get("i1") is None and cache.get("i2") is not None

    def test_cache_budget_formula(self):
        assert solve_cache_budget(16 * 1024, holders=1) == pytest.approx(8 * 1024 * 0.6)
        assert solve_cache_budget(16 * 1024, holders=4) == pytest.approx(8 * 1024 * 0.6 / 4)
        assert solve_cache_budget(4 * 1024) == 0.0        # 内存不足 → 不缓存

    def test_multi_zone_mutex(self):
        ts = 128
        zones = {
            "A": {"mask": np.zeros((ts, ts), bool)},
            "B": {"mask": np.zeros((ts, ts), bool)},
        }
        zones["A"]["mask"][10:60, 10:60] = True
        zones["B"]["mask"][10:60, 70:120] = True

        def sampler(zname, rng):
            m = np.zeros((ts, ts), bool)
            if zname == "A":
                m[30:50, 50:75] = True                    # 越界进 B 的地带
            else:
                m[30:50, 75:95] = True
            return m
        occupied, pasted, placed = paste_multi_zone(zones, sampler, random.Random(0))
        assert pasted == ["A", "B"]
        assert int((occupied & zones["B"]["mask"]).sum()) == 0 or \
            occupied[30:50, 70:75].sum() == 0              # A 的越界部分被互斥截掉
        a_part = occupied[30:50, 50:60]
        assert a_part.all()                               # A 在自己区内的部分保留

    def test_jitter_bounds_and_annotation(self):
        H = W = 1000
        ann = np.zeros((H, W), dtype=bool)
        ann[300:320, 400:420] = True
        union = np.ones((H, W), dtype=bool)
        rng = random.Random(1)
        moved = 0
        for _ in range(100):
            x, y = jitter_coord(384, 384, 96, W, H, 128, rng, ann, union)
            assert 0 <= x <= W - 128 and 0 <= y <= H - 128
            assert not ann[y:y + 128, x:x + 128].any()
            moved += (x, y) != (384, 384)
        assert moved > 50                                 # 抖动确实生效

    def test_jitter_falls_back_when_blocked(self):
        # 候选全被标注挡住时回退原坐标（而不是死循环）
        H = W = 300
        ann = np.zeros((H, W), dtype=bool)
        ann[:, :] = True                                  # 全图都是标注
        union = np.ones((H, W), dtype=bool)
        rng = random.Random(0)
        x, y = jitter_coord(80, 80, 96, W, H, 128, rng, ann, union)
        assert (x, y) == (80, 80)
