"""课时 3：牌靴无放回抽样 + 逐样本确定性 rng + EpochShardSampler —— dataset.py/assets.py 采样工程。

多进程/多卡训练里"每个缺陷素材被抽中多少次"必须精确可控，否则小素材被背下来、
大素材饿死。真工程三板斧：

  1. 牌靴（deck）确定性无放回：seed = md5("{区}/{缺陷}/{池}/{N}/{牌靴轮号}")，
     default_rng(seed).permutation(N) —— 不依赖进程内随机状态，任意进程/任意卡
     对同一牌号得到**完全一致**的洗牌；每连续 N 次抽取 100% 覆盖全部素材（无放回），
     抽完自动换下一副牌（deck_id = g // N）；
  2. 逐样本确定性 rng：random.Random(seed·1000003 + epoch·100003 + variant·987654
     + pool_idx)——同一 (样本, 轮) 在所有卡上生成**完全相同**的增强与贴图，
     跨卡结果可复现、可回放排错；
  3. EpochShardSampler 轮号编码：采样器产出 index = epoch·len(pool) + pool_idx，
     让 persistent_workers 驻留的 worker 子进程从索引反解出"当前轮号"——
     不发广播也能每轮换分片（修复跨进程不同步）；DDP 时按 rank 切条带 + drop_last。

对应主仓库：src/train/assets.py::_get_deck_perm / sample_for_bind、
src/train/dataset.py::__getitem__(rng 行) / EpochShardSampler。

运行：python -m stage9.deck_sampling
"""
import hashlib
import math
import random

import numpy as np
import torch


def deck_perm(zone: str, defect: str, pool_tag: str, n_pool: int, deck_id: int):
    """无状态确定性洗牌：md5(五元组) 播种 permutation。同牌号任意进程结果一致。"""
    raw = f"{zone}/{defect}/{pool_tag}/{n_pool}/{deck_id}".encode("utf-8")
    seed_val = int.from_bytes(hashlib.md5(raw).digest()[:8], "little")
    return np.random.default_rng(seed_val).permutation(n_pool).tolist()


def deck_draw(zone, defect, pool_tag, n_pool, g: int) -> int:
    """第 g 次（全局连续序号）抽牌：perm[g % N]，抽完 N 张自动换下一副牌。"""
    deck_id = g // max(1, n_pool)
    return deck_perm(zone, defect, pool_tag, n_pool, deck_id)[g % n_pool]


def sample_rng(seed: int, epoch: int, variant: int, pool_idx: int) -> random.Random:
    """逐样本确定性 rng：dataset.__getitem__ 同款公式。"""
    return random.Random(seed * 1000003 + epoch * 100003 + variant * 987654 + pool_idx)


class ShardDataset:
    """迷你数据集：只提供 sampler 需要的 pool_items 与分片接口（替身 TileDataset）。"""

    def __init__(self, n_pairs: int, num_shards: int = 1):
        self.pool_items = [None] * (2 * n_pairs)
        self.num_shards = max(1, num_shards)

    def shard_indices_for_epoch(self, epoch: int):
        cur_shard = max(0, int(epoch) - 1) % self.num_shards
        if self.num_shards == 1:
            return list(range(len(self.pool_items)))
        n_pairs = len(self.pool_items) // 2
        indices = []
        for p_idx in range(cur_shard, n_pairs, self.num_shards):
            indices.append(2 * p_idx)
            indices.append(2 * p_idx + 1)
        return indices


class EpochShardSampler(torch.utils.data.Sampler):
    """轮号编码采样器：产出 epoch·len(pool) + idx，正负样本成对不被拆散。"""

    def __init__(self, dataset, num_replicas: int = 1, rank: int = 0,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = True):
        self.dataset = dataset
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 1

    def set_epoch(self, epoch: int):
        self.epoch = max(1, int(epoch))

    def __iter__(self):
        indices = self.dataset.shard_indices_for_epoch(self.epoch)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch * 1009)
            order = rng.permutation(len(indices)).tolist()
            indices = [indices[i] for i in order]

        if self.drop_last and len(indices) % self.num_replicas != 0:
            total_size = (len(indices) // self.num_replicas) * self.num_replicas
            indices = indices[:total_size]
        else:
            total_size = int(math.ceil(len(indices) / float(self.num_replicas))) * self.num_replicas
            while len(indices) < total_size:
                indices.extend(indices[:total_size - len(indices)])

        my_indices = indices[self.rank:total_size:self.num_replicas]
        n_pool = len(self.dataset.pool_items)
        offset = self.epoch * n_pool
        return iter(offset + int(p_idx) for p_idx in my_indices)

    def __len__(self):
        n_shard = len(self.dataset.shard_indices_for_epoch(self.epoch))
        if self.drop_last:
            return n_shard // self.num_replicas
        return int(math.ceil(n_shard / float(self.num_replicas)))


def decode_index(i: int, n_pool: int):
    """worker 侧从编码索引反解 (epoch, pool_idx)——persistent_workers 的感知机制。"""
    return int(i // n_pool), int(i % n_pool)


if __name__ == "__main__":
    # 1. 牌靴：10 张素材抽 25 次——前 10 次全覆盖，25 次里每张被抽 2~3 次
    n_pool = 10
    draws = [deck_draw("加工面", "划痕", "all", n_pool, g) for g in range(25)]
    first_deck = draws[:10]
    print(f"第 1 副牌 10 次抽取: {first_deck}")
    print(f"  覆盖全部素材: {sorted(first_deck) == list(range(n_pool))} (无放回洗牌)")
    from collections import Counter
    counts = Counter(draws)
    print(f"  25 次抽取各素材次数: {sorted(counts.values())} (每张 2~3 次, 无人饿死/暴富)")

    # 跨"进程"一致性：不依赖任何全局状态，重算一遍逐位相同
    again = [deck_draw("加工面", "划痕", "all", n_pool, g) for g in range(25)]
    print(f"  任意进程重算一致: {draws == again}")

    # 2. 逐样本确定性 rng：同 (样本,轮) 同增强；换轮/换样本立即不同
    r1 = sample_rng(42, 1, 0, 7)
    r2 = sample_rng(42, 1, 0, 7)
    r3 = sample_rng(42, 2, 0, 7)
    a = [r1.random() for _ in range(3)]
    b = [r2.random() for _ in range(3)]
    c = [r3.random() for _ in range(3)]
    print(f"逐样本 rng: 同键重放一致={a == b}, 换轮立即不同={a != c}")

    # 3. EpochShardSampler：2 分片轮转 + DDP 2 卡切条带
    ds = ShardDataset(n_pairs=8, num_shards=2)
    for rank in (0, 1):
        s = EpochShardSampler(ds, num_replicas=2, rank=rank, shuffle=True, seed=0)
        s.set_epoch(1)
        idxs = list(iter(s))
        epochs, pool_idxs = zip(*[decode_index(i, len(ds.pool_items)) for i in idxs])
        print(f"rank{rank} epoch1: 产出 {len(idxs)} 个, 轮号全=1={set(epochs) == {1}}, "
              f"条带={sorted(pool_idxs)}")
    s0 = EpochShardSampler(ds, num_replicas=1, seed=0)
    s0.set_epoch(2)
    idxs2 = list(iter(s0))
    print(f"epoch2 全量: 轮号编码全=2={all(decode_index(i, len(ds.pool_items))[0] == 2 for i in idxs2)}")
