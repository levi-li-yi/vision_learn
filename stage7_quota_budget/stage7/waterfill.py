"""课时 2：带上下界约束的注水分配（Water-Filling）—— budget.py::_waterfill_allocate。

单轮正样本图块预算 B_epoch 要分给各个 (点位, 检测区)，同时满足三方约束：

  · 按权重成比例：W = √(T·D)（干净块数 × 缺陷种类数开根号）——面积大的区多分，
    但开根号压平差距，小检测区保底放大；
  · 下界保底 L = min(5T, max(100·D, T/epochs))：每种缺陷每轮至少 100 块正样本，
    否则小类别学不会；
  · 上界封顶 U = 5T：单块底图每轮最多重复 5 次——极小检测区靠无限复读会过拟合。

注水（water-filling）算法：按权重把预算倒进池子，水位溢出上下界的池子先被"钉死"
在边界上并退出自由分配，剩余预算在余下池子间继续按权重分，直到没有池子越界。
逐行复刻 _waterfill_allocate（含 max(1e-6, w) 防零权重除零、越界池子逐轮钉死循环）。

对应主仓库：src/train/budget.py::_waterfill_allocate / compute_product_budget(权重段)。

运行：python -m stage7.waterfill
"""
import numpy as np

EPS_W = 1e-6   # 对应真工程 max(1e-6, weights[k])：零权重池子仍给残口水，不除零


def waterfill_allocate(weights: dict, lows: dict, highs: dict, total_budget: float) -> dict:
    """带上下界约束的连续注水分配：lows[k] <= alloc[k] <= highs[k]，常规配置 Σalloc = total_budget。

    注：与真工程一致，同一轮内 w_sum 不即时更新——对抗性的上下界组合可能留少量
    预算碎屑没分掉（Σalloc ≤ 预算恒成立）；真实素材配置（L ≪ 比例份额 ≪ U）精确收敛。
    """
    keys = sorted(weights.keys())
    if not keys:
        return {}
    sum_low = sum(lows[k] for k in keys)
    sum_high = sum(highs[k] for k in keys)
    if total_budget <= sum_low:
        return {k: float(lows[k]) for k in keys}          # 预算不足：全部按下界躺平
    if total_budget >= sum_high:
        return {k: float(highs[k]) for k in keys}         # 预算充裕：全部顶到上界

    alloc: dict = {}
    free = set(keys)
    rem_budget = float(total_budget)

    for _ in range(len(keys) + 2):
        if not free:
            break
        w_sum = sum(max(EPS_W, weights[k]) for k in free)
        clamped_any = False
        for k in list(free):
            share = rem_budget * (max(EPS_W, weights[k]) / w_sum)
            if share < lows[k]:
                alloc[k] = float(lows[k])                  # 水位不够下界 → 钉在下界
                rem_budget -= alloc[k]
                free.remove(k)
                clamped_any = True
            elif share > highs[k]:
                alloc[k] = float(highs[k])                 # 水位漫过上界 → 钉在上界
                rem_budget -= alloc[k]
                free.remove(k)
                clamped_any = True
        if not clamped_any:                                # 本轮无人越界 → 按权重分完剩余
            for k in free:
                alloc[k] = rem_budget * (max(EPS_W, weights[k]) / w_sum)
            break
    for k in keys:
        if k not in alloc:
            alloc[k] = float(np.clip(lows[k], lows[k], highs[k]))
    return alloc


def zone_weights(tile_counts: dict, defect_counts: dict) -> dict:
    """W_{p,z} = √(T_{p,z} · D_{p,z})：compute_product_budget 的权重公式。"""
    return {k: float(np.sqrt(max(1, tile_counts[k]) * max(1, defect_counts[k])))
            for k in tile_counts}


def zone_bounds(tile_counts: dict, defect_counts: dict, epochs: int = 20,
                max_variants: float = 5.0, min_per_defect: float = 100.0) -> dict:
    """下界 L = min(5T, max(100D, T/epochs))、上界 U = 5T——budget.py 同名公式。"""
    lows, highs = {}, {}
    for k, t in tile_counts.items():
        d = max(1, defect_counts[k])
        highs[k] = max_variants * float(t)
        lows[k] = min(highs[k],
                      max(min_per_defect * float(d), float(t) / float(max(1, epochs))))
    return lows, highs


if __name__ == "__main__":
    # 一个大区 + 一个中区 + 一个极小区：开根号权重 + 下界保底 + 上界 5T 封顶
    tile_counts = {"P1/加工面": 1200, "P1/装配面": 300, "P1/槽底": 40}
    defect_counts = {"P1/加工面": 2, "P1/装配面": 1, "P1/槽底": 1}
    weights = zone_weights(tile_counts, defect_counts)
    lows, highs = zone_bounds(tile_counts, defect_counts)
    print(f"{'区':<10}{'T':>6}{'D':>3}{'W=√(TD)':>9}{'下界L':>8}{'上界U=5T':>9}")
    for k in sorted(tile_counts):
        print(f"{k:<10}{tile_counts[k]:>6}{defect_counts[k]:>3}"
              f"{weights[k]:>9.1f}{lows[k]:>8.0f}{highs[k]:>9.0f}")

    for budget in (300, 1200, 3000, 8000):
        alloc = waterfill_allocate(weights, lows, highs, budget)
        s = " | ".join(f"{k.split('/')[1]}={alloc[k]:.0f}" for k in sorted(alloc))
        print(f"预算 {budget:>5}: {s} | 合计 {sum(alloc.values()):.0f}")

    print("\n看三个极端：预算 300 时极小区被下界(=min(5T,100)=100)保住；"
          "预算 8000 时大区顶到上界 5T=6000 封顶防过拟合；"
          "中区在两个极端之间始终按权重平滑伸缩。")
