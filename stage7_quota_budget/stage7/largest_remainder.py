"""课时 3：最大余数法取整（Largest Remainder）—— budget.py::largest_remainder_round。

把浮点配额（注水分配的结果、跨点位合成张数）取整成整数，要求**总和严格等于
目标值**——四舍五入各自为政会差 1~3 张，一天几百张的合成任务里这就是"永远差一点"
的增量补齐循环。

规则：floor 取整后按小数部分从大到小把差额逐个 +1（"谁差得多给谁"）；
小数部分并列（比如三个区都是 .50 差 2 个名额）用**种子化随机平票扰动**打破——
同一 seed 下结果确定，但不同 seed 不总偏袒排在前面的区。

对应主仓库：src/train/budget.py::largest_remainder_round
（被 compute_product_budget 用于把 [60, 600] 钳位后的合成总配额分回各点位）。

运行：python -m stage7.largest_remainder
"""
import numpy as np


def largest_remainder_round(weights, target_total: int, seed: int = 0):
    """非负浮点权重 → 严格加和等于 target_total 的非负整数列表。"""
    n = len(weights)
    if n == 0 or target_total <= 0:
        return [0] * n
    w_arr = np.maximum(0.0, np.asarray(weights, dtype=np.float64))
    w_sum = float(w_arr.sum())
    if w_sum <= 0.0:
        return [0] * n

    scaled = w_arr * (float(target_total) / w_sum)
    floors = np.floor(scaled).astype(np.int64)
    rem = scaled - floors
    deficit = int(target_total - int(floors.sum()))
    if deficit > 0:
        rng = np.random.default_rng(seed)
        tie = rng.random(n) * 1e-9                      # 种子化平票扰动
        order = np.argsort(-(rem + tie))
        floors[order[:deficit]] += 1
    return floors.tolist()


if __name__ == "__main__":
    # 经典案例：3 个区按权重分 100 张，权重和 300
    alloc = largest_remainder_round([100.0, 100.0, 100.0], 100, seed=0)
    print(f"均分: {alloc} 合计 {sum(alloc)}")

    alloc = largest_remainder_round([0.3, 0.3, 0.3, 0.1], 10, seed=0)
    print(f"3:3:3:1 分 10: {alloc} 合计 {sum(alloc)} (各 3.0→3 张, 1.0→1 张, 零损失)")

    # 平票扰动：三个区小数全 .5 只差 2 个名额，不同 seed 分给不同区
    for s in (0, 1, 2, 3):
        print(f"平票 [1.5, 1.5, 1.5] 分 5, seed={s}: "
              f"{largest_remainder_round([1.5, 1.5, 1.5], 5, seed=s)}")

    # 对照：朴素 round 各自为政 → 总和漂移
    naive = [round(w / 300 * 100) for w in (100.0, 100.0, 100.0)]
    print(f"朴素 round 各自为政: {naive}（本例恰好也对，换 [1.5,1.5,1.5] 分 5 就漂了："
          f"{[round(x) for x in [5/3, 5/3, 5/3]]} 合计 {sum(round(x) for x in [5/3, 5/3, 5/3])}）")
