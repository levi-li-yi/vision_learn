"""课时 4：阈值扫描与 Micro 指标——eval.py run_sweep 的复刻。

对应主仓库 eval.py 129-255 行：
  一次前向 -> 多阈值累计 tp/fp/fn/tn/交/并 -> Global Micro-Averaging
  -> Acc/P/R/F1/IoU/Dice 表 + best-F1 标注（eps=1e-7 防除零）。

Micro(全局合并) vs 逐样本平均的差别也在此定量复现：
  micro 里大图话语权天然大——与训练期 iou_ng(逐样本平均)不是一把尺。

运行：python -m stage3.threshold_sweep
"""
import numpy as np


def sweep_metrics(scores: np.ndarray, gts: np.ndarray,
                  thresholds=None, eps=1e-7):
    """scores/gts: 展平的像素级 (分数, 0/1 答案)。返回逐阈值指标表。"""
    if thresholds is None:
        thresholds = sorted({round(i * 0.1, 2) for i in range(1, 10)})
    table = {}
    for thr in thresholds:
        pred = scores > thr
        tp = float((pred & (gts > 0.5)).sum())
        fp = float((pred & (gts <= 0.5)).sum())
        fn = float((~pred & (gts > 0.5)).sum())
        tn = float((~pred & (gts <= 0.5)).sum())
        inter = tp; union = tp + fp + fn
        p = tp / (tp + fp + eps); r = tp / (tp + fn + eps)
        table[thr] = {
            "pixel_acc": (tp + tn) / (tp + tn + fp + fn + eps),
            "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r + eps),
            "iou": inter / (union + eps),
        }
    return table


def best_f1_threshold(table: dict) -> float:
    """best-F1 档（调参后'最优阈值有没有漂移'就看这行）。"""
    return max(table, key=lambda t: table[t]["f1"])


def micro_vs_per_sample() -> dict:
    """构造案例：图A 大缺陷(1000px)、图B 小缺陷(10px)。
    同样的检测质量下，micro 里 A 的贡献是 B 的 100 倍；
    逐样本平均则两图等权——两把尺的分歧定量呈现。"""
    case = {
        "A_scores": np.concatenate([np.full(9000, 0.1), np.full(1000, 0.9)]),
        "A_gt": np.concatenate([np.zeros(9000), np.ones(1000)]),
        "B_scores": np.concatenate([np.full(9990, 0.1), np.full(10, 0.9)]),
        "B_gt": np.concatenate([np.zeros(9990), np.ones(10)]),
    }
    micro = sweep_metrics(
        np.concatenate([case["A_scores"], case["B_scores"]]),
        np.concatenate([case["A_gt"], case["B_gt"]]), thresholds=[0.5])
    # 逐样本：A 图 IoU = 1.0，B 图 IoU = 1.0 —— 两图都完美
    per_sample_iou = [1.0, 1.0]
    return {"micro_at_0.5": micro[0.5], "per_sample_mean_iou":
            float(np.mean(per_sample_iou))}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 合成"分数图"：缺陷区中心 0.8、边缘 0.5、背景 0.1
    n_ok, n_def = 10000, 200
    scores = np.concatenate([
        rng.normal(0.1, 0.05, n_ok),
        rng.normal(0.5, 0.15, n_def // 2),      # 边缘带（难）
        rng.normal(0.8, 0.08, n_def // 2),      # 核心
    ]).clip(0, 1)
    gts = np.concatenate([np.zeros(n_ok), np.ones(n_def)])
    table = sweep_metrics(scores, gts)
    print(f"{'thr':<5} {'Acc':<7} {'P':<7} {'R':<7} {'F1':<7} {'IoU':<7}")
    for thr, m in table.items():
        mark = " <-- best F1" if thr == best_f1_threshold(table) else ""
        print(f"{thr:<5} {m['pixel_acc']:<7.4f} {m['precision']:<7.4f} "
              f"{m['recall']:<7.4f} {m['f1']:<7.4f} {m['iou']:<7.4f}{mark}")
    r = micro_vs_per_sample()
    print("\nMicro vs 逐样本：micro 下大缺陷话语权大；逐样本两图等权 ->",
          {k: round(v if isinstance(v, float) else v.get('f1', v), 4)
           for k, v in r.items()})
