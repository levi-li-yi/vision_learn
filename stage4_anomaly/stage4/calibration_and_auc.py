"""课时 4+5：分位标定 + 退化兜底阶梯 + AUC/TopK——core/utils.py 的复刻。
这个文件教你无监督线最含金量的部分：怎么把'距离'变成一把可比的尺子——分位标定、退化兜底、逐桶独立标定、TopK100 打分、AUC。


对应主仓库：
  - _map_quantiles：OK 池异常图的 90%/99.5% 分位（q_start/q_end）；
  - _robust_quantiles：ungated -> region 门控 -> 相对 eps 三级兜底
    （大面积同值像素会把两分位吞成同一个数 -> 除零 -> inf/nan）；
  - predict：score = 0.1 * (m - q_start) / (q_end - q_start)（保序仿射）；
  - MIN_BUCKET_SAMPLES=4 的逐桶门槛与全局回退；
  - test() 的 TopK100 均值打分（防单噪点劫持最大值）。

运行：python -m stage4.calibration_and_auc
"""
import numpy as np
import torch

Q_SCALE_MIN_REL = 1e-6          # 对应主仓库 _Q_SCALE_MIN_REL
MIN_BUCKET_SAMPLES = 4          # 对应主仓库同名常量


def map_quantiles(maps: torch.Tensor, q_start=0.9, q_end=0.995):
    """kthvalue 版分位（k 钳位到 [1,n] 防小样本越界）。"""
    n = maps.numel()
    ks = min(max(int(n * q_start), 1), n)
    ke = min(max(int(n * q_end), 1), n)
    flat = maps.flatten()
    return (torch.kthvalue(flat, ks).values,
            torch.kthvalue(flat, ke).values)


def robust_quantiles(maps, envs=None, label="global"):
    """三级兜底：ungated -> gated -> eps；恒保证 q_end > q_start。"""
    qs, qe = map_quantiles(maps)
    if qe > qs:
        return qs, qe, "ungated"
    if envs is not None and envs.shape == maps.shape:
        gated = maps[envs > 0]
        if gated.numel() > 0:
            gs, ge = map_quantiles(gated)
            if ge > gs:
                return gs, ge, "gated"
    eps = torch.maximum(torch.maximum(qs.abs(), maps.max()),
                        torch.ones_like(qs)) * Q_SCALE_MIN_REL
    return qs, qs + eps, "eps"


def score_map(m, qs, qe):
    """主仓库 predict 的归一化行：0.1*(m-qs)/(qe-qs)，分母相对 eps 钳制。"""
    denom = (qe - qs).clamp_min(Q_SCALE_MIN_REL * qs.abs().clamp_min(1.0))
    return 0.1 * (m - qs) / denom


def bucket_key(x0: int, y0: int) -> str:
    """桶键 x{X}_y{Y}——四方共用同一函数，禁止各自拼串。"""
    return f"x{x0}_y{y0}"


def build_bucket_table(bucket_maps: dict):
    """逐桶标定：n >= 4 才产出，否则整桶回退全局（主仓库语义）。"""
    all_maps = torch.cat([m.flatten() for m in bucket_maps.values()])
    g_qs, g_qe, g_mode = robust_quantiles(all_maps)
    table, modes = {}, {"global": g_mode}
    for key, maps in bucket_maps.items():
        if maps.numel() < MIN_BUCKET_SAMPLES * maps.shape[-1] * maps.shape[-2]:
            continue
        if len(maps.flatten()) / maps[0].numel() < MIN_BUCKET_SAMPLES:
            continue
        qs, qe, mode = robust_quantiles(maps)
        if mode == "eps":
            continue                                   # 带病桶不入表
        table[key] = {"q_start": qs, "q_end": qe}
        modes[key] = mode
    return table, (g_qs, g_qe), modes


# ---------------------------------------------------------------------------
# AUC 与 TopK 打分
# ---------------------------------------------------------------------------

def roc_auc(labels, scores) -> float:
    """Mann-Whitney：随机(正,负)对中正分更高的比例。"""
    labels = np.asarray(labels); scores = np.asarray(scores)
    pos, neg = scores[labels == 1], scores[labels == 0]
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def best_f1_threshold(labels, scores):
    """PR 曲线上 F1 最大的阈值（主仓库 find_optimal_threshold 同思想）。"""
    labels = np.asarray(labels); scores = np.asarray(scores)
    order = np.argsort(-scores)
    tp = fp = 0; n_pos = int((labels == 1).sum())
    best_f1, best_thr = 0.0, scores[order[0]]
    for i in order:
        if labels[i] == 1: tp += 1
        else: fp += 1
        prec = tp / (tp + fp); rec = tp / n_pos
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        if f1 > best_f1:
            best_f1, best_thr = f1, scores[i]
    return float(best_thr), float(best_f1)


def topk_mean(amap: np.ndarray, k: int = 100) -> float:
    """图像分数 = 最烫 k 个像素的均值（主仓库 test() 同款）。"""
    flat = np.sort(amap.flatten())
    return float(flat[-k:].mean())


if __name__ == "__main__":
    # 正常路径：连续分布的分位
    maps = torch.rand(4, 1, 64, 64)
    qs, qe, mode = robust_quantiles(maps)
    print(f"正常路径: mode={mode}, q_start={qs:.3f}, q_end={qe:.3f}")
    print(f"归一化分数范围 ≈ [{score_map(maps.min(), qs, qe):.3f}, "
          f"{score_map(maps.max(), qs, qe):.3f}]（90%分位记 0）")

    # 退化路径：大面积同值像素（纯色背景切片）
    const_maps = torch.full((4, 1, 64, 64), 0.5)
    envs = torch.zeros_like(const_maps); envs[..., :32] = 1
    qs2, qe2, mode2 = robust_quantiles(const_maps, envs)
    print(f"退化路径: mode={mode2}, q_end-q_start={float(qe2-qs2):.2e}"
          "（eps 只救数值，分数不可跨模型比较）")

    # TopK vs max 的抗噪性
    rng = np.random.default_rng(0)
    amap = rng.normal(0.1, 0.02, (64, 64))
    noisy = amap.copy(); noisy[10, 10] = 5.0        # 单点尖峰噪声
    print(f"TopK100 分数(干净/带噪): {topk_mean(amap):.3f} / {topk_mean(noisy):.3f}")
    print(f"max 分数(干净/带噪):     {amap.max():.3f} / {noisy.max():.3f}"
          " <- max 被单噪点劫持")
