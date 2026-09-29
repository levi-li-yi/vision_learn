"""数学基础：线性代数（矩阵乘 / 向量距离） + 概率统计（均值方差 / 分位数）。

为什么学这些——每一节都直通仓库代码：
- 矩阵乘 / 距离：卷积 = 邻域窗口与核的点积；无监督线的异常分数 =
  teacher 与 student 输出向量的逐像素距离平方（modeling.py / utils.py::predict）。
- E[x^2] - E[x]^2 求方差：unsupervised_training/core/utils.py::teacher_normalization
  统计 teacher 输出通道均值方差用的正是这个公式（sum_sq/count - mean^2）。
- 分位数（kthvalue 风格）：无监督线的两处命脉——
    a) 训练损失取最难的"前 0.1%"像素（trainer.py, k = int(numel*0.999)）
    b) 异常图用 90%/99.5% 分位做归一化标定（utils.py::_map_quantiles）

运行方式：
    python -m stage0.math_basics
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. 线性代数
# ---------------------------------------------------------------------------

def matmul_by_hand(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """手写矩阵乘法（三重循环定义版）。

    定义：c[i,j] = sum_k a[i,k] * b[k,j]。
    仓库里的卷积层（PDN 的 Conv2d）本质就是对每个输出位置做这样的
    "局部窗口点积"——先懂矩阵乘，才懂卷积在算什么。
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"形状不匹配: {a.shape} @ {b.shape}")
    rows, inner = a.shape
    inner2, cols = b.shape

    c = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            acc = 0.0
            for k in range(inner):
                acc += a[i, k] * b[k, j]
            c[i, j] = acc
    return c


def l2_distance_by_hand(u: np.ndarray, v: np.ndarray) -> float:
    """手写欧氏距离：sqrt(sum_i (u_i - v_i)^2)。

    仓库的逐像素异常分数 = mean_c (teacher_c - student_c)^2，
    就是"距离平方的平均"——本函数去掉 sqrt 的版本。
    """
    diff = np.asarray(u, dtype=np.float64) - np.asarray(v, dtype=np.float64)
    return float(np.sqrt(np.sum(diff ** 2)))


def squared_channel_distance(teacher_patch: np.ndarray, student_patch: np.ndarray) -> np.ndarray:
    """迷你版异常图：两组"特征图"逐位置距离平方再对通道取平均。

    teacher_patch/student_patch 形状 (H, W, C)，
    输出 (H, W)——每个位置的"模仿差距"。
    对照 unsupervised_training/core/utils.py::predict 的
    map_combined = mean((teacher_output - student_output)^2, dim=通道)。
    """
    diff = teacher_patch.astype(np.float32) - student_patch.astype(np.float32)
    return np.mean(diff ** 2, axis=-1)


# ---------------------------------------------------------------------------
# 2. 概率统计
# ---------------------------------------------------------------------------

def mean_std_by_sum_sq(x: np.ndarray):
    """用 E[x^2] - E[x]^2 一步算均值和标准差（仓库 teacher_normalization 同款）。

    为什么不直接 x.std()？因为流式/分桶累计时只需维护 sum 和 sum_sq 两个
    累计量即可，不必存全部数据——仓库正是这样逐桶累计的。
    注意：该公式有浮点误差，方差可能算出微小的负数，所以仓库用了
    torch.sqrt(torch.relu(var)) 钳成 0；此处用 max(var, 0) 同款防御。
    """
    x = x.astype(np.float64)
    mean = x.sum() / x.size
    mean_sq = (x ** 2).sum() / x.size
    var = max(mean_sq - mean ** 2, 0.0)
    return float(mean), float(np.sqrt(var))


def kth_smallest(x: np.ndarray, q: float) -> float:
    """按分位比例取第 k 小的元素——torch.kthvalue 的手写版。

    仓库口径（务必一致）：
      k = int(n * q)，取第 k 小（1 起数），并钳位到 [1, n]。
    对照 unsupervised_training/core/utils.py::_map_quantiles 的
      k_start = min(max(int(n * 0.9), 1), n)
    """
    n = x.size
    k = int(n * q)
    k = min(max(k, 1), n)
    sorted_asc = np.sort(np.asarray(x, dtype=np.float64).ravel())
    return float(sorted_asc[k - 1])


def map_quantiles(anomaly_map: np.ndarray):
    """迷你版分位标定：返回异常图的 90% / 99.5% 分位 (q_start, q_end)。

    这两个数在仓库里的用途：推理时把原始差距图做
      (map - q_start) / (q_end - q_start)
    仿射归一化——"先看正常图自己的差距一般多大，以此为尺子刻度"。
    """
    return kth_smallest(anomaly_map, 0.9), kth_smallest(anomaly_map, 0.995)


def hard_mining_quantile(distance_map: np.ndarray, q: float = 0.999) -> float:
    """迷你版困难挖掘阈值：返回前 (1-q) 最难像素的下界值。

    仓库训练损失：只对 distance >= kth_smallest(0.999) 的像素求均值。
    """
    return kth_smallest(distance_map, q)


def simulate_anomaly_scores(n_ok: int = 1000, n_ng: int = 200, seed: int = 42):
    """模拟无监督线的考试场景：OK 图分数低而集中、NG 图分数高而分散。

    用于直观感受"分位数怎么从分布里切出阈值"。
    """
    rng = np.random.default_rng(seed)
    ok_scores = rng.normal(0.2, 0.05, size=n_ok).clip(0, None)   # 好品分数
    ng_scores = rng.normal(0.7, 0.12, size=n_ng).clip(0, None)   # 坏品分数
    return ok_scores, ng_scores


def auc_by_hand(labels: np.ndarray, scores: np.ndarray) -> float:
    """手写 AUC：随机抽一正一负，正样本分数更高的概率。

    用两两比较 O(n_pos * n_neg) 的定义式实现（教学用），
    对照 sklearn.metrics.roc_auc_score——仓库用它评估异常检测模型。
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = scores[labels == 1], scores[labels == 0]

    wins = ties = 0
    for p in pos:
        wins += int((p > neg).sum())
        ties += int((p == neg).sum())
    return (wins + 0.5 * ties) / (pos.size * neg.size)


# ---------------------------------------------------------------------------
# 演示入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("1) 线性代数：手写矩阵乘 vs np.matmul")
    print("-" * 60)
    a = np.array([[1., 2.], [3., 4.]])
    b = np.array([[5., 6.], [7., 8.]])
    hand = matmul_by_hand(a, b)
    print(f"手写结果 =\n{hand}")
    print(f"np.matmul 结果 =\n{a @ b}")
    assert np.allclose(hand, a @ b), "两者应完全一致"
    print("一致 ✓（以后一律用 @ / np.matmul，但要懂它算的是什么）")

    print("=" * 60)
    print("2) 向量距离：teacher vs student 特征")
    print("-" * 60)
    teacher_feat = np.array([1.0, 0.5, -0.3, 2.0])
    student_normal = np.array([1.02, 0.48, -0.35, 1.97])        # 模仿得好
    student_anomaly = np.array([-0.8, 1.9, 0.4, -1.1])          # 模仿不上
    print(f"正常位置距离 = {l2_distance_by_hand(teacher_feat, student_normal):.4f} （小）")
    print(f"异常位置距离 = {l2_distance_by_hand(teacher_feat, student_anomaly):.4f} （大）")
    print("-> 异常检测的全部直觉就在这两个数的大小差异里")

    print("=" * 60)
    print("3) 异常图：逐位置通道距离（迷你版 predict）")
    print("-" * 60)
    rng = np.random.default_rng(0)
    t = rng.normal(0, 1, size=(8, 8, 4)).astype(np.float32)
    s = t + rng.normal(0, 0.05, size=(8, 8, 4)).astype(np.float32)
    s[5, 5] += 4.0                                               # 埋一个异常
    amap = squared_channel_distance(t, s)
    print(f"异常图 shape={amap.shape}（H,W），埋异常处值 = {amap[5, 5]:.2f}，"
          f"其余均值 = {np.delete(amap, 5*8+5).mean():.4f}")

    print("=" * 60)
    print("4) 均值/方差：E[x^2]-E[x]^2 与 np 的对照")
    print("-" * 60)
    x = rng.normal(50, 7, size=10000)
    m1, s1 = mean_std_by_sum_sq(x)
    print(f"手写累计式: mean={m1:.4f}, std={s1:.4f}")
    print(f"NumPy 内置: mean={x.mean():.4f}, std={x.std():.4f}")

    print("=" * 60)
    print("5) 分位数：标定与困难挖掘（无监督线的命脉）")
    print("-" * 60)
    q_start, q_end = map_quantiles(amap)
    hard_thr = hard_mining_quantile(amap, q=0.999)
    print(f"异常图 90% 分位  q_start = {q_start:.6f}")
    print(f"异常图 99.5%分位 q_end  = {q_end:.6f}")
    print(f"99.9% 分位（困难挖掘阈值） = {hard_thr:.6f}")
    print(f"归一化后异常位置分数 = {(amap[5,5] - q_start) / (q_end - q_start):.2f}"
          f" （跨模型可比的标准分）")

    print("=" * 60)
    print("6) AUC：手写定义式")
    print("-" * 60)
    ok_scores, ng_scores = simulate_anomaly_scores()
    labels = np.concatenate([np.zeros(len(ok_scores)), np.ones(len(ng_scores))])
    scores = np.concatenate([ok_scores, ng_scores])
    print(f"模拟 OK 分数带: [{ok_scores.min():.2f}, {ok_scores.max():.2f}]")
    print(f"模拟 NG 分数带: [{ng_scores.min():.2f}, {ng_scores.max():.2f}]")
    print(f"手写 AUC = {auc_by_hand(labels, scores):.4f}")
    try:
        from sklearn.metrics import roc_auc_score
        print(f"sklearn AUC = {roc_auc_score(labels, scores):.4f} （对照）")
    except ImportError:
        print("（未装 sklearn，跳过对照——含义：随机一好一坏，坏品分数更高的概率）")
