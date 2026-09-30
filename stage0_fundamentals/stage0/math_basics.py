"""数学基础：线性代数（矩阵乘 / 向量距离） + 概率统计（均值方差 / 分位数）。

为什么学这些——每一节都直通仓库代码：
- 矩阵乘 / 距离：卷积 = 邻域窗口与核的点积；无监督线的异常分数 =
  teacher 与 student 输出向量的逐像素距离平方（modeling.py / utils.py::predict）。
- E[x^2] - E[x]^2 求方差：unsupervised_training/core/utils.py::teacher_normalization
  统计 teacher 输出通道均值方差用的正是这个公式（sum_sq/count - mean^2）。
- 分位数（kthvalue 风格）：无监督线的两处命脉——
    a) 训练损失取最难的"前 0.1%"像素（trainer.py, k = int(numel*0.999)）
    b) 异常图用 90%/99.5% 分位做归一化标定（utils.py::_map_quantiles）

注释体例同 numpy_basics.py：每个 API 配"作用 + 参数 + 例: 入参 -> 出参"，
均可在交互环境验证（.venv/Scripts/python.exe）。

运行方式：
    python -m stage0.math_basics
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. 线性代数
# ---------------------------------------------------------------------------

def matmul_by_hand(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """手写矩阵乘法（三重循环定义版）。

    定义：结果 c 的第 i 行第 j 列 = a 的第 i 行 与 b 的第 j 列 对应相乘再求和。
    比喻：a 的每一"横排"去和 b 的每一"竖列"握手，握手的费用 = 对应相乘再累加。
    例（2x2）：
      [[1, 2],   @   [[5, 6],   =   [[1*5+2*7, 1*6+2*8],   =   [[19, 22],
       [3, 4]]        [7, 8]]         [3*5+4*7, 3*6+4*8]]        [43, 50]]
    工程连接：卷积层的每个输出位置就是"局部窗口 x 卷积核"的点积——先懂
    矩阵乘，才懂卷积在算什么。以后一律用 a @ b（np.matmul，BLAS 加速百倍）。
    """
    # 安检：矩阵乘要求 a 的列数 == b 的行数（"中间维"相等），否则握手对不上。
    #   例: (2,3) @ (3,4) 合法（中间都是 3）；(2,3) @ (2,3) -> raise
    # .ndim：维数标签（2 = 矩阵）；.shape[i]：shape 元组第 i 个数
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"形状不匹配: {a.shape} @ {b.shape}")
    # .shape 解包：一行拿两个数。
    #   例: a.shape (2,3) -> rows=2, inner=3；b.shape (3,4) -> cols=4
    rows, inner = a.shape
    inner2, cols = b.shape

    # np.zeros(形状, dtype=类型)：造全 0 表当结果容器（形状参数是元组）。
    #   例: np.zeros((2, 4), dtype=np.float64) -> 2 行 4 列全 0. 的小数表
    c = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):           # 遍历结果的每一行
        for j in range(cols):       # 遍历结果的每一列
            acc = 0.0               # 累加器：本次握手的合计
            for k in range(inner):  # a 第 i 行与 b 第 j 列逐对相乘累加
                acc += a[i, k] * b[k, j]   # a[i,k]：a 表第 i 行第 k 列那个数
            c[i, j] = acc           # 合计填进结果格
    return c


def l2_distance_by_hand(u: np.ndarray, v: np.ndarray) -> float:
    """手写欧氏距离：sqrt(sum_i (u_i - v_i)^2)。

    勾股定理的高维版：(3,4) 到原点 = sqrt(3^2+4^2) = 5；384 维同理——
    各维差平方求和再开方。
    例: u=[0,0], v=[3,4] -> sqrt(9+16) = 5.0
    工程连接：无监督线异常分数 = teacher 与 student 384 维特征"距离平方的
    平均"（本函数去掉 sqrt 的版本）。距离大 = 模仿不上 = 异常——整条
    无监督线就建立在这一个数的大小差异上。
    """
    # np.asarray(数据, dtype=类型)：统一升 float64 再运算（u 可能是列表；
    # 高精度小数让 384 维累加的浮点误差最小）。逐元素减：[3,4]-[0,0]->[3,4]
    diff = np.asarray(u, dtype=np.float64) - np.asarray(v, dtype=np.float64)
    # diff ** 2：逐元素平方 [3,4]->[9,16]（负差翻正，差距不论方向）
    # np.sum(表)：全压成一个数（9+16=25）
    # np.sqrt(数或表)：开方。例: np.sqrt(np.array([9., 25.])) -> array([3.,  5.])
    return float(np.sqrt(np.sum(diff ** 2)))


def squared_channel_distance(teacher_patch: np.ndarray, student_patch: np.ndarray) -> np.ndarray:
    """迷你版异常图：逐位置算"模仿差距"，输出 (H,W) 差距热图。

    输入两组特征图 (H, W, C)——每个位置 C 个数（迷你例 C=4，仓库 C=384）。
    三步：广播逐元素减 -> 平方 -> mean(axis=-1) 压掉通道轴。
    例（某位置，C=4）：
      模仿到位: teacher=[1,0,1,0], student=[1,0,1,0]
                差=0 -> 距离平方=0（正常位置分数约 0）
      埋了异常: student=[-1,0,1,0]
                差平方=[4,0,0,0] -> mean=1.0（该位置分数大）
    对照 unsupervised_training/core/utils.py::predict 的
    map_combined = mean((teacher - student)^2, dim=通道)。
    """
    # 广播逐元素减 (H,W,C)；.astype(np.float32) 先升小数（同 safe_brightness 首步）
    diff = teacher_patch.astype(np.float32) - student_patch.astype(np.float32)
    # ** 2 逐元素平方；np.mean(..., axis=-1)：压掉最后一根轴（通道）。
    #   axis=-1 = 倒数第一维（编号规则见 numpy_basics.channel_means 注释）
    #   例: [[4,0,0,0]] (1,1,4) --mean(axis=-1)--> [[1.0]] (1,1)
    return np.mean(diff ** 2, axis=-1)


# ---------------------------------------------------------------------------
# 2. 概率统计
# ---------------------------------------------------------------------------

def mean_std_by_sum_sq(x: np.ndarray):
    """用 E[x^2] - E[x]^2 一步算均值和标准差（仓库 teacher_normalization 同款）。

    为什么不直接 x.std()：流式/分桶累计只需维护两个累加器 sum 与 sum_sq
    （数据流过一遍就扔，不必存全量）——仓库逐桶统计 teacher 输出正是这样累计。
    数学：方差 = 平方的平均 - 平均的平方（分配律展开可证）。
    陷阱：浮点误差可能算出 -1e-16 的负方差，开方得 NaN -> max(var,0) 钳制，
    与仓库 torch.sqrt(torch.relu(var)) 同款防御。
    例: 全 3.14 的数组 -> mean=3.14, std 残渣 < 1e-6（非精确 0）。
    """
    # .astype(np.float64)：升双精度减小累加误差
    x = x.astype(np.float64)
    # x.size：元素总个数（属性）。例: np.arange(4).size -> 4
    # x.sum()：全压成一个数
    mean = x.sum() / x.size              # 均值 = 总和 / 个数
    mean_sq = (x ** 2).sum() / x.size    # "平方的平均" E[x^2]
    # 方差 = E[x^2] - E[x]^2；max(..., 0.0) 钳负保平安
    var = max(mean_sq - mean ** 2, 0.0)
    # np.sqrt 开方得标准差（波动幅度）
    return float(mean), float(np.sqrt(var))


def kth_smallest(x: np.ndarray, q: float) -> float:
    """按分位比例取"第 k 小"——torch.kthvalue 的手写版（仓库口径）。

    比喻：n 个数从小到大排队，第 k 个出列。
    换算：k = int(n * q)，钳位到 [1, n]（k=1 是最小值，k=n 是最大值）。
    例: x = [0..9] 共 10 个
      kth_smallest(x, 0.9)  k = int(10*0.9) = 9  -> 第 9 小 = 8.0
      kth_smallest(x, 0.5)  k = 5                -> 第 5 小 = 4.0
    仓库两处命脉：训练损失取前 0.1% 最难像素（k=int(n*0.999)）；
    异常图 90%/99.5% 分位标定（_map_quantiles: min(max(int(n*0.9),1),n)）。
    """
    # int(n * q)：分位比例换算成名次（直接砍小数，与仓库一致——不是四舍五入）
    n = x.size
    k = int(n * q)
    # min/max 双向钳位：q=0 时 k=0 -> 提到 1；q 过大乘出界 -> 压到 n
    k = min(max(k, 1), n)
    # np.sort(表)：升序排序（原表不动，返回新排队结果）。
    #   例: np.sort([3, 1, 2]) -> array([1, 2, 3])
    # .ravel()：拉平成一维（几维的表都铺成一条队）。
    #   例: np.array([[1,2],[3,4]]).ravel() -> array([1, 2, 3, 4])
    sorted_asc = np.sort(np.asarray(x, dtype=np.float64).ravel())
    # 排队后取"第 k 个"：下标从 0 数，第 k 个 = 下标 k-1
    return float(sorted_asc[k - 1])


def map_quantiles(anomaly_map: np.ndarray):
    """迷你版分位标定：返回异常图的 90% / 99.5% 分位 (q_start, q_end)。

    用途（仓库 predict）：推理时 (map - q_start) / (q_end - q_start) 把原始
    差距图仿射到"标准分"——先看"正常图自己的差距一般多大"，以此为尺子
    刻度，异常分数才跨模型/跨点位可比。
    """
    return kth_smallest(anomaly_map, 0.9), kth_smallest(anomaly_map, 0.995)


def hard_mining_quantile(distance_map: np.ndarray, q: float = 0.999) -> float:
    """迷你版困难挖掘阈值：前 (1-q) 最难像素的下界。

    仓库 trainer.py：损失只对 distance >= kth_smallest(0.999) 的像素求均值
    ——把学习精力集中在"正常内部最难解释"的位置（分位数困难挖掘）。
    例: n=1000, q=0.999 -> k=999 -> 返回第 999 小，约 1 个像素高过它。
    """
    return kth_smallest(distance_map, q)


def simulate_anomaly_scores(n_ok: int = 1000, n_ng: int = 200, seed: int = 42):
    """模拟无监督线的"考试"场景：OK 图分数低而集中、NG 图分数高而分散。"""
    # np.random.default_rng(种子)：造一台"随机器"。同一种子 -> 同一批随机数
    #   （可复现；本工程所有测试的确定性全靠固定种子）。
    rng = np.random.default_rng(seed)
    # rng.normal(均值, 标准差, size=个数)：从正态分布（钟形曲线）撒数。
    #   例: rng.normal(0.2, 0.05, size=3) -> 约 [0.19, 0.24, 0.15] 一带的小数
    # .clip(0, None)：夹紧，下限 0、上限不限（None = 该侧不设限）——分数非负。
    ok_scores = rng.normal(0.2, 0.05, size=n_ok).clip(0, None)   # 好品：低分带
    ng_scores = rng.normal(0.7, 0.12, size=n_ng).clip(0, None)   # 坏品：高分带
    return ok_scores, ng_scores


def auc_by_hand(labels: np.ndarray, scores: np.ndarray) -> float:
    """手写 AUC：随机抽一好一坏，坏品分数更高的概率。

    0.5 = 瞎猜水平（两带完全重叠）；1.0 = 完美分离。
    例: 好品分数 [0.1, 0.3]、坏品 [0.8] -> 2 场全赢 -> AUC = 1.0
    仓库用 sklearn.roc_auc_score（同义）；此处用定义式两两比较作教学版。
    """
    # 布尔索引（掩码）——新姿势：比较运算先得一张对/错表，方括号拿它"挑行"。
    #   例: scores = [0.1, 0.8, 0.3], labels = [0, 1, 0]
    #       labels == 1          -> [False, True, False]（对/错表）
    #       scores[labels == 1]  -> array([0.8])（只挑坏品那行）
    #       scores[labels == 0]  -> array([0.1, 0.3])
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = scores[labels == 1], scores[labels == 0]

    # 每个好品与整批坏品打擂台（向量化计数，不写内层循环）：
    #   (p > neg)：一个好品对全部坏品的对/错表
    #   .sum()：True 记 1 求和 = 赢的场数（例: 0.8 > [0.1,0.3] -> [T,T] -> 2）
    wins = ties = 0
    for p in pos:
        wins += int((p > neg).sum())
        ties += int((p == neg).sum())
    # 赢 1 分、平 0.5 分、输 0 分，除以总场数 = 概率
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
