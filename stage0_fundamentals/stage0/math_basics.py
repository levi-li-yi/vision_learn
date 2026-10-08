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

    顺序规则（没有交换律，也不存在"以维度大的为准"）：
      谁在前谁出"行数"、谁在后谁出"列数"，中间维必须咬合（相等且被消耗）：
        (2,3) @ (3,4) -> (2,4) ✓；  (3,4) @ (2,3) -> 中间 4!=2 直接报错
      数值同样随顺序变（还是上面这对 2x2）：
        a @ b = [[19, 22],   而   b @ a = [[23, 34],
                 [43, 50]]                [31, 46]]     两者完全不同
      直觉：矩阵 = 变换（旋转/平移/缩放），动作有先后——先旋转再平移
      不等于先平移再旋转（先穿袜子再穿鞋）。仓库配准矩阵的相乘顺序
      也是精心排的，排错坐标系就搬歪。
      唯一例外：单位矩阵 I（对角线全 1、其余 0），a @ I = I @ a = a
      ——"什么都不做"的变换，矩阵世界的 1。

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
    """手写欧氏距离：sqrt(sum_i (u_i - v_i)^2)——两列数差多少的"直线距离"。

    勾股定理的高维版，但务必带着"先求差"这一步看（差值是直角三角形
    的两条直角边）：
      例: u=[1,2], v=[4,6]
        第1步 逐维求差: u-v = [-3, -4]
        第2步 平方再求和: 9+16 = 25        <- 负差被平方翻正，方向不重要
        第3步 开方: sqrt(25) = 5.0          <- 两点间直线距离
      384 维同理：384 维各自求差平方，求和开方。
    （教学注意：别拿"到原点"举例——减 [0,0] 等于没减，容易让人忘了
      距离的定义里本来就有一步求差。）
    工程连接：无监督线异常分数 = teacher 与 student 384 维特征"距离平方的
    平均"（本函数去掉 sqrt 的版本）。距离大 = 模仿不上 = 异常——整条
    无监督线就建立在这一个数的大小差异上。
    """
    # np.asarray(数据, dtype=类型)：统一升 float64 再运算（u 可能是列表；
    # 高精度小数让 384 维累加的浮点误差最小）。
    # 逐元素减：[1,2] - [4,6] -> [-3, -4]（差的正负无所谓，下一步平方翻正）
    diff = np.asarray(u, dtype=np.float64) - np.asarray(v, dtype=np.float64)
    # diff ** 2：逐元素平方 [-3,-4] -> [9,16]（负差翻正，差距不论方向）
    # np.sum(表)：全压成一个数（9+16=25）
    # np.sqrt(数或表)：开方。例: np.sqrt(np.array([9., 25.])) -> array([3., 5.])
    return float(np.sqrt(np.sum(diff ** 2)))


def squared_channel_distance(teacher_patch: np.ndarray, student_patch: np.ndarray) -> np.ndarray:
    """迷你版异常图：逐位置算"模仿差距"，输出 (H,W) 差距热图——一场自动找茬。

    剧情像"大家来找茬"：teacher = 标准答案图（冻结不动），student = 学生的
    临摹（正常处摹得像，"看不懂"处会摹崩）。逐位置比较两张图，差得多的
    地方标出来——student 摹崩处 = 异常。

    每个位置三步，全是小学算术（设该位置 teacher 说内容是 [10, 20, 30]）：
      情况一 student 摹得像 [10, 21, 30]：
        相减 [0,-1,0] -> 平方 [0,1,0] -> 平均 0.33    低分 = 正常 ✓
      情况二 student 摹崩了 [40, 20, 90]：
        相减 [-30,0,-60] -> 平方 [900,0,3600] -> 平均 1500   爆表 = 异常 ✗
      0.33 vs 1500——不用设门槛，一眼看出第二处出事。

    三个为什么：
      平方：差 -1 和 +1 都是"差 1"，不平方会互相抵消；顺带小差缩小、
            大差急剧放大（1²=1 vs 30²=900），摹崩处格外显眼；
      平均：每位置 C 个数（迷你例 C=3，仓库 C=384）压成 1 个分 -> 整张图
            变一张 (H,W) 分数地图：正常暗、异常亮。shape (H,W,C) -> (H,W)；
      不开方：sqrt 不改变大小顺序（1500>0.33 开方后仍大），阈值/AUC
            结论不变，省一次计算。
    热图下一站：map_quantiles 做 90%/99.5% 分位标定。
    对照仓库 utils.py::predict: mean((teacher - student)^2, dim=通道)。
    """
    # 第1步 相减：广播逐元素减；.astype(np.float32) 先升小数（同 safe_brightness 首步）
    #   例: [10,20,30] - [40,20,90] -> [-30, 0, -60]
    diff = teacher_patch.astype(np.float32) - student_patch.astype(np.float32)
    # 第2、3步 平方 + 平均：** 2 逐元素平方（[-30,0,-60] -> [900,0,3600]）；
    # np.mean(..., axis=-1) 压掉最后一根轴 = 通道（axis=-1 = 倒数第一维，
    # 编号规则见 numpy_basics.channel_means 注释）-> 每位置一个分 (H,W)
    return np.mean(diff ** 2, axis=-1)


# ---------------------------------------------------------------------------
# 2. 概率统计
# ---------------------------------------------------------------------------

def mean_std_by_sum_sq(x: np.ndarray):
    """用"两本流水账"一步算出均值和标准差——仓库 teacher_normalization 同款。

    比喻：班主任管流动班级——学生报完分数就走（不留档），老师只记两本账：
    总分、分数的平方和。期末不记得任何人，平均分（mean）和班级波动（std）
    照样算得出来。仓库统计 teacher 输出（成千上万批特征图流过即扔）正是
    这套记账法，不必存全量数据——这也是为什么不直接 x.std()（它要摊开
    全部卷子才能算）。

    小学算术验证（分数 [1, 2, 3]）：
      第一本账: sum = 1+2+3 = 6, n = 3        -> mean = 6/3 = 2
      第二本账: 1²+2²+3² = 14                 -> 平方的平均 = 14/3 ≈ 4.667
      方差 = 4.667 - 2² = 0.667               -> std = sqrt(0.667) ≈ 0.816
      （按课本定义验算: (1-2)²+(2-2)²+(3-2)² 平均 = 2/3 = 0.667 ✓）

    恒等式：方差 = 平方的平均 - 平均的平方——"先平方再平均"比"先平均再
    平方"多出来的部分恰好是波动；所有值相同时两者相等，方差为 0。

    陷阱：浮点数只有 ~15-16 位有效数字，两个几乎相等的数相减可能算出
    "-0.000...1" 的负方差 -> 开方得 NaN -> 一处 NaN 传染整条打分链。
    max(var, 0) 钳制与仓库 torch.sqrt(torch.relu(var)) 同款防御。
    例: 全 3.14 的数组 -> std 残渣 < 1e-6（非精确 0，但被钳住不为负）。
    """
    # .astype(np.float64)：升双精度，几万个数累加时浮点误差最小
    x = x.astype(np.float64)
    # x.size：元素总个数（属性），例: [1,2,3].size -> 3；x.sum()：全压成一个数
    mean = x.sum() / x.size              # 第一本账 -> 平均分：6/3 = 2
    mean_sq = (x ** 2).sum() / x.size    # 第二本账 -> 平方的平均：14/3 ≈ 4.667
    # 方差 = E[x^2] - mean^2 = 4.667-4 = 0.667；max(..., 0.0) 钳负（防浮点残渣出 NaN）
    var = max(mean_sq - mean ** 2, 0.0)
    # np.sqrt 开方 -> std ≈ 0.816（波动幅度）
    return float(mean), float(np.sqrt(var))


def kth_smallest(x: np.ndarray, q: float) -> float:
    """按分位比例取"第 k 小"——torch.kthvalue 手写版，无监督线的命脉。

    比喻：n 个人按个子从矮到高排成一队，喊"第 k 个出列"。
    分位比例换名次：k = int(n * q)（直接砍小数，与仓库一致，非四舍五入），
    再钳位到 [1, n]——k=1 是最矮、k=n 是最高。

    小学算术（x = [0..9] 共 10 个，已排队 0,1,2,...,9）：
      kth_smallest(x, 0.9) : k = int(10*0.9) = 9 -> 第 9 小 = 8.0
      kth_smallest(x, 0.5) : k = 5               -> 第 5 小 = 4.0
      kth_smallest(x, 0.0) : k=0 被钳到 1        -> 最小值 0.0
    业务含义：q=0.9 的结果 = "90% 的数都不超过它"（90 分位线）。

    为什么不用 np.quantile：库版默认在两数之间"插值"（第 9.5 名取第 9、10
    名的中间值），返回一个可能不存在的数；仓库要"实实在在排在那的第 k 名"
    （kthvalue 语义）——标定阈值必须是真实出现过的差距值。

    仓库两处命脉：
      训练损失取前 0.1% 最难像素（trainer.py, k = int(numel*0.999)）；
      异常图 90%/99.5% 分位标定（_map_quantiles 同款钳位）。
    """
    # x.size：元素总个数（属性）
    n = x.size
    # int(n * q)：分位比例 -> 名次（砍小数）
    k = int(n * q)
    # min/max 双向钳位：k=0 -> 提到 1；乘出界 -> 压回 n
    k = min(max(k, 1), n)
    # np.sort(表)：升序排队（原表不动，返回新队）。例: np.sort([3,1,2]) -> [1,2,3]
    # .ravel()：拉平成一维（几维的表都铺成一条队）。例: [[1,2],[3,4]] -> [1,2,3,4]
    sorted_asc = np.sort(np.asarray(x, dtype=np.float64).ravel())
    # 排队后取"第 k 个"：下标从 0 数，第 k 个 = 下标 k-1（第 5 个下标是 4）
    return float(sorted_asc[k - 1])


def map_quantiles(anomaly_map: np.ndarray):
    """迷你版分位标定：返回异常图的 90% / 99.5% 分位——恰好 2 个数。

    return a, b 语法上返回一个元组 (q_start, q_end)（"多个数打包"），
    调用端按形状解包：
        q_start, q_end = map_quantiles(amap)     # 一对一接住
    （演示 8x8 异常图的真实输出示例：q_start=0.0049, q_end=0.0102）

    两个数的业务身份 = 把异常图换成"标准分"那把尺子的两道刻度：
      q_start（90% 分位）= "日常水位"——90% 的正常像素都不超过它；
      q_end  （99.5% 分位）= "极端水位"——99.5% 都在其下，几乎到顶。
    用途（仓库 predict）：score = (x - q_start) / (q_end - q_start)——
    比 q_start 高 = 比日常离谱，比 q_end 高 = 比极端还离谱；两道刻度一立，
    任何原始差距都能换算成"离谱多少倍"的标准分，跨模型/跨点位可比。
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
    """直接"编"出两批异常分数，代替稀缺的真实坏品——给 auc_by_hand 出考卷。

    在故事里的位置（出卷人）：模型给每张图打一个"异常分"（找茬热图的
    汇总），好品分低、坏品分高。要考模型（算 AUC）需要一批已知好/坏的
    分数，但真实坏品稀缺（这正是无监督线上线的理由）——于是跳过图片，
    直接按两边的分数"长什么样"各编一批：
      好品: normal(0.2, 0.05) -> 1000 个挤在 0.1~0.3 的低分（差距小而齐）
      坏品: normal(0.7, 0.12) -> 200 个散在 0.46~0.94 的高分（差距大而乱）
    两批基本不重叠 -> 直方图上两座"钟"分开站 -> AUC 接近 1。

    正态分布大白话：一种"中间多、两头少"的撒数规律（如身高集中在平均
    附近）。两个旋钮：均值 = 钟的中心（数堆在哪）；标准差 = 钟的宽窄
    （数散得多开，约 95% 落在 均值 ± 2 倍标准差 内）。

    seed 种子：同一台随机器 + 同一种子 -> 永远编出同一批数（可复现，
    本工程全部测试的确定性都靠它）；换种子 = 换一套卷子。
    """
    # np.random.default_rng(种子)：造一台"随机器"。
    #   例: default_rng(42).normal(0, 1, 2) 每次运行结果一模一样
    rng = np.random.default_rng(seed)
    # rng.normal(均值, 标准差, size=个数)：从正态分布撒 size 个数。
    #   例: normal(0.2, 0.05, 5) -> 约 [0.19, 0.24, 0.15, 0.22, 0.26]
    # .clip(0, None)：单侧夹紧（下限 0、上限不限）——分数不能为负。
    ok_scores = rng.normal(0.2, 0.05, size=n_ok).clip(0, None)   # 好品：又矮又窄的低分钟
    ng_scores = rng.normal(0.7, 0.12, size=n_ng).clip(0, None)   # 坏品：散得开的高分钟
    # 返回 2 个数组（return a, b = 元组打包）。下一站喂给 auc_by_hand：
    # 1000 x 200 = 20 万场擂台，两钟不重叠 -> 坏品几乎场场赢 -> AUC≈1
    return ok_scores, ng_scores


def auc_by_hand(labels: np.ndarray, scores: np.ndarray) -> float:
    """手写 AUC：坏品好品两两配对打擂台，坏品分数更高的场数占比。

    为什么需要它：评价模型打分不能只看坏品考多少（0.8 算高吗？要看好品
    都考多少才知道）——分数的意义来自对比。所以不设及格线、考排序：
    随机抽一坏一好亮分，坏品更高 = 这场模型答对。穷举全部组合
    （x 个坏 x y 个好 = x*y 场）就是打擂台。

    最小考卷全程手算：
      好 [0.1, 0.3] 坏 [0.8]（共 1x2 = 2 场）：
        0.8 vs 0.1 赢 / 0.8 vs 0.3 赢 -> (2+0)/2 = 1.0   两队完全分开
      好 [0.2, 0.8] 坏 [0.5]（一输一赢）：
        0.5 vs 0.2 赢 / 0.5 vs 0.8 输  -> 1/2 = 0.5      瞎猜水平
    三个锚点：1.0 = 完美排序 / 0.5 = 瞎猜（两队混作一团，抽一对如抛
    硬币）/ 0.0 = 全反着判。平局（分数相等）算半对记 0.5。

    对应仓库环节——无监督线训练结束后的"期末考试"（utils.py::test +
    trainer 的 AUC 门控保存），本函数的 labels/scores 就是那场考试的产物：
      1. 组考卷：合成 NG（Cut-Paste 与 OK 1:1 成对）或真实 NG 切片
         （fetch_ng_by_qr 从生产库拉图入池）——labels 即 0=OK / 1=NG；
      2. 打分：每张图过 teacher/student -> 找茬热图 -> 高斯模糊 ->
         region 门控 -> 最热 TopK100 像素平均 = 图像级异常分（scores）；
      3. 算 AUC（仓库用 sklearn.roc_auc_score，本函数是其手写教学版）->
         ComparativeSaveManager 门控保存：新模型 AUC 赢过历史最佳，
         才删旧 pth 写新部署权重——部署盘上的模型只会越来越好；
      4. 同场考试还顺带按 PR-F1 选判 NG 的阈值（find_optimal_threshold），
         逐桶再各选各的（bucket_metrics）。
    """
    # 拆两队（布尔索引：对/错表当筛子挑行）。
    #   例: labels = [0, 1, 0], scores = [0.1, 0.8, 0.3]
    #       labels == 1         -> [False, True, False]
    #       scores[labels == 1] -> [0.8]（坏品队 pos，label 1 = NG）
    #       scores[labels == 0] -> [0.1, 0.3]（好品队 neg）
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = scores[labels == 1], scores[labels == 0]

    # 每个坏品轮流上台，向全部好品同时叫阵（向量化，无内层循环）：
    #   (p > neg)：这个坏品对所有好品逐场判胜负
    #   .sum()：True 记 1 相加 = 赢的场数
    #   例: p=0.8, neg=[0.1, 0.3] -> [True, True] -> 2 胜
    wins = ties = 0
    for p in pos:
        wins += int((p > neg).sum())
        ties += int((p == neg).sum())
    # (赢场 x 1 + 平局 x 0.5) / 总场数 pos.size x neg.size = 答对概率
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
