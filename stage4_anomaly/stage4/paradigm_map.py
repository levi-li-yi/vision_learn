"""课时 1：无监督异常检测范式地图——四大门派一文件走完。

对应提纲：重建类 AE -> 教师学生类 -> 记忆库类 PatchCore -> 合成类
Cut-Paste/DRAEM。本模块在同一个合成 2D 特征数据集上各跑一遍"最小实现"，
用同一个 roc_auc 打分——四大范式在 10 分钟内可比。

特征设定（可可视化想象）：正常样本聚成一团；异常样本偏离团心。
四种打分方式：
  AE：只学重建正常团 -> 重建误差大 = 异常；
  Teacher-Student：teacher 固定、student 学抄 -> 抄不像 = 异常
                   （本工程路线，细节在 demo_l）；
  记忆库(PatchCore mini)：存正常样本特征，新样本到最近邻的距离
                   = 异常分；
  合成异常(CutPaste 特征版)：正常特征里"挖一块换别处"造伪异常训练
                   二分类（本工程 test 考卷的思想源头）。

运行：python -m stage4.paradigm_map
"""
import torch
from torch import nn


def make_features(n_ok=400, n_ng=60, dim=8, seed=0):
    """正常特征聚团，异常特征离群（模拟深度特征的几何形状）。"""
    g = torch.Generator().manual_seed(seed)
    ok = torch.randn(n_ok, dim, generator=g) * 0.5
    direction = torch.randn(dim, generator=g)
    direction = direction / direction.norm()
    ng = ok[:n_ng] + direction * 4.0 + torch.randn(n_ng, dim, generator=g) * 0.3
    return ok, ng


def roc_auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """Mann-Whitney 版 AUC：随机抽(正,负)对，正分更高的比例（并列记 0.5）。"""
    pos = scores[labels == 1]; neg = scores[labels == 0]
    gt = (pos[:, None] > neg[None, :]).float().sum()
    eq = (pos[:, None] == neg[None, :]).float().sum()
    return ((gt + 0.5 * eq) / (len(pos) * len(neg))).item()


# ---------------------------------------------------------------------------
# 范式一：重建（最小自编码器）
# ---------------------------------------------------------------------------

class TinyAE(nn.Module):
    def __init__(self, dim=8, hidden=2):
        super().__init__()
        self.enc = nn.Linear(dim, hidden)
        self.dec = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.dec(torch.tanh(self.enc(x)))


def score_autoencoder(ok_train, all_x, epochs=300, seed=0):
    torch.manual_seed(seed)
    ae = TinyAE(ok_train.shape[1])
    opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
    for _ in range(epochs):
        loss = ((ae(ok_train) - ok_train) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return ((ae(all_x) - all_x) ** 2).sum(1)


# ---------------------------------------------------------------------------
# 范式二：教师-学生（本工程路线的特征版）
# ---------------------------------------------------------------------------

def score_teacher_student(ok_train, all_x, epochs=300, seed=0):
    torch.manual_seed(seed)
    teacher = nn.Linear(ok_train.shape[1], 16); teacher.requires_grad_(False)
    student = nn.Linear(ok_train.shape[1], 16)
    opt = torch.optim.Adam(student.parameters(), lr=1e-2)
    with torch.no_grad():
        t_ok = teacher(ok_train)
    for _ in range(epochs):
        loss = ((student(ok_train) - t_ok) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return ((student(all_x) - teacher(all_x)) ** 2).sum(1)


# ---------------------------------------------------------------------------
# 范式三：记忆库（PatchCore mini——最近邻距离）
# ---------------------------------------------------------------------------

def score_memory_bank(bank: torch.Tensor, all_x: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(all_x, bank)                        # [N, bank]
    return d.min(1).values                              # 到最近正常样本的距离


# ---------------------------------------------------------------------------
# 范式四：合成异常（CutPaste 特征版——造伪异常训二分类）
# ---------------------------------------------------------------------------

def score_cutpaste(ok_train, all_x, epochs=200, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    n, dim = ok_train.shape
    # 特征版"剪贴"：随机抽 30% 维度，把该维度值替换成另一样本的
    # （再乘 1.5 推离流形）——图像空间的"剪一块贴别处"在特征空间的对应
    fake_ng = ok_train.clone()
    for i in range(n):
        j = int(torch.randint(0, n, (1,), generator=g))
        dims = torch.randperm(dim, generator=g)[:dim // 3]
        fake_ng[i, dims] = ok_train[j, dims] * 1.5
    clf = nn.Linear(dim, 2)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
    x = torch.cat([ok_train, fake_ng]); y = torch.cat(
        [torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    for _ in range(epochs):
        loss = nn.functional.cross_entropy(clf(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return torch.softmax(clf(all_x), 1)[:, 1]       # 判为"异常"的概率


if __name__ == "__main__":
    ok, ng = make_features()
    all_x = torch.cat([ok, ng]); labels = torch.cat(
        [torch.zeros(len(ok)), torch.ones(len(ng))])
    scorers = {
        "重建 AE": score_autoencoder(ok, all_x),
        "师生 T-S": score_teacher_student(ok, all_x),
        "记忆库": score_memory_bank(ok[:200], all_x),
        "合成 CutPaste": score_cutpaste(ok, all_x),
    }
    print(f"{'范式':<14} AUC")
    for name, s in scorers.items():
        print(f"{name:<14} {roc_auc(labels, s):.3f}")
    print("\n四条路殊途同归：都以'好件训练、偏离即异常'为纲；")
    print("本工程无监督线 = 师生范式（像素级距离图）+ 本仓库的标定/桶机制。")
