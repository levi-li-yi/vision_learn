"""课时 2：nn.Module / Dataset / DataLoader——主仓库三件套的微缩版。
这个文件教你 PyTorch 的三件套怎么用：模型(nn.Module)、数据集(Dataset)、搬运工(DataLoader)——这是所有深度学习代码的骨架。


对应主仓库：
  - core/seg.py 的 EfficientViTSeg(nn.Module)——forward 输入输出张量；
  - data_utils/data.py 的 FixtureSupervisedDataset(Dataset)——__len__/__getitem__；
  - train_entry 的 DataLoader(num_workers=4, collate_fn=allocate)——搬运工。

合成任务：给 16x16 的单通道图分类"亮块在左上还是在右下"（数据即答案，
无需下载；真实数据集在 Demo C/D）。

运行：python -m stage2.nn_module_dataset_loader
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# 1. 模型：两段卷积 + 池化 + 全连接（麻雀虽小，五脏俱全）
# ---------------------------------------------------------------------------

class TinyCNN(nn.Module):
    """对应主仓库三段式的影子：层在 __init__ 声明，forward 只写数据流。

    任务：判断 16x16 图里的亮块在左上还是右下。整个模型是一条算数流水线：
    __init__ 只登记零件（此时还没有任何图片流过），forward 写数据怎么流。

    层数/通道怎么定（真实业务同三原则）：①缩几次由输入尺寸定——16→8→4，
    缩到够小再摊平打分；②深度宽度由任务复杂度定——此任务 2 组、4→8 通道
    即够，真实网络图每缩半通道翻倍（64→128→256→512）；③上限由数据量与
    算力定。真实业务几乎不从零设计：拿成熟骨干（ResNet/EfficientViT 等）
    换头 + 迁移学习（见 Demo D）。另：Sequential 只能表达直线流水线，
    带残差跳线的网络（如 ResNet）要手写 forward。

    参数量 594 的来历（卷积旋钮=探测器数×通道×3x3 窗面积，打分器=128x2 权重
    表，各输出再配 1 个偏置；ReLU/池化是固定规则、0 参数）：卷积① 4x1x9+4=40，
    卷积② 8x4x9+8=296，打分器 128x2+2=258，合计 40+296+258=594
    （named_parameters 共 6 个张量，sum(p.numel()) 数的就是它们）。
    """

    def __init__(self):
        super().__init__()  # 先让父类把参数登记簿初始化好（固定写法，不写报错）
        # Sequential 接收任意个参数：每个参数是一个已造好的层对象，按传入顺序
        # 串成流水线；调用时数据挨个流过，最后一个的输出就是总输出。
        self.features = nn.Sequential(
            # 卷积盒=3x3 小窗口扫全图找"哪里亮"（括号数依次：进 1 张灰度图、
            # 4 个探测器、窗口 3x3、补一圈 0 保尺寸）；ReLU 盒=负数归零；
            # 池化盒=每 2x2 格只留最大数，图缩半。
            nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16->8
            # 第二遍同款三连：探测器 4->8，在上一遍的发现上找更复杂的图案，再缩半
            nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 8->4
        )
        # Linear 两副面孔：nn.Linear(128,2) 造出"打分器对象"（不是数字！）——肚里是
        # 128x2 权重表+2 偏置，初始全随机；真正算分是 forward 里调 self.head(清单)：
        # 每类一套权重做加权求和 -> [B,2]。训练的意义=把随机表调成会偏向正确答案。
        self.head = nn.Linear(8 * 4 * 4, 2)

    def forward(self, x):                # x: [B,1,16,16] -> logits: [B,2]
        # 一行三步从里往外读：①过流水线得 8 张 4x4 痕迹图；②flatten(1) 每张摊成
        # 16 个数共 128（打分器只认清单；从第 1 维摊，"第几张图"绝不搅混）；③打分。
        return self.head(self.features(x).flatten(1))


# ---------------------------------------------------------------------------
# 2. 数据集：合成"亮块位置"二分类
# ---------------------------------------------------------------------------

class BrightCornerDataset(Dataset):
    """对应主仓库 Dataset 契约：__len__ 报课表行数，__getitem__ 出一份样本。

    合成数据的"印刷厂"：图不是照片，是代码现印的——先定答案再照着画（数据即答案）。
    继承 Dataset = 签上岗合同：会报库存 + 会出样本，DataLoader 就能雇佣它；
    Demo C 换真数据集时只需新写一个同样合同的类，下游 DataLoader/训练循环不用改。
    """

    def __init__(self, n=256, size=16, seed=0):
        # 种子钉死在 seed 上：每次运行的"随机"数都一样，图可复现、测试可对拍
        g = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, 2, (n,), generator=g)  # 先抽答案：扔 n 次硬币，各得 0/1
        self.noise = torch.rand(n, 1, size, size, generator=g) * 0.3  # 再印底噪：n 张雪花屏压到 0~0.3
        self.size = size  # 记下边长，__getitem__ 切亮块时要用

    def __len__(self):
        return len(self.labels)  # 合同方法①：DataLoader 问"库存几张"就调它（此刻图还没造）

    def __getitem__(self, idx):
        # 合同方法②：取第 idx 张，图在这里现做
        img = self.noise[idx].clone()  # 抽底噪并复制：要在上面贴亮块，库存必须保持干净
        lab = int(self.labels[idx])    # 查这张图预定好的答案
        q = 0 if lab == 0 else self.size // 2          # 0=左上 1=右下；亮块起始坐标 0 或 8
        # 切片贴亮块：[通道, 行, 列] 中选 8x8 区域（q:q+8 含头不含尾，共 8 行/列），
        # 每像素 +0.7——灰底噪(0~0.3)上浮出亮块(0.7~1.0)
        img[:, q:q + self.size // 2, q:q + self.size // 2] += 0.7
        return img, lab  # 交货：(图, 答案) 一对，永远一致


def collate_stack(batch):
    """对应主仓库 allocate：把一批 (图,标签) 拼成两个批量张量。

    三件套的"打包员"：__getitem__ 一份份出餐（单图+单标签），模型一次要吃
    一整批（GPU 并行、梯度取批平均更稳），中间由它把散装样本码成规格化张量。
    batch 是列表 [(图1,标签1),...,(图B,标签B)]。DataLoader 其实自带默认打包，
    手写一份是为了看清机制（也对应主仓库自定义 allocate 的做法）。
    """
    imgs = torch.stack([b[0] for b in batch], 0)  # 取每对第 0 个(图)，8 张码成一摞 -> [B,1,16,16]
    labs = torch.tensor([b[1] for b in batch])    # 取每对第 1 个(标签)，8 个整数直接造张量 -> [B]
    return imgs, labs  # 顺序天然对齐：imgs[i] 必须配 labs[i]，弄乱 = 给图贴错答案


if __name__ == "__main__":  # 只在直接运行本文件时执行；被 import 复用类时不跑演示
    # 自检演示：三件套串起来通一次水，只验证"管道通、形状对"——训练是 train_loop.py 的事
    ds = BrightCornerDataset()  # 开印刷厂：备好 256 个答案+底噪，此刻还没造出一张成品图
    # batch_size=8：教学演示顺手挑的小 2 的幂，整除 256 只是整洁、非必需——
    # 不整除时 DataLoader 默认把最后一批凑小发出。真实项目里批大小是
    # "2 的幂习惯 + 显存上限 + 梯度稳定性"的权衡，还可用梯度累积凑等效大批。
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_stack)
    model = TinyCNN()  # 装配模型：594 个参数全随机 = 瞎猜状态，还没见过任何数据
    # 取第一批（一行三件事）：iter=搬运工开始发货；next=随机挑 8 个下标→
    # 挨个 __getitem__ 现做 8 对样本→collate_stack 打包；再拆成 imgs/labs
    imgs, labs = next(iter(loader))
    # model(imgs) 触发 forward：[8,1,16,16]→流水线[8,8,4,4]→摊平[8,128]→打分[8,2]；
    # 此刻分数是随机权重算出来的、毫无意义——这行只验证形状管道通不通
    print("batch:", imgs.shape, labs.shape, "| 前向输出:", model(imgs).shape)
    print("参数量:", sum(p.numel() for p in model.parameters()))  # 数旋钮：40+296+258=594
