"""★Demo D：迁移学习——冻结 vs 差异化学习率，体会 backbone_lr_scale=0.1。
这个文件教你'迁移学习'：拿一个已经学过看图的预训练模型，只换最后的分类头，在小数据集上微调——体验'骨干小步、新头大步'的效果。


对应主仓库：fresh 模式加载 pretrained.pth.tar（563 键骨干）+ 头部随机初始化
+ AdamW 两组旋钮（骨干 lr×0.1）——"转学生底子只精修不重学"的用意。

本模块可测试的核心（离线）：
  split_transfer_groups()：预训练骨干组 ×scale + 新头组 ×1.0；
  freeze_backbone()：冻结的另一种做法（requires_grad=False，梯度不产生）。
真实练习（作业）：--dataset 取 torchvision Flowers102 或自备猫狗小数据集，
观察"冻结快但上限低 / 差异化慢热但上限高"。

运行：python -m stage2.demo_d_transfer
"""
import torch
from torch import nn


class PretrainedLikeBackbone(nn.Module):
    """替身演员：扮演"预训练骨干"这个角色(真实剧本是 ResNet18 那种在百万图上
    学过看图的大块头，主仓库 pretrained.pth.tar 有 563 个键；本课练的是迁移
    流程不是骨干本身，迷你假货替身够用)。名字带 backbone. 姓——分组小步/
    冻结全按姓找人。一人分饰两角(__main__ 里)：①当"捐献者"——把学过的知识
    (W 矩阵)塞进骨干、存成假预训练文件=人造预训练模型；②当"学生模板"——M
    继承它，load_pretrained_backbone 把捐献者骨干装进来(头保持随机)=迁移本身。
    核心设定：骨干=别人练好的旧知识、头=本任务新装随机起步——"转学生底子
    好、新课本从零学"，Demo D 全程就在讲怎么对待这两种旋钮。"""

    def __init__(self):
        super().__init__()  # 老规矩：先初始化父类登记簿
        # 骨干三步迷你流水线(参数名册自动带姓：backbone.1.weight 等)：
        # Flatten 摊平成 32 个数的清单 -> Linear(32,64) 32进64出打分器(骨干
        # 的本事全存这张 32x64 表里) -> ReLU 负数归零。共 2112 个旋钮
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(32, 64), nn.ReLU())
        self.head = nn.Linear(64, 2)          # 新头：随机初始化(64进2出二分类；130 个旋钮)

    def forward(self, x):
        return self.head(self.backbone(x))  # 数据流：32 个数 -> 骨干提炼成 64 个特征数 -> 新头打 2 个分


def load_pretrained_backbone(model: nn.Module) -> int:
    """器官移植主刀：把捐献者的骨干装进学生模型，头部保持随机("只穿旧底子、新头从零学")。
    关键认知——学生早就有同名键：学生是同款图纸(M 继承本类)造的同款模型，造出来那刻
    零件全齐、只是数值随机。键=零件名、形状=规格，由图纸(结构)决定；"学没学过"的
    差别只在格子里的数值。所以"装骨干"不是添新键，而是同名键对号入座地**覆盖数值**——
    这也是迁移学习铁律：想装谁的知识就得用谁的同款骨干结构(键对不上装不进)。
    返回命中键数。(注：现状 donor 是随机新造的，会把学生已装好的知识覆盖成随机数——
    忠实模拟主仓库应从预训练文件加载捐献者，待修)"""
    donor = PretrainedLikeBackbone()  # 现场造捐献者(真实场景=从硬盘加载预训练文件；现状随机，见 docstring 注)
    # 只挑 backbone. 姓的条目做成 {名字:数值}：只捐骨干不捐头——别人的头对本任务没用
    donor_keys = {k: v for k, v in donor.state_dict().items()
                  if k.startswith("backbone.")}
    target = model.state_dict()  # 取学生名册；里面是参数的"引用"——待会 copy_ 会直接改到模型身上
    hit = 0  # 命中计数器：成功装了几个键
    with torch.no_grad():  # 盖"不许记账"章：搬家不求导，省内存
        for k, v in donor_keys.items():
            # 配型核对(不是查"有没有"——同款图纸必然同名同形状；这是防图纸不一致的防御)：
            # 学生有同名键吗 + 形状一致吗(32x64 的表不能硬塞 64x32 的位置)；不过关静默跳过，绝不硬塞
            if k in target and target[k].shape == v.shape:
                target[k].copy_(v)  # 原地抄入(名册是引用=直接写进学生模型)——覆盖的是值，键从来都在
                hit += 1
    model.load_state_dict(target)  # 名册正式装回(copy_ 已实际生效，这步是走正式手续)
    return hit


def split_transfer_groups(model: nn.Module, lr=1e-3, backbone_lr_scale=0.1):
    """差异化学习率：转学生组小步、新头组大步（主仓库同款结构）。"""
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    return [{"params": bb, "lr": lr * backbone_lr_scale, "name": "backbone"},
            {"params": hd, "lr": lr, "name": "head"}]


def freeze_backbone(model: nn.Module) -> nn.Module:
    """冻结式：骨干参数完全不更新（对应'冻结与否两种方式'的前者）。"""
    for n, p in model.named_parameters():
        if n.startswith("backbone"):
            p.requires_grad_(False)
    return model


def train_two_ways(model_ctor, loader, epochs=5, lr=1e-3):
    """PK 台：让"冻结派 vs 差异化派"两种迁移方案在同样数据、同样起点下各训一遍比分。
    入参：model_ctor=造模型的工厂(传类不传实例——两位选手必须是独立的新模型)；
    loader=训练题；epochs/lr=轮数与步长。返回成绩单 {"freeze": x, "differential": y}。
    (诚实提醒：考卷=训练题本身=背题式考试，教学演示偷懒，真实对比作业用 Flowers102；
    实跑 1.0 vs 1.0 区分不出优劣——骨干被随机捐献者覆盖+背题+任务小，修好才能看出
    README 说的"冻结快但上限低 / 差异化慢热但上限高")"""
    results = {}  # 空成绩单，待会记两派分数
    for mode in ("freeze", "differential"):  # 两位选手依次上台，各走完整流程：造->装->配方案->训->考
        torch.manual_seed(0)  # 公平第一招：钉死种子——两派拿到一模一样的随机初始旋钮，成绩差异才能归因于方案
        model = model_ctor()  # 工厂开工：调类造全新实例
        load_pretrained_backbone(model)  # 装预训练骨干(现状坑：随机捐献者把 W 覆盖掉、两派其实都随机起步，待修)
        if mode == "freeze":  # 冻结派方案①：冰冻术——骨干旋钮焊死(requires_grad=False，梯度根本不产生)
            freeze_backbone(model)
            opt = torch.optim.AdamW(   # 方案②：调整师只收没冻住的(筛 requires_grad=True，实际只剩 head 130 个)，
                [p for p in model.parameters() if p.requires_grad], lr=lr)  # 单一 lr——只有一类学生
        else:  # 差异化派方案：分筐员分两筐——骨干 lr*0.1 小步精修、头部原价大步(骨干照常学，只是步子小)
            opt = torch.optim.AdamW(
                split_transfer_groups(model, lr=lr, backbone_lr_scale=0.1))
        model.train()  # 挂营业牌
        # 最简训练循环(对比全家桶：无裁剪/调度/攒批——PK 台控制变量，只比教学方案，其它从简)。
        # 每批两行：猜 model(x)->罚 cross_entropy；zero_grad->backward->step(擦黑板->记账->拧旋钮)
        for _ in range(epochs):
            for x, y in loader:
                loss = nn.functional.cross_entropy(model(x), y)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()  # 切考场模式
        hit = tot = 0  # 小本本：对几题 / 共几题
        with torch.no_grad():  # 不许记账：考试不求导
            # 内联版考官=课时 3 accuracy 的五步挤在两行：打分->挑大分座位号->对答案->数✓->记本
            for x, y in loader:
                hit += int((model(x).argmax(1) == y).sum()); tot += len(y)
        results[mode] = hit / tot  # 该选手的考分记入成绩单
    return results


if __name__ == "__main__":  # 直接运行才执行。剧本三幕：造题(答案由 W 决定)->人造预训练(塞 W 存档)->造学生(从档装 W)->开赛
    from torch.utils.data import DataLoader, TensorDataset  # TensorDataset=最简数据集容器：现成 x/y 张量对扔进去即合法 Dataset，免写 __len__/__getitem__

    torch.manual_seed(42)  # 钉死种子：W/x 全固定，谁跑结果都一样
    # 第一幕·造题("合金数据"=题目按知识配制)：W 是核心道具——32x64 随机矩阵，
    # 扮演"预训练学到的知识"；x 是 256 道题、每题 32 个数(匹配骨干进料口径)
    W = torch.randn(32, 64)
    x = torch.randn(256, 32)
    # 出题公式 = 骨干的前向计算公式：x@W 矩阵乘(每题 64 个中间数，恰是骨干 Linear
    # 的输出形状)->relu 归零(恰是骨干的 ReLU)->每题求和->分正负得 True/False 标签。
    # 答案完全由 W 决定，而"预训练骨干"就被塞入这个 W——出题人和预习课本同一本：
    # 谁身上有 W 谁就提前知道答案规律，预训练优势从数学上成立
    y = (x @ W).relu().sum(1) > 0
    loader = DataLoader(TensorDataset(x, y.long()), batch_size=32)  # 打包发货；y.long() 把 True/False 转 1/0 整数(cross_entropy 要整型标签的老合同)

    # 第二幕·人造"预训练模型"：把 W 的转置塞进捐献者骨干——出题用 x@W，而
    # nn.Linear 内部按 x@权重的转置 算(表"横着存"、题"竖着用")，塞 W.T 才等效。
    # 此刻捐献者"真的学过"答案规律；no_grad=改数值不求导
    donor = PretrainedLikeBackbone()
    with torch.no_grad():
        donor.backbone[1].weight.copy_(W.T)      # "预训练"真的学过这个映射
    torch.save({"weight": donor.state_dict()}, "./_pretrained_like.pth")  # 存成预训练文件=模仿"下载预训练权重"(跑完可删)

    # 第三幕·造学生：不直接拿替身当学生——真实流程是"先造随机模型、再从文件装
    # 知识"(主仓库 fresh 模式)，造模型的类和知识来源分开，剧情才像真的
    class M(PretrainedLikeBackbone):
        def __init__(self):
            super().__init__()  # 先按图纸造随机模型(零件全齐、值随机——"学生早就有键")
            sd = torch.load("./_pretrained_like.pth", weights_only=False)["weight"]  # 读预训练文件取名册(load_five 同款读法)
            self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(32, 64), nn.ReLU())  # 显式重建骨干(super 已造过；把"重造零件、再装知识"两步分开走给你看)
            # 从名册挑 backbone.1.weight/bias(筛子 if 'backbone.1' in k)；k.split('.')[-1]
            # 取名字最后一段(backbone.1.weight->weight)——Linear 的 load_state_dict 只认
            # 短名。装完=学生带着真 W 知识出厂(可惜进 train_two_ways 后被随机捐献者覆盖，待修)
            self.backbone[1].load_state_dict({k.split('.')[-1]: v for k, v in sd.items()
                                              if 'backbone.1' in k})

    print(train_two_ways(M, loader, epochs=5))  # 开赛：M 当工厂，两派各训 5 轮，打印成绩单
