"""★Demo C：CIFAR-10 完整训练脚本（不用任何高层框架）。
这个文件是阶段 2 的毕业作品：在 CIFAR-10 图片分类任务上写一个完整的训练脚本——热身调度、梯度裁剪、混合精度、断点续训全都有，不用任何高级框架，纯 PyTorch 从零写。


要求清单（提纲原文）：warmup+余弦调度、checkpoint 保存/续训、梯度裁剪——
再加本仓库同款：bf16 autocast、梯度累积、两组学习率（backbone 前缀可开关）。

两种运行模式：
  smoke（默认） : torchvision.FakeData 合成图，CPU 数秒跑完——验证骨架正确；
  real          : --data <目录> 真实 CIFAR-10（自动下载）。

运行：
  python -m stage2.demo_c_cifar                       # smoke 自检
  python -m stage2.demo_c_cifar --data ./cifar_data --epochs 30
通关练习：把 smoke 换成 real 跑到 test acc >= 85%（ResNet 结构或更长轮次）。
"""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .optim_scheduler import build_scheduler
from .train_loop import train_one_epoch, accuracy


# ---------------------------------------------------------------------------
# 模型：小 ResNet 风格 CNN（CIFAR 尺寸 32x32）
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """一块"原图+补丁"的改图积木：进一张图，出"原图+补丁"。公式只有一个——

        出图 = 原图 + 补丁          (forward 里那个 + 号)

    - 补丁(self.net 算)：哪里改、改多少——车间真正学的东西；没什么好学的
      就全打 0，出图=原图原样放行，零成本；
    - 原图(self.short 递)：进来的图带到出口。
    为什么不学"一张新图"只学"补丁"？——"不改"必须容易：补丁打空白人人都会；
    若每块都被迫从零画新图，想画得和原图一模一样反而极难。深网络能一层层
    叠起来不学崩，靠的就是这招(残差=补丁，ResNet 的核心思想)。
    cin/cout：进/出的"图层数"(彩色图=3 层纸叠的；网络里逐块变 32/64/128 层)。"""

    def __init__(self, cin, cout):
        super().__init__()
        # 算补丁的三步：①卷积=拿 3x3 放大镜扫图找图案；②BatchNorm=过秤，把数字
        # 拉回正常范围防越算越歪；③ReLU=负数归零。(bias=False：卷积自带的起步数
        # 与过秤功能重复，省掉)
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(cout), nn.ReLU())
        # 递原图的通道：层数相同 -> Identity=直通管，原样递过；
        # 层数不同(如 3 层原图 + 32 层补丁，层数对不上没法相加) ->
        # 1x1 卷积=调色盘：把 cin 层按不同比例调成 cout 层，对上号才能加
        self.short = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.short(x)  # 出图 = 原图 + 补丁(两张图逐位置相加)


class SmallResNet(nn.Module):
    """整栋楼(105,322 个旋钮；TinyCNN 594 的"小高楼"版)：三层 ConvBlock 楼层+楼梯
    +收尾总结台拼成骨干，再配打分器。套路与 TinyCNN 一致(骨干提特征+打分器出分)，
    零件全升级。属性名 backbone=起姓：分组学习率(老师傅小步/学徒大步)按姓认骨干筐。
    数据流：[B,3,32,32] -骨干-> [B,128,1,1] -摊平-> [B,128] -head-> [B,10]。
    nc=给几个类打分(默认 CIFAR-10 十类；换数据集只改这一个数)。"""

    def __init__(self, nc=10):
        super().__init__()  # 老规矩：先初始化父类登记簿
        # 骨干 = 三层楼 + 两部楼梯 + 收尾总结台：
        #   一楼 ConvBlock(3,32)：彩色图 3 层进、32 张"发现图"出(找基础图案=认笔画)；
        #   楼梯 MaxPool2d(2)：每 2x2 只留最大，图缩半 32->16(省算力、保"在哪片区域")；
        #   二楼(32->64)：在一楼发现上组合更复杂图案(=认偏旁)；楼梯 16->8；
        #   三楼(64->128)：128 个探测器认抽象部件(=认整字)；
        #   总结台 AdaptiveAvgPool2d(1)：每层求平均，128 层 -> 128 个数——位置细节
        #   不要只要总量(分类知道"有轮子有眼睛"就够)；"自适应"=多大的图都吃
        self.backbone = nn.Sequential(
            ConvBlock(3, 32), nn.MaxPool2d(2),      # 32->16
            ConvBlock(32, 64), nn.MaxPool2d(2),     # 16->8
            ConvBlock(64, 128), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, nc)  # 打分器(学徒)：128 个数 -> nc 个分，分高者胜

    def forward(self, x):
        # 一行从里往外读：过三层楼 -> flatten(1) 摊平(128x1x1 方块堆压成 128 清单，
        # 第 0 维"第几张图"不动) -> 打分
        return self.head(self.backbone(x).flatten(1))


# ---------------------------------------------------------------------------
# 数据：smoke 用 FakeData（无需下载、可复现）；real 用 CIFAR-10
# ---------------------------------------------------------------------------

def make_loaders(batch_size=64, smoke=True, data_dir=None):
    """备料员：把"题库+考卷"两个搬运工一次备齐。真假双仓共用同一出口——冒烟用
    FakeData 假图空转验骨架(新车先怠速热车)，真跑用 CIFAR-10(自动下载)，下游一行
    不改(Dataset 合同兑现"换供料仓不动下游")。训练题和考卷必须不同图：考背过的
    原题分数虚高。入参：batch_size=每批几张；smoke=冒烟开关；data_dir=真数据目录。
    返回(训练工, 考官工)一对。"""
    import torchvision                     # 图片工具箱：现成数据集+加工工序
    import torchvision.transforms as T     # T=加工工序的缩写别名
    # 训练图加工流水线(Compose=把工序串成一条线)：①ToTensor 图片变张量、像素 0~255
    # 压成 0~1；②Normalize 每通道减 0.5 再除 0.5 -> 压到 -1~1((0.5,)*3 即复制三份
    # (0.5,0.5,0.5)，红绿蓝各一套)——数字以 0 为中心两边对称，网络吃着舒服学得稳
    tf_train = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    tf_test = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])  # 考卷同款流水线——必须同样加工，否则考试时模型看到"陌生格式"
    if smoke:  # 冒烟假仓：512 张训练假题+128 张假考卷，随机噪声随机贴答案——机器空转验骨架
        train = torchvision.datasets.FakeData(512, (3, 32, 32), 10,
                                              transform=tf_train)
        test = torchvision.datasets.FakeData(128, (3, 32, 32), 10,
                                             transform=tf_test)
    else:  # 真仓：train=True 拿 5 万张训练分卷 / train=False 拿 1 万张考卷(两拨必须
        # 是不同的图)；download=True 目录没有就自动下载；取图时自动过 transform 流水线
        train = torchvision.datasets.CIFAR10(data_dir, train=True,
                                             download=True, transform=tf_train)
        test = torchvision.datasets.CIFAR10(data_dir, train=False,
                                            download=True, transform=tf_test)
    # 搬运工公共配置：num_workers=0 单线程搬运(教学演示不起多进程)；pin_memory 有
    # GPU 时数据放"锁页内存"(系统不挪动的专区，GPU 直取更快)
    kw = dict(num_workers=0, pin_memory=torch.cuda.is_available())
    # 雇两个搬运工：训练工打乱发货(每轮题序不同、防背顺序)；考官工按序发(考试可
    # 复现)；**kw 把配置字典原样拆包倒进参数
    return (DataLoader(train, batch_size, shuffle=True, **kw),
            DataLoader(test, batch_size, shuffle=False, **kw))


# ---------------------------------------------------------------------------
# 训练主流程：调度/裁剪/AMP/断点 全家桶
# ---------------------------------------------------------------------------

def run_training(epochs=2, smoke=True, data_dir=None, ckpt=None,
                 resume=False, accumulation_steps=1, lr=1e-3,
                 backbone_lr_scale=1.0, warmup_steps=50, device=None):
    """总装车间(主仓库 train_entry.run() 的单卡版)：前五课所有零件在此合体——
    模型/备料/两筐学习率/变速器/断点续训/bf16 混合精度/攒批/裁剪/每轮存档。
    入参：epochs=轮数；smoke/data_dir=真假仓；ckpt=存档路径；resume=续训开关；
    accumulation_steps=攒批 N；lr=基准步长；backbone_lr_scale=老师傅折扣(1.0=不打
    折)；warmup_steps=热身步数；device=None 则自动挑。返回最终考卷命中率。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # 挑设备：传了用传的，没传自动(有 N 卡 cuda 否则 cpu；A or B = A 空才用 B)
    train_loader, test_loader = make_loaders(smoke=smoke, data_dir=data_dir)  # 备料员开工，领回训练工+考官工
    model = SmallResNet().to(device)  # 造 10.5 万旋钮的楼并搬上设备(模型和数据必须同机才能一起算)

    # 分筐(课时 4 split_param_groups 的手写内联)：点名册翻两遍按 backbone 姓分两筐——
    # 骨干筐(老师傅)lr*scale 打折、头部筐(学徒)原价 lr
    groups = [
        {"params": [p for n, p in model.named_parameters()
                    if n.startswith("backbone")], "lr": lr * backbone_lr_scale},
        {"params": [p for n, p in model.named_parameters()
                    if not n.startswith("backbone")], "lr": lr},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)  # 调整师接管两筐；weight_decay=每步顺手把旋钮往 0 按一小下(防死记硬背的"减肥针")
    steps = epochs * len(train_loader)  # 总里程 = 轮数 x 每轮批数(冒烟 2x8=16；真数据 30x781≈两万多)
    # 变速器(复用课时 4)：total_steps÷N——变速器按"优化器步"计，攒批时 N 批才拧一次
    # (README 坑清单第一条现场)；warmup 取 min(.., steps//4) 防呆——热身不超总里程
    # 四分之一；底价=lr/10
    sch = build_scheduler(opt, total_steps=steps // accumulation_steps,
                          warmup_steps=min(warmup_steps, steps // 4), min_lr=lr / 10)
    start_epoch, start_step = 0, 0  # 起点默认从头开始，续训时下面改写

    ckpt_path = Path(ckpt) if ckpt else None  # 有给路径才转 Path，否则 None(不存档)
    if resume and ckpt_path and ckpt_path.exists():  # 三条件串联(开关开+有路径+文件真在)才续训；短路求值前假后不看
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # 开箱读档(load_five 同款动作)
        model.load_state_dict(state["weight"])   # 三件记忆装回：①旋钮值
        opt.load_state_dict(state["optimizer"])  # ②调整师账本
        sch.load_state_dict(state["scheduler"])  # ③变速器进度(没存 scaler：bf16 用不上，五件套按需裁剪)
        start_step = state["step"]               # 取出存档步数
        start_epoch = start_step // max(1, len(train_loader))  # 由总步数反推从第几轮续(÷每轮批数；max 防除零)
        print(f"[resume] 从 step={start_step} (epoch {start_epoch}) 续训")  # 打个招呼

    step = start_step  # 计步器从存档值起步
    for ep in range(start_epoch, epochs):  # 外层轮次循环：从续训轮跑到目标轮
        model.train()  # 挂营业牌——这次不是走形式！SmallResNet 有 BatchNorm 双面层，必须真切训练模式
        for i, (x, y) in enumerate(train_loader):  # 内层批循环；enumerate 没带 1、批号从 0 数(下面 +1 的伏笔)
            x, y = x.to(device), y.to(device)  # 这批的题和答案也搬上设备
            # "三元版 with"：GPU 开 bf16 混合精度区(重活自动半精度)，CPU 明确关掉。
            # 区域内一行三件事：猜 model(x) -> 罚 cross_entropy -> ÷N 备攒批。
            # 无 scaler——bf16 天生不溢出，放大器免了
            with torch.autocast(device_type=device, dtype=torch.bfloat16
                                ) if device == "cuda" else torch.autocast("cpu", enabled=False):
                loss = nn.functional.cross_entropy(model(x), y) / accumulation_steps
            loss.backward()  # 记账(梯度累加)
            if (i + 1) % accumulation_steps == 0:  # 攒满判断；批号从 0 数故 +1 凑"第几批"(与课时 3"从 1 数起 i%N"同一件事两种写法)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ①安检：梯度总长>1.0 就压到 1.0
                opt.step()       # ②拧旋钮
                sch.step()       # ③变速器上发条！(教学骨架省略的那步；主仓库 step->sched.step->zero_grad 原序)
                opt.zero_grad()  # ④清账本
                step += 1        # ⑤计步器+1(只在真拧时+1，与变速器同口径=都按"优化器步")
        acc = accuracy(model, test_loader)  # 每轮考官在考卷(非训练题)上考一场
        print(f"epoch {ep+1}/{epochs} step={step} lr={opt.param_groups[1]['lr']:.2e} "
              f"test_acc={acc:.2%}")  # 成绩单：第几轮/步数/当前步长([1]=head 筐)/考卷命中率
        if ckpt_path:  # 每轮存一次四件套档——训练中断最多损失一轮
            torch.save({"weight": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": sch.state_dict(),
                        "step": step}, ckpt_path)
            print(f"[ckpt] -> {ckpt_path}")
    return accuracy(model, test_loader)  # 全部轮数跑完，最后考一场交最终成绩(通关就看它冲没冲 85%)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="CIFAR-10 目录(给了即真实模式)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run_training(epochs=args.epochs, smoke=args.data is None,
                 data_dir=args.data, ckpt=args.ckpt, resume=args.resume)
