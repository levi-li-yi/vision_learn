"""课时 4：AdamW 双组旋钮 + warmup→cosine 调度——trainer.__init__ 的复刻。
这个文件教你两件事：①为什么'骨干(老手)用小步、头部(新手)用大步'；②学习率为什么要'先热身再缓降'而不是一直全速。


对应主仓库 trainer.py 39-88 行：
  - 分组：名字以 "backbone" 开头进骨干组（lr×backbone_lr_scale），其余进 head 组；
  - 断言：backbone_lr >= min_lr，否则当场报错（信息里写明三种修法）；
  - 调度：LinearLR(0.1→1.0, warmup_steps) --SequentialLR--> CosineAnnealingLR(min_lr)。
本模块把参数名前缀参数化（主仓库前缀是 image_encoder.）。

运行：python -m stage2.optim_scheduler
"""
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def split_param_groups(model: nn.Module, lr: float, backbone_lr_scale: float,
                       backbone_prefix: str = "backbone"):
    """分筐员：把旋钮按"资历"分成两筐、各配不同步长——差异化学习率的全部秘密。
    背景(迁移学习)：骨干=别人预训练好的"老师傅"旋钮(已会认边/纹理，只能小步微调，
    配大步会把功力拧乱=灾难性遗忘)；头部=本任务新装的"学徒"旋钮(全随机，要大步快学)。
    入参：model=待分筐模型；lr=基准步长(学徒筐原价)；backbone_lr_scale=老师傅折扣
    (0.1=打一折)；backbone_prefix=老师傅"姓氏"——名字以此开头算骨干(主仓库实际是
    "image_encoder."，此处参数化可传)。
    实际工程怎么定姓氏(姓氏由"最后写模型的人"决定)：
    ①自己套薄壳(DemoNet/主仓库模式)：self.image_encoder=骨干(...) + self.head=..
    这两个属性名就是命名动作，骨干内部几百层自动冠 image_encoder. 前缀——> 传
    backbone_prefix="image_encoder"，按骨干姓正向挑(本函数的写法)；
    ②直接用库模型(resnet50(num_classes=2))：头部姓氏是库作者起的(fc/classifier/
    head 三种常见)，骨干反而没统一姓氏——> 本函数照镜子用：hd=startswith("fc")
    按头部姓挑、其余全进骨干筐。
    接手任何模型第一件事：打印名册 for n,p in model.named_parameters(): print(n)，
    翻到最底下，最后几页就是输出层，看它姓什么再定分组方向。"""
    # 点名册(named_parameters 每页一对(名字n,旋钮组p))翻一遍查姓氏：
    # 名字以 backbone 开头的，只留旋钮本身 p 装进"骨干筐" bb——名字认完人即扔
    bb = [p for n, p in model.named_parameters() if n.startswith(backbone_prefix)]
    # 同一本册子反着挑：不带 backbone 姓氏的(如 head.weight/head.bias)进"头部筐" hd
    # ——两筐合起来=全部旋钮，不漏不重
    hd = [p for n, p in model.named_parameters() if not n.startswith(backbone_prefix)]
    # 交货：AdamW 想吃的"参数组"格式(整包塞进 torch.optim.AdamW(...) 即可)，每组三样——
    # params=这筐旋钮；lr=这筐步长(骨干筐 lr*scale 打折、头部筐原价)；name=标签方便打印。
    # 例 lr=4e-4,scale=0.1：骨干筐步长 4e-5、头部筐 4e-4(学徒十倍速)；此后同一个
    # optimizer.step() 里，两筐各按各的步长拧
    return [
        {"params": bb, "lr": lr * backbone_lr_scale, "name": "backbone"},
        {"params": hd, "lr": lr, "name": "head"},
    ]


def assert_backbone_floor(lr: float, scale: float, min_lr: float):
    """安检员(保险丝)：开跑前检查"骨干组的折扣步长不许低于调度器的退火底价"。
    背景：调度器是缩步器，承诺步长最后恰好落在底价 min_lr；若骨干起步价(lr*scale)
    已在底价之下，余弦公式会反转成"越走越大"往底价爬——老师傅步子不减反增、还不
    报错、悄悄学坏。本断言把"悄悄学坏"变成"启动即报错"(主仓库 trainer.py 39-44 同款)。
    入参：lr=基准步长；scale=老师傅折扣；min_lr=退火底价。"""
    if lr * scale < min_lr:  # 骨干实际起步价(基准×折扣)低于退火底价 = 要穿底
        raise AssertionError(  # 主动抛错、当场停(断言=保险丝)：一行训练都不跑
            f"lr*scale ({lr*scale}) must stay >= min_lr ({min_lr}); "
            f"raise lr/scale or lower min_lr (min_lr==lr*scale => 骨干恒定)")
        # 报错信息自带修法：调大 lr/scale 让起步价回到线上 / 调小 min_lr 把底价挖低；
        # 恰好相等=压线允许，骨干全程恒定在底价(__main__ 的 4e-4*0.1=4e-5 即此配置)


def build_scheduler(optimizer, total_steps: int, warmup_steps: int = 2000,
                    start_factor: float = 0.1, min_lr: float = 4e-5):
    """造一台"自动变速器"：步长(学习率)按剧本自动换挡——前段热身爬坡、后段余弦缓降。
    嵌套关系：optimizer 拧模型旋钮，本调度器拧的是 optimizer 手里的"步长刻度"。
    为什么要两段：①刚开练头部旋钮全随机、梯度又乱又噪，全速拧会把骨干老师傅的
    预训练功力拧坏("起步旋钮乱+梯度噪声大->全速拧坏老底子")，故先 0.1 倍慢速提
    速；②练到后期接近好答案，大步容易一脚迈过最佳点，故余弦缩到底价小步微调。
    入参：optimizer=被调步长的调整师；total_steps=总步数(总里程表)；warmup_steps=
    前 2000 步热身；start_factor=热身起步倍速 0.1；min_lr=退火终点底价。
    整条曲线(README 通关要求"默画 lr 曲线"；变速器缩的是倍数，各组同涨同跌)：
      倍速
     1.0 |             .----.
         |            /       \\
         |           /         \\__
     0.1 | .-------/               '---- min_lr
         +-+--------+------------------+--> 步数
           1      warmup          total_steps
           热身爬坡   余弦缓降
    另：调度器不会自己走——每拧一次旋钮要顺手 scheduler.step() 上发条(主仓库排在
    五连击 step->sched.step->zero_grad；stage2 教学骨架省略，由下方 lr_at 代演示)。"""
    # 第一段·热身器：直线斜坡——前 warmup_steps 步，步长从 0.1 倍直线爬到 1.0 倍
    warm = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0,
                    total_iters=warmup_steps)
    # 第二段·余弦缩步器：热身结束后接管，步长沿余弦曲线从满速缩到底价 eta_min。
    # T_max=这一段要走的步数(总步数-热身步数)；max(1,·)=保险丝，防总步数≤热身步数
    # 时算出 0/负数崩掉；用余弦不用直线：余弦两头都缓——开降温柔、结尾趋平稳落底价
    decay = CosineAnnealingLR(optimizer,
                              T_max=max(1, total_steps - warmup_steps),
                              eta_min=min_lr)
    # 焊接员：两段拼成一台；milestones=[warmup_steps] 是换挡点——
    # 第 1~warmup_steps 步听 warm 的，之后无缝切到 decay
    return SequentialLR(optimizer, [warm, decay], milestones=[warmup_steps])


class DemoNet(nn.Module):
    """道具模型：专为分筐演示而生——存在的唯一意义是给两层起好 backbone/head 两个名。

    姓氏由谁命名(实际业务同理)：写模型的人在 __init__ 里给零件起的【属性名】=参数
    姓氏，PyTorch 按规则自动拼全名——
      self.backbone = nn.Linear(..)  -> backbone.weight / backbone.bias
      self.backbone = 大骨干()       -> 骨干内部几百层全部自动冠上 backbone. 前缀
                                        (backbone.stages.0.blocks.1.proj.weight...)，
                                        你只起最外层名，内部姓氏由骨干作者的名接续拼
      nn.Sequential(卷积,ReLU,..)    -> 没单独起名的层用序号顶替：features.0.weight
    真实项目两层分工：骨干作者起内部名；你只负责最外层关键属性名(如 self.image_encoder
    =骨干(...)，主仓库 trainer 正按 "image_encoder" 前缀分组)。起名是配置接口/纪律：
    分组学习率、冻结、加载预训练权重(load_state_dict 按名字对位)、断点存档全靠名字。
    随时核对名册一行：for n,p in model.named_parameters(): print(n, p.numel())。"""

    def __init__(self):
        super().__init__()  # 老规矩：先初始化父类的参数登记簿(固定写法，不写报错)
        # 骨干层：进 4 出 8。重点不在数字在名字——属性叫 backbone，点名册上参数自动
        # 姓 backbone(backbone.weight/bias)，被分筐员认作"老师傅"进打折筐
        self.backbone = nn.Linear(4, 8)
        # 头部层：进 8 出 2(骨干出的 8 个数正好喂它)。属性叫 head -> 进"学徒筐"原价。
        # 注：本类故意不写 forward——整场演示从不真喂图，它只是"旋钮名册的载体"
        self.head = nn.Linear(8, 2)


def lr_at(scheduler, optimizer, step: int) -> float:
    """快进采样器："变速器走到第 step 步时，头部组步长是多少"——替人看数/给测试断言用，不参与训练。
    入参：scheduler=变速器；optimizer=调整师(变速器拧的就是它手里的步长刻度，读数也从它身上读)；
    step=想知道第几步。__main__ 用它采曲线上 5 个点(1/1000/2000/6000/10000 步)，测试用它断言变速器行为。"""
    for _ in range(step):  # 快进：把发条拧 step 下(每拧一次里程+1 并按曲线更新步长)；_ = "下标用不上"惯例
        scheduler.step()
    # 读表：param_groups 就是 split_param_groups 交的两筐(0=骨干筐 1=头部筐)，取头部筐此刻的步长刻度。
    # 注意：快进不回零——每次调用在现有进度上继续，演示按从小到大连调才和标签吻合；
    # 测试要精确读数时需重造调度器从头快进
    return optimizer.param_groups[1]["lr"]


if __name__ == "__main__":
    lr, scale, min_lr = 4e-4, 0.1, 4e-5
    assert_backbone_floor(lr, scale, min_lr)
    opt = torch.optim.AdamW(split_param_groups(DemoNet(), lr, scale))
    sch = build_scheduler(opt, total_steps=10000, warmup_steps=2000, min_lr=min_lr)
    for s in (1, 1000, 2000, 6000, 10000):
        print(f"step {s:>5}: head lr = {lr_at(sch, opt, s if s == 1 else s - (1 if s==1 else 0)):.2e}")
