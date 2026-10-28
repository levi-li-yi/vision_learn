"""课时 6：断点五件套与 fresh/resume/finetune——base_utils/utils.py 的复刻。
这个文件教你训练中断后怎么恢复：断点文件里存了什么(五件套)、三种重启方式(fresh/resume/finetune)各恢复哪些东西。


对应主仓库：
  - save_trainer_state / load_trainer_state（五件套：weight/optimizer/
    scheduler_last_epoch/scaler/step，任一失败 warning 降级不崩）；
  - 三模式语义：fresh 删旧断点重来；resume 全恢复；finetune 只穿权重。
断点的本质：一个 dict 存进 .pth 文件——没有魔法，只有字典。

运行：python -m stage2.checkpoint_modes
"""
import copy
import tempfile
from pathlib import Path

import torch
from torch import nn


def save_five(model, optimizer, scheduler, scaler, step, path):
    """训练存档器(游戏存档)：把"进行到一半"的五样状态打包写进一个文件，断电/崩溃后
    从这儿原地满血复活(午睡存档：五件齐=睡醒接着干仿佛没停过；缺一件=醒来带点失忆)。
    为什么恰好五件(丢了会怎样)：①weight 旋钮当前值——丢了从随机重来全白学；
    ②optimizer 调整师手法记忆(AdamW 动量)——丢了拧法突变 loss 跳；③scheduler
    里程表读数——丢了变速器归零重新热身；④scaler 当前放大倍数——丢了回出厂值
    重新爬挡；⑤step 总步数——丢了日志/计划对不上表。
    入参：前四个=四位"状态持有人"对象(不传数值传对象，函数自己问它们要存档)；
    step=裸数字走到第几步；path=存档文件写到哪。
    断点本质：一个 dict 存进 .pth 文件——没有魔法，只有字典(读档 load_five 按标签
    逐件取回)。DDP 多卡时模型要先拆 .module 壳再打包(主仓库有此步，教学版单卡略)。"""
    torch.save({  # 把一个 Python 对象整个序列化成字节写进文件；这里只存下面这个五格字典
        "weight": model.state_dict(),             # ①模型旋钮当前值：state_dict()=标准打包格式(名字->数值的册子)，抄一份、模型本身不动
        "optimizer": optimizer.state_dict(),      # ②调整师内部账本(AdamW 给每个参数记的动量/历史)
        "scheduler_last_epoch": scheduler.last_epoch,  # ③变速器里程表读数(一个 int，有它就能对表)
        "scaler": scaler.state_dict(),            # ④放大器状态(当前放大倍数等)
        "step": step,                             # ⑤总步数，裸整数原样入包
    }, path)                                      # 五格字典整个写进 path(.pth/.pth.tar 后缀是惯例)


def load_five(path, model, optimizer=None, scheduler=None, scaler=None):
    """读档器(save_five 的镜像)：开箱五格字典逐件装回，训练"原地满血复活"。
    入参不对称是刻意的：model 必填(权重必须有身体可装=读档核心目的)；opt/sch/
    scaler 可选默认 None——谁传了顺便恢复谁的记忆，不传当全新的(finetune 只要
    权重不要旧手法，resume 才五位全复)。设计哲学"一硬四软"：①weight 必恢复
    (strict=False 宽容位：名册对不上也不炸，模型多出的键保持新值、存档多余的键
    跳过——断点兼容，新键旧键都能活)；②③④尽力恢复，坏了降级成全新、训练照跑
    (对应主仓库 warning 降级不崩：醒来带点失忆，绝不堵死训练)；⑤step 读出返回。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # 开箱：.pth 反序列化回五格字典；map_location="cpu"先落 CPU(防读档机器没同款 GPU 就崩)；weights_only=False 放行含普通对象的字典(自家文件)
    model.load_state_dict(ckpt["weight"], strict=False)  # 穿权重：按名字把旋钮值装回模型(strict=False=宽容模式)
    if optimizer is not None:            # ②恢复调整师记忆(传了才做)
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            pass                                   # 降级：全新优化器(账本对不上静默放弃，权重还在只是手法从头练)
    if scheduler is not None:            # ③恢复里程表
        try:
            scheduler.last_epoch = ckpt.get("scheduler_last_epoch", 0)  # .get(键,0)=保险：老存档缺这个键就当 0
        except Exception:
            pass
    if scaler is not None:               # ④恢复放大器倍数
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass
    return ckpt.get("step", 0)           # ⑤交回总步数(缺键当 0)，日志/续期计划对表


def apply_mode(mode: str, has_old_state: bool) -> str:
    """三模式对断点的处置（主仓库 train_entry+trainer 的分支逻辑）。"""
    if mode == "fresh":
        return "delete_old_state"                 # 只删断点，不删部署权重
    if mode in ("resume", "finetune"):
        return "restore_five" if has_old_state else "warn_from_scratch"
    raise ValueError(f"unknown mode: {mode}")


if __name__ == "__main__":  # 直接运行才执行；被 import 借函数不跑。全流程彩排：凑人->存档->换机读档->逐位对账
    torch.manual_seed(0)  # 钉死种子：初始权重固定，谁跑结果都一样
    # 演员就位(五件套的四位持有人；主角是断点机制不是看图，用最小零件 4进2出=10 个旋钮代替 TinyCNN)
    m = nn.Linear(4, 2)                                              # 模型
    opt = torch.optim.AdamW(m.parameters())                          # 调整师(接管 10 个旋钮)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)  # 变速器(裸造一个，演示里程表怎么存)
    sc = torch.amp.GradScaler(enabled=True)                          # 放大器(CPU 上自禁用成恒等，不影响演示)
    with tempfile.TemporaryDirectory() as d:  # 租临时仓库：with 一结束自动拆房，演示不留垃圾文件
        p = Path(d) / "trainer_state.pth.tar"  # 拼存档路径(Path / 是 pathlib 跨平台拼法；文件名对齐主仓库真实断点)
        save_five(m, opt, sch, sc, step=1234, path=p)  # 存档！step 故意填 1234 好认的数——读回一眼验真伪
        m2 = nn.Linear(4, 2)  # 造"另一台机器的新模型"：同款但另一套随机旋钮，模拟重启后从零装配、与 m 无血缘
        step = load_five(p, m2, optimizer=None, scheduler=None, scaler=None)  # 读档：三位传 None(演示可选恢复=只穿权重)，m2 旋钮被存档值整体覆盖
        # 逐位对账五层从里往外：两本"名字->数值"册子 -> .values() 只取数值列 -> zip 拉链逐对配
        # -> torch.equal 逐位精确相等(存/读=照抄字节，敢用最严比法；与 grads_close 的"近似相等"相反)
        # -> all 收尾：Python 内置"质检员"——一串 True/False 全 True 才 True，遇第一个 False 立刻
        #    短路判负(空串=True，没人反对=通过；兄弟 any 相反=有真即真)；括号里是生成器边产边喂
        same = all(torch.equal(a, b) for a, b in zip(m.state_dict().values(),
                                                    m2.state_dict().values()))
        print(f"恢复 step={step}, 权重逐位一致={same}")  # 成绩单：step=1234 + 存进去的每一位==读回来的每一位
    print("三模式处置:", {mm: apply_mode(mm, True) for mm in ("fresh", "resume", "finetune")})  # 顺带演示三模式处置清单
