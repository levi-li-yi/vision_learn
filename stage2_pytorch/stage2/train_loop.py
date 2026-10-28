"""课时 3：完整训练循环 + 梯度累积——trainer.train() 的可测试骨架。
这个文件教你写一个完整的训练循环——就是'喂图、猜答案、算错多少、调旋钮'这四步的代码版，还教'显存不够时怎么分小批攒着当一大批用'。


对应主仓库 trainer.py 114-157 行逐句：
  前向 -> loss_masks -> 总分÷accumulation_steps -> scale 反传 ->
  (累积最后一步) unscale -> 裁剪 -> step -> sched.step -> zero_grad。
梯度累积的数学：N 小批各自 backward 梯度会**相加**，故每小批 loss 先 ÷N，
总效果恰等价于一大批——本模块用测试钉死这个等价性。

运行：python -m stage2.train_loop
"""
import torch
from torch import nn


def train_one_epoch(model, loader, optimizer, accumulation_steps=1,
                    max_norm=1.0, use_scaler=None):
    """单 epoch 训练：loader 所有批过一遍，四步循环"喂图→猜答案→算错多少→调旋钮"。

    入参：model=模型(nn.Module，旋钮在它身上)；loader=搬运工，每批出 x[float32,
    B,1,16,16]+y[int64,B]；optimizer=旋钮调整师(AdamW 等)；accumulation_steps=
    累积步数 N——攒 N 个小批当一大批再更新(显存不够时用)；max_norm=梯度安全带
    阈值；use_scaler=fp16 小梯度放大器 GradScaler 或 None(fp32/bf16 不用)。

    与主仓库一致的三个细节：
      1) loss ÷ accumulation_steps；
      2) 只有累积的最后一步才 step/zero（中间步只 backward）；
      3) step 前先 unscale 再裁剪 max_norm。

      
      用“故事 + 箭头图 + 真实数字”的形式，把 train_one_epoch 从进门到出门完整讲一遍。设定：B=32 张/批、共 16 批、N=1（先讲普通模式）：
      
      【进门】
          model.train()          挂"营业中"牌
          total, nbatch = 0, 0   拿出两个空流水账（罚分总和 / 批数）
                │
                │  ←────────── 搬运工送来第 1 批：x[32 张图] + y[32 个答案]
                ▼
        ① 猜答案    logits = model(x)          每图 2 个分数 [32,2]
                │                              （fp16 时前向裹在 autocast 里跑，快且省显存）
                ▼
        ② 打罚分    loss = cross_entropy(logits, y)   → 0.69（瞎猜固定起点）
                │
                ▼
        ③ 除以 N    0.69 / 1 = 0.69            （为攒批模式预留）
                │
                ▼
        ★ 先放大    loss = scaler.scale(loss)   → 0.69 × 65536 ≈ 45225
                │                                 （给小梯度上保险，防止 fp16 里变 0）
                ▼
        ④ 反向记账  loss.backward()             594 个梯度全部 ×65536 写进账本 .grad
                │                                 例：真实 0.002 的小梯度 → 存成 131，得救
                ▼
        ⑤ 攒满了吗  1 % 1 == 0 → 满！进五连击
                │
        ┌───────▼──────────────────────────────────────────────┐
        │ 五连击                                                │
        │ ★ 还钱   scaler.unscale_(opt)   131 ÷ 65536 = 0.002   │
        │                                       账本恢复真实数字 │
        │   限速   clip(≤1.0)             量真实箭头，超 1.0 压回│
        │ ★ 把关   scaler.step(opt)       先查账本有无 inf/NaN：│
        │                                   干净 → 真正拧 594 个 │
        │                                   旋钮；脏 → 这批不拧！│
        │ ★ 调挡   scaler.update()        连续干净多步→放大倍数  │
        │                                   试着加大；刚跳坏步→  │
        │                                   倍数减半(65536→32768)│
        │   撕账   zero_grad()            账本清零，下批白纸     │
        └───────┬──────────────────────────────────────────────┘
                ▼
        ⑥ 记流水账  total += 0.69（真实罚分）；nbatch += 1
                │
                ▼
          回到 ① 迎接第 2 批…… 共 16 批
                │
                ▼
        【出门】return total / 16   →  交出平均罚分（门外 print + accuracy 考试）
    """
    model.train()  # 切"训练模式"(影响 Dropout/BatchNorm；TinyCNN 没这类层，纯规范动作)
    total, nbatch = 0.0, 0  # 两个账本：损失总和(Python float)+批数(int)——故意不用张量，账本不参与求导
    # x/y = 一批考题的两半(命名沿用数学 y=f(x) 惯例：x 题面、y 答案)：
    # x 是 4 维张量 float32[B,1,16,16]——B 张、每张 1 通道 16 行 16 列，格子装 0~1 像素亮度；
    # y 是 1 维张量 int64[B]——每个 0/1(0=亮块左上 1=右下)；y[i] 配 x[i] 一一对位绝不乱序
    for i, (x, y) in enumerate(loader, 1):  # i 从 1 数起(配合取余判断)；双层解包：先剥批号 i，再剥题面和答案
        logits = model(x)  # 猜答案(前向)：图过模型 -> logits[float32,B,2]，每图 2 个分，带记账(grad_fn)
        loss = nn.functional.cross_entropy(logits, y)  # 算错多少：对照答案打总罚分 -> 标量张量；瞎猜期≈ln(2)≈0.69
        loss = loss / accumulation_steps  # ÷N 备累积：N 小批梯度相加，各自先÷N、加完恰等一大批
        if use_scaler is not None:
            loss = use_scaler.scale(loss)  # fp16 先放大再反传(主仓库"scale 反传"步)：小梯度 0.002->131 防
                                           # fp16 里变 0；恒等 scaler(enabled=False/CPU 自禁用)原样返回零影响
        loss.backward()  # 反传记账：594 个参数的 .grad 被"累加"(非覆盖！累积能成立的根基)；只记账不拧旋钮
        if i % accumulation_steps == 0:  # 攒满一组才触发"五连击"；N=1 时每批都触发=普通训练
            if use_scaler is not None:
                use_scaler.unscale_(optimizer)  # 还原：把放大的梯度除回真实值——下一步裁剪要拿真实值比
            if max_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # 安全带：梯度总长>1.0 就整体等比例压到 1.0
            if use_scaler is not None:
                use_scaler.step(optimizer)  # 拧旋钮(先查 inf/NaN，干净才真更新)
                use_scaler.update()         # 动态调下次放大倍数
            else:
                optimizer.step()  # 拧旋钮：参数值朝让 loss 变小的方向挪一点(值变、类型形状不变)
            optimizer.zero_grad()  # 清梯度账本(清 .grad、不是清参数)——只在累积最后一步调，中间清=白攒
        total += loss.item() * accumulation_steps  # 乘回 N 还原真实损失入账；.item() 取出 Python float 并切断计算图
        nbatch += 1
    return total / max(1, nbatch)  # 返回 Python float：平均每批损失；max(1,·) 防空 loader 除零


@torch.no_grad()  # "草稿纸不用留"：考试不学习、永远不用回头改旋钮，不留记号省内存还快
def accuracy(model, loader):
    """考官：让模型把题库全做一遍，数蒙对几题，报命中率(0.0~1.0)。
    例：模型答卷 [1,0,1,0] vs 标准答案 [1,0,0,1] -> 对 2 题 -> 返回 0.5。
    与训练的分工：训练问"错得多离谱"(罚分 loss，要照着改旋钮)；考试只问"对没对"(数数即可)。"""
    model.eval()  # 进考场前换模式(Dropout 这类"训练/考试两副面孔"的零件需要切换；TinyCNN 没有纯走过场)
    hit = tot = 0  # 小本本：蒙对几题 / 总共考几题
    for x, y in loader:  # 一批批把题做完
        # 一行就是考试五步：模型每题打 2 个分 -> argmax(1) 挑大分坐在几号位(0=左上 1=右下)当答卷
        # -> ==y 逐题对答案出 ✓✗ -> sum() 数 ✓(一个算 1) -> int() 拿出数字记小本本
        hit += int((model(x).argmax(1) == y).sum())
        tot += len(y)  # 这批考了几题
    return hit / max(1, tot)  # 命中率 = 对的÷总的；max(1,·) 防空题库除 0 崩溃


def grads_close(m1: nn.Module, m2: nn.Module, tol=1e-5) -> bool:
    """质检员：验收"攒批"承诺——分 N 小批攒的梯度 ≈ 一大批直接算的梯度。
    背景：训练任一时刻只走一条路(accumulation_steps 开关拨几就走哪条)。"一大批"
    是理想，"攒批"是显存(桌面)不够时的替代打法：每趟只摊 16 张草稿，梯度记账本
    攒着。数学保证殊途同归——全班平均分原理：批内平均÷N 再相加 = 总平均，这正是
    train_one_epoch 里 ÷N 的来历；忘 ÷N 则攒出来正好差 N 倍。本函数不参与训练，
    只住在测试里：每次 pytest 重演双胞胎实验——老大 64 张一口气算，老二 4 趟(各
    ÷4)攒，逐组比对两边账本，几乎一样才盖章"承诺没写坏"。比"差不多"不比"一模
    一样"：计算器算小数天生有微小抖动，== 几乎必失败，只能比"差得够不够小"(每格<0.00001)。"""
    # named_parameters()=模型的点名册：594 个旋钮按名字分成 6 组，每翻一页交出一对
    # (名字, 旋钮组本身)——k=页标题/部位名(如 features.0.weight=流水线第 0 盒的卷积核)，
    # 只用来核对身份(别拿视力对血压)；p=那组旋钮本身，要比的梯度挂在它身上(p.grad)；
    # 下标 1/2 = 老大/老二。zip 把两本点名册拉在一起逐页对：老大第 1 组对老二第 1 组，不错位
    for (k1, p1), (k2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        # 开比先验三样：两页名字对不上=不是同款、比个寂寞；谁的草稿还空白=没backward过没得比
        if k1 != k2 or p1.grad is None or p2.grad is None:
            return False
        # 正式逐页比：这一组一个格一个格对，每格差 <0.00001 才算过；一个格子差太多整场判负
        if not torch.allclose(p1.grad, p2.grad, atol=tol):
            return False
    return True  # 6 组全部对完、没被中途判负 -> 两边梯度几乎一样，承诺兑现


if __name__ == "__main__":  # 只在直接运行本文件时执行；被 import 借零件时不跑演示
    # 从隔壁课时 2 借三样零件：印刷厂数据集、打包员、迷你模型（开头的点=同工程内相对导入）
    from .nn_module_dataset_loader import BrightCornerDataset, collate_stack, TinyCNN
    from torch.utils.data import DataLoader  # 搬运工是 PyTorch 自带的，直接领一个

    torch.manual_seed(0)  # 钉死随机种子：初始旋钮/题目噪声/打乱顺序全固定，谁跑结果都一样
    ds = BrightCornerDataset(512)  # 开印刷厂：备 512 道题（此刻还没印，取货时现做）
    # 雇搬运工并交代规矩：每批发 32 张、下标打乱、散装样本交给打包员拼成两个张量
    # 512 ÷ 32 = 一轮 16 批
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_stack)
    model = TinyCNN()                                     # 先装配模型：594 个旋钮全随机 = 瞎猜状态
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)  # 再让调整师接管它的 594 个参数；lr=拧旋钮的步子大小
    for ep in range(1, 4):  # 轮次循环：同一本题库从头学 3 遍（一遍=一个 epoch）——别和函数内的批循环混淆
        loss = train_one_epoch(model, loader, opt)  # 学一遍：16 批"猜→罚→记账→拧旋钮"，返回平均罚分
        # 交成绩单：平均罚分 loss（越低越好）+ 考官 accuracy 当场考一场报命中率（越高越好）
        print(f"epoch {ep}: loss={loss:.4f} acc={accuracy(model, loader):.2%}")
