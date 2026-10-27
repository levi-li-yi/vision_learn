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
    """
    model.train()  # 切"训练模式"(影响 Dropout/BatchNorm；TinyCNN 没这类层，纯规范动作)
    total, nbatch = 0.0, 0  # 两个账本：损失总和(Python float)+批数(int)——故意不用张量，账本不参与求导
    for i, (x, y) in enumerate(loader, 1):  # i 从 1 数起(配合取余判断)；x:float32[B,1,16,16]，y:int64[B]
        logits = model(x)  # 猜答案(前向)：图过模型 -> logits[float32,B,2]，每图 2 个分，带记账(grad_fn)
        loss = nn.functional.cross_entropy(logits, y)  # 算错多少：对照答案打总罚分 -> 标量张量；瞎猜期≈ln(2)≈0.69
        loss = loss / accumulation_steps  # ÷N 备累积：N 小批梯度相加，各自先÷N、加完恰等一大批
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


@torch.no_grad()  # 全函数"不许记账"：考试只算分对答案、永不 backward，关掉计算图省内存又快
def accuracy(model, loader):
    """考官：出题->对答案->报命中率。训练循环的精简版——只剩喂图+判卷两步
    （没有 loss/backward/step，考试不用学）。分工：训练用 cross_entropy(可导罚分)，评估用 argmax(硬判决)。"""
    model.eval()  # 切"评估模式"(影响 Dropout/BatchNorm 等双面层；与 no_grad 是两件不同的事)
    hit = tot = 0  # 两个账本(Python int)：蒙对几题 / 总共考几题——纯统计不参与求导
    for x, y in loader:  # 所有批过一遍；不用 enumerate——考试不需要批号做取余判断
        # 浓缩一行从里往外剥：①model(x) 前向得 logits[float32,B,2]；②argmax(1) 沿每图
        # 自己的 2 个分取最大分的下标=网络判的类别(int64[B])；③==y 逐题比对得布尔[B]；
        # ④sum() 数 True(按 1 计)；⑤int() 取出成 Python int 累进账本
        hit += int((model(x).argmax(1) == y).sum())
        tot += len(y)  # 这批考了 B 题，累进总题数
    return hit / max(1, tot)  # 命中率 Python float 0.0~1.0；max(1,·) 防空 loader 除零


def grads_close(m1: nn.Module, m2: nn.Module, tol=1e-5) -> bool:
    """裁判：比两个同款模型的梯度是否"近似相等"——给累积等价性实验判卷。
    剧情见 tests/test_train_loop.py：m_big 一大批直接 backward，m_acc 分 4 小批(各
    loss/4)攒梯度，本函数断言两边 .grad 几乎一样，从而钉死主仓库 loss÷N 的合法性。
    叫 close 不叫 equal：浮点无精确相等(求和顺序不同都差一丝)，只能比"差得够不够小"。"""
    # named_parameters() 把 594 个数按名字分成 6 个张量(卷积核/偏置/打分表)逐个列出；
    # zip 拉链配对：同款模型清单顺序固定，保证"卷积核对卷积核"不错位
    for (k1, p1), (k2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        # 资格预审：名字不同 = 比错对象；某边 .grad 还是 None = 没_backward过没账可比
        if k1 != k2 or p1.grad is None or p2.grad is None:
            return False
        # 正式比对：allclose = 带容差的逐格相等(每格 |a-b|<=tol 全过才 True)；
        # 若训练循环忘了 ÷N，攒出的梯度差 N 倍，这里立刻 False、测试变红
        if not torch.allclose(p1.grad, p2.grad, atol=tol):
            return False
    return True  # 6 对参数全过关、没被中途判负打断 -> 等价性成立


if __name__ == "__main__":
    from .nn_module_dataset_loader import BrightCornerDataset, collate_stack, TinyCNN
    from torch.utils.data import DataLoader

    torch.manual_seed(0)
    ds = BrightCornerDataset(512)
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_stack)
    model, opt = TinyCNN(), torch.optim.AdamW(model.parameters(), lr=3e-3)
    for ep in range(1, 4):
        loss = train_one_epoch(model, loader, opt)
        print(f"epoch {ep}: loss={loss:.4f} acc={accuracy(model, loader):.2%}")
