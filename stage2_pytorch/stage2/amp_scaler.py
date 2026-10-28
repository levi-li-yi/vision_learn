"""课时 5：混合精度(autocast bf16) 与 GradScaler——"快的部分快，准的部分准"。
这个文件教你'怎么让 GPU 算得快还不出错'：用 bf16 低精度提速，但关键运算保持高精度——'快的部分快，准的部分准'。


对应主仓库：
  - trainer.py 120-121 行 `autocast(dtype=torch.bfloat16)` 前向；
  - GradScaler 配 bf16 近似恒等、配 fp16 才真正干活（放大损失防小梯度下溢）；
  - 4.5 章真实事故：推理侧 fp16 溢出产 NaN —— 本模块亲手复现"溢出"。

三个实验：
  1) autocast 下矩阵乘输出 dtype 变 bf16（算得快的证据）；
  2) fp16 存大数变 inf，bf16 有限（位分配的实验版）；
  3) GradScaler 基本动作：scale(loss) 放大 -> backward -> unscale 还原。

运行：python -m stage2.amp_scaler（CUDA 可用则用 GPU，否则 CPU 模拟 dtype 行为）
"""
import torch
from torch import nn


def matmul_dtype_under_autocast(device: str) -> torch.dtype:
    """验钞灯(试纸实验)：验证 autocast 区域内"重活"(矩阵乘)自动换成 bf16 半精度干。
    背景(混合精度哲学"快的部分快，准的部分准")：矩阵乘/卷积这类重活用 bf16(16 位，
    指数位与 fp32 同宽=范围不牺牲、只牺牲一点尾数精度)——GPU 算得快还省一半显存；
    softmax/算 loss 等精细活保持 fp32。autocast=划区域的开关：with 包住的代码内部
    自动按"重活/精细活"分类换精度，计算代码一行不用改；主仓库 trainer.py 120-121
    行前向就裹在 autocast(dtype=bfloat16) 里，本函数是那机制的微缩实验。
    入参：device="cuda"/"cpu"；返回数据格式标签，预期 torch.bfloat16。"""
    # 造试纸：全连接层 Linear(16,16)=16 个数进、16 个数出。肚里一张 16x16 权重表+16 个
    # 偏置(共 272 个旋钮)；"全连接"=每个输出和全部 16 个输入各拉一根线(共 256 根)，
    # 每个输出=16 个输入加权求和+自己的偏置；16 个输出并行一起算就是一个矩阵乘
    # (函数名 matmul 的来历)。本实验只借它的矩阵乘当试纸，不需要它真会打分
    net = nn.Linear(16, 16).to(device)
    x = torch.randn(4, 16, device=device)  # 造输入：4 份各 16 个数的随机清单——此刻是默认 float32
    # (实验巧处：fp32 进去，看什么格式出来)。
    # ↓ 以下 6 行是历史遗留死代码、什么都不干：三元表达式两个分支都是 bfloat16(等于
    #   没绕)，内嵌 autocast 配 pass 也没算任何东西。作者原想按设备挑格式，后发现 CPU
    #   也支持 bf16 autocast(torch>=2.0)不用挑了，改完的尾巴没删——真正干活的在下面
    with torch.autocast(device_type=device, dtype=torch.bfloat16
                        if device != "cpu" or torch.backends.mps.is_available() else torch.bfloat16):
        # CPU 也支持 bf16 autocast（torch>=2.0）
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            pass
    # 真正的实验：开 bf16 的 autocast 区域，区域内过线性层——autocast 认出"重活"
    # (矩阵乘)，自动把这次乘法换成 bf16 算(输入 fp32 进区域后被就地转换)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        # out=试纸吐出的"新清单"：[4,16] 的 bf16 张量，每行=对应输入行 16 个数过权重表
        # 混合出的 16 个新数。数值本身是随机权重下的垃圾值、没人看——它存在的全部
        # 意义是把"bf16"这个格式标签带出来给 return(验钞灯看荧光记号，不看钞票金额)
        out = net(x)
    return out.dtype  # 交证据：进去 fp32、出来 bf16，坐实"低精度干重活"(运行打印的就是它)


def fp16_gradient_underflow(lr_scale_needed: bool = True) -> dict:
    """案卷：复现"小梯度下溢事故" + GradScaler 救法的最小证明。
    背景：fp16 指数位只有 5 位，安全区窄(约 6e-8 ~ 65504)——比 6e-8 还小就四舍五入
    成 0(下溢，本函数演的)；超 65504 变 inf(上溢，推理侧事故)。很多真实梯度就是
    1e-8 量级：fp16 一转变 0 -> 旋钮不动 -> 无声白学且不报错(对比 bf16：指数位 8 位
    同 fp32，两头宽裕天生不怕，故主仓库用 bf16)。返回对照表：直接转 vs 先放大再转。"""
    # (注：参数 lr_scale_needed 在函数体里从未用到——摆设参数，历史遗留的尾巴)
    g_fp32 = torch.tensor(1e-8, dtype=torch.float32)  # 造案发样本：真实量级的小梯度，fp32 格子细放得下(对照组)
    g_fp16 = g_fp32.to(torch.float16)          # 事故现场：1e-8 < fp16 最小刻度(约6e-8)装不下 -> 四舍五入成 0(下溢)
    scaled = (g_fp32 * 2 ** 16).to(torch.float16)  # 救法演示(GradScaler 思路最小版)：先 x65536 再转 = 6.55e-4，稳落安全区保得住
    return {"raw_fp16": g_fp16.item(), "scaled_fp16": scaled.item()}  # 交对照表：0.0(无声白学) vs 0.000655(先放大的功劳)


def scaler_round_trip(device: str = "cpu"):
    """三步曲最小闭环演示：借 1 个旋钮的迷你模型走一遍 scale->backward->unscale，
    验证"放大一个来回(round trip=一去一回)不丢数"——账本梯度应和手算的 2.0 一致。
    衔接：上承下溢实验(为什么要放大)，下接训练循环五连击里 unscale 那步(何时还原)。"""
    # (注：参数 device 在函数体里从未用到——又一个摆设参数，历史遗留尾巴)
    w = torch.tensor([1.0], requires_grad=True)  # 迷你模型：单旋钮 w=1.0 贴求导标签；只用 1 个旋钮是为了数学能手算对答案
    scaler = torch.amp.GradScaler(enabled=True)  # 请出放大器本尊(CPU 上自动禁用成恒等、放大倍数=1，本演示跑恒等版数值照样对)
    loss = (w * 2).sum()  # 手造最简前向 loss=2w；标准答案手算可得：d(loss)/dw = 2
    # 三步曲①+②连写：scale 先放大 loss(真启用时 x65536)再立刻 backward 记账——
    # 放大版账本存 2x65536=131072 的"假数字"(对照下溢实验：不放大小梯度会丢)
    scaler.scale(loss).backward()
    # 三步曲③还钱：账本除回放大倍数恢复真实值；临时造的 SGD 纯当道具
    # (unscale_ 合同要求传 optimizer，它从头到尾没被 step、不拧任何旋钮)
    scaler.unscale_(torch.optim.SGD([w]))
    return w.grad.item()      # 读账本交卷：预期 2.0=手算答案——放大再还原分毫不差


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备:", dev)
    print("autocast 输出 dtype:", matmul_dtype_under_autocast(dev))
    print("fp16 梯度下溢实验:", fp16_gradient_underflow())
    print("GradScaler 还原梯度:", scaler_round_trip())
