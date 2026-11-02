"""课时 1：编码器-解码器与上采样——core/seg.py 级联头的微缩复刻。
这个文件教你语义分割模型的'喇叭结构'：先把图压缩看大局，再逐级放大恢复细节——两种放大方式(反卷积 vs 双线性)的差别用手感体验。


对应主仓库 CascadedSegHead 的"喇叭结构"：64->128(反卷积)->256(反卷积)
->1024(双线性 x4)。本模块用 64->128->256 三级复刻同一形状链，
并实验"双线性插值的软边 vs 反卷积可学边"。

两个核心问题：
  1) 为什么要逐级放大而不是一步到位？——一步拉伸=糊，逐级+卷积=锐；
  2) 反卷积(ConvTranspose2d)和双线性插值差在哪？——前者带可学参数
     （边放大边学怎么修），后者零参数纯几何拉伸。

运行：python -m stage3.encoder_decoder
"""
import torch
import torch.nn.functional as F
from torch import nn


class CascadeHead(nn.Module):
    """喇叭口(解码器)：把压缩过的特征图逐级放大回原图尺寸——主仓库 CascadedSegHead
    的缩小同构版(主仓库 64->128->256->1024，这里到 256、通道减半)。
    为什么逐级放大不一步到位：一步 x4 拉伸=糊，逐级 x2+每次卷积修边=锐(像放大照片：
    直接 4 倍糊成一团；放大 2 倍->锐化->再放大->再锐化清楚得多)。
    形状链：[B,32,64,64] --up1--> [B,16,128,128] --up2--> [B,1,256,256]。

    "通道"(channel)=一摞胶片的张数(注意不是"网络层"那个层)：输入的 32 = 32 张发现图
    叠成一摞，每张是一个探测器画的"哪里有我找的图案"。通道数由卷积参数写死、与输入
    无关——Conv2d(1,4) 就是"4 个探测器各画 1 张新胶片"，多出来的胶片是探测器拿自己
    3x3 窗表在图上滑动加权求和现场画的(窗表 3x3=每次看一片，输出图=输入同尺寸、每格
    一个匹配分)。压缩侧图越压越小、摞越叠越厚(灰度 1->128)；本类是放大侧：逐级把摞
    拆薄(32->16->8)，最后收口成 1 张"疑心值图"——每个位置一个 0~1 的数：这颗像素
    是缺陷吗(分割任务的输出签名)。
    入参：in_ch/mid_ch/out_ch=进/中间/出的胶片张数。"""

    def __init__(self, in_ch=32, mid_ch=16, out_ch=1):
        super().__init__()  # 老规矩：先初始化父类登记簿
        # 第一级放大 up1(64->128)：反卷积(可学放大)+过秤归零+精修卷积
        self.up1 = nn.Sequential(                       # 对应主仓库 up1
            # 反卷积 ConvTranspose2d=边放大边学的"可学放大器"：普通卷积是"扫"(多个输入
            # 凑一个输出、图变小)，反卷积是"铺"(每个输入位置往输出画一小块 4x4，块块
            # 重叠拼出大图；块上带可学权重=能学怎么把边画锐——隔壁 BilinearHead 的双线
            # 性是零参数纯几何拉伸，两者对比正是本文件主题)。4/2/1(kernel/stride/
            # padding)=尺寸正好翻倍的标准配方；通道 32->16 顺带减=越接近最终答案摞越薄；
            # bias=False 因紧跟 BN 过秤自带平移，重复的省掉
            nn.ConvTranspose2d(in_ch, mid_ch, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(),   # 过秤(数字拉回正常范围)+负数归零
            # 精修卷积：3x3 窗扫一遍刚放大的图——"拉伸会糊，卷积修边"，每放大一级立刻
            # 修一级不让糊累积；padding=1 补圈 0 保尺寸不变
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(),
        )
        # 第二级放大 up2(128->256)：同款反卷积翻倍，通道 16->8(4,2,1 是位置参数缩写)
        self.up2 = nn.Sequential(                       # 对应主仓库 up2
            nn.ConvTranspose2d(mid_ch, mid_ch // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid_ch // 2), nn.ReLU(),
            # 收口：1x1 卷积不看邻域、纯通道混合(调色盘/接线板同款)，把 8 张胶片调和成
            # 1 张疑心值图。这次带 bias——后面没有 BN 了，起步数得自己带
            nn.Conv2d(mid_ch // 2, out_ch, 1),
        )

    def forward(self, x):
        x = self.up1(x)                                 # 64->128(摞 32->16)
        x = self.up2(x)                                 # 128->256(摞 16->1)
        return x                                        # [B,32,64,64]->[B,1,256,256](实跑已验)


class BilinearHead(nn.Module):
    """对照组：与 CascadeHead 打擂台——精修卷积/口径(32->16->1)/形状链全保持一致，
    唯一变量是"放大"这一步：反卷积(带旋钮、边放大边学修边) vs 双线性(零旋钮、纯
    几何拉伸)，单变量对照。双线性=放大时新格子的值取周围 4 个最近老格子按距离
    加权平均(离谁近多听谁的)——权重是距离算出的固定比例、一个旋钮都没有，只能
    掰开摊大、变不出新细节，边缘变软。
    形状链(终点与 CascadeHead 相同)：[B,32,64,64]->精修[B,16,64,64]->拉伸[B,16,
    128,128]->精修[B,1,128,128]->拉伸[B,1,256,256]。主仓库真实选择是混搭：前几级
    反卷积(可学预算花在刀刃上)、末级 1024 用双线性 x4(图太大、反卷积参数/算力贵)。
    双线性的过渡值=软边——对输入图是好事(边缘自然)，对 0/1 答案掩码是坏事(见下方
    soft_edge_fraction，主仓库"图像可以柔、答案必须硬")。"""

    def __init__(self, in_ch=32, mid_ch=16, out_ch=1):
        super().__init__()  # 口径与 CascadeHead 完全一致——公平起见的对齐
        # 第一站精修：普通 3x3 卷积扫一遍(通道 32->16)+过秤+归零。不放大——
        # padding=1 保尺寸，只改内容不碰大小(与 CascadeHead 的精修卷积同款)
        self.refine1 = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 3, padding=1),
                                     nn.BatchNorm2d(mid_ch), nn.ReLU())
        # 第二站精修：3x3 直接 16->1 收口成疑心值图(用 3x3 不用 1x1=顺带看邻域
        # 再下结论)。没有 BN/ReLU——最终疑心值是 logit、可正可负，不能截成非负
        self.refine2 = nn.Sequential(nn.Conv2d(mid_ch, out_ch, 3, padding=1))

    def forward(self, x):
        # 数据流"精修->拉伸->精修->拉伸"。F.interpolate=改图尺寸的工具函数：
        # scale_factor=2 放大一倍；mode="bilinear" 双线性填新格子(零旋钮)；
        # align_corners=False=按格心对位的现代默认约定(防整体偏移半个格)
        x = self.refine1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False)          # 64->128 纯拉伸
        x = self.refine2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False)          # 128->256 纯拉伸
        return x


def soft_edge_fraction(mask: torch.Tensor) -> float:
    """软边度量尺：把非黑即白的 0/1 掩码放大 4 倍，数多少格子变"半黑半白"——
    用数字坐实"双线性会把答案弄软、最近邻不会"。
    这就是主仓库铁律"图像可以柔、答案必须硬"的位级证据：输入图像放大用双线性
    (边缘自然)；答案掩码放大永远用最近邻(新格子直接抄最近老格子，0 抄 0、1 抄 1，
    绝不产中间值)——分割答案要和 0/1 标签逐像素比对，中间值既不是"是"也不是
    "不是"=把标准答案涂脏(照片可以修朦胧，答题卡必须 2B 铅笔涂满非黑即白)。
    实测：双线性约 75%(演示掩码无固定种子，每次略变) / 最近邻恒 0.00%——75% 夸张
    是因为演示用随机椒盐掩码(几乎每格都贴黑白交界)；真实缺陷掩码是一整块斑，
    软边只在边缘一圈(约百分之几)。"""
    # 放大 4 倍用双线性(新格子=4 个老邻居按距离加权平均)。mask.float() 先转小数
    # 容器——平均会产出 0.3/0.7 的中间值，整数容器装不下。黑白交界的新格子就此变软
    up = F.interpolate(mask.float(), scale_factor=4, mode="bilinear",
                       align_corners=False)
    # 一行数占比，从里往外：①(up>0.05)&(up<0.95) 找"既不太黑也不太白"的过渡格
    # (0.05/0.95 留容差，浮点的"纯黑"可能是 0.0001 擦边值)；②.float() True/False
    # 变 1/0；③.mean() 平均=过渡格占比；④.item() 取出 Python float
    soft = ((up > 0.05) & (up < 0.95)).float().mean().item()
    return soft


if __name__ == "__main__":
    print("== 级联头形状链（应与主仓库 64->128->256 一致）==")
    head = CascadeHead()
    y = head(torch.randn(1, 32, 64, 64))
    print(f"输入 [1,32,64,64] -> 输出 {list(y.shape)}")

    print("\n== 双线性 vs 最近邻：软边占比 ==")
    mask = (torch.rand(1, 1, 32, 32) > 0.5).float()
    near = F.interpolate(mask, scale_factor=4, mode="nearest")
    up_bil = F.interpolate(mask, scale_factor=4, mode="bilinear",
                           align_corners=False)
    soft_bil = ((up_bil > 0.05) & (up_bil < 0.95)).float().mean().item()
    soft_near = ((near > 0.05) & (near < 0.95)).float().mean().item()
    print(f"双线性软边占比 = {soft_bil:.2%}（图像用：过渡自然）")
    print(f"最近邻软边占比 = {soft_near:.2%}（答案用：非黑即白）")
