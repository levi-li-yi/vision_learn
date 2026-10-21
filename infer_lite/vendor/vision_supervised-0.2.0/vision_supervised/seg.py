#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""seg.py —— 分割模型: 官方 EfficientViT-SAM L1 骨干 (SA-1B 预训练热启动)
+ 浅层跳连级联分割头。

  · 3 通道输入 (灰度×3, 预训练权重原样灌入)
  · backbone.stages[0..2] 完全冻结 (含 BN eval), stages[3..4] 挂 LoRA 旁路
    (旁路零初始化起步恒等, 官方权重可原样载入), neck + head 全参数开放
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


class CascadedSegHead(nn.Module):
    """渐进式上采样分割头: 64 -> 128 -> 256 -> (Bilinear) -> 输入分辨率。
    final_scale: 末段双线性上采样倍率 —— 官方 L1 骨干 1/8 步距用 2.0。

    浅层高分辨率跳连: 9~13px 小缺陷在 1/8 特征仅 ~1 cell, 被深层池化抹平
    —— 引出骨干 stage1(1/4) 与 stage0(1/2) 特征, 各经 1x1 Conv 瓶颈降维后
    Concat 进解码器 (严禁 Add —— 浅层铸造噪点/走刀纹会被无条件注入引发误报;
    Concat 让网络自学门控阻尼)。
    """

    def __init__(
        self,
        in_channels: int = 256,
        mid_channels: int = 64,
        out_channels: int = 1,
        final_scale: float = 2.0,
        skip1_channels: int = 0,
        skip0_channels: int = 0
    ):
        self.final_scale = final_scale
        super().__init__()

        self.skip1_proj = self.skip0_proj = None
        if skip1_channels:
            s1 = mid_channels // 2                      # 64->32
            self.skip1_proj = nn.Sequential(
                nn.Conv2d(skip1_channels, s1, 1, bias=False),
                nn.BatchNorm2d(s1), nn.SiLU(inplace=True))
        if skip0_channels:
            s0c = mid_channels // 4                     # 64->16
            self.skip0_proj = nn.Sequential(
                nn.Conv2d(skip0_channels, s0c, 1, bias=False),
                nn.BatchNorm2d(s0c), nn.SiLU(inplace=True))

        up1_out = mid_channels + (mid_channels // 2 if skip1_channels else 0)
        up2_out = mid_channels // 2 + (mid_channels // 4 if skip0_channels else 0)
        self._has_skips = bool(skip1_channels or skip0_channels)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, mid_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 有跳连: up2 只做上采样+激活, 通道融合延后到 final_conv
        # (Concat 必须发生在降维到 out_channels 之前)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                up1_out, mid_channels // 2,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(up2_out, out_channels, kernel_size=1)

    def forward(self, x, skip1: torch.Tensor = None,
                skip0: torch.Tensor = None):
        x = self.up1(x)                                  # ↑2
        if self.skip1_proj is not None and skip1 is not None:
            x = torch.cat([x, self.skip1_proj(skip1)], dim=1)

        x = self.up2(x)                                  # ↑2
        if self.skip0_proj is not None and skip0 is not None:
            x = torch.cat([x, self.skip0_proj(skip0)], dim=1)

        x = self.final_conv(x)
        logits = F.interpolate(
            x,
            scale_factor=self.final_scale,
            mode='bilinear',
            align_corners=False
        )
        return logits


def _attach_lora(cl: nn.Module, r: int = 16, alpha: float = 32.0,
                 dropout: float = 0.05) -> bool:
    """给 ConvLayer 原位挂 LoRA 旁路 (键名不变, 官方权重可原样灌入):
    冻结 cl.conv, 旁路 lora_a/lora_b 注册在 cl 上, 前向被替换为
    conv(x) + α/r · B(A(x))。B 零初始化 → 起步恒等。"""
    conv = getattr(cl, "conv", None)
    if not isinstance(conv, nn.Conv2d) or conv.kernel_size != (1, 1) \
            or conv.in_channels < 64 or conv.out_channels < 64:
        return False
    for p in conv.parameters():
        p.requires_grad_(False)
    a = nn.Conv2d(conv.in_channels, r, 1, bias=False)
    b = nn.Conv2d(r, conv.out_channels, 1, bias=False)
    nn.init.kaiming_uniform_(a.weight, a=5 ** 0.5)
    nn.init.zeros_(b.weight)
    cl.add_module("lora_a", a)
    cl.add_module("lora_b", b)
    scale = alpha / r
    drop = nn.Dropout(dropout)
    cl.add_module("lora_drop", drop)

    # 前向替换必须用 partial 挂: deepcopy 会经 memo 把 partial 捕获的 cl
    # 重绑到拷贝自身。若用默认参数/闭包捕获 (函数对象在 deepcopy 中按引用
    # 共享, 捕获不随拷贝走), EMA 拷贝的前向会执行训练侧原模块 —— EMA 权重
    # 对这些层不生效, 且原模块 training 态 SyncBatchNorm 会在单 rank 前向
    # 中发 all_gather, 与其他 rank 的集合通信错位 → 死锁。
    def _lora_fwd(cl_: nn.Module, s: float, x):
        return type(cl_).forward(cl_, x) + s * cl_.lora_b(cl_.lora_drop(cl_.lora_a(x)))
    cl.forward = functools.partial(_lora_fwd, cl, scale)
    return True


def _apply_lora(module: nn.Module, r: int = 16, alpha: float = 32.0,
                dropout: float = 0.05) -> int:
    n = 0
    for name, child in module.named_children():
        if _attach_lora(child, r, alpha, dropout):
            n += 1                     # 已挂载, 其子无需再递归
        n += _apply_lora(child, r, alpha, dropout)
    return n


class OfficialEfficientViTSeg(nn.Module):
    """官方骨干版分割模型 (3 通道: 灰度×3 预训练)"""

    def __init__(self, input_size: int = 512, in_channels: int = 3):
        super().__init__()
        from efficientvit.models.efficientvit.sam import efficientvit_sam_l1
        sam = efficientvit_sam_l1(image_size=input_size)
        self.image_encoder = sam.image_encoder

        bb = self.image_encoder.backbone
        with torch.no_grad():
            feed = bb(torch.zeros(1, in_channels, 64, 64))
            skip0_ch, skip1_ch = int(feed["stage0"].shape[1]), int(feed["stage1"].shape[1])
        self.head = CascadedSegHead(final_scale=2.0,
                                    skip1_channels=skip1_ch,
                                    skip0_channels=skip0_ch)
        # 冻结 stages[0..2] (BN 保持 eval, running stats 不更新)
        for st in bb.stages[:3]:
            for p in st.parameters():
                p.requires_grad_(False)
        # stages[3..4] 挂 LoRA
        self.n_lora = _apply_lora(bb.stages[3]) + _apply_lora(bb.stages[4])

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            bb = self.image_encoder.backbone
            for st in bb.stages[:3]:
                st.eval()
        return self

    def forward(self, x):
        # 引出浅层特征 (backbone 返回逐 stage 字典), neck/norm 与官方前向一致
        feed = self.image_encoder.backbone(x)
        skip0, skip1 = feed["stage0"], feed["stage1"]
        enc = self.image_encoder.norm(self.image_encoder.neck(feed)["sam_encoder"])
        return self.head(enc, skip1=skip1, skip0=skip0)


def build_model(input_size: int = 512) -> nn.Module:
    return OfficialEfficientViTSeg(input_size=input_size, in_channels=3)
