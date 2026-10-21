#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mini_model.py —— TinyUNet:train_mini 产物的推理侧构造器(与 train_mini/model.py 逐字一致)。

用途:服务 train_mini(迷你同构训练工程)交付的 best_int*.pth 时,无需构建
vision_supervised wheel,直接以环境变量切换构造器:

    $env:SUPERVISED_MODEL_MODULE = "mini_model"
    $env:SUPERVISED_MODEL_BUILDER = "build_model"
    python app_run.py

ckpt 契约与真工程一致({"weight","step","epoch","intersection","hit_defects",
"quality_score"}),state_dict 键与本文件 TinyUNet 构造严格一致(strict=True 可载)。
全卷积:任意输入尺寸输出同尺寸 logits。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """~0.4M 参数的迷你 U-Net(3 级下采样/上采样 + 跳连 concat)。

    输入 [B,3,H,W](灰度×3,与真工程同预处理);输出 [B,1,H,W] logits。
    """

    def __init__(self, in_ch: int = 3, base: int = 16):
        super().__init__()
        c1, c2, c3, cb = base, base * 2, base * 4, base * 8
        self.e1 = ConvBlock(in_ch, c1)
        self.e2 = ConvBlock(c1, c2)
        self.e3 = ConvBlock(c2, c3)
        self.pool = nn.MaxPool2d(2)
        self.bridge = ConvBlock(c3, cb)
        self.u3 = nn.ConvTranspose2d(cb, c3, 2, stride=2)
        self.d3 = ConvBlock(c3 * 2, c3)
        self.u2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.d2 = ConvBlock(c2 * 2, c2)
        self.u1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.d1 = ConvBlock(c1 * 2, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bridge(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.head(d1)


def build_model(input_size: int = 256) -> nn.Module:
    """与真工程 build_model 同签名(input_size 仅校验,网络全卷积不吃尺寸)。"""
    if int(input_size) < 64 or int(input_size) % 32 != 0:
        raise ValueError(f"input_size 需为 ≥64 的 32 倍数: {input_size}")
    return TinyUNet()


if __name__ == "__main__":
    m = build_model(256)
    n = sum(p.numel() for p in m.parameters())
    with torch.no_grad():
        y = m(torch.zeros(1, 3, 256, 256))
    print(f"TinyUNet params={n/1e6:.2f}M out={tuple(y.shape)}")
