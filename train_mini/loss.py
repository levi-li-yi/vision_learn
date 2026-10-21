#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""loss.py —— Sigmoid Focal + Dice(与真工程"稠密 Focal+Dice 统一"同权重:α=0.4, γ=2.0, Dice×3)。

标签口径:缺陷=1 / 正常=0 / 忽略=255(检测区外不计误差)。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE = 255


def focal_loss(logits: torch.Tensor, target: torch.Tensor,
               alpha: float = 0.4, gamma: float = 2.0) -> torch.Tensor:
    valid = target != IGNORE
    if not valid.any():
        return logits.sum() * 0.0
    lg = logits[valid]
    tg = target[valid].float()
    p = torch.sigmoid(lg)
    ce = F.binary_cross_entropy_with_logits(lg, tg, reduction="none")
    pt = p * tg + (1 - p) * (1 - tg)
    alpha_t = alpha * tg + (1 - alpha) * (1 - tg)
    return (alpha_t * (1 - pt).pow(gamma) * ce).mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != IGNORE
    if not valid.any():
        return logits.sum() * 0.0
    lg = logits[valid].unsqueeze(0)
    tg = target[valid].float().unsqueeze(0)
    p = torch.sigmoid(lg)
    inter = (p * tg).sum()
    denom = p.sum() + tg.sum() + 1e-6
    return 1.0 - 2.0 * inter / denom


def seg_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 3.0,
             focal_alpha: float = 0.4, focal_gamma: float = 2.0) -> torch.Tensor:
    return focal_loss(logits, target, focal_alpha, focal_gamma) \
        + dice_weight * dice_loss(logits, target)
