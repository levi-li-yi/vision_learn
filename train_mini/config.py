#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""config.py —— 全局配置(config/app_config.jsonc,支持 // 注释)。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from io_utils import load_jsonc

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "app_config.jsonc")


@dataclass
class AppConfig:
    seed: int = 20260927
    tile_size: int = 256          # 迷你切块(真工程 512;256 使训练+推理 <1GB)
    epochs: int = 20              # 与真工程同:上限 20,练满 8 轮后早停
    min_epochs: int = 8
    early_stop_patience: int = 4
    batch_size: int = 4           # 2GB 显存下 256 片的安全批
    lr: float = 3e-4
    weight_decay: float = 1e-4
    dice_weight: float = 3.0      # 与真工程同:Focal + Dice×3
    focal_alpha: float = 0.4
    focal_gamma: float = 2.0
    amp: str = "auto"             # auto=bf16(支持时)否则 fp16;可强制 bf16/fp16/fp32
    device: str = "auto"          # auto=cuda(可用时)否则 cpu
    data_dir: str = "datasets"    # datasets/{产品}/{点位}/
    models_dir: str = "models"    # models/{产品}/best_*.pth + deploy/
    val_ratio: float = 0.2        # 训练池 8:2 切早停验证(固定种子)
    paste_dilate_min: int = 5     # 泊松克隆前掩码外扩 px(盖过 OpenCV 内部 4px 腐蚀)
    paste_dilate_max: int = 7

    @staticmethod
    def load(path: str = DEFAULT_CONFIG) -> "AppConfig":
        cfg = AppConfig()
        if os.path.isfile(path):
            for k, v in load_jsonc(path).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg
