#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rle.py —— define_masks 分区 RLE 编解码(训练/推理共享,★ D5 同一套)。

逐字提取自训练仓 supervised_apps/train 的 src/train/io_utils.py @ origin/dev(3cdfdcb)。
RLE 格式: {"size": [H, W], "runs": [y, x_start, x_end, ...]} 逐行行程。
"""

from __future__ import annotations

import numpy as np


def rle_to_mask(rle: dict) -> np.ndarray:
    """RLE {size, runs=[y, x_start, x_end, ...]} → 二值蒙版 uint8 0/1

    define_masks {基准图名}.json 分区存储格式, 逐行行程解码。
    """
    m = np.zeros(rle["size"], np.uint8)
    r = rle["runs"]
    for i in range(0, len(r), 3):
        y, s, e = r[i], r[i + 1], r[i + 2]
        m[y, s:e + 1] = 1
    return m


def mask_to_rle(mask: np.ndarray) -> dict:
    """二值蒙版 (bool/uint8) → RLE {size: [H, W], runs: [y, x_start, x_end, ...]}"""
    h, w = mask.shape[:2]
    b = (mask > 0).astype(np.uint8)
    pad = np.zeros((h, w + 2), dtype=np.uint8)
    pad[:, 1:-1] = b
    diff = pad[:, 1:] != pad[:, :-1]
    ys, xs = np.nonzero(diff)
    runs = np.empty((len(ys) // 2) * 3, dtype=np.int32)
    runs[0::3] = ys[0::2]
    runs[1::3] = xs[0::2]
    runs[2::3] = xs[1::2] - 1
    return {"size": [int(h), int(w)], "runs": runs.tolist()}
