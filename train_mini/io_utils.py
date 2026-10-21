#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""io_utils.py —— 图像/标注/分区读写(中文路径安全)。

.tmp 标注格式与 define_masks 分区 json 格式,均与真训练仓(supervised_apps/train)
逐字段一致(checkup.py 同款解析),保证学习工程可直接消费现场标注工具的产物。
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Tuple

import cv2
import numpy as np

cv2.setNumThreads(0)

DOMAIN = "domain"        # 对齐依据分区(每点位必备)
EXCLUDE_LABEL = "排除"   # .tmp 中非缺陷的排除区标注
VAL_DIRNAME = "val"      # 独立考卷子目录(100% 物理隔离)

_BMP_EXTS = (".bmp",)


# ------------------------------------------------------------------ #
# JSONC(支持 // 注释)与 JSON
# ------------------------------------------------------------------ #
def load_jsonc(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read()
    text = re.sub(r"//[^\n]*", "", text)
    return json.loads(text)


def load_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ #
# 图像(内存二进制流解码,免疫中文路径)
# ------------------------------------------------------------------ #
def read_gray(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"图像读取失败: {path}")
    return img


def write_image(path: str, img: np.ndarray) -> None:
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"图像编码失败: {path}")
    buf.tofile(path)


def image_size(path: str) -> Tuple[int, int]:
    """(H, W) —— 只读头(PIL 惰性解码)。"""
    from PIL import Image
    with Image.open(path) as im:
        return im.height, im.width


# ------------------------------------------------------------------ #
# 分区 RLE(define_masks {基准图名}.json 同款)
# ------------------------------------------------------------------ #
def rle_to_mask(rle: dict) -> np.ndarray:
    m = np.zeros(rle["size"], np.uint8)
    r = rle["runs"]
    for i in range(0, len(r), 3):
        y, s, e = r[i], r[i + 1], r[i + 2]
        m[y, s:e + 1] = 1
    return m


def mask_to_rle(mask: np.ndarray) -> dict:
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


def load_zones(json_path: str) -> dict:
    """分区 json → {分区名: bool 蒙版(基准图坐标系)}。"""
    data = load_json(json_path)
    return {r["name"]: rle_to_mask(r["rle"]) > 0 for r in data.get("regions", [])}


# ------------------------------------------------------------------ #
# .tmp 标注(checkup.load_polygons 同款格式)
# ------------------------------------------------------------------ #
def load_polygons(img_path: str, defect_only: bool = False) -> List[np.ndarray]:
    """`.bmp.tmp` → [np.array(N,2) int32](xy 坐标,原图坐标系)。

    label == "排除" → 非缺陷排除区;其余 label(如"砂眼"/"划痕")→ 真实缺陷。
    """
    tmp = img_path + ".tmp"
    if not os.path.isfile(tmp):
        return []
    data = load_json(tmp)
    polys = []
    for ent in data:
        if ent.get("type") != "poly":
            continue
        lbl = str(ent.get("label") or "").strip()
        if not lbl:
            continue
        if defect_only and lbl == EXCLUDE_LABEL:
            continue
        pts = np.asarray(ent.get("coord") or [], np.int32)
        if len(pts) >= 3:
            polys.append(pts)
    return polys


def load_polygons_with_labels(img_path: str) -> List[Tuple[str, np.ndarray]]:
    tmp = img_path + ".tmp"
    if not os.path.isfile(tmp):
        return []
    data = load_json(tmp)
    out = []
    for ent in data:
        if ent.get("type") != "poly":
            continue
        lbl = str(ent.get("label") or "").strip()
        pts = ent.get("coord") or []
        if lbl and len(pts) >= 3:
            arr = np.asarray(pts, dtype=np.int32)
            if arr.ndim == 2 and arr.shape[1] == 2:
                out.append((lbl, arr))
    return out
