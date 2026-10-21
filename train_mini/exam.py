#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exam.py —— 独立考卷整图推理与择优交付(真工程 exam 的迷你同构,契约逐字段一致)。

· 推理:原始 .bmp 上全图网格(stride=0.75×tile,clamp)+ 帐篷权重窗加权平均
  (heat×win ÷ acc)+ 投影分区并集门控 + 阈值 0.5 —— 与 infer_lite 推理逐口径同款;
· 计分:总交集(报警∩标注∩分区)、逐缺陷命中数(hit)、报警面积(alarm);
· 选型:quality_score = hit×1000 + 交集 − 0.1×误报面积(与真工程 dev 版同公式),
  新高超旧才刷新唯一 best_int{交集}_ep{轮:02d}.pth;
· 交付:models/{产品}/ + deploy/(manifest.json/tile_size.txt/{pid}/{基准图名}.json+bmp),
  manifest 字段与真工程完全一致(weight/tile_size/threshold/points…zones)。
"""

from __future__ import annotations

import os
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from budget import grid_coords
from io_utils import load_polygons, read_gray
from zones import PointMat

THRESHOLD = 0.5


def blend_window(ts: int, stride: int, device) -> torch.Tensor:
    """帐篷窗(与真工程 exam.blend_window 逐公式一致)。"""
    ramp = max(1, ts - stride)
    i = torch.arange(ts, dtype=torch.float32, device=device) + 0.5
    w1 = torch.clamp(torch.minimum(i / ramp, (ts - i) / ramp), max=1.0)
    return w1[:, None] * w1[None, :]


def predict_heatmap(model, gray: np.ndarray, zone_union: np.ndarray, ts: int,
                    device: str, batch_size: int = 8) -> np.ndarray:
    """整图切块推理 + 帐篷窗加权拼接 + 分区门控,返回 (heat/acc × zone) [H,W]。"""
    h, w = gray.shape[:2]
    coords = grid_coords(h, w, ts)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    win = blend_window(ts, int(ts * 0.75), device)
    gray_t = torch.from_numpy(gray).to(device)

    heat = torch.zeros((h, w), dtype=torch.float32, device=device)
    acc = torch.zeros((h, w), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(coords), batch_size):
            chunk = coords[s:s + batch_size]
            t = torch.stack([
                gray_t[y:y + ts, x:x + ts].repeat(3, 1, 1) for (x, y) in chunk
            ]).float().div_(255.0)
            t = (t - mean) / std
            logits = model(t)
            if logits.shape[-1] != ts:
                logits = torch.nn.functional.interpolate(
                    logits, size=(ts, ts), mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits).squeeze(1).float()
            for (x, y), prob in zip(chunk, probs):
                heat[y:y + ts, x:x + ts] += prob * win
                acc[y:y + ts, x:x + ts] += win
    acc[acc == 0] = 1.0
    zone_t = torch.from_numpy(np.ascontiguousarray(zone_union)).to(device)
    return (heat / acc * zone_t).cpu().numpy()


def run_exam(model, mats: List[PointMat], eff_by_img: Dict[str, Dict[str, np.ndarray]],
             ts: int, device: str) -> dict:
    """全部点位全部 val 考卷跑一遍 → {total, hit_defects, total_defects, alarm, per_point}。"""
    total = 0
    hit_defects = 0
    total_defects = 0
    alarm = 0
    per_point: Dict[str, int] = {}
    for mat in mats:
        for path in mat.val_paths:
            gray = read_gray(path)
            eff = eff_by_img.get(path) or {z: mat.zones[z] for z in mat.detect_zones}
            zone_union = np.zeros_like(eff[mat.detect_zones[0]])
            for zm in eff.values():
                zone_union |= zm
            heat = predict_heatmap(model, gray, zone_union, ts, device)
            binary = heat > THRESHOLD
            alarm += int(binary.sum())
            ann = np.zeros(gray.shape, bool)
            polys = load_polygons(path, defect_only=True)
            for pts in polys:
                cv2.fillPoly(ann, [pts], 1)
            ann &= zone_union
            total += int((binary & ann).sum())
            for pts in polys:
                sub = np.zeros(gray.shape, np.uint8)
                cv2.fillPoly(sub, [pts], 1)
                pm = (sub > 0) & zone_union
                if not pm.any():
                    pm = sub > 0
                total_defects += 1
                if (binary & pm).any():
                    hit_defects += 1
            per_point[mat.name] = per_point.get(mat.name, 0) + int((binary & ann).sum())
    fp_area = max(0, alarm - total)
    quality = hit_defects * 1000 + total - int(round(0.10 * fp_area))
    return {"total": total, "hit_defects": hit_defects, "total_defects": total_defects,
            "alarm": alarm, "quality_score": quality, "per_point": per_point}


def save_best(models_dir: str, product: str, res: dict, epoch: int,
              model, mats: List[PointMat], zone_defects: Dict[str, List[str]],
              ts: int, prev_best: Tuple[int, int, int]) -> Optional[str]:
    """新高超旧 → 删旧存新 + 整体重建 deploy/(v2 契约)。返回文件名或 None。"""
    cur = (int(res["quality_score"]), int(res["hit_defects"]), int(res["total"]))
    if res["total"] <= 0 or cur <= prev_best:
        return None
    ckpt_dir = os.path.join(models_dir, product)
    os.makedirs(ckpt_dir, exist_ok=True)
    for f in os.listdir(ckpt_dir):
        if f.startswith("best_int") and f.endswith(".pth"):
            os.remove(os.path.join(ckpt_dir, f))
    name = f"best_int{int(res['total'])}_ep{epoch:02d}.pth"
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"weight": sd, "step": -1, "epoch": epoch,
                "intersection": int(res["total"]),
                "hit_defects": int(res["hit_defects"]),
                "quality_score": int(res["quality_score"])},
               os.path.join(ckpt_dir, name))

    deploy = os.path.join(ckpt_dir, "deploy")
    if os.path.isdir(deploy):
        shutil.rmtree(deploy)
    os.makedirs(deploy, exist_ok=True)
    manifest = {"weight": name, "tile_size": int(ts), "threshold": THRESHOLD, "points": {}}
    for mat in mats:
        pd = os.path.join(deploy, mat.name)
        os.makedirs(pd, exist_ok=True)
        shutil.copy2(os.path.join(mat.dir, mat.ref_name + ".json"), pd)
        shutil.copy2(os.path.join(mat.dir, mat.ref_name + ".bmp"), pd)
        manifest["points"][mat.name] = {
            "ref_name": mat.ref_name,
            "json": f"{mat.name}/{mat.ref_name}.json",
            "ref_image": f"{mat.name}/{mat.ref_name}.bmp",
            "zones": {z: list(defects) for z, defects in zone_defects.items()},
        }
    with open(os.path.join(deploy, "tile_size.txt"), "w") as f:
        f.write(str(ts))
    from io_utils import save_json
    save_json(manifest, os.path.join(deploy, "manifest.json"))
    return name
