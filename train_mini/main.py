#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""main.py —— 迷你同构训练事件线(真工程五段事件线的完整缩影,无增量缓存从简)。

  0 体检+索引   scan_point:分区 json/基准图/domain/同分辨率/考卷隔离
  1 逆向对齐    OrbAligner(基准↔每张原图)→ 分区投影到原图坐标系(图不动,蒙版动)
  2 真实缺陷小图  原图零失真抠取(外扩 8px,掩码=多边形∩投影分区)+ zone↔defect 归属
  3 干净切块    全图网格(0.75 步长)剔除含标注脏块,保留命中分区块;训练池 8:2 切早停验证
  4 在线贴图训练  每 epoch 正负 1:1 泊松贴题(Focal+Dice×3,AMP,EMA 从简);
                val_loss 练满 8 轮后连续 4 轮不降早停
  5 独立考卷    val/ 原图整图推理(帐篷窗+门控+0.5)→ 交集/hit/quality_score
                → 新高超旧刷新唯一 best + deploy/(v2 契约,infer_lite 可直接服务)

用法:
  python main.py --product HT12 [--epochs 20] [--data datasets] [--models models]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

import model as model_mod
from budget import build_tiles
from config import AppConfig
from dataset import OnlinePasteDataset, build_fixed_val, normalize_gray3
from exam import run_exam, save_best
from io_utils import read_gray
from loss import seg_loss
from patches import DefectPatch, extract_patches
from runtime_state import Monitor
from zones import PointMat, effective_zones, scan_point, zone_defect_map
from align import OrbAligner


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_device(cfg: AppConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_amp(cfg: AppConfig, device: torch.device) -> str:
    if cfg.amp != "auto":
        return cfg.amp
    if device.type != "cuda":
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


@torch.no_grad()
def val_loss(model, val_set, device) -> float:
    model.eval()
    losses = []
    for img, label in val_set:
        x = torch.from_numpy(img).unsqueeze(0).to(device)
        y = torch.from_numpy(label).unsqueeze(0).to(device)
        logits = model(x)
        losses.append(seg_loss(logits, y).item())
    model.train()
    return float(np.mean(losses)) if losses else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--models", default=None)
    args = ap.parse_args()

    cfg = AppConfig.load()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.data:
        cfg.data_dir = args.data
    if args.models:
        cfg.models_dir = args.models
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = pick_device(cfg)
    amp = pick_amp(cfg, device)
    ts = int(cfg.tile_size)
    mon = Monitor(args.product, {"tile_size": ts, "batch_size": cfg.batch_size,
                                 "epochs": cfg.epochs, "device": str(device),
                                 "amp": amp, "seed": cfg.seed})

    def emit(msg: str):
        """控制台 log + 监控状态文件双写(web.py 据此渲染事件线)。"""
        log(msg)
        mon.event(msg)

    emit(f"配置: tile={ts} batch={cfg.batch_size} device={device} amp={amp} seed={cfg.seed}")
    emit(f"监控: 另开终端 python web.py --product {args.product} → "
         f"http://127.0.0.1:{cfg.web_port}")

    # ---- 0 体检+索引 ----
    prod_root = os.path.join(cfg.data_dir, args.product)
    if not os.path.isdir(prod_root):
        raise SystemExit(f"产品目录不存在: {prod_root}")
    mats: List[PointMat] = [scan_point(os.path.join(prod_root, d))
                            for d in sorted(os.listdir(prod_root))
                            if os.path.isdir(os.path.join(prod_root, d))
                            and not d.startswith(".") and d not in ("val", "assets")]
    total_val = sum(len(m.val_paths) for m in mats)
    emit(f"0.体检: {len(mats)} 个点位, 训练池 {sum(len(m.train_paths) for m in mats)} 张, "
         f"独立考卷 {total_val} 张")

    # ---- 1 逆向对齐 + 分区投影;2 真实缺陷小图 ----
    patches: List[DefectPatch] = []
    eff_by_img: Dict[str, Dict[str, np.ndarray]] = {}
    for mat in mats:
        ref_gray = read_gray(mat.ref_path)
        aligner = OrbAligner(ref_gray, mat.zones.get("domain"))
        for p in mat.train_paths + mat.val_paths:
            gray = read_gray(p)
            eff = effective_zones(mat, gray, aligner)
            eff_by_img[p] = eff
            if p in mat.train_paths:
                patches.extend(extract_patches(p, gray, eff))
    if not patches:
        raise SystemExit("未抽到任何真实缺陷小图(检查 .tmp 标注与分区归属)")
    emit(f"1.对齐+投影: {len(eff_by_img)} 张图; 2.真实缺陷小图: {len(patches)} 个")

    zdef = zone_defect_map(mats, eff_by_img)
    emit(f"   分区↔缺陷归属: {json.dumps(zdef, ensure_ascii=False)}")

    # ---- 3 干净切块(训练池 8:2) ----
    all_tiles = []
    for mat in mats:
        for p in mat.train_paths:
            all_tiles.extend(build_tiles(p, read_gray(p), eff_by_img[p], ts))
    rng = random.Random(cfg.seed)
    rng.shuffle(all_tiles)
    n_val = max(1, int(len(all_tiles) * cfg.val_ratio)) if len(all_tiles) > 1 else 0
    val_tiles, train_tiles = all_tiles[:n_val], all_tiles[n_val:]
    if not train_tiles:
        raise SystemExit("干净块池为空(标注覆盖全图?)")
    emit(f"3.切块: 干净块 {len(all_tiles)}(训练 {len(train_tiles)} / 早停验证 {len(val_tiles)})")

    # ---- 4 在线贴图训练 ----
    net = model_mod.build_model(ts).to(device)
    n_params = sum(p_.numel() for p_ in net.parameters())
    emit(f"4.TinyUNet 参数 {n_params/1e6:.2f}M")
    train_ds = OnlinePasteDataset(train_tiles, patches, seed=cfg.seed + 1,
                                  dilate_min=cfg.paste_dilate_min,
                                  dilate_max=cfg.paste_dilate_max)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=0, drop_last=True)
    val_set = build_fixed_val(val_tiles or train_tiles[:4], patches,
                              seed=cfg.seed + 2, n=16)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16" and device.type == "cuda"))

    best_prev = (-10**9, 0, 0)
    best_name = None
    no_gain = 0
    last_vl = float("inf")
    stop_reason = "reached_epoch_limit"
    for epoch in range(1, cfg.epochs + 1):
        net.train()
        running = []
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16 if amp == "bf16" else torch.float16,
                                enabled=amp != "fp32"):
                logits = net(imgs)
                loss = seg_loss(logits, labels, cfg.dice_weight,
                                cfg.focal_alpha, cfg.focal_gamma)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running.append(loss.item())
        vl = val_loss(net, val_set, device)
        emit(f"4.epoch {epoch:02d}: train_loss={np.mean(running):.4f} val_loss={vl:.4f}")
        rec = {"epoch": epoch,
               "train_loss": round(float(np.mean(running)), 4),
               "val_loss": round(vl, 4)}

        if total_val > 0:
            res = run_exam(net, mats, eff_by_img, ts, device)
            emit(f"5.考试 ep{epoch:02d}: 交集={res['total']} 检出={res['hit_defects']}/"
                 f"{res['total_defects']} 报警面积={res['alarm']} "
                 f"quality={res['quality_score']}")
            rec.update(inter=res["total"], hit=res["hit_defects"],
                       hit_total=res["total_defects"], alarm=res["alarm"],
                       quality=res["quality_score"])
            name = save_best(cfg.models_dir, args.product, res, epoch, net,
                             mats, zdef, ts, best_prev)
            if name:
                best_prev = (res["quality_score"], res["hit_defects"], res["total"])
                best_name = name
                mon.best(name, res["quality_score"])
                emit(f"5.★ best 刷新: {name}")
        mon.epoch(rec)
        if epoch >= cfg.min_epochs:
            if vl >= last_vl - 1e-4:
                no_gain += 1
            else:
                no_gain = 0
            if no_gain >= cfg.early_stop_patience:
                stop_reason = f"early_stop(ep{epoch}, 连续{no_gain}轮不降)"
                break
        last_vl = vl

    emit(f"训练结束({stop_reason}) best={best_name or '无(总交集恒0,请检查数据/标注)'}")
    emit(f"交付目录: {os.path.join(cfg.models_dir, args.product)} → 拷入 infer_lite 的 "
         f"{{SUPERVISED_CKPT_DIR}}/{{产品}}/ 即可服务")
    mon.finish(stop_reason)


if __name__ == "__main__":
    main()
