from __future__ import annotations

import json
import math
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from common.logger import setup_logger
from engine.defect_size import calculate_defect_size
from engine.inference.base import BaseInferEngine, PostResult, PreparedInput
from engine.paths import Resolved
from engine.registry import ModelEntry
from schemas import Measurements

logger = setup_logger("supervised")

# 归一化常数（与训练 dataset.py / exam.py 逐位一致：灰度×3 → ImageNet mean/std）
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

DOMAIN_ZONE = "domain"
_TPL_CACHE_MAX = 16  # 标件(基准图)预备实例 LRU 缓存上限（按点位，约 2MB/点位）


def _read_gray(path: str) -> np.ndarray:
    """灰度读图（内存二进制流解码，免疫中文/Windows 路径；失败退 PIL）。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    except Exception:
        pass
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


def warn_once(seen: set, key: tuple, msg: str) -> None:
    """同一失败原因进程级只告警一次（每请求刷屏的告警等于没有告警）。"""
    if key in seen:
        return
    seen.add(key)
    logger.warning(msg)


class SupervisedInferEngine(BaseInferEngine):
    """v2 监督推理引擎（与训练考试 exam.py 同口径）。

    · 输入：灰度 → 复制 3 通道 → ImageNet 归一化；
    · 切块：tile=manifest.tile_size，stride=int(tile×0.75) 全图网格 + 边界 clamp
      （无 included_coords/无分区过滤/无补块——朴素考试网格）；
    · 拼接：帐篷权重窗加权平均（heat×win 累加 ÷ acc 累加，重叠带两窗和恒 1）；
    · 门控：训练同款逆向对齐 + ±10px 吸附 + 圆滑 = "有效检测区"
      （Point8DofAligner 解 H → align_and_compute_effective_zones(smooth_for_infer=True)）；
      失败四情形（总闸关/基准图缺/尺寸不符/配准不置信）→ 基准系朴素分区蒙版 + warn_once，
      与训练考卷"缺 zones_rle 回退朴素分区"完全同款；
    · 阈值：manifest.threshold(0.5)，SEG_THRESHOLD 显式设置时覆盖；
    · defect_class：命中轮廓 → 所在有效检测区分图（重叠 argmax）→ manifest 缺陷名
      以 " / " 拼接（Detail 级按分区名稳定排序去重并集；不落任何分区兜底 "NG"）；
    · 测量：HALCON 像素→世界平面，diameter < min_defect_diameter_mm 的轮廓连同
      测量值一并丢弃（全被滤则降级 OK；轮廓与 measurements 严格按序等长）。
    """

    def __init__(
        self,
        registry,
        gpu_lock,
        device: str = "cuda",
        *,
        threshold_override: Optional[float] = None,
        calib=None,
        min_defect_diameter_mm: float = 0.5,
        region_align: bool = True,
        amp_dtype: str = "bf16",
        batch_max: int = 24,
        gpu_mem_safety: float = 0.75,
    ):
        super().__init__(registry, gpu_lock, device,
                         amp_dtype=amp_dtype, batch_max=batch_max,
                         gpu_mem_safety=gpu_mem_safety)
        self.threshold_override = threshold_override
        self.calib = calib
        self.min_diameter = float(min_defect_diameter_mm)
        self.region_align = bool(region_align)
        self._warned: set = set()
        self._tpl_lock = threading.Lock()
        self._tpl_cache: "OrderedDict[tuple, object]" = OrderedDict()
        self._win_cache: dict = {}

    # ------------------------------------------------------------------ #
    # 备料（锁外）
    # ------------------------------------------------------------------ #
    def _tent_window(self, ts: int, stride: int, device) -> torch.Tensor:
        """帐篷窗 [ts,ts]：中心 1.0、四边在重叠带 (ts-stride) 内线性趋 0（exam.blend_window 同款）。"""
        key = (ts, stride, str(device))
        win = self._win_cache.get(key)
        if win is None:
            ramp = max(1, ts - stride)
            i = torch.arange(ts, dtype=torch.float32, device=device) + 0.5
            w1 = torch.clamp(torch.minimum(i / ramp, (ts - i) / ramp), max=1.0)
            win = w1[:, None] * w1[None, :]
            self._win_cache[key] = win
        return win

    @staticmethod
    def _grid(h: int, w: int, ts: int) -> list:
        """全图网格（exam.py:73-77 同款）：stride=int(ts*0.75)，边界 clamp 平移不补零。"""
        stride = int(ts * 0.75)
        xs = [min(i * stride, max(0, w - ts)) for i in range(int(math.ceil(w / stride)))]
        ys = [min(j * stride, max(0, h - ts)) for j in range(int(math.ceil(h / stride)))]
        return [(x, y) for y in ys for x in xs]

    @staticmethod
    def _rasterize_zone(zone_rle: dict, target_hw: tuple) -> np.ndarray:
        """RLE 栅格化为 bool 蒙版；RLE size 与现场图不符时 INTER_NEAREST 缩放（沿旧语义）。"""
        from vision_supervised.rle import rle_to_mask
        m = rle_to_mask(zone_rle) > 0
        if m.shape != target_hw:
            logger.warning(
                f"zone RLE size {m.shape} != 图像 {target_hw}，INTER_NEAREST resize 对齐"
            )
            m = cv2.resize(m.astype(np.uint8), (target_hw[1], target_hw[0]),
                           interpolation=cv2.INTER_NEAREST) > 0
        return m

    def _load_zones(self, resolved: Resolved, gray_hw: tuple):
        """读点位分区 json，栅格化 zones 分区（u8 0/255）+ domain 分区（bool）。"""
        point = resolved.point_meta
        zones_def: dict = point.get("zones") or {}
        json_rel = point.get("json")
        deploy = Path(resolved.deploy_dir)
        json_path = deploy / json_rel if json_rel else None
        if json_path is None or not json_path.is_file():
            warn_once(self._warned, (resolved.uid, "zone_json", resolved.point_id),
                      f"分区 json 缺失，region 门控退化为不门控 uid={resolved.uid} "
                      f"pid={resolved.point_id} path={json_path}")
            return {}, None

        with open(json_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        regions = {r.get("name"): r.get("rle") for r in data.get("regions", [])}

        zones_u8: dict = {}
        for name in zones_def:
            rle = regions.get(name)
            if not rle:
                warn_once(self._warned, (resolved.uid, "zone_missing", resolved.point_id, name),
                          f"分区 {name!r} 不在 json regions 内，跳过 uid={resolved.uid} "
                          f"pid={resolved.point_id}")
                continue
            zones_u8[name] = self._rasterize_zone(rle, gray_hw).astype(np.uint8) * 255

        domain = None
        if DOMAIN_ZONE in regions and regions[DOMAIN_ZONE]:
            domain = self._rasterize_zone(regions[DOMAIN_ZONE], gray_hw)
        return zones_u8, domain

    def _get_aligner(self, resolved: Resolved, ref_path: Path, domain: Optional[np.ndarray]):
        """基准图预备实例缓存（按 (path, mtime)，LRU 上限 16 位点）。"""
        try:
            mtime = os.stat(ref_path).st_mtime_ns
        except OSError:
            return None
        key = (str(ref_path), mtime)
        with self._tpl_lock:
            hit = self._tpl_cache.get(key)
            if hit is not None:
                self._tpl_cache.move_to_end(key)
                return hit
        from vision_supervised.align8dof import Point8DofAligner
        ref_gray = _read_gray(str(ref_path))
        aligner = Point8DofAligner(ref_gray, domain)
        with self._tpl_lock:
            self._tpl_cache[key] = (aligner, ref_gray.shape[:2])
            self._tpl_cache.move_to_end(key)
            while len(self._tpl_cache) > _TPL_CACHE_MAX:
                self._tpl_cache.popitem(last=False)
        return self._tpl_cache[key]

    def _compute_effective_zones(self, resolved: Resolved, zones_u8: dict,
                                 domain: Optional[np.ndarray], gray: np.ndarray) -> dict:
        """有效检测区：配准成功 → 投影+±10px吸附+圆滑(smooth_for_infer=True)；否则朴素分区。"""
        if zones_u8 and self.region_align:
            point = resolved.point_meta
            ref_rel = point.get("ref_image")
            ref_path = Path(resolved.deploy_dir) / ref_rel if ref_rel else None
            if ref_path is None or not ref_path.is_file():
                warn_once(self._warned, (resolved.uid, "tpl_missing", resolved.point_id),
                          f"基准图缺失，有效检测区退化为朴素分区 uid={resolved.uid} "
                          f"pid={resolved.point_id}（deploy 随包 {ref_rel} 为配准参照物）")
            else:
                cached = self._get_aligner(resolved, ref_path, domain)
                if cached is None:
                    warn_once(self._warned, (resolved.uid, "tpl_load", resolved.point_id),
                              f"基准图预备失败，退化为朴素分区 uid={resolved.uid} pid={resolved.point_id}")
                else:
                    aligner, ref_shape = cached
                    if gray.shape[:2] != tuple(ref_shape):
                        warn_once(self._warned, (resolved.uid, "tpl_shape", resolved.point_id),
                                  f"基准图尺寸 {ref_shape} 与现场图 {gray.shape[:2]} 不符，"
                                  f"退化为朴素分区 uid={resolved.uid} pid={resolved.point_id}")
                    else:
                        res = aligner.align(gray, warp_image=False)
                        if not res.get("ok"):
                            warn_once(self._warned, (resolved.uid, "align_fail", resolved.point_id),
                                      f"配准不置信({res.get('reason')})，退化为朴素分区 "
                                      f"uid={resolved.uid} pid={resolved.point_id}")
                        else:
                            from vision_supervised.align8dof import align_and_compute_effective_zones
                            eff = align_and_compute_effective_zones(
                                zones_u8, res["H"], gray, smooth_for_infer=True
                            )
                            logger.info(
                                f"[align] uid={resolved.uid} pid={resolved.point_id} "
                                f"inliers={res.get('inliers')} shift={res.get('max_shift_px')}px "
                                f"det={res.get('det')} ms={res.get('ms_total')}"
                            )
                            return eff
        elif zones_u8:
            warn_once(self._warned, (resolved.uid, "align_off"),
                      f"REGION_ALIGN=0，直接使用基准系朴素分区蒙版 uid={resolved.uid}")
        # 硬回退：朴素分区（与训练考卷"缺 zones_rle 回退朴素分区"同款）
        return {z: (m > 0) for z, m in zones_u8.items()}

    def _prepare(self, resolved: Resolved, entry: ModelEntry) -> PreparedInput:
        gray = _read_gray(resolved.image_path)
        h, w = gray.shape[:2]
        ts = int(resolved.tile_size)

        zones_u8, domain = self._load_zones(resolved, (h, w))
        eff_zones = self._compute_effective_zones(resolved, zones_u8, domain, gray)
        env = None
        if eff_zones:
            env = np.zeros((h, w), dtype=bool)
            for m in eff_zones.values():
                env |= m
        else:
            warn_once(self._warned, (resolved.uid, "no_zones", resolved.point_id),
                      f"该点位无可用分区蒙版，region 门控退化为不门控 uid={resolved.uid} "
                      f"pid={resolved.point_id}")

        coords = self._grid(h, w, ts)
        crops = np.empty((len(coords), ts, ts), dtype=np.uint8)
        for i, (x, y) in enumerate(coords):
            x0, y0 = int(x), int(y)
            patch = gray[y0:y0 + ts, x0:x0 + ts]
            if patch.shape != (ts, ts):  # 图比 tile 还小的极端情况：右/下零填充
                padded = np.zeros((ts, ts), dtype=np.uint8)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded
            crops[i] = patch

        # 灰度×3 → /255 → ImageNet 归一化（CPU float32，锁外完成）
        batch = torch.from_numpy(crops).float().div_(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
        batch = (batch - _MEAN) / _STD
        batch = batch.contiguous()

        threshold = resolved.threshold
        if self.threshold_override is not None:
            threshold = self.threshold_override

        return PreparedInput(
            batch=batch,
            patches=coords,
            extra={
                "gray_hw": (h, w),
                "eff_zones": eff_zones,      # {分区名: bool 蒙版（现场系）}
                "env": env,                  # 并集门控蒙版（现场系）或 None（不门控）
                "threshold": threshold,
                "ts": ts,
            },
        )

    def _per_item_bytes(self, prepared: PreparedInput) -> int:
        ts = prepared.extra["ts"]
        return ts * ts * 96 * self._act_bytes  # 监督经验系数 96（启发式，实测再调）

    # ------------------------------------------------------------------ #
    # 前向与锁内 GPU 后处理
    # ------------------------------------------------------------------ #
    def _forward(self, model, prepared: PreparedInput) -> torch.Tensor:
        return self._chunked_forward(model, prepared)  # [N, 1, ts, ts] fp32 logits

    def _postprocess_gpu(self, raw: torch.Tensor, prepared: PreparedInput,
                         resolved: Resolved) -> dict:
        device = raw.device
        ts = raw.shape[-1]
        stride = int(ts * 0.75)
        h, w = prepared.extra["gray_hw"]
        win = self._tent_window(ts, stride, device)

        heat = torch.zeros((h, w), dtype=torch.float32, device=device)
        acc = torch.zeros((h, w), dtype=torch.float32, device=device)
        for (x, y), logits_i in zip(prepared.patches, raw):
            prob = torch.sigmoid(logits_i[0])
            if prob.shape[-1] != ts:  # 全卷积兜底：输出尺寸≠输入时双线性对齐
                prob = torch.nn.functional.interpolate(
                    prob.unsqueeze(0).unsqueeze(0), size=(ts, ts),
                    mode="bilinear", align_corners=False,
                )[0, 0]
            heat[y:y + ts, x:x + ts] += prob * win
            acc[y:y + ts, x:x + ts] += win
        acc[acc == 0] = 1.0
        prob_full = heat / acc

        pred = prob_full > float(prepared.extra["threshold"])
        env = prepared.extra.get("env")
        if env is not None:
            pred = pred & torch.from_numpy(np.ascontiguousarray(env)).to(device)
        bin_mask = pred.to(torch.uint8).cpu().numpy()
        return {"bin": bin_mask}

    # ------------------------------------------------------------------ #
    # 锁外 CPU 后处理：轮廓 + defect_class 判类
    # ------------------------------------------------------------------ #
    def _postprocess_cpu(self, mid: dict, prepared: PreparedInput,
                         resolved: Resolved) -> PostResult:
        contours, _ = cv2.findContours(
            mid["bin"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        pts_list = [c.reshape(-1, 2).tolist() for c in contours]
        if not pts_list:
            return PostResult(ok=True, defect_class="OK", contours=[])

        eff_zones = prepared.extra.get("eff_zones") or {}
        zones_def: dict = resolved.point_meta.get("zones") or {}
        h, w = prepared.extra["gray_hw"]

        hit_zone_names: list = []       # 命中分区（稳定顺序：按分区名排序）
        has_orphan = False              # 有轮廓不落任何分区（理论不可能，门控已裁）
        for pts in pts_list:
            arr = np.asarray(pts, dtype=np.int32)
            x0 = max(0, int(arr[:, 0].min())); x1 = min(w, int(arr[:, 0].max()) + 1)
            y0 = max(0, int(arr[:, 1].min())); y1 = min(h, int(arr[:, 1].max()) + 1)
            if x1 <= x0 or y1 <= y0:
                has_orphan = True
                continue
            sub = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.fillPoly(sub, [arr - [x0, y0]], 1)
            sub = sub > 0
            best_zone, best_n = None, 0
            for z, zm in eff_zones.items():
                n = int(np.logical_and(sub, zm[y0:y1, x0:x1]).sum())
                if n > best_n:
                    best_zone, best_n = z, n
            if best_zone is None:
                has_orphan = True
            elif best_zone not in hit_zone_names:
                hit_zone_names.append(best_zone)

        names: list = []
        for z in sorted(hit_zone_names):
            for name in zones_def.get(z, []):
                if name and name not in names:
                    names.append(name)
        if has_orphan and "NG" not in names:
            names.append("NG")
        defect_class = " / ".join(names) if names else "NG"

        return PostResult(ok=defect_class == "OK", defect_class=defect_class,
                          contours=pts_list)

    # ------------------------------------------------------------------ #
    # 锁外测量：HALCON 毫米 + 直径过滤
    # ------------------------------------------------------------------ #
    def _measure(self, resolved: Resolved, post: PostResult,
                 prepared: PreparedInput) -> list:
        if not post.contours:
            return []
        if self.calib is None:
            # 保护性兜底：无量尺配置时保留轮廓、不带测量，不误判降级
            logger.warning(f"calib registry 缺失，measurements 置零 uid={resolved.uid}")
            return [Measurements() for _ in post.contours]

        cam_par, pose = self.calib.get(resolved.product_code)  # 缺参 → CalibNotFound(500)

        kept_contours: list = []
        measurements: list = []
        for pts in post.contours:
            arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
            try:
                peri = cv2.arcLength(arr, True)
                eps = max(1.0, peri * 0.01)  # 简化仅用于 HALCON 投影输入；回传仍用原始轮廓
                approx = cv2.approxPolyDP(arr, eps, True).reshape(-1, 2).tolist()
                diameter, area = calculate_defect_size(approx, cam_par, pose)
            except Exception as e:
                logger.warning(f"defect_size 单条计算异常(保留不过滤): {e}")
                kept_contours.append(pts)
                measurements.append(Measurements())
                continue
            if diameter < self.min_diameter:
                continue  # 直径过小：轮廓与测量值一并丢弃
            kept_contours.append(pts)
            measurements.append(Measurements(
                diameter_mm=round(float(diameter), 4),
                area_mm2=round(float(area), 4),
                radius_mm=round(float(diameter) / 2.0, 4),
            ))

        if kept_contours:
            post.contours = kept_contours
            return measurements
        # 全部被过滤 → 整件降级 OK
        post.contours = []
        post.defect_class = "OK"
        post.ok = True
        return []
