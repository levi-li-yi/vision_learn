from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from common.logger import setup_logger
from engine.paths import Resolved
from engine.registry import ModelEntry, ModelRegistry
from schemas import Detail

logger = setup_logger("infer")


@dataclass
class Timings:
    prepare_ms: float = 0.0
    migrate_ms: float = 0.0
    forward_ms: float = 0.0
    postprocess_ms: float = 0.0  # 锁内 GPU 后处理
    post_cpu_ms: float = 0.0     # 锁外 CPU 后处理（轮廓/判类）
    measure_ms: float = 0.0      # 锁外 HALCON 尺寸

    @property
    def lock_ms(self) -> float:
        """串行锁窗口 = migrate + forward + postprocess（吞吐优化核心指标）。"""
        return self.migrate_ms + self.forward_ms + self.postprocess_ms

    @property
    def total_ms(self) -> float:
        return (self.prepare_ms + self.lock_ms + self.post_cpu_ms + self.measure_ms)


@dataclass
class PreparedInput:
    """锁外备料产物：CPU float32 归一化批次 + 切片坐标 + 引擎私有中间量。"""

    batch: torch.Tensor                    # [N, 3, tile, tile] float32（已归一化）
    patches: list                          # [(x, y), ...] 全图网格坐标（clamp）
    extra: dict = field(default_factory=dict)


@dataclass
class PostResult:
    ok: bool = True
    defect_class: str = "OK"
    contours: list = field(default_factory=list)  # 全图绝对坐标 [[[x,y],...], ...]


class BaseInferEngine:
    """模板方法：prepare（锁外）→ 锁内（迁移/前向/GPU 后处理）→ 锁外（CPU 后处理/测量）。

    分界线唯一原则：**碰 GPU 张量的活都在锁内**，且 _postprocess_gpu 结束前把全部
    中间结果搬回 CPU——出锁那一刻起，后续所有活都与 GPU 无关。
    """

    def __init__(
        self,
        registry: ModelRegistry,
        gpu_lock,
        device: str = "cuda",
        amp_dtype: str = "bf16",
        batch_max: int = 24,
        gpu_mem_safety: float = 0.75,
    ):
        self.registry = registry
        self.gpu_lock = gpu_lock
        self.device = device
        self.amp_dtype = amp_dtype
        self.batch_max = int(batch_max)
        self.gpu_mem_safety = float(gpu_mem_safety)
        self._act_bytes = 2 if amp_dtype in ("bf16", "fp16") else 4

    # ------------------------------------------------------------------ #
    # 子类接口
    def _prepare(self, resolved: Resolved, entry: ModelEntry) -> PreparedInput:
        raise NotImplementedError

    def _forward(self, model, prepared: PreparedInput) -> Any:
        raise NotImplementedError

    def _postprocess_gpu(self, raw: Any, prepared: PreparedInput, resolved: Resolved) -> dict:
        """锁内：依赖 GPU 张量的后处理；返回纯 CPU 中间产物。"""
        raise NotImplementedError

    def _postprocess_cpu(self, mid: dict, prepared: PreparedInput, resolved: Resolved) -> PostResult:
        raise NotImplementedError

    def _measure(self, resolved: Resolved, post: PostResult, prepared: PreparedInput) -> list:
        """锁外：尺寸测量；返回 measurements（与 post.contours 按序等长）。"""
        return []

    def _per_item_bytes(self, prepared: PreparedInput) -> int:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    def _chunked_forward(self, model, prepared: PreparedInput) -> torch.Tensor:
        """动态分块批量前向：批大小按当前空闲显存现算，夹在 [1, batch_max]。

        chunk = clamp( floor(空闲显存 × gpu_mem_safety ÷ 每片激活占用), 1, batch_max, N )
        各块结果 .float() 回 fp32 后按序拼接。
        """
        batch = prepared.batch
        n = batch.shape[0]
        per_item = max(1, self._per_item_bytes(prepared))
        chunk = n
        if self.device.startswith("cuda") and self.batch_max > 1 and n > 1:
            try:
                free, _total = torch.cuda.mem_get_info()
                chunk = int(free * self.gpu_mem_safety // per_item)
            except Exception:
                chunk = self.batch_max
            chunk = max(1, min(chunk, self.batch_max, n))
        outs = []
        for s in range(0, n, chunk):
            piece = batch[s:s + chunk].to(self.device, non_blocking=False)
            outs.append(model(piece).float())
        return torch.cat(outs, dim=0)

    # ------------------------------------------------------------------ #
    def infer(self, resolved: Resolved, entry: ModelEntry) -> list[Detail]:
        import os
        if not os.path.isfile(resolved.image_path):
            from common.exceptions import InputImageNotFound
            raise InputImageNotFound(os.path.dirname(resolved.image_path))

        t = Timings()
        cuda = self.device.startswith("cuda")
        if cuda:
            torch.cuda.reset_peak_memory_stats()

        # ---- ① 锁外备料（读图/网格裁片/归一化/有效检测区配准） ----
        t0 = time.perf_counter()
        prepared = self._prepare(resolved, entry)
        t.prepare_ms = (time.perf_counter() - t0) * 1000

        if len(prepared.patches) == 0 or prepared.batch.shape[0] == 0:
            # 无有效切片（如点位未绑定任何分区）：单条 OK 占位，避免空 forward
            return [Detail(task_id=resolved.task_id, type=resolved.type.value,
                           defect_class="OK", global_contours=[], measurements=[])]

        # ---- ② 锁内：迁移 → 前向（autocast）→ GPU 后处理（结束前全部搬回 CPU） ----
        with self.gpu_lock:
            t0 = time.perf_counter()
            with self.registry.acquire_gpu(entry) as model:
                t.migrate_ms = (time.perf_counter() - t0) * 1000
                with torch.inference_mode():
                    t0 = time.perf_counter()
                    with torch.autocast(
                        device_type="cuda" if cuda else "cpu",
                        dtype=torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16,
                        enabled=self.amp_dtype != "fp32" and cuda,
                    ):
                        raw = self._forward(model, prepared)
                    t.forward_ms = (time.perf_counter() - t0) * 1000

                    t0 = time.perf_counter()
                    mid = self._postprocess_gpu(raw, prepared, resolved)
                    t.postprocess_ms = (time.perf_counter() - t0) * 1000
            # acquire_gpu 退出（模型回 CPU）；锁内窗口到此为止

        # ---- ③ 锁外 CPU 后处理（轮廓/defect_class 判类） ----
        t0 = time.perf_counter()
        post = self._postprocess_cpu(mid, prepared, resolved)
        t.post_cpu_ms = (time.perf_counter() - t0) * 1000

        # ---- ④ 锁外尺寸测量（HALCON 像素→世界，CPU 密集） ----
        t0 = time.perf_counter()
        measurements = self._measure(resolved, post, prepared)
        t.measure_ms = (time.perf_counter() - t0) * 1000

        peak = 0.0
        if cuda:
            try:
                peak = torch.cuda.max_memory_allocated() / 1e6
            except Exception:
                pass
        logger.info(
            f"[CT] uid={resolved.uid} task={resolved.task_id} "
            f"prepare={t.prepare_ms:.1f} migrate={t.migrate_ms:.1f} forward={t.forward_ms:.1f} "
            f"post_gpu={t.postprocess_ms:.1f} post_cpu={t.post_cpu_ms:.1f} measure={t.measure_ms:.1f} "
            f"lock={t.lock_ms:.1f} total={t.total_ms:.1f}ms peak_gpu={peak:.0f}MB"
        )
        return [Detail(
            task_id=resolved.task_id,
            type=resolved.type.value,
            defect_class=post.defect_class,
            global_contours=post.contours,
            measurements=measurements,
        )]
