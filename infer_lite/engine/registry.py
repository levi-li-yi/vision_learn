from __future__ import annotations

import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch

from common.logger import setup_logger
from engine.loaders import load_weight
from engine.paths import ModelPathResolver, Resolved
from schemas import DetectType, ModelStatus

logger = setup_logger("registry")


@dataclass
class ModelEntry:
    uid: str
    type: DetectType
    ckpt_path: str
    ckpt_sig: tuple  # (路径, mtime_ns, size) —— v2 热更新签名
    manifest: dict
    load_time: datetime
    last_access: datetime
    in_gpu: bool = False
    model: Optional[torch.nn.Module] = field(default=None, repr=False)


def _ckpt_signature(ckpt_path: str) -> tuple:
    """v2 热更新签名 = (路径, mtime_ns, size)。

    best 文件名按分数命名（与 quality_score 脱钩）：重训后只要交集与轮次相同，
    文件名完全不变——仅比对路径会让旧权重无声续役，必须叠加 mtime/size。
    （发布规范：覆盖拷贝不得保留源 mtime，否则签名失效。）
    """
    st = os.stat(ckpt_path)
    return (ckpt_path, st.st_mtime_ns, st.st_size)


class ModelRegistry:
    """模型注册中心: 模型常驻 CPU/eval, 推理时经 acquire_gpu 上下文取放。

    - torch.load / build / 构造在注册锁**外**执行（double-check），首载不阻塞其它查询。
    - LRU 用 OrderedDict O(1)：命中 move_to_end，淘汰 popitem(last=False)。
    - acquire_gpu 热路径不 empty_cache（仅 _drop 时调用）。
    - _evict/_drop 跳过 in_gpu 项，规避推理中淘汰竞态。
    - 权重精度按 amp_dtype：fp16 旧模式半精度常驻；bf16（默认）/fp32 保持 fp32。
    """

    def __init__(
        self,
        model_num_limit: int = 100,
        device: str = "cuda",
        amp_dtype: str = "bf16",
    ):
        self.limit = model_num_limit
        self.device = device
        self.amp_dtype = amp_dtype
        self.models: "OrderedDict[str, ModelEntry]" = OrderedDict()
        self.lock = threading.RLock()

    def set_model_num(self, n: int) -> None:
        with self.lock:
            self.limit = n
            self._evict()

    def get_or_load(self, resolved: Resolved) -> ModelEntry:
        uid = resolved.uid
        sig = _ckpt_signature(resolved.ckpt_path)
        with self.lock:
            entry = self.models.get(uid)
            if entry is not None and entry.ckpt_sig == sig:
                entry.last_access = datetime.now()
                self.models.move_to_end(uid)
                return entry

        # ---- 注册锁外：昂贵的磁盘 IO + 构造 + 载权重 ----
        model = load_weight(
            resolved.type,
            resolved.ckpt_path,
            amp_dtype=self.amp_dtype,
            manifest=resolved.manifest,
        )
        now = datetime.now()
        new_entry = ModelEntry(
            uid=uid,
            type=resolved.type,
            ckpt_path=resolved.ckpt_path,
            ckpt_sig=sig,
            manifest=resolved.manifest,
            load_time=now,
            last_access=now,
            model=model,
        )

        with self.lock:
            # double-check：并发首载时首者获胜，丢弃本线程重复加载产物
            existing = self.models.get(uid)
            if existing is not None and existing.ckpt_sig == sig:
                existing.last_access = datetime.now()
                self.models.move_to_end(uid)
                return existing
            self.models[uid] = new_entry
            self.models.move_to_end(uid)
            self._evict()
            logger.info(f"model loaded uid={uid} ckpt={resolved.ckpt_path}")
            return new_entry

    def _evict(self) -> None:
        """调用方需持锁。从最旧端淘汰，跳过 in_gpu 项。"""
        while len(self.models) > self.limit:
            oldest_uid, entry = next(iter(self.models.items()))
            if entry.in_gpu:
                # 推理中：移到最近端，继续看下一个，避免淘汰活跃模型
                self.models.move_to_end(oldest_uid)
                if all(e.in_gpu for e in self.models.values()):
                    logger.warning(
                        f"evict aborted: all {len(self.models)} models in_gpu (limit={self.limit})"
                    )
                    return
                continue
            logger.info(f"evicting oldest model uid={oldest_uid}")
            self._drop(oldest_uid)

    def _drop(self, uid: str) -> None:
        entry = self.models.pop(uid, None)
        if entry is None:
            return
        if entry.in_gpu:
            self.models[uid] = entry
            logger.warning(f"skip drop: uid={uid} is in_gpu")
            return
        try:
            if entry.model is not None:
                del entry.model
                entry.model = None
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"drop uid={uid} cleanup error: {e}")

    def unload(self, uid: str) -> None:
        with self.lock:
            self._drop(uid)

    def clear(self) -> None:
        with self.lock:
            for uid in list(self.models.keys()):
                self._drop(uid)

    @contextmanager
    def acquire_gpu(self, entry: ModelEntry):
        """将模型迁至 GPU, 退出时迁回 CPU。调用方需在推理锁内使用。"""
        if entry.model is None:
            raise RuntimeError(f"model missing for uid={entry.uid}")
        entry.model.to(self.device)
        entry.in_gpu = True
        try:
            yield entry.model
        finally:
            entry.model.cpu()
            entry.in_gpu = False

    # ---- 监控查询 (只读, 不动模型) ----
    def _to_status(self, entry: Optional[ModelEntry], uid: str) -> ModelStatus:
        return ModelStatus(
            uid=uid,
            type=entry.type.value if entry else DetectType.SUPERVISED.value,
            ckpt_path=entry.ckpt_path if entry else "",
            registered=entry is not None,
            load_time=entry.load_time if entry else None,
            last_access=entry.last_access if entry else None,
            in_gpu=entry.in_gpu if entry else False,
        )

    def status(self, uid: str) -> ModelStatus:
        with self.lock:
            return self._to_status(self.models.get(uid), uid)

    def list_all(self) -> list[ModelStatus]:
        with self.lock:
            return [self._to_status(e, e.uid) for e in self.models.values()]

    def status_by_fields(self, product_code: str) -> ModelStatus:
        return self.status(ModelPathResolver.build_uid(product_code))
