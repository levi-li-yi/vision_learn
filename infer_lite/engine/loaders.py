from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

from common.exceptions import CkptKeyError, ModelBuildError, StateDictMismatch
from common.logger import setup_logger
from engine import frameworks
from schemas import DetectType

logger = setup_logger("loaders")

# v2 ckpt 必备键（训练 exam._save_best 产出：{"weight", "step", "epoch",
# "intersection", "hit_defects", "quality_score"}；weight 为 EMA 权重，其余为附加键，容忍）
_SUPERVISED_REQUIRED_KEYS = ("weight",)

#: 程序化注入自定义构造函数（高级用法，绕开 frameworks 默认导入）
_BUILDERS: Dict[DetectType, Optional[Callable]] = {DetectType.SUPERVISED: None}


def register_builder(det_type: DetectType, builder: Callable) -> None:
    _BUILDERS[det_type] = builder


def _default_build(det_type: DetectType, input_size: int):
    return frameworks.build_supervised_segmenter(input_size=input_size)


def _load_ckpt(ckpt_path: str) -> dict:
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise ModelBuildError(f"torch.load 失败: {ckpt_path}: {e}")


def _apply_half(model: torch.nn.Module) -> None:
    """fp16 旧模式：仅半化 parameters（buffer 不动——统计量半化会漂移口径）。"""
    for p in model.parameters():
        p.data = p.data.half()


def load_supervised(ckpt_path: str, amp_dtype: str = "bf16", manifest: Optional[dict] = None):
    manifest = manifest or {}
    try:
        tile_size = int(manifest.get("tile_size") or 512)
    except (TypeError, ValueError):
        tile_size = 512

    builder = _BUILDERS[DetectType.SUPERVISED]
    model = builder(input_size=tile_size) if builder else _default_build(DetectType.SUPERVISED, tile_size)

    ckpt = _load_ckpt(ckpt_path)
    if not isinstance(ckpt, dict) or any(k not in ckpt for k in _SUPERVISED_REQUIRED_KEYS):
        raise CkptKeyError(ckpt_path, list(_SUPERVISED_REQUIRED_KEYS))
    try:
        model.load_state_dict(ckpt["weight"], strict=True)
    except RuntimeError as e:
        raise StateDictMismatch(ckpt_path, str(e))

    model.eval()
    model.cpu()
    if amp_dtype == "fp16":
        _apply_half(model)

    extras = {k: ckpt.get(k) for k in ("epoch", "intersection", "hit_defects", "quality_score")
              if ckpt.get(k) is not None}
    logger.info(f"supervised model loaded ckpt={ckpt_path} tile={tile_size} amp={amp_dtype} {extras}")
    return model


def load_weight(det_type: DetectType, ckpt_path: str, amp_dtype: str = "bf16",
                manifest: Optional[dict] = None):
    """按检测类型分发（v2：仅监督）。"""
    if det_type == DetectType.SUPERVISED:
        return load_supervised(ckpt_path, amp_dtype=amp_dtype, manifest=manifest)
    raise ModelBuildError(f"不支持的检测类型: {det_type}")
