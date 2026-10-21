from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from common.logger import setup_logger

logger = setup_logger("config")

# 推理精度模式（AMP_DTYPE）别名表：统一规范化为 bf16 / fp16 / fp32 三档
_AMP_DTYPE_ALIASES = {
    "bf16": "bf16", "bfloat16": "bf16",
    "fp16": "fp16", "float16": "fp16", "half": "fp16",
    "fp32": "fp32", "float32": "fp32",
}

#: 精度模式 → torch dtype（推理 autocast 共用一处映射，防各处漂移）
AMP_TORCH_DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def normalize_amp_dtype(raw: str) -> str:
    """把用户输入规范化为 bf16/fp16/fp32；非法值抛 ValueError。"""
    key = str(raw).strip().lower()
    if key not in _AMP_DTYPE_ALIASES:
        raise ValueError(f"未知精度模式: {raw!r}（可选 bf16|fp16|fp32）")
    return _AMP_DTYPE_ALIASES[key]


def amp_torch_dtype(mode: str) -> torch.dtype:
    return AMP_TORCH_DTYPES[normalize_amp_dtype(mode)]


def _default_amp_dtype() -> str:
    """解析默认精度模式：AMP_DTYPE 优先；兼容旧 HALF_PRECISION（1→fp16 / 0→fp32）；缺省 bf16。

    bf16 为默认：指数位宽与 fp32 一致，规避 fp16 溢出；bf16/fp32 模式下权重保持
    fp32（autocast 对 BN 走 fp32 策略，半精度权重在 BN 处会 dtype 冲突）。
    非法值告警回退，不阻断启动。
    """
    raw = os.environ.get("AMP_DTYPE")
    if raw is not None and raw.strip():
        try:
            return normalize_amp_dtype(raw)
        except ValueError as e:
            logger.warning(f"AMP_DTYPE={raw!r} 非法（{e}），按 HALF_PRECISION/默认值回退")
    legacy = os.environ.get("HALF_PRECISION")
    if legacy is not None and legacy.strip():
        return "fp16" if legacy.strip() == "1" else "fp32"
    return "bf16"


def _seg_threshold_override() -> Optional[float]:
    """SEG_THRESHOLD 覆盖语义（v2）：未设置 → None（以 manifest.threshold 为准）；
    显式设置 → 覆盖 manifest 携带阈值（服务端自配口径）。"""
    raw = os.environ.get("SEG_THRESHOLD")
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"SEG_THRESHOLD={raw!r} 非法，忽略（以 manifest.threshold 为准）")
        return None


@dataclass
class ManagerArgs:
    supervised_ckpt_dir: str = field(
        default=os.environ.get("SUPERVISED_CKPT_DIR", "D:/vision/vision-ai-pretrained/supervised"),
        metadata={"help": "监督模型 ckpt 根目录 ({dir}/{product}/，v2 布局：best 权重 + deploy/ 拷入即发布)"},
    )
    calib_dir: str = field(
        default=os.environ.get("CALIB_DIR", "D:/vision/vision-ai-pretrained/calib"),
        metadata={"help": "相机内外参根目录（{dir}/{product}/）；现场示教一机种一套，推理共用"},
    )
    model_num_limit: int = field(
        default=int(os.environ.get("MODEL_NUM_LIMIT", "100")),
        metadata={"help": "注册中心最多驻留模型数"},
    )
    device: str = field(
        default=os.environ.get("DEVICE", "cuda"),
        metadata={"help": "推理设备"},
    )
    amp_dtype: str = field(
        default=_default_amp_dtype(),
        metadata={
            "help": "推理精度模式 bf16|fp16|fp32（默认 bf16）：bf16/fp32 权重保持 fp32，"
                    "forward autocast bf16（fp32 为纯单精度不 autocast）；fp16 兼容旧行为"
                    "（权重 fp16 常驻 + autocast fp16，有溢出风险）；bf16 需 Ampere+(SM80) GPU。"
                    "旧 HALF_PRECISION=1/0 仍生效（映射 fp16/fp32）",
        },
    )
    seg_threshold: Optional[float] = field(
        default=_seg_threshold_override(),
        metadata={"help": "v2：缺省 None=以 manifest.threshold(0.5)为准；显式设置则覆盖（服务端自配口径）"},
    )
    batch_max: int = field(
        default=int(os.environ.get("BATCH_MAX", "24")),
        metadata={"help": "单次 forward 最大 patch 数（安全天花板）"},
    )
    gpu_mem_safety: float = field(
        default=float(os.environ.get("GPU_MEM_SAFETY", "0.75")),
        metadata={"help": "动态分块时可用显存安全系数"},
    )
    min_defect_diameter_mm: float = field(
        default=float(os.environ.get("MIN_DEFECT_DIAMETER_MM", "0.5")),
        metadata={"help": "缺陷直径过滤阈值(mm)，diameter < 该值的轮廓与测量值一并丢弃"},
    )
    region_align: bool = field(
        default=os.environ.get("REGION_ALIGN", "1") == "1",
        metadata={"help": "有效检测区配准总闸（0=直接用基准系朴素分区蒙版，不做逆向对齐；"
                          "配准失败时无论开关如何均回退朴素分区，仅告警不改判）"},
    )
    host: str = field(default="0.0.0.0", metadata={"help": "uvicorn host"})
    port: int = field(default=8090, metadata={"help": "uvicorn port"})
