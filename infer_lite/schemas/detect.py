from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DetectType(str, Enum):
    """v2：纯监督——旧非监督枚举值请求将 422（调用方须停发）。"""

    SUPERVISED = "two_d_supervised_detection"


class DetectRequest(BaseModel):
    device_code: str
    product_code: str
    fixture_code: str  # v2：保留必填、仅透传（寻址不再消费，零改动兼容调用方）
    point_id: str      # 选 manifest.points[pid] 的点位配置（json/基准图/zones）
    code: str
    name: str
    type: DetectType
    dir: str


class Measurements(BaseModel):
    diameter_mm: float = 0.0  # 最小包围圆直径（mm），与 global_contours 逐条对齐
    area_mm2: float = 0.0
    radius_mm: float = 0.0  # = diameter_mm / 2


class Detail(BaseModel):
    task_id: str
    type: str
    defect_class: str
    global_contours: list = Field(default_factory=list)
    measurements: list[Measurements] = Field(default_factory=list)


class DetectData(BaseModel):
    is_ok: bool
    details: list[Detail] = Field(default_factory=list)


class DetectResponse(BaseModel):
    code: int = 200
    message: str = "success"
    data: DetectData | None = None


class ModelStatus(BaseModel):
    uid: str
    type: str
    ckpt_path: str
    registered: bool
    load_time: datetime | None = None
    last_access: datetime | None = None
    in_gpu: bool = False


class ResolveQuery(BaseModel):
    product_code: str
    fixture_code: str | None = None  # v2：可选，不再参与寻址
    point_id: str | None = None      # 监督忽略（按 uid 反查）
    type: DetectType = DetectType.SUPERVISED


# --------------------------------------------------------------------------- #
# 拓扑发布前校验 (preflight) 与模型卸载 (unload)
# --------------------------------------------------------------------------- #
class PreflightStatus(str, Enum):
    UNCHANGED = "unchanged"  # registry 现役签名与候选一致，运行时无感
    UPDATED = "updated"  # 现役签名不同，下一请求 JIT 重载新权重
    ADDED = "added"  # 候选有、registry 无，首请求 JIT 加载
    REMOVED = "removed"  # registry 有、候选无，内存残留可 unload 释放


class PreflightRequest(BaseModel):
    dir: str  # 候选拓扑根目录（须含 supervised/ 子目录）


class PreflightItem(BaseModel):
    uid: str
    type: str
    product_code: str
    fixture_code: str | None = None  # v2 布局无 fixture 层
    point_id: str | None = None
    status: PreflightStatus
    current_ckpt_filename: str | None = None  # registry 现役权重文件名
    candidate_ckpt_filename: str | None = None  # 候选拓扑选中权重文件名（manifest.weight）
    manifest_present: bool = True
    registry_loaded: bool = False
    in_gpu: bool = False
    note: str | None = None


class PreflightSummary(BaseModel):
    unchanged: int = 0
    updated: int = 0
    added: int = 0
    removed: int = 0


class PreflightData(BaseModel):
    dir: str
    summary: PreflightSummary
    items: list[PreflightItem] = Field(default_factory=list)


class UnloadModelRequest(BaseModel):
    uid: str
