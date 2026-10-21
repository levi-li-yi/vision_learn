from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, Query

from api.deps import get_registry, get_resolver
from common.exceptions import TopologyInvalid
from engine.paths import ModelPathResolver, TopoEntry, expand
from engine.registry import ModelRegistry
from schemas import (
    DetectType,
    ModelStatus,
    PreflightData,
    PreflightItem,
    PreflightRequest,
    PreflightStatus,
    PreflightSummary,
    UnloadModelRequest,
)

router = APIRouter(prefix="/models", tags=["模型监控"])

_BEST_INT_PATTERN = re.compile(r"best_int(\d+)_ep(\d+)\.pth(?:\.tar)?")


def _split_uid(uid: str) -> tuple[str, str | None]:
    """uid 反解为 (product_code, fixture_code=None)。uid 形如 `sup::{product}`。"""
    _, rest = uid.split("::", 1)
    return rest.split("/")[0], None


def _version_rank(filename: str | None, manifest: dict | None):
    """版本排序键：manifest.quality_score（📋 训练侧增写后直接可用）优先，
    退化 best_int 数值（文件名与质量脱钩，字典序会错）。"""
    if manifest and isinstance(manifest.get("quality_score"), (int, float)):
        return (0, -float(manifest["quality_score"]), 0)
    if filename:
        m = _BEST_INT_PATTERN.fullmatch(filename)
        if m:
            return (0, -int(m.group(1)), -int(m.group(2)))
    return (1, 0.0, 0)


@router.get("", response_model=list[ModelStatus])
async def list_models(registry: ModelRegistry = Depends(get_registry)):
    return registry.list_all()


@router.get("/resolve", response_model=ModelStatus)
async def resolve_model(
    product_code: str = Query(...),
    fixture_code: str | None = Query(None),  # v2：兼容保留，不参与寻址
    point_id: str | None = Query(None),      # 监督忽略
    type: DetectType = Query(DetectType.SUPERVISED),
    registry: ModelRegistry = Depends(get_registry),
):
    return registry.status_by_fields(product_code)


@router.post("/preflight", response_model=PreflightData, summary="发布前拓扑校验")
async def preflight_topology(
    body: PreflightRequest,
    resolver: ModelPathResolver = Depends(get_resolver),
    registry: ModelRegistry = Depends(get_registry),
) -> PreflightData:
    """对比候选拓扑目录与 registry 现役内存态，输出 unchanged/updated/added/removed 清单。

    纯磁盘只读 + registry 只读快照，不加载权重、不进 gpu_lock。
    基线为 registry cache（现役驻留模型），非落盘生产目录。
    v2：候选拓扑根只须含 supervised/ 子目录。
    """
    cand_root = Path(expand(body.dir))
    if not cand_root.is_dir():
        raise TopologyInvalid(f"候选拓扑目录不存在或非目录: {cand_root}")

    sup_dir = cand_root / "supervised"
    if not sup_dir.is_dir():
        raise TopologyInvalid("候选拓扑必须含 supervised/ 子目录")

    candidate: dict[str, TopoEntry] = {
        e.uid: e for e in ModelPathResolver.enumerate_topology(sup_dir)
    }
    registry_map: dict[str, ModelStatus] = {s.uid: s for s in registry.list_all()}

    items: list[PreflightItem] = []
    summary = PreflightSummary()
    seen: set[str] = set()

    for uid, ce in candidate.items():
        seen.add(uid)
        rs = registry_map.get(uid)
        cur_fn = Path(rs.ckpt_path).name if rs and rs.registered else None
        if rs is None or not rs.registered:
            status = PreflightStatus.ADDED
            summary.added += 1
        elif cur_fn == ce.ckpt_filename:
            status = PreflightStatus.UNCHANGED
            summary.unchanged += 1
        else:
            status = PreflightStatus.UPDATED
            summary.updated += 1

        note: str | None = None
        if status == PreflightStatus.UPDATED:
            try:
                cand_manifest = ModelPathResolver.read_manifest(
                    Path(ce.ckpt_path).parent / "deploy" / "manifest.json"
                )
            except Exception:
                cand_manifest = None
            if _version_rank(ce.ckpt_filename, cand_manifest) > _version_rank(cur_fn, None):
                note = "候选权重签名早于现役，疑似降级"

        items.append(
            PreflightItem(
                uid=uid,
                type=ce.type.value,
                product_code=ce.product_code,
                fixture_code=None,
                point_id=None,
                status=status,
                current_ckpt_filename=cur_fn,
                candidate_ckpt_filename=ce.ckpt_filename,
                manifest_present=ce.manifest_present,
                registry_loaded=bool(rs and rs.registered),
                in_gpu=bool(rs and rs.in_gpu),
                note=note,
            )
        )

    for uid, rs in registry_map.items():
        if uid in seen or not rs.registered:
            continue
        product_code, fixture_code = _split_uid(uid)
        items.append(
            PreflightItem(
                uid=uid,
                type=rs.type,
                product_code=product_code,
                fixture_code=fixture_code,
                point_id=None,
                status=PreflightStatus.REMOVED,
                current_ckpt_filename=Path(rs.ckpt_path).name,
                candidate_ckpt_filename=None,
                manifest_present=True,
                registry_loaded=True,
                in_gpu=rs.in_gpu,
                note="新拓扑无此模型，可 POST /models/unload 释放内存",
            )
        )
        summary.removed += 1

    return PreflightData(dir=str(cand_root), summary=summary, items=items)


@router.post("/unload", response_model=ModelStatus, summary="卸载驻留模型")
async def unload_model(
    body: UnloadModelRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> ModelStatus:
    """从 registry 内存中卸载指定 uid 的模型（推理中的模型受保护，不会被移除）。"""
    registry.unload(body.uid)
    return registry.status(body.uid)


@router.get("/{uid}", response_model=ModelStatus)
async def get_model(uid: str, registry: ModelRegistry = Depends(get_registry)):
    return registry.status(uid)
