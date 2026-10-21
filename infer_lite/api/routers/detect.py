from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import get_infer_engine, get_registry, resolve_request
from engine.inference.base import BaseInferEngine
from engine.paths import Resolved
from engine.registry import ModelRegistry
from schemas import DetectData, DetectRequest, DetectResponse

router = APIRouter(prefix="/detect", tags=["检测推理"])


@router.post("", response_model=DetectResponse)
async def detect(
    body: DetectRequest,
    resolved: Resolved = Depends(resolve_request),
    registry: ModelRegistry = Depends(get_registry),
    engine: BaseInferEngine = Depends(get_infer_engine),
):
    entry = registry.get_or_load(resolved)  # 命中复用；(路径,mtime,size) 变化 → 热更新
    details = engine.infer(resolved, entry)
    is_ok = all(d.defect_class == "OK" for d in details)
    return DetectResponse(data=DetectData(is_ok=is_ok, details=details))
