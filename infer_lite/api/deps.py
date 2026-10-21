from __future__ import annotations

from fastapi import Request

from engine.inference.base import BaseInferEngine
from engine.inference.supervised import SupervisedInferEngine
from engine.paths import ModelPathResolver, Resolved, expand
from engine.registry import ModelRegistry
from schemas import DetectRequest


def get_resolver(request: Request) -> ModelPathResolver:
    return request.app.state.resolver


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def resolve_request(body: DetectRequest, request: Request) -> Resolved:
    resolver: ModelPathResolver = request.app.state.resolver
    return resolver.resolve(
        product_code=body.product_code,
        point_id=body.point_id,
        dir_path=expand(body.dir),
    )


def get_infer_engine(request: Request, body: DetectRequest) -> BaseInferEngine:
    """每请求新建引擎实例（轻对象），参数自 app.state 配置注入。"""
    registry: ModelRegistry = request.app.state.registry
    gpu_lock = request.app.state.gpu_lock
    config = request.app.state.config
    return SupervisedInferEngine(
        registry, gpu_lock, request.app.state.device,
        threshold_override=config.seg_threshold,
        calib=request.app.state.calib,
        min_defect_diameter_mm=config.min_defect_diameter_mm,
        region_align=config.region_align,
        amp_dtype=config.amp_dtype,
        batch_max=config.batch_max,
        gpu_mem_safety=config.gpu_mem_safety,
    )
