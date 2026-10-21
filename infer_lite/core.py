from __future__ import annotations

import threading
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from common.config import ManagerArgs
from common.exceptions import ServiceException
from common.logger import setup_logger
from engine import frameworks
from engine.calib import CalibRegistry
from engine.paths import ModelPathResolver
from engine.registry import ModelRegistry
from schemas import DetectResponse

logger = setup_logger("core")


class AppCore:
    def __init__(self, args: ManagerArgs):
        self.args = args

    def create_app(self) -> FastAPI:
        args = self.args

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            app.state.config = args
            app.state.device = args.device
            app.state.resolver = ModelPathResolver(args.supervised_ckpt_dir)
            app.state.registry = ModelRegistry(
                model_num_limit=args.model_num_limit,
                device=args.device,
                amp_dtype=args.amp_dtype,
            )
            app.state.gpu_lock = threading.RLock()
            app.state.calib = CalibRegistry(args.calib_dir)
            # 预热共享 wheel 导入（segment_anything/efficientvit 首次 import 较慢），移到启动期
            frameworks.prewarm()
            # bf16 需 Ampere+(SM80) GPU；旧卡告警不自动改判（操作者显式设 AMP_DTYPE=fp16/fp32 回退）
            if args.amp_dtype == "bf16" and str(args.device).startswith("cuda"):
                try:
                    if torch.cuda.get_device_capability(0) < (8, 0):
                        logger.warning(
                            f"AMP_DTYPE=bf16 但 GPU 算力 {torch.cuda.get_device_capability(0)} "
                            f"< SM80（Ampere+），bf16 内核可能不可用/回退异常，"
                            f"建议 AMP_DTYPE=fp16 或 fp32"
                        )
                except Exception:
                    pass

            logger.info(
                f"service started (v2 · supervised-only · ckpt_root={args.supervised_ckpt_dir})"
            )
            yield

            app.state.registry.clear()
            logger.info("service stopped")

        app = FastAPI(lifespan=lifespan)

        from api.routers import detect, health, models

        app.include_router(health.router)
        app.include_router(detect.router)
        app.include_router(models.router)

        @app.exception_handler(ServiceException)
        async def service_exception_handler(request: Request, exc: ServiceException):
            logger.error(f"service error: {exc.message}")
            return JSONResponse(
                status_code=exc.code if 200 <= exc.code < 600 else 500,
                content=DetectResponse(
                    code=exc.code, message=exc.message, data=None
                ).model_dump(),
            )

        @app.exception_handler(Exception)
        async def unhandled_exception_handler(request: Request, exc: Exception):
            # 兜底：推理期非业务异常（CUDA OOM / RuntimeError / 维度不匹配等）
            # 统一成 {code, message, data: None}，避免泄漏裸 500。
            # ServiceException 由上面的专用处理器接住（Starlette 按异常 MRO 最精确匹配优先）。
            logger.exception(f"unhandled error: {exc!r}")
            return JSONResponse(
                status_code=500,
                content=DetectResponse(
                    code=500, message=f"内部错误: {exc}", data=None
                ).model_dump(),
            )

        return app
