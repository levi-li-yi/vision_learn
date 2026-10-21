from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["健康检查"])


@router.get("/health")
async def health():
    return {"status": "ok"}
