from __future__ import annotations

import uvicorn

from common.config import ManagerArgs
from core import AppCore


def build_app():
    return AppCore(ManagerArgs()).create_app()


app = build_app()


if __name__ == "__main__":
    args = ManagerArgs()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False,
    )
