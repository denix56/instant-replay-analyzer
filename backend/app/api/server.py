from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional
from threading import Thread

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import create_router
from ..config import get_settings
from ..model_downloader import ensure_models
from ..operations.manager import OperationManager
from ..operations.schemas import AppConfig


def create_app(config: Optional[AppConfig] = None, manager: Optional[OperationManager] = None) -> FastAPI:
    app_config = config or AppConfig()
    operations = manager or OperationManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _start_model_download(app)
        yield

    app = FastAPI(title="Instant Replay Analyzer API", version="0.1.0", lifespan=lifespan)
    app.state.operations = operations

    if app_config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=app_config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    router = create_router(operations)
    app.include_router(router, prefix="/api")
    app.include_router(router)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


def _start_model_download(app: FastAPI) -> None:
    settings = get_settings()
    if not settings.auto_download_models:
        app.state.model_download_status = {"status": "disabled"}
        return

    def download() -> None:
        app.state.model_download_status = {"status": "running"}
        results = ensure_models(settings.model_tier, settings.models_dir, gpu_backend=settings.gpu_backend)
        app.state.model_download_status = {
            "status": "failed" if any(result.status == "failed" for result in results) else "ready",
            "results": [result.__dict__ for result in results],
        }

    Thread(target=download, name="model-download", daemon=True).start()


def run_server(config: Optional[AppConfig] = None) -> None:
    import uvicorn

    app_config = config or AppConfig()
    uvicorn.run(
        "app.main:app",
        host=app_config.host,
        port=app_config.port,
        log_level=app_config.log_level,
        reload=False,
    )
