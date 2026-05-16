from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse

from ..config import get_settings, normalize_model_tier, normalize_runtime_profile
from ..db import Database
from ..hf_pipeline.model_registry import model_for_role, quantization_for_backend
from ..knowledge.hunt_runtime import HuntKnowledgeService
from ..pipeline import run_analysis, run_search
from ..operations.events import sse_format
from ..operations.manager import OperationManager, OperationNotFound
from ..operations.schemas import (
    ClipSearchRequest,
    IndexRequest,
    OperationKind,
    OperationRequest,
    OperationStartResponse,
    OperationStatus,
)


def create_router(manager: Optional[OperationManager] = None) -> APIRouter:
    router = APIRouter()
    operations = manager or OperationManager()

    def start_response(status: OperationStatus) -> OperationStartResponse:
        return OperationStartResponse(
            operation_id=status.operation_id,
            state=status.state,
            status_url=f"/api/operations/{status.operation_id}",
            events_url=f"/api/operations/{status.operation_id}/events",
        )

    @router.get("/health")
    async def health() -> dict:
        return {"status": "ok", "state": "online", "version": "0.1.0", "message": "Backend reachable"}

    @router.get("/status")
    async def status() -> dict:
        settings = get_settings()
        db = Database(settings.db_path)
        try:
            clips = db.list_clips()
            return {
                "status": "ok",
                "clips": len(clips),
                "indexed_clips": sum(1 for clip in clips if clip["status"] == "indexed"),
                "runtime_profile": settings.runtime_profile,
                "gpu_backend": settings.gpu_backend,
                "model_tier": settings.model_tier,
            }
        finally:
            db.close()

    @router.get("/settings")
    async def settings() -> dict:
        app_settings = get_settings()
        tier = app_settings.tier
        indexing = app_settings.indexing
        return {
            "clips_dir": str(app_settings.clips_dir),
            "data_dir": str(app_settings.data_dir),
            "models_dir": str(app_settings.models_dir),
            "model_tier": app_settings.model_tier,
            "indexing_profile": app_settings.indexing_profile,
            "runtime_profile": app_settings.runtime_profile,
            "gpu_backend": app_settings.gpu_backend,
            "qdrant_url": app_settings.qdrant_url,
            "active_models": {
                "retrieval": model_for_role("embedder", app_settings.model_tier, device_backend=app_settings.gpu_backend).model_id,
                "asr": model_for_role("asr", app_settings.model_tier, device_backend=app_settings.gpu_backend).model_id,
                "audio_captioner": tier.audio_captioner_model,
                "reranker": model_for_role("reranker", app_settings.model_tier, device_backend=app_settings.gpu_backend).model_id,
                "reasoning": model_for_role("summarizer", app_settings.model_tier, device_backend=app_settings.gpu_backend).model_id,
            },
            "segment_settings": {
                "segment_seconds": indexing.segment_seconds,
                "segment_stride_seconds": indexing.segment_stride_seconds,
                "representative_frames_per_segment": indexing.representative_frames_per_segment,
                "store_video_segment_files": indexing.store_video_segment_files,
                "store_audio_segment_files": indexing.store_audio_segment_files,
            },
            "video_embedding": {
                "fps": app_settings.video_embedding_fps,
                "max_frames": app_settings.video_embedding_max_frames,
            },
            "torch_compile": {
                "mode": app_settings.torch_compile_mode,
                "backend": app_settings.torch_compile_backend,
                "profile": app_settings.torch_compile_profile,
            },
        }

    @router.put("/settings")
    async def save_settings(payload: dict[str, Any] = Body(default_factory=dict)) -> dict:
        app_settings = get_settings()
        db = Database(app_settings.db_path)
        try:
            for key, value in payload.items():
                db.set_setting(str(key), str(value))
            return {"ok": True}
        finally:
            db.close()

    @router.post("/settings/model-tier")
    async def set_model_tier(payload: dict[str, Any] = Body(default_factory=dict)) -> dict:
        tier = normalize_model_tier(str(payload.get("tier") or payload.get("model_tier") or "default"))
        app_settings = get_settings()
        db = Database(app_settings.db_path)
        try:
            db.set_setting("model_tier", tier)
            return {"ok": True, "model_tier": tier, "requires_reindex": True}
        finally:
            db.close()

    @router.post("/settings/runtime-profile")
    async def set_runtime_profile(payload: dict[str, Any] = Body(default_factory=dict)) -> dict:
        profile = normalize_runtime_profile(str(payload.get("profile") or payload.get("runtime_profile") or "auto"))
        app_settings = get_settings()
        db = Database(app_settings.db_path)
        try:
            db.set_setting("runtime_profile", profile)
            return {"ok": True, "runtime_profile": profile}
        finally:
            db.close()

    @router.get("/service-status")
    async def service_status() -> dict:
        app_settings = get_settings()
        tier = app_settings.tier
        embedder_spec = model_for_role("embedder", app_settings.model_tier, device_backend=app_settings.gpu_backend)
        asr_spec = model_for_role("asr", app_settings.model_tier, device_backend=app_settings.gpu_backend)
        reranker_spec = model_for_role("reranker", app_settings.model_tier, device_backend=app_settings.gpu_backend)
        summarizer_spec = model_for_role("summarizer", app_settings.model_tier, device_backend=app_settings.gpu_backend)
        return {
            "backend": "online",
            "qdrant": "embedded" if _is_local_qdrant(app_settings.qdrant_url) else _http_status(app_settings.qdrant_url),
            "model_tier": app_settings.model_tier,
            "runtime_profile": app_settings.runtime_profile,
            "gpu_backend": app_settings.gpu_backend,
            "models": {
                "qwen3_vl_embedder": _model_status(
                    embedder_spec.model_id,
                    "retrieval",
                    quantization_for_backend(embedder_spec, app_settings.gpu_backend),
                    app_settings.gpu_backend,
                ),
                "whisper_asr": _model_status(asr_spec.model_id, "asr", "transformers", app_settings.gpu_backend),
                "midashenglm_audio_captioner": _model_status(
                    tier.audio_captioner_model,
                    "audio_captioner",
                    "transformers",
                    app_settings.gpu_backend,
                ),
                "qwen3_vl_reranker": _model_status(
                    reranker_spec.model_id,
                    "reranker",
                    quantization_for_backend(reranker_spec, app_settings.gpu_backend),
                    app_settings.gpu_backend,
                ),
                "qwen35_video_fusion": _model_status(
                    summarizer_spec.model_id,
                    "reasoning",
                    quantization_for_backend(summarizer_spec, app_settings.gpu_backend),
                    app_settings.gpu_backend,
                ),
            },
        }

    @router.get("/dashboard")
    async def dashboard() -> dict:
        app_settings = get_settings()
        db = Database(app_settings.db_path)
        try:
            clips = db.list_clips()
            active = await operations.list()
            return {
                "clipCount": len(clips),
                "indexedClipCount": sum(1 for clip in clips if clip["status"] == "indexed"),
                "groupCount": len(db.groups()),
                "queuedJobs": sum(1 for item in active if str(item.state) == "OperationState.PENDING" or str(item.state).endswith("pending")),
                "activeJobs": sum(1 for item in active if str(item.state) == "OperationState.RUNNING" or str(item.state).endswith("running")),
                "storageGb": round(_dir_size(app_settings.data_dir) / 1024**3, 3),
            }
        finally:
            db.close()

    @router.get("/operations", response_model=list[OperationStatus])
    async def list_operations() -> list[OperationStatus]:
        return await operations.list()

    @router.post("/operations", response_model=OperationStartResponse, status_code=202)
    async def start_operation(request: OperationRequest) -> OperationStartResponse:
        status = await operations.start(request.kind.value, request.params)
        return start_response(status)

    @router.post("/scan", response_model=OperationStartResponse, status_code=202)
    async def scan_clips(request: dict[str, Any] = Body(default_factory=dict)) -> OperationStartResponse:
        status = await operations.start(OperationKind.SCAN.value, request)
        return start_response(status)

    @router.get("/operations/{operation_id}", response_model=OperationStatus)
    async def get_operation(operation_id: str) -> OperationStatus:
        try:
            return await operations.get(operation_id)
        except OperationNotFound as exc:
            raise HTTPException(status_code=404, detail="operation not found") from exc

    @router.delete("/operations/{operation_id}", response_model=OperationStatus)
    async def cancel_operation(operation_id: str) -> OperationStatus:
        try:
            return await operations.cancel(operation_id)
        except OperationNotFound as exc:
            raise HTTPException(status_code=404, detail="operation not found") from exc

    @router.get("/operations/{operation_id}/events")
    async def operation_events(operation_id: str) -> StreamingResponse:
        try:
            await operations.get(operation_id)
        except OperationNotFound as exc:
            raise HTTPException(status_code=404, detail="operation not found") from exc

        async def stream():
            async for event in operations.events.subscribe(operation_id):
                yield sse_format(event)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @router.post("/index", response_model=OperationStartResponse, status_code=202)
    async def index_clips(request: IndexRequest) -> OperationStartResponse:
        params = request.dict()
        status = await operations.start(OperationKind.INDEX.value, params)
        return start_response(status)

    @router.get("/index/jobs")
    async def index_jobs() -> list[dict]:
        statuses = await operations.list()
        jobs = []
        for status in statuses:
            if status.kind != OperationKind.INDEX.value:
                continue
            jobs.append(
                {
                    "id": status.operation_id,
                    "folder": status.params.get("source") or status.params.get("replay_dir") or status.params.get("input") or "",
                    "status": _ui_job_status(status.state.value),
                    "progress": status.progress,
                    "clipsFound": (status.result or {}).get("total_candidates", 0) if isinstance(status.result, dict) else 0,
                    "startedAt": status.started_at.isoformat() if status.started_at else status.created_at.isoformat(),
                }
            )
        return jobs

    @router.post("/index/jobs", response_model=OperationStartResponse, status_code=202)
    async def start_index_job(request: dict[str, Any] = Body(default_factory=dict)) -> OperationStartResponse:
        folder = request.get("folder") or request.get("input") or request.get("source")
        params = {"input": folder, "force": bool(request.get("force", False))}
        status = await operations.start(OperationKind.INDEX.value, params)
        return start_response(status)

    @router.get("/groups")
    async def groups() -> list[dict]:
        app_settings = get_settings()
        db = Database(app_settings.db_path)
        try:
            rows = db.groups()
            return [
                {
                    "group_name": row["group_name"],
                    "total_videos": row["total_videos"] or 0,
                    "indexed_videos": row["indexed_videos"] or 0,
                    "failed_videos": row["failed_videos"] or 0,
                    "missing_videos": row["missing_videos"] or 0,
                    "last_indexed_at": row["last_indexed_at"],
                    "id": row["group_name"],
                    "name": row["group_name"],
                    "clipCount": row["total_videos"] or 0,
                    "lastUpdated": row["last_indexed_at"] or "Never",
                    "rules": [],
                    "color": "#3b82f6",
                }
                for row in rows
            ]
        finally:
            db.close()

    @router.get("/clips")
    async def clips(group_name: str | None = None, status: str | None = None) -> list[dict]:
        app_settings = get_settings()
        db = Database(app_settings.db_path)
        try:
            return [dict(row) for row in db.list_clips(group_name=group_name, status=status)]
        finally:
            db.close()

    @router.get("/clips/{clip_id}")
    async def clip_detail(clip_id: int) -> dict:
        app_settings = get_settings()
        db = Database(app_settings.db_path)
        try:
            clip = db.get_clip(clip_id)
            if clip is None:
                raise HTTPException(status_code=404, detail="clip not found")
            transcripts = db.get_transcripts(clip_id)
            segments = db.list_segments(clip_id)
            tags = db.get_clip_tags(clip_id)
            hud_detections = [dict(row) for row in db.list_hud_detections(clip_id=clip_id)]
            hud_summary = db.hud_loadout_summary(clip_id)
            death_detections = [dict(row) for row in db.list_death_screen_detections(clip_id=clip_id)]
            death_summary = db.death_screen_summary(clip_id)
            return {
                **dict(clip),
                "tags": tags,
                "segments": [dict(row) for row in segments],
                "transcripts": [dict(row) for row in transcripts],
                "hud_detections": hud_detections,
                "death_detections": death_detections,
                "active_weapon": hud_summary.get("active_weapon"),
                "active_equipment": hud_summary.get("active_equipment"),
                "active_equipment_type": hud_summary.get("active_equipment_type"),
                "detected_loadout": hud_summary.get("loadout", []),
                "death_status": death_summary.get("status"),
                "killed_by_weapon": death_summary.get("killed_with"),
                "killer_name": death_summary.get("killer_name"),
                "id": str(clip["id"]),
                "title": clip["filename"],
                "game": clip["group_name"],
                "timestamp": "00:00",
                "durationSec": clip["duration"] or 0,
                "score": 1,
                "path": clip["path"],
                "summary": clip["summary"] or "",
                "transcript": [row["text"] for row in transcripts],
                "events": [
                    {"time": f"{segment['start_time']:.1f}s", "label": segment["modality"], "confidence": 1.0}
                    for segment in segments
                ],
                "technical": {
                    "resolution": f"{clip['width'] or 0}x{clip['height'] or 0}",
                    "fps": clip["fps"] or 0,
                    "codec": clip["codec"] or "",
                    "sizeMb": round((clip["size_bytes"] or 0) / 1024 / 1024, 2),
                },
            }
        finally:
            db.close()

    @router.post("/search")
    async def search_clips(request: dict[str, Any] = Body(default_factory=dict)) -> dict:
        return run_search(params=request)

    @router.post("/clips/{clip_id}/analyze")
    async def analyze_clip(clip_id: int, request: dict[str, Any] = Body(default_factory=dict)) -> dict:
        return run_analysis(params={**request, "clip_id": clip_id})

    @router.get("/knowledge/hunt/status")
    async def hunt_knowledge_status() -> dict:
        app_settings = get_settings()
        service = HuntKnowledgeService(app_settings.data_dir / "packs" / "hunt-knowledge-pack")
        return service.status()

    @router.post("/knowledge/hunt/search")
    async def hunt_knowledge_search(request: dict[str, Any] = Body(default_factory=dict)) -> dict:
        app_settings = get_settings()
        service = HuntKnowledgeService(app_settings.data_dir / "packs" / "hunt-knowledge-pack")
        query = str(request.get("query") or "")
        top_k = int(request.get("top_k") or request.get("limit") or 5)
        entity_types = set(request.get("entity_types") or [])
        hits = service.search(query, top_k=top_k, entity_types=entity_types or None)
        return {"query": query, "hits": [hit.__dict__ for hit in hits], "status": service.status()}

    @router.get("/logs")
    async def logs(limit: int = 200) -> list[dict]:
        statuses = await operations.list()
        rows = []
        for status in statuses[-limit:]:
            rows.append(
                {
                    "id": status.operation_id,
                    "level": "error" if status.error else "info",
                    "source": status.kind,
                    "message": status.error or status.message or status.state.value,
                    "time": status.updated_at.isoformat(),
                }
            )
        return rows

    return router


def _http_status(url: str) -> str:
    import urllib.request

    if _is_local_qdrant(url) or not url:
        return "embedded"
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=0.8) as response:
            return "online" if response.status < 500 else "degraded"
    except Exception:
        return "offline"


def _is_local_qdrant(url: str | None) -> bool:
    value = (url or "local").strip().lower()
    return value in {"", "local", "embedded", "qdrant-local"} or value.startswith("local:")


def _model_status(model_name: str, task: str, precision: str, runtime_backend: str) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "task": task,
        "local": False,
        "status": "configured" if model_name != "disabled" else "disabled",
        "precision": precision,
        "runtime_backend": runtime_backend,
        "warning": "Model availability is checked by the runtime adapter during loading.",
    }


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _ui_job_status(state: str) -> str:
    return {
        "pending": "queued",
        "running": "running",
        "succeeded": "complete",
        "failed": "failed",
        "canceled": "failed",
    }.get(state, state)
