from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .analysis.hit_marker import detect_hit_marker_evidence
from .analysis.hud_loadout import HudLoadoutDetector, detections_to_rows, summarize_detections
from .analysis.deep_reasoning import GameplayReasoner
from .config import AppSettings, get_settings, parse_bool
from .db import Database
from .embeddings.hf_multimodal_embedder import (
    DEFAULT_EMBEDDING_DIMENSION,
    EmbeddingConfig,
    HuggingFaceMultimodalEmbedder,
    TransformersEmbeddingBackend,
)
from .embeddings.vector_store import QdrantVectorStore, QdrantVectorStoreConfig
from .hf_pipeline.adapters import (
    AudioCaptionerAdapter,
    FusionSummarizerAdapter,
    build_asr_transcript,
    metadata_with_qwen_visual_ocr,
)
from .hf_pipeline.evidence_ledger import build_evidence_ledger, visual_events_to_observations
from .hf_pipeline.model_registry import model_for_role, quantization_for_backend, validate_tier_for_backend
from .hf_pipeline.retrieval import RetrievalSettings, late_fuse_hits
from .hf_pipeline.schemas import (
    ASRTranscriptV1,
    ClipManifestV1,
    ClipTimebaseV1,
    EvidenceLedgerV1,
    EmbeddingRecordV1,
    FusedSummaryV1,
    MediaWindowV1,
    MetadataPayloadV1,
    VideoPayloadBudgetV1,
    VisualEventV1,
    payload_hash,
)
from .hf_pipeline.video_budget import fit_qwen_video_budget, qwen_budget_profiles
from .ingestion.scanner import scan_directory
from .model_downloader import ensure_models
from .models import AnalyzeResponse, SearchResponse, SearchResult, ScanSummary, utc_now
from .knowledge.hunt_runtime import HuntKnowledgeService, format_knowledge_hits
from .processing.av_segments import extract_audio_segment_to_path, extract_clip_segments
from .processing.qwen_video import (
    QwenVideoInput,
    SUMMARY_KILL_FOCUS_END_SEC,
    SUMMARY_KILL_FOCUS_START_SEC,
    SUMMARY_KILL_FOCUS_WINDOW_ID,
    prepare_hit_marker_video_input,
    prepare_qwen_video_input,
)
from .processing.timebase import analysis_start_sec
from .processing.transcription import Transcriber, TranscriptionConfig, TransformersASRBackend
from .runtime.transformers_runtime import transformers_runtime_manager
from .search.reranking import SearchReranker


HUD_DETECTION_WINDOW_SECONDS = 15.0
HUD_KILL_DETECTION_START_SECONDS = 18.0
HUD_KILL_DETECTION_END_SECONDS = 20.0
AUDIO_CAPTION_MIN_WINDOW_SECONDS = 0.25


@dataclass
class _IndexingClipState:
    index: int
    clip: Any
    clip_id: int
    base_progress: float
    segment_summary: Any = None
    segments: list[Any] = field(default_factory=list)
    death_summary: dict[str, Any] = field(
        default_factory=lambda: {"status": None, "killed_with": None, "killer_name": None}
    )
    death_text: str = ""
    hud_summary: dict[str, Any] = field(
        default_factory=lambda: {
            "active_weapon": None,
            "active_equipment": None,
            "active_equipment_type": None,
            "loadout": [],
        }
    )
    hit_marker_summary: dict[str, Any] = field(default_factory=lambda: {"detected": False, "evidence": []})
    hud_text: str = ""
    asr_audio_path: str | None = None
    asr_model_id: str | None = None
    asr_language: str | None = None
    transcript_text: str = ""
    transcript_segments: list[dict[str, Any]] = field(default_factory=list)
    metadata_payload: MetadataPayloadV1 | None = None
    manifest: ClipManifestV1 | None = None
    timebase: ClipTimebaseV1 | None = None
    qwen_video_input: QwenVideoInput | None = None
    qwen_focus_video_input: QwenVideoInput | None = None
    qwen_embedding_video_input: QwenVideoInput | None = None
    video_payload_budgets: list[VideoPayloadBudgetV1] = field(default_factory=list)
    qwen_visual_ocr: dict[str, Any] = field(default_factory=dict)
    windows: list[MediaWindowV1] = field(default_factory=list)
    visual_events: list[VisualEventV1] = field(default_factory=list)
    evidence_ledger: EvidenceLedgerV1 | None = None
    video_observations: list[Any] = field(default_factory=list)
    audio_caption_windows: list[MediaWindowV1] = field(default_factory=list)
    audio_captions: list[Any] = field(default_factory=list)
    transcript_artifact: ASRTranscriptV1 | None = None
    fused_summary: FusedSummaryV1 | None = None
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    completed: bool = False
    failed: bool = False
    error: str | None = None


def run_scan(
    params: dict[str, Any] | None = None,
    *,
    input: str | None = None,
    source: str | None = None,
    force_verify: bool = False,
    progress_callback: object | None = None,
    **_: Any,
) -> dict[str, Any]:
    settings = get_settings()
    root = input or source or (params or {}).get("input") or (params or {}).get("source") or str(settings.clips_dir)
    db = Database(settings.db_path)
    try:
        summary = scan_directory(root, db, force_verify=force_verify, progress_callback=progress_callback)
        db.set_setting("clips_dir", str(root))
        return summary.model_dump()
    finally:
        db.close()


def run_indexing(
    params: dict[str, Any] | None = None,
    *,
    input: str | None = None,
    source: str | None = None,
    replay_dir: str | None = None,
    group_name: str | None = None,
    force: bool = False,
    progress_callback: object | None = None,
    cancel_event: asyncio.Event | None = None,
    **_: Any,
) -> dict[str, Any]:
    settings = get_settings()
    payload = params or {}
    root = input or source or replay_dir or payload.get("input") or payload.get("source") or payload.get("replay_dir") or str(settings.clips_dir)
    group_name = group_name or payload.get("group_name")
    force = bool(force or payload.get("force"))
    db = Database(settings.db_path)
    try:
        validate_tier_for_backend(settings.model_tier, settings.gpu_backend)
        if settings.auto_download_models:
            _progress(progress_callback, "checking/downloading Hugging Face models", 0.01, {"stage": "model download"})
            model_results = ensure_models(settings.model_tier, settings.models_dir, gpu_backend=settings.gpu_backend)
            failures = [result for result in model_results if result.status == "failed"]
            if failures and not settings.allow_mock_models:
                raise RuntimeError("; ".join(f"{item.role}: {item.error}" for item in failures))
        _progress(progress_callback, "scanning library", 0.02, {"stage": "scan"})
        scan_summary = scan_directory(root, db, force_verify=False, progress_callback=progress_callback)
        db.set_setting("clips_dir", str(root))
        db.set_setting("model_tier", settings.model_tier)
        db.set_setting("indexing_profile", settings.indexing_profile)
        db.set_setting("runtime_profile", settings.runtime_profile)

        clips = db.list_clips(group_name=group_name)
        candidates = [
            clip
            for clip in clips
            if force
            or str(clip["scan_status"]) in {"new", "changed"}
            or str(clip["status"]) in {"pending", "partial", "failed"}
            or not str(clip["summary"] or "").strip()
            or _clip_needs_embedding_rebuild(db, int(clip["id"]), settings)
        ]
        manager = _model_runtime_manager(settings)
        skipped = len(clips) - len(candidates)
        total = len(candidates)
        states = [
            _IndexingClipState(
                index=clip_index,
                clip=clip,
                clip_id=int(clip["id"]),
                base_progress=clip_index / max(total, 1),
            )
            for clip_index, clip in enumerate(candidates, start=1)
        ]
        vector_store = _vector_store(settings, _embedding_dimension(settings))

        try:
            if states:
                _prepare_indexing_states(settings, db, states, force, progress_callback, cancel_event)
                manager.unload()
                _run_asr_indexing_stage(settings, db, states, progress_callback, cancel_event)
                manager.unload()
                _run_audio_caption_indexing_stage(settings, db, states, progress_callback, cancel_event)
                manager.unload()
                _run_fusion_indexing_stage(settings, db, states, progress_callback, cancel_event)
                manager.unload()
                _run_embedding_indexing_stage(settings, db, states, vector_store, progress_callback, cancel_event)
        finally:
            manager.unload()

        completed = sum(1 for state in states if state.completed)
        failed = sum(1 for state in states if state.failed)

        return {
            "scan": scan_summary.model_dump(),
            "total_candidates": total,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "qdrant_active": vector_store.using_qdrant,
        }
    finally:
        manager = _model_runtime_manager(settings)
        if getattr(manager, "loaded", None) is not None:
            manager.unload()
        db.close()


def _prepare_indexing_states(
    settings: AppSettings,
    db: Database,
    states: list[_IndexingClipState],
    force: bool,
    progress_callback: object | None,
    cancel_event: asyncio.Event | None,
) -> None:
    embedder = _embedder(settings)
    hunt_knowledge = _hunt_knowledge(settings, embedder)
    hud_detector = HudLoadoutDetector(hunt_knowledge, embedder=embedder) if hunt_knowledge.available else None
    total = len(states)
    for state in states:
        _check_cancel(cancel_event)
        if state.failed:
            continue
        try:
            _progress(
                progress_callback,
                f"preparing clip media and metadata: {state.clip['filename']}",
                _stage_progress(0.04, 0.18, state.index, total),
                {"clip_id": state.clip_id, "stage": "prepare"},
            )
            state.segment_summary = extract_clip_segments(
                db,
                state.clip,
                settings.indexing,
                settings.data_dir,
                overwrite=force,
                progress_callback=progress_callback,
            )
            state.segments = db.list_segments(state.clip_id)
            db.execute(
                """
                UPDATE av_segments
                SET embedding_id=NULL, embedding_model=NULL, embedding_precision=NULL, runtime_backend=NULL
                WHERE clip_id=?
                """,
                (state.clip_id,),
            )
            db.execute("DELETE FROM death_screen_detections WHERE clip_id=?", (state.clip_id,))
            if hud_detector is not None:
                _progress(
                    progress_callback,
                    f"HUD loadout detection: {state.clip['filename']}",
                    _stage_progress(0.22, 0.26, state.index, total),
                    {"clip_id": state.clip_id, "stage": "hud_loadout"},
                )
                state.hud_summary = _detect_hud_for_segments(
                    db,
                    state.clip_id,
                    state.segments,
                    hud_detector,
                    clip=state.clip,
            )
            state.hud_text = _hud_summary_text(state.hud_summary)
            full_video_path = str(state.clip["path"] or "")
            clip_duration = float(state.clip["duration"] or 0.0)
            state.timebase = _clip_timebase(state.clip, settings)
            state.video_payload_budgets = [
                fit_qwen_video_budget("full_visual", settings=settings, max_allocated_gb=settings.qwen_vram_max_allocated_gb),
                fit_qwen_video_budget("focus_visual", settings=settings, max_allocated_gb=settings.qwen_vram_max_allocated_gb),
                fit_qwen_video_budget("ocr", settings=settings, max_allocated_gb=settings.qwen_vram_max_allocated_gb),
                fit_qwen_video_budget("embedding", settings=settings, max_allocated_gb=settings.qwen_vram_max_allocated_gb),
            ]
            analysis_start = state.timebase.analysis_start_sec
            if full_video_path:
                if analysis_start < clip_duration:
                    full_budget = fit_qwen_video_budget(
                        "full_visual",
                        settings=settings,
                        max_allocated_gb=settings.qwen_vram_max_allocated_gb,
                    )
                    focus_budget = fit_qwen_video_budget(
                        "focus_visual",
                        settings=settings,
                        max_allocated_gb=settings.qwen_vram_max_allocated_gb,
                    )
                    embed_budget = fit_qwen_video_budget(
                        "embedding",
                        settings=settings,
                        max_allocated_gb=settings.qwen_vram_max_allocated_gb,
                    )
                    state.qwen_video_input = prepare_qwen_video_input(
                        full_video_path,
                        settings.data_dir / "qwen_video_inputs",
                        fps=settings.video_embedding_fps,
                        max_frames=full_budget.max_frames,
                        start_sec=analysis_start,
                        end_sec=clip_duration,
                        accelerator=settings.gpu_backend,
                    )
                    focus_start = max(analysis_start, SUMMARY_KILL_FOCUS_START_SEC)
                    focus_end = min(clip_duration, SUMMARY_KILL_FOCUS_END_SEC)
                    if focus_end > focus_start:
                        state.qwen_focus_video_input = prepare_qwen_video_input(
                            full_video_path,
                            settings.data_dir / "qwen_video_focus_inputs",
                            fps=settings.video_embedding_fps,
                            max_frames=focus_budget.max_frames,
                            start_sec=focus_start,
                            end_sec=focus_end,
                            accelerator=settings.gpu_backend,
                        )
                    state.qwen_embedding_video_input = prepare_qwen_video_input(
                        full_video_path,
                        settings.data_dir / "qwen_video_embedding_inputs",
                        fps=settings.video_embedding_fps,
                        max_frames=embed_budget.max_frames,
                        start_sec=analysis_start,
                        end_sec=clip_duration,
                        accelerator=settings.gpu_backend,
                    )
                if hud_detector is not None:
                    if state.qwen_video_input is not None:
                        state.hud_summary = _merge_hud_summaries(
                            state.hud_summary,
                            _detect_hud_for_qwen_frames(state.clip_id, state.qwen_video_input, hud_detector),
                        )
                if settings.allow_mock_models and not Path(full_video_path).exists() and state.qwen_video_input is not None:
                    hit_marker_video_input = state.qwen_video_input
                elif analysis_start < clip_duration:
                    hit_marker_video_input = prepare_hit_marker_video_input(
                        full_video_path,
                        settings.data_dir / "hit_marker_video_inputs",
                        fps=settings.video_embedding_fps,
                        max_frames=settings.qwen_full_video_max_frames,
                        start_sec=analysis_start,
                        end_sec=clip_duration,
                        accelerator=settings.gpu_backend,
                    )
                else:
                    hit_marker_video_input = None
                if hit_marker_video_input is not None:
                    state.hit_marker_summary = detect_hit_marker_evidence(
                        hit_marker_video_input.frame_paths,
                        sample_fps=hit_marker_video_input.sample_fps,
                        frame_timestamps=hit_marker_video_input.metadata.get("hit_marker_frame_timestamps_sec"),
                        active_weapon=state.hud_summary.get("active_weapon"),
                        equipment_timeline=state.hud_summary.get("equipment_timeline"),
                    )
            state.metadata_payload = _metadata_payload(
                state.clip,
                user_metadata={
                    "hud": state.hud_summary,
                    "death_screen": state.death_summary,
                    "hit_marker": state.hit_marker_summary,
                    "known_outcome": _known_clip_outcome(state.clip),
                },
            )
            state.metadata_payload = _metadata_with_timebase(state.metadata_payload, state.timebase)
            state.manifest = _clip_manifest(state.clip, state.metadata_payload)
            state.windows = [_media_window(state.clip, segment) for segment in state.segments]
        except Exception as exc:  # noqa: BLE001 - one failed clip must not stop the batch.
            _fail_indexing_state(db, state, "prepare", exc)


def _run_asr_indexing_stage(
    settings: AppSettings,
    db: Database,
    states: list[_IndexingClipState],
    progress_callback: object | None,
    cancel_event: asyncio.Event | None,
) -> None:
    active_states = _active_indexing_states(states)
    if not active_states:
        return
    transcriber = _transcriber(settings)
    total = len(active_states)
    for position, state in enumerate(active_states, start=1):
        _check_cancel(cancel_event)
        _progress(
            progress_callback,
            f"preparing full-clip Whisper ASR audio: {state.clip['filename']}",
            _stage_progress(0.26, 0.30, position, total),
            {"clip_id": state.clip_id, "stage": "asr_audio"},
        )
        audio_source = None
        if settings.enable_transcription:
            try:
                audio_source = _asr_audio_source_for_state(settings, state)
            except Exception as exc:  # noqa: BLE001 - ASR should degrade without killing AV indexing.
                db.add_processing_event(state.clip_id, "asr_audio", "failed", str(exc))
        if settings.enable_transcription and audio_source:
            try:
                _progress(
                    progress_callback,
                    f"Whisper timestamped ASR transcription: {state.clip['filename']}",
                    _stage_progress(0.30, 0.40, position, total),
                    {
                        "clip_id": state.clip_id,
                        "stage": "asr",
                        "return_timestamps": True,
                    },
                )
                result = transcriber.transcribe(audio_source)
                state.asr_audio_path = str(audio_source)
                state.asr_model_id = result.engine or transcriber.config.engine
                state.asr_language = result.language or settings.asr_language
                state.transcript_text = result.text
                timestamp_offset = state.timebase.analysis_start_sec if state.timebase is not None else 0.0
                state.transcript_segments = [
                    {
                        "start_time": item.start + timestamp_offset,
                        "end_time": item.end + timestamp_offset,
                        "text": item.text,
                        "confidence": item.confidence,
                        "model_name": state.asr_model_id,
                    }
                    for item in result.segments
                    if item.text.strip()
                ]
            except Exception as exc:  # noqa: BLE001 - ASR should degrade without killing AV indexing.
                db.add_processing_event(state.clip_id, "asr_transcription", "failed", str(exc))
        if not state.transcript_segments and state.transcript_text:
            state.transcript_segments = [
                {"text": state.transcript_text, "model_name": state.asr_model_id or _transcriber_engine(transcriber, settings)}
            ]
        db.replace_transcripts(state.clip_id, state.transcript_segments)
        state.transcript_artifact = _asr_transcript_artifact(
            state.clip,
            state.transcript_text,
            state.transcript_segments,
            settings,
            model_id=state.asr_model_id or _transcriber_engine(transcriber, settings),
            language=state.asr_language or settings.asr_language,
        )


def _run_audio_caption_indexing_stage(
    settings: AppSettings,
    db: Database,
    states: list[_IndexingClipState],
    progress_callback: object | None,
    cancel_event: asyncio.Event | None,
) -> None:
    active_states = _active_indexing_states(states)
    if not active_states:
        return
    manager = _model_runtime_manager(settings)
    audio_captioner = AudioCaptionerAdapter(
        model_for_role("audio_captioner", settings.model_tier, device_backend=settings.gpu_backend),
        manager=manager,
        mock_fallback=settings.allow_mock_models,
    )
    total = len(active_states)
    for position, state in enumerate(active_states, start=1):
        _check_cancel(cancel_event)
        try:
            if state.manifest is None:
                raise RuntimeError("clip manifest is unavailable before audio captioning")
            _progress(
                progress_callback,
                f"preparing non-speech audio caption windows: {state.clip['filename']}",
                _stage_progress(0.40, 0.44, position, total),
                {
                    "clip_id": state.clip_id,
                    "stage": "audio_caption_windowing",
                    "window_sec": audio_captioner.window_sec,
                    "stride_sec": audio_captioner.stride_sec,
                },
            )
            state.audio_caption_windows = _audio_caption_windows_for_state(settings, state, audio_captioner)
        except Exception as exc:  # noqa: BLE001 - one failed clip must not stop the batch.
            _fail_indexing_state(db, state, "audio_caption", exc)
    caption_states = _active_indexing_states(active_states)
    total = len(caption_states)
    for position, state in enumerate(caption_states, start=1):
        _check_cancel(cancel_event)
        try:
            if state.manifest is None:
                raise RuntimeError("clip manifest is unavailable before audio captioning")
            _progress(
                progress_callback,
                f"MiDashengLM timestamped audio captions: {state.clip['filename']}",
                _stage_progress(0.44, 0.52, position, total),
                {
                    "clip_id": state.clip_id,
                    "stage": "audio_caption",
                    "window_count": len(state.audio_caption_windows),
                },
            )
            state.audio_captions = audio_captioner.caption_windows(state.manifest, state.audio_caption_windows)
        except Exception as exc:  # noqa: BLE001 - one failed clip must not stop the batch.
            _fail_indexing_state(db, state, "audio_caption", exc)


def _run_fusion_indexing_stage(
    settings: AppSettings,
    db: Database,
    states: list[_IndexingClipState],
    progress_callback: object | None,
    cancel_event: asyncio.Event | None,
) -> None:
    active_states = _active_indexing_states(states)
    if not active_states:
        return
    manager = _model_runtime_manager(settings)
    summarizer_spec = _summarizer_spec(settings)
    hunt_knowledge = _hunt_knowledge(settings)
    fusion_summarizer = FusionSummarizerAdapter(
        summarizer_spec,
        model_id=summarizer_spec.model_id,
        manager=manager,
        mock_fallback=settings.allow_mock_models,
        reasoning_mode=settings.qwen_reasoning_mode,
        reasoning_budget_tokens=settings.qwen_reasoning_budget_tokens,
        weapon_resolver=_hunt_weapon_skin_resolver(hunt_knowledge),
        weapon_skin_map=_hunt_weapon_skin_map(hunt_knowledge),
    )
    total = len(active_states)
    for position, state in enumerate(active_states, start=1):
        _check_cancel(cancel_event)
        try:
            if state.manifest is None or state.metadata_payload is None:
                raise RuntimeError("clip manifest or metadata is unavailable before fusion")
            if state.transcript_artifact is None:
                state.transcript_artifact = _asr_transcript_artifact(
                    state.clip,
                    state.transcript_text,
                    state.transcript_segments,
                    settings,
                )
            if _should_scan_death_screen(state.clip):
                _progress(
                    progress_callback,
                    f"Qwen3.5 death screen analysis: {state.clip['filename']}",
                    _stage_progress(0.50, 0.52, position, total),
                    {"clip_id": state.clip_id, "stage": "qwen_death_screen"},
                )
                state.death_summary = _detect_death_screen_with_qwen(
                    db,
                    state,
                    fusion_summarizer,
                    hunt_knowledge,
                )
                state.death_text = _death_summary_text(state.death_summary, expected=True)
                state.metadata_payload = _metadata_with_user_metadata(
                    state.metadata_payload,
                    death_screen=state.death_summary,
                )
                state.manifest = state.manifest.model_copy(update={"metadata": state.metadata_payload})
            _progress(
                progress_callback,
                f"Qwen3.5 visual OCR over prepared frames: {state.clip['filename']}",
                _stage_progress(0.52, 0.58, position, total),
                {"clip_id": state.clip_id, "stage": "qwen_visual_ocr"},
            )
            media_windows = _fusion_media_windows(state)
            state.qwen_visual_ocr = fusion_summarizer.extract_visual_ocr(
                state.manifest,
                media_windows=_ocr_media_windows(media_windows),
                metadata=state.metadata_payload,
            )
            state.metadata_payload = metadata_with_qwen_visual_ocr(state.metadata_payload, state.qwen_visual_ocr)
            state.manifest = state.manifest.model_copy(update={"metadata": state.metadata_payload})
            _progress(
                progress_callback,
                f"Qwen3.5 visual event extraction and ledger composition: {state.clip['filename']}",
                _stage_progress(0.58, 0.72, position, total),
                {
                    "clip_id": state.clip_id,
                    "stage": "evidence_ledger_fusion",
                    "qwen_visual_ocr_observation_count": len(state.qwen_visual_ocr.get("observations") or []),
                },
            )
            if hasattr(fusion_summarizer, "extract_visual_events") and hasattr(fusion_summarizer, "summarize_from_ledger"):
                state.visual_events = fusion_summarizer.extract_visual_events(
                    state.manifest,
                    media_windows=media_windows,
                    metadata=state.metadata_payload,
                    video_payload_budgets=state.video_payload_budgets,
                )
                state.evidence_ledger = build_evidence_ledger(
                    state.manifest,
                    timebase=state.timebase or _clip_timebase(state.clip, settings),
                    metadata=state.metadata_payload,
                    transcript=state.transcript_artifact,
                    audio_captions=state.audio_captions,
                    visual_events=state.visual_events,
                    hit_marker_summary=state.hit_marker_summary,
                    death_screen_summary=state.death_summary,
                    video_payload_budgets=state.video_payload_budgets,
                    weapon_resolver=_hunt_weapon_skin_resolver(hunt_knowledge),
                )
                state.fused_summary = fusion_summarizer.summarize_from_ledger(
                    state.manifest,
                    ledger=state.evidence_ledger,
                )
                state.video_observations = visual_events_to_observations(
                    state.evidence_ledger,
                    model_id=summarizer_spec.model_id,
                )
            elif hasattr(fusion_summarizer, "summarize_with_observations"):
                state.video_observations, state.fused_summary = fusion_summarizer.summarize_with_observations(
                    state.manifest,
                    media_windows=media_windows,
                    transcript=state.transcript_artifact,
                    audio_captions=state.audio_captions,
                    metadata=state.metadata_payload,
                )
            else:
                state.video_observations = fusion_summarizer.observe_video_windows(state.manifest, media_windows)
                state.fused_summary = fusion_summarizer.summarize(
                    state.manifest,
                    video_observations=state.video_observations,
                    media_windows=media_windows,
                    transcript=state.transcript_artifact,
                    audio_captions=state.audio_captions,
                    metadata=state.metadata_payload,
                )
            state.summary = state.fused_summary.detailed_summary
            state.tags = sorted(set(state.fused_summary.tags))
            db.execute("UPDATE clips SET summary=? WHERE id=?", (state.summary, state.clip_id))
            db.set_clip_tags(state.clip_id, state.tags, source="hf_fusion")
        except Exception as exc:  # noqa: BLE001 - one failed clip must not stop the batch.
            _fail_indexing_state(db, state, "fusion", exc)


def _run_embedding_indexing_stage(
    settings: AppSettings,
    db: Database,
    states: list[_IndexingClipState],
    vector_store: QdrantVectorStore,
    progress_callback: object | None,
    cancel_event: asyncio.Event | None,
) -> None:
    active_states = _active_indexing_states(states)
    if not active_states:
        return
    embedder = _embedder(settings)
    total = len(active_states)
    for position, state in enumerate(active_states, start=1):
        _check_cancel(cancel_event)
        try:
            if state.manifest is None or state.metadata_payload is None or state.fused_summary is None:
                raise RuntimeError("fusion artifacts are unavailable before embedding")
            if state.transcript_artifact is None:
                state.transcript_artifact = _asr_transcript_artifact(
                    state.clip,
                    state.transcript_text,
                    state.transcript_segments,
                    settings,
                )
            _progress(
                progress_callback,
                f"Qwen3-VL multimodal embeddings: {state.clip['filename']}",
                _stage_progress(0.72, 0.98, position, total),
                {"clip_id": state.clip_id, "stage": "embedding"},
            )
            _embed_indexing_state(settings, db, state, embedder, vector_store)
            state.completed = True
            db.update_clip_status(state.clip_id, "indexed", indexed=True)
            _progress(
                progress_callback,
                f"completed: {state.clip['filename']}",
                _stage_progress(0.98, 1.0, position, total),
                {"clip_id": state.clip_id, "segments": getattr(state.segment_summary, "total_segments", len(state.segments))},
            )
        except Exception as exc:  # noqa: BLE001 - one failed clip must not stop the batch.
            _fail_indexing_state(db, state, "embedding", exc)


def _embed_indexing_state(
    settings: AppSettings,
    db: Database,
    state: _IndexingClipState,
    embedder: HuggingFaceMultimodalEmbedder,
    vector_store: QdrantVectorStore,
) -> None:
    clip = state.clip
    manifest = state.manifest
    metadata_payload = state.metadata_payload
    fused_summary = state.fused_summary
    transcript_artifact = state.transcript_artifact
    if manifest is None or metadata_payload is None or fused_summary is None or transcript_artifact is None:
        raise RuntimeError("embedding requires manifest, metadata, transcript, and fused summary artifacts")

    vector_store.delete_by_clip_id(state.clip_id)
    full_video_path = str(clip["path"] or "")
    if full_video_path:
        vector_id = f"clip-{state.clip_id}-video-full"
        payload_text = _video_embedding_payload_text(manifest, fused_summary, scope="full_clip")
        qwen_video_input = state.qwen_embedding_video_input or state.qwen_video_input or prepare_qwen_video_input(
            full_video_path,
            settings.data_dir / "qwen_video_embedding_inputs",
            fps=settings.video_embedding_fps,
            max_frames=settings.qwen_video_embed_max_frames,
            start_sec=state.timebase.analysis_start_sec if state.timebase is not None else 0.0,
            end_sec=manifest.duration_sec,
            accelerator=settings.gpu_backend,
        )
        video_embedding_metadata = {
            "embedding_scope": "full_clip",
            "video_input_mode": qwen_video_input.mode,
            "video_fps": settings.video_embedding_fps,
            "video_max_frames": settings.qwen_video_embed_max_frames,
            "video_max_pixels": settings.qwen_video_embed_max_pixels,
            **qwen_video_input.metadata,
        }
        vector = embedder.embed_video_frames(
            qwen_video_input.frame_paths,
            text=payload_text,
            fps=settings.video_embedding_fps,
            max_frames=settings.qwen_video_embed_max_frames,
        )
        vector_store.add_vector(
            "video",
            vector_id,
            vector,
            _embedding_payload(
                clip,
                state.tags,
                field="video",
                vector_id=vector_id,
                payload_text=payload_text,
                model_id=settings.tier.multimodal_retrieval_model,
                start_sec=state.timebase.analysis_start_sec if state.timebase is not None and manifest.duration_sec > 0 else (0.0 if manifest.duration_sec > 0 else None),
                end_sec=manifest.duration_sec if manifest.duration_sec > 0 else None,
                video_path=full_video_path,
                payload_ref=full_video_path,
                extra_metadata=video_embedding_metadata,
            ),
        )
    text_items = []
    if state.summary:
        vector_id = f"clip-{state.clip_id}-summary"
        vector_store.add_vector(
            "summary",
            vector_id,
            embedder.embed_text(state.summary),
            _embedding_payload(
                clip,
                state.tags,
                field="summary",
                vector_id=vector_id,
                payload_text=state.summary,
                model_id=settings.tier.multimodal_retrieval_model,
            ),
        )
        text_items.append(_text_item_payload("summary", state.summary, vector_id, settings))
    if state.transcript_text:
        vector_id = f"clip-{state.clip_id}-speech"
        vector_store.add_vector(
            "speech",
            vector_id,
            embedder.embed_text(state.transcript_text),
            _embedding_payload(
                clip,
                state.tags,
                field="speech",
                vector_id=vector_id,
                payload_text=state.transcript_text,
                model_id=settings.tier.multimodal_retrieval_model,
            ),
        )
        text_items.append(_text_item_payload("speech", state.transcript_text, vector_id, settings))
    for caption in state.audio_captions:
        vector_id = f"clip-{state.clip_id}-audio-{caption.window_id}"
        vector_store.add_vector(
            "audio_caption",
            vector_id,
            embedder.embed_text(caption.text),
            _embedding_payload(
                clip,
                state.tags,
                field="audio_caption",
                vector_id=vector_id,
                payload_text=caption.text,
                model_id=settings.tier.multimodal_retrieval_model,
                window_id=caption.window_id,
                start_sec=caption.start_sec,
                end_sec=caption.end_sec,
            ),
        )
    metadata_text = _metadata_embedding_text(metadata_payload, tags=state.tags, summary=state.summary)
    metadata_vector_id = f"clip-{state.clip_id}-metadata"
    vector_store.add_vector(
        "metadata",
        metadata_vector_id,
        embedder.embed_text(metadata_text),
        _embedding_payload(
            clip,
            state.tags,
            field="metadata",
            vector_id=metadata_vector_id,
            payload_text=metadata_text,
            model_id=settings.tier.multimodal_retrieval_model,
            extra_metadata=metadata_payload.model_dump(),
        ),
    )
    text_items.append(_text_item_payload("metadata", metadata_text, metadata_vector_id, settings))
    fused_text = _fused_embedding_text(fused_summary, transcript_artifact, state.audio_captions, metadata_payload)
    fused_vector_id = f"clip-{state.clip_id}-fused"
    vector_store.add_vector(
        "fused",
        fused_vector_id,
        embedder.embed_text(fused_text),
        _embedding_payload(
            clip,
            state.tags,
            field="fused",
            vector_id=fused_vector_id,
            payload_text=fused_text,
            model_id=settings.tier.multimodal_retrieval_model,
        ),
    )
    text_items.append(_text_item_payload("fused", fused_text, fused_vector_id, settings))
    db.replace_text_items(state.clip_id, text_items)


def _fusion_media_windows(state: _IndexingClipState) -> list[MediaWindowV1]:
    manifest = state.manifest
    qwen_video_input = state.qwen_video_input
    if manifest is None or qwen_video_input is None:
        return state.windows
    analysis_start = state.timebase.analysis_start_sec if state.timebase is not None else 0.0
    windows = [
        MediaWindowV1(
            clip_id=manifest.clip_id,
            file_name=manifest.file_name,
            window_id="window_full_clip",
            start_sec=analysis_start,
            end_sec=manifest.duration_sec,
            duration_sec=max(manifest.duration_sec - analysis_start, 0.001),
            prepared_video_frame_paths=qwen_video_input.frame_paths,
            prepared_video_sample_fps=qwen_video_input.sample_fps,
            prepared_video_metadata=qwen_video_input.metadata,
        )
    ]
    if state.qwen_focus_video_input is not None:
        focus_input = state.qwen_focus_video_input
        start_sec = max(analysis_start, min(manifest.duration_sec, SUMMARY_KILL_FOCUS_START_SEC))
        end_sec = max(start_sec, min(manifest.duration_sec, SUMMARY_KILL_FOCUS_END_SEC))
        if end_sec > start_sec:
            windows.append(
                MediaWindowV1(
                    clip_id=manifest.clip_id,
                    file_name=manifest.file_name,
                    window_id=SUMMARY_KILL_FOCUS_WINDOW_ID,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration_sec=max(end_sec - start_sec, 0.001),
                    prepared_video_frame_paths=focus_input.frame_paths,
                    prepared_video_sample_fps=focus_input.sample_fps,
                    prepared_video_metadata=focus_input.metadata,
                )
            )
    return windows


def _ocr_media_windows(media_windows: list[MediaWindowV1]) -> list[MediaWindowV1]:
    base_windows = [window for window in media_windows if window.window_id != SUMMARY_KILL_FOCUS_WINDOW_ID]
    return base_windows or media_windows


def _asr_audio_source_for_state(
    settings: AppSettings,
    state: _IndexingClipState,
    *,
    overwrite: bool = False,
) -> str | None:
    manifest = state.manifest
    if manifest is None:
        raise RuntimeError("clip manifest is unavailable before ASR audio preparation")
    if manifest.duration_sec <= 0:
        return None
    source_path = str(state.clip["path"] or manifest.file_path or "")
    if not source_path:
        return None
    output_path = settings.data_dir / "asr_audio" / f"clip_{state.clip_id}" / "full_clip_16k_mono.wav"
    start_seconds = state.timebase.analysis_start_sec if state.timebase is not None else 0.0
    if start_seconds >= manifest.duration_sec:
        return None
    duration_seconds = max(0.001, manifest.duration_sec - start_seconds)
    ok, error = extract_audio_segment_to_path(
        source_path,
        output_path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        overwrite=overwrite,
    )
    if not ok:
        raise RuntimeError(f"failed to prepare full-clip Whisper ASR audio for {state.clip['filename']}: {error}")
    return str(output_path)


def _audio_caption_windows_for_state(
    settings: AppSettings,
    state: _IndexingClipState,
    audio_captioner: AudioCaptionerAdapter,
    *,
    overwrite: bool = False,
) -> list[MediaWindowV1]:
    manifest = state.manifest
    if manifest is None:
        raise RuntimeError("clip manifest is unavailable before audio caption windowing")
    source_path = str(state.clip["path"] or manifest.file_path or "")
    if not source_path:
        return []
    ranges = _audio_caption_window_ranges(
        manifest.duration_sec,
        audio_captioner.window_sec,
        audio_captioner.stride_sec,
        start_sec=state.timebase.analysis_start_sec if state.timebase is not None else 0.0,
    )
    output_root = (
        settings.data_dir
        / "audio_caption_windows"
        / f"clip_{state.clip_id}"
        / _audio_caption_window_profile(audio_captioner.window_sec, audio_captioner.stride_sec)
    )
    windows: list[MediaWindowV1] = []
    for index, (start_sec, end_sec) in enumerate(ranges):
        duration_sec = max(0.001, end_sec - start_sec)
        audio_path = output_root / (
            f"audio_caption_{index:06d}_"
            f"{int(round(start_sec * 1000)):09d}_{int(round(end_sec * 1000)):09d}.wav"
        )
        ok, error = extract_audio_segment_to_path(
            source_path,
            audio_path,
            start_seconds=start_sec,
            duration_seconds=duration_sec,
            overwrite=overwrite,
        )
        if not ok:
            raise RuntimeError(
                "failed to prepare MiDashengLM audio caption window "
                f"{index} ({start_sec:.3f}-{end_sec:.3f}s) for {state.clip['filename']}: {error}"
            )
        windows.append(
            MediaWindowV1(
                clip_id=manifest.clip_id,
                file_name=manifest.file_name,
                window_id=f"audio_caption_{index:06d}",
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=duration_sec,
                audio_path=str(audio_path),
            )
        )
    return windows


def _audio_caption_window_ranges(
    duration_sec: float,
    window_sec: float,
    stride_sec: float,
    *,
    start_sec: float = 0.0,
    min_window_sec: float = AUDIO_CAPTION_MIN_WINDOW_SECONDS,
) -> list[tuple[float, float]]:
    duration = max(0.0, float(duration_sec or 0.0))
    if duration <= 0.0:
        return []
    start_floor = max(0.0, min(duration, float(start_sec or 0.0)))
    if start_floor >= duration:
        return []
    window = min(duration, 30.0, max(float(window_sec), min_window_sec))
    stride = min(window, max(float(stride_sec), min_window_sec))
    ranges: list[tuple[float, float]] = []
    start_sec = start_floor
    epsilon = 0.001
    while start_sec < duration - epsilon:
        end_sec = min(duration, start_sec + window)
        if end_sec - start_sec >= min_window_sec or not ranges:
            ranges.append((round(start_sec, 3), round(end_sec, 3)))
        if end_sec >= duration - epsilon:
            break
        next_start = start_sec + stride
        if next_start <= start_sec + epsilon:
            break
        start_sec = next_start
    if ranges and ranges[-1][1] < duration - epsilon:
        tail_start = max(start_floor, duration - window)
        if tail_start > ranges[-1][0] + epsilon:
            ranges.append((round(tail_start, 3), round(duration, 3)))
    return ranges


def _audio_caption_window_profile(window_sec: float, stride_sec: float) -> str:
    return f"win_{_seconds_token(window_sec)}_stride_{_seconds_token(stride_sec)}"


def _seconds_token(value: float) -> str:
    compact = f"{value:.3f}".rstrip("0").rstrip(".")
    return re.sub(r"[^0-9A-Za-z]+", "_", compact).strip("_") or "0"


def _active_indexing_states(states: list[_IndexingClipState]) -> list[_IndexingClipState]:
    return [state for state in states if not state.failed]


def _fail_indexing_state(
    db: Database | None,
    state: _IndexingClipState,
    stage: str,
    exc: Exception,
) -> None:
    state.failed = True
    state.error = str(exc)
    if db is not None:
        db.update_clip_status(state.clip_id, "failed", state.error)
        try:
            db.add_processing_event(state.clip_id, stage, "failed", state.error)
        except Exception:
            pass


def _stage_progress(start: float, end: float, position: int, total: int) -> float:
    if total <= 0:
        return end
    return start + (end - start) * min(max(position / total, 0.0), 1.0)


def run_search(
    params: dict[str, Any] | None = None,
    *,
    query: str | None = None,
    top_k: int | None = None,
    limit: int | None = None,
    group_name: str | None = None,
    modalities: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    progress_callback: object | None = None,
    **_: Any,
) -> dict[str, Any]:
    settings = get_settings()
    payload = params or {}
    query = query or payload.get("query") or ""
    if not query.strip():
        return SearchResponse(query=query, results=[], warnings=["empty query"]).model_dump()
    reranking_enabled = _effective_reranking_enabled(settings, payload)
    top = _bounded_int(top_k or limit or payload.get("top_k") or payload.get("limit") or 10, default=10, lower=1, upper=500)
    candidate_limit = _bounded_int(
        _first_present(payload, "candidate_limit", "max_results", default=settings.search_candidate_limit),
        default=settings.search_candidate_limit,
        lower=max(top, 1),
        upper=500,
    )
    min_score = _bounded_float(
        _first_present(payload, "min_score", "score_threshold", default=settings.search_min_score),
        default=settings.search_min_score,
        lower=0.0,
        upper=1.0,
    )
    filters = filters or payload.get("filters") or {}
    group_name = group_name or payload.get("group_name") or filters.get("game") or filters.get("group_name")
    modalities = modalities or payload.get("modalities") or filters.get("modalities") or []

    db = Database(settings.db_path)
    try:
        embedder = _embedder(settings)
        vector_store = _vector_store(settings, embedder.dimension)
        query_vector = embedder.embed_query(query)
        qfilter: dict[str, Any] = {}
        if group_name:
            qfilter["group_name"] = group_name
        warnings: list[str] = []
        retrieval_settings = RetrievalSettings(per_field_top_k=candidate_limit, rerank_top_n=candidate_limit, final_top_k=top)
        hits_by_field = {}
        for index, field in enumerate(("video", "summary", "speech", "audio_caption", "metadata", "fused"), start=1):
            _progress(progress_callback, f"searching {field} vector field", 0.15 + index * 0.10)
            hits_by_field[field] = vector_store.search(field, query_vector, retrieval_settings.per_field_top_k, qfilter)
        retrieval = late_fuse_hits(query, hits_by_field, settings=retrieval_settings)
        hits = [(candidate.combined_score, candidate.payload) for candidate in retrieval.candidates]
        if modalities:
            allowed = set(modalities)
            hits = [item for item in hits if str(item[1].get("field")) in allowed or str(item[1].get("modality")) in allowed]
        if not hits:
            warnings.append("No vector hits found; returning lightweight SQLite text matches.")
            results = _sqlite_search(db, query, candidate_limit, group_name)
        else:
            results = [_hit_to_result(db, score, payload, query=query) for score, payload in hits[:candidate_limit]]
        results = [result for result in results if result is not None]
        ranked_results = results
        if reranking_enabled and results:
            reranked = _rerank_search_results(settings, query, results)
            ranked_results = reranked.results
            if reranked.warning:
                warnings.append(reranked.warning)
        results = _filter_search_results_by_threshold(ranked_results, min_score=min_score)
        if ranked_results and not results:
            warnings.append(f"No results met the relevance threshold ({min_score:.2f}).")
        _progress(progress_callback, "preparing result cards", 1.0)
        response = SearchResponse(query=query, results=results, warnings=warnings).model_dump()
        response["fallback"] = bool(warnings)
        response["min_score"] = min_score
        response["candidate_limit"] = candidate_limit
        response["result_count"] = len(results)
        return response
    finally:
        manager = _model_runtime_manager(settings)
        if manager.loaded is not None:
            manager.unload()
        db.close()


def run_analysis(
    params: dict[str, Any] | None = None,
    *,
    clip_id: int | str | None = None,
    progress_callback: object | None = None,
    **_: Any,
) -> dict[str, Any]:
    settings = get_settings()
    payload = params or {}
    clip_id = clip_id or payload.get("clip_id")
    question = str(payload.get("question") or payload.get("query") or "")
    if clip_id is None:
        raise ValueError("clip_id is required")
    db = Database(settings.db_path)
    try:
        clip = db.get_clip(int(clip_id))
        if clip is None:
            raise ValueError(f"clip not found: {clip_id}")
        _progress(progress_callback, "loading transcript and segments", 0.25)
        transcripts = db.get_transcripts(int(clip_id))
        transcript_text = " ".join(str(row["text"]) for row in transcripts)
        tags = db.get_clip_tags(int(clip_id))
        hud_summary = db.hud_loadout_summary(int(clip_id))
        hud_text = _hud_summary_text(hud_summary)
        embedder = _embedder(settings)
        knowledge = _hunt_knowledge(settings, embedder)
        death_summary = _resolve_death_summary(knowledge, db.death_screen_summary(int(clip_id)))
        death_text = _death_summary_text(death_summary)
        knowledge_hits = _hunt_facts_for_analysis(knowledge, question, hud_summary, death_summary)
        knowledge_text = format_knowledge_hits(knowledge_hits, max_chars=700)
        _progress(progress_callback, "calling reasoning model", 0.75)
        if settings.enable_deep_reasoning:
            reasoner = GameplayReasoner()
            prompt = " ".join(
                part
                for part in [
                    question or f"Analyze clip {clip_id}",
                    transcript_text or str(clip["filename"]),
                    hud_text,
                    death_text,
                    knowledge_text,
                ]
                if part
            )
            answer = reasoner.answer(prompt, []).answer
        else:
            answer = str(clip["summary"] or _metadata_text(clip, tags, summary="", extra=hud_text))
            if hud_text:
                answer = f"{answer}\n\nDetected HUD loadout: {hud_text}"
            if death_text:
                answer = f"{answer}\n\nDetected death screen: {death_text}"
            if knowledge_text:
                answer = f"{answer}\n\nHunt knowledge:\n{knowledge_text}"
        response = AnalyzeResponse(
            clip_id=int(clip_id),
            description=answer,
            important_events=tags[:5],
            cues=[tag for tag in tags if tag in {"footsteps", "gunshots", "explosion", "dogs", "crows"}],
            tactical_observations=[item for item in [hud_text, death_text] if item],
            uncertainty_notes=[] if transcript_text else ["No transcript evidence was available."],
            model_name=(
                model_for_role("summarizer", settings.model_tier, device_backend=settings.gpu_backend).model_id
                if settings.enable_deep_reasoning
                else "heuristic"
            ),
            runtime_backend=_runtime_backend(settings),
        )
        dumped = response.model_dump()
        dumped["detected_loadout"] = hud_summary.get("loadout", [])
        dumped["active_weapon"] = hud_summary.get("active_weapon")
        dumped["active_equipment"] = hud_summary.get("active_equipment")
        dumped["active_equipment_type"] = hud_summary.get("active_equipment_type")
        dumped["death_status"] = death_summary.get("status")
        dumped["killed_by_weapon"] = death_summary.get("killed_with")
        dumped["killer_name"] = death_summary.get("killer_name")
        dumped["knowledge_facts"] = [hit.__dict__ for hit in knowledge_hits]
        _progress(progress_callback, "analysis complete", 1.0)
        return dumped
    finally:
        manager = _model_runtime_manager(settings)
        if manager.loaded is not None:
            manager.unload()
        db.close()


def _embedder(settings: AppSettings) -> HuggingFaceMultimodalEmbedder:
    backend = None
    spec = _embedder_spec(settings)
    if not settings.allow_mock_models:
        backend = TransformersEmbeddingBackend(
            spec,
            manager=_model_runtime_manager(settings),
            video_fps=settings.video_embedding_fps,
            video_max_frames=settings.qwen_video_embed_max_frames,
        )
    return HuggingFaceMultimodalEmbedder(
        EmbeddingConfig(
            model_name=settings.tier.multimodal_retrieval_model,
            dimension=_embedding_dimension(settings),
            modality="multimodal",
            mock_fallback=settings.allow_mock_models,
            model_path=settings.models_dir / "hub",
            runtime_backend=_runtime_backend(settings),
            precision=_embedding_precision(settings),
            video_fps=settings.video_embedding_fps,
            video_max_frames=settings.qwen_video_embed_max_frames,
        ),
        backend=backend,
    )


def _summarizer_spec(settings: AppSettings):
    spec = model_for_role("summarizer", settings.model_tier, device_backend=settings.gpu_backend)
    max_input = dict(spec.max_input or {})
    max_input.update(
        {
            "video_fps": settings.video_embedding_fps,
            "video_max_frames": settings.qwen_full_video_max_frames,
            "video_max_pixels": settings.qwen_full_video_max_pixels,
            "max_pixels": settings.qwen_full_video_max_pixels,
            "focus_video_max_frames": settings.qwen_focus_video_max_frames,
            "focus_video_max_pixels": settings.qwen_focus_video_max_pixels,
            "ocr_video_max_frames": settings.qwen_ocr_video_max_frames,
            "ocr_video_max_pixels": settings.qwen_ocr_video_max_pixels,
        }
    )
    return replace(spec, max_input=max_input)


def _embedder_spec(settings: AppSettings):
    spec = model_for_role("embedder", settings.model_tier, device_backend=settings.gpu_backend)
    max_input = dict(spec.max_input or {})
    max_input.update(
        {
            "video_fps": settings.video_embedding_fps,
            "video_max_frames": settings.qwen_video_embed_max_frames,
            "video_max_pixels": settings.qwen_video_embed_max_pixels,
            "max_pixels": settings.qwen_video_embed_max_pixels,
        }
    )
    return replace(spec, max_input=max_input)


def _transcriber(settings: AppSettings) -> Transcriber:
    backend = None
    asr_spec = model_for_role("asr", settings.model_tier, device_backend=settings.gpu_backend)
    if not settings.allow_mock_models and settings.enable_transcription:
        backend = TransformersASRBackend(
            asr_spec,
            manager=_model_runtime_manager(settings),
        )
    return Transcriber(
        TranscriptionConfig(
            engine=asr_spec.model_id,
            language=settings.asr_language,
            mock_fallback=settings.allow_mock_models,
        ),
        backend=backend,
    )


def _transcriber_engine(transcriber: Any, settings: AppSettings) -> str:
    config = getattr(transcriber, "config", None)
    engine = getattr(config, "engine", None)
    return str(engine or settings.tier.asr_model)


def _vector_store(settings: AppSettings, dimension: int) -> QdrantVectorStore:
    return QdrantVectorStore(
        QdrantVectorStoreConfig(
            url=settings.qdrant_url,
            dimension=dimension,
            mock_fallback=settings.allow_mock_models,
            local_path=str(settings.data_dir / "qdrant"),
        )
    )


def _payload(
    clip: Any,
    tags: list[str],
    *,
    collection_kind: str,
    modality: str,
    precision: str,
    runtime_backend: str,
    embedding_model: str,
    start_time: float | None = None,
    end_time: float | None = None,
    text: str | None = None,
    representative_frame_path: str | None = None,
    video_path: str | None = None,
    hud_loadout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "clip_id": int(clip["id"]),
        "group_name": str(clip["group_name"] or "Ungrouped"),
        "source_path": str(clip["path"]),
        "relative_path": clip["relative_path"],
        "modality": modality,
        "start_time": start_time,
        "end_time": end_time,
        "filename": str(clip["filename"]),
        "tags": tags,
        "embedding_model": embedding_model,
        "collection_kind": collection_kind,
        "scan_status": str(clip["scan_status"]),
        "clip_status": str(clip["status"]),
        "precision": precision,
        "runtime_backend": runtime_backend,
        "model_tier": get_settings().model_tier,
        "summary": clip["summary"],
        "text": text,
        "representative_frame_path": representative_frame_path,
        "video_path": video_path,
        "hud_loadout": hud_loadout or {},
        "active_weapon": (hud_loadout or {}).get("active_weapon") if hud_loadout else None,
        "active_equipment": (hud_loadout or {}).get("active_equipment") if hud_loadout else None,
        "active_equipment_type": (hud_loadout or {}).get("active_equipment_type") if hud_loadout else None,
    }


def _metadata_payload(clip: Any, *, user_metadata: dict[str, Any] | None = None, tags: list[str] | None = None) -> MetadataPayloadV1:
    file_name = str(clip["filename"])
    return MetadataPayloadV1(
        clip_id=int(clip["id"]),
        file_name=file_name,
        file_path=str(clip["path"]) if clip["path"] else None,
        title=file_name,
        description=str(clip["summary"] or "") or None,
        tags=tags or [],
        user_metadata=user_metadata or {},
        technical_metadata={
            "duration": clip["duration"],
            "width": clip["width"],
            "height": clip["height"],
            "fps": clip["fps"],
            "codec": clip["codec"],
            "size_bytes": clip["size_bytes"],
        },
        ingest_metadata={
            "relative_path": clip["relative_path"],
            "source_root": clip["source_root"],
            "group_name": clip["group_name"],
            "scan_status": clip["scan_status"],
            "status": clip["status"],
        },
    )


def _clip_timebase(clip: Any, settings: AppSettings) -> ClipTimebaseV1:
    duration = float(clip["duration"] or 0.0)
    start = analysis_start_sec(duration, settings.video_analysis_skip_start_sec)
    return ClipTimebaseV1(
        clip_id=int(clip["id"]),
        file_name=str(clip["filename"]),
        source_duration_sec=duration,
        analysis_start_sec=start,
        analysis_end_sec=duration,
    )


def _metadata_with_timebase(metadata: MetadataPayloadV1, timebase: ClipTimebaseV1 | None) -> MetadataPayloadV1:
    if timebase is None:
        return metadata
    technical = dict(metadata.technical_metadata)
    technical["timebase"] = timebase.model_dump()
    return metadata.model_copy(update={"technical_metadata": technical})


def _known_clip_outcome(clip: Any) -> str | None:
    filename = str(clip["filename"] or "").lower()
    if "_killed" in filename or "hunter killed" in filename or re.search(r"\bkilled\b", filename):
        return "confirmed_hunter_kill"
    return None


def _metadata_with_user_metadata(metadata: MetadataPayloadV1, **items: Any) -> MetadataPayloadV1:
    user_metadata = dict(metadata.user_metadata)
    user_metadata.update(items)
    return metadata.model_copy(update={"user_metadata": user_metadata})


def _clip_manifest(clip: Any, metadata: MetadataPayloadV1) -> ClipManifestV1:
    return ClipManifestV1(
        clip_id=int(clip["id"]),
        file_name=metadata.file_name,
        file_path=metadata.file_path,
        duration_sec=float(clip["duration"] or 0.0),
        media_type="video",
        metadata=metadata,
        created_at=clip["created_at"] or utc_now(),
        ingest_timestamp=clip["last_seen_at"] or utc_now(),
    )


def _media_window(clip: Any, segment: Any) -> MediaWindowV1:
    return MediaWindowV1(
        clip_id=int(clip["id"]),
        file_name=str(clip["filename"]),
        window_id=f"window_{int(segment['id']):06d}",
        start_sec=float(segment["start_time"] or 0.0),
        end_sec=float(segment["end_time"] or segment["start_time"] or 0.0),
        duration_sec=float(segment["duration"] or 0.0),
        frame_paths=[str(segment["representative_frame_path"])] if segment["representative_frame_path"] else [],
        video_path=str(segment["video_segment_path"]) if segment["video_segment_path"] else None,
        audio_path=str(segment["audio_segment_path"]) if segment["audio_segment_path"] else None,
    )


def _asr_transcript_artifact(
    clip: Any,
    transcript_text: str,
    transcript_segments: list[dict[str, Any]],
    settings: AppSettings,
    *,
    model_id: str | None = None,
    language: str | None = None,
) -> ASRTranscriptV1:
    metadata = _metadata_payload(clip)
    manifest = _clip_manifest(clip, metadata)
    segment_objects = [
        type(
            "Segment",
            (),
            {
                "start": item.get("start_time", 0.0) or 0.0,
                "end": item.get("end_time", item.get("start_time", 0.0) or 0.0) or 0.0,
                "text": item.get("text", ""),
                "confidence": item.get("confidence"),
            },
        )()
        for item in transcript_segments
    ]
    return build_asr_transcript(
        manifest,
        model_id=model_id or settings.tier.asr_model,
        text=transcript_text,
        language=language if language is not None else settings.asr_language,
        segments=segment_objects,
    )


def _embedding_payload(
    clip: Any,
    tags: list[str],
    *,
    field: str,
    vector_id: str,
    payload_text: str,
    model_id: str,
    window_id: str | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    representative_frame_path: Any = None,
    video_path: Any = None,
    payload_ref: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = EmbeddingRecordV1(
        clip_id=int(clip["id"]),
        file_name=str(clip["filename"]),
        field=field,  # type: ignore[arg-type]
        window_id=window_id,
        start_sec=start_sec,
        end_sec=end_sec,
        model_id=model_id,
        embedding_dim=_embedding_dimension(get_settings()),
        payload_hash=payload_hash(payload_text),
        payload_text=payload_text,
        payload_ref=payload_ref,
        metadata=extra_metadata or {},
    )
    payload = _payload(
        clip,
        tags,
        collection_kind=field,
        modality=field,
        start_time=start_sec,
        end_time=end_sec,
        text=payload_text,
        precision=_embedding_record_precision(get_settings()),
        runtime_backend=_runtime_backend(get_settings()),
        embedding_model=model_id,
        representative_frame_path=representative_frame_path,
        video_path=str(video_path) if video_path else None,
    )
    payload.update(record.model_dump())
    payload["field"] = field
    payload["vector_id"] = vector_id
    payload["file_name"] = str(clip["filename"])
    return payload


def _metadata_embedding_text(metadata: MetadataPayloadV1, *, tags: list[str], summary: str) -> str:
    hud_text = _hud_summary_text(metadata.user_metadata.get("hud") or {})
    hit_text = _hit_marker_summary_text(metadata.user_metadata.get("hit_marker") or {})
    return " ".join(
        part
        for part in [
            f"file_name: {metadata.file_name}",
            f"title: {metadata.title}" if metadata.title else "",
            f"description: {metadata.description}" if metadata.description else "",
            f"hud_loadout: {hud_text}" if hud_text else "",
            f"hit_marker: {hit_text}" if hit_text else "",
            f"tags: {', '.join(tags)}" if tags else "",
            f"summary: {summary}" if summary else "",
        ]
        if part
    )


def _hit_marker_summary_text(hit_marker: dict[str, Any]) -> str:
    if not isinstance(hit_marker, dict) or not hit_marker.get("detected"):
        return ""
    description = str(hit_marker.get("description") or "").strip()
    if description:
        return description
    timestamp = hit_marker.get("timestamp")
    confidence = hit_marker.get("confidence")
    parts = ["probable hit marker detected"]
    if timestamp is not None:
        parts.append(f"at {float(timestamp):.2f}s")
    if confidence is not None:
        parts.append(f"confidence {float(confidence):.2f}")
    return " ".join(parts)


def _video_embedding_payload_text(manifest: ClipManifestV1, summary: FusedSummaryV1, *, scope: str) -> str:
    return " ".join(
        part
        for part in [
            f"file_name: {manifest.file_name}",
            f"video_embedding_scope: {scope}",
            f"duration_sec: {manifest.duration_sec:.3f}",
            f"summary: {summary.short_summary}" if summary.short_summary else "",
            "evidence_source: video",
        ]
        if part
    )


def _fused_embedding_text(
    summary: Any,
    transcript: ASRTranscriptV1,
    captions: list[Any],
    metadata: MetadataPayloadV1,
) -> str:
    return " ".join(
        part
        for part in [
            f"file_name: {metadata.file_name}",
            summary.short_summary,
            summary.detailed_summary,
            transcript.text,
            " ".join(caption.text for caption in captions),
            _metadata_embedding_text(metadata, tags=summary.tags, summary=summary.short_summary),
        ]
        if part
    )


def _clip_needs_embedding_rebuild(db: Database, clip_id: int, settings: AppSettings) -> bool:
    expected_model = settings.tier.multimodal_retrieval_model
    expected_precision = _embedding_record_precision(settings)
    if db.query("SELECT 1 FROM av_segments WHERE clip_id=? AND embedding_id IS NOT NULL LIMIT 1", (clip_id,)):
        return True
    rows = db.query(
        """
        SELECT embedding_model, embedding_precision FROM text_items WHERE clip_id=? AND embedding_id IS NOT NULL
        """,
        (clip_id,),
    )
    if not rows:
        return False
    return any(
        str(row["embedding_model"] or "") != expected_model
        or str(row["embedding_precision"] or "") != expected_precision
        for row in rows
    )


def _text_item_payload(kind: str, text: str, embedding_id: str | None, settings: AppSettings) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": kind, "text": text, "embedding_id": embedding_id}
    if embedding_id is not None:
        payload.update(
            {
                "embedding_model": settings.tier.multimodal_retrieval_model,
                "embedding_precision": _embedding_record_precision(settings),
                "runtime_backend": _runtime_backend(settings),
            }
        )
    return payload


def _combine_hits(settings: AppSettings, av_hits: list[Any], transcript_hits: list[Any], metadata_hits: list[Any]) -> list[tuple[float, dict[str, Any]]]:
    by_clip: dict[int, dict[str, Any]] = {}
    scores: dict[int, dict[str, float]] = {}
    for kind, weight, hits in [
        ("av", settings.av_segment_weight, av_hits),
        ("transcript", settings.transcript_weight, transcript_hits),
        ("metadata", settings.metadata_weight, metadata_hits),
    ]:
        for hit in hits:
            clip_id = int(hit.payload.get("clip_id"))
            by_clip.setdefault(clip_id, dict(hit.payload))
            scores.setdefault(clip_id, {})
            scores[clip_id][kind] = max(scores[clip_id].get(kind, 0.0), float(hit.score) * weight)
    combined = []
    for clip_id, payload in by_clip.items():
        final_score = sum(scores.get(clip_id, {}).values())
        if scores.get(clip_id, {}).keys() >= {"av", "transcript"}:
            payload["modality"] = "hybrid"
        combined.append((final_score, payload))
    return sorted(combined, key=lambda item: item[0], reverse=True)


def _resolve_search_timing(
    db: Database,
    clip_id: int,
    payload: dict[str, Any],
    *,
    query: str = "",
) -> dict[str, Any]:
    start_time = _float_or_none(payload.get("start_time"))
    end_time = _float_or_none(payload.get("end_time"))
    if start_time is not None:
        if end_time is None:
            end_time = start_time
        best_timestamp = _float_or_none(payload.get("best_timestamp"))
        return _timing_payload(
            db,
            clip_id,
            best_timestamp=best_timestamp if best_timestamp is not None else start_time,
            segment_start=start_time,
            segment_end=end_time,
            preview_frame=payload.get("representative_frame_path"),
            preserve_segment_bounds=True,
        )

    collection_kind = str(payload.get("collection_kind") or "")
    should_try_transcripts = collection_kind in {"transcript_text", "sqlite_fallback"}
    transcript_anchor = _transcript_timing_anchor(db, clip_id, query) if should_try_transcripts else None
    if transcript_anchor is not None:
        transcript_start, transcript_end = transcript_anchor
        return _timing_payload(
            db,
            clip_id,
            best_timestamp=transcript_start,
            segment_start=transcript_start,
            segment_end=transcript_end,
            preview_frame=payload.get("representative_frame_path"),
            preserve_segment_bounds=False,
        )

    return {}


def _timing_payload(
    db: Database,
    clip_id: int,
    *,
    best_timestamp: float,
    segment_start: float | None,
    segment_end: float | None,
    preview_frame: Any = None,
    preserve_segment_bounds: bool = False,
) -> dict[str, Any]:
    nearest = _nearest_segment(db, clip_id, best_timestamp)
    if nearest is not None:
        if not preserve_segment_bounds or segment_start is None:
            segment_start = _float_or_none(nearest["start_time"])
        if not preserve_segment_bounds or segment_end is None:
            segment_end = _float_or_none(nearest["end_time"])
        preview_frame = preview_frame or nearest["representative_frame_path"]
    return {
        "best_timestamp": best_timestamp,
        "segment_start": segment_start,
        "segment_end": segment_end,
        "preview_frame": preview_frame,
    }


def _transcript_timing_anchor(db: Database, clip_id: int, query: str) -> tuple[float, float | None] | None:
    rows = db.get_transcripts(clip_id)
    timed_rows = [
        row for row in rows
        if _float_or_none(row["start_time"]) is not None
    ]
    if not timed_rows:
        return None
    terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2]
    if terms:
        for row in timed_rows:
            text = str(row["text"] or "").lower()
            if any(term in text for term in terms):
                start = _float_or_none(row["start_time"])
                if start is not None:
                    return start, _float_or_none(row["end_time"])
    row = timed_rows[0]
    start = _float_or_none(row["start_time"])
    return (start, _float_or_none(row["end_time"])) if start is not None else None


def _nearest_segment(db: Database, clip_id: int, timestamp: float) -> Any | None:
    segments = db.list_segments(clip_id)
    if not segments:
        return None
    return min(
        segments,
        key=lambda segment: _segment_distance(segment, timestamp),
    )


def _segment_distance(segment: Any, timestamp: float) -> float:
    start = _float_or_none(segment["start_time"])
    end = _float_or_none(segment["end_time"])
    if start is None and end is None:
        return float("inf")
    if start is None:
        return abs(float(end) - timestamp)
    if end is None:
        return abs(float(start) - timestamp)
    if start <= timestamp <= end:
        return 0.0
    return min(abs(start - timestamp), abs(end - timestamp))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _hit_to_result(db: Database, score: float, payload: dict[str, Any], *, query: str = "") -> SearchResult | None:
    clip = db.get_clip(int(payload["clip_id"]))
    if clip is None:
        return None
    hud_summary = db.hud_loadout_summary(int(clip["id"]))
    death_summary = db.death_screen_summary(int(clip["id"]))
    timing = _resolve_search_timing(db, int(clip["id"]), payload, query=query)
    return SearchResult(
        clip_id=int(clip["id"]),
        clip_filename=str(clip["filename"]),
        source_path=str(clip["path"]),
        group_name=str(clip["group_name"] or "Ungrouped"),
        relative_path=clip["relative_path"],
        best_timestamp=timing.get("best_timestamp"),
        segment_start=timing.get("segment_start"),
        segment_end=timing.get("segment_end"),
        preview_frame=timing.get("preview_frame") or payload.get("representative_frame_path"),
        summary=clip["summary"],
        tags=db.get_clip_tags(int(clip["id"])),
        score=float(score),
        matched_modality=str(payload.get("modality") or "hybrid"),
        matched_reason=_matched_reason(payload),
        transcript_snippet=payload.get("text"),
        active_weapon=hud_summary.get("active_weapon"),
        active_equipment=hud_summary.get("active_equipment"),
        active_equipment_type=hud_summary.get("active_equipment_type"),
        detected_loadout=list(hud_summary.get("loadout") or []),
        death_status=death_summary.get("status"),
        killed_by_weapon=death_summary.get("killed_with"),
        killer_name=death_summary.get("killer_name"),
    )


def _matched_reason(payload: dict[str, Any]) -> str:
    collection_kind = str(payload.get("collection_kind") or "unknown")
    if collection_kind == "player_kill_intent":
        return "Matched player-kill intent from Hunter-killed clip metadata, loadout, and segment timing."
    fields = payload.get("matched_fields") or payload.get("field") or payload.get("collection_kind")
    return f"Matched {fields} with Qwen3-VL multimodal query embedding."


def _rerank_search_results(settings: AppSettings, query: str, results: list[SearchResult]) -> Any:
    reranker = SearchReranker(
        model_name=settings.tier.reranker_model,
        spec=model_for_role("reranker", settings.model_tier, device_backend=settings.gpu_backend),
        manager=_model_runtime_manager(settings),
        runtime_backend=_runtime_backend(settings),
        precision=_reranker_precision(settings),
        mock_fallback=settings.allow_mock_models,
    )
    return reranker.rerank(query, results)


def _filter_search_results_by_threshold(results: list[SearchResult], *, min_score: float) -> list[SearchResult]:
    return [result for result in sorted(results, key=lambda item: item.score, reverse=True) if result.score >= min_score]


def _bounded_int(value: Any, *, default: int, lower: int, upper: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(lower, min(upper, parsed))


def _bounded_float(value: Any, *, default: float, lower: float, upper: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return max(lower, min(upper, parsed))


def _first_present(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def _effective_reranking_enabled(settings: AppSettings, payload: dict[str, Any]) -> bool:
    default = True if settings.enable_reranking is None else bool(settings.enable_reranking)
    value = _first_present(payload, "enable_reranking", "reranking", "use_reranking", default=default)
    return parse_bool(value, default)


def _sqlite_search(db: Database, query: str, top: int, group_name: str | None) -> list[SearchResult]:
    terms = _search_terms(query)
    rows = db.list_clips(group_name=group_name)
    scored: list[tuple[float, Any]] = []
    for row in rows:
        tags = db.get_clip_tags(int(row["id"]))
        transcripts = " ".join(str(item["text"]) for item in db.get_transcripts(int(row["id"])))
        haystack = _metadata_text(row, tags, summary=row["summary"], extra=transcripts).lower()
        compact_haystack = _compact_search_text(haystack)
        score = sum(1.0 for term in terms if term in haystack or _compact_search_text(term) in compact_haystack)
        if score > 0:
            scored.append((score / max(len(terms), 1), row))
    results = []
    for score, row in sorted(scored, key=lambda item: item[0], reverse=True)[:top]:
        hud_summary = db.hud_loadout_summary(int(row["id"]))
        death_summary = db.death_screen_summary(int(row["id"]))
        timing = _resolve_search_timing(
            db,
            int(row["id"]),
            {"clip_id": int(row["id"]), "collection_kind": "sqlite_fallback", "modality": "metadata"},
            query=query,
        )
        results.append(
            SearchResult(
                clip_id=int(row["id"]),
                clip_filename=str(row["filename"]),
                source_path=str(row["path"]),
                group_name=str(row["group_name"] or "Ungrouped"),
                relative_path=row["relative_path"],
                best_timestamp=timing.get("best_timestamp"),
                segment_start=timing.get("segment_start"),
                segment_end=timing.get("segment_end"),
                preview_frame=timing.get("preview_frame"),
                summary=row["summary"],
                tags=db.get_clip_tags(int(row["id"])),
                score=score,
                matched_modality="metadata",
                matched_reason="SQLite text fallback matched metadata, tags, or transcript text.",
                active_weapon=hud_summary.get("active_weapon"),
                active_equipment=hud_summary.get("active_equipment"),
                active_equipment_type=hud_summary.get("active_equipment_type"),
                detected_loadout=list(hud_summary.get("loadout") or []),
                death_status=death_summary.get("status"),
                killed_by_weapon=death_summary.get("killed_with"),
                killer_name=death_summary.get("killer_name"),
            )
        )
    return results


def _metadata_text(clip: Any, tags: list[str], *, summary: str | None = None, extra: str = "") -> str:
    return " ".join(
        str(part)
        for part in [
            clip["filename"],
            clip["group_name"],
            clip["relative_path"],
            summary or clip["summary"],
            " ".join(tags),
            extra,
        ]
        if part
    )


def _hunt_pack_dir(settings: AppSettings) -> Path:
    return settings.data_dir / "packs" / "hunt-knowledge-pack"


def _hunt_knowledge(settings: AppSettings, embedder: HuggingFaceMultimodalEmbedder | None = None) -> HuntKnowledgeService:
    return HuntKnowledgeService(_hunt_pack_dir(settings), embedder=embedder)


def _detect_hud_for_segments(
    db: Database,
    clip_id: int,
    segments: list[Any],
    detector: HudLoadoutDetector,
    *,
    clip: Any | None = None,
) -> dict[str, Any]:
    use_kill_window = _is_player_kill_clip(clip)
    death_segment_ids = {int(row["segment_id"]) for row in db.list_death_screen_detections(clip_id=clip_id)}
    first_death_start: float | None = None
    if death_segment_ids:
        by_id = {int(segment["id"]): segment for segment in segments}
        death_starts = [
            float(by_id[segment_id]["start_time"] or 0.0)
            for segment_id in death_segment_ids
            if segment_id in by_id
        ]
        first_death_start = min(death_starts) if death_starts else None
    frame_paths: list[str] = []
    first_segment_id: int | None = None
    first_timestamp = 0.0
    selected_segment_ids: set[int] = set()
    for segment in segments:
        segment_id = int(segment["id"])
        if _is_death_or_post_death_segment(segment, death_segment_ids, first_death_start) or not _is_hud_detection_window_segment(segment, kill_clip=use_kill_window):
            db.replace_hud_detections(clip_id, int(segment["id"]), [])
            continue
        frame_path = segment["representative_frame_path"]
        if not frame_path:
            db.replace_hud_detections(clip_id, int(segment["id"]), [])
            continue
        frame_paths.append(str(frame_path))
        selected_segment_ids.add(segment_id)
        if first_segment_id is None:
            first_segment_id = segment_id
            first_timestamp = float(segment["start_time"] or 0.0)
        result = detector.detect_frame(frame_path)
        if result is None:
            continue
        rows = detections_to_rows(
            clip_id,
            int(segment["id"]),
            float(segment["start_time"] or 0.0),
            result,
        )
        db.replace_hud_detections(clip_id, int(segment["id"]), rows)
    if first_segment_id is not None and len(frame_paths) > 1:
        aggregate = detector.detect_frames(frame_paths)
        if aggregate is not None and aggregate.loadout_names():
            db.replace_hud_detections(
                clip_id,
                first_segment_id,
                detections_to_rows(clip_id, first_segment_id, first_timestamp, aggregate),
            )
    rows = [
        row
        for row in db.list_hud_detections(clip_id=clip_id)
        if int(row["segment_id"]) in selected_segment_ids
        and int(row["segment_id"]) not in death_segment_ids
        and (first_death_start is None or float(row["timestamp"] or 0.0) < first_death_start)
    ]
    return summarize_detections(rows)


def _detect_hud_for_qwen_frames(
    clip_id: int,
    qwen_video_input: QwenVideoInput,
    detector: HudLoadoutDetector,
) -> dict[str, Any]:
    timestamps = qwen_video_input.metadata.get("qwen_video_frame_timestamps_sec")
    if not isinstance(timestamps, list):
        timestamps = []
    rows: list[dict[str, Any]] = []
    for index, frame_path in enumerate(qwen_video_input.frame_paths):
        timestamp = _qwen_frame_timestamp(index, qwen_video_input.sample_fps, timestamps)
        result = detector.detect_frame(frame_path)
        if result is None:
            continue
        for row in detections_to_rows(clip_id, 0, timestamp, result):
            row["frame_index"] = index
            row["source"] = "qwen_prepared_frame_ocr"
            rows.append(row)
    summary = summarize_detections(rows)
    summary["prepared_frame_evidence"] = rows
    summary["equipment_timeline"] = _equipment_timeline_from_hud_rows(rows)
    summary["qwen_prepared_frame_count"] = len(qwen_video_input.frame_paths)
    return summary


def _merge_hud_summaries(base: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    if not prepared.get("prepared_frame_evidence") and not prepared.get("equipment_timeline"):
        return base
    merged = dict(base or {})
    prepared_evidence = [row for row in prepared.get("prepared_frame_evidence") or [] if isinstance(row, dict)]
    representative_evidence = [row for row in prepared.get("evidence") or [] if isinstance(row, dict)]
    merged["prepared_frame_evidence"] = prepared_evidence
    merged["qwen_prepared_frame_count"] = prepared.get("qwen_prepared_frame_count")
    merged["equipment_timeline"] = _equipment_timeline_from_hud_rows(
        [*prepared_evidence, *[row for row in merged.get("evidence") or [] if isinstance(row, dict)]]
    )
    merged["evidence"] = _dedupe_hud_rows([*[row for row in merged.get("evidence") or [] if isinstance(row, dict)], *representative_evidence])
    for key in ("active_weapon", "active_equipment", "active_equipment_type"):
        if prepared.get(key):
            merged[key] = prepared[key]
    merged["loadout"] = _dedupe_names([*[str(item) for item in merged.get("loadout") or []], *[str(item) for item in prepared.get("loadout") or []]])
    return merged


def _equipment_timeline_from_hud_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    last_key: tuple[str, str] | None = None
    for row in sorted(rows, key=lambda item: float(item.get("timestamp") or 0.0)):
        if not row.get("is_active"):
            continue
        name = str(row.get("entity_name") or "").strip()
        entity_type = str(row.get("entity_type") or "").strip()
        if not name:
            continue
        key = (name.lower(), entity_type.lower())
        timestamp = float(row.get("timestamp") or 0.0)
        if key == last_key and timeline:
            timeline[-1]["end_timestamp"] = round(timestamp, 3)
            timeline[-1]["end_frame_index"] = row.get("frame_index")
            timeline[-1]["sample_count"] = int(timeline[-1].get("sample_count") or 1) + 1
            timeline[-1]["confidence"] = max(float(timeline[-1].get("confidence") or 0.0), float(row.get("confidence") or 0.0))
            continue
        timeline.append(
            {
                "timestamp": round(timestamp, 3),
                "start_timestamp": round(timestamp, 3),
                "end_timestamp": round(timestamp, 3),
                "frame_index": row.get("frame_index"),
                "end_frame_index": row.get("frame_index"),
                "frame_path": row.get("frame_path"),
                "entity_id": row.get("entity_id"),
                "entity_name": name,
                "entity_type": entity_type or None,
                "confidence": float(row.get("confidence") or 0.0),
                "sample_count": 1,
            }
        )
        last_key = key
    return timeline


def _qwen_frame_timestamp(index: int, sample_fps: float, timestamps: list[Any]) -> float:
    if index < len(timestamps):
        try:
            return float(timestamps[index])
        except (TypeError, ValueError):
            pass
    return index / sample_fps if sample_fps > 0 else float(index)


def _dedupe_hud_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, Any, Any]] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("timestamp"), row.get("frame_path"), row.get("slot_key"), row.get("entity_name"))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _dedupe_names(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def _is_first_hud_window_segment(segment: Any) -> bool:
    start = float(segment["start_time"] or 0.0)
    end = float(segment["end_time"] or start)
    return start >= 0.0 and end <= HUD_DETECTION_WINDOW_SECONDS


def _is_hud_detection_window_segment(segment: Any, *, kill_clip: bool = False) -> bool:
    if not kill_clip:
        return _is_first_hud_window_segment(segment)
    start = float(segment["start_time"] or 0.0)
    end = float(segment["end_time"] or start)
    return start < HUD_KILL_DETECTION_END_SECONDS and end > HUD_KILL_DETECTION_START_SECONDS


def _is_player_kill_clip(clip: Any | None) -> bool:
    if clip is None:
        return False
    try:
        filename = str(clip["filename"]).lower()
    except Exception:
        filename = str(clip).lower()
    return "hunter killed" in filename or "enemy killed" in filename or "hunter_killed" in filename


def _is_death_or_post_death_segment(segment: Any, death_segment_ids: set[int], first_death_start: float | None) -> bool:
    if int(segment["id"]) in death_segment_ids:
        return True
    if first_death_start is None:
        return False
    return float(segment["end_time"] or segment["start_time"] or 0.0) >= first_death_start


def _detect_death_screen_with_qwen(
    db: Database,
    state: _IndexingClipState,
    summarizer: FusionSummarizerAdapter,
    knowledge: HuntKnowledgeService | None = None,
) -> dict[str, Any]:
    if state.manifest is None:
        raise RuntimeError("clip manifest is unavailable before Qwen death-screen analysis")
    candidates = _death_screen_frame_candidates(
        state.segments,
        start_sec=state.timebase.analysis_start_sec if state.timebase is not None else 0.0,
    )
    result = summarizer.extract_death_screen(state.manifest, frame_candidates=candidates)
    detected_segment_ids: set[int] = set()
    for detection in result.get("detections") or []:
        if not isinstance(detection, dict):
            continue
        try:
            segment_id = int(detection["segment_id"])
        except (KeyError, TypeError, ValueError):
            continue
        detected_segment_ids.add(segment_id)
        killed_with = detection.get("killed_with")
        resolved = _resolve_weapon_name(knowledge, killed_with)
        db.replace_death_screen_detection(
            state.clip_id,
            segment_id,
            {
                "frame_path": detection.get("frame_path"),
                "timestamp": detection.get("timestamp"),
                "status": detection.get("status"),
                "killed_with": resolved or killed_with,
                "killer_name": detection.get("killer_name"),
                "raw_text": detection.get("raw_text"),
                "confidence": detection.get("confidence"),
            },
        )
    for candidate in candidates:
        try:
            segment_id = int(candidate["segment_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if segment_id not in detected_segment_ids:
            db.replace_death_screen_detection(state.clip_id, segment_id, None)
    return db.death_screen_summary(state.clip_id)


def _death_screen_frame_candidates(segments: list[Any], *, start_sec: float = 0.0) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, segment in enumerate(sorted(segments, key=lambda row: float(row["start_time"] or 0.0), reverse=True)[:6], start=1):
        frame_path = segment["representative_frame_path"]
        if not frame_path:
            continue
        start = float(segment["start_time"] or 0.0)
        end = float(segment["end_time"] or start)
        if end < start_sec:
            continue
        candidates.append(
            {
                "frame_id": f"death_candidate_{index:03d}",
                "segment_id": int(segment["id"]),
                "frame_path": str(frame_path),
                "timestamp": start,
                "start": start,
                "end": end,
            }
        )
    return candidates


def _resolve_death_summary(knowledge: HuntKnowledgeService, death_summary: dict[str, Any]) -> dict[str, Any]:
    resolved = _resolve_weapon_name(knowledge, death_summary.get("killed_with"))
    if not resolved:
        return death_summary
    output = dict(death_summary)
    output["killed_with"] = resolved
    return output


def _resolve_weapon_name(knowledge: HuntKnowledgeService | None, value: Any) -> str | None:
    if knowledge is None or not knowledge.available or not value:
        return None
    resolution = knowledge.resolve_equipment(str(value), entity_types={"weapon"})
    return resolution.display_name if resolution is not None else None


def _hunt_weapon_skin_resolver(knowledge: HuntKnowledgeService | None):
    if knowledge is None or not knowledge.weapon_skin_map():
        return None

    def resolve(text: str) -> str | None:
        return knowledge.resolve_weapon_skin_display(text)

    return resolve


def _hunt_weapon_skin_map(knowledge: HuntKnowledgeService | None) -> dict[str, str]:
    if knowledge is None:
        return {}
    return knowledge.weapon_skin_map()


def _hud_summary_text(hud_summary: dict[str, Any]) -> str:
    loadout = [str(item) for item in hud_summary.get("loadout") or [] if str(item).strip()]
    active = str(hud_summary.get("active_equipment") or hud_summary.get("active_weapon") or "").strip()
    parts = []
    if active:
        parts.append(f"active weapon or item: {active}")
    if loadout:
        parts.append("detected loadout: " + ", ".join(loadout))
    return "; ".join(parts)


def _death_summary_text(death_summary: dict[str, Any], *, expected: bool = False) -> str:
    status = str(death_summary.get("status") or "").strip()
    killed_with = str(death_summary.get("killed_with") or "").strip()
    killer_name = str(death_summary.get("killer_name") or "").strip()
    if killed_with:
        parts = [f"player was {status}" if status else "player was killed/downed", f"killed with: {killed_with}"]
        if killer_name:
            parts.append(f"killer: {killer_name}")
        return "; ".join(parts)
    if expected:
        return "player was killed/downed; killed-with weapon: unknown"
    return ""


def _append_detection_context(
    summary: str,
    hud_summary: dict[str, Any],
    death_summary: dict[str, Any],
    *,
    expected_death: bool = False,
) -> str:
    base = summary.strip()
    hud_text = _hud_summary_text(hud_summary) or "detected loadout: unknown; active weapon or item: unknown"
    death_text = _death_summary_text(death_summary, expected=expected_death)
    context = f"Equipped info: {hud_text}."
    if death_text:
        context = f"{context} Death info: {death_text}."
    if context.lower() in base.lower():
        return base
    return f"{base} {context}".strip()


def _hud_tags(hud_summary: dict[str, Any]) -> list[str]:
    tags = []
    for name in hud_summary.get("loadout") or []:
        slug = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
        if slug:
            tags.append(f"loadout:{slug}")
    active = hud_summary.get("active_equipment") or hud_summary.get("active_weapon")
    if active:
        slug = re.sub(r"[^a-z0-9]+", "-", str(active).lower()).strip("-")
        if slug:
            tags.append(f"active:{slug}")
    return sorted(set(tags))


def _death_tags(death_summary: dict[str, Any]) -> list[str]:
    tags = []
    status = death_summary.get("status")
    if status:
        tags.append(f"death-status:{status}")
    killed_with = death_summary.get("killed_with")
    if killed_with:
        slug = re.sub(r"[^a-z0-9]+", "-", str(killed_with).lower()).strip("-")
        if slug:
            tags.append(f"killed-with:{slug}")
    killer_name = death_summary.get("killer_name")
    if killer_name:
        slug = re.sub(r"[^a-z0-9]+", "-", str(killer_name).lower()).strip("-")
        if slug:
            tags.append(f"killer:{slug}")
    return sorted(set(tags))


def _should_scan_death_screen(clip: Any) -> bool:
    filename = str(clip["filename"] or "").lower()
    return any(token in filename for token in ("player downed", "you're down", "youre down", "you're dead", "youre dead", "death"))


def _search_terms(query: str) -> list[str]:
    terms: list[str] = []
    for term in re.findall(r"[a-z0-9]+", query.lower()):
        if len(term) <= 2:
            continue
        terms.append(term)
        match = re.fullmatch(r"([a-z]+)([0-9]+)", term)
        if match and len(match.group(1)) > 2:
            terms.append(match.group(1))
            terms.append(f"{match.group(1)}-{match.group(2)}")
            terms.append(f"{match.group(1)} {match.group(2)}")
    return _dedupe_search_terms(terms)


def _dedupe_search_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for term in terms:
        cleaned = term.strip().lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            output.append(cleaned)
    return output


def _compact_search_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _query_has_death_intent(query: str) -> bool:
    normalized = query.lower()
    return bool(
        re.search(
            r"\b(killed me|kill me|killed by|killed with|what killed|who killed|i was killed|i got killed|"
            r"player downed|i was downed|i got downed|downed me|you'?re down|death|dead)\b",
            normalized,
        )
    )


def _query_has_player_kill_intent(query: str) -> bool:
    normalized = query.lower()
    if _query_has_death_intent(normalized):
        return False
    if re.search(r"\b(hunter killed|enemy killed|my kill|my down|final kill)\b", normalized):
        return True
    if re.search(r"\b(i|we)\s+(kill|killed|down|downed|shot|headshot)\b", normalized):
        return True
    return bool(re.search(r"\b(kill|killed|downed|headshot)\b", normalized))


def _boost_player_kill_intent(
    db: Database,
    query: str,
    hits: list[tuple[float, dict[str, Any]]],
    *,
    top: int,
    group_name: str | None,
) -> list[tuple[float, dict[str, Any]]]:
    if not _query_has_player_kill_intent(query):
        return hits
    by_clip = {int(payload["clip_id"]): (score, dict(payload)) for score, payload in hits}
    for clip in db.list_clips(group_name=group_name):
        clip_id = int(clip["id"])
        is_player_kill = _is_player_kill_clip(clip)
        death_summary = db.death_screen_summary(clip_id)
        has_player_death = bool(death_summary.get("status") or death_summary.get("killed_with"))
        if not is_player_kill and not has_player_death:
            continue
        score, payload = by_clip.get(clip_id, (0.0, _player_kill_payload(db, clip)))
        payload = dict(payload)
        if is_player_kill:
            boost = 4.5
            hud_summary = db.hud_loadout_summary(clip_id)
            if _hud_matches_query(hud_summary, query):
                boost += 1.0
            if "window" in query.lower() and _clip_mentions(clip, db.get_clip_tags(clip_id), "window"):
                boost += 0.5
            payload.setdefault("modality", "player_kill")
            payload["collection_kind"] = "player_kill_intent"
            if payload.get("start_time") is None:
                payload.update(_player_kill_window_payload(db, clip_id))
            by_clip[clip_id] = (score + boost, payload)
        elif has_player_death:
            payload["player_death_penalty"] = True
            by_clip[clip_id] = (score - 2.5, payload)
    return sorted(by_clip.values(), key=lambda item: item[0], reverse=True)[: max(top, len(hits))]


def _player_kill_payload(db: Database, clip: Any) -> dict[str, Any]:
    payload = {
        "clip_id": int(clip["id"]),
        "group_name": clip["group_name"],
        "filename": clip["filename"],
        "modality": "player_kill",
        "collection_kind": "player_kill_intent",
        "text": clip["summary"],
    }
    payload.update(_player_kill_window_payload(db, int(clip["id"])))
    return payload


def _player_kill_window_payload(db: Database, clip_id: int) -> dict[str, Any]:
    segments = db.list_segments(clip_id)
    candidates = [
        segment
        for segment in segments
        if _float_or_none(segment["start_time"]) is not None
        and _float_or_none(segment["end_time"]) is not None
        and float(segment["start_time"]) < HUD_KILL_DETECTION_END_SECONDS
        and float(segment["end_time"]) > HUD_KILL_DETECTION_START_SECONDS
    ]
    if not candidates:
        return {}
    segment = min(
        candidates,
        key=lambda row: abs(float(row["end_time"] or row["start_time"] or 0.0) - HUD_KILL_DETECTION_END_SECONDS),
    )
    return {
        "best_timestamp": _float_or_none(segment["end_time"]),
        "start_time": _float_or_none(segment["start_time"]),
        "end_time": _float_or_none(segment["end_time"]),
        "representative_frame_path": segment["representative_frame_path"],
    }


def _hud_matches_query(hud_summary: dict[str, Any], query: str) -> bool:
    compact_query = _compact_search_text(query)
    names = [
        hud_summary.get("active_weapon"),
        hud_summary.get("active_equipment"),
        *(hud_summary.get("loadout") or []),
    ]
    return any(name and _compact_search_text(name) in compact_query for name in names)


def _clip_mentions(clip: Any, tags: list[str], term: str) -> bool:
    haystack = _metadata_text(clip, tags, summary=clip["summary"]).lower()
    return term.lower() in haystack


def _boost_hud_matches(
    db: Database,
    query: str,
    hits: list[tuple[float, dict[str, Any]]],
    *,
    top: int,
    group_name: str | None,
) -> list[tuple[float, dict[str, Any]]]:
    terms = _search_terms(query)
    rows = db.search_hud_detections(terms, group_name=group_name)
    by_clip = {int(payload["clip_id"]): (score, dict(payload)) for score, payload in hits}
    for row in rows:
        clip_id = int(row["clip_id"])
        boost = 0.35 + (0.15 if int(row["is_active"] or 0) else 0.0) + min(float(row["confidence"] or 0.0), 1.0) * 0.1
        if clip_id in by_clip:
            score, payload = by_clip[clip_id]
            payload = dict(payload)
            payload["hud_match"] = row["entity_name"]
            if payload.get("start_time") is None and row["timestamp"] is not None:
                payload["start_time"] = row["timestamp"]
                payload["end_time"] = row["timestamp"]
            by_clip[clip_id] = (score + boost, payload)
        else:
            by_clip[clip_id] = (
                boost,
                {
                    "clip_id": clip_id,
                    "group_name": row["group_name"],
                    "filename": row["filename"],
                    "modality": "hud_loadout",
                    "collection_kind": "hud_loadout",
                    "start_time": row["timestamp"],
                    "end_time": row["timestamp"],
                    "text": row["entity_name"],
                    "hud_match": row["entity_name"],
                },
            )
    return sorted(by_clip.values(), key=lambda item: item[0], reverse=True)[: max(top, len(hits))]


def _boost_death_matches(
    db: Database,
    query: str,
    hits: list[tuple[float, dict[str, Any]]],
    *,
    top: int,
    group_name: str | None,
) -> list[tuple[float, dict[str, Any]]]:
    terms = _search_terms(query)
    rows = db.search_death_screen_detections(terms, group_name=group_name)
    by_clip = {int(payload["clip_id"]): (score, dict(payload)) for score, payload in hits}
    for row in rows:
        clip_id = int(row["clip_id"])
        boost = 0.7 + min(float(row["confidence"] or 0.0), 1.0) * 0.15
        if clip_id in by_clip:
            score, payload = by_clip[clip_id]
            payload = dict(payload)
            payload["death_match"] = row["killed_with"]
            if payload.get("start_time") is None and row["timestamp"] is not None:
                payload["start_time"] = row["timestamp"]
                payload["end_time"] = row["timestamp"]
            by_clip[clip_id] = (score + boost, payload)
        else:
            by_clip[clip_id] = (
                boost,
                {
                    "clip_id": clip_id,
                    "group_name": row["group_name"],
                    "filename": row["filename"],
                    "modality": "death_screen",
                    "collection_kind": "death_screen",
                    "start_time": row["timestamp"],
                    "end_time": row["timestamp"],
                    "text": row["killed_with"],
                    "death_match": row["killed_with"],
                },
            )
    return sorted(by_clip.values(), key=lambda item: item[0], reverse=True)[: max(top, len(hits))]


def _hunt_facts_for_analysis(
    knowledge: HuntKnowledgeService,
    question: str,
    hud_summary: dict[str, Any],
    death_summary: dict[str, Any] | None = None,
) -> list[Any]:
    if not knowledge.available:
        return []
    query_parts = [question]
    query_parts.extend(str(item) for item in hud_summary.get("loadout") or [])
    active_equipment = hud_summary.get("active_equipment")
    if active_equipment:
        query_parts.insert(0, str(active_equipment))
    active = hud_summary.get("active_weapon")
    if active and active != active_equipment:
        query_parts.insert(0, str(active))
    if death_summary:
        killed_with = death_summary.get("killed_with")
        if killed_with:
            query_parts.insert(0, str(killed_with))
    return knowledge.search(" ".join(part for part in query_parts if part), top_k=5)


def _segment_modality_counts(segments: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for segment in segments:
        modality = str(segment["modality"] or "unknown")
        counts[modality] = counts.get(modality, 0) + 1
    return counts


def _runtime_backend(settings: AppSettings) -> str:
    if settings.gpu_backend == "macos-metal":
        return "metal"
    if settings.gpu_backend in {"cuda", "rocm", "cpu"}:
        return settings.gpu_backend
    if settings.runtime_profile == "macos":
        return "metal"
    return "unknown"


def _embedding_dimension(settings: AppSettings) -> int:
    return 4096 if settings.tier.multimodal_retrieval_model.endswith("-8B") else DEFAULT_EMBEDDING_DIMENSION


def _embedding_precision(settings: AppSettings) -> str:
    spec = model_for_role("embedder", settings.model_tier, device_backend=settings.gpu_backend)
    return quantization_for_backend(spec, settings.gpu_backend)


def _embedding_record_precision(settings: AppSettings) -> str:
    return (
        f"{_embedding_precision(settings)}|qwen3-vl-embedding-v2|"
        f"video_fps={settings.video_embedding_fps:g}|video_max_frames={settings.video_embedding_max_frames}"
    )


def _asr_precision(settings: AppSettings) -> str:
    return "transformers"


def _reranker_precision(settings: AppSettings) -> str:
    spec = model_for_role("reranker", settings.model_tier, device_backend=settings.gpu_backend)
    return quantization_for_backend(spec, settings.gpu_backend)


def _model_runtime_manager(settings: AppSettings):
    return transformers_runtime_manager(
        models_dir=settings.models_dir,
        logs_dir=settings.logs_dir,
        gpu_backend=settings.gpu_backend,
        one_model_at_a_time=True,
        torch_compile_mode=settings.torch_compile_mode,
        torch_compile_backend=settings.torch_compile_backend,
        torch_compile_profile=settings.torch_compile_profile,
        generation_cache_implementation=settings.qwen_cache_implementation,
    )


def _progress(callback: object | None, message: str, progress: float, data: dict[str, Any] | None = None) -> None:
    if callable(callback):
        callback(message=message, progress=progress, data=data or {})


def _check_cancel(cancel_event: asyncio.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("operation cancelled")
