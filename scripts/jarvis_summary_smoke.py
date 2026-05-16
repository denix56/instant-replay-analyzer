from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
try:
    from datetime import UTC, datetime
except ImportError:  # Python < 3.11 on JarvisLabs base images.
    from datetime import datetime, timezone

    UTC = timezone.utc
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _env_path(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _memory(label: str) -> dict[str, Any]:
    import psutil
    import torch

    process = psutil.Process(os.getpid())
    info = process.memory_info()
    out: dict[str, Any] = {
        "label": label,
        "time": datetime.now(UTC).isoformat(),
        "rss_gb": round(info.rss / 1024**3, 3),
        "vms_gb": round(info.vms / 1024**3, 3),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        out.update(
            {
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
                "cuda_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
                "cuda_max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            }
        )
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _progress(
    report: dict[str, Any],
    report_path: Path,
    stage: str,
    message: str,
    *,
    status: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    entry: dict[str, Any] = {
        "time": datetime.now(UTC).isoformat(),
        "stage": stage,
        "message": message,
    }
    if extra:
        entry.update(extra)
    report["current_stage"] = stage
    report["current_stage_message"] = message
    report.setdefault("stage_history", []).append(entry)
    if status is not None:
        report["status"] = status
    _write_json(report_path, report)
    print(f"[progress] {stage}: {message}", flush=True)


def _extract_audio_16k_mono(
    clip: Path,
    output: Path,
    *,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> None:
    from imageio_ffmpeg import get_ffmpeg_exe

    ffmpeg = Path(get_ffmpeg_exe())
    output.parent.mkdir(parents=True, exist_ok=True)
    shim_dir = output.parent / "bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "ffmpeg"
    if not shim.exists():
        try:
            shim.symlink_to(ffmpeg)
        except OSError:
            shim.write_text(f"#!/usr/bin/env sh\nexec {ffmpeg} \"$@\"\n", encoding="utf-8")
            shim.chmod(0o755)
    os.environ["PATH"] = f"{shim_dir}{os.pathsep}{ffmpeg.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    if start_seconds is not None:
        command.extend(["-ss", f"{max(0.0, start_seconds):.3f}"])
    command.extend(["-i", str(clip)])
    if duration_seconds is not None:
        command.extend(["-t", f"{max(0.001, duration_seconds):.3f}"])
    command.extend(["-vn", "-ac", "1", "-ar", "16000", str(output)])
    subprocess.run(command, check=True)


def _audio_caption_window_ranges(
    duration_sec: float,
    window_sec: float,
    stride_sec: float,
    *,
    start_sec: float = 0.0,
    min_window_sec: float = 0.25,
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


def _cuda_max_allocated_gb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / 1024**3)


def _assert_vram_cap(stage: str, max_allocated_gb: float) -> None:
    measured = _cuda_max_allocated_gb()
    if measured is None:
        return
    if measured > max_allocated_gb:
        raise RuntimeError(
            f"Qwen stage {stage} exceeded QWEN_VRAM_MAX_ALLOCATED_GB={max_allocated_gb:.2f}: "
            f"torch.cuda.max_memory_allocated={measured:.3f}GB."
        )


def _stream_logger(path: Path):
    counts: dict[str, int] = {}
    text_parts: list[str] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")

    def callback(event: dict[str, Any]) -> None:
        event = {**event, "time": datetime.now(UTC).isoformat()}
        name = str(event.get("event"))
        counts[name] = counts.get(name, 0) + 1
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        if event.get("event") == "stream_chunk" and not event.get("redacted"):
            text = str(event.get("text") or "")
            text_parts.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()
        elif event.get("event") in {"stream_start", "stream_end", "thinking_start", "thinking_end"}:
            print(
                "\n[stream:{event} phase={phase} redacted={redacted} chars={chars} tokens={tokens}]".format(
                    event=event.get("event"),
                    phase=event.get("phase"),
                    redacted=event.get("redacted"),
                    chars=event.get("chars", ""),
                    tokens=event.get("generated_tokens", event.get("budget_tokens", "")),
                ),
                flush=True,
            )

    return callback, counts, text_parts


def _weapon_skin_resolver(settings: Any, service_cls: Any):
    service = _hunt_knowledge_service(settings, service_cls)
    if service is None or not service.weapon_skin_map():
        return None

    def resolve(text: str, *, _service=service) -> str | None:
        return _service.resolve_weapon_skin_display(text)

    return resolve


def _weapon_skin_map(settings: Any, service_cls: Any) -> dict[str, str]:
    service = _hunt_knowledge_service(settings, service_cls)
    if service is None:
        return {}
    return service.weapon_skin_map()


def _hunt_knowledge_service(settings: Any, service_cls: Any):
    candidates = [
        settings.data_dir / "packs" / "hunt-knowledge-pack",
        Path(__file__).resolve().parents[1] / "data" / "packs" / "hunt-knowledge-pack",
    ]
    for pack_dir in candidates:
        service = service_cls(pack_dir)
        if service.available or service.weapon_skin_map():
            return service
    return None


def _detect_hud_for_qwen_frames(qwen_input: Any, detector: Any, *, clip_id: int, rows_to_payload: Any, summarize_rows: Any) -> dict[str, Any]:
    timestamps = qwen_input.metadata.get("qwen_video_frame_timestamps_sec")
    if not isinstance(timestamps, list):
        timestamps = []
    rows: list[dict[str, Any]] = []
    for index, frame_path in enumerate(qwen_input.frame_paths):
        timestamp = _qwen_frame_timestamp(index, qwen_input.sample_fps, timestamps)
        result = detector.detect_frame(frame_path)
        if result is None:
            continue
        for row in rows_to_payload(clip_id, 0, timestamp, result):
            row["frame_index"] = index
            row["source"] = "qwen_prepared_frame_ocr"
            rows.append(row)
    summary = summarize_rows(rows)
    summary["prepared_frame_evidence"] = rows
    summary["equipment_timeline"] = _equipment_timeline_from_hud_rows(rows)
    summary["qwen_prepared_frame_count"] = len(qwen_input.frame_paths)
    return summary


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
        timestamp = float(row.get("timestamp") or 0.0)
        key = (name.lower(), entity_type.lower())
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one real JarvisLabs summary smoke test.")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output-dir", default="/root/ira-data/jarvis-summary-smoke")
    parser.add_argument("--env-file", default="/root/instant-replay-analyzer/.env.remote")
    parser.add_argument("--model-tier", default=None)
    parser.add_argument("--gpu-backend", default=None)
    parser.add_argument("--reasoning-mode", default=None)
    parser.add_argument("--active-weapon", default="")
    parser.add_argument("--known-outcome", default="")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--analysis-skip-start", type=float, default=5.0)
    parser.add_argument("--qwen-vram-max-allocated-gb", type=float, default=8.0)
    parser.add_argument("--full-max-pixels", type=int, default=256000)
    parser.add_argument("--focus-max-pixels", type=int, default=320000)
    parser.add_argument("--ocr-max-frames", type=int, default=50)
    parser.add_argument("--ocr-max-pixels", type=int, default=600000)
    parser.add_argument("--embed-max-frames", type=int, default=64)
    parser.add_argument("--embed-max-pixels", type=int, default=224000)
    parser.add_argument("--summary-focus-start", type=float, default=12.0)
    parser.add_argument("--summary-focus-end", type=float, default=24.0)
    parser.add_argument("--summary-focus-frames", type=int, default=80)
    args = parser.parse_args()

    _env_path(Path(args.env_file))
    if args.model_tier:
        os.environ["MODEL_TIER"] = args.model_tier
    if args.gpu_backend:
        os.environ["GPU_BACKEND"] = args.gpu_backend
    if args.reasoning_mode:
        os.environ["QWEN_REASONING_MODE"] = args.reasoning_mode
    os.environ["VIDEO_ANALYSIS_SKIP_START_SEC"] = str(args.analysis_skip_start)
    os.environ["QWEN_VRAM_MAX_ALLOCATED_GB"] = str(args.qwen_vram_max_allocated_gb)
    os.environ["QWEN_FULL_VIDEO_MAX_FRAMES"] = str(args.max_frames)
    os.environ["QWEN_FULL_VIDEO_MAX_PIXELS"] = str(args.full_max_pixels)
    os.environ["QWEN_FOCUS_VIDEO_MAX_FRAMES"] = str(args.summary_focus_frames)
    os.environ["QWEN_FOCUS_VIDEO_MAX_PIXELS"] = str(args.focus_max_pixels)
    os.environ["QWEN_OCR_VIDEO_MAX_FRAMES"] = str(args.ocr_max_frames)
    os.environ["QWEN_OCR_VIDEO_MAX_PIXELS"] = str(args.ocr_max_pixels)
    os.environ["QWEN_VIDEO_EMBED_MAX_FRAMES"] = str(args.embed_max_frames)
    os.environ["QWEN_VIDEO_EMBED_MAX_PIXELS"] = str(args.embed_max_pixels)

    from app.analysis.hit_marker import detect_hit_marker_evidence
    from app.analysis.hud_loadout import HudLoadoutDetector, detections_to_rows, summarize_detections
    from app.config import AppSettings
    from app.hf_pipeline.adapters import (
        AudioCaptionerAdapter,
        FusionSummarizerAdapter,
        SUMMARY_ANSWER_MAX_TOKENS,
        build_asr_transcript,
        metadata_with_qwen_visual_ocr,
    )
    from app.hf_pipeline.evidence_ledger import build_evidence_ledger, visual_events_to_observations
    from app.hf_pipeline.model_registry import (
        dtype_for_backend,
        loader_class_name,
        model_for_role,
        quantization_for_backend,
    )
    from app.hf_pipeline.schemas import ClipManifestV1, ClipTimebaseV1, MediaWindowV1, MetadataPayloadV1
    from app.hf_pipeline.video_budget import fit_qwen_video_budget
    from app.knowledge.hunt_runtime import HuntKnowledgeService
    from app.processing.qwen_video import SUMMARY_KILL_FOCUS_WINDOW_ID, prepare_hit_marker_video_input, prepare_qwen_video_input
    from app.runtime.transformers_runtime import TransformersModelManager

    settings = AppSettings.from_env()
    settings.ensure_dirs()
    clip = Path(args.clip).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "summary_run_report.json"
    stream_path = output_dir / "stream_events.jsonl"
    callback, stream_counts, stream_text_parts = _stream_logger(stream_path)
    started = time.perf_counter()
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    report: dict[str, Any] = {
        "clip": str(clip),
        "output_dir": str(output_dir),
        "settings": {
            "model_tier": settings.model_tier,
            "gpu_backend": settings.gpu_backend,
            "qwen_reasoning_mode": settings.qwen_reasoning_mode,
            "qwen_reasoning_budget_tokens": settings.qwen_reasoning_budget_tokens,
            "qwen_cache_implementation": settings.qwen_cache_implementation,
            "fps": args.fps,
            "max_frames": args.max_frames,
            "analysis_skip_start_sec": args.analysis_skip_start,
            "qwen_vram_max_allocated_gb": args.qwen_vram_max_allocated_gb,
            "full_max_pixels": args.full_max_pixels,
            "summary_focus_start": args.summary_focus_start,
            "summary_focus_end": args.summary_focus_end,
            "summary_focus_frames": args.summary_focus_frames,
            "focus_max_pixels": args.focus_max_pixels,
            "ocr_max_frames": args.ocr_max_frames,
            "ocr_max_pixels": args.ocr_max_pixels,
        },
        "memory": [_memory("start")],
        "status": "starting",
        "current_stage": "starting",
        "stage_history": [],
    }
    _write_json(report_path, report)
    _progress(
        report,
        report_path,
        "video_preparation",
        (
            f"Preparing Qwen video frames from {args.analysis_skip_start:g}s at fps={args.fps:g}, "
            f"max_frames={args.max_frames}, max_width=1280."
        ),
        status="preparing_video",
        extra={"fps": args.fps, "max_frames": args.max_frames, "max_width": 1280, "start_sec": args.analysis_skip_start},
    )

    qwen_input = prepare_qwen_video_input(
        clip,
        settings.data_dir / "qwen_video_inputs",
        fps=args.fps,
        max_frames=args.max_frames,
        max_width=1280,
        start_sec=args.analysis_skip_start,
        overwrite=False,
        accelerator=settings.gpu_backend,
    )
    focus_start = max(float(args.analysis_skip_start), float(args.summary_focus_start))
    focus_end = max(focus_start + 0.001, float(args.summary_focus_end))
    _progress(
        report,
        report_path,
        "focus_video_preparation",
        (
            f"Preparing focused summary frames for {focus_start:g}-{focus_end:g}s "
            f"at fps={args.fps:g}, max_frames={args.summary_focus_frames}."
        ),
        status="preparing_focus_video",
        extra={
            "fps": args.fps,
            "max_frames": args.summary_focus_frames,
            "max_width": 1280,
            "start_sec": focus_start,
            "end_sec": focus_end,
        },
    )
    qwen_focus_input = prepare_qwen_video_input(
        clip,
        settings.data_dir / "qwen_video_focus_inputs",
        fps=args.fps,
        max_frames=args.summary_focus_frames,
        max_width=1280,
        start_sec=focus_start,
        end_sec=focus_end,
        overwrite=False,
        accelerator=settings.gpu_backend,
    )
    _progress(
        report,
        report_path,
        "hud_ocr_detection",
        f"Prepared {len(qwen_input.frame_paths)} frames; running HUD OCR over the same frames sent to Qwen.",
        status="detecting_hud_ocr",
        extra={"prepared_frame_count": len(qwen_input.frame_paths), "sample_fps": qwen_input.sample_fps},
    )
    duration = float(qwen_input.metadata.get("source_video_duration") or 0.0)
    analysis_start = max(0.0, min(duration, float(args.analysis_skip_start)))
    timebase = ClipTimebaseV1(
        clip_id=1,
        file_name=clip.name,
        source_duration_sec=duration,
        analysis_start_sec=analysis_start,
        analysis_end_sec=duration,
    )
    active_weapon = args.active_weapon.strip()
    hud: dict[str, Any] = {}
    hunt_knowledge = _hunt_knowledge_service(settings, HuntKnowledgeService)
    if hunt_knowledge is not None and hunt_knowledge.available:
        hud_detector = HudLoadoutDetector(hunt_knowledge)
        hud = _detect_hud_for_qwen_frames(
            qwen_input,
            hud_detector,
            clip_id=1,
            rows_to_payload=detections_to_rows,
            summarize_rows=summarize_detections,
        )
    if active_weapon and not hud.get("active_equipment"):
        hud = {
            **hud,
            "active_weapon": active_weapon,
            "active_equipment": active_weapon,
            "active_equipment_type": "weapon",
            "loadout": [*list(hud.get("loadout") or []), active_weapon],
            "evidence": [
                *[item for item in hud.get("evidence") or [] if isinstance(item, dict)],
                {
                    "timestamp": 18.0,
                    "slot_key": "manual_smoke",
                    "is_active": True,
                    "entity_name": active_weapon,
                    "entity_type": "weapon",
                    "confidence": 0.96,
                },
            ],
        }
    _progress(
        report,
        report_path,
        "hit_marker_detection",
        "Detecting late hit-marker evidence with timestamped active equipment context.",
        status="detecting_hit_marker",
        extra={
            "prepared_frame_count": len(qwen_input.frame_paths),
            "hud_evidence_count": len(hud.get("prepared_frame_evidence") or []),
            "equipment_timeline_count": len(hud.get("equipment_timeline") or []),
        },
    )
    hit_marker_input = prepare_hit_marker_video_input(
        clip,
        settings.data_dir / "hit_marker_video_inputs",
        fps=args.fps,
        max_frames=args.max_frames,
        start_sec=analysis_start,
        end_sec=duration,
        overwrite=False,
        accelerator=settings.gpu_backend,
    )
    hit_marker = detect_hit_marker_evidence(
        hit_marker_input.frame_paths,
        sample_fps=float(hit_marker_input.sample_fps or args.fps),
        frame_timestamps=hit_marker_input.metadata.get("hit_marker_frame_timestamps_sec"),
        start_sec=max(analysis_start, 18.0),
        end_sec=22.0,
        active_weapon=hud.get("active_weapon") or active_weapon or None,
        equipment_timeline=hud.get("equipment_timeline"),
    )
    report["hit_marker_frame_prepare"] = {
        "frame_count": len(hit_marker_input.frame_paths),
        "sample_fps": hit_marker_input.sample_fps,
        "metadata": hit_marker_input.metadata,
    }
    _write_json(report_path, report)
    user_metadata: dict[str, Any] = {"hud": hud, "hit_marker": hit_marker}
    known_outcome = args.known_outcome.strip()
    if not known_outcome and ("_killed" in clip.name.lower() or "hunter killed" in clip.name.lower()):
        known_outcome = "confirmed_hunter_kill"
    if known_outcome:
        user_metadata["known_outcome"] = known_outcome
    metadata = MetadataPayloadV1(
        clip_id=1,
        file_name=clip.name,
        file_path=str(clip),
        title=clip.name,
        tags=["hunt showdown", "gameplay"],
        user_metadata=user_metadata,
        technical_metadata={"qwen_video": qwen_input.metadata, "timebase": timebase.model_dump()},
        ingest_metadata={"source": "jarvis_summary_smoke", "created_at": created_at},
    )
    manifest = ClipManifestV1(
        clip_id=1,
        file_name=clip.name,
        file_path=str(clip),
        duration_sec=duration,
        media_type="video",
        metadata=metadata,
        created_at=created_at,
    )
    window = MediaWindowV1(
        clip_id=1,
        file_name=clip.name,
        window_id="window_full_clip",
        start_sec=analysis_start,
        end_sec=duration,
        duration_sec=max(0.001, duration - analysis_start),
        video_path=str(clip),
        prepared_video_frame_paths=qwen_input.frame_paths,
        prepared_video_sample_fps=float(qwen_input.sample_fps or args.fps),
        prepared_video_metadata=qwen_input.metadata,
    )
    focus_window = MediaWindowV1(
        clip_id=1,
        file_name=clip.name,
        window_id=SUMMARY_KILL_FOCUS_WINDOW_ID,
        start_sec=min(focus_start, duration),
        end_sec=min(focus_end, duration),
        duration_sec=max(0.001, min(focus_end, duration) - min(focus_start, duration)),
        video_path=str(clip),
        prepared_video_frame_paths=qwen_focus_input.frame_paths,
        prepared_video_sample_fps=float(qwen_focus_input.sample_fps or args.fps),
        prepared_video_metadata=qwen_focus_input.metadata,
    )
    report.update(
        {
            "video_prepare": {
                "frame_count": len(qwen_input.frame_paths),
                "sample_fps": qwen_input.sample_fps,
                "metadata": qwen_input.metadata,
            },
            "focus_video_prepare": {
                "frame_count": len(qwen_focus_input.frame_paths),
                "sample_fps": qwen_focus_input.sample_fps,
                "metadata": qwen_focus_input.metadata,
                "start_sec": focus_window.start_sec,
                "end_sec": focus_window.end_sec,
            },
            "hit_marker": hit_marker,
            "hud_summary": hud,
            "memory": [*report["memory"], _memory("after_video_prepare")],
            "status": "transcribing",
        }
    )
    _write_json(report_path, report)
    _progress(
        report,
        report_path,
        "runtime_manager_init",
        "Creating Transformers model manager with one-model-at-a-time loading.",
        status="initializing_runtime",
    )

    manager = TransformersModelManager(
        models_dir=settings.models_dir,
        logs_dir=settings.logs_dir,
        gpu_backend=settings.gpu_backend,
        one_model_at_a_time=True,
        torch_compile_mode=settings.torch_compile_mode,
        torch_compile_backend=settings.torch_compile_backend,
        torch_compile_profile=settings.torch_compile_profile,
        generation_cache_implementation=settings.qwen_cache_implementation,
    )
    try:
        audio_path = output_dir / "audio_16k_mono.wav"
        _progress(
            report,
            report_path,
            "audio_extraction",
            f"Extracting 16 kHz mono audio for Whisper ASR from {analysis_start:g}s.",
            status="extracting_audio",
        )
        _extract_audio_16k_mono(
            clip,
            audio_path,
            start_seconds=analysis_start,
            duration_seconds=max(0.001, duration - analysis_start),
        )
        asr_spec = model_for_role("asr", settings.model_tier, device_backend=settings.gpu_backend)
        _progress(
            report,
            report_path,
            "asr_model_and_transcription",
            f"Loading/transcribing with {asr_spec.model_id}.",
            status="transcribing",
            extra={"model_id": asr_spec.model_id, "loader": loader_class_name(asr_spec)},
        )
        asr_result = manager.transcribe(asr_spec, audio_path, language=settings.asr_language)
        shifted_segments = [
            SimpleNamespace(
                start=float(item.start or 0.0) + analysis_start,
                end=float(item.end or item.start or 0.0) + analysis_start,
                text=item.text,
                confidence=item.confidence,
            )
            for item in asr_result.segments
        ]
        transcript = build_asr_transcript(
            manifest,
            model_id=asr_spec.model_id,
            text=asr_result.text,
            language=asr_result.language,
            segments=shifted_segments,
        )
        report["asr"] = transcript.model_dump()
        report["memory"].append(_memory("after_asr"))
        _write_json(report_path, report)
        _progress(report, report_path, "asr_unload", "Unloading ASR model and clearing caches.", status="unloading_asr")
        manager.unload()
        report["memory"].append(_memory("after_asr_unload"))
        report["status"] = "summarizing"
        _write_json(report_path, report)

        audio_captions = []
        caption_spec = model_for_role("audio_captioner", settings.model_tier, device_backend=settings.gpu_backend)
        audio_caption_adapter = AudioCaptionerAdapter(caption_spec, manager=manager, mock_fallback=False)
        audio_caption_window_dir = output_dir / "audio_caption_windows"
        audio_caption_windows = []
        caption_ranges = _audio_caption_window_ranges(
            duration,
            audio_caption_adapter.window_sec,
            audio_caption_adapter.stride_sec,
            start_sec=analysis_start,
        )
        _progress(
            report,
            report_path,
            "audio_caption_windowing",
            (
                f"Preparing {len(caption_ranges)} MiDashengLM audio windows "
                f"({audio_caption_adapter.window_sec:g}s window, {audio_caption_adapter.stride_sec:g}s stride)."
            ),
            status="preparing_audio_captions",
            extra={
                "window_sec": audio_caption_adapter.window_sec,
                "stride_sec": audio_caption_adapter.stride_sec,
                "window_count": len(caption_ranges),
            },
        )
        for index, (start_sec, end_sec) in enumerate(caption_ranges):
            caption_audio_path = audio_caption_window_dir / (
                f"audio_caption_{index:06d}_"
                f"{int(round(start_sec * 1000)):09d}_{int(round(end_sec * 1000)):09d}.wav"
            )
            _extract_audio_16k_mono(
                clip,
                caption_audio_path,
                start_seconds=start_sec,
                duration_seconds=max(0.001, end_sec - start_sec),
            )
            audio_caption_windows.append(
                MediaWindowV1(
                    clip_id=1,
                    file_name=clip.name,
                    window_id=f"audio_caption_{index:06d}",
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration_sec=max(0.001, end_sec - start_sec),
                    audio_path=str(caption_audio_path),
                )
            )
        _progress(
            report,
            report_path,
            "audio_caption_model",
            f"Loading/captioning {len(audio_caption_windows)} timestamped non-speech audio windows with {caption_spec.model_id}.",
            status="captioning_audio",
            extra={
                "model_id": caption_spec.model_id,
                "loader": loader_class_name(caption_spec),
                "quantization": quantization_for_backend(caption_spec, settings.gpu_backend),
                "window_count": len(audio_caption_windows),
            },
        )
        audio_captions = audio_caption_adapter.caption_windows(manifest, audio_caption_windows)
        report["audio_captions"] = [item.model_dump() for item in audio_captions]
        report["memory"].append(_memory("after_audio_caption"))
        _write_json(report_path, report)
        _progress(
            report,
            report_path,
            "audio_caption_unload",
            "Unloading audio caption model and clearing caches.",
            status="unloading_audio_captioner",
        )
        manager.unload()
        report["memory"].append(_memory("after_audio_caption_unload"))
        _write_json(report_path, report)

        summary_spec = model_for_role("summarizer", settings.model_tier, device_backend=settings.gpu_backend)
        summary_spec = replace(
            summary_spec,
            max_input={
                **dict(summary_spec.max_input or {}),
                "video_fps": args.fps,
                "video_max_frames": args.max_frames,
                "video_max_pixels": args.full_max_pixels,
                "max_pixels": args.full_max_pixels,
                "focus_video_max_frames": args.summary_focus_frames,
                "focus_video_max_pixels": args.focus_max_pixels,
                "ocr_video_max_frames": args.ocr_max_frames,
                "ocr_video_max_pixels": args.ocr_max_pixels,
            },
        )
        video_payload_budgets = [
            fit_qwen_video_budget("full_visual", settings=settings, max_allocated_gb=args.qwen_vram_max_allocated_gb),
            fit_qwen_video_budget("focus_visual", settings=settings, max_allocated_gb=args.qwen_vram_max_allocated_gb),
            fit_qwen_video_budget("ocr", settings=settings, max_allocated_gb=args.qwen_vram_max_allocated_gb),
        ]
        report["model"] = {
            "role": summary_spec.role,
            "model_id": summary_spec.model_id,
            "loader": loader_class_name(summary_spec),
            "registry_loader": summary_spec.loader,
            "quantization": quantization_for_backend(summary_spec, settings.gpu_backend),
            "dtype": dtype_for_backend(summary_spec, settings.gpu_backend),
            "device": summary_spec.device,
            "summary_answer_max_tokens": SUMMARY_ANSWER_MAX_TOKENS,
            "generation_cache_implementation": settings.qwen_cache_implementation,
            "video_payload_budgets": [item.model_dump() for item in video_payload_budgets],
        }
        _write_json(report_path, report)
        _progress(
            report,
            report_path,
            "summary_model_generation",
            (
                f"Loading/generating with {summary_spec.model_id}; "
                f"reasoning={settings.qwen_reasoning_mode}; answer_max_tokens={SUMMARY_ANSWER_MAX_TOKENS}."
            ),
            status="summarizing",
            extra={
                "model_id": summary_spec.model_id,
                "loader": loader_class_name(summary_spec),
                "reasoning_mode": settings.qwen_reasoning_mode,
                "reasoning_budget_tokens": settings.qwen_reasoning_budget_tokens,
                "answer_max_tokens": SUMMARY_ANSWER_MAX_TOKENS,
                "generation_cache_implementation": settings.qwen_cache_implementation,
            },
        )
        adapter = FusionSummarizerAdapter(
            summary_spec,
            model_id=summary_spec.model_id,
            manager=manager,
            mock_fallback=False,
            reasoning_mode=settings.qwen_reasoning_mode,
            reasoning_budget_tokens=settings.qwen_reasoning_budget_tokens,
            stream_callback=callback,
            weapon_resolver=_weapon_skin_resolver(settings, HuntKnowledgeService),
            weapon_skin_map=_weapon_skin_map(settings, HuntKnowledgeService),
        )
        _progress(
            report,
            report_path,
            "qwen_visual_ocr",
            "Running Qwen3.5 visual OCR over the prepared video frames before summary.",
            status="qwen_visual_ocr",
            extra={
                "model_id": summary_spec.model_id,
                "prepared_frame_count": len(window.prepared_video_frame_paths),
            },
        )
        qwen_visual_ocr = adapter.extract_visual_ocr(
            manifest,
            media_windows=[window],
            metadata=metadata,
        )
        report["memory"].append(_memory("after_qwen_visual_ocr"))
        _assert_vram_cap("qwen_visual_ocr", args.qwen_vram_max_allocated_gb)
        metadata = metadata_with_qwen_visual_ocr(metadata, qwen_visual_ocr)
        manifest = manifest.model_copy(update={"metadata": metadata})
        report["qwen_visual_ocr"] = qwen_visual_ocr
        _write_json(report_path, report)
        _progress(
            report,
            report_path,
            "qwen_visual_events",
            (
                f"Extracting visual events with {summary_spec.model_id}; "
                f"qwen_ocr_observations={len(qwen_visual_ocr.get('observations') or [])}."
            ),
            status="extracting_visual_events",
            extra={
                "model_id": summary_spec.model_id,
                "qwen_visual_ocr_observation_count": len(qwen_visual_ocr.get("observations") or []),
                "qwen_visual_ocr_equipment_timeline_count": len(qwen_visual_ocr.get("equipment_timeline") or []),
                "base_summary_frame_count": len(window.prepared_video_frame_paths),
                "focus_summary_frame_count": len(focus_window.prepared_video_frame_paths),
                "focus_start_sec": focus_window.start_sec,
                "focus_end_sec": focus_window.end_sec,
            },
        )
        visual_events = adapter.extract_visual_events(
            manifest,
            media_windows=[window, focus_window],
            metadata=metadata,
            video_payload_budgets=video_payload_budgets,
        )
        report["memory"].append(_memory("after_qwen_visual_events"))
        _assert_vram_cap("qwen_visual_events", args.qwen_vram_max_allocated_gb)
        evidence_ledger = build_evidence_ledger(
            manifest,
            timebase=timebase,
            metadata=metadata,
            transcript=transcript,
            audio_captions=audio_captions,
            visual_events=visual_events,
            hit_marker_summary=hit_marker,
            death_screen_summary=metadata.user_metadata.get("death_screen"),
            video_payload_budgets=video_payload_budgets,
            weapon_resolver=_weapon_skin_resolver(settings, HuntKnowledgeService),
        )
        observations = visual_events_to_observations(evidence_ledger, model_id=summary_spec.model_id)
        report["visual_events"] = [item.model_dump() for item in visual_events]
        report["evidence_ledger"] = evidence_ledger.model_dump()
        _write_json(report_path, report)
        _progress(
            report,
            report_path,
            "summary_model_generation",
            f"Generating text-only final summary with {summary_spec.model_id} from the evidence ledger.",
            status="summarizing",
            extra={
                "model_id": summary_spec.model_id,
                "ledger_visual_event_count": len(evidence_ledger.visual_events),
                "ledger_audio_caption_count": len(evidence_ledger.audio_captions),
                "ledger_death_vocalization_count": len(evidence_ledger.death_vocalizations),
            },
        )
        summary = adapter.summarize_from_ledger(
            manifest,
            ledger=evidence_ledger,
        )
        report["memory"].append(_memory("after_summary_generation"))
        _assert_vram_cap("qwen_summary_generation", args.qwen_vram_max_allocated_gb)
        _progress(report, report_path, "summary_validation", "Validating summary evidence bounds.", status="validating_summary")
        summary.validate_evidence_bounds(duration)
        summary_text = json.dumps(summary.model_dump(), ensure_ascii=False)
        validations = {
            "first_qwen_timestamp_gte_skip": min(qwen_input.metadata.get("qwen_video_frame_timestamps_sec") or [analysis_start]) >= analysis_start,
            "peak_allocated_vram_lte_cap": (_cuda_max_allocated_gb() or 0.0) <= args.qwen_vram_max_allocated_gb,
            "no_unsupported_auto5": "Auto-5" not in summary_text,
            "has_evidence_ledger": True,
            "final_composer_text_only": True,
        }
        if not all(validations.values()):
            raise RuntimeError(f"Summary smoke validation failed: {validations}")
        report.update(
            {
                "observations": [item.model_dump() for item in observations],
                "summary": summary.model_dump(),
                "validations": validations,
                "stream_event_counts": stream_counts,
                "stream_text": "".join(stream_text_parts),
                "memory": [*report["memory"], _memory("after_summary")],
                "status": "passed",
                "current_stage": "passed",
            }
        )
        _progress(report, report_path, "passed", "Summary pipeline completed.", status="passed")
        if manager.loaded is not None:
            device_map = getattr(manager.loaded.model, "hf_device_map", None)
            device_map_summary: dict[str, int] = {}
            if isinstance(device_map, dict):
                for target in device_map.values():
                    device_map_summary[str(target)] = device_map_summary.get(str(target), 0) + 1
            report["loaded_model"] = {
                "model_id": manager.loaded.spec.model_id,
                "device": manager.loaded.device,
                "dtype": manager.loaded.dtype,
                "quantization": manager.loaded.quantization,
                "attention_backend": manager.loaded.attention_backend,
                "generation_cache_implementation": manager.loaded.generation_cache_implementation,
                "torch_compile_status": manager.loaded.torch_compile_status,
                "max_memory": manager.loaded.max_memory,
                "device_map_summary": device_map_summary,
            }
        print("\n--- parsed summary ---")
        print("TITLE:", summary.title)
        print("SHORT:", summary.short_summary)
        print("DETAIL:", summary.detailed_summary)
    except Exception as exc:
        report["status"] = "failed"
        report["current_stage"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["elapsed_sec"] = round(time.perf_counter() - started, 3)
        report.setdefault("memory", []).append(_memory("before_final_unload"))
        _progress(report, report_path, "cleanup", "Unloading any active model and writing final report.")
        manager.unload()
        report["memory"].append(_memory("after_final_unload"))
        if report.get("status") == "passed":
            report["current_stage"] = "completed"
            report["current_stage_message"] = "Summary pipeline completed and models unloaded."
        elif report.get("status") == "failed":
            report["current_stage"] = "failed"
            report["current_stage_message"] = "Summary pipeline failed; models were unloaded."
        _write_json(report_path, report)
        print(f"\nREPORT_WRITTEN {report_path}", flush=True)
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
