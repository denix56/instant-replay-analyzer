from __future__ import annotations

import json
import math
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

PROXY_VERSION = "qwen-video-frame-sequence-v16"
HIT_MARKER_PROXY_VERSION = "hit-marker-frame-sequence-v2"
TONE_MAPPING_ALGORITHM = "pyav_direct_autocontrast"
SDR_DARK_ENHANCEMENT_ALGORITHM = "adaptive_sdr_low_light_v1"
FRAME_ENCODING = "png_compress_level_1"
TEMPORAL_SAMPLING_STRATEGY = "equal_time_midpoint_v1"
SUMMARY_KILL_FOCUS_WINDOW_ID = "window_kill_focus_12_24"
SUMMARY_KILL_FOCUS_START_SEC = 12.0
SUMMARY_KILL_FOCUS_END_SEC = 24.0


@dataclass(frozen=True)
class QwenVideoInput:
    source_path: str
    frame_paths: list[str]
    mode: str
    sample_fps: float
    metadata: dict[str, Any] = field(default_factory=dict)


def prepare_qwen_video_input(
    source_path: str | Path,
    cache_dir: str | Path,
    *,
    fps: float,
    max_frames: int,
    max_width: int = 1280,
    start_sec: float | None = None,
    end_sec: float | None = None,
    overwrite: bool = False,
    accelerator: str | None = None,
) -> QwenVideoInput:
    """Prepare one bounded Qwen video payload for both Qwen3.5 and Qwen3-VL embeddings.

    The output is a cached list of SDR RGB frame images generated through PyAV. Passing a
    list keeps qwen-vl-utils on its video path while avoiding unbounded source-video decode
    inside the model runtime.
    """

    source = Path(source_path).expanduser().resolve()
    selected_accelerator = _normalize_video_accelerator(accelerator)
    probe = probe_video_pyav(source)
    window_start_sec, window_end_sec = _sampling_window(
        float(probe.get("duration") or 0.0),
        start_sec=start_sec,
        end_sec=end_sec,
    )
    target_frames = _target_frame_count(probe, fps=fps, max_frames=max_frames)
    metadata = _metadata(
        probe,
        fps=fps,
        max_frames=max_frames,
        target_frames=target_frames,
        max_width=max_width,
        accelerator=selected_accelerator,
        start_sec=window_start_sec,
        end_sec=window_end_sec,
    )
    output_dir = Path(cache_dir).expanduser().resolve() / _cache_key(source, metadata)
    sidecar = output_dir / "metadata.json"
    existing = sorted(output_dir.glob("frame_*.png"))
    if existing and sidecar.exists() and not overwrite:
        cached_metadata = _read_json(sidecar)
        return QwenVideoInput(
            source_path=str(source),
            frame_paths=[str(path) for path in existing],
            mode="sampled_sdr_frame_sequence",
            sample_fps=float(fps),
            metadata={
                **metadata,
                **{
                    key: value
                    for key, value in cached_metadata.items()
                    if key.startswith("qwen_video_")
                },
                "qwen_video_input_mode": "sampled_sdr_frame_sequence",
                "qwen_video_frame_count": len(existing),
                "qwen_video_frame_dir": str(output_dir),
            },
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("frame_*.png"):
        stale.unlink()
    frame_paths, processing_metadata = _sample_frames_to_pngs(
        source,
        output_dir,
        probe=probe,
        fps=fps,
        target_frames=target_frames,
        max_width=max_width,
        accelerator=selected_accelerator,
        start_sec=window_start_sec,
        end_sec=window_end_sec,
    )
    if not frame_paths:
        raise RuntimeError(f"Unable to prepare Qwen video frames: no decodable video frames in {source}")
    frame_timestamps = _read_frame_timestamps(output_dir, len(frame_paths), fps)
    sidecar.write_text(
        json.dumps(
            {
                **metadata,
                **processing_metadata,
                "source_path": str(source),
                "frame_paths": [str(path) for path in frame_paths],
                "qwen_video_frame_timestamps_sec": frame_timestamps,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return QwenVideoInput(
        source_path=str(source),
        frame_paths=[str(path) for path in frame_paths],
        mode="sampled_sdr_frame_sequence",
        sample_fps=float(fps),
        metadata={
            **metadata,
            **processing_metadata,
            "qwen_video_input_mode": "sampled_sdr_frame_sequence",
            "qwen_video_frame_count": len(frame_paths),
            "qwen_video_frame_dir": str(output_dir),
            "qwen_video_frame_timestamps_sec": frame_timestamps,
        },
    )


def prepare_hit_marker_video_input(
    source_path: str | Path,
    cache_dir: str | Path,
    *,
    fps: float,
    max_frames: int,
    start_sec: float | None = None,
    end_sec: float | None = None,
    overwrite: bool = False,
    accelerator: str | None = None,
) -> QwenVideoInput:
    """Prepare full-source-width SDR frames for deterministic hit-marker detection.

    Qwen video inputs are bounded for model memory, but hit-marker detection benefits from
    the original frame resolution. This path uses the same temporal sampling and SDR/HDR
    conversion as Qwen preparation, while avoiding the Qwen max-width downscale.
    """

    source = Path(source_path).expanduser().resolve()
    selected_accelerator = _normalize_video_accelerator(accelerator)
    probe = probe_video_pyav(source)
    window_start_sec, window_end_sec = _sampling_window(
        float(probe.get("duration") or 0.0),
        start_sec=start_sec,
        end_sec=end_sec,
    )
    target_frames = _target_frame_count(probe, fps=fps, max_frames=max_frames)
    source_width = int(probe.get("width") or 0)
    max_width = source_width if source_width > 0 else 100_000
    metadata = {
        **_metadata(
            probe,
            fps=fps,
            max_frames=max_frames,
            target_frames=target_frames,
            max_width=max_width,
            accelerator=selected_accelerator,
            start_sec=window_start_sec,
            end_sec=window_end_sec,
        ),
        "hit_marker_frame_preparation_version": HIT_MARKER_PROXY_VERSION,
        "hit_marker_frame_input_mode": "sampled_sdr_source_width_frame_sequence",
        "hit_marker_frame_preserve_source_width": True,
        "hit_marker_frame_preparation_fps": fps,
        "hit_marker_frame_preparation_max_frames": max_frames,
        "hit_marker_frame_preparation_target_frames": target_frames,
        "hit_marker_frame_preparation_max_width": max_width,
    }
    output_dir = Path(cache_dir).expanduser().resolve() / _cache_key(source, metadata)
    sidecar = output_dir / "metadata.json"
    existing = sorted(output_dir.glob("frame_*.png"))
    if existing and sidecar.exists() and not overwrite:
        cached_metadata = _read_json(sidecar)
        timestamps = cached_metadata.get("hit_marker_frame_timestamps_sec")
        if not isinstance(timestamps, list):
            timestamps = _read_frame_timestamps(output_dir, len(existing), fps)
        return QwenVideoInput(
            source_path=str(source),
            frame_paths=[str(path) for path in existing],
            mode="sampled_sdr_source_width_frame_sequence",
            sample_fps=float(fps),
            metadata={
                **metadata,
                **cached_metadata,
                "hit_marker_frame_input_mode": "sampled_sdr_source_width_frame_sequence",
                "hit_marker_frame_count": len(existing),
                "hit_marker_frame_dir": str(output_dir),
                "hit_marker_frame_timestamps_sec": timestamps,
            },
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("frame_*.png"):
        stale.unlink()
    frame_paths, processing_metadata = _sample_frames_to_pngs(
        source,
        output_dir,
        probe=probe,
        fps=fps,
        target_frames=target_frames,
        max_width=max_width,
        accelerator=selected_accelerator,
        start_sec=window_start_sec,
        end_sec=window_end_sec,
    )
    if not frame_paths:
        raise RuntimeError(f"Unable to prepare hit-marker frames: no decodable video frames in {source}")
    frame_timestamps = _read_frame_timestamps(output_dir, len(frame_paths), fps)
    sidecar.write_text(
        json.dumps(
            {
                **metadata,
                **processing_metadata,
                "source_path": str(source),
                "frame_paths": [str(path) for path in frame_paths],
                "hit_marker_frame_timestamps_sec": frame_timestamps,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return QwenVideoInput(
        source_path=str(source),
        frame_paths=[str(path) for path in frame_paths],
        mode="sampled_sdr_source_width_frame_sequence",
        sample_fps=float(fps),
        metadata={
            **metadata,
            **processing_metadata,
            "hit_marker_frame_input_mode": "sampled_sdr_source_width_frame_sequence",
            "hit_marker_frame_count": len(frame_paths),
            "hit_marker_frame_dir": str(output_dir),
            "hit_marker_frame_timestamps_sec": frame_timestamps,
        },
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_frame_timestamps(output_dir: Path, count: int, fps: float) -> list[float]:
    try:
        values = json.loads((output_dir / "frame_timestamps.json").read_text(encoding="utf-8"))
    except Exception:
        values = []
    if not isinstance(values, list) or len(values) != count:
        step = 1.0 / fps if fps > 0 else 0.0
        return [round(index * step, 6) for index in range(count)]
    timestamps: list[float] = []
    for index, value in enumerate(values):
        try:
            timestamps.append(round(float(value), 6))
        except (TypeError, ValueError):
            step = 1.0 / fps if fps > 0 else 0.0
            timestamps.append(round(index * step, 6))
    return timestamps


def probe_video_pyav(path: str | Path) -> dict[str, Any]:
    try:
        import av
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyAV is required for Qwen video preparation.") from exc

    source = Path(path).expanduser().resolve()
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        duration = _stream_duration(container, stream)
        fps = float(stream.average_rate) if stream.average_rate else None
        frame_count = int(stream.frames or 0) or (int(round(duration * fps)) if duration and fps else None)
        codec = getattr(stream.codec_context, "name", None)
        pix_fmt = getattr(getattr(stream.codec_context, "format", None), "name", None)
        first_frame = next(container.decode(stream), None)
        if first_frame is not None:
            pix_fmt = getattr(first_frame.format, "name", pix_fmt)
            color_space = str(getattr(first_frame, "colorspace", "") or "")
            color_range = str(getattr(first_frame, "color_range", "") or "")
            color_primaries = str(getattr(first_frame, "color_primaries", "") or "")
            color_transfer = str(getattr(first_frame, "color_trc", "") or "")
        else:
            color_space = color_range = color_primaries = color_transfer = ""
        return {
            "path": str(source),
            "duration": duration,
            "width": int(stream.codec_context.width or 0) or None,
            "height": int(stream.codec_context.height or 0) or None,
            "fps": fps,
            "frame_count": frame_count,
            "codec": codec,
            "pix_fmt": pix_fmt,
            "color_space": color_space,
            "color_range": color_range,
            "color_transfer": color_transfer,
            "color_primaries": color_primaries,
        }


def _sample_frames_to_pngs(
    source: Path,
    output_dir: Path,
    *,
    probe: dict[str, Any],
    fps: float,
    target_frames: int,
    max_width: int,
    accelerator: str,
    start_sec: float,
    end_sec: float,
) -> tuple[list[Path], dict[str, Any]]:
    try:
        import av
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyAV is required for Qwen video preparation.") from exc

    duration = float(probe.get("duration") or 0.0)
    target_times = _target_times(duration, target_frames, start_sec=start_sec, end_sec=end_sec)
    selected: list[Path] = []
    selected_timestamps: list[float] = []
    next_index = 0
    hdr_or_10bit = _is_hdr_or_10bit(probe)
    dark_enhanced_count = 0
    worker_count = _video_prepare_workers(target_frames)
    futures: list[Future[bool]] = []
    open_kwargs, accelerator_metadata = _pyav_open_kwargs(av, accelerator)
    try:
        container = av.open(str(source), **open_kwargs)
    except Exception as exc:
        if not open_kwargs:
            raise
        accelerator_metadata = _pyav_hwaccel_unavailable_metadata(
            accelerator,
            _hwaccel_device_type(accelerator),
            f"PyAV HWAccel open failed: {exc}",
        )
        container = av.open(str(source))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        with container:
            stream = container.streams.video[0]
            try:
                stream.thread_type = "AUTO"
            except Exception:
                pass
            accelerator_metadata = _pyav_hwaccel_stream_metadata(stream, accelerator, accelerator_metadata)
            for frame in container.decode(stream):
                if next_index >= len(target_times):
                    break
                timestamp = float(frame.time or 0.0)
                if timestamp + 1e-6 < target_times[next_index]:
                    continue
                path = output_dir / f"frame_{next_index:04d}.png"
                futures.append(
                    executor.submit(
                        _prepare_and_save_frame,
                        frame.to_image().convert("RGB"),
                        path,
                        probe=probe,
                        hdr_or_10bit=hdr_or_10bit,
                        max_width=max_width,
                    )
                )
                selected.append(path)
                selected_timestamps.append(timestamp)
                next_index += 1

        dark_enhanced_count = sum(1 for future in futures if future.result())

    if selected:
        while len(selected) < target_frames:
            path = output_dir / f"frame_{len(selected):04d}.png"
            _copy_frame_image(selected[-1], path)
            selected.append(path)
            selected_timestamps.append(selected_timestamps[-1] if selected_timestamps else 0.0)

    (output_dir / "frame_timestamps.json").write_text(json.dumps(selected_timestamps), encoding="utf-8")
    return selected, {
        **accelerator_metadata,
        "qwen_video_conversion_backend": "pyav",
        "qwen_video_frame_postprocess_workers": worker_count,
        "qwen_video_frame_encoding": FRAME_ENCODING,
        "qwen_video_sdr_dark_enhanced_frame_count": dark_enhanced_count,
    }


def _frame_to_sdr_image(frame: Any, *, probe: dict[str, Any], hdr_or_10bit: bool) -> tuple[Image.Image, bool]:
    image = frame.to_image().convert("RGB")
    return _image_to_sdr_image(image, probe=probe, hdr_or_10bit=hdr_or_10bit)


def _image_to_sdr_image(image: Image.Image, *, probe: dict[str, Any], hdr_or_10bit: bool) -> tuple[Image.Image, bool]:
    if not hdr_or_10bit:
        return _maybe_enhance_dark_sdr_frame(image)

    return _enhance_sdr_proxy_for_vision(image), False


def _prepare_and_save_frame(
    image: Image.Image,
    path: Path,
    *,
    probe: dict[str, Any],
    hdr_or_10bit: bool,
    max_width: int,
) -> bool:
    try:
        prepared, dark_enhanced = _image_to_sdr_image(image, probe=probe, hdr_or_10bit=hdr_or_10bit)
        prepared = _downscale_image(prepared, max_width=max_width)
        _save_frame_image(prepared, path)
        return dark_enhanced
    finally:
        image.close()


def _save_frame_image(image: Image.Image, path: Path) -> None:
    image.save(path, format="PNG", optimize=False, compress_level=1)


def _copy_frame_image(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        _save_frame_image(image.convert("RGB"), destination)


def _enhance_sdr_proxy_for_vision(image: Image.Image) -> Image.Image:
    enhanced = ImageOps.autocontrast(image.convert("RGB"), cutoff=0.25)
    enhanced = ImageEnhance.Color(enhanced).enhance(1.08)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.18)
    return enhanced.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=3))


def _maybe_enhance_dark_sdr_frame(image: Image.Image) -> tuple[Image.Image, bool]:
    rgb = image.convert("RGB")
    luma = np.asarray(rgb.convert("L"), dtype=np.float32)
    mean = float(luma.mean())
    median = float(np.percentile(luma, 50))
    p95 = float(np.percentile(luma, 95))
    too_dark = (mean < 52.0 and p95 < 150.0) or (median < 38.0 and mean < 70.0)
    if not too_dark:
        return rgb, False

    brightness = min(1.35, max(1.10, 72.0 / max(mean, 1.0)))
    enhanced = ImageEnhance.Brightness(rgb).enhance(brightness)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.08)
    enhanced = ImageEnhance.Color(enhanced).enhance(1.03)
    return enhanced.filter(ImageFilter.UnsharpMask(radius=0.8, percent=45, threshold=4)), True


def _downscale_image(image: Image.Image, *, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    height = max(2, int(round(image.height * (max_width / image.width))))
    return image.resize((max_width, height), Image.Resampling.LANCZOS)


def _video_prepare_workers(target_frames: int) -> int:
    raw = os.getenv("QWEN_VIDEO_PREPARE_WORKERS", "").strip()
    if raw:
        try:
            requested = int(raw)
        except ValueError:
            requested = 0
        if requested > 0:
            return max(1, min(target_frames, requested))
    return max(1, min(target_frames, os.cpu_count() or 1, 8))


def _normalize_video_accelerator(value: str | None) -> str:
    raw = (value or os.getenv("GPU_BACKEND") or os.getenv("RUNTIME_PROFILE") or "").strip().lower()
    if raw in {"", "unknown"}:
        return "cpu"
    if raw == "auto":
        return "auto"
    if raw in {"cpu", "none", "off", "false", "0"}:
        return "cpu"
    if raw in {"cuda", "nvidia"}:
        return "cuda"
    if raw in {"rocm", "amd", "hip"}:
        return "rocm"
    if raw in {"macos-metal", "metal", "mps", "macos"}:
        return "macos-metal"
    return raw


def _pyav_open_kwargs(av: Any, accelerator: str) -> tuple[dict[str, Any], dict[str, Any]]:
    device_type = _hwaccel_device_type(accelerator)
    if device_type is None:
        return {}, _pyav_hwaccel_not_requested_metadata(accelerator)
    try:
        from av.codec.hwaccel import HWAccel
    except Exception as exc:
        return {}, _pyav_hwaccel_unavailable_metadata(accelerator, device_type, f"PyAV HWAccel unavailable: {exc}")
    try:
        hwaccel = HWAccel(device_type=device_type, allow_software_fallback=False)
    except Exception as exc:
        return {}, _pyav_hwaccel_unavailable_metadata(accelerator, device_type, f"PyAV HWAccel init failed: {exc}")
    return {
        "hwaccel": hwaccel,
    }, {
        "qwen_video_accelerator_status": "requested",
        "qwen_video_accelerator_api": "pyav_hwaccel",
        "qwen_video_accelerator_backend": accelerator,
        "qwen_video_accelerator_device_type": device_type,
    }


def _pyav_hwaccel_stream_metadata(stream: Any, accelerator: str, metadata: dict[str, Any]) -> dict[str, Any]:
    if metadata.get("qwen_video_accelerator_status") != "requested":
        return metadata
    is_hwaccel = bool(getattr(stream.codec_context, "is_hwaccel", False))
    if is_hwaccel:
        return {
            **metadata,
            "qwen_video_accelerator_status": "used",
            "qwen_video_pyav_codec_context_is_hwaccel": True,
        }
    return {
        **metadata,
        "qwen_video_accelerator_status": "unavailable_fallback_to_pyav",
        "qwen_video_pyav_codec_context_is_hwaccel": False,
        "qwen_video_accelerator_error": "PyAV opened the stream but codec_context.is_hwaccel is false.",
    }


def _hwaccel_device_type(accelerator: str) -> str | None:
    if accelerator == "cpu":
        return None
    if accelerator == "cuda":
        return "cuda"
    if accelerator == "macos-metal":
        return "videotoolbox"
    if accelerator == "rocm":
        return "d3d11va" if os.name == "nt" else "vaapi"
    if accelerator == "auto":
        if os.name == "nt":
            return "d3d11va"
        if sys_platform := os.getenv("PYAV_HWACCEL_DEVICE_TYPE"):
            return sys_platform.strip().lower()
        return "cuda"
    return accelerator


def _pyav_hwaccel_not_requested_metadata(accelerator: str) -> dict[str, Any]:
    return {
        "qwen_video_accelerator_status": "not_requested",
        "qwen_video_accelerator_api": "pyav_hwaccel",
        "qwen_video_accelerator_backend": accelerator,
    }


def _pyav_hwaccel_unavailable_metadata(accelerator: str, device_type: str, reason: str) -> dict[str, Any]:
    return {
        "qwen_video_accelerator_status": "unavailable_fallback_to_pyav",
        "qwen_video_accelerator_api": "pyav_hwaccel",
        "qwen_video_accelerator_backend": accelerator,
        "qwen_video_accelerator_device_type": device_type,
        "qwen_video_accelerator_error": reason[:1000],
    }


def _target_times(
    duration: float,
    target_frames: int,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> list[float]:
    duration = max(0.0, float(duration or 0.0))
    target_frames = max(1, int(target_frames or 1))
    start, end = _sampling_window(duration, start_sec=start_sec, end_sec=end_sec)
    if end <= start:
        return [round(start, 6) for _ in range(target_frames)]
    if target_frames == 1:
        return [round(start + (end - start) * 0.5, 6)]
    step = (end - start) / target_frames
    return [round(start + step * (index + 0.5), 6) for index in range(target_frames)]


def _sampling_window(
    duration: float,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> tuple[float, float]:
    duration = max(0.0, float(duration or 0.0))
    start = 0.0 if start_sec is None else max(0.0, min(duration, float(start_sec)))
    end = duration if end_sec is None else max(0.0, min(duration, float(end_sec)))
    if end <= start:
        end = duration
        start = 0.0
    return round(start, 6), round(end, 6)


def _target_time_distribution(
    duration: float,
    target_frames: int,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> dict[str, Any]:
    duration = max(0.0, float(duration or 0.0))
    window_start, window_end = _sampling_window(duration, start_sec=start_sec, end_sec=end_sec)
    times = _target_times(duration, target_frames, start_sec=window_start, end_sec=window_end)
    midpoint = duration / 2.0
    frames_per_second = _frames_per_second_distribution(duration, times)
    return {
        "qwen_video_temporal_sampling_start_sec": window_start,
        "qwen_video_temporal_sampling_end_sec": window_end,
        "qwen_video_temporal_sampling_window_duration_sec": round(max(0.0, window_end - window_start), 6),
        "qwen_video_temporal_sampling_first_half_frames": sum(1 for timestamp in times if timestamp < midpoint),
        "qwen_video_temporal_sampling_second_half_frames": sum(1 for timestamp in times if timestamp >= midpoint),
        "qwen_video_temporal_sampling_first_second_frames": sum(1 for timestamp in times if 0.0 <= timestamp < 1.0),
        "qwen_video_temporal_sampling_frames_per_second": frames_per_second,
        "qwen_video_temporal_sampling_target_times_sec": times,
    }


def _frames_per_second_distribution(duration: float, times: list[float]) -> list[dict[str, Any]]:
    seconds = max(1, int(math.ceil(max(0.0, float(duration or 0.0)))))
    output: list[dict[str, Any]] = []
    for second in range(seconds):
        start = float(second)
        end = min(float(second + 1), float(duration or second + 1))
        count = sum(1 for timestamp in times if start <= timestamp < end)
        if second == seconds - 1:
            count = sum(1 for timestamp in times if start <= timestamp <= end)
        output.append({"start_sec": round(start, 3), "end_sec": round(end, 3), "frame_count": count})
    return output


def _target_frame_count(probe: dict[str, Any], *, fps: float, max_frames: int) -> int:
    return max(1, int(max_frames))


def _stream_duration(container: Any, stream: Any) -> float | None:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration / 1_000_000)
    return None


def _is_hdr_or_10bit(probe: dict[str, Any]) -> bool:
    pix_fmt = str(probe.get("pix_fmt") or "").lower()
    transfer = str(probe.get("color_transfer") or "").lower()
    primaries = str(probe.get("color_primaries") or "").lower()
    color_space = str(probe.get("color_space") or "").lower()
    return (
        "10" in pix_fmt
        or "12" in pix_fmt
        or transfer in {"16", "smpte2084", "arib-std-b67"}
        or primaries in {"9", "bt2020"}
        or color_space in {"9", "bt2020nc", "bt2020"}
    )


def _metadata(
    probe: dict[str, Any],
    *,
    fps: float,
    max_frames: int,
    target_frames: int,
    max_width: int,
    accelerator: str,
    start_sec: float,
    end_sec: float,
) -> dict[str, Any]:
    duration = float(probe.get("duration") or 0.0)
    return {
        "qwen_video_preparation_version": PROXY_VERSION,
        "qwen_video_preparation_fps": fps,
        "qwen_video_preparation_max_frames": max_frames,
        "qwen_video_preparation_target_frames": target_frames,
        "qwen_video_preparation_max_width": max_width,
        "qwen_video_conversion_backend": "pyav",
        "qwen_video_requested_accelerator": accelerator,
        "qwen_video_accelerator_status": "not_requested" if accelerator == "cpu" else "unavailable_fallback_to_pyav",
        "qwen_video_decode_threading": "auto",
        "qwen_video_tone_mapping": TONE_MAPPING_ALGORITHM,
        "qwen_video_sdr_dark_enhancement_algorithm": SDR_DARK_ENHANCEMENT_ALGORITHM,
        "qwen_video_sdr_dark_enhancement_enabled": True,
        "qwen_video_temporal_sampling_strategy": TEMPORAL_SAMPLING_STRATEGY,
        **_target_time_distribution(duration, target_frames, start_sec=start_sec, end_sec=end_sec),
        "source_video_codec": probe.get("codec"),
        "source_video_pix_fmt": probe.get("pix_fmt"),
        "source_video_color_space": probe.get("color_space"),
        "source_video_color_transfer": probe.get("color_transfer"),
        "source_video_color_primaries": probe.get("color_primaries"),
        "source_video_width": probe.get("width"),
        "source_video_height": probe.get("height"),
        "source_video_duration": probe.get("duration"),
        "source_video_frame_count": probe.get("frame_count"),
        "source_video_hdr_or_10bit": _is_hdr_or_10bit(probe),
    }


def _cache_key(source: Path, metadata: dict[str, Any]) -> str:
    stat = source.stat()
    payload = {
        "path": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "metadata": metadata,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
