from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path


def resolve_ffmpeg(
    ffmpeg_bin: str = "ffmpeg",
    *,
    required_filters: tuple[str, ...] = (),
) -> str | None:
    candidates: list[str] = []
    if ffmpeg_bin in {"ffmpeg", "ffmpeg.exe"}:
        configured = os.getenv("FFMPEG_BINARY") or os.getenv("IMAGEIO_FFMPEG_EXE")
        if configured and Path(configured).is_file():
            candidates.append(configured)
    resolved = shutil.which(ffmpeg_bin)
    if resolved:
        candidates.append(resolved)
    if ffmpeg_bin in {"ffmpeg", "ffmpeg.exe"}:
        static_candidate = _static_ffmpeg_candidate()
        if static_candidate:
            candidates.append(static_candidate)
    if ffmpeg_bin not in {"ffmpeg", "ffmpeg.exe"}:
        return None
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    for candidate in dict.fromkeys(candidates):
        if _ffmpeg_supports_filters(candidate, required_filters):
            return candidate
    return None


def ensure_ffmpeg(
    runtime_dir: str | Path | None = None,
    *,
    required_filters: tuple[str, ...] = (),
) -> str:
    """Return a usable FFmpeg binary, copying the app-managed fallback into app storage if needed."""

    resolved = resolve_ffmpeg(required_filters=required_filters)
    if resolved is None:
        raise RuntimeError(
            "FFmpeg is required for audio and segment extraction. Run `uv sync` so the bundled "
            "imageio-ffmpeg fallback is available, or install FFmpeg system-wide."
        )
    if runtime_dir is None or _is_system_ffmpeg(resolved):
        _assert_ffmpeg_usable(resolved, required_filters=required_filters)
        return resolved

    runtime_path = Path(runtime_dir)
    runtime_path.mkdir(parents=True, exist_ok=True)
    target = runtime_path / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    source = Path(resolved)
    if source.resolve() != target.resolve():
        if not target.exists() or source.stat().st_size != target.stat().st_size:
            shutil.copy2(source, target)
        mode = target.stat().st_mode
        target.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    _assert_ffmpeg_usable(str(target), required_filters=required_filters)
    return str(target)


def ffmpeg_version(ffmpeg_bin: str = "ffmpeg") -> str:
    resolved = resolve_ffmpeg(ffmpeg_bin)
    if resolved is None:
        return "missing"
    try:
        completed = subprocess.run(
            [resolved, "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    text = (completed.stdout or completed.stderr).strip()
    return text.splitlines()[0] if text else "available"


def _is_system_ffmpeg(path: str) -> bool:
    system = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    return bool(system and Path(system).resolve() == Path(path).resolve())


def _assert_ffmpeg_usable(path: str, *, required_filters: tuple[str, ...] = ()) -> None:
    try:
        completed = subprocess.run(
            [path, "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"FFmpeg binary is not executable: {path}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip() or "unknown error"
        raise RuntimeError(f"FFmpeg binary failed health check: {path}: {detail}")
    if not _ffmpeg_supports_filters(path, required_filters):
        missing = ", ".join(filter_name for filter_name in required_filters if not _ffmpeg_has_filter(path, filter_name))
        raise RuntimeError(
            f"FFmpeg binary is missing required filter(s): {missing}. "
            "Install an FFmpeg build compiled with --enable-libzimg or run `uv sync` to fetch the "
            "app-managed static-ffmpeg binary."
        )


def _ffmpeg_supports_filters(path: str, required_filters: tuple[str, ...]) -> bool:
    return all(_ffmpeg_has_filter(path, filter_name) for filter_name in required_filters)


def _ffmpeg_has_filter(path: str, filter_name: str) -> bool:
    if not filter_name:
        return True
    try:
        completed = subprocess.run(
            [path, "-hide_banner", "-filters"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if completed.returncode != 0:
        return False
    return any(line.split()[1:2] == [filter_name] for line in completed.stdout.splitlines())


def _static_ffmpeg_candidate() -> str | None:
    try:
        from static_ffmpeg import run

        ffmpeg, _ffprobe = run.get_or_fetch_platform_executables_else_raise()
        return str(ffmpeg)
    except Exception:
        return None
