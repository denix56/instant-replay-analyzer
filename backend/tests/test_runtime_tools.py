from __future__ import annotations

import os
from pathlib import Path

from backend.app import runtime_tools


def test_resolve_ffmpeg_prefers_configured_binary(tmp_path: Path, monkeypatch) -> None:
    binary = _fake_ffmpeg(tmp_path / "custom_ffmpeg")
    monkeypatch.setenv("FFMPEG_BINARY", str(binary))

    assert runtime_tools.resolve_ffmpeg() == str(binary)


def test_ensure_ffmpeg_copies_configured_fallback_into_runtime_dir(tmp_path: Path, monkeypatch) -> None:
    source = _fake_ffmpeg(tmp_path / "source_ffmpeg", filters=("fps", "scale", "zscale", "tonemap"))
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv("FFMPEG_BINARY", str(source))
    monkeypatch.setattr(runtime_tools.shutil, "which", lambda _name: None)

    ensured = Path(runtime_tools.ensure_ffmpeg(runtime_dir))

    assert ensured == runtime_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    assert ensured.exists()
    assert os.access(ensured, os.X_OK)


def test_resolve_ffmpeg_skips_binaries_without_required_filters(tmp_path: Path, monkeypatch) -> None:
    missing_zscale = _fake_ffmpeg(tmp_path / "system_ffmpeg", filters=("fps", "scale", "tonemap"))
    full_static = _fake_ffmpeg(tmp_path / "static_ffmpeg", filters=("fps", "scale", "zscale", "tonemap"))
    monkeypatch.delenv("FFMPEG_BINARY", raising=False)
    monkeypatch.delenv("IMAGEIO_FFMPEG_EXE", raising=False)
    monkeypatch.setattr(runtime_tools.shutil, "which", lambda _name: str(missing_zscale))
    monkeypatch.setattr(runtime_tools, "_static_ffmpeg_candidate", lambda: str(full_static))

    resolved = runtime_tools.resolve_ffmpeg(required_filters=("zscale", "tonemap"))

    assert resolved == str(full_static)


def _fake_ffmpeg(path: Path, *, filters: tuple[str, ...] = ()) -> Path:
    filter_lines = "\n".join(f" TS {name} V->V fake filter" for name in filters)
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "if '-filters' in sys.argv:",
                f"    print({filter_lines!r})",
                "else:",
                "    print('ffmpeg fake')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path
