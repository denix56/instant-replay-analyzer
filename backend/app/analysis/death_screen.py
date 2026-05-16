from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


_EASYOCR_READER: Any | None = None
_EASYOCR_FAILED = False


def recognize_text(frame_path: str | Path, *, timeout_seconds: float = 20.0) -> list[OCRLine]:
    easyocr_lines = recognize_text_easyocr(frame_path)
    if easyocr_lines:
        return easyocr_lines
    return recognize_text_macos_vision(frame_path, timeout_seconds=timeout_seconds)


def recognize_text_easyocr(frame_path: str | Path) -> list[OCRLine]:
    global _EASYOCR_FAILED, _EASYOCR_READER
    if _EASYOCR_FAILED:
        return []
    try:
        from PIL import Image
    except Exception:
        return []
    try:
        if _EASYOCR_READER is None:
            _EASYOCR_READER = _build_easyocr_reader()
        image_path = str(frame_path)
        width, height = Image.open(image_path).size
        rows = _EASYOCR_READER.readtext(image_path, detail=1, paragraph=False)
    except Exception:
        _EASYOCR_FAILED = True
        return []
    output = [easyocr_result_to_line(row, width=width, height=height) for row in rows]
    return [line for line in output if line is not None]


def recognize_text_macos_vision(frame_path: str | Path, *, timeout_seconds: float = 20.0) -> list[OCRLine]:
    if platform.system().lower() != "darwin" or shutil.which("swift") is None:
        return []
    script = Path(__file__).with_name("macos_vision_ocr.swift")
    if not script.exists():
        return []
    try:
        completed = subprocess.run(
            ["swift", str(script), str(frame_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode not in {0, 2}:
        return []
    try:
        payload = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError:
        return []
    output: list[OCRLine] = []
    for item in payload if isinstance(payload, list) else []:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        box_values = item.get("box") or []
        box = tuple(float(value) for value in box_values[:4])
        if len(box) != 4:
            box = (0.0, 0.0, 0.0, 0.0)
        output.append(OCRLine(text=text, confidence=float(item.get("confidence") or 0.0), box=box))
    return output


def easyocr_result_to_line(row: Any, *, width: int, height: int) -> OCRLine | None:
    if not isinstance(row, (list, tuple)) or len(row) < 3:
        return None
    box_points, text, confidence = row[0], row[1], row[2]
    if not text:
        return None
    try:
        xs = [float(point[0]) for point in box_points]
        ys = [float(point[1]) for point in box_points]
    except Exception:
        return None
    if not xs or not ys or width <= 0 or height <= 0:
        return None
    x_min = max(0.0, min(xs))
    x_max = min(float(width), max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(float(height), max(ys))
    normalized_box = (
        x_min / width,
        1.0 - (y_max / height),
        max(0.0, x_max - x_min) / width,
        max(0.0, y_max - y_min) / height,
    )
    return OCRLine(text=str(text).strip(), confidence=float(confidence or 0.0), box=normalized_box)


def _build_easyocr_reader() -> Any:
    import easyocr  # type: ignore

    model_dir = os.getenv("EASYOCR_MODEL_DIR") or os.getenv("EASYOCR_MODULE_PATH")
    kwargs: dict[str, Any] = {
        "gpu": _easyocr_gpu_setting(),
        "verbose": False,
        "quantize": True,
    }
    if model_dir:
        kwargs["model_storage_directory"] = str(Path(model_dir) / "model")
        kwargs["user_network_directory"] = str(Path(model_dir) / "user_network")
    return easyocr.Reader(["en"], **kwargs)


def _easyocr_gpu_setting() -> bool | str:
    raw = os.getenv("EASYOCR_GPU", "").strip().lower()
    if raw in {"0", "false", "no", "off", "cpu"}:
        return False
    if raw in {"cuda", "mps", "cpu"}:
        return raw
    return True
