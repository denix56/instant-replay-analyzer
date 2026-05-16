from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from ..llm.schemas import OCRResult


class OCRBackend(Protocol):
    def read_text(self, image_path: str | Path) -> OCRResult:
        ...


@dataclass(frozen=True)
class OCRConfig:
    sidecar_suffix: str = ".ocr.txt"
    mock_fallback: bool = True


class OCRReader:
    def __init__(self, config: Optional[OCRConfig] = None, backend: Optional[OCRBackend] = None) -> None:
        self.config = config or OCRConfig()
        self._backend = backend

    def read_text(self, image_path: str | Path) -> OCRResult:
        if self._backend is not None:
            try:
                return self._backend.read_text(image_path)
            except Exception:
                if not self.config.mock_fallback:
                    raise
        path = Path(image_path)
        sidecar = path.with_name(path.name + self.config.sidecar_suffix)
        if sidecar.exists():
            return OCRResult(text=sidecar.read_text(encoding="utf-8").strip(), engine="sidecar")
        text = re.sub(r"[_\-]+", " ", path.stem).strip()
        return OCRResult(text=text, confidence=1.0 if text else 0.0, engine="mock")


def read_ocr(image_path: str | Path) -> OCRResult:
    return OCRReader().read_text(image_path)
