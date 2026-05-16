from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class HitMarkerEvidence:
    frame_path: str
    timestamp: float
    confidence: float
    marker_pixel_score: int
    centered_target_score: int
    region: dict[str, float]


def detect_hit_marker_evidence(
    frame_paths: Sequence[str | Path],
    *,
    sample_fps: float,
    frame_timestamps: Sequence[float] | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    active_weapon: str | None = None,
    equipment_timeline: Sequence[dict[str, Any]] | None = None,
    max_evidence: int = 3,
) -> dict[str, Any]:
    evidence: list[HitMarkerEvidence] = []
    for index, frame_path in enumerate(frame_paths):
        timestamp = _timestamp_for_frame(index, sample_fps=sample_fps, frame_timestamps=frame_timestamps)
        if start_sec is not None and timestamp < start_sec:
            continue
        if end_sec is not None and timestamp > end_sec:
            continue
        scored = _score_hit_marker_frame(Path(frame_path), timestamp=timestamp)
        if scored is not None:
            evidence.append(scored)
    evidence.sort(key=lambda item: (-item.confidence, item.timestamp))
    kept = evidence[: max(1, max_evidence)]
    detected = bool(kept and kept[0].confidence >= 0.55)
    result: dict[str, Any] = {
        "detected": detected,
        "active_weapon": active_weapon,
        "evidence": [item.__dict__ for item in kept],
        "uncertainties": [
            "Hit-marker detection is deterministic visual evidence for a probable hit marker or impact cue; it does not confirm a kill by itself."
        ],
    }
    if detected:
        best = kept[0]
        weapon_for_timestamp = _active_weapon_from_timeline(equipment_timeline, timestamp=best.timestamp) or active_weapon
        result.update(
            {
                "timestamp": best.timestamp,
                "confidence": best.confidence,
                "frame_path": best.frame_path,
                "description": _description(best, active_weapon=weapon_for_timestamp),
            }
        )
        if weapon_for_timestamp:
            result["active_weapon"] = weapon_for_timestamp
    return result


def _score_hit_marker_frame(path: Path, *, timestamp: float) -> HitMarkerEvidence | None:
    if not path.is_file():
        return None
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        return None
    array = _enhanced_array(image)
    height, width, _ = array.shape
    x1, x2 = int(width * 0.35), int(width * 0.65)
    y1, y2 = int(height * 0.25), int(height * 0.68)
    roi = array[y1:y2, x1:x2]
    marker_score = _marker_pixel_score(roi)
    target_score = _centered_target_score(roi)
    if marker_score < 120 or target_score < 500:
        return None
    confidence = min(0.99, 0.45 + marker_score / 900.0 + target_score / 50000.0)
    return HitMarkerEvidence(
        frame_path=str(path),
        timestamp=round(float(timestamp), 3),
        confidence=round(float(confidence), 3),
        marker_pixel_score=int(marker_score),
        centered_target_score=int(target_score),
        region={"x_min": 0.35, "x_max": 0.65, "y_min": 0.25, "y_max": 0.68},
    )


def _enhanced_array(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    low = float(np.percentile(array, 1.0))
    high = float(np.percentile(array, 99.7))
    if high - low > 1e-4:
        array = np.clip((array - low) / (high - low), 0.0, 1.0)
    array = np.clip(np.power(array, 0.55) * 1.18, 0.0, 1.0)
    return (array * 255.0 + 0.5).astype(np.uint8)


def _marker_pixel_score(roi: np.ndarray) -> int:
    values = roi.astype(np.int16)
    red = values[:, :, 0]
    green = values[:, :, 1]
    blue = values[:, :, 2]
    white = (red > 190) & (green > 170) & (blue > 170)
    pink = (red > 170) & (blue > 80) & ((red - green) > 55)
    return int(np.count_nonzero(white | pink))


def _centered_target_score(roi: np.ndarray) -> int:
    values = roi.astype(np.int16)
    maximum = values.max(axis=2)
    minimum = values.min(axis=2)
    mean = values.mean(axis=2)
    neutral_target = ((maximum - minimum) < 40) & (mean > 35) & (mean < 190)
    return int(np.count_nonzero(neutral_target))


def _timestamp_for_frame(
    index: int,
    *,
    sample_fps: float,
    frame_timestamps: Sequence[float] | None,
) -> float:
    if frame_timestamps is not None and index < len(frame_timestamps):
        try:
            return float(frame_timestamps[index])
        except (TypeError, ValueError):
            pass
    return index / sample_fps if sample_fps > 0 else float(index)


def _active_weapon_from_timeline(
    equipment_timeline: Sequence[dict[str, Any]] | None,
    *,
    timestamp: float,
    max_distance_sec: float = 2.5,
) -> str | None:
    if not equipment_timeline:
        return None
    candidates: list[tuple[float, float, str]] = []
    for item in equipment_timeline:
        if str(item.get("entity_type") or "").strip().lower() != "weapon":
            continue
        name = str(item.get("entity_name") or "").strip()
        if not name:
            continue
        start = _float_or_none(item.get("start_timestamp", item.get("timestamp")))
        end = _float_or_none(item.get("end_timestamp", item.get("timestamp")))
        if start is None:
            continue
        end = start if end is None else end
        if start <= timestamp <= end:
            distance = 0.0
        elif timestamp < start:
            distance = start - timestamp
        else:
            distance = timestamp - end
        if distance <= max_distance_sec:
            candidates.append((distance, -end, name))
    if not candidates:
        return None
    _, _, name = min(candidates, key=lambda item: (item[0], item[1]))
    return name


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _description(evidence: HitMarkerEvidence, *, active_weapon: str | None) -> str:
    weapon = str(active_weapon or "").strip()
    weapon_text = f" while HUD/loadout evidence indicates active weapon {weapon}" if weapon else ""
    return (
        f"Probable hit marker or impact cue detected near screen center at {evidence.timestamp:.2f}s"
        f"{weapon_text}; confidence {evidence.confidence:.2f}. This supports a hit cue, not a confirmed kill."
    )
