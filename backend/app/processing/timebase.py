from __future__ import annotations

from dataclasses import replace
from typing import Any


def analysis_start_sec(duration_sec: float, configured_skip_sec: float) -> float:
    duration = max(0.0, float(duration_sec or 0.0))
    skip = max(0.0, float(configured_skip_sec or 0.0))
    return round(min(skip, duration), 6)


def source_to_analysis_time(source_sec: float, offset_sec: float) -> float:
    return round(max(0.0, float(source_sec or 0.0) - max(0.0, float(offset_sec or 0.0))), 6)


def analysis_to_source_time(analysis_sec: float, offset_sec: float) -> float:
    return round(max(0.0, float(analysis_sec or 0.0) + max(0.0, float(offset_sec or 0.0))), 6)


def apply_source_timestamp_offset(item: Any, offset_sec: float) -> Any:
    """Return item with ASR/audio timestamps shifted from analysis-relative to source time."""

    offset = max(0.0, float(offset_sec or 0.0))
    if offset <= 0:
        return item
    if isinstance(item, dict):
        output = dict(item)
        for start_key, end_key in (("start", "end"), ("start_sec", "end_sec"), ("start_time", "end_time")):
            if start_key in output:
                output[start_key] = analysis_to_source_time(float(output.get(start_key) or 0.0), offset)
            if end_key in output:
                output[end_key] = analysis_to_source_time(float(output.get(end_key) or output.get(start_key) or 0.0), offset)
        return output
    if hasattr(item, "start") or hasattr(item, "end"):
        try:
            return replace(
                item,
                start=analysis_to_source_time(float(getattr(item, "start", 0.0) or 0.0), offset),
                end=analysis_to_source_time(float(getattr(item, "end", getattr(item, "start", 0.0)) or 0.0), offset),
            )
        except Exception:
            item.start = analysis_to_source_time(float(getattr(item, "start", 0.0) or 0.0), offset)
            item.end = analysis_to_source_time(float(getattr(item, "end", item.start) or item.start), offset)
            return item
    return item
