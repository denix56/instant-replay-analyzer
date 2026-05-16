from __future__ import annotations

from typing import Any

from .schemas import QwenVideoStage, VideoPayloadBudgetV1


DEFAULT_QWEN_BUDGETS: dict[QwenVideoStage, tuple[tuple[int, int], ...]] = {
    "full_visual": ((80, 256000), (64, 224000), (48, 196000)),
    "focus_visual": ((80, 320000), (64, 256000)),
    "ocr": ((50, 600000), (40, 480000)),
    "embedding": ((64, 224000),),
}


def fit_qwen_video_budget(
    stage: QwenVideoStage,
    *,
    settings: Any | None = None,
    max_allocated_gb: float = 8.0,
    measured_allocated_gb: float | None = None,
    fallback_index: int = 0,
) -> VideoPayloadBudgetV1:
    profiles = _profiles_for_stage(stage, settings=settings)
    index = max(0, int(fallback_index or 0))
    if measured_allocated_gb is not None and measured_allocated_gb > max_allocated_gb:
        index += 1
    if index >= len(profiles):
        raise RuntimeError(
            f"No Qwen video payload budget for stage={stage} fits under "
            f"{max_allocated_gb:.2f}GB allocated VRAM. Last measured: "
            f"{measured_allocated_gb:.3f}GB."
            if measured_allocated_gb is not None
            else f"No Qwen video payload budget for stage={stage} is configured."
        )
    frames, pixels = profiles[index]
    return VideoPayloadBudgetV1(
        stage=stage,
        max_frames=int(frames),
        max_pixels=int(pixels),
        fallback_index=index,
        max_allocated_gb=float(max_allocated_gb),
        measured_allocated_gb=measured_allocated_gb,
    )


def qwen_budget_profiles(stage: QwenVideoStage, *, settings: Any | None = None) -> list[VideoPayloadBudgetV1]:
    max_allocated = float(getattr(settings, "qwen_vram_max_allocated_gb", 8.0) if settings is not None else 8.0)
    return [
        VideoPayloadBudgetV1(
            stage=stage,
            max_frames=int(frames),
            max_pixels=int(pixels),
            fallback_index=index,
            max_allocated_gb=max_allocated,
        )
        for index, (frames, pixels) in enumerate(_profiles_for_stage(stage, settings=settings))
    ]


def _profiles_for_stage(stage: QwenVideoStage, *, settings: Any | None) -> tuple[tuple[int, int], ...]:
    defaults = DEFAULT_QWEN_BUDGETS[stage]
    if settings is None:
        return defaults
    if stage == "full_visual":
        return (
            (int(getattr(settings, "qwen_full_video_max_frames", defaults[0][0])), int(getattr(settings, "qwen_full_video_max_pixels", defaults[0][1]))),
            *defaults[1:],
        )
    if stage == "focus_visual":
        return (
            (int(getattr(settings, "qwen_focus_video_max_frames", defaults[0][0])), int(getattr(settings, "qwen_focus_video_max_pixels", defaults[0][1]))),
            *defaults[1:],
        )
    if stage == "ocr":
        return (
            (int(getattr(settings, "qwen_ocr_video_max_frames", defaults[0][0])), int(getattr(settings, "qwen_ocr_video_max_pixels", defaults[0][1]))),
            *defaults[1:],
        )
    if stage == "embedding":
        return (
            (int(getattr(settings, "qwen_video_embed_max_frames", defaults[0][0])), int(getattr(settings, "qwen_video_embed_max_pixels", defaults[0][1]))),
        )
    return defaults
