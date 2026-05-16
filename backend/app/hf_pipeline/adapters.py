from __future__ import annotations

import json
import os
import re
import wave
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..config import DEFAULT_QWEN_REASONING_BUDGET_TOKENS, FULL_QWEN_REASONING_BUDGET_TOKENS
from ..processing.qwen_video import (
    SUMMARY_KILL_FOCUS_END_SEC,
    SUMMARY_KILL_FOCUS_START_SEC,
    SUMMARY_KILL_FOCUS_WINDOW_ID,
    TEMPORAL_SAMPLING_STRATEGY,
)
from .model_registry import HFModelSpec, VIDEO_MAX_FRAMES_QWEN_DEFAULT, model_for_role
from ..runtime.transformers_runtime import TransformersModelManager
from .schemas import (
    ASRSegmentV1,
    ASRTranscriptV1,
    AudioCaptionV1,
    ClipManifestV1,
    EvidenceLedgerV1,
    EvidencePointerV1,
    FusedSummaryV1,
    KeyMomentV1,
    MediaWindowV1,
    MetadataPayloadV1,
    VideoPayloadBudgetV1,
    VisualEventV1,
    VideoObservationV1,
)
from .evidence_ledger import ledger_to_compact_text, validate_evidence_ledger

def _positive_int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


SUMMARY_ANSWER_MAX_TOKENS = _positive_int_from_env("SUMMARY_ANSWER_MAX_TOKENS", 2000)
SUMMARY_INTERMEDIATE_MAX_TOKENS = _positive_int_from_env(
    "SUMMARY_INTERMEDIATE_MAX_TOKENS",
    min(2048, SUMMARY_ANSWER_MAX_TOKENS),
)
SUMMARY_DETAILED_MAX_CHARS = _positive_int_from_env("SUMMARY_DETAILED_MAX_CHARS", 3200)
SUMMARY_VISUAL_OBSERVATION_MAX_ITEMS = 8
SUMMARY_VISUAL_OBSERVATION_MAX_CHARS = 900
QWEN_OCR_MAX_TOKENS = 1536
QWEN_DEATH_SCREEN_MAX_TOKENS = 1024
SUMMARY_FOCUSED_VISUAL_MAX_IMAGES = 8
SUMMARY_FOCUSED_VISUAL_VERSION = "summary-focus-crops-v1"
QWEN_OCR_VIDEO_MAX_FRAMES = _positive_int_from_env("QWEN_OCR_VIDEO_MAX_FRAMES", 50)
QWEN_OCR_VIDEO_MAX_PIXELS = _positive_int_from_env("QWEN_OCR_VIDEO_MAX_PIXELS", 600000)
QWEN_OCR_VIDEO_SAMPLING_STRATEGY = "equal_time_50_frames_high_pixels_v1"
FOCUS_WINDOW_START_SEC = SUMMARY_KILL_FOCUS_START_SEC
FOCUS_WINDOW_END_SEC = SUMMARY_KILL_FOCUS_END_SEC


class AdapterRuntimeError(RuntimeError):
    """Raised when a Hugging Face adapter cannot run with the configured runtime."""


class AudioCaptionerAdapter:
    def __init__(
        self,
        spec: HFModelSpec | None = None,
        *,
        manager: TransformersModelManager | None = None,
        mock_fallback: bool = True,
    ) -> None:
        self.spec = spec or model_for_role("audio_captioner")
        self.manager = manager
        self.mock_fallback = mock_fallback
        max_input = self.spec.max_input or {}
        self.sample_rate = int(max_input.get("sample_rate", 16000))
        self.channels = int(max_input.get("channels", 1))
        self.max_chunk_sec = float(max_input.get("max_chunk_sec", 30))
        self.window_sec = min(self.max_chunk_sec, float(max_input.get("window_sec", 5.0)))
        self.stride_sec = min(self.window_sec, float(max_input.get("stride_sec", self.window_sec / 2.0)))
        if self.window_sec <= 0:
            raise AdapterRuntimeError("Audio caption window_sec must be positive.")
        if self.stride_sec <= 0:
            raise AdapterRuntimeError("Audio caption stride_sec must be positive.")

    def caption(self, manifest: ClipManifestV1, window: MediaWindowV1) -> AudioCaptionV1 | None:
        if not window.audio_path:
            return None
        audio_path = Path(window.audio_path)
        self.validate_audio(audio_path)
        if window.duration_sec > self.max_chunk_sec:
            raise AdapterRuntimeError("Audio caption windows must be <=30 seconds.")
        if not self.mock_fallback:
            if self.manager is None:
                raise AdapterRuntimeError("MiDashengLM audio captioning requires a TransformersModelManager.")
            text = self.manager.caption_audio(
                self.spec,
                audio_path,
                prompt=_audio_caption_prompt(manifest.file_name, window.start_sec, window.end_sec),
                max_new_tokens=192,
            )
        else:
            text = (
                f"Audio caption evidence for {manifest.file_name} from {window.start_sec:.1f}s "
                f"to {window.end_sec:.1f}s."
            )
        text = text.strip()
        if not text:
            return None
        return AudioCaptionV1(
            clip_id=manifest.clip_id,
            file_name=manifest.file_name,
            window_id=window.window_id,
            start_sec=window.start_sec,
            end_sec=window.end_sec,
            model_id=self.spec.model_id,
            text=text,
            confidence=None,
            uncertainties=["MiDashengLM audio caption evidence is uncertain unless corroborated."],
            raw_payload={
                "audio_path": str(audio_path),
                "runtime": "transformers",
                "loader": self.spec.loader,
                "window_sec": self.window_sec,
                "stride_sec": self.stride_sec,
            },
        )

    def caption_windows(self, manifest: ClipManifestV1, windows: Sequence[MediaWindowV1]) -> list[AudioCaptionV1]:
        captions: list[AudioCaptionV1] = []
        for window in sorted(windows, key=lambda item: (item.start_sec, item.end_sec, item.window_id)):
            caption = self.caption(manifest, window)
            if caption is not None:
                captions.append(caption)
        return captions

    def validate_audio(self, path: Path) -> None:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            frames = handle.getnframes()
        duration = frames / sample_rate if sample_rate else 0.0
        if sample_rate != self.sample_rate:
            raise AdapterRuntimeError(f"Audio captioner expects 16 kHz audio, got {sample_rate} Hz: {path}")
        if channels != self.channels:
            raise AdapterRuntimeError(f"Audio captioner expects mono audio, got {channels} channels: {path}")
        if duration > self.max_chunk_sec + 0.001:
            raise AdapterRuntimeError(f"Audio caption chunk exceeds 30 seconds: {duration:.3f}s")


class FusionSummarizerAdapter:
    def __init__(
        self,
        spec: HFModelSpec | None = None,
        *,
        model_id: str | None = None,
        manager: TransformersModelManager | None = None,
        mock_fallback: bool = True,
        reasoning_mode: str = "off",
        reasoning_budget_tokens: int = DEFAULT_QWEN_REASONING_BUDGET_TOKENS,
        stream_callback: Callable[[dict[str, Any]], None] | None = None,
        weapon_resolver: Callable[[str], str | None] | None = None,
        weapon_skin_map: Mapping[str, str] | None = None,
    ) -> None:
        self.spec = spec or model_for_role("summarizer")
        self.model_id = model_id or self.spec.model_id
        self.manager = manager
        self.mock_fallback = mock_fallback
        self.reasoning_mode = _normalize_reasoning_mode(reasoning_mode)
        self.reasoning_budget_tokens = max(0, int(reasoning_budget_tokens))
        self.stream_callback = stream_callback
        self.weapon_resolver = weapon_resolver
        self.weapon_skin_map = dict(weapon_skin_map or {})

    def observe_video_windows(self, manifest: ClipManifestV1, windows: Sequence[MediaWindowV1]) -> list[VideoObservationV1]:
        observations: list[VideoObservationV1] = []
        for window in windows:
            prepared_frame_count = len(window.prepared_video_frame_paths)
            if not window.video_path and not prepared_frame_count:
                text = "Visual evidence is unavailable because no video segment is stored for this window."
                uncertainties = ["No direct or prepared video input available; representative frames are not used as model evidence."]
                video_input_mode = "no_video_input"
            elif not self.mock_fallback:
                text = self._complete(
                    _video_observation_messages(manifest, window, self.spec),
                    max_tokens=384,
                    empty_response_instruction="Return one detailed visual evidence paragraph only.",
                )
                if not text:
                    text = "Visual evidence is unclear in this window."
                uncertainties = []
                video_input_mode = "qwen35_prepared_video_frames" if prepared_frame_count else "qwen35_direct_video"
            else:
                text = (
                    f"Qwen3.5 video evidence from {manifest.file_name} between {window.start_sec:.1f}s and "
                    f"{window.end_sec:.1f}s. Prepared video frames: {prepared_frame_count or 'none'}."
                )
                uncertainties = ["Mock Qwen3.5 video evidence."]
                video_input_mode = "qwen35_prepared_video_frames" if prepared_frame_count else "qwen35_direct_video"
            observations.append(
                VideoObservationV1(
                    clip_id=manifest.clip_id,
                    file_name=manifest.file_name,
                    window_id=window.window_id,
                    start_sec=window.start_sec,
                    end_sec=window.end_sec,
                    model_id=self.model_id,
                    text=text,
                    confidence=None,
                    uncertainties=uncertainties,
                    raw_payload={
                        "video_path": window.video_path,
                        "audio_path": window.audio_path,
                        "video_input_mode": video_input_mode,
                        "prepared_video_frame_count": prepared_frame_count,
                        "prepared_video_metadata": window.prepared_video_metadata,
                    },
                )
            )
        return observations

    def extract_visual_ocr(
        self,
        manifest: ClipManifestV1,
        *,
        media_windows: Sequence[MediaWindowV1],
        metadata: MetadataPayloadV1,
    ) -> dict[str, Any]:
        """Use Qwen3.5 to read HUD/OCR text from the same prepared video frames sent to summary."""

        windows = [window for window in media_windows if window.prepared_video_frame_paths or window.video_path]
        if not windows:
            return {}
        if self.mock_fallback:
            return {
                "schema_version": "1.0",
                "source": "qwen35_visual_ocr",
                "model_id": self.model_id,
                "observations": [],
                "equipment_timeline": [],
                "uncertainties": ["Mock mode: Qwen3.5 visual OCR was not run."],
            }
        messages = _qwen_visual_ocr_messages(
            manifest,
            media_windows=windows,
            metadata=metadata,
            spec=self.spec,
            weapon_skin_map=self.weapon_skin_map,
        )
        raw = self._complete(
            messages,
            max_tokens=QWEN_OCR_MAX_TOKENS,
            empty_response_instruction="Return the requested strict OCR JSON only.",
        )
        try:
            return _parse_qwen_visual_ocr_json(
                raw,
                manifest=manifest,
                model_id=self.model_id,
                media_windows=windows,
                weapon_resolver=self.weapon_resolver,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            repaired = self._complete(
                _summary_repair_messages(messages, raw=raw, error=str(exc)),
                max_tokens=QWEN_OCR_MAX_TOKENS,
                empty_response_instruction="Return repaired strict OCR JSON only.",
            )
            try:
                return _parse_qwen_visual_ocr_json(
                    repaired,
                    manifest=manifest,
                    model_id=self.model_id,
                    media_windows=windows,
                    weapon_resolver=self.weapon_resolver,
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as repair_exc:
                raise AdapterRuntimeError(
                    "Qwen3.5 visual OCR returned invalid JSON after one repair attempt: "
                    f"{repair_exc}. Raw preview: {_raw_preview(repaired)}"
                ) from repair_exc

    def extract_visual_events(
        self,
        manifest: ClipManifestV1,
        *,
        media_windows: Sequence[MediaWindowV1],
        metadata: MetadataPayloadV1,
        video_payload_budgets: Sequence[VideoPayloadBudgetV1] = (),
    ) -> list[VisualEventV1]:
        """Extract conservative visual events before final text-only composition."""

        windows = [window for window in media_windows if window.prepared_video_frame_paths or window.video_path]
        if not windows:
            return []
        if self.mock_fallback:
            return [
                VisualEventV1(
                    clip_id=manifest.clip_id,
                    file_name=manifest.file_name,
                    window_id=window.window_id,
                    start_sec=window.start_sec,
                    end_sec=window.end_sec,
                    description=(
                        f"Mock Qwen3.5 visual event for {manifest.file_name} from "
                        f"{window.start_sec:.1f}s to {window.end_sec:.1f}s."
                    ),
                    evidence_pointers=[
                        EvidencePointerV1(
                            source="video",
                            window_id=window.window_id,
                            start=window.start_sec,
                            end=window.end_sec,
                            quote_or_observation="Mock visual event.",
                        )
                    ],
                    uncertainties=["Mock mode: Qwen3.5 visual event extraction was not run."],
                    raw_payload={"prepared_video_frame_count": len(window.prepared_video_frame_paths)},
                )
                for window in windows
            ]
        messages = _qwen_visual_event_messages(
            manifest,
            media_windows=windows,
            metadata=metadata,
            spec=self.spec,
            video_payload_budgets=video_payload_budgets,
            weapon_skin_map=self.weapon_skin_map,
        )
        raw = self._complete(
            messages,
            max_tokens=SUMMARY_INTERMEDIATE_MAX_TOKENS,
            empty_response_instruction="Return the requested strict visual-events JSON only.",
        )
        try:
            return _parse_qwen_visual_events_json(
                raw,
                manifest=manifest,
                model_id=self.model_id,
                media_windows=windows,
                weapon_resolver=self.weapon_resolver,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            repaired = self._complete(
                _summary_repair_messages(messages, raw=raw, error=str(exc)),
                max_tokens=SUMMARY_INTERMEDIATE_MAX_TOKENS,
                empty_response_instruction="Return repaired strict visual-events JSON only.",
            )
            try:
                return _parse_qwen_visual_events_json(
                    repaired,
                    manifest=manifest,
                    model_id=self.model_id,
                    media_windows=windows,
                    weapon_resolver=self.weapon_resolver,
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as repair_exc:
                raise AdapterRuntimeError(
                    "Qwen3.5 visual event extraction returned invalid JSON after one repair attempt: "
                    f"{repair_exc}. Raw preview: {_raw_preview(repaired)}"
                ) from repair_exc

    def extract_death_screen(
        self,
        manifest: ClipManifestV1,
        *,
        frame_candidates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Use Qwen3.5 to read death-screen frames into the existing death-screen contract."""

        candidates = _valid_death_screen_candidates(frame_candidates)
        if not candidates:
            return {"schema_version": "1.0", "source": "qwen35_death_screen", "model_id": self.model_id, "detections": []}
        if self.mock_fallback:
            return {
                "schema_version": "1.0",
                "source": "qwen35_death_screen",
                "model_id": self.model_id,
                "detections": [],
                "uncertainties": ["Mock mode: Qwen3.5 death-screen analysis was not run."],
            }
        messages = _qwen_death_screen_messages(manifest, candidates, spec=self.spec)
        raw = self._complete(
            messages,
            max_tokens=QWEN_DEATH_SCREEN_MAX_TOKENS,
            empty_response_instruction="Return the requested strict death-screen JSON only.",
        )
        try:
            return _parse_qwen_death_screen_json(raw, manifest=manifest, model_id=self.model_id, candidates=candidates)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            repaired = self._complete(
                _summary_repair_messages(messages, raw=raw, error=str(exc)),
                max_tokens=QWEN_DEATH_SCREEN_MAX_TOKENS,
                empty_response_instruction="Return repaired strict death-screen JSON only.",
            )
            try:
                return _parse_qwen_death_screen_json(
                    repaired,
                    manifest=manifest,
                    model_id=self.model_id,
                    candidates=candidates,
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as repair_exc:
                raise AdapterRuntimeError(
                    "Qwen3.5 death-screen analysis returned invalid JSON after one repair attempt: "
                    f"{repair_exc}. Raw preview: {_raw_preview(repaired)}"
                ) from repair_exc

    def summarize(
        self,
        manifest: ClipManifestV1,
        *,
        video_observations: Sequence[VideoObservationV1] | None = None,
        media_windows: Sequence[MediaWindowV1] = (),
        transcript: ASRTranscriptV1,
        audio_captions: Sequence[AudioCaptionV1],
        metadata: MetadataPayloadV1,
    ) -> FusedSummaryV1:
        observations = _with_hud_loadout_observation(
            manifest,
            video_observations or self.observe_video_windows(manifest, media_windows),
            metadata,
        )
        if not self.mock_fallback:
            messages = _summary_messages(
                manifest,
                observations=observations,
                transcript=transcript,
                audio_captions=audio_captions,
                metadata=metadata,
                weapon_resolver=self.weapon_resolver,
                weapon_skin_map=self.weapon_skin_map,
            )
            raw = self._complete(
                messages,
                max_tokens=SUMMARY_ANSWER_MAX_TOKENS,
                empty_response_instruction="Return the requested strict JSON summary only.",
            )
            try:
                summary = _parse_summary_contract(
                    raw,
                    manifest=manifest,
                    model_id=self.model_id,
                    metadata=metadata,
                    weapon_resolver=self.weapon_resolver,
                )
                return _ensure_deterministic_observation_key_moments(
                    summary,
                    observations,
                    manifest,
                    transcript=transcript,
                    audio_captions=audio_captions,
                    metadata=metadata,
                    weapon_resolver=self.weapon_resolver,
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                repaired = self._complete(
                    _summary_repair_messages(messages, raw=raw, error=str(exc)),
                    max_tokens=SUMMARY_ANSWER_MAX_TOKENS,
                    empty_response_instruction="Return repaired strict JSON only.",
                )
                try:
                    summary = _parse_summary_contract(
                        repaired,
                        manifest=manifest,
                        model_id=self.model_id,
                        metadata=metadata,
                        weapon_resolver=self.weapon_resolver,
                    )
                    return _ensure_deterministic_observation_key_moments(
                        summary,
                        observations,
                        manifest,
                        transcript=transcript,
                        audio_captions=audio_captions,
                        metadata=metadata,
                        weapon_resolver=self.weapon_resolver,
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as repair_exc:
                    raise AdapterRuntimeError(
                        "Qwen3.5 summary returned invalid JSON after one repair attempt: "
                        f"{repair_exc}. Raw preview: {_raw_preview(repaired)}"
                    ) from repair_exc
        evidence = _evidence_lines(observations, transcript.segments, audio_captions, metadata)
        title = metadata.title or manifest.file_name
        short = _compact(" ".join(evidence) or f"Clip metadata for {manifest.file_name}", max_words=28)
        detailed = _compact(" ".join(evidence) or f"No model evidence was available for {manifest.file_name}.", max_words=90)
        pointer = _first_pointer(observations, transcript.segments, audio_captions, metadata)
        sources = sorted({pointer.source})
        key_moment = KeyMomentV1(
            start=max(0.0, pointer.start),
            end=min(manifest.duration_sec, pointer.end),
            description=_compact(pointer.quote_or_observation, max_words=24),
            evidence=sources,
            evidence_pointers=[pointer],
        )
        uncertainties = []
        if audio_captions:
            uncertainties.append("Audio captions are potentially uncertain unless corroborated by video, speech, or metadata.")
        if metadata.file_name and not (observations or transcript.segments or audio_captions):
            uncertainties.append("Summary is based only on metadata.")
        summary = FusedSummaryV1(
            clip_id=manifest.clip_id,
            file_name=manifest.file_name,
            model_id=self.model_id,
            title=title,
            short_summary=short,
            detailed_summary=detailed,
            key_moments=[key_moment],
            tags=_tags_from_evidence(evidence),
            detected_language=transcript.language,
            uncertainties=uncertainties,
            raw_payload={
                "evidence_count": len(evidence),
                "fusion_mode": "mock_qwen35_video_aware_evidence_only",
                "video_window_count": len(media_windows) or len(observations),
            },
        )
        summary.validate_evidence_bounds(manifest.duration_sec)
        return summary

    def summarize_from_ledger(
        self,
        manifest: ClipManifestV1,
        *,
        ledger: EvidenceLedgerV1,
    ) -> FusedSummaryV1:
        """Compose the final summary from normalized text evidence only."""

        ledger = validate_evidence_ledger(ledger)
        if not self.mock_fallback:
            messages = _summary_from_ledger_messages(
                manifest,
                ledger=ledger,
                weapon_skin_map=self.weapon_skin_map,
            )
            raw = self._complete(
                messages,
                max_tokens=SUMMARY_ANSWER_MAX_TOKENS,
                empty_response_instruction="Return the requested strict JSON summary only.",
            )
            try:
                return _parse_summary_contract(
                    raw,
                    manifest=manifest,
                    model_id=self.model_id,
                    metadata=ledger.metadata,
                    weapon_resolver=self.weapon_resolver,
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                repaired = self._complete(
                    _summary_repair_messages(messages, raw=raw, error=str(exc)),
                    max_tokens=SUMMARY_ANSWER_MAX_TOKENS,
                    empty_response_instruction="Return repaired strict JSON only.",
                )
                try:
                    return _parse_summary_contract(
                        repaired,
                        manifest=manifest,
                        model_id=self.model_id,
                        metadata=ledger.metadata,
                        weapon_resolver=self.weapon_resolver,
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as repair_exc:
                    raise AdapterRuntimeError(
                        "Qwen3.5 text-only ledger summary returned invalid JSON after one repair attempt: "
                        f"{repair_exc}. Raw preview: {_raw_preview(repaired)}"
                    ) from repair_exc
        pointers = ledger.evidence_pointers()
        pointer = pointers[0] if pointers else EvidencePointerV1(
            source="metadata",
            window_id="metadata",
            start=0.0,
            end=0.0,
            quote_or_observation=f"file_name: {manifest.file_name}",
        )
        details = _mock_ledger_summary_text(ledger)
        summary = FusedSummaryV1(
            clip_id=manifest.clip_id,
            file_name=manifest.file_name,
            model_id=self.model_id,
            title=ledger.metadata.title or manifest.file_name,
            short_summary=_compact(details, max_words=28) or f"Evidence summary for {manifest.file_name}.",
            detailed_summary=details or f"No normalized evidence was available for {manifest.file_name}.",
            key_moments=[
                KeyMomentV1(
                    start=pointer.start,
                    end=min(manifest.duration_sec, max(pointer.start, pointer.end)),
                    description=_compact(pointer.quote_or_observation, max_words=24),
                    evidence=[pointer.source],
                    evidence_pointers=[pointer],
                )
            ],
            tags=_tags_from_evidence([details]),
            detected_language=None,
            uncertainties=list(ledger.uncertainties),
            raw_payload={"fusion_mode": "mock_qwen35_text_only_evidence_ledger", "ledger": ledger.model_dump()},
        )
        summary.validate_evidence_bounds(manifest.duration_sec)
        return summary

    def summarize_with_observations(
        self,
        manifest: ClipManifestV1,
        *,
        media_windows: Sequence[MediaWindowV1],
        transcript: ASRTranscriptV1,
        audio_captions: Sequence[AudioCaptionV1],
        metadata: MetadataPayloadV1,
    ) -> tuple[list[VideoObservationV1], FusedSummaryV1]:
        if self.mock_fallback:
            observations = self.observe_video_windows(manifest, media_windows)
            summary = self.summarize(
                manifest,
                video_observations=observations,
                media_windows=media_windows,
                transcript=transcript,
                audio_captions=audio_captions,
                metadata=metadata,
            )
            return observations, summary
        deterministic_observations = _with_hud_loadout_observation(manifest, [], metadata)
        base_windows, focus_windows = _split_summary_media_windows(media_windows)
        require_base_visual_observations = not focus_windows
        base_contract_windows = base_windows if require_base_visual_observations else []
        base_max_tokens = SUMMARY_ANSWER_MAX_TOKENS if require_base_visual_observations else SUMMARY_INTERMEDIATE_MAX_TOKENS
        messages = _summary_messages(
            manifest,
            observations=deterministic_observations,
            transcript=transcript,
            audio_captions=audio_captions,
            metadata=metadata,
            media_windows=base_windows,
            require_visual_observations=require_base_visual_observations,
            spec=self.spec,
            weapon_resolver=self.weapon_resolver,
            weapon_skin_map=self.weapon_skin_map,
        )
        raw = self._complete(
            messages,
            max_tokens=base_max_tokens,
            empty_response_instruction="Return the requested strict JSON summary only.",
        )
        try:
            summary = _parse_summary_with_video_contract(
                raw,
                manifest=manifest,
                model_id=self.model_id,
                media_windows=base_contract_windows,
                metadata=metadata,
                weapon_resolver=self.weapon_resolver,
            )
            summary = _ensure_deterministic_observation_key_moments(
                summary,
                deterministic_observations,
                manifest,
                transcript=transcript,
                audio_captions=audio_captions,
                metadata=metadata,
                weapon_resolver=self.weapon_resolver,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            repaired = self._complete(
                _summary_repair_messages(messages, raw=raw, error=str(exc)),
                max_tokens=base_max_tokens,
                empty_response_instruction="Return repaired strict JSON only.",
            )
            try:
                summary = _parse_summary_with_video_contract(
                    repaired,
                    manifest=manifest,
                    model_id=self.model_id,
                    media_windows=base_contract_windows,
                    metadata=metadata,
                    weapon_resolver=self.weapon_resolver,
                )
                summary = _ensure_deterministic_observation_key_moments(
                    summary,
                    deterministic_observations,
                    manifest,
                    transcript=transcript,
                    audio_captions=audio_captions,
                    metadata=metadata,
                    weapon_resolver=self.weapon_resolver,
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as repair_exc:
                raise AdapterRuntimeError(
                    "Qwen3.5 summary returned invalid JSON after one repair attempt: "
                    f"{repair_exc}. Raw preview: {_raw_preview(repaired)}"
                ) from repair_exc
        if focus_windows:
            summary = self._refine_summary_with_focus(
                summary,
                manifest=manifest,
                deterministic_observations=deterministic_observations,
                focus_windows=focus_windows,
                transcript=transcript,
                audio_captions=audio_captions,
                metadata=metadata,
            )
        all_windows = [*base_windows, *focus_windows]
        observations = [
            *deterministic_observations,
            *_summary_visual_observations(
                summary,
                manifest=manifest,
                media_windows=all_windows,
                model_id=self.model_id,
            ),
        ]
        return observations, summary

    def _refine_summary_with_focus(
        self,
        initial_summary: FusedSummaryV1,
        *,
        manifest: ClipManifestV1,
        deterministic_observations: Sequence[VideoObservationV1],
        focus_windows: Sequence[MediaWindowV1],
        transcript: ASRTranscriptV1,
        audio_captions: Sequence[AudioCaptionV1],
        metadata: MetadataPayloadV1,
    ) -> FusedSummaryV1:
        messages = _summary_focus_refinement_messages(
            manifest,
            initial_summary=initial_summary,
            observations=deterministic_observations,
            transcript=transcript,
            audio_captions=audio_captions,
            metadata=metadata,
            focus_windows=focus_windows,
            spec=self.spec,
            weapon_resolver=self.weapon_resolver,
            weapon_skin_map=self.weapon_skin_map,
        )
        raw = self._complete(
            messages,
            max_tokens=SUMMARY_ANSWER_MAX_TOKENS,
            empty_response_instruction="Return the refined strict JSON summary and focus visual_observations only.",
        )
        try:
            summary = _parse_summary_with_video_contract(
                raw,
                manifest=manifest,
                model_id=self.model_id,
                media_windows=focus_windows,
                metadata=metadata,
                weapon_resolver=self.weapon_resolver,
            )
            return _ensure_deterministic_observation_key_moments(
                summary,
                deterministic_observations,
                manifest,
                transcript=transcript,
                audio_captions=audio_captions,
                metadata=metadata,
                weapon_resolver=self.weapon_resolver,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            repaired = self._complete(
                _summary_repair_messages(messages, raw=raw, error=str(exc)),
                max_tokens=SUMMARY_ANSWER_MAX_TOKENS,
                empty_response_instruction="Return repaired refined strict JSON only.",
            )
            try:
                summary = _parse_summary_with_video_contract(
                    repaired,
                    manifest=manifest,
                    model_id=self.model_id,
                    media_windows=focus_windows,
                    metadata=metadata,
                    weapon_resolver=self.weapon_resolver,
                )
                return _ensure_deterministic_observation_key_moments(
                    summary,
                    deterministic_observations,
                    manifest,
                    transcript=transcript,
                    audio_captions=audio_captions,
                    metadata=metadata,
                    weapon_resolver=self.weapon_resolver,
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as repair_exc:
                raise AdapterRuntimeError(
                    "Qwen3.5 focus summary refinement returned invalid JSON after one repair attempt: "
                    f"{repair_exc}. Raw preview: {_raw_preview(repaired)}"
                ) from repair_exc

    def _complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        empty_response_instruction: str,
        stop_after_json: bool = True,
    ) -> str:
        if self.manager is None:
            raise AdapterRuntimeError("Qwen3.5 fusion requires a TransformersModelManager.")
        thinking_budget = self._reasoning_budget_for_request()
        kwargs: dict[str, Any] = {}
        if self.stream_callback is not None:
            kwargs["stream_callback"] = self.stream_callback
        raw = self.manager.generate_chat(
            self.spec,
            messages,
            temperature=0.0,
            max_new_tokens=max_tokens,
            chat_template_kwargs=self._chat_template_kwargs(),
            thinking_budget_tokens=thinking_budget if thinking_budget > 0 else None,
            stop_after_json=stop_after_json,
            **kwargs,
        )
        content = _strip_thinking(raw)
        if content:
            return content
        if raw:
            repaired = self.manager.generate_chat(
                self.spec,
                _final_answer_messages(
                    messages,
                    reasoning=raw,
                    instruction=empty_response_instruction,
                ),
                temperature=0.0,
                max_new_tokens=max_tokens,
                chat_template_kwargs={"enable_thinking": False},
                stop_after_json=stop_after_json,
                **kwargs,
            )
            return _strip_thinking(repaired)
        retry = self.manager.generate_chat(
            self.spec,
            _empty_response_messages(messages, instruction=empty_response_instruction),
            temperature=0.0,
            max_new_tokens=max_tokens,
            chat_template_kwargs={"enable_thinking": False},
            stop_after_json=stop_after_json,
            **kwargs,
        )
        return _strip_thinking(retry)

    def _chat_template_kwargs(self) -> dict[str, Any]:
        if self.reasoning_mode == "off":
            return {"enable_thinking": False}
        return {"enable_thinking": True}

    def _reasoning_budget_for_request(self) -> int:
        if self.reasoning_mode == "off":
            return 0
        if self.reasoning_mode == "full":
            return max(self.reasoning_budget_tokens, FULL_QWEN_REASONING_BUDGET_TOKENS)
        return self.reasoning_budget_tokens


def build_asr_transcript(
    manifest: ClipManifestV1,
    *,
    model_id: str,
    text: str,
    language: str | None,
    segments: Iterable[Any],
) -> ASRTranscriptV1:
    output_segments: list[ASRSegmentV1] = []
    for index, item in enumerate(segments, start=1):
        start = float(getattr(item, "start", 0.0))
        end = float(getattr(item, "end", start))
        segment_text = str(getattr(item, "text", "")).strip()
        if not segment_text:
            continue
        output_segments.append(
            ASRSegmentV1(
                clip_id=manifest.clip_id,
                file_name=manifest.file_name,
                window_id=f"speech_{index:03d}",
                start_sec=max(0.0, start),
                end_sec=max(start, end),
                model_id=model_id,
                text=segment_text,
                confidence=getattr(item, "confidence", None),
            )
        )
    return ASRTranscriptV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        model_id=model_id,
        language=language,
        text=text,
        segments=output_segments,
    )


def metadata_with_qwen_visual_ocr(metadata: MetadataPayloadV1, qwen_ocr: Mapping[str, Any] | None) -> MetadataPayloadV1:
    if not qwen_ocr or not qwen_ocr.get("observations") and not qwen_ocr.get("equipment_timeline"):
        return metadata
    user_metadata = dict(metadata.user_metadata)
    user_metadata["qwen_visual_ocr"] = dict(qwen_ocr)
    return metadata.model_copy(update={"user_metadata": user_metadata})


def _with_hud_loadout_observation(
    manifest: ClipManifestV1,
    observations: Sequence[VideoObservationV1],
    metadata: MetadataPayloadV1,
) -> list[VideoObservationV1]:
    output = list(observations)
    for observation in (
        _hud_loadout_observation(manifest, metadata),
        _qwen_visual_ocr_observation(manifest, metadata),
        _hit_marker_observation(manifest, metadata),
    ):
        if observation is None:
            continue
        if any(item.window_id == observation.window_id and item.model_id == observation.model_id for item in output):
            continue
        output.append(observation)
    return output


def _hud_loadout_observation(manifest: ClipManifestV1, metadata: MetadataPayloadV1) -> VideoObservationV1 | None:
    hud = metadata.user_metadata.get("hud")
    if not isinstance(hud, dict):
        return None
    loadout = [str(item).strip() for item in hud.get("loadout") or [] if str(item).strip()]
    active_weapon = str(hud.get("active_weapon") or "").strip()
    active_equipment = str(hud.get("active_equipment") or active_weapon or "").strip()
    active_equipment_type = str(hud.get("active_equipment_type") or "").strip()
    if not active_equipment and not loadout:
        return None

    evidence = _hud_evidence_items(hud)
    prepared_evidence = _hud_prepared_evidence_items(hud)
    all_evidence = [*evidence, *prepared_evidence]
    equipment_timeline = _hud_equipment_timeline(hud, all_evidence)
    confidence = _hud_best_confidence(evidence, active_equipment or active_weapon)
    if confidence is None:
        confidence = _hud_best_confidence(prepared_evidence, active_equipment or active_weapon)
    timestamp = _hud_evidence_timestamp(evidence)
    if timestamp is None:
        timestamp = _hud_evidence_timestamp(prepared_evidence)
    if timestamp is None:
        timestamp = 0.0
    timestamp = _clip_timestamp(max(0.0, float(timestamp)), manifest.duration_sec)
    start = max(0.0, timestamp - 0.25)
    end = min(manifest.duration_sec, timestamp + 0.25)
    if end <= start:
        end = min(manifest.duration_sec, start + 0.001)

    parts: list[str] = []
    if active_equipment:
        if active_equipment_type == "weapon" or active_weapon:
            parts.append(f"HUD/loadout extraction identifies active weapon: {active_equipment}")
        else:
            parts.append(f"HUD/loadout extraction identifies active {active_equipment_type or 'item'}: {active_equipment}")
    if loadout:
        parts.append("visible loadout: " + ", ".join(loadout))
    timeline_text = _equipment_timeline_text(equipment_timeline)
    if timeline_text:
        parts.append("timestamped current equipment: " + timeline_text)
    if confidence is not None:
        parts.append(f"confidence: {confidence:.2f}")
    text = "; ".join(parts).strip()
    if not text:
        return None
    if not text.endswith("."):
        text += "."

    uncertainties: list[str] = []
    if confidence is None:
        uncertainties.append("HUD/loadout extraction has no confidence score; treat equipment evidence as uncertain.")
    elif confidence < 0.70:
        uncertainties.append(f"HUD/loadout extraction confidence is {confidence:.2f}; mark equipment claim uncertain.")
    return VideoObservationV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="hud_loadout_detection",
        start_sec=round(start, 3),
        end_sec=round(end, 3),
        model_id="hud_loadout_detector",
        text=text,
        confidence=confidence,
        uncertainties=uncertainties,
        raw_payload={
            "extraction": "prepared_frame_hud_loadout",
            "hud_loadout": hud,
            "evidence": evidence,
            "prepared_frame_evidence": prepared_evidence,
            "equipment_timeline": equipment_timeline,
        },
    )


def _qwen_visual_ocr_observation(manifest: ClipManifestV1, metadata: MetadataPayloadV1) -> VideoObservationV1 | None:
    qwen_ocr = metadata.user_metadata.get("qwen_visual_ocr")
    if not isinstance(qwen_ocr, dict):
        return None
    observations = [item for item in qwen_ocr.get("observations") or [] if isinstance(item, dict)]
    timeline = [item for item in qwen_ocr.get("equipment_timeline") or [] if isinstance(item, dict)]
    if not observations and not timeline:
        return None
    start_values = [_float_or_none(item.get("start", item.get("timestamp"))) for item in [*observations, *timeline]]
    end_values = [_float_or_none(item.get("end", item.get("end_timestamp", item.get("timestamp")))) for item in [*observations, *timeline]]
    start_candidates = [value for value in start_values if value is not None]
    end_candidates = [value for value in end_values if value is not None]
    start = _clip_timestamp(min(start_candidates) if start_candidates else 0.0, manifest.duration_sec)
    end = _clip_timestamp(max(end_candidates) if end_candidates else start, manifest.duration_sec)
    if end <= start:
        end = min(manifest.duration_sec, start + 0.001)
    parts: list[str] = []
    for item in observations[:8]:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        item_start = _float_or_none(item.get("start"))
        item_end = _float_or_none(item.get("end"))
        if item_start is not None and item_end is not None and abs(item_end - item_start) > 0.05:
            time_text = f"{item_start:.2f}-{item_end:.2f}s"
        elif item_start is not None:
            time_text = f"{item_start:.2f}s"
        else:
            time_text = "unknown time"
        parts.append(f"{time_text}: {text}")
    timeline_text = _equipment_timeline_text(timeline, max_items=8)
    if timeline_text:
        parts.append("Qwen3.5 timestamped current equipment: " + timeline_text)
    if not parts:
        return None
    text = "Qwen3.5 visual OCR/HUD evidence: " + "; ".join(parts)
    if not text.endswith("."):
        text += "."
    uncertainties = [
        str(value)
        for value in qwen_ocr.get("uncertainties", [])
        if str(value).strip()
    ] if isinstance(qwen_ocr.get("uncertainties"), list) else []
    return VideoObservationV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="qwen35_visual_ocr",
        start_sec=round(float(start), 3),
        end_sec=round(float(end), 3),
        model_id=str(qwen_ocr.get("model_id") or "qwen35_visual_ocr"),
        text=text,
        confidence=None,
        uncertainties=uncertainties,
        raw_payload={
            "extraction": "qwen35_visual_ocr",
            "qwen_visual_ocr": qwen_ocr,
            "equipment_timeline": timeline,
        },
    )


def _hud_evidence_items(hud: dict[str, Any]) -> list[dict[str, Any]]:
    raw = hud.get("evidence")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _hud_prepared_evidence_items(hud: dict[str, Any]) -> list[dict[str, Any]]:
    raw = hud.get("prepared_frame_evidence")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _hud_equipment_timeline(hud: dict[str, Any], evidence: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = hud.get("equipment_timeline")
    if isinstance(raw, list):
        items = [item for item in raw if isinstance(item, dict)]
        if items:
            return items
    timeline: list[dict[str, Any]] = []
    for item in sorted(evidence, key=lambda row: float(row.get("timestamp") or 0.0)):
        if not item.get("is_active"):
            continue
        name = str(item.get("entity_name") or "").strip()
        if not name:
            continue
        timestamp = _float_or_none(item.get("timestamp"))
        timeline.append(
            {
                "timestamp": timestamp,
                "entity_name": name,
                "entity_type": item.get("entity_type"),
                "confidence": _float_or_none(item.get("confidence")),
                "frame_path": item.get("frame_path"),
                "frame_index": item.get("frame_index"),
            }
        )
    return _compact_equipment_timeline(timeline)


def _compact_equipment_timeline(items: Sequence[dict[str, Any]], *, max_items: int = 24) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    last_key: tuple[str, str] | None = None
    for item in sorted(items, key=lambda row: float(row.get("timestamp") or 0.0)):
        name = str(item.get("entity_name") or "").strip()
        entity_type = str(item.get("entity_type") or "").strip()
        if not name:
            continue
        key = (name.lower(), entity_type.lower())
        if key == last_key and output:
            output[-1]["end_timestamp"] = item.get("timestamp")
            output[-1]["end_frame_index"] = item.get("frame_index")
            output[-1]["sample_count"] = int(output[-1].get("sample_count") or 1) + 1
            if item.get("confidence") is not None:
                output[-1]["confidence"] = max(float(output[-1].get("confidence") or 0.0), float(item["confidence"]))
            continue
        row = dict(item)
        row.setdefault("start_timestamp", row.get("timestamp"))
        row.setdefault("end_timestamp", row.get("timestamp"))
        row.setdefault("sample_count", 1)
        output.append(row)
        last_key = key
    if len(output) <= max_items:
        return output
    return [*output[: max_items // 2], *output[-(max_items - max_items // 2) :]]


def _equipment_timeline_text(items: Sequence[dict[str, Any]], *, max_items: int = 12) -> str:
    parts: list[str] = []
    for item in list(items)[:max_items]:
        name = str(item.get("entity_name") or "").strip()
        if not name:
            continue
        entity_type = str(item.get("entity_type") or "item").strip() or "item"
        start = _float_or_none(item.get("start_timestamp", item.get("timestamp")))
        end = _float_or_none(item.get("end_timestamp", item.get("timestamp")))
        if start is None:
            time_text = "unknown time"
        elif end is not None and abs(end - start) > 0.05:
            time_text = f"{start:.2f}-{end:.2f}s"
        else:
            time_text = f"{start:.2f}s"
        confidence = _float_or_none(item.get("confidence"))
        confidence_text = f", confidence {confidence:.2f}" if confidence is not None else ""
        parts.append(f"{time_text}: {name} ({entity_type}{confidence_text})")
    return "; ".join(parts)


def _weapon_resolution_evidence(
    metadata: MetadataPayloadV1,
    observations: Sequence[VideoObservationV1],
    weapon_resolver: Callable[[str], str | None] | None,
) -> list[dict[str, Any]]:
    if weapon_resolver is None:
        return []
    candidates: dict[tuple[str, str], dict[str, Any]] = {}

    def add(raw_name: str, *, timestamp: Any = None, source: str = "metadata", frame_path: Any = None) -> None:
        cleaned = raw_name.strip()
        if not cleaned:
            return
        resolved = weapon_resolver(cleaned)
        if not resolved:
            return
        key = (cleaned.lower(), resolved.lower())
        item = candidates.setdefault(
            key,
            {
                "raw_name": cleaned,
                "display_name": resolved,
                "source": source,
                "allowed_use": "weapon/equipment name only",
                "forbidden_use": "do not use as teammate, enemy, hunter, player, or person identity",
                "timestamps": [],
            },
        )
        time_value = _float_or_none(timestamp)
        if time_value is not None:
            time_row = {"timestamp": round(time_value, 3), "source": source}
            if frame_path:
                time_row["frame_path"] = str(frame_path)
            if time_row not in item["timestamps"]:
                item["timestamps"].append(time_row)

    hud = metadata.user_metadata.get("hud")
    if isinstance(hud, dict):
        for key in ("active_weapon", "active_equipment"):
            add(str(hud.get(key) or ""), source=f"hud.{key}")
        for name in hud.get("loadout") or []:
            add(str(name), source="hud.loadout")
        for field in ("evidence", "prepared_frame_evidence", "equipment_timeline"):
            raw_items = hud.get(field)
            if not isinstance(raw_items, list):
                continue
            for row in raw_items:
                if not isinstance(row, dict):
                    continue
                add(
                    str(row.get("entity_name") or ""),
                    timestamp=row.get("timestamp", row.get("start_timestamp")),
                    source=f"hud.{field}",
                    frame_path=row.get("frame_path"),
                )

    qwen_ocr = metadata.user_metadata.get("qwen_visual_ocr")
    if isinstance(qwen_ocr, dict):
        for row in qwen_ocr.get("observations") or []:
            if not isinstance(row, dict):
                continue
            timestamp = row.get("start")
            for equipment in row.get("resolved_equipment") or []:
                if isinstance(equipment, dict):
                    add(str(equipment.get("raw_name") or equipment.get("display_name") or ""), timestamp=timestamp, source="qwen_visual_ocr")
            add(str(row.get("raw_text") or row.get("text") or ""), timestamp=timestamp, source="qwen_visual_ocr")
        for row in qwen_ocr.get("equipment_timeline") or []:
            if isinstance(row, dict):
                add(
                    str(row.get("entity_name") or ""),
                    timestamp=row.get("timestamp", row.get("start_timestamp")),
                    source="qwen_visual_ocr.equipment_timeline",
                )

    for observation in observations:
        if observation.window_id not in {"hud_loadout_detection", "qwen35_visual_ocr"}:
            continue
        add(observation.text, timestamp=observation.start_sec, source="video.hud_observation")

    output = sorted(
        candidates.values(),
        key=lambda item: (str(item["display_name"]).lower(), str(item["raw_name"]).lower()),
    )
    for item in output:
        item["timestamps"] = sorted(item["timestamps"], key=lambda row: float(row["timestamp"]))
    return output[:16]


def _authoritative_equipment_timeline(
    metadata: MetadataPayloadV1,
    *,
    weapon_resolver: Callable[[str], str | None] | None,
) -> list[dict[str, Any]]:
    """Return the compact timestamped local-player equipment timeline used by summary."""

    rows: list[dict[str, Any]] = []

    def add_rows(source: Mapping[str, Any], *, source_name: str) -> None:
        raw_items = source.get("equipment_timeline")
        if not isinstance(raw_items, list):
            return
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            raw_name = str(item.get("entity_name") or "").strip()
            if not raw_name:
                continue
            timestamp = _float_or_none(item.get("timestamp", item.get("start_timestamp")))
            start = _float_or_none(item.get("start_timestamp", item.get("timestamp")))
            end = _float_or_none(item.get("end_timestamp", item.get("timestamp")))
            if timestamp is None:
                timestamp = start
            entity_type = str(item.get("entity_type") or "item").strip() or "item"
            resolved = _canonical_hunt_weapon_from_text(raw_name, weapon_resolver=weapon_resolver)
            display_name = (resolved.removeprefix("the ") if resolved else raw_name).strip()
            rows.append(
                {
                    "timestamp": timestamp,
                    "start_timestamp": start,
                    "end_timestamp": end,
                    "raw_name": raw_name,
                    "display_name": display_name,
                    "entity_type": entity_type,
                    "source": source_name,
                    "confidence": _float_or_none(item.get("confidence")),
                }
            )

    hud = metadata.user_metadata.get("hud")
    if isinstance(hud, dict):
        add_rows(hud, source_name="hud.equipment_timeline")
    qwen_ocr = metadata.user_metadata.get("qwen_visual_ocr")
    if isinstance(qwen_ocr, dict):
        add_rows(qwen_ocr, source_name="qwen_visual_ocr.equipment_timeline")

    compacted: list[dict[str, Any]] = []
    seen: set[tuple[float | None, str, str, str]] = set()
    for row in sorted(rows, key=lambda item: float(item.get("timestamp") or item.get("start_timestamp") or 0.0)):
        key = (
            _float_or_none(row.get("timestamp")),
            str(row.get("display_name") or "").lower(),
            str(row.get("entity_type") or "").lower(),
            str(row.get("source") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        compacted.append(row)
    return compacted[:16]


def _weapon_name_resolution_rules(
    weapon_skin_map: Mapping[str, str] | None,
    *,
    max_rules: int = 96,
) -> list[dict[str, str]]:
    if not weapon_skin_map:
        return []
    rules: list[dict[str, str]] = []
    # Keep the prompt bounded: short visual/OCR skin labels are the highest-risk cases for
    # Qwen writing a skin name as if it were the weapon name. The full map remains available
    # to deterministic postprocessing through weapon_resolver.
    for raw_name, display_name in sorted(weapon_skin_map.items(), key=lambda item: (len(item[0]), item[0].lower())):
        raw = str(raw_name).strip()
        display = str(display_name).strip()
        if not raw or not display:
            continue
        rules.append(
            {
                "raw_name": raw,
                "display_name": display,
                "allowed_use": "weapon/equipment name normalization only",
                "forbidden_use": "do not use raw_name as a teammate, enemy, hunter, player, or person identity",
            }
        )
        if len(rules) >= max_rules:
            break
    return rules


def _hit_marker_observation(manifest: ClipManifestV1, metadata: MetadataPayloadV1) -> VideoObservationV1 | None:
    hit_marker = metadata.user_metadata.get("hit_marker")
    if not isinstance(hit_marker, dict) or not hit_marker.get("detected"):
        return None
    evidence = hit_marker.get("evidence")
    evidence_items = [item for item in evidence if isinstance(item, dict)] if isinstance(evidence, list) else []
    timestamp = _clip_timestamp(float(hit_marker.get("timestamp") or 0.0), manifest.duration_sec)
    start = max(0.0, timestamp - 0.25)
    end = min(manifest.duration_sec, timestamp + 0.25)
    if end <= start:
        end = min(manifest.duration_sec, start + 0.001)
    confidence = hit_marker.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None
    text = str(hit_marker.get("description") or "").strip()
    if not text:
        text = f"Hit-marker detector identifies a probable hit marker or impact cue near screen center at {timestamp:.2f}s."
    if not text.endswith("."):
        text += "."
    uncertainties = [str(item) for item in hit_marker.get("uncertainties", []) if str(item).strip()]
    if not uncertainties:
        uncertainties.append("Hit-marker evidence supports a probable hit cue, not a confirmed kill by itself.")
    return VideoObservationV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        window_id="hit_marker_detection",
        start_sec=round(start, 3),
        end_sec=round(end, 3),
        model_id="hit_marker_detector",
        text=text,
        confidence=confidence_value,
        uncertainties=uncertainties,
        raw_payload={
            "extraction": "center_screen_hit_marker_detection",
            "hit_marker": hit_marker,
            "evidence": evidence_items,
        },
    )


def _hud_best_confidence(evidence: Sequence[dict[str, Any]], active_name: str) -> float | None:
    if not evidence:
        return None
    active_key = active_name.strip().lower()
    candidates = [
        item
        for item in evidence
        if item.get("confidence") is not None
        and (
            bool(item.get("is_active"))
            or not active_key
            or str(item.get("entity_name") or "").strip().lower() == active_key
        )
    ]
    if not candidates:
        candidates = [item for item in evidence if item.get("confidence") is not None]
    confidences: list[float] = []
    for item in candidates:
        try:
            confidences.append(float(item["confidence"]))
        except (TypeError, ValueError):
            continue
    return max(confidences) if confidences else None


def _hud_evidence_timestamp(evidence: Sequence[dict[str, Any]]) -> float | None:
    selected = next((item for item in evidence if item.get("is_active") and item.get("timestamp") is not None), None)
    if selected is None:
        selected = next((item for item in evidence if item.get("timestamp") is not None), None)
    if selected is None:
        return None
    try:
        return float(selected["timestamp"])
    except (TypeError, ValueError):
        return None


def _evidence_lines(
    observations: Sequence[VideoObservationV1],
    speech: Sequence[ASRSegmentV1],
    captions: Sequence[AudioCaptionV1],
    metadata: MetadataPayloadV1,
) -> list[str]:
    lines: list[str] = []
    lines.extend(f"video: {item.text}" for item in observations)
    lines.extend(f"speech: {item.text}" for item in speech)
    lines.extend(f"audio: {item.text}" for item in captions)
    meta_parts = [metadata.file_name, metadata.title or "", metadata.description or "", " ".join(metadata.tags)]
    metadata_text = " ".join(part for part in meta_parts if part).strip()
    if metadata_text:
        lines.append(f"metadata: {metadata_text}")
    return lines


def _video_observation_messages(
    manifest: ClipManifestV1,
    window: MediaWindowV1,
    spec: HFModelSpec | None = None,
) -> list[dict[str, Any]]:
    max_input = spec.max_input if spec is not None and spec.max_input else {}
    video_fps = float(max_input.get("video_fps", 6.0))
    video_max_frames = int(max_input.get("video_max_frames", VIDEO_MAX_FRAMES_QWEN_DEFAULT))
    video_min_frames = int(max_input.get("video_min_frames", min(4, video_max_frames)))
    video_max_pixels = max_input.get("video_max_pixels", max_input.get("max_pixels"))
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Observe this gameplay media window for visual evidence only. "
                "Treat the file name as an identifier, not visual evidence. "
                "Use only visible content, including OCR/scoreboard text if clearly visible. "
                "Do not invent names, teams, scores, places, brands, calls, weapons, or events. "
                "Do not infer intent or outcome unless it is directly visible. "
                "Describe the clip chronologically and spatially. Cover the player/hunter actions, "
                "camera or hunter position, visible surroundings and room/layout, visible enemy positions, "
                "visible teammate positions, HUD state, weapon/equipment only if visible or extracted, "
                "and how the scene changes across the window. "
                "Do not identify hunter or teammate skin/cosmetic names from appearance, clothing, silhouettes, or name tags; "
                "describe people by position and appearance only. "
                "If enemies, teammates, location names, or exact weapons are not visible, say they are not visible "
                "instead of guessing. "
                f"File: {manifest.file_name}. Window: {window.start_sec:.3f}-{window.end_sec:.3f}s. "
                "Return one plain-text paragraph, 80-180 words when evidence allows. If evidence is unclear, say so explicitly. "
                "No markdown and no JSON."
            ),
        }
    ]
    if window.prepared_video_frame_paths:
        content.append(_window_video_payload(window, spec))
    elif window.video_path:
        content.append(_window_video_payload(window, spec))
    return [
        {
            "role": "system",
            "content": "You produce conservative visual evidence for video retrieval and summary fusion.",
        },
        {"role": "user", "content": content},
    ]


def _window_video_payload(window: MediaWindowV1, spec: HFModelSpec | None = None) -> dict[str, Any]:
    max_input = spec.max_input if spec is not None and spec.max_input else {}
    video_fps = float(max_input.get("video_fps", 6.0))
    video_max_frames = int(max_input.get("video_max_frames", VIDEO_MAX_FRAMES_QWEN_DEFAULT))
    video_min_frames = int(max_input.get("video_min_frames", min(4, video_max_frames)))
    video_max_pixels = max_input.get("video_max_pixels", max_input.get("max_pixels"))
    if window.prepared_video_frame_paths:
        video_payload: dict[str, Any] = {
            "type": "video",
            "video": window.prepared_video_frame_paths,
            "sample_fps": float(window.prepared_video_sample_fps or video_fps),
            "raw_fps": float(window.prepared_video_sample_fps or video_fps),
            "max_frames": min(video_max_frames, len(window.prepared_video_frame_paths)),
        }
    else:
        video_payload = {
            "type": "video",
            "video": window.video_path,
            "fps": video_fps,
            "min_frames": video_min_frames,
            "max_frames": video_max_frames,
        }
    if video_max_pixels is not None:
        video_payload["max_pixels"] = int(video_max_pixels)
    return video_payload


def _window_visual_ocr_video_payload(window: MediaWindowV1, spec: HFModelSpec | None = None) -> dict[str, Any]:
    max_pixels = _qwen_ocr_video_max_pixels(spec)
    if window.prepared_video_frame_paths:
        timestamps = _prepared_frame_timestamps(window)
        indices = _visual_ocr_frame_indices(window, timestamps=timestamps)
        frame_paths = [window.prepared_video_frame_paths[index] for index in indices]
        selected_timestamps = [timestamps[index] for index in indices if index < len(timestamps)]
        sample_fps = _sample_fps_for_timestamps(
            selected_timestamps,
            default=float(window.prepared_video_sample_fps or 3.0),
        )
        return {
            "type": "video",
            "video": frame_paths,
            "sample_fps": sample_fps,
            "raw_fps": sample_fps,
            "max_frames": len(frame_paths),
            "max_pixels": max_pixels,
        }
    payload = _window_video_payload(window, spec)
    payload["fps"] = min(float(payload.get("fps", 3.0)), 3.0)
    payload["max_frames"] = QWEN_OCR_VIDEO_MAX_FRAMES
    payload["max_pixels"] = max_pixels
    return payload


def _visual_ocr_frame_indices(window: MediaWindowV1, *, timestamps: Sequence[float] | None = None) -> list[int]:
    frame_count = len(window.prepared_video_frame_paths)
    if frame_count <= QWEN_OCR_VIDEO_MAX_FRAMES:
        return list(range(frame_count))
    resolved_timestamps = list(timestamps) if timestamps is not None else _prepared_frame_timestamps(window)
    if len(resolved_timestamps) != frame_count:
        return _evenly_spaced_indices(frame_count, QWEN_OCR_VIDEO_MAX_FRAMES)
    return _equally_sample_frame_indices_by_time(resolved_timestamps, QWEN_OCR_VIDEO_MAX_FRAMES)


def _equally_sample_frame_indices_by_time(timestamps: Sequence[float], max_frames: int) -> list[int]:
    if not timestamps or max_frames <= 0:
        return []
    if len(timestamps) <= max_frames:
        return list(range(len(timestamps)))
    start = float(min(timestamps))
    end = float(max(timestamps))
    if end <= start:
        return _evenly_spaced_indices(len(timestamps), max_frames)
    target_times = [start + (end - start) * index / (max_frames - 1) for index in range(max_frames)]
    indices: list[int] = []
    for target in target_times:
        nearest = min(range(len(timestamps)), key=lambda index: abs(float(timestamps[index]) - target))
        if nearest not in indices:
            indices.append(nearest)
    if len(indices) < max_frames:
        for index in _evenly_spaced_indices(len(timestamps), max_frames):
            if index not in indices:
                indices.append(index)
            if len(indices) >= max_frames:
                break
    return sorted(indices[:max_frames])


def _evenly_spaced_indices(length: int, max_items: int) -> list[int]:
    if length <= 0 or max_items <= 0:
        return []
    if length <= max_items:
        return list(range(length))
    if max_items == 1:
        return [0]
    return sorted({round(index * (length - 1) / (max_items - 1)) for index in range(max_items)})


def _sample_fps_for_timestamps(timestamps: Sequence[float], *, default: float) -> float:
    if len(timestamps) < 2:
        return max(default, 0.1)
    duration = max(float(max(timestamps)) - float(min(timestamps)), 1e-3)
    return max(min((len(timestamps) - 1) / duration, default), 0.1)


def _qwen_ocr_video_max_pixels(spec: HFModelSpec | None) -> int:
    max_input = spec.max_input if spec is not None and spec.max_input else {}
    configured = max_input.get("ocr_video_max_pixels")
    if configured is not None:
        return int(configured)
    base = int(max_input.get("video_max_pixels", max_input.get("max_pixels", QWEN_OCR_VIDEO_MAX_PIXELS)))
    return max(QWEN_OCR_VIDEO_MAX_PIXELS, base)


def _valid_death_screen_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, item in enumerate(candidates, start=1):
        frame_path = str(item.get("frame_path") or "").strip()
        if not frame_path:
            continue
        path = Path(frame_path).expanduser()
        if not path.exists():
            continue
        frame_id = str(item.get("frame_id") or f"death_candidate_{index:03d}")
        timestamp = _float_or_none(item.get("timestamp")) or 0.0
        start = _float_or_none(item.get("start")) or timestamp
        end = _float_or_none(item.get("end")) or timestamp
        output.append(
            {
                "frame_id": frame_id,
                "segment_id": item.get("segment_id"),
                "frame_path": str(path),
                "timestamp": timestamp,
                "start": start,
                "end": end,
            }
        )
    return output[:6]


def _qwen_death_screen_messages(
    manifest: ClipManifestV1,
    candidates: Sequence[Mapping[str, Any]],
    *,
    spec: HFModelSpec | None = None,
) -> list[dict[str, Any]]:
    max_input = spec.max_input if spec is not None and spec.max_input else {}
    image_max_pixels = int(max_input.get("image_max_pixels", max_input.get("max_pixels", 786432)))
    payload = {
        "clip": {
            "clip_id": manifest.clip_id,
            "file_name": manifest.file_name,
            "duration_sec": manifest.duration_sec,
        },
        "candidate_frames": [
            {
                "frame_id": item["frame_id"],
                "segment_id": item.get("segment_id"),
                "timestamp": item.get("timestamp"),
                "start": item.get("start"),
                "end": item.get("end"),
            }
            for item in candidates
        ],
    }
    user_text = (
        "Analyze only whether the attached candidate images are Hunt: Showdown death/down screens. "
        "Read visible death-screen text and return structured JSON. "
        "Only extract status, killed_with weapon, killer_name, and raw visible text from death-screen UI. "
        "Do not infer hunter skin names, weapons, people, places, teams, causes, or events from appearance; "
        "only use readable death-screen text. If an image is normal gameplay or unreadable, mark is_death_screen false. "
        "Allowed status values are \"downed\", \"dead\", \"unknown\", or null. "
        "Return exactly this JSON shape: {"
        "\"detections\": [{\"frame_id\": string, \"is_death_screen\": boolean, "
        "\"status\": \"downed\"|\"dead\"|\"unknown\"|null, "
        "\"killed_with\": string|null, \"killer_name\": string|null, "
        "\"raw_visible_text\": string, \"confidence\": number, \"uncertainties\": [string]}], "
        "\"uncertainties\": [string]}. "
        "Use one detection per candidate frame. Return valid JSON only, with no markdown and no prose outside JSON. "
        "Evidence payload: "
        + json.dumps(payload, ensure_ascii=True)
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for item in candidates:
        content.append(
            {
                "type": "text",
                "text": (
                    f"Candidate frame_id={item['frame_id']} "
                    f"timestamp={float(item.get('timestamp') or 0.0):.3f}s."
                ),
            }
        )
        content.append(
            {
                "type": "image",
                "image": item["frame_path"],
                "max_pixels": image_max_pixels,
            }
        )
    return [
        {
            "role": "system",
            "content": (
                "You extract death-screen UI text into strict JSON. Use only readable text in the supplied image. "
                "Do not guess. Return JSON only."
            ),
        },
        {"role": "user", "content": content},
    ]


def _parse_qwen_death_screen_json(
    raw: str,
    *,
    manifest: ClipManifestV1,
    model_id: str,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    data = _loads_json_object(raw)
    candidates_by_id = {str(item["frame_id"]): item for item in candidates}
    detections: list[dict[str, Any]] = []
    raw_items = data.get("detections")
    if not isinstance(raw_items, list):
        raise TypeError("death-screen JSON must contain detections list")
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        frame_id = str(item.get("frame_id") or "").strip()
        candidate = candidates_by_id.get(frame_id)
        if candidate is None:
            continue
        is_death_screen = _bool_or_false(item.get("is_death_screen"))
        status = _normalized_death_status(item.get("status"))
        killed_with = _clean_death_field(item.get("killed_with"))
        killer_name = _clean_death_field(item.get("killer_name"))
        raw_text = _clean_death_field(item.get("raw_visible_text"), max_chars=600) or ""
        if not is_death_screen and not status and not killed_with and not killer_name:
            continue
        timestamp = _clip_timestamp(float(candidate.get("timestamp") or 0.0), manifest.duration_sec)
        confidence = _float_or_none(item.get("confidence"))
        detections.append(
            {
                "schema_version": "1.0",
                "frame_id": frame_id,
                "segment_id": candidate.get("segment_id"),
                "frame_path": candidate["frame_path"],
                "timestamp": round(float(timestamp), 3),
                "status": status or "unknown",
                "killed_with": killed_with,
                "killer_name": killer_name,
                "raw_text": raw_text,
                "confidence": max(0.0, min(1.0, confidence if confidence is not None else 0.0)),
                "source": "qwen35_death_screen",
                "model_id": model_id,
                "uncertainties": [
                    str(value)
                    for value in item.get("uncertainties", [])
                    if str(value).strip()
                ] if isinstance(item.get("uncertainties"), list) else [],
            }
        )
    uncertainties = [
        str(value)
        for value in data.get("uncertainties", [])
        if str(value).strip()
    ] if isinstance(data.get("uncertainties"), list) else []
    return {
        "schema_version": "1.0",
        "source": "qwen35_death_screen",
        "model_id": model_id,
        "detections": detections[:6],
        "uncertainties": uncertainties,
        "raw_response": data,
    }


def _normalized_death_status(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("'", "")
    if not text or text in {"none", "null"}:
        return None
    if "dead" in text:
        return "dead"
    if "down" in text:
        return "downed"
    if text == "unknown":
        return "unknown"
    return None


def _bool_or_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def _clean_death_field(value: Any, *, max_chars: int = 160) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        text = "\n".join(str(item) for item in value if str(item).strip())
    else:
        text = str(value)
    cleaned = re.sub(r"[ \t]+", " ", text).strip(" \n\r\t.:-")
    if not cleaned:
        return None
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 3].rstrip(" ,.;") + "..."
    return cleaned


def _qwen_visual_ocr_messages(
    manifest: ClipManifestV1,
    *,
    media_windows: Sequence[MediaWindowV1],
    metadata: MetadataPayloadV1,
    spec: HFModelSpec | None,
    weapon_skin_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    weapon_name_resolution_rules = _weapon_name_resolution_rules(weapon_skin_map)
    payload = {
        "clip": {
            "clip_id": manifest.clip_id,
            "file_name": manifest.file_name,
            "duration_sec": manifest.duration_sec,
        },
        "metadata": {
            "file_name": metadata.file_name,
            "hud": metadata.user_metadata.get("hud"),
        },
        "weapon_name_resolution_rules": weapon_name_resolution_rules,
        "media_windows": [
            {
                "window_id": window.window_id,
                "start_sec": window.start_sec,
                "end_sec": window.end_sec,
                "prepared_video_frame_count": len(window.prepared_video_frame_paths),
                "prepared_video_metadata": _compact_video_metadata(window.prepared_video_metadata),
                "ocr_video_frame_count": len(_visual_ocr_frame_indices(window)),
                "ocr_video_max_frames": QWEN_OCR_VIDEO_MAX_FRAMES,
                "ocr_video_max_pixels": _qwen_ocr_video_max_pixels(spec),
                "ocr_video_sampling_strategy": QWEN_OCR_VIDEO_SAMPLING_STRATEGY,
                "ocr_video_frame_timestamps_sec": [
                    round(float(timestamps[index]), 3)
                    for timestamps in [_prepared_frame_timestamps(window)]
                    for index in _visual_ocr_frame_indices(window, timestamps=timestamps)
                ],
            }
            for window in media_windows
        ],
    }
    user_text = (
        "Read only the local player's currently selected or held weapon/tool/consumable text from the attached gameplay video frames. "
        "Focus on the current equipment HUD label and weapon skin markings only. "
        "Do not extract kill feed, death-screen text, teammate tags, status prompts, locations, scores, player names, "
        "or general HUD text; those belong to dedicated death-screen/hit-marker detectors or the final Qwen visual summary. "
        "Ignore repeated ammo counters, health-bar numbers, crosshair ticks, UI decoration, unreadable noise, and repeated numeric strings. "
        "Never repeat a token or number more than twice. Keep each text field under 120 characters. "
        "Prefer omitting uncertain/noisy OCR over returning long strings. "
        "Do not summarize the clip and do not infer people, teams, places, or outcomes from visual appearance. "
        "Do not downgrade an exact weapon model to a generic class: if Auto-5 is visible or recognizable, return Auto-5, "
        "not only shotgun; if the model is visually recognized but no text is legible, add an uncertainty that the weapon name is "
        "visual-model evidence. "
        "Use timestamps from the supplied media_windows metadata when possible; otherwise estimate a conservative time range. "
        "Normalize weapon/equipment names using weapon_name_resolution_rules before returning text. "
        "If the raw text or visible weapon skin says Rougarou and the rule maps Rougarou to Mosin Obrez "
        "(Rougarou skin), return Mosin Obrez (Rougarou skin), not Rougarou as a standalone weapon/person name. "
        "Never use weapon skin names as teammate, enemy, hunter, player, or person identities. "
        "Return exactly this JSON shape: {"
        "\"ocr_observations\": [{\"window_id\": string, \"start\": number, \"end\": number, "
        "\"text\": string, \"source_area\": string|null, \"raw_text\": string|null, "
        "\"resolved_equipment\": [{\"raw_name\": string, \"display_name\": string, \"entity_type\": string}], "
        "\"uncertainties\": [string]}], "
        "\"equipment_timeline\": [{\"timestamp\": number, \"entity_name\": string, \"entity_type\": string, "
        "\"source\": \"qwen35_visual_ocr\", \"confidence\": number|null}], "
        "\"uncertainties\": [string]}. "
        "Use at most 8 ocr_observations and at most 8 equipment_timeline rows. "
        "If no currently selected weapon/tool/consumable text is visible, return empty arrays and explain that in uncertainties. "
        "Return valid JSON only, with no markdown and no prose outside JSON. Evidence payload: "
        + json.dumps(payload, ensure_ascii=True)
    )
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for window in media_windows:
        if window.prepared_video_frame_paths or window.video_path:
            user_content.append(_window_visual_ocr_video_payload(window, spec))
    return [
        {
            "role": "system",
            "content": (
                "You are a conservative OCR and HUD text extractor for gameplay frames. "
                "Output strict JSON only. Read text that is visible or strongly legible; mark uncertainty instead of guessing."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def _qwen_visual_event_messages(
    manifest: ClipManifestV1,
    *,
    media_windows: Sequence[MediaWindowV1],
    metadata: MetadataPayloadV1,
    spec: HFModelSpec | None,
    video_payload_budgets: Sequence[VideoPayloadBudgetV1] = (),
    weapon_skin_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    weapon_name_resolution_rules = _weapon_name_resolution_rules(weapon_skin_map)
    budget_rows = [item.model_dump() for item in video_payload_budgets]
    payload = {
        "clip": {
            "clip_id": manifest.clip_id,
            "file_name": manifest.file_name,
            "duration_sec": manifest.duration_sec,
        },
        "metadata": {
            "file_name": metadata.file_name,
            "known_outcome": metadata.user_metadata.get("known_outcome"),
            "confirmed_outcome": metadata.user_metadata.get("confirmed_outcome"),
            "clip_context": metadata.user_metadata.get("clip_context"),
            "hud": metadata.user_metadata.get("hud"),
            "qwen_visual_ocr": metadata.user_metadata.get("qwen_visual_ocr"),
            "hit_marker": metadata.user_metadata.get("hit_marker"),
            "death_screen": metadata.user_metadata.get("death_screen"),
        },
        "weapon_name_resolution_rules": weapon_name_resolution_rules,
        "video_payload_budgets": budget_rows,
        "media_windows": [
            {
                "window_id": window.window_id,
                "start_sec": window.start_sec,
                "end_sec": window.end_sec,
                "prepared_video_frame_count": len(window.prepared_video_frame_paths),
                "prepared_video_metadata": _compact_video_metadata(window.prepared_video_metadata),
            }
            for window in media_windows
        ],
    }
    user_text = (
        "Extract conservative visual gameplay events from the attached video inputs before final summary composition. "
        "Return structured evidence only; do not write the final summary. "
        "Describe action, surroundings, player position, visible enemy position, visible teammate position, and apparent hit/fall/death-screen cues. "
        "Use the media window timestamps and source-original time. Do not analyze or mention media before the supplied window start. "
        "Use only direct visual evidence plus the supplied metadata fields. Do not invent names, teams, exact locations, scores, or events. "
        "Do not identify hunter skin names or teammate/enemy cosmetic names from appearance or name tags. "
        "When equipment is visible or supplied by OCR/HUD, write exactly the canonical weapon/tool/consumable display name from the metadata or "
        "weapon_name_resolution_rules; do not substitute a generic class like shotgun or rifle unless it is part of the supplied name. "
        "If the file name or metadata indicates a confirmed kill, treat the kill outcome as metadata-confirmed, but still describe only the "
        "visible position/timing/weapon evidence you can support from video/OCR/HUD/hit-marker evidence. "
        "Audio is not attached here; do not infer sounds. Return exactly this JSON shape: {"
        "\"visual_events\": [{\"window_id\": string, \"start\": number, \"end\": number, "
        "\"description\": string, \"player_position\": string|null, \"enemy_position\": string|null, "
        "\"teammate_positions\": [string], \"equipment_name\": string|null, \"action\": string|null, "
        "\"confidence\": number|null, \"uncertainties\": [string], "
        "\"evidence_pointers\": [{\"source\": \"video\", \"window_id\": string, \"start\": number, "
        "\"end\": number, \"quote_or_observation\": string}]}], \"uncertainties\": [string]}. "
        "Use at most 8 events total. Return valid JSON only, no markdown. Evidence payload: "
        + json.dumps(payload, ensure_ascii=True)
    )
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for window in media_windows:
        if window.prepared_video_frame_paths or window.video_path:
            user_content.append(_window_visual_event_video_payload(window, spec, video_payload_budgets))
    return [
        {
            "role": "system",
            "content": (
                "You extract visual event evidence from gameplay video into strict JSON. "
                "You are conservative, spatial, timestamped, and do not write unsupported facts."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def _window_visual_event_video_payload(
    window: MediaWindowV1,
    spec: HFModelSpec | None,
    budgets: Sequence[VideoPayloadBudgetV1],
) -> dict[str, Any]:
    payload = _window_video_payload(window, spec)
    budget = _budget_for_window(window, budgets)
    if budget is not None:
        payload["max_frames"] = min(int(payload.get("max_frames") or budget.max_frames), budget.max_frames)
        payload["max_pixels"] = budget.max_pixels
    return payload


def _budget_for_window(window: MediaWindowV1, budgets: Sequence[VideoPayloadBudgetV1]) -> VideoPayloadBudgetV1 | None:
    preferred_stage = "focus_visual" if _is_summary_focus_window(window) else "full_visual"
    for budget in budgets:
        if budget.stage == preferred_stage:
            return budget
    return None


def _parse_qwen_visual_events_json(
    raw: str,
    *,
    manifest: ClipManifestV1,
    model_id: str,
    media_windows: Sequence[MediaWindowV1],
    weapon_resolver: Callable[[str], str | None] | None = None,
) -> list[VisualEventV1]:
    data = _loads_json_object(raw)
    windows = {window.window_id: window for window in media_windows}
    raw_items = data.get("visual_events")
    if not isinstance(raw_items, list):
        raise TypeError("visual-events JSON must contain visual_events list")
    events: list[VisualEventV1] = []
    for index, item in enumerate(raw_items[:12], start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("description") or item.get("text") or "").strip()
        if not text:
            continue
        window_id = str(item.get("window_id") or (media_windows[0].window_id if media_windows else f"visual_event_{index:03d}"))
        window = windows.get(window_id)
        start = _clip_timestamp(float(item.get("start", window.start_sec if window else 0.0)), manifest.duration_sec)
        end = _clip_timestamp(float(item.get("end", window.end_sec if window else start)), manifest.duration_sec)
        if end < start:
            end = start
        pointers = _parse_visual_event_pointers(
            item.get("evidence_pointers"),
            manifest=manifest,
            window_id=window_id,
            start=start,
            end=end,
            fallback=text,
        )
        equipment_name = _clean_equipment_name(item.get("equipment_name"), weapon_resolver=weapon_resolver)
        events.append(
            VisualEventV1(
                clip_id=manifest.clip_id,
                file_name=manifest.file_name,
                window_id=window_id,
                start_sec=start,
                end_sec=max(start, end),
                description=text,
                player_position=_clean_optional_text(item.get("player_position")),
                enemy_position=_clean_optional_text(item.get("enemy_position")),
                teammate_positions=[
                    str(value).strip()
                    for value in item.get("teammate_positions", [])
                    if str(value).strip()
                ] if isinstance(item.get("teammate_positions"), list) else [],
                equipment_name=equipment_name,
                action=_clean_optional_text(item.get("action")),
                confidence=_float_or_none(item.get("confidence")),
                evidence_pointers=pointers,
                uncertainties=[
                    str(value).strip()
                    for value in item.get("uncertainties", [])
                    if str(value).strip()
                ] if isinstance(item.get("uncertainties"), list) else [],
                raw_payload={"model_id": model_id, "raw_event": item},
            )
        )
    return events


def _parse_visual_event_pointers(
    raw: Any,
    *,
    manifest: ClipManifestV1,
    window_id: str,
    start: float,
    end: float,
    fallback: str,
) -> list[EvidencePointerV1]:
    pointers: list[EvidencePointerV1] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "video").strip()
            if source != "video":
                continue
            pointer_start = _clip_timestamp(float(item.get("start", start)), manifest.duration_sec)
            pointer_end = _clip_timestamp(float(item.get("end", end)), manifest.duration_sec)
            quote = str(item.get("quote_or_observation") or fallback).strip()
            if not quote:
                continue
            pointers.append(
                EvidencePointerV1(
                    source="video",
                    window_id=str(item.get("window_id") or window_id),
                    start=pointer_start,
                    end=max(pointer_start, pointer_end),
                    quote_or_observation=quote,
                )
            )
    if pointers:
        return pointers
    return [
        EvidencePointerV1(
            source="video",
            window_id=window_id,
            start=start,
            end=max(start, end),
            quote_or_observation=fallback,
        )
    ]


def _summary_from_ledger_messages(
    manifest: ClipManifestV1,
    *,
    ledger: EvidenceLedgerV1,
    weapon_skin_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    weapon_name_resolution_rules = _weapon_name_resolution_rules(weapon_skin_map)
    payload = {
        "ledger": json.loads(ledger_to_compact_text(ledger)),
        "weapon_name_resolution_rules": weapon_name_resolution_rules,
    }
    user_text = (
        "Compose the final gameplay clip summary using only the supplied text-only evidence ledger. "
        "No video, image, or audio payload is attached to this final call; do not claim you inspected media in this step. "
        "Return exactly this JSON shape: {\"title\": string, \"short_summary\": string, "
        "\"detailed_summary\": string, \"key_moments\": [{\"start\": number, \"end\": number, "
        "\"description\": string, \"evidence\": [\"video\"|\"speech\"|\"audio\"|\"metadata\"], "
        "\"evidence_pointers\": [{\"source\": \"video\"|\"speech\"|\"audio\"|\"metadata\", "
        "\"window_id\": string, \"start\": number, \"end\": number, \"quote_or_observation\": string}]}], "
        "\"tags\": [string], \"detected_language\": string|null, \"uncertainties\": [string]}. "
        f"All timestamps must be source-original and within 0 and {manifest.duration_sec:.3f}. "
        f"Do not mention media before {ledger.timebase.analysis_start_sec:.3f}s except to say it was skipped for analysis. "
        "Every key moment and every detailed-summary claim must be supported by ledger evidence pointers. "
        "Use only evidence labels video, speech, audio, metadata. "
        "Speech evidence must stay verbatim in evidence pointers. "
        "Audio-caption-only claims are uncertain unless corroborated. "
        "Death vocalizations with classification other_death_audio mean someone died elsewhere/off-screen/nearby; do not attribute them "
        "to the player's kill. Death vocalizations classified player_kill_candidate can support the player-kill timing only when paired "
        "with hit marker or metadata-confirmed kill evidence. "
        "If ledger.known_outcome is confirmed_hunter_kill, report the kill as confirmed by clip context/metadata, then use video, "
        "hit-marker, death vocalization, and equipment evidence for timing, position, and weapon. "
        "Never write unsupported weapon names. For the local player's equipment, use only ledger.equipment_timeline.display_name. "
        "If a visual event's equipment_name is null or marked unsupported, do not infer another weapon. "
        "Never use a weapon skin name as a teammate, enemy, hunter, player, or person identity. "
        "Do not identify hunter skins or cosmetic names except death-screen OCR killer/hunter identity. "
        "The detailed_summary must be a scene reconstruction: chronological actions, surroundings/layout, the player's position, "
        "visible enemy position, visible teammate positions, active equipment, hit/death evidence, and uncertainty for missing details. "
        "Keep it concise but detailed, no repeated sentences. Return JSON only. Evidence payload: "
        + json.dumps(payload, ensure_ascii=True)
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a strict text-only evidence-ledger composer. Use only ledger facts and evidence pointers. "
                "Do not invent gameplay details. Return valid JSON only."
            ),
        },
        {"role": "user", "content": user_text},
    ]


def _mock_ledger_summary_text(ledger: EvidenceLedgerV1) -> str:
    parts: list[str] = []
    if ledger.known_outcome:
        parts.append(f"Known outcome: {ledger.known_outcome}.")
    for event in ledger.visual_events[:4]:
        parts.append(f"{event.start_sec:.1f}-{event.end_sec:.1f}s video: {event.description}")
    for hit_marker in ledger.hit_markers[:2]:
        parts.append(f"{hit_marker.timestamp_sec:.1f}s hit marker: {hit_marker.evidence_pointers[0].quote_or_observation}")
    for equipment in ledger.equipment_timeline[:4]:
        parts.append(f"{equipment.start_sec:.1f}s equipment: {equipment.display_name}")
    for vocalization in ledger.death_vocalizations[:3]:
        parts.append(f"{vocalization.start_sec:.1f}-{vocalization.end_sec:.1f}s audio {vocalization.classification}: {vocalization.text}")
    if not parts:
        parts.append(f"Metadata-only evidence for {ledger.file_name}.")
    return " ".join(parts)


def _clean_equipment_name(
    value: Any,
    *,
    weapon_resolver: Callable[[str], str | None] | None,
) -> str | None:
    text = _clean_optional_text(value)
    if not text:
        return None
    if weapon_resolver is not None:
        resolved = weapon_resolver(text)
        if resolved:
            return resolved
    if text.lower() == "rougarou":
        return "Mosin Obrez (Rougarou skin)"
    return text


def _clean_optional_text(value: Any) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text or text.lower() in {"none", "null", "unknown"}:
        return None
    return text


def _split_summary_media_windows(media_windows: Sequence[MediaWindowV1]) -> tuple[list[MediaWindowV1], list[MediaWindowV1]]:
    focus_windows: list[MediaWindowV1] = []
    base_windows: list[MediaWindowV1] = []
    for window in media_windows:
        if _is_summary_focus_window(window):
            focus_windows.append(window)
        else:
            base_windows.append(window)
    if not base_windows and focus_windows:
        base_windows = [focus_windows[0]]
        focus_windows = focus_windows[1:]
    return base_windows, focus_windows


def _is_summary_focus_window(window: MediaWindowV1) -> bool:
    window_id = str(window.window_id or "").lower()
    if window_id == SUMMARY_KILL_FOCUS_WINDOW_ID.lower() or "kill_focus" in window_id:
        return True
    start = float(window.start_sec or 0.0)
    end = float(window.end_sec or 0.0)
    return abs(start - SUMMARY_KILL_FOCUS_START_SEC) <= 0.25 and abs(end - SUMMARY_KILL_FOCUS_END_SEC) <= 0.25


def _summary_focus_refinement_messages(
    manifest: ClipManifestV1,
    *,
    initial_summary: FusedSummaryV1,
    observations: Sequence[VideoObservationV1],
    transcript: ASRTranscriptV1,
    audio_captions: Sequence[AudioCaptionV1],
    metadata: MetadataPayloadV1,
    focus_windows: Sequence[MediaWindowV1],
    spec: HFModelSpec | None = None,
    weapon_resolver: Callable[[str], str | None] | None = None,
    weapon_skin_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    messages = _summary_messages(
        manifest,
        observations=observations,
        transcript=transcript,
        audio_captions=audio_captions,
        metadata=metadata,
        media_windows=focus_windows,
        require_visual_observations=True,
        spec=spec,
        weapon_resolver=weapon_resolver,
        weapon_skin_map=weapon_skin_map,
    )
    return [
        *messages,
        {
            "role": "user",
            "content": (
                "Second-pass focus refinement. The previous JSON summary below is an intermediate draft, "
                "not new evidence. Use it only as the baseline structure, then re-inspect the attached "
                f"{SUMMARY_KILL_FOCUS_START_SEC:g}-{SUMMARY_KILL_FOCUS_END_SEC:g}s focus video window(s), "
                "which contain dense equal-time frames at the same quality as the main summary input. "
                "Return one complete final JSON object with the same schema, not a patch. "
                "Improve the kill-focused part: describe the player's firing position, visible enemy position, "
                "enemy movement/reaction, hit-marker or death-scream evidence, and weapon/equipment using the "
                "authoritative equipment timeline and weapon resolution rules. If the clip context or file name "
                "confirms a kill, state it as metadata-confirmed when the body drop or kill feed is not visible, "
                "then cite video/audio/metadata evidence for timing and position. "
                "Do not carry forward draft claims that conflict with the focus video, OCR, HUD, or equipment timeline. "
                "Keep the summary non-repetitive and use at most 5 key_moments and 8 visual_observations. "
                "Intermediate draft JSON: "
                + json.dumps(initial_summary.model_dump(), ensure_ascii=True)
                + " /no_think"
            ),
        },
    ]


def _summary_messages(
    manifest: ClipManifestV1,
    *,
    observations: Sequence[VideoObservationV1],
    transcript: ASRTranscriptV1,
    audio_captions: Sequence[AudioCaptionV1],
    metadata: MetadataPayloadV1,
    media_windows: Sequence[MediaWindowV1] = (),
    require_visual_observations: bool = False,
    spec: HFModelSpec | None = None,
    weapon_resolver: Callable[[str], str | None] | None = None,
    weapon_skin_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    death_screen_visual_inputs = _death_screen_visual_inputs(metadata, manifest.duration_sec)
    focused_visual_inputs = _focused_visual_inputs(media_windows, metadata)
    weapon_resolution_evidence = _weapon_resolution_evidence(metadata, observations, weapon_resolver)
    weapon_name_resolution_rules = _weapon_name_resolution_rules(weapon_skin_map)
    authoritative_equipment_timeline = _authoritative_equipment_timeline(
        metadata,
        weapon_resolver=weapon_resolver,
    )
    payload = {
        "clip": manifest.model_dump(),
        "video_observations": [item.model_dump() for item in observations],
        "speech_transcript": transcript.model_dump(),
        "audio_captions": [item.model_dump() for item in audio_captions],
        "metadata": metadata.model_dump(),
        "authoritative_equipment_timeline": authoritative_equipment_timeline,
        "weapon_resolution_evidence": weapon_resolution_evidence,
        "weapon_name_resolution_rules": weapon_name_resolution_rules,
        "focused_visual_inputs": [
            {key: value for key, value in item.items() if key != "image_path"}
            for item in focused_visual_inputs
        ],
        "death_screen_visual_inputs": [
            {key: value for key, value in item.items() if key != "image_path"}
            for item in death_screen_visual_inputs
        ],
        "media_windows": [
            {
                "schema_version": item.schema_version,
                "window_id": item.window_id,
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
                "duration_sec": item.duration_sec,
                "prepared_video_frame_count": len(item.prepared_video_frame_paths),
                "prepared_video_metadata": _compact_video_metadata(item.prepared_video_metadata),
            }
            for item in media_windows
        ],
    }
    visual_observation_shape = (
        "\"visual_observations\": [{\"window_id\": string, \"start\": number, \"end\": number, "
        "\"text\": string, \"uncertainties\": [string]}], "
        if require_visual_observations
        else ""
    )
    user_text = (
        "Return exactly this JSON shape: {"
        + visual_observation_shape
        + "\"title\": string, \"short_summary\": string, "
        "\"detailed_summary\": string, \"key_moments\": [{\"start\": number, \"end\": number, "
        "\"description\": string, \"evidence\": [\"video\"|\"speech\"|\"audio\"|\"metadata\"], "
        "\"evidence_pointers\": [{\"source\": \"video\"|\"speech\"|\"audio\"|\"metadata\", "
        "\"window_id\": string, \"start\": number, \"end\": number, "
        "\"quote_or_observation\": string}]}], \"tags\": [string], "
        "\"detected_language\": string|null, \"uncertainties\": [string]}. "
        f"Timestamp rules: all start/end values must be within 0 and {manifest.duration_sec:.3f}. "
        f"Use at most 5 key_moments and at most {SUMMARY_VISUAL_OBSERVATION_MAX_ITEMS} visual_observations. "
        + (
            "visual_observations is required and must describe only directly visible visual evidence from the attached video input; "
            "treat each visual_observations item as one unique timestamped event or interval, keep it chronological, spatial, "
            "and conservative. Each visual_observations[].text must be compact, no more than 45 words, and summarize the clip "
            "as intervals instead of enumerating individual frames or repeated timestamps. "
            "Include at least one direct-video visual_observation for each "
            "attached media window using that media window_id; do not satisfy visual_observations only by restating supplied "
            "HUD/loadout, hit-marker, death-screen, or other detector observations. "
            if require_visual_observations
            else ""
        )
        + (
            "The attached video input is direct visual evidence. Use source label video for claims supported by it. "
            "Inspect the final engagement frames carefully, especially the 16-21 second range when present. If an enemy hunter, "
            "enemy position, hit reaction, downing, kill animation, kill feed/OCR, or death confirmation is visible in the video, "
            "describe it with relative position and timestamped evidence. If a visible enemy is not clear enough to locate, say "
            "that the enemy position is not established rather than omitting the field entirely. "
            "Focused engagement crop images may be attached after the video input; they are enlarged direct-video evidence from "
            "the same prepared frames and include timestamp labels. Inspect them for tiny enemy silhouettes, crosshair alignment, "
            "hit markers, muzzle flash, disappearance/reaction after the shot, or other kill/death confirmation. "
            if media_windows
            else ""
        )
        + "short_summary should be 1-2 dense factual sentences. "
        "detailed_summary must reconstruct the clip in detail from the supplied evidence: chronological actions, "
        "surroundings and room/layout, player/hunter position and movement, visible enemy positions, visible teammate "
        "positions, weapon/equipment if supported, HUD state, and the apparent outcome. "
        "Do not name hunter skins, teammate skins, enemy skins, or character cosmetics from visual appearance, clothing, "
        "silhouette, or teammate name tags. Describe teammates/enemies as 'a teammate', 'enemy hunter', 'hooded figure', "
        "'blue-cloaked teammate', etc. The only allowed hunter/person skin name is one explicitly read from death-screen "
        "OCR as the killer/hunter identity; weapon skin names are allowed only as weapon/equipment evidence. "
        "For required details that are not visible or not supplied, explicitly say they are not visible or not established; "
        "do not fill gaps with guesses. "
        "Use detected_language only from speech_transcript.language; otherwise null. "
        "When HUD/loadout extraction evidence identifies active weapon, active item, or visible loadout, "
        "include that equipment in short_summary or detailed_summary with a video evidence pointer. "
        "The payload field authoritative_equipment_timeline is the canonical timestamped local-player equipment table "
        "built from HUD and Qwen visual OCR. Before writing the summary, resolve the local player's active equipment "
        "from this table: for any timestamp t, use the latest row at or before t until a later row changes it. "
        "For every weapon/equipment claim in visual_observations, short_summary, detailed_summary, key_moments, and "
        "evidence pointers, use authoritative_equipment_timeline[].display_name for the relevant timestamp. "
        "A contradiction with authoritative_equipment_timeline is invalid and will be rejected for repair. "
        "When metadata contains HUD equipment_timeline or prepared_frame_evidence, use it as timestamped evidence for "
        "which local-player weapon, tool, or consumable is active at those times. "
        "When metadata or Qwen visual OCR contains equipment_timeline rows, treat the latest row at or before a timestamp "
        "as authoritative for the local player's active equipment until a later supplied timeline row changes it. "
        "Do not claim the player switched back to a previous weapon at or after a timestamp if the latest supplied "
        "equipment_timeline row names a different weapon; if direct video appears ambiguous, mark the weapon state uncertain "
        "instead of asserting an unsupported switch. "
        "If HUD/loadout, Qwen visual OCR, or weapon_resolution_evidence supplies a concrete weapon model such as Auto-5, "
        "use that exact model name in the summary and key moments; do not replace it with only a generic class like shotgun, "
        "rifle, pistol, firearm, or visible weapon. "
        "When an equipment name is supplied by authoritative_equipment_timeline, HUD/loadout, Qwen visual OCR, or "
        "weapon_resolution_evidence, write exactly that supplied display_name for the weapon/tool/consumable. Do not append, "
        "prepend, or substitute an item type or weapon class such as shotgun, rifle, pistol, revolver, firearm, melee, tool, "
        "or consumable unless that word is already part of the supplied display_name. "
        "Before writing visual_observations, short_summary, detailed_summary, key_moments, or evidence pointer text, normalize "
        "weapon/equipment names using weapon_name_resolution_rules and weapon_resolution_evidence. "
        "If a visible HUD label, weapon skin marking, OCR line, or model-recognized weapon name matches raw_name, you must write "
        "display_name instead of raw_name; for example, do not write 'holding a Rougarou' when the supplied rule says "
        "'Rougarou' resolves to 'Mosin Obrez (Rougarou skin)'. "
        "weapon_resolution_evidence is authoritative only for weapon/equipment naming: use display_name for the weapon or "
        "equipment when raw_name appears in HUD/OCR/equipment evidence, and cite the supplied timestamps. "
        "Never use raw_name or a weapon skin name as a teammate, enemy, hunter, player, or person identity. "
        "HUD/loadout active weapon evidence belongs to the local first-person player/hunter, not to the enemy; "
        "do not say an enemy used that weapon unless direct visible evidence separately supports it. "
        "Use explicit actors such as 'the player' and 'the enemy hunter' instead of ambiguous pronouns when describing who shoots, "
        "is hit, falls, heals, or moves. "
        "If HUD/loadout confidence is missing or marked uncertain, keep the equipment claim explicitly uncertain. "
        "When hit-marker detector evidence is supplied, it may support a probable hit-marker or impact-cue claim. "
        "If a hit-marker detector timestamp is supplied, include it as a key moment unless stronger evidence proves it is irrelevant. "
        "A screen-center hit marker is local-player impact feedback; it is not evidence that the enemy shot or hit the player. "
        "The 'Stop Bleeding' or 'Stopping Bleeding' prompt means the player is bleeding or wounded; it is not evidence that "
        "the player is downed, dead, unable to move, or incapacitated. "
        "Do not say the local player takes damage, is hit, stops bleeding, or sees blood splatter "
        "unless those exact facts are supplied by transcript, OCR/HUD, death-screen, metadata, or an explicit video observation; "
        "healing-item claims such as Devil's Salve are allowed only when visible in video/OCR/HUD evidence. "
        "do not infer them from a hit marker or from generic red/dark pixels. "
        "Do not convert hit-marker evidence into a confirmed kill unless death-screen, kill-feed/OCR, transcript, "
        "death scream or pain/death vocalization audio caption, focused visual crop evidence, or explicit metadata separately "
        "supports the kill. App-generated clip names containing Hunter killed or killed are explicit metadata outcome evidence: "
        "treat the kill outcome as confirmed by metadata, then use video/audio evidence to report timing, player position, "
        "enemy position, and weapon. If no body drop or kill feed is visible, say the kill is metadata-confirmed rather than "
        "visually confirmed. If metadata.user_metadata.known_outcome, confirmed_outcome, or clip_context says "
        "confirmed_hunter_kill, hunter_killed, or killed, treat the kill as confirmed by the clip context itself; do not "
        "downgrade it to only a possible hit. If active weapon evidence is nearby, you may say the "
        "hit marker occurred while that weapon was active, with uncertainty if not corroborated. "
        "This restriction applies to detector evidence alone; if the attached video itself visibly shows an enemy being hit, "
        "falling, downed, killed, or otherwise defeated, describe that visible outcome directly and cite video evidence. "
        "For metadata-confirmed kills, the summary must state the firing position and weapon when they are visible or supplied; "
        "if exact weapon name is not visible, describe the visible weapon conservatively. "
        "Speech evidence must stay verbatim: when citing speech_transcript.segments, copy the exact segment text "
        "into evidence_pointers.quote_or_observation and do not translate, paraphrase, or reinterpret it unless "
        "an explicit translation is supplied as evidence. If you mention an inferred meaning from speech, mark it "
        "uncertain and keep the original transcript quote beside it. "
        "Attached death-screen images, when present, are direct visual evidence for the death screen only. "
        "Use source label video for claims supported by those images, use the supplied death-screen window_id, "
        "and do not infer unrelated names, places, teams, weapons, or events from the image. Evidence payload: "
        + json.dumps(payload, ensure_ascii=True)
    )
    user_content: str | list[dict[str, Any]]
    if death_screen_visual_inputs or media_windows:
        user_content = [{"type": "text", "text": user_text}]
        for window in media_windows:
            if window.prepared_video_frame_paths or window.video_path:
                user_content.append(_window_video_payload(window, spec))
        if focused_visual_inputs:
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "Focused engagement crops follow. They are enlarged direct-video frame crops from the "
                        f"{FOCUS_WINDOW_START_SEC:g}-{FOCUS_WINDOW_END_SEC:g}s focus interval and hit-marker neighborhood. "
                        "Use them only as visual evidence; each crop contains its timestamp label."
                    ),
                }
            )
        image_max_pixels = _focused_image_max_pixels(spec)
        for item in focused_visual_inputs:
            user_content.append(
                {
                    "type": "image",
                    "image": item["image_path"],
                    "max_pixels": image_max_pixels,
                }
            )
        for item in death_screen_visual_inputs:
            user_content.append(
                {
                    "type": "image",
                    "image": item["image_path"],
                }
            )
    else:
        user_content = user_text
    return [
        {
            "role": "system",
            "content": (
                "You fuse video observations, speech transcript segments, uncertain audio captions, and metadata "
                "into strict JSON for gameplay clip retrieval. Use only supplied evidence. Do not invent names, "
                "teams, scores, locations, calls, brands, weapons, people, or events. Treat file_name as metadata "
                "only; any claim based only on file_name must be marked metadata-derived and uncertain. "
                "Audio-caption-only claims must be explicitly uncertain unless corroborated by video, speech, or metadata. "
                "Speech transcript text is evidence in its original language. Preserve it verbatim in speech evidence pointers; "
                "do not translate, paraphrase, or convert speech into an English claim unless that translation is supplied "
                "as evidence, and mark any inferred meaning as uncertain. "
                "authoritative_equipment_timeline is binding timestamped local-player equipment evidence; derive weapon/tool "
                "state from it before writing action descriptions, and never contradict its latest row at a timestamp. "
                "HUD/loadout observations are extracted from prepared Qwen frames or representative video frames and may support weapon/equipment "
                "claims only for the exact detected names and loadout items supplied in the evidence. "
                "weapon_name_resolution_rules are authoritative weapon/equipment normalization rules. If visual content, OCR, "
                "HUD text, or your model recognition produces a raw weapon skin name that appears in those rules, output the "
                "corresponding display_name everywhere, including visual_observations and evidence pointers. "
                "For supplied weapons, tools, and consumables, use only the supplied display_name as the equipment name. Do not "
                "add weapon/item type words such as shotgun, rifle, pistol, revolver, firearm, tool, item, or consumable unless "
                "the supplied display_name already contains that word. "
                "Hit-marker observations are deterministic frame evidence for a probable hit marker or impact cue. "
                "They support hit-cue claims only, not confirmed kills unless corroborated by death-screen, kill-feed/OCR, "
                "speech, death scream/pain-vocalization audio captions, focused visual crops, or explicit metadata. "
                "'Stop Bleeding' and 'Stopping Bleeding' HUD prompts mean bleeding/wounded, not downed, dead, unable to move, or incapacitated. "
                "Do not say the local player takes damage, is hit, stops bleeding, or sees blood splatter "
                "unless those exact facts are supplied by transcript, OCR/HUD, death-screen, metadata, or an explicit video observation. "
                "Healing-item claims such as Devil's Salve are allowed only when visible in video/OCR/HUD evidence. "
                "App-generated clip names containing Hunter killed or killed are explicit metadata outcome evidence; report the "
                "kill as metadata-confirmed and use visual/audio evidence for position, weapon, and timing. "
                "metadata.user_metadata.known_outcome, confirmed_outcome, or clip_context values such as confirmed_hunter_kill, "
                "hunter_killed, or killed are explicit clip-context evidence that the kill is confirmed by the nature of the clip; "
                "use that context to report a confirmed kill while still citing video/audio evidence for timing and weapon. "
                "Do not let that detector-only rule suppress direct visible video evidence: if the attached video shows an enemy "
                "hunter after the hit marker, a fall/down/death animation, kill confirmation, or kill-feed text, describe it with "
                "the enemy's relative position and a timestamped video evidence pointer. "
                "The detailed_summary should be a scene reconstruction, not a label: include what happens, where the hunter "
                "is positioned, what is around them, where enemies and teammates are visible or unknown, and how the action unfolds. "
        "Never identify teammate/enemy/player hunter-skin or character-cosmetic names from visual appearance or name tags; "
        "the only person skin/name exception is death-screen OCR for who killed the player. "
        "Do not enumerate every sampled frame; group repeated aiming, turning, firing, movement, or reloading into time ranges. "
        "Never repeat the same sentence or near-identical action claim. If several sampled frames show the same posture, "
        "reload, aim, or movement, describe it once as a continuous interval. "
        "If you start producing a repeated loop, stop immediately and return the concise JSON completed so far. "
        "Key moments must be unique events, not frame-by-frame narration. "
        "Every key moment and every factual claim in detailed_summary must be supported by at least one "
                "evidence pointer. If evidence is weak or missing, say that in uncertainties instead of guessing. "
                "Return valid JSON only: start with { and end with }. No markdown, no code fences, no comments, no prose outside JSON."
                " Do not output thinking text or <think> tags."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _compact_video_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            "frame_paths",
            "qwen_video_frame_paths",
            "prepared_video_frame_paths",
            "qwen_video_frame_dir",
        }
    }


def _focused_visual_inputs(
    media_windows: Sequence[MediaWindowV1],
    metadata: MetadataPayloadV1,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    hit_marker = metadata.user_metadata.get("hit_marker")
    hit_timestamps = _hit_marker_timestamps(hit_marker)
    for window in media_windows:
        if not window.prepared_video_frame_paths:
            continue
        timestamps = _prepared_frame_timestamps(window)
        if not timestamps:
            continue
        selected_indices = _focused_frame_indices(timestamps, hit_timestamps)
        raw_frame_dir = str(window.prepared_video_metadata.get("qwen_video_frame_dir") or "").strip()
        if raw_frame_dir:
            frame_dir = Path(raw_frame_dir).expanduser()
        else:
            frame_dir = Path(window.prepared_video_frame_paths[0]).expanduser().resolve().parent
        crop_dir = frame_dir / SUMMARY_FOCUSED_VISUAL_VERSION
        for index in selected_indices:
            if index < 0 or index >= len(window.prepared_video_frame_paths):
                continue
            source = Path(window.prepared_video_frame_paths[index]).expanduser()
            if not source.exists():
                continue
            timestamp = round(float(timestamps[index]), 3)
            try:
                image_path = _write_focused_visual_crop(source, crop_dir, index=index, timestamp=timestamp)
            except Exception:
                continue
            key = str(image_path)
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "window_id": window.window_id,
                    "source": "video",
                    "timestamp": timestamp,
                    "frame_index": index,
                    "image_path": str(image_path),
                    "purpose": "focused_engagement_crop",
                    "sampling_strategy": window.prepared_video_metadata.get(
                        "qwen_video_temporal_sampling_strategy",
                        TEMPORAL_SAMPLING_STRATEGY,
                    ),
                }
            )
            if len(output) >= SUMMARY_FOCUSED_VISUAL_MAX_IMAGES:
                return output
    return output


def _hit_marker_timestamps(hit_marker: Any) -> list[float]:
    if not isinstance(hit_marker, dict):
        return []
    timestamps: list[float] = []
    if hit_marker.get("timestamp") is not None:
        try:
            timestamps.append(float(hit_marker["timestamp"]))
        except (TypeError, ValueError):
            pass
    evidence = hit_marker.get("evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict) or item.get("timestamp") is None:
                continue
            try:
                timestamps.append(float(item["timestamp"]))
            except (TypeError, ValueError):
                continue
    return sorted({round(value, 3) for value in timestamps})


def _prepared_frame_timestamps(window: MediaWindowV1) -> list[float]:
    raw = window.prepared_video_metadata.get("qwen_video_frame_timestamps_sec")
    if isinstance(raw, list) and len(raw) == len(window.prepared_video_frame_paths):
        timestamps: list[float] = []
        for index, value in enumerate(raw):
            try:
                timestamps.append(float(value))
            except (TypeError, ValueError):
                timestamps.append(index / window.prepared_video_sample_fps if window.prepared_video_sample_fps else float(index))
        return timestamps
    step = 1.0 / window.prepared_video_sample_fps if window.prepared_video_sample_fps else 0.0
    return [index * step for index in range(len(window.prepared_video_frame_paths))]


def _focused_frame_indices(timestamps: Sequence[float], hit_timestamps: Sequence[float]) -> list[int]:
    target_times = [
        FOCUS_WINDOW_START_SEC,
        FOCUS_WINDOW_START_SEC + 0.75,
        FOCUS_WINDOW_START_SEC + 1.5,
        FOCUS_WINDOW_START_SEC + 2.0,
        FOCUS_WINDOW_END_SEC - 1.0,
        FOCUS_WINDOW_END_SEC - 0.5,
    ]
    for timestamp in hit_timestamps:
        target_times.extend([timestamp - 0.25, timestamp, timestamp + 0.25])
    bounded = [
        max(min(float(timestamp), max(timestamps)), min(timestamps))
        for timestamp in target_times
        if FOCUS_WINDOW_START_SEC - 0.5 <= float(timestamp) <= FOCUS_WINDOW_END_SEC + 0.5
    ]
    indices: list[int] = []
    for target in bounded:
        nearest = min(range(len(timestamps)), key=lambda index: abs(float(timestamps[index]) - target))
        if nearest not in indices:
            indices.append(nearest)
    return sorted(indices)[:SUMMARY_FOCUSED_VISUAL_MAX_IMAGES]


def _write_focused_visual_crop(source: Path, crop_dir: Path, *, index: int, timestamp: float) -> Path:
    from PIL import Image, ImageDraw, ImageEnhance, ImageOps

    key = sha256(f"{source.resolve()}|{source.stat().st_mtime_ns}|{index}|{timestamp}|{SUMMARY_FOCUSED_VISUAL_VERSION}".encode()).hexdigest()[:12]
    crop_dir.mkdir(parents=True, exist_ok=True)
    destination = crop_dir / f"focus_{index:04d}_{timestamp:08.3f}_{key}.jpg"
    if destination.exists():
        return destination
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        left = int(width * 0.24)
        top = int(height * 0.04)
        right = int(width * 0.86)
        bottom = int(height * 0.82)
        crop = rgb.crop((left, top, right, bottom))
        crop = ImageOps.autocontrast(crop, cutoff=0.5)
        crop = ImageEnhance.Contrast(crop).enhance(1.28)
        crop = ImageEnhance.Sharpness(crop).enhance(1.25)
        if crop.width < 900:
            scale = min(2.0, 900 / max(crop.width, 1))
            crop = crop.resize((int(round(crop.width * scale)), int(round(crop.height * scale))), Image.Resampling.LANCZOS)
        if crop.width > 960:
            crop = crop.resize((960, max(2, int(round(crop.height * (960 / crop.width))))), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(crop)
        label = f"focus frame {index:04d} t={timestamp:.3f}s"
        draw.rectangle([0, 0, min(crop.width, 420), 30], fill=(0, 0, 0))
        draw.text((8, 7), label, fill=(255, 255, 255))
        crop.save(destination, format="JPEG", quality=94)
    return destination


def _focused_image_max_pixels(spec: HFModelSpec | None) -> int:
    max_input = spec.max_input if spec is not None and spec.max_input else {}
    configured = max_input.get("focused_image_max_pixels", max_input.get("image_max_pixels", max_input.get("video_max_pixels")))
    if configured is None:
        return 524288
    return max(262144, min(int(configured) * 2, 786432))


def _death_screen_visual_inputs(metadata: MetadataPayloadV1, duration_sec: float) -> list[dict[str, Any]]:
    death_screen = metadata.user_metadata.get("death_screen")
    if not isinstance(death_screen, dict):
        return []
    raw_path = str(death_screen.get("frame_path") or "").strip()
    if not raw_path:
        return []
    path = Path(raw_path).expanduser()
    if not path.exists():
        return []
    try:
        timestamp = _clip_timestamp(float(death_screen.get("timestamp") or 0.0), duration_sec)
    except (TypeError, ValueError):
        timestamp = 0.0
    return [
        {
            "window_id": "death_screen_frame",
            "source": "video",
            "image_path": str(path),
            "start": timestamp,
            "end": timestamp,
            "status": death_screen.get("status"),
            "killed_with": death_screen.get("killed_with"),
            "killer_name": death_screen.get("killer_name"),
            "confidence": death_screen.get("confidence"),
            "raw_text": death_screen.get("raw_text"),
            "uncertainty": "Death-screen image evidence is limited to visible death-screen/OCR content.",
        }
    ]


def _summary_repair_messages(
    original_messages: Sequence[dict[str, Any]],
    *,
    raw: str,
    error: str,
) -> list[dict[str, Any]]:
    messages = _with_system_instruction(
        original_messages,
        (
            "Repair mode. The next assistant message must be exactly one JSON object and nothing else. "
            "Do not describe the repair. Do not include markdown, code fences, comments, or prose outside JSON. "
            "Use only these evidence source labels: video, speech, audio, metadata. Never use focus as an "
            "evidence source label. Use exact schema keys only."
        ),
    )
    return [
        *messages,
        {
            "role": "user",
            "content": (
                "Repair the previous response using only the original evidence. "
                "The invalid response is intentionally omitted; do not copy malformed keys, unsupported facts, "
                "or repeated text from it. "
                f"Validation error: {error}. Return valid JSON only with the exact requested schema. "
                "Allowed evidence source labels are exactly video, speech, audio, metadata. "
                "Use at most 3 key_moments and at most 2 visual_observations. "
                "Every key_moment must include evidence_pointers, and every evidence pointer must use keys "
                "source, window_id, start, end, quote_or_observation. Do not use start_sec or end_sec in "
                "evidence_pointers. "
                "Remove repeated or near-duplicate sentences, collapse repeated actions into one interval, "
                "and keep key moments unique. Do not mention player damage, bleeding, enemy-hit-player claims, "
                "or similar HUD states unless supplied evidence explicitly states them. Healing-item claims such as "
                "Devil's Salve are allowed only when visible in video/OCR/HUD evidence. "
                "Start with { and end with }. Do not add unsupported facts. /no_think"
            ),
        },
    ]


def _empty_response_messages(
    original_messages: Sequence[dict[str, Any]],
    *,
    instruction: str,
) -> list[dict[str, Any]]:
    messages = _with_system_instruction(
        original_messages,
        (
            "Retry mode. The previous model response was empty. The next assistant message must be exactly "
            "one JSON object and nothing else. Do not include markdown, comments, explanations, repair notes, "
            "or prose outside the JSON object."
        ),
    )
    return [
        *messages,
        {
            "role": "user",
            "content": (
                f"{instruction} Use only the original evidence. Start with {{ and end with }}. "
                "Do not add unsupported facts. /no_think"
            ),
        },
    ]


def _with_system_instruction(original_messages: Sequence[dict[str, Any]], instruction: str) -> list[dict[str, Any]]:
    messages = [dict(message) for message in original_messages]
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = f"{messages[0].get('content', '')}\n\n{instruction}"
        return messages
    return [{"role": "system", "content": instruction}, *messages]


def _final_answer_messages(
    original_messages: Sequence[dict[str, Any]],
    *,
    reasoning: str,
    instruction: str,
) -> list[dict[str, Any]]:
    trimmed_reasoning = reasoning.strip()
    if len(trimmed_reasoning) > 4000:
        trimmed_reasoning = trimmed_reasoning[-4000:]
    return [
        *original_messages,
        {"role": "assistant", "content": f"<think>\n{trimmed_reasoning}\n</think>\n\n"},
        {
            "role": "user",
            "content": (
                "Your previous response contained reasoning but no final answer. "
                f"{instruction} Do not include reasoning, markdown, code fences, or <think> tags. /no_think"
            ),
        },
    ]


def _audio_caption_prompt(file_name: str, start_sec: float, end_sec: float) -> str:
    return (
        "Caption only the audible non-speech gameplay sounds in this window. "
        "Mention gunshots, footsteps, impacts, UI/game sounds, ambience, and especially any human pain cry, "
        "death scream, or downed/death vocalization if audible. "
        "If speech is present, say speech is present but do not transcribe it. "
        "Do not infer speakers, names, teams, weapon names, directions, or visual outcomes. "
        "Return one cautious plain-text sentence."
    )


def _first_pointer(
    observations: Sequence[VideoObservationV1],
    speech: Sequence[ASRSegmentV1],
    captions: Sequence[AudioCaptionV1],
    metadata: MetadataPayloadV1,
) -> EvidencePointerV1:
    for items in (observations, speech, captions):
        if items:
            item = items[0]
            return EvidencePointerV1(
                source=item.source,
                window_id=item.window_id,
                start=item.start_sec,
                end=item.end_sec,
                quote_or_observation=item.text,
            )
    return EvidencePointerV1(
        source="metadata",
        window_id="metadata",
        start=0.0,
        end=0.0,
        quote_or_observation=f"file_name: {metadata.file_name}",
    )


def _compact(text: str, *, max_words: int) -> str:
    words = re.sub(r"\s+", " ", text).strip().split()
    if not words:
        return ""
    suffix = "..." if len(words) > max_words else ""
    return " ".join(words[:max_words]).rstrip(".,;") + suffix


def _normalize_reasoning_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"off", "none", "false", "0"}:
        return "off"
    if normalized in {"full", "high", "true", "1", "on"}:
        return "full"
    return "low"


def _strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    return cleaned


def _summary_visual_observations(
    summary: FusedSummaryV1,
    *,
    manifest: ClipManifestV1,
    media_windows: Sequence[MediaWindowV1],
    model_id: str,
) -> list[VideoObservationV1]:
    raw_items = summary.raw_payload.get("visual_observations")
    windows_by_id = {window.window_id: window for window in media_windows}
    observations: list[VideoObservationV1] = []
    if isinstance(raw_items, list):
        seen_signatures: set[str] = set()
        for index, item in enumerate(raw_items, start=1):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("observation") or "").strip()
            if not text:
                continue
            signature = _canonical_summary_sentence(text, drop_timestamps=True)
            if signature and signature in seen_signatures:
                continue
            if signature:
                seen_signatures.add(signature)
            window_id = str(item.get("window_id") or (media_windows[0].window_id if media_windows else f"summary_video_{index:03d}"))
            window = windows_by_id.get(window_id)
            start = _clip_timestamp(float(item.get("start", window.start_sec if window else 0.0)), manifest.duration_sec)
            end = _clip_timestamp(float(item.get("end", window.end_sec if window else manifest.duration_sec)), manifest.duration_sec)
            observations.append(
                VideoObservationV1(
                    clip_id=manifest.clip_id,
                    file_name=manifest.file_name,
                    window_id=window_id,
                    start_sec=start,
                    end_sec=max(start, end),
                    model_id=model_id,
                    text=text,
                    uncertainties=[str(value) for value in item.get("uncertainties", [])]
                    if isinstance(item.get("uncertainties"), list)
                    else [],
                    raw_payload={
                        "video_input_mode": "qwen35_combined_summary_visual_observation",
                        "combined_summary": True,
                    },
                )
            )
    if observations:
        return observations
    if not media_windows:
        return []
    window = media_windows[0]
    return [
        VideoObservationV1(
            clip_id=manifest.clip_id,
            file_name=manifest.file_name,
            window_id=window.window_id,
            start_sec=window.start_sec,
            end_sec=window.end_sec,
            model_id=model_id,
            text=summary.detailed_summary,
            uncertainties=["Visual observation was reconstructed from the combined summary because visual_observations was missing."],
            raw_payload={
                "video_input_mode": "qwen35_combined_summary_fallback",
                "combined_summary": True,
            },
        )
    ]


def _parse_summary_with_video_contract(
    raw: str,
    *,
    manifest: ClipManifestV1,
    model_id: str,
    media_windows: Sequence[MediaWindowV1],
    metadata: MetadataPayloadV1 | None = None,
    weapon_resolver: Callable[[str], str | None] | None = None,
) -> FusedSummaryV1:
    return _parse_summary_contract(
        raw,
        manifest=manifest,
        model_id=model_id,
        media_windows=media_windows,
        metadata=metadata,
        weapon_resolver=weapon_resolver,
    )


def _parse_summary_contract(
    raw: str,
    *,
    manifest: ClipManifestV1,
    model_id: str,
    media_windows: Sequence[MediaWindowV1] = (),
    metadata: MetadataPayloadV1 | None = None,
    weapon_resolver: Callable[[str], str | None] | None = None,
) -> FusedSummaryV1:
    summary = parse_summary_json(raw, manifest=manifest, model_id=model_id)
    errors = _summary_contract_errors(
        summary,
        media_windows=media_windows,
        metadata=metadata,
        weapon_resolver=weapon_resolver,
    )
    if errors:
        raise ValueError("; ".join(errors))
    return summary


def _summary_contract_errors(
    summary: FusedSummaryV1,
    *,
    media_windows: Sequence[MediaWindowV1] = (),
    metadata: MetadataPayloadV1 | None = None,
    weapon_resolver: Callable[[str], str | None] | None = None,
) -> list[str]:
    return [
        *_summary_video_contract_errors(summary, media_windows=media_windows),
        *_summary_equipment_timeline_contract_errors(
            summary,
            metadata=metadata,
            weapon_resolver=weapon_resolver,
        ),
        *_summary_repetition_contract_errors(summary),
    ]


def _summary_video_contract_errors(
    summary: FusedSummaryV1,
    *,
    media_windows: Sequence[MediaWindowV1],
) -> list[str]:
    errors: list[str] = []
    required_window_ids = {
        window.window_id
        for window in media_windows
        if window.window_id and (window.prepared_video_frame_paths or window.video_path)
    }
    if required_window_ids:
        raw_items = summary.raw_payload.get("visual_observations")
        observed_window_ids = {
            str(item.get("window_id"))
            for item in raw_items
            if isinstance(item, dict) and str(item.get("text") or item.get("observation") or "").strip()
        } if isinstance(raw_items, list) else set()
        missing = sorted(required_window_ids - observed_window_ids)
        if missing:
            errors.append(
                "visual_observations must include direct attached video observations for window_id(s): "
                + ", ".join(missing)
                + "; detector-only windows are insufficient"
            )

    for sentence in re.split(r"(?<=[.!?])\s+", f"{summary.short_summary} {summary.detailed_summary}"):
        lowered = sentence.lower()
        if (
            ("stop bleeding" in lowered or "stopping bleeding" in lowered)
            and any(phrase in lowered for phrase in ("downed", "unable to move", "incapacitated", "dead"))
        ):
            errors.append(
                "Stop Bleeding/Stopping Bleeding prompts indicate bleeding or wounded state; do not call the player downed, dead, unable to move, or incapacitated from that prompt"
            )
            break
        if "hit marker" not in lowered:
            continue
        if (
            "player was hit" in lowered
            or "player is hit" in lowered
            or "enemy hunter fires at the player" in lowered
            or "enemy fires at the player" in lowered
            or "enemy hunter shoots the player" in lowered
            or "enemy shoots the player" in lowered
        ):
            errors.append(
                "hit-marker evidence is local-player impact feedback; do not describe it as the enemy hitting or shooting the player"
            )
            break
    return errors


def _summary_repetition_contract_errors(summary: FusedSummaryV1) -> list[str]:
    errors: list[str] = []
    if len(summary.detailed_summary) > SUMMARY_DETAILED_MAX_CHARS:
        errors.append(
            "detailed_summary is too long and likely contains frame-by-frame narration or repetition; "
            f"keep it under {SUMMARY_DETAILED_MAX_CHARS} characters and group repeated actions into intervals"
        )
    duplicate = _first_duplicate_sentence(summary.detailed_summary)
    if duplicate:
        errors.append(f"detailed_summary repeats a sentence; rewrite once only: {duplicate}")
    action_repeat = _first_repeated_action_signature(summary.detailed_summary)
    if action_repeat:
        errors.append(
            "detailed_summary repeats the same action pattern too many times; "
            f"collapse it into one interval instead of listing sampled frames: {action_repeat}"
        )

    raw_items = summary.raw_payload.get("visual_observations")
    if isinstance(raw_items, list):
        if len(raw_items) > SUMMARY_VISUAL_OBSERVATION_MAX_ITEMS:
            errors.append(
                "visual_observations contains too many items; return at most "
                f"{SUMMARY_VISUAL_OBSERVATION_MAX_ITEMS} unique events or intervals"
            )
        seen_observations: dict[str, int] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("observation") or "").strip()
            if not text:
                continue
            if len(text) > SUMMARY_VISUAL_OBSERVATION_MAX_CHARS:
                errors.append(
                    "visual_observations text is too long; make each observation compact and interval-based"
                )
                break
            duplicate = _first_duplicate_sentence(text)
            if duplicate:
                errors.append(f"visual_observations repeats a sentence; rewrite once only: {duplicate}")
                break
            action_repeat = _first_repeated_action_signature(text)
            if action_repeat:
                errors.append(
                    "visual_observations repeats the same action pattern too many times; "
                    f"collapse it into one interval: {action_repeat}"
                )
                break
            signature = _canonical_summary_sentence(text, drop_timestamps=True)
            if signature:
                seen_observations[signature] = seen_observations.get(signature, 0) + 1
                if seen_observations[signature] > 1:
                    errors.append("visual_observations contains duplicate or near-duplicate observations")
                    break

    seen_moments: dict[str, int] = {}
    for moment in summary.key_moments:
        signature = _canonical_summary_sentence(moment.description, drop_timestamps=True)
        if not signature:
            continue
        seen_moments[signature] = seen_moments.get(signature, 0) + 1
        if seen_moments[signature] > 1:
            errors.append("key_moments contains duplicate or near-duplicate descriptions")
            break
    return errors


def _summary_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
        if sentence.strip()
    ]


def _first_duplicate_sentence(text: str) -> str | None:
    seen: dict[str, int] = {}
    for sentence in _summary_sentences(text):
        signature = _canonical_summary_sentence(sentence, drop_timestamps=False)
        if len(signature.split()) < 6 or len(signature) < 36:
            continue
        seen[signature] = seen.get(signature, 0) + 1
        if seen[signature] > 1:
            return sentence[:180]
    return None


def _first_repeated_action_signature(text: str) -> str | None:
    seen: dict[str, int] = {}
    for sentence in _summary_sentences(text):
        lowered = sentence.lower()
        if not re.search(r"\b(reload(?:s|ed|ing)?|aim(?:s|ed|ing)?|turn(?:s|ed|ing)?|move(?:s|d|ing)?)\b", lowered):
            continue
        signature = _canonical_summary_sentence(sentence, drop_timestamps=True)
        if len(signature.split()) < 4:
            continue
        seen[signature] = seen.get(signature, 0) + 1
        if seen[signature] > 2:
            return signature[:180]
    return None


def _canonical_summary_sentence(text: str, *, drop_timestamps: bool) -> str:
    normalized = re.sub(r"\s+", " ", str(text).lower()).strip()
    if drop_timestamps:
        normalized = re.sub(r"\b\d+(?:\.\d+)?\s*s(?:ec(?:ond)?s?)?\b", "<time>", normalized)
        normalized = re.sub(r"\bat\s+<time>\b", "at <time>", normalized)
    normalized = re.sub(r"[^a-z0-9()<>.' -]+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .")
    return normalized


def _summary_equipment_timeline_contract_errors(
    summary: FusedSummaryV1,
    *,
    metadata: MetadataPayloadV1 | None,
    weapon_resolver: Callable[[str], str | None] | None,
) -> list[str]:
    if metadata is None:
        return []
    context = _equipment_timeline_conflict_context(metadata, weapon_resolver=weapon_resolver)
    if context is None:
        return []
    text = _summary_text_for_contract(summary)
    latest_name = str(context.get("latest_name") or "").strip()
    latest_time = _float_or_none(context.get("latest_time"))
    if not latest_name or latest_time is None:
        return []
    for previous_name in context.get("previous_names") or []:
        for pattern in _switch_back_patterns(str(previous_name)):
            if re.search(pattern, text, flags=re.IGNORECASE):
                return [
                    "summary contradicts authoritative_equipment_timeline: "
                    f"at and after {latest_time:.2f}s the latest local-player weapon is {latest_name}; "
                    f"do not say the player switched back to {previous_name}. Use the OCR timestamped display_name "
                    "or mark direct-video weapon state uncertain."
                ]
    return []


def _summary_text_for_contract(summary: FusedSummaryV1) -> str:
    parts = [summary.short_summary, summary.detailed_summary]
    parts.extend(moment.description for moment in summary.key_moments)
    parts.extend(pointer.quote_or_observation for moment in summary.key_moments for pointer in moment.evidence_pointers)
    raw_observations = summary.raw_payload.get("visual_observations")
    if isinstance(raw_observations, list):
        for item in raw_observations:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("observation") or ""))
    return " ".join(part for part in parts if part)


def _ensure_deterministic_observation_key_moments(
    summary: FusedSummaryV1,
    observations: Sequence[VideoObservationV1],
    manifest: ClipManifestV1,
    *,
    transcript: ASRTranscriptV1 | None = None,
    audio_captions: Sequence[AudioCaptionV1] = (),
    metadata: MetadataPayloadV1 | None = None,
    weapon_resolver: Callable[[str], str | None] | None = None,
) -> FusedSummaryV1:
    additions: list[KeyMomentV1] = []
    detail_additions: list[str] = []
    uncertainties = list(summary.uncertainties)
    hit_marker_observation: VideoObservationV1 | None = None
    for observation in observations:
        if observation.window_id != "hit_marker_detection":
            continue
        hit_marker_observation = observation
        if _summary_contains_observation_pointer(summary, observation):
            continue
        pointer = EvidencePointerV1(
            source="video",
            window_id=observation.window_id,
            start=_clip_timestamp(observation.start_sec, manifest.duration_sec),
            end=_clip_timestamp(observation.end_sec, manifest.duration_sec),
            quote_or_observation=observation.text,
        )
        additions.append(
            KeyMomentV1(
                start=pointer.start,
                end=pointer.end,
                description=_compact(observation.text, max_words=28),
                evidence=["video"],
                evidence_pointers=[pointer],
            )
        )
        if observation.text and observation.text not in summary.detailed_summary:
            detail_additions.append(observation.text)
        for uncertainty in observation.uncertainties:
            if uncertainty not in uncertainties:
                uncertainties.append(uncertainty)
    confirmed_kill = _metadata_confirmed_kill_key_moment(
        summary,
        hit_marker_observation,
        audio_captions=audio_captions,
        metadata=metadata,
        manifest=manifest,
        weapon_resolver=weapon_resolver,
    )
    short_summary = summary.short_summary
    if confirmed_kill is not None:
        additions.append(confirmed_kill["moment"])
        short_summary = confirmed_kill["short_summary"]
        if confirmed_kill["detail"] not in summary.detailed_summary:
            detail_additions.append(confirmed_kill["detail"])
        for uncertainty in confirmed_kill["uncertainties"]:
            if uncertainty not in uncertainties:
                uncertainties.append(uncertainty)
    detailed_summary = summary.detailed_summary
    if detail_additions:
        detailed_summary = f"{detailed_summary.rstrip()} {' '.join(detail_additions)}".strip()
    key_moments = [*summary.key_moments, *additions]
    canonical_weapon = _canonical_hunt_weapon_from_text(
        f"{short_summary} {detailed_summary} "
        + " ".join(moment.description for moment in key_moments)
        + " ".join(pointer.quote_or_observation for moment in key_moments for pointer in moment.evidence_pointers),
        weapon_resolver=weapon_resolver,
    )
    if canonical_weapon:
        short_summary = _canonicalize_weapon_terms(short_summary, canonical_weapon)
        detailed_summary = _canonicalize_weapon_terms(detailed_summary, canonical_weapon)
        key_moments = [_canonicalize_key_moment_weapon_terms(moment, canonical_weapon) for moment in key_moments]
    short_summary = _normalize_weapon_type_mismatches(short_summary)
    detailed_summary = _normalize_weapon_type_mismatches(detailed_summary)
    key_moments = [_normalize_key_moment_weapon_type_mismatches(moment) for moment in key_moments]
    short_summary = _sanitize_hunter_identity_claims(short_summary)
    detailed_summary = _sanitize_hunter_identity_claims(detailed_summary)
    key_moments = [_sanitize_key_moment_hunter_identity_claims(moment) for moment in key_moments]
    raw_payload = _sanitize_summary_raw_payload_hunter_identity_claims(summary.raw_payload, canonical_weapon=canonical_weapon)
    tags = _sanitize_hunter_identity_tags(
        _summary_tags_with_kill(summary.tags, confirmed_kill is not None),
        weapon_resolver=weapon_resolver,
    )
    updated = summary.model_copy(
        update={
            "short_summary": short_summary,
            "detailed_summary": detailed_summary,
            "key_moments": key_moments,
            "tags": tags,
            "uncertainties": uncertainties,
            "raw_payload": raw_payload,
        }
    )
    updated.validate_evidence_bounds(manifest.duration_sec)
    return updated


def _summary_contains_observation_pointer(summary: FusedSummaryV1, observation: VideoObservationV1) -> bool:
    for moment in summary.key_moments:
        for pointer in moment.evidence_pointers:
            if pointer.window_id != observation.window_id:
                continue
            if pointer.end >= observation.start_sec and pointer.start <= observation.end_sec:
                return True
    return False


def _metadata_confirmed_kill_key_moment(
    summary: FusedSummaryV1,
    hit_marker: VideoObservationV1 | None,
    *,
    audio_captions: Sequence[AudioCaptionV1],
    metadata: MetadataPayloadV1 | None,
    manifest: ClipManifestV1,
    weapon_resolver: Callable[[str], str | None] | None = None,
) -> dict[str, Any] | None:
    if hit_marker is None or metadata is None:
        return None
    if not _metadata_claims_kill(metadata):
        return None
    audio_caption = _nearby_death_vocalization_caption(audio_captions, hit_marker.start_sec, hit_marker.end_sec)
    if _summary_already_reports_confirmed_kill(summary):
        return None
    hit_time = (hit_marker.start_sec + hit_marker.end_sec) / 2.0
    weapon = _kill_weapon_from_evidence(summary, metadata, hit_time=hit_time, weapon_resolver=weapon_resolver)
    position = _kill_position_from_evidence(summary)
    audio_clause = (
        " Nearby audio evidence also reports a death scream or pain/death vocalization."
        if audio_caption is not None
        else ""
    )
    metadata_source = _kill_metadata_source_description(metadata)
    detail = (
        f"Metadata-confirmed hunter kill near {hit_time:.2f}s: {metadata_source}, and the local player receives a "
        f"hit marker after firing {weapon} from {position}."
        f"{audio_clause} No body drop or kill feed is visible in the sampled frames, so visual evidence supports the shot/hit "
        "while metadata confirms the kill outcome."
    )
    pointers = [
        EvidencePointerV1(
            source="video",
            window_id=hit_marker.window_id,
            start=hit_marker.start_sec,
            end=hit_marker.end_sec,
            quote_or_observation=hit_marker.text,
        ),
        EvidencePointerV1(
            source="metadata",
            window_id="metadata",
            start=0.0,
            end=0.0,
            quote_or_observation=f"file_name: {metadata.file_name}",
        ),
    ]
    evidence = ["video", "metadata"]
    if audio_caption is not None:
        pointers.insert(
            1,
            EvidencePointerV1(
                source="audio",
                window_id=audio_caption.window_id,
                start=audio_caption.start_sec,
                end=audio_caption.end_sec,
                quote_or_observation=audio_caption.text,
            ),
        )
        evidence.insert(1, "audio")
    return {
        "moment": KeyMomentV1(
            start=hit_marker.start_sec,
            end=hit_marker.end_sec,
            description=detail,
            evidence=evidence,
            evidence_pointers=pointers,
        ),
        "short_summary": (
            f"{_ensure_sentence_end(summary.short_summary)} The clip metadata confirms a hunter kill near {hit_time:.2f}s; "
            f"the player fires {weapon} from {position}, with the hit marker as visual timing evidence."
        ),
        "detail": detail,
        "uncertainties": [
            "Kill outcome is confirmed by app-generated metadata, while body drop or kill-feed confirmation is not visible in the sampled frames."
        ],
    }


def _ensure_sentence_end(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    if stripped[-1] in ".!?":
        return stripped
    return f"{stripped}."


def _metadata_claims_kill(metadata: MetadataPayloadV1) -> bool:
    explicit_values = [
        str(metadata.user_metadata.get(key) or "")
        for key in ("known_outcome", "confirmed_outcome", "clip_context", "outcome")
        if isinstance(metadata.user_metadata, dict)
    ]
    text = " ".join(
        str(value or "")
        for value in (
            metadata.file_name,
            metadata.title,
            metadata.description,
            " ".join(metadata.tags or []),
            " ".join(explicit_values),
        )
    ).lower()
    return bool(re.search(r"\b(kill|killed|confirmed_hunter_kill|hunter_killed)\b", text))


def _kill_metadata_source_description(metadata: MetadataPayloadV1) -> str:
    user_metadata = metadata.user_metadata if isinstance(metadata.user_metadata, dict) else {}
    for key in ("known_outcome", "confirmed_outcome", "clip_context", "outcome"):
        value = str(user_metadata.get(key) or "").strip()
        if value and re.search(r"\b(kill|killed|confirmed_hunter_kill|hunter_killed)\b", value.lower()):
            return f"explicit clip-context metadata ({key}: {value}) marks this clip as a confirmed killed-hunter highlight"
    return "the app-generated file name marks this clip as a killed-hunter highlight"


def _nearby_death_vocalization_caption(
    captions: Sequence[AudioCaptionV1],
    start_sec: float,
    end_sec: float,
) -> AudioCaptionV1 | None:
    center = (start_sec + end_sec) / 2.0
    candidates = [caption for caption in captions if _audio_caption_mentions_death_vocalization(caption.text)]
    if not candidates:
        return None

    def score(caption: AudioCaptionV1) -> tuple[int, float, float]:
        prompt_echo = int(_looks_like_audio_prompt_echo(caption.text))
        if caption.start_sec <= center <= caption.end_sec:
            distance = 0.0
        else:
            distance = min(abs(caption.start_sec - center), abs(caption.end_sec - center))
        return prompt_echo, distance, abs((caption.start_sec + caption.end_sec) / 2.0 - center)

    best = min(candidates, key=score)
    prompt_echo, distance, _ = score(best)
    if prompt_echo and distance > 0.75:
        return None
    return best if distance <= 3.0 else None


def _audio_caption_mentions_death_vocalization(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "death scream",
            "death screams",
            "death sound",
            "pain vocalization",
            "pain cries",
            "human pain",
            "downed/death vocalization",
            "downed vocalization",
        )
    )


def _looks_like_audio_prompt_echo(text: str) -> bool:
    lowered = text.lower()
    return "return one plain-text" in lowered or "do not infer speakers" in lowered or "if a death scream" in lowered


def _kill_weapon_from_evidence(
    summary: FusedSummaryV1,
    metadata: MetadataPayloadV1,
    *,
    hit_time: float | None = None,
    weapon_resolver: Callable[[str], str | None] | None = None,
) -> str:
    qwen_ocr_weapon = _active_weapon_from_qwen_ocr(metadata, timestamp=hit_time, weapon_resolver=weapon_resolver)
    if qwen_ocr_weapon:
        return f"with {qwen_ocr_weapon}"
    hud = metadata.user_metadata.get("hud")
    if isinstance(hud, dict):
        active = str(hud.get("active_weapon") or hud.get("active_equipment") or "").strip()
        if active:
            canonical = _canonical_hunt_weapon_from_text(active, weapon_resolver=weapon_resolver)
            return f"with {canonical or active}"
    text = _summary_evidence_text(summary)
    lowered = text.lower()
    summary_weapon = _kill_weapon_from_summary_action_text(text, weapon_resolver=weapon_resolver)
    if summary_weapon:
        return f"with {summary_weapon}"
    canonical = _canonical_hunt_weapon_from_text(text, weapon_resolver=weapon_resolver)
    if canonical:
        return f"with {canonical}"
    has_revolver = "revolver" in lowered
    if has_revolver:
        return "with the visible revolver"
    weapon_match = re.search(r"(?:holding|fires?|using)\s+(?:a|an|the)?\s*([A-Z][A-Za-z0-9 .'-]{2,40})", text)
    if weapon_match:
        return f"with the visible {weapon_match.group(1).strip()}"
    return "with the visible first-person firearm"


def _kill_weapon_from_summary_action_text(
    text: str,
    *,
    weapon_resolver: Callable[[str], str | None] | None,
) -> str | None:
    patterns = [
        r"\b(?:fires?|shoots|kills?\s+[^,.;]{0,32}\s+with)\s+(?:a|an|the)?\s*([^,.;]{2,48})",
        r"\b(?:switches|swaps|changes)\s+(?:their\s+)?(?:active\s+)?weapon\s+(?:to|for)\s+(?:a|an|the)?\s*([^,.;]{2,48})",
        r"\b(?:switches|swaps|changes)\s+(?:to|for)\s+(?:a|an|the)?\s*([^,.;]{2,48})",
        r"\b(?:active weapon|weapon)\s+(?:is|to|:)\s+(?:a|an|the)?\s*([^,.;]{2,48})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidate = _clean_weapon_candidate(match.group(1))
            if not candidate:
                continue
            canonical = _canonical_hunt_weapon_from_text(candidate, weapon_resolver=weapon_resolver)
            return canonical or _weapon_display_with_article(candidate)
    return None


def _clean_weapon_candidate(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" -:()[]'\"")
    cleaned = re.sub(r"\b(?:shotgun|rifle|pistol|revolver|firearm|weapon)\b.*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\b(?:at|near|toward|while|after|before|and|with)\b.*$", "", cleaned, flags=re.IGNORECASE).strip()
    if re.search(r"\b(salve|healing|heal|item|consumable|tool|prompt|progress)\b", cleaned, flags=re.IGNORECASE):
        return ""
    if not cleaned or len(cleaned) > 40:
        return ""
    if cleaned.lower().split(" ", 1)[0] in {"enemy", "player", "hunter", "teammate", "target", "through", "from", "inside", "toward"}:
        return ""
    return cleaned


def _active_weapon_from_qwen_ocr(
    metadata: MetadataPayloadV1,
    *,
    timestamp: float | None,
    weapon_resolver: Callable[[str], str | None] | None,
) -> str | None:
    qwen_ocr = metadata.user_metadata.get("qwen_visual_ocr")
    if not isinstance(qwen_ocr, dict):
        return None
    rows = [row for row in qwen_ocr.get("equipment_timeline") or [] if isinstance(row, dict)]
    candidates: list[tuple[int, float, str]] = []
    for row in rows:
        entity_type = str(row.get("entity_type") or "").lower()
        if entity_type and entity_type != "weapon":
            continue
        name = str(row.get("entity_name") or "").strip()
        if not name:
            continue
        start = _float_or_none(row.get("start_timestamp", row.get("timestamp")))
        end = _float_or_none(row.get("end_timestamp", row.get("timestamp")))
        point = _float_or_none(row.get("timestamp"))
        if start is None:
            start = point
        if end is None:
            end = point if point is not None else start
        if timestamp is not None and start is not None and end is not None and start - 0.5 <= timestamp <= end + 0.75:
            distance = 0.0
            priority = 0
        elif timestamp is not None and end is not None and end <= timestamp:
            distance = timestamp - end
            priority = 1 if distance <= 8.0 else 3
        elif timestamp is not None and start is not None:
            distance = abs(start - timestamp)
            if start > timestamp and distance > 0.5:
                continue
            priority = 2
        else:
            distance = 0.0
            priority = 2
        canonical = _canonical_hunt_weapon_from_text(name, weapon_resolver=weapon_resolver)
        display = canonical or _weapon_display_with_article(name)
        candidates.append((priority, distance, display))
    if not candidates:
        return None
    priority, distance, display = min(candidates, key=lambda item: (item[0], item[1]))
    if priority >= 3:
        return None
    return display


def _weapon_display_with_article(name: str) -> str:
    stripped = name.strip()
    if not stripped:
        return stripped
    if stripped.lower().startswith(("the ", "a ", "an ")):
        return stripped
    return f"the {stripped}"


def _canonical_hunt_weapon_from_text(
    text: str,
    *,
    weapon_resolver: Callable[[str], str | None] | None = None,
) -> str | None:
    if weapon_resolver is None:
        return None
    try:
        resolved = weapon_resolver(text)
    except Exception:
        return None
    if not resolved:
        return None
    stripped = resolved.strip()
    if not stripped:
        return None
    if stripped.lower().startswith(("the ", "a ", "an ")):
        return stripped
    return f"the {stripped}"


def _canonicalize_key_moment_weapon_terms(moment: KeyMomentV1, canonical_weapon: str) -> KeyMomentV1:
    return moment.model_copy(
        update={
            "description": _canonicalize_weapon_terms(moment.description, canonical_weapon),
            "evidence_pointers": [
                pointer.model_copy(
                    update={"quote_or_observation": _canonicalize_weapon_terms(pointer.quote_or_observation, canonical_weapon)}
                )
                for pointer in moment.evidence_pointers
            ],
        }
    )


def _normalize_key_moment_weapon_type_mismatches(moment: KeyMomentV1) -> KeyMomentV1:
    return moment.model_copy(
        update={
            "description": _normalize_weapon_type_mismatches(moment.description),
            "evidence_pointers": [
                pointer.model_copy(update={"quote_or_observation": _normalize_weapon_type_mismatches(pointer.quote_or_observation)})
                for pointer in moment.evidence_pointers
            ],
        }
    )


def _normalize_weapon_type_mismatches(text: str) -> str:
    if not text:
        return text
    replacements = [
        (r"\bAuto-5\s+(?:revolver|pistol|rifle|weapon|firearm)\b", "Auto-5"),
        (r"\bMosin Obrez\s+\(Rougarou skin\)\s+(?:rifle|pistol|weapon|firearm)\b", "Mosin Obrez (Rougarou skin)"),
        (r"\bthe Mosin Obrez\s+\(Rougarou skin\)\s+(?:rifle|pistol|weapon|firearm)\b", "the Mosin Obrez (Rougarou skin)"),
    ]
    updated = text
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)
    return updated


def _canonicalize_weapon_terms(text: str, canonical_weapon: str) -> str:
    if not text:
        return text
    weapon_name = canonical_weapon.removeprefix("the ")
    replacements = [
        (r"\bRougarou-marked revolver/firearm\b", weapon_name),
        (r"\bRougarou-marked revolver\b", weapon_name),
        (r"\bRougarou-marked firearm\b", weapon_name),
    ]
    skin_match = re.search(r"\(([^()]+?) skin\)", weapon_name, flags=re.IGNORECASE)
    if skin_match:
        skin_name = skin_match.group(1).strip()
        if skin_name:
            escaped_skin = re.escape(skin_name)
            base_weapon_name = weapon_name.split("(", 1)[0].strip()
            escaped_base_weapon = re.escape(base_weapon_name)
            replacements.extend(
                [
                    (
                        rf"\b(?:a|an|the)?\s*{escaped_base_weapon}(?:\s+(?:rifle|shotgun|pistol|revolver|firearm|weapon))?"
                        rf"\s+with\s+(?:a|an|the)?\s*['\"]?{escaped_skin}['\"]?\s+skin\b",
                        f"the {weapon_name}",
                    ),
                    (
                        rf"\b{escaped_base_weapon}\s+\({escaped_skin}\s+skin\)\s+(?:rifle|shotgun|pistol|revolver|firearm|weapon)\b",
                        weapon_name,
                    ),
                    (
                        rf"\b(?:a|an|the)?\s*(weapon|firearm|gun)\s+with\s+(?:a|an|the)?\s*['\"]?{escaped_skin}['\"]?\s+skin\b",
                        f"the {weapon_name}",
                    ),
                    (
                        rf"\b(?:a|an|the)?\s*(weapon|firearm|gun)\s+with\s+(?:a|an|the)?\s*['\"]?{escaped_skin}['\"]?\s+skin\s+visible\b",
                        f"the {weapon_name} visible",
                    ),
                    (
                        rf"\b['\"]?{escaped_skin}['\"]?\s+skin\s+visible\s+on\s+(?:the\s+)?HUD\b",
                        f"{weapon_name} visible on the HUD",
                    ),
                    (
                        rf"\b(holding|holds|using|uses|carrying|wielding|firing|fires|switches to|equipped with)\s+"
                        rf"(?:a|an|the)?\s*{escaped_skin}\b",
                        rf"\1 the {weapon_name}",
                    ),
                    (
                        rf"\b(active weapon|active equipment|visible weapon|weapon|firearm|gun):?\s+{escaped_skin}\b",
                        rf"\1: {weapon_name}",
                    ),
                ]
            )
    updated = text
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)
    return updated


def _sanitize_key_moment_hunter_identity_claims(moment: KeyMomentV1) -> KeyMomentV1:
    return moment.model_copy(
        update={
            "description": _sanitize_hunter_identity_claims(moment.description),
            "evidence_pointers": [
                pointer.model_copy(
                    update={"quote_or_observation": _sanitize_hunter_identity_claims(pointer.quote_or_observation)}
                )
                for pointer in moment.evidence_pointers
            ],
        }
    )


def _sanitize_summary_raw_payload_hunter_identity_claims(
    raw_payload: dict[str, Any],
    *,
    canonical_weapon: str | None,
) -> dict[str, Any]:
    output = dict(raw_payload)
    raw_observations = output.get("visual_observations")
    if isinstance(raw_observations, list):
        observations = []
        for item in raw_observations:
            if not isinstance(item, dict):
                observations.append(item)
                continue
            updated = dict(item)
            for key in ("text", "observation"):
                if key in updated:
                    text = str(updated[key])
                    if canonical_weapon:
                        text = _canonicalize_weapon_terms(text, canonical_weapon)
                    updated[key] = _sanitize_hunter_identity_claims(text)
            observations.append(updated)
        output["visual_observations"] = observations
    return output


def _equipment_timeline_conflict_context(
    metadata: MetadataPayloadV1 | None,
    *,
    weapon_resolver: Callable[[str], str | None] | None,
) -> dict[str, Any] | None:
    if metadata is None:
        return None
    rows: list[tuple[float, str]] = []
    for source in (metadata.user_metadata.get("hud"), metadata.user_metadata.get("qwen_visual_ocr")):
        if not isinstance(source, dict):
            continue
        for row in source.get("equipment_timeline") or []:
            if not isinstance(row, dict):
                continue
            entity_type = str(row.get("entity_type") or "weapon").lower()
            if entity_type != "weapon":
                continue
            name = str(row.get("entity_name") or "").strip()
            if not name:
                continue
            timestamp = _float_or_none(row.get("timestamp", row.get("start_timestamp")))
            if timestamp is None:
                continue
            display = _canonical_hunt_weapon_from_text(name, weapon_resolver=weapon_resolver) or _weapon_display_with_article(name)
            rows.append((float(timestamp), display.removeprefix("the ")))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda item: item[0])
    latest_time, latest_name = rows[-1]
    previous_names = []
    for _, name in rows[:-1]:
        if name.lower() != latest_name.lower() and name not in previous_names:
            previous_names.append(name)
    if not previous_names:
        return None
    return {"latest_time": latest_time, "latest_name": latest_name, "previous_names": previous_names}


def _switch_back_patterns(previous_name: str) -> list[str]:
    names = [previous_name.strip()]
    base = previous_name.split("(", 1)[0].strip()
    if base and base.lower() != previous_name.lower():
        names.append(base)
    patterns = []
    for name in names:
        escaped = re.escape(name)
        patterns.append(rf"\bswitch(?:es|ed|ing)?\s+back\s+to\s+(?:the\s+)?{escaped}(?=\W|$)")
        patterns.append(rf"\b(?:the\s+)?(?:player'?s\s+)?weapon\s+is\s+back\s+to\s+(?:the\s+)?{escaped}(?=\W|$)")
        patterns.append(rf"\b(?:the\s+)?player\s+is\s+back\s+on\s+(?:the\s+)?{escaped}(?=\W|$)")
    return patterns


def _sanitize_hunter_identity_tags(
    tags: Sequence[str],
    *,
    weapon_resolver: Callable[[str], str | None] | None,
) -> list[str]:
    output: list[str] = []
    for tag in tags:
        cleaned = str(tag).strip()
        if not cleaned:
            continue
        resolved_weapon_skin = _canonical_hunt_weapon_from_text(cleaned, weapon_resolver=weapon_resolver)
        if resolved_weapon_skin:
            base_weapon = resolved_weapon_skin.removeprefix("the ").split("(", 1)[0].strip()
            if base_weapon:
                output.append(_tag_slug(base_weapon))
            continue
        output.append(cleaned)
    return list(dict.fromkeys(output))


def _tag_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _sanitize_hunter_identity_claims(text: str) -> str:
    if not text:
        return text
    updated = text
    actor = (
        r"(?:teammates?|team mates?|partners?|enemy hunters?|enemies|opponents?|characters?|"
        r"second teammate|first teammate|second character|first character)"
    )
    updated = re.sub(
        rf"\b(?P<actor>{actor})\s*,?\s*(?:both|all)\s+[A-Z][A-Za-z0-9' -]{{1,40}}\b,?",
        lambda match: match.group("actor"),
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        rf"\b(?P<actor>(?:a|an|the|one|another|second|first)?\s*{actor})\s*,?\s*"
        r"(?:identified|recognized|label(?:ed|led)|named)\s+as\s+(?:a\s+|an\s+|the\s+)?"
        r"[A-Z][A-Za-z0-9' -]{1,40}"
        r"(?:\s+by\s+(?:the\s+)?(?:name ?tag|label|HUD text|metadata|visual appearance|clothing|silhouette))?,?",
        lambda match: match.group("actor"),
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        rf"\b(?P<actor>(?:a|an|the|one|another|second|first)?\s*{actor})\s*\((?![^)]*\bskin\b)[^)]+\)",
        lambda match: match.group("actor"),
        updated,
        flags=re.IGNORECASE,
    )
    updated = re.sub(
        r"\b(?:also\s+)?identified\s+as\s+(?:a\s+|an\s+|the\s+)?[A-Z][A-Za-z0-9' -]{1,40}\s+(?=(?:but\s+)?wearing|with\s+)",
        "",
        updated,
    )
    updated = re.sub(r"\s+,", ",", updated)
    updated = re.sub(r",\s*,", ",", updated)
    updated = re.sub(r"\s{2,}", " ", updated)
    return updated.strip()


def _kill_position_from_evidence(summary: FusedSummaryV1) -> str:
    text = _summary_evidence_text(summary).lower()
    if "inside" in text and "window" in text and ("wooden" in text or "room" in text or "building" in text):
        return "inside the wooden/windowed room, firing through the window toward the outdoor walkway"
    if "window" in text:
        return "the window position, firing through the window"
    if "inside" in text:
        return "the indoor first-person position shown in the clip"
    return "the visible first-person firing position"


def _summary_evidence_text(summary: FusedSummaryV1) -> str:
    raw_observations = summary.raw_payload.get("visual_observations")
    raw_text = ""
    if isinstance(raw_observations, list):
        raw_text = " ".join(str(item.get("text") or item.get("observation") or "") for item in raw_observations if isinstance(item, dict))
    moments = " ".join(moment.description for moment in summary.key_moments)
    return f"{summary.short_summary} {summary.detailed_summary} {moments} {raw_text}"


def _summary_already_reports_confirmed_kill(summary: FusedSummaryV1) -> bool:
    text = f"{summary.short_summary} {summary.detailed_summary} " + " ".join(moment.description for moment in summary.key_moments)
    return bool(re.search(r"\b(metadata-confirmed|confirmed)\s+(hunter\s+)?kill", text.lower()))


def _summary_tags_with_kill(tags: Sequence[str], include_kill: bool) -> list[str]:
    output = list(tags)
    if include_kill and "confirmed-kill" not in output:
        output.append("confirmed-kill")
    return output


def _tags_from_evidence(lines: Sequence[str]) -> list[str]:
    stop = {"the", "and", "for", "with", "from", "that", "this", "between", "metadata", "video", "speech", "audio"}
    counts: dict[str, int] = {}
    for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", " ".join(lines).lower()):
        if token in stop:
            continue
        counts[token] = counts.get(token, 0) + 1
    return [token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]]


def parse_summary_json(raw: str, *, manifest: ClipManifestV1, model_id: str) -> FusedSummaryV1:
    data = _loads_json_object(raw)
    moments = []
    for item in data.get("key_moments", []):
        evidence, pointers = _coerce_key_moment_evidence(item, manifest)
        moments.append(
            KeyMomentV1(
                start=_clip_timestamp(float(item["start"]), manifest.duration_sec),
                end=_clip_timestamp(float(item["end"]), manifest.duration_sec),
                description=str(item["description"]),
                evidence=evidence,
                evidence_pointers=pointers,
            )
        )
    summary = FusedSummaryV1(
        clip_id=manifest.clip_id,
        file_name=manifest.file_name,
        model_id=model_id,
        title=str(data.get("title") or manifest.file_name),
        short_summary=str(data.get("short_summary") or ""),
        detailed_summary=str(data.get("detailed_summary") or ""),
        key_moments=moments,
        tags=[str(item) for item in data.get("tags", [])],
        detected_language=data.get("detected_language"),
        uncertainties=[str(item) for item in data.get("uncertainties", [])],
        raw_payload=data,
    )
    summary.validate_evidence_bounds(manifest.duration_sec)
    return summary


def _parse_qwen_visual_ocr_json(
    raw: str,
    *,
    manifest: ClipManifestV1,
    model_id: str,
    media_windows: Sequence[MediaWindowV1],
    weapon_resolver: Callable[[str], str | None] | None,
) -> dict[str, Any]:
    data = _loads_json_object(raw)
    windows_by_id = {window.window_id: window for window in media_windows}
    observations: list[dict[str, Any]] = []
    for item in data.get("ocr_observations", []):
        if not isinstance(item, dict):
            continue
        text = _clean_qwen_ocr_text(str(item.get("text") or ""))
        raw_text = _clean_qwen_ocr_text(str(item.get("raw_text") or ""))
        if not text and raw_text:
            text = raw_text
        window_id = str(item.get("window_id") or (media_windows[0].window_id if media_windows else "window_full_clip"))
        window = windows_by_id.get(window_id)
        start = _clip_timestamp(_float_or_none(item.get("start")) or (window.start_sec if window else 0.0), manifest.duration_sec)
        end = _clip_timestamp(_float_or_none(item.get("end")) or (window.end_sec if window else start), manifest.duration_sec)
        resolved_equipment = _qwen_ocr_resolved_equipment(item, text, weapon_resolver)
        if not resolved_equipment:
            continue
        canonical_weapon = _canonical_weapon_from_resolved_equipment(resolved_equipment)
        if not text and resolved_equipment:
            text = "Visible HUD equipment: " + ", ".join(row["display_name"] for row in resolved_equipment[:3])
        if not text:
            continue
        if canonical_weapon:
            text = _canonicalize_weapon_terms(text, canonical_weapon)
            raw_text = _canonicalize_weapon_terms(raw_text, canonical_weapon) if raw_text else raw_text
        observations.append(
            {
                "schema_version": "1.0",
                "window_id": window_id,
                "start": round(float(start), 3),
                "end": round(float(max(start, end)), 3),
                "text": text,
                "source": "video",
                "source_area": item.get("source_area"),
                "raw_text": raw_text or None,
                "resolved_equipment": resolved_equipment,
                "uncertainties": [str(value) for value in item.get("uncertainties", [])]
                if isinstance(item.get("uncertainties"), list)
                else [],
            }
        )
    equipment_timeline = _qwen_ocr_equipment_timeline(
        data.get("equipment_timeline", []),
        observations=observations,
        manifest=manifest,
        weapon_resolver=weapon_resolver,
    )
    uncertainties = [
        str(value)
        for value in data.get("uncertainties", [])
        if str(value).strip()
    ] if isinstance(data.get("uncertainties"), list) else []
    return {
        "schema_version": "1.0",
        "source": "qwen35_visual_ocr",
        "model_id": model_id,
        "observations": observations[:16],
        "equipment_timeline": equipment_timeline[:12],
        "uncertainties": uncertainties,
        "raw_response": data,
    }


def _qwen_ocr_resolved_equipment(
    item: dict[str, Any],
    text: str,
    weapon_resolver: Callable[[str], str | None] | None,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    raw_items = item.get("resolved_equipment")
    if isinstance(raw_items, list):
        for row in raw_items:
            if not isinstance(row, dict):
                continue
            raw_name = str(row.get("raw_name") or "").strip()
            display_name = str(row.get("display_name") or "").strip()
            entity_type = str(row.get("entity_type") or "weapon").strip() or "weapon"
            if not raw_name and display_name:
                raw_name = display_name
            if raw_name and weapon_resolver is not None:
                try:
                    display_name = weapon_resolver(raw_name) or display_name
                except Exception:
                    pass
            if raw_name and display_name:
                output.append({"raw_name": raw_name, "display_name": display_name, "entity_type": entity_type})
    if weapon_resolver is not None:
        try:
            resolved = weapon_resolver(text)
        except Exception:
            resolved = None
        if resolved:
            raw_name = _raw_weapon_name_from_resolved_text(text, resolved)
            output.append({"raw_name": raw_name, "display_name": resolved, "entity_type": "weapon"})
    deduped: dict[tuple[str, str], dict[str, str]] = {}
    for row in output:
        key = (row["raw_name"].lower(), row["display_name"].lower())
        deduped[key] = row
    return list(deduped.values())[:8]


def _clean_qwen_ocr_text(text: str, *, max_chars: int = 240) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ""
    parts = [part.strip() for part in re.split(r"[,;|]", cleaned) if part.strip()]
    if len(parts) >= 12:
        compact_parts: list[str] = []
        counts: dict[str, int] = {}
        for part in parts:
            normalized = part.lower()
            if re.fullmatch(r"\d+(?:/\d+)?x?", normalized) or re.fullmatch(r"\d{2,}", normalized):
                continue
            counts[normalized] = counts.get(normalized, 0) + 1
            if counts[normalized] > 2:
                continue
            compact_parts.append(part)
            if len(", ".join(compact_parts)) >= max_chars:
                break
        cleaned = ", ".join(compact_parts).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 3].rstrip(" ,.;") + "..."
    if _looks_like_numeric_ocr_noise(cleaned):
        return ""
    return cleaned


def _looks_like_numeric_ocr_noise(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z]+|\d+(?:/\d+)?x?", text)
    if not tokens:
        return False
    numeric = sum(1 for token in tokens if re.fullmatch(r"\d+(?:/\d+)?x?", token.lower()))
    alpha = sum(1 for token in tokens if re.fullmatch(r"[A-Za-z]+", token))
    return numeric >= 8 and numeric > alpha * 3


def _raw_weapon_name_from_resolved_text(text: str, resolved: str) -> str:
    skin_match = re.search(r"\(([^()]+?) skin\)", resolved, flags=re.IGNORECASE)
    if skin_match and re.search(rf"\b{re.escape(skin_match.group(1))}\b", text, flags=re.IGNORECASE):
        return skin_match.group(1)
    return text.strip()[:80] or resolved


def _canonical_weapon_from_resolved_equipment(rows: Sequence[dict[str, str]]) -> str | None:
    for row in rows:
        if str(row.get("entity_type") or "").lower() != "weapon":
            continue
        display_name = str(row.get("display_name") or "").strip()
        if display_name:
            return display_name if display_name.lower().startswith(("the ", "a ", "an ")) else f"the {display_name}"
    return None


def _qwen_ocr_equipment_timeline(
    raw_items: Any,
    *,
    observations: Sequence[dict[str, Any]],
    manifest: ClipManifestV1,
    weapon_resolver: Callable[[str], str | None] | None,
) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("entity_name") or "").strip()
            if not name:
                continue
            entity_type = str(item.get("entity_type") or "weapon").strip().lower() or "weapon"
            if entity_type not in {"weapon", "tool", "consumable"}:
                continue
            if weapon_resolver is not None:
                try:
                    name = weapon_resolver(name) or name
                except Exception:
                    pass
            timestamp = _clip_timestamp(_float_or_none(item.get("timestamp")) or 0.0, manifest.duration_sec)
            timeline.append(
                {
                    "timestamp": round(float(timestamp), 3),
                    "start_timestamp": round(float(timestamp), 3),
                    "end_timestamp": round(float(timestamp), 3),
                    "entity_name": name,
                    "entity_type": entity_type,
                    "source": "qwen35_visual_ocr",
                    "confidence": _float_or_none(item.get("confidence")),
                }
            )
    for observation in observations:
        for row in observation.get("resolved_equipment") or []:
            if not isinstance(row, dict):
                continue
            display_name = str(row.get("display_name") or "").strip()
            if not display_name:
                continue
            entity_type = str(row.get("entity_type") or "weapon").strip().lower() or "weapon"
            if entity_type not in {"weapon", "tool", "consumable"}:
                continue
            timestamp = _clip_timestamp(_float_or_none(observation.get("start")) or 0.0, manifest.duration_sec)
            timeline.append(
                {
                    "timestamp": round(float(timestamp), 3),
                    "start_timestamp": round(float(timestamp), 3),
                    "end_timestamp": round(float(_float_or_none(observation.get("end")) or timestamp), 3),
                    "entity_name": display_name,
                    "entity_type": entity_type,
                    "source": "qwen35_visual_ocr",
                    "confidence": None,
                }
            )
    return _compact_equipment_timeline(timeline, max_items=12)


def _coerce_key_moment_evidence(item: dict[str, Any], manifest: ClipManifestV1) -> tuple[list[str], list[EvidencePointerV1]]:
    raw_evidence = item.get("evidence", [])
    raw_pointers = item.get("evidence_pointers", [])
    pointer_payloads: list[dict[str, Any]] = []
    evidence_sources: list[str] = []
    valid_sources = {"video", "speech", "audio", "metadata"}

    if isinstance(raw_evidence, str):
        source = raw_evidence.strip().lower()
        if source in valid_sources:
            evidence_sources.append(source)
    elif isinstance(raw_evidence, list):
        for value in raw_evidence:
            if isinstance(value, str):
                source = value.strip().lower()
                if source in valid_sources:
                    evidence_sources.append(source)
            elif isinstance(value, dict):
                pointer_payloads.append(value)
                source = value.get("source")
                if source is not None:
                    normalized = str(source).strip().lower()
                    if normalized in valid_sources:
                        evidence_sources.append(normalized)

    if isinstance(raw_pointers, list):
        pointer_payloads.extend(pointer for pointer in raw_pointers if isinstance(pointer, dict))

    if not pointer_payloads and evidence_sources:
        pointer_payloads.append(
            {
                "source": "video" if "video" in evidence_sources else evidence_sources[0],
                "window_id": item.get("window_id") or "window_full_clip",
                "start": item.get("start"),
                "end": item.get("end"),
                "quote_or_observation": item.get("description"),
            }
        )

    pointers: list[EvidencePointerV1] = []
    for pointer in pointer_payloads:
        source = str(pointer.get("source") or "").strip().lower()
        if source not in valid_sources:
            continue
        start_value = pointer.get("start", pointer.get("start_sec", item.get("start")))
        end_value = pointer.get("end", pointer.get("end_sec", item.get("end", start_value)))
        quote = (
            pointer.get("quote_or_observation")
            or pointer.get("observation")
            or pointer.get("text")
            or item.get("description")
        )
        if start_value is None or end_value is None or quote is None:
            continue
        pointers.append(
            EvidencePointerV1(
                source=source,
                window_id=pointer.get("window_id"),
                start=_clip_timestamp(float(start_value), manifest.duration_sec),
                end=_clip_timestamp(float(end_value), manifest.duration_sec),
                quote_or_observation=str(quote),
            )
        )
    if not evidence_sources:
        evidence_sources = [str(pointer.source) for pointer in pointers]
    return list(dict.fromkeys(evidence_sources)), pointers


def _clip_timestamp(value: float, duration_sec: float) -> float:
    tolerance = max(0.05, min(0.25, duration_sec * 0.02))
    if value < 0 and abs(value) <= tolerance:
        return 0.0
    if duration_sec >= 0 and value > duration_sec and value - duration_sec <= tolerance:
        return duration_sec
    return value


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _loads_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        extracted = _first_complete_json_object_text(text)
        if extracted is None:
            raise
        data = json.loads(extracted)
    if not isinstance(data, dict):
        raise TypeError("summary JSON must be an object")
    return data


def _first_complete_json_object_text(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for offset, char in enumerate(text[start:]):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : start + offset + 1].strip()
    return None


def _raw_preview(raw: str, *, limit: int = 500) -> str:
    collapsed = " ".join(str(raw or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."
