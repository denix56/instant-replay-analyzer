from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from ..llm.llama_client import LlamaClient
from ..llm.prompts import CLIP_SUMMARY_SYSTEM_PROMPT, build_summary_prompt
from ..llm.schemas import SummaryResult


@dataclass(frozen=True)
class SummarizationConfig:
    max_summary_words: int = 95
    use_llm: bool = False
    mock_fallback: bool = True


class ClipSummarizer:
    def __init__(
        self,
        config: Optional[SummarizationConfig] = None,
        llama_client: Optional[LlamaClient] = None,
    ) -> None:
        self.config = config or SummarizationConfig()
        self._llama_client = llama_client or LlamaClient()

    def summarize(
        self,
        transcript: str,
        *,
        ocr_text: str = "",
        audio_events: Iterable[str] = (),
        filename: str = "",
        group_name: str = "",
        tags: Sequence[str] = (),
        av_semantic_hints: Sequence[str] = (),
        duration_seconds: float | None = None,
        segment_count: int | None = None,
        modality_counts: dict[str, int] | None = None,
    ) -> SummaryResult:
        if self.config.use_llm:
            try:
                raw = self._llama_client.complete(
                    build_summary_prompt(
                        _context_text(
                            transcript=transcript,
                            filename=filename,
                            group_name=group_name,
                            tags=tags,
                            av_semantic_hints=av_semantic_hints,
                            duration_seconds=duration_seconds,
                            segment_count=segment_count,
                            modality_counts=modality_counts,
                        ),
                        ocr_text,
                        audio_events,
                    ),
                    system=CLIP_SUMMARY_SYSTEM_PROMPT,
                    temperature=0.0,
                )
                parsed = json.loads(raw)
                return SummaryResult(
                    title=str(parsed.get("title") or "Gameplay clip"),
                    summary=str(parsed.get("summary") or ""),
                    key_moments=[str(item) for item in parsed.get("key_moments", [])],
                    engine="llama",
                )
            except Exception:
                if not self.config.mock_fallback:
                    raise
        return self._extractive_summary(
            transcript,
            ocr_text,
            audio_events,
            filename=filename,
            group_name=group_name,
            tags=tags,
            av_semantic_hints=av_semantic_hints,
            duration_seconds=duration_seconds,
            segment_count=segment_count,
            modality_counts=modality_counts,
        )

    def _extractive_summary(
        self,
        transcript: str,
        ocr_text: str,
        audio_events: Iterable[str],
        *,
        filename: str = "",
        group_name: str = "",
        tags: Sequence[str] = (),
        av_semantic_hints: Sequence[str] = (),
        duration_seconds: float | None = None,
        segment_count: int | None = None,
        modality_counts: dict[str, int] | None = None,
    ) -> SummaryResult:
        source = " ".join(part.strip() for part in [transcript, ocr_text] if part.strip())
        cleaned_filename = _humanize_filename(filename)
        selected_tags = [tag for tag in tags if tag][:8]
        selected_hints = _dedupe([hint for hint in av_semantic_hints if hint])[:5]
        duration_text = _duration_text(duration_seconds)
        modalities = _modality_text(modality_counts)
        events = [event for event in audio_events if event]
        structured = _structured_gameplay_summary(
            selected_tags,
            selected_hints,
            duration_text=duration_text,
            filename=cleaned_filename,
            group_name=group_name,
            evidence_text=source,
        )
        if structured:
            summary = structured
            source_for_title = source or cleaned_filename or "Gameplay clip"
            title = _title_from_text(source_for_title)
            key_moments = _sentences(source)[:3]
            if selected_tags:
                key_moments.append(f"Tags: {', '.join(selected_tags[:6])}")
            if selected_hints:
                key_moments.append(f"Detected context: {', '.join(selected_hints[:4])}")
            if segment_count:
                key_moments.append(f"Indexed into {segment_count} {_dominant_modality_label(modality_counts)} segments")
            key_moments.extend(f"Audio: {event}" for event in events)
            return SummaryResult(title=title, summary=summary, key_moments=key_moments[:5], engine="automatic")

        if source:
            spoken = _compact_text(source, max_words=28)
            action = _action_phrase(selected_tags)
            embedding_phrase = _hint_phrase(selected_hints)
            position_movement = _position_movement_phrase(selected_tags, selected_hints, source)
            summary_parts = [
                "Gameplay clip",
                f"from {group_name}" if group_name else "",
                f"around {action}" if action else "",
                f"with transcript cues: {spoken}",
                f"Audio-video embeddings suggest {embedding_phrase}" if embedding_phrase else "",
                f"Position/movement: {position_movement}" if position_movement else "",
            ]
            summary = _sentence(" ".join(part for part in summary_parts if part))
        else:
            basis = cleaned_filename or "an untitled capture"
            action = _action_phrase(selected_tags)
            embedding_phrase = _hint_phrase(selected_hints)
            position_movement = _position_movement_phrase(selected_tags, selected_hints, source)
            summary_parts = [
                "Gameplay clip",
                f"from {group_name}" if group_name else "",
                f"named '{basis}'",
                f"likely involving {action}" if action else "",
                f"with audio-video embedding cues for {embedding_phrase}" if embedding_phrase else "",
                f"Position/movement: {position_movement}" if position_movement else "",
                modalities,
            ]
            summary = _sentence(" ".join(part for part in summary_parts if part))

        words = summary.split()
        if len(words) > self.config.max_summary_words:
            summary = " ".join(words[: self.config.max_summary_words]).rstrip(".,;") + "..."

        title_source = source or cleaned_filename or "Gameplay clip"
        title = _title_from_text(title_source)
        key_moments = _sentences(source)[:3]
        if selected_tags:
            key_moments.append(f"Tags: {', '.join(selected_tags[:6])}")
        if selected_hints:
            key_moments.append(f"Audio-video embedding hints: {', '.join(selected_hints[:4])}")
        if segment_count:
            key_moments.append(f"Indexed into {segment_count} {_dominant_modality_label(modality_counts)} segments")
        key_moments.extend(f"Audio: {event}" for event in events)
        return SummaryResult(title=title, summary=summary, key_moments=key_moments[:5], engine="automatic")


def summarize_clip(transcript: str, **kwargs: object) -> SummaryResult:
    return ClipSummarizer().summarize(transcript, **kwargs)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _title_from_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "Gameplay Clip"
    return " ".join(cleaned.split()[:8]).rstrip(".,!?").title()


def _context_text(
    *,
    transcript: str,
    filename: str,
    group_name: str,
    tags: Sequence[str],
    av_semantic_hints: Sequence[str],
    duration_seconds: float | None,
    segment_count: int | None,
    modality_counts: dict[str, int] | None,
) -> str:
    parts = [
        f"Filename: {filename}" if filename else "",
        f"Group: {group_name}" if group_name else "",
        f"Duration: {_duration_text(duration_seconds)}" if duration_seconds else "",
        f"Segments: {segment_count}" if segment_count else "",
        f"Modalities: {_modality_text(modality_counts)}" if modality_counts else "",
        f"Tags: {', '.join(tags)}" if tags else "",
        f"Audio-video embedding hints: {', '.join(av_semantic_hints)}" if av_semantic_hints else "",
        f"Transcript: {transcript}" if transcript else "Transcript: [none]",
    ]
    return "\n".join(part for part in parts if part)


def _humanize_filename(filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = re.sub(r"\.[a-zA-Z0-9]{2,5}$", "", stem)
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\b\d{4}[.\-_ ]\d{1,2}[.\-_ ]\d{1,2}\b", "", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem


def _compact_text(text: str, *, max_words: int) -> str:
    words = re.sub(r"\s+", " ", text).strip().split()
    compact = " ".join(words[:max_words])
    if len(words) > max_words:
        compact += "..."
    return compact


def _duration_text(duration_seconds: float | None) -> str:
    if duration_seconds is None or duration_seconds <= 0:
        return ""
    rounded = int(round(duration_seconds))
    if rounded < 60:
        return f"{rounded}-second"
    minutes, seconds = divmod(rounded, 60)
    return f"{minutes}m {seconds:02d}s"


def _action_phrase(tags: Sequence[str]) -> str:
    priority = [
        "death",
        "kill",
        "boss",
        "lair",
        "shotgun",
        "rifle",
        "teammate comms",
        "footsteps",
        "gunshots",
        "explosion",
        "dogs",
        "crows",
        "window",
        "extract",
        "funny",
        "clutch",
    ]
    selected = [tag for tag in priority if tag in set(tags)]
    if not selected:
        selected = list(tags[:3])
    if not selected:
        return ""
    if len(selected) == 1:
        return selected[0]
    return ", ".join(selected[:-1]) + f" and {selected[-1]}"


def _hint_phrase(hints: Sequence[str]) -> str:
    selected = [hint for hint in hints if hint]
    if not selected:
        return ""
    if len(selected) == 1:
        return selected[0]
    if len(selected) == 2:
        return f"{selected[0]} and {selected[1]}"
    return ", ".join(selected[:-1]) + f" and {selected[-1]}"


def _structured_gameplay_summary(
    tags: Sequence[str],
    hints: Sequence[str],
    *,
    duration_text: str,
    filename: str,
    group_name: str,
    evidence_text: str = "",
) -> str:
    hint_text = " | ".join(hints)
    active, loadout = _parse_hud_hint(hint_text)
    death_status, killed_with, killer = _parse_death_hint(hint_text)
    if not active and not loadout and not death_status and not killed_with and not killer:
        return ""
    context = _context_phrase(tags, hints)
    action = _player_action_phrase(tags, hints, active)
    position_movement = _position_movement_phrase(tags, hints, evidence_text)
    notable = _notable_phrase(tags, hints)
    parts = []
    intro = f"This happens in {context}" if context else "Gameplay moment"
    parts.append(intro)
    if action:
        parts.append(action)
    if position_movement:
        parts.append(f"Position/movement: {position_movement}")
    weapon_parts = []
    if active:
        weapon_parts.append(f"Current weapon/item: {active}")
    if loadout:
        weapon_parts.append(f"visible loadout: {', '.join(loadout)}")
    if weapon_parts:
        parts.append("; ".join(weapon_parts))
    if death_status or killed_with or killer:
        outcome = f"You were {death_status}" if death_status else "The clip ends with a death/down"
        if killed_with:
            outcome += f" by {killed_with}"
        if killer:
            outcome += f" from {killer}"
        parts.append(outcome)
    if notable:
        parts.append(f"Notable detail: {notable}")
    if len(parts) < 2:
        return ""
    return _sentence(". ".join(parts))


def _parse_hud_hint(text: str) -> tuple[str, list[str]]:
    active = ""
    loadout: list[str] = []
    match = re.search(r"active weapon or item:\s*([^;|.]+)", text, flags=re.IGNORECASE)
    if match:
        active = match.group(1).strip()
    match = re.search(r"detected loadout:\s*([^|.]+)", text, flags=re.IGNORECASE)
    if match:
        loadout = [part.strip() for part in match.group(1).split(",") if part.strip()]
    return active, loadout


def _parse_death_hint(text: str) -> tuple[str, str, str]:
    status = ""
    killed_with = ""
    killer = ""
    match = re.search(r"player was\s+([^;|.]+)", text, flags=re.IGNORECASE)
    if match:
        status = match.group(1).strip()
    match = re.search(r"killed with:\s*([^;|.]+)", text, flags=re.IGNORECASE)
    if match:
        killed_with = match.group(1).strip()
    match = re.search(r"killer:\s*([^;|.]+)", text, flags=re.IGNORECASE)
    if match:
        killer = match.group(1).strip()
    return status, killed_with, killer


def _context_phrase(tags: Sequence[str], hints: Sequence[str]) -> str:
    haystack = " ".join([*tags, *hints]).lower()
    if "dark indoor fight" in haystack:
        return "a dark indoor fight"
    if "indoors" in haystack or "indoor" in haystack:
        return "an indoor fight"
    if "window" in haystack:
        return "a fight around a window angle"
    if "night" in haystack:
        return "a night fight"
    return ""


def _player_action_phrase(tags: Sequence[str], hints: Sequence[str], active: str) -> str:
    haystack = " ".join([*tags, *hints]).lower()
    if active and "shotgun" in haystack:
        return f"You were playing a close-range shotgun position with the {active}"
    if active:
        return f"You were using the {active}"
    if "enemy visible in window" in haystack:
        return "You were watching an enemy near a window"
    if "gunfight" in haystack:
        return "You were in a gunfight"
    return ""


def _position_movement_phrase(
    tags: Sequence[str],
    hints: Sequence[str],
    evidence_text: str = "",
) -> str:
    haystack = " ".join([*tags, *hints, evidence_text]).lower()
    details: list[str] = []
    if "boss lair" in haystack:
        details.append("the fight centered near a boss lair")
    elif "extraction fight" in haystack or "extract" in haystack:
        details.append("the fight was near extraction")
    elif "dark indoor fight" in haystack:
        details.append("you were inside or just inside a dark compound interior")
    elif "indoors" in haystack or "indoor" in haystack:
        details.append("you were fighting from indoor cover")
    elif "night" in haystack:
        details.append("visibility was low in a night fight")
    if "enemy visible in window" in haystack or "window" in haystack:
        details.append("the enemy pressure was around a window angle")
    if "enemy pushing through doorway" in haystack:
        details.append("an enemy appeared to push through a doorway")
    if "enemy crossing open ground" in haystack:
        details.append("an enemy was likely crossing exposed ground")
    if "footsteps before death" in haystack or "enemy footsteps approaching" in haystack:
        details.append("enemy footsteps or movement closed in before the outcome")
    if "player rotating" in haystack or "rotation" in haystack or "rotating" in haystack:
        details.append("you were rotating through the area")
    if re.search(r"\b(i'?m|i am|we are|we'?re|i was|we were)\s+(holding|watching)\b", haystack):
        details.append("you were holding or watching an angle")
    if re.search(r"\b(i'?m|i am|we are|we'?re|i was|we were)\s+(pushing|moving in)\b", haystack):
        details.append("you were pushing or closing distance")
    if re.search(r"\b(backing up|falling back|repositioning)\b", haystack):
        details.append("you were backing up or repositioning")
    if re.search(r"\b(peeking|peeked|wide peek)\b", haystack):
        details.append("you were peeking an angle")
    if "close-range push" in haystack or "shotgun fight" in haystack:
        details.append("the engagement stayed close range")
    if "reload under pressure" in haystack:
        details.append("you had reload or reposition pressure")
    if "revive attempt" in haystack:
        details.append("movement centered on a revive attempt")
    details.extend(_enemy_position_details(tags, hints, evidence_text))
    details.extend(_teammate_details(tags, hints, evidence_text))
    return "; ".join(_dedupe(details)[:5])


def _enemy_position_details(
    tags: Sequence[str],
    hints: Sequence[str],
    evidence_text: str = "",
) -> list[str]:
    haystack = " ".join([*tags, *hints, evidence_text]).lower()
    details: list[str] = []
    for direction, phrase in [
        ("left", "an enemy was to your left"),
        ("right", "an enemy was to your right"),
        ("front", "an enemy was in front of you"),
        ("behind", "an enemy was behind you"),
        ("above", "an enemy was above you"),
    ]:
        if _has_enemy_direction(haystack, direction):
            details.append(phrase)
    if "enemy visible in window" in haystack:
        details.append("an enemy was visible near a window")
    if "enemy pushing through doorway" in haystack:
        details.append("an enemy was pushing through a doorway")
    if "enemy crossing open ground" in haystack:
        details.append("an enemy was crossing exposed ground")
    return details


def _has_enemy_direction(text: str, direction: str) -> bool:
    direction_patterns = {
        "left": [r"left"],
        "right": [r"right"],
        "front": [r"front", r"ahead"],
        "behind": [r"behind", r"back"],
        "above": [r"above", r"up top", r"high ground"],
    }
    terms = r"(?:enemy|enemies|one|hunter|guy|he|they)"
    for chunk in _evidence_chunks(text):
        has_enemy_term = re.search(rf"\b{terms}\b", chunk) is not None
        if "teammate" in chunk and not has_enemy_term:
            continue
        for direction_pattern in direction_patterns[direction]:
            if re.search(
                rf"\b{terms}\b[^.!?;]{{0,40}}\b(?:on|to|at|in|from|my|our|your|the|is|are|was|were|coming|pushing|got|right)?\s*(?:my|our|your|the)?\s*{direction_pattern}\b",
                chunk,
            ):
                return True
            if has_enemy_term and re.search(
                rf"\b(?:on|to|at|in|from)\s+(?:my|our|your|the)?\s*{direction_pattern}\b",
                chunk,
            ):
                return True
            if has_enemy_term and re.search(rf"\b{direction_pattern}\s+(?:of|side)\b", chunk):
                return True
            if re.search(rf"\bon\s+(?:my|our|your)\s+{direction_pattern}\b", chunk):
                return True
    return False


def _teammate_details(
    tags: Sequence[str],
    hints: Sequence[str],
    evidence_text: str = "",
) -> list[str]:
    haystack = " ".join([*tags, *hints, evidence_text]).lower()
    if not re.search(r"\b(teammate|team mate|partner|buddy)\b", haystack):
        return []
    details: list[str] = []
    if re.search(r"\b(teammate|team mate|partner|buddy)\b[^.!?;]{0,50}\b(nearby|near me|close|with me)\b", haystack):
        details.append("a teammate was nearby")
    if re.search(r"\b(teammate|team mate|partner|buddy)\b[^.!?;]{0,50}\b(downed|down|dead|killed)\b", haystack):
        details.append("a teammate was downed or dead")
    if re.search(r"\b(teammate|team mate|partner|buddy)\b[^.!?;]{0,50}\b(pushing|push|pushed|moving in)\b", haystack):
        details.append("a teammate was pushing")
    for direction, phrase in [
        ("left", "a teammate was on your left"),
        ("right", "a teammate was on your right"),
        ("front", "a teammate was in front of you"),
        ("behind", "a teammate was behind you"),
        ("above", "a teammate was above you"),
    ]:
        if _has_teammate_direction(haystack, direction):
            details.append(phrase)
    if "teammate comms" in haystack or re.search(
        r"\b(teammate|team mate|partner|buddy)\b[^.!?;]{0,50}\b(comms|called|callout|said)\b",
        haystack,
    ):
        details.append("teammate comms were present")
    return details


def _has_teammate_direction(text: str, direction: str) -> bool:
    direction_patterns = {
        "left": [r"left"],
        "right": [r"right"],
        "front": [r"front", r"ahead"],
        "behind": [r"behind", r"back"],
        "above": [r"above", r"up top", r"high ground"],
    }
    terms = r"(?:teammate|team mate|partner|buddy)"
    for pattern in direction_patterns[direction]:
        if re.search(rf"\b{terms}\b[^.!?;]{{0,50}}\b(?:on|to|at|in)?\s*(?:my|our|your|the)?\s*{pattern}\b", text):
            return True
    return False


def _evidence_chunks(text: str) -> list[str]:
    return [chunk.strip() for chunk in re.split(r"[.!?;|,]+", text.lower()) if chunk.strip()]


def _notable_phrase(tags: Sequence[str], hints: Sequence[str]) -> str:
    haystack = " ".join([*tags, *hints]).lower()
    details: list[str] = []
    if "enemy visible in window" in haystack:
        details.append("an enemy was visible through or near a window")
    if "explosion" in haystack:
        details.append("there were explosion cues")
    if "teammate comms" in haystack:
        details.append("teammate communication was detected")
    if "funny" in haystack:
        details.append("the clip was tagged as funny")
    return "; ".join(details[:2])


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", value).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def _modality_text(modality_counts: dict[str, int] | None) -> str:
    if not modality_counts:
        return ""
    parts = [f"{count} {name.replace('_', '-')}" for name, count in sorted(modality_counts.items()) if count]
    if not parts:
        return ""
    return "indexed as " + ", ".join(parts)


def _dominant_modality_label(modality_counts: dict[str, int] | None) -> str:
    if not modality_counts:
        return "audio-video"
    modality, _ = max(modality_counts.items(), key=lambda item: item[1])
    return modality.replace("_", "-")


def _sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" ,;")
    if not cleaned:
        return "Gameplay clip awaiting transcript or visual details."
    return cleaned[0].upper() + cleaned[1:].rstrip(".") + "."
