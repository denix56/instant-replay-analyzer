from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from ..llm.schemas import TagResult


DEFAULT_TAG_RULES: Dict[str, tuple[str, ...]] = {
    "kill": ("kill", "eliminated", "frag", "downed", "headshot"),
    "death": ("died", "dead", "death", "killed me", "got me", "traded"),
    "boss": ("boss", "butcher", "spider", "assassin", "scrapbeak", "rotjaw"),
    "extract": ("extract", "extraction", "extracting"),
    "compound": ("compound", "yard", "barn", "church", "station", "fort"),
    "night": ("night", "dark", "low light"),
    "indoors": ("inside", "indoor", "building", "room", "stairs"),
    "shotgun": ("shotgun", "romero", "specter", "crown", "terminus", "slate"),
    "rifle": ("rifle", "long ammo", "mosin", "lebel", "berthier", "sparks"),
    "teammate comms": ("teammate", "on the left", "on right", "callout", "comms"),
    "gunfight": ("gunfight", "gunshot", "shooting", "shot", "duel"),
    "funny": ("funny", "chaos", "laugh", "panic"),
    "fail": ("fail", "missed", "whiff", "mistake"),
    "clutch": ("clutch", "last one", "solo", "saved"),
    "reload": ("reload", "reloading"),
    "revive": ("revive", "reviving", "picked up"),
    "burning": ("burn", "burning", "fire", "lantern"),
    "bounty": ("bounty", "token", "banish"),
    "window": ("window", "peek", "re-peek", "repeek"),
    "lair": ("lair", "boss room"),
    "forest": ("forest", "trees", "woods"),
    "sniper": ("sniper", "scope", "scoped", "marksman"),
    "melee": ("melee", "knife", "saber", "bomblance"),
    "footsteps": ("footstep", "steps", "walking", "running"),
    "gunshots": ("gunshot", "shots", "shooting", "fired"),
    "explosion": ("explosion", "dynamite", "frag", "bomb"),
    "dogs": ("dog", "dogs", "kennel"),
    "crows": ("crow", "crows", "birds"),
}


@dataclass(frozen=True)
class TaggingConfig:
    rules: Dict[str, tuple[str, ...]] = field(default_factory=lambda: DEFAULT_TAG_RULES)
    include_games: bool = True
    max_tags: int = 12


class GameplayTagger:
    def __init__(self, config: Optional[TaggingConfig] = None) -> None:
        self.config = config or TaggingConfig()

    def tag(
        self,
        *,
        transcript: str = "",
        summary: str = "",
        ocr_text: str = "",
        audio_events: Iterable[str] = (),
        game: str = "",
    ) -> TagResult:
        text = " ".join([transcript, summary, ocr_text, " ".join(audio_events)]).lower()
        scores: Dict[str, float] = {}
        for tag, keywords in self.config.rules.items():
            matches = sum(1 for keyword in keywords if re.search(rf"\b{re.escape(keyword)}\b", text))
            if matches:
                scores[tag] = min(1.0, 0.45 + matches * 0.2)
        if self.config.include_games and game and game.strip().lower() not in {"ungrouped", "unknown"}:
            scores[_slug(game)] = max(scores.get(_slug(game), 0.0), 0.75)
        tags = sorted(scores, key=lambda tag: (-scores[tag], tag))[: self.config.max_tags]
        return TagResult(tags=tags, confidence_by_tag={tag: scores[tag] for tag in tags}, engine="rules")


def tag_clip(**kwargs: object) -> TagResult:
    return GameplayTagger().tag(**kwargs)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
