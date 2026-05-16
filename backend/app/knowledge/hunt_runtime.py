from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence
from urllib.parse import unquote

import numpy as np

HF_RETRIEVAL_PROFILE = "qwen3-vl-embedding-v1"
BUNDLED_WEAPON_SKIN_MAP_PATH = Path(__file__).with_name("hunt_weapon_skin_map.json")


class TextEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]:
        ...


HUNT_ENTITY_TYPES = {
    "weapon",
    "tool",
    "consumable",
    "ammo",
    "trait",
    "map",
    "target",
    "monster",
    "world_item",
}


@dataclass(frozen=True)
class HuntEntity:
    id: str
    type: str
    name: str
    aliases: tuple[str, ...]
    description: str
    source_url: str
    image_paths: tuple[str, ...]
    key_values: dict[str, str]


@dataclass(frozen=True)
class HuntKnowledgeHit:
    entity_id: str
    entity_name: str
    entity_type: str
    text: str
    source_url: str
    score: float
    image_paths: tuple[str, ...] = ()
    license: str = ""
    license_url: str = ""


@dataclass(frozen=True)
class HuntEquipmentResolution:
    entity: HuntEntity
    matched_name: str
    display_name: str


class HuntKnowledgeService:
    """Runtime reader/searcher for the local Hunt wiki knowledge pack."""

    def __init__(self, pack_dir: str | Path, *, embedder: TextEmbedder | None = None) -> None:
        self.pack_dir = Path(pack_dir)
        self.embedder = embedder
        self.available = False
        self.manifest: dict[str, Any] = {}
        self.entities: dict[str, HuntEntity] = {}
        self.chunks: list[dict[str, Any]] = []
        self.media: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.alias_index: dict[str, set[str]] = {}
        self.equipment_alias_index: dict[str, str] = {}
        self._equipment_alias_names: dict[str, str] = {}
        self._equipment_alias_display_names: dict[str, str] = {}
        self.weapon_skin_index: dict[str, str] = {}
        self._weapon_skin_names: dict[str, str] = {}
        self._weapon_skin_display_names: dict[str, str] = {}
        self._load()

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "pack_dir": str(self.pack_dir),
            "page_count": self.manifest.get("page_count", 0),
            "entity_count": len(self.entities),
            "chunk_count": len(self.chunks),
            "embedding_dimension": self.manifest.get("embedding_dimension"),
            "embedding_profile": self.manifest.get("embedding_profile"),
            "weapon_skin_count": len(self.weapon_skin_map()),
        }

    def lookup(self, text: str, *, entity_types: set[str] | None = None) -> list[HuntEntity]:
        if not self.available:
            return []
        tokens = _tokens(text)
        matched: dict[str, HuntEntity] = {}
        for token in tokens:
            for entity_id in self.alias_index.get(token, set()):
                entity = self.entities.get(entity_id)
                if entity is None:
                    continue
                if entity_types and entity.type not in entity_types:
                    continue
                matched[entity.id] = entity
        return sorted(matched.values(), key=lambda item: (item.type, item.name))

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        entity_types: set[str] | None = None,
    ) -> list[HuntKnowledgeHit]:
        if not self.available or not query.strip():
            return []
        lexical = self._lexical_scores(query, entity_types=entity_types)
        semantic = self._semantic_scores(query) if self.embedder is not None else {}
        scored: dict[int, float] = {}
        for index, score in lexical.items():
            scored[index] = max(scored.get(index, 0.0), score)
        for index, score in semantic.items():
            scored[index] = max(scored.get(index, 0.0), score)
        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        hits: list[HuntKnowledgeHit] = []
        for index, score in ranked:
            if len(hits) >= top_k:
                break
            chunk = self.chunks[index]
            entity = self.entities.get(str(chunk.get("entity_id")))
            if entity is None:
                continue
            if entity_types and entity.type not in entity_types:
                continue
            hits.append(
                HuntKnowledgeHit(
                    entity_id=entity.id,
                    entity_name=entity.name,
                    entity_type=entity.type,
                    text=str(chunk.get("text") or entity.description),
                    source_url=str(chunk.get("source_url") or entity.source_url),
                    score=float(score),
                    image_paths=entity.image_paths,
                    license=str(chunk.get("license") or self.manifest.get("license") or ""),
                    license_url=str(chunk.get("license_url") or self.manifest.get("license_url") or ""),
                )
            )
        return hits

    def references(self, entity_types: set[str]) -> list[HuntEntity]:
        if not self.available:
            return []
        return [
            entity
            for entity in self.entities.values()
            if entity.type in entity_types and entity.image_paths
        ]

    def reference_image_records(self, entity_types: set[str]) -> list[tuple[HuntEntity, str]]:
        if not self.available:
            return []
        records: list[tuple[HuntEntity, str]] = []
        seen: set[tuple[str, str]] = set()
        for row in self.media:
            entity = self.entities.get(str(row.get("entity_id")))
            if entity is None or entity.type not in entity_types:
                continue
            local_path = str(row.get("local_path") or "")
            if not local_path or not _is_equipment_reference_media(row):
                continue
            key = (entity.id, local_path)
            if key in seen:
                continue
            seen.add(key)
            records.append((entity, local_path))
        if records:
            return records
        for entity in self.references(entity_types):
            for image_path in entity.image_paths:
                key = (entity.id, image_path)
                if key in seen:
                    continue
                seen.add(key)
                records.append((entity, image_path))
        return records

    def entity(self, entity_id: str) -> HuntEntity | None:
        return self.entities.get(entity_id)

    def resolve_equipment_alias(self, text: str, *, entity_types: set[str] | None = None) -> HuntEquipmentResolution | None:
        if not self.available or not text.strip():
            return None
        allowed = entity_types or {"weapon", "tool", "consumable"}
        normalized = _alias_key(text)
        candidates: list[tuple[int, str, HuntEntity]] = []
        for alias, entity_id in self.equipment_alias_index.items():
            entity = self.entities.get(entity_id)
            if entity is None or entity.type not in allowed:
                continue
            if alias == normalized:
                candidates.append((len(alias), alias, entity))
            elif alias and f" {alias} " in f" {normalized} ":
                candidates.append((len(alias), alias, entity))
        if not candidates:
            return None
        _, alias, entity = max(candidates, key=lambda item: (item[0], item[2].type == "weapon"))
        matched = self._equipment_alias_match_name(alias) or text.strip()
        display = self._equipment_alias_display_name(alias) or entity.name
        return HuntEquipmentResolution(entity=entity, matched_name=matched, display_name=display)

    def resolve_equipment(self, text: str, *, entity_types: set[str] | None = None) -> HuntEquipmentResolution | None:
        resolved = self.resolve_equipment_alias(text, entity_types=entity_types)
        if resolved is not None:
            return resolved
        allowed = entity_types or {"weapon", "tool", "consumable"}
        if "weapon" not in allowed:
            return None
        return self.resolve_weapon_skin(text)

    def resolve_weapon_skin(self, text: str) -> HuntEquipmentResolution | None:
        if not self.available or not text.strip():
            return None
        normalized = _alias_key(text)
        candidates: list[tuple[int, str, HuntEntity]] = []
        for alias, entity_id in self.weapon_skin_index.items():
            entity = self.entities.get(entity_id)
            if entity is None or entity.type != "weapon":
                continue
            if alias == normalized:
                candidates.append((len(alias), alias, entity))
            elif alias and f" {alias} " in f" {normalized} ":
                candidates.append((len(alias), alias, entity))
        if not candidates:
            return None
        _, alias, entity = max(candidates, key=lambda item: (item[0], not _is_variant_equipment(item[2])))
        matched = self._weapon_skin_names.get(alias) or text.strip()
        display = self._weapon_skin_display_names.get(alias) or f"{entity.name} ({matched} skin)"
        return HuntEquipmentResolution(entity=entity, matched_name=matched, display_name=display)

    def resolve_weapon_skin_display(self, text: str) -> str | None:
        if not text.strip():
            return None
        normalized = _alias_key(text)
        candidates = [
            (len(alias), alias)
            for alias in self._weapon_skin_display_names
            if alias == normalized or (alias and f" {alias} " in f" {normalized} ")
        ]
        if not candidates:
            return None
        _, alias = max(candidates, key=lambda item: item[0])
        return self._weapon_skin_display_names.get(alias)

    def weapon_skin_map(self) -> dict[str, str]:
        return {
            self._weapon_skin_names[alias]: self._weapon_skin_display_names[alias]
            for alias in sorted(self._weapon_skin_display_names)
            if alias in self._weapon_skin_names and alias in self._weapon_skin_display_names
        }

    def _load(self) -> None:
        required = [
            self.pack_dir / "manifest.json",
            self.pack_dir / "entities.jsonl",
            self.pack_dir / "chunks.jsonl",
        ]
        if any(not path.exists() for path in required):
            self._build_weapon_skin_index()
            return
        self.manifest = _read_json(self.pack_dir / "manifest.json")
        self.entities = {}
        for row in _read_jsonl(self.pack_dir / "entities.jsonl"):
            entity = HuntEntity(
                id=str(row.get("id")),
                type=str(row.get("type")),
                name=str(row.get("name")),
                aliases=tuple(str(item) for item in row.get("aliases") or [] if str(item).strip()),
                description=str(row.get("description") or ""),
                source_url=str(row.get("source_url") or ""),
                image_paths=tuple(str(item) for item in row.get("image_paths") or [] if str(item).strip()),
                key_values={str(key): str(value) for key, value in dict(row.get("key_values") or {}).items()},
            )
            self.entities[entity.id] = entity
        self.chunks = _read_jsonl(self.pack_dir / "chunks.jsonl")
        self.media = _read_jsonl(self.pack_dir / "media_index.jsonl")
        embedding_path = self.pack_dir / "embeddings.npy"
        if embedding_path.exists() and self.manifest.get("embedding_profile") in {HF_RETRIEVAL_PROFILE, None, ""}:
            loaded = np.load(embedding_path)
            self.embeddings = _normalize_rows(loaded.astype(np.float32, copy=False))
        self._build_alias_index()
        self._build_equipment_alias_index()
        self._build_weapon_skin_index()
        self.available = bool(self.entities and self.chunks)

    def _build_alias_index(self) -> None:
        self.alias_index = {}
        for entity in self.entities.values():
            values = [entity.id, entity.name, entity.description, *entity.aliases]
            for key, value in entity.key_values.items():
                if key.lower() in {"ammo type", "size", "source"}:
                    values.append(value)
            for token in _tokens(" ".join(values)):
                self.alias_index.setdefault(token, set()).add(entity.id)

    def _build_equipment_alias_index(self) -> None:
        self.equipment_alias_index = {}
        self._equipment_alias_names: dict[str, str] = {}
        self._equipment_alias_display_names: dict[str, str] = {}
        for entity in self.entities.values():
            if entity.type not in {"weapon", "tool", "consumable"}:
                continue
            for value in [entity.name, *entity.aliases]:
                self._add_equipment_alias(value, entity.id, display_name=entity.name)
        for row in self.media:
            entity = self.entities.get(str(row.get("entity_id")))
            if entity is None or entity.type not in {"weapon", "tool", "consumable"}:
                continue
            for value, display_name in _media_equipment_aliases(row, entity.name):
                self._add_equipment_alias(value, entity.id, display_name=display_name)
        for alias, target in EQUIPMENT_ALIAS_OVERRIDES.items():
            target_key = _alias_key(target)
            entity_id = self.equipment_alias_index.get(target_key)
            if entity_id:
                self._add_equipment_alias(alias, entity_id, display_name=target)

    def _build_weapon_skin_index(self) -> None:
        self.weapon_skin_index = {}
        self._weapon_skin_names = {}
        self._weapon_skin_display_names = {}
        for entity in self.entities.values():
            if entity.type != "weapon":
                continue
            for skin_name in _weapon_skin_names_from_entity(entity):
                self._add_weapon_skin_alias(skin_name, entity.id)
        for row in self.media:
            entity = self.entities.get(str(row.get("entity_id")))
            if entity is None or entity.type != "weapon" or not _is_skin_media(row):
                continue
            for value, _ in _media_equipment_aliases(row, entity.name):
                self._add_weapon_skin_alias(value, entity.id)
        for row in _read_bundled_weapon_skin_map():
            skin_name = str(row.get("skin") or "")
            weapon_name = str(row.get("weapon") or "")
            display_name = str(row.get("display_name") or "")
            if not skin_name or not weapon_name:
                continue
            self._add_weapon_skin_display_alias(
                skin_name,
                display_name or f"{weapon_name} ({skin_name} skin)",
                weapon_name=weapon_name,
            )

    def _add_equipment_alias(self, value: str, entity_id: str, *, display_name: str) -> None:
        cleaned = _clean_alias_name(value)
        key = _alias_key(cleaned)
        if not key:
            return
        existing_id = self.equipment_alias_index.get(key)
        if existing_id and not _alias_key(cleaned).endswith("shorty"):
            existing = self.entities.get(existing_id)
            incoming = self.entities.get(entity_id)
            if existing is not None and incoming is not None:
                existing_is_variant = _is_variant_equipment(existing)
                incoming_is_variant = _is_variant_equipment(incoming)
                if existing_is_variant and not incoming_is_variant:
                    pass
                elif incoming_is_variant and not existing_is_variant:
                    return
                else:
                    return
        self.equipment_alias_index[key] = entity_id
        self._equipment_alias_names[key] = cleaned
        self._equipment_alias_display_names[key] = display_name

    def _add_weapon_skin_alias(self, value: str, entity_id: str) -> None:
        entity = self.entities.get(entity_id)
        if entity is None or entity.type != "weapon":
            return
        cleaned = _clean_skin_name(value, entity.name)
        key = _alias_key(cleaned)
        if not key:
            return
        existing_id = self.weapon_skin_index.get(key)
        if existing_id:
            existing = self.entities.get(existing_id)
            if existing is not None:
                existing_is_variant = _is_variant_equipment(existing)
                incoming_is_variant = _is_variant_equipment(entity)
                if existing_is_variant and not incoming_is_variant:
                    pass
                elif incoming_is_variant and not existing_is_variant:
                    return
                else:
                    return
        self.weapon_skin_index[key] = entity_id
        self._weapon_skin_names[key] = cleaned
        self._weapon_skin_display_names[key] = f"{entity.name} ({cleaned} skin)"

    def _add_weapon_skin_display_alias(self, value: str, display_name: str, *, weapon_name: str) -> None:
        cleaned = _clean_skin_name(value, weapon_name)
        key = _alias_key(cleaned)
        if not key or key in self._weapon_skin_display_names:
            return
        self._weapon_skin_names[key] = cleaned
        self._weapon_skin_display_names[key] = display_name
        entity_id = self.equipment_alias_index.get(_alias_key(weapon_name))
        if entity_id and key not in self.weapon_skin_index:
            self.weapon_skin_index[key] = entity_id

    def _equipment_alias_match_name(self, alias: str) -> str | None:
        return self._equipment_alias_names.get(alias)

    def _equipment_alias_display_name(self, alias: str) -> str | None:
        return self._equipment_alias_display_names.get(alias)

    def _lexical_scores(self, query: str, *, entity_types: set[str] | None) -> dict[int, float]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return {}
        scores: dict[int, float] = {}
        for index, chunk in enumerate(self.chunks):
            entity = self.entities.get(str(chunk.get("entity_id")))
            if entity is None:
                continue
            if entity_types and entity.type not in entity_types:
                continue
            haystack = " ".join([entity.name, " ".join(entity.aliases), str(chunk.get("text") or "")])
            hay_tokens = set(_tokens(haystack))
            overlap = sum(1 for token in query_tokens if token in hay_tokens)
            if overlap:
                exact_bonus = 0.25 if query.lower() in haystack.lower() else 0.0
                scores[index] = (overlap / max(len(query_tokens), 1)) + exact_bonus
        return scores

    def _semantic_scores(self, query: str) -> dict[int, float]:
        if self.embeddings is None or self.embedder is None or len(self.chunks) == 0:
            return {}
        try:
            vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        except Exception:
            return {}
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return {}
        vector = vector / norm
        scores = self.embeddings @ vector
        top_n = min(20, len(scores))
        if top_n <= 0:
            return {}
        indices = np.argpartition(scores, -top_n)[-top_n:]
        return {int(index): float(scores[index]) for index in indices if math.isfinite(float(scores[index]))}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            rows.append(loaded)
    return rows


def _read_bundled_weapon_skin_map() -> list[dict[str, Any]]:
    try:
        loaded = json.loads(BUNDLED_WEAPON_SKIN_MAP_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = loaded.get("skins") if isinstance(loaded, dict) else None
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2:
        return values
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return values / norms


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1
    ]


def _alias_key(text: str) -> str:
    return " ".join(_tokens(text))


def _clean_alias_name(value: str) -> str:
    cleaned = unquote(str(value))
    cleaned = cleaned.rsplit("/", 1)[-1]
    cleaned = cleaned.split("?", 1)[0]
    cleaned = re.sub(r"\.(png|jpe?g|webp)$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[_-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for prefix in (
        "Weapon 3D ",
        "Weapon ",
        "Tool 3D ",
        "Tool ",
        "Consumable 3D ",
        "Consumable ",
        "Ammo ",
        "Item ",
    ):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
            break
    return cleaned


def _weapon_skin_names_from_entity(entity: HuntEntity) -> list[str]:
    raw = str(entity.key_values.get("Skins") or "")
    return _dedupe(
        cleaned
        for value in re.split(r"\s*,\s*", raw)
        if (cleaned := _clean_skin_name(value, entity.name))
    )


def _clean_skin_name(value: str, entity_name: str) -> str:
    cleaned = _clean_alias_name(value)
    if not cleaned:
        return ""
    if _alias_key(cleaned) == _alias_key(entity_name):
        return ""
    if cleaned.lower() in {"base", "base weapon", "base tool", "base consumable", "weapon", "tool", "consumable"}:
        return ""
    return cleaned


def _is_variant_equipment(entity: HuntEntity) -> bool:
    page_name = entity.source_url.rsplit("/wiki/", 1)[-1] if "/wiki/" in entity.source_url else entity.id
    return "/" in page_name


def _media_equipment_aliases(row: dict[str, Any], entity_name: str) -> list[tuple[str, str]]:
    values = [
        str(row.get("alt") or ""),
        str(row.get("source_url") or ""),
        str(row.get("local_path") or ""),
    ]
    aliases: list[tuple[str, str]] = []
    is_skin = _is_skin_media(row)
    for value in values:
        cleaned = _clean_alias_name(value)
        if not cleaned or cleaned.lower() in {"dlc art", "promo", "model", "concept dlc art"}:
            continue
        display_name = entity_name if is_skin else cleaned
        aliases.append((cleaned, display_name))
        entity_key = _alias_key(entity_name)
        cleaned_key = _alias_key(cleaned)
        if entity_key and cleaned_key.startswith(entity_key + " "):
            suffix_words = cleaned.split()[len(entity_name.split()) :]
            suffix = " ".join(suffix_words).strip()
            if suffix:
                aliases.append((suffix, display_name))
    return _dedupe_aliases(aliases)


def _is_skin_media(row: dict[str, Any]) -> bool:
    alt = str(row.get("alt") or "")
    source = str(row.get("source_url") or "")
    local = str(row.get("local_path") or "")
    haystack = f"{alt} {source} {local}".lower()
    return (
        " 3d " in f" {haystack} "
        or "_3d_" in haystack
        or "weapon 3d" in haystack
        or "dlc art" in haystack
        or "/thumb/dlc_" in haystack
    )


def _is_equipment_reference_media(row: dict[str, Any]) -> bool:
    alt = str(row.get("alt") or "")
    title = str(row.get("title") or "")
    source = str(row.get("source_url") or "")
    content_type = str(row.get("content_type") or "").lower()
    entity_type = str(row.get("entity_type") or "").lower()
    haystack = f"{alt} {title} {source}".lower()
    if "dlc art" in haystack or "/thumb/dlc_" in haystack:
        return False
    if " 3d " in f" {haystack} " or "_3d_" in haystack or "weapon 3d" in haystack:
        return False
    if entity_type in {"weapon", "tool", "consumable"} and f"{entity_type} " not in haystack:
        return False
    if entity_type == "weapon" and re.search(r"\bammo\b|/ammo_", haystack):
        return False
    looks_like_icon = bool(re.search(r"\b(weapon|tool|consumable|ammo)\b", haystack)) and ".png" in f"{alt} {source}".lower()
    if content_type and content_type != "image/png" and not looks_like_icon:
        return False
    return any(marker in haystack for marker in ("weapon", "tool", "consumable", "ammo"))


EQUIPMENT_ALIAS_OVERRIDES = {
    "Chock Bomb": "Choke Bombs",
    "Chock Bombs": "Choke Bombs",
    "Choke Bomb": "Choke Bombs",
    "Centering Breath": "First Aid Kit",
    "Devil's Salve": "First Aid Kit",
    "Dying Breath": "First Aid Kit",
    "Health Satchel": "First Aid Kit",
    "Healing Satchel": "First Aid Kit",
    "Mosin Obrez Drum": "Mosin Obrez Extended",
    "Regen Shot": "Regeneration Shot",
    "Rooted Apothecary": "First Aid Kit",
    "Scarfskin Satchel": "First Aid Kit",
    "Signal Flair": "Flare Pistol",
    "Signal Flare": "Flare Pistol",
    "The Marrow": "First Aid Kit",
    "The Show Must Go On": "First Aid Kit",
    "The Waxwing": "First Aid Kit",
}


def _dedupe_aliases(values: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    output: list[tuple[str, str]] = []
    for value, display_name in values:
        cleaned = value.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append((cleaned, display_name))
    return output


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def format_knowledge_hits(hits: Sequence[HuntKnowledgeHit], *, max_chars: int = 900) -> str:
    parts: list[str] = []
    remaining = max_chars
    for hit in hits:
        if remaining <= 0:
            break
        text = re.sub(r"\s+", " ", hit.text).strip()
        snippet = text[: max(0, remaining - len(hit.entity_name) - 8)]
        if not snippet:
            break
        parts.append(f"{hit.entity_name}: {snippet}")
        remaining -= len(parts[-1])
    return "\n".join(parts)
