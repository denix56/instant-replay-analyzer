from __future__ import annotations

import json
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .death_screen import OCRLine, recognize_text
from ..embeddings.hf_multimodal_embedder import HuggingFaceMultimodalEmbedder
from ..knowledge.hunt_runtime import HuntEntity, HuntEquipmentResolution, HuntKnowledgeService


HUD_REFERENCE_DESCRIPTOR_VERSION = 6
HUD_REFERENCE_DESCRIPTOR_FILE = "hud_reference_descriptors.npz"
HUD_REFERENCE_DESCRIPTOR_METADATA_FILE = "hud_reference_descriptors.jsonl"
HUD_REFERENCE_EMBEDDING_VERSION = 4
HUD_REFERENCE_EMBEDDING_FILE = "hud_reference_hf_embeddings.npz"
HUD_REFERENCE_EMBEDDING_METADATA_FILE = "hud_reference_hf_embeddings.jsonl"
HUD_MATCH_ICON_BOTTOM_RATIO = 0.76
HUD_MATCH_EMBEDDING_WEIGHT = 0.18
HUD_EMPTY_SLOT_PRESENCE_THRESHOLD = 0.20
HUD_WEAPON_CONTOUR_PADDING_RATIO = 0.04
HUD_EQUIPMENT_CONTOUR_PADDING_RATIO = 0.24
HUD_QUANTITY_HINT_SCORE_BONUS = 0.085
HUD_LOADED_EXTRA_HINT_SCORE_BONUS = 0.11
HUD_CALIBRATION_REFERENCE_SCORE_BONUS = 0.18
HUD_CONSUMABLE_CALIBRATION_CONFIDENCE_THRESHOLD = 0.70


@dataclass(frozen=True)
class _HudIconCropVariant:
    name: str
    top: float
    bottom: float
    left: float
    right: float


HUD_MATCH_ICON_CROP_VARIANTS: tuple[_HudIconCropVariant, ...] = (
    _HudIconCropVariant("long", 0.24, 0.86, 0.12, 0.90),
    _HudIconCropVariant("short", 0.30, 0.78, 0.16, 0.88),
    _HudIconCropVariant("mid", 0.30, 0.70, 0.12, 0.90),
    _HudIconCropVariant("high", 0.18, 0.62, 0.16, 0.88),
)

HUD_EQUIPMENT_ICON_CROP_VARIANTS: tuple[_HudIconCropVariant, ...] = (
    _HudIconCropVariant("equipment-main", 0.12, 1.00, 0.00, 1.00),
    _HudIconCropVariant("equipment-main-tight", 0.12, 0.94, 0.00, 1.00),
    _HudIconCropVariant("equipment-lower", 0.18, 1.00, 0.00, 1.00),
    _HudIconCropVariant("equipment-lower-tight", 0.18, 0.96, 0.00, 1.00),
    _HudIconCropVariant("equipment-wide", 0.06, 1.00, 0.00, 1.00),
    _HudIconCropVariant("equipment-wide-tight", 0.06, 0.98, 0.00, 1.00),
)

HUD_TOOL_ICON_CROP_VARIANTS: tuple[_HudIconCropVariant, ...] = HUD_EQUIPMENT_ICON_CROP_VARIANTS
HUD_CONSUMABLE_ICON_CROP_VARIANTS: tuple[_HudIconCropVariant, ...] = HUD_EQUIPMENT_ICON_CROP_VARIANTS


@dataclass(frozen=True)
class RelativeBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class HudSlotConfig:
    key: str
    box: RelativeBox
    allowed_types: tuple[str, ...]
    box_variants: tuple[RelativeBox, ...] = ()


def _slot_sequence(
    start_x: float,
    y_min: float,
    y_max: float,
    specs: tuple[tuple[str, float, tuple[str, ...]], ...],
    *,
    gap: float = 0.0,
) -> tuple[HudSlotConfig, ...]:
    slots: list[HudSlotConfig] = []
    x_min = start_x
    for key, width, allowed_types in specs:
        x_max = min(1.0, round(x_min + width, 6))
        slots.append(HudSlotConfig(key, RelativeBox(round(x_min, 6), y_min, x_max, y_max), allowed_types))
        x_min = round(x_max + gap, 6)
    return tuple(slots)


@dataclass(frozen=True)
class HudGeometryConfig:
    hud_crop: RelativeBox = RelativeBox(0.0, 0.74, 1.0, 0.99)
    current_weapon_name: RelativeBox = RelativeBox(0.30, 0.735, 0.70, 0.86)
    slots: tuple[HudSlotConfig, ...] = (
        HudSlotConfig(
            "1",
            RelativeBox(0.085, 0.755, 0.205, 0.89),
            ("weapon",),
            (
                RelativeBox(0.085, 0.755, 0.190, 0.89),
                RelativeBox(0.095, 0.755, 0.205, 0.89),
                RelativeBox(0.090, 0.755, 0.175, 0.89),
                RelativeBox(0.075, 0.755, 0.215, 0.89),
            ),
        ),
        HudSlotConfig(
            "2",
            RelativeBox(0.195, 0.755, 0.325, 0.89),
            ("weapon",),
            (
                RelativeBox(0.195, 0.755, 0.310, 0.89),
                RelativeBox(0.210, 0.755, 0.325, 0.89),
                RelativeBox(0.205, 0.755, 0.300, 0.89),
                RelativeBox(0.185, 0.755, 0.335, 0.89),
            ),
        ),
        *_slot_sequence(
            0.335,
            0.755,
            0.89,
            (
                ("3", 0.066, ("tool",)),
                ("4", 0.066, ("tool",)),
                ("5", 0.066, ("tool",)),
                ("6", 0.066, ("tool",)),
                ("7", 0.066, ("consumable",)),
                ("8", 0.066, ("consumable",)),
                ("9", 0.066, ("consumable",)),
                ("0", 0.066, ("consumable",)),
            ),
            gap=0.006,
        ),
    )


@dataclass(frozen=True)
class HudMatch:
    slot_key: str
    is_active: bool
    entity_id: str | None
    entity_name: str | None
    entity_type: str | None
    confidence: float
    matched_image_path: str | None
    highlight_score: float


@dataclass(frozen=True)
class HudDetectionResult:
    frame_path: str
    frame_width: int
    frame_height: int
    matches: tuple[HudMatch, ...]

    @property
    def active_match(self) -> HudMatch | None:
        return next((match for match in self.matches if match.is_active), None)

    def loadout_names(self) -> list[str]:
        names = [match.entity_name for match in self.matches if match.entity_name]
        return _dedupe(str(name) for name in names)


@dataclass(frozen=True)
class _ReferenceImage:
    entity_id: str
    entity_name: str
    entity_type: str
    local_path: Path
    descriptor: "_VisualDescriptor"
    embedding: np.ndarray | None = None


@dataclass(frozen=True)
class _VisualDescriptor:
    icon: np.ndarray
    edges: np.ndarray
    feature: np.ndarray
    aspect_ratio: float


class HudLoadoutDetector:
    def __init__(
        self,
        knowledge: HuntKnowledgeService,
        *,
        geometry: HudGeometryConfig | None = None,
        confidence_threshold: float = 0.28,
        embedder: HuggingFaceMultimodalEmbedder | None = None,
    ) -> None:
        self.knowledge = knowledge
        self.geometry = geometry or HudGeometryConfig()
        self.confidence_threshold = confidence_threshold
        self.embedder = embedder if embedder is not None and embedder.uses_real_backend else None
        self._references_by_type: dict[str, list[_ReferenceImage]] | None = None

    def detect_frame(self, frame_path: str | Path) -> HudDetectionResult | None:
        image = _read_image(frame_path)
        if image is None:
            return None
        height, width = image.shape[:2]
        current_resolution = self._current_weapon_from_ocr(image)
        matches = (self._current_weapon_match(current_resolution),) if current_resolution is not None else ()
        return HudDetectionResult(
            frame_path=str(frame_path),
            frame_width=width,
            frame_height=height,
            matches=matches,
        )

    def detect_frames(self, frame_paths: Iterable[str | Path]) -> HudDetectionResult | None:
        loaded: list[tuple[str, np.ndarray]] = []
        for frame_path in frame_paths:
            image = _read_image(frame_path)
            if image is not None:
                loaded.append((str(frame_path), image))
        if not loaded:
            return None
        if len(loaded) == 1:
            return self.detect_frame(loaded[0][0])
        first_path, first_image = loaded[0]
        height, width = first_image.shape[:2]
        compatible = [(path, image) for path, image in loaded if image.shape[:2] == (height, width)]
        if not compatible:
            return None
        current_resolution = self._current_weapon_from_ocr_frames([image for _, image in compatible])
        matches = (self._current_weapon_match(current_resolution),) if current_resolution is not None else ()
        return HudDetectionResult(frame_path=first_path, frame_width=width, frame_height=height, matches=matches)

    def _current_weapon_from_ocr(self, image: np.ndarray) -> HuntEquipmentResolution | None:
        return self._resolve_current_weapon_text(crop_relative(image, self.geometry.current_weapon_name))

    def _current_weapon_from_ocr_frames(self, images: Iterable[np.ndarray]) -> HuntEquipmentResolution | None:
        for image in images:
            resolution = self._current_weapon_from_ocr(image)
            if resolution is not None:
                return resolution
        return None

    def _resolve_current_weapon_text(self, crop: np.ndarray) -> HuntEquipmentResolution | None:
        if crop.size == 0:
            return None
        for line in _recognize_current_weapon_lines(crop):
            if line.confidence < 0.35:
                continue
            resolution = self.knowledge.resolve_equipment(line.text, entity_types={"weapon", "tool", "consumable"})
            if resolution is not None:
                return resolution
        return None

    @staticmethod
    def _current_weapon_match(resolution: HuntEquipmentResolution) -> HudMatch:
        return HudMatch(
            slot_key="current_ocr",
            is_active=True,
            entity_id=resolution.entity.id,
            entity_name=resolution.display_name,
            entity_type=resolution.entity.type,
            confidence=0.96,
            matched_image_path=None,
            highlight_score=1.0,
        )

    def _match_slot(
        self,
        crop: np.ndarray,
        allowed_types: Iterable[str],
    ) -> tuple[str | None, str | None, str | None, float, str | None]:
        if crop.size == 0 or not self._allowed_references(allowed_types):
            return None, None, None, 0.0, None
        return self._match_crops_average([crop], allowed_types)

    def _match_crops_average(
        self,
        crops: Iterable[np.ndarray],
        allowed_types: Iterable[str],
    ) -> tuple[str | None, str | None, str | None, float, str | None]:
        allowed_type_tuple = tuple(allowed_types)
        crop_list = [crop for crop in crops if crop.size]
        if not _crop_groups_have_visible_icon([crop_list], allowed_type_tuple):
            return None, None, None, 0.0, None
        descriptors = multi_crop_hud_visual_descriptors(
            crop_list,
            variants=_hud_icon_crop_variants_for_types(allowed_type_tuple),
            use_ocr_key_binding_mask=allowed_type_tuple != ("weapon",),
        )
        quantity_hint = _slot_quantity_hint(crop_list) if allowed_type_tuple != ("weapon",) else None
        return self._guard_calibration_overmatch(
            self._match_descriptors(
                descriptors,
                allowed_type_tuple,
                allow_embedding=True,
                quantity_hint=quantity_hint,
            ),
            allowed_type_tuple,
        )

    def _match_crop_groups(
        self,
        crop_groups: Iterable[Iterable[np.ndarray]],
        allowed_types: Iterable[str],
    ) -> tuple[str | None, str | None, str | None, float, str | None]:
        allowed_type_tuple = tuple(allowed_types)
        groups = [list(crops) for crops in crop_groups]
        if not _crop_groups_have_visible_icon(groups, allowed_type_tuple):
            return None, None, None, 0.0, None
        variants = _hud_icon_crop_variants_for_types(allowed_type_tuple)
        descriptors: list[_VisualDescriptor] = []
        for crops in groups:
            descriptors.extend(
                multi_crop_hud_visual_descriptors(
                    crops,
                    variants=variants,
                    use_ocr_key_binding_mask=allowed_type_tuple != ("weapon",),
                )
            )
        quantity_hint = _slot_quantity_hint(groups[0]) if groups and allowed_type_tuple != ("weapon",) else None
        return self._guard_calibration_overmatch(
            self._match_descriptors(
                descriptors,
                allowed_type_tuple,
                allow_embedding=True,
                quantity_hint=quantity_hint,
            ),
            allowed_type_tuple,
        )

    def _guard_calibration_overmatch(
        self,
        result: tuple[str | None, str | None, str | None, float, str | None],
        allowed_types: Iterable[str],
    ) -> tuple[str | None, str | None, str | None, float, str | None]:
        entity_id, entity_name, entity_type, confidence, matched_path = result
        if _is_weak_hud_calibration_match(matched_path, tuple(allowed_types), confidence):
            return None, None, None, 0.0, None
        return entity_id, entity_name, entity_type, confidence, matched_path

    def _match_descriptor(
        self,
        descriptor: _VisualDescriptor,
        allowed_types: Iterable[str],
    ) -> tuple[str | None, str | None, str | None, float, str | None]:
        return self._match_descriptors([descriptor], allowed_types, allow_embedding=True)

    def _match_descriptors(
        self,
        descriptors: Iterable[_VisualDescriptor],
        allowed_types: Iterable[str],
        *,
        allow_embedding: bool,
        quantity_hint: str | None = None,
    ) -> tuple[str | None, str | None, str | None, float, str | None]:
        descriptor_list = list(descriptors)
        references = self._allowed_references(allowed_types)
        if not descriptor_list or not references:
            return None, None, None, 0.0, None
        scored: list[tuple[float, _ReferenceImage]] = []
        for reference in references:
            best_score = 0.0
            for descriptor in descriptor_list:
                best_score = max(best_score, hybrid_visual_score(descriptor, reference.descriptor))
            best_score = self._apply_quantity_hint(best_score, reference, quantity_hint)
            scored.append((best_score, reference))
        score, margin = _top_entity_score_and_margin(scored)
        if not allow_embedding or self.embedder is None or _visual_match_is_decisive(score, margin):
            return self._best_scored_reference(scored)

        query_embeddings = self._embed_descriptors(descriptor_list)
        if not any(query_embedding is not None for query_embedding in query_embeddings):
            return self._best_scored_reference(scored)

        scored = []
        for reference in references:
            best_score = 0.0
            for descriptor, query_embedding in zip(descriptor_list, query_embeddings):
                best_score = max(best_score, self._score_descriptor_against_reference(descriptor, reference, query_embedding))
            best_score = self._apply_quantity_hint(best_score, reference, quantity_hint)
            scored.append((best_score, reference))
        return self._best_scored_reference(scored)

    def _apply_quantity_hint(self, score: float, reference: _ReferenceImage, quantity_hint: str | None) -> float:
        adjusted = float(score)
        if _is_hud_calibration_reference(reference) and adjusted >= 0.24:
            adjusted += HUD_CALIBRATION_REFERENCE_SCORE_BONUS
        if not quantity_hint:
            return adjusted
        entity = self.knowledge.entity(reference.entity_id)
        if entity is None:
            return adjusted
        markers = _equipment_quantity_markers(entity)
        if not markers:
            return adjusted
        if quantity_hint in markers.get("loaded_extra", set()):
            return float(adjusted + HUD_LOADED_EXTRA_HINT_SCORE_BONUS)
        if quantity_hint in markers.get("quantity", set()):
            return float(adjusted + HUD_QUANTITY_HINT_SCORE_BONUS)
        return adjusted

    def _allowed_references(self, allowed_types: Iterable[str]) -> list[_ReferenceImage]:
        references: list[_ReferenceImage] = []
        by_type = self._references()
        for entity_type in allowed_types:
            references.extend(by_type.get(entity_type, []))
        return references

    def _score_descriptor_against_reference(
        self,
        descriptor: _VisualDescriptor,
        reference: _ReferenceImage,
        query_embedding: np.ndarray | None,
    ) -> float:
        visual_score = hybrid_visual_score(descriptor, reference.descriptor)
        if query_embedding is not None and reference.embedding is not None:
            embedding_score = max(0.0, cosine(query_embedding, reference.embedding))
            return float(embedding_score * HUD_MATCH_EMBEDDING_WEIGHT + visual_score * (1.0 - HUD_MATCH_EMBEDDING_WEIGHT))
        return float(visual_score)

    def _best_scored_reference(
        self,
        scored: list[tuple[float, _ReferenceImage]],
    ) -> tuple[str | None, str | None, str | None, float, str | None]:
        if not scored:
            return None, None, None, 0.0, None
        best_by_entity: dict[str, tuple[float, _ReferenceImage]] = {}
        for score, reference in scored:
            existing = best_by_entity.get(reference.entity_id)
            if existing is None or score > existing[0]:
                best_by_entity[reference.entity_id] = (score, reference)
        ranked = sorted(best_by_entity.values(), key=lambda item: item[0], reverse=True)
        if not ranked:
            return None, None, None, 0.0, None
        score, reference = ranked[0]
        margin = score - ranked[1][0] if len(ranked) > 1 else score
        confidence = _margin_adjusted_confidence(score, margin)
        return (
            reference.entity_id,
            reference.entity_name,
            reference.entity_type,
            confidence,
            str(reference.local_path),
        )

    def _embed_descriptor(self, descriptor: _VisualDescriptor) -> np.ndarray | None:
        if self.embedder is None:
            return None
        try:
            return embed_icon_with_model(self.embedder, descriptor.icon)
        except Exception:
            return None

    def _embed_descriptors(self, descriptors: list[_VisualDescriptor]) -> list[np.ndarray | None]:
        if self.embedder is None:
            return [None for _ in descriptors]
        try:
            return list(embed_icons_with_model(self.embedder, [descriptor.icon for descriptor in descriptors]))
        except Exception:
            return [self._embed_descriptor(descriptor) for descriptor in descriptors]

    def _references(self) -> dict[str, list[_ReferenceImage]]:
        if self._references_by_type is not None:
            return self._references_by_type
        embedding_by_path = load_reference_embedding_index(
            self.knowledge,
            expected_dimension=self.embedder.dimension if self.embedder is not None else None,
        )
        cached = load_reference_descriptor_index(self.knowledge, embedding_by_path=embedding_by_path)
        if cached is not None:
            self._references_by_type = cached
            return cached
        output: dict[str, list[_ReferenceImage]] = {}
        for entity, relative_path, path, display_name in _reference_inputs(self.knowledge):
            descriptor = _reference_descriptor_for_path(path)
            if descriptor is None:
                continue
            output.setdefault(entity.type, []).append(
                _ReferenceImage(
                    entity_id=entity.id,
                    entity_name=display_name,
                    entity_type=entity.type,
                    local_path=path,
                    descriptor=descriptor,
                    embedding=embedding_by_path.get(relative_path),
                )
            )
        self._references_by_type = output
        return output

    def _reference_display_name(self, entity: HuntEntity, relative_path: str) -> str | None:
        for row in self.knowledge.media:
            if str(row.get("local_path") or "") != relative_path:
                continue
            reference_name = str(row.get("alt") or row.get("source_url") or relative_path)
            resolved = self.knowledge.resolve_equipment(reference_name, entity_types={entity.type})
            if resolved and resolved.entity.id != entity.id:
                return None
            return resolved.display_name if resolved else entity.name
        return entity.name


def recompute_reference_descriptor_index(knowledge: HuntKnowledgeService) -> int:
    references: list[_ReferenceImage] = []
    for entity, relative_path, path, display_name in _reference_inputs(knowledge):
        descriptor = _reference_descriptor_for_path(path)
        if descriptor is None:
            continue
        references.append(
            _ReferenceImage(
                entity_id=entity.id,
                entity_name=display_name,
                entity_type=entity.type,
                local_path=path,
                descriptor=descriptor,
            )
        )
    _write_reference_descriptor_index(knowledge.pack_dir, references)
    return len(references)


def recompute_reference_embedding_index(
    knowledge: HuntKnowledgeService,
    embedder: HuggingFaceMultimodalEmbedder,
    *,
    batch_size: int = 32,
    progress_callback: Any | None = None,
) -> int:
    rows: list[dict[str, Any]] = []
    icons: list[np.ndarray] = []
    for entity, relative_path, path, display_name in _reference_inputs(knowledge):
        image = _read_image(path)
        if image is None:
            continue
        icon = reference_icon_for_matching(image)
        icons.append(icon)
        rows.append(
            {
                "version": HUD_REFERENCE_EMBEDDING_VERSION,
                "preprocessing": "hud-reference-grayscale-v2",
                "model": embedder.config.model_name,
                "dimension": embedder.dimension,
                "entity_id": entity.id,
                "entity_name": display_name,
                "entity_type": entity.type,
                "relative_path": relative_path,
            }
        )
    vectors: list[np.ndarray] = []
    safe_batch_size = max(1, int(batch_size))
    total = len(icons)
    for start in range(0, total, safe_batch_size):
        batch = icons[start : start + safe_batch_size]
        vectors.extend(embed_icons_with_model(embedder, batch))
        if callable(progress_callback):
            progress_callback(min(start + len(batch), total), total)
    _write_reference_embedding_index(knowledge.pack_dir, rows, vectors)
    return len(rows)


def load_reference_embedding_index(
    knowledge: HuntKnowledgeService,
    *,
    expected_dimension: int | None = None,
) -> dict[str, np.ndarray]:
    embedding_path = knowledge.pack_dir / HUD_REFERENCE_EMBEDDING_FILE
    metadata_path = knowledge.pack_dir / HUD_REFERENCE_EMBEDDING_METADATA_FILE
    if not embedding_path.exists() or not metadata_path.exists():
        return {}
    try:
        loaded = np.load(embedding_path)
        embeddings = loaded["embeddings"].astype(np.float32, copy=False)
    except Exception:
        return {}
    rows: list[dict[str, Any]] = []
    try:
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return {}
    if len(rows) != embeddings.shape[0]:
        return {}
    if expected_dimension is not None and embeddings.ndim == 2 and int(embeddings.shape[1]) != int(expected_dimension):
        return {}
    output: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows):
        if int(row.get("version") or 0) != HUD_REFERENCE_EMBEDDING_VERSION:
            continue
        relative_path = str(row.get("relative_path") or "")
        if relative_path:
            output[relative_path] = embeddings[index]
    return output


def _write_reference_embedding_index(pack_dir: Path, rows: list[dict[str, Any]], vectors: list[np.ndarray]) -> None:
    pack_dir.mkdir(parents=True, exist_ok=True)
    if vectors:
        embeddings = np.stack(vectors).astype(np.float32)
    else:
        embeddings = np.zeros((0, 0), dtype=np.float32)
    np.savez_compressed(pack_dir / HUD_REFERENCE_EMBEDDING_FILE, embeddings=embeddings)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    (pack_dir / HUD_REFERENCE_EMBEDDING_METADATA_FILE).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def embed_icon_with_model(embedder: HuggingFaceMultimodalEmbedder, icon: np.ndarray) -> np.ndarray:
    return embed_icons_with_model(embedder, [icon])[0]


def embed_icons_with_model(embedder: HuggingFaceMultimodalEmbedder, icons: list[np.ndarray]) -> list[np.ndarray]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to render HUD icons for embedding.") from exc
    if not icons:
        return []
    with tempfile.TemporaryDirectory() as directory:
        paths: list[str] = []
        for index, icon in enumerate(icons):
            rendered = np.clip(icon * 255.0, 0, 255).astype(np.uint8)
            canvas = np.zeros((rendered.shape[0], rendered.shape[1], 3), dtype=np.uint8)
            canvas[:, :, 0] = rendered
            canvas[:, :, 1] = rendered
            canvas[:, :, 2] = rendered
            path = str(Path(directory) / f"hud-icon-{index:04d}.png")
            if not cv2.imwrite(path, canvas):
                raise RuntimeError("Unable to render temporary HUD icon image for embedding.")
            paths.append(path)
        vectors = [np.asarray(vector, dtype=np.float32) for vector in embedder.embed_image_paths(paths)]
    output: list[np.ndarray] = []
    for vector in vectors:
        norm = float(np.linalg.norm(vector))
        output.append(vector / norm if norm else vector)
    return output


def load_reference_descriptor_index(
    knowledge: HuntKnowledgeService,
    *,
    embedding_by_path: dict[str, np.ndarray] | None = None,
) -> dict[str, list[_ReferenceImage]] | None:
    descriptor_path = knowledge.pack_dir / HUD_REFERENCE_DESCRIPTOR_FILE
    metadata_path = knowledge.pack_dir / HUD_REFERENCE_DESCRIPTOR_METADATA_FILE
    if not descriptor_path.exists() or not metadata_path.exists():
        return None
    try:
        loaded = np.load(descriptor_path)
        icons = loaded["icons"].astype(np.float32, copy=False)
        edges = loaded["edges"].astype(np.float32, copy=False)
        features = loaded["features"].astype(np.float32, copy=False)
        aspect_ratios = loaded["aspect_ratios"].astype(np.float32, copy=False)
    except Exception:
        return None
    rows: list[dict[str, Any]] = []
    try:
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return None
    count = len(rows)
    if not count or icons.shape[0] != count or edges.shape[0] != count or features.shape[0] != count or aspect_ratios.shape[0] != count:
        return None
    output: dict[str, list[_ReferenceImage]] = {}
    for index, row in enumerate(rows):
        if int(row.get("version") or 0) != HUD_REFERENCE_DESCRIPTOR_VERSION:
            return None
        entity_type = str(row.get("entity_type") or "")
        relative_path = str(row.get("relative_path") or "")
        if not entity_type or not relative_path:
            return None
        reference = _ReferenceImage(
            entity_id=str(row.get("entity_id") or ""),
            entity_name=str(row.get("entity_name") or ""),
            entity_type=entity_type,
            local_path=knowledge.pack_dir / relative_path,
            descriptor=_VisualDescriptor(
                icon=icons[index],
                edges=edges[index],
                feature=features[index],
                aspect_ratio=float(aspect_ratios[index]),
            ),
            embedding=(embedding_by_path or {}).get(relative_path),
        )
        output.setdefault(entity_type, []).append(reference)
    return output


def _write_reference_descriptor_index(pack_dir: Path, references: list[_ReferenceImage]) -> None:
    pack_dir.mkdir(parents=True, exist_ok=True)
    if references:
        icons = np.stack([reference.descriptor.icon for reference in references]).astype(np.float32)
        edges = np.stack([reference.descriptor.edges for reference in references]).astype(np.float32)
        features = np.stack([reference.descriptor.feature for reference in references]).astype(np.float32)
        aspect_ratios = np.asarray([reference.descriptor.aspect_ratio for reference in references], dtype=np.float32)
    else:
        icons = np.zeros((0, 48, 96), dtype=np.float32)
        edges = np.zeros((0, 48, 96), dtype=np.float32)
        features = np.zeros((0, 48 * 96 * 2 + 96 + 48), dtype=np.float32)
        aspect_ratios = np.zeros((0,), dtype=np.float32)
    np.savez_compressed(
        pack_dir / HUD_REFERENCE_DESCRIPTOR_FILE,
        icons=icons,
        edges=edges,
        features=features,
        aspect_ratios=aspect_ratios,
    )
    lines = []
    for reference in references:
        try:
            relative_path = str(reference.local_path.relative_to(pack_dir))
        except ValueError:
            relative_path = str(reference.local_path)
        lines.append(
            json.dumps(
                {
                    "version": HUD_REFERENCE_DESCRIPTOR_VERSION,
                    "preprocessing": "hud-reference-grayscale-v2",
                    "entity_id": reference.entity_id,
                    "entity_name": reference.entity_name,
                    "entity_type": reference.entity_type,
                    "relative_path": relative_path,
                },
                sort_keys=True,
            )
        )
    (pack_dir / HUD_REFERENCE_DESCRIPTOR_METADATA_FILE).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _reference_inputs(knowledge: HuntKnowledgeService) -> list[tuple[HuntEntity, str, Path, str]]:
    output: list[tuple[HuntEntity, str, Path, str]] = []
    resolver = HudLoadoutDetector(knowledge)
    records = knowledge.reference_image_records({"weapon", "tool", "consumable"})
    for entity, relative_path in records:
        path = knowledge.pack_dir / relative_path
        if not path.exists() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        display_name = resolver._reference_display_name(entity, relative_path)
        if display_name is None:
            continue
        output.append((entity, relative_path, path, display_name))
    return output


def _reference_descriptor_for_path(path: Path) -> _VisualDescriptor | None:
    image = _read_image(path)
    if image is None:
        return None
    return reference_visual_descriptor(image)


def _primary_image_path(pack_dir: Path, entity: HuntEntity) -> Path | None:
    for relative in entity.image_paths:
        candidate = pack_dir / relative
        if candidate.exists() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            return candidate
    return None


def relative_box_to_pixels(box: RelativeBox, width: int, height: int) -> tuple[int, int, int, int]:
    x_min = int(round(_clamp(box.x_min) * width))
    y_min = int(round(_clamp(box.y_min) * height))
    x_max = int(round(_clamp(box.x_max) * width))
    y_max = int(round(_clamp(box.y_max) * height))
    x_min, x_max = sorted((max(0, x_min), min(width, x_max)))
    y_min, y_max = sorted((max(0, y_min), min(height, y_max)))
    return x_min, y_min, x_max, y_max


def crop_relative(image: np.ndarray, box: RelativeBox) -> np.ndarray:
    height, width = image.shape[:2]
    x_min, y_min, x_max, y_max = relative_box_to_pixels(box, width, height)
    return image[y_min:y_max, x_min:x_max]


def _slot_candidate_boxes(slot: HudSlotConfig) -> tuple[RelativeBox, ...]:
    return (slot.box, *slot.box_variants)


def _hud_icon_crop_variants_for_types(allowed_types: Iterable[str]) -> tuple[_HudIconCropVariant, ...]:
    type_set = set(allowed_types)
    if type_set == {"weapon"}:
        return HUD_MATCH_ICON_CROP_VARIANTS
    if type_set == {"tool"}:
        return HUD_TOOL_ICON_CROP_VARIANTS
    if type_set == {"consumable"}:
        return HUD_CONSUMABLE_ICON_CROP_VARIANTS
    return HUD_EQUIPMENT_ICON_CROP_VARIANTS


def _crop_groups_have_visible_icon(
    crop_groups: Iterable[Iterable[np.ndarray]],
    allowed_types: Iterable[str],
) -> bool:
    scores = [_crop_group_icon_presence_score(crops, allowed_types) for crops in crop_groups]
    return max(scores, default=0.0) >= HUD_EMPTY_SLOT_PRESENCE_THRESHOLD


def _crop_group_icon_presence_score(crops: Iterable[np.ndarray], allowed_types: Iterable[str]) -> float:
    per_frame: list[float] = []
    variants = _hud_icon_crop_variants_for_types(allowed_types)
    for crop in crops:
        if crop.size == 0:
            continue
        region_scores = []
        for variant in variants:
            region = _crop_hud_match_icon_region(crop, variant=variant)
            region_scores.append(_region_icon_presence_score(region))
        per_frame.append(max(region_scores, default=0.0))
    if not per_frame:
        return 0.0
    if len(per_frame) >= 3:
        return float(np.median(per_frame))
    return float(max(per_frame))


def _region_icon_presence_score(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0
    try:
        import cv2  # type: ignore
    except ImportError:
        gray = _to_gray01(region)
        return float(min(1.0, gray.std() / 0.06))

    gray = _to_gray01(region)
    if gray.size == 0:
        return 0.0
    low = float(np.percentile(gray, 5))
    high = float(np.percentile(gray, 99.5))
    contrast = max(0.0, high - low)
    std = float(np.std(gray))
    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    edges = cv2.Canny(gray_u8, 22, 80).astype(np.float32) / 255.0
    edge_density = float(edges.mean())
    foreground_threshold = low + max(0.08, contrast * 0.62)
    foreground_density = float((gray > foreground_threshold).mean())
    if contrast < 0.025 and edge_density < 0.004 and foreground_density < 0.002:
        return 0.0
    return float(
        min(1.0, contrast / 0.18) * 0.42
        + min(1.0, edge_density / 0.045) * 0.32
        + min(1.0, std / 0.045) * 0.18
        + min(1.0, foreground_density / 0.035) * 0.08
    )


def active_highlight_score(crop: np.ndarray) -> float:
    if crop.size == 0:
        return 0.0
    rgb = _to_rgb(crop).astype(np.float32) / 255.0
    brightness = rgb.mean(axis=2)
    channel_spread = rgb.max(axis=2) - rgb.min(axis=2)
    grayish = 1.0 - np.clip(channel_spread * 3.0, 0.0, 1.0)
    mid_bright = np.clip((brightness - 0.18) / 0.5, 0.0, 1.0)
    return float((grayish * mid_bright).mean())


def visual_feature(image: np.ndarray, *, size: tuple[int, int] = (96, 48)) -> np.ndarray:
    return visual_descriptor(image, size=size).feature


def visual_descriptor(image: np.ndarray, *, size: tuple[int, int] = (96, 48)) -> _VisualDescriptor:
    try:
        import cv2  # type: ignore
    except ImportError:
        feature = _fallback_feature(image, size=size)
        empty = np.zeros((size[1], size[0]), dtype=np.float32)
        return _VisualDescriptor(icon=empty, edges=empty, feature=feature, aspect_ratio=size[0] / size[1])
    icon, aspect_ratio = preprocess_icon_for_matching(image, size=size, include_aspect=True)
    return _descriptor_from_icon(icon, aspect_ratio)


def average_auto_levels_visual_descriptor(crops: Iterable[np.ndarray], *, size: tuple[int, int] = (96, 48)) -> _VisualDescriptor:
    maps: list[np.ndarray] = []
    aspect_ratios: list[float] = []
    first_shape: tuple[int, int] | None = None
    try:
        import cv2  # type: ignore
    except ImportError:
        first = next(iter(crops), np.zeros((size[1], size[0]), dtype=np.float32))
        return visual_descriptor(first, size=size)
    for crop in crops:
        if crop.size == 0:
            continue
        region = _crop_hud_match_icon_region(crop)
        icon_map = _hud_auto_levels_map(region)
        if first_shape is None:
            first_shape = icon_map.shape
        if icon_map.shape != first_shape:
            icon_map = cv2.resize(icon_map, (first_shape[1], first_shape[0]), interpolation=cv2.INTER_AREA)
        maps.append(icon_map)
        aspect_ratios.append(icon_map.shape[1] / max(icon_map.shape[0], 1))
    if not maps:
        empty = np.zeros((size[1], size[0]), dtype=np.float32)
        return _VisualDescriptor(icon=empty, edges=empty, feature=empty.flatten(), aspect_ratio=size[0] / size[1])
    averaged = np.mean(np.stack(maps, axis=0), axis=0)
    aspect_ratio = float(np.median(aspect_ratios)) if aspect_ratios else averaged.shape[1] / max(averaged.shape[0], 1)
    icon = np.clip(_resize_letterbox(averaged, size).astype(np.float32), 0.0, 1.0)
    return _descriptor_from_icon(icon, aspect_ratio)


def multi_crop_hud_visual_descriptors(
    crops: Iterable[np.ndarray],
    *,
    size: tuple[int, int] = (96, 48),
    variants: tuple[_HudIconCropVariant, ...] = HUD_MATCH_ICON_CROP_VARIANTS,
    contour_padding_ratio: float | None = None,
    use_ocr_key_binding_mask: bool = False,
) -> list[_VisualDescriptor]:
    crop_list = [crop for crop in crops if crop.size]
    if not crop_list:
        empty = np.zeros((size[1], size[0]), dtype=np.float32)
        return [_VisualDescriptor(icon=empty, edges=empty, feature=empty.flatten(), aspect_ratio=size[0] / size[1])]
    padding_ratio = (
        HUD_WEAPON_CONTOUR_PADDING_RATIO
        if contour_padding_ratio is None and variants == HUD_MATCH_ICON_CROP_VARIANTS
        else HUD_EQUIPMENT_CONTOUR_PADDING_RATIO
        if contour_padding_ratio is None
        else contour_padding_ratio
    )
    suppress_key_binding_text = variants == HUD_EQUIPMENT_ICON_CROP_VARIANTS
    descriptors = []
    for variant in variants:
        descriptor = _persistent_detail_visual_descriptor(
            crop_list,
            variant=variant,
            size=size,
            contour_padding_ratio=padding_ratio,
            suppress_key_binding_text=suppress_key_binding_text,
            use_ocr_key_binding_mask=use_ocr_key_binding_mask,
        )
        descriptors.append(descriptor)
    usable = [descriptor for descriptor in descriptors if float(descriptor.icon.max()) > 0.02]
    if usable:
        return usable
    fallback = average_auto_levels_visual_descriptor(crop_list, size=size)
    return [fallback]


def _persistent_detail_visual_descriptor(
    crops: Iterable[np.ndarray],
    *,
    variant: _HudIconCropVariant,
    size: tuple[int, int],
    contour_padding_ratio: float = HUD_WEAPON_CONTOUR_PADDING_RATIO,
    suppress_key_binding_text: bool = False,
    use_ocr_key_binding_mask: bool = False,
) -> _VisualDescriptor:
    try:
        import cv2  # type: ignore
    except ImportError:
        return average_auto_levels_visual_descriptor(crops, size=size)

    maps: list[np.ndarray] = []
    first_shape: tuple[int, int] | None = None
    ocr_mask: np.ndarray | None = None
    ocr_attempts = 0
    used_ocr_mask = False
    for crop in crops:
        if crop.size == 0:
            continue
        region = _crop_hud_match_icon_region(crop, variant=variant)
        detail = _hud_local_detail_map(region)
        if suppress_key_binding_text and use_ocr_key_binding_mask:
            if ocr_mask is None and ocr_attempts < 2:
                candidate_mask = _key_binding_text_mask_from_ocr(region)
                ocr_attempts += 1
                if candidate_mask is not None and int(candidate_mask.sum()) > 0:
                    ocr_mask = candidate_mask
            if ocr_mask is not None:
                mask = ocr_mask
                if mask.shape != detail.shape:
                    mask = cv2.resize(mask, (detail.shape[1], detail.shape[0]), interpolation=cv2.INTER_NEAREST)
                detail = detail.copy()
                detail[mask > 0] = 0.0
                used_ocr_mask = True
        if first_shape is None:
            first_shape = detail.shape
        if detail.shape != first_shape:
            detail = cv2.resize(detail, (first_shape[1], first_shape[0]), interpolation=cv2.INTER_AREA)
        maps.append(detail)
    if not maps:
        return average_auto_levels_visual_descriptor(crops, size=size)

    stack = np.stack(maps, axis=0)
    foreground = np.median(stack, axis=0) * np.exp(-np.std(stack, axis=0) * 4.0)
    baseline = float(np.percentile(foreground, 55))
    foreground = np.maximum(foreground - baseline, 0.0)
    high = float(np.percentile(foreground, 99.2))
    if high > 1e-6:
        foreground = np.clip(foreground / high, 0.0, 1.0)
    if suppress_key_binding_text and not used_ocr_mask:
        foreground = _suppress_upper_key_binding_text(foreground)
    lower_text_start_ratio = 0.72 if suppress_key_binding_text and variant.bottom >= 0.995 else 0.45
    foreground = _crop_primary_contour(
        foreground,
        prefer_upper=True,
        pad_ratio=contour_padding_ratio,
        lower_text_start_ratio=lower_text_start_ratio,
        bottom_text_start_ratio=lower_text_start_ratio,
        include_weak_support=suppress_key_binding_text,
    )
    if suppress_key_binding_text:
        boosted_foreground = _normalize_selected_icon_foreground(foreground)
        foreground = np.maximum(foreground, boosted_foreground * 0.55)
    aspect_ratio = foreground.shape[1] / max(foreground.shape[0], 1)
    icon = np.clip(_resize_letterbox(foreground, size).astype(np.float32), 0.0, 1.0)
    return _descriptor_from_icon(icon, float(aspect_ratio))


def reference_visual_descriptor(image: np.ndarray, *, size: tuple[int, int] = (96, 48)) -> _VisualDescriptor:
    try:
        import cv2  # type: ignore
    except ImportError:
        feature = _fallback_feature(image, size=size)
        empty = np.zeros((size[1], size[0]), dtype=np.float32)
        return _VisualDescriptor(icon=empty, edges=empty, feature=feature, aspect_ratio=size[0] / size[1])
    icon, aspect_ratio = reference_icon_for_matching(image, size=size, include_aspect=True)
    return _descriptor_from_icon(icon, aspect_ratio)


def _descriptor_from_icon(icon: np.ndarray, aspect_ratio: float) -> _VisualDescriptor:
    try:
        import cv2  # type: ignore
    except ImportError:
        feature = icon.flatten()
        norm = np.linalg.norm(feature)
        return _VisualDescriptor(icon=icon, edges=np.zeros_like(icon), feature=feature / norm if norm else feature, aspect_ratio=aspect_ratio)
    edges = cv2.Canny((icon * 255.0).astype(np.uint8), 30, 100).astype(np.float32) / 255.0
    horizontal_projection = icon.mean(axis=0)
    vertical_projection = icon.mean(axis=1)
    feature = np.concatenate(
        [
            icon.flatten() * 0.55,
            edges.flatten() * 0.65,
            horizontal_projection * 0.35,
            vertical_projection * 0.35,
        ]
    )
    norm = np.linalg.norm(feature)
    normalized = feature / norm if norm else feature
    return _VisualDescriptor(icon=icon, edges=edges, feature=normalized, aspect_ratio=aspect_ratio)


def temporal_visual_descriptor(
    crops: Iterable[np.ndarray],
    *,
    size: tuple[int, int] = (96, 48),
    use_robust_pca: bool = True,
) -> _VisualDescriptor:
    try:
        import cv2  # type: ignore
    except ImportError:
        first = next(iter(crops), np.zeros((size[1], size[0]), dtype=np.float32))
        return visual_descriptor(first, size=size)
    crop_list = [crop for crop in crops if crop.size]
    if not crop_list:
        empty = np.zeros((size[1], size[0]), dtype=np.float32)
        return _VisualDescriptor(icon=empty, edges=empty, feature=empty.flatten(), aspect_ratio=size[0] / size[1])
    if len(crop_list) == 1:
        return visual_descriptor(crop_list[0], size=size)
    icon, aspect_ratio = temporal_foreground_icon_map(crop_list, size=size, include_aspect=True, use_robust_pca=use_robust_pca)
    return _descriptor_from_icon(icon, aspect_ratio)


def temporal_foreground_icon_map(
    crops: Iterable[np.ndarray],
    *,
    size: tuple[int, int] = (96, 48),
    include_aspect: bool = False,
    use_robust_pca: bool = True,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Separate fixed HUD icon strokes from changing scene background across frames."""
    try:
        import cv2  # type: ignore
    except ImportError:
        first = next(iter(crops), np.zeros((size[1], size[0]), dtype=np.float32))
        icon = _fallback_icon_map(first, size=size)
        return (icon, size[0] / size[1]) if include_aspect else icon

    detail_maps: list[np.ndarray] = []
    first_shape: tuple[int, int] | None = None
    for crop in crops:
        if crop.size == 0:
            continue
        detail = _hud_local_detail_map(crop)
        if first_shape is None:
            first_shape = detail.shape
        if detail.shape != first_shape:
            detail = cv2.resize(detail, (first_shape[1], first_shape[0]), interpolation=cv2.INTER_AREA)
        detail_maps.append(detail)
    if not detail_maps:
        empty = np.zeros((size[1], size[0]), dtype=np.float32)
        return (empty, size[0] / size[1]) if include_aspect else empty

    stack = np.stack(detail_maps, axis=0)
    foreground = _temporal_hud_foreground_score(stack, use_robust_pca=use_robust_pca)
    baseline = float(np.percentile(foreground, 58))
    foreground = np.maximum(foreground - baseline, 0.0)
    percentile = float(np.percentile(foreground, 99))
    if percentile > 1e-6:
        foreground = np.clip(foreground / percentile, 0.0, 1.0)
    foreground = _crop_primary_contour(foreground, prefer_upper=True)
    aspect_ratio = foreground.shape[1] / max(foreground.shape[0], 1)
    icon = np.clip(_resize_letterbox(foreground, size).astype(np.float32), 0.0, 1.0)
    return (icon, float(aspect_ratio)) if include_aspect else icon


def hybrid_visual_score(left: _VisualDescriptor, right: _VisualDescriptor) -> float:
    feature_score = cosine(left.feature, right.feature)
    icon_score = cosine(left.icon.flatten(), right.icon.flatten())
    edge_score = cosine(left.edges.flatten(), right.edges.flatten())
    aspect_score = min(left.aspect_ratio, right.aspect_ratio) / max(left.aspect_ratio, right.aspect_ratio, 1e-6)
    return float(feature_score * 0.58 + icon_score * 0.22 + edge_score * 0.15 + aspect_score * 0.05)


def _temporal_hud_foreground_score(stack: np.ndarray, *, use_robust_pca: bool) -> np.ndarray:
    median_detail = np.median(stack, axis=0)
    temporal_std = np.std(stack, axis=0)
    stable_weight = np.exp(-temporal_std * 4.5)
    edge_persistence = _edge_persistence_map(stack)
    foreground = median_detail * stable_weight * (0.35 + edge_persistence * 0.65)

    if use_robust_pca and stack.shape[0] >= 4:
        low_rank = _robust_low_rank_detail(stack)
        low_rank_edges = _edge_persistence_map(low_rank[None, :, :])
        robust_foreground = low_rank * stable_weight * (0.40 + low_rank_edges * 0.60)
        foreground = np.maximum(foreground, robust_foreground * 0.85)

    return np.clip(foreground.astype(np.float32), 0.0, 1.0)


def _edge_persistence_map(stack: np.ndarray) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError:
        return np.zeros(stack.shape[1:], dtype=np.float32)
    edge_maps: list[np.ndarray] = []
    for frame in stack:
        frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        edges = cv2.Canny(frame_u8, 20, 80).astype(np.float32) / 255.0
        edges = cv2.dilate(edges.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1).astype(np.float32)
        edge_maps.append(edges)
    if not edge_maps:
        return np.zeros(stack.shape[1:], dtype=np.float32)
    persistence = np.mean(np.stack(edge_maps, axis=0), axis=0)
    return np.clip(persistence / 0.45, 0.0, 1.0).astype(np.float32)


def _robust_low_rank_detail(stack: np.ndarray) -> np.ndarray:
    frame_count, height, width = stack.shape
    matrix = stack.reshape(frame_count, height * width).T.astype(np.float32)
    low_rank = _robust_pca_low_rank(matrix, max_iter=35, tol=1e-5)
    low_rank_stack = low_rank.T.reshape(frame_count, height, width)
    stable = np.median(low_rank_stack, axis=0)
    stable = np.maximum(stable, 0.0)
    percentile = float(np.percentile(stable, 99))
    if percentile > 1e-6:
        stable = np.clip(stable / percentile, 0.0, 1.0)
    return stable.astype(np.float32)


def _robust_pca_low_rank(matrix: np.ndarray, *, max_iter: int, tol: float) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norm = float(np.linalg.norm(matrix, ord="fro"))
    if norm <= 1e-8:
        return np.zeros_like(matrix, dtype=np.float32)
    rows, cols = matrix.shape
    lam = 1.0 / np.sqrt(float(max(rows, cols)))
    try:
        spectral_norm = float(np.linalg.norm(matrix, ord=2))
    except np.linalg.LinAlgError:
        spectral_norm = norm
    inf_norm = float(np.max(np.abs(matrix))) / max(lam, 1e-8)
    dual_norm = max(spectral_norm, inf_norm, 1e-8)
    y = matrix / dual_norm
    mu = 1.25 / max(spectral_norm, 1e-8)
    mu_bar = mu * 1e7
    rho = 1.5
    low_rank = np.zeros_like(matrix, dtype=np.float32)
    sparse = np.zeros_like(matrix, dtype=np.float32)

    for _ in range(max_iter):
        u, singular_values, vt = np.linalg.svd(matrix - sparse + y / mu, full_matrices=False)
        thresholded = np.maximum(singular_values - 1.0 / mu, 0.0)
        rank = int(np.sum(thresholded > 0.0))
        if rank:
            low_rank = (u[:, :rank] * thresholded[:rank]) @ vt[:rank, :]
        else:
            low_rank = np.zeros_like(matrix, dtype=np.float32)
        sparse = _soft_threshold(matrix - low_rank + y / mu, lam / mu)
        residual = matrix - low_rank - sparse
        if float(np.linalg.norm(residual, ord="fro")) / norm < tol:
            break
        y = y + mu * residual
        mu = min(mu * rho, mu_bar)
    return low_rank.astype(np.float32)


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _margin_adjusted_confidence(score: float, margin: float) -> float:
    if margin >= 0.035:
        return float(score)
    return float(score * max(0.25, margin / 0.035))


def _top_entity_score_and_margin(scored: Iterable[tuple[float, _ReferenceImage]]) -> tuple[float, float]:
    best_by_entity: dict[str, float] = {}
    for score, reference in scored:
        best_by_entity[reference.entity_id] = max(best_by_entity.get(reference.entity_id, 0.0), float(score))
    ranked = sorted(best_by_entity.values(), reverse=True)
    if not ranked:
        return 0.0, 0.0
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
    return float(ranked[0]), float(margin)


def _visual_match_is_decisive(score: float, margin: float) -> bool:
    return score >= 0.50 and margin >= 0.06


def preprocess_icon_for_matching(
    image: np.ndarray,
    *,
    size: tuple[int, int] = (96, 48),
    include_aspect: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Extract a foreground icon map from either a wiki reference or a HUD crop."""
    try:
        import cv2  # type: ignore
    except ImportError:
        icon = _fallback_icon_map(image, size=size)
        return (icon, size[0] / size[1]) if include_aspect else icon

    if _has_alpha(image):
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        foreground = alpha * (0.45 + gray * 0.55)
        foreground = _crop_primary_contour(foreground, prefer_upper=True)
        aspect_ratio = foreground.shape[1] / max(foreground.shape[0], 1)
    else:
        foreground = _hud_auto_levels_map(_crop_hud_match_icon_region(image))
        aspect_ratio = foreground.shape[1] / max(foreground.shape[0], 1)
    resized = _resize_letterbox(foreground, size)
    icon = np.clip(resized.astype(np.float32), 0.0, 1.0)
    return (icon, float(aspect_ratio)) if include_aspect else icon


def reference_icon_for_matching(
    image: np.ndarray,
    *,
    size: tuple[int, int] = (96, 48),
    include_aspect: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Render a wiki reference image into the same grayscale HUD-icon domain as slot crops."""
    try:
        import cv2  # type: ignore
    except ImportError:
        icon = _fallback_icon_map(image, size=size)
        return (icon, size[0] / size[1]) if include_aspect else icon

    foreground = _reference_foreground_map(image)
    foreground = _crop_primary_contour(foreground, prefer_upper=False)
    aspect_ratio = foreground.shape[1] / max(foreground.shape[0], 1)
    foreground = _hud_auto_levels_map(foreground)
    icon = np.clip(_resize_letterbox(foreground, size).astype(np.float32), 0.0, 1.0)
    return (icon, float(aspect_ratio)) if include_aspect else icon


def _reference_foreground_map(image: np.ndarray) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError:
        return _to_rgb(image).mean(axis=2).astype(np.float32) / 255.0

    gray = _to_gray(image).astype(np.float32) / 255.0
    if _has_alpha(image):
        mask = image[:, :, 3].astype(np.float32) / 255.0
    else:
        rgb = _to_rgb(image).astype(np.float32) / 255.0
        border_gray = _border_values(gray)
        background_gray = float(np.median(border_gray)) if border_gray.size else float(np.median(gray))
        gray_delta = np.abs(gray - background_gray)
        border_rgb = _border_pixels(rgb)
        background_rgb = np.median(border_rgb, axis=0) if border_rgb.size else np.median(rgb.reshape(-1, 3), axis=0)
        color_delta = np.linalg.norm(rgb - background_rgb.reshape(1, 1, 3), axis=2) / np.sqrt(3.0)
        mask = np.maximum(gray_delta, color_delta)
        threshold = max(float(np.percentile(mask, 65)), 0.03)
        spread = max(float(np.percentile(mask, 98)) - threshold, 1e-6)
        mask = np.clip((mask - threshold) / spread, 0.0, 1.0)

    masked_values = gray[mask > 0.05]
    if masked_values.size:
        low = float(np.percentile(masked_values, 5))
        high = float(np.percentile(masked_values, 95))
        normalized_gray = np.clip((gray - low) / max(high - low, 1e-6), 0.0, 1.0)
    else:
        normalized_gray = gray
    body = mask * (0.55 + normalized_gray * 0.30)
    edges = cv2.Canny((mask * 255.0).astype(np.uint8), 20, 80).astype(np.float32) / 255.0
    detail = cv2.Canny((normalized_gray * mask * 255.0).astype(np.uint8), 30, 100).astype(np.float32) / 255.0
    hud_icon = np.maximum(body, np.maximum(edges, detail) * 0.95)
    if hud_icon.max() > 1e-6:
        hud_icon = hud_icon / hud_icon.max()
    return np.clip(hud_icon.astype(np.float32), 0.0, 1.0)


def _hud_local_detail_map(image: np.ndarray) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError:
        return _to_rgb(image).mean(axis=2).astype(np.float32) / 255.0
    gray = _to_gray(image).astype(np.float32)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 4))
    equalized = clahe.apply(gray_u8).astype(np.float32)
    kernel_width = max(31, int(equalized.shape[1] * 0.22) | 1)
    kernel_height = max(9, int(equalized.shape[0] * 0.45) | 1)
    background = cv2.GaussianBlur(equalized, (kernel_width, kernel_height), 0)
    light_detail = np.maximum(equalized - background, 0.0)
    dark_detail = np.maximum(background - equalized, 0.0)
    if image.ndim == 3 and image.shape[2] >= 3:
        channels = image[:, :, :3].astype(np.float32)
        chroma = channels.max(axis=2) - channels.min(axis=2)
        chroma_background = cv2.GaussianBlur(chroma, (kernel_width, kernel_height), 0)
        chroma_detail = np.maximum(chroma - chroma_background, 0.0)
    else:
        chroma_detail = np.zeros_like(light_detail, dtype=np.float32)
    edges = cv2.Canny(gray_u8, 22, 80).astype(np.float32) / 255.0
    edges = cv2.dilate(edges.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1).astype(np.float32)
    detail = np.maximum.reduce([light_detail, dark_detail * 0.82, chroma_detail * 0.95, edges * 55.0])
    if detail.shape[0] >= 20:
        y = np.linspace(0.0, 1.0, detail.shape[0], dtype=np.float32)[:, None]
        detail *= np.where(y > 0.82, 0.15, 1.0)
    percentile = float(np.percentile(detail, 99.4))
    if percentile > 1e-6:
        detail = np.clip(detail / percentile, 0.0, 1.0)
    return detail.astype(np.float32)


def _crop_hud_match_icon_region(
    image: np.ndarray,
    *,
    bottom_ratio: float = HUD_MATCH_ICON_BOTTOM_RATIO,
    variant: _HudIconCropVariant | None = None,
) -> np.ndarray:
    if image.size == 0 or image.shape[0] < 4:
        return image
    if variant is not None:
        height, width = image.shape[:2]
        y_min = max(0, min(height - 1, int(round(height * variant.top))))
        y_max = max(y_min + 1, min(height, int(round(height * variant.bottom))))
        x_min = max(0, min(width - 1, int(round(width * variant.left))))
        x_max = max(x_min + 1, min(width, int(round(width * variant.right))))
        return image[y_min:y_max, x_min:x_max, ...]
    y_max = max(1, min(image.shape[0], int(round(image.shape[0] * bottom_ratio))))
    return image[:y_max, ...]


def _hud_auto_levels_map(image: np.ndarray, *, low_percentile: float = 1.0, high_percentile: float = 99.5) -> np.ndarray:
    gray = _to_gray01(image)
    low = float(np.percentile(gray, low_percentile))
    high = float(np.percentile(gray, high_percentile))
    return np.clip((gray - low) / max(high - low, 1e-6), 0.0, 1.0).astype(np.float32)


def _to_gray01(image: np.ndarray) -> np.ndarray:
    gray = _to_gray(image).astype(np.float32)
    if gray.size and float(gray.max()) <= 1.5:
        return np.clip(gray, 0.0, 1.0)
    return np.clip(gray / 255.0, 0.0, 1.0)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    numerator = float(np.dot(left, right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return numerator / denominator if denominator else 0.0


def _recognize_current_weapon_lines(crop: np.ndarray) -> list[OCRLine]:
    if crop.size == 0:
        return []
    try:
        import cv2  # type: ignore
    except ImportError:
        return []
    variants: list[np.ndarray] = [crop]
    gray = _to_gray(crop)
    if gray.size:
        scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        low = float(np.percentile(scaled, 1.0))
        high = float(np.percentile(scaled, 99.5))
        leveled = np.clip((scaled.astype(np.float32) - low) / max(high - low, 1e-6), 0.0, 1.0)
        leveled_u8 = np.clip(leveled * 255.0, 0, 255).astype(np.uint8)
        variants.append(cv2.cvtColor(leveled_u8, cv2.COLOR_GRAY2BGR))
        inverted = cv2.bitwise_not(leveled_u8)
        variants.append(cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR))
    lines: list[OCRLine] = []
    with tempfile.TemporaryDirectory() as directory:
        for index, image in enumerate(variants):
            path = Path(directory) / f"current-weapon-{index}.png"
            if not cv2.imwrite(str(path), image):
                continue
            try:
                lines.extend(recognize_text(path, timeout_seconds=8.0))
            except Exception:
                continue
    return sorted(_dedupe_ocr_lines(lines), key=lambda line: line.confidence, reverse=True)


def _slot_quantity_hint(crops: Iterable[np.ndarray]) -> str | None:
    crop_list = [crop for crop in crops if crop.size]
    if not crop_list:
        return None
    candidates: dict[str, float] = {}
    for crop in _slot_quantity_sample_crops(crop_list):
        for line in _recognize_slot_quantity_lines(crop):
            if line.confidence < 0.20:
                continue
            normalized = _normalize_slot_quantity_text(line.text)
            if normalized is None:
                continue
            candidates[normalized] = max(candidates.get(normalized, 0.0), float(line.confidence))
    if not candidates:
        return None
    return max(candidates.items(), key=lambda item: item[1])[0]


def _slot_quantity_sample_crops(crops: list[np.ndarray]) -> list[np.ndarray]:
    if len(crops) <= 4:
        return crops
    indices = sorted({0, len(crops) // 3, (len(crops) * 2) // 3, len(crops) - 1})
    return [crops[index] for index in indices]


def _recognize_slot_quantity_lines(crop: np.ndarray) -> list[OCRLine]:
    if crop.size == 0:
        return []
    try:
        import cv2  # type: ignore
    except ImportError:
        return []
    height, width = crop.shape[:2]
    if height < 10 or width < 10:
        return []
    quantity_region = crop[int(height * 0.58) : height, :]
    if quantity_region.size == 0:
        return []
    gray = _to_gray(quantity_region)
    scaled = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    low = float(np.percentile(scaled, 2.0))
    high = float(np.percentile(scaled, 99.4))
    leveled = np.clip((scaled.astype(np.float32) - low) / max(high - low, 1e-6), 0.0, 1.0)
    leveled_u8 = np.clip(leveled * 255.0, 0, 255).astype(np.uint8)
    variants = [
        cv2.cvtColor(leveled_u8, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(cv2.bitwise_not(leveled_u8), cv2.COLOR_GRAY2BGR),
    ]
    lines: list[OCRLine] = []
    with tempfile.TemporaryDirectory() as directory:
        for index, image in enumerate(variants):
            path = Path(directory) / f"slot-quantity-{index}.png"
            if not cv2.imwrite(str(path), image):
                continue
            try:
                lines.extend(recognize_text(path, timeout_seconds=4.0))
            except Exception:
                continue
    return sorted(_dedupe_ocr_lines(lines), key=lambda line: line.confidence, reverse=True)


def _normalize_slot_quantity_text(text: str) -> str | None:
    compact = re.sub(r"\s+", "", text.upper())
    if not compact:
        return None
    compact = compact.translate(str.maketrans({"I": "1", "L": "1", "|": "1", "Z": "2", "O": "0"}))
    compact = compact.replace("\\", "/")
    match = re.search(r"([0-9])[/]([0-9])", compact)
    if match:
        return f"{int(match.group(1))}/{int(match.group(2))}"
    match = re.search(r"([0-9]+)", compact)
    if match:
        return str(int(match.group(1)))
    return None


def _equipment_quantity_markers(entity: HuntEntity) -> dict[str, set[str]]:
    markers: dict[str, set[str]] = {"quantity": set(), "loaded_extra": set()}
    quantity = _normalize_slot_quantity_text(str(entity.key_values.get("Quantity") or ""))
    if quantity is not None:
        markers["quantity"].add(quantity)
    loaded = _normalize_slot_quantity_text(str(entity.key_values.get("Loaded") or ""))
    extra = _normalize_slot_quantity_text(str(entity.key_values.get("Extra") or ""))
    if loaded is not None and extra is not None:
        markers["loaded_extra"].add(f"{loaded}/{extra}")
    return markers


def _is_hud_calibration_reference(reference: _ReferenceImage) -> bool:
    return _is_hud_calibration_path(str(reference.local_path))


def _is_hud_calibration_path(path: str | None) -> bool:
    return "media/images/hud_calibration/" in str(path or "").replace("\\", "/")


def _is_weak_hud_calibration_match(matched_path: str | None, allowed_types: Iterable[str], confidence: float) -> bool:
    return (
        tuple(allowed_types) == ("consumable",)
        and _is_hud_calibration_path(matched_path)
        and confidence < HUD_CONSUMABLE_CALIBRATION_CONFIDENCE_THRESHOLD
    )


def _dedupe_ocr_lines(lines: Iterable[OCRLine]) -> list[OCRLine]:
    by_text: dict[str, OCRLine] = {}
    for line in lines:
        text = line.text.strip()
        key = text.lower()
        if not text:
            continue
        existing = by_text.get(key)
        if existing is None or line.confidence > existing.confidence:
            by_text[key] = line
    return list(by_text.values())


def detections_to_rows(
    clip_id: int,
    segment_id: int,
    timestamp: float,
    result: HudDetectionResult,
) -> list[dict[str, Any]]:
    snapshot = json.dumps(
        {
            "active": asdict(result.active_match) if result.active_match else None,
            "loadout": [asdict(match) for match in result.matches],
        },
        sort_keys=True,
    )
    rows = []
    for match in result.matches:
        rows.append(
            {
                "clip_id": clip_id,
                "segment_id": segment_id,
                "frame_path": result.frame_path,
                "timestamp": timestamp,
                "slot_key": match.slot_key,
                "is_active": int(match.is_active),
                "entity_id": match.entity_id,
                "entity_name": match.entity_name,
                "entity_type": match.entity_type,
                "confidence": match.confidence,
                "matched_image_path": match.matched_image_path,
                "loadout_snapshot": snapshot,
            }
        )
    return rows


def summarize_detections(rows: Iterable[Any]) -> dict[str, Any]:
    row_list = list(rows)
    if not row_list:
        return _empty_hud_summary()
    selected_rows = _select_representative_hud_rows(row_list)
    active_weapons: list[str] = []
    active_equipment: list[tuple[str, str | None]] = []
    loadout: list[str] = []
    weapon_slots: list[str] = []
    for row in selected_rows:
        name = _row_get(row, "entity_name")
        if not name:
            continue
        entity_type = str(_row_get(row, "entity_type") or "") or None
        loadout.append(str(name))
        if entity_type == "weapon" and str(_row_get(row, "slot_key") or "") in {"1", "2", "3"}:
            weapon_slots.append(str(name))
        if int(_row_get(row, "is_active") or 0):
            active_equipment.append((str(name), entity_type))
            if entity_type == "weapon":
                active_weapons.append(str(name))
    credible_weapon_loadout = len(_dedupe(weapon_slots)) >= 2
    deduped_active_weapons = _dedupe(active_weapons)
    active_equipment_name = active_equipment[0][0] if active_equipment else (deduped_active_weapons[0] if deduped_active_weapons else None)
    active_equipment_type = active_equipment[0][1] if active_equipment else ("weapon" if deduped_active_weapons else None)
    if active_equipment and any(str(_row_get(row, "slot_key") or "") == "current_ocr" for row in selected_rows):
        return {
            "active_weapon": deduped_active_weapons[0] if deduped_active_weapons else None,
            "active_equipment": active_equipment_name,
            "active_equipment_type": active_equipment_type,
            "loadout": _dedupe(loadout),
            "evidence": _hud_evidence_payload(selected_rows),
        }
    if not credible_weapon_loadout and not active_weapons and not active_equipment:
        return _empty_hud_summary()
    return {
        "active_weapon": deduped_active_weapons[0] if deduped_active_weapons else None,
        "active_equipment": active_equipment_name,
        "active_equipment_type": active_equipment_type,
        "loadout": _dedupe(loadout),
        "evidence": _hud_evidence_payload(selected_rows),
    }


def _empty_hud_summary() -> dict[str, Any]:
    return {
        "active_weapon": None,
        "active_equipment": None,
        "active_equipment_type": None,
        "loadout": [],
        "evidence": [],
    }


def _hud_evidence_payload(rows: Iterable[Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for row in rows:
        name = _row_get(row, "entity_name")
        if not name:
            continue
        evidence.append(
            {
                "segment_id": _int_or_none(_row_get(row, "segment_id")),
                "frame_path": _row_get(row, "frame_path"),
                "timestamp": _float_or_none(_row_get(row, "timestamp")),
                "slot_key": _row_get(row, "slot_key"),
                "is_active": bool(int(_row_get(row, "is_active") or 0)),
                "entity_id": _row_get(row, "entity_id"),
                "entity_name": str(name),
                "entity_type": _row_get(row, "entity_type"),
                "confidence": _float_or_none(_row_get(row, "confidence")),
                "matched_image_path": _row_get(row, "matched_image_path"),
            }
        )
    return evidence


def _select_representative_hud_rows(rows: list[Any]) -> list[Any]:
    by_frame: dict[tuple[float, str], list[Any]] = {}
    for row in rows:
        key = (float(_row_get(row, "timestamp") or 0.0), str(_row_get(row, "frame_path") or ""))
        by_frame.setdefault(key, []).append(row)
    candidates: list[tuple[float, float, int, list[Any]]] = []
    for (timestamp, _frame_path), frame_rows in by_frame.items():
        named = [row for row in frame_rows if _row_get(row, "entity_name")]
        if not named:
            continue
        avg_confidence = sum(float(_row_get(row, "confidence") or 0.0) for row in named) / len(named)
        current_ocr_active = any(
            int(_row_get(row, "is_active") or 0)
            and str(_row_get(row, "slot_key") or "") == "current_ocr"
            for row in named
        )
        active_equipment = any(
            int(_row_get(row, "is_active") or 0)
            for row in named
        )
        weapon_count = len(_dedupe(str(_row_get(row, "entity_name") or "") for row in named if _row_get(row, "entity_type") == "weapon"))
        active_rank = 2.0 if current_ocr_active else 1.0 if active_equipment else 0.0
        candidates.append((active_rank, float(weapon_count), avg_confidence, int(timestamp * 1000), frame_rows))
    if not candidates:
        return rows
    _, _, _, _, selected = max(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))
    return sorted(selected, key=lambda row: _slot_sort_key(str(_row_get(row, "slot_key") or "")))


def _slot_sort_key(slot_key: str) -> int:
    order = {"current_ocr": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "mouse5": 6, "7": 7, "8": 8, "9": 9, "0": 10}
    return order.get(slot_key, 99)


def _read_image(path: str | Path) -> np.ndarray | None:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return image
    return image


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image, image, image], axis=2)
    if image.shape[2] >= 3:
        return image[:, :, :3][:, :, ::-1]
    return np.repeat(image[:, :, :1], 3, axis=2)


def _to_gray(image: np.ndarray) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError:
        return _to_rgb(image).mean(axis=2).astype(np.uint8)
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)


def _has_alpha(image: np.ndarray) -> bool:
    return image.ndim == 3 and image.shape[2] == 4 and int(image[:, :, 3].max()) > 0


def _border_values(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=np.float32)
    top = values[0, :]
    bottom = values[-1, :]
    left = values[:, 0]
    right = values[:, -1]
    return np.concatenate([top, bottom, left, right]).astype(np.float32)


def _border_pixels(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=np.float32).reshape(0, 3)
    top = values[0, :, :]
    bottom = values[-1, :, :]
    left = values[:, 0, :]
    right = values[:, -1, :]
    return np.concatenate([top, bottom, left, right], axis=0).astype(np.float32)


def _trim_empty_border(gray: np.ndarray) -> np.ndarray:
    values = gray.astype(np.float32)
    mask = values < max(float(values.mean()) + 20.0, 245.0)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return gray
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    return gray[y_min:y_max, x_min:x_max]


def _trim_foreground_border(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    threshold = max(float(np.percentile(values, 88)) * 0.30, float(values.mean()) + float(values.std()) * 0.10)
    mask = values > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return values
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    pad = 3
    y_min = max(0, int(y_min) - pad)
    x_min = max(0, int(x_min) - pad)
    y_max = min(values.shape[0], int(y_max) + pad)
    x_max = min(values.shape[1], int(x_max) + pad)
    return values[y_min:y_max, x_min:x_max]


def _key_binding_text_mask_from_ocr(region: np.ndarray) -> np.ndarray | None:
    if region.size == 0 or region.shape[0] < 16 or region.shape[1] < 16:
        return None
    try:
        import cv2  # type: ignore
    except ImportError:
        return None

    height, width = region.shape[:2]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "hud-key-binding.png"
        prepared = _prepare_key_binding_ocr_image(region)
        if not cv2.imwrite(str(path), prepared):
            return None
        try:
            lines = recognize_text(path, timeout_seconds=4.0)
        except Exception:
            return None

    mask = np.zeros((height, width), dtype=np.uint8)
    for line in lines:
        if line.confidence < 0.22 or not _is_key_binding_text(line.text):
            continue
        x_min, y_min, x_max, y_max = _ocr_line_box_to_pixels(line, width=width, height=height)
        if y_min > int(round(height * 0.38)):
            continue
        if y_max - y_min > int(round(height * 0.34)):
            continue
        x_max = min(width, x_max + _key_binding_right_extension(line.text, y_max - y_min))
        text_mask = _tight_key_binding_text_mask(region, (x_min, y_min, x_max, y_max))
        if text_mask is not None:
            mask = np.maximum(mask, text_mask)
    if not int(mask.sum()):
        return None
    return mask


def _tight_key_binding_text_mask(region: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    height, width = region.shape[:2]
    x_min, y_min, x_max, y_max = box
    if x_min >= x_max or y_min >= y_max:
        return None

    pad_x = max(2, int(round(width * 0.020)))
    pad_top = max(1, int(round(height * 0.010)))
    x_min = max(0, x_min - pad_x)
    x_max = min(width, x_max + pad_x)
    y_min = max(0, y_min - pad_top)
    y_max = min(height, y_max)
    if x_min >= x_max or y_min >= y_max:
        return None

    roi = region[y_min:y_max, x_min:x_max]
    local = _hud_auto_levels_map(roi)
    if local.size == 0:
        return None
    scan_height = max(1, min(local.shape[0], int(round(height * 0.16))))
    scan = local[:scan_height, :]
    threshold = max(
        float(np.percentile(scan, 78)) * 0.80,
        float(scan.mean()) + float(scan.std()) * 0.28,
        0.38,
    )
    stroke_mask = (local >= threshold).astype(np.uint8)
    row_activity = stroke_mask.mean(axis=1)
    if not row_activity.size or float(row_activity.max()) <= 0.0:
        return None

    active_floor = max(0.010, float(row_activity.max()) * 0.08)
    active_rows = np.flatnonzero(row_activity >= active_floor)
    if not active_rows.size:
        return None
    text_top = int(active_rows[0])
    max_text_height = max(6, int(round(height * 0.13)))
    scan_end = min(local.shape[0], text_top + max_text_height + 4)
    text_bottom = min(local.shape[0], text_top + max_text_height)
    gap_floor = max(0.006, float(row_activity.max()) * 0.04)
    gap_run = 0
    for row in range(text_top + 4, scan_end):
        if row_activity[row] <= gap_floor:
            gap_run += 1
        else:
            gap_run = 0
        if gap_run >= 2:
            text_bottom = max(text_top + 1, row - gap_run + 1)
            break

    capped = np.zeros_like(stroke_mask, dtype=np.uint8)
    top = max(0, text_top - 1)
    bottom = min(stroke_mask.shape[0], text_bottom + 1)
    capped[top:bottom, :] = stroke_mask[top:bottom, :]
    labels, stats = _component_stats(capped)
    if not stats:
        return None
    kept = np.zeros_like(capped, dtype=np.uint8)
    for label, stat in stats:
        x, y, comp_width, comp_height, area = stat
        if area <= 1:
            continue
        if y >= bottom or comp_height > max_text_height:
            continue
        if comp_width > max(6, int(round(width * 0.62))):
            continue
        kept[labels == label] = 1
    if not int(kept.sum()):
        return None
    kept = cv2.dilate(kept, np.ones((2, 2), np.uint8), iterations=1)
    if bottom < kept.shape[0]:
        kept[bottom:, :] = 0
    coords = np.argwhere(kept > 0)
    if coords.size == 0:
        return None
    y_text_min, x_text_min = coords.min(axis=0)
    y_text_max, x_text_max = coords.max(axis=0) + 1
    x_text_pad = max(1, int(round(width * 0.010)))
    y_text_min = max(0, int(y_text_min) - 1)
    x_text_min = max(0, int(x_text_min) - x_text_pad)
    x_text_max = min(kept.shape[1], int(x_text_max) + x_text_pad)
    y_text_max = max(y_text_min + 1, min(bottom, int(y_text_max) - 2))
    tight_box = np.zeros_like(kept, dtype=np.uint8)
    tight_box[y_text_min:y_text_max, x_text_min:x_text_max] = 1

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = tight_box
    return mask


def _key_binding_right_extension(text: str, box_height: int) -> int:
    compact = re.sub(r"[^A-Z0-9]+", "", text.upper())
    if "OUSE" in compact and not compact[-1:].isdigit():
        return max(5, int(round(max(box_height, 1) * 0.75)))
    return 0


def _prepare_key_binding_ocr_image(region: np.ndarray) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError:
        return _to_rgb(region)
    gray = _to_gray(region)
    scale = 3 if max(gray.shape[:2], default=0) < 240 else 2
    enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    low = float(np.percentile(enlarged, 1.0))
    high = float(np.percentile(enlarged, 99.5))
    leveled = np.clip((enlarged.astype(np.float32) - low) / max(high - low, 1e-6), 0.0, 1.0)
    leveled_u8 = np.clip(leveled * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(leveled_u8, cv2.COLOR_GRAY2BGR)


def _ocr_line_box_to_pixels(line: OCRLine, *, width: int, height: int) -> tuple[int, int, int, int]:
    x, y_from_bottom, box_width, box_height = line.box
    x_min = int(round(_clamp(x) * width))
    x_max = int(round(_clamp(x + box_width) * width))
    y_min = int(round(_clamp(1.0 - y_from_bottom - box_height) * height))
    y_max = int(round(_clamp(1.0 - y_from_bottom) * height))
    x_min, x_max = sorted((max(0, x_min), min(width, x_max)))
    y_min, y_max = sorted((max(0, y_min), min(height, y_max)))
    return x_min, y_min, x_max, y_max


def _is_key_binding_text(text: str) -> bool:
    compact = re.sub(r"[^A-Z0-9]+", "", text.upper())
    if not compact:
        return False
    compact = compact.replace("0USE", "OUSE").replace("M0USE", "MOUSE")
    if compact.isdigit() and len(compact) <= 2:
        return True
    if len(compact) == 1 and compact.isalnum():
        return True
    key_words = {
        "ALT",
        "CTRL",
        "CONTROL",
        "SHIFT",
        "SPACE",
        "TAB",
        "CAPS",
        "ENTER",
        "LMB",
        "RMB",
        "MMB",
        "MB4",
        "MB5",
        "M4",
        "M5",
    }
    if compact in key_words:
        return True
    if "OUSE" in compact and len(compact) <= 8:
        return True
    if compact.startswith("MB") and compact[2:].isdigit():
        return True
    if compact.startswith("F") and compact[1:].isdigit() and len(compact) <= 3:
        return True
    return False


def _suppress_upper_key_binding_text(values: np.ndarray) -> np.ndarray:
    if values.size == 0 or values.ndim != 2:
        return values
    try:
        import cv2  # type: ignore
    except ImportError:
        return values

    normalized = values.astype(np.float32, copy=False)
    peak = float(normalized.max())
    if peak <= 1e-6:
        return normalized
    normalized = normalized / peak
    frame_height, frame_width = normalized.shape
    if frame_height < 18 or frame_width < 18:
        return values.astype(np.float32, copy=False)

    threshold = max(
        float(np.percentile(normalized, 86)) * 0.55,
        float(normalized.mean()) + float(normalized.std()) * 0.35,
        0.045,
    )
    mask = (normalized >= threshold).astype(np.uint8)
    labels, stats = _component_stats(mask)
    if not stats:
        return values.astype(np.float32, copy=False)

    remove_mask = np.zeros_like(mask, dtype=np.uint8)
    upper_limit = int(round(frame_height * 0.42))
    text_band_limit = int(round(frame_height * 0.24))
    for label, stat in stats:
        x, y, width, height, area = stat
        y_max = y + height
        if y >= upper_limit:
            continue
        center_y = (y + height / 2.0) / max(frame_height, 1)
        area_ratio = area / max(frame_width * frame_height, 1)
        width_ratio = width / max(frame_width, 1)
        height_ratio = height / max(frame_height, 1)
        top_line_like = y <= max(2, int(frame_height * 0.06)) and height_ratio <= 0.13
        glyph_like = width_ratio <= 0.22 and height_ratio <= 0.32 and area_ratio <= 0.045
        word_like = width_ratio <= 0.68 and height_ratio <= 0.30 and area_ratio <= 0.10
        tall_binding_like = width_ratio <= 0.75 and height_ratio <= 0.55 and area_ratio <= 0.14
        does_not_reach_icon_body = y_max <= int(round(frame_height * 0.50))
        upper_text_like = center_y <= 0.40 and (glyph_like or word_like or top_line_like or tall_binding_like)
        if not upper_text_like:
            continue
        component_mask = labels == label
        if does_not_reach_icon_body:
            remove_mask[component_mask] = 1
            continue
        if y <= int(round(frame_height * 0.24)) and width_ratio <= 0.75 and height_ratio <= 0.55:
            band_mask = np.zeros_like(remove_mask, dtype=bool)
            band_mask[:text_band_limit, :] = True
            remove_mask[component_mask & band_mask] = 1

    if not int(remove_mask.sum()):
        return values.astype(np.float32, copy=False)
    remove_mask = cv2.dilate(remove_mask, np.ones((3, 3), np.uint8), iterations=1)
    cleaned = values.astype(np.float32, copy=True)
    cleaned[remove_mask > 0] = 0.0
    return cleaned


def _normalize_selected_icon_foreground(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32, copy=False)
    normalized = values.astype(np.float32, copy=True)
    active = normalized[normalized > 0.01]
    if active.size < 4:
        return normalized
    low = float(np.percentile(active, 5.0)) * 0.35
    high = float(np.percentile(active, 99.0))
    if high <= low + 1e-6:
        return normalized
    output = np.zeros_like(normalized, dtype=np.float32)
    support = normalized > 0.0
    output[support] = np.clip((normalized[support] - low) / (high - low), 0.0, 1.0)
    return np.sqrt(output, dtype=np.float32)


def _crop_primary_contour(
    values: np.ndarray,
    *,
    prefer_upper: bool,
    pad_ratio: float = HUD_WEAPON_CONTOUR_PADDING_RATIO,
    lower_text_start_ratio: float = 0.45,
    bottom_text_start_ratio: float = 0.46,
    include_weak_support: bool = False,
) -> np.ndarray:
    if values.size == 0 or values.ndim != 2:
        return values
    try:
        import cv2  # type: ignore
    except ImportError:
        return _trim_foreground_border(values)

    normalized = values.astype(np.float32, copy=False)
    peak = float(normalized.max())
    if peak <= 1e-6:
        return normalized
    normalized = normalized / peak

    threshold = max(
        float(np.percentile(normalized, 86)) * 0.55,
        float(normalized.mean()) + float(normalized.std()) * 0.35,
        0.045,
    )
    mask = (normalized >= threshold).astype(np.uint8)
    if prefer_upper and mask.shape[0] >= 24:
        height = mask.shape[0]
        lower_start = int(round(height * lower_text_start_ratio))
        lower = mask[lower_start:, :]
        lower_labels, lower_stats = _component_stats(lower)
        for label, stat in lower_stats:
            x, y, width, comp_height, area = stat
            bottom_text_like = comp_height <= max(6, int(height * 0.30))
            if bottom_text_like:
                lower[lower_labels == label] = 0
        mask[lower_start:, :] = lower

    close_width = max(3, int(round(mask.shape[1] * 0.035)) | 1)
    close_height = max(3, int(round(mask.shape[0] * 0.055)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_height, close_width), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((2, 3), np.uint8), iterations=1)

    labels, stats = _component_stats(mask)
    if not stats:
        return _trim_foreground_border(values)

    frame_height, frame_width = mask.shape
    candidates: list[tuple[float, int, tuple[int, int, int, int, int]]] = []
    min_area = max(4, int(frame_width * frame_height * 0.002))
    for label, stat in stats:
        x, y, width, height, area = stat
        if area < min_area:
            continue
        touches_side = x <= 1 or x + width >= frame_width - 1
        border_line = touches_side and height > frame_height * 0.55 and width <= max(4, frame_width * 0.035)
        bottom_text = prefer_upper and y > frame_height * bottom_text_start_ratio and height < frame_height * 0.32
        if border_line or bottom_text:
            continue
        center_y = (y + height / 2.0) / max(frame_height, 1)
        upper_bias = 1.25 if not prefer_upper or center_y <= 0.70 else 0.55
        elongation = min(width / max(height, 1), 5.0) / 5.0
        score = float(area) * upper_bias * (1.0 + elongation * 0.45)
        candidates.append((score, label, stat))
    if not candidates:
        return _trim_foreground_border(values)

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, primary_label, primary_stat = candidates[0]
    selected_labels = {primary_label}
    x, y, width, height, primary_area = primary_stat
    x_min, y_min, x_max, y_max = x, y, x + width, y + height
    band_y_min = max(0, int(y_min - frame_height * 0.16))
    band_y_max = min(frame_height, int(y_max + frame_height * 0.22))
    area_floor = max(3, int(primary_area * 0.05))

    for _, label, stat in candidates[1:]:
        cx, cy, comp_width, comp_height, area = stat
        if area < area_floor:
            continue
        comp_y_max = cy + comp_height
        vertical_overlap = min(y_max, comp_y_max) - max(y_min, cy)
        same_icon_band = cy < band_y_max and comp_y_max > band_y_min
        close_horizontally = cx <= x_max + frame_width * 0.20 and cx + comp_width >= x_min - frame_width * 0.20
        if vertical_overlap > 0 or (same_icon_band and close_horizontally):
            selected_labels.add(label)
            x_min = min(x_min, cx)
            y_min = min(y_min, cy)
            x_max = max(x_max, cx + comp_width)
            y_max = max(y_max, comp_y_max)

    selected_mask = np.isin(labels, tuple(selected_labels)).astype(np.uint8)
    if include_weak_support and y_min > frame_height * 0.20:
        weak_threshold = max(threshold * 0.28, 0.012)
        weak_mask = (normalized >= weak_threshold).astype(np.uint8)
        if prefer_upper and weak_mask.shape[0] >= 24:
            weak_lower_start = int(round(weak_mask.shape[0] * lower_text_start_ratio))
            weak_mask[weak_lower_start:, :] = mask[weak_lower_start:, :]
        weak_labels, weak_stats = _component_stats(weak_mask)
        support_labels: set[int] = set()
        for label, stat in weak_stats:
            sx, sy, support_width, support_height, support_area = stat
            if support_area < 3:
                continue
            support_density = support_area / max(support_width * support_height, 1)
            large_sparse_frame = (
                support_width >= frame_width * 0.48
                and support_height >= frame_height * 0.34
                and support_density <= 0.28
            )
            if large_sparse_frame:
                continue
            support_y_max = sy + support_height
            touches_side = sx <= 1 or sx + support_width >= frame_width - 1
            border_line = touches_side and support_height > frame_height * 0.55 and support_width <= max(4, frame_width * 0.035)
            bottom_text = prefer_upper and sy > frame_height * bottom_text_start_ratio and support_height < frame_height * 0.32
            if border_line or bottom_text:
                continue
            close_horizontally = sx <= x_max + frame_width * 0.26 and sx + support_width >= x_min - frame_width * 0.26
            near_icon_vertically = support_y_max >= y_min - frame_height * 0.42 and sy <= y_max + frame_height * 0.20
            if close_horizontally and near_icon_vertically:
                support_labels.add(label)
        if support_labels:
            support_mask = np.isin(weak_labels, tuple(support_labels)).astype(np.uint8)
            selected_mask = np.maximum(selected_mask, support_mask)
    coords = np.argwhere(selected_mask > 0)
    if coords.size == 0:
        return _trim_foreground_border(values)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    pad = max(2, int(round(max(y_max - y_min, x_max - x_min) * max(0.0, pad_ratio))))
    y_min = max(0, int(y_min) - pad)
    x_min = max(0, int(x_min) - pad)
    y_max = min(frame_height, int(y_max) + pad)
    x_max = min(frame_width, int(x_max) + pad)
    cropped = values[y_min:y_max, x_min:x_max].astype(np.float32, copy=True)
    contour = selected_mask[y_min:y_max, x_min:x_max].astype(np.float32)
    contour = cv2.dilate(contour.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)
    return cropped * contour


def _component_stats(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, tuple[int, int, int, int, int]]]]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return np.zeros_like(mask, dtype=np.int32), []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    output: list[tuple[int, tuple[int, int, int, int, int]]] = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        output.append((label, (x, y, width, height, area)))
    return labels, output


def _resize_letterbox(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError:
        return _fallback_icon_map(values, size=size)
    target_width, target_height = size
    height, width = values.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((target_height, target_width), dtype=np.float32)
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, min(target_width, int(round(width * scale))))
    resized_height = max(1, min(target_height, int(round(height * scale))))
    resized = cv2.resize(values, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width), dtype=np.float32)
    y_min = (target_height - resized_height) // 2
    x_min = (target_width - resized_width) // 2
    canvas[y_min : y_min + resized_height, x_min : x_min + resized_width] = resized
    return canvas


def _fallback_feature(image: np.ndarray, *, size: tuple[int, int]) -> np.ndarray:
    rgb = _to_rgb(image).astype(np.float32)
    gray = rgb.mean(axis=2)
    y_idx = np.linspace(0, max(gray.shape[0] - 1, 0), size[1]).astype(int)
    x_idx = np.linspace(0, max(gray.shape[1] - 1, 0), size[0]).astype(int)
    resized = gray[np.ix_(y_idx, x_idx)] / 255.0
    feature = resized.flatten()
    norm = np.linalg.norm(feature)
    return feature / norm if norm else feature


def _fallback_icon_map(image: np.ndarray, *, size: tuple[int, int]) -> np.ndarray:
    rgb = _to_rgb(image).astype(np.float32)
    gray = rgb.mean(axis=2) / 255.0
    y_idx = np.linspace(0, max(gray.shape[0] - 1, 0), size[1]).astype(int)
    x_idx = np.linspace(0, max(gray.shape[1] - 1, 0), size[0]).astype(int)
    return gray[np.ix_(y_idx, x_idx)].astype(np.float32)


def _clamp(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _row_get(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except Exception:
        return getattr(row, key, None)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
