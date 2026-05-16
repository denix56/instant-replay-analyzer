import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.analysis.hud_loadout import (
    HudGeometryConfig,
    HudLoadoutDetector,
    HudMatch,
    HudDetectionResult,
    RelativeBox,
    active_highlight_score,
    detections_to_rows,
    multi_crop_hud_visual_descriptors,
    preprocess_icon_for_matching,
    reference_icon_for_matching,
    relative_box_to_pixels,
    temporal_foreground_icon_map,
    _crop_primary_contour,
    _crop_groups_have_visible_icon,
    _hud_icon_crop_variants_for_types,
    _hud_local_detail_map,
    _is_weak_hud_calibration_match,
    _key_binding_right_extension,
    _normalize_selected_icon_foreground,
    _normalize_slot_quantity_text,
    _equipment_quantity_markers,
    _suppress_upper_key_binding_text,
    _tight_key_binding_text_mask,
)
from backend.app.db import Database
from backend.app.knowledge.hunt_runtime import HuntEntity, HuntEquipmentResolution, HuntKnowledgeService


def test_relative_box_to_pixels_scales_across_resolutions():
    box = RelativeBox(0.1, 0.75, 0.2, 0.9)

    assert relative_box_to_pixels(box, 1920, 1080) == (192, 810, 384, 972)
    assert relative_box_to_pixels(box, 2560, 1440) == (256, 1080, 512, 1296)


def test_active_highlight_score_prefers_gray_highlight():
    dark = np.zeros((100, 180, 3), dtype=np.uint8)
    highlighted = np.full((100, 180, 3), 128, dtype=np.uint8)

    assert active_highlight_score(highlighted) > active_highlight_score(dark) + 0.4


def test_hud_geometry_slots_are_normalized():
    geometry = HudGeometryConfig()

    assert geometry.hud_crop.y_min >= 0.7
    assert all(0.0 <= slot.box.x_min < slot.box.x_max <= 1.0 for slot in geometry.slots)
    assert all(0.0 <= slot.box.y_min < slot.box.y_max <= 1.0 for slot in geometry.slots)
    by_key = {slot.key: slot.allowed_types for slot in geometry.slots}
    assert by_key["1"] == ("weapon",)
    assert by_key["2"] == ("weapon",)
    assert by_key["3"] == ("tool",)
    assert by_key["4"] == ("tool",)
    assert by_key["5"] == ("tool",)
    assert by_key["6"] == ("tool",)
    assert by_key["7"] == ("consumable",)
    assert by_key["8"] == ("consumable",)
    assert by_key["9"] == ("consumable",)
    assert by_key["0"] == ("consumable",)


def test_weapon_slots_have_normalized_size_variants():
    geometry = HudGeometryConfig()
    weapon_slots = [slot for slot in geometry.slots if slot.allowed_types == ("weapon",)]

    assert len(weapon_slots) == 2
    for slot in weapon_slots:
        assert len(slot.box_variants) >= 3
        base_width = slot.box.x_max - slot.box.x_min
        variant_widths = [box.x_max - box.x_min for box in slot.box_variants]
        assert min(variant_widths) < base_width
        assert max(variant_widths) > base_width
        assert all(0.0 <= box.x_min < box.x_max <= 1.0 for box in slot.box_variants)
        assert all(0.0 <= box.y_min < box.y_max <= 1.0 for box in slot.box_variants)


def test_tool_and_consumable_slots_keep_equal_width_and_gutters():
    geometry = HudGeometryConfig()
    slots = [slot for slot in geometry.slots if slot.key in {"3", "4", "5", "6", "7", "8", "9", "0"}]
    weapon_2 = next(slot for slot in geometry.slots if slot.key == "2")

    widths = [round(slot.box.x_max - slot.box.x_min, 6) for slot in slots]
    gaps = [round(current.box.x_min - previous.box.x_max, 6) for previous, current in zip(slots, slots[1:])]
    assert len(set(widths)) == 1
    assert len(set(gaps)) == 1
    assert gaps[0] > 0.0
    assert slots[0].box.x_min > weapon_2.box.x_max
    for previous, current in zip(slots, slots[1:]):
        assert current.box.x_min > previous.box.x_max
    by_key = {slot.key: slot for slot in slots}
    assert by_key["6"].box.x_max < by_key["7"].box.x_min


def test_empty_slot_gate_rejects_blank_crop_and_keeps_icon_crop():
    blank = np.full((90, 180, 3), 42, dtype=np.uint8)
    icon = blank.copy()
    icon[28:33, 34:148] = 215
    icon[33:40, 58:128] = 180

    assert not _crop_groups_have_visible_icon([[blank, blank, blank]], ("weapon",))
    assert _crop_groups_have_visible_icon([[icon, icon, icon]], ("weapon",))


def test_detect_frame_uses_only_current_item_ocr(tmp_path, monkeypatch):
    cv2 = __import__("cv2")
    frame_path = tmp_path / "frame.png"
    frame = np.full((240, 320, 3), 48, dtype=np.uint8)
    assert cv2.imwrite(str(frame_path), frame)
    detector = HudLoadoutDetector(HuntKnowledgeService(tmp_path / "missing-pack"), confidence_threshold=0.99)
    entity = HuntEntity(
        id="tool:flare-pistol",
        type="tool",
        name="Flare Pistol",
        aliases=(),
        description="",
        source_url="",
        image_paths=(),
        key_values={},
    )

    monkeypatch.setattr(
        detector,
        "_current_weapon_from_ocr",
        lambda image: HuntEquipmentResolution(entity=entity, matched_name="Signal Flair", display_name="Flare Pistol"),
    )
    monkeypatch.setattr(
        detector,
        "_match_crop_groups",
        lambda crop_groups, allowed_types: (_ for _ in ()).throw(AssertionError("slot image matching should be disabled")),
    )

    result = detector.detect_frame(frame_path)

    assert result is not None
    assert len(result.matches) == 1
    assert result.matches[0].slot_key == "current_ocr"
    assert result.matches[0].entity_name == "Flare Pistol"
    assert result.matches[0].entity_type == "tool"


def test_tool_descriptor_uses_tight_equipment_variants_and_suppresses_background():
    assert len(_hud_icon_crop_variants_for_types(("tool",))) >= 3
    crops = []
    for index in range(5):
        crop = np.full((80, 80, 3), 44 + index * 7, dtype=np.uint8)
        gradient = ((np.arange(80, dtype=np.uint8)[None, :] * 2 + index * 19) % 70).astype(np.uint8)
        crop[:, :, 0] = np.clip(crop[:, :, 0] + gradient, 0, 255)
        crop[:, :, 1] = np.clip(crop[:, :, 1] + gradient // 2, 0, 255)
        crop[20:60, 22:58] = [185, 185, 185]
        crop[28:52, 30:50] = [225, 225, 225]
        crop[66:72, 18:62] = [160, 160, 160]
        crops.append(crop)

    descriptor = multi_crop_hud_visual_descriptors(
        crops,
        variants=_hud_icon_crop_variants_for_types(("tool",)),
    )[0]

    y_indices, x_indices = np.nonzero(descriptor.icon > 0.2)
    assert len(x_indices) > 0
    foreground = float(
        descriptor.icon[
            y_indices.min() : y_indices.max() + 1,
            x_indices.min() : x_indices.max() + 1,
        ].mean()
    )
    border = float(np.concatenate([descriptor.icon[:, :8].ravel(), descriptor.icon[:, -8:].ravel()]).mean())
    lower_binding = float(descriptor.icon[38:, :].mean())
    assert foreground > 0.25
    assert border < foreground * 0.55
    assert lower_binding < foreground * 0.75


def test_hud_local_detail_map_preserves_dark_and_colored_icon_detail():
    crop = np.full((90, 130, 3), [88, 88, 88], dtype=np.uint8)
    crop[:, :, 0] += np.linspace(0, 34, crop.shape[1], dtype=np.uint8)[None, :]
    crop[20:27, 24:102] = [42, 42, 42]
    crop[42:70, 58:66] = [190, 42, 32]
    crop[52:60, 46:78] = [190, 42, 32]

    detail = _hud_local_detail_map(crop)

    background = float(detail[:12, :12].mean())
    dark_stroke = float(detail[20:27, 24:102].mean())
    colored_stroke = float(detail[48:64, 52:72].mean())
    assert dark_stroke > background + 0.12
    assert colored_stroke > background + 0.12
    assert len(np.unique(detail)) > 2


def test_slot_quantity_text_normalizes_ocr_confusions():
    assert _normalize_slot_quantity_text("3x") == "3"
    assert _normalize_slot_quantity_text("Zx") == "2"
    assert _normalize_slot_quantity_text("Ix") == "1"
    assert _normalize_slot_quantity_text("Lx") == "1"
    assert _normalize_slot_quantity_text("1/2x") == "1/2"


def test_equipment_quantity_markers_include_quantity_and_loaded_extra():
    entity = HuntEntity(
        id="tool:flare-pistol",
        type="tool",
        name="Flare Pistol",
        aliases=(),
        description="",
        source_url="",
        image_paths=(),
        key_values={"Quantity": "2", "Loaded": "1", "Extra": "2"},
    )

    markers = _equipment_quantity_markers(entity)

    assert markers["quantity"] == {"2"}
    assert markers["loaded_extra"] == {"1/2"}


def test_weak_consumable_calibration_match_is_rejected():
    calibration_path = "/pack/media/images/hud_calibration/consumable-consumables-regeneration-shot-slot7.png"

    assert _is_weak_hud_calibration_match(calibration_path, ("consumable",), 0.68)
    assert not _is_weak_hud_calibration_match(calibration_path, ("consumable",), 0.72)
    assert not _is_weak_hud_calibration_match(calibration_path, ("tool",), 0.58)


def test_upper_key_binding_text_is_removed_before_equipment_contour():
    foreground = np.zeros((90, 120), dtype=np.float32)
    foreground[6:18, 10:18] = 0.95
    foreground[8:20, 52:94] = 0.92
    foreground[38:72, 40:82] = 0.78
    foreground[48:63, 54:98] = 0.68

    cleaned = _suppress_upper_key_binding_text(foreground)

    assert float(cleaned[:24, :].max()) < 0.1
    assert float(cleaned[38:72, 40:98].mean()) > 0.45


def test_ocr_key_binding_mask_keeps_bottom_tight_to_letters():
    region = np.zeros((100, 140, 3), dtype=np.uint8)
    region[8:18, 8:76] = [235, 235, 235]
    region[24:50, 32:112] = [210, 210, 210]

    mask = _tight_key_binding_text_mask(region, (2, 2, 118, 56))

    assert mask is not None
    assert int(mask[8:20, 8:76].sum()) > 0
    assert int(mask[26:50, 32:112].sum()) == 0


def test_mouse_key_binding_box_extends_right_for_missing_digit():
    assert _key_binding_right_extension("IOUSE", 20) >= 12
    assert _key_binding_right_extension("MOUSE 5", 20) == 0


def test_hud_descriptors_keep_grayscale_variant_for_each_crop_variant():
    crop = np.full((80, 80, 3), 42, dtype=np.uint8)
    crop[24:58, 18:54] = [190, 190, 190]
    crop[32:48, 28:46] = [235, 235, 235]
    variants = _hud_icon_crop_variants_for_types(("tool",))

    descriptors = multi_crop_hud_visual_descriptors([crop, crop], variants=variants)

    assert len(descriptors) == len(variants)
    for descriptor in descriptors:
        unique_values = np.unique(descriptor.icon)
        assert float(descriptor.icon.max()) > 0.0
        assert len(unique_values) > 2
        assert not set(unique_values).issubset({0.0, 1.0})


def test_preprocess_icon_preserves_aspect_with_letterbox():
    wide = np.ones((24, 120, 4), dtype=np.uint8) * 255
    wide[:, :, 3] = 255

    icon, aspect = preprocess_icon_for_matching(wide, size=(96, 48), include_aspect=True)
    nonzero = np.argwhere(icon > 0.05)
    y_min, _ = nonzero.min(axis=0)
    y_max, _ = nonzero.max(axis=0)

    assert aspect >= 4.5
    assert y_min > 0
    assert y_max < 47


def test_reference_icon_converts_transparent_art_to_hud_grayscale():
    reference = np.zeros((32, 128, 4), dtype=np.uint8)
    reference[10:22, 20:108, :3] = [80, 120, 180]
    reference[10:22, 20:108, 3] = 255

    icon, aspect = reference_icon_for_matching(reference, size=(96, 48), include_aspect=True)
    nonzero = np.argwhere(icon > 0.05)
    y_min, _ = nonzero.min(axis=0)
    y_max, _ = nonzero.max(axis=0)

    assert aspect >= 4.5
    assert float(icon.max()) > 0.9
    assert 2 < len(np.unique(icon)) < 64
    assert y_min > 0
    assert y_max < 47


def test_reference_icon_extracts_foreground_from_black_background():
    reference = np.zeros((40, 120, 3), dtype=np.uint8)
    reference[15:25, 25:100] = [95, 120, 145]

    icon = reference_icon_for_matching(reference, size=(96, 48))

    assert float(icon.max()) > 0.9
    assert 2 < len(np.unique(icon)) < 64
    assert int((icon > 0.1).sum()) > 100


def test_temporal_foreground_keeps_fixed_icon_over_changing_background():
    crops = []
    for index in range(5):
        crop = np.full((80, 180, 3), 35 + index * 15, dtype=np.uint8)
        crop[:, :, 0] += (np.arange(180, dtype=np.uint8)[None, :] + index * 17) % 30
        crop[28:35, 35:145] = 210
        crop[35:42, 58:132] = 185
        crops.append(crop)

    icon = temporal_foreground_icon_map(crops, size=(90, 40), use_robust_pca=False)
    rpca_icon = temporal_foreground_icon_map(crops, size=(90, 40), use_robust_pca=True)

    assert float(icon.max()) > 0.8
    assert len(np.unique(icon)) > 2
    assert not set(np.unique(icon)).issubset({0.0, 1.0})
    assert int((icon[:, 18:72] > 0.5).sum()) > int((icon[:, :12] > 0.5).sum())
    assert float(rpca_icon.max()) > 0.8
    assert len(np.unique(rpca_icon)) > 2
    assert not set(np.unique(rpca_icon)).issubset({0.0, 1.0})


def test_primary_contour_crop_removes_lower_ammo_text():
    foreground = np.zeros((80, 180), dtype=np.float32)
    foreground[24:31, 30:150] = 0.75
    foreground[31:38, 55:135] = 0.55
    foreground[62:68, 74:108] = 1.0

    cropped = _crop_primary_contour(foreground, prefer_upper=True)

    assert cropped.shape[0] < 42
    assert float(cropped.max()) < 0.9
    assert int((cropped > 0.1).sum()) > 300


def test_equipment_contour_keeps_lower_icon_parts_before_ammo_text():
    foreground = np.zeros((100, 120), dtype=np.float32)
    foreground[20:48, 42:72] = 0.75
    foreground[58:70, 54:84] = 0.55
    foreground[88:94, 44:76] = 1.0

    cropped = _crop_primary_contour(
        foreground,
        prefer_upper=True,
        pad_ratio=0.10,
        lower_text_start_ratio=0.72,
        bottom_text_start_ratio=0.72,
    )

    assert cropped.shape[0] >= 50
    assert float(cropped.max()) < 0.9
    assert int((cropped > 0.1).sum()) > 1100


def test_equipment_contour_recovers_weak_upper_icon_parts():
    foreground = np.zeros((100, 120), dtype=np.float32)
    foreground[18:27, 50:78] = 0.08
    foreground[54:72, 42:86] = 0.80

    cropped = _crop_primary_contour(
        foreground,
        prefer_upper=True,
        pad_ratio=0.08,
        include_weak_support=True,
    )

    assert cropped.shape[0] >= 48
    assert float(cropped[:14, :].max()) > 0.05
    assert float(cropped.max()) > 0.70


def test_equipment_contour_ignores_large_sparse_slot_border():
    foreground = np.zeros((100, 120), dtype=np.float32)
    foreground[10:90, 8:10] = 0.06
    foreground[10:12, 8:112] = 0.06
    foreground[88:90, 8:112] = 0.06
    foreground[10:90, 110:112] = 0.06
    foreground[20:28, 50:78] = 0.08
    foreground[54:72, 42:86] = 0.80

    cropped = _crop_primary_contour(
        foreground,
        prefer_upper=True,
        pad_ratio=0.08,
        include_weak_support=True,
    )

    assert cropped.shape[0] < 80
    assert cropped.shape[1] < 80
    assert float(cropped[:4, :].mean()) < 0.03
    assert float(cropped.max()) > 0.70


def test_equipment_icon_normalization_lifts_weak_upper_strokes():
    foreground = np.zeros((80, 80), dtype=np.float32)
    foreground[10:16, 30:50] = 0.18
    foreground[32:54, 24:58] = 0.82

    normalized = _normalize_selected_icon_foreground(foreground)

    assert float(normalized[10:16, 30:50].mean()) > 0.15
    assert float(normalized[32:54, 24:58].mean()) > 0.85
    assert float(normalized[:5, :].max()) == 0.0


def test_hud_reference_display_name_uses_variant_alias(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    entity = {
        "id": "weapon:weapons-mosin-obrez",
        "type": "weapon",
        "name": "Mosin Obrez",
        "aliases": ["Mosin Obrez"],
        "description": "Medium-slot Mosin variant.",
        "source_url": "https://huntshowdown.wiki.gg/wiki/Weapons/Mosin_Obrez",
        "image_paths": [],
        "key_values": {},
    }
    media = {
        "entity_id": "weapon:weapons-mosin-obrez",
        "entity_type": "weapon",
        "local_path": "media/images/weapon/mosin-extended.png",
        "alt": "Weapon Mosin Obrez Extended.png",
        "source_url": "https://huntshowdown.wiki.gg/images/Weapon_Mosin_Obrez_Extended.png",
        "title": "Mosin Obrez",
        "content_type": "image/png",
    }
    (pack / "manifest.json").write_text(json.dumps({"embedding_dimension": 8}), encoding="utf-8")
    (pack / "entities.jsonl").write_text(json.dumps(entity) + "\n", encoding="utf-8")
    (pack / "chunks.jsonl").write_text(
        json.dumps(
            {
                "id": "weapon:weapons-mosin-obrez:chunk:0",
                "entity_id": "weapon:weapons-mosin-obrez",
                "entity_type": "weapon",
                "text": "Mosin Obrez Mosin Obrez Extended",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (pack / "media_index.jsonl").write_text(json.dumps(media) + "\n", encoding="utf-8")

    service = HuntKnowledgeService(pack)
    detector = HudLoadoutDetector(service)

    assert detector._reference_display_name(service.entities["weapon:weapons-mosin-obrez"], media["local_path"]) == "Mosin Obrez Extended"


def test_db_persists_and_replaces_hud_detections(tmp_path):
    db = Database(tmp_path / "app.db")
    try:
        clip_id = db.upsert_clip({"filename": "clip.mp4", "path": "clip.mp4"})
        segment_id = db.upsert_segment(
            {
                "clip_id": clip_id,
                "group_name": "Hunt",
                "start_time": 0.0,
                "end_time": 2.0,
                "duration": 2.0,
                "modality": "video_only",
                "segment_settings_hash": "settings",
            }
        )
        result = HudDetectionResult(
            frame_path="frame.jpg",
            frame_width=1920,
            frame_height=1080,
            matches=(
                HudMatch(
                    slot_key="1",
                    is_active=True,
                    entity_id="weapon:dolch-96",
                    entity_name="Dolch 96",
                    entity_type="weapon",
                    confidence=0.9,
                    matched_image_path="dolch.png",
                    highlight_score=0.5,
                ),
            ),
        )
        db.replace_hud_detections(clip_id, segment_id, detections_to_rows(clip_id, segment_id, 0.0, result))
        db.replace_hud_detections(clip_id, segment_id, detections_to_rows(clip_id, segment_id, 0.0, result))

        rows = db.list_hud_detections(clip_id=clip_id)
        assert len(rows) == 1
        summary = db.hud_loadout_summary(clip_id)
        assert summary["active_weapon"] == "Dolch 96"
        assert summary["active_equipment"] == "Dolch 96"
        assert summary["active_equipment_type"] == "weapon"
        assert summary["loadout"] == ["Dolch 96"]
        assert summary["evidence"] == [
            {
                "segment_id": segment_id,
                "frame_path": "frame.jpg",
                "timestamp": 0.0,
                "slot_key": "1",
                "is_active": True,
                "entity_id": "weapon:dolch-96",
                "entity_name": "Dolch 96",
                "entity_type": "weapon",
                "confidence": 0.9,
                "matched_image_path": "dolch.png",
            }
        ]

        current_item_result = HudDetectionResult(
            frame_path="frame.jpg",
            frame_width=1920,
            frame_height=1080,
            matches=(
                HudMatch(
                    slot_key="current_ocr",
                    is_active=True,
                    entity_id="consumable:dynamite-bundle",
                    entity_name="Dynamite Bundle",
                    entity_type="consumable",
                    confidence=0.96,
                    matched_image_path=None,
                    highlight_score=1.0,
                ),
            ),
        )
        db.replace_hud_detections(clip_id, segment_id, detections_to_rows(clip_id, segment_id, 0.0, current_item_result))

        current_item_summary = db.hud_loadout_summary(clip_id)
        assert current_item_summary["active_weapon"] is None
        assert current_item_summary["active_equipment"] == "Dynamite Bundle"
        assert current_item_summary["active_equipment_type"] == "consumable"
        assert current_item_summary["loadout"] == ["Dynamite Bundle"]
        assert current_item_summary["evidence"][0]["slot_key"] == "current_ocr"
        assert current_item_summary["evidence"][0]["entity_name"] == "Dynamite Bundle"
    finally:
        db.close()
