from pathlib import Path

from PIL import Image, ImageDraw

from backend.app.analysis.hit_marker import detect_hit_marker_evidence


def test_hit_marker_detector_finds_centered_marker_with_target(tmp_path: Path) -> None:
    frame = tmp_path / "frame_0040.png"
    image = Image.new("RGB", (640, 360), (90, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((270, 125, 370, 260), fill=(105, 105, 105))
    for xy in [
        (305, 145, 315, 185),
        (330, 145, 340, 185),
        (285, 205, 325, 215),
        (345, 205, 385, 215),
        (290, 130, 305, 145),
        (360, 130, 375, 145),
    ]:
        draw.rectangle(xy, fill=(255, 80, 200))
    image.save(frame)

    result = detect_hit_marker_evidence(
        [frame],
        sample_fps=2.0,
        frame_timestamps=[20.0],
        start_sec=18.0,
        end_sec=22.0,
        active_weapon="Auto-5",
    )

    assert result["detected"] is True
    assert result["timestamp"] == 20.0
    assert result["active_weapon"] == "Auto-5"
    assert "Auto-5" in result["description"]
    assert result["evidence"][0]["confidence"] >= 0.55


def test_hit_marker_detector_uses_timestamped_equipment_timeline(tmp_path: Path) -> None:
    frame = tmp_path / "frame_0040.png"
    image = Image.new("RGB", (640, 360), (90, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((270, 125, 370, 260), fill=(105, 105, 105))
    for xy in [
        (305, 145, 315, 185),
        (330, 145, 340, 185),
        (285, 205, 325, 215),
        (345, 205, 385, 215),
        (290, 130, 305, 145),
        (360, 130, 375, 145),
    ]:
        draw.rectangle(xy, fill=(255, 80, 200))
    image.save(frame)

    result = detect_hit_marker_evidence(
        [frame],
        sample_fps=2.0,
        frame_timestamps=[20.0],
        active_weapon="Fallback Rifle",
        equipment_timeline=[
            {
                "start_timestamp": 18.0,
                "end_timestamp": 21.0,
                "entity_name": "Mosin Obrez (Rougarou skin)",
                "entity_type": "weapon",
            }
        ],
    )

    assert result["detected"] is True
    assert result["active_weapon"] == "Mosin Obrez (Rougarou skin)"
    assert "Mosin Obrez (Rougarou skin)" in result["description"]


def test_hit_marker_detector_ignores_plain_dark_frame(tmp_path: Path) -> None:
    frame = tmp_path / "frame_0000.png"
    Image.new("RGB", (640, 360), (90, 0, 0)).save(frame)

    result = detect_hit_marker_evidence([frame], sample_fps=2.0, frame_timestamps=[20.0], start_sec=18.0, end_sec=22.0)

    assert result["detected"] is False
    assert result["evidence"] == []
