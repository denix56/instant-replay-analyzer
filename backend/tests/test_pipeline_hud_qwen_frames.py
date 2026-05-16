from backend.app.analysis.hud_loadout import HudDetectionResult, HudMatch
from backend.app.pipeline import _detect_hud_for_qwen_frames
from backend.app.processing.qwen_video import QwenVideoInput


def test_detect_hud_for_qwen_frames_runs_ocr_on_every_prepared_frame() -> None:
    frames = ["/tmp/frame_0000.png", "/tmp/frame_0001.png", "/tmp/frame_0002.png"]

    class Detector:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def detect_frame(self, frame_path: str) -> HudDetectionResult | None:
            self.seen.append(frame_path)
            if frame_path.endswith("0001.png"):
                return None
            name = "Mosin Obrez (Rougarou skin)" if frame_path.endswith("0000.png") else "First Aid Kit"
            entity_type = "weapon" if "Mosin" in name else "tool"
            return HudDetectionResult(
                frame_path=frame_path,
                frame_width=1280,
                frame_height=720,
                matches=(
                    HudMatch(
                        slot_key="current_ocr",
                        is_active=True,
                        entity_id=f"{entity_type}:example",
                        entity_name=name,
                        entity_type=entity_type,
                        confidence=0.96,
                        matched_image_path=None,
                        highlight_score=1.0,
                    ),
                ),
            )

    detector = Detector()
    qwen_input = QwenVideoInput(
        source_path="/tmp/clip.mp4",
        frame_paths=frames,
        mode="sampled_sdr_frame_sequence",
        sample_fps=2.0,
        metadata={"qwen_video_frame_timestamps_sec": [0.0, 0.5, 1.0]},
    )

    summary = _detect_hud_for_qwen_frames(7, qwen_input, detector)  # type: ignore[arg-type]

    assert detector.seen == frames
    assert [row["timestamp"] for row in summary["prepared_frame_evidence"]] == [0.0, 1.0]
    assert [item["entity_name"] for item in summary["equipment_timeline"]] == [
        "Mosin Obrez (Rougarou skin)",
        "First Aid Kit",
    ]
    assert summary["qwen_prepared_frame_count"] == 3

