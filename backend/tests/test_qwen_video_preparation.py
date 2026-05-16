from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from backend.app.processing.qwen_video import (
    TEMPORAL_SAMPLING_STRATEGY,
    _enhance_sdr_proxy_for_vision,
    _maybe_enhance_dark_sdr_frame,
    _target_frame_count,
    _target_time_distribution,
    _target_times,
    prepare_hit_marker_video_input,
    prepare_qwen_video_input,
)


def test_qwen_video_preparation_samples_downscales_and_caches_frames(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _write_test_video(source)

    prepared = prepare_qwen_video_input(source, tmp_path / "cache", fps=2.0, max_frames=4, max_width=80, accelerator="cpu")
    cached = prepare_qwen_video_input(source, tmp_path / "cache", fps=2.0, max_frames=4, max_width=80, accelerator="cpu")

    assert prepared.mode == "sampled_sdr_frame_sequence"
    assert len(prepared.frame_paths) == 4
    assert prepared.frame_paths == cached.frame_paths
    assert prepared.metadata["qwen_video_input_mode"] == "sampled_sdr_frame_sequence"
    assert prepared.metadata["qwen_video_frame_count"] == 4
    assert prepared.metadata["qwen_video_conversion_backend"] == "pyav"
    assert prepared.metadata["qwen_video_decode_threading"] == "auto"
    assert prepared.metadata["qwen_video_requested_accelerator"] == "cpu"
    assert prepared.metadata["qwen_video_accelerator_api"] == "pyav_hwaccel"
    assert prepared.metadata["qwen_video_accelerator_status"] == "not_requested"
    assert prepared.metadata["qwen_video_frame_encoding"] == "png_compress_level_1"
    assert prepared.metadata["qwen_video_frame_postprocess_workers"] >= 1
    assert prepared.metadata["qwen_video_tone_mapping"] == "pyav_direct_autocontrast"
    assert prepared.metadata["qwen_video_sdr_dark_enhancement_algorithm"] == "adaptive_sdr_low_light_v1"
    assert prepared.metadata["qwen_video_temporal_sampling_strategy"] == TEMPORAL_SAMPLING_STRATEGY
    assert prepared.metadata["qwen_video_temporal_sampling_start_sec"] == 0.0
    assert prepared.metadata["qwen_video_temporal_sampling_end_sec"] == 2.0
    assert prepared.metadata["qwen_video_temporal_sampling_target_times_sec"] == [0.25, 0.75, 1.25, 1.75]
    assert "qwen_video_sdr_dark_enhanced_frame_count" in prepared.metadata
    assert len(prepared.metadata["qwen_video_frame_timestamps_sec"]) == 4
    with Image.open(prepared.frame_paths[0]) as image:
        assert image.mode == "RGB"
        assert image.width <= 80


def test_hit_marker_video_preparation_preserves_source_width(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _write_test_video(source)

    prepared = prepare_hit_marker_video_input(source, tmp_path / "hit-cache", fps=2.0, max_frames=4, accelerator="cpu")
    cached = prepare_hit_marker_video_input(source, tmp_path / "hit-cache", fps=2.0, max_frames=4, accelerator="cpu")

    assert prepared.mode == "sampled_sdr_source_width_frame_sequence"
    assert prepared.frame_paths == cached.frame_paths
    assert prepared.metadata["hit_marker_frame_input_mode"] == "sampled_sdr_source_width_frame_sequence"
    assert prepared.metadata["hit_marker_frame_count"] == 4
    assert prepared.metadata["hit_marker_frame_preserve_source_width"] is True
    assert len(prepared.metadata["hit_marker_frame_timestamps_sec"]) == 4
    with Image.open(prepared.frame_paths[0]) as image:
        assert image.mode == "RGB"
        assert image.width == 160


def test_qwen_video_sampler_uses_equal_time_schedule_and_focus_windows() -> None:
    duration = 25.09113333333333
    target_frames = _target_frame_count({"duration": duration}, fps=6.0, max_frames=100)
    timestamps = _target_times(duration=duration, target_frames=target_frames)

    first_half = [timestamp for timestamp in timestamps if timestamp < duration / 2.0]
    second_half = [timestamp for timestamp in timestamps if timestamp >= duration / 2.0]
    per_second = {
        item["start_sec"]: item["frame_count"]
        for item in _target_time_distribution(duration, target_frames)["qwen_video_temporal_sampling_frames_per_second"]
    }
    distribution = _target_time_distribution(duration, target_frames)
    focus_timestamps = _target_times(duration=duration, target_frames=100, start_sec=12.0, end_sec=24.0)
    focus_distribution = _target_time_distribution(duration, 100, start_sec=12.0, end_sec=24.0)

    assert target_frames == 100
    assert len(timestamps) == 100
    assert timestamps == sorted(timestamps)
    assert abs(len(first_half) - len(second_half)) <= 1
    assert per_second[0.0] in {3, 4}
    assert per_second[18.0] in {3, 4}
    assert per_second[24.0] in {3, 4, 5}
    assert distribution["qwen_video_temporal_sampling_start_sec"] == 0.0
    assert distribution["qwen_video_temporal_sampling_end_sec"] == round(duration, 6)
    assert len(focus_timestamps) == 100
    assert min(focus_timestamps) >= 12.0
    assert max(focus_timestamps) <= 24.0
    assert focus_distribution["qwen_video_temporal_sampling_start_sec"] == 12.0
    assert focus_distribution["qwen_video_temporal_sampling_end_sec"] == 24.0


def test_hdr_proxy_enhancement_increases_local_contrast_without_red_cast() -> None:
    frame = np.full((32, 32, 3), 92, dtype=np.uint8)
    frame[8:24, 12:20, :] = 126
    image = Image.fromarray(frame)

    enhanced = np.asarray(_enhance_sdr_proxy_for_vision(image)).astype(np.float32)

    assert enhanced.std() > frame.std()
    channel_means = enhanced.reshape(-1, 3).mean(axis=0)
    assert max(channel_means) - min(channel_means) < 3.0


def test_dark_sdr_frame_gets_mild_adaptive_lift_without_color_cast() -> None:
    frame = np.full((32, 32, 3), 24, dtype=np.uint8)
    frame[10:22, 12:20, :] = 48
    image = Image.fromarray(frame)

    enhanced, changed = _maybe_enhance_dark_sdr_frame(image)
    enhanced_array = np.asarray(enhanced).astype(np.float32)

    assert changed is True
    assert enhanced_array.mean() > frame.mean()
    assert enhanced_array.std() >= frame.std()
    channel_means = enhanced_array.reshape(-1, 3).mean(axis=0)
    assert max(channel_means) - min(channel_means) < 3.0


def test_normal_sdr_frame_is_not_modified() -> None:
    frame = np.full((32, 32, 3), 112, dtype=np.uint8)
    frame[:, :, 1] = 128
    image = Image.fromarray(frame)

    enhanced, changed = _maybe_enhance_dark_sdr_frame(image)

    assert changed is False
    assert np.array_equal(np.asarray(enhanced), frame)


def _write_test_video(path: Path) -> None:
    import av

    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=8)
    stream.width = 160
    stream.height = 90
    stream.pix_fmt = "yuv420p"
    for index in range(16):
        frame_rgb = np.zeros((90, 160, 3), dtype=np.uint8)
        frame_rgb[:, :, 0] = min(index * 12, 255)
        frame_rgb[:, :, 1] = np.linspace(0, 255, 160, dtype=np.uint8)
        frame_rgb[:, :, 2] = 128
        frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
