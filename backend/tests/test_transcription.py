import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.config import AppSettings
from backend.app.pipeline import _transcriber
from backend.app.processing.transcription import Transcriber, TranscriptionConfig, _whisper_language


def test_transcriber_uses_sidecar(tmp_path):
    audio = tmp_path / "round_ending.wav"
    sidecar = tmp_path / "round_ending.wav.transcript.txt"
    sidecar.write_text("clutch win with final elimination", encoding="utf-8")

    result = Transcriber().transcribe(audio)

    assert result.text == "clutch win with final elimination"
    assert result.engine == "sidecar"
    assert result.segments[0].start == 0.0


def test_transcriber_filename_fallback_is_deterministic():
    result = Transcriber(TranscriptionConfig(engine="mock")).transcribe("final-kill_cam.mp4")

    assert result.text == "final kill cam"
    assert result.engine == "mock"


def test_transcription_default_language_is_auto():
    assert TranscriptionConfig().language == "auto"

    result = Transcriber().transcribe("final-kill_cam.mp4")

    assert result.language == "auto"


def test_whisper_auto_language_disables_forced_language():
    assert _whisper_language("auto") is None


def test_pipeline_transcriber_uses_configured_asr_language(tmp_path):
    settings = AppSettings(data_dir=tmp_path, models_dir=tmp_path / "models", asr_language="ru", allow_mock_models=True)

    transcriber = _transcriber(settings)

    assert transcriber.config.language == "ru"


def test_pipeline_transcriber_uses_turbo_only_on_cpu_or_metal(tmp_path):
    cpu_settings = AppSettings(data_dir=tmp_path, models_dir=tmp_path / "models", gpu_backend="cpu", allow_mock_models=True)
    cuda_settings = AppSettings(data_dir=tmp_path, models_dir=tmp_path / "models", gpu_backend="cuda", allow_mock_models=True)

    assert _transcriber(cpu_settings).config.engine == "openai/whisper-large-v3-turbo"
    assert _transcriber(cuda_settings).config.engine == "openai/whisper-large-v3"
