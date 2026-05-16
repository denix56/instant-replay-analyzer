from __future__ import annotations

import hashlib
import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .hf_pipeline.model_registry import (
    DEFAULT_MODEL_REGISTRY,
    VIDEO_EMBED_MAX_FRAMES_QWEN_DEFAULT,
    VIDEO_EMBED_MAX_PIXELS_QWEN_LIMIT,
    VIDEO_FOCUS_MAX_FRAMES_QWEN_DEFAULT,
    VIDEO_FOCUS_MAX_PIXELS_QWEN_LIMIT,
    VIDEO_MAX_FRAMES_QWEN_DEFAULT,
    VIDEO_MAX_PIXELS_QWEN_LIMIT,
    VIDEO_OCR_MAX_FRAMES_QWEN_DEFAULT,
    VIDEO_OCR_MAX_PIXELS_QWEN_LIMIT,
    ModelTier,
    normalize_model_tier,
)

try:
    import yaml
except Exception:  # pragma: no cover - optional at import time
    yaml = None


RuntimeProfile = Literal["auto", "nvidia", "amd", "macos", "cpu"]
GpuBackend = Literal["cuda", "rocm", "macos-metal", "cpu", "unknown"]
IndexingProfile = Literal["fast", "balanced", "detailed"]
QwenReasoningMode = Literal["off", "low", "full"]
TorchCompileMode = Literal["off", "auto", "on"]
QwenCacheImplementation = Literal["auto", "dynamic", "offloaded", "static", "quantized"]

EMBEDDING_MODEL = DEFAULT_MODEL_REGISTRY["embedder"].default_model_id
RERANKER_MODEL = DEFAULT_MODEL_REGISTRY["reranker"].default_model_id
SUMMARIZER_MODEL = DEFAULT_MODEL_REGISTRY["summarizer"].default_model_id

PRECISION_VALUES = {
    "nf4_4bit",
    "int8_8bit",
    "bf16",
    "fp16",
    "fp32",
    "none",
    "unknown",
}

RUNTIME_VALUES = {"transformers", "cpu", "mps", "metal", "macos-metal", "cuda", "rocm", "unknown"}
DEFAULT_QWEN_REASONING_BUDGET_TOKENS = 1024
FULL_QWEN_REASONING_BUDGET_TOKENS = 2048
SUPPORTED_VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}


@dataclass(frozen=True)
class ModelTierSettings:
    name: ModelTier
    multimodal_retrieval_model: str
    multimodal_retrieval_precision: str
    asr_model: str
    audio_captioner_model: str
    enable_reranking: bool
    reranker_model: str
    deep_reasoning_model: str
    enable_deep_reasoning: bool


@dataclass(frozen=True)
class IndexingSettings:
    name: IndexingProfile
    segment_seconds: float
    segment_stride_seconds: float
    representative_frames_per_segment: int
    store_video_segment_files: bool
    store_audio_segment_files: bool

    def segment_settings_hash(self) -> str:
        payload = (
            f"{self.name}|{self.segment_seconds}|{self.segment_stride_seconds}|"
            f"{self.representative_frames_per_segment}|{self.store_video_segment_files}|"
            f"{self.store_audio_segment_files}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


MODEL_TIERS: dict[str, ModelTierSettings] = {
    "default": ModelTierSettings(
        name="default",
        multimodal_retrieval_model=DEFAULT_MODEL_REGISTRY["embedder"].default_model_id,
        multimodal_retrieval_precision="nf4_4bit",
        asr_model=DEFAULT_MODEL_REGISTRY["asr"].default_model_id,
        audio_captioner_model=DEFAULT_MODEL_REGISTRY["audio_captioner"].default_model_id,
        enable_reranking=True,
        reranker_model=DEFAULT_MODEL_REGISTRY["reranker"].default_model_id,
        deep_reasoning_model=DEFAULT_MODEL_REGISTRY["summarizer"].default_model_id,
        enable_deep_reasoning=False,
    ),
    "quality": ModelTierSettings(
        name="quality",
        multimodal_retrieval_model=DEFAULT_MODEL_REGISTRY["embedder"].quality_model_id,
        multimodal_retrieval_precision="nf4_4bit",
        asr_model=DEFAULT_MODEL_REGISTRY["asr"].quality_model_id,
        audio_captioner_model=DEFAULT_MODEL_REGISTRY["audio_captioner"].quality_model_id,
        enable_reranking=True,
        reranker_model=DEFAULT_MODEL_REGISTRY["reranker"].quality_model_id,
        deep_reasoning_model=DEFAULT_MODEL_REGISTRY["summarizer"].quality_model_id,
        enable_deep_reasoning=True,
    ),
}

INDEXING_PROFILES: dict[str, IndexingSettings] = {
    "fast": IndexingSettings(
        name="fast",
        segment_seconds=2.0,
        segment_stride_seconds=2.0,
        representative_frames_per_segment=1,
        store_video_segment_files=False,
        store_audio_segment_files=True,
    ),
    "balanced": IndexingSettings(
        name="balanced",
        segment_seconds=2.0,
        segment_stride_seconds=1.0,
        representative_frames_per_segment=2,
        store_video_segment_files=False,
        store_audio_segment_files=True,
    ),
    "detailed": IndexingSettings(
        name="detailed",
        segment_seconds=1.5,
        segment_stride_seconds=0.5,
        representative_frames_per_segment=3,
        store_video_segment_files=True,
        store_audio_segment_files=True,
    ),
}


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def normalize_indexing_profile(value: str | None) -> IndexingProfile:
    raw = (value or "balanced").strip().lower()
    if raw not in INDEXING_PROFILES:
        return "balanced"
    return raw  # type: ignore[return-value]


def normalize_runtime_profile(value: str | None) -> RuntimeProfile:
    raw = (value or "auto").strip().lower()
    if raw in {"nvidia", "amd", "macos", "cpu", "auto"}:
        return raw  # type: ignore[return-value]
    return "auto"


def normalize_qwen_reasoning_mode(value: str | None) -> QwenReasoningMode:
    raw = (value or "off").strip().lower()
    if raw in {"off", "none", "false", "0"}:
        return "off"
    if raw in {"full", "high", "on", "true", "1"}:
        return "full"
    return "low"


def normalize_torch_compile_mode(value: str | None) -> TorchCompileMode:
    raw = (value or "off").strip().lower()
    if raw in {"on", "true", "1", "yes"}:
        return "on"
    if raw in {"auto", "try"}:
        return "auto"
    return "off"


def normalize_qwen_cache_implementation(value: str | None) -> QwenCacheImplementation:
    raw = (value or "auto").strip().lower().replace("-", "_")
    if raw in {"", "auto", "none", "off", "false", "0"}:
        return "auto"
    if raw in {"dynamic", "offloaded", "static", "quantized"}:
        return raw  # type: ignore[return-value]
    return "auto"


def normalize_asr_language(value: Any) -> str:
    normalized = str(value or "auto").strip().lower()
    return normalized or "auto"


def gpu_backend_for_profile(profile: RuntimeProfile) -> GpuBackend:
    if profile == "auto":
        return detect_gpu_backend()
    return {
        "nvidia": "cuda",
        "amd": "rocm",
        "macos": "macos-metal",
        "cpu": "cpu",
    }[profile]


def detect_gpu_backend() -> GpuBackend:
    if platform.system().lower() == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return "macos-metal"
    if shutil.which("nvidia-smi") or shutil.which("nvcc"):
        return "cuda"
    if shutil.which("hipcc") or shutil.which("rocm-smi") or Path("/dev/kfd").exists():
        return "rocm"
    return "cpu"


@dataclass
class AppSettings:
    clips_dir: Path = Path("clips")
    data_dir: Path = Path("/data")
    models_dir: Path = Path("/models")
    model_tier: ModelTier = "default"
    indexing_profile: IndexingProfile = "balanced"
    runtime_profile: RuntimeProfile = "auto"
    gpu_backend: GpuBackend = "unknown"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    qdrant_url: str = "local"
    vector_backend: str = "qdrant"
    cuda_version: str = "12.8"
    enable_transcription: bool = True
    enable_reranking: bool | None = None
    enable_deep_reasoning: bool | None = None
    enable_audio_event_detection: bool = False
    asr_language: str = "auto"
    allow_mock_models: bool = True
    auto_download_models: bool = True
    av_segment_weight: float = 0.55
    transcript_weight: float = 0.25
    metadata_weight: float = 0.05
    reranker_weight: float = 0.15
    search_min_score: float = 0.35
    search_candidate_limit: int = 100
    video_embedding_fps: float = 6.0
    video_embedding_max_frames: int = VIDEO_EMBED_MAX_FRAMES_QWEN_DEFAULT
    video_analysis_skip_start_sec: float = 5.0
    qwen_vram_max_allocated_gb: float = 8.0
    qwen_full_video_max_frames: int = VIDEO_MAX_FRAMES_QWEN_DEFAULT
    qwen_full_video_max_pixels: int = VIDEO_MAX_PIXELS_QWEN_LIMIT
    qwen_focus_video_max_frames: int = VIDEO_FOCUS_MAX_FRAMES_QWEN_DEFAULT
    qwen_focus_video_max_pixels: int = VIDEO_FOCUS_MAX_PIXELS_QWEN_LIMIT
    qwen_ocr_video_max_frames: int = VIDEO_OCR_MAX_FRAMES_QWEN_DEFAULT
    qwen_ocr_video_max_pixels: int = VIDEO_OCR_MAX_PIXELS_QWEN_LIMIT
    qwen_video_embed_max_frames: int = VIDEO_EMBED_MAX_FRAMES_QWEN_DEFAULT
    qwen_video_embed_max_pixels: int = VIDEO_EMBED_MAX_PIXELS_QWEN_LIMIT
    qwen_reasoning_mode: QwenReasoningMode = "off"
    qwen_reasoning_budget_tokens: int = DEFAULT_QWEN_REASONING_BUDGET_TOKENS
    qwen_cache_implementation: QwenCacheImplementation = "auto"
    torch_compile_mode: TorchCompileMode = "off"
    torch_compile_backend: str = "inductor"
    torch_compile_profile: str = "reduce-overhead"
    cors_origins: list[str] = field(default_factory=lambda: ["tauri://localhost", "http://localhost:1420", "http://127.0.0.1:1420"])

    @property
    def tier(self) -> ModelTierSettings:
        return MODEL_TIERS[self.model_tier]

    @property
    def indexing(self) -> IndexingSettings:
        return INDEXING_PROFILES[self.indexing_profile]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def config_path(self) -> Path:
        return self.data_dir / "config.yaml"

    @property
    def segment_cache_dir(self) -> Path:
        return self.data_dir / "segments"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @classmethod
    def from_env(cls) -> "AppSettings":
        project_dir = _default_project_dir()
        data_dir = Path(os.getenv("DATA_DIR", str(project_dir / "data")))
        persisted = _load_yaml(data_dir / "config.yaml")

        def get(key: str, default: Any = None) -> Any:
            return os.getenv(key, persisted.get(key.lower(), default))

        runtime_profile = normalize_runtime_profile(get("COMPOSE_PROFILE", get("RUNTIME_PROFILE", "auto")))
        model_tier = normalize_model_tier(str(get("MODEL_TIER", "default")))
        indexing_profile = normalize_indexing_profile(get("INDEXING_PROFILE", "balanced"))
        gpu_backend = str(get("GPU_BACKEND", "") or "").strip().lower()
        if not gpu_backend or (gpu_backend == "unknown" and runtime_profile == "auto"):
            gpu_backend = gpu_backend_for_profile(runtime_profile)
        if gpu_backend in {"metal", "mps"}:
            gpu_backend = "macos-metal"
        elif gpu_backend == "nvidia":
            gpu_backend = "cuda"
        elif gpu_backend in {"amd", "hip"}:
            gpu_backend = "rocm"
        if gpu_backend == "macos-metal":
            gpu_backend = "macos-metal"
        elif gpu_backend not in {"cuda", "rocm", "cpu", "unknown"}:
            gpu_backend = gpu_backend_for_profile(runtime_profile)

        tier = MODEL_TIERS[model_tier]
        default_reasoning_mode = "low" if model_tier == "quality" else "off"
        return cls(
            clips_dir=Path(get("CLIPS_DIR", get("HOST_CLIPS_DIR", "clips"))),
            data_dir=data_dir,
            models_dir=Path(get("MODELS_DIR", str(project_dir / "models"))),
            model_tier=model_tier,
            indexing_profile=indexing_profile,
            runtime_profile=runtime_profile,
            gpu_backend=gpu_backend,
            backend_host=str(get("BACKEND_HOST", "0.0.0.0")),
            backend_port=int(get("BACKEND_PORT", 8000)),
            qdrant_url=str(get("QDRANT_URL", "local")),
            vector_backend=str(get("VECTOR_BACKEND", "qdrant")),
            cuda_version=str(get("CUDA_VERSION", "12.8")),
            enable_transcription=parse_bool(get("ENABLE_TRANSCRIPTION", True), True),
            enable_reranking=parse_bool(get("ENABLE_RERANKING", tier.enable_reranking), tier.enable_reranking),
            enable_deep_reasoning=parse_bool(
                get("ENABLE_DEEP_REASONING", tier.enable_deep_reasoning),
                tier.enable_deep_reasoning,
            ),
            enable_audio_event_detection=parse_bool(get("ENABLE_AUDIO_EVENT_DETECTION", False), False),
            asr_language=normalize_asr_language(get("ASR_LANGUAGE", "auto")),
            allow_mock_models=parse_bool(get("ALLOW_MOCK_MODELS", True), True),
            auto_download_models=parse_bool(get("AUTO_DOWNLOAD_MODELS", True), True),
            av_segment_weight=float(get("AV_SEGMENT_WEIGHT", 0.55)),
            transcript_weight=float(get("TRANSCRIPT_WEIGHT", 0.25)),
            metadata_weight=float(get("METADATA_WEIGHT", 0.05)),
            reranker_weight=float(get("RERANKER_WEIGHT", 0.15)),
            search_min_score=float(get("SEARCH_MIN_SCORE", 0.35)),
            search_candidate_limit=int(get("SEARCH_CANDIDATE_LIMIT", 100)),
            video_embedding_fps=float(get("VIDEO_EMBEDDING_FPS", 6.0)),
            video_embedding_max_frames=int(
                get("VIDEO_EMBEDDING_MAX_FRAMES", VIDEO_EMBED_MAX_FRAMES_QWEN_DEFAULT)
            ),
            video_analysis_skip_start_sec=max(0.0, float(get("VIDEO_ANALYSIS_SKIP_START_SEC", 5.0))),
            qwen_vram_max_allocated_gb=max(0.1, float(get("QWEN_VRAM_MAX_ALLOCATED_GB", 8.0))),
            qwen_full_video_max_frames=max(1, int(get("QWEN_FULL_VIDEO_MAX_FRAMES", VIDEO_MAX_FRAMES_QWEN_DEFAULT))),
            qwen_full_video_max_pixels=max(1, int(get("QWEN_FULL_VIDEO_MAX_PIXELS", VIDEO_MAX_PIXELS_QWEN_LIMIT))),
            qwen_focus_video_max_frames=max(1, int(get("QWEN_FOCUS_VIDEO_MAX_FRAMES", VIDEO_FOCUS_MAX_FRAMES_QWEN_DEFAULT))),
            qwen_focus_video_max_pixels=max(1, int(get("QWEN_FOCUS_VIDEO_MAX_PIXELS", VIDEO_FOCUS_MAX_PIXELS_QWEN_LIMIT))),
            qwen_ocr_video_max_frames=max(1, int(get("QWEN_OCR_VIDEO_MAX_FRAMES", VIDEO_OCR_MAX_FRAMES_QWEN_DEFAULT))),
            qwen_ocr_video_max_pixels=max(1, int(get("QWEN_OCR_VIDEO_MAX_PIXELS", VIDEO_OCR_MAX_PIXELS_QWEN_LIMIT))),
            qwen_video_embed_max_frames=max(1, int(get("QWEN_VIDEO_EMBED_MAX_FRAMES", VIDEO_EMBED_MAX_FRAMES_QWEN_DEFAULT))),
            qwen_video_embed_max_pixels=max(1, int(get("QWEN_VIDEO_EMBED_MAX_PIXELS", VIDEO_EMBED_MAX_PIXELS_QWEN_LIMIT))),
            qwen_reasoning_mode=normalize_qwen_reasoning_mode(
                get("QWEN_REASONING_MODE", default_reasoning_mode)
            ),
            qwen_reasoning_budget_tokens=max(
                0,
                int(get("QWEN_REASONING_BUDGET_TOKENS", DEFAULT_QWEN_REASONING_BUDGET_TOKENS)),
            ),
            qwen_cache_implementation=normalize_qwen_cache_implementation(
                get("QWEN_CACHE_IMPLEMENTATION", "auto")
            ),
            torch_compile_mode=normalize_torch_compile_mode(get("TORCH_COMPILE_MODE", "off")),
            torch_compile_backend=str(get("TORCH_COMPILE_BACKEND", "inductor")),
            torch_compile_profile=str(get("TORCH_COMPILE_PROFILE", "reduce-overhead")),
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.segment_cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def persist(self) -> None:
        self.ensure_dirs()
        payload = {
            "clips_dir": str(self.clips_dir),
            "data_dir": str(self.data_dir),
            "models_dir": str(self.models_dir),
            "model_tier": self.model_tier,
            "indexing_profile": self.indexing_profile,
            "runtime_profile": self.runtime_profile,
            "gpu_backend": self.gpu_backend,
            "backend_port": self.backend_port,
            "qdrant_url": self.qdrant_url,
            "asr_language": self.asr_language,
            "video_embedding_fps": self.video_embedding_fps,
            "video_embedding_max_frames": self.video_embedding_max_frames,
            "video_analysis_skip_start_sec": self.video_analysis_skip_start_sec,
            "qwen_vram_max_allocated_gb": self.qwen_vram_max_allocated_gb,
            "qwen_full_video_max_frames": self.qwen_full_video_max_frames,
            "qwen_full_video_max_pixels": self.qwen_full_video_max_pixels,
            "qwen_focus_video_max_frames": self.qwen_focus_video_max_frames,
            "qwen_focus_video_max_pixels": self.qwen_focus_video_max_pixels,
            "qwen_ocr_video_max_frames": self.qwen_ocr_video_max_frames,
            "qwen_ocr_video_max_pixels": self.qwen_ocr_video_max_pixels,
            "qwen_video_embed_max_frames": self.qwen_video_embed_max_frames,
            "qwen_video_embed_max_pixels": self.qwen_video_embed_max_pixels,
            "qwen_reasoning_mode": self.qwen_reasoning_mode,
            "qwen_reasoning_budget_tokens": self.qwen_reasoning_budget_tokens,
            "qwen_cache_implementation": self.qwen_cache_implementation,
            "torch_compile_mode": self.torch_compile_mode,
            "torch_compile_backend": self.torch_compile_backend,
            "torch_compile_profile": self.torch_compile_profile,
        }
        if yaml is None:
            lines = [f"{key}: {value}\n" for key, value in payload.items()]
            self.config_path.write_text("".join(lines), encoding="utf-8")
        else:
            self.config_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        return {}
    return {str(key).lower(): value for key, value in loaded.items()}


def get_settings() -> AppSettings:
    settings = AppSettings.from_env()
    settings.ensure_dirs()
    return settings


def _default_project_dir() -> Path:
    cwd = Path.cwd()
    if cwd.name == "backend":
        return cwd.parent
    return cwd
