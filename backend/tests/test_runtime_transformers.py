from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from backend.app.config import normalize_qwen_cache_implementation
from backend.app.hf_pipeline.model_registry import model_for_role
from backend.app.runtime.transformers_runtime import (
    LoadedModel,
    ModelRuntimeError,
    TransformersModelManager,
    _chat_inputs,
    _first_complete_json_object_text,
    _generated_tokens_only,
    _has_complete_json_object,
)


def test_transformers_manager_loads_one_model_at_a_time(monkeypatch, tmp_path):
    manager = TransformersModelManager(models_dir=tmp_path / "models", logs_dir=tmp_path / "logs", gpu_backend="cpu")
    loads: list[str] = []

    def fake_load(spec):  # noqa: ANN001, ANN202 - test double.
        loads.append(spec.model_id)
        return LoadedModel(spec=spec, model=object(), device="cpu", dtype=spec.dtype, quantization="none")

    monkeypatch.setattr(manager, "_load", fake_load)
    first = model_for_role("summarizer", "default", device_backend="cpu")
    second = model_for_role("embedder", "default", device_backend="cpu")

    assert manager.ensure_loaded(first).spec.model_id == "Qwen/Qwen3.5-2B"
    assert manager.ensure_loaded(first).spec.model_id == "Qwen/Qwen3.5-2B"
    assert manager.ensure_loaded(second).spec.model_id == "Qwen/Qwen3-VL-Embedding-2B"
    assert loads == ["Qwen/Qwen3.5-2B", "Qwen/Qwen3-VL-Embedding-2B"]


def test_transformers_manager_reuses_same_role_and_model(monkeypatch, tmp_path):
    manager = TransformersModelManager(models_dir=tmp_path / "models", logs_dir=tmp_path / "logs", gpu_backend="cuda")
    loads = 0

    def fake_load(spec):  # noqa: ANN001, ANN202 - test double.
        nonlocal loads
        loads += 1
        return LoadedModel(spec=spec, model=object(), device="cuda", dtype=spec.dtype, quantization="nf4_4bit")

    monkeypatch.setattr(manager, "_load", fake_load)
    spec = model_for_role("embedder", "default", device_backend="cuda")

    manager.ensure_loaded(spec)
    manager.ensure_loaded(replace(spec))

    assert loads == 1


def test_transformers_manager_loads_qwen35_2b_unquantized_bf16(monkeypatch, tmp_path):
    records: dict[str, object] = {}

    class FakeTorch:
        bfloat16 = "bf16"
        float16 = "fp16"
        float32 = "fp32"
        cuda = SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)
        backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))

    class FakeLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):  # noqa: ANN002, ANN003, ANN205
            return object()

    class FakeModel:
        def eval(self):  # noqa: ANN201
            records["eval"] = True

    class FakeModelClass:
        @staticmethod
        def from_pretrained(model_id, **kwargs):  # noqa: ANN001, ANN003, ANN205
            records["model_id"] = model_id
            records["kwargs"] = kwargs
            return FakeModel()

    fake_transformers = SimpleNamespace(
        AutoProcessor=FakeLoader,
        AutoTokenizer=FakeLoader,
        AutoModelForImageTextToText=FakeModelClass,
    )
    monkeypatch.setattr("backend.app.runtime.transformers_runtime._torch", lambda: FakeTorch)
    monkeypatch.setattr("backend.app.runtime.transformers_runtime._transformers", lambda: fake_transformers)
    monkeypatch.setattr("backend.app.runtime.transformers_runtime._attention_candidates", lambda *_args: ["sdpa"])
    manager = TransformersModelManager(models_dir=tmp_path / "models", logs_dir=tmp_path / "logs", gpu_backend="cuda")
    spec = model_for_role("summarizer", "default", device_backend="cuda")
    loaded = manager._load(spec)

    kwargs = records["kwargs"]
    assert records["model_id"] == "Qwen/Qwen3.5-2B"
    assert "quantization_config" not in kwargs
    assert kwargs["device_map"] == {"": "cuda"}
    assert kwargs["dtype"] == "bf16"
    assert kwargs["attn_implementation"] == "sdpa"
    assert records["eval"] is True
    assert loaded.quantization == "none"


def test_qwen_cache_implementation_normalization() -> None:
    assert normalize_qwen_cache_implementation(None) == "auto"
    assert normalize_qwen_cache_implementation("off") == "auto"
    assert normalize_qwen_cache_implementation("offloaded") == "offloaded"
    assert normalize_qwen_cache_implementation("static") == "static"
    assert normalize_qwen_cache_implementation("unsupported") == "auto"


def test_json_completion_detector_handles_nested_strings() -> None:
    assert not _has_complete_json_object("")
    assert not _has_complete_json_object('{"title": "unfinished"')
    assert _has_complete_json_object('{"title": "done", "items": [{"text": "brace } inside string"}]}')
    assert _first_complete_json_object_text('prefix {"title": "done"} trailing') == '{"title": "done"}'


def test_transformers_manager_compiles_qwen_forward_when_enabled(monkeypatch, tmp_path):
    records: dict[str, object] = {}

    class FakeModel:
        def forward(self, value):  # noqa: ANN001, ANN201
            return f"forward:{value}"

    class FakeConfig:
        capture_dynamic_output_shape_ops = False
        cache_size_limit = 8

    class FakeTorch:
        _dynamo = SimpleNamespace(config=FakeConfig())
        cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))

        @staticmethod
        def compile(function, **kwargs):  # noqa: ANN001, ANN003, ANN205
            records["kwargs"] = kwargs

            def compiled(*args, **inner_kwargs):  # noqa: ANN002, ANN003, ANN202
                return function(*args, **inner_kwargs)

            return compiled

    monkeypatch.setattr("backend.app.runtime.transformers_runtime._torch", lambda: FakeTorch)
    manager = TransformersModelManager(
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        gpu_backend="cpu",
        torch_compile_mode="on",
        torch_compile_backend="inductor",
        torch_compile_profile="reduce-overhead",
    )
    spec = model_for_role("reranker", "default", device_backend="cpu")
    monkeypatch.setattr(
        manager,
        "_load",
        lambda loaded_spec: LoadedModel(spec=loaded_spec, model=FakeModel(), device="cpu"),
    )

    loaded = manager.ensure_loaded(spec)

    assert loaded.torch_compile_status == "compiled"
    assert loaded.torch_compile_target == "forward"
    assert records["kwargs"] == {
        "backend": "inductor",
        "fullgraph": False,
        "dynamic": False,
        "mode": "reduce-overhead",
    }
    assert FakeTorch._dynamo.config.capture_dynamic_output_shape_ops is True
    assert FakeTorch._dynamo.config.cache_size_limit == 1000
    assert loaded.model.forward("ok") == "forward:ok"


def test_transformers_manager_prefers_module_compile_method(monkeypatch, tmp_path):
    records: dict[str, object] = {"torch_compile_called": False}

    class FakeModel:
        def compile(self, **kwargs):  # noqa: ANN003, ANN201
            records["module_compile_kwargs"] = kwargs

        def forward(self, value):  # noqa: ANN001, ANN201
            return value

    class FakeTorch:
        cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))

        @staticmethod
        def compile(function, **kwargs):  # noqa: ANN001, ANN003, ANN205
            records["torch_compile_called"] = True
            return function

    monkeypatch.setattr("backend.app.runtime.transformers_runtime._torch", lambda: FakeTorch)
    manager = TransformersModelManager(
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        gpu_backend="cpu",
        torch_compile_mode="on",
        torch_compile_profile="default",
    )
    spec = model_for_role("reranker", "default", device_backend="cpu")
    monkeypatch.setattr(
        manager,
        "_load",
        lambda loaded_spec: LoadedModel(spec=loaded_spec, model=FakeModel(), device="cpu"),
    )

    loaded = manager.ensure_loaded(spec)

    assert loaded.torch_compile_status == "compiled"
    assert records["torch_compile_called"] is False
    assert records["module_compile_kwargs"] == {
        "backend": "inductor",
        "fullgraph": False,
        "dynamic": False,
    }


def test_transformers_manager_loads_and_saves_torch_compile_cache(monkeypatch, tmp_path):
    records: list[tuple[str, object]] = []
    cache_file = tmp_path / "compile-cache.bin"
    cache_file.write_bytes(b"old-cache")

    class FakeModel:
        def compile(self, **kwargs):  # noqa: ANN003, ANN201
            records.append(("compile", kwargs))

        def forward(self, value):  # noqa: ANN001, ANN201
            return value

    class FakeCompiler:
        @staticmethod
        def load_cache_artifacts(serialized_artifacts):  # noqa: ANN001, ANN205
            records.append(("load", serialized_artifacts))
            return SimpleNamespace(hit=True)

        @staticmethod
        def save_cache_artifacts():  # noqa: ANN205
            records.append(("save", None))
            return b"new-cache", SimpleNamespace(saved=True)

    class FakeTorch:
        compiler = FakeCompiler
        cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))

        @staticmethod
        def compile(function, **kwargs):  # noqa: ANN001, ANN003, ANN205
            return function

    monkeypatch.setattr("backend.app.runtime.transformers_runtime._torch", lambda: FakeTorch)
    manager = TransformersModelManager(
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        gpu_backend="cpu",
        torch_compile_mode="on",
        torch_compile_profile="default",
    )
    manager._compile_cache_path = lambda loaded, torch=None: cache_file  # type: ignore[method-assign]
    spec = model_for_role("reranker", "default", device_backend="cpu")
    monkeypatch.setattr(
        manager,
        "_load",
        lambda loaded_spec: LoadedModel(spec=loaded_spec, model=FakeModel(), device="cpu"),
    )

    loaded = manager.ensure_loaded(spec)

    assert [name for name, _ in records] == ["load", "compile", "save"]
    assert records[0][1] == b"old-cache"
    assert cache_file.read_bytes() == b"new-cache"
    assert loaded.torch_compile_cache_status == "saved"
    assert loaded.torch_compile_cache_path == str(cache_file)


def test_transformers_manager_saves_torch_compile_cache_after_first_execution(monkeypatch, tmp_path):
    records: list[str] = []
    cache_file = tmp_path / "compile-cache.bin"

    class InnerModel:
        def compile(self, **kwargs):  # noqa: ANN003, ANN201
            records.append("compile")

        def forward(self, value):  # noqa: ANN001, ANN201
            return value

    class FakeReranker:
        model = InnerModel()

        def process(self, payload):  # noqa: ANN001, ANN201
            records.append("process")
            return [0.5]

    class FakeCompiler:
        calls = 0

        @staticmethod
        def load_cache_artifacts(serialized_artifacts):  # noqa: ANN001, ANN205
            raise AssertionError("cache file should be absent")

        @classmethod
        def save_cache_artifacts(cls):  # noqa: ANN206
            cls.calls += 1
            records.append(f"save:{cls.calls}")
            if cls.calls == 1:
                return None
            return b"runtime-cache", SimpleNamespace(saved=True)

    class FakeTorch:
        compiler = FakeCompiler
        cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))

        @staticmethod
        def compile(function, **kwargs):  # noqa: ANN001, ANN003, ANN205
            return function

    monkeypatch.setattr("backend.app.runtime.transformers_runtime._torch", lambda: FakeTorch)
    manager = TransformersModelManager(
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        gpu_backend="cpu",
        torch_compile_mode="on",
        torch_compile_profile="default",
    )
    manager._compile_cache_path = lambda loaded, torch=None: cache_file  # type: ignore[method-assign]
    spec = model_for_role("reranker", "default", device_backend="cpu")
    monkeypatch.setattr(
        manager,
        "_load",
        lambda loaded_spec: LoadedModel(spec=loaded_spec, model=FakeReranker(), device="cpu"),
    )

    assert manager.rerank(spec, "query", ["document"]) == [0.5]

    loaded = manager.loaded
    assert loaded is not None
    assert records == ["compile", "save:1", "process", "save:2"]
    assert cache_file.read_bytes() == b"runtime-cache"
    assert loaded.torch_compile_cache_status == "saved"


def test_transformers_manager_skips_qwen35_summarizer_compile(monkeypatch, tmp_path):
    records: dict[str, int] = {"compile_calls": 0}

    class FakeTorch:
        cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))

        @staticmethod
        def compile(function, **kwargs):  # noqa: ANN001, ANN003, ANN205
            records["compile_calls"] += 1
            return function

    class FakeModel:
        def compile(self, **kwargs):  # noqa: ANN003, ANN201
            records["compile_calls"] += 1

        def forward(self):  # noqa: ANN201
            return None

    monkeypatch.setattr("backend.app.runtime.transformers_runtime._torch", lambda: FakeTorch)
    manager = TransformersModelManager(
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        gpu_backend="cpu",
        torch_compile_mode="on",
    )
    spec = model_for_role("summarizer", "default", device_backend="cpu")
    monkeypatch.setattr(
        manager,
        "_load",
        lambda loaded_spec: LoadedModel(spec=loaded_spec, model=FakeModel(), device="cpu"),
    )

    loaded = manager.ensure_loaded(spec)

    assert loaded.torch_compile_status == "skipped_qwen35_video_dynamic_shapes"
    assert records["compile_calls"] == 0


def test_transformers_manager_skips_qwen3_vl_embedding_compile(monkeypatch, tmp_path):
    records: dict[str, int] = {"compile_calls": 0}

    class FakeTorch:
        cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))

        @staticmethod
        def compile(function, **kwargs):  # noqa: ANN001, ANN003, ANN205
            records["compile_calls"] += 1
            return function

    class FakeModel:
        def compile(self, **kwargs):  # noqa: ANN003, ANN201
            records["compile_calls"] += 1

        def forward(self):  # noqa: ANN201
            return None

    monkeypatch.setattr("backend.app.runtime.transformers_runtime._torch", lambda: FakeTorch)
    manager = TransformersModelManager(
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        gpu_backend="cpu",
        torch_compile_mode="on",
    )
    spec = model_for_role("embedder", "default", device_backend="cpu")
    monkeypatch.setattr(
        manager,
        "_load",
        lambda loaded_spec: LoadedModel(spec=loaded_spec, model=FakeModel(), device="cpu"),
    )

    loaded = manager.ensure_loaded(spec)

    assert loaded.torch_compile_status == "skipped_qwen3_vl_embedding_dynamic_shapes"
    assert records["compile_calls"] == 0


def test_transformers_manager_skips_compile_for_asr_even_when_enabled(monkeypatch, tmp_path):
    records: dict[str, int] = {"compile_calls": 0}

    class FakeTorch:
        cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))

        @staticmethod
        def compile(function, **kwargs):  # noqa: ANN001, ANN003, ANN205
            records["compile_calls"] += 1
            return function

    class FakeModel:
        def forward(self):  # noqa: ANN201
            return None

    monkeypatch.setattr("backend.app.runtime.transformers_runtime._torch", lambda: FakeTorch)
    manager = TransformersModelManager(
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        gpu_backend="cpu",
        torch_compile_mode="on",
    )
    spec = model_for_role("asr", "default", device_backend="cpu")
    monkeypatch.setattr(
        manager,
        "_load",
        lambda loaded_spec: LoadedModel(spec=loaded_spec, model=FakeModel(), device="cpu"),
    )

    loaded = manager.ensure_loaded(spec)

    assert loaded.torch_compile_status == "skipped_unsupported_role"
    assert records["compile_calls"] == 0


def test_transformers_manager_compile_on_fails_for_supported_role(monkeypatch, tmp_path):
    class FakeTorch:
        cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))

        @staticmethod
        def compile(function, **kwargs):  # noqa: ANN001, ANN003, ANN205
            raise RuntimeError("backend cannot compile")

    class FakeModel:
        def forward(self):  # noqa: ANN201
            return None

    monkeypatch.setattr("backend.app.runtime.transformers_runtime._torch", lambda: FakeTorch)
    manager = TransformersModelManager(
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        gpu_backend="cpu",
        torch_compile_mode="on",
    )
    spec = model_for_role("reranker", "default", device_backend="cpu")
    monkeypatch.setattr(
        manager,
        "_load",
        lambda loaded_spec: LoadedModel(spec=loaded_spec, model=FakeModel(), device="cpu"),
    )

    with pytest.raises(ModelRuntimeError, match="torch.compile failed"):
        manager.ensure_loaded(spec)


def test_transformers_embedding_manager_contract(monkeypatch, tmp_path):
    manager = TransformersModelManager(models_dir=tmp_path / "models", logs_dir=tmp_path / "logs", gpu_backend="cuda")
    spec = model_for_role("embedder", "default", device_backend="cuda")

    monkeypatch.setattr(manager, "embed", lambda loaded_spec, values: [[0.1, 0.9] for _ in values])

    assert manager.embed(spec, ["query"])[0] == [0.1, 0.9]


class _FakeTensor:
    shape = (1, 3)


class _FakeBatch(dict):
    def to(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201 - test double.
        self["to_args"] = args
        self["to_kwargs"] = kwargs
        return self


def test_qwen_vl_chat_inputs_pass_video_metadata_and_disable_resize(monkeypatch, tmp_path):
    processor_calls: list[dict[str, object]] = []
    process_calls: list[dict[str, object]] = []
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")

    class FakeProcessor:
        image_processor = SimpleNamespace(patch_size=14)

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):  # noqa: ANN001, ANN201
            process_calls.append({"messages": messages, "template_kwargs": kwargs})
            return "prompt"

        def __call__(self, **kwargs):  # noqa: ANN003, ANN201
            processor_calls.append(kwargs)
            return _FakeBatch(input_ids=_FakeTensor())

    def fake_process_vision_info(messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
        process_calls.append({"messages": messages, "vision_kwargs": kwargs})
        metadata = {"fps": 2.0, "frames_indices": [0, 1], "total_num_frames": 2}
        return None, [("video_tensor", metadata)], {"do_sample_frames": False}

    monkeypatch.setitem(
        __import__("sys").modules,
        "qwen_vl_utils",
        SimpleNamespace(process_vision_info=fake_process_vision_info),
    )
    spec = model_for_role("summarizer")
    loaded = LoadedModel(spec=spec, model=object(), processor=FakeProcessor(), device="cpu")

    _chat_inputs(
        loaded,
        [{"role": "user", "content": [{"type": "video", "video": str(video_path)}, {"type": "text", "text": "Describe."}]}],
        {},
    )

    assert process_calls[1]["vision_kwargs"] == {
        "return_video_kwargs": True,
        "return_video_metadata": True,
        "image_patch_size": 14,
    }
    assert process_calls[1]["messages"][0]["content"][0]["video"].startswith("file://")
    assert processor_calls[0]["videos"] == ["video_tensor"]
    assert processor_calls[0]["video_metadata"] == [{"fps": 2.0, "frames_indices": [0, 1], "total_num_frames": 2}]
    assert processor_calls[0]["do_resize"] is False


def test_qwen_vl_chat_inputs_disable_resize_for_image_only_media(monkeypatch, tmp_path):
    processor_calls: list[dict[str, object]] = []
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake-image")

    class FakeProcessor:
        image_processor = SimpleNamespace(patch_size=14)

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):  # noqa: ANN001, ANN201
            return "prompt"

        def __call__(self, **kwargs):  # noqa: ANN003, ANN201
            processor_calls.append(kwargs)
            return _FakeBatch(input_ids=_FakeTensor())

    def fake_process_vision_info(messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
        return ["image_tensor"], None, {"do_sample_frames": False}

    monkeypatch.setitem(
        __import__("sys").modules,
        "qwen_vl_utils",
        SimpleNamespace(process_vision_info=fake_process_vision_info),
    )
    spec = model_for_role("summarizer")
    loaded = LoadedModel(spec=spec, model=object(), processor=FakeProcessor(), device="cpu")

    _chat_inputs(
        loaded,
        [{"role": "user", "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": "Describe."}]}],
        {},
    )

    assert processor_calls[0]["images"] == ["image_tensor"]
    assert processor_calls[0]["do_resize"] is False


def _manager_with_loaded(tmp_path, loaded: LoadedModel, **kwargs) -> TransformersModelManager:
    manager = TransformersModelManager(
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        gpu_backend="cpu",
        **kwargs,
    )
    manager.ensure_loaded = lambda spec: loaded  # type: ignore[method-assign]
    return manager


def test_generate_chat_passes_configured_cache_implementation(tmp_path):
    class FakeProcessor:
        tokenizer = None

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):  # noqa: ANN001, ANN201
            return "prompt"

        def __call__(self, text, **kwargs):  # noqa: ANN001, ANN003, ANN201
            return _FakeBatch(input_ids=torch.tensor([[10, 11]]), attention_mask=torch.tensor([[1, 1]]))

        def batch_decode(self, generated, *, skip_special_tokens):  # noqa: ANN001, ANN201
            return ["ok"]

    class FakeModel:
        def __init__(self) -> None:
            self.generate_calls: list[dict[str, object]] = []

        def generate(self, **kwargs):  # noqa: ANN003, ANN201
            self.generate_calls.append(kwargs)
            return torch.cat([kwargs["input_ids"], torch.tensor([[777]])], dim=-1)

    model = FakeModel()
    processor = FakeProcessor()
    spec = model_for_role("summarizer", "default", device_backend="cpu")
    loaded = LoadedModel(spec=spec, model=model, processor=processor, tokenizer=processor, device="cpu", dtype="fp32")

    result = _manager_with_loaded(
        tmp_path,
        loaded,
        generation_cache_implementation="offloaded",
    ).generate_chat(
        spec,
        [{"role": "user", "content": "Summarize."}],
        max_new_tokens=128,
        temperature=0.0,
    )

    assert result == "ok"
    assert model.generate_calls[0]["cache_implementation"] == "offloaded"


def test_generate_chat_bounds_json_generation_and_trims_after_first_object(tmp_path):
    class FakeProcessor:
        tokenizer = None

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):  # noqa: ANN001, ANN201
            return "prompt"

        def __call__(self, text, **kwargs):  # noqa: ANN001, ANN003, ANN201
            return _FakeBatch(input_ids=torch.tensor([[10, 11]]), attention_mask=torch.tensor([[1, 1]]))

        def decode(self, token_ids, *, skip_special_tokens):  # noqa: ANN001, ANN201
            return '{"title": "done"}'

        def batch_decode(self, generated, *, skip_special_tokens):  # noqa: ANN001, ANN201
            return ['{"title": "done"} repeated text that should be ignored']

    class FakeModel:
        def __init__(self) -> None:
            self.generate_calls: list[dict[str, object]] = []

        def generate(self, **kwargs):  # noqa: ANN003, ANN201
            self.generate_calls.append(kwargs)
            return torch.cat([kwargs["input_ids"], torch.tensor([[777]])], dim=-1)

    model = FakeModel()
    processor = FakeProcessor()
    spec = model_for_role("summarizer", "default", device_backend="cpu")
    loaded = LoadedModel(spec=spec, model=model, processor=processor, tokenizer=processor, device="cpu", dtype="fp32")

    result = _manager_with_loaded(tmp_path, loaded).generate_chat(
        spec,
        [{"role": "user", "content": "Summarize as JSON."}],
        max_new_tokens=128,
        temperature=0.0,
        stop_after_json=True,
    )

    call = model.generate_calls[0]
    assert result == '{"title": "done"}'
    assert call["repetition_penalty"] == 1.08
    assert "no_repeat_ngram_size" not in call
    assert call["renormalize_logits"] is True
    assert "stopping_criteria" in call


def test_generate_chat_applies_qwen_thinking_budget_with_continuation(tmp_path):
    calls: list[dict[str, object]] = []

    class FakeProcessor:
        tokenizer = None

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):  # noqa: ANN001, ANN201
            calls.append({"template_kwargs": kwargs})
            return "prompt"

        def __call__(self, text, **kwargs):  # noqa: ANN001, ANN003, ANN201
            if text == ["prompt"]:
                return _FakeBatch(
                    input_ids=torch.tensor([[10, 11]]),
                    attention_mask=torch.tensor([[1, 1]]),
                    mm_token_type_ids=torch.tensor([[0, 0]]),
                )
            return SimpleNamespace(input_ids=torch.tensor([[301, 151668]]))

        def convert_tokens_to_ids(self, token):  # noqa: ANN001, ANN201
            return {"<|im_end|>": 151645, "</think>": 151668}[token]

        def decode(self, token_ids, *, skip_special_tokens):  # noqa: ANN001, ANN201
            visible = [item for item in token_ids if item not in {151645, 151668}]
            if visible == [777]:
                return "final answer"
            return " ".join(str(item) for item in visible)

        def batch_decode(self, generated, *, skip_special_tokens):  # noqa: ANN001, ANN201
            return ["unused"]

    class FakeModel:
        def __init__(self) -> None:
            self.generate_calls: list[dict[str, object]] = []

        def generate(self, **kwargs):  # noqa: ANN003, ANN201
            self.generate_calls.append(kwargs)
            input_ids = kwargs["input_ids"]
            if len(self.generate_calls) == 1:
                return torch.cat([input_ids, torch.tensor([[201, 202]])], dim=-1)
            return torch.cat([input_ids, torch.tensor([[777, 151645]])], dim=-1)

    model = FakeModel()
    processor = FakeProcessor()
    processor.tokenizer = processor
    spec = model_for_role("summarizer", "default", device_backend="cpu")
    loaded = LoadedModel(spec=spec, model=model, processor=processor, tokenizer=processor, device="cpu", dtype="fp32")

    result = _manager_with_loaded(tmp_path, loaded).generate_chat(
        spec,
        [{"role": "user", "content": "Summarize."}],
        max_new_tokens=1536,
        temperature=0.0,
        chat_template_kwargs={"enable_thinking": True},
        thinking_budget_tokens=64,
    )

    assert result == "final answer"
    assert calls[0]["template_kwargs"] == {"enable_thinking": True}
    assert model.generate_calls[0]["max_new_tokens"] == 64
    assert model.generate_calls[1]["max_new_tokens"] == 1536
    assert model.generate_calls[1]["input_ids"].tolist()[0] == [10, 11, 201, 202, 301, 151668]
    assert model.generate_calls[1]["mm_token_type_ids"].tolist()[0] == [0, 0, 0, 0, 0, 0]


def test_generate_chat_streams_answer_chunks_without_reasoning(tmp_path):
    events: list[dict[str, object]] = []

    class FakeProcessor:
        tokenizer = None

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):  # noqa: ANN001, ANN201
            return "prompt"

        def __call__(self, text, **kwargs):  # noqa: ANN001, ANN003, ANN201
            return _FakeBatch(input_ids=torch.tensor([[10, 11]]), attention_mask=torch.tensor([[1, 1]]))

        def decode(self, token_ids, *, skip_special_tokens):  # noqa: ANN001, ANN201
            visible = [int(item) for item in token_ids if int(item) != 151645]
            if visible == [777]:
                return "final answer\n"
            return " ".join(str(item) for item in visible)

        def batch_decode(self, generated, *, skip_special_tokens):  # noqa: ANN001, ANN201
            return ["final answer"]

    class FakeModel:
        def generate(self, **kwargs):  # noqa: ANN003, ANN201
            input_ids = kwargs["input_ids"]
            streamer = kwargs.get("streamer")
            if streamer is not None:
                streamer.put(input_ids)
                streamer.put(torch.tensor([[777]]))
                streamer.end()
            return torch.cat([input_ids, torch.tensor([[777, 151645]])], dim=-1)

    processor = FakeProcessor()
    processor.tokenizer = processor
    spec = model_for_role("summarizer", "default", device_backend="cpu")
    loaded = LoadedModel(spec=spec, model=FakeModel(), processor=processor, tokenizer=processor, device="cpu", dtype="fp32")

    result = _manager_with_loaded(tmp_path, loaded).generate_chat(
        spec,
        [{"role": "user", "content": "Summarize."}],
        max_new_tokens=128,
        temperature=0.0,
        chat_template_kwargs={"enable_thinking": False},
        stream_callback=events.append,
    )

    assert result == "final answer"
    assert [event["event"] for event in events] == ["stream_start", "stream_chunk", "stream_end"]
    assert events[1]["phase"] == "answer"
    assert events[1]["text"] == "final answer\n"
    assert events[1]["redacted"] is False


def test_whisper_transcribe_uses_pipeline_chunking_and_timestamps(monkeypatch, tmp_path):
    records: dict[str, object] = {}

    class FakeProcessor:
        tokenizer = object()
        feature_extractor = object()

    def fake_pipeline(task, **kwargs):  # noqa: ANN001, ANN003, ANN201
        records["pipeline"] = {"task": task, **kwargs}

        def run(audio, **call_kwargs):  # noqa: ANN001, ANN003, ANN202
            records["call"] = {"audio": audio, **call_kwargs}
            return {"text": "rotate left", "chunks": [{"timestamp": (0.0, 1.2), "text": "rotate left"}]}

        return run

    monkeypatch.setattr(
        "backend.app.runtime.transformers_runtime._transformers",
        lambda: SimpleNamespace(pipeline=fake_pipeline),
    )
    monkeypatch.setattr(
        "backend.app.runtime.transformers_runtime._audio_array_for_asr",
        lambda path, *, sample_rate: {"array": [0.0], "sampling_rate": sample_rate},
    )
    monkeypatch.setattr(
        "backend.app.runtime.transformers_runtime._audio_array_for_asr",
        lambda path, *, sample_rate: {"array": [0.0, 0.1, 0.0], "sampling_rate": sample_rate},
    )
    spec = model_for_role("asr")
    loaded = LoadedModel(spec=spec, model=object(), processor=FakeProcessor(), device="cpu", dtype="fp32")
    result = _manager_with_loaded(tmp_path, loaded).transcribe(spec, tmp_path / "audio.wav", language="auto")

    assert records["pipeline"]["task"] == "automatic-speech-recognition"
    assert records["pipeline"]["chunk_length_s"] == 30
    assert "dtype" in records["pipeline"]
    assert "torch_dtype" not in records["pipeline"]
    assert records["call"]["audio"] == {"array": [0.0, 0.1, 0.0], "sampling_rate": 16000}
    assert records["call"]["return_timestamps"] is True
    assert "language" not in records["call"]["generate_kwargs"]
    assert records["call"]["generate_kwargs"]["max_new_tokens"] == 445
    assert result.text == "rotate left"
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 1.2


def test_whisper_transcribe_passes_configured_language(monkeypatch, tmp_path):
    records: dict[str, object] = {}

    class FakeProcessor:
        tokenizer = object()
        feature_extractor = object()

    def fake_pipeline(task, **kwargs):  # noqa: ANN001, ANN003, ANN201
        def run(path, **call_kwargs):  # noqa: ANN001, ANN003, ANN202
            records.update(call_kwargs)
            return {"text": "hallo", "chunks": [{"timestamp": (0.0, 0.7), "text": "hallo"}]}

        return run

    monkeypatch.setattr(
        "backend.app.runtime.transformers_runtime._transformers",
        lambda: SimpleNamespace(pipeline=fake_pipeline),
    )
    monkeypatch.setattr(
        "backend.app.runtime.transformers_runtime._audio_array_for_asr",
        lambda path, *, sample_rate: {"array": [0.0], "sampling_rate": sample_rate},
    )
    spec = model_for_role("asr")
    loaded = LoadedModel(spec=spec, model=object(), processor=FakeProcessor(), device="cpu", dtype="fp32")
    result = _manager_with_loaded(tmp_path, loaded).transcribe(spec, tmp_path / "audio.wav", language="de")

    assert records["generate_kwargs"]["language"] == "de"
    assert result.language == "de"


def test_midashenglm_caption_audio_uses_chat_template_audio_path(monkeypatch, tmp_path):
    records: dict[str, object] = {}
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
            records["messages"] = messages
            records["template_kwargs"] = kwargs
            return _FakeBatch(input_ids=_FakeTensor())

        def __call__(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
            raise AssertionError("text-only processor path should not be used")

    class FakeModel:
        dtype = "float32"

        def generate(self, **kwargs):  # noqa: ANN003, ANN201
            records["generate_kwargs"] = kwargs
            return [[1, 2, 3]]

    class FakeTokenizer:
        def batch_decode(self, values, *, skip_special_tokens):  # noqa: ANN001, ANN201
            records["decoded"] = values
            return ["possible footsteps"]

    spec = model_for_role("audio_captioner")
    loaded = LoadedModel(
        spec=spec,
        model=FakeModel(),
        processor=FakeProcessor(),
        tokenizer=FakeTokenizer(),
        device="cuda",
        dtype="fp32",
    )

    caption = _manager_with_loaded(tmp_path, loaded).caption_audio(spec, audio_path, prompt="Caption audio", max_new_tokens=192)

    assert records["messages"][0]["content"] == [
        {"type": "text", "text": "Caption audio"},
        {"type": "audio", "path": str(audio_path)},
    ]
    assert records["template_kwargs"] == {
        "tokenize": True,
        "add_generation_prompt": True,
        "add_special_tokens": True,
        "return_dict": True,
    }
    assert records["generate_kwargs"]["do_sample"] is True
    assert records["generate_kwargs"]["top_p"] == 0.8
    assert records["generate_kwargs"]["top_k"] == 50
    assert records["generate_kwargs"]["temperature"] == 1.0
    assert records["generate_kwargs"]["repetition_penalty"] == 1.05
    assert records["generate_kwargs"]["max_new_tokens"] == 192
    assert caption == "possible footsteps"


def test_generated_tokens_only_preserves_midasheng_new_token_outputs() -> None:
    inputs = {"input_ids": torch.arange(12).reshape(1, 12)}
    output_ids = torch.tensor([[200, 201]])

    generated = _generated_tokens_only(output_ids, inputs)

    assert torch.equal(generated, output_ids)


def test_generated_tokens_only_trims_prompt_prefixed_outputs() -> None:
    inputs = {"input_ids": torch.tensor([[10, 11, 12]])}
    output_ids = torch.tensor([[10, 11, 12, 200, 201]])

    generated = _generated_tokens_only(output_ids, inputs)

    assert torch.equal(generated, torch.tensor([[200, 201]]))


def test_embedding_uses_qwen3_vl_process_for_video_payload(tmp_path):
    calls: list[object] = []

    class FakeQwen3VLEmbedder:
        def process(self, values):  # noqa: ANN001, ANN201
            calls.append(values)
            return [[0.0, 2.0]]

        def __call__(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
            raise AssertionError("generic hidden-state fallback used")

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    spec = model_for_role("embedder")
    loaded = LoadedModel(spec=spec, model=FakeQwen3VLEmbedder())

    vector = _manager_with_loaded(tmp_path, loaded).embed(
        spec,
        [
            {
                "instruction": "Represent the gameplay video content for retrieval.",
                "text": "full clip",
                "video_path": str(video),
                "video_fps": 2.0,
                "video_max_frames": 64,
            }
        ],
    )[0]

    assert vector == [0.0, 1.0]
    payload = calls[0][0]
    assert payload["instruction"] == "Represent the gameplay video content for retrieval."
    assert payload["text"] == "full clip"
    assert payload["video"] == str(video)
    assert payload["fps"] == 2.0
    assert payload["max_frames"] == 64


def test_embedding_qwen3_vl_rejects_frame_sequence_payload(tmp_path):
    class FakeQwen3VLEmbedder:
        def process(self, values):  # noqa: ANN001, ANN201
            raise AssertionError("frame_paths should be rejected before model.process")

    spec = model_for_role("embedder")
    loaded = LoadedModel(spec=spec, model=FakeQwen3VLEmbedder())

    with pytest.raises(ModelRuntimeError, match="frame_paths are reserved"):
        _manager_with_loaded(tmp_path, loaded).embed(spec, [{"frame_paths": ["frame.jpg"]}])


def test_embedding_qwen3_vl_accepts_prepared_video_frames(tmp_path):
    calls: list[object] = []

    class FakeQwen3VLEmbedder:
        def process(self, values):  # noqa: ANN001, ANN201
            calls.append(values)
            return [[1.0, 0.0]]

    frame_a = tmp_path / "frame_a.png"
    frame_b = tmp_path / "frame_b.png"
    frame_a.write_bytes(b"fake")
    frame_b.write_bytes(b"fake")
    spec = model_for_role("embedder")
    loaded = LoadedModel(spec=spec, model=FakeQwen3VLEmbedder())

    vector = _manager_with_loaded(tmp_path, loaded).embed(
        spec,
        [
            {
                "instruction": "Represent the gameplay video content for retrieval.",
                "text": "prepared full clip",
                "video_frames": [str(frame_a), str(frame_b)],
                "video_fps": 2.0,
                "video_max_frames": 2,
            }
        ],
    )[0]

    assert vector == [1.0, 0.0]
    payload = calls[0][0]
    assert payload["video"] == [str(frame_a), str(frame_b)]
    assert payload["fps"] == 2.0
    assert payload["max_frames"] == 2


def test_embedding_unsupported_model_api_raises(tmp_path):
    spec = model_for_role("embedder")
    loaded = LoadedModel(spec=spec, model=object())

    with pytest.raises(ModelRuntimeError, match="supported Qwen3-VL embedding API"):
        _manager_with_loaded(tmp_path, loaded).embed(spec, ["query"])


def test_reranker_uses_qwen3_vl_process(tmp_path):
    calls: list[object] = []

    class FakeQwen3VLReranker:
        def process(self, payload):  # noqa: ANN001, ANN201
            calls.append(payload)
            return [0.3, 0.7]

    spec = model_for_role("reranker")
    loaded = LoadedModel(spec=spec, model=FakeQwen3VLReranker())

    assert _manager_with_loaded(tmp_path, loaded).rerank(spec, "query", ["doc a", "doc b"]) == [0.3, 0.7]
    assert calls[0]["query"] == {"text": "query"}
    assert calls[0]["documents"] == [{"text": "doc a"}, {"text": "doc b"}]


def test_reranker_unsupported_model_api_raises(tmp_path):
    spec = model_for_role("reranker")
    loaded = LoadedModel(spec=spec, model=object())

    with pytest.raises(ModelRuntimeError, match="supported Qwen3-VL reranker API"):
        _manager_with_loaded(tmp_path, loaded).rerank(spec, "query", ["doc"])
