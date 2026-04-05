import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import urllib.error

import pytest

import manuscript.api.base as base_module
from manuscript.api.base import BaseArtifactModel


class EdgeModel(BaseArtifactModel):
    default_weights_name = "default_model.onnx"
    pretrained_registry = {}

    def _initialize_session(self):
        self.session = object()

    def predict(self, value):
        return value


def _make_model(tmp_path, *, device="cpu"):
    weights_file = tmp_path / "model.onnx"
    weights_file.write_text("mock", encoding="utf-8")
    return EdgeModel(weights=str(weights_file), device=device)


class _FakeUrlopenResponse:
    def __init__(self, size):
        self.headers = {"content-length": str(size)}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTqdm:
    def __init__(self, *args, **kwargs):
        self.n = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, amount):
        self.n += amount


class TestBaseArtifactModelRuntimeEdges:
    def test_runtime_providers_coreml(self, tmp_path):
        model = _make_model(tmp_path, device="coreml")
        assert model.runtime_providers() == [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]

    def test_log_device_info_warns_when_cuda_provider_unavailable(
        self, tmp_path, capsys
    ):
        model = _make_model(tmp_path, device="cuda")
        session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

        model._log_device_info(session)

        captured = capsys.readouterr().out
        assert "CUDA requested but not available" in captured
        assert "onnxruntime-gpu" in captured

    def test_log_device_info_warns_when_coreml_provider_unavailable(
        self, tmp_path, capsys
    ):
        model = _make_model(tmp_path, device="coreml")
        session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

        model._log_device_info(session)

        captured = capsys.readouterr().out
        assert "CoreML requested but not available" in captured
        assert "onnxruntime-silicon" in captured

    def test_download_http_uses_progress_bar_when_tqdm_available(self, tmp_path):
        model = _make_model(tmp_path)
        cache_root = tmp_path / "cache_home"

        def fake_urlretrieve(url, target, reporthook=None):
            if reporthook is not None:
                reporthook(1, 4, 10)
                reporthook(2, 4, 10)
            Path(target).write_text("payload", encoding="utf-8")

        with patch.object(base_module.Path, "home", return_value=cache_root):
            with patch.object(base_module, "tqdm", _FakeTqdm):
                with patch(
                    "manuscript.api.base.urllib.request.urlopen",
                    return_value=_FakeUrlopenResponse(size=10),
                ):
                    with patch(
                        "manuscript.api.base.urllib.request.urlretrieve",
                        side_effect=fake_urlretrieve,
                    ) as mock_urlretrieve:
                        result = model._download_http("https://example.com/model.onnx")

        assert Path(result).exists()
        assert Path(result).read_text(encoding="utf-8") == "payload"
        assert mock_urlretrieve.call_count == 1

    def test_download_http_falls_back_when_progress_bar_setup_fails(
        self, tmp_path, capsys
    ):
        model = _make_model(tmp_path)
        cache_root = tmp_path / "cache_home"

        def fake_urlretrieve(url, target, reporthook=None):
            assert reporthook is None
            Path(target).write_text("payload", encoding="utf-8")

        with patch.object(base_module.Path, "home", return_value=cache_root):
            with patch.object(base_module, "tqdm", _FakeTqdm):
                with patch(
                    "manuscript.api.base.urllib.request.urlopen",
                    side_effect=RuntimeError("broken progress setup"),
                ):
                    with patch(
                        "manuscript.api.base.urllib.request.urlretrieve",
                        side_effect=fake_urlretrieve,
                    ):
                        result = model._download_http("https://example.com/fallback.onnx")

        captured = capsys.readouterr().out
        assert "downloading without progress" in captured
        assert Path(result).exists()

    def test_download_http_without_tqdm_uses_plain_urlretrieve(self, tmp_path):
        model = _make_model(tmp_path)
        cache_root = tmp_path / "cache_home"

        def fake_urlretrieve(url, target):
            Path(target).write_text("payload", encoding="utf-8")

        with patch.object(base_module.Path, "home", return_value=cache_root):
            with patch.object(base_module, "tqdm", None):
                with patch(
                    "manuscript.api.base.urllib.request.urlretrieve",
                    side_effect=fake_urlretrieve,
                ) as mock_urlretrieve:
                    result = model._download_http("https://example.com/plain.onnx")

        assert Path(result).exists()
        assert mock_urlretrieve.call_count == 1

    def test_download_http_retries_retryable_http_error_and_cleans_temp_file(
        self, tmp_path
    ):
        model = _make_model(tmp_path)
        cache_root = tmp_path / "cache_home"
        temp_file = tmp_path / "retry_http.tmp"
        attempts = {"count": 0}

        def fake_urlretrieve(url, target):
            attempts["count"] += 1
            Path(target).write_text("payload", encoding="utf-8")
            if attempts["count"] == 1:
                raise urllib.error.HTTPError(url, 429, "retry", None, None)

        with patch.object(base_module.Path, "home", return_value=cache_root):
            with patch.object(base_module, "tqdm", None):
                with patch(
                    "manuscript.api.base.tempfile.NamedTemporaryFile",
                    return_value=SimpleNamespace(name=str(temp_file)),
                ):
                    with patch(
                        "manuscript.api.base.urllib.request.urlretrieve",
                        side_effect=fake_urlretrieve,
                    ):
                        with patch("manuscript.api.base.time.sleep") as mock_sleep:
                            result = model._download_http("https://example.com/retry.onnx")

        assert attempts["count"] == 2
        assert mock_sleep.call_count == 1
        assert not temp_file.exists()
        assert Path(result).exists()

    def test_download_http_retries_url_error(self, tmp_path):
        model = _make_model(tmp_path)
        cache_root = tmp_path / "cache_home"
        attempts = {"count": 0}

        def fake_urlretrieve(url, target):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise urllib.error.URLError("temporary outage")
            Path(target).write_text("payload", encoding="utf-8")

        with patch.object(base_module.Path, "home", return_value=cache_root):
            with patch.object(base_module, "tqdm", None):
                with patch(
                    "manuscript.api.base.urllib.request.urlretrieve",
                    side_effect=fake_urlretrieve,
                ):
                    with patch("manuscript.api.base.time.sleep") as mock_sleep:
                        result = model._download_http(
                            "https://example.com/retry-url-error.onnx"
                        )

        assert attempts["count"] == 2
        assert mock_sleep.call_count == 1
        assert Path(result).exists()

    def test_download_gdrive_raises_when_gdown_returns_none(self, tmp_path):
        model = _make_model(tmp_path)

        with patch.dict("sys.modules", {"gdown": SimpleNamespace(download=lambda **_: None)}):
            with pytest.raises(RuntimeError, match="Failed to download from Google Drive"):
                model._download_gdrive("gdrive:missing-file")

    def test_download_gdrive_force_download_replaces_cached_file(self, tmp_path):
        cache_root = tmp_path / "cache_home"
        cached_file = cache_root / ".manuscript" / "weights" / "abc123.bin"
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        cached_file.write_text("stale", encoding="utf-8")

        model = _make_model(tmp_path)
        model.force_download = True

        def fake_download(*, id, output, quiet):
            assert id == "abc123"
            assert quiet is False
            Path(output).write_text("fresh", encoding="utf-8")
            return output

        with patch.object(base_module.Path, "home", return_value=cache_root):
            with patch.dict(
                "sys.modules",
                {"gdown": SimpleNamespace(download=fake_download)},
            ):
                result = model._download_gdrive("gdrive:abc123")

        assert result == str(cached_file)
        assert cached_file.read_text(encoding="utf-8") == "fresh"


def test_base_module_falls_back_when_tqdm_is_missing(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    base_path = repo_root / "src" / "manuscript" / "api" / "base.py"
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tqdm":
            raise ImportError("tqdm unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    spec = importlib.util.spec_from_file_location(
        "manuscript.api._base_without_tqdm", base_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.tqdm is None
