from tests.support.modules import patch_modules
import builtins
import io
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

    def close(self):
        pass

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
        self, tmp_path, caplog
    ):
        model = _make_model(tmp_path, device="cuda")
        session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

        model._log_device_info(session)

        captured = caplog.text
        assert "requested CUDAExecutionProvider" in captured
        assert "CPUExecutionProvider" in captured

    def test_log_device_info_warns_when_coreml_provider_unavailable(
        self, tmp_path, caplog
    ):
        model = _make_model(tmp_path, device="coreml")
        session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

        model._log_device_info(session)

        captured = caplog.text
        assert "requested CoreMLExecutionProvider" in captured
        assert "CPUExecutionProvider" in captured

    def test_prepare_runtime_dependencies_adds_windows_nvidia_dirs_and_preloads(
        self, tmp_path
    ):
        model = _make_model(tmp_path, device="cuda")

        cudnn_dir = tmp_path / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin"
        cublas_dir = tmp_path / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin"
        cudnn_dir.mkdir(parents=True)
        cublas_dir.mkdir(parents=True)

        added_dirs = []
        handles = [object(), object()]

        def fake_add_dll_directory(path):
            added_dirs.append(path)
            return handles[len(added_dirs) - 1]

        with patch.object(base_module.sys, "prefix", str(tmp_path)):
            # Keep filesystem paths native while exercising the Windows branch.
            with (
                patch.object(base_module.os, "name", "nt"),
                patch.object(base_module, "Path", type(tmp_path)),
            ):
                with patch.object(
                    base_module.os,
                    "add_dll_directory",
                    side_effect=fake_add_dll_directory,
                    create=True,
                ):
                    with patch.object(
                        base_module.get_native_runtime(), "preload_dlls", create=True
                    ) as mock_preload:
                        model._prepare_runtime_dependencies()

        assert added_dirs == [str(cudnn_dir), str(cublas_dir)]
        assert model._runtime_dll_dir_handles == handles
        assert model._runtime_deps_preloaded is True
        assert "Added Windows DLL directories:" in model._runtime_preload_message
        mock_preload.assert_called_once_with(directory="")

    def test_download_http_uses_progress_bar_when_tqdm_available(self, tmp_path):
        model = _make_model(tmp_path)
        response = io.BytesIO(b'payload'); response.headers = {'content-length':'7'}
        with patch.object(base_module.Path, 'home', return_value=tmp_path):
            with patch.object(base_module, 'tqdm', _FakeTqdm), patch.object(base_module.http, 'urlopen', return_value=response):
                result = model._download_http('https://example.com/model.onnx')
        assert Path(result).read_bytes() == b'payload'
        assert response.closed

    def test_download_http_falls_back_when_progress_bar_setup_fails(self, tmp_path, capsys):
        model = _make_model(tmp_path)
        response = io.BytesIO(b'payload'); response.headers = {}
        with patch.object(base_module.Path, 'home', return_value=tmp_path):
            with patch.object(base_module, 'tqdm', side_effect=RuntimeError('broken progress')), patch.object(base_module.http, 'urlopen', return_value=response) as open_url:
                result = model._download_http('https://example.com/fallback.onnx')
        assert Path(result).read_bytes() == b'payload'
        assert open_url.call_count == 1

    def test_download_http_without_tqdm_uses_plain_stream(self, tmp_path):
        model = _make_model(tmp_path)
        response = io.BytesIO(b'payload'); response.headers = {}
        with patch.object(base_module.Path, 'home', return_value=tmp_path):
            with patch.object(base_module, 'tqdm', None), patch.object(base_module.http, 'urlopen', return_value=response) as open_url:
                result = model._download_http('https://example.com/plain.onnx')
        assert Path(result).read_bytes() == b'payload'
        assert open_url.call_count == 1

    def test_download_http_retries_retryable_http_error_and_cleans_temp_file(self, tmp_path):
        model = _make_model(tmp_path)
        url = 'https://example.com/retry.onnx'
        response = io.BytesIO(b'payload'); response.headers = {}
        with patch.object(base_module.Path, 'home', return_value=tmp_path):
            with patch.object(base_module.http, 'urlopen', side_effect=[urllib.error.HTTPError(url, 429, "retry", None, None), response]) as open_url, patch.object(base_module.http, 'sleep') as sleep:
                result = model._download_http(url)
        assert open_url.call_count == 2
        sleep.assert_called_once_with(1)
        assert Path(result).read_bytes() == b'payload'
        assert not list(Path(result).parent.glob('*.part'))

    def test_download_http_retries_url_error(self, tmp_path):
        model = _make_model(tmp_path)
        url = 'https://example.com/retry.onnx'
        response = io.BytesIO(b'payload'); response.headers = {}
        with patch.object(base_module.Path, 'home', return_value=tmp_path):
            with patch.object(base_module.http, 'urlopen', side_effect=[urllib.error.URLError("temporary outage"), response]) as open_url, patch.object(base_module.http, 'sleep') as sleep:
                result = model._download_http(url)
        assert open_url.call_count == 2
        sleep.assert_called_once_with(1)
        assert Path(result).read_bytes() == b'payload'
        assert not list(Path(result).parent.glob('*.part'))

    def test_download_gdrive_raises_when_gdown_returns_none(self, tmp_path):
        model = _make_model(tmp_path)

        with patch_modules({"gdown": SimpleNamespace(download=lambda **_: None)}
        ):
            with pytest.raises(
                RuntimeError, match="Failed to download from Google Drive"
            ):
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
            with patch_modules({"gdown": SimpleNamespace(download=fake_download)},
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


@pytest.mark.parametrize("status,attempts", [(404, 1), (503, 5)])
def test_failed_refresh_preserves_cache_and_removes_partial_files(tmp_path, monkeypatch, status, attempts):
    model = _make_model(tmp_path)
    model.force_download = True
    monkeypatch.setattr(type(model), '_cache_dir', property(lambda self: tmp_path / 'cache'))
    url = 'https://example.com/model.onnx'
    cache = model._http_cache_path(url); cache.write_bytes(b'working weights')
    with patch.object(base_module.http, 'urlopen', side_effect=urllib.error.HTTPError(url, status, 'unavailable', None, None)) as open_url, patch.object(base_module.http, 'sleep') as sleep:
        with pytest.raises(urllib.error.HTTPError) as error:
            model._download_http(url)
    assert error.value.code == status
    assert open_url.call_count == attempts
    assert sleep.call_count == attempts - 1
    assert cache.read_bytes() == b'working weights'
    assert not list(cache.parent.glob('*.part'))
