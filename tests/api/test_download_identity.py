import io
from pathlib import Path
from unittest.mock import patch

import pytest


def test_http_cache_does_not_alias_distinct_urls(tmp_path):
    import manuscript.api.base as base

    class Model(base.BaseArtifactModel):
        def _initialize_session(self):
            pass

        def predict(self):
            pass

        @property
        def _cache_dir(self):
            return tmp_path

    model = object.__new__(Model)
    model.force_download = False
    calls = []

    def download(url, **kwargs):
        calls.append(url)
        response = io.BytesIO(url.encode()); response.headers = {}
        return response

    with (
        patch.object(base, "tqdm", None),
        patch.object(base.http, "urlopen", download),
    ):
        first = model._download_http("https://host-a/model.onnx")
        second = model._download_http("https://host-b/model.onnx")
    assert first != second and len(calls) == 2, (first, second, calls)


@pytest.mark.parametrize(
    "second",
    [
        "https://host/model.onnx?version=2",
        "https://host/other/model.onnx",
        "https://other/model.onnx?version=1",
    ],
)
def test_url_cache_preserves_extension_and_identity(tmp_path, monkeypatch, second):
    from tests.api.test_base_runtime_edges import _make_model
    import manuscript.api.base as module

    model = _make_model(tmp_path)
    calls = []

    def download(url, **kwargs):
        calls.append(url)
        response = io.BytesIO(url.encode()); response.headers = {}
        return response

    monkeypatch.setattr(module, "tqdm", None)
    monkeypatch.setattr(module.http, "urlopen", download)
    first = model._download_http("https://host/model.onnx?version=1")
    other = model._download_http(second)
    assert Path(first).suffix == Path(other).suffix == ".onnx"
    assert first != other
    assert model._download_http(second) == other
    assert calls == ["https://host/model.onnx?version=1", second]
    model.force_download = True
    assert model._download_http(second) == other
    assert calls[-1] == second and len(calls) == 3
