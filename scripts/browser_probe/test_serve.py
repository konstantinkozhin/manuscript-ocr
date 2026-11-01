import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('browser_serve', Path(__file__).with_name('serve.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def server(tmp_path, monkeypatch):
    body = b'model weights'
    artifact = {'filename': 'weights.onnx', 'size': len(body),
                'sha256': hashlib.sha256(body).hexdigest(),
                'urls': ['https://example.org/weights.onnx']}
    catalog = {'models': {'example': {'artifacts': {'weights': artifact}}}}
    handler = module.create_handler(tmp_path, tmp_path / 'cache', catalog)
    calls = []

    def open_url(url, **kwargs):
        calls.append(url)
        return io.BytesIO(body)

    monkeypatch.setattr(module, 'urlopen', open_url)

    def request(path):
        instance = object.__new__(handler)
        instance.path, instance.headers = path, {'Host': '127.0.0.1:8014'}
        instance.wfile, instance.status, instance.response_headers = io.BytesIO(), None, {}
        instance.send_response = lambda status: setattr(instance, 'status', status)
        instance.send_error = lambda status, *args: setattr(instance, 'status', status)
        instance.send_header = instance.response_headers.__setitem__
        instance.end_headers = lambda: None
        instance.log_error = lambda *args: None
        instance.do_GET()
        return instance

    return request, artifact, calls, tmp_path / 'cache', body


def test_catalog_points_at_same_origin_without_mutating_source(server):
    request, artifact, calls, _, _ = server
    result = request('/catalog.json')
    catalog = json.loads(result.wfile.getvalue())
    url = catalog['models']['example']['artifacts']['weights']['urls'][0]
    assert url == 'http://127.0.0.1:8014/_artifacts/' + module.artifact_id(artifact)
    assert artifact['urls'] == ['https://example.org/weights.onnx']
    assert not calls


def test_streams_and_caches_verified_artifact(server):
    request, artifact, calls, cache, body = server
    path = '/_artifacts/' + module.artifact_id(artifact)
    assert request(path).wfile.getvalue() == body
    assert request(path).wfile.getvalue() == body
    assert calls == ['https://example.org/weights.onnx']
    assert len(list(cache.iterdir())) == 1


def test_unknown_artifact_cannot_turn_server_into_arbitrary_proxy(server):
    request, _, calls, _, _ = server
    assert request('/_artifacts/https://other.example/file').status == 404
    assert not calls


def test_corrupt_upstream_is_not_cached(server, monkeypatch):
    request, artifact, _, cache, _ = server
    monkeypatch.setattr(module, 'urlopen', lambda *a, **k: io.BytesIO(b'corrupt'))
    result = request('/_artifacts/' + module.artifact_id(artifact))
    assert result.close_connection
    assert not list(cache.iterdir())


def test_all_mirrors_failing_returns_gateway_error(server, monkeypatch):
    request, artifact, _, cache, _ = server

    def fail(*args, **kwargs):
        raise OSError('network unavailable')

    monkeypatch.setattr(module, 'urlopen', fail)
    assert request('/_artifacts/' + module.artifact_id(artifact)).status == 502
    assert not list(cache.iterdir())
