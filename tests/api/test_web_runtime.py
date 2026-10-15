"""Boundary checks for optional browser inference, without faking model results."""
import subprocess
import sys
from importlib.metadata import distribution

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

from manuscript.runtimes.web import WebSession


def test_public_imports_do_not_require_native_onnxruntime():
    source = '''
import sys
from importlib.abc import MetaPathFinder
class BlockNativeOrt(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "onnxruntime" or fullname.startswith("onnxruntime."):
            raise ModuleNotFoundError("native runtime unavailable", name="onnxruntime")
sys.meta_path.insert(0, BlockNativeOrt())
from manuscript import Pipeline, Page, SimpleSorting
from manuscript.recognizers import TRBA, PPOCRRec, TrOCR
from manuscript.detectors import YOLO
from manuscript.correctors import CharLM
page = Page(image_width=100, image_height=50)
assert SimpleSorting().predict(page) is page
assert "onnxruntime" not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_web_runtime_reports_missing_pyodide(tmp_path):
    weights = tmp_path / 'model.onnx'
    weights.write_bytes(b'not read outside Pyodide')
    with pytest.raises(RuntimeError, match='requires Python running in Pyodide'):
        WebSession(weights)


def test_web_runtime_rejects_native_devices(tmp_path):
    with pytest.raises(ValueError, match='Web runtime device'):
        WebSession(tmp_path / 'model.onnx', device='cuda')


def test_web_dependency_markers_preserve_native_installation():
    package = distribution('manuscript-ocr')
    assert 'web' in package.metadata.get_all('Provides-Extra')
    requirements = list(map(Requirement, package.requires))
    base = {r.name for r in requirements if r.marker is None or r.marker.evaluate({'extra': ''})}
    assert not base & {'rapidfuzz', 'onnxruntime', 'onnxruntime-gpu', 'numba', 'scikit-image'}
    for extra, name in [('cpu', 'onnxruntime'), ('gpu', 'onnxruntime-gpu')]:
        for platform in ['emscripten', 'linux', 'win32', 'darwin']:
            environment = {**default_environment(), 'extra': extra, 'sys_platform': platform}
            selected = any(r.name == name and r.marker.evaluate(environment) for r in requirements)
            assert selected == (platform != 'emscripten')


def test_explicit_web_runtime_routes_session_without_loading_native(tmp_path, monkeypatch):
    from manuscript.api.base import BaseArtifactModel
    import manuscript.runtimes.web as web
    class Model(BaseArtifactModel):
        def _initialize_session(self):
            pass
        def predict(self):
            pass
    weights = tmp_path / 'model.onnx'
    weights.write_bytes(b'test')
    def no_native_device(*args):
        raise AssertionError('Web runtime attempted native device detection')
    monkeypatch.setattr(Model, '_resolve_device', no_native_device)
    calls = []
    from types import SimpleNamespace
    sentinel = SimpleNamespace(get_providers=lambda: ["WebGPUExecutionProvider"])
    def create(path, device, external_data=None, log_severity_level=None):
        calls.append((path, device))
        return sentinel
    monkeypatch.setattr(web, 'WebSession', create)
    model = Model(weights=weights, runtime='web', device='gpu')
    assert model._create_onnx_session() is sentinel
    assert calls == [(str(weights), 'gpu')]
    with pytest.raises(ValueError, match='Native ONNX session options'):
        model._create_onnx_session(sess_options=object())


def test_ensembles_and_east_import_without_optional_native_accelerators():
    source = '''
import sys
from importlib.abc import MetaPathFinder
class BlockNative(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in {"onnxruntime", "rapidfuzz", "numba"}:
            raise ModuleNotFoundError(root + " unavailable", name=root)
sys.meta_path.insert(0, BlockNative())
import numpy as np
from manuscript.recognizers import MajorityVotingRecognizer, MedoidRecognizer
from manuscript.recognizers._medoid import _distance
from manuscript.detectors import EAST
from manuscript.detectors._east.lanms import polygon_iou, locality_aware_nms
assert _distance("кот", "кит") == 1
polygon = np.array([[0,0],[10,0],[10,10],[0,10]], dtype=np.float64)
assert polygon_iou(polygon, polygon) == 1
boxes = np.array([list(polygon.flatten()) + [.8], list(polygon.flatten()) + [.9]])
assert len(locality_aware_nms(boxes, .3)) == 1
'''
    result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_browser_parallel_mode_reports_sequential_alternative(monkeypatch):
    import manuscript._stack as stack
    import manuscript.recognizers._majority_voting as voting
    from types import SimpleNamespace
    monkeypatch.setattr(stack, 'sys', SimpleNamespace(platform='emscripten'))
    monkeypatch.setattr(voting, 'sys', SimpleNamespace(platform='emscripten'))
    class Processor:
        def predict(self, image):
            pass
    for cls in [stack.Stack, voting.MajorityVotingRecognizer]:
        with pytest.raises(ValueError, match='parallel=False'):
            cls([Processor()], parallel=True)
        assert cls([Processor()], parallel=False).parallel is False
