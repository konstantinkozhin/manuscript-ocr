"""Session ownership and option translation across runtime backends."""
from types import SimpleNamespace

import pytest

from manuscript.api.base import BaseArtifactModel, managed_session_initialization
from manuscript.runtimes import create_session


class Session:
    def __init__(self):
        self.closed = 0

    def get_providers(self):
        return ['CPUExecutionProvider']

    def close(self):
        self.closed += 1


class Model(BaseArtifactModel):
    @managed_session_initialization
    def _initialize_session(self, fail=False):
        if self.session is not None:
            return
        self.session = self._create_onnx_session()
        if fail:
            self.decoder_session = self._create_onnx_session()
            raise ValueError('invalid graph metadata')

    def predict(self):
        self._initialize_session()
        return self.session


@pytest.fixture
def model(tmp_path, monkeypatch):
    weights = tmp_path / 'model.onnx'
    weights.write_bytes(b'test')
    sessions = []
    def create(*args, **kwargs):
        session = Session()
        sessions.append(session)
        return session
    monkeypatch.setattr('manuscript.api.base.create_session', create)
    return Model(weights=weights, device='cpu'), sessions


def test_close_is_idempotent_and_prediction_reinitializes(model):
    instance, sessions = model
    first = instance.predict()
    instance.close()
    instance.close()
    assert first.closed == 1
    assert instance.session is None
    second = instance.predict()
    assert second is not first
    assert second.closed == 0
    assert len(sessions) == 2


def test_partial_initialization_releases_all_created_sessions(model):
    instance, sessions = model
    with pytest.raises(ValueError, match='invalid graph metadata'):
        instance._initialize_session(fail=True)
    assert [s.closed for s in sessions] == [1, 1]
    assert instance.session is instance.decoder_session is None
    assert instance._runtime_sessions == []
    assert instance.predict() is sessions[-1]


def test_context_manager_releases_on_exception(model):
    instance, sessions = model
    with pytest.raises(RuntimeError):
        with instance:
            instance.predict()
            raise RuntimeError('processing failed')
    assert sessions[0].closed == 1
    assert instance.session is None


def test_cleanup_continues_when_one_session_release_fails(model):
    instance, sessions = model
    instance.predict()
    second = instance._create_onnx_session()
    def fail():
        raise RuntimeError('release failed')
    second.close = fail
    instance.close()
    assert sessions[0].closed == 1
    assert instance.session is None
    assert instance._runtime_sessions == [second]
    second.close = lambda: None
    instance.close()
    assert instance._runtime_sessions == []


def test_common_log_options_translate_to_native_session_options(monkeypatch):
    calls = []
    ort = SimpleNamespace(SessionOptions=SimpleNamespace,
                          InferenceSession=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr('manuscript.runtimes._native.get_native_runtime', lambda: ort)
    create_session('model.onnx', runtime='native', device='cpu',
                   providers=['CPUExecutionProvider'], log_severity_level=3)
    assert calls[0][1]['sess_options'].log_severity_level == 3
    assert calls[0][1]['providers'] == ['CPUExecutionProvider']


@pytest.mark.parametrize('level', [True, -1, 5, '3'])
def test_common_log_options_reject_invalid_values(level):
    with pytest.raises(ValueError, match='log_severity_level'):
        create_session('model.onnx', runtime='native', device='cpu', log_severity_level=level)
