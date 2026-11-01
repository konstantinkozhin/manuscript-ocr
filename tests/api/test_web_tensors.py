"""The bridge must preserve tensor dimensions, integer precision and FP16 bits."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from manuscript.runtimes import web


@pytest.mark.parametrize('value', [
    np.array(3.25, dtype=np.float32),
    np.array([], dtype=np.float32).reshape(0, 2),
    np.array([2**60 + 7, -2**60], dtype=np.int64),
    np.array([True, False], dtype=np.bool_),
    np.array([1.5, -2, np.inf, -0.0], dtype=np.float16),
    np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2],
])
def test_tensor_round_trip_preserves_data_and_shape(value, tmp_path, monkeypatch):
    def wrapped(value):
        return SimpleNamespace(to_py=lambda: value)
    inputs = wrapped([{'name': 'x', 'type': str(value.dtype), 'shape': list(value.shape)}])
    outputs = wrapped([{'name': 'y', 'type': str(value.dtype), 'shape': list(value.shape)}])
    handle = SimpleNamespace(inputs=inputs, outputs=outputs)
    def run(handle, descriptors, names):
        assert names == ['y']
        item = descriptors[0]
        assert item['shape'] == list(value.shape)
        expected = value.reshape(-1).view(np.uint16) if value.dtype == np.float16 else value.reshape(-1)
        np.testing.assert_array_equal(item['data'], expected)
        return [SimpleNamespace(type=item['type'], shape=wrapped(item['shape']),
                                data=wrapped(item['data'].copy()))]
    bridge = SimpleNamespace(create_session=lambda *args: handle, run_session=run,
                             release_session=lambda handle: None)
    monkeypatch.setattr(web, '_context', lambda: (bridge, lambda value: value, lambda value, **kwargs: value))
    monkeypatch.setitem(sys.modules, 'js', SimpleNamespace(Object=SimpleNamespace(fromEntries=object())))
    weights = tmp_path / 'model.onnx'
    weights.write_bytes(b'bridge conversion test')
    session = web.WebSession(weights)
    result, = session.run(None, {'x': value})
    assert result.shape == value.shape
    assert result.dtype == value.dtype
    np.testing.assert_array_equal(result, value)
    assert not np.shares_memory(result, value)
    session.close()
