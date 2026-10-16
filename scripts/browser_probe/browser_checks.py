"""Browser regressions for ensembles, EAST geometry and external ONNX data."""
import json
from pathlib import Path

import numpy as np
from manuscript.data import Page, TextRegion
from manuscript.recognizers import MajorityVotingRecognizer, MedoidRecognizer
from manuscript.detectors._east.lanms import polygon_iou, locality_aware_nms
from manuscript.models.registry import Registry
from manuscript._internal.http import urlopen
from manuscript.runtimes.web import WebSession


def check_browser_features(browser_base_url):
    page = Page(image_width=100, image_height=20)
    page.children = [TextRegion(type='word', polygon=[(0,0),(100,0),(100,20),(0,20)])]
    class Answer:
        def __init__(self, text):
            self.text = text
        def predict(self, page, **kwargs):
            result = page.clone()
            result.children[0].text = self.text
            return result
    models = [Answer(value) for value in ['aaaaaa', 'aaabbb', 'bbbbbb']]
    assert MajorityVotingRecognizer(models).predict(page).get_text() == 'aaaaaa'
    assert MedoidRecognizer(models).predict(page).get_text() == 'aaabbb'
    print('✓ Голосование и медоид работают без RapidFuzz')
    polygon = np.array([[0,0],[10,0],[10,10],[0,10]], dtype=np.float64)
    assert polygon_iou(polygon, polygon) == 1.0
    boxes = np.array([list(polygon.flatten()) + [.8], list(polygon.flatten()) + [.9]])
    assert len(locality_aware_nms(boxes, .3)) == 1
    print('✓ Геометрия EAST и NMS работают без Numba')
    for ensemble in [MajorityVotingRecognizer, MedoidRecognizer]:
        try:
            ensemble(models, parallel=True)
        except ValueError as exc:
            assert 'parallel=False' in str(exc)
        else:
            raise AssertionError('Python threads must not be enabled in Pyodide')
    with urlopen(browser_base_url + 'probe-config.json', timeout=30) as response:
        assert 'wheel' in json.load(response)
    registry = Registry(root='/tmp/browser-registry-check', sources=[browser_base_url + 'selfcheck-registry.json'], timeout=30)
    registry.refresh()
    files = registry.resolve('external_test', 'Test')
    for external in [None, files['external_data'], [files['external_data']], {'external.onnx.data':files['external_data']}]:
        session = WebSession(files['weights'], external_data=external)
        try:
            np.testing.assert_array_equal(session.run(None, {'x':np.array([[4,5]],dtype=np.float32)})[0], [[8,15]])
        finally:
            session.close()
    # Cached files must work without another network request or a bridge change.
    assert registry.resolve('external_test', 'Test') == files
    print('✓ Реестр, проверка SHA-256, кеш и внешние ONNX-данные работают в браузере')

    tensor_path = Path('/tmp/tensor-check.onnx')
    with urlopen(browser_base_url + 'tensor-check.onnx', timeout=30) as response:
        tensor_path.write_bytes(response.read())
    values = [
        np.array(3.25, dtype=np.float32), np.empty((0, 2), dtype=np.float32),
        np.array([2**60 + 7, -2**60], dtype=np.int64), np.array([True, False]),
        np.array([1.5, -2, np.inf, -0.0], dtype=np.float16),
        np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2],
    ]
    session = WebSession(tensor_path)
    try:
        results = session.run(None, {f'x{i}': value for i, value in enumerate(values)})
        for actual, expected in zip(results, values):
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
    finally:
        session.close()
    print('✓ Скалярные, пустые, FP16, int64 и bool тензоры проходят без потери данных')
