"""Batch assembly must preserve inputs, padding and release temporary arrays."""
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from manuscript.recognizers import TRBA, TrOCR
from manuscript.recognizers._common.image_batch import prepare_image_batch
from manuscript.recognizers._common.region_types import RecognitionPrediction


@pytest.mark.parametrize('kind', ['trba', 'trocr'])
@pytest.mark.parametrize('dtype', [np.float32, np.float16])
def test_batch_matches_previous_assembly_with_real_preprocessing(kind, dtype):
    model = (TRBA if kind == 'trba' else TrOCR).__new__(TRBA if kind == 'trba' else TrOCR)
    if kind == 'trba':
        model.img_h, model.img_w, model._input_dtype = 32, 64, dtype
    else:
        model.config = {'preprocess': {'height': 32, 'width': 64, 'resample': 3,
                                      'mean': [.5] * 3, 'std': [.5] * 3}}
        model.onnx_session = SimpleNamespace(get_inputs=lambda: [
            SimpleNamespace(type='tensor(float16)' if dtype == np.float16 else 'tensor(float)')])
    rng = np.random.default_rng(42)
    regions = [SimpleNamespace(image=rng.integers(0, 256, (h, w, 3), np.uint8))
               for h, w in [(12, 50), (80, 31), (32, 64)]]
    prepared = [model._preprocess_image(r.image) for r in regions]
    expected = np.concatenate(prepared)
    expected = np.concatenate([expected, np.repeat(expected[-1:], 2, axis=0)])
    actual = prepare_image_batch(regions, model._preprocess_image, 5)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == dtype and actual.flags.c_contiguous


@pytest.mark.parametrize('kind,fixed,device,fallback', [
    ('trba', None, 'cpu', False), ('trba', None, 'cuda', False),
    ('trba', 4, 'cpu', False), ('trba', None, 'cpu', True),
    ('trocr', None, 'cpu', False), ('trocr', 4, 'cpu', False),
])
def test_prediction_batches_release_arrays_and_preserve_results(kind, fixed, device, fallback):
    cls = TRBA if kind == 'trba' else TrOCR
    model = cls.__new__(cls)
    model.batch_size, model.runtime, model.device = 4, 'native', device
    model._supports_multi_batch_inference = True
    model._initialize_session = lambda: None
    model._warn_single_batch_only = lambda: None
    model._is_batch_shape_error = lambda exc: str(exc) == 'batch not supported'
    regions = [SimpleNamespace(image=np.full((1, 3, 2, 4), i, np.float32)) for i in range(7)]
    model._preprocess_image = lambda image: image.copy()
    previous = []
    seen = []

    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name='pixels', shape=[fixed or 'N', 3, 2, 4])]

        def get_outputs(self):
            return [SimpleNamespace(name='logits', shape=[fixed or 'N', 1, 1])]

        def run(self, names, feed):
            pixels = feed['pixels']
            if len(pixels) > 1 or not fallback:
                assert all(ref() is None for ref in previous)
            if fallback and len(pixels) > 1:
                raise ValueError('batch not supported')
            seen.append(pixels[:, 0, 0, 0].tolist())
            output = pixels[:, :1, :1, :1].reshape(len(pixels), 1, 1).copy()
            if not fallback:
                previous[:] = [weakref.ref(pixels), weakref.ref(output)]
            return [output]

    model.onnx_session = Session()
    if kind == 'trba':
        model._decode_predictions = lambda logits, raw: [
            RecognitionPrediction(str(int(row[0, 0])), meta={'output': row.copy()} if raw else {})
            for row in logits]
    else:
        model.decoder_session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(shape=['N', 'T'])])
        model.decoder_with_past_session = None
        model._generate_batch = lambda hidden: [([int(row[0, 0])], 1.) for row in hidden]
        model.tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: str(tokens[0]))
    results = model._predict_text_images(regions, return_raw=True)
    assert [p.text for p in results] == [str(i) for i in range(7)]
    assert all(ref() is None for ref in previous)
    if not fallback:
        last = [4., 5., 6., 6.] if fixed or device == 'cuda' else [4., 5., 6.]
        assert seen == [[0., 1., 2., 3.], last]
    else:
        assert seen == [[float(i)] for i in range(7)]
