"""Device bridge ownership, beam parity, and explicit capability selection."""
import gc
from types import SimpleNamespace

import numpy as np
import pytest

from manuscript.recognizers import TrOCR
from manuscript.recognizers._trocr.generation import generate_batch, NumpyBackend


@pytest.mark.parametrize('size,k', [(1, 1), (100, 3), (100, 200), (50000, 8)])
def test_partial_top_k_preserves_full_stable_sort(size, k):
    rng = np.random.default_rng(42)
    flat = rng.integers(-3, 4, size).astype(np.float64)
    flat[::5] = -np.inf
    indices, values = NumpyBackend().top(flat, k)
    expected = np.argsort(-flat, kind='stable')[:k]
    assert indices == expected.tolist()
    assert values == flat[expected].tolist()


@pytest.mark.parametrize('beams', [1, 3])
@pytest.mark.parametrize('early', [True, False, 'never'])
def test_tensor_backend_matches_numpy_beams_and_cache_gathers(beams, early):
    torch = pytest.importorskip('torch')
    from manuscript.recognizers._trocr.cuda import TorchBackend
    hidden = np.arange(3, dtype=np.float32).reshape(3, 1, 1)
    generation = dict(num_beams=beams, max_length=9, decoder_start_token_id=1,
                      eos_token_id=[2, 6], length_penalty=1.5,
                      early_stopping=early, no_repeat_ngram_size=2)

    def decode(ids, states, cache):
        owners = states[:, 0, 0].astype(int)
        if cache is not None:
            np.testing.assert_array_equal(cache['past'][:, 0, 0, 0], owners)
        logits = np.full((len(ids), 2, 7), -8., np.float32)
        for index, owner in enumerate(owners):
            # Exact ties and different finish times exercise stable top-k and pruning.
            logits[index, -1, 3:6] = 2.
            logits[index, -1, 2] = 7. if ids.shape[1] > owner + 2 else -4.
        return logits, {'past': states[:, None].copy()}

    expected = generate_batch(hidden, generation, decode, TrOCR._banned)
    def tensor_decode(ids, states, cache):
        values, present = decode(ids.numpy(), states.numpy(),
                                 None if cache is None else {k: v.numpy() for k, v in cache.items()})
        return torch.from_numpy(values), {k: torch.from_numpy(v) for k, v in present.items()}
    actual = generate_batch(torch.from_numpy(hidden), generation, tensor_decode,
                            TrOCR._banned, backend=TorchBackend())
    assert [row[0] for row in actual] == [row[0] for row in expected]
    np.testing.assert_allclose([row[1] for row in actual], [row[1] for row in expected], rtol=1e-12)


def test_torch_top_k_preserves_boundary_ties():
    torch = pytest.importorskip('torch')
    from manuscript.recognizers._trocr.cuda import TorchBackend
    scores = torch.tensor([1., 2., 2., -torch.inf, 2., 1.], dtype=torch.float64)
    actual = TorchBackend().top(scores, 2)
    assert actual == ([1, 2], [2., 2.])


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.int64])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_ort_binding_dlpack_outputs_outlive_binding(tmp_path, dtype, device):
    torch = pytest.importorskip('torch')
    ort = pytest.importorskip('onnxruntime')
    onnx = pytest.importorskip('onnx')
    from manuscript.recognizers._trocr.cuda import run_bound
    if device == 'cuda' and (not torch.cuda.is_available() or 'CUDAExecutionProvider' not in ort.get_available_providers()):
        pytest.skip('CUDA Torch and ONNX Runtime GPU are required')
    tensor_type = {np.float16: onnx.TensorProto.FLOAT16, np.float32: onnx.TensorProto.FLOAT,
                   np.int64: onnx.TensorProto.INT64}[dtype]
    graph = onnx.helper.make_graph([onnx.helper.make_node('Identity', ['x'], ['y'])], 'bridge',
        [onnx.helper.make_tensor_value_info('x', tensor_type, ['N', 3])],
        [onnx.helper.make_tensor_value_info('y', tensor_type, ['N', 3])])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('', 17)])
    model.ir_version = 9
    path = tmp_path / 'bridge.onnx'
    onnx.save(model, path)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    session = ort.InferenceSession(str(path), providers=providers)
    # Noncontiguous input also tests that bound temporary buffers remain alive.
    values = torch.from_numpy(np.arange(18, dtype=dtype).reshape(3, 6)).to(device)[:, ::2]
    expected = values.cpu().numpy().copy()
    result = run_bound(session, {'x': values}, ['y'], torch)['y']
    del values, session
    gc.collect()
    np.testing.assert_array_equal(result.cpu().numpy(), expected)
    assert result.device.type == device


@pytest.mark.parametrize('device,runtime', [('cpu', 'native'), ('coreml', 'native'), ('gpu', 'web')])
def test_auto_on_other_devices_does_not_load_cuda_backend(monkeypatch, device, runtime):
    model = TrOCR.__new__(TrOCR)
    model.cuda_decoding, model.device, model.runtime = 'auto', device, runtime
    monkeypatch.setattr('manuscript.recognizers._trocr.cuda.unavailable_reason',
                        lambda _: pytest.fail('CUDA support must not be checked'))
    model._select_cuda_decoding()
    assert model._cuda_decoding_enabled is False


def test_cuda_selection_warns_once_and_required_rejects(monkeypatch):
    model = TrOCR.__new__(TrOCR)
    model.cuda_decoding, model.device, model.runtime = 'auto', 'cuda', 'native'
    model._cuda_warning_reason = None
    monkeypatch.setattr('manuscript.recognizers._trocr.cuda.unavailable_reason', lambda _: 'missing DLPack')
    with pytest.warns(RuntimeWarning, match='missing DLPack'):
        model._select_cuda_decoding()
    import warnings
    with warnings.catch_warnings(record=True) as seen:
        model._select_cuda_decoding()
    assert not seen and model._cuda_decoding_enabled is False
    model.cuda_decoding = 'required'
    with pytest.raises(RuntimeError, match='missing DLPack'):
        model._select_cuda_decoding()
    monkeypatch.setattr('manuscript.recognizers._trocr.cuda.unavailable_reason', lambda _: None)
    model._select_cuda_decoding()
    assert model._cuda_decoding_enabled is True
    model.cuda_decoding = 'off'
    model._select_cuda_decoding()
    assert model._cuda_decoding_enabled is False


def test_real_cuda_cached_decoder_matches_host_path(tmp_path):
    torch = pytest.importorskip('torch')
    ort = pytest.importorskip('onnxruntime')
    onnx = pytest.importorskip('onnx')
    if not torch.cuda.is_available() or 'CUDAExecutionProvider' not in ort.get_available_providers():
        pytest.skip('CUDA Torch and ONNX Runtime GPU are required')
    from manuscript.recognizers._trocr.cuda import unavailable_reason
    info = onnx.helper.make_tensor_value_info
    def save(name, nodes, inputs, outputs, tensors):
        graph = onnx.helper.make_graph(nodes, name, inputs, outputs,
                                      [onnx.numpy_helper.from_array(v, k) for k, v in tensors.items()])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('', 17)])
        model.ir_version = 9
        path = tmp_path / (name + '.onnx')
        onnx.save(model, path)
        return ort.InferenceSession(str(path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    encoder = save('encoder', [onnx.helper.make_node('Reshape', ['pixels', 'shape'], ['hidden'])],
                   [info('pixels', onnx.TensorProto.FLOAT, ['N', 3, 8, 16])],
                   [info('hidden', onnx.TensorProto.FLOAT, ['N', 1, 384])],
                   {'shape': np.array([0, 1, -1], np.int64)})
    weights = np.full((5, 5), -20, np.float32)
    weights[:, 2] = 10
    weights[1, 3] = 20
    sessions = []
    for cached in (False, True):
        inputs = [info('input_ids', onnx.TensorProto.INT64, ['N', 'T']),
                  info('encoder_hidden_states', onnx.TensorProto.FLOAT, ['N', 1, 384])]
        nodes = [onnx.helper.make_node('Gather', ['weights', 'input_ids'], ['logits'], axis=0)]
        tensors = {'weights': weights}
        if cached:
            inputs.append(info('past', onnx.TensorProto.FLOAT, ['N', 1, 1, 384]))
            nodes.append(onnx.helper.make_node('Identity', ['past'], ['present']))
        else:
            tensors['axis'] = np.array([1], np.int64)
            nodes.append(onnx.helper.make_node('Unsqueeze', ['encoder_hidden_states', 'axis'], ['present']))
        sessions.append(save('cached' if cached else 'decoder', nodes, inputs,
                             [info('logits', onnx.TensorProto.FLOAT, ['N', 'T', 5]),
                              info('present', onnx.TensorProto.FLOAT, ['N', 1, 1, 384])], tensors))
    model = TrOCR.__new__(TrOCR)
    model.onnx_session = encoder
    model.decoder_session, model.decoder_with_past_session = sessions
    model.runtime, model.device, model.decoder_format = 'native', 'cuda', 'kv_cache'
    reason = unavailable_reason(model)
    if reason:
        pytest.skip('CUDA bridge is unsupported by this ONNX Runtime: ' + reason)
    model.cache_mapping = [{'input': 'past', 'output': 'present'}]
    model.generation = dict(num_beams=2, max_length=6, decoder_start_token_id=1,
                            eos_token_id=2, length_penalty=1., early_stopping=False,
                            no_repeat_ngram_size=0)
    model.config = {'preprocess': {'height': 8, 'width': 16, 'mean': [.5]*3, 'std': [.5]*3}}
    model.tokenizer = SimpleNamespace(decode=lambda tokens, **kw: str(tokens))
    regions = [SimpleNamespace(image=np.full((8, 16, 3), value, np.uint8)) for value in (50, 200)]
    model._cuda_decoding_enabled = False
    expected = model._infer_image_batch(regions, 2, 'pixels', True)
    model._cuda_decoding_enabled = True
    actual = model._infer_image_batch(regions, 2, 'pixels', True)
    assert [p.text for p in actual] == [p.text for p in expected]
    np.testing.assert_allclose([p.confidence for p in actual], [p.confidence for p in expected], rtol=1e-10)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a.meta['encoder_output'], b.meta['encoder_output'])


@pytest.mark.parametrize('case,match', [
    ('ok', None), ('format', 'KV-cache'), ('provider', 'all three'),
    ('static', 'dynamic-batch'), ('binding', 'I/O binding'), ('dlpack', 'DLPack'),
])
def test_cuda_capability_checks_before_execution(monkeypatch, case, match):
    torch = pytest.importorskip('torch')
    pytest.importorskip('onnxruntime')
    from manuscript.recognizers._trocr.cuda import unavailable_reason
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    def session():
        return SimpleNamespace(get_providers=lambda: ['CUDAExecutionProvider'],
                               get_inputs=lambda: [SimpleNamespace(shape=['N', 2])],
                               io_binding=lambda: None)
    model = SimpleNamespace(runtime='native', device='cuda', decoder_format='kv_cache',
                            onnx_session=session(), decoder_session=session(),
                            decoder_with_past_session=session())
    if case == 'format': model.decoder_format = 'full_prefix'
    elif case == 'provider': model.decoder_session.get_providers = lambda: ['CPUExecutionProvider']
    elif case == 'static': model.onnx_session.get_inputs = lambda: [SimpleNamespace(shape=[1, 2])]
    elif case == 'binding': model.decoder_with_past_session.io_binding = None
    elif case == 'dlpack':
        fake = SimpleNamespace(OrtValue=SimpleNamespace(ortvalue_from_numpy=lambda value: SimpleNamespace(_ortvalue=object())))
        monkeypatch.setattr('manuscript.runtimes._native.get_native_runtime', lambda: fake)
    reason = unavailable_reason(model)
    assert reason is None if match is None else match in reason


def test_active_cuda_inference_error_is_not_silently_retried_on_cpu(monkeypatch):
    model = TrOCR.__new__(TrOCR)
    model._cuda_decoding_enabled = True
    model._preprocess_image = lambda image: np.zeros((1, 3, 2, 2), np.float32)
    def fail(*args):
        raise RuntimeError('CUDA inference failed')
    monkeypatch.setattr('manuscript.recognizers._trocr.cuda.generate_cuda', fail)
    with pytest.raises(RuntimeError, match='CUDA inference failed'):
        model._infer_image_batch([SimpleNamespace(image=None)], 1, 'pixels', False)
