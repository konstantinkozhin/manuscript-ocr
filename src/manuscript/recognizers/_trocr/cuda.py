"""Optional device-resident decoding. Torch owns buffers, ONNX executes models."""
import numpy as np


class TorchBackend:
    def __init__(self):
        import torch
        self.torch = torch

    def ids(self, rows, hidden):
        return self.torch.tensor(rows, dtype=self.torch.int64, device=hidden.device)

    def gather(self, values, indices):
        indices = self.torch.tensor(indices, dtype=self.torch.int64, device=values.device)
        return values.index_select(0, indices)

    def scores(self, logits, count):
        torch = self.torch
        if (logits.ndim != 3 or logits.shape[0] != count or not logits.shape[1]
                or not logits.shape[2] or not bool(torch.isfinite(logits).all())):
            raise ValueError('Decoder emitted invalid or nonfinite logits')
        return torch.log_softmax(logits[:, -1].to(torch.float64), dim=-1)

    def block(self, scores, index, blocked, prior):
        if blocked:
            scores[index, blocked] = -self.torch.inf
        scores[index] += prior

    def top(self, scores, k):
        torch = self.torch
        flat = scores.flatten()
        k = min(k, flat.numel())
        threshold = torch.topk(flat, k, sorted=False).values.min()
        greater = torch.nonzero(flat > threshold, as_tuple=False).flatten()
        ties = torch.nonzero(flat == threshold, as_tuple=False).flatten()[:k - greater.numel()]
        indices = torch.cat((greater, ties)).sort().values
        order = torch.argsort(flat[indices], descending=True, stable=True)
        indices = indices[order]
        # Only the small candidate list crosses to the host, not full logits/cache.
        return indices.cpu().tolist(), flat[indices].cpu().tolist()


def from_ort_value(value, torch):
    """Support the public DLPack protocol and older ORT's C OrtValue bridge."""
    if hasattr(value, '__dlpack__'):
        return torch.utils.dlpack.from_dlpack(value)
    native = getattr(value, '_ortvalue', value)
    if hasattr(native, 'to_dlpack'):
        return torch.utils.dlpack.from_dlpack(native.to_dlpack())
    raise RuntimeError('Device decoding requires ONNX Runtime DLPack export support')


def run_bound(session, feed, names, torch):
    """Bind device buffers; DLPack tensors retain ORT output allocations."""
    binding = session.io_binding()
    keep_alive = {}
    device = next(iter(feed.values())).device
    dtypes = {torch.float32: np.float32, torch.float16: np.float16, torch.int64: np.int64}
    for name, tensor in feed.items():
        if tensor.device != device:
            raise ValueError('All ONNX inference inputs must reside on the same device')
        value = tensor.contiguous()
        keep_alive[name] = value
        binding.bind_input(name, device.type, device.index or 0,
                           dtypes[value.dtype], tuple(value.shape), value.data_ptr())
    for name in names:
        binding.bind_output(name, device.type, device.index or 0)
    # Correct across ORT's and Torch's independently managed CUDA streams.
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    values = binding.get_outputs()
    return {name: from_ort_value(value, torch) for name, value in zip(names, values)}


def generate_cuda(model, pixels):
    from .generation import generate_batch

    backend = TorchBackend()
    torch = backend.torch
    options = model.onnx_session.get_provider_options().get('CUDAExecutionProvider', {})
    device = torch.device('cuda', int(options.get('device_id', 0)))
    with torch.inference_mode():
        images = torch.from_numpy(pixels).to(device=device)
        output_name = model.onnx_session.get_outputs()[0].name
        hidden = run_bound(model.onnx_session, {model.onnx_session.get_inputs()[0].name: images},
                           [output_name], torch)[output_name]
        del images

        def decode(ids, states, cache):
            session = model.decoder_session if cache is None else model.decoder_with_past_session
            feed = {'input_ids': ids if cache is None else ids[:, -1:].contiguous()}
            inputs = {item.name: item for item in session.get_inputs()}
            if 'encoder_hidden_states' in inputs:
                dtype = torch.float16 if inputs['encoder_hidden_states'].type == 'tensor(float16)' else torch.float32
                feed['encoder_hidden_states'] = states.to(dtype=dtype)
            if cache is not None:
                feed.update(cache)
            names = ['logits', *[entry['output'] for entry in model.cache_mapping]]
            outputs = run_bound(session, feed, names, torch)
            return outputs['logits'], {entry['input']: outputs[entry['output']] for entry in model.cache_mapping}

        predictions = generate_batch(hidden, model.generation, decode, model._banned, backend=backend)
        return hidden, predictions


def unavailable_reason(model):
    """Check support before selecting the CUDA path; inference errors are not hidden."""
    if model.runtime != 'native' or model.device != 'cuda':
        return 'native NVIDIA CUDA execution is required'
    try:
        import torch
    except ImportError:
        return 'PyTorch is not installed (install manuscript-ocr[trocr-cuda,gpu])'
    if not torch.cuda.is_available():
        return 'PyTorch CUDA is unavailable'
    if model.decoder_format != 'kv_cache':
        return 'a KV-cache model export is required'
    sessions = (model.onnx_session, model.decoder_session, model.decoder_with_past_session)
    for session in sessions:
        if not session.get_providers() or session.get_providers()[0] != 'CUDAExecutionProvider':
            return 'CUDAExecutionProvider must be active in all three ONNX sessions'
        if any(isinstance(item.shape[0], int) for item in session.get_inputs()):
            return 'dynamic-batch encoder and decoder graphs are required'
        if not callable(getattr(session, 'io_binding', None)):
            return 'ONNX Runtime I/O binding is unavailable'
    from manuscript.runtimes._native import get_native_runtime
    ort = get_native_runtime()
    probe = ort.OrtValue.ortvalue_from_numpy(np.zeros(1, np.float32))
    if not (hasattr(probe, '__dlpack__') or hasattr(probe._ortvalue, 'to_dlpack')):
        return 'ONNX Runtime DLPack export support is unavailable'
    return None
