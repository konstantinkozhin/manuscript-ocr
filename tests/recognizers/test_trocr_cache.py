"""Exercise real cached ONNX graphs against the source generator."""
import numpy as np
import pytest

from manuscript.recognizers import TrOCR


@pytest.fixture(scope='module', params=['trocr', 'roberta'])
def bundle(request, tmp_path_factory):
    torch = pytest.importorskip('torch')
    hf = pytest.importorskip('transformers')
    tokenizers = pytest.importorskip('tokenizers')
    pytest.importorskip('onnx')
    torch.set_num_threads(2)
    torch.manual_seed(7)
    source = tmp_path_factory.mktemp('trocr_export') / 'source'
    encoder = hf.ViTModel(hf.ViTConfig(image_size=16, patch_size=8, hidden_size=16,
                                     num_hidden_layers=1, num_attention_heads=2, intermediate_size=24))
    if request.param == 'trocr':
        decoder = hf.TrOCRForCausalLM(hf.TrOCRConfig(vocab_size=9, d_model=16, decoder_layers=1,
            decoder_attention_heads=2, decoder_ffn_dim=24, cross_attention_hidden_size=16,
            decoder_start_token_id=1, eos_token_id=2, pad_token_id=0))
    else:
        decoder = hf.RobertaForCausalLM(hf.RobertaConfig(vocab_size=9, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=24, is_decoder=True, add_cross_attention=True,
            bos_token_id=1, eos_token_id=2, pad_token_id=0))
    model = hf.VisionEncoderDecoderModel(encoder=encoder, decoder=decoder).eval()
    model.generation_config.decoder_start_token_id = 1
    model.generation_config.eos_token_id = 2
    model.generation_config.pad_token_id = 0
    model.generation_config.max_length = 12
    model.generation_config.no_repeat_ngram_size = 3
    model.save_pretrained(source)
    hf.ViTImageProcessor(size={'height': 16, 'width': 16}).save_pretrained(source)
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({str(i): i for i in range(9)}, unk_token='0'))
    hf.PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='0', bos_token='1', eos_token='2', unk_token='0').save_pretrained(source)
    # Both formats and FP16 must preserve the same generation contract.
    for cached in (True, False):
        TrOCR.export(source, source.parent / str(cached), use_cache=cached, fp16=cached)
    return model, source.parent


@pytest.mark.parametrize('beams', [1, 4])
@pytest.mark.parametrize('cached,precision', [(False, 'fp32'), (True, 'fp32'), (True, 'fp16')])
def test_exported_decoder_matches_source_for_multiple_images(bundle, beams, cached, precision):
    import torch
    model, root = bundle
    recognizer = TrOCR(root / str(cached) / f'encoder.{precision}.onnx', device='cpu', batch_size=3,
                       generation={'num_beams': beams})
    recognizer._initialize_session()
    pixels = np.random.default_rng(13).normal(size=(3, 3, 16, 16)).astype(np.float32)
    dtype = np.float16 if recognizer.onnx_session.get_inputs()[0].type == 'tensor(float16)' else np.float32
    hidden = recognizer.onnx_session.run(None, {'pixel_values': pixels.astype(dtype)})[0]
    result = recognizer._generate_batch(hidden)
    with torch.no_grad():
        expected = model.generate(torch.from_numpy(pixels), num_beams=beams, use_cache=cached).tolist()
    for (tokens, confidence), reference in zip(result, expected):
        assert tokens == reference[:len(tokens)]
        assert 0 < confidence <= 1


def test_cache_tracks_beam_parents_and_finished_samples():
    from manuscript.recognizers._trocr.generation import generate_batch
    hidden = np.arange(3, dtype=np.float32)[:, None, None]
    generation = dict(num_beams=3, max_length=8, decoder_start_token_id=1, eos_token_id=2,
                      length_penalty=1.0, no_repeat_ngram_size=2, early_stopping=True)
    calls = []

    def decode(ids, hidden, cache):
        # Cache must contain exactly the prefix consumed at the preceding step.
        if cache is not None:
            np.testing.assert_array_equal(cache['history'], ids[:, :-1])
            np.testing.assert_array_equal(cache['owner'], hidden[:, 0, 0])
        calls.append(len(ids))
        logits = np.full((len(ids), 1, 7), -10.0)
        for i, owner in enumerate(hidden[:, 0, 0].astype(int)):
            logits[i, 0, 2 if len(ids[i]) >= owner + 2 else 3] = 3
            logits[i, 0, 4] = 2
        return logits, {'history': ids.copy(), 'owner': hidden[:, 0, 0].copy()}

    cached = generate_batch(hidden, generation, decode, TrOCR._banned)
    plain = generate_batch(hidden, generation, lambda ids, h, cache: (decode(ids, h, None)[0], None), TrOCR._banned)
    assert cached == plain
    assert len({len(tokens) for tokens, _ in cached}) > 1
    assert max(calls) > len(hidden)


def test_predict_batches_images_and_keeps_order(bundle):
    from types import SimpleNamespace
    model, root = bundle
    recognizer = TrOCR(root / 'True' / 'encoder.fp32.onnx', device='cpu', batch_size=2)
    recognizer._initialize_session()
    session = recognizer.onnx_session
    calls = []

    def run(outputs, feed):
        calls.append(next(iter(feed.values())).shape[0])
        return session.run(outputs, feed)

    recognizer.onnx_session = SimpleNamespace(get_inputs=session.get_inputs, run=run)
    images = np.random.default_rng(3).integers(0, 256, (3, 16, 16, 3), dtype=np.uint8)
    regions = [SimpleNamespace(image=image) for image in images]
    batch = recognizer._predict_text_images(regions, return_raw=True)
    assert calls == [2, 1]
    single = recognizer._predict_text_images(regions, batch_size=1)
    assert [p.meta['token_ids'] for p in batch] == [p.meta['token_ids'] for p in single]
    assert all(p.meta['encoder_output'].shape[0] == 1 for p in batch)


def test_export_preserves_source_image_processing(bundle):
    from transformers import AutoImageProcessor
    model, root = bundle
    recognizer = TrOCR(root / 'True' / 'encoder.fp32.onnx', device='cpu')
    recognizer._initialize_session()
    image = np.random.default_rng(1).integers(0, 256, (11, 29, 3), dtype=np.uint8)
    processor = AutoImageProcessor.from_pretrained(root / 'source', local_files_only=True)
    expected = processor(image, return_tensors='pt').pixel_values.numpy()
    np.testing.assert_allclose(recognizer._preprocess_image(image), expected, atol=2 / 255, rtol=0)


def test_export_module_import_does_not_load_training_dependencies():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c', '''
import sys
import manuscript.recognizers._trocr.export
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('original,target', [((37, 83), (19, 46)), ((37, 83), (73, 151)), ((300, 500), (30, 60))])
def test_native_uint8_bicubic_resize_matches_torchvision(original, target):
    torch = pytest.importorskip('torch')
    functional = pytest.importorskip('torchvision.transforms.v2.functional')
    from manuscript.recognizers._trocr import _resize_antialiased
    image = np.random.default_rng(2).integers(0, 256, (*original, 3), dtype=np.uint8)
    expected = functional.resize(torch.from_numpy(image.transpose(2, 0, 1)), list(target),
                                 interpolation=functional.InterpolationMode.BICUBIC, antialias=True).numpy().transpose(1, 2, 0)
    np.testing.assert_array_equal(_resize_antialiased(image, *target, 3, uint8_rounding=True), expected)
