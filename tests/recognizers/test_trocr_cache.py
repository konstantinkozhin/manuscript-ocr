import numpy as np
import pytest
from manuscript.recognizers import TrOCR

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


@pytest.mark.parametrize('original,target', [((37, 83), (19, 46)), ((37, 83), (73, 151)), ((300, 500), (30, 60))])
def test_native_uint8_bicubic_resize_matches_torchvision(original, target):
    torch = pytest.importorskip('torch')
    functional = pytest.importorskip('torchvision.transforms.v2.functional')
    from manuscript.recognizers._trocr import _resize_antialiased
    image = np.random.default_rng(2).integers(0, 256, (*original, 3), dtype=np.uint8)
    expected = functional.resize(torch.from_numpy(image.transpose(2, 0, 1)), list(target),
                                 interpolation=functional.InterpolationMode.BICUBIC, antialias=True).numpy().transpose(1, 2, 0)
    np.testing.assert_array_equal(_resize_antialiased(image, *target, 3, uint8_rounding=True), expected)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_generation_uses_last_position_without_mutating_decoder_output(dtype):
    from manuscript.recognizers._trocr.generation import generate_batch

    generation = dict(num_beams=1, max_length=4, decoder_start_token_id=1,
                      eos_token_id=2, length_penalty=1., early_stopping=False,
                      no_repeat_ngram_size=0)
    logits = np.zeros((1, 3, 5), dtype)
    logits[:, -1, 2] = 10
    original = logits.copy()
    result = generate_batch(np.zeros((1, 1, 2)), generation,
                            lambda *args: (logits, None), TrOCR._banned)
    assert result[0][0] == [1, 2]
    np.testing.assert_array_equal(logits, original)
    logits[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        generate_batch(np.zeros((1, 1, 2)), generation,
                       lambda *args: (logits, None), TrOCR._banned)
