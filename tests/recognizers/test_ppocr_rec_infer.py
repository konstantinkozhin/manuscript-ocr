import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from manuscript.recognizers import PPOCRRec


@pytest.fixture
def bundle(tmp_path):
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"fake-session")
    config = {
        "rec_image_shape": [3, 32, 64],
        "preprocess": {"color_order": "RGB", "normalization": "invert",
                       "dynamic_width": True, "min_width": 0, "padding": 16,
                       "interpolation": "lanczos"},
        "postprocess": {"output_is_logits": True},
        "PostProcess": {"character_dict": ["а", "б", " "]},
        "input_name": "images", "output_name": "logits",
        "length_input_name": "lengths", "length_input_units": "pixels", "length_output_name": "output_lengths",
    }
    weights.with_suffix(".json").write_text(json.dumps(config))
    return weights, config


def test_kraken_preprocessing_preserves_long_rgb_lines(bundle):
    model = PPOCRRec(str(bundle[0]))
    image = np.full((16, 150, 3), [255, 128, 0], np.uint8)
    data = model._preprocess_image(image)
    assert data.shape == (1, 3, 32, 332)
    np.testing.assert_allclose(data[0, :, 0, 16], [0, 127 / 255, 1])
    assert not data[0, :, :, :16].any()
    assert not data[0, :, :, -16:].any()
    assert data.flags.c_contiguous


def test_kraken_preprocessing_matches_lanczos_resize(bundle):
    image = np.random.default_rng(42).integers(0, 256, (29, 123, 3), np.uint8)
    resized = np.asarray(Image.fromarray(image).resize((int(123 * 32 / 29), 32), Image.Resampling.LANCZOS))
    expected = np.pad(resized, ((0, 0), (16, 16), (0, 0)), constant_values=255)
    expected = (expected.max() - expected.astype(np.float32)) / 255
    np.testing.assert_array_equal(PPOCRRec(str(bundle[0]))._preprocess_image(image)[0], expected.transpose(2, 0, 1))


def test_ctc_logits_blank_repetition_spaces_and_confidence(bundle):
    model = PPOCRRec(str(bundle[0]))
    logits = np.full((1, 7, 4), -10., np.float32)
    logits[0, np.arange(7), [1, 1, 0, 1, 3, 2, 2]] = 10
    result = model._decode_recognition_logits(logits)[0]
    assert result.text == "аа б"
    assert 0.999 < result.confidence <= 1
    with pytest.raises(ValueError, match="dictionary"):
        model._decode_recognition_logits(np.zeros((1, 2, 5)))
    with pytest.raises(ValueError, match="Invalid"):
        model._decode_recognition_logits(np.full((1, 2, 4), np.nan))


@pytest.mark.parametrize("dtype,fixed_batch", [("tensor(float)", None), ("tensor(float16)", 2)])
def test_mixed_width_batches_use_lengths_and_trim_ctc_output(bundle, monkeypatch, dtype, fixed_batch):
    feeds = []

    class Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="images", type=dtype, shape=[fixed_batch or "N", 3, 32, "W"]),
                    SimpleNamespace(name="lengths", type="tensor(int64)", shape=[fixed_batch or "N"])]

        def get_outputs(self):
            return [SimpleNamespace(name="logits", shape=["N", "T", 4]),
                    SimpleNamespace(name="output_lengths", shape=["N"])]

        def run(self, names, feed):
            assert names == ["logits", "output_lengths"]
            feeds.append(feed)
            images = feed["images"]
            assert images.dtype == (np.float16 if "float16" in dtype else np.float32)
            if fixed_batch:
                assert len(images) == fixed_batch
            lengths = feed["lengths"] // 8
            logits = np.full((len(images), images.shape[3] // 8, 4), -10., np.float32)
            logits[:, :, 2] = 10  # Padding emits б; trimmed before decoding.
            for index, length in enumerate(lengths):
                logits[index, :length, 2] = -10
                logits[index, :length, 1] = 10
            return [logits, lengths]

    monkeypatch.setattr("manuscript.recognizers._ppocr_rec.ort.InferenceSession", lambda *a, **kw: Session())
    model = PPOCRRec(str(bundle[0]), batch_size=2)
    images = [np.zeros((32, width, 3), np.uint8) for width in [80, 40, 64]]
    results = model._predict_word_images(images)
    assert [result["text"] for result in results] == ["а", "а", "а"]
    np.testing.assert_array_equal(feeds[0]["lengths"], [112, 72])
    assert feeds[0]["images"].shape == (2, 3, 32, 112)
    assert len(results) == 3


def test_future_registry_bundle_loads_config_and_charset(bundle, tmp_path, monkeypatch):
    weights, config = bundle
    config_path = tmp_path / "separate-config.json"
    config_path.write_text(json.dumps(config))
    charset = tmp_path / "dictionary.txt"
    charset.write_text("x\ny\n \n")
    calls = []

    def resolve(name, model_class, force_download=False):
        calls.append((name, model_class, force_download))
        return {"weights": weights, "config": config_path, "charset": charset}

    monkeypatch.setattr("manuscript.models.resolve", resolve)
    model = PPOCRRec("future_rec_model", force_download=True)
    assert calls == [("future_rec_model", "PPOCRRec", True)]
    assert model.config_path == str(config_path)
    assert model.characters == ["blank", "x", "y", " "]
    assert PPOCRRec.registry_model_class == "PPOCRRec"


def test_native_paddle_config_adds_space_and_finds_resize_op(bundle):
    weights, _ = bundle
    weights.with_suffix(".json").write_text(json.dumps({
        "PreProcess": {"transform_ops": [{"RecResizeImg": {"image_shape": [3, 48, 320]}}]},
        "PostProcess": {"name": "CTCLabelDecode", "character_dict": ["а", "б"]},
    }))
    model = PPOCRRec(str(weights))
    assert model.characters == ["blank", "а", "б", " "]
    assert model.rec_image_shape == [3, 48, 320]
    assert model._preprocess_image(np.zeros((48, 800, 3), np.uint8)).shape == (1, 3, 48, 800)


def test_paddle_downsampling_uses_official_linear_bgr_normalization(bundle):
    weights, _ = bundle
    weights.with_suffix(".json").write_text(json.dumps({
        "rec_image_shape": [3, 48, 320], "PostProcess": {"character_dict": ["а"]},
    }))
    image = np.random.default_rng(3).integers(0, 256, (96, 800, 3), np.uint8)
    expected = cv2.resize(image[:, :, ::-1], (400, 48)).astype(np.float32)
    expected = (expected / 255 - 0.5) / 0.5
    np.testing.assert_array_equal(PPOCRRec(str(weights))._preprocess_image(image)[0], expected.transpose(2, 0, 1))


@pytest.mark.parametrize("field,value", [("normalization", "other"), ("padding", -1), ("color_order", "GRAY")])
def test_invalid_preprocessing_fails_early(bundle, field, value):
    weights, config = bundle
    config["preprocess"][field] = value
    weights.with_suffix(".json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        PPOCRRec(str(weights))


@pytest.mark.parametrize("recognizer_class", ["PPOCRRec", "TrOCR", "TRBA"])
def test_page_prediction_and_debug_crops_use_shared_methods(bundle, tmp_path, monkeypatch, recognizer_class):
    from manuscript.data import Block, Line, Page, TextSpan
    from manuscript.recognizers import TRBA, TrOCR
    from manuscript.recognizers._common.region_types import RecognitionPrediction

    model = PPOCRRec(str(bundle[0]), rotate_threshold=None)
    if recognizer_class != "PPOCRRec":
        # Exercise inherited Page integration without running model backends.
        recognizer = object.__new__({"TrOCR": TrOCR, "TRBA": TRBA}[recognizer_class])
        recognizer.__dict__.update(model.__dict__)
        model = recognizer
    page = Page(blocks=[Block(lines=[Line(text_spans=[TextSpan(
        polygon=[(5., 5.), (45., 5.), (45., 25.), (5., 25.)],
        detection_confidence=0.9,
    )])])])
    crops = []

    def predict_regions(regions, batch_size):
        crops.extend(regions)
        return [RecognitionPrediction(text="аб", confidence=0.95)]

    monkeypatch.setattr(model, "_predict_text_images", predict_regions)
    debug_dir = tmp_path / "debug"
    result = model.predict(page, np.zeros((40, 60, 3), np.uint8), debug_save_dir=debug_dir)
    assert page.blocks[0].lines[0].text_spans[0].text is None
    span = result.blocks[0].lines[0].text_spans[0]
    assert span.text == "аб"
    assert span.recognition_confidence == 0.95
    assert crops[0].text_span is span
    assert list(debug_dir.glob("*.png"))
    assert (debug_dir / "index.json").exists()


def test_invalid_graph_is_not_cached_after_failed_initialization(bundle, monkeypatch):
    attempts = []

    class Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="images", type="tensor(int64)", shape=[1, 3, 32, 64])]

    def create(*args, **kwargs):
        attempts.append(1)
        return Session()

    monkeypatch.setattr("manuscript.recognizers._ppocr_rec.ort.InferenceSession", create)
    model = PPOCRRec(str(bundle[0]))
    for _ in range(2):
        with pytest.raises(ValueError, match="float32/float16"):
            model._initialize_session()
        assert model.onnx_session is None
    assert len(attempts) == 2


def test_missing_config_is_not_guessed(bundle):
    bundle[0].with_suffix('.json').unlink()
    with pytest.raises(ValueError, match='requires a config'):
        PPOCRRec(str(bundle[0]))


def test_unknown_length_units_fail(bundle):
    weights, config = bundle
    config.pop('length_input_units')
    weights.with_suffix('.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='length_input_units'):
        PPOCRRec(str(weights))


def test_raw_decode_retains_original_logits(bundle):
    model = PPOCRRec(str(bundle[0]))
    logits = np.zeros((1, 2, len(model.characters)), np.float32)
    results = model._decode_predictions(logits, return_raw=True)
    np.testing.assert_array_equal(results[0].meta['model_output'], logits[0])
