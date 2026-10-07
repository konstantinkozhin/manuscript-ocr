import json
from types import SimpleNamespace

import numpy as np
import pytest
import cv2

from manuscript.api.detector import BaseDetector
from manuscript.data import Page
from manuscript.detectors._ppocr import PPOCR



@pytest.fixture
def config():
    return {
        "schema_version": 1, "task": "text_detection", "algorithm": "DB",
        "preprocess": {
            "color_order": "BGR", "scale": 1 / 255,
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "limit_side_len": 960, "limit_type": "max", "max_side_limit": 4000,
        },
        "postprocess": {
            "threshold": 0.3, "box_threshold": 0.6, "unclip_ratio": 1.5,
            "max_candidates": 1000, "min_size": 3, "box_type": "poly",
            "unclip_join": "round", "output_is_logits": False, "output_channel": 0,
        },
    }

@pytest.fixture
def weights(tmp_path, config):
    path = tmp_path / "detector.onnx"
    path.write_bytes(b"test-session")
    path.with_suffix(".json").write_text(json.dumps(config))
    return path


def test_sidecar_normalization_and_constructor_overrides(weights, config):
    config["preprocess"].update(limit_type="min", limit_side_len=64)
    config["postprocess"].update(threshold=0.2, box_threshold=0.4)
    weights.with_suffix(".json").write_text(json.dumps(config))
    model = PPOCR(weights, box_threshold=0.5)
    assert isinstance(model, BaseDetector)
    assert model._resize_shape(32, 48) == (64, 96)
    image = np.full((32, 48, 3), [255, 128, 0], dtype=np.uint8)
    result = model._preprocess(image)
    np.testing.assert_allclose(result[0, :, 0, 0],
                               (np.array([0, 128, 255]) / 255 - model.preprocess["mean"]) / model.preprocess["std"],
                               atol=1e-6)
    assert model.postprocess["threshold"] == 0.2
    assert model.postprocess["box_threshold"] == 0.5
    assert result.flags.c_contiguous


@pytest.mark.parametrize("input_type,dtype", [("tensor(float)", np.float32), ("tensor(float16)", np.float16)])
def test_predict_uses_graph_names_precision_and_fixed_shape(weights, monkeypatch, input_type, dtype):
    class Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="x", type=input_type, shape=[1, 3, 64, 96])]

        def get_outputs(self):
            return [SimpleNamespace(name="maps")]

        def run(self, names, feed):
            assert names == ["maps"]
            assert feed["x"].dtype == dtype
            assert feed["x"].shape == (1, 3, 64, 96)
            result = np.zeros((1, 1, 32, 48), np.float32)
            result[0, 0, 8:20, 10:35] = 0.9
            return [result]

    monkeypatch.setattr("manuscript.detectors._ppocr.ort.InferenceSession", lambda *a, **kw: Session())
    model = PPOCR(weights, unclip_ratio=0)
    page = model.predict(np.zeros((128, 192, 3), np.uint8))
    assert isinstance(page, Page)
    spans = page.blocks[0].lines[0].text_spans
    assert len(spans) == 1
    assert spans[0].detection_confidence == pytest.approx(0.9)
    np.testing.assert_allclose(spans[0].polygon, [[40, 32], [136, 32], [136, 76], [40, 76]])
    raw = model.predict(np.zeros((128, 192, 3), np.uint8), return_raw=True, return_outputs=True)
    assert raw["page"] == page
    assert raw["outputs"]["maps"].shape == (1, 1, 32, 48)
    assert len(raw["detections"]) == 1
    assert raw["image_size"] == {"height": 128, "width": 192}
    compact = model.predict(np.zeros((128, 192, 3), np.uint8), return_raw=True)
    assert "outputs" not in compact and "masks" not in compact
    assert compact["detections"] == raw["detections"]
    masked = model.predict(np.zeros((128, 192, 3), np.uint8), return_masks=True)
    assert "outputs" not in masked
    assert masked["masks"]["text"].shape == (128, 192)
    assert masked["masks"]["text"].dtype == np.bool_
    from manuscript import Pipeline

    pipeline_page = Pipeline(detector=model, layout=None, recognizer=None).predict(
        np.zeros((128, 192, 3), np.uint8)
    )["page"]
    assert len(pipeline_page.blocks[0].lines[0].text_spans) == 1


def test_db_score_filtering_unclip_and_polygon_mode(weights):
    maps = np.zeros((1, 1, 64, 96), np.float32)
    maps[0, 0, 12:26, 20:70] = 0.9
    maps[0, 0, 40:50, 5:20] = 0.4
    model = PPOCR(weights, box_threshold=0.6, unclip_ratio=1.5)
    page = model._postprocess(maps, (128, 192))
    spans = page.blocks[0].lines[0].text_spans
    assert len(spans) == 1
    assert min(x for x, _ in spans[0].polygon) < 40
    assert max(x for x, _ in spans[0].polygon) > 138
    poly = model._postprocess(maps, (128, 192)).blocks[0].lines[0].text_spans[0].polygon
    assert len(poly) > 4
    assert all(0 <= x <= 192 and 0 <= y <= 128 for x, y in poly)


def test_empty_output_and_invalid_maps(weights):
    model = PPOCR(weights)
    assert model._postprocess(np.zeros((1, 1, 64, 64)), (64, 64)).blocks[0].lines[0].text_spans == []
    for value in [np.ones((2, 1, 64, 64)), np.full((64, 64), np.nan), np.full((64, 64), 2.0)]:
        with pytest.raises(ValueError):
            model._postprocess(value, (64, 64))
    model.postprocess["output_is_logits"] = True
    result = model._postprocess(np.full((64, 64), -10.0), (64, 64))
    assert result.blocks[0].lines[0].text_spans == []


@pytest.mark.parametrize("invalid", [
    {"algorithm": "CTC"}, {"schema_version": 2},
    {"preprocess": {"std": [1, 0, 1]}},
    {"preprocess": {"color_order": "GRAY"}},
    {"postprocess": {"box_threshold": 1.5}},
    {"postprocess": {"unclip_ratio": -1}},
])
def test_bad_configs_fail_before_inference(weights, config, invalid):
    for key, value in invalid.items():
        if isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    with pytest.raises(ValueError):
        PPOCR(weights, config=config)


def test_missing_explicit_weights():
    with pytest.raises(ValueError, match="explicit ONNX"):
        PPOCR(None)


@pytest.mark.parametrize("explicit", [False, True])
def test_registry_weights_and_config(weights, tmp_path, monkeypatch, config, explicit):
    # Registry config need not share the ONNX filename; it also beats a sidecar.
    config_path = tmp_path / "registry_config.json"
    config["postprocess"]["threshold"] = 0.2
    config_path.write_text(json.dumps(config))
    config["postprocess"]["threshold"] = 0.9
    weights.with_suffix(".json").write_text(json.dumps(config))
    config["postprocess"]["threshold"] = 0.7
    explicit_config = config if explicit else None
    calls = []

    def resolve(name, model_class, force_download=False):
        calls.append((name, model_class, force_download))
        return {"weights": weights, "config": config_path}

    monkeypatch.setattr("manuscript.models.resolve", resolve)
    model = PPOCR("future_ppocr_model", config=explicit_config, force_download=True)
    assert calls == [("future_ppocr_model", "PPOCR", True)]
    assert model.weights == str(weights)
    assert model.postprocess["threshold"] == (0.2 if explicit_config is None else 0.7)


def test_registry_without_config_fails(weights, monkeypatch):
    weights.with_suffix(".json").unlink()
    monkeypatch.setattr("manuscript.models.resolve", lambda *a, **kw: {"weights": weights})
    with pytest.raises(ValueError, match="requires a configuration"):
        PPOCR("future_ppocr_model")


def test_local_weights_without_config_fail(weights):
    weights.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="requires a configuration"):
        PPOCR(weights)


@pytest.mark.parametrize("section,key", [
    ("preprocess", "mean"), ("preprocess", "scale"),
    ("postprocess", "threshold"), ("postprocess", "output_channel"),
])
def test_incomplete_config_fails_even_with_override(weights, config, section, key):
    del config[section][key]
    with pytest.raises(ValueError, match=f"missing {section} fields: {key}"):
        PPOCR(weights, config=config, threshold=0.4)


def test_overrides_do_not_mutate_config(weights, config):
    model = PPOCR(weights, config=config, threshold=0.8)
    assert model.postprocess["threshold"] == 0.8
    assert config["postprocess"]["threshold"] == 0.3


def test_exactly_45_degree_regions_keep_four_distinct_vertices(weights):
    maps = np.zeros((64, 64), np.float32)
    cv2.fillPoly(maps, [np.array([[32, 8], [56, 32], [32, 56], [8, 32]], np.int32)], 0.9)
    model = PPOCR(weights, unclip_ratio=0)
    spans = model._postprocess(maps, (64, 64)).blocks[0].lines[0].text_spans
    assert len(spans) == 1
    assert len(set(spans[0].polygon)) == 4
