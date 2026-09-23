import json
from pathlib import Path

import numpy as np

from manuscript.api.detector import BaseDetector
from manuscript.detectors import Mask2Former
from manuscript.layouts import SimpleSorting


class _FakeIO:
    def __init__(self, name, type_name):
        self.name = name
        self.type = type_name


class _FakeSession:
    def __init__(self, providers=None):
        self.providers = providers or ["CPUExecutionProvider"]
        self.feed = None

    def get_inputs(self):
        return [_FakeIO("pixel_values", "tensor(float)"),
                _FakeIO("pixel_mask", "tensor(int64)")]

    def get_providers(self):
        return self.providers

    def run(self, output_names, feed):
        self.feed = feed
        classes = np.array([[[6.0, -6.0], [5.0, -5.0]]], dtype=np.float32)
        masks = np.full((1, 2, 256, 256), -8.0, dtype=np.float32)
        masks[0, 0, 60:85, 30:220] = 8.0
        masks[0, 1, 145:175, 45:210] = 8.0
        return [classes, masks]


def _artifacts(tmp_path):
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"fake")
    config = tmp_path / "model.json"
    config.write_text(json.dumps({
        "schema_version": 1,
        "image_size": 1024,
        "mask_interpolation_size": 384,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "score_threshold": 0.15,
        "mask_logit_threshold": 0.0,
        "nms_iou_threshold": 0.5,
        "min_mask_area": 16,
    }), encoding="utf-8")
    return weights, config


def test_mask2former_is_public_detector():
    assert issubclass(Mask2Former, BaseDetector)
    assert Mask2Former.default_weights_name == "mask2former_line_v0_prev"


def test_config_defaults_and_explicit_overrides(tmp_path):
    weights, config = _artifacts(tmp_path)
    detector = Mask2Former(weights=weights, config=config,
                           score_threshold=0.25, min_mask_area=32)
    assert detector.image_size == 1024
    assert detector.score_threshold == 0.25
    assert detector.min_mask_area == 32


def test_predict_returns_one_text_span_per_line(monkeypatch, tmp_path):
    weights, config = _artifacts(tmp_path)
    session = _FakeSession()
    session_kwargs = {}

    def make_session(*_args, **kwargs):
        session_kwargs.update(kwargs)
        return session

    monkeypatch.setattr(
        "manuscript.detectors._mask2former.ort.InferenceSession",
        make_session,
    )
    detector = Mask2Former(weights=weights, config=config, device="cpu")
    page = detector.predict(np.full((400, 600, 3), 255, dtype=np.uint8))

    assert len(page.blocks) == 1
    assert len(page.blocks[0].lines) == 2
    assert all(len(line.text_spans) == 1 for line in page.blocks[0].lines)
    assert all(line.order is None for line in page.blocks[0].lines)
    assert all(line.text_spans[0].order is None for line in page.blocks[0].lines)
    assert session.feed["pixel_values"].shape == (1, 3, 1024, 1024)
    assert session.feed["pixel_values"].dtype == np.float32
    assert session.feed["pixel_mask"].dtype == np.int64
    assert session_kwargs["sess_options"].log_severity_level == 3
    assert all(len(line.text_spans[0].polygon) >= 4 for line in page.blocks[0].lines)

    ordered = SimpleSorting(use_columns=False).predict(page)
    assert [line.order for line in ordered.blocks[0].lines] == [0, 1]
    assert all(len(line.text_spans) == 1 for line in ordered.blocks[0].lines)
    assert all(line.text_spans[0].order == 0 for line in ordered.blocks[0].lines)


def test_default_preset_uses_registry_bundle(monkeypatch, tmp_path):
    weights, config = _artifacts(tmp_path)
    monkeypatch.setattr(
        "manuscript.models.resolve",
        lambda *args, **kwargs: {"weights": weights, "config": config},
    )
    detector = Mask2Former()
    assert Path(detector.weights) == weights
    assert Path(detector.config_path) == config
