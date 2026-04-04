from pathlib import Path

import numpy as np
import pytest

from manuscript.api.detector import BaseDetector
from manuscript.data import Page
from manuscript.detectors import YOLO


def _mock_download_http(model_path, yaml_path):
    def _download_http(self, url):
        if url.endswith(".onnx"):
            return str(model_path)
        if url.endswith(".yaml"):
            return str(yaml_path)
        raise AssertionError(f"Unexpected download URL: {url}")

    return _download_http


class _FakeIO:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _FakeSessionDetect:
    def __init__(self, providers=None):
        self._providers = providers or ["CPUExecutionProvider"]
        self.last_inputs = None

    def get_inputs(self):
        return [_FakeIO("images", [1, 3, 1280, 1280])]

    def get_outputs(self):
        return [_FakeIO("output0", [1, None, 6])]

    def get_providers(self):
        return list(self._providers)

    def run(self, output_names, feed_dict):
        self.last_inputs = {"output_names": output_names, "feed_dict": feed_dict}
        return [
            np.array(
                [
                    [
                        [200.0, 420.0, 600.0, 620.0, 0.9, 0.0],
                        [10.0, 10.0, 10.0, 20.0, 0.8, 0.0],
                        [40.0, 180.0, 80.0, 260.0, 0.2, 1.0],
                        [800.0, 440.0, 1000.0, 560.0, 0.7, 1.0],
                        [500.0, 500.0, 520.0, 520.0, np.nan, 0.0],
                    ]
                ],
                dtype=np.float32,
            )
        ]


class _FakeSessionOBB:
    def __init__(self, providers=None):
        self._providers = providers or ["CPUExecutionProvider"]
        self.last_inputs = None

    def get_inputs(self):
        return [_FakeIO("images", [1, 3, 1280, 1280])]

    def get_outputs(self):
        return [_FakeIO("output0", [1, None, 7])]

    def get_providers(self):
        return list(self._providers)

    def run(self, output_names, feed_dict):
        self.last_inputs = {"output_names": output_names, "feed_dict": feed_dict}
        return [
            np.array(
                [
                    [
                        [400.0, 520.0, 200.0, 100.0, 0.9, 0.0, 0.0],
                        [10.0, 10.0, 0.0, 20.0, 0.8, 0.0, 0.0],
                        [880.0, 520.0, 200.0, 120.0, 0.7, 1.0, 0.0],
                        [500.0, 500.0, 20.0, 20.0, np.nan, 0.0, 0.0],
                    ]
                ],
                dtype=np.float32,
            )
        ]


def test_yolo_is_exported():
    assert YOLO is not None


def test_yolo_inherits_base_detector():
    assert issubclass(YOLO, BaseDetector)


def test_yolo_has_default_preset():
    assert YOLO.default_weights_name == "yolo26s_obb_text_g1"
    assert YOLO.pretrained_registry["yolo26s_obb_text_g1"] == (
        "https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26s_obb_text_g1.raw.onnx"
    )
    assert YOLO.pretrained_registry["yolo26x_obb_text_g1"] == (
        "https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26x_obb_text_g1.raw.onnx"
    )


def test_yolo_uses_default_preset_when_weights_missing(monkeypatch, tmp_path):
    model_path = tmp_path / "preset.onnx"
    model_path.write_bytes(b"fake")
    yaml_path = tmp_path / "downloaded-default.yaml"
    yaml_path.write_text("imgsz: 1280\n", encoding="utf-8")

    monkeypatch.setattr(
        "manuscript.api.base.BaseArtifactModel._download_http",
        _mock_download_http(model_path, yaml_path),
    )

    detector = YOLO(weights=None)

    assert Path(detector.weights) == model_path


def test_yolo_default_preset_uses_1280_target_size(monkeypatch, tmp_path):
    model_path = tmp_path / "preset.onnx"
    model_path.write_bytes(b"fake")
    yaml_path = tmp_path / "downloaded-default.yaml"
    yaml_path.write_text("imgsz: 1280\n", encoding="utf-8")

    monkeypatch.setattr(
        "manuscript.api.base.BaseArtifactModel._download_http",
        _mock_download_http(model_path, yaml_path),
    )

    detector = YOLO(weights=None)

    assert detector.target_size == 1280


def test_yolo_defaults_to_01_score_thresh(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")

    detector = YOLO(weights=str(model_path))

    assert detector.score_thresh == 0.1


def test_yolo26x_preset_uses_1024_target_size(monkeypatch, tmp_path):
    model_path = tmp_path / "preset.onnx"
    model_path.write_bytes(b"fake")
    yaml_path = tmp_path / "downloaded-x.yaml"
    yaml_path.write_text("imgsz: 1024\n", encoding="utf-8")

    monkeypatch.setattr(
        "manuscript.api.base.BaseArtifactModel._download_http",
        _mock_download_http(model_path, yaml_path),
    )

    detector = YOLO(weights="yolo26x_obb_text_g1")

    assert detector.target_size == 1024


def test_yolo_reads_local_yaml_imgsz_for_target_size(tmp_path):
    model_path = tmp_path / "custom.raw.onnx"
    model_path.write_bytes(b"fake")
    model_path.with_suffix(".yaml").write_text("imgsz: [960, 960]\n", encoding="utf-8")

    detector = YOLO(weights=str(model_path))

    assert detector.target_size == 960


def test_yolo_suppresses_mostly_contained_boxes_by_default():
    boxes = np.array(
        [
            [10.0, 10.0, 100.0, 100.0, 0.95, 0.0],
            [18.0, 18.0, 82.0, 82.0, 0.99, 0.0],
            [120.0, 20.0, 180.0, 80.0, 0.90, 1.0],
        ],
        dtype=np.float32,
    )

    detector = YOLO.__new__(YOLO)
    detector.containment_threshold = 0.9
    filtered = detector._suppress_contained_boxes(boxes)

    assert filtered.shape == (2, 6)
    assert (filtered[:, 5] == np.array([0.0, 1.0], dtype=np.float32)).all()


def test_yolo_can_disable_containment_cleanup(monkeypatch, tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")

    fake_session = _FakeSessionDetect()
    monkeypatch.setattr(
        "manuscript.detectors._yolo.ort.InferenceSession",
        lambda *_args, **_kwargs: fake_session,
    )

    detector = YOLO(
        weights=str(model_path),
        containment_threshold=None,
    )

    nested_output = np.array(
        [
            [
                [100.0, 100.0, 400.0, 300.0, 0.9, 0.0],
                [150.0, 150.0, 200.0, 220.0, 0.8, 0.0],
            ]
        ],
        dtype=np.float32,
    )

    boxes, polygons = detector._postprocess(
        nested_output,
        image_hw=(640, 640),
        ratio=1.0,
        pad=(0.0, 0.0),
    )

    assert boxes.shape == (2, 6)
    assert polygons.shape == (2, 4, 2)


def test_yolo_predict_returns_axis_aligned_rows_for_obb_by_default(
    monkeypatch,
    tmp_path,
):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")

    fake_session = _FakeSessionOBB()
    monkeypatch.setattr(
        "manuscript.detectors._yolo.ort.InferenceSession",
        lambda *_args, **_kwargs: fake_session,
    )

    detector = YOLO(
        weights=str(model_path),
        device="cpu",
        score_thresh=0.25,
        class_ids=[0],
    )

    image = np.zeros((320, 640, 3), dtype=np.uint8)
    page = detector.predict(image)

    assert fake_session.last_inputs is not None
    tensor = fake_session.last_inputs["feed_dict"]["images"]
    assert tensor.shape == (1, 3, 1280, 1280)
    assert tensor.dtype == np.float32

    assert isinstance(page, Page)
    assert len(page.blocks) == 1
    assert len(page.blocks[0].lines) == 1
    assert len(page.blocks[0].lines[0].text_spans) == 1
    assert page.blocks[0].lines[0].text_spans[0].polygon == [
        (150.0, 75.0),
        (250.0, 75.0),
        (250.0, 125.0),
        (150.0, 125.0),
    ]


def test_yolo_predict_returns_rotated_rows_when_requested(monkeypatch, tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")

    fake_session = _FakeSessionOBB()
    monkeypatch.setattr(
        "manuscript.detectors._yolo.ort.InferenceSession",
        lambda *_args, **_kwargs: fake_session,
    )

    detector = YOLO(
        weights=str(model_path),
        device="cpu",
        score_thresh=0.25,
        class_ids=[0],
        axis_aligned_output=False,
    )

    image = np.zeros((320, 640, 3), dtype=np.uint8)
    page = detector.predict(image)

    assert isinstance(page, Page)
    polygons = page.blocks[0].lines[0].text_spans[0].polygon
    assert polygons == [
        (150.0, 75.0),
        (250.0, 75.0),
        (250.0, 125.0),
        (150.0, 125.0),
    ]


def test_yolo_supports_legacy_axis_aligned_output_format(monkeypatch, tmp_path):
    model_path = tmp_path / "legacy.onnx"
    model_path.write_bytes(b"fake")

    fake_session = _FakeSessionDetect()
    monkeypatch.setattr(
        "manuscript.detectors._yolo.ort.InferenceSession",
        lambda *_args, **_kwargs: fake_session,
    )

    detector = YOLO(
        weights=str(model_path),
        device="cpu",
        score_thresh=0.25,
        class_ids=[0],
    )

    image = np.zeros((320, 640, 3), dtype=np.uint8)
    page = detector.predict(image)

    assert isinstance(page, Page)
    polygon = page.blocks[0].lines[0].text_spans[0].polygon
    assert polygon == [
        (100.0, 50.0),
        (300.0, 50.0),
        (300.0, 150.0),
        (100.0, 150.0),
    ]


def test_yolo_supports_dynamic_onnx_input_with_default_1280(monkeypatch, tmp_path):
    class _DynamicSession(_FakeSessionOBB):
        def get_inputs(self):
            return [_FakeIO("images", [1, 3, "height", "width"])]

    model_path = tmp_path / "dynamic.onnx"
    model_path.write_bytes(b"fake")

    monkeypatch.setattr(
        "manuscript.detectors._yolo.ort.InferenceSession",
        lambda *_args, **_kwargs: _DynamicSession(),
    )

    detector = YOLO(weights=str(model_path))
    detector._initialize_session()

    assert detector.target_size == 1280


def test_yolo_rejects_static_shape_mismatch(monkeypatch, tmp_path):
    model_path = tmp_path / "mismatch.onnx"
    model_path.write_bytes(b"fake")

    monkeypatch.setattr(
        "manuscript.detectors._yolo.ort.InferenceSession",
        lambda *_args, **_kwargs: _FakeSessionOBB(),
    )

    detector = YOLO(weights=str(model_path), target_size=640)

    with pytest.raises(ValueError, match="expects input size"):
        detector._initialize_session()
