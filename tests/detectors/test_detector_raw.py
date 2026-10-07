from types import SimpleNamespace

import numpy as np
import pytest

from manuscript.detectors import EAST, YOLO


@pytest.mark.parametrize("detector_class", [EAST, YOLO])
def test_axis_aligned_option_is_deprecated(detector_class, tmp_path):
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"test")
    with pytest.warns(DeprecationWarning, match="axis_aligned_output"):
        detector_class(weights=str(weights), axis_aligned_output=True)
    assert not detector_class(weights=str(weights)).axis_aligned_output


def test_east_raw_preserves_native_quads_and_model_outputs(tmp_path, monkeypatch):
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"test")
    with pytest.warns(DeprecationWarning):
        model = EAST(weights=str(weights), target_size=100, axis_aligned_output=True,
                     expand_ratio_w=0, expand_ratio_h=0)
    model.onnx_session = object()
    quad = np.array([[10, 10, 60, 20, 55, 40, 5, 30, .9]], np.float32)
    score_map, geo_map = np.zeros((25, 25)), np.zeros((5, 25, 25))
    monkeypatch.setattr(model, "_run_inference_on_image", lambda image, return_raw=False: (quad, score_map, geo_map, {"score_map": score_map, "geo_map": geo_map}) if return_raw else (quad, score_map, geo_map))
    raw = model.predict(np.zeros((100, 100, 3), np.uint8), return_raw=True, return_outputs=True)
    compact = model.predict(np.zeros((100, 100, 3), np.uint8), return_raw=True)
    assert "outputs" not in compact
    assert compact["detections"] == raw["detections"]
    polygon = np.asarray(raw["detections"][0]["polygon"])
    assert len(set(polygon[:, 1])) > 2
    assert raw["outputs"]["score_map"] is score_map
    assert raw["page"] == model.predict(np.zeros((100, 100, 3), np.uint8))


def test_yolo_raw_preserves_obb_angle_and_all_outputs(tmp_path, monkeypatch):
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"test")
    output = np.array([[[50, 50, 40, 10, .9, 2, .4]]], np.float32)
    auxiliary = np.ones((1, 3), np.float16)
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="images")],
        get_outputs=lambda: [SimpleNamespace(name="boxes"), SimpleNamespace(name="aux")],
        run=lambda *args: [output, auxiliary],
    )
    with pytest.warns(DeprecationWarning):
        model = YOLO(weights=str(weights), axis_aligned_output=True)
    model.onnx_session = session
    model._output_layout = "obb"
    monkeypatch.setattr(model, "_preprocess", lambda image: (image, 1., (0., 0.)))
    raw = model.predict(np.zeros((100, 100, 3), np.uint8), return_raw=True, return_outputs=True)
    assert raw["outputs"]["boxes"] is output
    assert raw["outputs"]["aux"] is auxiliary
    assert raw["detections"][0]["angle"] == pytest.approx(.4)
    compact = model.predict(np.zeros((100, 100, 3), np.uint8), return_raw=True)
    assert "outputs" not in compact
    assert compact["detections"] == raw["detections"]
    polygon = np.asarray(raw["detections"][0]["polygon"])
    assert len(set(polygon[:, 1])) > 2
