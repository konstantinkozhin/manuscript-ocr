import pytest
import json
from pathlib import Path

import numpy as np

from manuscript.api.detector import BaseDetector
from manuscript.detectors._mask2former import Mask2Former
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
        "region_classes": {"0": {"type": "line"}},
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

    assert len(page.children) == 2
    assert all(line.type == "line" and not line.children for line in page.children)
    assert (page.image_width, page.image_height) == (600, 400)
    assert session.feed["pixel_values"].shape == (1, 3, 1024, 1024)
    assert session.feed["pixel_values"].dtype == np.float32
    assert session.feed["pixel_mask"].dtype == np.int64
    assert session_kwargs["sess_options"].log_severity_level == 3
    assert all(len(line.polygon) >= 4 for line in page.children)
    ordered = SimpleSorting(use_columns=False).predict(page)
    assert len(ordered.children) == 2
    assert all(line.type == "line" for line in ordered.children)
    assert min(y for _, y in ordered.children[0].polygon) < min(y for _, y in ordered.children[1].polygon)


def test_default_preset_uses_registry_bundle(monkeypatch, tmp_path):
    weights, config = _artifacts(tmp_path)
    monkeypatch.setattr(
        "manuscript.models.resolve",
        lambda *args, **kwargs: {"weights": weights, "config": config},
    )
    detector = Mask2Former()
    assert Path(detector.weights) == weights
    assert Path(detector.config_path) == config


def test_raw_preserves_model_tensors_masks_and_holes(monkeypatch, tmp_path):
    weights, config = _artifacts(tmp_path)
    session = _FakeSession()
    monkeypatch.setattr("manuscript.detectors._mask2former.ort.InferenceSession",
                        lambda *args, **kwargs: session)
    model = Mask2Former(weights=weights, config=config)
    image = np.zeros((128, 256, 3), np.uint8)
    raw = model.predict(image, return_raw=True, return_masks=True, return_outputs=True)
    assert set(raw) == {"page", "outputs", "detections", "image_size", "metadata"}
    assert set(raw["outputs"]) == {model.output_class_logits, model.output_mask_logits}
    for row in raw["detections"]:
        assert row["mask"].shape == image.shape[:2]
        assert "contours" in row and "contour_hierarchy" in row
    compact = model.predict(image, return_raw=True)
    assert "outputs" not in compact
    assert all("mask" not in row for row in compact["detections"])
    assert [row["polygon"] for row in compact["detections"]] == [row["polygon"] for row in raw["detections"]]
    masks_only = model.predict(image, return_masks=True)
    assert "outputs" not in masks_only and "mask" in masks_only["detections"][0]
    outputs_only = model.predict(image, return_outputs=True)
    assert "outputs" in outputs_only and all("mask" not in row for row in outputs_only["detections"])
    assert raw["page"].to_tree_dict() == model.predict(image).to_tree_dict()


def test_reordered_inputs_preserve_image_float16(monkeypatch, tmp_path):
    weights, config = _artifacts(tmp_path)

    class Session(_FakeSession):
        def get_inputs(self):
            return [_FakeIO("pixel_mask", "tensor(int64)"),
                    _FakeIO("pixel_values", "tensor(float16)")]

    session = Session()
    monkeypatch.setattr("manuscript.detectors._mask2former.ort.InferenceSession",
                        lambda *args, **kwargs: session)
    model = Mask2Former(weights=weights, config=config)
    model.predict(np.zeros((128, 256, 3), np.uint8))
    assert session.feed["pixel_values"].dtype == np.float16
    assert session.feed["pixel_mask"].dtype == np.int64


def test_multiclass_nms_and_no_object(tmp_path):
    weights, config = _artifacts(tmp_path)
    values = json.loads(config.read_text())
    values['region_classes'] = {'0': {'type': 'line'}, '1': {'type': 'table'}}
    detector = Mask2Former(weights=weights, config=values)
    logits = np.array([[8,-8,-8],[-8,8,-8],[7,-8,-8],[-8,-8,8]],np.float32)
    masks = np.full((4,16,16),-8,np.float32)
    masks[:,4:12,2:14] = 8
    detections = detector._postprocess(logits,masks,(100,100,0,0,1024,1024))
    assert len(detections) == 2
    assert {class_id for _,_,class_id in detections} == {0,1}


def test_tree_serialization_and_line_recognition_levels(tmp_path,monkeypatch):
    import pytest
    from manuscript.data import Page
    from manuscript.api.recognizer import BaseRecognizer
    from manuscript.recognizers._common.region_types import RecognitionPrediction
    weights,config = _artifacts(tmp_path)
    monkeypatch.setattr('manuscript.detectors._mask2former.ort.InferenceSession',lambda *a,**k:_FakeSession())
    detector=Mask2Former(weights=weights,config=config)
    image=np.zeros((400,600,3),np.uint8)
    page=detector.predict(image)
    path=tmp_path/'page.json'
    page.save(path)
    assert Page.load(path).to_tree_dict() == page.to_tree_dict()
    class Recognizer(BaseRecognizer):
        def _initialize_session(self): pass
        def _predict_text_images(self,regions,batch_size=None):
            return [RecognitionPrediction(text='line text',confidence=.9) for _ in regions]
    recognizer=Recognizer(weights=weights,recognition_levels=['word'])
    with pytest.warns(UserWarning,match='recognition skipped'):
        untouched=recognizer.predict(page,image)
    assert all(line.text is None for line in untouched.children)
    recognizer=Recognizer(weights=weights,recognition_levels=['line'])
    result=recognizer.predict(page,image)
    assert all(line.text == 'line text' for line in result.children)
    assert all(line.text is None for line in page.children)


def test_create_config_save_and_reuse(tmp_path):
    import pytest
    weights, _ = _artifacts(tmp_path)
    output = tmp_path / 'generated.json'
    config = Mask2Former.create_config(image_size=512,score_threshold=.3,output_path=output)
    assert config['region_classes'] == {'0':{'type':'line'}}
    assert json.loads(output.read_text()) == config
    detector = Mask2Former(weights=weights,config=output)
    assert detector.image_size == 512 and detector.score_threshold == .3
    for kwargs in ({'image_size':0},{'image_std':[0,1,1]}, {'score_threshold':2}):
        with pytest.raises(ValueError):
            Mask2Former.create_config(**kwargs)



@pytest.mark.parametrize("case", ["no_queries", "no_object", "empty_masks", "low_score", "small_area"])
def test_postprocess_empty_and_filtered_regions(tmp_path, case):
    weights, config = _artifacts(tmp_path)
    model = Mask2Former(weights=weights, config=config, image_size=16,
                        mask_interpolation_size=16, min_mask_area=4)
    classes = np.array([[8., -8.]], np.float32)
    masks = np.full((1, 16, 16), -8., np.float32)
    masks[0, 4:12, 4:12] = 8.
    if case == "no_queries": classes, masks = classes[:0], masks[:0]
    elif case == "no_object": classes = classes[:, ::-1]
    elif case == "empty_masks": masks[:] = -8.
    elif case == "low_score": model.score_threshold = 1.
    elif case == "small_area": model.min_mask_area = 100
    assert model._postprocess(classes, masks, (16, 16, 0, 0, 16, 16)) == []


@pytest.mark.parametrize("case", ["class_shape", "mask_shape", "query_count", "nan_class", "inf_mask"])
def test_invalid_model_outputs_fail_explicitly(tmp_path, case):
    weights, config = _artifacts(tmp_path)
    model = Mask2Former(weights=weights, config=config)
    classes = np.zeros((1, 2), np.float32)
    masks = np.zeros((1, 16, 16), np.float32)
    if case == "class_shape": classes = classes[0]
    elif case == "mask_shape": masks = masks[0]
    elif case == "query_count": masks = np.zeros((2, 16, 16), np.float32)
    elif case == "nan_class": classes[0, 0] = np.nan
    elif case == "inf_mask": masks[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="Invalid Mask2Former"):
        model._postprocess(classes, masks, (16, 16, 0, 0, 16, 16))


def test_mask2former_constructor_validates_config(tmp_path):
    from manuscript.detectors._mask2former import Mask2Former
    weights = tmp_path / 'model.onnx'
    weights.write_bytes(b'test')
    config = Mask2Former.create_config()
    config['image_std'] = [0, 0, 0]
    with pytest.raises(ValueError):
        Mask2Former(weights=weights, config=config)


@pytest.mark.parametrize("key,value", [
    ("image_size", 0), ("image_size", True), ("image_mean", [float('nan'), 0, 0]),
    ("image_std", [0, 1, 1]), ("min_mask_area", -1), ("score_threshold", 1.1),
    ("nms_iou_threshold", -0.1), ("mask_logit_threshold", float('nan')),
])
def test_constructor_and_config_builder_reject_invalid_values(tmp_path, key, value):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"model")
    with pytest.raises(ValueError):
        Mask2Former.create_config(**{key: value})
    config = Mask2Former.create_config()
    config[key] = value
    with pytest.raises(ValueError):
        Mask2Former(weights=path, config=config)


@pytest.mark.parametrize("key,value", [("score_threshold", -1), ("min_mask_area", 0),
                                      ("nms_iou_threshold", float('nan'))])
def test_constructor_validates_threshold_overrides(tmp_path, key, value):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"model")
    with pytest.raises(ValueError):
        Mask2Former(weights=path, config=Mask2Former.create_config(), **{key: value})
