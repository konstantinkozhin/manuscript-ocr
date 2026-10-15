from pathlib import Path

import numpy as np
import pytest

from manuscript.api.detector import BaseDetector
from manuscript.data import Page, TextRegion, Table
from manuscript.detectors import YOLO


class _FakeIO:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.type = "tensor(float)"


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
    assert YOLO.default_weights_name == "yolo26x_obb_text_g1"
    assert YOLO.registry_model_class == "YOLO"


def test_yolo_uses_default_preset_when_weights_missing(monkeypatch, tmp_path):
    model_path = tmp_path / "preset.onnx"
    model_path.write_bytes(b"fake")
    model_path.with_suffix(".yaml").write_text("region_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")
    yaml_path = tmp_path / "downloaded-default.yaml"
    yaml_path.write_text("imgsz: 1280\nregion_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")

    monkeypatch.setattr(
        "manuscript.models.resolve",
        lambda *args, **kwargs: {'weights': model_path, 'config': yaml_path},
    )

    detector = YOLO(weights=None)

    assert Path(detector.weights) == model_path


def test_yolo_default_preset_uses_1024_target_size(monkeypatch, tmp_path):
    model_path = tmp_path / "preset.onnx"
    model_path.write_bytes(b"fake")
    model_path.with_suffix(".yaml").write_text("region_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")
    yaml_path = tmp_path / "downloaded-default.yaml"
    yaml_path.write_text("imgsz: 1024\nregion_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")

    monkeypatch.setattr(
        "manuscript.models.resolve",
        lambda *args, **kwargs: {'weights': model_path, 'config': yaml_path},
    )

    detector = YOLO(weights=None)

    assert detector.target_size == 1024


def test_yolo_defaults_to_01_score_thresh(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")
    model_path.with_suffix(".yaml").write_text("region_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")

    detector = YOLO(weights=str(model_path), config=model_path.with_suffix(".yaml"))

    assert detector.score_thresh == 0.1


def test_yolo26x_preset_uses_1024_target_size(monkeypatch, tmp_path):
    model_path = tmp_path / "preset.onnx"
    model_path.write_bytes(b"fake")
    model_path.with_suffix(".yaml").write_text("region_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")
    yaml_path = tmp_path / "downloaded-x.yaml"
    yaml_path.write_text("imgsz: 1024\nregion_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")

    monkeypatch.setattr(
        "manuscript.models.resolve",
        lambda *args, **kwargs: {'weights': model_path, 'config': yaml_path},
    )

    detector = YOLO(weights="yolo26x_obb_text_g1")

    assert detector.target_size == 1024


def test_yolo_reads_local_yaml_imgsz_for_target_size(tmp_path):
    model_path = tmp_path / "custom.raw.onnx"
    model_path.write_bytes(b"fake")
    model_path.with_suffix(".yaml").write_text("region_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")
    model_path.with_suffix(".yaml").write_text("imgsz: [960, 960]\nregion_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")

    detector = YOLO(weights=str(model_path), config=model_path.with_suffix(".yaml"))

    assert detector.target_size == 960


def test_yolo_reads_explicit_config_imgsz_for_target_size(tmp_path):
    model_path = tmp_path / "custom.raw.onnx"
    model_path.write_bytes(b"fake")
    model_path.with_suffix(".yaml").write_text("region_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")
    model_path.with_suffix(".yaml").write_text("imgsz: [960, 960]\nregion_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")

    config_path = tmp_path / "manual.yaml"
    config_path.write_text("imgsz: 768\nregion_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")

    detector = YOLO(weights=str(model_path), config=str(config_path))

    assert Path(detector.config_path) == config_path.resolve()
    assert detector.target_size == 768


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
    model_path.with_suffix(".yaml").write_text("region_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")

    fake_session = _FakeSessionDetect()
    monkeypatch.setattr(
        "manuscript.detectors._yolo.ort.InferenceSession",
        lambda *_args, **_kwargs: fake_session,
    )

    detector = YOLO(
        weights=str(model_path),
        containment_threshold=None,
    config=model_path.with_suffix(".yaml"))

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


def test_yolo_predict_returns_flat_tree_for_obb_by_default(
    monkeypatch,
    tmp_path,
):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")
    model_path.with_suffix(".yaml").write_text("region_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")

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
    config=model_path.with_suffix(".yaml"))

    image = np.zeros((320, 640, 3), dtype=np.uint8)
    page = detector.predict(image)

    assert fake_session.last_inputs is not None
    tensor = fake_session.last_inputs["feed_dict"]["images"]
    assert tensor.shape == (1, 3, 1280, 1280)
    assert tensor.dtype == np.float32

    assert isinstance(page, Page)
    assert len(page.children) == 1
    assert page.children[0].type == "word"
    assert (page.image_width, page.image_height) == (640, 320)
    assert page.children[0].polygon == [
        (150.0, 75.0),
        (250.0, 75.0),
        (250.0, 125.0),
        (150.0, 125.0),
    ]


def test_yolo_predict_returns_rotated_rows_when_requested(monkeypatch, tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")
    model_path.with_suffix(".yaml").write_text("region_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")

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
    config=model_path.with_suffix(".yaml"))

    image = np.zeros((320, 640, 3), dtype=np.uint8)
    page = detector.predict(image)

    assert isinstance(page, Page)
    polygons = page.children[0].polygon
    assert polygons == [
        (150.0, 75.0),
        (250.0, 75.0),
        (250.0, 125.0),
        (150.0, 125.0),
    ]


def test_yolo_supports_legacy_axis_aligned_output_format(monkeypatch, tmp_path):
    model_path = tmp_path / "legacy.onnx"
    model_path.write_bytes(b"fake")
    model_path.with_suffix(".yaml").write_text("region_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")

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
    config=model_path.with_suffix(".yaml"))

    image = np.zeros((320, 640, 3), dtype=np.uint8)
    page = detector.predict(image)

    assert isinstance(page, Page)
    polygon = page.children[0].polygon
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
    model_path.with_suffix(".yaml").write_text("region_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")

    monkeypatch.setattr(
        "manuscript.detectors._yolo.ort.InferenceSession",
        lambda *_args, **_kwargs: _DynamicSession(),
    )

    detector = YOLO(weights=str(model_path), config=model_path.with_suffix(".yaml"))
    detector._initialize_session()

    assert detector.target_size == 1280


def test_yolo_rejects_static_shape_mismatch(monkeypatch, tmp_path):
    model_path = tmp_path / "mismatch.onnx"
    model_path.write_bytes(b"fake")
    model_path.with_suffix(".yaml").write_text("region_classes: {0: {type: word}, 1: {type: table}}\n", encoding="utf-8")

    monkeypatch.setattr(
        "manuscript.detectors._yolo.ort.InferenceSession",
        lambda *_args, **_kwargs: _FakeSessionOBB(),
    )

    detector = YOLO(weights=str(model_path), target_size=640, config=model_path.with_suffix(".yaml"))

    with pytest.raises(ValueError, match="expects input size"):
        detector._initialize_session()


def test_float16_graph_uses_float16_input(monkeypatch, tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"test")
    model_path.with_suffix(".yaml").write_text("region_classes: {0: {type: word}}\n", encoding="utf-8")
    session = _FakeSessionDetect()
    original_inputs = session.get_inputs

    def get_inputs():
        items = original_inputs()
        items[0].type = "tensor(float16)"
        return items

    monkeypatch.setattr(session, "get_inputs", get_inputs)
    monkeypatch.setattr("manuscript.detectors._yolo.ort.InferenceSession", lambda *a, **k: session)
    model = YOLO(weights=model_path, config=model_path.with_suffix(".yaml"))
    model._initialize_session()
    tensor, _, _ = model._preprocess(np.zeros((20, 30, 3), np.uint8))
    assert tensor.dtype == np.float16


def test_yolo_multiclass_tree_and_raw_outputs(monkeypatch, tmp_path):
    model_path = tmp_path / 'model.onnx'
    model_path.write_bytes(b'fake')
    session = _FakeSessionDetect()
    monkeypatch.setattr('manuscript.detectors._yolo.ort.InferenceSession', lambda *a, **k: session)
    detector = YOLO(weights=model_path, config={'region_classes': {
        0: {'type': 'word', 'label': 'handwritten'}, 1: {'type': 'table'}}},
        score_thresh=.25, containment_threshold=None)
    raw = detector.predict(np.zeros((320, 640, 3), dtype=np.uint8), return_raw=True)
    page = raw['page']
    assert isinstance(page.children[0], TextRegion)
    assert page.children[0].label == 'handwritten'
    assert isinstance(page.children[1], Table)
    assert all(child.parent is page for child in page.children)
    assert [d['class_id'] for d in raw['detections']] == [0, 1]
    assert 'outputs' not in raw and 'masks' not in raw
    heavy = detector.predict(np.zeros((320, 640, 3), dtype=np.uint8), return_outputs=True)
    assert 'output0' in heavy['outputs']
    assert Page.from_tree_dict(page.to_tree_dict()).to_dict() == page.to_dict()


@pytest.mark.parametrize('obb', [False, True])
def test_yolo_containment_preserves_different_classes(obb):
    detector = YOLO.__new__(YOLO)
    detector.containment_threshold = .9
    boxes = np.array([[0, 0, 100, 100, .9, 1], [10, 10, 20, 20, .8, 0],
                      [12, 12, 18, 18, .7, 0]], dtype=np.float32)
    if obb:
        from manuscript.utils.geometry import boxes_to_polygons
        rows, polygons = detector._suppress_contained_polygons(boxes, boxes_to_polygons(boxes))
        assert len(polygons) == 2
    else:
        rows = detector._suppress_contained_boxes(boxes)
    assert list(rows[:, 5]) == [1, 0]


def test_yolo_requires_explicit_region_mapping(tmp_path):
    model_path = tmp_path / 'model.onnx'
    model_path.write_bytes(b'fake')
    with pytest.raises(ValueError, match='requires a configuration'):
        YOLO(weights=model_path)
    with pytest.raises(ValueError, match='region_classes'):
        YOLO(weights=model_path, config={'imgsz': 1280})


def test_yolo_empty_page_and_unconfigured_class(monkeypatch, tmp_path):
    model_path = tmp_path / 'model.onnx'
    model_path.write_bytes(b'fake')
    session = _FakeSessionDetect()
    monkeypatch.setattr('manuscript.detectors._yolo.ort.InferenceSession', lambda *a, **k: session)
    detector = YOLO(weights=model_path, config={'region_classes': {0: {'type': 'word'}}}, score_thresh=1)
    page = detector.predict(np.zeros((320,640,3), dtype=np.uint8))
    assert page.children == ()
    assert (page.image_width,page.image_height) == (640,320)
    detector.score_thresh = .25
    with pytest.raises(ValueError, match='Unconfigured detection class'):
        detector.predict(np.zeros((320,640,3), dtype=np.uint8))


def test_yolo_rotated_polygon_preserved_in_tree(monkeypatch, tmp_path):
    model_path = tmp_path / 'model.onnx'
    model_path.write_bytes(b'fake')
    class Rotated(_FakeSessionOBB):
        def run(self, output_names, feed_dict):
            return [np.array([[[400,520,200,100,.9,0,np.pi/4]]], dtype=np.float32)]
    monkeypatch.setattr('manuscript.detectors._yolo.ort.InferenceSession', lambda *a, **k: Rotated())
    detector = YOLO(weights=model_path, config={'region_classes': {0: {'type':'word'}}})
    image = np.zeros((320,640,3), dtype=np.uint8)
    raw = detector.predict(image, return_raw=True)
    polygon = raw['page'].children[0].polygon
    assert len({round(x,3) for x,y in polygon}) == 4
    assert np.allclose(polygon,raw['detections'][0]['polygon'])
    detector.axis_aligned_output=True
    legacy = detector.predict(image, return_raw=True)
    assert len({round(x,3) for x,y in legacy['page'].children[0].polygon}) == 2
    assert np.allclose(legacy['detections'][0]['polygon'],polygon)


def test_yolo_create_json_config_from_training_yaml(tmp_path):
    import json
    source = tmp_path / 'args.yaml'
    source.write_text('imgsz: [1408, 1408]\nmodel: C:\\training\\last.pt\nepochs: 100\n', encoding='utf-8')
    dest = tmp_path / 'model.json'
    config = YOLO.create_config(source, output_path=dest)
    assert config == {'imgsz':1408, 'region_classes':{'0':{'type':'word'}}}
    assert json.loads(dest.read_text()) == config
    weights = tmp_path / 'model.onnx'
    weights.write_bytes(b'fake')
    detector = YOLO(weights=weights, config=weights.with_suffix(".json"))
    assert detector.target_size == 1408
    assert detector.config_path == str(dest.resolve())


def test_yolo_create_config_custom_mapping_and_json_priority(tmp_path):
    weights = tmp_path / 'model.onnx'
    weights.write_bytes(b'fake')
    weights.with_suffix('.yaml').write_text('imgsz: 640\n', encoding='utf-8')
    mapping = {0:{'type':'line'}, 1:{'type':'table'}}
    config = YOLO.create_config({'imgsz':640}, imgsz=1024, region_classes=mapping,
                                output_path=weights.with_suffix('.json'))
    assert YOLO.create_config(weights.with_suffix('.json')) == config
    detector = YOLO(weights=weights, config=weights.with_suffix(".json"))
    assert detector.target_size == 1024
    assert detector.region_classes[1]['type'] == 'table'
    with pytest.warns(UserWarning, match='Unknown element type'):
        config = YOLO.create_config(region_classes={0:{'type':'unknown'}})
    assert config['region_classes']['0']['type'] == 'unknown'
    with pytest.raises(ValueError, match='positive'):
        YOLO.create_config(imgsz=0)


def test_yolo_raw_and_page_agree_after_outside_boxes_are_removed(tmp_path):
    from types import SimpleNamespace
    from manuscript.detectors import YOLO
    weights = tmp_path / 'model.onnx'
    weights.write_bytes(b'test')
    model = YOLO(weights=weights, config={'imgsz': 100, 'region_classes': {0: {'type':'word'}}})
    class Session:
        def get_inputs(self): return [SimpleNamespace(name='images')]
        def run(self, *args): return [np.array([[[200, 200, 220, 220, .9, 0]]], np.float32)]
    model.onnx_session = Session()
    raw = model.predict(np.zeros((100,100,3), np.uint8), return_raw=True)
    assert len(raw['page'].children) == len(raw['detections']), (raw['page'].children, raw['detections'])
