import json
import numpy as np
import pytest
from manuscript.api.detector import BaseDetector
from manuscript.data import Element


class Detector(BaseDetector):
    def _initialize_session(self):
        pass

    def predict(self, image, **kwargs):
        raise NotImplementedError


@pytest.fixture
def detector():
    model = object.__new__(Detector)
    model._configure_region_types({"region_classes": {"0": {"type": "line", "orientation": 90}}})
    return model


@pytest.mark.parametrize("mapping", [None, {}, {"0": {"type": "page"}}])
def test_invalid_region_mapping_fails(mapping):
    with pytest.raises(ValueError):
        Detector._validate_region_types({"region_classes": mapping})


def test_config_requires_explicit_artifact_and_accepts_registry(detector, tmp_path):
    with pytest.raises(ValueError, match="provide config explicitly"):
        detector._load_detector_config()
    assert detector._load_detector_config(required=False) == (None, None)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"region_classes": {"0": {"type": "word"}}}))
    detector._resolved_model_artifacts = {"config": config}
    values, path = detector._load_detector_config()
    assert values["region_classes"]["0"]["type"] == "word"
    assert str(path) == str(config)
    config.write_text('[]')
    with pytest.raises(ValueError, match="must be an object"):
        detector._load_detector_config()


def test_geometry_repair_preserves_region_fields(detector):
    page = detector._tree_from_regions([[[0, 0], [4, 0], [2, 3]]], [.8], [0], (10, 20))
    region = page.children[0]
    assert region.type == "line" and region.orientation == 90
    assert region.detection_confidence == .8 and region.parent is page
    assert len(region.polygon) >= 4
    assert (page.image_height, page.image_width) == (10, 20)


def test_empty_and_unrecoverable_geometry(detector):
    assert not detector._tree_from_regions([], [], [], (10, 20)).children
    assert not detector._tree_from_regions([[[0, 0], [1, 0]]], [.8], [0], (10, 20)).children


def test_inconsistent_detection_arrays_and_unconfigured_class_fail(detector):
    with pytest.raises(ValueError, match="equal lengths"):
        detector._tree_from_regions([], [.8], [0], (10, 20))
    with pytest.raises(ValueError, match="Unconfigured"):
        detector._tree_from_regions([[[0, 0], [4, 0], [2, 3]]], [.8], [5], (10, 20))


def test_unknown_type_is_retained(detector):
    with pytest.warns(UserWarning, match="Unknown element type"):
        detector._configure_region_types({"region_classes": {"0": {"type": "custom_api_region"}}})
    page = detector._tree_from_regions([[[0, 0], [4, 0], [4, 3], [0, 3]]], [.8], [0], (10, 20))
    assert type(page.children[0]) is Element
    assert page.children[0].type == "custom_api_region"


def test_raw_outputs_are_optional(detector):
    page = detector._tree_from_regions([], [], [], (10, 20))
    raw = detector._raw_result(page, None, [], (10, 20), backend="test")
    assert "outputs" not in raw and raw["page"] is page
    outputs = {"scores": np.array([.8])}
    assert detector._raw_result(page, outputs, [], (10, 20))["outputs"] is outputs


def test_mask_details_keep_holes_without_retaining_mask(detector):
    mask = np.ones((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 0
    details = detector._mask_details(mask, include_mask=False)
    assert "mask" not in details and len(details["contours"]) == 2
    assert any(item[3] >= 0 for item in details["contour_hierarchy"])
    assert detector._mask_details(mask)["mask"] is mask


def test_local_detector_does_not_discover_adjacent_config(tmp_path):
    model = object.__new__(Detector)
    model.weights = str(tmp_path / "model.onnx")
    (tmp_path / "model.json").write_text('{}')
    with pytest.raises(ValueError, match="provide config explicitly"):
        model._load_detector_config()
    assert model._load_detector_config({"example": True}) == ({"example": True}, None)



def test_region_mapping_rejects_runtime_geometry_overrides():
    from manuscript.api.detector import BaseDetector
    with pytest.raises(ValueError):
        BaseDetector._validate_region_types({'region_classes': {0: {
            'type': 'word', 'polygon': [(0,0),(1,0),(1,1),(0,1)]}}})


@pytest.mark.parametrize("mapping", [
    {True: {"type": "word"}}, {-1: {"type": "word"}}, {1.5: {"type": "word"}},
    {0: {"type": "word"}, "0": {"type": "line"}},
    {0: {"type": "word", "detection_confidence": .5}}, {0: "word"},
])
def test_region_mapping_rejects_ambiguous_classes(mapping):
    with pytest.raises(ValueError):
        BaseDetector._validate_region_types({"region_classes": mapping})
