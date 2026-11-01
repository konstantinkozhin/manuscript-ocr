from tests.support.tree import make_page, make_region
import numpy as np
import pytest
from manuscript.api.recognizer import BaseRecognizer
from manuscript.data import TextRegion


class CropRecognizer(BaseRecognizer):

    def _initialize_session(self):
        pass

    def _predict_text_images(self, regions, batch_size=None):
        self.calls.append((regions, batch_size))
        return self.predictions


@pytest.fixture
def recognizer(tmp_path):
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"test")
    model = CropRecognizer(weights=str(weights), batch_size=7, rotate_threshold=None)
    model.calls = []
    model.predictions = [{"text": "test", "confidence": 0.8}]
    return model


@pytest.fixture
def page():
    return make_page(
        children=[
            make_region(
                "block",
                children=[
                    make_region(
                        "line",
                        children=[
                            make_region(
                                "word",
                                polygon=[
                                    (2.0, 2.0),
                                    (20.0, 2.0),
                                    (20.0, 12.0),
                                    (2.0, 12.0),
                                ],
                                detection_confidence=0.9,
                            )
                        ],
                    )
                ],
            )
        ]
    )


def test_prediction_preserves_source_and_passes_batch_size(recognizer, page):
    image = np.zeros((25, 30, 3), np.uint8)
    result = recognizer.predict(page, image)
    assert recognizer.calls[0][1] == 7
    assert result.children[0].children[0].children[0].text == "test"
    assert page.children[0].children[0].children[0].text is None
    recognizer.predict(page, image, batch_size=2)
    assert recognizer.calls[1][1] == 2


def test_missing_image_and_empty_regions_skip_backend(recognizer, page):
    result = recognizer.predict(page)
    assert result.to_tree_dict() == page.to_tree_dict() and result is not page
    recognizer.predict(make_page(children=[]), np.zeros((25, 30, 3), np.uint8))
    assert recognizer.calls == []


def test_prediction_count_mismatch_fails(recognizer, page):
    recognizer.predictions = []
    with pytest.raises(ValueError, match="same number"):
        recognizer.predict(page, np.zeros((25, 30, 3), np.uint8))


def test_page_level_override_does_not_require_crop_backend(tmp_path, page):

    class PageRecognizer(BaseRecognizer):

        def _initialize_session(self):
            pass

        def predict(self, page, image=None, **kwargs):
            return page.clone()

    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"test")
    model = PageRecognizer(weights=str(weights))
    assert model.predict(page).to_tree_dict() == page.to_tree_dict()


def test_raw_keeps_details_and_page_contract(recognizer, page):
    recognizer.predictions = [
        {"text": "test", "confidence": 0.8, "meta": {"token_ids": [1, 2]}}
    ]
    raw = recognizer.predict(page, np.zeros((25, 30, 3), np.uint8), return_raw=True)
    assert raw["page"].children[0].children[0].children[0].text == "test"
    assert raw["predictions"][0]["metadata"]["token_ids"] == [1, 2]
    assert page.children[0].children[0].children[0].text is None
    assert recognizer.predict(page, return_raw=True)["predictions"] == []


@pytest.mark.parametrize(
    "name,value",
    [
        ("batch_size", True),
        ("batch_size", 0),
        ("batch_size", 1.5),
        ("min_text_size", -1),
        ("rotate_threshold", float("nan")),
    ],
)
def test_invalid_common_settings_fail(tmp_path, name, value):
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"test")
    with pytest.raises(ValueError, match=name):
        CropRecognizer(weights=str(weights), **{name: value})


@pytest.mark.parametrize("angle", [0, 90, -90, 180, 270])
def test_known_orientation_precedes_shape_heuristic(recognizer, angle):
    crop = np.arange(18, dtype=np.uint8).reshape(6, 3)
    region = TextRegion(type="word", orientation=angle)
    result = recognizer._apply_region_rotation(crop, region=region)
    np.testing.assert_array_equal(result, np.rot90(crop, k=int(angle / 90)))


def test_orientation_can_be_disabled(recognizer):
    recognizer.use_orientation = False
    recognizer.rotate_threshold = None
    crop = np.arange(18, dtype=np.uint8).reshape(6, 3)
    result = recognizer._apply_region_rotation(crop, TextRegion(type="word", orientation=90))
    np.testing.assert_array_equal(result, crop)


def test_builtin_preparer_applies_region_orientation(recognizer):
    recognizer.min_text_size = 0
    page = make_page(children=[make_region("word", orientation=180,
        polygon=[(0, 0), (3, 0), (3, 2), (0, 2)])])
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    regions = recognizer._call_region_preparer(page, image)
    np.testing.assert_array_equal(regions[0].image, np.rot90(image, 2))


def test_arbitrary_orientation_expands_crop(recognizer):
    crop = np.zeros((10, 20, 3), dtype=np.uint8)
    result = recognizer._apply_region_rotation(crop, TextRegion(type="line", orientation=45))
    assert result.shape[0] > 10 and result.shape[1] > 20


def test_multiple_regions_keep_prediction_order_and_confidence(recognizer):
    polygon = [(1, 1), (10, 1), (10, 10), (1, 10)]
    page = make_page(children=[make_region("word", polygon=polygon),
                               make_region("word", polygon=polygon)])
    recognizer.predictions = [{"text": "first", "confidence": .9},
                              {"text": "second", "confidence": .4}]
    result = recognizer.predict(page, np.zeros((20, 20, 3), np.uint8))
    assert [region.text for region in result.children] == ["first", "second"]
    assert [region.recognition_confidence for region in result.children] == [.9, .4]
    assert all(region.parent is result for region in result.children)
    assert all(region.text is None for region in page.children)


@pytest.mark.parametrize("batch_size", [0, True, 1.5])
def test_prediction_rejects_invalid_batch_override(recognizer, page, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        recognizer.predict(page, np.zeros((25, 30, 3), np.uint8), batch_size=batch_size)
    assert not recognizer.calls


def test_prediction_rejects_non_page(recognizer):
    with pytest.raises(TypeError, match="document-tree Page"):
        recognizer.predict({})
