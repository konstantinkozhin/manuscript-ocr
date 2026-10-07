import numpy as np
import pytest

from manuscript.api.recognizer import BaseRecognizer
from manuscript.data import Block, Line, Page, TextSpan


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
    return Page(blocks=[Block(lines=[Line(text_spans=[TextSpan(
        polygon=[(2., 2.), (20., 2.), (20., 12.), (2., 12.)],
        detection_confidence=0.9,
    )])])])


def test_prediction_preserves_source_and_passes_batch_size(recognizer, page):
    image = np.zeros((25, 30, 3), np.uint8)
    result = recognizer.predict(page, image)
    assert recognizer.calls[0][1] == 7
    assert result.blocks[0].lines[0].text_spans[0].text == "test"
    assert page.blocks[0].lines[0].text_spans[0].text is None
    recognizer.predict(page, image, batch_size=2)
    assert recognizer.calls[1][1] == 2


def test_missing_image_and_empty_regions_skip_backend(recognizer, page):
    result = recognizer.predict(page)
    assert result == page and result is not page
    recognizer.predict(Page(blocks=[]), np.zeros((25, 30, 3), np.uint8))
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
            return page.model_copy(deep=True)

    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"test")
    model = PageRecognizer(weights=str(weights))
    assert model.predict(page) == page


def test_raw_keeps_details_and_page_contract(recognizer, page):
    recognizer.predictions = [{"text": "test", "confidence": .8, "meta": {"token_ids": [1, 2]}}]
    raw = recognizer.predict(page, np.zeros((25, 30, 3), np.uint8), return_raw=True)
    assert raw["page"].blocks[0].lines[0].text_spans[0].text == "test"
    assert raw["predictions"][0]["metadata"]["token_ids"] == [1, 2]
    assert page.blocks[0].lines[0].text_spans[0].text is None
    assert recognizer.predict(page, return_raw=True)["predictions"] == []


@pytest.mark.parametrize('name,value', [('batch_size', True), ('batch_size', 0), ('batch_size', 1.5),
                                      ('min_text_size', -1), ('rotate_threshold', float('nan'))])
def test_invalid_common_settings_fail(tmp_path, name, value):
    weights = tmp_path / 'model.onnx'
    weights.write_bytes(b'test')
    with pytest.raises(ValueError, match=name):
        CropRecognizer(weights=str(weights), **{name: value})
