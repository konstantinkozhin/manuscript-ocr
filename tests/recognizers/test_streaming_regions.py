from pathlib import Path
import weakref

import numpy as np
import pytest

from manuscript.api.recognizer import BaseRecognizer
from manuscript.data import Page, TextRegion
from manuscript.recognizers._common.region_types import RecognitionPrediction


class StreamingRecognizer(BaseRecognizer):
    def __init__(self, preset, debug_save_dir=None):
        self.region_preparer = preset
        self.region_preparer_options = {'levels': ['word']}
        self.min_text_size = 1
        self.batch_size = 3
        self.default_debug_save_dir = debug_save_dir
        self.levels = ('word',)
        self.crop_refs = []
        self.calls = []

    def _initialize_session(self):
        pass

    def _region_preparation_batch_size(self, batch_size):
        return batch_size

    def _apply_region_rotation(self, crop, region=None):
        self.crop_refs.append(weakref.ref(crop))
        return crop

    def _predict_text_images(self, regions, batch_size=None, return_raw=False):
        self.calls.append((len(regions), sum(ref() is not None for ref in self.crop_refs)))
        return [RecognitionPrediction(str(int(r.image.mean())), .9) for r in regions]


def make_input():
    image = np.zeros((10, 70, 3), np.uint8)
    page = Page(image_width=70, image_height=10)
    for i in range(7):
        image[:, i * 10:(i + 1) * 10] = i * 20
        node = TextRegion(type='word', polygon=[
            (i * 10, 0), ((i + 1) * 10, 0), ((i + 1) * 10, 10), (i * 10, 10)])
        node.parent = page
    return page, image


@pytest.mark.parametrize('preset', ['bbox', 'polygon_mask', 'quad_warp'])
def test_streamed_crops_match_eager_results_without_retaining_page_images(preset):
    page, image = make_input()
    streamed = StreamingRecognizer(preset)
    actual = streamed.predict(page, image=image, return_raw=True)
    eager = StreamingRecognizer(preset)
    eager._region_preparation_batch_size = lambda batch: None
    expected = eager.predict(page, image=image, return_raw=True)
    assert actual['page'].to_dict() == expected['page'].to_dict()
    for a, b in zip(actual['predictions'], expected['predictions']):
        assert a['text'] == b['text'] and a['confidence'] == b['confidence']
        np.testing.assert_array_equal(a['polygon'], b['polygon'])
    assert [n for n, _ in streamed.calls] == [3, 3, 1]
    assert all(live <= 3 for _, live in streamed.calls)
    assert all(ref() is None for ref in streamed.crop_refs)
    assert [node.text for node in page.children] == [None] * 7


def test_custom_preparer_keeps_whole_page_contract():
    page, image = make_input()
    model = StreamingRecognizer('bbox')
    from manuscript.recognizers._common.region_preparers import prepare_bbox_regions
    model.region_preparer = lambda page, image: prepare_bbox_regions(
        page, image, min_text_size=1, rotate_region=model._apply_region_rotation)
    model.predict(page, image=image)
    assert [n for n, _ in model.calls] == [7]


def test_debug_crops_keep_global_file_numbering(tmp_path):
    page, image = make_input()
    model = StreamingRecognizer('bbox', debug_save_dir=tmp_path)
    model.predict(page, image=image)
    assert [n for n, _ in model.calls] == [7]
    assert len(list(Path(tmp_path).glob('*.png'))) == 7


def test_empty_selection_does_not_initialize_model():
    page, image = make_input()
    model = StreamingRecognizer('bbox')
    model.region_preparer_options = {'levels': ['line']}
    model.levels = ('line',)
    def unexpected_initialization(batch_size):
        raise AssertionError('Model must remain unloaded without eligible crops')
    model._region_preparation_batch_size = unexpected_initialization
    with pytest.warns(UserWarning, match='do not match'):
        result = model.predict(page, image=image)
    assert result.to_dict() == page.to_dict()
    assert model.calls == []
