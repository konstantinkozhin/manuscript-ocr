import numpy as np
import pytest
from manuscript.api.recognizer import BaseRecognizer
from manuscript.data import Page, TextRegion
from manuscript.recognizers._common.region_types import RecognitionPrediction


def test_tree_recognition_levels_and_copy():
    class Recognizer(BaseRecognizer):
        def _initialize_session(self):
            pass
        def _predict_text_images(self, regions, batch_size=None):
            return [RecognitionPrediction(text='recognized', confidence=.8) for _ in regions]
    recognizer = object.__new__(Recognizer)
    recognizer.levels = None
    recognizer.region_preparer = 'bbox'
    recognizer.region_preparer_options = {}
    recognizer.min_text_size = 0
    recognizer.rotate_threshold = None
    recognizer.batch_size = 2
    recognizer.default_debug_save_dir = None
    recognizer._configure_recognition_levels({'recognition_levels':['word']})
    page = Page(image_width=100,image_height=100)
    polygon=[(1,1),(20,1),(20,20),(1,20)]
    line=TextRegion(type='line',polygon=polygon)
    word=TextRegion(type='word',polygon=polygon)
    line.children=(word,);page.children=(line,)
    result=recognizer.predict(page,np.zeros((100,100,3),dtype=np.uint8))
    assert result.children[0].text is None
    output=result.children[0].children[0]
    assert output.text=='recognized' and output.recognition_confidence==.8
    assert output.parent is result.children[0]
    assert word.text is None
    with pytest.raises(ValueError):
        recognizer._validate_levels(['table'])


def make_recognizer(tmp_path, **kwargs):
    class Recognizer(BaseRecognizer):
        def _initialize_session(self):
            pass
        def _predict_text_images(self, regions, batch_size=None):
            return [RecognitionPrediction(text='recognized', confidence=.8) for _ in regions]
    weights = tmp_path / 'model.onnx'
    weights.write_bytes(b'fake')
    return Recognizer(weights=str(weights), min_text_size=0, **kwargs)


def test_recognition_level_mismatch_warns_and_preserves_page(tmp_path):
    recognizer = make_recognizer(tmp_path, recognition_levels=['word'])
    recognizer._configure_recognition_levels({'recognition_levels':['line']})
    assert recognizer.levels == ('word',)
    page = Page(image_width=100, image_height=100)
    page.children = (TextRegion(type='line',polygon=[(1,1),(20,1),(20,20),(1,20)]),)
    with pytest.warns(UserWarning, match='recognition skipped.*line.*recognition_levels'):
        raw = recognizer.predict(page,np.zeros((100,100,3),dtype=np.uint8),return_raw=True)
    assert raw['predictions'] == []
    assert raw['page'].to_tree_dict() == page.to_tree_dict()
    recognizer = make_recognizer(tmp_path, recognition_levels=['line'])
    result = recognizer.predict(page,np.zeros((100,100,3),dtype=np.uint8))
    assert result.children[0].text == 'recognized'


def test_empty_or_filtered_page_does_not_warn_about_levels(tmp_path):
    import warnings
    recognizer = make_recognizer(tmp_path, recognition_levels=['word'])
    page = Page(image_width=100,image_height=100)
    image = np.zeros((100,100,3),dtype=np.uint8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        recognizer.predict(page,image)
        page.children = (TextRegion(type='word',text='already recognized',polygon=[(1,1),(20,1),(20,20),(1,20)]),)
        recognizer.predict(page,image)
        page.children = (TextRegion(type='word'),)
        recognizer.predict(page,image)
        page.children = (TextRegion(type='line'),)
        recognizer.predict(page)  # No image means recognition was not requested.
    assert not caught
