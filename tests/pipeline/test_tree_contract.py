import numpy as np
from manuscript import Pipeline
from manuscript.data import Page, TextRegion
from manuscript.layouts import SimpleSorting


def test_pipeline_tree_snapshots_text_and_raw_forwarding():
    class Detector:
        def predict(self,image,**kwargs):
            page=Page(image_width=100,image_height=100)
            page.children=(TextRegion(type='word',polygon=[(1,1),(20,1),(20,20),(1,20)]),)
            return {'page':page,'detections':[{'class_id':0}]}
    class Recognizer:
        def predict(self,page,image=None,detector_result=None):
            assert detector_result['detections'][0]['class_id']==0
            result=page.clone()
            next(n for n in result.descendants if n.type=='word').text='hello'
            return result
    pipeline=Pipeline(detector=Detector(),layout=SimpleSorting(use_columns=False),recognizer=Recognizer(),corrector=None)
    result=pipeline.predict(np.zeros((100,100,3),dtype=np.uint8))['page']
    assert pipeline.get_text(result)=='hello'
    assert pipeline.last_detection_page.children[0].type=='word'
    assert pipeline.last_detection_page.children[0].text is None
    for snapshot in (pipeline.last_detection_page,pipeline.last_layout_page,pipeline.last_recognition_page):
        assert all(c.parent is n for n in (snapshot,*snapshot.descendants) for c in n.children)
    next(n for n in result.descendants if n.type=='word').text='changed'
    assert pipeline.get_text(pipeline.last_recognition_page)=='hello'


def test_pipeline_warns_once_for_text_on_parent_and_children():
    import pytest

    class Detector:
        def predict(self, image):
            page = Page(image_width=100, image_height=100)
            line = TextRegion(type='line', text='parent')
            line.children = (TextRegion(type='word', text='one'),
                             TextRegion(type='word', text='two'))
            page.children = (line,)
            return page

    pipeline = Pipeline(detector=Detector(), layout=None, recognizer=None, corrector=None)
    with pytest.warns(UserWarning, match='both parent and child') as records:
        result = pipeline.predict(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(records) == 1
    assert result['page'].children[0].text == 'parent'


def test_invalid_detector_results_have_explicit_contract_error():
    import pytest
    class Detector:
        def __init__(self, result): self.result = result
        def predict(self, image): return self.result
    image = np.zeros((10, 10, 3), np.uint8)
    for result in (None, {}, {"page": "invalid"}):
        pipeline = Pipeline(detector=Detector(result), layout=None, recognizer=None, corrector=None)
        with pytest.raises(TypeError, match="Detector must return Page"):
            pipeline.predict(image)
        assert pipeline.last_detection_page is None


def test_input_file_name_is_recorded_without_overwriting_detector_metadata(tmp_path):
    from PIL import Image
    class Detector:
        def __init__(self, file_name): self.file_name = file_name
        def predict(self, image): return Page(image_width=10, image_height=10, file_name=self.file_name)
    source = tmp_path / "page.png"
    Image.new("RGB", (10,10)).save(source)
    for supplied, expected in (("", "page.png"), ("original.tif", "original.tif")):
        pipeline = Pipeline(detector=Detector(supplied), layout=None, recognizer=None, corrector=None)
        assert pipeline.predict(source)["page"].file_name == expected
