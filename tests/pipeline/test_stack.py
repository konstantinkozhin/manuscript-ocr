import numpy as np
import pytest
from PIL import Image
from manuscript import Stack, Pipeline
from manuscript.data import Page, TextRegion

IMAGE = np.zeros((10, 20, 3), dtype=np.uint8)


class Detector:
    def __init__(self, kind="word", width=20):
        self.page = Page(image_width=width, image_height=10)
        self.page.children = [TextRegion(type=kind)]
        self.flags = None

    def predict(
        self, image, return_raw=False, *, return_masks=False, return_outputs=False
    ):
        self.flags = (return_raw, return_masks, return_outputs)
        if return_raw:
            return {
                "page": self.page,
                "detections": [{"class_id": 0, "kind": self.page.children[0].type}],
                "metadata": {"model": "dummy"},
            }
        return self.page


class PageDetector:
    def predict(self, img_or_path):
        page = Page(image_width=20, image_height=10)
        page.children = [TextRegion(type="line")]
        return page


class Recognizer:
    def __init__(self, text, batch_size):
        self.text = text
        self.batch_size = batch_size
        self.observed = None

    def predict(self, page, image=None, batch_size=None, detector_result=None):
        self.observed = (
            image,
            batch_size if batch_size is not None else self.batch_size,
            detector_result,
        )
        result = page.clone()
        for region in result.children:
            region.text = self.text
        return result


class Corrector:
    def predict(self, page):
        for region in page.children:
            region.text = (region.text or "") + "!"
        return page


def test_accumulation_and_source_isolation():
    a, b = Detector("block"), Detector("line")
    result = Stack([a, b]).predict(IMAGE)
    assert [n.type for n in result.children] == ["block", "line"]
    assert all(n.parent is result for n in result.children)
    assert a.page.children[0].parent is a.page
    assert b.page.children[0].parent is b.page
    result.children[0].text = "changed"
    assert a.page.children[0].text is None


def test_mixed_and_raw_context():
    a, b, r = Detector(), Detector("line"), Recognizer("hello", 7)
    result = Stack([a, b, r, Corrector()]).predict(image=IMAGE, return_raw=True)
    assert [n.text for n in result["page"].children] == ["hello!", "hello!"]
    assert len(result["detections"]) == 2
    assert len(result["detector_results"]) == 2
    assert r.observed[1] == 7
    assert len(r.observed[2]["detections"]) == 2
    assert a.flags == (True, False, False)


def test_nested_and_page_only_detector():
    result = Stack(
        [Detector(), Stack([PageDetector(), Corrector()]), Corrector()]
    ).predict(IMAGE)
    assert [n.type for n in result.children] == ["word", "line"]
    assert [n.text for n in result.children] == ["!!", "!!"]


def test_pipeline_all_slots_preserves_defaults():
    a, b = Recognizer("first", 7), Recognizer("second", 13)
    pipeline = Pipeline(
        detector=Stack([Detector(), PageDetector()]),
        layout=Stack([Corrector()]),
        recognizer=Stack([a, b]),
        corrector=Stack([Corrector()]),
    )
    result = pipeline.predict(IMAGE)["page"]
    assert [n.text for n in result.children] == ["second!", "second!"]
    assert a.observed[1] == 7 and b.observed[1] == 13
    assert len(a.observed[2]["detections"]) == 1
    assert len(pipeline.last_detection_page.children) == 2


def test_page_input_and_batch_override():
    page = Detector().page
    r = Recognizer("text", 7)
    result = Stack([r, Corrector()]).predict(page, IMAGE, batch_size=3)
    assert result.children[0].text == "text!"
    assert page.children[0].text is None
    assert r.observed[1] == 3


def test_flags_propagate_to_nested_detectors():
    d = Detector()
    result = Stack([Stack([d])]).predict(IMAGE, return_masks=True, return_outputs=True)
    assert d.flags == (True, True, True)
    assert isinstance(result["page"], Page)


def test_keyword_only_and_annotation():
    class D:
        def predict(self, *, image):
            return Detector().page

    class R:
        def predict(self, document: Page, /):
            document.children[0].text = "annotated"
            return document

    assert Stack([D(), R()]).predict(IMAGE).children[0].text == "annotated"


def test_load_image_for_page_processors_once(tmp_path):
    path = tmp_path / "input.png"
    Image.fromarray(IMAGE).save(path)
    a, b = Recognizer("a", 7), Recognizer("b", 8)
    Stack([Detector(), a, b]).predict(path)
    assert isinstance(a.observed[0], np.ndarray)
    assert a.observed[0] is b.observed[0]


@pytest.mark.parametrize(
    "stages,kwargs,error",
    [
        ([], {}, ValueError),
        ([object()], {}, TypeError),
        ([Corrector()], {}, ValueError),
        ([Detector()], {}, ValueError),
        ([Detector(), Detector(width=30)], {"image": IMAGE}, ValueError),
    ],
)
def test_validation(stages, kwargs, error):
    with pytest.raises(error):
        Stack(stages).predict(**kwargs)


def test_invalid_returns():
    class InvalidDetector:
        def predict(self, image):
            return {"page": None}

    class InvalidPageStage:
        def predict(self, page):
            return {"page": page}

    with pytest.raises(TypeError):
        Stack([InvalidDetector()]).predict(IMAGE)
    with pytest.raises(TypeError):
        Stack([Detector(), InvalidPageStage()]).predict(IMAGE)


def test_configured_batch_size_overrides_signature_default():
    class R:
        batch_size = 11

        def predict(self, page, batch_size=32):
            assert batch_size == 11
            return page

    Stack([R()]).predict(Detector().page)


def test_parallel_nested_detectors_run_together_and_merge_in_list_order():
    import threading

    barrier = threading.Barrier(3)

    class ConcurrentDetector(Detector):
        def predict(self, image, return_raw=False, **kwargs):
            barrier.wait(timeout=5)
            return super().predict(image, return_raw=return_raw, **kwargs)

    detectors = [ConcurrentDetector(kind) for kind in ("block", "line", "word")]
    result = Stack([detectors[0], Stack(detectors[1:])], parallel=True).predict(IMAGE)
    assert [node.type for node in result.children] == ["block", "line", "word"]
    assert all(
        detector.page.children[0].parent is detector.page for detector in detectors
    )


@pytest.mark.parametrize("nested", [False, True])
def test_parallel_stack_rejects_page_processors(nested):
    stages = [Corrector()]
    if nested:
        stages = [Stack(stages)]
    with pytest.raises(ValueError, match="only image detectors"):
        Stack(stages, parallel=True)
