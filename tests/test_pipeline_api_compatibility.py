from typing import Any, Dict, List, Union

import numpy as np
import pytest
from PIL import Image

import manuscript._pipeline as pipeline_module
from manuscript import Pipeline
from manuscript.data import Block, Line, Page, Word


def _make_test_page() -> Page:
    # Intentionally shuffled order by x to verify layout stage behavior.
    words = [
        Word(
            polygon=[(210.0, 10.0), (300.0, 10.0), (300.0, 50.0), (210.0, 50.0)],
            detection_confidence=0.88,
            order=2,
        ),
        Word(
            polygon=[(10.0, 10.0), (100.0, 10.0), (100.0, 50.0), (10.0, 50.0)],
            detection_confidence=0.95,
            order=0,
        ),
        Word(
            polygon=[(110.0, 10.0), (200.0, 10.0), (200.0, 50.0), (110.0, 50.0)],
            detection_confidence=0.92,
            order=1,
        ),
    ]
    return Page(blocks=[Block(lines=[Line(words=words, order=0)], order=0)])


class DummyDetector:
    def __init__(self, events: List[str] = None):
        self.events = events

    def predict(
        self,
        img_or_path: Union[str, np.ndarray, Image.Image],
        return_maps: bool = False,
    ) -> Dict[str, Any]:
        if self.events is not None:
            self.events.append("detector")
        page = _make_test_page()
        return {
            "page": page,
            "score_map": None if not return_maps else np.zeros((10, 10)),
            "geo_map": None if not return_maps else np.zeros((10, 10, 5)),
        }


class DummyLayout:
    def __init__(self, events: List[str] = None):
        self.events = events
        self.call_count = 0
        self.last_image = None

    def predict(self, page: Page, image: np.ndarray = None) -> Page:
        if self.events is not None:
            self.events.append("layout")
        self.call_count += 1
        self.last_image = image

        result = page.model_copy(deep=True)
        for block in result.blocks:
            for line in block.lines:
                line.words.sort(key=lambda w: min(p[0] for p in w.polygon))
                for idx, word in enumerate(line.words):
                    word.order = idx
        return result


class DummyRecognizer:
    def __init__(self, events: List[str] = None, min_text_size: int = 5):
        self.events = events
        self.call_count = 0
        self.last_image = None
        self.last_min_text_size = None
        self.last_orders = None
        self.min_text_size = min_text_size

    def predict(
        self,
        page: Page,
        image: np.ndarray = None,
        batch_size: int = 32,
    ) -> Page:
        if self.events is not None:
            self.events.append("recognizer")
        self.call_count += 1
        self.last_image = image
        self.last_min_text_size = self.min_text_size

        result = page.model_copy(deep=True)
        self.last_orders = [
            word.order
            for block in result.blocks
            for line in block.lines
            for word in line.words
        ]

        recognized_idx = 0
        for block in result.blocks:
            for line in block.lines:
                for word in line.words:
                    poly = np.array(word.polygon, dtype=np.float32)
                    x_min, y_min = np.min(poly, axis=0)
                    x_max, y_max = np.max(poly, axis=0)
                    width = x_max - x_min
                    height = y_max - y_min

                    if width < self.min_text_size or height < self.min_text_size:
                        continue

                    recognized_idx += 1
                    word.text = f"word{recognized_idx}"
                    word.recognition_confidence = 0.9 - (recognized_idx - 1) * 0.05

        return result


class DummyRecognizerNoExtraArgs:
    def __init__(self):
        self.call_count = 0
        self.last_image = None

    def predict(self, page: Page, image: np.ndarray = None) -> Page:
        self.call_count += 1
        self.last_image = image
        result = page.model_copy(deep=True)
        for block in result.blocks:
            for line in block.lines:
                for idx, word in enumerate(line.words, start=1):
                    word.text = f"r{idx}"
                    word.recognition_confidence = 0.8
        return result


class DummyCorrector:
    def __init__(self, events: List[str] = None):
        self.events = events
        self.call_count = 0
        self.last_image = None

    def predict(self, page: Page, image: np.ndarray = None) -> Page:
        if self.events is not None:
            self.events.append("corrector")
        self.call_count += 1
        self.last_image = image

        result = page.model_copy(deep=True)
        for block in result.blocks:
            for line in block.lines:
                for word in line.words:
                    if word.text:
                        word.text = f"{word.text}_corr"
        return result


class TestPipelineAPICompatibility:
    def test_pipeline_basic_usage(self):
        detector = DummyDetector()
        layout = DummyLayout()
        recognizer = DummyRecognizer()
        pipeline = Pipeline(detector=detector, layout=layout, recognizer=recognizer)

        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result = pipeline.predict(img)

        assert isinstance(result, dict)
        assert "page" in result
        page = result["page"]
        assert isinstance(page, Page)

        words = page.blocks[0].lines[0].words
        assert [w.text for w in words] == ["word1", "word2", "word3"]
        assert [w.order for w in words] == [0, 1, 2]
        assert recognizer.last_orders == [0, 1, 2]

    def test_pipeline_returns_dict_structure(self):
        pipeline = Pipeline(
            detector=DummyDetector(),
            layout=DummyLayout(),
            recognizer=DummyRecognizer(),
        )
        result = pipeline.predict(np.zeros((80, 120, 3), dtype=np.uint8))
        assert isinstance(result, dict)
        assert isinstance(result["page"], Page)

    def test_pipeline_without_recognizer(self):
        layout = DummyLayout()
        pipeline = Pipeline(
            detector=DummyDetector(),
            layout=layout,
            recognizer=None,
        )

        result = pipeline.predict(np.zeros((100, 400, 3), dtype=np.uint8))
        page = result["page"]
        words = page.blocks[0].lines[0].words

        assert layout.call_count == 1
        assert all(word.text is None for word in words)
        assert pipeline.last_recognition_page is None
        assert pipeline.last_layout_page is not None

    def test_pipeline_without_layout(self):
        recognizer = DummyRecognizer()
        pipeline = Pipeline(
            detector=DummyDetector(),
            layout=None,
            recognizer=recognizer,
        )

        result = pipeline.predict(np.zeros((100, 400, 3), dtype=np.uint8))
        words = result["page"].blocks[0].lines[0].words
        assert recognizer.call_count == 1
        assert [w.text for w in words] == ["word1", "word2", "word3"]
        assert pipeline.last_layout_page is None

    def test_pipeline_with_corrector(self):
        corrector = DummyCorrector()
        pipeline = Pipeline(
            detector=DummyDetector(),
            layout=DummyLayout(),
            recognizer=DummyRecognizer(),
            corrector=corrector,
        )

        result = pipeline.predict(np.zeros((100, 400, 3), dtype=np.uint8))
        words = result["page"].blocks[0].lines[0].words
        assert [w.text for w in words] == ["word1_corr", "word2_corr", "word3_corr"]
        assert corrector.call_count == 1
        assert pipeline.last_correction_page is not None

    def test_pipeline_with_visualization(self):
        pipeline = Pipeline(
            detector=DummyDetector(),
            layout=DummyLayout(),
            recognizer=DummyRecognizer(),
        )

        result, vis_img = pipeline.predict(
            np.zeros((100, 400, 3), dtype=np.uint8), vis=True
        )
        assert isinstance(result, dict)
        assert isinstance(result["page"], Page)
        assert isinstance(vis_img, Image.Image)

    def test_pipeline_profile_mode(self, capsys):
        pipeline = Pipeline(
            detector=DummyDetector(),
            layout=DummyLayout(),
            recognizer=DummyRecognizer(),
            corrector=DummyCorrector(),
        )

        _ = pipeline.predict(np.zeros((100, 400, 3), dtype=np.uint8), profile=True)
        out = capsys.readouterr().out
        assert "Detection:" in out
        assert "Recognition:" in out
        assert "Correction:" in out
        assert "Pipeline total:" in out

    def test_pipeline_get_text(self):
        pipeline = Pipeline(
            detector=DummyDetector(),
            layout=DummyLayout(),
            recognizer=DummyRecognizer(),
        )
        result = pipeline.predict(np.zeros((100, 400, 3), dtype=np.uint8))
        text = pipeline.get_text(result["page"])
        assert "word1" in text
        assert "word2" in text
        assert "word3" in text

    def test_pipeline_min_text_size_filtering(self):
        class SmallBoxDetector(DummyDetector):
            def predict(self, img_or_path, return_maps=False):
                words = [
                    Word(
                        polygon=[(10.0, 10.0), (12.0, 10.0), (12.0, 12.0), (10.0, 12.0)],
                        detection_confidence=0.95,
                        order=0,
                    )
                ]
                page = Page(blocks=[Block(lines=[Line(words=words, order=0)], order=0)])
                return {"page": page}

        recognizer = DummyRecognizer(min_text_size=5)
        pipeline = Pipeline(
            detector=SmallBoxDetector(),
            layout=DummyLayout(),
            recognizer=recognizer,
        )

        result = pipeline.predict(np.zeros((100, 400, 3), dtype=np.uint8))
        assert recognizer.call_count == 1
        assert result["page"].blocks[0].lines[0].words[0].text is None
        assert recognizer.last_min_text_size == 5

    def test_pipeline_stage_image_forwarding(self):
        layout = DummyLayout()
        recognizer = DummyRecognizer()
        corrector = DummyCorrector()
        pipeline = Pipeline(
            detector=DummyDetector(),
            layout=layout,
            recognizer=recognizer,
            corrector=corrector,
        )

        image = np.zeros((40, 120, 3), dtype=np.uint8)
        _ = pipeline.predict(image)

        assert isinstance(layout.last_image, np.ndarray)
        assert isinstance(recognizer.last_image, np.ndarray)
        assert isinstance(corrector.last_image, np.ndarray)

    def test_pipeline_recognizer_without_batch_min_text_signature(self):
        recognizer = DummyRecognizerNoExtraArgs()
        pipeline = Pipeline(
            detector=DummyDetector(),
            layout=DummyLayout(),
            recognizer=recognizer,
        )
        result = pipeline.predict(np.zeros((60, 200, 3), dtype=np.uint8))
        words = result["page"].blocks[0].lines[0].words
        assert recognizer.call_count == 1
        assert [w.text for w in words] == ["r1", "r2", "r3"]

    def test_pipeline_last_pages_when_stage_skipped(self):
        pipeline = Pipeline(
            detector=DummyDetector(),
            layout=None,
            recognizer=None,
            corrector=None,
        )
        _ = pipeline.predict(np.zeros((100, 200, 3), dtype=np.uint8))
        assert pipeline.last_detection_page is not None
        assert pipeline.last_layout_page is None
        assert pipeline.last_recognition_page is None
        assert pipeline.last_correction_page is None


class TestLayoutAfterRouting:
    def test_layout_after_detector(self):
        events: List[str] = []
        pipeline = Pipeline(
            detector=DummyDetector(events=events),
            layout=DummyLayout(events=events),
            recognizer=DummyRecognizer(events=events),
            corrector=DummyCorrector(events=events),
            layout_after="detector",
        )
        _ = pipeline.predict(np.zeros((100, 200, 3), dtype=np.uint8))
        assert events == ["detector", "layout", "recognizer", "corrector"]

    def test_layout_after_recognizer(self):
        events: List[str] = []
        pipeline = Pipeline(
            detector=DummyDetector(events=events),
            layout=DummyLayout(events=events),
            recognizer=DummyRecognizer(events=events),
            corrector=DummyCorrector(events=events),
            layout_after="recognizer",
        )
        _ = pipeline.predict(np.zeros((100, 200, 3), dtype=np.uint8))
        assert events == ["detector", "recognizer", "layout", "corrector"]

    def test_layout_after_corrector(self):
        events: List[str] = []
        pipeline = Pipeline(
            detector=DummyDetector(events=events),
            layout=DummyLayout(events=events),
            recognizer=DummyRecognizer(events=events),
            corrector=DummyCorrector(events=events),
            layout_after="corrector",
        )
        _ = pipeline.predict(np.zeros((100, 200, 3), dtype=np.uint8))
        assert events == ["detector", "recognizer", "corrector", "layout"]

    def test_layout_after_recognizer_when_recognizer_is_none(self):
        events: List[str] = []
        pipeline = Pipeline(
            detector=DummyDetector(events=events),
            layout=DummyLayout(events=events),
            recognizer=None,
            corrector=DummyCorrector(events=events),
            layout_after="recognizer",
        )
        _ = pipeline.predict(np.zeros((100, 200, 3), dtype=np.uint8))
        assert events == ["detector", "layout", "corrector"]

    def test_layout_after_corrector_when_corrector_is_none(self):
        events: List[str] = []
        pipeline = Pipeline(
            detector=DummyDetector(events=events),
            layout=DummyLayout(events=events),
            recognizer=DummyRecognizer(events=events),
            corrector=None,
            layout_after="corrector",
        )
        _ = pipeline.predict(np.zeros((100, 200, 3), dtype=np.uint8))
        assert events == ["detector", "recognizer", "layout"]

    def test_invalid_layout_after(self):
        with pytest.raises(ValueError):
            Pipeline(
                detector=DummyDetector(),
                layout=DummyLayout(),
                recognizer=DummyRecognizer(),
                layout_after="unknown",
            )


class TestPipelineInitialization:
    def test_detector_cannot_be_none(self):
        with pytest.raises(ValueError):
            Pipeline(detector=None)

    def test_pipeline_default_initialization(self, monkeypatch):
        class FakeEAST:
            def predict(self, image, return_maps=False):
                return {"page": _make_test_page()}

        class FakeLayout:
            def predict(self, page, image=None):
                return page

        class FakeTRBA:
            def predict(self, page, image=None):
                return page

        monkeypatch.setattr(pipeline_module, "EAST", FakeEAST)
        monkeypatch.setattr(pipeline_module, "SimpleSorting", FakeLayout)
        monkeypatch.setattr(pipeline_module, "TRBA", FakeTRBA)

        pipeline = Pipeline()
        assert isinstance(pipeline.detector, FakeEAST)
        assert isinstance(pipeline.layout, FakeLayout)
        assert isinstance(pipeline.recognizer, FakeTRBA)
        assert pipeline.corrector is None
        assert pipeline.layout_after == "detector"

    def test_pipeline_partial_initialization(self, monkeypatch):
        class FakeEAST:
            def predict(self, image, return_maps=False):
                return {"page": _make_test_page()}

        class FakeLayout:
            def predict(self, page, image=None):
                return page

        class FakeTRBA:
            def predict(self, page, image=None):
                return page

        monkeypatch.setattr(pipeline_module, "EAST", FakeEAST)
        monkeypatch.setattr(pipeline_module, "SimpleSorting", FakeLayout)
        monkeypatch.setattr(pipeline_module, "TRBA", FakeTRBA)

        custom_detector = DummyDetector()
        pipeline1 = Pipeline(detector=custom_detector)
        assert pipeline1.detector is custom_detector
        assert isinstance(pipeline1.layout, FakeLayout)
        assert isinstance(pipeline1.recognizer, FakeTRBA)

        custom_recognizer = DummyRecognizer()
        pipeline2 = Pipeline(recognizer=custom_recognizer)
        assert isinstance(pipeline2.detector, FakeEAST)
        assert isinstance(pipeline2.layout, FakeLayout)
        assert pipeline2.recognizer is custom_recognizer
