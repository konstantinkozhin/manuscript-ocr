import numpy as np
from PIL import Image
from typing import List, Dict, Any, Union

from manuscript import Pipeline
from manuscript.data import Word, Line, Block, Page


class DummyDetector:
    def __init__(self):
        pass

    def predict(
        self,
        img_or_path: Union[str, np.ndarray, Image.Image],
        return_maps: bool = False,
        sort_reading_order: bool = True,
    ) -> Dict[str, Any]:
        # Create 3 words in reading order
        words = [
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
            Word(
                polygon=[(210.0, 10.0), (300.0, 10.0), (300.0, 50.0), (210.0, 50.0)],
                detection_confidence=0.88,
                order=2,
            ),
        ]
        line = Line(words=words, order=0)
        
        block = Block(lines=[line], order=0)
        
        page = Page(blocks=[block])

        return {
            "page": page,
            "score_map": None if not return_maps else np.zeros((100, 100)),
            "geo_map": None if not return_maps else np.zeros((100, 100, 5)),
        }


class DummyRecognizer:
    def __init__(self):
        self.call_count = 0

    def predict(
        self, images: List[np.ndarray], batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        self.call_count += 1

        results = []
        for i, img in enumerate(images):
            results.append({
                "text": f"word{i + 1}",
                "confidence": 0.9 - i * 0.05
            })

        return results


class TestPipelineAPICompatibility:
    """Tests for Pipeline compatibility with new BaseModel API"""

    def test_pipeline_basic_usage(self):
        """Test basic Pipeline usage with new API"""
        detector = DummyDetector()
        recognizer = DummyRecognizer()

        pipeline = Pipeline(detector=detector, recognizer=recognizer)

        # Create test image
        img = np.zeros((100, 400, 3), dtype=np.uint8)

        result = pipeline.predict(img, recognize_text=True, vis=False)

        # Check result is dict with "page" key
        assert isinstance(result, dict)
        assert "page" in result
        
        page = result["page"]
        assert isinstance(page, Page)
        assert len(page.blocks) == 1
        assert len(page.blocks[0].lines) == 1
        assert len(page.blocks[0].lines[0].words) == 3

        # Check recognition was performed
        assert page.blocks[0].lines[0].words[0].text == "word1"
        assert page.blocks[0].lines[0].words[1].text == "word2"
        assert page.blocks[0].lines[0].words[2].text == "word3"

    def test_pipeline_returns_dict_structure(self):
        """Test that Pipeline returns consistent dict structure"""
        detector = DummyDetector()
        recognizer = DummyRecognizer()

        pipeline = Pipeline(detector=detector, recognizer=recognizer)

        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result = pipeline.predict(img)

        # New API: always returns dict
        assert isinstance(result, dict)
        assert "page" in result
        assert isinstance(result["page"], Page)

    def test_pipeline_without_recognition(self):
        """Test Pipeline detection-only mode"""
        detector = DummyDetector()
        recognizer = DummyRecognizer()

        pipeline = Pipeline(detector=detector, recognizer=recognizer)

        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result = pipeline.predict(img, recognize_text=False, vis=False)

        # Recognizer should not be called
        assert recognizer.call_count == 0

        # Words should have no text
        page = result["page"]
        assert page.blocks[0].lines[0].words[0].text is None

    def test_pipeline_with_visualization(self):
        """Test Pipeline with visualization output"""
        detector = DummyDetector()
        recognizer = DummyRecognizer()

        pipeline = Pipeline(detector=detector, recognizer=recognizer)

        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result, vis_img = pipeline.predict(img, recognize_text=True, vis=True)

        # Should return tuple when vis=True
        assert isinstance(result, dict)
        assert isinstance(result["page"], Page)
        assert isinstance(vis_img, Image.Image)

    def test_pipeline_get_text(self):
        """Test get_text method with new Line structure"""
        detector = DummyDetector()
        recognizer = DummyRecognizer()

        pipeline = Pipeline(detector=detector, recognizer=recognizer)

        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result = pipeline.predict(img, recognize_text=True, vis=False)

        page = result["page"]
        text = pipeline.get_text(page)

        # Should return combined text from all lines
        assert "word1" in text
        assert "word2" in text
        assert "word3" in text

    def test_pipeline_hierarchical_structure(self):
        """Test that Pipeline preserves Page → Block → Line → Word hierarchy"""
        detector = DummyDetector()
        recognizer = DummyRecognizer()

        pipeline = Pipeline(detector=detector, recognizer=recognizer)

        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result = pipeline.predict(img)

        page = result["page"]
        
        # Check hierarchy
        assert isinstance(page, Page)
        assert len(page.blocks) > 0
        
        block = page.blocks[0]
        assert isinstance(block, Block)
        assert len(block.lines) > 0
        
        line = block.lines[0]
        assert isinstance(line, Line)
        assert len(line.words) > 0
        
        word = line.words[0]
        assert isinstance(word, Word)
        assert word.polygon is not None
        assert word.detection_confidence is not None

    def test_pipeline_word_ordering(self):
        """Test that words maintain their reading order"""
        detector = DummyDetector()
        recognizer = DummyRecognizer()

        pipeline = Pipeline(detector=detector, recognizer=recognizer)

        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result = pipeline.predict(img)

        page = result["page"]
        words = page.blocks[0].lines[0].words
        
        # Check words have order attribute
        assert all(w.order is not None for w in words)
        
        # Check order is sequential
        assert words[0].order == 0
        assert words[1].order == 1
        assert words[2].order == 2

    def test_pipeline_min_text_size_filtering(self):
        """Test filtering by minimum text size"""

        class SmallBoxDetector(DummyDetector):
            """Detector with very small boxes"""

            def predict(self, img_or_path, return_maps=False, sort_reading_order=True):
                # Create words with tiny boxes (smaller than min_text_size)
                words = [
                    Word(
                        polygon=[
                            (10.0, 10.0),
                            (12.0, 10.0),
                            (12.0, 12.0),
                            (10.0, 12.0),
                        ],
                        detection_confidence=0.95,
                        order=0,
                    ),
                ]
                line = Line(words=words, order=0)
                block = Block(lines=[line], order=0)
                page = Page(blocks=[block])
                return {"page": page}

        detector = SmallBoxDetector()
        recognizer = DummyRecognizer()

        # min_text_size = 5 (default)
        pipeline = Pipeline(detector=detector, recognizer=recognizer, min_text_size=5)

        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result = pipeline.predict(img, recognize_text=True, vis=False)

        # Recognizer should not be called because all boxes are filtered out
        assert recognizer.call_count == 0

    def test_pipeline_confidence_preservation(self):
        """Test that confidence scores are properly preserved"""
        detector = DummyDetector()
        recognizer = DummyRecognizer()

        pipeline = Pipeline(detector=detector, recognizer=recognizer)

        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result = pipeline.predict(img)

        page = result["page"]
        words = page.blocks[0].lines[0].words

        # Check detection confidence is preserved
        assert words[0].detection_confidence == 0.95
        assert words[1].detection_confidence == 0.92
        assert words[2].detection_confidence == 0.88

        # Check recognition confidence is added
        assert words[0].recognition_confidence == 0.9
        assert words[1].recognition_confidence == 0.85
        assert words[2].recognition_confidence == 0.8


class TestDummyImplementations:
    """Tests for dummy detector and recognizer implementations"""

    def test_dummy_detector_returns_correct_format(self):
        """Test DummyDetector returns correct dict format"""
        detector = DummyDetector()
        result = detector.predict(np.zeros((100, 100, 3), dtype=np.uint8))

        assert isinstance(result, dict)
        assert "page" in result
        assert isinstance(result["page"], Page)
        assert "score_map" in result
        assert "geo_map" in result

    def test_dummy_detector_creates_hierarchical_structure(self):
        """Test DummyDetector creates proper Page hierarchy"""
        detector = DummyDetector()
        result = detector.predict(np.zeros((100, 100, 3), dtype=np.uint8))

        page = result["page"]
        assert len(page.blocks) == 1
        assert len(page.blocks[0].lines) == 1
        assert len(page.blocks[0].lines[0].words) == 3

    def test_dummy_detector_with_return_maps(self):
        """Test DummyDetector with return_maps=True"""
        detector = DummyDetector()
        result = detector.predict(
            np.zeros((100, 100, 3), dtype=np.uint8),
            return_maps=True
        )

        assert result["score_map"] is not None
        assert result["geo_map"] is not None
        assert isinstance(result["score_map"], np.ndarray)
        assert isinstance(result["geo_map"], np.ndarray)

    def test_dummy_recognizer_returns_correct_format(self):
        """Test DummyRecognizer returns list of dicts"""
        recognizer = DummyRecognizer()
        images = [np.zeros((64, 256, 3), dtype=np.uint8) for _ in range(3)]

        results = recognizer.predict(images)

        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        assert all("text" in r and "confidence" in r for r in results)

    def test_dummy_recognizer_correct_content(self):
        """Test DummyRecognizer returns correct text and confidence"""
        recognizer = DummyRecognizer()
        images = [np.zeros((64, 256, 3), dtype=np.uint8) for _ in range(3)]

        results = recognizer.predict(images)

        assert results[0]["text"] == "word1"
        assert results[0]["confidence"] == 0.9
        assert results[1]["text"] == "word2"
        assert results[1]["confidence"] == 0.85
        assert results[2]["text"] == "word3"
        assert results[2]["confidence"] == 0.8


def test_pipeline_integration_example():
    """
    Test that Pipeline API works as documented.
    This ensures the API is truly universal and user-friendly.
    """
    # Use dummy implementations
    detector = DummyDetector()
    recognizer = DummyRecognizer()

    # Example from documentation
    pipeline = Pipeline(detector, recognizer)

    # Create test image
    img = np.zeros((100, 400, 3), dtype=np.uint8)

    # Full image processing
    result = pipeline.predict(img)

    # Get recognized text
    text = pipeline.get_text(result["page"])

    assert text is not None
    assert len(text) > 0

    # Detailed information for each word
    page = result["page"]
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                assert word.text is not None
                assert word.detection_confidence is not None
                assert word.recognition_confidence is not None


def test_pipeline_default_initialization():
    """
    Test that Pipeline() can be created without parameters.
    Should auto-initialize EAST and TRBA.
    """
    # Create without parameters
    pipeline = Pipeline()

    # Check detector and recognizer are created
    assert pipeline.detector is not None
    assert pipeline.recognizer is not None

    # Check types (should be EAST and TRBA)
    from manuscript.detectors import EAST
    from manuscript.recognizers import TRBA

    assert isinstance(pipeline.detector, EAST)
    assert isinstance(pipeline.recognizer, TRBA)

    # Check default min_text_size
    assert pipeline.min_text_size == 5


def test_pipeline_partial_initialization():
    """
    Test that Pipeline can be created with one parameter,
    the other is initialized by default.
    """
    from manuscript.detectors import EAST
    from manuscript.recognizers import TRBA

    # Only detector
    custom_detector = DummyDetector()
    pipeline1 = Pipeline(detector=custom_detector)
    assert pipeline1.detector is custom_detector
    assert isinstance(pipeline1.recognizer, TRBA)

    # Only recognizer
    custom_recognizer = DummyRecognizer()
    pipeline2 = Pipeline(recognizer=custom_recognizer)
    assert isinstance(pipeline2.detector, EAST)
    assert pipeline2.recognizer is custom_recognizer


class TestPipelineCropIntegration:
    """Integration tests for crop extraction in full pipeline."""

    def test_pipeline_with_vertical_word(self):
        """Test pipeline correctly processes vertical word boxes."""
        
        class VerticalWordDetector:
            """Detector that returns a vertical word box."""
            def predict(self, img, return_maps=False, sort_reading_order=True):
                # Vertical box: width=20, height=100
                words = [
                    Word(
                        polygon=[
                            (10.0, 10.0), (30.0, 10.0), 
                            (30.0, 110.0), (10.0, 110.0)
                        ],
                        detection_confidence=0.95,
                        order=0,
                    ),
                ]
                line = Line(words=words, order=0)
                block = Block(lines=[line], order=0)
                page = Page(blocks=[block])
                return {"page": page}
        
        class ShapeTrackingRecognizer:
            """Recognizer that tracks input shapes."""
            def __init__(self):
                self.received_shapes = []
            
            def predict(self, images, batch_size=32):
                for img in images:
                    self.received_shapes.append(img.shape)
                return [{"text": "test", "confidence": 0.9} for _ in images]
        
        detector = VerticalWordDetector()
        recognizer = ShapeTrackingRecognizer()
        
        pipeline = Pipeline(detector=detector, recognizer=recognizer)
        
        # Create test image large enough for the box
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        
        result = pipeline.predict(img, recognize_text=True)
        
        # Pipeline does not rotate crops; recognizer handles orientation.
        assert len(recognizer.received_shapes) == 1
        received_h, received_w = recognizer.received_shapes[0][:2]
        assert received_h > received_w


