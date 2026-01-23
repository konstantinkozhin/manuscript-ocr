import pytest
from copy import deepcopy

from manuscript.data import Word, Line, Block, Page
from manuscript.correctors import CharLM
from manuscript.api.base import BaseModel as BaseCorrector


# ============================================================================
# Helper functions
# ============================================================================


def create_sample_page() -> Page:
    """Create a sample Page for testing."""
    word1 = Word(
        polygon=[(10.0, 20.0), (100.0, 20.0), (100.0, 40.0), (10.0, 40.0)],
        detection_confidence=0.95,
        text="Hello",
        recognition_confidence=0.98,
    )
    word2 = Word(
        polygon=[(110.0, 20.0), (200.0, 20.0), (200.0, 40.0), (110.0, 40.0)],
        detection_confidence=0.92,
        text="World",
        recognition_confidence=0.96,
    )
    line = Line(words=[word1, word2])
    block = Block(lines=[line])
    return Page(blocks=[block])


def create_empty_page() -> Page:
    """Create an empty Page for testing."""
    return Page(blocks=[])


# ============================================================================
# Tests for BaseCorrector
# ============================================================================


class TestBaseCorrector:
    """Tests asserting BaseModel usage as corrector base class."""

    def test_base_model_is_abstract(self):
        """BaseModel (used as corrector base) is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseCorrector()

    def test_subclass_must_implement_predict_and_initialize(self):
        """A subclass implementing required methods can be instantiated and used."""

        class MinimalCorrector(BaseCorrector):
            default_weights_name = None
            
            def __init__(self):
                self.device = "cpu"
                self.weights = None
                self.extra_config = {}
                self.session = None
            
            def _initialize_session(self):
                pass

            def predict(self, page: Page) -> Page:
                return page.model_copy(deep=True)

        corrector = MinimalCorrector()
        page = create_sample_page()

        result = corrector.predict(page)
        assert isinstance(result, Page)

    def test_base_model_call_invokes_predict(self):
        """__call__ should forward to predict()."""

        class CallableCorrector(BaseCorrector):
            default_weights_name = None
            
            def __init__(self):
                self.device = "cpu"
                self.weights = None
                self.extra_config = {}
                self.session = None
                self.call_count = 0
            
            def _initialize_session(self):
                pass

            def predict(self, page: Page) -> Page:
                self.call_count += 1
                return page.model_copy(deep=True)

        corrector = CallableCorrector()
        page = create_sample_page()

        result = corrector(page)

        assert corrector.call_count == 1
        assert isinstance(result, Page)


# ============================================================================
# Tests for CharLM
# ============================================================================


class TestCharLM:
    """Tests for CharLM corrector implementation."""

    def test_charlm_creation(self):
        """CharLM can be instantiated with API-compatible signature."""
        corrector = CharLM(weights=None, device="cpu")
        assert corrector is not None
        assert hasattr(corrector, "device")
        assert hasattr(corrector, "weights")

    def test_charlm_inherits_base(self):
        """CharLM inherits from BaseModel (used as corrector base)."""
        corrector = CharLM(weights=None)
        assert isinstance(corrector, BaseCorrector)

    def test_charlm_returns_page(self):
        """CharLM.predict returns a Page object."""
        corrector = CharLM(weights=None)
        page = create_sample_page()

        result = corrector.predict(page)

        assert isinstance(result, Page)

    def test_charlm_returns_copy(self):
        """CharLM returns a copy, not the same object."""
        corrector = CharLM(weights=None)
        page = create_sample_page()

        result = corrector.predict(page)

        assert result is not page

    def test_charlm_deep_copy(self):
        """CharLM returns a deep copy."""
        corrector = CharLM(weights=None)
        page = create_sample_page()

        result = corrector.predict(page)

        page.blocks[0].lines[0].words[0].text = "Modified"

        assert result.blocks[0].lines[0].words[0].text == "Hello"

    def test_charlm_preserves_text_without_weights(self):
        """CharLM without weights preserves all text content."""
        corrector = CharLM(weights=None)
        page = create_sample_page()

        result = corrector.predict(page)

        assert result.blocks[0].lines[0].words[0].text == "Hello"
        assert result.blocks[0].lines[0].words[1].text == "World"

    def test_charlm_preserves_structure(self):
        """CharLM preserves page structure."""
        corrector = CharLM(weights=None)
        page = create_sample_page()

        result = corrector.predict(page)

        assert len(result.blocks) == len(page.blocks)
        assert len(result.blocks[0].lines) == len(page.blocks[0].lines)
        assert len(result.blocks[0].lines[0].words) == len(page.blocks[0].lines[0].words)

    def test_charlm_preserves_coordinates(self):
        """CharLM preserves polygon coordinates."""
        corrector = CharLM(weights=None)
        page = create_sample_page()

        result = corrector.predict(page)

        original_polygon = page.blocks[0].lines[0].words[0].polygon
        result_polygon = result.blocks[0].lines[0].words[0].polygon
        assert result_polygon == original_polygon

    def test_charlm_preserves_confidence(self):
        """CharLM preserves confidence scores."""
        corrector = CharLM(weights=None)
        page = create_sample_page()

        result = corrector.predict(page)

        orig_word = page.blocks[0].lines[0].words[0]
        result_word = result.blocks[0].lines[0].words[0]
        assert result_word.detection_confidence == orig_word.detection_confidence
        assert result_word.recognition_confidence == orig_word.recognition_confidence

    def test_charlm_empty_page(self):
        """CharLM handles empty page."""
        corrector = CharLM(weights=None)
        page = create_empty_page()

        result = corrector.predict(page)

        assert isinstance(result, Page)
        assert len(result.blocks) == 0

    def test_charlm_callable(self):
        """CharLM can be called directly via __call__ alias."""
        corrector = CharLM(weights=None)
        page = create_sample_page()

        result = corrector(page)

        assert isinstance(result, Page)
        assert result.blocks[0].lines[0].words[0].text == "Hello"

    def test_charlm_multiple_calls(self):
        """CharLM can be called multiple times."""
        corrector = CharLM(weights=None)
        page1 = create_sample_page()
        page2 = create_sample_page()
        page2.blocks[0].lines[0].words[0].text = "Different"

        result1 = corrector.predict(page1)
        result2 = corrector.predict(page2)

        assert result1.blocks[0].lines[0].words[0].text == "Hello"
        assert result2.blocks[0].lines[0].words[0].text == "Different"

    def test_charlm_default_parameters(self):
        """CharLM has sensible default parameters."""
        corrector = CharLM(weights=None)
        
        assert hasattr(corrector, "mask_threshold")
        assert hasattr(corrector, "apply_threshold")
        assert hasattr(corrector, "max_edits")
        assert hasattr(corrector, "min_word_len")
        
    def test_charlm_custom_thresholds(self):
        """CharLM accepts custom threshold parameters."""
        corrector = CharLM(
            weights=None,
            mask_threshold=0.1,
            apply_threshold=0.9,
            max_edits=3,
            min_word_len=3
        )
        
        assert corrector.mask_threshold == 0.1
        assert corrector.apply_threshold == 0.9
        assert corrector.max_edits == 3
        assert corrector.min_word_len == 3


# ============================================================================
# Tests for module imports
# ============================================================================


class TestModuleImports:
    """Tests for module structure and imports."""

    def test_import_from_correctors(self):
        """Can import CharLM from manuscript.correctors."""
        from manuscript.correctors import CharLM

        assert CharLM is not None

    def test_import_from_manuscript(self):
        """Can import CharLM from manuscript root."""
        from manuscript import CharLM

        assert CharLM is not None

    def test_correctors_in_all(self):
        """Correctors are in __all__."""
        import manuscript.correctors

        assert "CharLM" in manuscript.correctors.__all__


# ============================================================================
# Tests for custom corrector implementation
# ============================================================================


class TestCustomCorrector:
    """Tests for implementing custom correctors."""

    def test_custom_corrector_modifies_text(self):
        """Custom corrector can modify word text."""

        class UppercaseCorrector(BaseCorrector):
            default_weights_name = None
            
            def __init__(self):
                self.device = "cpu"
                self.weights = None
                self.extra_config = {}
                self.session = None
            
            def _initialize_session(self):
                pass
            
            def predict(self, page: Page) -> Page:
                result = page.model_copy(deep=True)
                for block in result.blocks:
                    for line in block.lines:
                        for word in line.words:
                            if word.text:
                                word.text = word.text.upper()
                return result

        corrector = UppercaseCorrector()
        page = create_sample_page()

        result = corrector.predict(page)

        assert result.blocks[0].lines[0].words[0].text == "HELLO"
        assert result.blocks[0].lines[0].words[1].text == "WORLD"
        assert page.blocks[0].lines[0].words[0].text == "Hello"

    def test_custom_corrector_uses_context(self):
        """Custom corrector can use word context."""

        class ContextAwareCorrector(BaseCorrector):
            default_weights_name = None
            
            def __init__(self):
                self.device = "cpu"
                self.weights = None
                self.extra_config = {}
                self.session = None
            
            def _initialize_session(self):
                pass
            
            def predict(self, page: Page) -> Page:
                result = page.model_copy(deep=True)
                for block in result.blocks:
                    for line in block.lines:
                        texts = [w.text for w in line.words if w.text]
                        if line.words and line.words[0].text:
                            line.words[0].text = f"[{len(texts)} words] {line.words[0].text}"
                return result

        corrector = ContextAwareCorrector()
        page = create_sample_page()

        result = corrector.predict(page)

        assert result.blocks[0].lines[0].words[0].text == "[2 words] Hello"

    def test_custom_corrector_with_kwargs(self):
        """Custom corrector can accept additional kwargs."""

        class ConfigurableCorrector(BaseCorrector):
            default_weights_name = None
            
            def __init__(self, prefix: str = ""):
                self.device = "cpu"
                self.weights = None
                self.extra_config = {}
                self.session = None
                self.prefix = prefix
            
            def _initialize_session(self):
                pass
            
            def predict(self, page: Page) -> Page:
                result = page.model_copy(deep=True)
                for block in result.blocks:
                    for line in block.lines:
                        for word in line.words:
                            if word.text:
                                word.text = self.prefix + word.text
                return result

        corrector = ConfigurableCorrector(prefix=">>")
        page = create_sample_page()

        result = corrector.predict(page)

        assert result.blocks[0].lines[0].words[0].text == ">>Hello"


# ============================================================================
# Tests for CharLM specific functionality
# ============================================================================


class TestCharLMAdvanced:
    """Advanced tests for CharLM corrector."""

    def test_charlm_default_weights_name(self):
        """CharLM has default_weights_name attribute."""
        assert hasattr(CharLM, "default_weights_name")
        assert CharLM.default_weights_name == "prereform_charlm_g1"

    def test_charlm_pretrained_registry(self):
        """CharLM has pretrained_registry with available presets."""
        assert hasattr(CharLM, "pretrained_registry")
        assert "prereform_charlm_g1" in CharLM.pretrained_registry
        assert "modern_charlm_g1" in CharLM.pretrained_registry

    def test_charlm_vocab_registry(self):
        """CharLM has vocab_registry."""
        assert hasattr(CharLM, "vocab_registry")
        assert "prereform_charlm_g1" in CharLM.vocab_registry
        assert "modern_charlm_g1" in CharLM.vocab_registry

    def test_charlm_lexicon_registry(self):
        """CharLM has lexicon_registry."""
        assert hasattr(CharLM, "lexicon_registry")
        assert "prereform_words" in CharLM.lexicon_registry
        assert "modern_words" in CharLM.lexicon_registry

    def test_charlm_without_weights_returns_copy(self):
        """CharLM without weights returns a deep copy of the page."""
        corrector = CharLM(weights=None)
        page = create_sample_page()

        result = corrector.predict(page)

        assert result is not page
        assert result.blocks[0].lines[0].words[0].text == page.blocks[0].lines[0].words[0].text

    def test_charlm_accepts_lexicon_as_set(self):
        """CharLM accepts lexicon as a set."""
        lexicon = {"hello", "world"}
        corrector = CharLM(weights=None, lexicon=lexicon)
        
        assert corrector.lexicon is not None
        assert "hello" in corrector.lexicon
        assert "world" in corrector.lexicon

    def test_charlm_max_len_parameter(self):
        """CharLM accepts max_len parameter."""
        corrector = CharLM(weights=None, max_len=64)
        
        assert corrector.max_len == 64

    def test_charlm_handles_multiple_blocks(self):
        """CharLM handles pages with multiple blocks."""
        corrector = CharLM(weights=None)
        
        word1 = Word(
            polygon=[(10.0, 20.0), (100.0, 20.0), (100.0, 40.0), (10.0, 40.0)],
            detection_confidence=0.95,
            text="Block1",
            recognition_confidence=0.98,
        )
        word2 = Word(
            polygon=[(10.0, 60.0), (100.0, 60.0), (100.0, 80.0), (10.0, 80.0)],
            detection_confidence=0.92,
            text="Block2",
            recognition_confidence=0.96,
        )
        line1 = Line(words=[word1])
        line2 = Line(words=[word2])
        block1 = Block(lines=[line1])
        block2 = Block(lines=[line2])
        page = Page(blocks=[block1, block2])

        result = corrector.predict(page)

        assert len(result.blocks) == 2
        assert result.blocks[0].lines[0].words[0].text == "Block1"
        assert result.blocks[1].lines[0].words[0].text == "Block2"

    def test_charlm_handles_multiple_lines(self):
        """CharLM handles blocks with multiple lines."""
        corrector = CharLM(weights=None)
        
        word1 = Word(
            polygon=[(10.0, 20.0), (100.0, 20.0), (100.0, 40.0), (10.0, 40.0)],
            detection_confidence=0.95,
            text="Line1",
            recognition_confidence=0.98,
        )
        word2 = Word(
            polygon=[(10.0, 50.0), (100.0, 50.0), (100.0, 70.0), (10.0, 70.0)],
            detection_confidence=0.92,
            text="Line2",
            recognition_confidence=0.96,
        )
        line1 = Line(words=[word1])
        line2 = Line(words=[word2])
        block = Block(lines=[line1, line2])
        page = Page(blocks=[block])

        result = corrector.predict(page)

        assert len(result.blocks[0].lines) == 2
        assert result.blocks[0].lines[0].words[0].text == "Line1"
        assert result.blocks[0].lines[1].words[0].text == "Line2"

    def test_charlm_handles_words_without_text(self):
        """CharLM handles words with None text."""
        corrector = CharLM(weights=None)
        
        word = Word(
            polygon=[(10.0, 20.0), (100.0, 20.0), (100.0, 40.0), (10.0, 40.0)],
            detection_confidence=0.95,
            text=None,
            recognition_confidence=None,
        )
        line = Line(words=[word])
        block = Block(lines=[line])
        page = Page(blocks=[block])

        result = corrector.predict(page)

        assert result.blocks[0].lines[0].words[0].text is None

    def test_charlm_handles_empty_text(self):
        """CharLM handles words with empty text."""
        corrector = CharLM(weights=None)
        
        word = Word(
            polygon=[(10.0, 20.0), (100.0, 20.0), (100.0, 40.0), (10.0, 40.0)],
            detection_confidence=0.95,
            text="",
            recognition_confidence=0.98,
        )
        line = Line(words=[word])
        block = Block(lines=[line])
        page = Page(blocks=[block])

        result = corrector.predict(page)

        assert result.blocks[0].lines[0].words[0].text == ""

    def test_charlm_has_train_method(self):
        """CharLM has static train method."""
        assert hasattr(CharLM, "train")
        assert callable(CharLM.train)

    def test_charlm_has_export_method(self):
        """CharLM has static export method."""
        assert hasattr(CharLM, "export")
        assert callable(CharLM.export)


# ============================================================================
# Tests for create_page_from_text utility
# ============================================================================


class TestCreatePageFromText:
    """Tests for create_page_from_text utility function."""

    def test_create_page_from_text_import(self):
        """Can import create_page_from_text from manuscript.utils."""
        from manuscript.utils import create_page_from_text
        assert create_page_from_text is not None

    def test_create_page_from_text_single_line(self):
        """Creates page from single line."""
        from manuscript.utils import create_page_from_text
        
        page = create_page_from_text(["Hello world"])
        
        assert isinstance(page, Page)
        assert len(page.blocks) == 1
        assert len(page.blocks[0].lines) == 1
        assert len(page.blocks[0].lines[0].words) == 2
        assert page.blocks[0].lines[0].words[0].text == "Hello"
        assert page.blocks[0].lines[0].words[1].text == "world"

    def test_create_page_from_text_multiple_lines(self):
        """Creates page from multiple lines."""
        from manuscript.utils import create_page_from_text
        
        page = create_page_from_text(["Line one", "Line two", "Line three"])
        
        assert len(page.blocks[0].lines) == 3
        assert page.blocks[0].lines[0].words[0].text == "Line"
        assert page.blocks[0].lines[1].words[0].text == "Line"
        assert page.blocks[0].lines[2].words[0].text == "Line"

    def test_create_page_from_text_empty_lines(self):
        """Handles empty lines correctly."""
        from manuscript.utils import create_page_from_text
        
        page = create_page_from_text(["Hello", "", "World"])
        
        # Empty lines are skipped
        assert len(page.blocks[0].lines) == 2

    def test_create_page_from_text_default_confidence(self):
        """Uses default confidence of 1.0."""
        from manuscript.utils import create_page_from_text
        
        page = create_page_from_text(["Test"])
        
        word = page.blocks[0].lines[0].words[0]
        assert word.detection_confidence == 1.0
        assert word.recognition_confidence == 1.0

    def test_create_page_from_text_custom_confidence(self):
        """Accepts custom confidence parameter."""
        from manuscript.utils import create_page_from_text
        
        page = create_page_from_text(["Test"], confidence=0.5)
        
        word = page.blocks[0].lines[0].words[0]
        assert word.detection_confidence == 0.5
        assert word.recognition_confidence == 0.5

    def test_create_page_from_text_has_valid_polygons(self):
        """Words have valid polygon coordinates."""
        from manuscript.utils import create_page_from_text
        
        page = create_page_from_text(["Test"])
        
        word = page.blocks[0].lines[0].words[0]
        assert len(word.polygon) == 4
        for x, y in word.polygon:
            assert isinstance(x, float)
            assert isinstance(y, float)

    def test_create_page_from_text_empty_list(self):
        """Handles empty list input."""
        from manuscript.utils import create_page_from_text
        
        page = create_page_from_text([])
        
        assert len(page.blocks[0].lines) == 0

    def test_create_page_from_text_cyrillic(self):
        """Works with Cyrillic text."""
        from manuscript.utils import create_page_from_text
        
        page = create_page_from_text(["Привет мир"])
        
        assert page.blocks[0].lines[0].words[0].text == "Привет"
        assert page.blocks[0].lines[0].words[1].text == "мир"

    def test_create_page_from_text_with_charlm(self):
        """Integration test: create_page_from_text works with CharLM."""
        from manuscript.utils import create_page_from_text
        
        page = create_page_from_text(["Hello world"])
        corrector = CharLM(weights=None)
        
        result = corrector.predict(page)
        
        assert isinstance(result, Page)
        assert result.blocks[0].lines[0].words[0].text == "Hello"
        assert result.blocks[0].lines[0].words[1].text == "world"
