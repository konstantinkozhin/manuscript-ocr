import pytest
from copy import deepcopy

from manuscript.data import Word, Line, Block, Page
from manuscript.correctors import DummyCorrector
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
# Tests for DummyCorrector
# ============================================================================


class TestDummyCorrector:
    """Tests for DummyCorrector implementation."""

    def test_dummy_corrector_creation(self):
        """DummyCorrector can be instantiated with API-compatible signature."""
        corrector = DummyCorrector(weights=None, device="cpu")
        assert corrector is not None
        assert hasattr(corrector, "device")
        assert hasattr(corrector, "weights")

    def test_dummy_corrector_inherits_base(self):
        """DummyCorrector inherits from BaseModel (used as corrector base)."""
        corrector = DummyCorrector(weights=None)
        assert isinstance(corrector, BaseCorrector)

    def test_dummy_corrector_returns_page(self):
        """DummyCorrector.predict returns a Page object."""
        corrector = DummyCorrector()
        page = create_sample_page()

        result = corrector.predict(page)

        assert isinstance(result, Page)

    def test_dummy_corrector_returns_copy(self):
        """DummyCorrector returns a copy, not the same object."""
        corrector = DummyCorrector()
        page = create_sample_page()

        result = corrector.predict(page)

        assert result is not page

    def test_dummy_corrector_deep_copy(self):
        """DummyCorrector returns a deep copy."""
        corrector = DummyCorrector()
        page = create_sample_page()

        result = corrector.predict(page)

        page.blocks[0].lines[0].words[0].text = "Modified"

        assert result.blocks[0].lines[0].words[0].text == "Hello"

    def test_dummy_corrector_preserves_text(self):
        """DummyCorrector preserves all text content."""
        corrector = DummyCorrector()
        page = create_sample_page()

        result = corrector.predict(page)

        assert result.blocks[0].lines[0].words[0].text == "Hello"
        assert result.blocks[0].lines[0].words[1].text == "World"

    def test_dummy_corrector_preserves_structure(self):
        """DummyCorrector preserves page structure."""
        corrector = DummyCorrector()
        page = create_sample_page()

        result = corrector.predict(page)

        assert len(result.blocks) == len(page.blocks)
        assert len(result.blocks[0].lines) == len(page.blocks[0].lines)
        assert len(result.blocks[0].lines[0].words) == len(page.blocks[0].lines[0].words)

    def test_dummy_corrector_preserves_coordinates(self):
        """DummyCorrector preserves polygon coordinates."""
        corrector = DummyCorrector()
        page = create_sample_page()

        result = corrector.predict(page)

        original_polygon = page.blocks[0].lines[0].words[0].polygon
        result_polygon = result.blocks[0].lines[0].words[0].polygon
        assert result_polygon == original_polygon

    def test_dummy_corrector_preserves_confidence(self):
        """DummyCorrector preserves confidence scores."""
        corrector = DummyCorrector()
        page = create_sample_page()

        result = corrector.predict(page)

        orig_word = page.blocks[0].lines[0].words[0]
        result_word = result.blocks[0].lines[0].words[0]
        assert result_word.detection_confidence == orig_word.detection_confidence
        assert result_word.recognition_confidence == orig_word.recognition_confidence

    def test_dummy_corrector_empty_page(self):
        """DummyCorrector handles empty page."""
        corrector = DummyCorrector()
        page = create_empty_page()

        result = corrector.predict(page)

        assert isinstance(result, Page)
        assert len(result.blocks) == 0

    def test_dummy_corrector_callable(self):
        """DummyCorrector can be called directly via __call__ alias."""
        corrector = DummyCorrector(weights=None)
        page = create_sample_page()

        result = corrector(page)

        assert isinstance(result, Page)
        assert result.blocks[0].lines[0].words[0].text == "Hello"

    def test_dummy_corrector_multiple_calls(self):
        """DummyCorrector can be called multiple times."""
        corrector = DummyCorrector()
        page1 = create_sample_page()
        page2 = create_sample_page()
        page2.blocks[0].lines[0].words[0].text = "Different"

        result1 = corrector.predict(page1)
        result2 = corrector.predict(page2)

        assert result1.blocks[0].lines[0].words[0].text == "Hello"
        assert result2.blocks[0].lines[0].words[0].text == "Different"

    def test_dummy_corrector_correct_alias(self):
        """If present, DummyCorrector.correct() should be compatible with predict()."""
        corrector = DummyCorrector(weights=None)
        page = create_sample_page()

        if hasattr(corrector, "correct"):
            result = corrector.correct(page)
        else:
            result = corrector.predict(page)

        assert isinstance(result, Page)
        assert result.blocks[0].lines[0].words[0].text == "Hello"


# ============================================================================
# Tests for module imports
# ============================================================================


class TestModuleImports:
    """Tests for module structure and imports."""

    def test_import_from_correctors(self):
        """Can import DummyCorrector from manuscript.correctors."""
        from manuscript.correctors import DummyCorrector

        assert DummyCorrector is not None

    def test_import_from_manuscript(self):
        """Can import DummyCorrector from manuscript root."""
        from manuscript import DummyCorrector

        assert DummyCorrector is not None

    def test_correctors_in_all(self):
        """Correctors are in __all__."""
        import manuscript.correctors

        assert "DummyCorrector" in manuscript.correctors.__all__


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
