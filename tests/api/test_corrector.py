from manuscript.api.corrector import BaseCorrector
from manuscript.data import Page, TextRegion


class Corrector(BaseCorrector):
    def _initialize_session(self):
        pass

    def predict(self, page, image=None):
        result = self._copy_page(page)
        for region in self._text_regions(result):
            region.text = region.text.upper()
        return result


def test_correction_level_override():
    corrector = object.__new__(Corrector)
    corrector._configure_correction_levels(["line", "line"])
    page = Page(image_width=100, image_height=100)
    line = TextRegion(type="line", text="строка")
    line.children = [TextRegion(type="word", text="слово")]
    page.children = [line]
    assert corrector.correction_levels == ("line",)
    assert list(corrector._text_regions(page)) == [line]


def test_invalid_correction_levels():
    import pytest
    corrector = object.__new__(Corrector)
    for levels in [None, [], "word", ["table"]]:
        with pytest.raises(ValueError, match="correction_levels"):
            corrector._configure_correction_levels(levels)


def test_correction_copies_tree_and_ignores_empty_and_other_levels():
    model = object.__new__(Corrector)
    model._configure_correction_levels(["word"])
    page = Page(image_width=10, image_height=10)
    line = TextRegion(type="line", text="line")
    word = TextRegion(type="word", text="word")
    line.children = [word, TextRegion(type="word")]
    page.children = [line]
    result = model.predict(page)
    assert result.children[0].text == "line"
    assert result.children[0].children[0].text == "WORD"
    assert result.children[0].children[1].text is None
    assert word.text == "word"
    assert result.children[0].children[0].parent is result.children[0]


def test_corrector_rejects_non_page():
    import pytest
    with pytest.raises(TypeError, match="document-tree Page"):
        Corrector._copy_page({})
