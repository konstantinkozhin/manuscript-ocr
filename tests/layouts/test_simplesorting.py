from manuscript.data import Block, Line, Page, Word
from manuscript.layouts import SimpleSorting
from manuscript.utils import organize_page


def _collect_words(page: Page):
    return [
        word
        for block in page.blocks
        for line in block.lines
        for word in line.words
    ]


def test_simplesorting_empty_page():
    layout = SimpleSorting()
    result = layout.predict(Page(blocks=[]))
    assert len(result.blocks) == 1
    assert len(result.blocks[0].lines) == 1
    assert len(result.blocks[0].lines[0].words) == 0


def test_simplesorting_single_word():
    word = Word(
        polygon=[(10, 20), (100, 20), (100, 40), (10, 40)],
        detection_confidence=0.95,
    )
    page = Page(blocks=[Block(lines=[Line(words=[word], order=0)], order=0)])
    result = SimpleSorting(use_columns=False).predict(page)

    words = _collect_words(result)
    assert len(words) == 1
    assert words[0].order == 0


def test_simplesorting_multiple_words_in_line():
    words = [
        Word(polygon=[(120, 20), (180, 20), (180, 40), (120, 40)], detection_confidence=0.93),
        Word(polygon=[(10, 20), (50, 20), (50, 40), (10, 40)], detection_confidence=0.95),
        Word(polygon=[(60, 20), (110, 20), (110, 40), (60, 40)], detection_confidence=0.97),
    ]
    page = Page(blocks=[Block(lines=[Line(words=words, order=0)], order=0)])
    result = SimpleSorting(use_columns=False).predict(page)

    ordered = result.blocks[0].lines[0].words
    assert [word.order for word in ordered] == [0, 1, 2]
    assert [word.detection_confidence for word in ordered] == [0.95, 0.97, 0.93]


def test_simplesorting_multiple_lines():
    words = [
        Word(polygon=[(10, 20), (50, 20), (50, 40), (10, 40)], detection_confidence=0.95),
        Word(polygon=[(60, 20), (110, 20), (110, 40), (60, 40)], detection_confidence=0.97),
        Word(polygon=[(10, 50), (50, 50), (50, 70), (10, 70)], detection_confidence=0.93),
        Word(polygon=[(60, 50), (110, 50), (110, 70), (60, 70)], detection_confidence=0.91),
    ]
    page = Page(blocks=[Block(lines=[Line(words=words, order=0)], order=0)])
    result = SimpleSorting(use_columns=False).predict(page)

    assert len(result.blocks) == 1
    assert len(result.blocks[0].lines) == 2
    assert result.blocks[0].lines[0].order == 0
    assert result.blocks[0].lines[1].order == 1


def test_simplesorting_columns():
    words = [
        Word(polygon=[(10, 20), (50, 20), (50, 40), (10, 40)], detection_confidence=0.95),
        Word(polygon=[(10, 50), (50, 50), (50, 70), (10, 70)], detection_confidence=0.93),
        Word(polygon=[(200, 20), (250, 20), (250, 40), (200, 40)], detection_confidence=0.97),
        Word(polygon=[(200, 50), (250, 50), (250, 70), (200, 70)], detection_confidence=0.91),
    ]
    page = Page(blocks=[Block(lines=[Line(words=words, order=0)], order=0)])
    result = SimpleSorting(use_columns=True, max_splits=10).predict(page)

    assert len(result.blocks) >= 1
    for block in result.blocks:
        assert len(block.lines) > 0
        for line in block.lines:
            assert len(line.words) > 0


def test_simplesorting_preserves_word_attributes():
    word = Word(
        polygon=[(10, 20), (100, 20), (100, 40), (10, 40)],
        detection_confidence=0.95,
        text="Hello",
        recognition_confidence=0.98,
    )
    page = Page(blocks=[Block(lines=[Line(words=[word], order=0)], order=0)])
    result = SimpleSorting(use_columns=False).predict(page)

    result_word = result.blocks[0].lines[0].words[0]
    assert result_word.polygon == word.polygon
    assert result_word.detection_confidence == word.detection_confidence
    assert result_word.text == word.text
    assert result_word.recognition_confidence == word.recognition_confidence


def test_organize_page_wrapper_matches_simplesorting():
    words = [
        Word(polygon=[(120, 20), (180, 20), (180, 40), (120, 40)], detection_confidence=0.93),
        Word(polygon=[(10, 20), (50, 20), (50, 40), (10, 40)], detection_confidence=0.95),
        Word(polygon=[(60, 20), (110, 20), (110, 40), (60, 40)], detection_confidence=0.97),
        Word(polygon=[(10, 50), (50, 50), (50, 70), (10, 70)], detection_confidence=0.91),
    ]
    page = Page(blocks=[Block(lines=[Line(words=words, order=0)], order=0)])

    from_wrapper = organize_page(page, max_splits=10, use_columns=True)
    from_layout = SimpleSorting(max_splits=10, use_columns=True).predict(page)

    wrapper_words = _collect_words(from_wrapper)
    layout_words = _collect_words(from_layout)

    assert len(wrapper_words) == len(layout_words)
    for left, right in zip(wrapper_words, layout_words):
        assert left.polygon == right.polygon
        assert left.order == right.order
