from manuscript.api.layout import BaseLayout
from manuscript.data import Block, Line, Page, TextSpan
from manuscript.layouts import SimpleSorting
from manuscript.utils import organize_page


def _collect_text_spans(page: Page):
    return [
        text_span
        for block in page.blocks
        for line in block.lines
        for text_span in line.text_spans
    ]


def test_simplesorting_inherits_base_layout():
    assert issubclass(SimpleSorting, BaseLayout)


def test_simplesorting_empty_page():
    layout = SimpleSorting()
    result = layout.predict(Page(blocks=[]))
    assert len(result.blocks) == 1
    assert len(result.blocks[0].lines) == 1
    assert len(result.blocks[0].lines[0].text_spans) == 0


def test_simplesorting_single_text_span():
    text_span = TextSpan(
        polygon=[(10, 20), (100, 20), (100, 40), (10, 40)],
        detection_confidence=0.95,
    )
    page = Page(
        blocks=[Block(lines=[Line(text_spans=[text_span], order=0)], order=0)]
    )
    result = SimpleSorting(use_columns=False).predict(page)

    text_spans = _collect_text_spans(result)
    assert len(text_spans) == 1
    assert text_spans[0].order == 0


def test_simplesorting_multiple_text_spans_in_line():
    text_spans = [
        TextSpan(
            polygon=[(120, 20), (180, 20), (180, 40), (120, 40)],
            detection_confidence=0.93,
        ),
        TextSpan(
            polygon=[(10, 20), (50, 20), (50, 40), (10, 40)],
            detection_confidence=0.95,
        ),
        TextSpan(
            polygon=[(60, 20), (110, 20), (110, 40), (60, 40)],
            detection_confidence=0.97,
        ),
    ]
    page = Page(
        blocks=[Block(lines=[Line(text_spans=text_spans, order=0)], order=0)]
    )
    result = SimpleSorting(use_columns=False).predict(page)

    ordered = result.blocks[0].lines[0].text_spans
    assert [text_span.order for text_span in ordered] == [0, 1, 2]
    assert [text_span.detection_confidence for text_span in ordered] == [
        0.95,
        0.97,
        0.93,
    ]


def test_simplesorting_multiple_lines():
    text_spans = [
        TextSpan(
            polygon=[(10, 20), (50, 20), (50, 40), (10, 40)],
            detection_confidence=0.95,
        ),
        TextSpan(
            polygon=[(60, 20), (110, 20), (110, 40), (60, 40)],
            detection_confidence=0.97,
        ),
        TextSpan(
            polygon=[(10, 50), (50, 50), (50, 70), (10, 70)],
            detection_confidence=0.93,
        ),
        TextSpan(
            polygon=[(60, 50), (110, 50), (110, 70), (60, 70)],
            detection_confidence=0.91,
        ),
    ]
    page = Page(
        blocks=[Block(lines=[Line(text_spans=text_spans, order=0)], order=0)]
    )
    result = SimpleSorting(use_columns=False).predict(page)

    assert len(result.blocks) == 1
    assert len(result.blocks[0].lines) == 2
    assert result.blocks[0].lines[0].order == 0
    assert result.blocks[0].lines[1].order == 1


def test_simplesorting_columns():
    text_spans = [
        TextSpan(
            polygon=[(10, 20), (50, 20), (50, 40), (10, 40)],
            detection_confidence=0.95,
        ),
        TextSpan(
            polygon=[(10, 50), (50, 50), (50, 70), (10, 70)],
            detection_confidence=0.93,
        ),
        TextSpan(
            polygon=[(200, 20), (250, 20), (250, 40), (200, 40)],
            detection_confidence=0.97,
        ),
        TextSpan(
            polygon=[(200, 50), (250, 50), (250, 70), (200, 70)],
            detection_confidence=0.91,
        ),
    ]
    page = Page(
        blocks=[Block(lines=[Line(text_spans=text_spans, order=0)], order=0)]
    )
    result = SimpleSorting(use_columns=True, max_splits=10).predict(page)

    assert len(result.blocks) >= 1
    for block in result.blocks:
        assert len(block.lines) > 0
        for line in block.lines:
            assert len(line.text_spans) > 0


def test_simplesorting_preserves_text_span_attributes():
    text_span = TextSpan(
        polygon=[(10, 20), (100, 20), (100, 40), (10, 40)],
        detection_confidence=0.95,
        text="Hello",
        recognition_confidence=0.98,
    )
    page = Page(
        blocks=[Block(lines=[Line(text_spans=[text_span], order=0)], order=0)]
    )
    result = SimpleSorting(use_columns=False).predict(page)

    result_text_span = result.blocks[0].lines[0].text_spans[0]
    assert result_text_span.polygon == text_span.polygon
    assert result_text_span.detection_confidence == text_span.detection_confidence
    assert result_text_span.text == text_span.text
    assert (
        result_text_span.recognition_confidence
        == text_span.recognition_confidence
    )


def test_organize_page_wrapper_matches_simplesorting():
    text_spans = [
        TextSpan(
            polygon=[(120, 20), (180, 20), (180, 40), (120, 40)],
            detection_confidence=0.93,
        ),
        TextSpan(
            polygon=[(10, 20), (50, 20), (50, 40), (10, 40)],
            detection_confidence=0.95,
        ),
        TextSpan(
            polygon=[(60, 20), (110, 20), (110, 40), (60, 40)],
            detection_confidence=0.97,
        ),
        TextSpan(
            polygon=[(10, 50), (50, 50), (50, 70), (10, 70)],
            detection_confidence=0.91,
        ),
    ]
    page = Page(
        blocks=[Block(lines=[Line(text_spans=text_spans, order=0)], order=0)]
    )

    from_wrapper = organize_page(page, max_splits=10, use_columns=True)
    from_layout = SimpleSorting(max_splits=10, use_columns=True).predict(page)

    wrapper_text_spans = _collect_text_spans(from_wrapper)
    layout_text_spans = _collect_text_spans(from_layout)

    assert len(wrapper_text_spans) == len(layout_text_spans)
    for left, right in zip(wrapper_text_spans, layout_text_spans):
        assert left.polygon == right.polygon
        assert left.order == right.order
