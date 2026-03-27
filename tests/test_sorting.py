from manuscript.data import Block, Line, Page, TextSpan
from manuscript.utils import organize_page


def test_organize_page_empty():
    page = Page(blocks=[])
    result = organize_page(page)

    assert len(result.blocks) == 1
    assert len(result.blocks[0].lines) == 1
    assert len(result.blocks[0].lines[0].text_spans) == 0


def test_organize_page_single_text_span():
    text_span = TextSpan(
        polygon=[(10, 20), (100, 20), (100, 40), (10, 40)],
        detection_confidence=0.95,
    )
    page = Page(
        blocks=[Block(lines=[Line(text_spans=[text_span], order=0)], order=0)]
    )

    result = organize_page(page, use_columns=False)

    assert len(result.blocks) == 1
    assert len(result.blocks[0].lines) == 1
    assert len(result.blocks[0].lines[0].text_spans) == 1
    assert result.blocks[0].lines[0].text_spans[0].order == 0


def test_organize_page_multiple_text_spans():
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
            polygon=[(120, 20), (180, 20), (180, 40), (120, 40)],
            detection_confidence=0.93,
        ),
    ]
    page = Page(
        blocks=[Block(lines=[Line(text_spans=text_spans, order=0)], order=0)]
    )

    result = organize_page(page, use_columns=False)

    assert len(result.blocks) == 1
    assert len(result.blocks[0].lines) == 1
    assert len(result.blocks[0].lines[0].text_spans) == 3
    for idx, text_span in enumerate(result.blocks[0].lines[0].text_spans):
        assert text_span.order == idx


def test_organize_page_multiple_lines():
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

    result = organize_page(page, use_columns=False)

    assert len(result.blocks) == 1
    assert len(result.blocks[0].lines) == 2
    assert len(result.blocks[0].lines[0].text_spans) == 2
    assert len(result.blocks[0].lines[1].text_spans) == 2
    assert result.blocks[0].lines[0].order == 0
    assert result.blocks[0].lines[1].order == 1


def test_organize_page_with_columns():
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

    result = organize_page(page, use_columns=True, max_splits=10)

    assert len(result.blocks) >= 1
    for block in result.blocks:
        assert len(block.lines) > 0
        for line in block.lines:
            assert len(line.text_spans) > 0


def test_organize_page_preserves_text_span_attributes():
    text_span = TextSpan(
        polygon=[(10, 20), (100, 20), (100, 40), (10, 40)],
        detection_confidence=0.95,
        text="Hello",
        recognition_confidence=0.98,
    )
    page = Page(
        blocks=[Block(lines=[Line(text_spans=[text_span], order=0)], order=0)]
    )

    result = organize_page(page, use_columns=False)

    result_text_span = result.blocks[0].lines[0].text_spans[0]
    assert result_text_span.polygon == text_span.polygon
    assert result_text_span.detection_confidence == text_span.detection_confidence
    assert result_text_span.text == text_span.text
    assert (
        result_text_span.recognition_confidence
        == text_span.recognition_confidence
    )
