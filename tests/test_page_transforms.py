import pytest

from manuscript.data import Block, Line, Page, TextSpan
from manuscript.utils import (
    collapse_block_text_spans,
    collapse_line_text_spans,
    collapse_page_text_spans,
    merge_text_spans,
)


def _make_text_span(
    polygon,
    *,
    detection_confidence=1.0,
    text=None,
    recognition_confidence=None,
    order=None,
):
    return TextSpan(
        polygon=polygon,
        detection_confidence=detection_confidence,
        text=text,
        recognition_confidence=recognition_confidence,
        order=order,
    )


def test_merge_text_spans_bbox_merges_geometry_and_text():
    text_spans = [
        _make_text_span(
            [(10, 20), (30, 20), (30, 40), (10, 40)],
            detection_confidence=0.8,
            text="hello",
            recognition_confidence=0.9,
            order=0,
        ),
        _make_text_span(
            [(35, 22), (55, 22), (55, 38), (35, 38)],
            detection_confidence=1.0,
            text="world",
            recognition_confidence=0.7,
            order=1,
        ),
    ]

    merged = merge_text_spans(text_spans, method="bbox")

    assert merged is not None
    assert merged.polygon == [
        (10.0, 20.0),
        (55.0, 20.0),
        (55.0, 40.0),
        (10.0, 40.0),
    ]
    assert merged.detection_confidence == pytest.approx(0.9)
    assert merged.text == "hello world"
    assert merged.recognition_confidence == pytest.approx(0.8)
    assert merged.order == 0


def test_merge_text_spans_convex_hull_can_return_non_rectangular_polygon():
    text_spans = [
        _make_text_span([(1, 2), (2, 1), (3, 2), (2, 3)]),
        _make_text_span([(5, 2), (6, 1), (7, 2), (6, 3)]),
    ]

    merged = merge_text_spans(text_spans, method="convex_hull")

    assert merged is not None
    assert len(merged.polygon) == 6
    assert set(merged.polygon) == {
        (6.0, 3.0),
        (7.0, 2.0),
        (6.0, 1.0),
        (2.0, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
    }


def test_collapse_line_text_spans_returns_single_span_line():
    line = Line(
        text_spans=[
            _make_text_span([(0, 0), (10, 0), (10, 10), (0, 10)], text="a"),
            _make_text_span([(12, 0), (20, 0), (20, 10), (12, 10)], text="b"),
        ],
        order=3,
    )

    collapsed = collapse_line_text_spans(line)

    assert collapsed.order == 3
    assert len(collapsed.text_spans) == 1
    assert collapsed.text_spans[0].text == "a b"


def test_collapse_block_text_spans_returns_single_line_and_single_span():
    block = Block(
        lines=[
            Line(text_spans=[_make_text_span([(0, 0), (10, 0), (10, 10), (0, 10)], text="a")], order=0),
            Line(text_spans=[_make_text_span([(0, 20), (10, 20), (10, 30), (0, 30)], text="b")], order=1),
        ],
        order=4,
    )

    collapsed = collapse_block_text_spans(block)

    assert collapsed.order == 4
    assert len(collapsed.lines) == 1
    assert collapsed.lines[0].order == 0
    assert len(collapsed.lines[0].text_spans) == 1
    assert collapsed.lines[0].text_spans[0].text == "a b"


def test_collapse_page_text_spans_level_line_collapses_each_line_only():
    page = Page(
        blocks=[
            Block(
                lines=[
                    Line(
                        text_spans=[
                            _make_text_span([(0, 0), (10, 0), (10, 10), (0, 10)], text="a"),
                            _make_text_span([(12, 0), (20, 0), (20, 10), (12, 10)], text="b"),
                        ],
                        order=0,
                    ),
                    Line(
                        text_spans=[_make_text_span([(0, 20), (10, 20), (10, 30), (0, 30)], text="c")],
                        order=1,
                    ),
                ],
                order=2,
            )
        ]
    )

    collapsed = collapse_page_text_spans(page, level="line")

    assert len(collapsed.blocks) == 1
    assert len(collapsed.blocks[0].lines) == 2
    assert [len(line.text_spans) for line in collapsed.blocks[0].lines] == [1, 1]
    assert collapsed.blocks[0].lines[0].text_spans[0].text == "a b"
    assert collapsed.blocks[0].lines[1].text_spans[0].text == "c"


def test_collapse_page_text_spans_level_block_collapses_each_block():
    page = Page(
        blocks=[
            Block(
                lines=[
                    Line(text_spans=[_make_text_span([(0, 0), (10, 0), (10, 10), (0, 10)], text="a")], order=0),
                    Line(text_spans=[_make_text_span([(0, 20), (10, 20), (10, 30), (0, 30)], text="b")], order=1),
                ],
                order=0,
            ),
            Block(
                lines=[
                    Line(text_spans=[_make_text_span([(50, 0), (60, 0), (60, 10), (50, 10)], text="c")], order=0),
                ],
                order=1,
            ),
        ]
    )

    collapsed = collapse_page_text_spans(page, level="block")

    assert len(collapsed.blocks) == 2
    assert all(len(block.lines) == 1 for block in collapsed.blocks)
    assert collapsed.blocks[0].lines[0].text_spans[0].text == "a b"
    assert collapsed.blocks[1].lines[0].text_spans[0].text == "c"


def test_collapse_page_text_spans_preserves_empty_line():
    page = Page(blocks=[Block(lines=[Line(text_spans=[], order=5)], order=2)])

    collapsed = collapse_page_text_spans(page, level="line")

    assert len(collapsed.blocks) == 1
    assert len(collapsed.blocks[0].lines) == 1
    assert collapsed.blocks[0].order == 2
    assert collapsed.blocks[0].lines[0].order == 5
    assert collapsed.blocks[0].lines[0].text_spans == []


def test_collapse_page_text_spans_rejects_unknown_level():
    page = Page(blocks=[])

    with pytest.raises(ValueError, match="level must be 'line' or 'block'"):
        collapse_page_text_spans(page, level="paragraph")
