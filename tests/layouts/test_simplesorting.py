from tests.support.tree import make_page, make_region, sibling_index
from manuscript.api.layout import BaseLayout
from manuscript.data import Page, TextRegion, Image, Table, Cell
from manuscript.layouts import SimpleSorting
from manuscript.utils import organize_page


def _collect_text_spans(page: Page):
    return [
        text_span
        for block in page.children
        for line in block.children
        for text_span in line.children
    ]


def test_simplesorting_inherits_base_layout():
    assert issubclass(SimpleSorting, BaseLayout)


def test_simplesorting_empty_page():
    layout = SimpleSorting()
    result = layout.predict(make_page(children=[]))
    assert not result.children


def test_simplesorting_single_text_span():
    text_span = make_region(
        "word",
        polygon=[(10, 20), (100, 20), (100, 40), (10, 40)],
        detection_confidence=0.95,
    )
    page = make_page(
        children=[
            make_region("block", children=[make_region("line", children=[text_span])])
        ]
    )
    result = SimpleSorting(use_columns=False).predict(page)
    text_spans = _collect_text_spans(result)
    assert len(text_spans) == 1
    assert sibling_index(text_spans[0]) == 0


def test_simplesorting_multiple_text_spans_in_line():
    text_spans = [
        make_region(
            "word",
            polygon=[(120, 20), (180, 20), (180, 40), (120, 40)],
            detection_confidence=0.93,
        ),
        make_region(
            "word",
            polygon=[(10, 20), (50, 20), (50, 40), (10, 40)],
            detection_confidence=0.95,
        ),
        make_region(
            "word",
            polygon=[(60, 20), (110, 20), (110, 40), (60, 40)],
            detection_confidence=0.97,
        ),
    ]
    page = make_page(children=text_spans)
    result = SimpleSorting(use_columns=False).predict(page)
    ordered = result.children[0].children[0].children
    assert [sibling_index(text_span) for text_span in ordered] == [0, 1, 2]
    assert [text_span.detection_confidence for text_span in ordered] == [
        0.95,
        0.97,
        0.93,
    ]
    assert all(actual is expected for actual, expected in
               zip(ordered, [text_spans[1], text_spans[2], text_spans[0]]))


def test_simplesorting_multiple_lines():
    text_spans = [
        make_region(
            "word",
            polygon=[(10, 20), (50, 20), (50, 40), (10, 40)],
            detection_confidence=0.95,
        ),
        make_region(
            "word",
            polygon=[(60, 20), (110, 20), (110, 40), (60, 40)],
            detection_confidence=0.97,
        ),
        make_region(
            "word",
            polygon=[(10, 50), (50, 50), (50, 70), (10, 70)],
            detection_confidence=0.93,
        ),
        make_region(
            "word",
            polygon=[(60, 50), (110, 50), (110, 70), (60, 70)],
            detection_confidence=0.91,
        ),
    ]
    page = make_page(children=text_spans)
    result = SimpleSorting(use_columns=False).predict(page)
    assert len(result.children) == 1
    assert len(result.children[0].children) == 2
    assert sibling_index(result.children[0].children[0]) == 0
    assert sibling_index(result.children[0].children[1]) == 1


def test_simplesorting_columns():
    text_spans = [
        make_region(
            "word",
            polygon=[(10, 20), (50, 20), (50, 40), (10, 40)],
            detection_confidence=0.95,
        ),
        make_region(
            "word",
            polygon=[(10, 50), (50, 50), (50, 70), (10, 70)],
            detection_confidence=0.93,
        ),
        make_region(
            "word",
            polygon=[(200, 20), (250, 20), (250, 40), (200, 40)],
            detection_confidence=0.97,
        ),
        make_region(
            "word",
            polygon=[(200, 50), (250, 50), (250, 70), (200, 70)],
            detection_confidence=0.91,
        ),
    ]
    page = make_page(children=text_spans)
    result = SimpleSorting(use_columns=True, max_splits=10).predict(page)
    assert len(result.children) == 2
    assert [len(block.children) for block in result.children] == [2, 2]
    ordered_words = _collect_text_spans(result)
    assert all(actual is expected for actual, expected in zip(ordered_words, text_spans))


def test_simplesorting_preserves_text_span_attributes():
    text_span = make_region(
        "word",
        polygon=[(10, 20), (100, 20), (100, 40), (10, 40)],
        detection_confidence=0.95,
        text="Hello",
        recognition_confidence=0.98,
    )
    page = make_page(
        children=[
            make_region("block", children=[make_region("line", children=[text_span])])
        ]
    )
    before = text_span.model_dump()
    result = SimpleSorting(use_columns=False).predict(page)
    result_text_span = result.children[0].children[0].children[0]
    assert result_text_span is text_span
    assert result_text_span.model_dump() == before
    assert result_text_span.polygon == text_span.polygon
    assert result_text_span.detection_confidence == text_span.detection_confidence
    assert result_text_span.text == text_span.text
    assert result_text_span.recognition_confidence == text_span.recognition_confidence


def test_organize_page_wrapper_matches_simplesorting():
    text_spans = [
        make_region(
            "word",
            polygon=[(120, 20), (180, 20), (180, 40), (120, 40)],
            detection_confidence=0.93,
        ),
        make_region(
            "word",
            polygon=[(10, 20), (50, 20), (50, 40), (10, 40)],
            detection_confidence=0.95,
        ),
        make_region(
            "word",
            polygon=[(60, 20), (110, 20), (110, 40), (60, 40)],
            detection_confidence=0.97,
        ),
        make_region(
            "word",
            polygon=[(10, 50), (50, 50), (50, 70), (10, 70)],
            detection_confidence=0.91,
        ),
    ]
    page = make_page(children=text_spans)
    from_wrapper = organize_page(page.clone(), max_splits=10, use_columns=True)
    from_layout = SimpleSorting(max_splits=10, use_columns=True).predict(page.clone())
    assert from_wrapper is not from_layout
    assert from_wrapper.to_tree_dict() == from_layout.to_tree_dict()
    assert all(word.parent is page for word in page.children)
    wrapper_text_spans = _collect_text_spans(from_wrapper)
    layout_text_spans = _collect_text_spans(from_layout)
    assert len(wrapper_text_spans) == len(layout_text_spans)
    for left, right in zip(wrapper_text_spans, layout_text_spans):
        assert left.polygon == right.polygon
        assert sibling_index(left) == sibling_index(right)


def word(x, y, text=None):
    return TextRegion(type='word', polygon=[(x,y),(x+10,y),(x+10,y+10),(x,y+10)], text=text)


def test_grouping_preserves_duplicates_and_is_idempotent():
    page = Page(image_width=100, image_height=100)
    words = [word(25,10,'b'), word(5,10,'a'), word(5,10,'duplicate'), word(5,40)]
    image = Image(label='figure')
    page.children = (*words, image)
    layout = SimpleSorting(use_columns=False)
    assert layout.predict(page) is page
    assert all(any(n is w for n in page.descendants) for w in words)
    assert words[0].text == 'b' and image.parent is page
    assert len([n for n in page.descendants if isinstance(n,TextRegion) and n.type=='line']) == 2
    before = page.to_tree_dict()
    layout.predict(page)
    assert page.to_tree_dict() == before


def test_existing_regions_and_table_structure_preserved():
    page = Page(image_width=100, image_height=100)
    block = TextRegion(type='block', polygon=[(0,0),(90,0),(90,90),(0,90)],text='keep')
    block.children = (word(20,10),word(5,10))
    table = Table(rows=1, columns=1)
    cell = Cell(row=0,column=0)
    table.children = (cell,)
    page.children = (table,block)
    polygon = list(block.polygon)
    SimpleSorting(use_columns=False).predict(page)
    assert block.polygon == polygon and block.text == 'keep'
    assert table.children == (cell,) and cell.parent is table
    assert block.children[0].type == "line" and block.parent is page


def test_ready_lines_stay_separate_from_grouped_words():
    page = Page(image_width=100, image_height=100)
    line = TextRegion(type='line', polygon=[(0,30),(70,30),(70,40),(0,40)])
    w = word(5, 10)
    page.children = (line, w)
    SimpleSorting(use_columns=False).predict(page)
    assert line.type == 'line' and line.parent is page
    assert w.type == 'word' and w.parent.type == 'line'
    assert w.parent.parent.type == "block"
    assert w.parent.parent.parent is page


def test_table_subtree_and_ancestor_links_stay_unchanged():
    page = Page(image_width=100, image_height=100)
    block = TextRegion(type='block')
    table = Table(rows=1, columns=1)
    cell = Cell(row=0, column=0)
    w = word(5, 10)
    page.children = (block,)
    block.children = (table,)
    table.children = (cell,)
    cell.children = (w,)
    before = table.to_tree_dict()
    SimpleSorting().predict(page)
    assert table.to_tree_dict() == before
    assert table.parent is block and block.parent is page




def test_geometry_free_words_keep_their_data_and_sort_last():
    page = Page(image_width=100, image_height=100)
    unknown = TextRegion(type="word", text="no geometry")
    positioned = word(5, 10, "known")
    page.children = [unknown, positioned]
    SimpleSorting(use_columns=False).predict(page)
    assert page.children[-1] is unknown and unknown.parent is page
    assert unknown.text == "no geometry" and unknown.polygon is None
    assert positioned.parent.type == "line"


def test_existing_lines_without_words_do_not_create_blocks():
    page = Page(image_width=100, image_height=100)
    lower = TextRegion(type="line", polygon=[(0,40),(80,40),(80,50),(0,50)], text="lower")
    upper = TextRegion(type="line", polygon=[(0,10),(80,10),(80,20),(0,20)], text="upper")
    page.children = [lower, upper]
    SimpleSorting().predict(page)
    assert page.children == (upper, lower)
    assert all(node.type == "line" and node.parent is page for node in page.children)


def test_wrong_input_is_rejected():
    import pytest
    with pytest.raises(TypeError, match="document-tree Page"):
        SimpleSorting().predict({})


def test_zero_split_limit_accepts_fractional_coordinates():
    page = Page(image_width=100, image_height=100)
    left, right = word(1.5, 10.5), word(60.5, 10.5)
    page.children = [right, left]
    layout = SimpleSorting(max_splits=0)
    layout.predict(page)
    assert len(page.children) == 2
    assert _collect_text_spans(page) == [left, right]
    before = page.to_tree_dict()
    layout.predict(page)
    assert page.to_tree_dict() == before


def test_words_crossing_left_image_edge_are_grouped():
    page = Page(image_width=100, image_height=100)
    left, right = word(-5, 10), word(60, 10)
    page.children = [right, left]
    SimpleSorting().predict(page)
    assert len(page.children) == 2
    assert _collect_text_spans(page) == [left, right]
    assert left.parent.type == 'line' and left.parent.parent.type == 'block'
    assert left.polygon[0] == (-5, 10)


def test_invalid_column_settings_are_rejected():
    import pytest
    for value in (-1, 1.5, True):
        with pytest.raises(ValueError, match='max_splits'):
            SimpleSorting(max_splits=value)
    with pytest.raises(TypeError, match='use_columns'):
        SimpleSorting(use_columns='false')
