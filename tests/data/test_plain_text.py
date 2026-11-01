import pytest
from manuscript.data import Page, TextRegion, Table, Cell
from manuscript import Pipeline


def node(kind, text=None, children=()):
    result = TextRegion(type=kind,text=text)
    result.children = children
    return result


def page(*children):
    result = Page(image_width=100,image_height=100)
    result.children = children
    return result


def test_flat_words_and_mixed_boundaries():
    assert page(node('word','a'),node('word','b')).get_text() == 'a b'
    assert page(node('word','a'),node('block','B'),node('word','c')).get_text() == 'a\n\nB\n\nc'
    assert page(node('word','a'),node('line','B'),node('word','c')).get_text() == 'a\nB\nc'


def test_nested_text_parent_fallback_and_no_duplicates():
    result = page(node('block','duplicate',[
        node('line','duplicate',[node('word',' a '),node('word','b')]),
        node('line','c')]),node('block','d'))
    assert result.get_text() == 'a b\nc\n\nd'
    assert Pipeline.get_text(None,result) == result.get_text()
    assert page(node('line','fallback',[node('word')])).get_text() == 'fallback'


def test_custom_separators_have_no_external_delimiters():
    result = page(node('word','a'),node('block','b'),node('word','c'))
    assert result.get_text(separators={'word':'|','block':'<>'}) == 'a<>b<>c'
    assert result.get_text(separators={'word':'|'},separator_priorities={'word':3}) == 'a|b|c'


def test_table_row_major_empty_cells_and_merged_positions():
    table = Table(rows=3,columns=3)
    merged = Cell(row=0,column=0,column_span=2,row_span=2)
    merged.children = (node('line','a\nb'),)
    last = Cell(row=2,column=2)
    last.children = (node('word','last'),)
    table.children = (last,merged)
    result = page(node('word','before'),table,node('word','after'))
    before = result.to_tree_dict()
    assert result.get_text() == 'before\n\na b\t\t\n\t\t\n\t\tlast\n\nafter'
    assert result.to_tree_dict() == before


def test_inferred_table_dimensions_and_empty_table():
    table = Table()
    cell = Cell(row=1,column=1)
    cell.children = (node('word','x'),)
    table.children = (cell,)
    assert page(table).get_text() == '\t\n\tx'
    assert page(Table(rows=2,columns=2)).get_text() == ''


def test_unsupported_format_is_explicit():
    with pytest.raises(ValueError,match='plain'):
        page().get_text(format='markdown')


@pytest.mark.parametrize("options", [
    {"separators": []}, {"separators": {"word": 1}},
    {"separator_priorities": {"word": "high"}}, {"separators": {1: " "}},
])
def test_invalid_separator_settings_are_rejected(options):
    with pytest.raises(ValueError, match="Separators"):
        page(node("word", "a")).get_text(**options)


def test_serializer_rejects_non_page():
    from manuscript.data._text import get_text
    with pytest.raises(TypeError, match="requires Page"):
        get_text(node("word", "a"))


def test_table_without_cells_serializes_its_existing_children():
    table = Table()
    table.children = [node("line", "first"), node("line", "second")]
    assert page(table).get_text() == "first\nsecond"
