import pytest
from manuscript.data import TextRegion, Table
from manuscript.utils import (
    merge_text_spans,
    collapse_page_text_spans,
    create_page_from_text,
)


@pytest.mark.parametrize("method", ["bbox", "convex_hull"])
def test_merge_geometry_and_confidence(method):
    a = TextRegion(
        type="word",
        text="one",
        polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
        detection_confidence=0.8,
    )
    b = TextRegion(
        type="word",
        text="two",
        polygon=[(20, 0), (30, 0), (30, 10), (20, 10)],
        detection_confidence=1.0,
    )
    merged = merge_text_spans([a, b], method=method)
    assert merged.text == "one two"
    assert merged.detection_confidence == pytest.approx(0.9)
    assert min(x for x, y in merged.polygon) == 0
    assert max(x for x, y in merged.polygon) == 30
    assert merged.parent is None


def test_collapse_rejects_unknown_level():
    with pytest.raises(ValueError):
        collapse_page_text_spans(create_page_from_text("text"), level="unknown")


def test_empty_merge():
    assert merge_text_spans([]) is None


def test_collapse_lines_preserves_text_and_input():
    page = create_page_from_text(["Первое слово", "Вторая строка"])
    before = page.to_tree_dict()
    result = collapse_page_text_spans(page)
    lines = result.children[0].children
    assert [line.text for line in lines] == ["Первое слово", "Вторая строка"]
    assert all(not line.children for line in lines)
    assert all(line.parent is result.children[0] for line in lines)
    assert page.to_tree_dict() == before


def test_collapse_block_preserves_unrelated_table():
    page = create_page_from_text(["один два", "три"])
    table = Table(rows=1, columns=1)
    table.parent = page
    result = collapse_page_text_spans(page, level="block")
    assert result.children[0].text == "один два\nтри"
    assert not result.children[0].children
    assert result.children[1].type == "table"


def test_collapse_refuses_to_discard_nontext_descendants():
    page = create_page_from_text(["текст"])
    Table().parent = page.children[0]
    with pytest.raises(TypeError, match="Only text"):
        collapse_page_text_spans(page, level="block")
    assert page.children[0].children[-1].type == "table"


def test_merge_handles_missing_geometry_and_confidence():
    a = TextRegion(type="word", text="один")
    b = TextRegion(type="word", text="два", recognition_confidence=0.8)
    result = merge_text_spans([a, b], type="line")
    assert result.type == "line"
    assert result.text == "один два"
    assert result.polygon is None
    assert result.detection_confidence is None
    assert result.recognition_confidence == 0.8


@pytest.mark.parametrize("angles,expected", [([90,90],90), ([90,0],None), ([None,None],None)])
def test_merge_preserves_only_shared_orientation_without_changing_inputs(angles, expected):
    regions = [TextRegion(type="word", text=str(i), orientation=angle)
               for i, angle in enumerate(angles)]
    before = [region.to_tree_dict() for region in regions]
    merged = merge_text_spans(regions, type="line")
    assert merged.orientation == expected
    assert [region.to_tree_dict() for region in regions] == before


def test_collapse_retains_container_metadata_and_original_polygon():
    from manuscript.utils.page_transforms import collapse_line_text_spans
    polygon = [(0,0),(30,0),(30,20),(0,20)]
    line = TextRegion(type="line", polygon=polygon, label="sideways", orientation=90,
                      detection_confidence=.8, recognition_confidence=.7)
    word = TextRegion(type="word", text="inside")
    line.children = [word]
    result = collapse_line_text_spans(line)
    assert result.text == "inside" and not result.children
    assert result.polygon == polygon and result.label == "sideways"
    assert result.orientation == 90
    assert (result.detection_confidence, result.recognition_confidence) == (.8,.7)
    assert word.parent is line and line.children == (word,)


def test_triangular_convex_hull_preserves_outline_with_four_vertices():
    from shapely.geometry import Polygon
    region = TextRegion(type='word', text='triangle',
                        polygon=[(0, 0), (4, 0), (2, 3), (1, 1.5)])
    before = region.to_tree_dict()
    result = merge_text_spans([region], method='convex_hull')
    assert len(result.polygon) == 4
    assert Polygon(result.polygon).equals(Polygon(region.polygon))
    assert region.to_tree_dict() == before
