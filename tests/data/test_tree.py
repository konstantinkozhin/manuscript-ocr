import pytest
from pydantic import ValidationError
from manuscript.data import Page, TextRegion


@pytest.mark.parametrize("region_type", ["word", "line", "block"])
def test_text_region_optional_fields(region_type):
    region = TextRegion(type=region_type)
    assert region.polygon is None
    assert region.text is None
    assert region.detection_confidence is None


@pytest.mark.parametrize(
    "polygon", [[(0, 0)] * 3, [(0, 0), (1, 0), (1, float("nan")), (0, 1)]]
)
def test_invalid_coordinates(polygon):
    with pytest.raises(ValidationError):
        TextRegion(type="word", polygon=polygon)


def test_page_round_trip_with_flat_and_nested_regions(tmp_path):
    page = Page(image_width=100, image_height=100)
    line = TextRegion(type="line")
    word = TextRegion(type="word", text="текст", recognition_confidence=0.9)
    word.parent = line
    page.children = [line, TextRegion(type="word", text="отдельно")]
    path = tmp_path / "page.json"
    page.save(path)
    restored = Page.load(path)
    assert restored.to_tree_dict() == page.to_tree_dict()
    assert restored.children[0].children[0].parent is restored.children[0]


@pytest.mark.parametrize("field", ["detection_confidence", "recognition_confidence"])
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_confidence_bounds(field, value):
    with pytest.raises(ValidationError):
        TextRegion(type="word", **{field: value})


def test_polygon_preserves_more_than_four_vertices():
    polygon = [(0, 0), (5, 0), (10, 5), (5, 10), (0, 5)]
    region = TextRegion(type="line", polygon=polygon)
    assert region.polygon == polygon


def test_validation_applies_to_updates_and_unknown_fields():
    region = TextRegion(type="word")
    with pytest.raises(ValidationError):
        region.detection_confidence = 2
    with pytest.raises(ValidationError):
        TextRegion(type="word", unexpected=True)


def test_reparenting_and_clone_preserve_order_without_sharing_nodes():
    page = Page(image_width=100, image_height=100)
    first, second = TextRegion(type="block"), TextRegion(type="block")
    a, b = TextRegion(type="word", text="a"), TextRegion(type="word", text="b")
    first.children = [a, b]
    page.children = [first, second]
    a.parent = second
    assert first.children == (b,) and second.children == (a,)
    clone = page.clone()
    clone.children[1].children[0].text = "changed"
    assert a.text == "a"
    assert all(child.parent is node for node in (clone, *clone.descendants) for child in node.children)
    assert not {id(node) for node in (page, *page.descendants)}.intersection(
        id(node) for node in (clone, *clone.descendants))


def test_invalid_parent_and_nested_page_are_rejected():
    from manuscript.data import Element
    from anytree import NodeMixin
    word = TextRegion(type="word")
    with pytest.raises(TypeError, match="Parent must"):
        word.parent = NodeMixin()
    page = Page(image_width=10, image_height=10)
    with pytest.raises(ValueError, match="remain a root"):
        page.parent = Element(type="image")
    assert word.parent is None and page.parent is None


def test_registration_rejects_conflicts_and_non_element_classes():
    from manuscript.data import Element
    with pytest.raises(TypeError, match="inherit Element"):
        Element.register("invalid-class", str)
    with pytest.raises(ValueError, match="already registered"):
        Element.register("word", Element)


@pytest.mark.parametrize("field,value", [("row", -1), ("column", -1), ("row_span", 0), ("column_span", 0)])
def test_invalid_cell_grid_fields(field, value):
    from manuscript.data import Cell
    values = {"row": 0, "column": 0, field: value}
    with pytest.raises(ValidationError):
        Cell(**values)


@pytest.mark.parametrize("cell", [(1, 0, 2, 1), (0, 1, 1, 2)])
def test_spanning_cell_cannot_exceed_table_dimensions(cell):
    from manuscript.data import Table, Cell
    row, column, row_span, column_span = cell
    table = Table(rows=2, columns=2)
    table.children = [Cell(row=row, column=column, row_span=row_span, column_span=column_span)]
    with pytest.raises(ValueError, match="beyond table"):
        table.validate_tree()


def test_incomplete_grid_and_adjacent_merged_cells_are_valid():
    from manuscript.data import Table, Cell
    table = Table(rows=3, columns=4)
    table.children = [Cell(row=0, column=0, row_span=2, column_span=2),
                      Cell(row=0, column=2, row_span=2, column_span=2)]
    table.validate_tree()
    page = Page(image_width=100, image_height=100)
    table.parent = page
    restored = Page.from_tree_dict(page.to_tree_dict())
    restored.validate_tree()
    assert restored.to_tree_dict() == page.to_tree_dict()


def test_recognition_candidates_skip_recognized_and_geometry_free_nodes():
    from manuscript.data.tree import recognition_candidates
    polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
    page = Page(image_width=20, image_height=20)
    page.children = [TextRegion(type="word", text="known", polygon=polygon),
                     TextRegion(type="word"), TextRegion(type="line", polygon=polygon)]
    assert recognition_candidates(page, ("word",)) == []
    assert recognition_candidates(page, ("line",)) == [page.children[2]]
    with pytest.raises(ValueError, match="Unknown recognition levels"):
        recognition_candidates(page, ("cell",))


def test_element_type_must_not_be_whitespace():
    from manuscript.data import Element
    with pytest.raises(ValidationError, match="nonempty"):
        Element(type=" ")
