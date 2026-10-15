import pytest
from manuscript.data import Page, TextRegion, Table, Cell
from manuscript.layouts import ReadingSorting


def region(kind, x, y, width=20, height=10):
    return TextRegion(type=kind, polygon=[(x,y),(x+width,y),(x+width,y+height),(x,y+height)])


def test_mixed_types_recursive_order_preserves_objects_and_links():
    page = Page(image_width=500, image_height=500)
    first, second = region('block', 0, 0, 200, 100), region('block', 0, 200, 200, 100)
    left, right = region('word', 0, 10), region('line', 50, 12, height=20)
    first.children = (right, left)
    page.children = (second, first)
    nodes = list(page.descendants)
    parents = {id(node): node.parent for node in nodes}
    layout = ReadingSorting(infer_columns=False)
    assert layout.predict(page) is page
    assert page.children[0] is first and first.children[0] is left
    assert {id(node) for node in page.descendants} == {id(node) for node in nodes}
    assert all(node.parent is parents[id(node)] for node in nodes)
    before = page.to_tree_dict()
    layout.predict(page)
    assert page.to_tree_dict() == before


def test_columns_read_left_to_right():
    page = Page(image_width=500, image_height=500)
    left = region('line', 0, 100, width=100)
    right = region('line', 300, 0, width=100)
    page.children = (right, left)
    ReadingSorting(infer_columns=True).predict(page)
    assert page.children[0] is left


def test_scale_independent_row_order():
    for scale in (1, 100):
        page = Page(image_width=10000, image_height=10000)
        left = region('word', 0, 12*scale, 20*scale, 10*scale)
        right = region('line', 30*scale, 10*scale, 20*scale, 20*scale)
        page.children = (right, left)
        ReadingSorting(infer_columns=False).predict(page)
        assert page.children[0] is left


def test_table_indices_and_geometry_free_nodes():
    page = Page(image_width=100, image_height=100)
    table = Table(rows=2, columns=1)
    first, second = Cell(row=0,column=0), Cell(row=1,column=0)
    table.children = (second, first)
    positioned = region('line', 0, 0)
    page.children = (table, positioned)
    ReadingSorting().predict(page)
    assert page.children[0] is positioned
    assert table.children[0] is first and first.parent is table


@pytest.mark.parametrize('value', [0, -1, 2, float('nan')])
def test_invalid_overlap(value):
    with pytest.raises(ValueError):
        ReadingSorting(row_overlap=value)


def test_sort_and_containment_use_same_virtual_order():
    def make():
        page = Page(image_width=500, image_height=500)
        block = region('block', 0, 0, 200, 200)
        line = region('line', 10, 10, 100, 20)
        word = region('word', 20, 12, 20, 10)
        outside = region('line', 0, 300)
        table = Table(rows=1, columns=1, polygon=[(10,60),(150,60),(150,150),(10,150)])
        cell = Cell(row=0,column=0)
        table.children = (cell,)
        page.children = (outside, word, table, line, block)
        for i, node in enumerate(page.children):
            node.label = str(i)
        return page, block, line, word, table, cell

    flat, *_ = make()
    nested, block, line, word, table, cell = make()
    ReadingSorting().predict(flat)
    ReadingSorting(mode='containment').predict(nested)
    assert all(node.parent is flat for node in flat.children)
    assert line.parent is block and word.parent is line
    assert table.parent is block and cell.parent is table
    assert [node.label for node in flat.children] == [
        node.label for node in nested.descendants if node.label is not None]


def test_coverage_uses_child_polygon_and_preserves_existing_parents():
    page = Page(image_width=100, image_height=100)
    block = region('block', 0, 0, 100, 100)
    inside = region('line', 5, 5, 10, 10)
    partial = region('line', 96, 5, 10, 10)
    existing = region('word', 200, 200)
    inside.children = (existing,)
    page.children = (partial, inside, block)
    ReadingSorting(mode='containment').predict(page)
    assert inside.parent is block and partial.parent is page
    assert existing.parent is inside


def test_equal_blocks_do_not_form_cycles():
    page = Page(image_width=100, image_height=100)
    first, second = region('block', 0, 0, 50, 50), region('block', 0, 0, 50, 50)
    page.children = (first, second)
    ReadingSorting(mode='containment').predict(page)
    assert first.parent is page and second.parent is page


def test_inferred_columns_create_blocks_only_in_containment_mode():
    def make():
        page = Page(image_width=500, image_height=500)
        left = region('word', 0, 100, width=100)
        right = region('line', 300, 0, width=100)
        page.children = (right, left)
        return page, left, right
    page, left, right = make()
    ReadingSorting(mode='containment', infer_columns=True).predict(page)
    assert len(page.children) == 2
    assert all(n.type == 'block' for n in page.children)
    assert left.parent is page.children[0] and right.parent is page.children[1]
    before = page.to_tree_dict()
    ReadingSorting(mode='containment', infer_columns=True).predict(page)
    assert page.to_tree_dict() == before
    assert len([n for n in page.descendants if n.type == 'block']) == 2
    flat, left, right = make()
    ReadingSorting(infer_columns=True).predict(flat)
    assert len(flat.children) == 2 and left.parent is flat and right.parent is flat


def test_one_column_block_preserves_table_subtree_and_unpositioned_elements():
    page = Page(image_width=500, image_height=500)
    line = region('line', 0, 0, width=100)
    table = Table(rows=1, columns=1, polygon=[(0,50),(100,50),(100,100),(0,100)])
    cell = Cell(row=0, column=0)
    table.children = (cell,)
    unpositioned = TextRegion(type='word')
    page.children = (table, unpositioned, line)
    ReadingSorting(mode='containment', infer_columns=True).predict(page)
    assert len(page.children) == 2 and page.children[0].type == 'block'
    assert table.parent is line.parent is page.children[0]
    assert cell.parent is table and unpositioned.parent is page


@pytest.mark.parametrize('mode', ['sort', 'containment'])
def test_existing_column_blocks_keep_left_to_right_order(mode):
    page = Page(image_width=500,image_height=500)
    left = region('block',0,100,100,100)
    right = region('block',300,0,100,100)
    page.children = (right,left)
    layout = ReadingSorting(mode=mode,infer_columns=True)
    layout.predict(page)
    assert page.children[0] is left and page.children[1] is right
    before = page.to_tree_dict()
    layout.predict(page)
    assert page.to_tree_dict() == before


def test_blocks_without_column_inference_keep_row_order():
    page = Page(image_width=500,image_height=500)
    left = region('block',0,100,100,100)
    right = region('block',300,0,100,100)
    page.children = (left,right)
    ReadingSorting(infer_columns=False).predict(page)
    assert page.children[0] is right and page.children[1] is left


@pytest.mark.parametrize("coverage,inside", [(0.89, False), (0.9, True), (0.91, True)])
def test_containment_threshold_is_fraction_of_child_area(coverage, inside):
    page = Page(image_width=200, image_height=200)
    block = region("block", 0, 0, 100, 100)
    child = region("line", 100 - coverage * 100, 20, 100, 10)
    page.children = [child, block]
    ReadingSorting(mode="containment", containment_overlap=.9).predict(page)
    assert child.parent is (block if inside else page)


def test_best_coverage_precedes_smallest_container():
    page = Page(image_width=200, image_height=200)
    partial = region("block", 0, 0, 90, 30)
    full = region("block", 0, 0, 110, 50)
    child = region("line", 0, 10, 100, 10)
    page.children = [partial, child, full]
    ReadingSorting(mode="containment").predict(page)
    assert child.parent is full
    assert partial.parent is page


def test_overlapping_blocks_stay_siblings_and_line_chooses_smallest_container():
    page = Page(image_width=300, image_height=300)
    outer = region("block", 0, 0, 200, 200)
    inner = region("block", 10, 10, 100, 100)
    line = region("line", 20, 20, 70, 20)
    word = region("word", 30, 22, 10, 10)
    page.children = [word, outer, line, inner]
    ReadingSorting(mode="containment").predict(page)
    assert inner.parent is outer.parent is page
    assert line.parent is inner and word.parent is line
    before = page.to_tree_dict()
    ReadingSorting(mode="containment").predict(page)
    assert page.to_tree_dict() == before


@pytest.mark.parametrize("mode", ["sort", "containment"])
def test_identical_coordinates_keep_distinct_objects_and_stable_order(mode):
    page = Page(image_width=100, image_height=100)
    first, second = region("word", 10, 10), region("word", 10, 10)
    page.children = [second, first]
    layout = ReadingSorting(mode=mode)
    layout.predict(page)
    assert page.children[0] is second and page.children[1] is first
    layout.predict(page)
    assert page.children[0] is second and page.children[1] is first


def test_degenerate_polygon_does_not_become_container():
    page = Page(image_width=100, image_height=100)
    block = TextRegion(type="block", polygon=[(0,0),(1,0),(2,0),(3,0)])
    child = region("word", 0, 10)
    page.children = [child, block]
    ReadingSorting(mode="containment").predict(page)
    assert child.parent is page and block.parent is page


def test_table_cells_follow_indices_and_contents_follow_geometry():
    page = Page(image_width=100, image_height=100)
    table = Table(rows=1, columns=2)
    left, right = Cell(row=0, column=0), Cell(row=0, column=1)
    first, second = region("word", 0, 0), region("word", 30, 0)
    left.children = [second, first]
    table.children = [right, left]
    page.children = [table]
    ReadingSorting().predict(page)
    assert table.children == (left, right)
    assert left.children == (first, second)
    assert left.parent is table and first.parent is left


def test_invalid_mode_input_and_containment_threshold():
    with pytest.raises(ValueError, match="mode"):
        ReadingSorting(mode="group")
    for value in (0, -1, 1.1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="containment_overlap"):
            ReadingSorting(containment_overlap=value)
    with pytest.raises(TypeError, match="document-tree Page"):
        ReadingSorting().predict({})


def test_empty_and_geometry_free_page_creates_no_column_blocks():
    page = Page(image_width=100, image_height=100)
    layout = ReadingSorting(mode="containment", infer_columns=True)
    layout.predict(page)
    assert not page.children
    first, second = TextRegion(type="line"), TextRegion(type="word")
    page.children = [first, second]
    layout.predict(page)
    assert page.children[0] is first and page.children[1] is second


@pytest.mark.parametrize("overlap,expected", [(4.9, "right"), (5., "left"), (5.1, "left")])
def test_row_overlap_boundary_changes_reading_order(overlap, expected):
    page = Page(image_width=100, image_height=100)
    right = region("word", 40, 0, 10, 10)
    left = region("word", 0, 10 - overlap, 10, 10)
    page.children = [right, left]
    ReadingSorting(row_overlap=.5).predict(page)
    assert page.children[0] is (left if expected == "left" else right)


def test_self_intersecting_container_can_be_repaired_for_containment():
    page = Page(image_width=100, image_height=100)
    block = TextRegion(type="block", polygon=[(0,0),(100,100),(0,100),(100,0)])
    child = region("word", 45, 5, 10, 5)
    page.children = [child, block]
    ReadingSorting(mode="containment").predict(page)
    assert child.parent is block


def test_word_prefers_eligible_line_over_full_block_coverage():
    page = Page(image_width=200, image_height=200)
    block = region('block', 0, 0, 150, 100)
    line = region('line', 0, 10, 50, 20)
    word = region('word', 40, 12, 20, 10)
    page.children = [word, block, line]
    ReadingSorting(mode='containment', containment_overlap=.5).predict(page)
    assert word.parent is line and line.parent is block


def test_tall_region_does_not_join_separate_rows():
    page = Page(image_width=100, image_height=100)
    upper = region('word', 40, 0, 10, 10)
    tall = region('word', 20, 0, 10, 30)
    lower = region('word', 0, 20, 10, 10)
    page.children = [lower, upper, tall]
    ReadingSorting().predict(page)
    assert all(actual is expected for actual, expected in
               zip(page.children, [tall, upper, lower]))
