import json
from typing import Literal

import pytest
from anytree import LoopError
from manuscript.data import Element, Page, TextRegion, Table, Cell, to_coco, from_coco
from manuscript.data.tree import recognition_candidates

POLYGON = [(1.2, 2.3), (10, 2), (12, 4), (10, 8), (1, 8)]


def test_tree_coco_custom_class_and_reparenting(tmp_path):
    class Custom(Element):
        type: Literal['test-custom'] = 'test-custom'
        note: str
    Element.register('test-custom', Custom)
    page = Page(image_width=100, image_height=100)
    block = TextRegion(type='block', polygon=POLYGON, text='original')
    line = TextRegion(type='line', polygon=POLYGON, label='handwritten')
    page.children = (line, block, Custom(note='kept'))
    line.parent = block
    assert block.children[0] is line and block.text == 'original'
    with pytest.raises(LoopError):
        block.parent = line
    path = tmp_path / 'page.json'
    page.save(path)
    restored = Page.load(path)
    assert restored.to_tree_dict() == page.to_tree_dict()
    assert isinstance(restored.children[1], Custom)
    data = json.loads(path.read_text())
    assert {c['name'] for c in data['categories']} == {'block', 'line', 'test-custom'}
    line_annotation = next(a for a in data['annotations'] if a['manuscript']['type'] == 'line')
    assert line_annotation['manuscript']['label'] == 'handwritten'
    assert 'kind' not in line_annotation['manuscript'] and 'level' not in line_annotation['manuscript']
    data['annotations'][0]['parent_id'] = data['annotations'][0]['id']
    with pytest.raises(ValueError, match='Cyclic'):
        from_coco(data)
    data['annotations'][0]['parent_id'] = 999
    with pytest.raises(ValueError, match='Unknown parent'):
        from_coco(data)


def test_levels_and_table_grid():
    line = TextRegion(type='line', polygon=POLYGON)
    word = TextRegion(type='word', polygon=POLYGON)
    line.children = (word,)
    assert recognition_candidates(line, ('line',)) == [line]
    assert recognition_candidates(line, ('line', 'word')) == [word]
    table = Table(rows=2, columns=2)
    table.children = (Cell(row=0, column=0, column_span=2), Cell(row=0, column=1))
    with pytest.raises(ValueError, match='overlap'):
        table.validate_tree()


def test_unversioned_coco_defaults_and_sibling_order():
    data = {'images':[{'id':1,'width':100,'height':100,'file_name':'test.png'}],
            'categories':[{'id':7,'name':'block'},{'id':8,'name':'line'}],
            'annotations':[
                {'id':3,'image_id':1,'category_id':8,'parent_id':1,
                 'segmentation':[[1,1,20,1,20,20,1,20]],'value':''},
                {'id':1,'image_id':1,'category_id':7,'parent_id':None,'segmentation':[]},
                {'id':2,'image_id':1,'category_id':8,'parent_id':1,
                 'segmentation':[[1,30,20,30,20,40,1,40]],'value':'known'}]}
    page = from_coco(data)
    lines = page.children[0].children
    assert [line.text for line in lines] == [None,'known']
    assert all(line.detection_confidence is None and line.recognition_confidence is None for line in lines)
    assert from_coco(to_coco(page)).to_tree_dict() == page.to_tree_dict()
    data['annotations'][0]['parent_id']=999
    with pytest.raises(ValueError,match='Unknown parent'):
        from_coco(data)


def test_unknown_element_roundtrip(tmp_path):
    page = Page(image_width=100, image_height=100)
    unknown = Element(type='бурмалда', polygon=POLYGON, detection_confidence=.9)
    unknown.parent = page
    unknown = page.children[0]
    assert type(unknown) is Element and unknown.type == 'бурмалда'
    unknown.children = (TextRegion(type='word',polygon=POLYGON,text='inside'),)
    path = tmp_path / 'unknown.json'
    page.save(path)
    with pytest.warns(UserWarning, match='Unknown element type'):
        restored = Page.load(path)
    assert restored.to_tree_dict() == page.to_tree_dict()
    assert restored.children[0].children[0].parent is restored.children[0]
    with pytest.warns(UserWarning, match='Unknown element type'):
        clone = page.clone()
    assert clone.to_tree_dict() == page.to_tree_dict()
    coco = page.to_dict()
    assert any(category['name']=='бурмалда' for category in coco['categories'])
    coco.pop('manuscript_schema')
    for annotation in coco['annotations']:
        annotation.pop('manuscript')
    with pytest.warns(UserWarning,match='Unknown element type'):
        imported = from_coco(coco)
    assert type(imported.children[0]) is Element
    assert imported.children[0].type == 'бурмалда'


def test_unknown_type_still_validates_common_fields():
    with pytest.raises(ValueError,match='nonempty'):
        Element.resolve_type('')
    with pytest.warns(UserWarning,match='Unknown element type'):
        cls = Element.resolve_type('unknown-invalid-confidence')
    with pytest.raises(ValueError):
        cls(type='unknown-invalid-confidence',detection_confidence=2)


def test_orientation_optional_finite_and_roundtrip():
    assert Element(type='image').orientation is None
    for angle in (0,90,-90,180,12.5):
        region=TextRegion(type='line',orientation=angle,polygon=POLYGON)
        page=Page(image_width=100,image_height=100);region.parent=page
        assert from_coco(to_coco(page)).children[0].orientation==angle
    for angle in (float('nan'),float('inf')):
        with pytest.raises(ValueError,match='finite'):
            TextRegion(type='line',orientation=angle)
    data=to_coco(page)
    for annotation in data['annotations']:
        annotation['manuscript'].pop('orientation')
    assert from_coco(data).children[0].orientation is None


def test_coco_schema_is_manuscript_extension_version():
    page = Page(image_width=100, image_height=100)
    data = to_coco(page)
    assert data['manuscript_schema'] == 'manuscript-coco-v1'
    data['manuscript_schema'] = 'manuscript-coco-v999'
    with pytest.raises(ValueError, match='Unsupported Manuscript COCO schema'):
        from_coco(data)


def test_register_rejects_empty_type():
    with pytest.raises(ValueError, match='nonempty'):
        Element.register('', Element)


@pytest.fixture
def native_coco():
    page = Page(image_width=100, image_height=100, file_name="page.png")
    page.children = [TextRegion(type="word", polygon=POLYGON, text="text")]
    return to_coco(page)


@pytest.mark.parametrize("case,match", [
    ("duplicate_category", "Duplicate category"),
    ("duplicate_annotation", "Invalid or duplicate annotation"),
    ("invalid_annotation_id", "Invalid or duplicate annotation"),
    ("wrong_image", "Unknown image"),
    ("wrong_category", "Conflicting category"),
    ("wrong_type", "Conflicting category"),
    ("page_annotation", "Invalid annotation type"),
    ("wrong_bbox", "Inconsistent geometry"),
    ("wrong_area", "Inconsistent geometry"),
    ("missing_image", "exactly one image"),
    ("multiple_images", "exactly one image"),
])
def test_invalid_coco_metadata_is_rejected(native_coco, case, match):
    from copy import deepcopy
    annotation = native_coco["annotations"][0]
    if case == "duplicate_category": native_coco["categories"].append(deepcopy(native_coco["categories"][0]))
    elif case == "duplicate_annotation": native_coco["annotations"].append(deepcopy(annotation))
    elif case == "invalid_annotation_id": annotation["id"] = True
    elif case == "wrong_image": annotation["image_id"] = 99
    elif case == "wrong_category": annotation["category_id"] = 99
    elif case == "wrong_type": annotation["manuscript"]["type"] = "line"
    elif case == "page_annotation": annotation["manuscript"]["type"] = "page"
    elif case == "wrong_bbox": annotation["bbox"] = [0, 0, 1, 1]
    elif case == "wrong_area": annotation["area"] = -1
    elif case == "missing_image": native_coco["images"] = []
    elif case == "multiple_images": native_coco["images"].append(deepcopy(native_coco["images"][0]))
    with pytest.raises(ValueError, match=match):
        from_coco(native_coco)


@pytest.mark.parametrize("segmentation", [
    {"counts": "rle", "size": [100, 100]},
    [[0, 0, 10, 0, 10, 10, 0, 10], [20, 20, 30, 20, 30, 30, 20, 30]],
    [[0, 0, 10, 0, 10, 10]],
    [[0, 0, 10, 0, 10, 10, 0, 10, 5]],
])
def test_unsupported_segmentation_is_rejected(native_coco, segmentation):
    native_coco["annotations"][0]["segmentation"] = segmentation
    with pytest.raises(ValueError, match="polygon"):
        from_coco(native_coco)


def test_parent_cycles_and_boolean_references_are_rejected(native_coco):
    from copy import deepcopy
    for parent in (1, True):
        data = deepcopy(native_coco)
        data["annotations"][0]["parent_id"] = parent
        with pytest.raises(ValueError, match="Cyclic|Unknown parent"):
            from_coco(data)


def test_geometry_free_containers_and_merged_cells_roundtrip():
    page = Page(image_width=100, image_height=100)
    table = Table(rows=2, columns=3)
    cell = Cell(row=0, column=0, row_span=2, column_span=2)
    word = TextRegion(type="word", text="inside", polygon=POLYGON)
    word.parent = cell
    cell.parent = table
    table.parent = page
    data = to_coco(page)
    assert data["annotations"][0]["segmentation"] == []
    restored = from_coco(data)
    assert restored.to_tree_dict() == page.to_tree_dict()
    assert restored.children[0].children[0].row_span == 2


def test_unversioned_bbox_does_not_invent_polygon(native_coco):
    native_coco.pop("manuscript_schema")
    annotation = native_coco["annotations"][0]
    annotation.pop("manuscript")
    annotation["segmentation"] = []
    annotation["value"] = "known"
    page = from_coco(native_coco)
    assert page.children[0].polygon is None
    assert page.children[0].text == "known"
