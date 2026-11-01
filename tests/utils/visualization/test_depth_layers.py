from tests.support.tree import make_page, make_region
import numpy as np
from PIL import Image
from manuscript.data import Page, TextRegion
from manuscript.utils import visualize_page


def test_visualize_page_accepts_polygon_with_more_than_four_points():
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    text_span = make_region(
        "word",
        polygon=[
            (10.0, 10.0),
            (40.0, 8.0),
            (60.0, 20.0),
            (55.0, 35.0),
            (30.0, 40.0),
            (12.0, 28.0),
        ],
        detection_confidence=0.95,
        text="example",
    )
    page = make_page(
        children=[
            make_region("block", children=[make_region("line", children=[text_span])])
        ]
    )
    result = visualize_page(
        image, page, numbers_levels=[1], connections_levels=[1], max_size=1024
    )
    assert isinstance(result, Image.Image)
    assert result.size == (120, 80)


def test_visualize_page_accepts_nested_regions_with_polygon_mask_shape():
    image = np.full((100, 140, 3), 255, dtype=np.uint8)
    text_span = make_region(
        "word",
        polygon=[
            (70.0, 20.0),
            (100.0, 15.0),
            (120.0, 30.0),
            (112.0, 55.0),
            (82.0, 60.0),
            (65.0, 40.0),
        ],
        detection_confidence=0.88,
        text="block",
    )
    page = make_page(children=[make_region("block", children=[] + [text_span])])
    result = visualize_page(image, page, numbers_levels=[])
    assert isinstance(result, Image.Image)
    assert result.size == (140, 100)




def test_second_level_cold_colors_mix_and_repeat_consistently():
    block = make_region('block', polygon=[(5,5),(115,5),(115,75),(5,75)])
    first = make_region('line', polygon=[(10,10),(70,10),(70,60),(10,60)])
    second = make_region('word', polygon=[(50,20),(100,20),(100,65),(50,65)])
    block.children = (first,second)
    page = make_page(children=[block])
    image = np.full((80,120,3),255,np.uint8)
    result = np.asarray(visualize_page(image,page, numbers_levels=[]))
    assert np.all(result[30,20] < 255)
    assert not np.array_equal(result[30,60], result[30,20])
    assert not np.array_equal(result[30,60], result[30,90])
    assert np.all(result[10,10] <= 98)
    assert np.array_equal(result,np.asarray(visualize_page(image,page, numbers_levels=[])))


def test_third_level_lightens_parent_tint_and_brightens_overlaps():
    block = make_region('block', polygon=[(5,5),(115,5),(115,75),(5,75)])
    line = make_region('line', polygon=[(10,10),(110,10),(110,70),(10,70)])
    first = make_region('word', polygon=[(20,20),(65,20),(65,55),(20,55)])
    second = make_region('word', polygon=[(45,25),(95,25),(95,60),(45,60)])
    line.children = (first,second)
    block.children = (line,)
    image = np.full((80,120,3),100,np.uint8)
    result = np.asarray(visualize_page(image,make_page(children=[block]), numbers_levels=[]))
    parent_color = result[15,15].astype(float)
    assert np.allclose(result[30,30], parent_color * 1.25, atol=1)
    assert np.allclose(result[35,55], parent_color * 1.50, atol=1)
    assert len(set(result[30,30])) > 1
    assert np.all(result[20,20] <= 98)


def test_depth_rule_extends_to_levels_four_five_and_six():
    image = np.full((80,120,3),80,np.uint8)
    colors = []
    for depth in range(1,7):
        page = make_page(children=[])
        parent = page
        for _ in range(depth):
            child = make_region('block', polygon=[(10,10),(110,10),(110,70),(10,70)])
            parent.children = (child,)
            parent = child
        colors.append(np.asarray(visualize_page(image,page, numbers_levels=[]))[40,60])
    assert np.array_equal(colors[0], (80,80,80))
    for odd in (2,4):
        assert np.allclose(colors[odd], colors[odd-1].astype(float)*1.25, atol=1)
    for even in (1,3,5):
        expected = colors[even-1].astype(float)*(180/255) + np.array((52,178,207))*(75/255)
        assert np.allclose(colors[even],expected,atol=1)


def test_outline_color_is_dark_and_inherited_from_first_level():
    first = make_region('block', polygon=[(5,5),(65,5),(65,75),(5,75)])
    child = make_region('line', polygon=[(15,15),(55,15),(55,65),(15,65)])
    grandchild = make_region('word', polygon=[(25,25),(45,25),(45,55),(25,55)])
    second = make_region('block', polygon=[(75,5),(115,5),(115,75),(75,75)])
    first.children = (child,)
    child.children = (grandchild,)
    result = np.asarray(visualize_page(np.full((80,120,3),255,np.uint8),
                                      make_page(children=[first,second]), numbers_levels=[]))
    outline = result[5,30]
    assert np.all(outline <= 98)
    assert np.array_equal(outline,result[15,30])
    assert np.array_equal(outline,result[25,30])
    assert np.array_equal(outline,result[6,30])  # Two-pixel stroke.
    assert not np.array_equal(outline,result[5,90])


def test_overlap_is_capped_at_two_regions_on_each_depth():
    image = np.full((80,120,3),80,np.uint8)
    def render(count, depth):
        page = make_page(children=[])
        parent = page
        for _ in range(depth-1):
            child = make_region('block', polygon=[(5,5),(115,5),(115,75),(5,75)])
            parent.children = (child,)
            parent = child
        parent.children = tuple(make_region('word',polygon=[(10,10),(110,10),(110,70),(10,70)])
                                for _ in range(count))
        return np.asarray(visualize_page(image,page, numbers_levels=[]))[40,60]
    for depth in (1,2,3):
        assert np.array_equal(render(2,depth),render(5,depth))


def test_markers_select_depth_and_empty_list_disables():
    block = make_region('block',polygon=[(5,5),(115,5),(115,75),(5,75)])
    block.children = (make_region('word',polygon=[(15,15),(40,15),(40,40),(15,40)]),)
    page = make_page(children=[block])
    image = np.full((80,120,3),100,np.uint8)
    plain = np.asarray(visualize_page(image,page, numbers_levels=[]))
    assert np.array_equal(plain,np.asarray(visualize_page(image,page,numbers_levels=[],connections_levels=[])))
    assert not np.array_equal(plain,np.asarray(visualize_page(image,page,numbers_levels=[2])))
    assert np.array_equal(plain,np.asarray(visualize_page(image,page,numbers_levels=[9])))


def test_default_marker_levels_and_multiple_selected_depths():
    block = make_region('block',polygon=[(5,5),(115,5),(115,75),(5,75)])
    block.children = (make_region('word',polygon=[(15,15),(40,15),(40,40),(15,40)]),)
    page = make_page(children=[block])
    image = np.full((80,120,3),100,np.uint8)
    default = np.asarray(visualize_page(image,page))
    explicit = np.asarray(visualize_page(image,page,numbers_levels=[1,2],connections_levels=[]))
    assert np.array_equal(default,explicit)
    assert not np.array_equal(default,np.asarray(visualize_page(image,page,numbers_levels=[])))


def test_upscale_normalizes_long_side_and_preserves_aspect_ratio():
    page = make_page(children=[])
    image = np.full((80,120,3),255,np.uint8)
    assert visualize_page(image,page,max_size=240).size == (120,80)
    assert visualize_page(image,page,max_size=240,upscale=True).size == (240,160)
    assert visualize_page(image,page,max_size=60,upscale=True).size == (60,40)


def test_first_level_polygon_reveals_original_with_dark_outline():
    page = Page(image_width=100, image_height=100)
    block = TextRegion(type='block', polygon=[(10,10),(90,10),(90,30),(30,30),(30,90),(10,90)])
    page.children = (block,)
    word = TextRegion(type='word', polygon=[(14,14),(24,14),(24,24),(14,24)])
    block.children = (word,)
    image = np.full((100,100,3),100,np.uint8)
    result = np.asarray(visualize_page(image,page,thickness=1, numbers_levels=[]))
    assert np.all(result[20,60] == 100)
    assert np.all(result[60,20] == 100)
    assert np.all(result[60,60] == 70)
    assert np.all(result[14,14] <= 98)


def test_overlap_is_brighter_and_children_do_not_add_brightness():
    page = Page(image_width=100,image_height=100)
    first = TextRegion(type='block',polygon=[(10,10),(60,10),(60,60),(10,60)])
    second = TextRegion(type='line',polygon=[(40,40),(90,40),(90,90),(40,90)])
    first.children = (TextRegion(type='word',polygon=[(20,20),(30,20),(30,30),(20,30)]),)
    page.children = (first,second)
    image = np.full((100,100,3),100,np.uint8)
    result = np.asarray(visualize_page(image,page, numbers_levels=[]))
    assert np.all(result[5,5] == 70)
    assert result[25,25,2] > result[25,25,1]
    assert np.all(result[50,50] == 115)
    assert np.all(result[80,80] == 100)


def test_invalid_marker_levels_and_thickness_are_rejected():
    import pytest
    page = make_page(children=[])
    image = np.zeros((10, 10, 3), np.uint8)
    for settings in ({"numbers_levels": [0]}, {"connections_levels": [True]},
                     {"numbers_levels": "1"}, {"thickness": 0}, {"thickness": True}):
        with pytest.raises(ValueError):
            visualize_page(image, page, **settings)


def test_visualization_requires_page():
    import pytest
    with pytest.raises(TypeError, match="document-tree Page"):
        visualize_page(np.zeros((10, 10, 3), np.uint8), {})
