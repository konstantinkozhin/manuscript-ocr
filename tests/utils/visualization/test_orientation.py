import numpy as np
from manuscript.data import Page, TextRegion
from manuscript.utils import visualize_page


def test_orientation_labels_are_optional_and_skip_zero_unknown():
    page=Page(image_width=160,image_height=100)
    node=TextRegion(type='line',polygon=[(30,50),(130,50),(130,80),(30,80)])
    node.parent=page
    image=np.zeros((100,160,3),np.uint8)
    plain=np.asarray(visualize_page(image,page, numbers_levels=[]))
    for angle in (None,0):
        node.orientation=angle
        assert np.array_equal(plain,np.asarray(visualize_page(image,page,show_orientation=True, numbers_levels=[])))
    for angle in (90,-90,180,12.5):
        node.orientation=angle
        labeled=np.asarray(visualize_page(image,page,show_orientation=True, numbers_levels=[]))
        assert not np.array_equal(labeled,plain)
        assert np.array_equal(labeled[60:70],plain[60:70])
        assert node.orientation==angle
        assert np.array_equal(np.asarray(visualize_page(image,page, numbers_levels=[])),plain)


def test_hidden_tables_and_cells_do_not_draw_orientation_labels():
    from manuscript.data import Table, Cell
    page = Page(image_width=160, image_height=100)
    polygon = [(30, 50), (130, 50), (130, 80), (30, 80)]
    table = Table(polygon=polygon, orientation=90)
    cell = Cell(row=0, column=0, polygon=polygon, orientation=180)
    table.children = [cell]
    page.children = [table]
    image = np.zeros((100, 160, 3), np.uint8)
    options = dict(numbers_levels=[], show_tables=False, show_cells=False)
    plain = np.asarray(visualize_page(image, page, **options))
    labeled = np.asarray(visualize_page(image, page, show_orientation=True, **options))
    np.testing.assert_array_equal(labeled, plain)


def test_visualization_normalizes_grayscale_without_modifying_input():
    page = Page(image_width=160, image_height=100)
    image = np.full((100, 160), 100, np.uint8)
    before = image.copy()
    actual = visualize_page(image, page, numbers_levels=[])
    expected = visualize_page(np.repeat(image[:, :, None], 3, axis=2), page, numbers_levels=[])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, before)
