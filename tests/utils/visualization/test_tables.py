import numpy as np

from manuscript.data import Page, Table, Cell, TextRegion
from manuscript.utils import visualize_page


def rect(x, y, w, h):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def test_empty_merged_cells_draw_without_text_and_scale():
    page = Page(image_width=100, image_height=100)
    table = Table(rows=2, columns=2, polygon=rect(10, 10, 80, 80))
    table.parent = page
    cell = Cell(row=0, column=0, column_span=2, polygon=rect(20, 20, 60, 20))
    cell.parent = table
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = np.asarray(visualize_page(image, page, thickness=1, max_size=50, numbers_levels=[]))
    assert result.shape == (50, 50, 3)
    assert np.all(result[5,5] <= 98)
    assert np.all(result[10,10] <= 98)
    hidden = visualize_page(image, page, show_tables=False, show_cells=False, numbers_levels=[])
    assert not np.asarray(hidden).any()


def test_nested_table_cell_and_word_outlines_are_visible():
    page = Page(image_width=100, image_height=100)
    table = Table(rows=1, columns=2, polygon=rect(5, 5, 90, 90))
    table.parent = page
    cell = Cell(row=0, column=0, column_span=2, polygon=rect(10, 10, 80, 80))
    cell.parent = table
    word = TextRegion(type='word', text='test', polygon=rect(30, 40, 20, 10))
    word.parent = cell
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = np.asarray(visualize_page(image, page, thickness=1, numbers_levels=[]))
    assert np.all(result[40,30] <= 98)
    assert np.all(result[10,10] <= 98)


def test_table_reveals_first_level_polygon_and_can_be_hidden():
    page = Page(image_width=100, image_height=100)
    table = Table(rows=1, columns=1, polygon=rect(5,5,90,90))
    word = TextRegion(type='word', polygon=rect(30,30,20,20))
    table.children = (word,)
    page.children = (table,)
    image = np.full((100,100,3),255,np.uint8)
    result = np.asarray(visualize_page(image,page,thickness=1, numbers_levels=[]))
    assert np.all(result[20,20] == 255)
    assert np.all(result[35,35] < 255)
    assert np.all(result[5,5] <= 98)
    hidden = np.asarray(visualize_page(image,page,show_tables=False,thickness=1, numbers_levels=[]))
    assert np.all(hidden[20,20] == 178)


def test_cell_indices_are_optional_and_respect_hidden_cells():
    page = Page(image_width=100,image_height=100)
    table = Table(rows=1,columns=2,polygon=rect(5,5,90,90))
    table.children = (Cell(row=0,column=0,column_span=2,polygon=rect(10,10,80,70)),)
    page.children = (table,)
    image = np.full((100,100,3),100,np.uint8)
    plain = np.asarray(visualize_page(image,page,numbers_levels=[]))
    labeled = np.asarray(visualize_page(image,page,numbers_levels=[],show_cell_indices=True))
    assert not np.array_equal(plain,labeled)
    hidden = np.asarray(visualize_page(image,page,numbers_levels=[],show_cells=False))
    assert np.array_equal(hidden,np.asarray(visualize_page(
        image,page,numbers_levels=[],show_cells=False,show_cell_indices=True)))
