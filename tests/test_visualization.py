import numpy as np
from PIL import Image

from manuscript.data import Block, Line, Page, TextSpan
from manuscript.utils import visualize_page


def test_visualize_page_accepts_polygon_with_more_than_four_points():
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    text_span = TextSpan(
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
        order=0,
    )
    page = Page(
        blocks=[Block(lines=[Line(text_spans=[text_span], order=0)], order=0)]
    )

    result = visualize_page(
        image,
        page,
        show_order=True,
        show_lines=True,
        show_numbers=True,
        max_size=1024,
    )

    assert isinstance(result, Image.Image)
    assert result.size == (120, 80)


def test_visualize_page_accepts_block_text_spans_with_polygon_mask_shape():
    image = np.full((100, 140, 3), 255, dtype=np.uint8)
    text_span = TextSpan(
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
    page = Page(blocks=[Block(lines=[], text_spans=[text_span], order=0)])

    result = visualize_page(image, page, show_order=True)

    assert isinstance(result, Image.Image)
    assert result.size == (140, 100)
