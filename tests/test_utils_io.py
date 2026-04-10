import numpy as np
from PIL import Image

from manuscript.data import Page
from manuscript.utils import create_page_from_image


def test_create_page_from_image_importable():
    assert create_page_from_image is not None


def test_create_page_from_image_wraps_numpy_array_into_single_text_span():
    image = np.zeros((32, 120, 3), dtype=np.uint8)

    page = create_page_from_image(image)

    assert isinstance(page, Page)
    assert len(page.blocks) == 1
    assert len(page.blocks[0].lines) == 1
    assert len(page.blocks[0].lines[0].text_spans) == 1

    span = page.blocks[0].lines[0].text_spans[0]
    assert span.polygon == [
        (0.0, 0.0),
        (120.0, 0.0),
        (120.0, 32.0),
        (0.0, 32.0),
    ]
    assert span.detection_confidence == 1.0
    assert span.text is None
    assert span.order == 0


def test_create_page_from_image_accepts_pil_and_custom_confidence():
    image = Image.fromarray(np.zeros((24, 80, 3), dtype=np.uint8))

    page = create_page_from_image(image, confidence=0.75)

    span = page.blocks[0].lines[0].text_spans[0]
    assert span.polygon == [
        (0.0, 0.0),
        (80.0, 0.0),
        (80.0, 24.0),
        (0.0, 24.0),
    ]
    assert span.detection_confidence == 0.75


def test_create_page_from_image_can_stack_multiple_images_and_return_canvas():
    image1 = np.zeros((20, 50, 3), dtype=np.uint8)
    image2 = np.zeros((30, 80, 3), dtype=np.uint8)

    page, canvas = create_page_from_image(
        [image1, image2],
        gap=10,
        return_image=True,
    )

    assert isinstance(page, Page)
    assert canvas.shape == (60, 80, 3)

    assert len(page.blocks) == 1
    assert len(page.blocks[0].lines) == 2

    first_span = page.blocks[0].lines[0].text_spans[0]
    second_span = page.blocks[0].lines[1].text_spans[0]

    assert first_span.polygon == [
        (0.0, 0.0),
        (50.0, 0.0),
        (50.0, 20.0),
        (0.0, 20.0),
    ]
    assert second_span.polygon == [
        (0.0, 30.0),
        (80.0, 30.0),
        (80.0, 60.0),
        (0.0, 60.0),
    ]
