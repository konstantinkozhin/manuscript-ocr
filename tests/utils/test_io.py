import numpy as np
import pytest
from PIL import Image

from manuscript.data import Page, from_coco
from manuscript.utils import create_page_from_image, create_page_from_text


def test_create_page_from_image_importable():
    assert create_page_from_image is not None


def test_create_page_from_image_wraps_numpy_array_into_single_word_region():
    image = np.zeros((32, 120, 3), dtype=np.uint8)

    page = create_page_from_image(image)

    assert isinstance(page, Page)
    assert len(page.children) == 1
    assert (page.image_width, page.image_height) == (120, 32)

    span = page.children[0]
    assert span.polygon == [
        (0.0, 0.0),
        (120.0, 0.0),
        (120.0, 32.0),
        (0.0, 32.0),
    ]
    assert span.detection_confidence == 1.0
    assert span.text is None
    assert span.type == "word"
    assert span.parent is page


def test_create_page_from_image_accepts_pil_and_custom_confidence():
    image = Image.fromarray(np.zeros((24, 80, 3), dtype=np.uint8))

    page = create_page_from_image(image, confidence=0.75)

    span = page.children[0]
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

    assert len(page.children) == 2
    assert (page.image_width, page.image_height) == (80, 60)
    assert np.all(canvas[20:30] == 255)
    assert np.all(canvas[:20, 50:] == 255)

    first_span = page.children[0]
    second_span = page.children[1]

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


def test_text_tree_geometry_and_text_order():
    from manuscript import Pipeline
    page = create_page_from_text(["Первое слово", "", "Вторая строка"], confidence=0.8)
    block = page.children[0]
    assert block.type == "block"
    assert [node.type for node in block.children] == ["line", "line"]
    assert [node.text for node in block.children[0].children] == ["Первое", "слово"]
    assert block.children[1].polygon[0][1] == 70
    assert Pipeline.get_text(None, page) == "Первое слово\nВторая строка"
    assert all(node.detection_confidence == 0.8 for node in block.children[0].children)
    assert page.image_height == 100
    assert from_coco(page.to_dict()).to_tree_dict() == page.to_tree_dict()


def test_multiline_string_and_empty_text():
    page = create_page_from_text("один\nдва")
    assert len(page.children[0].children) == 2
    empty = create_page_from_text([])
    assert not empty.children
    assert (empty.image_width, empty.image_height) == (1, 1)


def test_line_crops_and_file_name(tmp_path):
    path = tmp_path / "line.png"
    Image.fromarray(np.zeros((24, 80, 3), np.uint8)).save(path)
    page, image = create_page_from_image(path, region_type="line", return_image=True)
    assert page.file_name == str(path)
    assert page.children[0].type == "line"
    assert image.shape == (24, 80, 3)
    assert from_coco(page.to_dict()).to_tree_dict() == page.to_tree_dict()


def test_invalid_inputs():
    import pytest
    image = np.zeros((24, 80, 3), np.uint8)
    for gap in [-1, True, 1.5]:
        with pytest.raises(ValueError, match="gap"):
            create_page_from_image(image, gap=gap)
    with pytest.raises(ValueError, match="empty"):
        create_page_from_image([])
    with pytest.raises(ValueError):
        create_page_from_image(image, region_type="table")
    with pytest.raises(TypeError, match="strings"):
        create_page_from_text([1])
    with pytest.raises(ValueError):
        create_page_from_text([], confidence=2)


def test_text_page_assigns_default_and_custom_confidence():
    for confidence in (1.0, 0.5):
        page = create_page_from_text(["Привет мир"], confidence=confidence)
        words = page.children[0].children[0].children
        assert [word.text for word in words] == ["Привет", "мир"]
        assert all(word.detection_confidence == confidence
                   and word.recognition_confidence == confidence for word in words)


def test_read_image_decodes_png_bytes_and_unicode_path(tmp_path):
    from io import BytesIO
    from manuscript.utils.io import read_image
    pixels = np.full((5, 7, 3), [255, 32, 10], np.uint8)
    image = Image.fromarray(pixels)
    stream = BytesIO()
    image.save(stream, format="PNG")
    path = tmp_path / "страница.png"
    image.save(path)
    for source in (stream.getvalue(), path, image, pixels):
        np.testing.assert_array_equal(read_image(source), pixels)


def test_read_image_rejects_corrupted_bytes():
    import pytest
    from manuscript.utils.io import read_image
    with pytest.raises(ValueError, match="decode"):
        read_image(b"invalid image")


def test_read_image_normalizes_grayscale_array():
    from manuscript.utils.io import read_image
    value = read_image(np.zeros((12, 20), np.uint8))
    assert value.shape == (12, 20, 3), value.shape


@pytest.mark.parametrize("channels", [None, 1, 4])
def test_read_image_array_color_modes(channels):
    from manuscript.utils.io import read_image
    shape = (4, 6) if channels is None else (4, 6, channels)
    image = np.full(shape, 70, np.uint8)
    actual = read_image(image)
    assert actual.shape == (4, 6, 3)
    np.testing.assert_array_equal(actual, 70)


@pytest.mark.parametrize("image", [np.zeros((4, 6), np.float32), np.zeros((4, 6, 2), np.uint8),
                                    np.zeros((4,), np.uint8), np.zeros((0, 6, 3), np.uint8)])
def test_read_image_rejects_invalid_arrays(image):
    from manuscript.utils.io import read_image
    with pytest.raises(ValueError, match="Image arrays"):
        read_image(image)


def test_read_image_empty_bytes_has_decode_error():
    from manuscript.utils.io import read_image
    with pytest.raises(ValueError, match='decode'):
        read_image(b'')


def test_read_image_empty_file_has_read_error(tmp_path):
    from manuscript.utils.io import read_image
    path = tmp_path / 'empty.png'
    path.touch()
    with pytest.raises(FileNotFoundError, match='Cannot read image'):
        read_image(path)
