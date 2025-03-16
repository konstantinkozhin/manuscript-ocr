import numpy as np
from manuscript.detectors.east.utils import get_rotate_mat
from manuscript.detectors.east.utils import draw_boxes_on_image

# ==========================
# Тесты для get_rotate_mat
# ==========================


def test_rotate_identity():
    """Проверка: для theta=0 должна возвращаться единичная матрица."""
    theta = 0.0
    expected = np.eye(2)  # Должна быть единичная матрица
    assert np.allclose(get_rotate_mat(theta), expected, rtol=1e-5)


def test_rotate_pi_over_two():
    """Проверка: для theta=pi/2 должна возвращаться матрица [[0, -1], [1, 0]]."""
    theta = np.pi / 2
    expected = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert np.allclose(get_rotate_mat(theta), expected, atol=1e-6)


def test_rotate_pi():
    """Проверка: для theta=pi должна возвращаться матрица [[-1, 0], [0, -1]]."""
    theta = np.pi
    expected = np.array([[-1.0, 0.0], [0.0, -1.0]])
    assert np.allclose(get_rotate_mat(theta), expected, atol=1e-6)


# ==========================
# Тесты для draw_boxes_on_image
# ==========================


def test_draw_boxes_output_type():
    """Функция должна возвращать изображение в виде np.ndarray."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [np.array([[10, 10], [20, 10], [20, 20], [10, 20]])]
    result = draw_boxes_on_image(img, boxes)
    assert isinstance(result, np.ndarray)


def test_draw_boxes_output_shape():
    """Размерность выходного изображения должна совпадать с входным."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [np.array([[10, 10], [20, 10], [20, 20], [10, 20]])]
    result = draw_boxes_on_image(img, boxes)
    assert result.shape == img.shape


def test_draw_boxes_changes_image():
    """Функция должна изменять изображение при наличии боксов."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [np.array([[10, 10], [20, 10], [20, 20], [10, 20]])]
    result = draw_boxes_on_image(img, boxes)
    assert not np.array_equal(result, img)


def test_draw_boxes_alpha_blending():
    """Проверяем, что функция учитывает прозрачность."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [np.array([[10, 10], [20, 10], [20, 20], [10, 20]])]

    result_high_alpha = draw_boxes_on_image(img, boxes, alpha=1.0)
    result_low_alpha = draw_boxes_on_image(img, boxes, alpha=0.2)

    assert not np.array_equal(result_high_alpha, result_low_alpha)


def test_draw_boxes_empty_list():
    """Если список боксов пуст, изображение не должно изменяться."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = []
    result = draw_boxes_on_image(img, boxes)
    assert np.array_equal(result, img)
