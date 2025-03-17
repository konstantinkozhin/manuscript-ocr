import numpy as np
import numpy.testing as npt

from manuscript.detectors.east.utils import get_rotate_mat
from manuscript.detectors.east.utils import draw_boxes_on_image
from manuscript.detectors.east.utils import shrink_poly
from manuscript.detectors.east.utils import compute_box_angle


# ==========================
# Тесты для shrink_poly
# ==========================
def test_compute_box_angle_axis_aligned():
    """
    Test compute_box_angle with an axis-aligned rectangle.
    Ожидаемый угол: 0 градусов.
    """
    # Задаём полигон в произвольном порядке
    poly = np.array([[4, 3], [0, 3], [0, 0], [4, 0]], dtype=np.float32)
    angle = compute_box_angle(poly)
    npt.assert_allclose(angle, 0.0, rtol=1e-5)


def test_compute_box_angle_rotated_30():
    """
    Test compute_box_angle with a rectangle rotated by 30 degrees.
    Ожидаемый угол: 30 градусов.
    """
    # Создаём осесимметричный прямоугольник (центр в (0,0))
    poly = np.array([[-2, -1], [2, -1], [2, 1], [-2, 1]], dtype=np.float32)
    angle_deg = 30.0
    theta = np.radians(angle_deg)
    R = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    poly_rotated = np.dot(poly, R.T)
    # Перемешиваем порядок точек, чтобы проверить устойчивость функции к неупорядоченности
    idx = np.random.permutation(4)
    poly_rotated = poly_rotated[idx]
    computed_angle = compute_box_angle(poly_rotated)
    npt.assert_allclose(computed_angle, angle_deg, rtol=1e-2)


def test_compute_box_angle_negative_45():
    """
    Тест функции compute_box_angle для прямоугольника, повернутого на -45 градусов.
    Ожидаемый угол: 45 градусов (из-за нормализации угла).
    """

    # Создаем упорядоченный полигон: [верхний левый, верхний правый, нижний правый, нижний левый]
    poly = np.array([[-2, 1], [2, 1], [2, -1], [-2, -1]], dtype=np.float32)
    angle_deg = -45.0  # будем вращать на -45 градусов
    theta = np.radians(angle_deg)
    # Создаем матрицу поворота для угла -45 градусов
    R = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    # Поворачиваем полигон
    poly_rotated = np.dot(poly, R.T)
    # Вычисляем угол
    computed_angle = compute_box_angle(poly_rotated)
    # Согласно логике функции, нормализация приводит к тому, что результат равен 45°
    expected_angle = 45.0
    npt.assert_allclose(computed_angle, expected_angle, rtol=1e-2)


# ==========================
# Тесты для shrink_poly
# ==========================
def test_shrink_poly_square():
    # Тест для квадрата с shrink_ratio=0.4
    poly = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)
    shrunk = shrink_poly(poly, shrink_ratio=0.4)
    # Центроид квадрата = [1, 1]
    # Новая точка: [1, 1] + ([original] - [1, 1]) * 0.6
    expected = np.array(
        [
            [1 + (0 - 1) * 0.6, 1 + (0 - 1) * 0.6],  # [0.4, 0.4]
            [1 + (2 - 1) * 0.6, 1 + (0 - 1) * 0.6],  # [1.6, 0.4]
            [1 + (2 - 1) * 0.6, 1 + (2 - 1) * 0.6],  # [1.6, 1.6]
            [1 + (0 - 1) * 0.6, 1 + (2 - 1) * 0.6],  # [0.4, 1.6]
        ],
        dtype=np.float32,
    )
    npt.assert_allclose(shrunk, expected, rtol=1e-5)


def test_shrink_poly_flattened_input():
    # Тест для входа в виде одномерного массива (8 элементов)
    poly = [0, 0, 2, 0, 2, 2, 0, 2]
    shrunk = shrink_poly(poly, shrink_ratio=0.4)
    expected = np.array(
        [[0.4, 0.4], [1.6, 0.4], [1.6, 1.6], [0.4, 1.6]], dtype=np.float32
    )
    npt.assert_allclose(shrunk, expected, rtol=1e-5)


def test_shrink_poly_triangle():
    # Тест для треугольника с другим коэффициентом сжатия (50%)
    poly = np.array([[0, 0], [4, 0], [2, 3]], dtype=np.float32)
    shrink_ratio = 0.5
    shrunk = shrink_poly(poly, shrink_ratio=shrink_ratio)
    centroid = np.mean(poly, axis=0)
    expected = centroid + (poly - centroid) * (1 - shrink_ratio)
    npt.assert_allclose(shrunk, expected, rtol=1e-5)


def test_shrink_poly_dtype():
    # Проверка, что возвращаемый массив имеет тип float32
    poly = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float64)
    shrunk = shrink_poly(poly, shrink_ratio=0.4)
    assert (
        shrunk.dtype == np.float32
    ), "Ожидается, что тип данных результата будет float32"


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
