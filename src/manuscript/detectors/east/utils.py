import numpy as np
import cv2

# from typing import Optional, List


def get_rotate_mat(theta: float) -> np.ndarray:
    """
    Возвращает матрицу поворота для заданного угла в радианах.

    Функция вычисляет двумерную матрицу поворота, которая используется для поворота точек в плоскости
    на угол `theta`.

    Параметры
    ----------
    theta : float
        Угол поворота в радианах.

    Возвращаемое значение
    ----------------------
    np.ndarray
        Матрица поворота размером 2x2, вычисленная по формуле:
        [[cos(theta), -sin(theta)],
         [sin(theta),  cos(theta)]].

    Примеры
    --------
    >>> import numpy as np
    >>> get_rotate_mat(0)
    array([[1., 0.],
           [0., 1.]])
    >>> import numpy as np
    >>> np.set_printoptions(precision=2)
    >>> get_rotate_mat(np.pi / 2)
    array([[ 0., -1.],
           [ 1.,  0.]])
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def draw_boxes_on_image(
    image: np.ndarray,
    boxes: list[np.ndarray],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
    alpha: float = 0.8,
) -> np.ndarray:
    """
    Отрисовывает боксы (полигоны) на изображении с возможностью задания прозрачности.

    Parameters
    ----------
    image : np.ndarray
        Исходное изображение в формате RGB (трёхканальный массив NumPy).
    boxes : list of np.ndarray
        Список полигонов, представляющих боксы. Каждый полигон — это массив формы (4, 2),
        содержащий координаты четырёх углов бокса.
    color : tuple of int, optional
        Цвет линий бокса в формате (B, G, R). По умолчанию (0, 255, 0) — зелёный.
    thickness : int, optional
        Толщина линии боксов. По умолчанию 1.
    alpha : float, optional
        Коэффициент прозрачности для боксов (0 - полностью прозрачные, 1 - полностью непрозрачные).
        По умолчанию 0.8.

    Returns
    -------
    np.ndarray
        Изображение с нанесёнными боксами.

    Examples
    --------
    >>> import numpy as np
    >>> import cv2
    >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
    >>> boxes = [np.array([[10, 10], [20, 10], [20, 20], [10, 20]])]
    >>> result = draw_boxes_on_image(img, boxes)
    >>> isinstance(result, np.ndarray)
    True
    """
    img_out = image.copy()
    overlay = image.copy()
    for box in boxes:
        pts = np.round(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=thickness)
    cv2.addWeighted(overlay, alpha, img_out, 1 - alpha, 0, img_out)
    return img_out


def shrink_poly(poly, shrink_ratio=0.4):
    """
    Сужает полигон, перемещая его вершины к центру.

    Функция принимает массив координат вершин полигона и сжимает его относительно центра,
    используя указанный коэффициент сжатия.

    Параметры
    ----------
    poly : array_like
        Numpy-массив с координатами вершин полигона. Может быть задан в виде двумерного массива
        формы (N, 2) или в виде одномерного массива с 8 элементами (при условии, что N = 4).
    shrink_ratio : float, optional
        Коэффициент сжатия. Обычно задается равным 0.4.

    Возвращаемое значение
    ----------------------
    shrunk_poly : numpy.ndarray
        Суженный полигон, представленный в виде numpy-массива с формой (N, 2).

    Примеры
    --------
    >>> import numpy as np
    >>> poly = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    >>> shrink_poly(poly, shrink_ratio=0.4)
    array([[0.4, 0.4],
           [1.6, 0.4],
           [1.6, 1.6],
           [0.4, 1.6]], dtype=float32)
    """
    poly = np.array(poly).reshape(-1, 2)
    centroid = np.mean(poly, axis=0)
    shrunk_poly = centroid + (poly - centroid) * (1 - shrink_ratio)
    return shrunk_poly.astype(np.float32)


def compute_box_angle(poly):
    """
    Вычисляет угол поворота текстового блока по заданному полигону.

    Функция упорядочивает четыре точки (предполагается, что они неупорядочены) полигона
    в следующем порядке: [верхний левый, верхний правый, нижний правый, нижний левый],
    а затем вычисляет угол между верхней стороной и горизонтальной осью.
    Результат возвращается в градусах в диапазоне [-90, 90).

    Параметры
    ----------
    poly : array_like
        Массив координат, представляющих вершины полигона.
        Может быть представлен либо в виде двумерного массива формы (N, 2),
        либо в виде одномерного массива формы (8,) (при этом N ожидается равным 4).

    Возвращаемое значение
    ----------------------
    angle : float
        Угол поворота текстового блока в градусах, нормализованный в диапазоне [-90, 90).

    Примечания
    ---------
    Упорядочивание точек осуществляется следующим образом:
      - Верхний левый угол выбирается как точка с наименьшей суммой координат.
      - Нижний правый угол выбирается как точка с наибольшей суммой координат.
      - Верхний правый угол определяется как точка с наименьшей разностью координат (y - x).
      - Нижний левый угол определяется как точка с наибольшей разностью координат (y - x).

    Примеры
    --------
    >>> import numpy as np
    >>> poly = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=np.float32)
    >>> compute_box_angle(poly)
    0.0
    """
    # Сумма координат для нахождения верхнего левого и нижнего правого
    s = poly.sum(axis=1)
    tl = poly[np.argmin(s)]
    br = poly[np.argmax(s)]
    # Разность координат для нахождения верхнего правого и нижнего левого
    diff = np.diff(poly, axis=1)
    tr = poly[np.argmin(diff)]
    bl = poly[np.argmax(diff)]
    # Теперь упорядочим точки
    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    # Верхняя сторона – от tl до tr
    delta = tr - tl
    angle = np.degrees(np.arctan2(delta[1], delta[0]))
    # Приводим угол к диапазону [-90, 90)
    if angle < -90:
        angle += 180
    elif angle >= 90:
        angle -= 180
    return angle
