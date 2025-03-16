import numpy as np
import cv2
from typing import Optional, List


def get_rotate_mat(theta: float) -> np.ndarray:
    """Возвращает матрицу поворота для угла theta (в радианах)."""
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
