from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


def _box_iou(
    box1: Union[Tuple[float, float, float, float], np.ndarray],
    box2: Union[Tuple[float, float, float, float], np.ndarray]
) -> float:
    """
    Вычисляет Intersection over Union (IoU) для двух ограничивающих прямоугольников,
    выровненных по осям.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def polygon_to_bbox(
    polygon: Union[np.ndarray, Tuple[Tuple[float, float], ...]],
    image_shape: Optional[Tuple[int, ...]] = None,
    pad: float = 0.0,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Преобразует полигон с произвольным числом вершин в обрезанный ограничивающий
    прямоугольник, выровненный по осям.

    Параметры
    ----------
    polygon : array-like of shape (N, 2)
        Вершины полигона в координатах изображения.
    image_shape : tuple, optional
        Форма исходного изображения для обрезки.
    pad : float, optional
        Дополнительный отступ в пикселях вокруг полигона. По умолчанию ``0``.

    Возвращает
    -------
    tuple or None
        Ограничивающий прямоугольник в виде ``(x1, y1, x2, y2)`` или ``None``,
        если результат недопустим.
    """
    pts = np.asarray(polygon, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
        return None

    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)

    if pad:
        x1 = int(np.floor(x_min - pad))
        y1 = int(np.floor(y_min - pad))
        x2 = int(np.ceil(x_max + pad))
        y2 = int(np.ceil(y_max + pad))
    else:
        # Preserve legacy crop behavior for the default bbox preset.
        x1 = int(x_min)
        y1 = int(y_min)
        x2 = int(x_max)
        y2 = int(y_max)

    if image_shape is not None:
        height, width = image_shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def crop_axis_aligned(
    image: np.ndarray,
    polygon: Union[np.ndarray, Tuple[Tuple[float, float], ...]],
    pad: float = 0.0,
) -> Optional[np.ndarray]:
    """
    Вырезает выровненный по осям прямоугольник, охватывающий полигон.

    Параметры
    ----------
    image : numpy.ndarray
        Исходное изображение.
    polygon : array-like of shape (N, 2)
        Вершины полигона в координатах изображения.
    pad : float, optional
        Дополнительный отступ в пикселях. По умолчанию ``0``.

    Возвращает
    -------
    numpy.ndarray or None
        Вырезанный фрагмент изображения или ``None``, если bbox недопустим.
    """
    bbox = polygon_to_bbox(polygon, image_shape=image.shape, pad=pad)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop.copy()


def crop_polygon_mask(
    image: np.ndarray,
    polygon: Union[np.ndarray, Tuple[Tuple[float, float], ...]],
    pad: float = 0.0,
    background: int = 255,
) -> Optional[np.ndarray]:
    """
    Вырезает ограничивающий прямоугольник полигона и маскирует пиксели за
    пределами полигона.

    Работает с произвольными полигонами формы ``(N, 2)``.

    Параметры
    ----------
    image : numpy.ndarray
        Исходное изображение.
    polygon : array-like of shape (N, 2)
        Вершины полигона в координатах изображения.
    pad : float, optional
        Дополнительный отступ в пикселях. По умолчанию ``0``.
    background : int, optional
        Значение пикселей фона вне полигона. По умолчанию ``255``.

    Возвращает
    -------
    numpy.ndarray or None
        Вырезанный фрагмент с замаскированными пикселями или ``None``,
        если bbox недопустим.
    """
    pts = np.asarray(polygon, dtype=np.float32)
    bbox = polygon_to_bbox(pts, image_shape=image.shape, pad=pad)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None

    shifted = pts.copy()
    shifted[:, 0] -= x1
    shifted[:, 1] -= y1

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [shifted.astype(np.int32)], 255)

    result = np.full_like(crop, background)
    if crop.ndim == 2:
        result[mask == 255] = crop[mask == 255]
    else:
        result[mask == 255] = crop[mask == 255]
    return result


def order_quad_points(
    points: Union[np.ndarray, Tuple[Tuple[float, float], ...]]
) -> np.ndarray:
    """
    Упорядочивает ровно 4 точки полигона в порядке: верхний левый, верхний правый,
    нижний правый, нижний левый.

    Параметры
    ----------
    points : array-like of shape (4, 2)
        Четыре точки полигона в произвольном порядке.

    Возвращает
    -------
    numpy.ndarray of shape (4, 2)
        Точки, упорядоченные по часовой стрелке, начиная с верхнего левого угла.

    Raises
    ------
    ValueError
        Если передано не ровно 4 точки.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected 4 points with shape (4, 2), got: {pts.shape}")

    rect = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]

    diffs = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def warp_quad(
    image: np.ndarray,
    polygon: Union[np.ndarray, Tuple[Tuple[float, float], ...]],
    output_size: Optional[Tuple[int, int]] = None,
    background: int = 255,
) -> Optional[np.ndarray]:
    """
    Применяет перспективное преобразование к четырёхугольному полигону и возвращает
    выпрямленный кроп.

    Функция намеренно предназначена только для четырёхугольников. Для полигонов с
    другим числом вершин возвращает ``None``, чтобы вызывающий код мог выбрать
    запасную стратегию.

    Параметры
    ----------
    image : numpy.ndarray
        Исходное изображение.
    polygon : array-like of shape (4, 2)
        Четыре вершины четырёхугольника.
    output_size : tuple of (int, int), optional
        Целевой размер выходного кропа ``(ширина, высота)``. Если ``None``,
        размер вычисляется автоматически на основе длин сторон полигона.
    background : int, optional
        Значение пикселей фона. По умолчанию ``255``.

    Возвращает
    -------
    numpy.ndarray or None
        Выпрямленный кроп или ``None``, если полигон не является четырёхугольником
        или результат пустой.
    """
    pts = np.asarray(polygon, dtype=np.float32)
    if pts.shape != (4, 2):
        return None

    rect = order_quad_points(pts)

    if output_size is None:
        top_width = np.linalg.norm(rect[1] - rect[0])
        bottom_width = np.linalg.norm(rect[2] - rect[3])
        left_height = np.linalg.norm(rect[3] - rect[0])
        right_height = np.linalg.norm(rect[2] - rect[1])
        width = max(int(round(max(top_width, bottom_width))), 1)
        height = max(int(round(max(left_height, right_height))), 1)
    else:
        width = max(int(output_size[0]), 1)
        height = max(int(output_size[1]), 1)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    border_value = background if image.ndim == 2 else (background, background, background)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    if warped.size == 0:
        return None
    return warped


def merge_polygons(
    polygons: Sequence[Union[np.ndarray, Tuple[Tuple[float, float], ...]]],
    method: str = "bbox",
) -> Optional[List[Tuple[float, float]]]:
    """
    Объединяет несколько полигонов в один.

    Параметры
    ----------
    polygons : sequence of array-like polygons
        Входные полигоны формы ``(N, 2)``.
    method : {"bbox", "convex_hull"}, optional
        Стратегия объединения. ``"bbox"`` возвращает выровненный по осям прямоугольник,
        охватывающий все точки. ``"convex_hull"`` возвращает выпуклую оболочку над всеми
        точками.

    Возвращает
    -------
    list of tuple or None
        Объединённый полигон или ``None``, если ``polygons`` пустой.

    Raises
    ------
    ValueError
        Если какой-либо полигон имеет недопустимую форму или передан неизвестный метод.
    """
    if not polygons:
        return None

    normalized = []
    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
            raise ValueError("Each polygon must have shape (N, 2)")
        normalized.append(pts)

    points = np.concatenate(normalized, axis=0)

    if method == "bbox":
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        return [
            (float(x_min), float(y_min)),
            (float(x_max), float(y_min)),
            (float(x_max), float(y_max)),
            (float(x_min), float(y_max)),
        ]

    if method == "convex_hull":
        hull = cv2.convexHull(points)
        if hull is None or hull.size == 0:
            return None
        return [(float(x), float(y)) for x, y in hull.reshape(-1, 2)]

    raise ValueError(f"method must be 'bbox' or 'convex_hull', got: {method}")
