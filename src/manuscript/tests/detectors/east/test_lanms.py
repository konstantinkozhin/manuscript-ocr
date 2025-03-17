import numpy as np
from manuscript.detectors.east.lanms import (
    polygon_area,
    compute_intersection,
    clip_polygon,
    polygon_intersection,
    polygon_iou,
    should_merge,
    normalize_polygon,
    PolygonMerger,
    standard_nms,
    locality_aware_nms,
)


# --- Тесты для геометрических функций ---


def test_polygon_area_square():
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    area = polygon_area(poly)
    np.testing.assert_allclose(area, 1.0, rtol=1e-5)


def test_polygon_area_triangle():
    poly = np.array([[0, 0], [2, 0], [0, 2]], dtype=np.float64)
    area = polygon_area(poly)
    np.testing.assert_allclose(area, 2.0, rtol=1e-5)


def test_compute_intersection():
    # Пересечение двух отрезков, должно дать точку (1,1)
    p1 = np.array([0, 0], dtype=np.float64)
    p2 = np.array([2, 2], dtype=np.float64)
    A = np.array([0, 2], dtype=np.float64)
    B = np.array([2, 0], dtype=np.float64)
    inter = compute_intersection(p1, p2, A, B)
    np.testing.assert_allclose(inter, np.array([1, 1], dtype=np.float64), rtol=1e-5)


def test_clip_polygon():
    subject = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    # Отсекаем по линии x = 2
    A = np.array([2, 5], dtype=np.float64)
    B = np.array([2, -1], dtype=np.float64)
    clipped, count = clip_polygon(subject, A, B)
    expected = np.array([[2, 0], [4, 0], [4, 4], [2, 4]], dtype=np.float64)
    np.testing.assert_allclose(clipped, expected, rtol=1e-5)
    assert count == 4


def test_polygon_intersection():
    poly1 = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    poly2 = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.float64)
    inter_poly = polygon_intersection(poly1, poly2)
    expected = np.array([[2, 2], [4, 2], [4, 4], [2, 4]], dtype=np.float64)
    np.testing.assert_allclose(inter_poly, expected, rtol=1e-5)


def test_polygon_iou():
    poly1 = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    poly2 = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.float64)
    iou = polygon_iou(poly1, poly2)
    expected = 4 / (16 + 16 - 4)  # 4 / 28 ~ 0.142857
    assert np.isclose(iou, expected, rtol=1e-5)


def test_should_merge():
    poly1 = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    poly2 = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.float64)
    # Для порога 0.1 (0.142857 > 0.1) должна вернуться True
    assert should_merge(poly1, poly2, 0.1)
    # Для порога 0.2 (0.142857 < 0.2) должна вернуться False
    assert not should_merge(poly1, poly2, 0.2)


def test_normalize_polygon():
    ref = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    poly = np.array(
        [[4, 4], [0, 4], [0, 0], [4, 0]], dtype=np.float64
    )  # Перестановка точек
    normalized = normalize_polygon(ref, poly)
    np.testing.assert_allclose(normalized, ref, rtol=1e-5)


def test_polygon_merger():
    merger = PolygonMerger()
    poly1 = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    poly2 = np.array([[1, 1], [5, 1], [5, 5], [1, 5]], dtype=np.float64)
    merger.add(poly1, 1.0)
    merger.add(poly2, 2.0)
    merged = merger.get()
    expected = (poly1 + 2 * poly2) / 3
    np.testing.assert_allclose(merged, expected, rtol=1e-5)


# --- Тесты для NMS функций ---


def test_standard_nms():
    # Три бокса, два из них перекрываются
    polys = [
        np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64),
        np.array([[1, 1], [5, 1], [5, 5], [1, 5]], dtype=np.float64),
        np.array([[10, 10], [14, 10], [14, 14], [10, 14]], dtype=np.float64),
    ]
    scores = [0.9, 0.8, 0.7]
    iou_threshold = 0.1
    kept_polys, kept_scores = standard_nms(polys, scores, iou_threshold)
    # Первые два бокса перекрываются, третий не перекрывается => должно остаться 2 бокса
    assert len(kept_polys) == 2


def test_locality_aware_nms():
    # Четыре бокса в формате (n, 9), где каждая строка: [x0, y0, x1, y1, x2, y2, x3, y3, score]
    boxes = np.array(
        [
            [0, 0, 4, 0, 4, 4, 0, 4, 0.9],
            [1, 1, 5, 1, 5, 5, 1, 5, 0.8],
            [10, 10, 14, 10, 14, 14, 10, 14, 0.7],
            [11, 11, 15, 11, 15, 15, 11, 15, 0.6],
        ],
        dtype=np.float32,
    )
    iou_threshold = 0.1
    final_boxes = locality_aware_nms(boxes, iou_threshold)
    # Предполагаем, что первые два бокса объединятся, а вторые объединятся между собой => итог 2 бокса
    assert final_boxes.shape[0] == 2
