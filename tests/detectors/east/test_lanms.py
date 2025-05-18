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

import pytest


# --- Tests for geometric functions ---


def test_polygon_area_square():
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    area = polygon_area(poly)
    np.testing.assert_allclose(area, 1.0, rtol=1e-5)


def test_polygon_area_triangle():
    poly = np.array([[0, 0], [2, 0], [0, 2]], dtype=np.float64)
    area = polygon_area(poly)
    np.testing.assert_allclose(area, 2.0, rtol=1e-5)


def test_compute_intersection():
    # Intersection of two segments, should return point (1,1)
    p1 = np.array([0, 0], dtype=np.float64)
    p2 = np.array([2, 2], dtype=np.float64)
    A = np.array([0, 2], dtype=np.float64)
    B = np.array([2, 0], dtype=np.float64)
    inter = compute_intersection(p1, p2, A, B)
    np.testing.assert_allclose(inter, np.array([1, 1], dtype=np.float64), rtol=1e-5)


def test_clip_polygon():
    subject = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    # Clipping by the line x = 2
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
    # IoU threshold 0.1 (0.142857 > 0.1) should return True
    assert should_merge(poly1, poly2, 0.1)
    # IoU threshold 0.2 (0.142857 < 0.2) should return False
    assert not should_merge(poly1, poly2, 0.2)


def test_normalize_polygon():
    ref = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64)
    poly = np.array(
        [[4, 4], [0, 4], [0, 0], [4, 0]], dtype=np.float64
    )  # Permuted vertices
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


# --- Tests for NMS functions ---


def test_standard_nms():
    # Three boxes where two overlap and one does not
    polys = [
        np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64),
        np.array([[1, 1], [5, 1], [5, 5], [1, 5]], dtype=np.float64),
        np.array([[10, 10], [14, 10], [14, 14], [10, 14]], dtype=np.float64),
    ]
    scores = [0.9, 0.8, 0.7]
    iou_threshold = 0.1
    kept_polys, kept_scores = standard_nms(polys, scores, iou_threshold)
    # The first two boxes overlap, the third does not => expect 2 kept boxes
    assert len(kept_polys) == 2


def test_locality_aware_nms():
    # Four boxes in (n,9) format: [x0,y0,x1,y1,x2,y2,x3,y3,score]
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
    # Expect 2 final boxes after merging adjacent overlaps
    assert final_boxes.shape[0] == 2


def test_polygon_area_degenerate():
    # Fewer than three points => area should be 0
    poly = np.array([[0, 0], [1, 0]], dtype=np.float64)
    area = polygon_area(poly)
    assert area == pytest.approx(0.0)


def test_compute_intersection_parallel():
    # Parallel segments => returns starting point
    p1 = np.array([0, 0], dtype=np.float64)
    p2 = np.array([1, 1], dtype=np.float64)
    A = np.array([2, 2], dtype=np.float64)
    B = np.array([3, 3], dtype=np.float64)
    inter = compute_intersection(p1, p2, A, B)
    np.testing.assert_allclose(inter, p1, rtol=1e-5)


def test_polygon_intersection_no_overlap():
    # No overlapping region => empty intersection
    poly1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    poly2 = np.array([[2, 2], [3, 2], [3, 3], [2, 3]], dtype=np.float64)
    inter = polygon_intersection(poly1, poly2)
    assert inter.shape == (0, 2)


def test_polygon_iou_extremes():
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    # Complete overlap => IoU=1.0
    assert polygon_iou(poly, poly) == pytest.approx(1.0)
    # No overlap => IoU=0.0
    other = np.array([[2, 2], [3, 2], [3, 3], [2, 3]], dtype=np.float64)
    assert polygon_iou(poly, other) == pytest.approx(0.0)


def test_should_merge_at_threshold():
    # At threshold boundary
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    assert not should_merge(poly, poly, 1.0)
    assert should_merge(poly, poly, 0.999)


def test_clip_polygon_no_clip():
    # Line far away => no clipping
    subject = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    A = np.array([100, 0], dtype=np.float64)
    B = np.array([100, 1], dtype=np.float64)
    clipped, count = clip_polygon(subject, A, B)
    np.testing.assert_allclose(clipped, subject, rtol=1e-5)
    assert count == subject.shape[0]


def test_clip_polygon_full_clip():
    # Polygon entirely on one side => empty result
    subject = np.array([[1, 1], [2, 1], [2, 2], [1, 2]], dtype=np.float64)
    A = np.array([0, 0], dtype=np.float64)
    B = np.array([0, 1], dtype=np.float64)
    # Полигон справа от линии x=0 => все точки вне
    clipped, count = clip_polygon(subject, A, B)
    assert clipped.shape == (0, 2)
    assert count == 0


def test_normalize_polygon_variants():
    ref = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    variants = []
    # All cyclic shifts and reversals
    for start in range(4):
        variants.append(np.vstack([ref[(i + start) % 4] for i in range(4)]))
        variants.append(np.vstack([ref[(start - i) % 4] for i in range(4)]))
    for var in variants:
        norm = normalize_polygon(ref, var)
        np.testing.assert_allclose(norm, ref, rtol=1e-5)


def test_standard_nms_empty():
    # Empty input => empty output
    kept_polys, kept_scores = standard_nms([], [], 0.5)
    assert kept_polys == []
    assert kept_scores == []


def test_locality_aware_nms_empty():
    # Empty input => empty output
    out = locality_aware_nms(np.zeros((0, 9), dtype=np.float32), 0.5)
    assert out.shape == (0,)


def test_standard_nms_order():
    polys = [np.zeros((4, 2), dtype=np.float64) for _ in range(3)]
    scores = [0.2, 0.9, 0.5]
    kept_polys, kept_scores = standard_nms(polys, scores, 0.1)
    # Check that highest scores are kept first
    assert kept_scores[0] == pytest.approx(0.9)
