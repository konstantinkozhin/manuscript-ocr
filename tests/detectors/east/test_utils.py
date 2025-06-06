import pytest
import numpy as np
from shapely.geometry import Polygon
from manuscript.detectors.east.utils import (
    poly_iou,
    compute_f1
)

# Глобальные переменные, которые использует compute_f1
gt_segs = {}
processed_ids = []

def test_identical_polygons():
    seg = [0, 0, 1, 0, 1, 1, 0, 1]
    assert poly_iou(seg, seg) == 1.0


def test_non_overlapping_polygons():
    segA = [0, 0, 1, 0, 1, 1, 0, 1]
    segB = [2, 2, 3, 2, 3, 3, 2, 3]
    assert poly_iou(segA, segB) == 0.0


def test_partial_overlap():
    segA = [0, 0, 2, 0, 2, 2, 0, 2]
    segB = [1, 1, 3, 1, 3, 3, 1, 3]
    expected_iou = Polygon(np.array(segA).reshape(-1, 2)).intersection(
        Polygon(np.array(segB).reshape(-1, 2))).area / \
        Polygon(np.array(segA).reshape(-1, 2)).union(
        Polygon(np.array(segB).reshape(-1, 2))).area
    assert pytest.approx(poly_iou(segA, segB), 0.01) == expected_iou


def test_invalid_polygon():
    segA = [0, 0, 1, 1, 1, 1]  # Недопустимое количество точек
    segB = [0, 0, 1, 0, 1, 1, 0, 1]
    assert poly_iou(segA, segB) == 0.0


def test_zero_area_union():
    segA = [0, 0, 0, 0, 0, 0, 0, 0]  # Точки совпадают
    segB = [0, 0, 0, 0, 0, 0, 0, 0]
    assert poly_iou(segA, segB) == 0.0


def test_compute_f1_perfect_match():
    global gt_segs, processed_ids
    gt_segs = {"img1": [[0, 0, 1, 0, 1, 1, 0, 1]]}
    processed_ids = ["img1"]
    preds = [{"image_id": "img1", "segmentation": [0, 0, 1, 0, 1, 1, 0, 1]}]
    assert compute_f1(preds, 0.5, gt_segs, processed_ids) == 1.0


def test_compute_f1_no_match():
    gt_segs = {"img1": [[0, 0, 1, 0, 1, 1, 0, 1]]}
    processed_ids = ["img1"]
    preds = [{"image_id": "img1", "segmentation": [2, 2, 3, 2, 3, 3, 2, 3]}]
    assert compute_f1(preds, 0.5, gt_segs, processed_ids) == 0.0


def test_compute_f1_partial_match():
    gt_segs = {"img1": [[0, 0, 2, 0, 2, 2, 0, 2]]}
    processed_ids = ["img1"]
    preds = [{"image_id": "img1", "segmentation": [1, 1, 3, 1, 3, 3, 1, 3]}]
    iou = poly_iou(preds[0]["segmentation"], gt_segs["img1"][0])
    expected_f1 = 1.0 if iou >= 0.5 else 0.0
    assert compute_f1(preds, 0.5, gt_segs, processed_ids) == expected_f1
