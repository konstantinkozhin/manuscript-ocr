import numpy as np
import pytest

import manuscript.utils.geometry as geometry
from manuscript.utils import crop_axis_aligned, crop_polygon_mask, polygon_to_bbox, warp_quad


def test_polygon_to_bbox_accepts_polygon_with_more_than_four_points():
    polygon = np.array(
        [
            [5.0, 5.0],
            [15.0, 5.0],
            [25.0, 10.0],
            [20.0, 20.0],
            [10.0, 25.0],
            [0.0, 15.0],
        ],
        dtype=np.float32,
    )

    assert polygon_to_bbox(polygon) == (0, 5, 25, 25)


def test_crop_polygon_mask_accepts_polygon_with_more_than_four_points():
    polygon = np.array(
        [
            [10.0, 10.0],
            [30.0, 10.0],
            [40.0, 20.0],
            [30.0, 30.0],
            [10.0, 30.0],
            [0.0, 20.0],
        ],
        dtype=np.float32,
    )
    image = np.full((48, 48, 3), [10, 20, 30], dtype=np.uint8)

    crop = crop_polygon_mask(image, polygon, background=255)

    assert crop is not None
    assert crop[0, 0].tolist() == [255, 255, 255]
    assert crop[10, 20].tolist() == [10, 20, 30]


def test_warp_quad_returns_none_for_non_quad_polygon():
    polygon = np.array(
        [
            [10.0, 10.0],
            [30.0, 10.0],
            [40.0, 20.0],
            [30.0, 30.0],
            [10.0, 30.0],
        ],
        dtype=np.float32,
    )
    image = np.zeros((48, 48, 3), dtype=np.uint8)

    assert warp_quad(image, polygon) is None


def test_box_iou_returns_expected_overlap_ratio():
    iou = geometry._box_iou((0, 0, 10, 10), (5, 0, 15, 10))
    assert iou == pytest.approx(50 / 150)


def test_box_iou_returns_zero_for_degenerate_union():
    assert geometry._box_iou((0, 0, 0, 10), (0, 0, 0, 10)) == 0.0


def test_polygon_to_bbox_returns_none_for_invalid_polygon_shape():
    polygon = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert polygon_to_bbox(polygon) is None


def test_polygon_to_bbox_applies_padding_and_image_clipping():
    polygon = np.array(
        [[1.2, 1.3], [3.1, 1.2], [3.0, 3.2], [1.0, 3.0]],
        dtype=np.float32,
    )

    assert polygon_to_bbox(polygon, image_shape=(4, 4, 3), pad=2.0) == (0, 0, 4, 4)


def test_polygon_to_bbox_returns_none_after_clipping_to_empty_box():
    polygon = np.array(
        [[-5.0, -5.0], [-1.0, -5.0], [-1.0, -1.0], [-5.0, -1.0]],
        dtype=np.float32,
    )

    assert polygon_to_bbox(polygon, image_shape=(10, 10, 3)) is None


def test_crop_axis_aligned_returns_none_for_invalid_bbox():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    polygon = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    assert crop_axis_aligned(image, polygon) is None


def test_crop_axis_aligned_returns_none_for_empty_slice(monkeypatch):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(geometry, "polygon_to_bbox", lambda *args, **kwargs: (2, 2, 2, 5))

    assert crop_axis_aligned(image, polygon) is None


def test_crop_polygon_mask_returns_none_for_invalid_bbox():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    polygon = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    assert crop_polygon_mask(image, polygon) is None


def test_crop_polygon_mask_returns_none_for_empty_slice(monkeypatch):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(geometry, "polygon_to_bbox", lambda *args, **kwargs: (2, 2, 2, 5))

    assert crop_polygon_mask(image, polygon) is None


def test_crop_polygon_mask_supports_grayscale_images():
    polygon = np.array(
        [[2.0, 2.0], [6.0, 2.0], [6.0, 6.0], [2.0, 6.0]],
        dtype=np.float32,
    )
    image = np.full((8, 8), 7, dtype=np.uint8)

    crop = crop_polygon_mask(image, polygon, background=255)

    assert crop is not None
    assert crop[0, 0] == 7
    assert crop.dtype == np.uint8


def test_order_quad_points_rejects_non_quad_polygon():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="Expected 4 points"):
        geometry.order_quad_points(points)


def test_warp_quad_respects_explicit_output_size():
    polygon = np.array(
        [[5.0, 5.0], [25.0, 5.0], [25.0, 15.0], [5.0, 15.0]],
        dtype=np.float32,
    )
    image = np.full((32, 32, 3), 120, dtype=np.uint8)

    warped = warp_quad(image, polygon, output_size=(8, 4))

    assert warped is not None
    assert warped.shape == (4, 8, 3)


def test_warp_quad_returns_none_for_empty_warp(monkeypatch):
    polygon = np.array(
        [[5.0, 5.0], [25.0, 5.0], [25.0, 15.0], [5.0, 15.0]],
        dtype=np.float32,
    )
    image = np.full((32, 32, 3), 120, dtype=np.uint8)

    monkeypatch.setattr(geometry.cv2, "warpPerspective", lambda *args, **kwargs: np.empty((0, 0, 3), dtype=np.uint8))

    assert warp_quad(image, polygon) is None
