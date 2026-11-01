import numpy as np
import pytest
from shapely.geometry import Polygon
from manuscript.utils.geometry import repair_detection_polygon


def test_valid_concave_polygon_is_preserved():
    points = np.array([[0, 0], [4, 0], [4, 4], [2, 2], [0, 4]], float)
    np.testing.assert_array_equal(repair_detection_polygon(points), points)


def test_disconnected_repair_retains_extent_with_rectangle():
    points = np.array([[0, 0], [4, 4], [0, 4], [4, 0]], float)
    result = repair_detection_polygon(points)
    assert len(result) == 4
    assert Polygon(result).is_valid
    assert Polygon(result).bounds == pytest.approx((0, 0, 4, 4))


def test_triangle_gets_valid_four_point_fallback():
    result = repair_detection_polygon([[0, 0], [4, 0], [2, 3]])
    assert len(result) == 4
    assert Polygon(result).area > 0


def test_degenerate_contour_uses_supplied_detection_box():
    result = repair_detection_polygon([[1, 1]], fallback=[[0, 0], [4, 0], [4, 3], [0, 3]])
    assert Polygon(result).area == pytest.approx(12)
    assert repair_detection_polygon([[0, 0], [1, 0]]) is None


def test_nonfinite_output_is_reported():
    with pytest.raises(ValueError, match="finite"):
        repair_detection_polygon([[0, 0], [1, 0], [1, float('nan')], [0, 1]])
