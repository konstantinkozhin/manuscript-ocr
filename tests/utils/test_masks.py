import numpy as np
import pytest
from manuscript.utils.masks import mask_iou, mask_to_polygon


def test_mask_iou_and_empty_geometry():
    mask = np.zeros((10, 10), bool)
    assert mask_iou(mask, mask) == 0
    assert mask_to_polygon(mask) is None
    mask[1:5, 1:5] = True
    assert mask_iou(mask, mask) == 1
    other = np.zeros_like(mask)
    other[6:9, 6:9] = True
    assert mask_iou(mask, other) == 0
    with pytest.raises(ValueError, match='matching shapes'):
        mask_iou(mask, np.zeros((1, 1)))




def test_disconnected_mask_polygon_covers_both_components():
    from shapely.geometry import Polygon
    mask = np.zeros((40, 80), np.uint8)
    mask[10:20, 5:15] = 1
    mask[10:20, 60:70] = 1
    polygon = Polygon(mask_to_polygon(mask))
    from shapely.geometry import Point
    assert polygon.covers(Point(8, 15))
    assert polygon.covers(Point(65, 15))
