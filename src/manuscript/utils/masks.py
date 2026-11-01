"""Geometry operations for binary instance masks."""
import cv2
import numpy as np
from .geometry import repair_detection_polygon


def mask_iou(first, second):
    first, second = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("Masks must have matching shapes")
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def contours_to_polygon(contours):
    """Return one outline, covering all components when contours are disconnected."""
    if not contours:
        return None
    contour = contours[0] if len(contours) == 1 else np.concatenate(contours)
    polygon = contour.reshape(-1, 2)
    if len(contours) > 1 or len(polygon) < 4:
        polygon = cv2.boxPoints(cv2.minAreaRect(contour.astype(np.float32)))
    return repair_detection_polygon(polygon)


def mask_to_polygon(mask):
    """Extract external mask contours; holes cannot be represented by one polygon."""
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours_to_polygon(contours)
