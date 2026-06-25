from typing import List, Tuple

import cv2
import numpy as np
from skimage import measure, segmentation


def _squeeze_map(prob_map: np.ndarray) -> np.ndarray:
    arr = np.asarray(prob_map, dtype=np.float32)
    while arr.ndim > 2 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D map, got shape {prob_map.shape}")
    return arr


def decode_instance_maps(
    score_map: np.ndarray,
    boundary_map: np.ndarray,
    center_map: np.ndarray,
    *,
    score_thresh: float = 0.5,
    boundary_thresh: float = 0.5,
    center_thresh: float = 0.35,
    min_area: int = 4,
) -> np.ndarray:
    """
    Decode score/boundary/center maps into a word instance label map.

    Score defines the text support, center creates seeds, and boundary prevents
    seed regions from merging across word edges.
    """
    score = _squeeze_map(score_map)
    boundary = _squeeze_map(boundary_map)
    center = _squeeze_map(center_map)

    if score.shape != boundary.shape or score.shape != center.shape:
        raise ValueError("score, boundary and center maps must have the same shape")

    text_mask = score >= score_thresh
    grow_mask = text_mask & (boundary < boundary_thresh)
    seed_mask = (center >= center_thresh) & text_mask

    markers = measure.label(seed_mask, connectivity=2).astype(np.int32)
    if markers.max() == 0:
        markers = measure.label(grow_mask, connectivity=2).astype(np.int32)
        if markers.max() == 0:
            return np.zeros(score.shape, dtype=np.int32)

    labels = segmentation.watershed(-center, markers=markers, mask=grow_mask)
    labels = labels.astype(np.int32)

    if min_area > 1:
        cleaned = np.zeros_like(labels, dtype=np.int32)
        next_id = 1
        for label_id in range(1, int(labels.max()) + 1):
            mask = labels == label_id
            if int(mask.sum()) < min_area:
                continue
            cleaned[mask] = next_id
            next_id += 1
        labels = cleaned

    return labels


def labels_to_polygons(
    labels: np.ndarray,
    *,
    score_map: np.ndarray = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    min_area: float = 1.0,
) -> List[Tuple[List[Tuple[float, float]], float]]:
    """Convert an instance label map to contour polygons and mask confidence."""
    labels = np.asarray(labels)
    scores = _squeeze_map(score_map) if score_map is not None else None
    if scores is not None and scores.shape != labels.shape:
        raise ValueError("score_map must have the same shape as labels")
    polygons: List[Tuple[List[Tuple[float, float]], float]] = []

    for label_id in range(1, int(labels.max()) + 1):
        mask = (labels == label_id).astype(np.uint8)
        if int(mask.sum()) < min_area:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < min_area:
            continue

        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if approx.shape[0] < 4:
            x, y, w, h = cv2.boundingRect(contour)
            approx = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                dtype=np.float32,
            )

        poly = [
            (float(x) * scale_x, float(y) * scale_y)
            for x, y in approx.astype(np.float32)
        ]
        confidence = float(scores[mask.astype(bool)].mean()) if scores is not None else 1.0
        polygons.append((poly, confidence))

    return polygons
