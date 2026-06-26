from collections import deque
from typing import List, Tuple

import cv2
import numpy as np


def pse(kernels: np.ndarray, min_area: int = 16) -> np.ndarray:
    """
    Progressive Scale Expansion post-processing.

    This follows the PAN++/PSENet implementation: label the smallest kernel,
    then expand labels progressively through larger kernels up to the text map.
    ``kernels`` must be ordered from largest/text map to smallest kernel.
    """
    kernels = np.asarray(kernels, dtype=np.uint8)
    if kernels.ndim != 3:
        raise ValueError(f"kernels must have shape (K,H,W), got {kernels.shape}")
    if kernels.shape[0] < 2:
        raise ValueError("PSE requires at least text map + one kernel map")

    kernel_num = kernels.shape[0]
    label_num, label = cv2.connectedComponents(kernels[-1], connectivity=4)
    label = label.astype(np.int32)

    for label_idx in range(1, label_num):
        if int(np.sum(label == label_idx)) < min_area:
            label[label == label_idx] = 0

    pred = np.zeros_like(label, dtype=np.int32)
    queue = deque()
    points = np.array(np.where(label > 0)).transpose((1, 0))
    for x, y in points:
        pred[x, y] = label[x, y]
        queue.append((int(x), int(y)))

    next_queue = deque()
    h, w = label.shape
    for kernel_idx in range(kernel_num - 2, -1, -1):
        while queue:
            x, y = queue.popleft()
            cur_label = pred[x, y]
            is_edge = True
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= h or ny < 0 or ny >= w:
                    continue
                if kernels[kernel_idx, nx, ny] == 0 or pred[nx, ny] > 0:
                    continue
                pred[nx, ny] = cur_label
                queue.append((nx, ny))
                is_edge = False
            if is_edge:
                next_queue.append((x, y))
        queue, next_queue = next_queue, queue

    return pred


def labels_to_polygons(
    labels: np.ndarray,
    score_map: np.ndarray,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    min_area: int = 16,
    min_score: float = 0.85,
    bbox_type: str = "poly",
) -> List[Tuple[List[Tuple[float, float]], float]]:
    labels = np.asarray(labels, dtype=np.int32)
    score_map = np.asarray(score_map, dtype=np.float32)
    polygons: List[Tuple[List[Tuple[float, float]], float]] = []

    for label_id in range(1, int(labels.max()) + 1):
        mask = labels == label_id
        if int(mask.sum()) < min_area:
            continue
        score = float(score_map[mask].mean()) if np.any(mask) else 0.0
        if score < min_score:
            continue

        points = np.array(np.where(mask)).transpose((1, 0))
        if bbox_type == "rect":
            rect = cv2.minAreaRect(points[:, ::-1].astype(np.float32))
            bbox = cv2.boxPoints(rect)
        else:
            binary = np.zeros(labels.shape, dtype=np.uint8)
            binary[mask] = 1
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            bbox = max(contours, key=cv2.contourArea).reshape(-1, 2)
            if bbox.shape[0] < 4:
                x, y, w, h = cv2.boundingRect(bbox.reshape(-1, 1, 2))
                bbox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)

        poly = [(float(x) * scale_x, float(y) * scale_y) for x, y in bbox.astype(np.float32)]
        polygons.append((poly, score))

    return polygons
