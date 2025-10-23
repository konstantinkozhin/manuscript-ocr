import numpy as np
from numba import njit, float64, int64
from numba.types import Tuple


@njit("float64(float64[:,:])")
def polygon_area(poly):
    area = 0.0
    n = poly.shape[0]
    for i in range(n):
        j = (i + 1) % n
        area += poly[i, 0] * poly[j, 1] - poly[j, 0] * poly[i, 1]
    return np.abs(area) / 2.0


@njit("float64[:](float64[:], float64[:], float64[:], float64[:])")
def compute_intersection(p1, p2, A, B):
    BAx = p2[0] - p1[0]
    BAy = p2[1] - p1[1]
    DCx = B[0] - A[0]
    DCy = B[1] - A[1]
    denom = BAx * DCy - BAy * DCx
    CAx = A[0] - p1[0]
    CAy = A[1] - p1[1]
    if denom == 0:
        return p1
    t = (CAx * DCy - CAy * DCx) / denom
    return np.array([p1[0] + t * BAx, p1[1] + t * BAy])


@njit(Tuple((float64[:, :], int64))(float64[:, :], float64[:], float64[:]))
def clip_polygon(subject, A, B):
    out = np.empty((20, 2), dtype=np.float64)
    count = 0
    n = subject.shape[0]
    for i in range(n):
        curr = subject[i]
        prev = subject[(i - 1) % n]
        curr_inside = (B[0] - A[0]) * (curr[1] - A[1]) - (B[1] - A[1]) * (
            curr[0] - A[0]
        ) >= 0
        prev_inside = (B[0] - A[0]) * (prev[1] - A[1]) - (B[1] - A[1]) * (
            prev[0] - A[0]
        ) >= 0
        if curr_inside:
            if not prev_inside:
                inter = compute_intersection(prev, curr, A, B)
                out[count] = inter
                count += 1
            out[count] = curr
            count += 1
        elif prev_inside:
            inter = compute_intersection(prev, curr, A, B)
            out[count] = inter
            count += 1
    return out[:count], count


@njit("float64[:,:](float64[:,:], float64[:,:])")
def polygon_intersection(poly1, poly2):
    n = poly1.shape[0]
    current = poly1.copy()
    current_count = n
    m = poly2.shape[0]
    for i in range(m):
        A = poly2[i]
        B = poly2[(i + 1) % m]
        clipped, clipped_count = clip_polygon(current[:current_count], A, B)
        current = clipped
        current_count = clipped_count
        if current_count == 0:
            break
    result = np.empty((current_count, 2), dtype=np.float64)
    for i in range(current_count):
        result[i] = current[i]
    return result


@njit("float64(float64[:,:], float64[:,:])")
def polygon_iou(poly1, poly2):
    inter_poly = polygon_intersection(poly1, poly2)
    inter_area = 0.0
    if inter_poly.shape[0] > 2:
        inter_area = polygon_area(inter_poly)
    area1 = polygon_area(poly1)
    area2 = polygon_area(poly2)
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


@njit("boolean(float64[:,:], float64[:,:], float64)")
def should_merge(poly1, poly2, iou_threshold):
    return polygon_iou(poly1, poly2) > iou_threshold


@njit("float64[:,:](float64[:,:], float64[:,:])")
def normalize_polygon(ref, poly):
    best_order = 0
    best_start = 0
    min_d = 1e20
    for start in range(4):
        d = 0.0
        for i in range(4):
            dx = ref[i, 0] - poly[(start + i) % 4, 0]
            dy = ref[i, 1] - poly[(start + i) % 4, 1]
            d += dx * dx + dy * dy
        if d < min_d:
            min_d = d
            best_start = start
            best_order = 0
    for start in range(4):
        d = 0.0
        for i in range(4):
            idx = (start - i) % 4
            d += (ref[i, 0] - poly[idx, 0]) ** 2 + (ref[i, 1] - poly[idx, 1]) ** 2
        if d < min_d:
            min_d = d
            best_start = start
            best_order = 1
    new_poly = np.empty((4, 2), dtype=np.float64)
    if best_order == 0:
        for i in range(4):
            new_poly[i] = poly[(best_start + i) % 4]
    else:
        for i in range(4):
            new_poly[i] = poly[(best_start - i) % 4]
    return new_poly


def standard_nms(polys, scores, iou_threshold):
    polys_arr = np.ascontiguousarray(polys, dtype=np.float64)
    scores_arr = np.ascontiguousarray(scores, dtype=np.float64)
    if polys_arr.size == 0:
        return polys_arr, scores_arr
    order = np.argsort(-scores_arr)
    keep_idx = []
    suppressed = np.zeros(polys_arr.shape[0], dtype=np.bool_)
    for i in range(order.shape[0]):
        idx = order[i]
        if suppressed[idx]:
            continue
        keep_idx.append(idx)
        for j in range(i + 1, order.shape[0]):
            idx_j = order[j]
            if suppressed[idx_j]:
                continue
            if should_merge(polys_arr[idx], polys_arr[idx_j], iou_threshold):
                suppressed[idx_j] = True
    keep_idx = np.array(keep_idx, dtype=np.int64)
    return polys_arr[keep_idx], scores_arr[keep_idx]


def locality_aware_nms(boxes, iou_threshold):
    """
    boxes — numpy-массив shape (n,9), где каждая строка:
            [x0, y0, x1, y1, x2, y2, x3, y3, score]
    iou_threshold — порог для объединения (IoU)
    Возвращает итоговый numpy-массив боксов (m,9).
    """
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 9), dtype=np.float32)

    boxes_sorted = np.ascontiguousarray(boxes, dtype=np.float64)[
        np.argsort(boxes[:, 0])
    ]

    merged_polys = []
    merged_scores = []
    weight_sums = []

    for box in boxes_sorted:
        poly = box[:8].reshape((4, 2))
        score = float(box[8])

        if merged_polys:
            last_poly = merged_polys[-1]
            if should_merge(poly, last_poly, iou_threshold):
                aligned_poly = normalize_polygon(last_poly, poly)
                total_weight = weight_sums[-1] + score
                merged_polys[-1] = (
                    last_poly * weight_sums[-1] + aligned_poly * score
                ) / total_weight
                weight_sums[-1] = total_weight
                merged_scores[-1] = max(merged_scores[-1], score)
                continue

        merged_polys.append(poly.copy())
        merged_scores.append(score)
        weight_sums.append(score)

    merged_polys_arr = np.stack(merged_polys) if merged_polys else np.empty((0, 4, 2))
    merged_scores_arr = np.array(merged_scores, dtype=np.float64)

    kept_polys, kept_scores = standard_nms(
        merged_polys_arr, merged_scores_arr, iou_threshold
    )

    if kept_polys.size == 0:
        return np.zeros((0, 9), dtype=np.float32)

    final_boxes = np.concatenate(
        [kept_polys.reshape(kept_polys.shape[0], -1), kept_scores[:, None]], axis=1
    )
    return final_boxes.astype(np.float32)
