import numpy as np
from numba import njit, float64, int64, boolean
from numba.experimental import jitclass
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


spec = [("sum_poly", float64[:, :]), ("total_score", float64), ("count", int64)]


@jitclass(spec)
class PolygonMerger(object):
    def __init__(self):
        self.sum_poly = np.zeros((4, 2), dtype=np.float64)
        self.total_score = 0.0
        self.count = 0

    def add(self, poly, score):
        if self.count > 0:
            ref = self.get()
            poly = normalize_polygon(ref, poly)
        self.sum_poly += poly * score
        self.total_score += score
        self.count += 1

    def get(self):
        return self.sum_poly / self.total_score


def standard_nms(polys, scores, iou_threshold):
    if len(polys) == 0:
        return [], []
    order = np.argsort(-np.array(scores))
    keep = []
    suppressed = np.zeros(len(polys), dtype=np.bool_)
    for i in range(len(order)):
        if suppressed[order[i]]:
            continue
        keep.append(order[i])
        for j in range(i + 1, len(order)):
            if suppressed[order[j]]:
                continue
            if should_merge(polys[order[i]], polys[order[j]], iou_threshold):
                suppressed[order[j]] = True
    kept_polys = [polys[i] for i in keep]
    kept_scores = [scores[i] for i in keep]
    return kept_polys, kept_scores


def locality_aware_nms(boxes, iou_threshold):
    """
    Вход:
      boxes — numpy-массив shape (n,9), где каждая строка задается как:
              [x0, y0, x1, y1, x2, y2, x3, y3, score]
      iou_threshold — порог для объединения (IoU)
    Выход:
      numpy-массив итоговых боксов в формате (m,9)
    """
    n = boxes.shape[0]
    polys = []
    scores = []
    for i in range(n):
        poly = boxes[i, :8].reshape((4, 2)).astype(np.float64)
        score = boxes[i, 8]
        if len(polys) > 0:
            last_poly = polys[-1]
            if should_merge(poly, last_poly, iou_threshold):
                merger = PolygonMerger()
                merger.add(last_poly, 1.0)
                merger.add(poly, score)
                merged_poly = merger.get()
                polys[-1] = merged_poly
                scores[-1] += score
            else:
                polys.append(poly)
                scores.append(score)
        else:
            polys.append(poly)
            scores.append(score)
    kept_polys, kept_scores = standard_nms(polys, scores, iou_threshold)
    final_boxes = []
    for poly, score in zip(kept_polys, kept_scores):
        poly_rounded = np.round(poly).astype(np.int32)
        final_boxes.append(
            np.concatenate(
                [poly_rounded.flatten(), np.array([score], dtype=np.float32)]
            )
        )
    return np.array(final_boxes)
