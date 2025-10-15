import cv2
import numpy as np
import torch
from .lanms import locality_aware_nms
from tqdm import tqdm
from shapely.geometry import Polygon
import json
from collections import defaultdict
import time


def fast_nms(boxes, iou_threshold=0.2):
    """
    Быстрый классический NMS для quad-боксов.
    O(n²) сложность вместо O(n³) у locality_aware_nms.
    
    Args:
        boxes: np.array shape (N, 9) - [x0, y0, x1, y1, x2, y2, x3, y3, score]
        iou_threshold: float - порог IoU для подавления
        
    Returns:
        np.array - отфильтрованные боксы
    """
    if len(boxes) == 0:
        return np.zeros((0, 9), dtype=np.float32)
    
    # Сортируем по score (убывание) - гарантирует качество
    scores = boxes[:, 8]
    sorted_indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(sorted_indices) > 0:
        # Берем бокс с максимальным score
        current_idx = sorted_indices[0]
        keep.append(current_idx)
        
        if len(sorted_indices) == 1:
            break
            
        # Быстро вычисляем IoU через bounding rectangles
        current_box = boxes[current_idx]
        remaining_indices = sorted_indices[1:]
        
        # Конвертируем quad в bounding rect
        curr_coords = current_box[:8].reshape(4, 2)
        curr_x1, curr_y1 = curr_coords.min(axis=0)
        curr_x2, curr_y2 = curr_coords.max(axis=0)
        curr_area = (curr_x2 - curr_x1) * (curr_y2 - curr_y1)
        
        # Векторизованное вычисление IoU для всех оставшихся боксов
        other_boxes = boxes[remaining_indices]
        other_coords = other_boxes[:, :8].reshape(-1, 4, 2)
        
        other_mins = other_coords.min(axis=1)  # (N, 2)
        other_maxs = other_coords.max(axis=1)  # (N, 2)
        
        other_x1s, other_y1s = other_mins[:, 0], other_mins[:, 1]  
        other_x2s, other_y2s = other_maxs[:, 0], other_maxs[:, 1]
        other_areas = (other_x2s - other_x1s) * (other_y2s - other_y1s)
        
        # Векторизованное пересечение
        inter_x1s = np.maximum(curr_x1, other_x1s)
        inter_y1s = np.maximum(curr_y1, other_y1s)  
        inter_x2s = np.minimum(curr_x2, other_x2s)
        inter_y2s = np.minimum(curr_y2, other_y2s)
        
        # Только положительные пересечения
        inter_ws = np.maximum(0, inter_x2s - inter_x1s)
        inter_hs = np.maximum(0, inter_y2s - inter_y1s)
        inter_areas = inter_ws * inter_hs
        
        # IoU
        union_areas = curr_area + other_areas - inter_areas
        ious = np.where(union_areas > 0, inter_areas / union_areas, 0)
        
        # Оставляем только боксы с IoU ≤ threshold
        keep_mask = ious <= iou_threshold
        sorted_indices = remaining_indices[keep_mask]
    
    return boxes[keep]


def convert_rboxes_to_quad_boxes(rboxes, scores=None):
    quad_boxes = []
    if scores is None:
        scores = np.ones(len(rboxes), dtype=np.float32)
    for i, r in enumerate(rboxes):
        cx, cy, w, h, angle = r
        pts = cv2.boxPoints(((cx, cy), (w, h), angle))
        quad = np.concatenate([pts.flatten(), [scores[i]]]).astype(np.float32)
        quad_boxes.append(quad)
    return np.array(quad_boxes, dtype=np.float32)


def quad_to_rbox(quad):
    pts = quad[:8].reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    return np.array([cx, cy, w, h, angle], dtype=np.float32)


def tensor_to_image(tensor):
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5) + 0.5
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def draw_quads(
    image: np.ndarray,
    quads: np.ndarray,
    color: tuple = (0, 0, 0),
    thickness: int = 1,
    dark_alpha: float = 0.5,
    blur_ksize: int = 11,
) -> np.ndarray:
    """
    Рисует надписи в стиле EAST:
      - затемняет фон на dark_alpha,
      - внутри полигонов оставляет исходное изображение,
      - рисует контуры толщиной thickness и цветом color,
      - размывает границу полигонов blur_ksize.

    Args:
        image: H×W×3, исходный BGR или RGB.
        quads: N×M, массив, где первые 8 значений каждой строки —
               это [x1,y1,...,x4,y4].
        color: кортеж BGR/RGB для контура.
        thickness: толщина линии контура.
        dark_alpha: степень затемнения вне полигонов (0–1).
        blur_ksize: нечётный размер ядра для размывания маски.
    """
    img = image.copy()
    if quads is None or len(quads) == 0:
        return img

    # если Tensor, то в numpy
    if isinstance(quads, torch.Tensor):
        quads = quads.detach().cpu().numpy()

    h, w = img.shape[:2]
    # затемнённый фон
    dark_bg = (img.astype(np.float32) * (1 - dark_alpha)).astype(np.uint8)

    # 1) Создаём маску внутри полигонов
    mask = np.zeros((h, w), dtype=np.float32)
    for q in quads:
        pts = q[:8].reshape(4, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1.0)

    # 2) Размываем границу
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.clip(mask, 0.0, 1.0)
    mask_3 = mask[:, :, None]

    # 3) Смешиваем исход и затемнённый фон
    out = img.astype(np.float32) * mask_3 + dark_bg.astype(np.float32) * (1 - mask_3)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # 4) Рисуем контуры полигонов
    for q in quads:
        pts = q[:8].reshape(4, 2).astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)

    return out


def draw_rboxes(image, rboxes, color=(0, 255, 0), thickness=2, alpha=0.5):
    img = image.copy()
    if rboxes is None or len(rboxes) == 0:
        return img
    if isinstance(rboxes, torch.Tensor):
        rboxes = rboxes.detach().cpu().numpy()
    overlay = img.copy()
    for r in rboxes:
        cx, cy, w, h, angle = r
        pts = cv2.boxPoints(((cx, cy), (w, h), angle))
        pts = np.int32(pts)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=thickness)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2, alpha=0.5):
    if boxes is None or len(boxes) == 0:
        return image
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    # detect format by length
    first = boxes[0]
    if len(first) == 5:
        return draw_rboxes(image, boxes, color=color, thickness=thickness, alpha=alpha)
    elif len(first) in (8, 9):
        # quad with or without score
        return draw_quads(
            image, boxes, color=color, thickness=thickness, dark_alpha=alpha
        )
    else:
        raise ValueError(f"Unsupported box format with length {len(first)}")


def create_collage(
    img_tensor,
    gt_score_map,
    gt_geo_map,
    gt_rboxes,
    pred_score_map=None,
    pred_geo_map=None,
    pred_rboxes=None,
    cell_size=640,
):
    n_rows, n_cols = 2, 10
    collage = np.full((cell_size * n_rows, cell_size * n_cols, 3), 255, dtype=np.uint8)
    orig = tensor_to_image(img_tensor)

    # GT
    gt_img = draw_boxes(orig, gt_rboxes, color=(0, 255, 0))
    gt_score = (
        gt_score_map.detach().cpu().numpy().squeeze()
        if isinstance(gt_score_map, torch.Tensor)
        else gt_score_map
    )
    gt_score_vis = cv2.applyColorMap(
        (gt_score * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    gt_geo = (
        gt_geo_map.detach().cpu().numpy()
        if isinstance(gt_geo_map, torch.Tensor)
        else gt_geo_map
    )
    gt_cells = [gt_img, gt_score_vis]
    for i in range(gt_geo.shape[2]):
        ch = gt_geo[:, :, i]
        norm = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gt_cells.append(cv2.applyColorMap(norm, cv2.COLORMAP_JET))

    # Pred
    if pred_score_map is not None and pred_geo_map is not None:
        pred_img = draw_boxes(orig, pred_rboxes, color=(0, 0, 255))
        pred_score = (
            pred_score_map.detach().cpu().numpy().squeeze()
            if isinstance(pred_score_map, torch.Tensor)
            else pred_score_map
        )
        pred_score_vis = cv2.applyColorMap(
            (pred_score * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        pred_geo = (
            pred_geo_map.detach().cpu().numpy()
            if isinstance(pred_geo_map, torch.Tensor)
            else pred_geo_map
        )
        pred_cells = [pred_img, pred_score_vis]
        for i in range(pred_geo.shape[2]):
            ch = pred_geo[:, :, i]
            norm = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            pred_cells.append(cv2.applyColorMap(norm, cv2.COLORMAP_JET))
    else:
        pred_cells = [np.zeros((cell_size, cell_size, 3), dtype=np.uint8)] * n_cols

    # assemble
    for r in range(n_rows):
        cells = gt_cells if r == 0 else pred_cells
        for c in range(n_cols):
            cell = cv2.resize(cells[c], (cell_size, cell_size))
            y0, y1 = r * cell_size, (r + 1) * cell_size
            x0, x1 = c * cell_size, (c + 1) * cell_size
            collage[y0:y1, x0:x1] = cell

    return collage


def remove_duplicate_boxes(quads: np.ndarray) -> np.ndarray:
    """
    Убирает дубликаты боксов с одинаковыми координатами,
    оставляя бокс с максимальным score.
    
    Args:
        quads: массив (N, 9) с координатами и score
        
    Returns:
        массив с уникальными боксами
    """
    if len(quads) == 0:
        return quads
    
    # Группируем по координатам (первые 8 значений)
    coord_to_indices = {}
    
    for i, quad in enumerate(quads):
        # Используем tuple координат как ключ
        key = tuple(quad[:8])
        if key not in coord_to_indices:
            coord_to_indices[key] = []
        coord_to_indices[key].append(i)
    
    # Оставляем только боксы с максимальным score среди дубликатов
    unique_quads = []
    for indices in coord_to_indices.values():
        if len(indices) == 1:
            unique_quads.append(quads[indices[0]])
        else:
            # Берем бокс с максимальным score
            best_idx = max(indices, key=lambda idx: quads[idx, 8])
            unique_quads.append(quads[best_idx])
    
    return np.array(unique_quads, dtype=np.float32)


def compute_quad_score(score_map: np.ndarray, quad_coords: np.ndarray, scale: float, percentile: float = 80.0) -> float:
    """
    Вычисляет процентиль score внутри области квадрилатерала (полигона из 4 точек).
    
    Args:
        score_map: карта score (H, W)
        quad_coords: координаты квадрилатерала (4, 2) в пикселях
        scale: масштаб для преобразования обратно в координаты score_map
        percentile: процентиль для вычисления (по умолчанию 80.0)
        
    Returns:
        процентиль score внутри полигона
    """
    h, w = score_map.shape
    
    # Преобразуем координаты обратно в координаты score_map
    map_coords = quad_coords / scale
    
    # Находим bounding box полигона
    min_x = max(0, int(np.floor(map_coords[:, 0].min())))
    max_x = min(w - 1, int(np.ceil(map_coords[:, 0].max())))
    min_y = max(0, int(np.floor(map_coords[:, 1].min())))
    max_y = min(h - 1, int(np.ceil(map_coords[:, 1].max())))
    
    if min_x >= max_x or min_y >= max_y:
        # Если область пустая, возвращаем значение центральной точки
        center_x = int(np.clip(map_coords[:, 0].mean(), 0, w - 1))
        center_y = int(np.clip(map_coords[:, 1].mean(), 0, h - 1))
        return float(score_map[center_y, center_x])
    
    # Создаем маску для области внутри квадрилатерала
    bbox_h = max_y - min_y + 1
    bbox_w = max_x - min_x + 1
    
    # Сдвигаем координаты полигона относительно bounding box
    shifted_coords = map_coords.copy()
    shifted_coords[:, 0] -= min_x
    shifted_coords[:, 1] -= min_y
    
    # Создаем маску полигона с помощью cv2.fillPoly
    mask = np.zeros((bbox_h, bbox_w), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted_coords.astype(np.int32)], 1)
    
    # Извлекаем область score_map, соответствующую bounding box
    score_region = score_map[min_y:max_y+1, min_x:max_x+1]
    
    # Вычисляем процентиль score только внутри полигона
    mask_bool = mask > 0
    if mask_bool.any():
        return float(np.percentile(score_region[mask_bool], percentile))
    else:
        # Fallback: если маска пустая, берем центральную точку
        center_x = int(np.clip(map_coords[:, 0].mean(), 0, w - 1))
        center_y = int(np.clip(map_coords[:, 1].mean(), 0, h - 1))
        return float(score_map[center_y, center_x])


def decode_boxes_from_maps(
    score_map: np.ndarray,
    geo_map: np.ndarray,
    score_thresh: float = 0.9,
    scale: float = 4.0,
    iou_threshold: float = 0.2,
    expand_ratio: float = 0.0,
    profile: bool = False,
    score_percentile: float = 80.0,
) -> np.ndarray:
    """
    Декодирует quad-боксы из 8-канальной geo_map, с опциональным расширением (обратным shrink).

    Параметры:
      score_map           — карта вероятностей (H, W) или (1, H, W);
      geo_map             — гео-карта (H, W, 8);
      score_thresh        — порог для отбора пикселей;
      scale               — коэффициент восстановления в исходные пиксели (обычно = 1.0/score_geo_scale);
      iou_threshold       — порог IoU для NMS;
      expand_ratio        — коэффициент обратного расширения (обычно = shrink_ratio);
      profile             — если True, выводит время выполнения этапов;
      score_percentile       — процентиль для вычисления confidence (по умолчанию 80.0).

    Возвращает:
      quad-боксы (N, 9) — [x0, y0, …, x3, y3, score].
    """
    start_time = time.time()
    
    # убираем лишнюю первую размерность
    t0 = time.time()
    if score_map.ndim == 3 and score_map.shape[0] == 1:
        score_map = score_map.squeeze(0)
    if profile: print(f"    Squeeze score_map: {time.time() - t0:.3f}s")

    # найти пиксели выше порога
    t0 = time.time()
    ys, xs = np.where(score_map > score_thresh)
    if profile: print(f"    Find pixels > thresh: {time.time() - t0:.3f}s ({len(ys)} pixels)")
    
    # декодировать координаты боксов
    t0 = time.time()
    quads = []
    for y, x in zip(ys, xs):
        offs = geo_map[y, x]
        verts = []
        for i in range(4):
            dx_map, dy_map = offs[2 * i], offs[2 * i + 1]
            dx = dx_map * scale
            dy = dy_map * scale
            vx = round(x * scale + dx)
            vy = round(y * scale + dy)
            verts.extend([vx, vy])
        
        # Вычисляем процентиль score внутри области квадрата
        quad_coords = np.array(verts).reshape(4, 2)
        score_val = compute_quad_score(score_map, quad_coords, scale, score_percentile)
        
        # Фильтруем по финальному score после вычисления процентиля
        if score_val >= score_thresh:
            quads.append(verts + [score_val])
    if profile: 
        print(f"    Decode coordinates: {time.time() - t0:.3f}s ({len(quads)} quads)")
        if len(quads) > 0:
            scores = [q[-1] for q in quads]
            print(f"    Score range: {min(scores):.3f} - {max(scores):.3f}")
            print(f"    Score samples: {scores[:5]}")
            # Считаем сколько боксов с низким финальным score
            low_scores = [s for s in scores if s < score_thresh]
            if low_scores:
                print(f"    Boxes below thresh: {len(low_scores)}/{len(scores)} (lowest: {min(low_scores):.3f})")

    if not quads:
        if profile: print(f"    decode_boxes_from_maps total: {time.time() - start_time:.3f}s")
        return np.zeros((0, 9), dtype=np.float32)

    t0 = time.time()
    quads = np.array(quads, dtype=np.float32)
    if profile: print(f"    Convert to numpy: {time.time() - t0:.3f}s")



    # Убираем дубликаты с одинаковыми координатами
    t0 = time.time()
    if profile: print(f"      Before dedup: {len(quads)} boxes")
    quads = remove_duplicate_boxes(quads)
    if profile: print(f"    Remove duplicates: {time.time() - t0:.3f}s ({len(quads)} after dedup)")

    # NMS (быстрый классический O(n²) вместо медленного locality_aware O(n³))
    t0 = time.time()
    if profile: print(f"      Before NMS: {len(quads)} boxes")
    keep = fast_nms(quads, iou_threshold=iou_threshold)
    if profile: print(f"    Fast NMS: {time.time() - t0:.3f}s ({len(keep)} kept)")
    
    # Дополнительная фильтрация после NMS по финальному score
    t0 = time.time()
    if len(keep) > 0:
        if profile: print(f"      Before score filter: {len(keep)} boxes, score_thresh={score_thresh}")
        final_keep = []
        for quad in keep:
            if quad[8] >= score_thresh:  # score находится в позиции 8
                final_keep.append(quad)
            elif profile and len(final_keep) < 3:  # Показываем первые отброшенные
                print(f"      Rejecting box with score {quad[8]:.3f} < {score_thresh}")
        keep = np.array(final_keep, dtype=np.float32) if final_keep else np.zeros((0, 9), dtype=np.float32)
    if profile: print(f"    Filter by score: {time.time() - t0:.3f}s ({len(keep)} kept after score filter)")
    
    # обратное расширение shrink_poly (если нужно)
    if expand_ratio and len(keep) > 0:
        t0 = time.time()
        from .dataset import shrink_poly

        expanded = []
        for quad in keep:
            coords = quad[:8].reshape(4, 2)
            # применяем shrink с отрицательным коэффициентом
            exp_poly = shrink_poly(coords, shrink_ratio=-expand_ratio)
            expanded.append(list(exp_poly.flatten()) + [quad[8]])
        keep = np.array(expanded, dtype=np.float32)
        if profile: print(f"    Expand boxes: {time.time() - t0:.3f}s")

    if profile: print(f"    decode_boxes_from_maps total: {time.time() - start_time:.3f}s")
    return keep


def expand_boxes(quads: np.ndarray, expand_ratio: float) -> np.ndarray:
    """
    Расширяет каждый quad обратно с помощью shrink_poly с отрицательным коэффициентом.
    """
    if expand_ratio == 0 or len(quads) == 0:
        return quads

    from .dataset import shrink_poly

    expanded = []
    for quad in quads:
        coords = quad[:8].reshape(4, 2)
        exp_poly = shrink_poly(coords, shrink_ratio=-expand_ratio)
        expanded.append(list(exp_poly.flatten()) + [quad[8]])
    return np.array(expanded, dtype=np.float32)


def apply_nms(quads: np.ndarray, iou_threshold: float = 0.2) -> np.ndarray:
    """
    Применяет быстрый NMS к массиву quad-боксов.
    """
    if len(quads) == 0:
        return quads
    return fast_nms(quads, iou_threshold=iou_threshold)


def poly_iou(segA, segB):
    A = Polygon(np.array(segA).reshape(-1, 2))
    B = Polygon(np.array(segB).reshape(-1, 2))
    if not A.is_valid or not B.is_valid:
        return 0.0
    inter = A.intersection(B).area
    union = A.union(B).area
    return inter / union if union > 0 else 0.0


def compute_f1(preds, thresh, gt_segs, processed_ids):
    # Кэшируем полигоны
    gt_polys = {
        iid: [Polygon(np.array(seg).reshape(-1, 2)) for seg in gt_segs.get(iid, [])]
        for iid in processed_ids
    }
    pred_polys = [
        {
            "image_id": p["image_id"],
            "polygon": Polygon(np.array(p["segmentation"]).reshape(-1, 2)),
        }
        for p in preds
    ]

    used = {iid: [False] * len(gt_polys.get(iid, [])) for iid in processed_ids}
    tp = fp = 0
    for p, pred_poly in zip(preds, pred_polys):
        image_id = p["image_id"]
        pred_polygon = pred_poly["polygon"]
        if not pred_polygon.is_valid:
            fp += 1
            continue
        best_iou, bj = 0, -1
        for j, gt_polygon in enumerate(gt_polys.get(image_id, [])):
            if used[image_id][j] or not gt_polygon.is_valid:
                continue
            inter = pred_polygon.intersection(gt_polygon).area
            union = pred_polygon.union(gt_polygon).area
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou, bj = iou, j
        if best_iou >= thresh:
            tp += 1
            used[image_id][bj] = True
        else:
            fp += 1
    total_gt = sum(len(v) for v in gt_polys.values())
    fn = total_gt - tp
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0


def load_gt(gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_coco = json.load(f)
    gt_segs = defaultdict(list)
    for ann in gt_coco["annotations"]:
        seg = ann.get("segmentation", [])
        if seg:
            gt_segs[ann["image_id"]].append(seg[0])
    return gt_segs


def load_preds(pred_path):
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    preds_list = data.get("annotations", data)
    preds = []
    for p in preds_list:
        seg = p.get("segmentation", [])
        if not seg:
            continue
        preds.append(
            {
                "image_id": p["image_id"],
                "segmentation": seg[0],
                "score": p.get("score", 1.0),
            }
        )
    return preds


def compute_f1_metrics(
    preds, gt_segs, processed_ids, avg_range=(0.50, 0.95), avg_step=0.05
):
    f1_at_05 = compute_f1(preds, 0.5, gt_segs, processed_ids)

    iou_vals = np.arange(avg_range[0], avg_range[1] + 1e-9, avg_step)
    f1_list = []
    for t in tqdm(iou_vals, desc="F1 по IoU", unit="IoU"):
        f1_list.append(compute_f1(preds, t, gt_segs, processed_ids))

    f1_avg = float(np.mean(f1_list))
    return f1_at_05, f1_avg
