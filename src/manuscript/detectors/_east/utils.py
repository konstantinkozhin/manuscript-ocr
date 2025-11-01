import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from shapely.geometry import Polygon
from tqdm import tqdm


"""
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
"""


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
) -> Image.Image:
    """
    Draw quadrilateral boxes over the input image and return a PIL visualization.

    Args:
        image: Source image in RGB (preferred) or BGR order with shape HxWx3.
        quads: Array-like collection of quadrangles shaped `N x 9` or `N x 8`
            where the first eight values are `[x1, y1, ..., x4, y4]`.
        color: Polyline color passed to OpenCV.
        thickness: Polyline thickness in pixels.
        dark_alpha: Blending factor for darkening the background under polygons.
        blur_ksize: Odd kernel size for Gaussian blur applied to the soft mask.

    Returns:
        PIL.Image.Image: Image containing the visualization overlay.
    """
    img = image.copy()
    if quads is None or len(quads) == 0:
        return Image.fromarray(img)

    if isinstance(quads, torch.Tensor):
        quads = quads.detach().cpu().numpy()

    h, w = img.shape[:2]
    dark_bg = (img.astype(np.float32) * (1 - dark_alpha)).astype(np.uint8)

    mask = np.zeros((h, w), dtype=np.float32)
    for q in quads:
        pts = q[:8].reshape(4, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1.0)

    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.clip(mask, 0.0, 1.0)
    mask_3 = mask[:, :, None]

    out = img.astype(np.float32) * mask_3 + dark_bg.astype(np.float32) * (1 - mask_3)
    out = np.clip(out, 0, 255).astype(np.uint8)

    for q in quads:
        pts = q[:8].reshape(4, 2).astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)

    return Image.fromarray(out)


def visualize_page(
    image,
    page,
    *,
    show_order=False,
    color=(0, 0, 255),
    thickness=2,
    dark_alpha=0.3,
    blur_ksize=11,
    line_color=(0, 255, 0),
    number_color=(255, 255, 255),
    number_bg=(0, 0, 0),
) -> Image.Image:
    """
    Unified visualization function for Page objects with optional reading order display.

    This function draws text detection boxes on the image using the same style as EAST detector,
    with optional numbered boxes and connecting lines to show reading order.

    Parameters
    ----------
    image : np.ndarray or PIL.Image.Image
        Input image (RGB format preferred).
    page : Page
        Page object containing detected text blocks and words.
    show_order : bool, optional
        Whether to display reading order (numbered boxes with connecting lines), by default False.
    color : tuple, optional
        Color for polygon borders in RGB format, by default (0, 0, 255) - blue.
    thickness : int, optional
        Thickness of polygon borders in pixels, by default 2.
    dark_alpha : float, optional
        Blending factor for darkening background (0.0 to 1.0), by default 0.3.
    blur_ksize : int, optional
        Kernel size for Gaussian blur (must be odd), by default 11.
    line_color : tuple, optional
        Color for connecting lines in RGB format, by default (0, 255, 0) - green.
    number_color : tuple, optional
        Color for box numbers text in RGB format, by default (255, 255, 255) - white.
    number_bg : tuple, optional
        Background color for box numbers in RGB format, by default (0, 0, 0) - black.

    Returns
    -------
    PIL.Image.Image
        Visualized image with detection boxes and optional reading order annotations.

    Examples
    --------
    Basic visualization without reading order:

    >>> from manuscript import EAST, visualize_page
    >>> detector = EAST()
    >>> result = detector.predict("document.jpg")
    >>> vis = visualize_page(result["vis_image"], result["page"])
    >>> vis.save("output.jpg")

    Visualization with reading order display:

    >>> vis = visualize_page(
    ...     result["vis_image"],
    ...     result["page"],
    ...     show_order=True,
    ...     color=(255, 0, 0),
    ...     thickness=3
    ... )
    """
    from PIL import ImageDraw

    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        img = image.copy()

    # Collect all quads and words in order
    quads = []
    words_in_order = []

    for block in page.blocks:
        for w in block.words:
            poly = np.array(w.polygon).reshape(-1)
            quads.append(poly)
            words_in_order.append(w)

    if len(quads) == 0:
        return Image.fromarray(img) if isinstance(image, np.ndarray) else image

    quads = np.stack(quads, axis=0)

    # Draw polygons using EAST style
    out = draw_quads(
        image=img,
        quads=quads,
        color=color,
        thickness=thickness,
        dark_alpha=dark_alpha,
        blur_ksize=blur_ksize,
    )

    # Add reading order visualization if requested
    if show_order:
        draw = ImageDraw.Draw(out)

        # Calculate centers of all words
        centers = []
        for w in words_in_order:
            xs = [p[0] for p in w.polygon]
            ys = [p[1] for p in w.polygon]
            centers.append((sum(xs) / len(xs), sum(ys) / len(ys)))

        # Draw connecting lines between consecutive words
        if len(centers) > 1:
            for p, c in zip(centers, centers[1:]):
                draw.line([p, c], fill=line_color, width=3)

        # Draw numbered boxes at centers
        for idx, c in enumerate(centers, start=1):
            cx, cy = c
            draw.rectangle(
                [cx - 12, cy - 12, cx + 12, cy + 12],
                fill=number_bg,
            )
            draw.text((cx - 6, cy - 8), str(idx), fill=number_color)

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
        quad_img = draw_quads(
            image, boxes, color=color, thickness=thickness, dark_alpha=alpha
        )
        return np.array(quad_img)
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


def decode_quads_from_maps(
    score_map: np.ndarray,
    geo_map: np.ndarray,
    score_thresh: float,
    scale: float,
    quantization: int = 1,
    profile=False,
) -> np.ndarray:
    if score_map.ndim == 3 and score_map.shape[0] == 1:
        score_map = score_map.squeeze(0)

    t0 = time.time()
    ys, xs = np.where(score_map > score_thresh)
    if profile:
        print(f"    Find pixels > thresh: {time.time() - t0:.3f}s ({len(ys)} pixels)")

    if len(ys) == 0:
        return np.zeros((0, 9), dtype=np.float32)

    if quantization > 1:
        t0_quant = time.time()
        ys_quant = (ys // quantization) * quantization + quantization // 2
        xs_quant = (xs // quantization) * quantization + quantization // 2

        coords = np.column_stack([ys_quant, xs_quant])
        unique_coords = np.unique(coords, axis=0)

        ys = unique_coords[:, 0]
        xs = unique_coords[:, 1]

        if profile:
            print(
                f"    Quantization (step={quantization}): {time.time() - t0_quant:.3f}s"
            )
            print(
                f"    Points after quantization: {len(ys)} (removed {len(coords) - len(ys)})"
            )

    t0 = time.time()
    quads = []
    for y, x in zip(ys, xs):
        offs = geo_map[y, x]
        verts = []
        for i in range(4):
            dx_map, dy_map = offs[2 * i], offs[2 * i + 1]
            vx = x * scale + dx_map * scale
            vy = y * scale + dy_map * scale
            verts.extend([vx, vy])
        quads.append(verts + [float(score_map[y, x])])

    if profile:
        print(f"    Decode coordinates: {time.time() - t0:.3f}s ({len(quads)} quads)")

    return np.array(quads, dtype=np.float32)


def expand_boxes(
    quads: np.ndarray, expand_w: float = 0.0, expand_h: float = 0.0
) -> np.ndarray:

    if len(quads) == 0 or (expand_w == 0 and expand_h == 0):
        return quads

    coords = quads[:, :8].reshape(-1, 4, 2)
    scores = quads[:, 8:9]

    x, y = coords[:, :, 0], coords[:, :, 1]
    area = np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    sign = np.sign(area).reshape(-1, 1, 1)
    sign[sign == 0] = 1

    p_prev = np.roll(coords, 1, axis=1)
    p_curr = coords
    p_next = np.roll(coords, -1, axis=1)

    edge1 = p_curr - p_prev
    edge2 = p_next - p_curr
    len1 = np.linalg.norm(edge1, axis=2, keepdims=True)
    len2 = np.linalg.norm(edge2, axis=2, keepdims=True)

    n1 = sign * np.stack([edge1[..., 1], -edge1[..., 0]], axis=2) / (len1 + 1e-6)
    n2 = sign * np.stack([edge2[..., 1], -edge2[..., 0]], axis=2) / (len2 + 1e-6)
    n_avg = n1 + n2
    norm = np.linalg.norm(n_avg, axis=2, keepdims=True)
    n_avg = np.divide(n_avg, norm, out=np.zeros_like(n_avg), where=norm > 0)

    offset = np.minimum(len1, len2)

    scale_xy = np.array([1 + expand_w, 1 + expand_h], dtype=np.float32).reshape(1, 1, 2)
    delta = (scale_xy - 1.0) * offset

    new_coords = p_curr + delta * n_avg

    expanded = np.hstack([new_coords.reshape(-1, 8), scores])
    return expanded.astype(np.float32)


def poly_iou(segA, segB):
    A = Polygon(np.array(segA).reshape(-1, 2))
    B = Polygon(np.array(segB).reshape(-1, 2))
    if not A.is_valid or not B.is_valid:
        return 0.0
    inter = A.intersection(B).area
    union = A.union(B).area
    return inter / union if union > 0 else 0.0


def compute_f1(preds, thresh, gt_segs, processed_ids):
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


def read_image(img_or_path):
    if isinstance(img_or_path, (str, Path)):
        img = cv2.imread(str(img_or_path))
        if img is None:
            try:
                with Image.open(str(img_or_path)) as pil_img:
                    img = np.array(pil_img.convert("RGB"))
            except Exception as e:
                raise FileNotFoundError(
                    f"Cannot read image with cv2 or PIL: {img_or_path}. Error: {e}"
                )
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif isinstance(img_or_path, np.ndarray):
        img = img_or_path

    else:
        raise TypeError(f"Unsupported type for image input: {type(img_or_path)}")

    return img


def resolve_intersections(boxes):
    """
    Resolve intersecting boxes by shrinking them iteratively.

    Parameters
    ----------
    boxes : list of tuple
        List of boxes in format (x_min, y_min, x_max, y_max).

    Returns
    -------
    list of tuple
        List of resolved boxes.
    """

    def intersect(b1, b2):
        return not (
            b1[2] <= b2[0] or b2[2] <= b1[0] or b1[3] <= b2[1] or b2[3] <= b1[1]
        )

    resolved = list(boxes)
    max_iterations = 50

    for _ in range(max_iterations):
        changed = False
        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                if intersect(resolved[i], resolved[j]):
                    x0, y0, x1, y1 = resolved[i]
                    x0b, y0b, x1b, y1b = resolved[j]

                    resolved[i] = (
                        x0,
                        y0,
                        int(x1 - (x1 - x0) * 0.1),
                        int(y1 - (y1 - y0) * 0.1),
                    )
                    resolved[j] = (
                        x0b,
                        y0b,
                        int(x1b - (x1b - x0b) * 0.1),
                        int(y1b - (y1b - y0b) * 0.1),
                    )
                    changed = True
        if not changed:
            break

    return resolved


def sort_boxes_reading_order(boxes, y_tol_ratio=0.6, x_gap_ratio=np.inf):
    """
    Sort boxes in natural reading order (left-to-right, top-to-bottom).

    Groups boxes into lines based on vertical proximity, then sorts each line
    horizontally. This approximates the natural reading order for documents.

    Parameters
    ----------
    boxes : list of tuple
        List of boxes in format (x_min, y_min, x_max, y_max).
    y_tol_ratio : float, optional
        Vertical tolerance as a ratio of average box height for grouping boxes
        into the same line, by default 0.6.
    x_gap_ratio : float, optional
        Maximum horizontal gap as a ratio of average box height for boxes to be
        considered part of the same line, by default np.inf (no limit).

    Returns
    -------
    list of tuple
        Boxes sorted in reading order.

    Examples
    --------
    >>> boxes = [(10, 10, 50, 30), (60, 10, 100, 30), (10, 50, 50, 70)]
    >>> sorted_boxes = sort_boxes_reading_order(boxes)
    """
    if not boxes:
        return []

    avg_h = np.mean([b[3] - b[1] for b in boxes])
    lines = []

    for b in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        cy = (b[1] + b[3]) / 2
        placed = False

        for ln in lines:
            line_cy = np.mean([(v[1] + v[3]) / 2 for v in ln])
            last_x1 = max(v[2] for v in ln)

            if (
                abs(cy - line_cy) <= avg_h * y_tol_ratio
                and (b[0] - last_x1) <= avg_h * x_gap_ratio
            ):
                ln.append(b)
                placed = True
                break

        if not placed:
            lines.append([b])

    lines.sort(key=lambda ln: np.mean([(b[1] + b[3]) / 2 for b in ln]))
    for ln in lines:
        ln.sort(key=lambda b: b[0])

    return [b for ln in lines for b in ln]


def sort_boxes_reading_order_with_resolutions(
    boxes, y_tol_ratio=0.6, x_gap_ratio=np.inf
):
    """
    Sort boxes in reading order after resolving intersections.

    This function first resolves overlapping boxes by shrinking them, then applies
    reading order sorting. Useful when boxes may overlap slightly.

    Parameters
    ----------
    boxes : list of tuple
        List of boxes in format (x_min, y_min, x_max, y_max).
    y_tol_ratio : float, optional
        Vertical tolerance for line grouping, by default 0.6.
    x_gap_ratio : float, optional
        Maximum horizontal gap for line continuity, by default np.inf.

    Returns
    -------
    list of tuple
        Boxes sorted in reading order with intersections resolved.

    Examples
    --------
    >>> boxes = [(10, 10, 55, 30), (50, 10, 100, 30)]  # Overlapping
    >>> sorted_boxes = sort_boxes_reading_order_with_resolutions(boxes)
    """
    compressed = resolve_intersections(boxes)
    mapping = {c: o for c, o in zip(compressed, boxes)}

    sorted_compressed = sort_boxes_reading_order(
        compressed, y_tol_ratio=y_tol_ratio, x_gap_ratio=x_gap_ratio
    )
    return [mapping[b] for b in sorted_compressed]


"""
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
"""


"""
def compute_f1_metrics(
    preds,
    gt_segs,
    processed_ids,
    avg_range=(0.50, 0.95),
    avg_step=0.05,
    use_tqdm: bool = True,
):
    f1_at_05 = compute_f1(preds, 0.5, gt_segs, processed_ids)

    iou_vals = np.arange(avg_range[0], avg_range[1] + 1e-9, avg_step)
    f1_list = []
    iterator = tqdm(iou_vals, desc="F1 ?? IoU", unit="IoU") if use_tqdm else iou_vals
    for t in iterator:
        f1_list.append(compute_f1(preds, t, gt_segs, processed_ids))

    f1_avg = float(np.mean(f1_list))
    return f1_at_05, f1_avg
"""
