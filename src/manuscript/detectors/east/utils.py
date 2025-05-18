import cv2
import numpy as np
import torch
from .lanms import locality_aware_nms


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
    style: str = "highlight",
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    alpha: float = 0.5,
    dark_alpha: float = 0.5,
    blur_ksize: int = 11,
) -> np.ndarray:
    """
    Рисует полигоны двумя стилями:
      - style="border": контур с прозрачностью alpha.
      - style="highlight": затемняет всю картинку на dark_alpha,
        при этом внутри каждого полигона показывает оригинал.
        Для мягких углов маска размывается ядром blur_ksize.

    :param blur_ksize: нечётный размер ядра для GaussianBlur.
    """
    img = image.copy()
    if quads is None or len(quads) == 0:
        return img

    # Если передали тензор — приводим к numpy
    if isinstance(quads, torch.Tensor):
        quads = quads.detach().cpu().numpy()

    if style == "border":
        overlay = img.copy()
        for q in quads:
            pts = q[:8].reshape(4, 2).astype(np.int32)
            cv2.polylines(
                overlay, [pts], isClosed=True, color=color, thickness=thickness
            )
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    elif style == "highlight":
        h, w, _ = img.shape

        dark_bg = (img.astype(np.float32) * (1 - dark_alpha)).astype(np.uint8)

        mask = np.zeros((h, w), dtype=np.float32)
        for q in quads:
            pts = q[:8].reshape(4, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1.0)

        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask = np.clip(mask, 0.0, 1.0)

        mask_3 = mask[:, :, None]
        out = img.astype(np.float32) * mask_3 + dark_bg.astype(np.float32) * (
            1 - mask_3
        )
        return np.clip(out, 0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown style {style!r}, expected 'border' or 'highlight'")


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2, alpha=0.5):
    if boxes is None or len(boxes) == 0:
        return image
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()

    first = boxes[0]
    if len(first) in (8, 9):

        return draw_quads(image, boxes, color=color, thickness=thickness, alpha=alpha)
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

    for r in range(n_rows):
        cells = gt_cells if r == 0 else pred_cells
        for c in range(n_cols):
            cell = cv2.resize(cells[c], (cell_size, cell_size))
            y0, y1 = r * cell_size, (r + 1) * cell_size
            x0, x1 = c * cell_size, (c + 1) * cell_size
            collage[y0:y1, x0:x1] = cell

    return collage


def decode_boxes_from_maps(
    score_map: np.ndarray,
    geo_map: np.ndarray,
    score_thresh: float = 0.9,
    scale: float = 4.0,
    iou_threshold: float = 0.2,
    expand_ratio: float = 0.0,
) -> np.ndarray:
    """
    Декодирует quad-боксы из 8-канальной geo_map, с опциональным расширением (обратным shrink).

    Параметры:
      score_map     — карта вероятностей (H, W) или (1, H, W);
      geo_map       — гео-карта (H, W, 8);
      score_thresh  — порог для отбора пикселей;
      scale         — коэффициент восстановления в исходные пиксели (обычно = 1.0/score_geo_scale);
      iou_threshold — порог IoU для NMS;
      expand_ratio  — коэффициент обратного расширения (обычно = shrink_ratio).

    Возвращает:
      quad-боксы (N, 9) — [x0, y0, …, x3, y3, score].
    """

    if score_map.ndim == 3 and score_map.shape[0] == 1:
        score_map = score_map.squeeze(0)

    ys, xs = np.where(score_map > score_thresh)
    quads = []
    for y, x in zip(ys, xs):
        offs = geo_map[y, x]
        verts = []
        for i in range(4):
            dx_map, dy_map = offs[2 * i], offs[2 * i + 1]
            dx = dx_map * scale
            dy = dy_map * scale
            vx = x * scale + dx
            vy = y * scale + dy
            verts.extend([vx, vy])
        quads.append(verts + [float(score_map[y, x])])

    if not quads:
        return np.zeros((0, 9), dtype=np.float32)

    quads = np.array(quads, dtype=np.float32)

    # NMS
    keep = locality_aware_nms(quads, iou_threshold=iou_threshold)

    if expand_ratio and len(keep) > 0:
        from .dataset import shrink_poly

        expanded = []
        for quad in keep:
            coords = quad[:8].reshape(4, 2)
            exp_poly = shrink_poly(coords, shrink_ratio=-expand_ratio)
            expanded.append(list(exp_poly.flatten()) + [quad[8]])
        keep = np.array(expanded, dtype=np.float32)

    return keep
