import time

import cv2
import numpy as np

from ...utils.visualization import _draw_quads
from ...utils.io import _tensor_to_image


def create_collage(
    img_tensor,
    gt_score_map,
    gt_geo_map,
    gt_quads,
    pred_score_map=None,
    pred_geo_map=None,
    pred_quads=None,
    cell_size=640,
):
    """Create visualization collage for EAST training (requires torch)."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "create_collage requires PyTorch. Install with: pip install manuscript-ocr[dev]"
        ) from exc

    n_rows, n_cols = 2, 10
    collage = np.full((cell_size * n_rows, cell_size * n_cols, 3), 255, dtype=np.uint8)
    orig = _tensor_to_image(img_tensor)

    # GT
    gt_img = np.array(_draw_quads(orig, gt_quads, color=(0, 255, 0)))
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
        pred_img = np.array(_draw_quads(orig, pred_quads, color=(0, 0, 255)))
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
    quads: np.ndarray,
    expand_w: float = 0.0,
    expand_h: float = 0.0,
    expand_power: float = 1.0,
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

    n_normalized = np.divide(n_avg, norm, out=np.zeros_like(n_avg), where=norm > 1e-6)
    degenerate_mask = (norm <= 1e-6).squeeze(-1)  # shape: (N, 4)
    n_normalized[degenerate_mask] = n1[degenerate_mask]

    offset = np.minimum(len1, len2)

    if expand_power != 1.0:
        offset_scaled = np.power(offset, expand_power)
    else:
        offset_scaled = offset

    scale_xy = np.array([1 + expand_w, 1 + expand_h], dtype=np.float32).reshape(1, 1, 2)
    delta = (scale_xy - 1.0) * offset_scaled

    new_coords = p_curr + delta * n_normalized

    expanded = np.hstack([new_coords.reshape(-1, 8), scores])
    return expanded.astype(np.float32)
