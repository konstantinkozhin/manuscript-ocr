from typing import Tuple, Optional, Union
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..data import Page
from .io import read_image


def _draw_quads(
    image: Union[str, Path, np.ndarray, Image.Image],
    quads: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    dark_alpha: float = 0.3,
    blur_ksize: int = 5,
) -> Image.Image:
    """
    Draw quadrilateral boxes on an image with semi-transparent overlay.

    Parameters
    ----------
    image : str, Path, np.ndarray, or PIL.Image
        Input image. Can be:
        - Path to image file (str or Path)
        - RGB numpy array with shape (H, W, 3)
        - PIL Image object
    quads : np.ndarray
        Array of quad boxes with shape (N, 8) or (N, 9).
        Each row contains [x1, y1, x2, y2, x3, y3, x4, y4] or with score.
    color : tuple of int, default=(0, 255, 0)
        RGB color for drawing boxes.
    thickness : int, default=2
        Line thickness in pixels.
    dark_alpha : float, default=0.3
        Alpha value for darkening the image (0=no darkening, 1=fully dark).
    blur_ksize : int, default=5
        Kernel size for Gaussian blur (must be odd, 0=no blur).

    Returns
    -------
    PIL.Image.Image
        Image with drawn quadrilaterals.

    Examples
    --------
    >>> import numpy as np
    >>> from PIL import Image
    >>> # From numpy array
    >>> img = np.zeros((480, 640, 3), dtype=np.uint8)
    >>> quads = np.array([[100, 100, 200, 100, 200, 150, 100, 150]])
    >>> result = draw_quads(img, quads, color=(255, 0, 0))

    >>> # From file path
    >>> result = draw_quads("document.jpg", quads, color=(255, 0, 0))
    """
    img = read_image(image).copy()

    # Apply darkening if requested
    if dark_alpha > 0:
        overlay = (img * (1 - dark_alpha)).astype(np.uint8)
    else:
        overlay = img

    # Apply blur if requested
    if blur_ksize > 0:
        overlay = cv2.GaussianBlur(overlay, (blur_ksize, blur_ksize), 0)

    # Draw each quad
    for quad in quads:
        coords = quad[:8].reshape(4, 2).astype(np.int32)
        cv2.polylines(
            overlay, [coords], isClosed=True, color=color, thickness=thickness
        )

    return Image.fromarray(overlay)


def visualize_page(
    image: Union[str, Path, np.ndarray, Image.Image],
    page: Page,
    thickness=3,
    numbers_levels=(1, 2),
    connections_levels=(),
    line_color=(255, 165, 0),
    number_bg=(255, 255, 255),
    number_color=(0, 0, 0),
    max_size=4096,
    upscale=False,
    show_tables=True,
    show_cells=True,
    show_cell_indices=False,
    show_orientation=False,
    show_labels=False,
) -> Image.Image:
    """Render hierarchy depth with alternating brightness and cold-color layers.

    The background darkens by 30%. Level one restores original brightness,
    adding 15% per extra overlap. Even levels blend translucent cold colors,
    contrasting consecutive siblings. Odd levels from three onward lighten
    accumulated colors by 25% per overlap. Three-pixel dark contours share a
    color throughout each first-level subtree.
    numbers_levels and connections_levels select tree depths; empty lists
    disable their markers. Numbers default to levels one and two.
    Markers follow depth-first order among visible polygons at that depth.
    Overlap contributions are capped at two regions per depth.
    show_labels draws the original model label (or element type) on each region.
    """
    for name, levels in (("numbers_levels", numbers_levels),
                         ("connections_levels", connections_levels)):
        if not isinstance(levels, (list, tuple)) or any(
                isinstance(level, bool) or not isinstance(level, int) or level < 1
                for level in levels):
            raise ValueError(f"{name} must be a list or tuple of positive integers")
    if isinstance(thickness, bool) or not isinstance(thickness, int) or thickness < 1:
        raise ValueError("thickness must be a positive integer")
    # Load image using universal reader
    if isinstance(image, (str, Path)):
        img = read_image(image)
    elif isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        img = image.copy()

    if max_size is not None:
        h, w = img.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1 or upscale:
            img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                             interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        else:
            scale = 1.0
    else:
        scale = 1.0

    if not isinstance(page, Page):
        raise TypeError("visualize_page requires the document-tree Page")
    from manuscript.data import Table, Cell

    def visible(node):
        return not ((isinstance(node, Table) and not show_tables)
                    or (isinstance(node, Cell) and not show_cells))

    h, w = img.shape[:2]
    regions = [(node, np.asarray(node.polygon, dtype=np.float32) * scale, None)
               for node in page.descendants if node.polygon is not None and visible(node)]
    levels = {}
    for node, points, _ in regions:
        levels.setdefault(node.depth, []).append((node, points))

    palette = ((45, 155, 180), (145, 70, 185),
               (55, 170, 125), (75, 90, 195))
    palette = tuple(tuple(min(255, round(channel * 1.15)) for channel in tint)
                    for tint in palette)
    pixels = np.clip(img.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
    for depth, members in sorted(levels.items()):
        if depth % 2 == 0:
            overlay = Image.fromarray(pixels).convert("RGBA")
            sibling_indices = {}
            coverage = np.zeros((h, w), dtype=np.uint8)
            for node, points in members:
                parent_id = id(node.parent)
                index = sibling_indices.get(parent_id, 0)
                sibling_indices[parent_id] = index + 1
                layer = np.zeros((h, w, 4), dtype=np.uint8)
                cv2.fillPoly(layer, [points.astype(np.int32)],
                             (*palette[index % len(palette)], 75))
                selected = layer[:, :, 3] != 0
                layer[coverage >= 2, 3] = 0
                coverage[selected] = np.minimum(coverage[selected] + 1, 2)
                overlay = Image.alpha_composite(overlay, Image.fromarray(layer, mode="RGBA"))
            pixels = np.asarray(overlay.convert("RGB")).copy()
        else:
            coverage = np.zeros((h, w), dtype=np.uint32)
            for _, points in members:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [points.astype(np.int32)], 1)
                coverage += mask
            np.minimum(coverage, 2, out=coverage)
            selected = coverage != 0
            # Level one restores the source; later odd levels lighten the
            # accumulated image, retaining all underlying color contributions.
            source = img if depth == 1 else pixels
            step = 0.15 if depth == 1 else 0.25
            count = coverage.astype(np.float32) - (1 if depth == 1 else 0)
            factor = 1.0 + step * np.maximum(count, 0)
            brightened = np.clip(source.astype(np.float32) * factor[:, :, None],
                                 0, 255).astype(np.uint8)
            pixels[selected] = brightened[selected]

    out = Image.fromarray(pixels).convert("RGB")
    draw = ImageDraw.Draw(out)
    outline_palette = ((80, 20, 35), (15, 55, 85), (55, 20, 80),
                       (15, 70, 40), (85, 45, 10), (20, 25, 80))
    outline_palette = tuple(tuple(min(255, round(channel * 1.15)) for channel in tint)
                            for tint in outline_palette)
    outline_palette = tuple(outline_palette[index]
                            for index in np.random.default_rng(0).permutation(len(outline_palette)))
    root_colors = {id(node): outline_palette[index % len(outline_palette)]
                   for index, node in enumerate(page.children)}
    for node, points, _ in regions:
        root = node.ancestors[1] if node.depth > 1 else node
        contour = [(round(x), round(y)) for x, y in points]
        outline = root_colors[id(root)]
        if node.depth % 3:
            draw.line(contour + [contour[0]], fill=outline, width=thickness)
        else:
            # Carry dash phase around corners instead of restarting each edge.
            dash, gap = 9, 6
            distance = 0.0
            for start, end in zip(contour, contour[1:] + contour[:1]):
                start = np.asarray(start, dtype=float)
                delta = np.asarray(end, dtype=float) - start
                length = float(np.linalg.norm(delta))
                position = 0.0
                while position < length:
                    phase = distance % (dash + gap)
                    drawing = phase < dash
                    step = min((dash if drawing else dash + gap) - phase,
                               length - position)
                    if drawing:
                        first = start + delta * (position / length)
                        last = start + delta * ((position + step) / length)
                        draw.line([tuple(first), tuple(last)], fill=outline, width=thickness)
                    position += step
                    distance += step
    if show_cell_indices:
        for node, points, _ in regions:
            if not isinstance(node, Cell):
                continue
            label = f"r{node.row} c{node.column}"
            if node.row_span != 1 or node.column_span != 1:
                label += f" ({node.row_span}x{node.column_span})"
            position = (int(points[:, 0].min()) + 4, int(points[:, 1].min()) + 4)
            draw.rectangle(draw.textbbox(position, label), fill=tuple(number_bg))
            draw.text(position, label, fill=tuple(number_color))

    def centers_at(depths):
        return [tuple(points.mean(axis=0)) for node, points, _ in regions
                if node.depth in depths]

    if connections_levels:
        centers = centers_at(connections_levels)
        for start, end in zip(centers, centers[1:]):
            draw.line([start, end], fill=line_color, width=thickness)
    if numbers_levels:
        centers = centers_at(numbers_levels)
        overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
        labels = ImageDraw.Draw(overlay)
        for index, (cx, cy) in enumerate(centers, 1):
            text = str(index)
            position = (cx - 6, cy - 8)
            box = labels.textbbox(position, text)
            labels.rectangle((box[0]-3,box[1]-3,box[2]+3,box[3]+3),
                             fill=tuple(number_bg) + (140,))
            labels.text(position, text, fill=tuple(number_color) + (255,))
        out = Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")

    if show_labels:
        draw = ImageDraw.Draw(out)
        for node, points, _ in regions:
            label = node.label or node.type
            position = (max(0, int(points[:, 0].min()) + 4),
                        max(0, int(points[:, 1].min()) + 4))
            bounds = draw.textbbox(position, label)
            draw.rectangle(bounds, fill=tuple(number_bg))
            draw.text(position, label, fill=tuple(number_color))

    if show_orientation and isinstance(page, Page):
        draw = ImageDraw.Draw(out)
        for node, _, _ in regions:
            if node.orientation in (None, 0):
                continue
            label = f"{node.orientation:g}°"
            x, y = min(node.polygon, key=lambda point: (point[1], point[0]))
            bounds = draw.textbbox((0, 0), label)
            width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
            padding = 3
            left = max(0, min(round(x * scale), out.width - width - 2 * padding))
            top = max(0, min(round(y * scale) - height - 2 * padding,
                             out.height - height - 2 * padding))
            draw.rectangle((left, top, left + width + 2 * padding,
                            top + height + 2 * padding), fill=number_bg)
            draw.text((left + padding - bounds[0], top + padding - bounds[1]),
                      label, fill=number_color)

    return out
