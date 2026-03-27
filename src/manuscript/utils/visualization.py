from typing import Tuple, Optional, Union
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .io import read_image

try:
    import torch
except ImportError:
    torch = None


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
    # Load image using universal reader
    if isinstance(image, (str, Path)):
        img = read_image(image)
    elif isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        img = image.copy()

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
    page: "Page",  # type: ignore  # noqa: F821
    color=(0, 255, 0),
    thickness=2,
    show_order=True,
    show_lines=False,
    show_numbers=False,
    line_color=(255, 165, 0),
    number_bg=(255, 255, 255),
    number_color=(0, 0, 0),
    max_size=4096,
) -> Image.Image:
    """
    Visualize a Page object with detected text spans/blocks.

    This function draws all text spans from the Page structure on the image,
    optionally showing reading order with numbered markers and connecting lines.
    When show_order=True, it also visualizes blocks with semi-transparent
    bounding boxes, each block having a distinct color.

    Parameters
    ----------
    image : str, Path, np.ndarray, or PIL.Image
        Input image. Can be:
        - Path to image file (str or Path) - supports Unicode paths
        - RGB numpy array with shape (H, W, 3)
        - PIL Image object
    page : Page
        Page object from manuscript.data containing detected blocks/text spans.
    color : tuple of int, default=(0, 255, 0)
        RGB color for text span boundaries.
    thickness : int, default=2
        Line thickness for text span boundaries.
    show_order : bool, default=True
        If True, colors different text lines with different colors and shows
        semi-transparent block boundaries with different colors per block.
    show_lines : bool, default=False
        If True and show_order=True, draw connecting lines between consecutive
        text spans showing the reading sequence.
    show_numbers : bool, default=False
        If True and show_order=True, display numbered markers on each text span
        showing the reading order.
    line_color : tuple of int, default=(255, 165, 0)
        RGB color for connecting lines between text spans.
    number_bg : tuple of int, default=(255, 255, 255)
        Background color for order number boxes.
    number_color : tuple of int, default=(0, 0, 0)
        Text color for order numbers.
    max_size : int or None, default=4096
        Maximum size for the longer dimension of the output image.
        Image will be resized proportionally if larger. Set to None to
        keep original size.

    Returns
    -------
    PIL.Image.Image
        Visualized image with detection boxes and optional reading order annotations.
        When show_order=True, also includes semi-transparent block boundaries.

    Examples
    --------
    Basic visualization without reading order:

    >>> from manuscript import EAST
    >>> from manuscript.utils import visualize_page
    >>> detector = EAST()
    >>> result = detector.predict("document.jpg")
    >>> # Can pass path directly
    >>> vis = visualize_page("document.jpg", result["page"])
    >>> vis.save("output.jpg")

    Visualization with reading order and block boundaries:

    >>> # Can also use numpy array or PIL Image
    >>> from manuscript.utils import read_image
    >>> img = read_image("document.jpg")
    >>> vis = visualize_page(
    ...     img,
    ...     result["page"],
    ...     show_order=True,
    ...     color=(255, 0, 0),
    ...     thickness=3
    ... )

    Show connecting lines and numbers between text spans:

    >>> vis = visualize_page(
    ...     "document.jpg",
    ...     result["page"],
    ...     show_order=True,
    ...     show_lines=True,
    ...     show_numbers=True
    ... )
    """
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
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            scale = 1.0
    else:
        scale = 1.0

    def get_line_color(idx: int):
        hue = (idx * 0.618033988749895) % 1.0
        hsv = np.uint8([[[int(hue * 179), 220, 255]]])
        return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0])

    def get_block_color(idx: int):
        hue = ((idx * 0.618033988749895) + 0.5) % 1.0
        hsv = np.uint8([[[int(hue * 179), 180, 255]]])
        return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0])

    lines = []
    blocks = []
    line_index = 0

    for block_idx, block in enumerate(page.blocks):
        block_quads = []
        if block.lines:
            for line in block.lines:
                quads, text_spans = [], []
                for text_span in line.text_spans:
                    poly = np.array(text_span.polygon) * scale
                    quad = poly.reshape(-1)
                    quads.append(quad)
                    text_spans.append(text_span)
                    block_quads.append(quad)
                if quads:
                    lines.append((quads, text_spans, line_index))
                    line_index += 1
        elif block.text_spans:
            quads, text_spans = [], []
            for text_span in block.text_spans:
                poly = np.array(text_span.polygon) * scale
                quad = poly.reshape(-1)
                quads.append(quad)
                text_spans.append(text_span)
                block_quads.append(quad)
            if quads:
                lines.append((quads, text_spans, line_index))
                line_index += 1
        if block_quads:
            blocks.append((block_quads, block_idx))

    if not lines:
        return Image.fromarray(img)

    h, w = img.shape[:2]

    # ----- BLOCK LAYER (RGBA) -----
    block_layer = np.zeros((h, w, 4), dtype=np.uint8)

    for block_quads, block_idx in blocks:
        pts = np.vstack([q.reshape(4, 2) for q in block_quads])
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        color_b = get_block_color(block_idx)
        cv2.rectangle(
            block_layer, (int(x1), int(y1)), (int(x2), int(y2)), (*color_b, 75), -1
        )  # alpha=75
    
    # ----- TEXT-SPAN MASK (cut out text spans from block layers) -----
    text_span_mask = np.zeros((h, w), dtype=np.uint8)
    for quads, _, _ in lines:
        for quad in quads:
            coords = quad.reshape(4, 2).astype(np.int32)
            cv2.fillPoly(text_span_mask, [coords], 255)

    inv_text_span_mask = cv2.bitwise_not(text_span_mask)

     # cut out text spans -> blocks do not cover text spans
    block_layer[:, :, 3] = cv2.bitwise_and(block_layer[:, :, 3], inv_text_span_mask)

    # final image
    base = Image.fromarray(img).convert("RGBA")
    block_img = Image.fromarray(block_layer, mode="RGBA")
    out = Image.alpha_composite(base, block_img).convert("RGB")

    draw = ImageDraw.Draw(out)

    # ----- TEXT-SPAN BOXES -----
    for quads, _, idx in lines:
        col = get_line_color(idx) if show_order else color
        for quad in quads:
            pts = quad.reshape(4, 2)
            pts_py = [(int(x), int(y)) for x, y in pts]
            draw.line(pts_py + [pts_py[0]], fill=tuple(col), width=thickness)

    # ----- ORDER LINES & NUMBERS -----
    if show_order:
        text_spans = [span for _, spans, _ in lines for span in spans]
        centers = []
        for text_span in text_spans:
            xs = [p[0] * scale for p in text_span.polygon]
            ys = [p[1] * scale for p in text_span.polygon]
            centers.append((sum(xs) / 4, sum(ys) / 4))

        # Draw connecting lines only if show_lines is True
        if show_lines:
            for p, c in zip(centers, centers[1:]):
                draw.line([p, c], fill=line_color, width=3)

        # Draw numbers only if show_numbers is True
        if show_numbers:
            overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
            d2 = ImageDraw.Draw(overlay)
            for cx, cy in centers:
                d2.rectangle([cx - 12, cy - 12, cx + 12, cy + 12], fill=number_bg + (140,))
            out = Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")

            draw = ImageDraw.Draw(out)
            for i, (cx, cy) in enumerate(centers, 1):
                draw.text((cx - 6, cy - 8), str(i), fill=number_color)

    return out
