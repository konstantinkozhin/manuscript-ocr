"""Common utilities for manuscript-ocr."""

# I/O utilities
from .io import read_image, create_page_from_text

# Visualization utilities
from .visualization import visualize_page

# Sorting and postprocessing utilities
from .sorting import organize_page

# Geometry utilities
from .geometry import (
    crop_axis_aligned,
    crop_polygon_mask,
    merge_polygons,
    order_quad_points,
    polygon_to_bbox,
    warp_quad,
)
from .page_transforms import (
    collapse_block_text_spans,
    collapse_line_text_spans,
    collapse_page_text_spans,
    merge_text_spans,
)

# Training utilities
from .training import set_seed


__all__ = [
    # I/O
    "read_image",
    "create_page_from_text",
    # Visualization
    "visualize_page",
    # Sorting/Postprocessing
    "organize_page",
    # Geometry
    "crop_axis_aligned",
    "crop_polygon_mask",
    "merge_polygons",
    "order_quad_points",
    "polygon_to_bbox",
    "warp_quad",
    "merge_text_spans",
    "collapse_line_text_spans",
    "collapse_block_text_spans",
    "collapse_page_text_spans",
    # Training
    "set_seed",
]
