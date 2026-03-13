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
    order_quad_points,
    polygon_to_bbox,
    warp_quad,
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
    "order_quad_points",
    "polygon_to_bbox",
    "warp_quad",
    # Training
    "set_seed",
]
