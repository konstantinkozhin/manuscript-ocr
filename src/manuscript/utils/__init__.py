"""Common utilities for manuscript-ocr."""

# I/O utilities
from .io import read_image, create_page_from_text, create_page_from_image

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

__all__ = [
    # I/O
    "read_image",
    "create_page_from_text",
    "create_page_from_image",
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


def __getattr__(name):
    if name == "visualize_page":
        from .visualization import visualize_page

        return visualize_page

    if name == "set_seed":
        from .training import set_seed

        return set_seed

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
