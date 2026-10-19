import math
from numbers import Real
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from manuscript.api._page_helpers import filter_callable_kwargs
from manuscript.data import Page
from manuscript.utils.geometry import (
    crop_axis_aligned,
    crop_polygon_mask,
    polygon_to_bbox,
    warp_quad,
)

from .region_types import PreparedRegion, normalize_prepared_regions


_RELATIVE_BBOX_OPTIONS = ("padding_x", "padding_y", "inset_x", "inset_y")


def validate_region_preparer_options(preparer, options):
    """Validate relative bbox margins before downloading or loading a model."""
    if not isinstance(preparer, str):
        return
    if preparer != "bbox":
        if any(name in options for name in _RELATIVE_BBOX_OPTIONS):
            raise ValueError("padding_x, padding_y, inset_x and inset_y require region_preparer='bbox'")
        return
    for name in _RELATIVE_BBOX_OPTIONS:
        value = options.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a nonnegative finite number")
        if name.startswith("inset") and value >= 0.5:
            raise ValueError(f"{name} must be less than 0.5")
    for axis in ("x", "y"):
        if options.get(f"padding_{axis}", 0) and options.get(f"inset_{axis}", 0):
            raise ValueError(f"Specify padding_{axis} or inset_{axis}, not both")
    if any(options.get(name, 0) for name in _RELATIVE_BBOX_OPTIONS) and float(options.get("pad", 0)) != 0:
        raise ValueError("Pixel pad cannot be combined with relative bbox margins")


def _relative_bbox_crop(image, polygon, options):
    # Use floating-point bounds before clipping or rounding. In particular,
    # off-image portions still contribute to the original width and height.
    x1, y1 = polygon.min(axis=0).astype(np.float64)
    x2, y2 = polygon.max(axis=0).astype(np.float64)
    dx = (x2 - x1) * (options.get("padding_x", 0.0) - options.get("inset_x", 0.0))
    dy = (y2 - y1) * (options.get("padding_y", 0.0) - options.get("inset_y", 0.0))
    height, width = image.shape[:2]
    left = math.floor(np.clip(x1 - dx, 0, width))
    top = math.floor(np.clip(y1 - dy, 0, height))
    right = math.ceil(np.clip(x2 + dx, 0, width))
    bottom = math.ceil(np.clip(y2 + dy, 0, height))
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right].copy()


def _rotate_crop(rotate_region, crop, region):
    kwargs = filter_callable_kwargs(rotate_region, {"region": region})
    return rotate_region(crop, **kwargs)


def iter_candidate_text_spans(
    page: Page,
    *,
    min_text_size: int,
    levels=("word", "line"),
):
    from manuscript.data.tree import recognition_candidates
    for text_span in recognition_candidates(page, levels=levels):
        poly = np.asarray(text_span.polygon, dtype=np.float32)
        bbox = polygon_to_bbox(poly)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        if (x2 - x1) < min_text_size or (y2 - y1) < min_text_size:
            continue
        yield text_span, poly


def prepare_bbox_regions(
    page: Page,
    image: np.ndarray,
    *,
    min_text_size: int,
    rotate_region: Callable[[np.ndarray], np.ndarray],
    options: Optional[Dict[str, Any]] = None,
) -> List[PreparedRegion]:
    options = dict(options or {})
    validate_region_preparer_options("bbox", options)
    pad = float(options.get("pad", 0))
    relative = any(options.get(name, 0) for name in _RELATIVE_BBOX_OPTIONS)
    regions: List[PreparedRegion] = []

    for text_span, poly in iter_candidate_text_spans(page, min_text_size=min_text_size,
                                                    levels=options.get("levels", ("word", "line"))):
        crop = (_relative_bbox_crop(image, poly, options) if relative
                else crop_axis_aligned(image, poly, pad=pad))
        if crop is None:
            continue

        regions.append(
            PreparedRegion(
                text_span=text_span,
                image=_rotate_crop(rotate_region, crop, text_span),
                polygon=poly,
                meta={"region_preparer": "bbox", "pad": pad,
                      **({name: float(options.get(name, 0.0)) for name in _RELATIVE_BBOX_OPTIONS}
                         if relative else {})},
            )
        )

    return regions


def prepare_polygon_mask_regions(
    page: Page,
    image: np.ndarray,
    *,
    min_text_size: int,
    rotate_region: Callable[[np.ndarray], np.ndarray],
    options: Optional[Dict[str, Any]] = None,
) -> List[PreparedRegion]:
    options = dict(options or {})
    validate_region_preparer_options("polygon_mask", options)
    pad = float(options.get("pad", 0))
    background = int(options.get("background", 255))
    regions: List[PreparedRegion] = []

    for text_span, poly in iter_candidate_text_spans(page, min_text_size=min_text_size,
                                                    levels=options.get("levels", ("word", "line"))):
        crop = crop_polygon_mask(image, poly, pad=pad, background=background)
        if crop is None:
            continue

        regions.append(
            PreparedRegion(
                text_span=text_span,
                image=_rotate_crop(rotate_region, crop, text_span),
                polygon=poly,
                meta={
                    "region_preparer": "polygon_mask",
                    "pad": pad,
                    "background": background,
                },
            )
        )

    return regions


def prepare_quad_warp_regions(
    page: Page,
    image: np.ndarray,
    *,
    min_text_size: int,
    rotate_region: Callable[[np.ndarray], np.ndarray],
    options: Optional[Dict[str, Any]] = None,
) -> List[PreparedRegion]:
    options = dict(options or {})
    validate_region_preparer_options("quad_warp", options)
    raw_output_size = options.get("output_size")
    fallback_to_bbox = bool(options.get("fallback_to_bbox", True))
    output_size = None
    if raw_output_size is not None:
        if not isinstance(raw_output_size, (list, tuple)) or len(raw_output_size) != 2:
            raise ValueError(
                "quad_warp output_size must be a tuple/list like (width, height)"
            )
        output_size = (int(raw_output_size[0]), int(raw_output_size[1]))

    regions: List[PreparedRegion] = []

    for text_span, poly in iter_candidate_text_spans(page, min_text_size=min_text_size,
                                                    levels=options.get("levels", ("word", "line"))):
        crop = warp_quad(image, poly, output_size=output_size)
        used_fallback = False
        if crop is None and fallback_to_bbox:
            crop = crop_axis_aligned(image, poly, pad=0)
            used_fallback = True
        if crop is None:
            continue

        regions.append(
            PreparedRegion(
                text_span=text_span,
                image=_rotate_crop(rotate_region, crop, text_span),
                polygon=poly,
                meta={
                    "region_preparer": "quad_warp",
                    "output_size": output_size,
                    "fallback_to_bbox": used_fallback,
                },
            )
        )

    return regions


def prepare_text_regions(
    page: Page,
    image: np.ndarray,
    *,
    preset: str,
    min_text_size: int,
    rotate_region: Callable[[np.ndarray], np.ndarray],
    options: Optional[Dict[str, Any]] = None,
) -> List[PreparedRegion]:
    if preset == "bbox":
        return prepare_bbox_regions(
            page,
            image,
            min_text_size=min_text_size,
            rotate_region=rotate_region,
            options=options,
        )
    if preset == "polygon_mask":
        return prepare_polygon_mask_regions(
            page,
            image,
            min_text_size=min_text_size,
            rotate_region=rotate_region,
            options=options,
        )
    if preset == "quad_warp":
        return prepare_quad_warp_regions(
            page,
            image,
            min_text_size=min_text_size,
            rotate_region=rotate_region,
            options=options,
        )

    raise ValueError(f"Unsupported region_preparer preset: {preset}")


def call_region_preparer(
    page: Page,
    image: np.ndarray,
    *,
    preparer: Union[str, Callable[..., Sequence[Any]]],
    options: Optional[Dict[str, Any]],
    recognizer: Any,
    min_text_size: int,
    rotate_region: Callable[[np.ndarray], np.ndarray],
) -> List[PreparedRegion]:
    if isinstance(preparer, str):
        return prepare_text_regions(
            page=page,
            image=image,
            preset=preparer,
            min_text_size=min_text_size,
            rotate_region=rotate_region,
            options=options,
        )

    kwargs = filter_callable_kwargs(
        preparer,
        {
            "page": page,
            "image": image,
            "recognizer": recognizer,
            "options": dict(options or {}),
        },
    )
    return normalize_prepared_regions(preparer(**kwargs))


__all__ = [
    "call_region_preparer",
    "iter_candidate_text_spans",
    "prepare_bbox_regions",
    "prepare_polygon_mask_regions",
    "prepare_quad_warp_regions",
    "prepare_text_regions",
]
