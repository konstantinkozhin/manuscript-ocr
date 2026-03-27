from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from manuscript.api._page_helpers import filter_callable_kwargs
from manuscript.data import Page, TextSpan
from manuscript.utils import (
    crop_axis_aligned,
    crop_polygon_mask,
    polygon_to_bbox,
    warp_quad,
)

from .region_types import PreparedRegion, normalize_prepared_regions


def iter_candidate_text_spans(
    page: Page,
    *,
    min_text_size: int,
):
    for block in page.blocks:
        for line in block.lines:
            for text_span in line.text_spans:
                poly = np.asarray(text_span.polygon, dtype=np.float32)
                if poly.size == 0:
                    continue

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
    pad = float(options.get("pad", 0))
    regions: List[PreparedRegion] = []

    for text_span, poly in iter_candidate_text_spans(page, min_text_size=min_text_size):
        crop = crop_axis_aligned(image, poly, pad=pad)
        if crop is None:
            continue

        regions.append(
            PreparedRegion(
                text_span=text_span,
                image=rotate_region(crop),
                polygon=poly,
                meta={"region_preparer": "bbox", "pad": pad},
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
    pad = float(options.get("pad", 0))
    background = int(options.get("background", 255))
    regions: List[PreparedRegion] = []

    for text_span, poly in iter_candidate_text_spans(page, min_text_size=min_text_size):
        crop = crop_polygon_mask(image, poly, pad=pad, background=background)
        if crop is None:
            continue

        regions.append(
            PreparedRegion(
                text_span=text_span,
                image=rotate_region(crop),
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

    for text_span, poly in iter_candidate_text_spans(page, min_text_size=min_text_size):
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
                image=rotate_region(crop),
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
