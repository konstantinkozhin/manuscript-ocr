from .debug import make_json_safe, save_debug_regions
from .region_preparers import (
    call_region_preparer,
    iter_candidate_text_spans,
    prepare_bbox_regions,
    prepare_polygon_mask_regions,
    prepare_quad_warp_regions,
    prepare_text_regions,
)
from .region_types import (
    REGION_PREPARER_PRESETS,
    PreparedRegion,
    RecognitionPrediction,
    normalize_prepared_regions,
    normalize_recognition_predictions,
)

__all__ = [
    "PreparedRegion",
    "RecognitionPrediction",
    "REGION_PREPARER_PRESETS",
    "call_region_preparer",
    "iter_candidate_text_spans",
    "make_json_safe",
    "normalize_prepared_regions",
    "normalize_recognition_predictions",
    "prepare_bbox_regions",
    "prepare_polygon_mask_regions",
    "prepare_quad_warp_regions",
    "prepare_text_regions",
    "save_debug_regions",
]
