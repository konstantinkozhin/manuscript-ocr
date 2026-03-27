from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from manuscript.data import TextSpan
from manuscript.utils import read_image

REGION_PREPARER_PRESETS = frozenset({"bbox", "polygon_mask", "quad_warp"})


@dataclass(slots=True)
class PreparedRegion:
    """Prepared text crop associated with a specific ``TextSpan`` object."""

    text_span: Optional[TextSpan]
    image: np.ndarray
    polygon: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RecognitionPrediction:
    """Recognition result for a single prepared text crop."""

    text: Optional[str]
    confidence: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def normalize_prepared_regions(regions: Sequence[Any]) -> List[PreparedRegion]:
    normalized: List[PreparedRegion] = []
    for region in regions:
        if isinstance(region, PreparedRegion):
            normalized.append(
                PreparedRegion(
                    text_span=region.text_span,
                    image=read_image(region.image),
                    polygon=np.asarray(region.polygon, dtype=np.float32),
                    meta=dict(region.meta),
                )
            )
            continue

        if not isinstance(region, dict):
            raise TypeError(
                "Each prepared region must be a PreparedRegion or dict with at "
                "least 'text_span' and 'image' keys"
            )

        if "text_span" not in region or "image" not in region:
            raise ValueError(
                "Prepared region dict must contain 'text_span' and 'image' keys"
            )

        text_span = region["text_span"]
        polygon = region.get("polygon", getattr(text_span, "polygon", None))
        if polygon is None:
            raise ValueError("Prepared region must include polygon coordinates")

        normalized.append(
            PreparedRegion(
                text_span=text_span,
                image=read_image(region["image"]),
                polygon=np.asarray(polygon, dtype=np.float32),
                meta=dict(region.get("meta", {})),
            )
        )

    return normalized


def normalize_recognition_predictions(
    predictions: Sequence[Any],
) -> List[RecognitionPrediction]:
    normalized: List[RecognitionPrediction] = []
    for prediction in predictions:
        if prediction is None:
            normalized.append(RecognitionPrediction(text=None, confidence=None))
            continue

        if isinstance(prediction, RecognitionPrediction):
            normalized.append(
                RecognitionPrediction(
                    text=prediction.text,
                    confidence=prediction.confidence,
                    meta=dict(prediction.meta),
                )
            )
            continue

        if not isinstance(prediction, dict):
            raise TypeError(
                "Each prediction must be a RecognitionPrediction, dict, or None"
            )

        normalized.append(
            RecognitionPrediction(
                text=prediction.get("text"),
                confidence=prediction.get(
                    "confidence", prediction.get("recognition_confidence")
                ),
                meta=dict(prediction.get("meta", {})),
            )
        )

    return normalized


__all__ = [
    "PreparedRegion",
    "RecognitionPrediction",
    "REGION_PREPARER_PRESETS",
    "normalize_prepared_regions",
    "normalize_recognition_predictions",
]
