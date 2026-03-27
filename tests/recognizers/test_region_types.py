import numpy as np
import pytest

from manuscript.data import TextSpan
from manuscript.recognizers._common.region_types import (
    PreparedRegion,
    RecognitionPrediction,
    normalize_prepared_regions,
    normalize_recognition_predictions,
)


def _make_text_span():
    return TextSpan(
        polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)],
        detection_confidence=0.95,
    )


def test_normalize_prepared_regions_copies_prepared_region():
    text_span = _make_text_span()
    region = PreparedRegion(
        text_span=text_span,
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        polygon=np.asarray(text_span.polygon, dtype=np.int32),
        meta={"source": "prepared"},
    )

    normalized = normalize_prepared_regions([region])

    assert len(normalized) == 1
    assert normalized[0] is not region
    assert normalized[0].text_span == text_span
    assert normalized[0].polygon.dtype == np.float32
    assert normalized[0].meta == {"source": "prepared"}
    assert normalized[0].meta is not region.meta


def test_normalize_prepared_regions_accepts_dict_and_uses_text_span_polygon():
    text_span = _make_text_span()

    normalized = normalize_prepared_regions(
        [
            {
                "text_span": text_span,
                "image": np.zeros((4, 4, 3), dtype=np.uint8),
                "meta": {"source": "dict"},
            }
        ]
    )

    assert len(normalized) == 1
    assert normalized[0].text_span == text_span
    assert normalized[0].polygon.shape == (4, 2)
    assert normalized[0].meta == {"source": "dict"}


def test_normalize_prepared_regions_rejects_invalid_type():
    with pytest.raises(TypeError, match="PreparedRegion or dict"):
        normalize_prepared_regions([123])


def test_normalize_prepared_regions_requires_text_span_and_image_keys():
    with pytest.raises(ValueError, match="must contain 'text_span' and 'image' keys"):
        normalize_prepared_regions([{"image": np.zeros((4, 4, 3), dtype=np.uint8)}])


def test_normalize_prepared_regions_requires_polygon_when_missing_from_text_span():
    with pytest.raises(ValueError, match="must include polygon coordinates"):
        normalize_prepared_regions(
            [{"text_span": None, "image": np.zeros((4, 4, 3), dtype=np.uint8)}]
        )


def test_normalize_recognition_predictions_accepts_none():
    normalized = normalize_recognition_predictions([None])

    assert normalized == [RecognitionPrediction(text=None, confidence=None, meta={})]


def test_normalize_recognition_predictions_copies_prediction_dataclass():
    prediction = RecognitionPrediction(
        text="hello",
        confidence=0.9,
        meta={"source": "model"},
    )

    normalized = normalize_recognition_predictions([prediction])

    assert normalized[0] is not prediction
    assert normalized[0].text == "hello"
    assert normalized[0].confidence == 0.9
    assert normalized[0].meta == {"source": "model"}
    assert normalized[0].meta is not prediction.meta


def test_normalize_recognition_predictions_supports_recognition_confidence_alias():
    normalized = normalize_recognition_predictions(
        [{"text": "hello", "recognition_confidence": 0.77}]
    )

    assert normalized[0].text == "hello"
    assert normalized[0].confidence == 0.77


def test_normalize_recognition_predictions_rejects_invalid_type():
    with pytest.raises(TypeError, match="RecognitionPrediction, dict, or None"):
        normalize_recognition_predictions([object()])
