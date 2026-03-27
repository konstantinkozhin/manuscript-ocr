import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

from manuscript.utils import polygon_to_bbox

from .region_types import PreparedRegion, RecognitionPrediction


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_debug_regions(
    regions: Sequence[PreparedRegion],
    debug_save_dir: Union[str, Path],
    predictions: Optional[Sequence[RecognitionPrediction]] = None,
    *,
    write_images: bool = True,
) -> None:
    output_dir = Path(debug_save_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows: List[Dict[str, Any]] = []
    for idx, region in enumerate(regions):
        file_name = f"{idx:04d}.png"
        image_path = output_dir / file_name
        if write_images:
            image_to_save = np.asarray(region.image)
            if image_to_save.dtype != np.uint8:
                image_to_save = np.clip(image_to_save, 0, 255).astype(np.uint8)
            Image.fromarray(image_to_save).save(image_path)

        bbox = polygon_to_bbox(region.polygon)
        row: Dict[str, Any] = {
            "index": idx,
            "file_name": file_name,
            "region_preparer": region.meta.get("region_preparer"),
            "crop_shape": list(region.image.shape),
            "polygon": np.asarray(region.polygon, dtype=np.float32).tolist(),
            "bbox": list(bbox) if bbox is not None else None,
            "text_span_order": getattr(region.text_span, "order", None),
            "detection_confidence": getattr(region.text_span, "detection_confidence", None),
            "meta": make_json_safe(region.meta),
        }
        if predictions is not None and idx < len(predictions):
            row["prediction"] = {
                "text": predictions[idx].text,
                "confidence": predictions[idx].confidence,
                "meta": make_json_safe(predictions[idx].meta),
            }
        index_rows.append(row)

    (output_dir / "index.json").write_text(
        json.dumps(index_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


__all__ = ["make_json_safe", "save_debug_regions"]
