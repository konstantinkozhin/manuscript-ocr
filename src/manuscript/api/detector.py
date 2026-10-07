from abc import abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import onnxruntime as ort

from manuscript.data import Block, Line, Page, TextSpan
from .base import BaseArtifactModel


class BaseDetector(BaseArtifactModel):
    """Page by default; return_raw includes compact geometry and class details.

    Results contain page, detections, image_size and metadata. return_masks
    adds available masks; return_outputs adds named original model tensors,
    including mask logits when the graph emits them. Either flag also enables
    dictionary output. Heavy arrays are retained only when explicitly requested.
    """

    def _load_detector_config(self, config=None, *, suffixes=(".json",), required=True):
        if config is None:
            config = getattr(self, "_resolved_model_artifacts", {}).get("config")
            if config is None:
                config = next((Path(self.weights).with_suffix(suffix)
                               for suffix in suffixes
                               if Path(self.weights).with_suffix(suffix).is_file()), None)
        if config is None:
            if required:
                raise ValueError(f"{type(self).__name__} requires a configuration; "
                                 "provide config, a registry artifact, or a sidecar")
            return None, None
        if isinstance(config, dict):
            return dict(config), None
        path = self._resolve_extra_artifact(str(config), default_name=None,
                                           registry={}, description="config")
        with open(path, encoding="utf-8") as stream:
            if Path(path).suffix.lower() in (".yaml", ".yml"):
                import yaml
                values = yaml.safe_load(stream)
            else:
                values = json.load(stream)
        if not isinstance(values, dict):
            raise ValueError("Detector config must be an object")
        return values, path

    @staticmethod
    def _page_from_regions(polygons, scores, *, one_per_line=False, ordered=False):
        spans = [TextSpan(polygon=[(float(x), float(y)) for x, y in polygon],
                          detection_confidence=float(score), order=i if ordered else None)
                 for i, (polygon, score) in enumerate(zip(polygons, scores))]
        lines = ([Line(text_spans=[span]) for span in spans] if one_per_line else
                 [Line(text_spans=spans, order=0 if ordered else None)])
        return Page(blocks=[Block(lines=lines, order=0 if ordered else None)])

    @staticmethod
    def _raw_result(page, outputs, detections, image_hw, **metadata):
        result = {"page": page, "detections": detections,
                "image_size": {"height": int(image_hw[0]), "width": int(image_hw[1])},
                "metadata": metadata}
        if outputs is not None:
            result["outputs"] = outputs
        return result

    @staticmethod
    def _mask_details(mask, *, include_mask=True):
        import cv2
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        return {**({"mask": mask} if include_mask else {}),
                "contours": [contour[:, 0].astype(float).tolist() for contour in contours],
                "contour_hierarchy": hierarchy[0].tolist() if hierarchy is not None else []}

    @abstractmethod
    def predict(self, image: Any, return_raw: bool = False, *, return_masks: bool = False,
                return_outputs: bool = False, **kwargs: Any) -> Union[Page, Dict[str, Any]]: ...


__all__ = ["BaseDetector"]
