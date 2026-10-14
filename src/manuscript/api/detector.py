from abc import abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from manuscript.data import Page, Element
from .base import BaseArtifactModel


class BaseDetector(BaseArtifactModel):
    """Base class for detectors producing document-tree Pages.

    By default, predict() returns a Page. Optional return flags expose
    additional detection data:

    - return_raw: a dictionary containing the Page, detection geometry,
    class details, image size and metadata.
    - return_masks: also include available segmentation masks.
    - return_outputs: also include original model output tensors.

    Each flag enables dictionary output. Masks and model tensors are
    included only when explicitly requested.
    """

    def _load_detector_config(self, config=None, *, required=True):
        if config is None:
            config = getattr(self, "_resolved_model_artifacts", {}).get("config")
        if config is None:
            if required:
                raise ValueError(f"{type(self).__name__} requires a configuration; "
                                 "provide config explicitly for local weights, or use a registry preset")
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

    def _configure_region_types(self, config):
        self.region_classes = self._validate_region_types(config)

    @staticmethod
    def _validate_region_types(config):
        mappings = config.get("region_classes")
        if not isinstance(mappings, dict) or not mappings:
            raise ValueError("Detector config requires region_classes")
        region_classes = {}
        for class_id, fields in mappings.items():
            if (isinstance(class_id, bool) or not isinstance(class_id, (int, str))
                    or (isinstance(class_id, str) and not class_id.isdecimal())):
                raise ValueError("region_classes IDs must be nonnegative integers")
            class_id = int(class_id)
            if class_id < 0 or class_id in region_classes:
                raise ValueError("region_classes IDs must be nonnegative and unique")
            if not isinstance(fields, dict):
                raise ValueError("region_classes values must be objects")
            fields = dict(fields)
            if {"polygon", "detection_confidence"} & fields.keys():
                raise ValueError("region_classes cannot override detection geometry or confidence")
            kind = fields.get("type")
            if kind == "page":
                raise ValueError("Page cannot be a detection region")
            Element.resolve_type(kind)(**fields)
            region_classes[class_id] = fields
        return region_classes

    def _tree_from_regions(self, polygons, scores, class_ids, image_hw):
        if not (len(polygons) == len(scores) == len(class_ids)):
            raise ValueError("Detection arrays must have equal lengths")
        from manuscript.utils.geometry import repair_detection_polygon
        page = Page(image_height=int(image_hw[0]), image_width=int(image_hw[1]))
        nodes = []
        for polygon, score, class_id in zip(polygons, scores, class_ids):
            if int(class_id) not in self.region_classes:
                raise ValueError(f"Unconfigured detection class: {class_id}")
            polygon = repair_detection_polygon(polygon)
            if polygon is None:
                continue  # No two-dimensional geometry exists to recover.
            fields = self.region_classes[int(class_id)]
            nodes.append(Element.registry.get(fields["type"], Element)(
                **fields, polygon=np.asarray(polygon).astype(float).tolist(),
                detection_confidence=float(score)))
        page.children = nodes
        return page

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
