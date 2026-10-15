from pathlib import Path
import json
from copy import deepcopy

import cv2
import numpy as np

from manuscript.api.base import managed_session_initialization
from manuscript.api.detector import BaseDetector
from manuscript.utils import read_image
from manuscript.utils.geometry import repair_detection_polygon
from manuscript.utils.masks import contours_to_polygon
from manuscript.utils.numerics import softmax, sigmoid


class RFDETR(BaseDetector):
    """Initialize RF-DETR detection or instance segmentation with ONNX Runtime.

    Parameters
    ----------
    weights : str or Path
        Compatible ONNX model file or model registry key. Local files,
        HTTP/HTTPS URLs, ``github://owner/repo/tag/file.onnx`` and
        ``gdrive:FILE_ID`` are supported. There is no default preset.
    config : dict, str or Path, optional
        Inference configuration, or a path/identifier for its JSON/YAML file.
        Required for local weights; registry presets supply their config.
        ``class_names`` maps output class IDs to names, and ``region_classes``
        maps selected IDs to document fields, e.g. ``{0: {"type": "word"}}``.
        Preprocessing and tensor names must match the exported model.
    device : {"cuda", "coreml", "cpu"}, optional
        Execution device. ``None`` selects CUDA when ONNX Runtime reports GPU
        support, otherwise CPU. Select CoreML explicitly on supported macOS
        installations. Default is ``None``.
    score_thresh : float or None, optional
        Minimum detection confidence in [0, 1]. ``None`` uses the config's
        ``score_thresh``, falling back to 0.5 when absent.
    class_ids : sequence of int or None, optional
        Class IDs to retain. Each must appear in both ``class_names`` and
        ``region_classes``. ``None`` uses the config's ``class_ids``, or all
        mapped region IDs when that setting is absent.
    target_size : int or None, optional
        Square input size in pixels, without aspect-ratio preservation.
        ``None`` uses the config's ``target_size`` or the fixed ONNX input
        dimensions. Dynamic image dimensions require an explicit size.
        A supplied size must agree with a fixed ONNX input shape.
    force_download : bool, optional
        Download artifacts again instead of reusing cached files.
        Default is ``False``.
    mask_threshold : float or None, optional
        Foreground probability threshold for instance masks, strictly between
        0 and 1. ``None`` uses the config's ``mask_threshold``, falling back to
        0.5 when absent. Applies only to models with mask outputs.

    Examples
    --------
    Load compatible local weights with their matching inference config:

    >>> from manuscript.detectors._rfdetr import RFDETR
    >>> detector = RFDETR(
    ...     weights="model.onnx", config="config.json",
    ...     score_thresh=0.4, device="cpu",
    ... )
    >>> page = detector.predict("document.jpg")
    """

    registry_model_class = "RFDETR"
    def __init__(
        self,
        weights,
        config=None,
        device=None,
        score_thresh=None,
        class_ids=None,
        target_size=None,
        force_download=False,
        mask_threshold=None,
    ):
        if weights is None:
            raise ValueError("RFDETR requires explicit ONNX weights or a registry name")
        super().__init__(
            weights=str(weights), device=device, force_download=force_download
        )
        config, self.config_path = self._load_detector_config(config)
        self._configure_region_types(config)
        if config.get("schema_version", 1) != 1:
            raise ValueError("Unsupported RFDETR config schema_version")
        if config.get("task", "object_detection") != "object_detection":
            raise ValueError("RFDETR requires an object_detection config")
        self.score_thresh = float(
            config.get("score_thresh", 0.5) if score_thresh is None else score_thresh
        )
        names = config.get("class_names")
        if not isinstance(names, dict) or not names:
            raise ValueError("RFDETR config requires a nonempty class_names object")
        self.class_names = {}
        for key, name in names.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise ValueError("class_names keys must be nonnegative integer IDs")
            try:
                class_id = int(key)
            except ValueError as exc:
                raise ValueError("class_names keys must be nonnegative integer IDs") from exc
            if class_id < 0 or class_id in self.class_names:
                raise ValueError("class_names IDs must be nonnegative and unique")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("class_names values must be nonempty strings")
            self.class_names[class_id] = name
        selected = config.get("class_ids", list(self.region_classes)) if class_ids is None else class_ids
        if not isinstance(selected, (list, tuple)) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in selected
        ):
            raise ValueError("class_ids must contain nonnegative integers")
        self.class_ids = tuple(selected)
        if any(class_id not in self.class_names for class_id in self.class_ids):
            raise ValueError("Every selected class_id must be defined in class_names")
        if any(class_id not in self.region_classes for class_id in self.class_ids):
            raise ValueError("Every selected class_id must be defined in region_classes")
        self.num_select = int(config.get("num_select", 300))
        self.target_size = (
            target_size if target_size is not None else config.get("target_size")
        )
        if self.target_size is not None:
            if (
                isinstance(self.target_size, bool)
                or int(self.target_size) != self.target_size
                or self.target_size <= 0
            ):
                raise ValueError("target_size must be a positive integer")
            self.target_size = int(self.target_size)
        self.mean = np.asarray(config.get("mean", [0.485, 0.456, 0.406]), np.float32)
        self.std = np.asarray(config.get("std", [0.229, 0.224, 0.225]), np.float32)
        if (
            self.mean.shape != (3,)
            or self.std.shape != (3,)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or np.any(self.std <= 0)
        ):
            raise ValueError(
                "mean/std must contain three finite values and positive std"
            )
        if not 0 <= self.score_thresh <= 1 or self.num_select <= 0:
            raise ValueError("score_thresh must be in [0,1] and num_select positive")
        self.input_name = config.get("input_name", "images")
        self.boxes_output_name = config.get("boxes_output_name", "pred_boxes")
        self.logits_output_name = config.get("logits_output_name", "pred_logits")
        self.masks_output_name = config.get("masks_output_name")
        self.mask_threshold = float(
            config.get("mask_threshold", 0.5)
            if mask_threshold is None
            else mask_threshold
        )
        if (
            not 0 < self.mask_threshold < 1
        ):
            raise ValueError("Invalid geometry or mask_threshold")
        self.activation = config.get("activation", "sigmoid")
        if self.activation not in ("sigmoid", "softmax", "none"):
            raise ValueError("activation must be sigmoid, softmax or none")
        self.onnx_session = None
        self._input_dtype = np.float32
        self._input_hw = None

    @managed_session_initialization
    def _initialize_session(self):
        if self.onnx_session is not None:
            return
        session = self._create_onnx_session()
        inputs = session.get_inputs()
        if len(inputs) != 1 or inputs[0].name != self.input_name:
            raise ValueError("RFDETR expects one configured image input")
        item = inputs[0]
        if len(item.shape) != 4 or item.type not in (
            "tensor(float)",
            "tensor(float16)",
        ):
            raise ValueError("RFDETR expects a float32/float16 NCHW image input")
        if isinstance(item.shape[0], int) and item.shape[0] != 1:
            raise ValueError(
                "RFDETR.predict expects ONNX batch size 1 or dynamic batch"
            )
        if isinstance(item.shape[1], int) and item.shape[1] != 3:
            raise ValueError("RFDETR expects RGB input with 3 channels")
        if all(isinstance(value, int) and value > 0 for value in item.shape[2:]):
            self._input_hw = tuple(item.shape[2:])
            if self.target_size is not None and self._input_hw != (
                self.target_size,
                self.target_size,
            ):
                raise ValueError(
                    "target_size disagrees with the fixed ONNX image shape"
                )
        elif self.target_size:
            self._input_hw = (self.target_size, self.target_size)
        else:
            raise ValueError(
                "Dynamic ONNX images require target_size in config or constructor"
            )
        outputs = {output.name for output in session.get_outputs()}
        if self.masks_output_name is None and "pred_masks" in outputs:
            self.masks_output_name = "pred_masks"
        if self.masks_output_name and self.masks_output_name not in outputs:
            raise ValueError("Configured mask output is absent from ONNX graph")
        if not {self.boxes_output_name, self.logits_output_name} <= outputs:
            raise ValueError("Configured RFDETR outputs are absent from ONNX graph")
        self._input_dtype = np.float16 if item.type == "tensor(float16)" else np.float32
        self.onnx_session = session

    def _preprocess(self, image):
        height, width = self._input_hw
        # RF-DETR resizes float RGB tensors with bilinear, antialias=False.
        image = cv2.resize(
            image.astype(np.float32) / 255.0,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = (image - self.mean) / self.std
        return np.ascontiguousarray(
            tensor.transpose(2, 0, 1)[None], dtype=self._input_dtype
        )

    def _postprocess(
        self, boxes, logits, image_hw, return_raw=False, masks=None, *, return_masks=False, return_outputs=False
    ):
        raw_outputs = {self.boxes_output_name: boxes, self.logits_output_name: logits} if return_outputs else None
        if return_outputs and masks is not None:
            raw_outputs[self.masks_output_name or "pred_masks"] = masks
        boxes, logits = np.asarray(boxes, np.float32), np.asarray(logits, np.float32)
        if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[2] != 4:
            raise ValueError("RFDETR boxes must have shape [1, queries, 4]")
        if (
            logits.ndim != 3
            or logits.shape[:2] != boxes.shape[:2]
            or logits.shape[2] == 0
        ):
            raise ValueError("RFDETR logits must have shape [1, queries, classes]")
        if not np.isfinite(boxes).all() or not np.isfinite(logits).all():
            raise ValueError("RFDETR outputs contain non-finite values")
        if self.activation == "sigmoid":
            scores = sigmoid(logits)
        elif self.activation == "softmax":
            scores = softmax(logits, axis=2)
        else:
            scores = logits
            if np.any(scores < 0) or np.any(scores > 1):
                raise ValueError("RFDETR probabilities must be in [0,1]")
        # RF-DETR selects top query/class pairs globally and uses no NMS.
        indices = np.argsort(-scores.ravel(), kind="stable")[: self.num_select]
        height, width = image_hw
        if masks is not None:
            masks = np.asarray(masks, np.float32)
            if (
                masks.ndim != 4
                or masks.shape[:2] != boxes.shape[:2]
                or not np.isfinite(masks).all()
            ):
                raise ValueError(
                    "RFDETR masks must have shape [1, queries, height, width] and finite logits"
                )
        threshold = getattr(self, "mask_threshold", 0.5)
        mask_cache = {}
        detections, polygons, confidences, region_ids = [], [], [], []
        for index in indices:
            score = float(scores.ravel()[index])
            if score < self.score_thresh:
                break
            query, class_id = divmod(int(index), scores.shape[2])
            cx, cy, box_width, box_height = boxes[0, query]
            if box_width <= 0 or box_height <= 0:
                continue
            left, right = (
                np.clip([cx - box_width / 2, cx + box_width / 2], 0, 1) * width
            )
            top, bottom = (
                np.clip([cy - box_height / 2, cy + box_height / 2], 0, 1) * height
            )
            if right <= left or bottom <= top:
                continue
            polygon = [
                (float(left), float(top)),
                (float(right), float(top)),
                (float(right), float(bottom)),
                (float(left), float(bottom)),
            ]
            bbox_polygon = polygon
            extra = {"geometry_source": "bbox"}
            if masks is not None:
                if query not in mask_cache:
                    resized = cv2.resize(
                        masks[0, query], (width, height), interpolation=cv2.INTER_LINEAR
                    )
                    binary = (resized > np.log(threshold / (1 - threshold))).astype(
                        np.uint8
                    )
                    contours, hierarchy = cv2.findContours(
                        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                    )
                    valid = [
                        i
                        for i, contour in enumerate(contours)
                        if len(contour) >= 3 and cv2.contourArea(contour) > 0
                    ]
                    outer = [i for i in valid if hierarchy[0, i, 3] == -1]
                    info = {
                        "contours": [
                            contours[i][:, 0].astype(float).tolist() for i in valid
                        ],
                        "contour_hierarchy": [hierarchy[0, i].tolist() for i in valid],
                        "contour_indices": valid,
                    }
                    if return_masks:
                        info["mask"] = binary
                    mask_cache[query] = (
                        contours_to_polygon([contours[i] for i in outer]), info,
                    )
                outline, info = mask_cache[query]
                extra.update(info)
                if outline is not None:
                    polygon = outline.tolist()
                    extra["geometry_source"] = "mask_polygon"
            polygon = repair_detection_polygon(polygon, fallback=bbox_polygon).tolist()
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, str(class_id)),
                    "confidence": score,
                    "bbox": [float(left), float(top), float(right), float(bottom)],
                    "polygon": polygon,
                    "query_index": query,
                    **extra,
                }
            )
            if class_id in self.class_ids:
                polygons.append(polygon)
                confidences.append(score)
                region_ids.append(class_id)
        page = self._tree_from_regions(polygons, confidences, region_ids, image_hw)
        if return_raw or return_masks or return_outputs:
            return self._raw_result(page, raw_outputs, detections, image_hw,
                                    class_names=self.class_names.copy())
        return page

    def predict(self, image, return_raw=False, *, return_masks=False, return_outputs=False):
        """Return text Page or compact detections; masks/tensors are opt-in."""
        self._initialize_session()
        image = read_image(image)
        names = [self.boxes_output_name, self.logits_output_name]
        if self.masks_output_name:
            names.append(self.masks_output_name)
        outputs = self.onnx_session.run(
            names, {self.input_name: self._preprocess(image)}
        )
        return self._postprocess(
            outputs[0],
            outputs[1],
            image.shape[:2],
            return_raw=return_raw,
            masks=outputs[2] if len(outputs) > 2 else None,
            return_masks=return_masks, return_outputs=return_outputs,
        )


    @staticmethod
    def create_config(source, *, region_classes=None, output_path=None):
        """Preserve inference settings and add structural classes and angles.

        Class semantics are supplied explicitly, not inferred from class names.
        The deprecated text-only filter is removed; all mapped classes are
        included unless the source has an explicit ``class_ids`` filter.
        """
        config = deepcopy(source) if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
        mapping = region_classes if region_classes is not None else config.get("region_classes")
        mapping = BaseDetector._validate_region_types({"region_classes": mapping})
        config["region_classes"] = {str(key): fields for key, fields in mapping.items()}
        config.pop("text_class_ids", None)
        if output_path is not None:
            Path(output_path).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return config
