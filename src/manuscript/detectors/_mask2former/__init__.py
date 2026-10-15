from pathlib import Path
import json
from typing import Dict, Optional, Union

import cv2
import numpy as np
import onnxruntime as ort

from manuscript.api.detector import BaseDetector
from ...utils import read_image
from manuscript.utils.masks import mask_iou, mask_to_polygon
from manuscript.utils.numerics import softmax, sigmoid


class Mask2Former(BaseDetector):
    """Initialize Mask2Former instance segmentation with ONNX Runtime.

    Parameters
    ----------
    weights : str or Path, optional
        Compatible ONNX model file or model registry key. Local files,
        HTTP/HTTPS URLs, ``github://owner/repo/tag/file.onnx`` and
        ``gdrive:FILE_ID`` are supported. ``None`` selects the default preset
        ``mask2former_line_v0_prev`` and downloads its supporting artifacts.
    config : dict, str or Path, optional
        Inference configuration, or a path/identifier for its JSON/YAML file.
        Required for local weights; registry presets supply their config.
        ``region_classes`` maps output class IDs to document fields,
        e.g. ``{0: {"type": "line"}}``. Image normalization, input size and
        tensor names must match the exported graph. ``create_config()`` builds
        a config dictionary with standard defaults and an explicit class mapping.
    device : {"cuda", "coreml", "cpu"}, optional
        Execution device. ``None`` selects CUDA when ONNX Runtime reports GPU
        support, otherwise CPU. Select CoreML explicitly on supported macOS
        installations. Default is ``None``.
    force_download : bool, optional
        Download artifacts again instead of reusing cached files.
        Default is ``False``.
    score_threshold : float or None, optional
        Minimum confidence in [0, 1], applied to the class probability and the
        final score (class probability times mean foreground-mask probability).
        ``None`` uses the config value, or 0.15 when absent.
    mask_logit_threshold : float or None, optional
        Finite logit threshold for creating binary masks, before sigmoid.
        ``None`` uses the config value, or 0.0 when absent.
    nms_iou_threshold : float or None, optional
        Mask IoU threshold in [0, 1] for suppressing overlapping detections
        of the same class. ``None`` uses the config value, or 0.5 when absent.
    min_mask_area : int or None, optional
        Minimum foreground area in source-image pixels; must be a positive
        integer. ``None`` uses the config value, or 16 when absent.
    **kwargs
        Additional options passed to ``BaseArtifactModel``.

    Examples
    --------
    Load the text-line preset:

    >>> from manuscript.detectors._mask2former import Mask2Former
    >>> detector = Mask2Former(weights="mask2former_line_v0_prev", device="cpu")
    >>> page = detector.predict("document.jpg")

    Load compatible local weights with their matching config:

    >>> detector = Mask2Former(weights="model.onnx", config="config.json")
    """

    default_weights_name = "mask2former_line_v0_prev"
    registry_model_class = "Mask2Former"
    pretrained_registry: Dict[str, str] = {}
    config_registry: Dict[str, str] = {}

    @staticmethod
    def create_config(*, image_size=1024, mask_interpolation_size=384,
                      image_mean=(0.485, 0.456, 0.406),
                      image_std=(0.229, 0.224, 0.225), score_threshold=0.15,
                      mask_logit_threshold=0.0, nms_iou_threshold=0.5,
                      min_mask_area=16, region_classes=None, output_path=None):
        """Build an inference config; class 0 defaults to text lines.

        Optionally save JSON to ``output_path``. The returned dictionary can
        also be passed directly to ``Mask2Former(config=...)``.
        """
        for name, value in (("image_size", image_size),
                            ("mask_interpolation_size", mask_interpolation_size),
                            ("min_mask_area", min_mask_area)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        mean, std = np.asarray(image_mean, dtype=float), np.asarray(image_std, dtype=float)
        if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("image_mean and image_std require three finite values and positive std")
        for name, value in (("score_threshold", score_threshold), ("nms_iou_threshold", nms_iou_threshold)):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not np.isfinite(mask_logit_threshold):
            raise ValueError("mask_logit_threshold must be finite")
        mapping = BaseDetector._validate_region_types({"region_classes":
            {0: {"type": "line"}} if region_classes is None else region_classes})
        values = {
            "schema_version": 1, "image_size": image_size,
            "mask_interpolation_size": mask_interpolation_size,
            "image_mean": mean.tolist(), "image_std": std.tolist(),
            "score_threshold": float(score_threshold),
            "mask_logit_threshold": float(mask_logit_threshold),
            "nms_iou_threshold": float(nms_iou_threshold), "min_mask_area": min_mask_area,
            "input_pixel_values": "pixel_values", "input_pixel_mask": "pixel_mask",
            "output_class_logits": "class_queries_logits", "output_mask_logits": "masks_queries_logits",
            "region_classes": {str(key): fields for key, fields in mapping.items()},
        }
        if output_path is not None:
            Path(output_path).write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return values


    def __init__(self, weights: Optional[Union[str, Path]] = None,
                 config: Optional[Union[str, Path]] = None,
                 device: Optional[str] = None, force_download: bool = False, *,
                 score_threshold: Optional[float] = None,
                 mask_logit_threshold: Optional[float] = None,
                 nms_iou_threshold: Optional[float] = None,
                 min_mask_area: Optional[int] = None, **kwargs):
        super().__init__(weights=weights, device=device,
                         force_download=force_download, **kwargs)
        self.onnx_session = None
        values, self.config_path = self._load_detector_config(config)
        self._configure_region_types(values)
        if values.get("schema_version") != 1:
            raise ValueError("Unsupported Mask2Former config schema")
        options = {key: values[key] for key in (
            "image_size", "mask_interpolation_size", "image_mean", "image_std",
            "score_threshold", "mask_logit_threshold", "nms_iou_threshold",
            "min_mask_area", "region_classes") if key in values}
        for key, value in (("score_threshold", score_threshold),
                           ("mask_logit_threshold", mask_logit_threshold),
                           ("nms_iou_threshold", nms_iou_threshold),
                           ("min_mask_area", min_mask_area)):
            if value is not None:
                options[key] = value
        validated = self.create_config(**options)
        for key, value in validated.items():
            values.setdefault(key, value)
        self.image_size = int(values["image_size"])
        self.mask_interpolation_size = int(values.get("mask_interpolation_size", 384))
        self.image_mean = np.asarray(values["image_mean"], dtype=np.float32).reshape(1, 1, 3)
        self.image_std = np.asarray(values["image_std"], dtype=np.float32).reshape(1, 1, 3)
        self.score_threshold = float(values["score_threshold"] if score_threshold is None else score_threshold)
        self.mask_logit_threshold = float(values["mask_logit_threshold"] if mask_logit_threshold is None else mask_logit_threshold)
        self.nms_iou_threshold = float(values["nms_iou_threshold"] if nms_iou_threshold is None else nms_iou_threshold)
        self.min_mask_area = int(values["min_mask_area"] if min_mask_area is None else min_mask_area)
        self.input_pixel_values = values.get("input_pixel_values", "pixel_values")
        self.input_pixel_mask = values.get("input_pixel_mask", "pixel_mask")
        self.output_class_logits = values.get("output_class_logits", "class_queries_logits")
        self.output_mask_logits = values.get("output_mask_logits", "masks_queries_logits")

    def _initialize_session(self):
        if self.onnx_session is not None:
            return
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        session = self._create_onnx_session(sess_options=session_options)
        inputs = {item.name: item for item in session.get_inputs()}
        if set(inputs) != {self.input_pixel_values, self.input_pixel_mask}:
            raise ValueError("Mask2Former requires configured image and pixel-mask inputs")
        pixel_input = inputs[self.input_pixel_values]
        if pixel_input.type not in ("tensor(float)", "tensor(float16)"):
            raise ValueError("Mask2Former image input must be float32 or float16")
        if inputs[self.input_pixel_mask].type != "tensor(int64)":
            raise ValueError("Mask2Former pixel mask must be int64")
        self.pixel_dtype = np.float16 if pixel_input.type == "tensor(float16)" else np.float32
        self.onnx_session = session

    def _preprocess(self, image):
        height, width = image.shape[:2]
        scale = min(self.image_size / width, self.image_size / height)
        resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = cv2.resize(image, resized_size, interpolation=cv2.INTER_LINEAR)
        left = (self.image_size - resized_size[0]) // 2
        top = (self.image_size - resized_size[1]) // 2
        canvas = np.full((self.image_size, self.image_size, 3), 255, dtype=np.uint8)
        canvas[top:top + resized_size[1], left:left + resized_size[0]] = resized
        pixels = canvas.astype(np.float32) / 255.0
        pixels = ((pixels - self.image_mean) / self.image_std).transpose(2, 0, 1)[None]
        pixels = pixels.astype(self.pixel_dtype, copy=False)
        return pixels, left, top, resized_size

    def _postprocess(self, class_logits, mask_logits, geometry):
        original_height, original_width, left, top, resized_width, resized_height = geometry
        if (class_logits.ndim != 2 or class_logits.shape[1] < 2 or mask_logits.ndim != 3
                or class_logits.shape[0] != mask_logits.shape[0]
                or not np.isfinite(class_logits).all() or not np.isfinite(mask_logits).all()):
            raise ValueError("Invalid Mask2Former class or mask outputs")
        probabilities = softmax(class_logits.astype(np.float32))
        class_ids = probabilities.argmax(axis=-1)
        class_scores = probabilities[np.arange(len(class_ids)), class_ids]
        candidates = []
        for query, class_score in enumerate(class_scores):
            class_id = int(class_ids[query])
            if class_id == probabilities.shape[1] - 1:
                continue
            if class_score < self.score_threshold:
                continue
            logits = cv2.resize(mask_logits[query].astype(np.float32),
                                (self.mask_interpolation_size,) * 2,
                                interpolation=cv2.INTER_LINEAR)
            binary = logits > self.mask_logit_threshold
            if not binary.any():
                continue
            score = float(class_score * sigmoid(logits)[binary].mean())
            if score < self.score_threshold:
                continue
            model_mask = cv2.resize(binary.astype(np.uint8),
                                    (self.image_size,) * 2,
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
            cropped = model_mask[top:top + resized_height,
                                 left:left + resized_width]
            restored = cv2.resize(cropped.astype(np.uint8),
                                  (original_width, original_height),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            if int(restored.sum()) >= self.min_mask_area:
                candidates.append((restored, score, class_id))
        candidates.sort(key=lambda item: item[1], reverse=True)
        kept = []
        for mask, score, class_id in candidates:
            if all(old_class != class_id or mask_iou(mask, old) <= self.nms_iou_threshold
                   for old, _, old_class in kept):
                kept.append((mask, score, class_id))
        return kept

    def predict(self, image, return_raw=False, *, return_masks=False, return_outputs=False):
        if self.onnx_session is None:
            self._initialize_session()
        source = read_image(image)
        original_height, original_width = source.shape[:2]
        pixels, left, top, resized_size = self._preprocess(source)
        pixel_mask = np.ones((1, self.image_size, self.image_size), dtype=np.int64)
        class_logits, mask_logits = self.onnx_session.run(
            [self.output_class_logits, self.output_mask_logits],
            {self.input_pixel_values: pixels, self.input_pixel_mask: pixel_mask})
        detections = self._postprocess(
            class_logits[0], mask_logits[0],
            (original_height, original_width, left, top,
             resized_size[0], resized_size[1]))
        rows = []
        for mask, score, class_id in detections:
            polygon = mask_to_polygon(mask)
            if polygon is not None:
                rows.append((polygon.tolist(), score, class_id, mask))
        page = self._tree_from_regions([polygon for polygon, _, _, _ in rows],
                                       [score for _, score, _, _ in rows],
                                       [class_id for _, _, class_id, _ in rows], source.shape[:2])
        if return_raw or return_masks or return_outputs:
            details = [{"class_id": class_id, "confidence": score, "polygon": polygon,
                        **self._mask_details(mask, include_mask=return_masks)}
                       for polygon, score, class_id, mask in rows]
            return self._raw_result(page, {self.output_class_logits: class_logits,
                                          self.output_mask_logits: mask_logits} if return_outputs else None,
                                    details, source.shape[:2],
                                    padding=[left, top], resized_size=resized_size)
        return page


__all__ = ["Mask2Former"]
