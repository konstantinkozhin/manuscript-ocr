import json
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
import onnxruntime as ort

from manuscript.api.detector import BaseDetector
from ...data import Block, Line, Page, TextSpan
from ...utils import read_image


class Mask2Former(BaseDetector):
    """Detect text with a Mask2Former ONNX model."""

    default_weights_name = "mask2former_line_v0_prev"
    registry_model_class = "Mask2Former"
    pretrained_registry: Dict[str, str] = {}
    config_registry: Dict[str, str] = {}

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
        self.config_path = self._resolve_model_config(config)
        with open(self.config_path, "r", encoding="utf-8") as stream:
            values = json.load(stream)
        if values.get("schema_version") != 1:
            raise ValueError("Unsupported Mask2Former config schema")
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

    def _resolve_model_config(self, config):
        if config is not None:
            return self._resolve_extra_artifact(
                str(config), default_name=None, registry=self.config_registry,
                description="model config")
        artifacts = getattr(self, "_resolved_model_artifacts", None)
        if artifacts and artifacts.get("config"):
            return str(artifacts["config"])
        candidate = Path(self.weights).with_suffix(".json")
        if candidate.exists():
            return str(candidate.resolve())
        raise FileNotFoundError(
            f"Mask2Former config not found next to {self.weights}; pass config explicitly")

    def _initialize_session(self):
        if self.onnx_session is not None:
            return
        self._prepare_runtime_dependencies()
        self.onnx_session = ort.InferenceSession(
            self.weights, providers=self.runtime_providers())
        self._log_device_info(self.onnx_session)

    @staticmethod
    def _softmax(values):
        values = values - values.max(axis=-1, keepdims=True)
        exp = np.exp(values)
        return exp / exp.sum(axis=-1, keepdims=True)

    @staticmethod
    def _sigmoid(values):
        values = np.clip(values, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-values))

    @staticmethod
    def _mask_iou(first, second):
        intersection = np.logical_and(first, second).sum()
        union = np.logical_or(first, second).sum()
        return float(intersection / union) if union else 0.0

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
        return pixels, left, top, resized_size

    def _postprocess(self, class_logits, mask_logits, geometry):
        original_height, original_width, left, top, resized_width, resized_height = geometry
        class_scores = self._softmax(class_logits.astype(np.float32))[:, :-1].max(axis=-1)
        candidates = []
        for query, class_score in enumerate(class_scores):
            logits = cv2.resize(mask_logits[query].astype(np.float32),
                                (self.mask_interpolation_size,) * 2,
                                interpolation=cv2.INTER_LINEAR)
            binary = logits > self.mask_logit_threshold
            if not binary.any():
                continue
            score = float(class_score * self._sigmoid(logits)[binary].mean())
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
                candidates.append((restored, score))
        candidates.sort(key=lambda item: item[1], reverse=True)
        kept = []
        for mask, score in candidates:
            if all(self._mask_iou(mask, old) <= self.nms_iou_threshold
                   for old, _ in kept):
                kept.append((mask, score))
        return kept

    @staticmethod
    def _mask_polygon(mask):
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        polygon = cv2.approxPolyDP(
            contour, 0.002 * cv2.arcLength(contour, True), True)[:, 0, :]
        if len(polygon) < 4:
            polygon = cv2.boxPoints(cv2.minAreaRect(contour))
        return [(float(x), float(y)) for x, y in polygon]

    def predict(self, image) -> Page:
        if self.onnx_session is None:
            self._initialize_session()
        source = read_image(image)
        original_height, original_width = source.shape[:2]
        pixels, left, top, resized_size = self._preprocess(source)
        input_type = self.onnx_session.get_inputs()[0].type
        pixels = pixels.astype(
            np.float16 if input_type == "tensor(float16)" else np.float32)
        pixel_mask = np.ones((1, self.image_size, self.image_size), dtype=np.int64)
        class_logits, mask_logits = self.onnx_session.run(
            [self.output_class_logits, self.output_mask_logits],
            {self.input_pixel_values: pixels, self.input_pixel_mask: pixel_mask})
        detections = self._postprocess(
            class_logits[0], mask_logits[0],
            (original_height, original_width, left, top,
             resized_size[0], resized_size[1]))
        rows = []
        for mask, score in detections:
            polygon = self._mask_polygon(mask)
            if polygon:
                rows.append((polygon, score))
        lines = [
            Line(text_spans=[TextSpan(polygon=polygon,
                                      detection_confidence=score)])
            for polygon, score in rows
        ]
        return Page(blocks=[Block(lines=lines)])


__all__ = ["Mask2Former"]
