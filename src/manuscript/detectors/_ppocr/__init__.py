from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import onnxruntime as ort
from shapely.geometry import Polygon

from manuscript.api.detector import BaseDetector
from manuscript.data import TextSpan
from manuscript.utils import read_image


class PPOCR(BaseDetector):
    """ONNX inference for PP-OCR detectors with DB probability-map outputs."""

    registry_model_class = "PPOCR"

    def __init__(
        self, weights: Union[str, Path], config=None, device: Optional[str] = None,
        threshold=None, box_threshold=None, unclip_ratio=None,
        limit_side_len=None, limit_type=None, force_download=False,
    ):
        if weights is None:
            raise ValueError("PPOCR requires explicit ONNX weights")
        super().__init__(weights=str(weights), device=device, force_download=force_download)
        config, self.config_path = self._load_detector_config(config)
        if config.get("schema_version") != 1:
            raise ValueError("Unsupported PPOCR config schema_version")
        if config.get("task") != "text_detection" or config.get("algorithm") != "DB":
            raise ValueError("PPOCR supports DB text-detection configurations")
        required = {
            "preprocess": (
                "color_order", "scale", "mean", "std", "limit_side_len",
                "limit_type", "max_side_limit",
            ),
            "postprocess": (
                "threshold", "box_threshold", "unclip_ratio", "max_candidates",
                "min_size", "unclip_join", "output_is_logits", "output_channel",
            ),
        }
        for section, keys in required.items():
            values = config.get(section)
            if not isinstance(values, dict):
                raise ValueError(f"PPOCR config requires a {section} JSON object")
            missing = [key for key in keys if key not in values]
            if missing:
                raise ValueError(f"PPOCR config missing {section} fields: {', '.join(missing)}")
        self.preprocess = dict(config["preprocess"])
        self.postprocess = dict(config["postprocess"])
        self.postprocess.pop("box_type", None)
        for key, value in [("limit_side_len", limit_side_len), ("limit_type", limit_type)]:
            if value is not None:
                self.preprocess[key] = value
        for key, value in [("threshold", threshold), ("box_threshold", box_threshold),
                           ("unclip_ratio", unclip_ratio)]:
            if value is not None:
                self.postprocess[key] = value
        self._input_name = config.get("input_name")
        self._output_name = config.get("output_name")
        self._input_dtype = np.float32
        self._fixed_hw = None
        self._validate_config()

    def _validate_config(self):
        pre, post = self.preprocess, self.postprocess
        if pre["color_order"] not in ("RGB", "BGR"):
            raise ValueError("color_order must be RGB or BGR")
        if pre["limit_type"] not in ("min", "max", "resize_long"):
            raise ValueError("limit_type must be min, max or resize_long")
        for name in ["limit_side_len", "max_side_limit"]:
            if isinstance(pre[name], bool) or int(pre[name]) != pre[name] or int(pre[name]) < 32:
                raise ValueError(name + " must be an integer of at least 32")
            pre[name] = int(pre[name])
        for name in ["mean", "std"]:
            values = np.asarray(pre[name], dtype=np.float32)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(name + " must contain three finite values")
        if any(float(value) <= 0 for value in pre["std"]):
            raise ValueError("std values must be positive")
        if not np.isfinite(pre["scale"]) or pre["scale"] <= 0:
            raise ValueError("scale must be positive and finite")
        for name in ["threshold", "box_threshold"]:
            if not np.isfinite(post[name]) or not 0 <= post[name] <= 1:
                raise ValueError(name + " must be between 0 and 1")
        if not np.isfinite(post["unclip_ratio"]) or post["unclip_ratio"] < 0:
            raise ValueError("unclip_ratio must be finite and nonnegative")
        for name in ["max_candidates", "min_size", "output_channel"]:
            if isinstance(post[name], bool) or int(post[name]) != post[name] or post[name] < (0 if name == "output_channel" else 1):
                raise ValueError(name + " must be a valid positive integer (channel may be zero)")
            post[name] = int(post[name])
        if post["unclip_join"] not in ("round", "miter"):
            raise ValueError("unclip_join must be round or miter")
        if not isinstance(post["output_is_logits"], bool):
            raise ValueError("output_is_logits must be boolean")

    def _initialize_session(self):
        if self.session is not None:
            return
        session = self._create_onnx_session()
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("PPOCR requires a graph with one NCHW image input")
        item = inputs[0]
        if len(item.shape) != 4 or (isinstance(item.shape[1], int) and item.shape[1] != 3):
            raise ValueError("PPOCR input must have shape [batch, 3, height, width]")
        if isinstance(item.shape[0], int) and item.shape[0] != 1:
            raise ValueError("PPOCR single-image inference requires batch 1 or dynamic batch")
        types = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
        if item.type not in types:
            raise ValueError("PPOCR input must be float32 or float16")
        if self._input_name and self._input_name != item.name:
            raise ValueError("Configured input_name is absent from ONNX")
        outputs = session.get_outputs()
        output_name = self._output_name or outputs[0].name
        if output_name not in {output.name for output in outputs}:
            raise ValueError("Configured output_name is absent from ONNX")
        self._input_name, self._output_name = item.name, output_name
        self._input_dtype = types[item.type]
        self._fixed_hw = tuple(d if isinstance(d, int) and d > 0 else None for d in item.shape[2:])
        self.session = session

    def _resize_shape(self, height, width):
        pre = self.preprocess
        limit = pre["limit_side_len"]
        if pre["limit_type"] == "min":
            ratio = max(1.0, limit / min(height, width))
        elif pre["limit_type"] == "max":
            ratio = min(1.0, limit / max(height, width))
        else:
            ratio = limit / max(height, width)
        new_h, new_w = int(height * ratio), int(width * ratio)
        if max(new_h, new_w) > pre["max_side_limit"]:
            ratio = pre["max_side_limit"] / max(new_h, new_w)
            new_h, new_w = int(new_h * ratio), int(new_w * ratio)
        new_h, new_w = max(32, round(new_h / 32) * 32), max(32, round(new_w / 32) * 32)
        if self._fixed_hw:
            new_h = self._fixed_hw[0] or new_h
            new_w = self._fixed_hw[1] or new_w
        return new_h, new_w

    def _preprocess(self, image):
        height, width = self._resize_shape(*image.shape[:2])
        values = cv2.resize(image.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        if self.preprocess["color_order"] == "BGR":
            values = values[:, :, ::-1]
        mean = np.asarray(self.preprocess["mean"], dtype=np.float32)
        std = np.asarray(self.preprocess["std"], dtype=np.float32)
        values = (values * self.preprocess["scale"] - mean) / std
        return np.ascontiguousarray(values.transpose(2, 0, 1)[None], dtype=self._input_dtype)

    @staticmethod
    def _order_quad(points):
        # Angle sorting preserves four distinct vertices even at exactly 45°,
        # where independent sum/difference extrema can pick the same point.
        points = np.asarray(points, dtype=np.float32)
        centered = points - points.mean(axis=0)
        points = points[np.argsort(np.arctan2(centered[:, 1], centered[:, 0]))]
        first = np.lexsort((points[:, 0], points[:, 1], points.sum(axis=1)))[0]
        return np.roll(points, -int(first), axis=0)

    @staticmethod
    def _score(prediction, polygon):
        height, width = prediction.shape
        low = np.floor(polygon.min(axis=0)).astype(int)
        high = np.ceil(polygon.max(axis=0)).astype(int)
        x0, y0 = max(0, low[0]), max(0, low[1])
        x1, y1 = min(width - 1, high[0]), min(height - 1, high[1])
        if x1 < x0 or y1 < y0:
            return 0.0
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        points = (polygon - [x0, y0]).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
        return float(cv2.mean(prediction[y0:y1 + 1, x0:x1 + 1], mask)[0])

    def _postprocess(self, output, image_hw, return_raw=False, *, return_masks=False, return_outputs=False):
        values = np.asarray(output, dtype=np.float32)
        channel = self.postprocess["output_channel"]
        if values.ndim == 4 and values.shape[0] == 1 and channel < values.shape[1]:
            values = values[0, channel]
        elif values.ndim == 3 and values.shape[0] == 1 and channel == 0:
            values = values[0]
        elif values.ndim != 2 or channel != 0:
            raise ValueError("PPOCR output must be a single-image probability map [1,C,H,W], [1,H,W] or [H,W]")
        if not values.size or not np.isfinite(values).all():
            raise ValueError("PPOCR output contains empty or nonfinite probabilities")
        if self.postprocess["output_is_logits"]:
            values = 1.0 / (1.0 + np.exp(-np.clip(values, -80, 80)))
        elif values.min() < -1e-4 or values.max() > 1.0001:
            raise ValueError("PPOCR output is not probabilities; set output_is_logits if appropriate")
        values = np.clip(values, 0, 1)
        mask = (values > self.postprocess["threshold"]).astype(np.uint8) * 255
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        spans, detections = [], []
        map_h, map_w = values.shape
        image_h, image_w = image_hw
        for contour in contours[:self.postprocess["max_candidates"]]:
            rect = cv2.minAreaRect(contour)
            if min(rect[1]) < self.postprocess["min_size"]:
                continue
            polygon = contour.reshape(-1, 2)
            if len(polygon) < 4:
                continue
            score = self._score(values, polygon)
            if score < self.postprocess["box_threshold"]:
                continue
            geometry = Polygon(polygon)
            if not geometry.is_valid or geometry.area <= 0 or geometry.length <= 0:
                continue
            distance = geometry.area * self.postprocess["unclip_ratio"] / geometry.length
            expanded = geometry.buffer(distance, join_style=1 if self.postprocess["unclip_join"] == "round" else 2)
            if expanded.is_empty or expanded.geom_type != "Polygon":
                continue
            points = np.asarray(expanded.exterior.coords[:-1], np.float32)
            expanded_rect = cv2.minAreaRect(points)
            if min(expanded_rect[1]) < self.postprocess["min_size"] + 2:
                continue
            points[:, 0] = np.clip(np.rint(points[:, 0] * image_w / map_w), 0, image_w)
            points[:, 1] = np.clip(np.rint(points[:, 1] * image_h / map_h), 0, image_h)
            final_polygon = Polygon(points)
            if len(np.unique(points, axis=0)) < 4 or not final_polygon.is_valid or final_polygon.area <= 0:
                continue
            # Public geometry convention is clockwise in image coordinates.
            if not final_polygon.exterior.is_ccw:
                points = points[::-1]
            detections.append({"polygon": points.astype(float).tolist(), "confidence": score,
                               "contour": (polygon * [image_w / map_w, image_h / map_h]).tolist()})
            if len(points) == 4:
                points = self._order_quad(points)
            spans.append(TextSpan(polygon=[(float(x), float(y)) for x, y in points],
                                  detection_confidence=float(np.clip(score, 0, 1))))
        page = self._page_from_regions([span.polygon for span in spans],
                                       [span.detection_confidence for span in spans])
        if return_raw or return_masks or return_outputs:
            result = self._raw_result(page, {self._output_name or "probability_map": output} if return_outputs else None, detections, image_hw)
            if return_masks:
                result["masks"] = {"text": cv2.resize(mask, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)}
            return result
        return page

    def predict(self, img_or_path, return_raw=False, *, return_masks=False, return_outputs=False):
        """Detect text in an image/path and return Page with one TextSpan per region."""
        if self.session is None:
            self._initialize_session()
        image = read_image(img_or_path)
        tensor = self._preprocess(image)
        output = self.session.run([self._output_name], {self._input_name: tensor})[0]
        return self._postprocess(output, image.shape[:2], return_raw=return_raw,
                                 return_masks=return_masks, return_outputs=return_outputs)

    @staticmethod
    def export(model_dir, output_dir, model_id=None, revision=None):
        """Export a compatible source model to an ONNX bundle with configuration.

        Export dependencies are imported only when this method is called.
        """
        from .export import export

        return export(model_dir, output_dir, model_id=model_id, revision=revision)
