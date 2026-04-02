from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from shapely.geometry import Polygon

from manuscript.api.detector import BaseDetector

from ...data import Block, Line, Page, TextSpan
from ...utils import order_quad_points, read_image


class YOLO(BaseDetector):
    """
    Initialize YOLO text detector with ONNX Runtime.

    Parameters
    ----------
    weights : str or Path, optional
        Path or identifier for ONNX model weights. Supports:

        - Local file path: ``"path/to/model.onnx"``
        - HTTP/HTTPS URL: ``"https://example.com/model.onnx"``
        - GitHub release: ``"github://owner/repo/tag/file.onnx"``
        - Google Drive: ``"gdrive:FILE_ID"``
        - Preset name: ``"yolo26s_obb_text"`` or ``"yolo26x_obb_text"``
        - ``None``: auto-downloads default preset (``yolo26s_obb_text``)

        The ONNX model may return either standard detections in
        ``xyxy, score, class_id`` format with shape ``[N, 6]`` / ``[1, N, 6]``
        or oriented detections in ``cx, cy, w, h, score, class_id, angle``
        format with shape ``[N, 7]`` / ``[1, N, 7]``.
    device : str, optional
        Compute device: ``"cuda"``, ``"coreml"``, or ``"cpu"``. If None,
        automatically selects CPU. For GPU/CoreML acceleration:

        - CUDA (NVIDIA): ``pip install onnxruntime-gpu``
        - CoreML (Apple Silicon M1/M2/M3): ``pip install onnxruntime-silicon``

        Default is ``None`` (CPU).
    score_thresh : float, optional
        Confidence threshold applied to model outputs after ONNX inference and
        before the additional containment cleanup pass. Default is ``0.1``.
    class_ids : sequence of int or None, optional
        Optional whitelist of class IDs to keep. If ``None``, all classes are
        kept. Default is ``None``.
    target_size : int or None, optional
        Square inference size used for letterbox preprocessing. Images are
        resized into ``(target_size, target_size)`` before ONNX inference.
        If ``None``, preset defaults are used: ``1280`` for
        ``"yolo26s_obb_text"``, ``1024`` for ``"yolo26x_obb_text"``.
        Unknown/custom weights fall back to ``1280``.
    axis_aligned_output : bool, optional
        If ``True`` (default), OBB detections are converted to standard
        axis-aligned rectangles. If ``False``, OBB detections are returned as
        rotated polygons via ``page`` / ``polygons`` and as
        ``cx, cy, w, h, score, class_id, angle`` rows in ``boxes``.
        For non-OBB models this flag has no effect.
    containment_threshold : float or None, optional
        Removes a smaller box when at least this fraction of its area is
        covered by a larger box. For example, ``0.9`` removes boxes that are
        contained by ``90%`` or more. Set to ``None`` to disable this extra
        cleanup. Default is ``0.9``.

    Notes
    -----
    The class provides one main public method:

    - ``predict`` - run inference on a single image and return detections.

    Available presets:

    - ``"yolo26s_obb_text"`` - YOLO26-S OBB text detector
    - ``"yolo26x_obb_text"`` - YOLO26-X OBB text detector
    """

    default_weights_name = "yolo26s_obb_text"
    default_target_size = 1280
    default_target_size_registry: Dict[str, int] = {
        "yolo26s_obb_text": 1280,
        "yolo26x_obb_text": 1024,
    }
    pretrained_registry: Dict[str, str] = {
        "yolo26s_obb_text": "https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26s_obb_text.raw.onnx",
        "yolo26x_obb_text": "https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26x_obb_text.raw.onnx",
    }

    def __init__(
        self,
        weights: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        *,
        score_thresh: float = 0.1,
        class_ids: Optional[Sequence[int]] = None,
        target_size: Optional[int] = None,
        axis_aligned_output: bool = True,
        containment_threshold: Optional[float] = 0.9,
        **kwargs,
    ):
        super().__init__(weights=weights, device=device, **kwargs)

        self.onnx_session = None
        self.score_thresh = float(score_thresh)
        self.class_ids = None if class_ids is None else {int(v) for v in class_ids}
        self.target_size = int(
            self._resolve_default_target_size(weights)
            if target_size is None
            else target_size
        )
        self.axis_aligned_output = bool(axis_aligned_output)
        self.containment_threshold = (
            None
            if containment_threshold is None
            else float(containment_threshold)
        )
        self._output_layout = "detect"

    @classmethod
    def _candidate_weight_names(cls, value: Optional[Union[str, Path]]) -> List[str]:
        if value is None:
            return []

        raw = str(value).strip()
        if not raw:
            return []

        tail = raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        candidates = [raw, tail]
        for item in list(candidates):
            without_onnx = item.removesuffix(".onnx")
            candidates.append(without_onnx)
            candidates.append(without_onnx.removesuffix(".raw"))

        seen = set()
        unique: List[str] = []
        for item in candidates:
            if item and item not in seen:
                unique.append(item)
                seen.add(item)
        return unique

    def _resolve_default_target_size(self, weights: Optional[Union[str, Path]]) -> int:
        candidates: List[str] = []
        if weights is None and self.default_weights_name:
            candidates.append(self.default_weights_name)

        for value in (weights, self.weights):
            candidates.extend(self._candidate_weight_names(value))

        for candidate in candidates:
            if candidate in self.default_target_size_registry:
                return self.default_target_size_registry[candidate]

        return self.default_target_size

    def _initialize_session(self):
        if self.onnx_session is not None:
            return

        self.onnx_session = ort.InferenceSession(
            self.weights,
            providers=self.runtime_providers(),
        )
        self._log_device_info(self.onnx_session)

        input_info = self.onnx_session.get_inputs()[0]
        output_info = self.onnx_session.get_outputs()[0]

        if len(input_info.shape) != 4:
            raise ValueError(
                f"YOLO expects a 4D ONNX input tensor, got: {input_info.shape}"
            )

        input_h = input_info.shape[-2]
        input_w = input_info.shape[-1]
        if isinstance(input_h, int) and isinstance(input_w, int):
            if int(input_h) != self.target_size or int(input_w) != self.target_size:
                raise ValueError(
                    f"YOLO ONNX expects input size {(int(input_h), int(input_w))}, "
                    f"but detector is configured for {(self.target_size, self.target_size)}."
                )

        if len(output_info.shape) < 2:
            raise ValueError(
                f"YOLO expects output rank >= 2, got: {output_info.shape}"
            )

        output_last_dim = output_info.shape[-1]
        self._output_layout = self._resolve_output_layout(output_last_dim)
        expected_dim = 7 if self._output_layout == "obb" else 6
        if isinstance(output_last_dim, int) and output_last_dim < expected_dim:
            raise ValueError(
                f"YOLO expects output rows with at least {expected_dim} values, got: {output_info.shape}"
            )

    def _resolve_output_layout(self, output_last_dim: Any) -> str:
        task = ""
        try:
            metadata = self.onnx_session.get_modelmeta().custom_metadata_map
            task = str(metadata.get("task", "")).strip().lower()
        except Exception:
            task = ""

        if task == "obb" or output_last_dim == 7:
            return "obb"
        return "detect"

    def _letterbox(
        self,
        image: np.ndarray,
        new_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        shape = image.shape[:2]
        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (
            int(round(shape[1] * ratio)),
            int(round(shape[0] * ratio)),
        )
        dw = (new_shape[1] - new_unpad[0]) / 2.0
        dh = (new_shape[0] - new_unpad[1]) / 2.0

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return image, float(ratio), (float(dw), float(dh))

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        letterboxed, ratio, pad = self._letterbox(
            image,
            (int(self.target_size), int(self.target_size)),
        )
        chw = letterboxed.transpose(2, 0, 1)
        tensor = np.ascontiguousarray(chw, dtype=np.float32) / 255.0
        return tensor[None], ratio, pad

    @staticmethod
    def _scale_boxes(
        boxes: np.ndarray,
        image_hw: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float],
    ) -> np.ndarray:
        if len(boxes) == 0:
            return boxes

        scaled = boxes.copy()
        dw, dh = pad
        scaled[:, [0, 2]] -= dw
        scaled[:, [1, 3]] -= dh
        scaled[:, :4] /= max(float(ratio), 1e-12)

        image_h, image_w = image_hw
        scaled[:, 0] = np.clip(scaled[:, 0], 0, image_w)
        scaled[:, 1] = np.clip(scaled[:, 1], 0, image_h)
        scaled[:, 2] = np.clip(scaled[:, 2], 0, image_w)
        scaled[:, 3] = np.clip(scaled[:, 3], 0, image_h)
        return scaled

    @staticmethod
    def _scale_obb_boxes(
        boxes: np.ndarray,
        ratio: float,
        pad: Tuple[float, float],
    ) -> np.ndarray:
        if len(boxes) == 0:
            return boxes

        scaled = boxes.copy()
        dw, dh = pad
        gain = max(float(ratio), 1e-12)
        scaled[:, 0] = (scaled[:, 0] - dw) / gain
        scaled[:, 1] = (scaled[:, 1] - dh) / gain
        scaled[:, 2] /= gain
        scaled[:, 3] /= gain
        return scaled

    @staticmethod
    def _obb_rows_to_polygons(rows: np.ndarray) -> np.ndarray:
        if len(rows) == 0:
            return np.zeros((0, 4, 2), dtype=np.float32)

        ctr = rows[:, :2]
        w = rows[:, 2:3]
        h = rows[:, 3:4]
        angle = rows[:, 6:7]
        cos_value = np.cos(angle)
        sin_value = np.sin(angle)
        vec1 = np.concatenate([w / 2.0 * cos_value, w / 2.0 * sin_value], axis=1)
        vec2 = np.concatenate([-h / 2.0 * sin_value, h / 2.0 * cos_value], axis=1)

        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        polygons = np.stack([pt1, pt2, pt3, pt4], axis=1).astype(np.float32)
        return np.stack(
            [order_quad_points(polygon) for polygon in polygons],
            axis=0,
        ).astype(np.float32)

    @staticmethod
    def _axis_aligned_boxes_to_polygons(boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, 4, 2), dtype=np.float32)

        polygons = np.zeros((len(boxes), 4, 2), dtype=np.float32)
        polygons[:, 0, 0] = boxes[:, 0]
        polygons[:, 0, 1] = boxes[:, 1]
        polygons[:, 1, 0] = boxes[:, 2]
        polygons[:, 1, 1] = boxes[:, 1]
        polygons[:, 2, 0] = boxes[:, 2]
        polygons[:, 2, 1] = boxes[:, 3]
        polygons[:, 3, 0] = boxes[:, 0]
        polygons[:, 3, 1] = boxes[:, 3]
        return polygons

    @staticmethod
    def _clip_polygons(
        polygons: np.ndarray,
        image_hw: Tuple[int, int],
    ) -> np.ndarray:
        if len(polygons) == 0:
            return polygons

        clipped = polygons.copy()
        image_h, image_w = image_hw
        clipped[:, :, 0] = np.clip(clipped[:, :, 0], 0, image_w)
        clipped[:, :, 1] = np.clip(clipped[:, :, 1], 0, image_h)
        return clipped

    @staticmethod
    def _polygons_to_axis_aligned_rows(
        polygons: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> np.ndarray:
        if len(polygons) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        mins = polygons.min(axis=1)
        maxs = polygons.max(axis=1)
        return np.column_stack(
            [mins[:, 0], mins[:, 1], maxs[:, 0], maxs[:, 1], scores, class_ids]
        ).astype(np.float32)

    @staticmethod
    def _polygon_areas(polygons: np.ndarray) -> np.ndarray:
        if len(polygons) == 0:
            return np.zeros((0,), dtype=np.float32)

        shifted = np.roll(polygons, shift=-1, axis=1)
        cross = polygons[:, :, 0] * shifted[:, :, 1] - shifted[:, :, 0] * polygons[:, :, 1]
        return 0.5 * np.abs(cross.sum(axis=1)).astype(np.float32)

    def _postprocess(
        self,
        output: np.ndarray,
        image_hw: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(output, dtype=np.float32)

        if rows.ndim == 3 and rows.shape[0] == 1:
            rows = rows[0]

        expected_dim = 7 if self._output_layout == "obb" else 6
        if rows.ndim != 2 or rows.shape[1] < expected_dim:
            raise ValueError(
                f"Unexpected YOLO ONNX output shape: {rows.shape}. "
                f"Expected [N, {expected_dim}] or [1, N, {expected_dim}]."
            )

        rows = rows[:, :expected_dim]
        rows = rows[np.isfinite(rows).all(axis=1)]
        rows = rows[rows[:, 4] >= self.score_thresh]

        if self._output_layout == "obb":
            rows = rows[rows[:, 2] > 0]
            rows = rows[rows[:, 3] > 0]
        else:
            rows = rows[rows[:, 2] > rows[:, 0]]
            rows = rows[rows[:, 3] > rows[:, 1]]

        if self.class_ids is not None:
            class_ids = rows[:, 5].astype(np.int64)
            rows = rows[np.isin(class_ids, list(self.class_ids))]

        if len(rows) == 0:
            return (
                np.zeros((0, expected_dim), dtype=np.float32),
                np.zeros((0, 4, 2), dtype=np.float32),
            )

        rows = rows[np.argsort(rows[:, 4])[::-1]]

        if self._output_layout == "obb":
            rows = self._scale_obb_boxes(rows, ratio=ratio, pad=pad)
            polygons = self._clip_polygons(
                self._obb_rows_to_polygons(rows),
                image_hw=image_hw,
            )

            valid = self._polygon_areas(polygons) > 1e-6
            rows = rows[valid]
            polygons = polygons[valid]
            rows, polygons = self._suppress_contained_polygons(rows, polygons)
            rows[:, 4] = np.clip(rows[:, 4], 0.0, 1.0)

            if self.axis_aligned_output:
                boxes = self._polygons_to_axis_aligned_rows(
                    polygons,
                    scores=rows[:, 4],
                    class_ids=rows[:, 5],
                )
                boxes[:, 4] = np.clip(boxes[:, 4], 0.0, 1.0)
                return boxes, self._axis_aligned_boxes_to_polygons(boxes)

            return rows, polygons

        rows = self._scale_boxes(rows, image_hw=image_hw, ratio=ratio, pad=pad)
        rows = self._suppress_contained_boxes(rows)
        rows[:, 4] = np.clip(rows[:, 4], 0.0, 1.0)
        return rows, self._axis_aligned_boxes_to_polygons(rows)

    @staticmethod
    def _intersection_area(box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _suppress_contained_boxes(self, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) <= 1 or self.containment_threshold is None:
            return boxes

        threshold = float(self.containment_threshold)
        if threshold <= 0.0:
            return boxes

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        areas = np.maximum(areas.astype(np.float32), 1e-12)
        area_order = np.argsort(areas)[::-1]
        keep = np.ones(len(boxes), dtype=bool)

        for outer_pos, outer_idx in enumerate(area_order):
            if not keep[outer_idx]:
                continue

            outer_box = boxes[outer_idx]
            outer_area = areas[outer_idx]

            for inner_idx in area_order[outer_pos + 1 :]:
                if not keep[inner_idx]:
                    continue

                inner_area = areas[inner_idx]
                if outer_area <= inner_area:
                    continue

                overlap = self._intersection_area(outer_box, boxes[inner_idx])
                if overlap / inner_area >= threshold:
                    keep[inner_idx] = False

        return boxes[keep]

    def _suppress_contained_polygons(
        self,
        rows: np.ndarray,
        polygons: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(rows) <= 1 or self.containment_threshold is None:
            return rows, polygons

        threshold = float(self.containment_threshold)
        if threshold <= 0.0:
            return rows, polygons

        geometries = [Polygon(poly) for poly in polygons]
        areas = np.asarray(
            [max(float(geom.area), 1e-12) for geom in geometries],
            dtype=np.float32,
        )
        area_order = np.argsort(areas)[::-1]
        keep = np.ones(len(rows), dtype=bool)

        for outer_pos, outer_idx in enumerate(area_order):
            if not keep[outer_idx]:
                continue

            outer_geom = geometries[outer_idx]
            outer_area = areas[outer_idx]
            if outer_area <= 0.0:
                continue

            for inner_idx in area_order[outer_pos + 1 :]:
                if not keep[inner_idx]:
                    continue

                inner_area = areas[inner_idx]
                if outer_area <= inner_area:
                    continue

                overlap = float(outer_geom.intersection(geometries[inner_idx]).area)
                if overlap / inner_area >= threshold:
                    keep[inner_idx] = False

        return rows[keep], polygons[keep]

    def _polygons_to_page(
        self,
        polygons: np.ndarray,
        scores: np.ndarray,
    ) -> Page:
        text_spans: List[TextSpan] = []

        for idx, (polygon, score) in enumerate(zip(polygons, scores)):
            text_spans.append(
                TextSpan(
                    polygon=[(float(x), float(y)) for x, y in polygon],
                    detection_confidence=float(score),
                    order=idx,
                )
            )

        return Page(
            blocks=[Block(lines=[Line(text_spans=text_spans, order=0)], order=0)]
        )

    def _boxes_to_rows(
        self,
        boxes: np.ndarray,
        polygons: np.ndarray,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        if boxes.shape[1] == 7 and not self.axis_aligned_output:
            for (cx, cy, w, h, score, class_id, angle), polygon in zip(boxes, polygons):
                x_min, y_min = polygon.min(axis=0)
                x_max, y_max = polygon.max(axis=0)
                rows.append(
                    {
                        "class_id": int(class_id),
                        "confidence": round(float(score), 4),
                        "cx": round(float(cx), 2),
                        "cy": round(float(cy), 2),
                        "w": round(float(w), 2),
                        "h": round(float(h), 2),
                        "angle": round(float(angle), 6),
                        "x1": round(float(x_min), 2),
                        "y1": round(float(y_min), 2),
                        "x2": round(float(x_max), 2),
                        "y2": round(float(y_max), 2),
                        "polygon": [
                            (round(float(x), 2), round(float(y), 2))
                            for x, y in polygon
                        ],
                    }
                )
            return rows

        for x1, y1, x2, y2, score, class_id in boxes:
            rows.append(
                {
                    "class_id": int(class_id),
                    "confidence": round(float(score), 4),
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2),
                }
            )
        return rows

    def predict(
        self,
        img_or_path: Union[str, Path, np.ndarray],
    ) -> Page:
        """
        Run YOLO ONNX inference on a single image and return detected page structure.

        Parameters
        ----------
        img_or_path : str or pathlib.Path or numpy.ndarray
            Path to an image file or an RGB image provided as a NumPy array
            with shape ``(H, W, 3)`` in ``uint8`` format.

        Returns
        -------
        Page
            Parsed detection result as a Page object containing a single
            Block with a single Line of TextSpan objects.

        Examples
        --------
        Run inference and get structured output:

        >>> from manuscript.detectors import YOLO
        >>> model = YOLO(weights="yolo26s_obb_text")
        >>> page = model.predict("page.jpg")
        >>> first_text_span = page.blocks[0].lines[0].text_spans[0]
        >>> print(first_text_span.detection_confidence)
        """
        if self.onnx_session is None:
            self._initialize_session()

        image = read_image(img_or_path)
        tensor, ratio, pad = self._preprocess(image)
        input_name = self.onnx_session.get_inputs()[0].name
        output = self.onnx_session.run(None, {input_name: tensor})[0]
        boxes, polygons = self._postprocess(
            output,
            image_hw=image.shape[:2],
            ratio=ratio,
            pad=pad,
        )

        return self._polygons_to_page(polygons, boxes[:, 4])
