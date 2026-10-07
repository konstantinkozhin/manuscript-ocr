import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from shapely.geometry import Polygon

from manuscript.api.detector import BaseDetector

from ...data import Block, Line, Page, TextSpan
from ...utils import order_quad_points, read_image
from ...utils.geometry import (
    boxes_to_polygons, clip_polygons, polygon_areas,
    box_intersection_area, polygons_to_boxes,
)


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
        - Preset name: ``"yolo26s_obb_text_g1"`` or ``"yolo26x_obb_text_g1"``
        - ``None``: auto-downloads default preset (``yolo26x_obb_text_g1``)

        The ONNX model may return either standard detections in
        ``xyxy, score, class_id`` format with shape ``[N, 6]`` / ``[1, N, 6]``
        or oriented detections in ``cx, cy, w, h, score, class_id, angle``
        format with shape ``[N, 7]`` / ``[1, N, 7]``.
    config : str or Path, optional
        Path or identifier for model configuration YAML. Same URL schemes
        as ``weights``. If ``None``, attempts to infer a YAML file next to
        the weights or uses the config artifact from the model registry.
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
        If ``None``, the detector tries to read ``imgsz`` from a YAML config
        located next to the weights or downloaded from the preset registry.
        Unknown/custom weights without YAML fall back to ``1280``.
    axis_aligned_output : bool, optional, deprecated
        Deprecated. If explicitly ``True``, OBB detections are converted to standard
        axis-aligned rectangles. By default, OBB detections are returned as
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

    - ``"yolo26s_obb_text_g1"`` - YOLO26-S OBB text detector
    - ``"yolo26x_obb_text_g1"`` - YOLO26-X OBB text detector
    """

    default_weights_name = "yolo26x_obb_text_g1"
    registry_model_class = "YOLO"
    default_target_size = 1280
    def __init__(
        self,
        weights: Optional[Union[str, Path]] = None,
        config: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        force_download: bool = False,
        *,
        score_thresh: float = 0.1,
        class_ids: Optional[Sequence[int]] = None,
        target_size: Optional[int] = None,
        axis_aligned_output: Optional[bool] = None,
        containment_threshold: Optional[float] = 0.9,
        **kwargs,
    ):
        super().__init__(
            weights=weights,
            device=device,
            force_download=force_download,
            **kwargs,
        )

        self.onnx_session = None
        self.score_thresh = float(score_thresh)
        self.class_ids = None if class_ids is None else {int(v) for v in class_ids}
        self.config_path = self._resolve_model_config(weights=weights, config=config)
        self.target_size = int(
            self._resolve_default_target_size()
            if target_size is None
            else target_size
        )
        if axis_aligned_output is not None:
            warnings.warn("axis_aligned_output is deprecated; configure recognition crops instead",
                          DeprecationWarning, stacklevel=2)
        self.axis_aligned_output = bool(axis_aligned_output)
        self.containment_threshold = (
            None
            if containment_threshold is None
            else float(containment_threshold)
        )
        self._output_layout = "detect"

    def _resolve_default_target_size(self) -> int:
        if self.config_path is not None:
            return self._load_target_size_from_config(self.config_path)
        return self.default_target_size

    def _resolve_model_config(
        self,
        weights: Optional[Union[str, Path]],
        config: Optional[Union[str, Path]] = None,
    ) -> Optional[str]:
        _, path = self._load_detector_config(config, suffixes=(".yaml", ".yml"), required=False)
        return path

    @staticmethod
    def _parse_imgsz(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("imgsz must be numeric, got bool")

        if isinstance(value, (int, float)):
            return int(value)

        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
            raise ValueError(f"imgsz must be an integer or square pair, got: {value!r}")

        if isinstance(value, (list, tuple)):
            if any(isinstance(v, bool) for v in value):
                raise ValueError("imgsz must be numeric, got bool")
            values = [int(v) for v in value]
            if len(values) == 1:
                return values[0]
            if len(values) == 2 and values[0] == values[1]:
                return values[0]
            raise ValueError(
                f"imgsz must be a single integer or square pair, got: {value!r}"
            )

        raise ValueError(f"imgsz must be an integer or square pair, got: {value!r}")

    def _load_target_size_from_config(self, config_path: Union[str, Path]) -> int:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        if not isinstance(config, dict):
            raise ValueError(
                f"YOLO config must be a mapping, got: {type(config).__name__}"
            )

        if "imgsz" not in config:
            raise ValueError(f"YOLO config does not define imgsz: {config_path}")

        return self._parse_imgsz(config["imgsz"])

    def _initialize_session(self):
        if self.onnx_session is not None:
            return

        session = self._create_onnx_session()
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or not outputs:
            raise ValueError("YOLO requires one image input and at least one output")
        input_info, output_info = inputs[0], outputs[0]
        if input_info.type not in ("tensor(float)", "tensor(float16)"):
            raise ValueError("YOLO image input must be float32 or float16")

        if len(input_info.shape) != 4:
            raise ValueError(
                f"YOLO expects a 4D ONNX input tensor, got: {input_info.shape}"
            )

        if isinstance(input_info.shape[0], int) and input_info.shape[0] != 1:
            raise ValueError("YOLO single-image inference requires batch 1 or dynamic batch")
        if isinstance(input_info.shape[1], int) and input_info.shape[1] != 3:
            raise ValueError("YOLO expects three image channels")
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
        self._output_layout = self._resolve_output_layout(output_last_dim, session=session)
        expected_dim = 7 if self._output_layout == "obb" else 6
        if isinstance(output_last_dim, int) and output_last_dim < expected_dim:
            raise ValueError(
                f"YOLO expects output rows with at least {expected_dim} values, got: {output_info.shape}"
            )

        self._input_dtype = np.float16 if input_info.type == "tensor(float16)" else np.float32
        self.onnx_session = session

    def _resolve_output_layout(self, output_last_dim: Any, session=None) -> str:
        task = ""
        try:
            metadata = (session or self.onnx_session).get_modelmeta().custom_metadata_map
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
        return tensor[None].astype(getattr(self, "_input_dtype", np.float32), copy=False), ratio, pad

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
    def _polygons_to_axis_aligned_rows(
        polygons: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> np.ndarray:
        if len(polygons) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        return np.column_stack([polygons_to_boxes(polygons), scores, class_ids]).astype(np.float32)


    def _postprocess(
        self,
        output: np.ndarray,
        image_hw: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float],
        preserve_geometry: bool = False,
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
            polygons = clip_polygons(
                self._obb_rows_to_polygons(rows),
                image_hw=image_hw,
            )

            valid = polygon_areas(polygons) > 1e-6
            rows = rows[valid]
            polygons = polygons[valid]
            rows, polygons = self._suppress_contained_polygons(rows, polygons)
            rows[:, 4] = np.clip(rows[:, 4], 0.0, 1.0)

            if self.axis_aligned_output and not preserve_geometry:
                boxes = self._polygons_to_axis_aligned_rows(
                    polygons,
                    scores=rows[:, 4],
                    class_ids=rows[:, 5],
                )
                boxes[:, 4] = np.clip(boxes[:, 4], 0.0, 1.0)
                return boxes, boxes_to_polygons(boxes)

            return rows, polygons

        rows = self._scale_boxes(rows, image_hw=image_hw, ratio=ratio, pad=pad)
        rows = self._suppress_contained_boxes(rows)
        rows[:, 4] = np.clip(rows[:, 4], 0.0, 1.0)
        return rows, boxes_to_polygons(rows)


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

                overlap = box_intersection_area(outer_box, boxes[inner_idx])
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
        return self._page_from_regions(polygons, scores, ordered=True)


    def predict(
        self,
        img_or_path: Union[str, Path, np.ndarray],
        return_raw: bool = False,
        *, return_masks: bool = False, return_outputs: bool = False,
    ):
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
        >>> model = YOLO(weights="yolo26x_obb_text_g1")
        >>> page = model.predict("page.jpg")
        >>> first_text_span = page.blocks[0].lines[0].text_spans[0]
        >>> print(first_text_span.detection_confidence)
        """
        if self.onnx_session is None:
            self._initialize_session()

        image = read_image(img_or_path)
        tensor, ratio, pad = self._preprocess(image)
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: tensor})
        output = outputs[0]
        boxes, polygons = self._postprocess(
            output,
            image_hw=image.shape[:2],
            ratio=ratio,
            pad=pad,
            preserve_geometry=True,
        )

        page_polygons = polygons
        if self._output_layout == "obb" and self.axis_aligned_output:
            page_polygons = boxes_to_polygons(polygons_to_boxes(polygons))
        page = self._polygons_to_page(page_polygons, boxes[:, 4])
        if return_raw or return_masks or return_outputs:
            detections = []
            for row, polygon in zip(boxes, polygons):
                detail = {"class_id": int(row[5]), "confidence": float(row[4]),
                          "polygon": polygon.astype(float).tolist(), "box": row.tolist()}
                if len(row) == 7:
                    detail["angle"] = float(row[6])
                detections.append(detail)
            return self._raw_result(page, dict(zip(
                [item.name for item in self.onnx_session.get_outputs()], outputs)) if return_outputs else None,
                detections, image.shape[:2], resize_ratio=ratio, padding=pad)
        return page
