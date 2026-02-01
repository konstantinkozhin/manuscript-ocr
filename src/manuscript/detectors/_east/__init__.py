import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

from manuscript.api.base import BaseModel

from ...data import Block, Line, Page, Word
from ...utils import read_image, organize_page
from .lanms import locality_aware_nms
from .utils import (
    decode_quads_from_maps,
    expand_boxes,
)

# Optional imports for training (not needed for inference)
try:
    import torch
    from torch.utils.data import ConcatDataset
    from .dataset import EASTDataset
    from .east import EASTModel
    from .train_utils import _run_training

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    ConcatDataset = None
    EASTDataset = None
    EASTModel = None
    _run_training = None
    _TORCH_AVAILABLE = False


class EAST(BaseModel):
    """
    Initialize EAST text detector with ONNX Runtime.

    Parameters
    ----------
    weights : str or Path, optional
        Path or identifier for ONNX model weights. Supports:

        - Local file path: ``"path/to/model.onnx"``
        - HTTP/HTTPS URL: ``"https://example.com/model.onnx"``
        - GitHub release: ``"github://owner/repo/tag/file.onnx"``
        - Google Drive: ``"gdrive:FILE_ID"``
        - Preset name: ``"east_50_g1"``
        - ``None``: auto-downloads default preset (east_50_g1)

    device : str, optional
        Compute device: ``"cuda"``, ``"coreml"``, or ``"cpu"``. If None,
        automatically selects CPU. For GPU/CoreML acceleration:

        - CUDA (NVIDIA): ``pip install onnxruntime-gpu``
        - CoreML (Apple Silicon M1/M2/M3): ``pip install onnxruntime-silicon``

        Default is ``None`` (CPU).
    target_size : int, optional
        Input image size for inference. Images are resized to
        ``(target_size, target_size)``. Default is 1280.
    expand_ratio_w : float, optional
        Horizontal expansion factor applied to detected boxes after NMS.
        Default is 0.7.
    expand_ratio_h : float, optional
        Vertical expansion factor applied to detected boxes after NMS.
        Default is 0.7.
    expand_power : float, optional
        Power for non-linear box expansion. Controls how expansion scales with box size.
        - 1.0 = linear (small and large boxes expand equally)
        - <1.0 = small boxes expand more (e.g., 0.5, recommended for character-level detection)
        - >1.0 = large boxes expand more
        Default is 0.5.
    score_thresh : float, optional
        Confidence threshold for selecting candidate detections before NMS.
        Default is 0.7.
    iou_threshold : float, optional
        IoU threshold for locality-aware NMS merging phase. Default is 0.2.
    iou_threshold_standard : float, optional
        IoU threshold for standard NMS after locality-aware merging.
        If None, uses the same value as iou_threshold. Default is None.
    score_geo_scale : float, optional
        Scale factor for decoding geometry/score maps. Default is 0.25.
    quantization : int, optional
        Quantization resolution for point coordinates during decoding.
        Default is 2.
    axis_aligned_output : bool, optional
        If True, outputs axis-aligned rectangles instead of original quads.
        Default is True.
    remove_area_anomalies : bool, optional
        If True, removes quads with extremely large area relative to the
        distribution. Default is False.
    anomaly_sigma_threshold : float, optional
        Sigma threshold for anomaly area filtering. Default is 5.0.
    anomaly_min_box_count : int, optional
        Minimum number of boxes required before anomaly filtering.
        Default is 30.
    use_tta : bool, optional
        Enable Test-Time Augmentation (TTA). When enabled, inference is run
        on both the original and horizontally flipped image, and results are
        merged. This can improve detection of partially visible or edge text.
        Default is False.
    tta_iou_thresh : float, optional
        IoU threshold for merging boxes from original and flipped images
        during TTA. Boxes with IoU > threshold are considered matches and
        merged. Default is 0.1.

    Notes
    -----
    The class provides two main public methods:

    - ``predict`` — run inference on a single image and return detections.
    - ``train`` — high-level training entrypoint to train an EAST model on custom datasets.

    The detector uses ONNX Runtime for fast inference on CPU and GPU.
    For GPU acceleration, install: ``pip install onnxruntime-gpu``
    """

    default_weights_name = "east_50_g1"

    pretrained_registry = {
        "east_50_g1": "github://konstantinkozhin/manuscript-ocr/v0.1.0/east_50_g1.onnx",
    }

    def __init__(
        self,
        weights: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        *,
        target_size: int = 1280,
        expand_ratio_w: float = 1.4,
        expand_ratio_h: float = 1.5,
        expand_power: float = 0.6,
        score_thresh: float = 0.6,
        iou_threshold: float = 0.05,
        iou_threshold_standard: Optional[float] = 0.05,
        score_geo_scale: float = 0.25,
        quantization: int = 2,
        axis_aligned_output: bool = True,
        remove_area_anomalies: bool = False,
        anomaly_sigma_threshold: float = 5.0,
        anomaly_min_box_count: int = 30,
        use_tta: bool = False,
        tta_iou_thresh: float = 0.1,
        **kwargs,
    ):
        super().__init__(weights=weights, device=device, **kwargs)

        self.onnx_session = None

        self.target_size = target_size
        self.expand_ratio_w = expand_ratio_w
        self.expand_ratio_h = expand_ratio_h
        self.expand_power = expand_power

        self.score_thresh = score_thresh
        self.iou_threshold = iou_threshold
        self.iou_threshold_standard = iou_threshold_standard

        self.score_geo_scale = score_geo_scale
        self.quantization = quantization

        self.axis_aligned_output = axis_aligned_output
        self.remove_area_anomalies = remove_area_anomalies
        self.anomaly_sigma_threshold = anomaly_sigma_threshold
        self.anomaly_min_box_count = anomaly_min_box_count

        # TTA parameters
        self.use_tta = use_tta
        self.tta_iou_thresh = tta_iou_thresh

    def _initialize_session(self):
        if self.onnx_session is not None:
            return

        providers = self.runtime_providers()

        self.onnx_session = ort.InferenceSession(
            self.weights,
            providers=providers,
        )

        self._log_device_info(self.onnx_session)

    def _scale_boxes_to_original(
        self, boxes: np.ndarray, orig_size: Tuple[int, int]
    ) -> np.ndarray:
        if len(boxes) == 0:
            return boxes

        orig_h, orig_w = orig_size
        scale_x = orig_w / self.target_size
        scale_y = orig_h / self.target_size

        scaled = boxes.copy()
        scaled[:, 0:8:2] *= scale_x
        scaled[:, 1:8:2] *= scale_y
        return scaled

    def _convert_to_axis_aligned(self, quads: np.ndarray) -> np.ndarray:
        if len(quads) == 0:
            return quads
        aligned = quads.copy()
        coords = aligned[:, :8].reshape(-1, 4, 2)
        x_min = coords[:, :, 0].min(axis=1)
        x_max = coords[:, :, 0].max(axis=1)
        y_min = coords[:, :, 1].min(axis=1)
        y_max = coords[:, :, 1].max(axis=1)
        rects = np.stack(
            [
                x_min,
                y_min,
                x_max,
                y_min,
                x_max,
                y_max,
                x_min,
                y_max,
            ],
            axis=1,
        )
        aligned[:, :8] = rects.reshape(-1, 8)
        return aligned

    @staticmethod
    def _polygon_area_batch(polys: np.ndarray) -> np.ndarray:
        if polys.size == 0:
            return np.zeros((0,), dtype=np.float32)
        x = polys[:, :, 0]
        y = polys[:, :, 1]
        return 0.5 * np.abs(
            np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)
        )

    def _is_quad_inside(self, inner: np.ndarray, outer: np.ndarray) -> bool:
        contour = outer.reshape(-1, 1, 2).astype(np.float32)
        for point in inner.astype(np.float32):
            if (
                cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
                < 0
            ):
                return False
        return True

    def _remove_fully_contained_boxes(self, quads: np.ndarray) -> np.ndarray:
        if len(quads) <= 1:
            return quads
        coords = quads[:, :8].reshape(-1, 4, 2)
        areas = self._polygon_area_batch(coords)
        keep = np.ones(len(quads), dtype=bool)
        order = np.argsort(areas)
        for idx in order:
            if not keep[idx]:
                continue
            inner = coords[idx]
            inner_area = areas[idx]
            for jdx in range(len(quads)):
                if idx == jdx or not keep[jdx]:
                    continue
                if areas[jdx] + 1e-6 < inner_area:
                    continue
                if self._is_quad_inside(inner, coords[jdx]):
                    keep[idx] = False
                    break
        return quads[keep]

    def _remove_area_anomalies(self, quads: np.ndarray) -> np.ndarray:
        if (
            not self.remove_area_anomalies
            or len(quads) == 0
            or len(quads) <= self.anomaly_min_box_count
        ):
            return quads
        coords = quads[:, :8].reshape(-1, 4, 2)
        areas = self._polygon_area_batch(coords).astype(np.float32)
        mean = float(np.mean(areas))
        std = float(np.std(areas))
        if std == 0.0:
            return quads
        threshold = mean + self.anomaly_sigma_threshold * std
        keep = areas <= threshold
        if not np.any(keep):
            return quads
        return quads[keep]

    @staticmethod
    def _box_iou(
        box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate IoU between two axis-aligned boxes.

        Parameters
        ----------
        box1, box2 : tuple of (x0, y0, x1, y1)
            Bounding box coordinates.

        Returns
        -------
        float
            Intersection over Union value in [0, 1].
        """
        x0 = max(box1[0], box2[0])
        y0 = max(box1[1], box2[1])
        x1 = min(box1[2], box2[2])
        y1 = min(box1[3], box2[3])

        if x1 <= x0 or y1 <= y0:
            return 0.0

        inter = (x1 - x0) * (y1 - y0)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / float(area1 + area2 - inter)

    def _merge_tta_boxes(
        self,
        boxes_orig: List[Tuple[Tuple[int, int, int, int], float]],
        boxes_flipped: List[Tuple[Tuple[int, int, int, int], float]],
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Merge detection boxes from original and horizontally flipped images.

        For each box in the original image, find matching box in flipped image
        (based on IoU threshold). Matched boxes are merged by extending x-coordinates
        and keeping y-coordinates from original, with averaged scores.

        This approach keeps vertical positioning stable while potentially
        extending horizontal coverage when both views detect overlapping regions.

        Parameters
        ----------
        boxes_orig : list of ((x0, y0, x1, y1), score)
            Boxes from original image detection.
        boxes_flipped : list of ((x0, y0, x1, y1), score)
            Boxes from flipped image detection (already transformed back
            to original coordinate space).

        Returns
        -------
        list of ((x0, y0, x1, y1), score)
            Merged boxes - only boxes that have matches in both views.
        """
        merged = []

        for box1, score1 in boxes_orig:
            for box2, score2 in boxes_flipped:
                iou = self._box_iou(box1, box2)
                if iou > self.tta_iou_thresh:
                    # Merge: extend x-coordinates, keep y from original
                    merged_box = (
                        min(box1[0], box2[0]),  # x0: take minimum
                        box1[1],  # y0: keep from original
                        max(box1[2], box2[2]),  # x1: take maximum
                        box1[3],  # y1: keep from original
                    )
                    avg_score = (score1 + score2) / 2
                    merged.append((merged_box, avg_score))
                    break  # Found match, move to next original box

        return merged

    def _run_inference_on_image(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run ONNX inference on a single image.

        Parameters
        ----------
        img : np.ndarray
            RGB image with shape (H, W, 3).

        Returns
        -------
        tuple of (final_quads_nms, score_map, geo_map)
            Detected quads after NMS and raw maps.
        """
        resized = cv2.resize(img, (self.target_size, self.target_size))

        img_norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        img_input = img_norm.transpose(2, 0, 1)[np.newaxis, :, :, :]

        input_name = self.onnx_session.get_inputs()[0].name
        output_names = [out.name for out in self.onnx_session.get_outputs()]

        outputs = self.onnx_session.run(output_names, {input_name: img_input})

        score_map = outputs[0].squeeze(0).squeeze(0)
        geo_map = outputs[1].squeeze(0)

        final_quads = decode_quads_from_maps(
            score_map=score_map,
            geo_map=geo_map.transpose(1, 2, 0),
            score_thresh=self.score_thresh,
            scale=1.0 / self.score_geo_scale,
            quantization=self.quantization,
        )

        final_quads_nms = locality_aware_nms(
            final_quads,
            iou_threshold=self.iou_threshold,
            iou_threshold_standard=self.iou_threshold_standard,
        )

        return final_quads_nms, score_map, geo_map

    def predict(
        self,
        img_or_path: Union[str, Path, np.ndarray],
        return_maps: bool = False,
        sort_reading_order: bool = True,
        split_into_columns: bool = True,
        max_columns: int = 10,
    ) -> Dict[str, Any]:
        """
        Run EAST inference and return detection results.

        Parameters
        ----------
        img_or_path : str or pathlib.Path or numpy.ndarray
            Path to an image file or an RGB image provided as a NumPy array
            with shape ``(H, W, 3)`` in ``uint8`` format.
        return_maps : bool, optional
            If True, returns raw model score and geometry maps under keys
            ``"score_map"`` and ``"geo_map"``. Default is False.
        sort_reading_order : bool, optional
            If True, sorts detected words in natural reading order
            (left-to-right, top-to-bottom) and groups them into text lines.
            Default is True.
        split_into_columns : bool, optional
            If True and ``sort_reading_order=True``, segments the page into
            columns (separate Blocks). If False, treats entire page as single
            column. Only used when ``sort_reading_order=True``. Default is True.
        max_columns : int, optional
            Maximum number of columns to detect when ``split_into_columns=True``.
            Higher values allow more columns to be detected. Only used when
            ``sort_reading_order=True`` and ``split_into_columns=True``.
            Default is 10.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - ``"page"`` : Page
                Parsed detection result as a Page object containing Block(s) with
                Line(s) of Word objects. Each Word has polygon coordinates and
                confidence scores. Words and Lines have reading order indices.
            - ``"score_map"`` : numpy.ndarray or None
                Raw score map produced by the network if ``return_maps=True``.
            - ``"geo_map"`` : numpy.ndarray or None
                Raw geometry map if ``return_maps=True``.

        Notes
        -----
        The method performs:
        (1) image loading, (2) resizing and normalization, (3) model inference,
        (4) quad decoding, (5) NMS, (6) box expansion, (7) scaling coordinates
        back to original size, (8) optional reading order sorting into lines.

        **Test-Time Augmentation (TTA):**

        When ``use_tta=True`` is set during initialization, the method runs
        inference on both the original and horizontally flipped image, then
        merges results. Boxes from both views are matched by IoU and merged
        by taking the union of coordinates with averaged scores. This can
        improve detection of text near image edges or partially visible text.

        For visualization, use the external ``visualize_page`` utility:

        >>> from manuscript.utils import visualize_page
        >>> result = model.predict(img_path)
        >>> vis_img = visualize_page(img, result["page"])

        Examples
        --------
        Perform inference and get structured output:

        >>> from manuscript.detectors import EAST
        >>> model = EAST()
        >>> img_path = r"example/ocr_example_image.jpg"
        >>> result = model.predict(img_path)
        >>> page = result["page"]
        >>> # Access first line's first word
        >>> first_word = page.blocks[0].lines[0].words[0]
        >>> print(f"Confidence: {first_word.detection_confidence}")

        Visualize results separately:

        >>> from manuscript.utils import visualize_page, read_image
        >>> result = model.predict(img_path)
        >>> img = read_image(img_path)
        >>> vis_img = visualize_page(img, result["page"])
        >>> vis_img.show()

        """

        if self.onnx_session is None:
            self._initialize_session()

        img = read_image(img_or_path)
        orig_h, orig_w = img.shape[:2]

        # Run inference on original image
        final_quads_nms, score_map, geo_map = self._run_inference_on_image(img)

        # TTA: Run on horizontally flipped image and merge results
        if self.use_tta:
            # Flip image horizontally
            img_flipped = np.fliplr(img).copy()
            final_quads_flipped, _, _ = self._run_inference_on_image(img_flipped)

            # Scale both to original size first
            scaled_orig = self._scale_boxes_to_original(
                final_quads_nms, (orig_h, orig_w)
            )
            scaled_flipped = self._scale_boxes_to_original(
                final_quads_flipped, (orig_h, orig_w)
            )

            # Convert to axis-aligned boxes for IoU matching
            # Keep track of original quads for non-axis-aligned output
            boxes_orig = []
            quads_orig_list = []  # Store original quads
            for quad in scaled_orig:
                pts = quad[:8].reshape(4, 2)
                x0, y0 = pts.min(axis=0)
                x1, y1 = pts.max(axis=0)
                score = float(np.clip(quad[8], 0.0, 1.0))
                boxes_orig.append(((int(x0), int(y0), int(x1), int(y1)), score))
                quads_orig_list.append(quad.copy())

            boxes_flipped = []
            for quad in scaled_flipped:
                pts = quad[:8].reshape(4, 2)
                x0, y0 = pts.min(axis=0)
                x1, y1 = pts.max(axis=0)
                # Mirror x coordinates back
                x0_mirrored = orig_w - x1
                x1_mirrored = orig_w - x0
                score = float(np.clip(quad[8], 0.0, 1.0))
                boxes_flipped.append(
                    ((int(x0_mirrored), int(y0), int(x1_mirrored), int(y1)), score)
                )

            # Find which original boxes have matches in flipped view
            matched_orig_indices = []
            for idx, (box1, score1) in enumerate(boxes_orig):
                for box2, score2 in boxes_flipped:
                    iou = self._box_iou(box1, box2)
                    if iou > self.tta_iou_thresh:
                        matched_orig_indices.append(idx)
                        break  # Found match, move to next original box

            # Filter to keep only matched quads (preserves original 4-point polygons)
            if matched_orig_indices:
                matched_quads = np.stack(
                    [quads_orig_list[i] for i in matched_orig_indices], axis=0
                )
            else:
                matched_quads = np.empty((0, 9), dtype=np.float32)

            # Expand boxes
            expanded = expand_boxes(
                matched_quads,
                expand_w=self.expand_ratio_w,
                expand_h=self.expand_ratio_h,
                expand_power=self.expand_power,
            )
        else:
            # No TTA: standard processing
            expanded = expand_boxes(
                final_quads_nms,
                expand_w=self.expand_ratio_w,
                expand_h=self.expand_ratio_h,
                expand_power=self.expand_power,
            )
            expanded = self._scale_boxes_to_original(expanded, (orig_h, orig_w))

        processed_quads = self._remove_fully_contained_boxes(expanded)
        processed_quads = self._remove_area_anomalies(processed_quads)
        output_quads = (
            self._convert_to_axis_aligned(processed_quads)
            if self.axis_aligned_output
            else processed_quads
        )

        words: List[Word] = []
        for quad in output_quads:
            pts = quad[:8].reshape(4, 2)
            score = float(np.clip(quad[8], 0.0, 1.0))
            words.append(Word(polygon=pts.tolist(), detection_confidence=score))

        if sort_reading_order and len(words) > 0:
            initial_page = Page(
                blocks=[Block(lines=[Line(words=words, order=0)], order=0)]
            )
            page = organize_page(
                initial_page, max_splits=max_columns, use_columns=split_into_columns
            )
        else:
            for idx, w in enumerate(words):
                w.order = idx
            page = Page(blocks=[Block(lines=[Line(words=words, order=0)], order=0)])

        return {
            "page": page,
            "score_map": score_map if return_maps else None,
            "geo_map": geo_map if return_maps else None,
        }

    @staticmethod
    def train(
        train_images: Union[str, Path, Sequence[Union[str, Path]]],
        train_anns: Union[str, Path, Sequence[Union[str, Path]]],
        val_images: Union[str, Path, Sequence[Union[str, Path]]],
        val_anns: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        experiment_root: str = "./experiments",
        model_name: str = "resnet_quad",
        backbone_name: str = "resnet50",
        pretrained_backbone: bool = True,
        freeze_first: bool = False,
        target_size: int = 1024,
        score_geo_scale: Optional[float] = None,
        score_map_shrink_ratio: float = 0.3,
        epochs: int = 500,
        batch_size: int = 3,
        accumulation_steps: int = 1,
        lr: float = 1e-4,
        lr_scheduler: str = "cosine_restart",
        lr_scheduler_params: Optional[Dict[str, Any]] = None,
        augmentation_config: Optional[Dict[str, Any]] = None,
        grad_clip: float = 5.0,
        early_stop: int = 100,
        use_sam: bool = False,
        sam_type: str = "asam",
        use_lookahead: bool = False,
        use_ema: bool = False,
        use_multiscale: bool = True,
        use_ohem: bool = False,
        ohem_ratio: float = 0.5,
        use_focal_geo: bool = True,
        focal_gamma: float = 2.0,
        resume_from: Optional[Union[str, Path]] = None,
        val_interval: int = 1,
        num_workers: int = 0,
        device: Optional["torch.device"] = None,
    ) -> "torch.nn.Module":
        """
        Train EAST model on custom datasets.

        Parameters
        ----------
        train_images : str, Path or sequence of paths
            Path(s) to training image folders.
        train_anns : str, Path or sequence of paths
            Path(s) to COCO-format JSON annotation files corresponding to
            ``train_images``.
        val_images : str, Path or sequence of paths
            Path(s) to validation image folders.
        val_anns : str, Path or sequence of paths
            Path(s) to COCO-format JSON annotation files corresponding to
            ``val_images``.
        experiment_root : str, optional
            Base directory where experiment folders will be created.
            Default is ``"./experiments"``.
        model_name : str, optional
            Folder name inside ``experiment_root`` for logs and checkpoints.
            Default is ``"resnet_quad"``.
        backbone_name : {"resnet50", "resnet101"}, optional
            Backbone architecture to use. Options:

            - ``"resnet50"`` — ResNet-50 (faster, less parameters)
            - ``"resnet101"`` — ResNet-101 (slower, more capacity)

            Default is ``"resnet50"``.
        pretrained_backbone : bool, optional
            Use ImageNet-pretrained backbone weights. Default ``True``.
        freeze_first : bool, optional
            Freeze lowest layers of the backbone. Default ``True``.
        target_size : int, optional
            Resize shorter side of images to this size. Default ``1024``.
        score_geo_scale : float, optional
            Multiplier to recover original coordinates from score/geo maps.
            If None, automatically taken from the model. Default ``None``.
        score_map_shrink_ratio : float, optional
            Shrink ratio for score map. Default ``0.3``.
        epochs : int, optional
            Number of training epochs. Default ``500``.
        batch_size : int, optional
            Batch size per GPU. Default ``3``.
        accumulation_steps : int, optional
            Number of gradient accumulation steps. Effective batch size will be
            ``batch_size * accumulation_steps``. Use this to train with larger
            effective batch sizes when GPU memory is limited. For example:

            - ``batch_size=2, accumulation_steps=4`` → effective batch size = 8
            - ``batch_size=1, accumulation_steps=8`` → effective batch size = 8

            Default is ``1`` (no accumulation).
        lr : float, optional
            Learning rate. Default ``1e-3``.
        lr_scheduler : {"cosine_restart", "cosine", "linear", "step", "exponential", "plateau", "none"}, optional
            Learning rate scheduler type. Default ``"cosine_restart"``.
        lr_scheduler_params : dict, optional
            Extra scheduler parameters (depends on scheduler type). Common keys:

            - ``cosine_restart``: ``t0``, ``t_mult``, ``eta_min``
            - ``cosine``: ``t_max``, ``eta_min``
            - ``linear``: ``final_factor``
            - ``step``: ``step_size``, ``gamma``
            - ``exponential``: ``gamma``
            - ``plateau``: ``factor``, ``patience``, ``min_lr``
        augmentation_config : dict, optional
            Full augmentation configuration passed to ``EASTDataset``. This is the
            preferred place for all augmentation settings. Any keys here override the
            corresponding defaults. Supported keys include:

            - ``flip_prob``, ``color_jitter``
            - ``small_rotate_prob``, ``small_rotate_deg``
            - ``perspective_prob``, ``perspective_scale``
            - ``blur_prob``, ``blur_ksize_range``
            - ``noise_prob``, ``noise_std``, ``salt_pepper_prob``
            - ``jpeg_prob``, ``jpeg_quality_range``
            - ``shading_prob``, ``shading_strength``
            - ``gamma_prob``, ``gamma_range``
            - ``downscale_prob``, ``downscale_range``
            - ``negative_prob``
        grad_clip : float, optional
            Gradient clipping value (L2 norm). Default ``5.0``.
        early_stop : int, optional
            Patience (epochs without improvement) for early stopping.
            Default ``100``.
        use_sam : bool, optional
            Enable SAM optimizer. Default ``True``.
        sam_type : {"sam", "asam"}, optional
            Variant of SAM to use. Default ``"asam"``.
        use_lookahead : bool, optional
            Wrap optimizer with Lookahead. Default ``True``.
        use_ema : bool, optional
            Maintain EMA version of model weights. Default ``False``.
        use_multiscale : bool, optional
            Random multi-scale training. Default ``True``.
        use_ohem : bool, optional
            Online Hard Example Mining. Default ``True``.
        ohem_ratio : float, optional
            Ratio of hard negatives for OHEM. Default ``0.5``.
        use_focal_geo : bool, optional
            Apply focal loss to geometry channels. Default ``True``.
        focal_gamma : float, optional
            Gamma for focal geometry loss. Default ``2.0``.
        resume_from : str or Path, optional
            Resume training from a previous experiment:
            a) experiment directory,
            b) `.../checkpoints/`,
            c) direct path to `last_state.pt`.
            Default ``None``.
        val_interval : int, optional
            Run validation every N epochs. Default ``1``.
        num_workers : int, optional
            Number of worker processes for data loading. Set to 0 for single-process
            loading (safer on Windows). Default ``0``.
        device : torch.device, optional
            CUDA or CPU device. Auto-selects if None.

        Returns
        -------
        torch.nn.Module
            Best model weights (EMA if enabled, otherwise base model).

        Examples
        --------
        Train on two datasets with validation:

        >>> from manuscript.detectors import EAST
        >>>
        >>> train_images = [
        ...     "/data/archive/train_images",
        ...     "/data/ddi/train_images"
        ... ]
        >>> train_anns = [
        ...     "/data/archive/train.json",
        ...     "/data/ddi/train.json"
        ... ]
        >>> val_images = [
        ...     "/data/archive/test_images",
        ...     "/data/ddi/test_images"
        ... ]
        >>> val_anns = [
        ...     "/data/archive/test.json",
        ...     "/data/ddi/test.json"
        ... ]
        >>>
        >>> best_model = EAST.train(
        ...     train_images=train_images,
        ...     train_anns=train_anns,
        ...     val_images=val_images,
        ...     val_anns=val_anns,
        ...     backbone_name="resnet50",
        ...     target_size=256,
        ...     epochs=20,
        ...     batch_size=4,
        ...     use_sam=False,
        ...     freeze_first=False,
        ...     val_interval=3,
        ... )
        >>> print("Best checkpoint loaded:", best_model)

        Configure augmentations via ``augmentation_config``:

        >>> aug_cfg = {
        ...     "flip_prob": 0.02,
        ...     "color_jitter": (0.1, 0.1, 0.1, 0.05),
        ...     "small_rotate_prob": 0.15,
        ...     "small_rotate_deg": 1.5,
        ...     "perspective_prob": 0.08,
        ...     "perspective_scale": 0.015,
        ...     "blur_prob": 0.1,
        ...     "blur_ksize_range": (3, 5),
        ...     "noise_prob": 0.1,
        ...     "noise_std": 0.008,
        ...     "salt_pepper_prob": 0.0005,
        ...     "jpeg_prob": 0.1,
        ...     "jpeg_quality_range": (75, 95),
        ...     "shading_prob": 0.1,
        ...     "shading_strength": 0.1,
        ...     "gamma_prob": 0.2,
        ...     "gamma_range": (0.95, 1.05),
        ...     "downscale_prob": 0.1,
        ...     "downscale_range": (0.7, 0.95),
        ...     "negative_prob": 0.05,
        ... }
        >>> best_model = EAST.train(
        ...     train_images=train_images,
        ...     train_anns=train_anns,
        ...     val_images=val_images,
        ...     val_anns=val_anns,
        ...     augmentation_config=aug_cfg,
        ... )
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for training. "
                "Install with: pip install manuscript-ocr[dev]"
            )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = EASTModel(
            backbone_name=backbone_name,
            pretrained_backbone=pretrained_backbone,
            freeze_first=freeze_first,
        ).to(device)

        if score_geo_scale is None:
            score_geo_scale = model.score_scale

        base_augmentation_config = {
            "flip_prob": 0.01,
            "color_jitter": (0.1, 0.1, 0.1, 0.05),
            "small_rotate_prob": 0.2,
            "small_rotate_deg": 2.0,
            "perspective_prob": 0.1,
            "perspective_scale": 0.015,
            "blur_prob": 0.1,
            "blur_ksize_range": (3, 5),
            "noise_prob": 0.1,
            "noise_std": 0.008,
            "salt_pepper_prob": 0.0005,
            "jpeg_prob": 0.1,
            "jpeg_quality_range": (75, 95),
            "shading_prob": 0.1,
            "shading_strength": 0.1,
            "gamma_prob": 0.2,
            "gamma_range": (0.95, 1.05),
            "downscale_prob": 0.1,
            "downscale_range": (0.7, 0.95),
            "negative_prob": 0.05,
        }
        aug_cfg = dict(base_augmentation_config)
        if augmentation_config:
            aug_cfg.update(augmentation_config)

        def make_dataset(imgs, anns, name: Optional[str] = None):
            return EASTDataset(
                images_folder=imgs,
                coco_annotation_file=anns,
                target_size=target_size,
                score_geo_scale=score_geo_scale,
                shrink_ratio=score_map_shrink_ratio,
                **aug_cfg,
                dataset_name=name,
            )

        def _dataset_base_name(
            img_path: Union[str, Path], ann_path: Union[str, Path]
        ) -> str:
            ann = Path(os.fspath(ann_path))
            parts: List[str] = []
            if ann.parent.name:
                parts.append(ann.parent.name)
            if ann.stem:
                parts.append(ann.stem)
            if not parts:
                img = Path(os.fspath(img_path))
                if img.parent.name:
                    parts.append(img.parent.name)
                stem = img.stem or img.name
                if stem:
                    parts.append(stem)
            return "/".join(parts)

        def _unique_dataset_name(
            img_path: Union[str, Path],
            ann_path: Union[str, Path],
            counts: Dict[str, int],
            idx: int,
            kind: str,
        ) -> str:
            base = _dataset_base_name(img_path, ann_path)
            if not base:
                base = f"{kind}_{idx}"
            count = counts.get(base, 0)
            counts[base] = count + 1
            if count == 0:
                return base
            return f"{base}_{count + 1}"

        train_imgs_list = (
            train_images if isinstance(train_images, (list, tuple)) else [train_images]
        )
        train_anns_list = (
            train_anns if isinstance(train_anns, (list, tuple)) else [train_anns]
        )
        val_imgs_list = (
            val_images if isinstance(val_images, (list, tuple)) else [val_images]
        )
        val_anns_list = val_anns if isinstance(val_anns, (list, tuple)) else [val_anns]
        assert len(train_imgs_list) == len(
            train_anns_list
        ), "train_images and train_anns must have the same length"
        assert len(val_imgs_list) == len(
            val_anns_list
        ), "val_images and val_anns must have the same length"

        train_datasets = []
        train_name_counts: Dict[str, int] = {}
        for idx, (imgs, anns) in enumerate(
            zip(train_imgs_list, train_anns_list), start=1
        ):
            dataset_name = _unique_dataset_name(
                imgs, anns, train_name_counts, idx=idx, kind="train"
            )
            train_datasets.append(make_dataset(imgs, anns, name=dataset_name))

        val_datasets = []
        val_name_counts: Dict[str, int] = {}
        for idx, (imgs, anns) in enumerate(zip(val_imgs_list, val_anns_list), start=1):
            dataset_name = _unique_dataset_name(
                imgs, anns, val_name_counts, idx=idx, kind="val"
            )
            val_datasets.append(make_dataset(imgs, anns, name=dataset_name))

        train_ds = ConcatDataset(train_datasets)
        val_ds = ConcatDataset(val_datasets)
        val_dataset_names = [ds.dataset_name for ds in val_datasets]

        def _resolve_path(path: Union[str, Path]) -> Path:
            p = Path(path)
            if p.is_absolute():
                return p
            project_root = Path(__file__).resolve().parents[4]
            candidate = (project_root / p).resolve()
            if candidate.exists():
                return candidate
            return (Path.cwd() / p).resolve()

        def _is_experiment_checkpoint(file_path: Path) -> bool:
            parent = file_path.parent
            return (
                parent.name == "checkpoints"
                or (parent / "training_config.json").exists()
            )

        def _resolve_resume_target(
            target: Union[str, Path],
            default_experiment_dir: str,
        ) -> Tuple[str, Optional[Path]]:
            resolved = _resolve_path(target)
            if not resolved.exists():
                raise FileNotFoundError(
                    f"resume_from target does not exist: {resolved}"
                )

            if resolved.is_file():
                resume_state = resolved
                if _is_experiment_checkpoint(resolved):
                    checkpoints_dir = resolved.parent
                    if checkpoints_dir.name == "checkpoints":
                        experiment_dir = checkpoints_dir.parent
                    else:
                        experiment_dir = checkpoints_dir
                    return os.path.abspath(os.fspath(experiment_dir)), resume_state
                else:
                    return default_experiment_dir, resume_state

            experiment_dir = resolved
            checkpoints_dir = (
                resolved if resolved.name == "checkpoints" else resolved / "checkpoints"
            )
            default_state = checkpoints_dir / "last_state.pt"
            resume_state = default_state if default_state.exists() else None
            return os.path.abspath(os.fspath(experiment_dir)), resume_state

        default_experiment_dir = os.path.abspath(
            os.path.join(experiment_root, model_name)
        )
        resume_state_path: Optional[Path] = None
        if resume_from is None:
            experiment_dir = default_experiment_dir
            resume_flag = False
        else:
            experiment_dir, resume_state_path = _resolve_resume_target(
                resume_from, default_experiment_dir
            )
            resume_flag = True

        best_model = _run_training(
            experiment_dir=experiment_dir,
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            device=device,
            num_epochs=epochs,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            grad_clip=grad_clip,
            early_stop=early_stop,
            use_sam=use_sam,
            sam_type=sam_type,
            use_lookahead=use_lookahead,
            use_ema=use_ema,
            use_multiscale=use_multiscale,
            use_ohem=use_ohem,
            ohem_ratio=ohem_ratio,
            use_focal_geo=use_focal_geo,
            focal_gamma=focal_gamma,
            val_interval=val_interval,
            num_workers=num_workers,
            backbone_name=backbone_name,
            target_size=target_size,
            pretrained_backbone=pretrained_backbone,
            val_datasets=val_datasets,
            val_dataset_names=val_dataset_names,
            resume=resume_flag,
            resume_state_path=(
                os.fspath(resume_state_path) if resume_state_path else None
            ),
            score_map_shrink_ratio=score_map_shrink_ratio,
            augmentation_config=aug_cfg,
        )
        return best_model

    @staticmethod
    def export(
        weights_path: Union[str, Path],
        output_path: Union[str, Path],
        backbone_name: str = None,
        input_size: int = 1280,
        opset_version: int = 14,
        simplify: bool = True,
    ) -> None:
        """
        Export EAST PyTorch model to ONNX format.

        This method converts a trained EAST model from PyTorch to ONNX format,
        which can be used for faster inference with ONNX Runtime. The exported
        model can be loaded using ``EAST(weights_path="model.onnx", use_onnx=True)``.

        Parameters
        ----------
        weights_path : str or Path
            Path to the PyTorch model weights file (.pth).
        output_path : str or Path
            Path where the ONNX model will be saved (.onnx).
        backbone_name : {"resnet50", "resnet101"}, optional
            Backbone architecture of the model. If None, will be automatically
            detected from the checkpoint. Must match the architecture used during
            training. Default is None (auto-detect).
        input_size : int, optional
            Input image size (height and width). The model will accept
            images of shape ``(batch, 3, input_size, input_size)``.
            Default is 1280.
        opset_version : int, optional
            ONNX opset version to use for export. Default is 14.
        simplify : bool, optional
            If True, applies ONNX graph simplification using onnx-simplifier
            to optimize the model. Requires ``onnx-simplifier`` package.
            Default is True.

        Returns
        -------
        None
            The ONNX model is saved to ``output_path``.

        Raises
        ------
        ImportError
            If required packages (torch, onnx) are not installed.
        FileNotFoundError
            If ``weights_path`` does not exist.
        ValueError
            If backbone_name doesn't match the checkpoint architecture.

        Notes
        -----
        The exported ONNX model has two outputs:

        - ``score_map``: Text confidence map with shape ``(batch, 1, H, W)``
        - ``geo_map``: Geometry map with shape ``(batch, 8, H, W)``

        The model supports dynamic batch size and image dimensions through
        dynamic axes configuration.

        **Automatic Backbone Detection:**

        The method automatically detects the backbone architecture from the checkpoint
        by analyzing the number of parameters in layer4. This prevents mismatches between
        checkpoint and architecture that could lead to incorrect exports.

        Examples
        --------
        Export with automatic backbone detection:

        >>> from manuscript.detectors import EAST
        >>> EAST.export(
        ...     weights_path="east_resnet50.pth",
        ...     output_path="east_model.onnx"
        ... )
        Auto-detected backbone: resnet50
        Exporting to ONNX (opset 14)...
        [OK] ONNX model saved to: east_model.onnx

        Export with explicit backbone:

        >>> EAST.export(
        ...     weights_path="custom_weights.pth",
        ...     output_path="custom_model.onnx",
        ...     backbone_name="resnet101",
        ...     input_size=1024,
        ...     simplify=False
        ... )

        Use the exported model for inference:

        >>> detector = EAST(
        ...     weights_path="east_model.onnx",
        ...     use_onnx=True,
        ...     device="cuda"
        ... )
        >>> result = detector.predict("image.jpg")

        See Also
        --------
        EAST.__init__ : Initialize EAST detector with ONNX support using ``use_onnx=True``.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for exporting models. "
                "Install with: pip install manuscript-ocr[dev]"
            )

        class EASTWrapper(torch.nn.Module):
            def __init__(self, east_model):
                super().__init__()
                self.east = east_model

            def forward(self, x):
                output = self.east(x)
                return output["score"], output["geometry"]

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        print(f"Loading checkpoint from {weights_path}...")
        checkpoint = torch.load(str(weights_path), map_location="cpu")

        # Auto-detect backbone if not specified
        if backbone_name is None:
            print("Auto-detecting backbone architecture...")
            # Detect by checking layer3 parameters count
            # ResNet50: layer3 has 6 bottleneck blocks (layer3.0 - layer3.5)
            # ResNet101: layer3 has 23 bottleneck blocks (layer3.0 - layer3.22)

            # Count backbone.extractor.* parameters starting with "layer3"
            layer3_keys = [
                k for k in checkpoint.keys() if "backbone.extractor.layer3" in k
            ]

            # Check if layer3.10 exists (only in resnet101)
            has_layer3_10 = any("layer3.10" in k for k in layer3_keys)

            if has_layer3_10:
                detected_backbone = "resnet101"
            else:
                detected_backbone = "resnet50"

            print(f"   Detected {len(layer3_keys)} layer3 parameters")
            print(f"   Auto-detected backbone: {detected_backbone}")
            backbone_name = detected_backbone
        else:
            print(f"Using specified backbone: {backbone_name}")

            # Verify backbone matches checkpoint
            print("Verifying backbone matches checkpoint...")
            layer3_keys = [
                k for k in checkpoint.keys() if "backbone.extractor.layer3" in k
            ]
            has_layer3_10 = any("layer3.10" in k for k in layer3_keys)

            expected_backbone = "resnet101" if has_layer3_10 else "resnet50"

            if expected_backbone != backbone_name:
                raise ValueError(
                    f"Backbone mismatch! Checkpoint is {expected_backbone}, "
                    f"but you specified {backbone_name}. "
                    f"Either use backbone_name='{expected_backbone}' or set backbone_name=None for auto-detection."
                )
            print(f"   [OK] Backbone matches checkpoint ({backbone_name})")

        print(f"\nLoading PyTorch model...")
        east_model = EASTModel(
            backbone_name=backbone_name,
            pretrained_backbone=False,
            pretrained_model_path=str(weights_path),
        )
        east_model.eval()

        model = EASTWrapper(east_model)
        model.eval()

        dummy_input = torch.randn(1, 3, input_size, input_size)

        print(f"Model architecture: {model.__class__.__name__}")
        print(f"Input shape: {dummy_input.shape}")

        with torch.no_grad():
            score_map, geo_map = model(dummy_input)

        print(f"Output shapes:")
        print(f"  - score_map: {score_map.shape}")
        print(f"  - geo_map: {geo_map.shape}")

        print(f"\nExporting to ONNX (opset {opset_version})...")
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["score_map", "geo_map"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "score_map": {0: "batch_size", 2: "height", 3: "width"},
                "geo_map": {0: "batch_size", 2: "height", 3: "width"},
            },
            verbose=False,
        )

        print(f"[OK] ONNX model saved to: {output_path}")

        import onnx
        import onnxsim

        print("\nVerifying ONNX model...")
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX model is valid")

        if simplify:
            print("\nSimplifying ONNX model...")
            model_simplified, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(model_simplified, str(output_path))
                print("[OK] ONNX model simplified")
            else:
                print("[WARNING] Simplification failed, using original model")

        # Test ONNX inference and compare with PyTorch
        try:
            import onnxruntime as ort

            print(f"\nTesting ONNX inference...")
            session = ort.InferenceSession(
                str(output_path), providers=["CPUExecutionProvider"]
            )

            ort_inputs = {"input": dummy_input.numpy()}
            ort_outputs = session.run(None, ort_inputs)

            print(f"[OK] ONNX inference works!")
            print(f"  ONNX score_map shape: {ort_outputs[0].shape}")
            print(f"  ONNX geo_map shape: {ort_outputs[1].shape}")

            # Compare with PyTorch
            print(f"\nComparing ONNX vs PyTorch outputs...")
            torch_score = score_map.numpy()
            torch_geo = geo_map.numpy()
            onnx_score = ort_outputs[0]
            onnx_geo = ort_outputs[1]

            score_max_diff = abs(torch_score - onnx_score).max()
            geo_max_diff = abs(torch_geo - onnx_geo).max()

            print(f"  Max difference in score_map: {score_max_diff:.6f}")
            print(f"  Max difference in geo_map: {geo_max_diff:.6f}")

            if score_max_diff < 1e-4 and geo_max_diff < 1e-4:
                print(f"  [OK] Outputs match!")
            elif score_max_diff < 1e-3 and geo_max_diff < 1e-3:
                print(f"  [WARNING] Small differences detected (acceptable)")
            else:
                print(f"  [ERROR] Outputs differ significantly!")
                print(f"  This may indicate a problem with the export.")

        except ImportError:
            print(f"[WARNING] onnxruntime not installed, skipping inference test")
            print(f"  Install with: pip install onnxruntime")
        except Exception as e:
            print(f"[WARNING] ONNX inference test failed: {e}")

        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"\n[OK] Export complete! Model size: {file_size_mb:.1f} MB")
        print(f"\n=== Summary ===")
        print(f"Backbone: {backbone_name}")
        print(f"Input shape: [batch_size, 3, {input_size}, {input_size}]")
        print(f"Output shapes:")
        print(f"  - score_map: [batch_size, 1, H, W]")
        print(f"  - geo_map: [batch_size, 8, H, W]")
