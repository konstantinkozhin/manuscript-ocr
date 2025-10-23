import cv2
import numpy as np
import torch
from torchvision import transforms
from pathlib import Path
from typing import Union, Optional, List, Tuple
import time

from .east import TextDetectionFCN
from .utils import decode_quads_from_maps, draw_quads, read_image, expand_boxes
from .lanms import locality_aware_nms
from .train_utils import train
from .._types import Word, Block, Page
import os

import gdown


class EASTInfer:
    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        target_size: int = 1280,
        expand_ratio_w: float = 0.9,
        expand_ratio_h: float = 0.9,
        score_thresh: float = 0.6,
        iou_threshold: float = 0.2,
        score_geo_scale: float = 0.25,
        quantization: int = 2,
        axis_aligned_output: bool = True,
        remove_area_anomalies: bool = True,
        anomaly_sigma_threshold: float = 5.0,
        anomaly_min_box_count: int = 30,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if weights_path is None:
            url = (
                "https://github.com/konstantinkozhin/manuscript-ocr"
                "/releases/download/v0.1.0/east_quad_23_05.pth"
            )
            out = os.path.expanduser("~/.east_quad_23_05.pth")
            if not os.path.exists(out):
                print(f"Downloading EAST weights from {url} …")
                gdown.download(url, out, quiet=False)
            weights_path = out

        self.model = TextDetectionFCN(
            pretrained_backbone=False,
            pretrained_model_path=str(weights_path),
        ).to(self.device)
        self.model.eval()

        self.target_size = target_size
        self.score_geo_scale = score_geo_scale
        self.expand_ratio_w = expand_ratio_w
        self.expand_ratio_h = expand_ratio_h
        self.score_thresh = score_thresh
        self.iou_threshold = iou_threshold
        self.quantization = quantization
        self.axis_aligned_output = axis_aligned_output
        self.remove_area_anomalies = remove_area_anomalies
        self.anomaly_sigma_threshold = anomaly_sigma_threshold
        self.anomaly_min_box_count = anomaly_min_box_count

        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

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

    def infer(
        self,
        img_or_path: Union[str, Path, np.ndarray],
        vis: bool = False,
        profile: bool = False,
    ) -> Union[Page, Tuple[Page, np.ndarray, np.ndarray]]:
        """
        :param img_or_path: путь или RGB ndarray
        :param vis: если True, возвращает также изображение с боксами и score map
        :param profile: если True, выводит время выполнения этапов
        :return: Page или (Page, vis_image, score_map)
        """
        # 1) Read & RGB
        img = read_image(img_or_path)

        # 2) Resize + ToTensor + Normalize
        resized = cv2.resize(img, (self.target_size, self.target_size))
        img_t = self.tf(resized).to(self.device)

        # 3) Forward
        t0 = time.time()
        with torch.no_grad():
            out = self.model(img_t.unsqueeze(0))

            score_map = out["score"][0].cpu().numpy().squeeze(0)
            geo_map = out["geometry"][0].cpu().numpy()
        if profile:
            print(f"  Model inference: {time.time() - t0:.3f}s")

        # 4) Decode raw quads (с квантизацией на уровне точек)
        t0 = time.time()
        final_quads = decode_quads_from_maps(
            score_map=score_map,
            geo_map=geo_map.transpose(1, 2, 0),
            score_thresh=self.score_thresh,
            scale=1.0 / self.score_geo_scale,
            quantization=self.quantization,
            profile=profile,
        )
        if profile:
            print(f"  Decode boxes: {time.time() - t0:.3f}s")

        # 5) Apply NMS
        t0 = time.time()
        final_quads_nms = locality_aware_nms(
            final_quads, iou_threshold=self.iou_threshold
        )
        if profile:
            print(f"  NMS: {time.time() - t0:.3f}s")
            print(f"    Boxes after NMS: {len(final_quads_nms)}")

        # 6) Expand (inverse shrink)
        final_quads_nms_expanded = expand_boxes(
            final_quads_nms, expand_w=self.expand_ratio_w, expand_h=self.expand_ratio_h
        )

        # 7) Scale coordinates back to original image size
        orig_h, orig_w = img.shape[:2]
        scaled_quads = self._scale_boxes_to_original(
            final_quads_nms_expanded, (orig_h, orig_w)
        )

        processed_quads = self._remove_fully_contained_boxes(scaled_quads)
        processed_quads = self._remove_area_anomalies(processed_quads)
        output_quads = (
            self._convert_to_axis_aligned(processed_quads)
            if self.axis_aligned_output
            else processed_quads
        )

        # 8) Build Page with scaled coordinates (after NMS & expand)
        words: List[Word] = []
        for quad in output_quads:
            pts = quad[:8].reshape(4, 2)
            score = float(quad[8])
            words.append(Word(polygon=pts.tolist(), detection_confidence=score))
        page = Page(blocks=[Block(words=words)])

        # 10) Optional visualization
        if vis:
            vis_img = draw_quads(
                img, output_quads if self.axis_aligned_output else processed_quads
            )
            score_map_resized = cv2.resize(
                score_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )
            return page, vis_img, score_map_resized

        return page, None, None
