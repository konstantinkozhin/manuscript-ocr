import cv2
import numpy as np
import torch
from torchvision import transforms
from pathlib import Path
from typing import Union, Optional, List, Tuple
from PIL import Image
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
        target_size: int = 1536,
        expand_ratio_w: float = 0.8,
        expand_ratio_h: float = 0.8,
        score_thresh: float = 0.99,
        iou_threshold: float = 0.2,
        score_geo_scale: float = 0.25,
        quantization: int = 2,
        use_tta: bool = False,
        tta_merge_mode: str = "mean",
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
        self.use_tta = use_tta
        self.tta_merge_mode = tta_merge_mode  # "mean", "max", "min"

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

    def _merge_maps(self, map1: np.ndarray, map2: np.ndarray) -> np.ndarray:
        """Объединяет две карты согласно tta_merge_mode"""
        if self.tta_merge_mode == "mean":
            return (map1 + map2) / 2.0
        elif self.tta_merge_mode == "max":
            return np.maximum(map1, map2)
        elif self.tta_merge_mode == "min":
            return np.minimum(map1, map2)
        else:
            raise ValueError(f"Unknown tta_merge_mode: {self.tta_merge_mode}")

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

            if self.use_tta:
                # TTA: переворачиваем изображение горизонтально
                img_t_flipped = torch.flip(img_t, dims=[2])  # flip по ширине
                out_flipped = self.model(img_t_flipped.unsqueeze(0))

                # Разворачиваем карты обратно
                score_flipped = (
                    torch.flip(out_flipped["score"], dims=[3]).cpu().numpy()[0, 0]
                )
                geo_flipped = (
                    torch.flip(out_flipped["geometry"], dims=[3]).cpu().numpy()[0]
                )

                geo_flipped_corrected = geo_flipped.copy()

                # Переставляем вершины: 0↔1, 2↔3
                v0 = geo_flipped[0:2].copy()
                v1 = geo_flipped[2:4].copy()
                v2 = geo_flipped[4:6].copy()
                v3 = geo_flipped[6:8].copy()

                geo_flipped_corrected[0:2] = v1  # v0 <- v1
                geo_flipped_corrected[2:4] = v0  # v1 <- v0
                geo_flipped_corrected[4:6] = v3  # v2 <- v3
                geo_flipped_corrected[6:8] = v2  # v3 <- v2

                # Инвертируем dx (каналы 0, 2, 4, 6)
                geo_flipped_corrected[0::2] = -geo_flipped_corrected[0::2]

                # Объединяем карты
                score_orig = out["score"][0].cpu().numpy()[0]
                geo_orig = out["geometry"][0].cpu().numpy()

                score_map = self._merge_maps(score_orig, score_flipped)
                geo_map = geo_orig
            else:
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

        # 8) Build Page with scaled coordinates (after NMS & expand)
        words: List[Word] = []
        for quad in scaled_quads:
            pts = quad[:8].reshape(4, 2)
            score = float(quad[8])
            words.append(Word(polygon=pts.tolist(), detection_confidence=score))
        page = Page(blocks=[Block(words=words)])

        # 10) Optional visualization
        if vis:
            vis_img = draw_quads(img, scaled_quads)
            # Масштабируем score map до размера исходного изображения
            score_map_resized = cv2.resize(
                score_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )
            return page, vis_img, score_map_resized

        return page, None, None
