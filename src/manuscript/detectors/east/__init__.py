import cv2
import numpy as np
import torch
from torchvision import transforms
from pathlib import Path
from typing import Union, Optional, List, Tuple
from PIL import Image
import time

from .east import TextDetectionFCN
from .utils import decode_boxes_from_maps, expand_boxes, draw_quads
from .train_utils import train
from .._types import Word, Block, Page
import os

import gdown


class EASTInfer:
    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        target_size: int = 1024,
        shrink_ratio: float = 0.3,
        score_thresh: float = 0.9,
        iou_threshold: float = 0.2,
        score_geo_scale: float = 0.25,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if weights_path is None:
            url = (
                "https://github.com/konstantinkozhin/manuscript-ocr"
                "/releases/download/v0.1.0/east_quad_23_05.pth"
            )
            out = os.path.expanduser("~/.east_weights.pth")
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
        self.shrink_ratio = shrink_ratio
        self.score_thresh = score_thresh
        self.iou_threshold = iou_threshold

        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def infer(
        self, img_or_path: Union[str, Path, np.ndarray], vis: bool = False, profile: bool = False
    ) -> Union[Page, Tuple[Page, np.ndarray]]:
        """
        :param img_or_path: путь или RGB ndarray
        :param vis: если True, возвращает также изображение с боксами
        :param profile: если True, выводит время выполнения этапов
        :return: Page или (Page, vis_image)
        """
        start_time = time.time()
        
        # 1) Read & RGB
        t0 = time.time()
        if isinstance(img_or_path, (str, Path)):
            img = cv2.imread(str(img_or_path))
            if img is None:
                try:
                    pil_img = Image.open(str(img_or_path))
                    img = np.array(pil_img.convert('RGB'))
                except Exception as e:
                    raise FileNotFoundError(f"Cannot read image with cv2 or PIL: {img_or_path}. Error: {e}")
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img_or_path, np.ndarray):
            img = img_or_path
        else:
            raise TypeError(f"Unsupported type {type(img_or_path)}")
        if profile: print(f"  Image loading: {time.time() - t0:.3f}s")

        # 2) Resize + ToTensor + Normalize
        t0 = time.time()
        resized = cv2.resize(img, (self.target_size, self.target_size))
        img_t = self.tf(resized).to(self.device)
        if profile: print(f"  Preprocessing: {time.time() - t0:.3f}s")

        # 3) Forward
        t0 = time.time()
        with torch.no_grad():
            out = self.model(img_t.unsqueeze(0))
        if profile: print(f"  Model inference: {time.time() - t0:.3f}s")

        # 4) Extract maps
        t0 = time.time()
        score_map = out["score"][0].cpu().numpy().squeeze(0)
        geo_map = out["geometry"][0].cpu().numpy().transpose(1, 2, 0)
        if profile: print(f"  Extract maps: {time.time() - t0:.3f}s")

        # 5) Decode raw quads (до NMS и expand)
        t0 = time.time()
        raw_quads = decode_boxes_from_maps(
            score_map=score_map,
            geo_map=geo_map,
            score_thresh=self.score_thresh,
            scale=1.0 / self.score_geo_scale,
            profile=profile,
        )
        if profile: print(f"  Decode boxes: {time.time() - t0:.3f}s")

        # 6) Expand (inverse shrink)
        t0 = time.time()
        quads9 = expand_boxes(raw_quads, expand_ratio=self.shrink_ratio)
        if profile: print(f"  Expand boxes: {time.time() - t0:.3f}s")

        # 7) Scale coordinates back to original image size
        t0 = time.time()
        orig_h, orig_w = img.shape[:2]
        scale_x = orig_w / self.target_size
        scale_y = orig_h / self.target_size
        
        # 8) Build Page with scaled coordinates
        words: List[Word] = []
        for quad in quads9:
            pts = quad[:8].reshape(4, 2)
            # Scale coordinates back to original image size
            pts[:, 0] *= scale_x  # x coordinates
            pts[:, 1] *= scale_y  # y coordinates
            score = quad[8]
            words.append(Word(polygon=pts.tolist(), detection_confidence=score))
        page = Page(blocks=[Block(words=words)])
        if profile: print(f"  Build page: {time.time() - t0:.3f}s")

        # 9) Optional visualization
        vis_img = None
        if vis:
            t0 = time.time()
            # For visualization, use coordinates on resized image
            vis_img = draw_quads(img, quads9)
            if profile: print(f"  Draw quads: {time.time() - t0:.3f}s")

        if profile: print(f"EAST total: {time.time() - start_time:.3f}s")
        
        if vis:
            return page, vis_img
        return page
