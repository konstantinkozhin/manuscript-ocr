import cv2
import numpy as np
import torch
from torchvision import transforms
from pathlib import Path
from typing import Union, Optional, List, Tuple

from .east import TextDetectionFCN
from .utils import decode_boxes_from_maps, draw_boxes
from ..types import Word, Block, Page
import os

import gdown


class EASTInfer:
    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        target_size: int = 1024,
        score_geo_scale: float = 0.25,
        shrink_ratio: float = 0.5,
        score_thresh: float = 0.9,
        iou_threshold: float = 0.2,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if weights_path is None:
            url = (
                "https://github.com/konstantinkozhin/manuscript-ocr"
                "/releases/download/v0.1.0/east_quad_14_05.pth"
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
        self, img_or_path: Union[str, Path, np.ndarray], vis: bool = False
    ) -> Union[Page, Tuple[Page, np.ndarray]]:
        """
        :param img_or_path: путь или RGB ndarray
        :param vis: если True, возвращает также изображение с боксами
        :return: Page или (Page, vis_image)
        """
        # 1) Read & RGB
        if isinstance(img_or_path, (str, Path)):
            img = cv2.imread(str(img_or_path))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {img_or_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img_or_path, np.ndarray):
            img = img_or_path
        else:
            raise TypeError(f"Unsupported type {type(img_or_path)}")

        # 2) Resize + ToTensor + Normalize
        resized = cv2.resize(img, (self.target_size, self.target_size))
        img_t = self.tf(resized).to(self.device)

        # 3) Forward
        with torch.no_grad():
            out = self.model(img_t.unsqueeze(0))

        # 4) Extract maps
        score_map = out["score"][0].cpu().numpy().squeeze(0)
        geo_map = out["geometry"][0].cpu().numpy().transpose(1, 2, 0)

        # 5) Decode quads
        quads9 = decode_boxes_from_maps(
            score_map=score_map,
            geo_map=geo_map,
            score_thresh=self.score_thresh,
            scale=1.0 / self.score_geo_scale,
            iou_threshold=self.iou_threshold,
            expand_ratio=self.shrink_ratio,
        )

        # 6) Build Page
        words: List[Word] = []
        for quad in quads9:
            pts = quad[:8].reshape(4, 2).tolist()
            score = quad[8]
            words.append(Word(polygon=pts, score=score))
        page = Page(blocks=[Block(words=words)])

        # 7) Optional visualization
        if vis:
            vis_img = draw_boxes(resized, quads9)
            return page, vis_img

        return page
