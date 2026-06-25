import json
import os
import warnings
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
import skimage.draw
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .._east.dataset import EASTDataset


def _iter_annotation_polygons(ann: dict) -> Iterable[np.ndarray]:
    seg = ann.get("segmentation")
    if seg is None or isinstance(seg, dict):
        return []
    if not isinstance(seg, (list, tuple)) or len(seg) == 0:
        return []

    seg_parts = seg if isinstance(seg[0], (list, tuple, np.ndarray)) else [seg]
    polygons = []
    for part in seg_parts:
        raw = np.asarray(part, dtype=np.float32)
        if raw.ndim == 1:
            if raw.size < 8 or raw.size % 2 != 0:
                continue
            pts = raw.reshape(-1, 2)
        elif raw.ndim == 2 and raw.shape[1] == 2:
            pts = raw
        else:
            continue
        if pts.shape[0] >= 4:
            polygons.append(pts.astype(np.float32))
    return polygons


def _draw_gaussian(target: np.ndarray, center_x: float, center_y: float, sigma: float):
    h, w = target.shape
    radius = max(1, int(3 * sigma))
    x0 = max(0, int(center_x) - radius)
    x1 = min(w, int(center_x) + radius + 1)
    y0 = max(0, int(center_y) - radius)
    y1 = min(h, int(center_y) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    gaussian = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma**2))
    target[y0:y1, x0:x1] = np.maximum(target[y0:y1, x0:x1], gaussian)


class EASTV2Dataset(EASTDataset):
    """
    COCO polygon dataset for EASTV2 instance targets.

    Returns score_map, boundary_map, center_map and instance_map at
    target_size * map_scale resolution.
    """

    def __init__(
        self,
        images_folder,
        coco_annotation_file,
        target_size: int = 512,
        map_scale: float = 0.25,
        boundary_width: int = 2,
        center_sigma_ratio: float = 0.15,
        flip_prob: float = 0.0,
        rotate_prob: float = 0.0,
        rotate_deg: float = 2.0,
        small_rotate_prob: Optional[float] = None,
        small_rotate_deg: Optional[float] = None,
        color_jitter=(0.1, 0.1, 0.1, 0.05),
        transform=None,
        dataset_name: Optional[str] = None,
        **kwargs,
    ):
        self.images_folder = images_folder
        self.target_size = int(target_size)
        self.map_scale = float(map_scale)
        self.boundary_width = int(boundary_width)
        self.center_sigma_ratio = float(center_sigma_ratio)
        self.flip_prob = float(flip_prob)
        self.rotate_prob = float(rotate_prob if small_rotate_prob is None else small_rotate_prob)
        self.rotate_deg = float(rotate_deg if small_rotate_deg is None else small_rotate_deg)
        self.extra_augmentation_config = dict(kwargs)
        self.vflip_prob = float(kwargs.get("vflip_prob", 0.0))
        self.small_rotate_prob = self.rotate_prob
        self.small_rotate_deg = self.rotate_deg
        self.perspective_prob = float(kwargs.get("perspective_prob", 0.1))
        self.perspective_scale = float(kwargs.get("perspective_scale", 0.015))
        self.shear_prob = float(kwargs.get("shear_prob", 0.15))
        self.shear_deg = float(kwargs.get("shear_deg", 5.0))
        self.random_crop_prob = float(kwargs.get("random_crop_prob", 0.2))
        self.random_crop_scale = tuple(kwargs.get("random_crop_scale", (0.7, 1.0)))
        self.blur_prob = float(kwargs.get("blur_prob", 0.1))
        self.blur_ksize_range = tuple(kwargs.get("blur_ksize_range", (3, 5)))
        self.motion_blur_prob = float(kwargs.get("motion_blur_prob", 0.1))
        self.motion_blur_ksize_range = tuple(kwargs.get("motion_blur_ksize_range", (3, 9)))
        self.noise_prob = float(kwargs.get("noise_prob", 0.1))
        self.noise_std = float(kwargs.get("noise_std", 0.008))
        self.salt_pepper_prob = float(kwargs.get("salt_pepper_prob", 0.0005))
        self.jpeg_prob = float(kwargs.get("jpeg_prob", 0.1))
        self.jpeg_quality_range = tuple(kwargs.get("jpeg_quality_range", (75, 95)))
        self.shading_prob = float(kwargs.get("shading_prob", 0.1))
        self.shading_strength = float(kwargs.get("shading_strength", 0.1))
        self.gamma_prob = float(kwargs.get("gamma_prob", 0.2))
        self.gamma_range = tuple(kwargs.get("gamma_range", (0.95, 1.05)))
        self.downscale_prob = float(kwargs.get("downscale_prob", 0.1))
        self.downscale_range = tuple(kwargs.get("downscale_range", (0.7, 0.95)))
        self.negative_prob = float(kwargs.get("negative_prob", 0.05))
        self.hsv_prob = float(kwargs.get("hsv_prob", 0.15))
        self.hsv_h = float(kwargs.get("hsv_h", 0.015))
        self.hsv_s = float(kwargs.get("hsv_s", 0.3))
        self.hsv_v = float(kwargs.get("hsv_v", 0.2))
        self.cutout_prob = float(kwargs.get("cutout_prob", 0.15))
        self.cutout_num_holes = int(kwargs.get("cutout_num_holes", 2))
        self.cutout_hole_size_range = tuple(kwargs.get("cutout_hole_size_range", (0.05, 0.15)))
        self.elastic_prob = float(kwargs.get("elastic_prob", 0.1))
        self.elastic_alpha = float(kwargs.get("elastic_alpha", 20.0))
        self.elastic_sigma = float(kwargs.get("elastic_sigma", 4.0))
        self.fog_prob = float(kwargs.get("fog_prob", 0.1))
        self.fog_strength_range = tuple(kwargs.get("fog_strength_range", (0.1, 0.4)))
        self.fog_direction = str(kwargs.get("fog_direction", "random"))
        self.dataset_name = (
            dataset_name if dataset_name is not None else Path(images_folder).stem
        )

        if not (0.0 < self.map_scale <= 0.5):
            raise ValueError("map_scale must be in (0, 0.5]")
        if self.boundary_width < 1:
            raise ValueError("boundary_width must be >= 1")

        if transform is None:
            if color_jitter:
                jitter = transforms.ColorJitter(*color_jitter)
                self.transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        jitter,
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
                        ),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
                        ),
                    ]
                )
        else:
            self.transform = transform

        with open(coco_annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.images_info = {img["id"]: img for img in data["images"]}
        self.image_ids = list(self.images_info.keys())
        self.annots = {}
        for ann in data["annotations"]:
            self.annots.setdefault(ann["image_id"], []).append(ann)
        self._filter_invalid()

    def _filter_invalid(self):
        invalid = []
        for image_id in self.image_ids:
            if not any(_iter_annotation_polygons(ann) for ann in self.annots.get(image_id, [])):
                invalid.append(image_id)
        for image_id in invalid:
            self.image_ids.remove(image_id)
            self.annots.pop(image_id, None)
        if invalid:
            warnings.warn(
                f"EASTV2Dataset: found {len(invalid)} images without valid masks",
                UserWarning,
            )

    def __len__(self):
        return len(self.image_ids)

    def _load_image_and_polygons(self, idx):
        image_id = self.image_ids[idx]
        info = self.images_info[image_id]
        path = os.path.join(self.images_folder, info["file_name"])
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.target_size, self.target_size))

        scale_x = self.target_size / info["width"]
        scale_y = self.target_size / info["height"]
        polygons = []
        for ann in self.annots.get(image_id, []):
            for pts in _iter_annotation_polygons(ann):
                poly = pts.copy()
                poly[:, 0] *= scale_x
                poly[:, 1] *= scale_y
                polygons.append(poly.astype(np.float32))
        return img, polygons

    def _apply_augments(self, img, polygons):
        if polygons and len({poly.shape[0] for poly in polygons}) == 1:
            img, polygons = self._apply_geometric_augments(img, polygons)
            img = self._apply_photometric_augments(img)
            return self._ensure_uint8(img), polygons

        if self.flip_prob > 0 and np.random.rand() < self.flip_prob:
            h, w = img.shape[:2]
            img = np.fliplr(img).copy()
            for poly in polygons:
                poly[:, 0] = (w - 1) - poly[:, 0]

        if self.rotate_deg > 0 and self.rotate_prob > 0 and np.random.rand() < self.rotate_prob:
            h, w = img.shape[:2]
            angle = np.random.uniform(-self.rotate_deg, self.rotate_deg)
            matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            img = cv2.warpAffine(
                img,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
            for poly in polygons:
                ones = np.ones((poly.shape[0], 1), dtype=np.float32)
                poly[:] = np.concatenate([poly, ones], axis=1) @ matrix.T

        for poly in polygons:
            poly[:, 0] = np.clip(poly[:, 0], 0, self.target_size - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, self.target_size - 1)
        return img, polygons

    def __getitem__(self, idx):
        img, polygons = self._load_image_and_polygons(idx)
        img, polygons = self._apply_augments(img, polygons)
        target = self.compute_instance_maps(polygons)
        target["polygons"] = [
            torch.from_numpy(poly.astype(np.float32)) for poly in polygons
        ]
        return self.transform(img), target

    def compute_instance_maps(self, polygons: List[np.ndarray]):
        out_h = int(self.target_size * self.map_scale)
        out_w = int(self.target_size * self.map_scale)
        score = np.zeros((out_h, out_w), dtype=np.float32)
        boundary = np.zeros((out_h, out_w), dtype=np.float32)
        center = np.zeros((out_h, out_w), dtype=np.float32)
        instance = np.zeros((out_h, out_w), dtype=np.int32)

        kernel_size = self.boundary_width * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        for idx, poly in enumerate(polygons, start=1):
            coords = poly.astype(np.float32) * self.map_scale
            rr, cc = skimage.draw.polygon(coords[:, 1], coords[:, 0], shape=(out_h, out_w))
            if len(rr) == 0:
                continue

            inst_mask = np.zeros((out_h, out_w), dtype=np.uint8)
            inst_mask[rr, cc] = 1
            score[rr, cc] = 1.0
            instance[rr, cc] = idx

            dilated = cv2.dilate(inst_mask, kernel, iterations=1)
            eroded = cv2.erode(inst_mask, kernel, iterations=1)
            boundary = np.maximum(boundary, (dilated - eroded).astype(np.float32))

            m = cv2.moments(inst_mask, binaryImage=True)
            if m["m00"] > 0:
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
            else:
                cx = float(coords[:, 0].mean())
                cy = float(coords[:, 1].mean())
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0)
            sigma = max(1.0, min(x1 - x0, y1 - y0) * self.center_sigma_ratio)
            _draw_gaussian(center, cx, cy, sigma)

        return {
            "score_map": torch.from_numpy(score).unsqueeze(0),
            "boundary_map": torch.from_numpy(np.clip(boundary, 0.0, 1.0)).unsqueeze(0),
            "center_map": torch.from_numpy(np.clip(center, 0.0, 1.0)).unsqueeze(0),
            "instance_map": torch.from_numpy(instance),
        }
