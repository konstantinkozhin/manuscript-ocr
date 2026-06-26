import json
import os
import warnings
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from shapely.geometry import Polygon
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


def _perimeter(poly: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(poly - np.roll(poly, -1, axis=0), axis=1)))


def shrink_polygon(poly: np.ndarray, rate: float, max_shr: int = 20) -> np.ndarray:
    """PSENet shrink formula using Shapely negative buffer."""
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    polygon = Polygon(poly)
    area = float(abs(polygon.area))
    peri = _perimeter(poly)
    if area <= 1e-6 or peri <= 1e-6:
        return poly
    rate_sq = float(rate) * float(rate)
    offset = min(int(area * (1.0 - rate_sq) / (peri + 0.001) + 0.5), int(max_shr))
    if offset <= 0:
        return poly
    shrinked = polygon.buffer(-offset, join_style=1)
    if shrinked.is_empty:
        return poly
    if shrinked.geom_type == "MultiPolygon":
        shrinked = max(shrinked.geoms, key=lambda geom: geom.area)
    coords = np.asarray(shrinked.exterior.coords[:-1], dtype=np.float32)
    if coords.shape[0] <= 2:
        return poly
    return coords


class PSEDataset(EASTDataset):
    """COCO polygon dataset producing PSENet text + progressive kernel maps."""

    def __init__(
        self,
        images_folder,
        coco_annotation_file,
        target_size: int = 512,
        map_scale: float = 0.25,
        kernel_num: int = 7,
        min_scale: float = 0.5,
        max_shrink: int = 20,
        flip_prob: float = 0.01,
        vflip_prob: float = 0.0,
        small_rotate_prob: float = 0.2,
        small_rotate_deg: float = 2.0,
        perspective_prob: float = 0.1,
        perspective_scale: float = 0.015,
        shear_prob: float = 0.15,
        shear_deg: float = 5.0,
        random_crop_prob: float = 0.2,
        random_crop_scale=(0.7, 1.0),
        blur_prob: float = 0.1,
        blur_ksize_range=(3, 5),
        motion_blur_prob: float = 0.1,
        motion_blur_ksize_range=(3, 9),
        noise_prob: float = 0.1,
        noise_std: float = 0.008,
        salt_pepper_prob: float = 0.0005,
        jpeg_prob: float = 0.1,
        jpeg_quality_range=(75, 95),
        shading_prob: float = 0.1,
        shading_strength: float = 0.1,
        gamma_prob: float = 0.2,
        gamma_range=(0.95, 1.05),
        downscale_prob: float = 0.1,
        downscale_range=(0.7, 0.95),
        negative_prob: float = 0.05,
        color_jitter=(0.1, 0.1, 0.1, 0.05),
        hsv_prob: float = 0.15,
        hsv_h: float = 0.015,
        hsv_s: float = 0.3,
        hsv_v: float = 0.2,
        cutout_prob: float = 0.15,
        cutout_num_holes: int = 2,
        cutout_hole_size_range=(0.05, 0.15),
        elastic_prob: float = 0.1,
        elastic_alpha: float = 20.0,
        elastic_sigma: float = 4.0,
        fog_prob: float = 0.1,
        fog_strength_range=(0.1, 0.4),
        fog_direction: str = "random",
        transform=None,
        dataset_name: Optional[str] = None,
        **kwargs,
    ):
        del kwargs
        self.images_folder = images_folder
        self.target_size = int(target_size)
        self.map_scale = float(map_scale)
        self.kernel_num = int(kernel_num)
        self.min_scale = float(min_scale)
        self.max_shrink = int(max_shrink)
        self.flip_prob = float(flip_prob)
        self.vflip_prob = float(vflip_prob)
        self.small_rotate_prob = float(small_rotate_prob)
        self.small_rotate_deg = float(small_rotate_deg)
        self.perspective_prob = float(perspective_prob)
        self.perspective_scale = float(perspective_scale)
        self.shear_prob = float(shear_prob)
        self.shear_deg = float(shear_deg)
        self.random_crop_prob = float(random_crop_prob)
        self.random_crop_scale = tuple(random_crop_scale)
        self.blur_prob = float(blur_prob)
        self.blur_ksize_range = tuple(blur_ksize_range)
        self.motion_blur_prob = float(motion_blur_prob)
        self.motion_blur_ksize_range = tuple(motion_blur_ksize_range)
        self.noise_prob = float(noise_prob)
        self.noise_std = float(noise_std)
        self.salt_pepper_prob = float(salt_pepper_prob)
        self.jpeg_prob = float(jpeg_prob)
        self.jpeg_quality_range = tuple(jpeg_quality_range)
        self.shading_prob = float(shading_prob)
        self.shading_strength = float(shading_strength)
        self.gamma_prob = float(gamma_prob)
        self.gamma_range = tuple(gamma_range)
        self.downscale_prob = float(downscale_prob)
        self.downscale_range = tuple(downscale_range)
        self.negative_prob = float(negative_prob)
        self.hsv_prob = float(hsv_prob)
        self.hsv_h = float(hsv_h)
        self.hsv_s = float(hsv_s)
        self.hsv_v = float(hsv_v)
        self.cutout_prob = float(cutout_prob)
        self.cutout_num_holes = int(cutout_num_holes)
        self.cutout_hole_size_range = tuple(cutout_hole_size_range)
        self.elastic_prob = float(elastic_prob)
        self.elastic_alpha = float(elastic_alpha)
        self.elastic_sigma = float(elastic_sigma)
        self.fog_prob = float(fog_prob)
        self.fog_strength_range = tuple(fog_strength_range)
        self.fog_direction = str(fog_direction)
        self.dataset_name = dataset_name if dataset_name is not None else Path(images_folder).stem

        if self.kernel_num < 2:
            raise ValueError("kernel_num must be >= 2")
        if not (0.0 < self.min_scale < 1.0):
            raise ValueError("min_scale must be in (0, 1)")
        if not (0.0 < self.map_scale <= 0.5):
            raise ValueError("map_scale must be in (0, 0.5]")

        if transform is None:
            steps = [transforms.ToPILImage()]
            if color_jitter:
                steps.append(transforms.ColorJitter(*color_jitter))
            steps.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]
            )
            self.transform = transforms.Compose(steps)
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
                f"PSEDataset: found {len(invalid)} images without valid polygons",
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

    def __getitem__(self, idx):
        img, polygons = self._load_image_and_polygons(idx)
        if polygons and len({poly.shape[0] for poly in polygons}) == 1:
            img, polygons = self._apply_geometric_augments(img, polygons)
            img = self._apply_photometric_augments(img)
            img = self._ensure_uint8(img)
        target = self.compute_pse_maps(polygons)
        target["polygons"] = [torch.from_numpy(poly.astype(np.float32)) for poly in polygons]
        return self.transform(img), target

    def compute_pse_maps(self, polygons: List[np.ndarray]):
        out_h = int(self.target_size * self.map_scale)
        out_w = int(self.target_size * self.map_scale)
        text = np.zeros((out_h, out_w), dtype=np.uint8)
        instance = np.zeros((out_h, out_w), dtype=np.int32)
        kernels = np.zeros((self.kernel_num - 1, out_h, out_w), dtype=np.uint8)
        scaled_polys = [poly.astype(np.float32) * self.map_scale for poly in polygons]

        for label_id, poly in enumerate(scaled_polys, start=1):
            pts = np.round(poly).astype(np.int32)
            cv2.drawContours(text, [pts], -1, 1, -1)
            cv2.drawContours(instance, [pts], -1, label_id, -1)

        for i in range(1, self.kernel_num):
            rate = 1.0 - (1.0 - self.min_scale) / (self.kernel_num - 1) * i
            for poly in scaled_polys:
                shrunk = shrink_polygon(poly, rate, max_shr=self.max_shrink)
                pts = np.round(shrunk).astype(np.int32)
                cv2.drawContours(kernels[i - 1], [pts], -1, 1, -1)

        return {
            "text_map": torch.from_numpy(text.astype(np.float32)),
            "kernel_maps": torch.from_numpy(kernels.astype(np.float32)),
            "training_mask": torch.ones(out_h, out_w, dtype=torch.float32),
            "instance_map": torch.from_numpy(instance),
        }
