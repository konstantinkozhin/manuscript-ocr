import os
import json
import warnings
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
import skimage.draw
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def order_vertices_clockwise(poly):
    poly = np.array(poly).reshape(-1, 2)
    s = poly.sum(axis=1)
    diff = np.diff(poly, axis=1).flatten()
    tl = poly[np.argmin(s)]
    br = poly[np.argmax(s)]
    tr = poly[np.argmin(diff)]
    bl = poly[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def shrink_poly(poly, shrink_ratio=0.3):
    poly = np.array(poly, dtype=np.float32).reshape(-1, 2)
    N = poly.shape[0]
    if N != 4:
        raise ValueError("Expected quadrilateral with 4 vertices")

    area = 0.0
    for i in range(N):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % N]
        area += x1 * y2 - x2 * y1
    sign = 1.0 if area > 0 else -1.0
    new_poly = np.zeros_like(poly)
    for i in range(N):
        p_prev = poly[(i - 1) % N]
        p_curr = poly[i]
        p_next = poly[(i + 1) % N]
        edge1 = p_curr - p_prev
        len1 = np.linalg.norm(edge1)
        n1 = sign * np.array([edge1[1], -edge1[0]]) / (len1 + 1e-6)
        edge2 = p_next - p_curr
        len2 = np.linalg.norm(edge2)
        n2 = sign * np.array([edge2[1], -edge2[0]]) / (len2 + 1e-6)
        n_avg = n1 + n2
        norm_n = np.linalg.norm(n_avg)
        if norm_n > 1e-6:
            n_avg /= norm_n
        else:
            n_avg = np.zeros(2, dtype=np.float32)
        offset = shrink_ratio * min(len1, len2)
        new_poly[i] = p_curr - offset * n_avg
    result = new_poly.astype(np.float32)
    # Safety: if somehow NaN crept in, return original poly unchanged
    if not np.all(np.isfinite(result)):
        return poly.astype(np.float32)
    return result


class EASTDataset(Dataset):
    def __init__(
        self,
        images_folder,
        coco_annotation_file,
        target_size=512,
        score_geo_scale=0.25,
        shrink_ratio=0.3,
        quad_source="auto",
        flip_prob=0.01,
        vflip_prob=0.0,
        small_rotate_prob=0.2,
        small_rotate_deg=2.0,
        perspective_prob=0.1,
        perspective_scale=0.015,
        shear_prob=0.15,
        shear_deg=5.0,
        random_crop_prob=0.2,
        random_crop_scale=(0.7, 1.0),
        blur_prob=0.1,
        blur_ksize_range=(3, 5),
        motion_blur_prob=0.1,
        motion_blur_ksize_range=(3, 9),
        noise_prob=0.1,
        noise_std=0.008,
        salt_pepper_prob=0.0005,
        jpeg_prob=0.1,
        jpeg_quality_range=(75, 95),
        shading_prob=0.1,
        shading_strength=0.1,
        gamma_prob=0.2,
        gamma_range=(0.95, 1.05),
        downscale_prob=0.1,
        downscale_range=(0.7, 0.95),
        negative_prob=0.05,
        color_jitter=(0.1, 0.1, 0.1, 0.05),
        hsv_prob=0.15,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.2,
        cutout_prob=0.15,
        cutout_num_holes=2,
        cutout_hole_size_range=(0.05, 0.15),
        elastic_prob=0.1,
        elastic_alpha=20.0,
        elastic_sigma=4.0,
        fog_prob=0.1,
        fog_strength_range=(0.1, 0.4),
        fog_direction="random",
        mosaic_prob=0.0,
        mosaic_center_range=(0.3, 0.7),
        cutmix_prob=0.0,
        cutmix_alpha=1.0,
        ricap_prob=0.0,
        ricap_beta=0.3,
        resizemix_prob=0.0,
        resizemix_scale_range=(0.1, 0.8),
        transform=None,
        dataset_name=None,
    ):
        self.images_folder = images_folder
        self.target_size = target_size
        self.score_geo_scale = score_geo_scale
        self.shrink_ratio = float(shrink_ratio)
        self.quad_source = str(quad_source)
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
        self.mosaic_prob = float(mosaic_prob)
        self.mosaic_center_range = tuple(mosaic_center_range)
        self.cutmix_prob = float(cutmix_prob)
        self.cutmix_alpha = float(cutmix_alpha)
        self.ricap_prob = float(ricap_prob)
        self.ricap_beta = float(ricap_beta)
        self.resizemix_prob = float(resizemix_prob)
        self.resizemix_scale_range = tuple(resizemix_scale_range)
        self.dataset_name = (
            dataset_name if dataset_name is not None else Path(images_folder).stem
        )

        if self.shrink_ratio < 0:
            raise ValueError("shrink_ratio must be >= 0")
        if self.quad_source not in {"auto", "as_is", "min_area_rect"}:
            raise ValueError(
                "quad_source must be one of {'auto', 'as_is', 'min_area_rect'}"
            )
        if not (0.0 <= self.flip_prob <= 1.0):
            raise ValueError("flip_prob must be in [0, 1]")
        if not (0.0 <= self.vflip_prob <= 1.0):
            raise ValueError("vflip_prob must be in [0, 1]")
        if not (0.0 <= self.small_rotate_prob <= 1.0):
            raise ValueError("small_rotate_prob must be in [0, 1]")
        if self.small_rotate_deg < 0:
            raise ValueError("small_rotate_deg must be >= 0")
        if not (0.0 <= self.perspective_prob <= 1.0):
            raise ValueError("perspective_prob must be in [0, 1]")
        if self.perspective_scale < 0:
            raise ValueError("perspective_scale must be >= 0")
        if not (0.0 <= self.shear_prob <= 1.0):
            raise ValueError("shear_prob must be in [0, 1]")
        if self.shear_deg < 0:
            raise ValueError("shear_deg must be >= 0")
        if not (0.0 <= self.random_crop_prob <= 1.0):
            raise ValueError("random_crop_prob must be in [0, 1]")
        if not (0.0 <= self.blur_prob <= 1.0):
            raise ValueError("blur_prob must be in [0, 1]")
        if not (0.0 <= self.motion_blur_prob <= 1.0):
            raise ValueError("motion_blur_prob must be in [0, 1]")
        if not (0.0 <= self.noise_prob <= 1.0):
            raise ValueError("noise_prob must be in [0, 1]")
        if self.noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        if not (0.0 <= self.salt_pepper_prob <= 1.0):
            raise ValueError("salt_pepper_prob must be in [0, 1]")
        if not (0.0 <= self.jpeg_prob <= 1.0):
            raise ValueError("jpeg_prob must be in [0, 1]")
        if not (0.0 <= self.shading_prob <= 1.0):
            raise ValueError("shading_prob must be in [0, 1]")
        if not (0.0 <= self.gamma_prob <= 1.0):
            raise ValueError("gamma_prob must be in [0, 1]")
        if not (0.0 <= self.downscale_prob <= 1.0):
            raise ValueError("downscale_prob must be in [0, 1]")
        if not (0.0 <= self.negative_prob <= 1.0):
            raise ValueError("negative_prob must be in [0, 1]")
        if not (0.0 <= self.hsv_prob <= 1.0):
            raise ValueError("hsv_prob must be in [0, 1]")
        if not (0.0 <= self.cutout_prob <= 1.0):
            raise ValueError("cutout_prob must be in [0, 1]")
        if self.cutout_num_holes < 1:
            raise ValueError("cutout_num_holes must be >= 1")
        if not (0.0 <= self.elastic_prob <= 1.0):
            raise ValueError("elastic_prob must be in [0, 1]")
        if not (0.0 <= self.fog_prob <= 1.0):
            raise ValueError("fog_prob must be in [0, 1]")
        if self.fog_direction not in {"top", "bottom", "left", "right", "random"}:
            raise ValueError("fog_direction must be one of: top, bottom, left, right, random")
        if not (0.0 <= self.mosaic_prob <= 1.0):
            raise ValueError("mosaic_prob must be in [0, 1]")
        lo, hi = self.mosaic_center_range
        if not (0.0 < lo < hi < 1.0):
            raise ValueError("mosaic_center_range must satisfy 0 < lo < hi < 1")
        if not (0.0 <= self.cutmix_prob <= 1.0):
            raise ValueError("cutmix_prob must be in [0, 1]")
        if self.cutmix_alpha <= 0:
            raise ValueError("cutmix_alpha must be > 0")
        if not (0.0 <= self.ricap_prob <= 1.0):
            raise ValueError("ricap_prob must be in [0, 1]")
        if self.ricap_beta <= 0:
            raise ValueError("ricap_beta must be > 0")
        if not (0.0 <= self.resizemix_prob <= 1.0):
            raise ValueError("resizemix_prob must be in [0, 1]")
        rlo, rhi = self.resizemix_scale_range
        if not (0.0 < rlo < rhi < 1.0):
            raise ValueError("resizemix_scale_range must satisfy 0 < lo < hi < 1")

        if transform is None:
            self._color_jitter = None
            if color_jitter:
                if isinstance(color_jitter, (list, tuple)):
                    self._color_jitter = transforms.ColorJitter(*color_jitter)
                else:
                    self._color_jitter = transforms.ColorJitter(color_jitter)
            self._base_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]
            )
            if self._color_jitter is not None:
                self.transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        self._color_jitter,
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
                        ),
                    ]
                )
            else:
                self.transform = self._base_transform
        else:
            self.transform = transform
            self._base_transform = transform
            self._color_jitter = None

        with open(coco_annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.images_info = {img["id"]: img for img in data["images"]}
        self.image_ids = list(self.images_info.keys())
        self.annots = {}
        for ann in data["annotations"]:
            self.annots.setdefault(ann["image_id"], []).append(ann)
        self._filter_invalid()

    @staticmethod
    def _iter_annotation_polygons(ann: dict) -> Iterable[np.ndarray]:
        seg = ann.get("segmentation")
        if seg is None:
            return []

        if isinstance(seg, dict):
            # COCO RLE masks are not supported by EAST quad target generation.
            return []

        if not isinstance(seg, (list, tuple)) or len(seg) == 0:
            return []

        if isinstance(seg[0], (list, tuple, np.ndarray)):
            seg_parts = seg
        else:
            seg_parts = [seg]

        polygons: List[np.ndarray] = []
        for seg_poly in seg_parts:
            pts_raw = np.asarray(seg_poly, dtype=np.float32)
            if pts_raw.ndim == 1:
                if pts_raw.size < 8 or pts_raw.size % 2 != 0:
                    continue
                pts = pts_raw.reshape(-1, 2)
            elif pts_raw.ndim == 2 and pts_raw.shape[1] == 2:
                if pts_raw.shape[0] < 4:
                    continue
                pts = pts_raw
            else:
                continue

            polygons.append(pts.astype(np.float32))

        return polygons

    def _polygon_to_quad(self, pts: np.ndarray) -> Optional[np.ndarray]:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 4:
            return None

        if self.quad_source == "as_is":
            if pts.shape[0] != 4:
                return None
            return order_vertices_clockwise(pts)

        if self.quad_source == "auto" and pts.shape[0] == 4:
            return order_vertices_clockwise(pts)

        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        return order_vertices_clockwise(box)

    def _annotation_to_quads(
        self,
        ann: dict,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> List[np.ndarray]:
        quads: List[np.ndarray] = []
        for pts in self._iter_annotation_polygons(ann):
            quad = self._polygon_to_quad(pts)
            if quad is None:
                continue
            quad = quad.astype(np.float32).copy()
            quad[:, 0] *= scale_x
            quad[:, 1] *= scale_y
            quads.append(quad)
        return quads

    @staticmethod
    def _should_apply(prob: float, force: bool) -> bool:
        if force:
            return True
        if prob <= 0.0:
            return False
        return np.random.rand() < prob

    @staticmethod
    def _ensure_uint8(img):
        if img.dtype == np.uint8:
            return img
        return np.clip(img, 0, 255).astype(np.uint8)

    def _clip_quads(self, quads):
        if not quads:
            return quads
        pts = np.stack(quads, axis=0).astype(np.float32)
        pts[..., 0] = np.clip(pts[..., 0], 0, self.target_size - 1)
        pts[..., 1] = np.clip(pts[..., 1], 0, self.target_size - 1)
        return [p.astype(np.float32) for p in pts]

    def _apply_hflip(self, img, quads, force: bool = False):
        if not self._should_apply(self.flip_prob, force):
            return img, quads

        h, w = img.shape[:2]
        img_flip = np.fliplr(img).copy()

        if not quads:
            return img_flip, quads

        pts = np.stack(quads, axis=0).astype(np.float32)
        pts[..., 0] = (w - 1) - pts[..., 0]
        return img_flip, self._clip_quads([p.astype(np.float32) for p in pts])

    def _apply_vflip(self, img, quads, force: bool = False):
        if not self._should_apply(self.vflip_prob, force):
            return img, quads

        h, w = img.shape[:2]
        img_flip = np.flipud(img).copy()

        if not quads:
            return img_flip, quads

        pts = np.stack(quads, axis=0).astype(np.float32)
        pts[..., 1] = (h - 1) - pts[..., 1]
        return img_flip, self._clip_quads([p.astype(np.float32) for p in pts])

    def _apply_small_rotate(self, img, quads, force: bool = False):
        if self.small_rotate_deg <= 0:
            return img, quads
        if not self._should_apply(self.small_rotate_prob, force):
            return img, quads

        h, w = img.shape[:2]
        angle = np.random.uniform(-self.small_rotate_deg, self.small_rotate_deg)
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        img_rot = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        if not quads:
            return img_rot, quads

        pts = np.stack(quads, axis=0).astype(np.float32)
        ones = np.ones((pts.shape[0], pts.shape[1], 1), dtype=np.float32)
        pts_h = np.concatenate([pts, ones], axis=2)
        pts_rot = pts_h @ M.T
        return img_rot, self._clip_quads([p.astype(np.float32) for p in pts_rot])

    def _apply_perspective(self, img, quads, force: bool = False):
        if self.perspective_scale <= 0:
            return img, quads
        if not self._should_apply(self.perspective_prob, force):
            return img, quads

        h, w = img.shape[:2]
        max_dx = self.perspective_scale * w
        max_dy = self.perspective_scale * h

        src = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )
        dst = src + np.random.uniform(
            [-max_dx, -max_dy], [max_dx, max_dy], size=src.shape
        ).astype(np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        img_warp = cv2.warpPerspective(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        if not quads:
            return img_warp, quads

        pts = np.stack(quads, axis=0).astype(np.float32)
        ones = np.ones((pts.shape[0], pts.shape[1], 1), dtype=np.float32)
        pts_h = np.concatenate([pts, ones], axis=2)
        pts_w = pts_h @ M.T
        pts_w = pts_w[..., :2] / np.clip(pts_w[..., 2:3], 1e-6, None)
        return img_warp, self._clip_quads([p.astype(np.float32) for p in pts_w])

    def _apply_shear(self, img, quads, force: bool = False):
        if self.shear_deg <= 0:
            return img, quads
        if not self._should_apply(self.shear_prob, force):
            return img, quads

        h, w = img.shape[:2]
        sx = np.tan(np.radians(np.random.uniform(-self.shear_deg, self.shear_deg)))
        sy = np.tan(np.radians(np.random.uniform(-self.shear_deg, self.shear_deg)))

        M = np.array([[1.0, sx, -sx * h / 2.0],
                      [sy, 1.0, -sy * w / 2.0]], dtype=np.float32)
        img_shear = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        if not quads:
            return img_shear, quads

        pts = np.stack(quads, axis=0).astype(np.float32)
        ones = np.ones((pts.shape[0], pts.shape[1], 1), dtype=np.float32)
        pts_h = np.concatenate([pts, ones], axis=2)
        pts_shear = pts_h @ M.T
        return img_shear, self._clip_quads([p.astype(np.float32) for p in pts_shear])

    def _apply_random_crop(self, img, quads, force: bool = False):
        if not self._should_apply(self.random_crop_prob, force):
            return img, quads

        h, w = img.shape[:2]
        s_min, s_max = self.random_crop_scale
        scale = np.random.uniform(s_min, s_max)
        crop_w = max(1, int(w * scale))
        crop_h = max(1, int(h * scale))
        x0 = np.random.randint(0, max(1, w - crop_w + 1))
        y0 = np.random.randint(0, max(1, h - crop_h + 1))

        img_crop = img[y0:y0 + crop_h, x0:x0 + crop_w]
        img_out = cv2.resize(img_crop, (w, h), interpolation=cv2.INTER_LINEAR)

        sx = w / crop_w
        sy = h / crop_h

        if not quads:
            return img_out, quads

        pts = np.stack(quads, axis=0).astype(np.float32)
        pts[..., 0] = (pts[..., 0] - x0) * sx
        pts[..., 1] = (pts[..., 1] - y0) * sy
        return img_out, self._clip_quads([p.astype(np.float32) for p in pts])

    def _apply_mosaic(self, idx: int, force: bool = False):
        """Combine 4 images into one mosaic. Returns (img, quads) or None if skipped."""
        if not self._should_apply(self.mosaic_prob, force):
            return None

        n = len(self.image_ids)
        if n < 2:
            return None

        lo, hi = self.mosaic_center_range
        ts = self.target_size
        cx = int(np.random.uniform(lo, hi) * ts)
        cy = int(np.random.uniform(lo, hi) * ts)

        # pick 3 other random indices (may repeat if dataset is small)
        other_idxs = [np.random.randint(0, n) for _ in range(3)]
        indices = [idx] + other_idxs

        # quadrant positions: (x_start, y_start, x_end, y_end)
        quadrants = [
            (0,  0,  cx, cy),   # top-left
            (cx, 0,  ts, cy),   # top-right
            (0,  cy, cx, ts),   # bottom-left
            (cx, cy, ts, ts),   # bottom-right
        ]

        canvas = np.full((ts, ts, 3), 255, dtype=np.uint8)
        all_quads: List[np.ndarray] = []

        for tile_idx, (x0, y0, x1, y1) in zip(indices, quadrants):
            tile_w = x1 - x0
            tile_h = y1 - y0
            if tile_w <= 0 or tile_h <= 0:
                continue

            tile_img, tile_quads = self._load_image_and_quads(tile_idx)
            tile_img = cv2.resize(tile_img, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
            canvas[y0:y1, x0:x1] = tile_img

            orig_w = self.target_size
            orig_h = self.target_size
            sx = tile_w / orig_w
            sy = tile_h / orig_h

            for q in tile_quads:
                q_new = q.copy()
                q_new[:, 0] = q_new[:, 0] * sx + x0
                q_new[:, 1] = q_new[:, 1] * sy + y0
                all_quads.append(q_new)

        return canvas, self._clip_quads(all_quads)

    def _apply_cutmix(self, idx: int, force: bool = False):
        """Cut a rectangular patch from a second image and paste it over the current image.

        Quads that fall entirely inside the pasted patch region are replaced by the
        second image's quads mapped to that region.  Quads from the base image that
        overlap the patch are discarded (conservative — avoids partial-quad labels).

        Returns (img, quads) or None if skipped.
        """
        if not self._should_apply(self.cutmix_prob, force):
            return None

        n = len(self.image_ids)
        if n < 2:
            return None

        # Sample λ from Beta(alpha, alpha)
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        ts = self.target_size

        # Determine cut box proportional to √(1−λ)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(ts * cut_ratio)
        cut_h = int(ts * cut_ratio)

        cx = np.random.randint(0, max(1, ts - cut_w + 1))
        cy = np.random.randint(0, max(1, ts - cut_h + 1))
        x1, y1 = cx, cy
        x2, y2 = min(cx + cut_w, ts), min(cy + cut_h, ts)

        base_img, base_quads = self._load_image_and_quads(idx)
        other_idx = np.random.randint(0, n)
        other_img, other_quads = self._load_image_and_quads(other_idx)

        # Paste patch from other image
        result = base_img.copy()
        result[y1:y2, x1:x2] = other_img[y1:y2, x1:x2]

        # Keep base quads that are fully OUTSIDE the cut region
        kept_base = []
        for q in base_quads:
            qx_min, qy_min = q[:, 0].min(), q[:, 1].min()
            qx_max, qy_max = q[:, 0].max(), q[:, 1].max()
            # Discard if bounding box overlaps the cut region
            overlaps = not (qx_max <= x1 or qx_min >= x2 or qy_max <= y1 or qy_min >= y2)
            if not overlaps:
                kept_base.append(q)

        # Keep other quads that are fully INSIDE the cut region
        kept_other = []
        for q in other_quads:
            qx_min, qy_min = q[:, 0].min(), q[:, 1].min()
            qx_max, qy_max = q[:, 0].max(), q[:, 1].max()
            inside = (qx_min >= x1 and qy_min >= y1 and qx_max <= x2 and qy_max <= y2)
            if inside:
                kept_other.append(q)

        merged = kept_base + kept_other
        return result, self._clip_quads(merged)

    def _apply_ricap(self, idx: int, force: bool = False):
        """RICAP (Random Image Cropping and Patching): tile 4 random crops from 4 images
        at a random center split point, similar to mosaic but each tile is a crop
        from a potentially different region of its source image.

        Returns (img, quads) or None if skipped.
        """
        if not self._should_apply(self.ricap_prob, force):
            return None

        n = len(self.image_ids)
        if n < 2:
            return None

        ts = self.target_size
        beta = self.ricap_beta

        # Sample split point from Beta distribution (centered around 0.5)
        w_ratio = np.random.beta(beta, beta)
        h_ratio = np.random.beta(beta, beta)
        # Clamp to avoid degenerate tiles
        w_ratio = float(np.clip(w_ratio, 0.15, 0.85))
        h_ratio = float(np.clip(h_ratio, 0.15, 0.85))

        cx = int(w_ratio * ts)
        cy = int(h_ratio * ts)

        # Tile dimensions: (tile_w, tile_h)
        tile_dims = [
            (cx,      cy,      0,  0),   # top-left:     x in [0,cx),  y in [0,cy)
            (ts - cx, cy,      cx, 0),   # top-right:    x in [cx,ts), y in [0,cy)
            (cx,      ts - cy, 0,  cy),  # bottom-left:  x in [0,cx),  y in [cy,ts)
            (ts - cx, ts - cy, cx, cy),  # bottom-right: x in [cx,ts), y in [cy,ts)
        ]

        other_idxs = [np.random.randint(0, n) for _ in range(3)]
        indices = [idx] + other_idxs

        canvas = np.full((ts, ts, 3), 255, dtype=np.uint8)
        all_quads: List[np.ndarray] = []

        for src_idx, (tile_w, tile_h, dst_x, dst_y) in zip(indices, tile_dims):
            if tile_w <= 0 or tile_h <= 0:
                continue

            src_img, src_quads = self._load_image_and_quads(src_idx)
            src_h, src_w = src_img.shape[:2]

            # Random crop origin in source image
            max_ox = max(0, src_w - tile_w)
            max_oy = max(0, src_h - tile_h)
            ox = np.random.randint(0, max_ox + 1)
            oy = np.random.randint(0, max_oy + 1)

            crop = src_img[oy:oy + tile_h, ox:ox + tile_w]
            if crop.shape[0] != tile_h or crop.shape[1] != tile_w:
                crop = cv2.resize(crop, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)

            canvas[dst_y:dst_y + tile_h, dst_x:dst_x + tile_w] = crop

            # Transform quads: subtract crop origin, then offset to canvas position
            for q in src_quads:
                q_new = q.copy()
                q_new[:, 0] = q_new[:, 0] - ox + dst_x
                q_new[:, 1] = q_new[:, 1] - oy + dst_y
                # Keep only quads fully within the tile's canvas region
                qx_min, qy_min = q_new[:, 0].min(), q_new[:, 1].min()
                qx_max, qy_max = q_new[:, 0].max(), q_new[:, 1].max()
                if (qx_min >= dst_x and qy_min >= dst_y
                        and qx_max <= dst_x + tile_w and qy_max <= dst_y + tile_h):
                    all_quads.append(q_new)

        return canvas, self._clip_quads(all_quads)

    def _apply_resizemix(self, idx: int, force: bool = False):
        """ResizeMix: resize a second image to a small patch and paste it onto the base image.

        Unlike CutMix (which cuts from the same location in both images), ResizeMix
        downscales the entire source image to a small thumbnail and pastes it at a
        random location, providing scale-invariant context mixing.

        Quads from the base image that are not covered by the patch are kept.
        The resized patch's quads are scaled and offset into the canvas.

        Returns (img, quads) or None if skipped.
        """
        if not self._should_apply(self.resizemix_prob, force):
            return None

        n = len(self.image_ids)
        if n < 2:
            return None

        s_min, s_max = self.resizemix_scale_range
        scale = np.random.uniform(s_min, s_max)
        ts = self.target_size

        patch_w = max(8, int(ts * scale))
        patch_h = max(8, int(ts * scale))

        # Random paste location
        px = np.random.randint(0, max(1, ts - patch_w + 1))
        py = np.random.randint(0, max(1, ts - patch_h + 1))
        px2, py2 = min(px + patch_w, ts), min(py + patch_h, ts)
        actual_w = px2 - px
        actual_h = py2 - py

        base_img, base_quads = self._load_image_and_quads(idx)
        other_idx = np.random.randint(0, n)
        other_img, other_quads = self._load_image_and_quads(other_idx)

        # Resize source image to patch size
        patch = cv2.resize(other_img, (actual_w, actual_h), interpolation=cv2.INTER_LINEAR)
        result = base_img.copy()
        result[py:py2, px:px2] = patch

        # Scale factors from original target_size to patch size
        sx = actual_w / ts
        sy = actual_h / ts

        # Keep base quads outside the paste region
        kept_base = []
        for q in base_quads:
            qx_min, qy_min = q[:, 0].min(), q[:, 1].min()
            qx_max, qy_max = q[:, 0].max(), q[:, 1].max()
            overlaps = not (qx_max <= px or qx_min >= px2 or qy_max <= py or qy_min >= py2)
            if not overlaps:
                kept_base.append(q)

        # Scale and offset the source image's quads into the patch region
        kept_other = []
        for q in other_quads:
            q_new = q.copy()
            q_new[:, 0] = q_new[:, 0] * sx + px
            q_new[:, 1] = q_new[:, 1] * sy + py
            # Keep only quads fully within the pasted patch
            qx_min, qy_min = q_new[:, 0].min(), q_new[:, 1].min()
            qx_max, qy_max = q_new[:, 0].max(), q_new[:, 1].max()
            if qx_min >= px and qy_min >= py and qx_max <= px2 and qy_max <= py2:
                kept_other.append(q_new)

        merged = kept_base + kept_other
        return result, self._clip_quads(merged)

    def _apply_geometric_augments(self, img, quads):
        img, quads = self._apply_random_crop(img, quads)
        img, quads = self._apply_small_rotate(img, quads)
        img, quads = self._apply_shear(img, quads)
        img, quads = self._apply_perspective(img, quads)
        img, quads = self._apply_hflip(img, quads)
        img, quads = self._apply_vflip(img, quads)
        return img, quads

    # ------------------------------------------------------------------
    # NaN / sanity guard helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_nan(tensor) -> bool:
        """Return True if a torch.Tensor or np.ndarray contains any NaN / Inf."""
        if isinstance(tensor, np.ndarray):
            return bool(np.any(~np.isfinite(tensor)))
        try:
            return bool((~torch.isfinite(tensor)).any().item())
        except Exception:
            return False

    def _sample_is_valid(self, img_tensor, target: dict) -> bool:
        """Quick sanity-check: no NaN/Inf in image tensor or map targets."""
        if self._has_nan(img_tensor):
            return False
        for v in target.values():
            if self._has_nan(v):
                return False
        return True

    def _apply_gaussian_blur(self, img, force: bool = False):
        if not self._should_apply(self.blur_prob, force):
            return img
        k_min, k_max = self.blur_ksize_range
        ks = [k for k in range(k_min, k_max + 1) if k % 2 == 1]
        if not ks:
            k = max(1, k_min)
            if k % 2 == 0:
                k += 1
        else:
            k = np.random.choice(ks)
        return cv2.GaussianBlur(img, (k, k), 0)

    def _apply_noise(self, img, force: bool = False):
        if not self._should_apply(self.noise_prob, force):
            return img
        img_f = img.astype(np.float32)
        sigma = self.noise_std * 255.0
        if sigma > 0:
            noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
            img_f = img_f + noise
        img_f = np.clip(img_f, 0, 255)
        if self.salt_pepper_prob > 0:
            prob = self.salt_pepper_prob
            rnd = np.random.rand(img.shape[0], img.shape[1])
            img_f[rnd < prob] = 0
            img_f[rnd > 1 - prob] = 255
        return img_f.astype(np.uint8)

    def _apply_jpeg(self, img, force: bool = False):
        if not self._should_apply(self.jpeg_prob, force):
            return img
        q_min, q_max = self.jpeg_quality_range
        quality = int(np.random.uniform(q_min, q_max))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encimg = cv2.imencode(".jpg", img, encode_param)
        if not success:
            return img
        decoded = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        if decoded is None:
            return img
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    def _apply_shading(self, img, force: bool = False):
        if not self._should_apply(self.shading_prob, force):
            return img
        h, w = img.shape[:2]
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xv, yv = np.meshgrid(x, y)
        angle = np.random.uniform(0, 2 * np.pi)
        grad = np.cos(angle) * xv + np.sin(angle) * yv
        mask = 1.0 + self.shading_strength * grad
        img_f = img.astype(np.float32) * mask[..., None]
        return np.clip(img_f, 0, 255).astype(np.uint8)

    def _apply_gamma(self, img, force: bool = False):
        if not self._should_apply(self.gamma_prob, force):
            return img
        g_min, g_max = self.gamma_range
        gamma = np.random.uniform(g_min, g_max)
        if gamma <= 0:
            return img
        table = ((np.arange(256) / 255.0) ** gamma * 255.0).astype(np.uint8)
        return cv2.LUT(img, table)

    def _apply_downscale(self, img, force: bool = False):
        if not self._should_apply(self.downscale_prob, force):
            return img
        s_min, s_max = self.downscale_range
        scale = np.random.uniform(s_min, s_max)
        if scale >= 1.0:
            return img
        h, w = img.shape[:2]
        nw = max(2, int(w * scale))
        nh = max(2, int(h * scale))
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    def _apply_negative(self, img, force: bool = False):
        if not self._should_apply(self.negative_prob, force):
            return img
        return 255 - img

    def _apply_motion_blur(self, img, force: bool = False):
        if not self._should_apply(self.motion_blur_prob, force):
            return img
        k_min, k_max = self.motion_blur_ksize_range
        ks = [k for k in range(k_min, k_max + 1) if k % 2 == 1]
        k = int(np.random.choice(ks)) if ks else max(3, k_min | 1)
        angle = np.random.uniform(0, 180)
        M_rot = cv2.getRotationMatrix2D((k // 2, k // 2), angle, 1.0)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        kernel = cv2.warpAffine(kernel, M_rot, (k, k))
        s = kernel.sum()
        if s > 0:
            kernel /= s
        return cv2.filter2D(img, -1, kernel)

    def _apply_hsv(self, img, force: bool = False):
        if not self._should_apply(self.hsv_prob, force):
            return img
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        dh = np.random.uniform(-self.hsv_h, self.hsv_h) * 180.0
        ds = np.random.uniform(-self.hsv_s, self.hsv_s) * 255.0
        dv = np.random.uniform(-self.hsv_v, self.hsv_v) * 255.0
        img_hsv[..., 0] = (img_hsv[..., 0] + dh) % 180.0
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + ds, 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + dv, 0, 255)
        return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _apply_cutout(self, img, force: bool = False):
        if not self._should_apply(self.cutout_prob, force):
            return img
        img = img.copy()
        h, w = img.shape[:2]
        s_min, s_max = self.cutout_hole_size_range
        for _ in range(self.cutout_num_holes):
            hole_w = int(np.random.uniform(s_min, s_max) * w)
            hole_h = int(np.random.uniform(s_min, s_max) * h)
            x0 = np.random.randint(0, max(1, w - hole_w + 1))
            y0 = np.random.randint(0, max(1, h - hole_h + 1))
            # fill with mean color to avoid black rectangle artifacts
            fill = img.mean(axis=(0, 1)).astype(np.uint8)
            img[y0:y0 + hole_h, x0:x0 + hole_w] = fill
        return img

    def _apply_elastic(self, img, force: bool = False):
        if not self._should_apply(self.elastic_prob, force):
            return img
        h, w = img.shape[:2]
        alpha = self.elastic_alpha
        sigma = self.elastic_sigma
        dx = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1).astype(np.float32),
            (0, 0), sigma
        ) * alpha
        dy = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1).astype(np.float32),
            (0, 0), sigma
        ) * alpha
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    def _apply_fog(self, img, force: bool = False):
        if not self._should_apply(self.fog_prob, force):
            return img
        h, w = img.shape[:2]
        s_min, s_max = self.fog_strength_range
        strength = np.random.uniform(s_min, s_max)

        direction = self.fog_direction
        if direction == "random":
            direction = np.random.choice(["top", "bottom", "left", "right"])

        if direction == "top":
            # плотный туман сверху, рассеивается вниз
            gradient = np.linspace(strength, 0.0, h, dtype=np.float32)
            gradient = np.tile(gradient[:, None], (1, w))
        elif direction == "bottom":
            # плотный туман снизу, рассеивается вверх
            gradient = np.linspace(0.0, strength, h, dtype=np.float32)
            gradient = np.tile(gradient[:, None], (1, w))
        elif direction == "left":
            gradient = np.linspace(strength, 0.0, w, dtype=np.float32)
            gradient = np.tile(gradient[None, :], (h, 1))
        else:  # right
            gradient = np.linspace(0.0, strength, w, dtype=np.float32)
            gradient = np.tile(gradient[None, :], (h, 1))

        gradient = gradient[..., None]  # (H, W, 1) для broadcast по каналам
        fog = img.astype(np.float32) * (1.0 - gradient) + 255.0 * gradient
        return np.clip(fog, 0, 255).astype(np.uint8)

    def _apply_photometric_augments(self, img):
        img = self._apply_hsv(img)
        img = self._apply_shading(img)
        img = self._apply_gamma(img)
        img = self._apply_fog(img)
        img = self._apply_gaussian_blur(img)
        img = self._apply_motion_blur(img)
        img = self._apply_noise(img)
        img = self._apply_jpeg(img)
        img = self._apply_downscale(img)
        img = self._apply_cutout(img)
        img = self._apply_elastic(img)
        img = self._apply_negative(img)
        return img

    def _apply_color_jitter(self, img):
        if self._color_jitter is None:
            return img
        pil = transforms.ToPILImage()(img)
        jittered = self._color_jitter(pil)
        return np.array(jittered)

    @staticmethod
    def _draw_quads(img, quads, color=(0, 255, 0), thickness=2):
        vis = img.copy()
        for quad in quads:
            coords = quad.reshape(4, 2).astype(np.int32)
            cv2.polylines(vis, [coords], isClosed=True, color=color, thickness=thickness)
        return vis

    def _load_image_and_quads(self, idx):
        image_id = self.image_ids[idx]
        info = self.images_info[image_id]
        path = os.path.join(self.images_folder, info["file_name"])
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (self.target_size, self.target_size))

        anns = self.annots.get(image_id, [])
        quads = []

        scale_x = self.target_size / info["width"]
        scale_y = self.target_size / info["height"]

        for ann in anns:
            quads.extend(self._annotation_to_quads(ann, scale_x=scale_x, scale_y=scale_y))

        return img_resized, quads

    def list_augmentations(self):
        return [
            "original",
            "mosaic",
            "cutmix",
            "ricap",
            "resizemix",
            "flip",
            "vflip",
            "small_rotate",
            "shear",
            "random_crop",
            "perspective",
            "color_jitter",
            "hsv",
            "blur",
            "motion_blur",
            "noise",
            "jpeg",
            "shading",
            "gamma",
            "downscale",
            "cutout",
            "elastic",
            "fog",
            "negative",
        ]

    def preview_augmentation(self, idx: int, name: str):
        img, quads = self._load_image_and_quads(idx)
        name = name.lower()

        if name == "original":
            pass
        elif name == "mosaic":
            result = self._apply_mosaic(idx, force=True)
            if result is not None:
                img, quads = result
        elif name == "cutmix":
            result = self._apply_cutmix(idx, force=True)
            if result is not None:
                img, quads = result
        elif name == "ricap":
            result = self._apply_ricap(idx, force=True)
            if result is not None:
                img, quads = result
        elif name == "resizemix":
            result = self._apply_resizemix(idx, force=True)
            if result is not None:
                img, quads = result
        elif name == "flip":
            img, quads = self._apply_hflip(img, quads, force=True)
        elif name == "vflip":
            img, quads = self._apply_vflip(img, quads, force=True)
        elif name == "small_rotate":
            img, quads = self._apply_small_rotate(img, quads, force=True)
        elif name == "shear":
            img, quads = self._apply_shear(img, quads, force=True)
        elif name == "random_crop":
            img, quads = self._apply_random_crop(img, quads, force=True)
        elif name == "perspective":
            img, quads = self._apply_perspective(img, quads, force=True)
        elif name == "color_jitter":
            img = self._apply_color_jitter(img)
        elif name == "hsv":
            img = self._apply_hsv(img, force=True)
        elif name == "blur":
            img = self._apply_gaussian_blur(img, force=True)
        elif name == "motion_blur":
            img = self._apply_motion_blur(img, force=True)
        elif name == "noise":
            img = self._apply_noise(img, force=True)
        elif name == "jpeg":
            img = self._apply_jpeg(img, force=True)
        elif name == "shading":
            img = self._apply_shading(img, force=True)
        elif name == "gamma":
            img = self._apply_gamma(img, force=True)
        elif name == "downscale":
            img = self._apply_downscale(img, force=True)
        elif name == "cutout":
            img = self._apply_cutout(img, force=True)
        elif name == "elastic":
            img = self._apply_elastic(img, force=True)
        elif name == "fog":
            img = self._apply_fog(img, force=True)
        elif name == "negative":
            img = self._apply_negative(img, force=True)
        else:
            return None

        img = self._ensure_uint8(img)

        if quads:
            quads_array = np.stack([q.flatten() for q in quads], axis=0).astype(
                np.float32
            )
        else:
            quads_array = np.zeros((0, 8), dtype=np.float32)

        return self._draw_quads(img, quads_array)

    def _filter_invalid(self):
        invalid_ids = []
        for img_id in list(self.image_ids):
            anns = self.annots.get(img_id, [])

            has_valid = False
            for ann in anns:
                if self._annotation_to_quads(ann):
                    has_valid = True
                    break
            if not has_valid:
                invalid_ids.append(img_id)
        for img_id in invalid_ids:
            self.image_ids.remove(img_id)
            self.annots.pop(img_id, None)

        if invalid_ids:
            warnings.warn(
                f"EASTDataset: found {len(invalid_ids)} images without valid quads — they will be skipped",
                UserWarning,
            )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # NaN-safe wrapper: if any augmentation produces NaN/Inf, fall back to a
        # clean sample (load raw image, no augmentations) up to _MAX_RETRIES times.
        # If all retries fail, return a zero-filled dummy sample rather than crashing.
        _MAX_RETRIES = 3
        for _attempt in range(_MAX_RETRIES + 1):
            try:
                img_tensor, target = self._build_sample(idx)
                if self._sample_is_valid(img_tensor, target):
                    return img_tensor, target
                # NaN detected — fall through to retry with a different index or clean load
                warnings.warn(
                    f"EASTDataset[{idx}]: NaN/Inf detected in sample "
                    f"(attempt {_attempt + 1}/{_MAX_RETRIES}), retrying with fallback.",
                    UserWarning,
                )
                # On subsequent retries pick a random other index
                if _attempt < _MAX_RETRIES:
                    idx = np.random.randint(0, len(self.image_ids))
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"EASTDataset[{idx}]: exception during augmentation "
                    f"(attempt {_attempt + 1}/{_MAX_RETRIES}): {e}",
                    UserWarning,
                )
                if _attempt < _MAX_RETRIES:
                    idx = np.random.randint(0, len(self.image_ids))

        # All retries exhausted — return a zero dummy to keep the DataLoader alive
        warnings.warn(
            f"EASTDataset: all {_MAX_RETRIES} retries failed, returning dummy sample.",
            UserWarning,
        )
        ts = self.target_size
        out_hw = int(ts * self.score_geo_scale)
        dummy_img = torch.zeros(3, ts, ts, dtype=torch.float32)
        dummy_target = {
            "score_map": torch.zeros(1, out_hw, out_hw, dtype=torch.float32),
            "geo_map": torch.zeros(8, out_hw, out_hw, dtype=torch.float32),
            "quads": torch.zeros(0, 8, dtype=torch.float32),
        }
        return dummy_img, dummy_target

    def _build_sample(self, idx):
        """Core sample-building logic (without NaN retry wrapper)."""
        # ── Mixing augmentations ──────────────────────────────────────────────
        # Each mixing aug is independent: all enabled ones can fire in the same
        # sample.  Apply them in order: mosaic → cutmix → ricap → resizemix.
        # The first one that fires determines the base image+quads; subsequent
        # ones are applied on top (they too may or may not fire based on their
        # probabilities).
        img_resized, quads = self._load_image_and_quads(idx)

        for _apply_mixing in (
            self._apply_mosaic,
            self._apply_cutmix,
            self._apply_ricap,
            self._apply_resizemix,
        ):
            result = _apply_mixing(idx)
            if result is not None:
                img_resized, quads = result
                break  # one mixing per sample is enough — avoids compounding distortions

        # ── Geometric augmentations (all applied independently) ───────────────
        img_resized, quads = self._apply_geometric_augments(img_resized, quads)

        # ── Photometric augmentations (all applied independently) ─────────────
        img_resized = self._apply_photometric_augments(img_resized)
        img_resized = self._ensure_uint8(img_resized)

        # Convert quads to (N, 8) array
        if quads:
            quads_array = np.stack(
                [q.flatten() for q in quads], axis=0
            ).astype(np.float32)
        else:
            quads_array = np.zeros((0, 8), dtype=np.float32)

        score_map, geo_map = self.compute_quad_maps(quads)
        img_tensor = self.transform(img_resized)
        target = {
            "score_map": torch.tensor(score_map).unsqueeze(0),
            "geo_map": torch.tensor(geo_map),
            "quads": torch.from_numpy(quads_array),
        }
        return img_tensor, target

    def compute_quad_maps(self, quads):
        out_h = int(self.target_size * self.score_geo_scale)
        out_w = int(self.target_size * self.score_geo_scale)
        score_map = np.zeros((out_h, out_w), dtype=np.float32)
        geo_map = np.zeros((8, out_h, out_w), dtype=np.float32)
        for quad in quads:
            try:
                quad = order_vertices_clockwise(quad)
                shrunk = shrink_poly(quad, shrink_ratio=self.shrink_ratio)

                # Guard: skip quad if shrink produced NaN/Inf
                if not np.all(np.isfinite(shrunk)):
                    continue

                coords = shrunk * self.score_geo_scale

                # Guard: skip degenerate quads (all points identical / zero area)
                if not np.all(np.isfinite(coords)):
                    continue

                rr, cc = skimage.draw.polygon(
                    coords[:, 1], coords[:, 0], shape=(out_h, out_w)
                )
                if len(rr) == 0:
                    continue
                score_map[rr, cc] = 1

                for i, (vx, vy) in enumerate(coords):
                    dx = vx - cc
                    dy = vy - rr
                    # Guard: skip NaN geometry offsets
                    if not (np.all(np.isfinite(dx)) and np.all(np.isfinite(dy))):
                        continue
                    geo_map[2 * i, rr, cc] = dx
                    geo_map[2 * i + 1, rr, cc] = dy
            except Exception as e:  # noqa: BLE001
                # Single bad quad should never kill the whole sample
                warnings.warn(f"compute_quad_maps: skipping quad due to error: {e}", UserWarning)
                continue

        # Final NaN clamp — belt-and-suspenders
        np.nan_to_num(score_map, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(geo_map, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return score_map, geo_map
