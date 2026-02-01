import os
import json
import warnings
from pathlib import Path

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
        if norm_n > 0:
            n_avg /= norm_n
        offset = shrink_ratio * min(len1, len2)
        new_poly[i] = p_curr - offset * n_avg
    return new_poly.astype(np.float32)


class EASTDataset(Dataset):
    def __init__(
        self,
        images_folder,
        coco_annotation_file,
        target_size=512,
        score_geo_scale=0.25,
        shrink_ratio=0.3,
        flip_prob=0.01,
        small_rotate_prob=0.2,
        small_rotate_deg=2.0,
        perspective_prob=0.1,
        perspective_scale=0.015,
        blur_prob=0.1,
        blur_ksize_range=(3, 5),
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
        transform=None,
        dataset_name=None,
    ):
        self.images_folder = images_folder
        self.target_size = target_size
        self.score_geo_scale = score_geo_scale
        self.shrink_ratio = float(shrink_ratio)
        self.flip_prob = float(flip_prob)
        self.small_rotate_prob = float(small_rotate_prob)
        self.small_rotate_deg = float(small_rotate_deg)
        self.perspective_prob = float(perspective_prob)
        self.perspective_scale = float(perspective_scale)
        self.blur_prob = float(blur_prob)
        self.blur_ksize_range = tuple(blur_ksize_range)
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
        self.dataset_name = (
            dataset_name if dataset_name is not None else Path(images_folder).stem
        )

        if self.shrink_ratio < 0:
            raise ValueError("shrink_ratio must be >= 0")
        if not (0.0 <= self.flip_prob <= 1.0):
            raise ValueError("flip_prob must be in [0, 1]")
        if not (0.0 <= self.small_rotate_prob <= 1.0):
            raise ValueError("small_rotate_prob must be in [0, 1]")
        if self.small_rotate_deg < 0:
            raise ValueError("small_rotate_deg must be >= 0")
        if not (0.0 <= self.perspective_prob <= 1.0):
            raise ValueError("perspective_prob must be in [0, 1]")
        if self.perspective_scale < 0:
            raise ValueError("perspective_scale must be >= 0")
        if not (0.0 <= self.blur_prob <= 1.0):
            raise ValueError("blur_prob must be in [0, 1]")
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

    def _apply_geometric_augments(self, img, quads):
        img, quads = self._apply_small_rotate(img, quads)
        img, quads = self._apply_perspective(img, quads)
        img, quads = self._apply_hflip(img, quads)
        return img, quads

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

    def _apply_photometric_augments(self, img):
        img = self._apply_shading(img)
        img = self._apply_gamma(img)
        img = self._apply_gaussian_blur(img)
        img = self._apply_noise(img)
        img = self._apply_jpeg(img)
        img = self._apply_downscale(img)
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
            if "segmentation" not in ann:
                continue
            seg = ann["segmentation"]
            if len(seg) == 0:
                continue
            seg_parts = seg if isinstance(seg[0], list) else [seg]
            for seg_poly in seg_parts:
                pts = np.array(seg_poly, dtype=np.float32).reshape(-1, 2)
                if pts.size == 0:
                    continue
                rect = cv2.minAreaRect(pts)
                box = cv2.boxPoints(rect)
                quad = order_vertices_clockwise(box)
                quad[:, 0] *= scale_x
                quad[:, 1] *= scale_y
                quads.append(quad)

        return img_resized, quads

    def list_augmentations(self):
        return [
            "original",
            "flip",
            "small_rotate",
            "perspective",
            "color_jitter",
            "blur",
            "noise",
            "jpeg",
            "shading",
            "gamma",
            "downscale",
            "negative",
        ]

    def preview_augmentation(self, idx: int, name: str):
        img, quads = self._load_image_and_quads(idx)
        name = name.lower()

        if name == "original":
            pass
        elif name == "flip":
            img, quads = self._apply_hflip(img, quads, force=True)
        elif name == "small_rotate":
            img, quads = self._apply_small_rotate(img, quads, force=True)
        elif name == "perspective":
            img, quads = self._apply_perspective(img, quads, force=True)
        elif name == "color_jitter":
            img = self._apply_color_jitter(img)
        elif name == "blur":
            img = self._apply_gaussian_blur(img, force=True)
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
                seg = ann.get("segmentation")
                if seg:
                    pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    if pts.shape[0] >= 4:
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
        img_resized, quads = self._load_image_and_quads(idx)
        img_resized, quads = self._apply_geometric_augments(img_resized, quads)
        img_resized = self._apply_photometric_augments(img_resized)

        # Convert quads to the format expected by visualization (N, 8)
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
            quad = order_vertices_clockwise(quad)
            shrunk = shrink_poly(quad, shrink_ratio=self.shrink_ratio)
            coords = shrunk * self.score_geo_scale
            rr, cc = skimage.draw.polygon(
                coords[:, 1], coords[:, 0], shape=(out_h, out_w)
            )
            if len(rr) == 0:
                continue
            score_map[rr, cc] = 1

            for i, (vx, vy) in enumerate(coords):
                geo_map[2 * i, rr, cc] = vx - cc
                geo_map[2 * i + 1, rr, cc] = vy - rr
        return score_map, geo_map
