import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import skimage.draw
from .utils import quad_to_rbox


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
        transform=None,
    ):
        self.images_folder = images_folder
        self.target_size = target_size
        self.score_geo_scale = score_geo_scale

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]
            )
        else:
            self.transform = transform

        # load COCO annotations
        with open(coco_annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.images_info = {img["id"]: img for img in data["images"]}
        self.image_ids = list(self.images_info.keys())
        self.annots = {}
        for ann in data["annotations"]:
            self.annots.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        info = self.images_info[image_id]
        path = os.path.join(self.images_folder, info["file_name"])
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # resize
        img_resized = cv2.resize(img, (self.target_size, self.target_size))

        # scale annotations
        anns = self.annots.get(image_id, [])
        quads = []
        for ann in anns:
            if "segmentation" not in ann:
                continue
            seg = ann["segmentation"]
            pts = np.array(seg, dtype=np.float32).reshape(-1, 2)

            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)
            quad = order_vertices_clockwise(box)

            quad[:, 0] *= self.target_size / info["width"]
            quad[:, 1] *= self.target_size / info["height"]
            quads.append(quad)

        # generate maps
        score_map, geo_map = self.compute_quad_maps(quads)
        rboxes = np.stack([quad_to_rbox(q.flatten()) for q in quads], axis=0).astype(
            np.float32
        )

        img_tensor = self.transform(img_resized)
        target = {
            "score_map": torch.tensor(score_map).unsqueeze(0),
            "geo_map": torch.tensor(geo_map),
            "rboxes": torch.tensor(rboxes),
        }
        return img_tensor, target

    def compute_quad_maps(self, quads):
        """
        quads: list of (4,2) arrays
        returns score_map (H',W') and geo_map (8,H',W')
        """
        out_h = int(self.target_size * self.score_geo_scale)
        out_w = int(self.target_size * self.score_geo_scale)
        score_map = np.zeros((out_h, out_w), dtype=np.float32)
        geo_map = np.zeros((8, out_h, out_w), dtype=np.float32)
        for quad in quads:
            shrunk = shrink_poly(order_vertices_clockwise(quad), shrink_ratio=0.3)
            coords = shrunk * self.score_geo_scale
            rr, cc = skimage.draw.polygon(
                coords[:, 1], coords[:, 0], shape=(out_h, out_w)
            )
            score_map[rr, cc] = 1
            for i, (vx, vy) in enumerate(coords):
                geo_map[2 * i, rr, cc] = vx - cc
                geo_map[2 * i + 1, rr, cc] = vy - rr
        return score_map, geo_map
