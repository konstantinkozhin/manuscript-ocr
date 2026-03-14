"""
Prepare school_notebooks_RU dataset for OCR recognition training.

Reads annotations_train/val/test.json, crops word images from pages,
and writes ordered CSV files (one per split) suitable for context-aware training.

Usage:
    python scripts/prepare_school_notebooks.py \
        --src school_notebooks_RU \
        --dst school_notebooks_RU/recognition
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np


def find_image(file_name: str, src: str) -> str | None:
    """Locate an image file across train_images/ and test_images/ folders."""
    for folder in ("train_images", "test_images"):
        p = os.path.join(src, folder, file_name)
        if os.path.isfile(p):
            return p
    return None


def polygon_to_bbox(seg: list[float]) -> tuple[int, int, int, int]:
    """Convert flat polygon [x0,y0,x1,y1,...] to axis-aligned bbox (x,y,w,h)."""
    xs = seg[0::2]
    ys = seg[1::2]
    x_min, x_max = int(np.floor(min(xs))), int(np.ceil(max(xs)))
    y_min, y_max = int(np.floor(min(ys))), int(np.ceil(max(ys)))
    return x_min, y_min, x_max - x_min, y_max - y_min


def crop_word(img: np.ndarray, seg: list[float]) -> np.ndarray | None:
    """Crop a word from the page using polygon bounding box with polygon mask."""
    x, y, w, h = polygon_to_bbox(seg)
    if w <= 0 or h <= 0:
        return None

    ih, iw = img.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, iw - x)
    h = min(h, ih - y)
    if w <= 0 or h <= 0:
        return None

    crop = img[y : y + h, x : x + w].copy()

    # Apply polygon mask to zero out background
    pts = np.array(list(zip(seg[0::2], seg[1::2])), dtype=np.float32)
    pts[:, 0] -= x
    pts[:, 1] -= y
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    if len(crop.shape) == 3:
        crop[mask == 0] = 255  # white background where masked
    else:
        crop[mask == 0] = 255

    return crop


def order_annotations(anns: list[dict]) -> list[dict]:
    """
    Order annotations in reading order on a single page:
    1. Group by group_id (same group = same line)
    2. Sort lines by average Y (top to bottom)
    3. Within each line, sort words by centroid X (left to right)
    4. Annotations without group_id treated as standalone lines, sorted by Y
    """
    grouped = {}  # group_id -> list of (cx, cy, ann)
    ungrouped = []  # (cx, cy, ann) — no group_id

    for a in anns:
        seg = a["segmentation"][0]
        xs = seg[0::2]
        ys = seg[1::2]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        gid = a.get("group_id")
        if gid is not None:
            grouped.setdefault(gid, []).append((cx, cy, a))
        else:
            ungrouped.append((cx, cy, a))

    # Build ordered list of lines
    lines = []

    # Grouped lines: sort by avg Y, then words in each line by X
    for gid, words in grouped.items():
        avg_y = sum(w[1] for w in words) / len(words)
        words.sort(key=lambda w: w[0])  # sort by X within line
        lines.append((avg_y, [w[2] for w in words]))

    # Ungrouped: each word is its own line
    for cx, cy, a in ungrouped:
        lines.append((cy, [a]))

    # Sort all lines by Y
    lines.sort(key=lambda l: l[0])

    # Flatten
    result = []
    for _, words in lines:
        result.extend(words)
    return result


def process_split(
    ann_path: str,
    src: str,
    img_dir: str,
    split_name: str,
) -> list[tuple[str, str]]:
    """
    Process one split (train/val/test).

    Returns list of (crop_filename, translation) in reading order.
    """
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_image = {img["id"]: img for img in data["images"]}

    # Group text annotations by image_id
    # Only category 0 (pupil_text) with non-empty translation
    by_image: dict[int, list[dict]] = {}
    for a in data["annotations"]:
        if a["category_id"] != 0:
            continue
        trans = a.get("attributes", {}).get("translation")
        if not trans:
            continue
        by_image.setdefault(a["image_id"], []).append(a)

    rows = []
    skipped = 0
    total_words = 0
    crop_counter = 0

    for image_id in sorted(by_image.keys()):
        anns = by_image[image_id]
        img_info = id_to_image[image_id]
        file_name = img_info["file_name"]

        img_path = find_image(file_name, src)
        if img_path is None:
            print(f"  WARNING: image not found: {file_name}")
            continue

        page_img = cv2.imread(img_path)
        if page_img is None:
            print(f"  WARNING: cannot read image: {img_path}")
            continue

        ordered = order_annotations(anns)

        for ann in ordered:
            seg = ann["segmentation"][0]
            trans = ann["attributes"]["translation"]
            total_words += 1

            crop = crop_word(page_img, seg)
            if crop is None:
                skipped += 1
                continue

            crop_fname = f"{split_name}_{image_id}_{crop_counter}.jpg"
            crop_counter += 1
            out_path = os.path.join(img_dir, crop_fname)
            cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            rows.append((crop_fname, trans))

    print(
        f"  {split_name}: {total_words} words, {skipped} skipped, "
        f"{len(rows)} in CSV"
    )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Prepare school_notebooks_RU for OCR recognition training"
    )
    parser.add_argument(
        "--src",
        default="school_notebooks_RU",
        help="Path to school_notebooks_RU root",
    )
    parser.add_argument(
        "--dst",
        default="school_notebooks_RU/recognition",
        help="Output directory for crops and CSVs",
    )
    args = parser.parse_args()

    src = args.src
    dst = args.dst
    img_dir = os.path.join(dst, "img")
    os.makedirs(img_dir, exist_ok=True)

    for split in ("train", "val", "test"):
        ann_path = os.path.join(src, f"annotations_{split}.json")
        if not os.path.isfile(ann_path):
            print(f"Skipping {split}: {ann_path} not found")
            continue

        print(f"Processing {split}...")
        rows = process_split(ann_path, src, img_dir, split)

        csv_path = os.path.join(dst, f"{split}.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "text"])
            for fname, text in rows:
                writer.writerow([fname, text])

        print(f"  Wrote {csv_path} ({len(rows)} rows)")

    print(f"\nDone. Output: {dst}/")


if __name__ == "__main__":
    main()
