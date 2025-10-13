import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon
from manuscript.detectors import EASTInfer
from manuscript.detectors.east.utils import (compute_f1_metrics)

# ----------------------------------------
# Пути к файлам и параметры
# ----------------------------------------
INPUT_JSON = r"C:\Users\USER\Desktop\data02065\Archives020525\test.json"
IMAGES_DIR = r"C:\Users\USER\Desktop\data02065\Archives020525\test_images"
SAMPLE_COUNT = 5  # сколько изображений выбрать равномерно

# ---------------------------
# 1) Загрузка полного JSON (images+annotations)
# ---------------------------
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco.get("annotations", [])

# Словарь id → filename
id2fname = {img["id"]: img["file_name"] for img in images}

# ---------------------------
# 2) Равномерная выборка SAMPLE_COUNT изображений
# ---------------------------
total = len(images)
# равномерные индексы от 0 до total-1
idxs = np.linspace(0, total - 1, SAMPLE_COUNT, dtype=int)
processed_ids = [images[i]["id"] for i in idxs]

# ---------------------------
# 3) Сбор GT-полигонов для выбранных изображений
# ---------------------------
gt_segs = {}
for ann in annotations:
    iid = ann["image_id"]
    if iid in processed_ids:
        gt_segs.setdefault(iid, []).append(ann["segmentation"][0])





# ---------------------------
# 5) Сетка гиперпараметров
# ---------------------------
shrink_ratios = np.arange(0.1, 0.5 + 1e-9, 0.1)
score_threshs = np.arange(0.6, 0.9 + 1e-9, 0.1)
iou_thresholds = np.arange(0.0, 0.3 + 1e-9, 0.1)
target_sizes = [1024 + 256 + 256]

results = []

# ---------------------------
# 6) Grid search
# ---------------------------
for target in target_sizes:
    for shrink in shrink_ratios:
        for score_th in score_threshs:
            for iou_th in iou_thresholds:
                det = EASTInfer(
                    shrink_ratio=shrink,
                    score_thresh=score_th,
                    iou_threshold=iou_th,
                    target_size=target,
                )
                preds = []
                for iid in processed_ids:
                    fn = id2fname[iid]
                    path = os.path.join(IMAGES_DIR, fn)
                    if not os.path.exists(path):
                        continue

                    page, vis = det.infer(path, vis=True)
                    w0, h0 = Image.open(path).size
                    vw, vh = (
                        (vis.shape[1], vis.shape[0])
                        if isinstance(vis, np.ndarray)
                        else (vis.width, vis.height)
                    )
                    sx, sy = w0 / vw, h0 / vh

                    for block in page.blocks:
                        for word in block.words:
                            seg = [
                                c for px, py in word.polygon for c in (px * sx, py * sy)
                            ]
                            preds.append(
                                {
                                    "image_id": iid,
                                    "segmentation": seg,
                                    "score": getattr(word, "score", 1.0),
                                }
                            )

                preds.sort(key=lambda x: x["score"], reverse=True)

                f1_at_05, f1_avg = compute_f1_metrics(preds, gt_segs, processed_ids)

                print(
                    {
                        "sample_count": SAMPLE_COUNT,
                        "target_size": target,
                        "shrink_ratio": float(shrink),
                        "score_thresh": float(score_th),
                        "iou_threshold": float(iou_th),
                        "f1@0.5": f1_at_05,
                        "f1@0.50-0.95": f1_avg,
                    }
                )

                results.append(
                    {
                        "sample_count": SAMPLE_COUNT,
                        "target_size": target,
                        "shrink_ratio": float(shrink),
                        "score_thresh": float(score_th),
                        "iou_threshold": float(iou_th),
                        "f1@0.5": f1_at_05,
                        "f1@0.50-0.95": f1_avg,
                    }
                )

# ---------------------------
# 7) Сохранение результатов
# ---------------------------
df = pd.DataFrame(results)
df.to_csv("grid_search_results.csv", index=False)
print(df)
