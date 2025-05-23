import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon
from manuscript.detectors import EASTInfer

# ----------------------------------------
# Пути к файлам и параметры
# ----------------------------------------
INPUT_JSON = r"C:\data0205\Archives020525\test.json"
IMAGES_DIR = r"C:\data0205\Archives020525\test_images"
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
# 4) Метрика poly-IoU и F1
# ---------------------------
def poly_iou(segA, segB):
    A = Polygon(np.array(segA).reshape(-1, 2))
    B = Polygon(np.array(segB).reshape(-1, 2))
    if not A.is_valid or not B.is_valid:
        return 0.0
    inter = A.intersection(B).area
    union = A.union(B).area
    return inter / union if union > 0 else 0.0


def compute_f1(preds, thresh):
    used = {iid: [False] * len(gt_segs.get(iid, [])) for iid in processed_ids}
    tp = fp = 0
    for p in preds:
        best_iou, bj = 0, -1
        for j, gt in enumerate(gt_segs.get(p["image_id"], [])):
            if used[p["image_id"]][j]:
                continue
            iou = poly_iou(p["segmentation"], gt)
            if iou > best_iou:
                best_iou, bj = iou, j
        if best_iou >= thresh:
            tp += 1
            used[p["image_id"]][bj] = True
        else:
            fp += 1
    total_gt = sum(len(v) for v in gt_segs.values())
    fn = total_gt - tp
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0


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

                # F1 @ текущий iou_th
                f1_at_05 = compute_f1(preds, 0.5)
                # F1 среднее по 0.50–0.95
                iou_vals = np.arange(0.50, 0.95 + 1e-9, 0.05)
                f1_list = [compute_f1(preds, t) for t in iou_vals]
                f1_avg = float(np.mean(f1_list))

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
