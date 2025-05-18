import os
import json
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------------------
# 1) Пути к JSON
# ----------------------------------------
GT_JSON = r"C:\data0205\ICDAR2015\test.json"
PRED_JSON = r"C:\Users\pasha\OneDrive\Рабочий стол\ICDAR2015_test.json"


def load_gt(gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_coco = json.load(f)
    gt_segs = defaultdict(list)
    for ann in gt_coco["annotations"]:
        seg = ann.get("segmentation", [])
        if seg:
            gt_segs[ann["image_id"]].append(seg[0])
    return gt_segs


def load_preds(pred_path):
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    preds_list = data.get("annotations", data)
    preds = []
    for p in preds_list:
        seg = p.get("segmentation", [])
        if not seg:
            continue
        preds.append(
            {
                "image_id": p["image_id"],
                "segmentation": seg[0],
                "score": p.get("score", 1.0),
            }
        )
    return preds


def poly_iou(polyA_xy, polyB_xy):
    polyA = Polygon(np.array(polyA_xy).reshape(-1, 2))
    polyB = Polygon(np.array(polyB_xy).reshape(-1, 2))
    if not polyA.is_valid or not polyB.is_valid:
        return 0.0
    inter = polyA.intersection(polyB).area
    union = polyA.union(polyB).area
    return inter / union if union > 0 else 0.0


def poly_dice(polyA_xy, polyB_xy):
    polyA = Polygon(np.array(polyA_xy).reshape(-1, 2))
    polyB = Polygon(np.array(polyB_xy).reshape(-1, 2))
    if not polyA.is_valid or not polyB.is_valid:
        return 0.0
    inter = polyA.intersection(polyB).area
    return (
        2 * inter / (polyA.area + polyB.area) if (polyA.area + polyB.area) > 0 else 0.0
    )


def compute_ap_for_threshold(args):
    gt_segs, preds, t = args
    total = sum(len(v) for v in gt_segs.values())
    if total == 0:
        return t, 0.0

    preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)
    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))
    used = {img: np.zeros(len(gt_segs[img]), dtype=bool) for img in gt_segs}

    for i, p in enumerate(preds_sorted):
        best_iou, best_j = 0.0, -1
        for j, gt_poly in enumerate(gt_segs.get(p["image_id"], [])):
            if used[p["image_id"]][j]:
                continue
            iou_val = poly_iou(p["segmentation"], gt_poly)
            if iou_val > best_iou:
                best_iou, best_j = iou_val, j

        if best_iou >= t:
            tp[i] = 1
            used[p["image_id"]][best_j] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / total
    prec = tp_cum / (tp_cum + fp_cum + 1e-6)

    rec = np.concatenate(([0], rec, [1]))
    prec = np.concatenate(([0], prec, [0]))
    for k in range(len(prec) - 2, -1, -1):
        prec[k] = max(prec[k], prec[k + 1])
    idxs = np.where(rec[1:] != rec[:-1])[0]
    ap = np.sum((rec[idxs + 1] - rec[idxs]) * prec[idxs + 1])
    return t, ap


def main():
    print("Загрузка GT и предсказаний...")
    gt_segs = load_gt(GT_JSON)
    preds = load_preds(PRED_JSON)
    print(
        f"GT полигоны: {sum(len(v) for v in gt_segs.values())}, predictions: {len(preds)}"
    )

    # AP@IoU=0.50:0.95
    # thresholds = np.arange(0.50, 0.96, 0.05)
    # args = [(gt_segs, preds, float(t)) for t in thresholds]

    # print("Вычисление AP по порогам IoU:")
    # aps = {}
    # with ProcessPoolExecutor() as executor:
    #     futures = {executor.submit(compute_ap_for_threshold, a): a[2] for a in args}
    #     for fut in tqdm(
    #         as_completed(futures), total=len(futures), desc="IoU thresholds"
    #     ):
    #         t, ap_val = fut.result()
    #         aps[t] = ap_val
    #         print(f"  IoU={t:.2f}: AP={ap_val:.4f}")

    # mAP = np.mean(list(aps.values()))
    # print(f"\nmAP (0.50–0.95) = {mAP:.4f}")

    # # Средние poly-IoU и poly-Dice (TP@0.50)
    # print("\nСредние poly-IoU и poly-Dice (TP@0.50)...")
    # used = {img: np.zeros(len(gt_segs[img]), dtype=bool) for img in gt_segs}
    # iou_scores, dice_scores = [], []

    # for p in sorted(preds, key=lambda x: x["score"], reverse=True):
    #     best_iou, best_j = 0.0, -1
    #     for j, gt_poly in enumerate(gt_segs.get(p["image_id"], [])):
    #         if used[p["image_id"]][j]:
    #             continue
    #         iou_val = poly_iou(p["segmentation"], gt_poly)
    #         if iou_val > best_iou:
    #             best_iou, best_j = iou_val, j

    #     if best_iou >= 0.5:
    #         used[p["image_id"]][best_j] = True
    #         iou_scores.append(best_iou)
    #         dice_scores.append(
    #             poly_dice(p["segmentation"], gt_segs[p["image_id"]][best_j])
    #         )

    # if dice_scores:
    #     print(f"Average poly-IoU  (TP@0.50): {np.mean(iou_scores):.4f}")
    #     print(f"Average poly-Dice (TP@0.50): {np.mean(dice_scores):.4f}")
    # else:
    #     print("No TP@0.50 → poly-IoU/Dice undefined")

    # H-mean (F1) @ IoU=0.50
    print("\nВычисление Precision, Recall, F1 @ IoU=0.50...")
    used_gt = {img: np.zeros(len(gt_segs[img]), dtype=bool) for img in gt_segs}
    tp = 0
    fp = 0
    for p in preds:
        matched = False
        for j, gt_poly in enumerate(gt_segs.get(p["image_id"], [])):
            if used_gt[p["image_id"]][j]:
                continue
            if poly_iou(p["segmentation"], gt_poly) >= 0.5:
                tp += 1
                used_gt[p["image_id"]][j] = True
                matched = True
                break
        if not matched:
            fp += 1
    total_gt = sum(len(v) for v in gt_segs.values())
    fn = total_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(f"Precision @ IoU=0.50: {precision:.4f}")
    print(f"Recall    @ IoU=0.50: {recall:.4f}")
    print(f"F1-score  @ IoU=0.50: {f1_score:.4f}")


if __name__ == "__main__":
    main()
