"""Check RF-DETR ONNX objects by class and IoU against source predictions."""

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from manuscript import Pipeline
from manuscript.detectors._rfdetr import RFDETR
from manuscript.utils import read_image, visualize_page


def compare(reference, predicted, min_iou=0.8):
    candidates = []
    for i, source in enumerate(reference):
        a = source["bbox"]
        for j, result in enumerate(predicted):
            if result["class_id"] != source["class_id"]:
                continue
            b = result["bbox"]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1])
            )
            union = (
                (a[2] - a[0]) * (a[3] - a[1])
                + (b[2] - b[0]) * (b[3] - b[1])
                - intersection
            )
            iou = intersection / union if union > 0 else 0
            if iou >= min_iou:
                candidates.append((iou, i, j))
    used_source, used_result, ious, confidences = set(), set(), [], []
    for iou, i, j in sorted(candidates, reverse=True):
        if i in used_source or j in used_result:
            continue
        used_source.add(i)
        used_result.add(j)
        ious.append(iou)
        confidences.append(abs(reference[i]["confidence"] - predicted[j]["confidence"]))
    return {
        "source_objects": len(reference),
        "objects": len(predicted),
        "matched": len(ious),
        "recall": len(ious) / len(reference) if reference else 1.0,
        "precision": (
            len(ious) / len(predicted) if predicted else (1.0 if not reference else 0.0)
        ),
        "mean_iou": float(np.mean(ious)) if ious else 1.0,
        "min_iou": min(ious, default=1.0),
        "mean_confidence_difference": (
            float(np.mean(confidences)) if confidences else 0.0
        ),
        "source_classes": dict(Counter(row["class_id"] for row in reference)),
        "classes": dict(Counter(row["class_id"] for row in predicted)),
    }


def validate(bundle, images):
    info = json.loads((bundle / "export_info.json").read_text())
    paths = [
        bundle
        / next(name for name in info["files"] if name.endswith(f".{precision}.onnx"))
        for precision in ("fp32", "fp16")
    ]
    models = [RFDETR(path) for path in paths]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    for model in models:
        # Use bounded CPU threads during artifact verification.
        model.onnx_session = ort.InferenceSession(
            str(model.weights), sess_options=options, providers=["CPUExecutionProvider"]
        )
        model._input_hw = tuple(model.onnx_session.get_inputs()[0].shape[2:])
    comparisons = []
    for index, path in enumerate(images):
        image = read_image(path)
        raw = np.load(bundle / ("source_" + Path(path).stem + ".npz"))
        source = models[0]._postprocess(
            raw["pred_boxes"],
            raw["pred_logits"],
            image.shape[:2],
            return_raw=True,
            masks=raw["pred_masks"] if "pred_masks" in raw else None,
        )
        results = [model.predict(image, return_raw=True) for model in models]
        row = {"image": str(path)}
        for label, result in zip(("fp32", "fp16"), results):
            row[label] = compare(source["detections"], result["detections"])
            row[label]["text_objects"] = sum(
                len(line.text_spans)
                for block in result["page"].blocks
                for line in block.lines
            )
            if "pred_masks" in raw:
                model = models[0] if label == "fp32" else models[1]
                actual_masks = model.onnx_session.run(
                    [model.masks_output_name],
                    {model.input_name: model._preprocess(image)},
                )[0]
                ious = []
                for detection in source["detections"]:
                    a = np.asarray(detection["bbox"])
                    candidates = [
                        d
                        for d in result["detections"]
                        if d["class_id"] == detection["class_id"]
                    ]
                    if not candidates:
                        continue
                    match = min(candidates, key=lambda d: np.abs(a - d["bbox"]).sum())
                    left = raw["pred_masks"][0, detection["query_index"]] > 0
                    right = actual_masks[0, match["query_index"]] > 0
                    intersection = np.logical_and(left, right).sum()
                    union = np.logical_or(left, right).sum()
                    ious.append(float(intersection / union) if union else 1.0)
                row[label]["mask_iou"] = {
                    "samples": len(ious),
                    "mean": float(np.mean(ious)),
                    "minimum": min(ious),
                }
                assert min(ious) >= 0.9, row[label]["mask_iou"]
                assert all(
                    d["geometry_source"] == "mask_polygon" for d in result["detections"]
                )
        row["fp16_vs_fp32"] = compare(
            results[0]["detections"], results[1]["detections"]
        )
        comparisons.append(row)
        print(json.dumps(row), flush=True)
        if index == 0:
            visualize_page(
                image, results[1]["page"], show_order=False, max_size=1280
            ).convert("RGB").save(bundle / "preview_fp16.jpg", quality=90)
            full = {**results[1], "page": results[1]["page"].model_dump(mode="json")}
            (bundle / "full_result_sample.json").write_text(
                json.dumps(full, ensure_ascii=False, indent=2) + "\n"
            )
        # Every matching query/class pair must retain the documented label.
        assert all(
            row["class_name"]
            == models[0].class_names.get(row["class_id"], str(row["class_id"]))
            for row in results[0]["detections"]
        )
    page = Pipeline(detector=models[1], layout=None, recognizer=None).predict(
        images[0]
    )["page"]
    page.to_json(bundle / "pipeline_sample.json")
    report = {
        "provider": "CPUExecutionProvider",
        "gpu_verified": False,
        "mode": (
            "boxes, classes and mask contours"
            if models[0].masks_output_name
            else "boxes and classes"
        ),
        "matching": {"class_id_must_match": True, "minimum_iou": 0.8},
        "threshold": models[0].score_thresh,
        "pipeline_verified": True,
        "comparisons": comparisons,
    }
    (bundle / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    for row in comparisons:
        for label in ("fp32", "fp16"):
            result = row[label]
            if (
                result["recall"] < 0.95
                or result["precision"] < 0.95
                or result["mean_confidence_difference"] > 0.02
            ):
                raise AssertionError(
                    {"image": row["image"], "precision": label, **result}
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--images", nargs="+", required=True)
    args = parser.parse_args()
    validate(args.bundle, args.images)
