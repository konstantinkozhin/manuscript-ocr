import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import onnxruntime as ort

from common_metrics import evaluate_dataset


# =============================================================================
# PATHS / PARAMS (explicit)
# =============================================================================

SRC_PATH = r"C:\Users\USER\manuscript-ocr\src"
EAST_WEIGHTS_PATH = r"C:\Users\USER\Desktop\east_resnet50\checkpoints\best_dice.onnx"
OUTPUT_DIR = r"C:\Users\USER\EAST_TRBA_simple_exp\benchmark_detection\benchmark_results_east_g2_"

ARCHIVES020525_IMAGES = r"C:\shared\data0205\data02065\Archives020525\test_images"
ARCHIVES020525_ANN = r"C:\shared\data0205\data02065\Archives020525\test.json"

SCHOOL_NOTEBOOKS_RU_IMAGES = r"C:\shared\data0205\data02065\school_notebooks_RU\test_images"
SCHOOL_NOTEBOOKS_RU_ANN = r"C:\shared\data0205\data02065\school_notebooks_RU\test.json"

IAM_IMAGES = r"C:\shared\data0205\data02065\IAM\test_images"
IAM_ANN = r"C:\shared\data0205\data02065\IAM\test.json"


DATASETS = [
    {"name": "Archives020525", "folder": ARCHIVES020525_IMAGES, "annotations": ARCHIVES020525_ANN},
    {"name": "school_notebooks_RU", "folder": SCHOOL_NOTEBOOKS_RU_IMAGES, "annotations": SCHOOL_NOTEBOOKS_RU_ANN},
    {"name": "IAM", "folder": IAM_IMAGES, "annotations": IAM_ANN},
]

CONFIG = {
    "target_size": 1408,
    "score_thresh": 0.6,
    "warmup_runs": 3,
    "cpu_only": False,
    "gpu_only": True,
}

CUDA_ORT_AVAILABLE = "CUDAExecutionProvider" in ort.get_available_providers()


# =============================================================================
# INIT
# =============================================================================

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from manuscript.detectors import EAST


# =============================================================================
# HELPERS
# =============================================================================

def get_image_files(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = []
    base = Path(folder)
    for ext in exts:
        files.extend(base.glob(f"*{ext}"))
        files.extend(base.glob(f"*{ext.upper()}"))
    return sorted(list(dict.fromkeys(map(str, files))))


def load_ground_truth(annotation_file: str) -> Dict[str, List[tuple]]:
    with open(annotation_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    ground_truths: Dict[str, List[tuple]] = {}

    for ann in coco["annotations"]:
        filename = id_to_name.get(ann["image_id"])
        if not filename:
            continue
        seg = ann.get("segmentation")
        if not seg:
            continue
        seg_parts = seg if isinstance(seg[0], list) else [seg]
        for seg_poly in seg_parts:
            if len(seg_poly) < 8:
                continue
            pts = np.array(seg_poly, dtype=np.float32).reshape(-1, 2)
            x_min = float(np.min(pts[:, 0]))
            y_min = float(np.min(pts[:, 1]))
            x_max = float(np.max(pts[:, 0]))
            y_max = float(np.max(pts[:, 1]))
            ground_truths.setdefault(filename, []).append((x_min, y_min, x_max, y_max))

    return ground_truths


def get_memory_usage() -> Dict[str, float]:
    import psutil

    ram_mb = psutil.Process().memory_info().rss / 1024 / 1024
    return {"ram_mb": ram_mb, "gpu_mb": None}


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_device(image_files: List[str], device: str, collect_predictions: bool = False) -> Dict[str, Any]:
    gc.collect()

    detector = EAST(
        weights=EAST_WEIGHTS_PATH,
        device=device,
        target_size=CONFIG["target_size"],
        score_thresh=CONFIG["score_thresh"],
    )
    mem_after_load = get_memory_usage()

    for img_path in image_files[: min(CONFIG["warmup_runs"], len(image_files))]:
        _ = detector.predict(img_path)

    inference_times = []
    detection_counts = []
    predictions = {} if collect_predictions else None
    peak_memory = dict(mem_after_load)

    for img_path in image_files:
        start_time = time.time()
        result = detector.predict(img_path)
        inference_times.append(time.time() - start_time)

        num_detections = 0
        boxes = []
        for block in result["page"].blocks:
            for line in getattr(block, "lines", []):
                words = getattr(line, "words", [])
                num_detections += len(words)
                for word in words:
                    box = None
                    if hasattr(word, "polygon") and word.polygon:
                        xs = [pt[0] for pt in word.polygon]
                        ys = [pt[1] for pt in word.polygon]
                        box = (min(xs), min(ys), max(xs), max(ys))
                    elif hasattr(word, "bbox") and word.bbox:
                        box = tuple(word.bbox)
                    elif hasattr(word, "geometry") and hasattr(word.geometry, "bbox"):
                        box = tuple(word.geometry.bbox)
                    if box:
                        boxes.append(box)

        detection_counts.append(num_detections)
        if collect_predictions:
            predictions[Path(img_path).name] = boxes

        current_mem = get_memory_usage()
        peak_memory["ram_mb"] = max(peak_memory["ram_mb"], current_mem["ram_mb"])
        if peak_memory["gpu_mb"] is not None and current_mem["gpu_mb"] is not None:
            peak_memory["gpu_mb"] = max(peak_memory["gpu_mb"], current_mem["gpu_mb"])

    times = np.array(inference_times)
    dets = np.array(detection_counts)

    stats = {
        "device": device,
        "num_images": len(image_files),
        "target_size": CONFIG["target_size"],
        "mean_time_ms": float(np.mean(times) * 1000),
        "median_time_ms": float(np.median(times) * 1000),
        "std_time_ms": float(np.std(times) * 1000),
        "min_time_ms": float(np.min(times) * 1000),
        "max_time_ms": float(np.max(times) * 1000),
        "total_time_s": float(np.sum(times)),
        "throughput_fps": float(len(image_files) / np.sum(times)),
        "mean_detections": float(np.mean(dets)) if len(dets) else 0.0,
        "total_detections": int(np.sum(dets)) if len(dets) else 0,
        "ram_after_load_mb": mem_after_load["ram_mb"],
        "ram_peak_mb": peak_memory["ram_mb"],
        "ram_delta_mb": peak_memory["ram_mb"] - mem_after_load["ram_mb"],
    }

    if device == "cuda":
        stats["gpu_after_load_mb"] = mem_after_load["gpu_mb"]
        stats["gpu_peak_mb"] = peak_memory["gpu_mb"]
        stats["gpu_delta_mb"] = (
            peak_memory["gpu_mb"] - mem_after_load["gpu_mb"]
            if peak_memory["gpu_mb"] is not None and mem_after_load["gpu_mb"] is not None
            else None
        )

    if collect_predictions:
        stats["predictions"] = predictions

    return stats


def save_results(cpu_stats: Dict[str, Any], gpu_stats: Dict[str, Any], output_file: Path, dataset_name: str) -> None:
    out = {"dataset_name": dataset_name, "cpu": cpu_stats, "gpu": gpu_stats}
    if cpu_stats and "predictions" in cpu_stats:
        del cpu_stats["predictions"]
    if gpu_stats and "predictions" in gpu_stats:
        del gpu_stats["predictions"]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        print(f"\n### DATASET: {ds['name']}")
        folder = ds["folder"]
        annotations = ds["annotations"]

        if not Path(folder).exists() or not Path(annotations).exists():
            print(f"Skip: missing path for {ds['name']}")
            continue

        image_files = get_image_files(folder)
        if not image_files:
            print(f"Skip: no images in {folder}")
            continue

        ground_truths = load_ground_truth(annotations)
        cpu_stats = None
        gpu_stats = None

        if not CONFIG["gpu_only"]:
            cpu_stats = benchmark_device(image_files, "cpu", collect_predictions=True)
            cpu_stats["accuracy_metrics"] = evaluate_dataset(cpu_stats["predictions"], ground_truths)

        if CUDA_ORT_AVAILABLE and not CONFIG["cpu_only"]:
            gpu_stats = benchmark_device(image_files, "cuda", collect_predictions=True)
            gpu_stats["accuracy_metrics"] = evaluate_dataset(gpu_stats["predictions"], ground_truths)

        output_file = Path(OUTPUT_DIR) / f"{ds['name']}_east.json"
        save_results(cpu_stats, gpu_stats, output_file, ds["name"])
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
