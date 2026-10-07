"""Compare a local PP-OCR ONNX bundle against its source model on real images."""

import argparse
import json
from pathlib import Path


def validate(bundle, sources, images):
    import numpy as np
    import onnxruntime as ort
    from shapely.geometry import Polygon
    import torch
    from transformers import AutoModelForObjectDetection
    from manuscript import Pipeline
    from manuscript.detectors._ppocr import PPOCR
    from manuscript.utils import read_image, visualize_page

    torch.set_num_threads(2)
    model = AutoModelForObjectDetection.from_pretrained(sources, local_files_only=True).eval()
    exported = json.loads((bundle / "export_info.json").read_text())["files"]
    files = [bundle / next(name for name in exported if name.endswith("." + precision + ".onnx"))
             for precision in ["fp32", "fp16"]]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    sessions = [ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
                for path in files]
    detector = PPOCR(files[1])

    def compare_regions(reference, prediction):
        original = reference.blocks[0].lines[0].text_spans
        predicted = prediction.blocks[0].lines[0].text_spans
        ious, score_differences, used = [], [], set()
        for span in original:
            polygon = Polygon(span.polygon)
            overlaps = [polygon.intersection(Polygon(item.polygon)).area /
                        polygon.union(Polygon(item.polygon)).area if i not in used else -1
                        for i, item in enumerate(predicted)]
            if not overlaps:
                continue
            match = int(np.argmax(overlaps))
            if overlaps[match] < 0.8:
                continue
            used.add(match)
            ious.append(overlaps[match])
            score_differences.append(abs(span.detection_confidence - predicted[match].detection_confidence))
        result = {"reference_regions": len(original), "regions": len(predicted),
                  "matched_regions_at_iou_0_8": len(ious),
                  "region_recall": len(ious) / len(original) if original else 1.0,
                  "region_precision": len(ious) / len(predicted) if predicted else (1.0 if not original else 0.0),
                  "min_polygon_iou": min(ious, default=1.0),
                  "mean_polygon_iou": float(np.mean(ious)) if ious else 1.0,
                  "max_confidence_difference": max(score_differences, default=0.0),
                  "mean_confidence_difference": float(np.mean(score_differences)) if score_differences else 0.0}
        return result

    metrics = []
    for index, source in enumerate(images):
        image = read_image(source)
        data = detector._preprocess(image)
        with torch.inference_mode():
            reference = model(pixel_values=torch.from_numpy(data)).last_hidden_state.numpy()
        maps = [session.run(None, {"images": data})[0] for session in sessions]
        pages = [detector._postprocess(value, image.shape[:2]) for value in [reference, *maps]]
        row = {"image": str(source), "input_shape": list(data.shape),
               "source_regions": len(pages[0].blocks[0].lines[0].text_spans)}
        for label, values, page in zip(["fp32", "fp16"], maps, pages[1:]):
            row[label] = compare_regions(pages[0], page)
            row[label].update({"max_probability_difference": float(np.max(abs(values - reference))),
                               "mean_probability_difference": float(np.mean(abs(values - reference)))})
        row["fp16_vs_fp32"] = compare_regions(pages[1], pages[2])
        metrics.append(row)
        print(json.dumps(row), flush=True)
        if index == 0:
            annotated = visualize_page(image, pages[2], show_order=False, max_size=1280)
            annotated.convert("RGB").save(bundle / "preview_fp16.jpg", quality=90)
            pages[2].to_json(bundle / "sample_page_fp16.json")

    # Exercise the public class/session and Pipeline detector slot independently.
    page = Pipeline(detector=PPOCR(files[1]), layout=None, recognizer=None).predict(images[0])["page"]
    result = {"provider": "CPUExecutionProvider", "cuda_verified": False,
              "preprocessing": "official bundled defaults; reference receives the same preprocessed tensor",
              "region_verification": {"matching_iou": 0.8, "minimum_recall": 0.95,
                                      "minimum_precision": 0.95, "maximum_mean_confidence_difference": 0.01},
              "real_image_comparisons": metrics,
              "pipeline_regions": sum(len(line.text_spans) for block in page.blocks for line in block.lines)}
    (bundle / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in metrics:
        for label in ["fp32", "fp16", "fp16_vs_fp32"]:
            stats = row[label]
            if stats["region_recall"] < 0.95 or stats["region_precision"] < 0.95 or stats["mean_confidence_difference"] > 0.01:
                raise AssertionError({"image": row["image"], "precision": label, **stats})
    print("Pipeline inference verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-directory", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    validate(args.bundle_directory, args.source_directory, args.images)
