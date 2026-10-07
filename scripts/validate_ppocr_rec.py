"""Validate local recognition bundles and exercise their public Pipeline API."""

import argparse
import json
from pathlib import Path

import numpy as np

from manuscript import Pipeline
from manuscript.data import Block, Line, Page, TextSpan
from manuscript.recognizers import PPOCRRec
from manuscript.utils import read_image


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def validate(root):
    sample_dir = root / "samples"
    if not sample_dir.is_dir():
        sample_dir = root / "примеры"
    samples = sorted(sample_dir.glob("*.png"))
    if not samples:
        raise ValueError("No validation images found in samples/ or примеры/")
    summary = []
    for bundle in sorted(root.iterdir()):
        if not (bundle / "export_info.json").exists():
            continue
        source = json.loads((bundle / "export_info.json").read_text())
        paths = [bundle / next(name for name in source["files"] if name.endswith(f".{precision}.onnx"))
                 for precision in ("fp32", "fp16")]
        models = [PPOCRRec(str(path), device="cpu", rotate_threshold=None) for path in paths]
        predictions = [model._predict_word_images(samples, batch_size=1) for model in models]
        batch_predictions = [model._predict_word_images(samples, batch_size=4) for model in models]
        rows = []
        for index, path in enumerate(samples):
            a, b = [result[index] for result in predictions]
            rows.append({"image": path.name, "fp32": a, "fp16": b,
                         "text_equal": a["text"] == b["text"],
                         "edit_distance": edit_distance(a["text"], b["text"]),
                         "confidence_difference": abs(a["confidence"] - b["confidence"]),
                         "batch4": [result[index] for result in batch_predictions]})
        # Pipeline calls the public predict(Page, image), without needing a
        # detector download or registry entry. Preserve the caller's input Page.
        class WholeImageDetector:
            def predict(self, image):
                array = read_image(image)
                height, width = array.shape[:2]
                return Page(blocks=[Block(lines=[Line(text_spans=[TextSpan(
                    polygon=[(0, 0), (width, 0), (width, height), (0, height)],
                    detection_confidence=1,
                )])])])

        pipeline = Pipeline(detector=WholeImageDetector(), layout=None, recognizer=models[1])
        pipeline_page = pipeline.predict(samples[0])["page"]
        span = pipeline_page.blocks[0].lines[0].text_spans[0]
        assert span.text == predictions[1][0]["text"]
        assert span.recognition_confidence is not None and 0 <= span.recognition_confidence <= 1
        original = WholeImageDetector().predict(samples[0])
        recognized = models[0].predict(original, image=samples[0])
        assert original.blocks[0].lines[0].text_spans[0].text is None
        assert recognized.blocks[0].lines[0].text_spans[0].text == predictions[0][0]["text"]
        pipeline_page.to_json(bundle / "pipeline_sample.json")
        report = {
            "provider": "CPUExecutionProvider", "gpu_verified": False,
            "comparison": "FP16 against FP32; not recognition accuracy against ground truth",
            "samples": len(rows), "equal_texts": sum(row["text_equal"] for row in rows),
            "changed_characters": sum(row["edit_distance"] for row in rows),
            "fp32_characters": sum(len(row["fp32"]["text"]) for row in rows),
            "pipeline_verified": True, "page_copy_verified": True, "results": rows,
        }
        (bundle / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        summary.append({"model": bundle.name, **{key: report[key] for key in (
            "samples", "equal_texts", "changed_characters", "fp32_characters", "pipeline_verified")}})
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    (root / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundles", type=Path)
    validate(parser.parse_args().bundles)
