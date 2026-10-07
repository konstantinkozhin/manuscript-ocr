"""Run: env/bin/python scripts/demo_rfdetr_kraken.py [path/to/image]."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manuscript import Pipeline
from manuscript.detectors._rfdetr import RFDETR
from manuscript.recognizers import PPOCRRec


# Change these paths to test your image or another local model bundle.
IMAGE_PATH = ROOT / "example" / "images" / "img3.jpeg"
MODEL_ROOT = Path(
    os.environ.get("MANUSCRIPT_DEMO_MODELS", Path.home() / "Desktop" / "модельки")
).expanduser()
DETECTOR_WEIGHTS = (
    MODEL_ROOT / "rfdetr-textline-textregion-detection-2xl" / "segmentation"
    / "rfdetr-textline-textregion-detection-2xl.fp16.onnx"
)
RECOGNIZER_WEIGHTS = (
    MODEL_ROOT / "kraken-ppocrv6-small" / "kraken-ppocrv6-small.fp16.onnx"
)


def main():
    parser = argparse.ArgumentParser(description="Local RF-DETR → Kraken OCR test")
    parser.add_argument("image", nargs="?", type=Path, default=IMAGE_PATH)
    args = parser.parse_args()

    print(f"Изображение: {args.image}", flush=True)
    pipeline = Pipeline(
        detector=RFDETR(weights=DETECTOR_WEIGHTS, device="cpu"),
        recognizer=PPOCRRec(
            weights=RECOGNIZER_WEIGHTS, device="cpu",
            region_preparer="bbox", batch_size=8,
        ),
    )
    page = pipeline.predict(str(args.image.expanduser()), profile=True)["page"]

    print("\nРаспознанный текст:")
    for block in page.blocks:
        for line in block.lines:
            print(" ".join(span.text or "" for span in line.text_spans))
        print()

    print("Области (координаты округлены только для вывода):")
    for block_index, block in enumerate(page.blocks, 1):
        for line_index, line in enumerate(block.lines, 1):
            for span_index, span in enumerate(line.text_spans, 1):
                polygon = " ".join(
                    f"({x:.1f}, {y:.1f})" for x, y in span.polygon
                )
                if len(polygon) > 30:
                    polygon = polygon[:30] + "..."
                recognition = (
                    f"{span.recognition_confidence:.3f}"
                    if span.recognition_confidence is not None else "—"
                )
                print(f"\n[{block_index}.{line_index}.{span_index}] {span.text or ''}")
                print(
                    f"  detection={span.detection_confidence:.3f} "
                    f"recognition={recognition}"
                )
                print(f"  polygon: {polygon}")


if __name__ == "__main__":
    main()
