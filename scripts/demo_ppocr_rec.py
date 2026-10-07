"""Recognize a crop or a page with a local PP-OCR recognition bundle."""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default="kraken-ppocrv6-small")
    parser.add_argument("--bundles", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--detector", type=Path, help="Local PPOCR detector weights for a whole page")
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    args = parser.parse_args()

    # Use this checkout while the new class has not yet been released on PyPI.
    checkout = Path(__file__).resolve().parents[1] / "src"
    if checkout.is_dir():
        sys.path.insert(0, str(checkout))
    from manuscript import Pipeline
    from manuscript.data import Block, Line, Page, TextSpan
    from manuscript.recognizers import PPOCRRec
    from manuscript.utils import read_image

    precision = "fp32" if args.fp32 else "fp16"
    weights = args.bundles / args.model / f"{args.model}.{precision}.onnx"
    recognizer = PPOCRRec(str(weights), device=args.device, batch_size=args.batch_size, rotate_threshold=None)
    if args.detector:
        from manuscript.detectors._ppocr import PPOCR
        page = Pipeline(detector=PPOCR(args.detector, device=args.device), layout=None,
                        recognizer=recognizer).predict(args.image)["page"]
    else:
        image = read_image(args.image)
        height, width = image.shape[:2]
        page = Page(blocks=[Block(lines=[Line(text_spans=[TextSpan(
            polygon=[(0, 0), (width, 0), (width, height), (0, height)], detection_confidence=1,
        )])])])
        page = recognizer.predict(page, image=image)
    page.to_json(args.output)
    for block in page.blocks:
        for line in block.lines:
            for span in line.text_spans:
                print(span.text)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
