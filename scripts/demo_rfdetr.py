"""Run a local RF-DETR ONNX bundle and save text-only or full-class results."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--all", action="store_true", help="Return all classes and a filtered text Page")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--classes", type=int, nargs="+", help="Text classes used to build Page")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("result"))
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1] / "src"
    if checkout.is_dir():
        sys.path.insert(0, str(checkout))
    from manuscript.detectors._rfdetr import RFDETR
    from manuscript.utils import read_image, visualize_page

    precision = "fp32" if args.fp32 else "fp16"
    weights = args.weights or Path(__file__).resolve().parent / f"rfdetr-textline-textregion-detection-2xl.{precision}.onnx"
    detector = RFDETR(weights, device=args.device, score_thresh=args.threshold, class_ids=args.classes)
    result = detector.predict(args.image, return_raw=args.all)
    page = result["page"] if args.all else result
    if args.all:
        payload = {**result, "page": page.model_dump(mode="json")}
        args.output.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        from collections import Counter
        print("Classes:", dict(Counter(row["class_name"] for row in result["detections"])))
    else:
        page.to_json(args.output.with_suffix(".json"))
    visualize_page(read_image(args.image), page, show_order=False, max_size=1280).convert("RGB").save(
        args.output.with_suffix(".jpg"), quality=90)
    count = sum(len(line.text_spans) for block in page.blocks for line in block.lines)
    print(f"Text spans: {count}; saved {args.output}.json/.jpg")


if __name__ == "__main__":
    main()
