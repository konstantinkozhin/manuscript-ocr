"""Recognize a cropped text line with a local TrOCR ONNX bundle."""

import argparse
import json
from pathlib import Path
import sys

checkout = Path(__file__).resolve().parents[1] / "src"
if checkout.is_dir():
    sys.path.insert(0, str(checkout))
from manuscript.recognizers import TrOCR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument(
        "--weights", required=True, help="Path to encoder ONNX with companion JSON"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beams", type=int)
    parser.add_argument("--max-length", type=int)
    args = parser.parse_args()
    generation = {}
    if args.beams is not None:
        generation["num_beams"] = args.beams
    if args.max_length is not None:
        generation["max_length"] = args.max_length
    model = TrOCR(args.weights, device=args.device, generation=generation)
    print(
        json.dumps(
            model._predict_word_images([args.image])[0], ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
