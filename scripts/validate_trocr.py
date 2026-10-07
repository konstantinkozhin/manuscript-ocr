"""Compare an exported TrOCR bundle with saved source outputs."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from manuscript.recognizers import TrOCR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    root = Path(args.bundle)
    reference = json.loads((root / "source_predictions.json").read_text())
    saved = np.load(root / "decoder_reference.npz")
    results = {}
    for precision in ("fp32", "fp16"):
        model = TrOCR(str(root / f"encoder.{precision}.onnx"), device="cpu")
        # Limit threads for reproducible CPU measurements and avoid oversubscription.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.log_severity_level = 3
        model._initialize_session()
        model.onnx_session = ort.InferenceSession(
            str(root / f"encoder.{precision}.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        model.decoder_session = ort.InferenceSession(
            str(root / f"decoder.{precision}.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        numerical = []
        encoded = model.onnx_session.run(
            None,
            {
                model.onnx_session.get_inputs()[0].name: np.zeros(
                    (
                        1,
                        3,
                        model.config["preprocess"]["height"],
                        model.config["preprocess"]["width"],
                    ),
                    np.float32,
                )
            },
        )[0]
        encoder_difference = np.abs(encoded - saved["hidden"])
        for batch, length in ((1, 1), (4, 7)):
            actual = model.decoder_session.run(
                None,
                {
                    "input_ids": np.ones((batch, length), np.int64),
                    "encoder_hidden_states": np.repeat(saved["hidden"], batch, axis=0),
                },
            )[0]
            difference = np.abs(actual - saved[f"logits_{batch}_{length}"])
            numerical.append(
                {
                    "batch": batch,
                    "length": length,
                    "max_logit_difference": float(difference.max()),
                    "mean_logit_difference": float(difference.mean()),
                }
            )
        rows = []
        for row in reference:
            image = Path(row["image"])
            start = time.perf_counter()
            actual = model._predict_word_images([image])[0]
            pixels = model._preprocess_image(image)
            source_pixels = np.load(root / f"{image.stem}.pixels.npy")
            entry = {
                "image": str(image),
                "source_text": row["text"],
                "onnx_text": actual["text"],
                "text_equal": actual["text"] == row["text"],
                "tokens_equal": actual["meta"]["token_ids"] == row["token_ids"],
                "confidence": actual["confidence"],
                "seconds": time.perf_counter() - start,
                "max_pixel_difference": float(np.abs(pixels - source_pixels).max()),
                "mean_pixel_difference": float(np.abs(pixels - source_pixels).mean()),
            }
            rows.append(entry)
            print(precision, entry, flush=True)
            results[precision] = {
                "samples": len(rows),
                "equal_texts": sum(r["text_equal"] for r in rows),
                "decoder_numerical_checks": numerical,
                "encoder_numerical_check": {
                    "max_difference": float(encoder_difference.max()),
                    "mean_difference": float(encoder_difference.mean()),
                },
                "results": rows,
            }
            (root / "validation.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
