"""Experiment with per-channel INT8 weights and dynamic INT8 OCR exports.

Weight-only exports store Conv/Gemm/MatMul weights in INT8 and dequantize them
to FP32. This reduces files; it does not promise INT8 execution or lower RAM.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import onnx
from onnx import numpy_helper, helper
import onnxruntime as ort


def lift_constants(graph):
    """Paddle2ONNX puts many weights in Constant nodes rather than initializers."""
    retained = []
    lifted = 0
    for node in graph.graph.node:
        attrs = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        if node.op_type == "Constant" and len(node.output) == 1 and isinstance(attrs.get("value"), onnx.TensorProto):
            tensor = onnx.TensorProto()
            tensor.CopyFrom(attrs["value"])
            tensor.name = node.output[0]
            graph.graph.initializer.append(tensor)
            lifted += 1
        else:
            retained.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(retained)
    return lifted


def quantize_weights(source, destination, preserve_head=False):
    graph = onnx.load(source)
    lifted = lift_constants(graph)
    weights = {item.name: item for item in graph.graph.initializer}
    quantized = {}
    nodes, added = [], []
    for node in graph.graph.node:
        excluded = preserve_head and any(part in node.name for part in ("/head/", "class_embed", "bbox_embed"))
        if not excluded and node.op_type in ("Conv", "Gemm", "MatMul") and len(node.input) > 1 and node.input[1] in weights:
            name = node.input[1]
            value = numpy_helper.to_array(weights[name])
            if value.dtype == np.float32 and value.ndim >= 2:
                attrs = {item.name: helper.get_attribute_value(item) for item in node.attribute}
                axis = 0 if node.op_type == "Conv" else (0 if node.op_type == "Gemm" and attrs.get("transB", 0) else value.ndim - 1)
                key = (name, axis)
                if key not in quantized:
                    prefix = name + f"_int8_axis{axis}"
                    axes = tuple(i for i in range(value.ndim) if i != axis)
                    scale = np.max(np.abs(value), axis=axes) / 127.0
                    scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
                    shape = [1] * value.ndim
                    shape[axis] = value.shape[axis]
                    integers = np.clip(np.rint(value / scale.reshape(shape)), -127, 127).astype(np.int8)
                    added.extend([numpy_helper.from_array(integers, prefix),
                                  numpy_helper.from_array(scale, prefix + "_scale"),
                                  numpy_helper.from_array(np.zeros(scale.shape, np.int8), prefix + "_zero")])
                    output = prefix + "_dequantized"
                    nodes.append(helper.make_node("DequantizeLinear",
                        [prefix, prefix + "_scale", prefix + "_zero"], [output], axis=axis,
                        name=prefix + "_DequantizeLinear"))
                    quantized[key] = output
                node.input[1] = quantized[key]
        nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    used = {name for node in nodes for name in node.input} | {item.name for item in graph.graph.output}
    retained = [item for item in graph.graph.initializer if item.name in used]
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(retained + added)
    onnx.checker.check_model(graph)
    onnx.save(graph, destination)
    return {"quantized_weight_tensors": len(quantized), "lifted_constants": lifted, "preserve_head": preserve_head,
            "mode": "INT8 per-channel weights; FP32 activations and compute"}


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def load_model(kind, path):
    if kind == "recognition":
        from manuscript.recognizers import PPOCRRec
        model = PPOCRRec(str(path), device="cpu", rotate_threshold=None)
    elif kind == "rfdetr":
        from manuscript.detectors._rfdetr import RFDETR
        model = RFDETR(path, device="cpu")
    else:
        from manuscript.detectors._ppocr import PPOCR
        model = PPOCR(path, device="cpu")
    model.onnx_session = session(path)
    if kind == "rfdetr":
        model._input_hw = tuple(model.onnx_session.get_inputs()[0].shape[2:])
    elif kind == "ppocr":
        # PPOCR's runtime uses a different session attribute.
        model._initialize_session()
        model.session = model.onnx_session
    return model


def input_feed(kind, model, image):
    from manuscript.utils import read_image
    if kind == "recognition":
        model._initialize_session()
        images = model._preprocess_image(image)
        feed = {model.input_name: images}
        if model.length_input_name:
            feed[model.length_input_name] = np.array([images.shape[3]], np.int64)
        return feed
    if kind == "rfdetr":
        return {model.input_name: model._preprocess(read_image(image))}
    return {model._input_name: model._preprocess(read_image(image))}


def benchmark(model_session, feed):
    model_session.run(None, feed)
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        model_session.run(None, feed)
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def evaluate(kind, reference, candidate, images):
    from manuscript.utils import read_image
    from validate_ppocr_rec import edit_distance
    from validate_rfdetr import compare
    rows = []
    if kind == "recognition":
        original = reference._predict_word_images(images, batch_size=1)
        predictions = candidate._predict_word_images(images, batch_size=1)
        for path, a, b in zip(images, original, predictions):
            rows.append({"image": str(path), "fp32": a, "quantized": b,
                         "text_equal": a["text"] == b["text"],
                         "changed_characters": edit_distance(a["text"], b["text"]),
                         "confidence_difference": abs(a["confidence"] - b["confidence"])})
        result = {"samples": len(rows), "equal_texts": sum(row["text_equal"] for row in rows),
                  "changed_characters": sum(row["changed_characters"] for row in rows),
                  "fp32_characters": sum(len(row["fp32"]["text"]) for row in rows), "results": rows}
        result["changed_character_fraction"] = result["changed_characters"] / max(1, result["fp32_characters"])
        result["preliminary_gate_passed"] = result["changed_character_fraction"] <= 0.01
    else:
        def detections(model, image):
            if kind == "rfdetr":
                return model.predict(image, return_raw=True)["detections"]
            page = model.predict(image)
            output = []
            for block in page.blocks:
                for line in block.lines:
                    for span in line.text_spans:
                        points = np.asarray(span.polygon)
                        output.append({"bbox": [float(points[:, 0].min()), float(points[:, 1].min()),
                                                float(points[:, 0].max()), float(points[:, 1].max())],
                                       "confidence": span.detection_confidence, "class_id": 0})
            return output
        for path in images:
            image = read_image(path)
            rows.append({"image": str(path), **compare(detections(reference, image), detections(candidate, image))})
        result = {"samples": len(rows), "results": rows,
                  "source_objects": sum(row["source_objects"] for row in rows),
                  "objects": sum(row["objects"] for row in rows), "matched": sum(row["matched"] for row in rows)}
        result["preliminary_gate_passed"] = all(row["recall"] >= 0.95 and row["precision"] >= 0.95 and
                                                row["mean_confidence_difference"] <= 0.02 for row in rows)
    feed = input_feed(kind, reference, images[0])
    original_session = reference.onnx_session if kind != "ppocr" else reference.session
    quantized_session = candidate.onnx_session if kind != "ppocr" else candidate.session
    result["benchmark"] = {"sample": str(images[0]), "threads": 2, "warmup": 1, "runs": 3,
                           "fp32_seconds": benchmark(original_session, feed),
                           "quantized_seconds": benchmark(quantized_session, feed)}
    result["benchmark"]["speedup"] = result["benchmark"]["fp32_seconds"] / result["benchmark"]["quantized_seconds"]
    return result


def jobs():
    samples = sorted(Path("build/ppocr_rec_bundle/samples").glob("*.png"))
    for root in sorted(Path("build/ppocr_rec_bundle").iterdir()):
        files = list(root.glob("*.fp32.onnx")) if root.is_dir() else []
        if files:
            yield "recognition", files[0], samples
    root = Path("build/rfdetr_bundle/rfdetr-textline-textregion-detection-2xl")
    images = [Path(f"example/images/img{i}.jpeg") for i in (1, 2, 3)] + sorted(Path("build/rfdetr_samples").glob("*.png"))
    yield "rfdetr", next(root.glob("*.fp32.onnx")), images
    root = Path.home() / "Desktop" / "PP-OCRv6_tiny_det"
    yield "ppocr", next(root.glob("*.fp32.onnx")), [Path(f"example/images/img{i}.jpeg") for i in (1, 2, 3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/quantization"))
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--modes", nargs="+", choices=["int8w", "int8w_head_fp32", "int8_dynamic"], default=["int8w"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_results = []
    for kind, source, images in jobs():
        name = source.name.removesuffix(".fp32.onnx")
        if args.models and name not in args.models:
            continue
        output = args.output / name
        output.mkdir(exist_ok=True)
        reference = load_model(kind, source)
        for mode in args.modes:
            path = output / f"{name}.{mode}.onnx"
            row = {"model": name, "kind": kind, "mode": mode, "source": str(source),
                   "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "fp32_size": source.stat().st_size}
            try:
                if mode in ("int8w", "int8w_head_fp32"):
                    row["quantization"] = quantize_weights(source, path, preserve_head=mode == "int8w_head_fp32")
                else:
                    from onnxruntime.quantization import quantize_dynamic, QuantType
                    lifted = onnx.load(source)
                    lift_constants(lifted)
                    quantize_dynamic(lifted, path, op_types_to_quantize=["MatMul", "Gemm"],
                                     per_channel=True, weight_type=QuantType.QInt8,
                                     extra_options={"MatMulConstBOnly": True})
                    row["quantization"] = {"mode": "Dynamic INT8 MatMul/Gemm; convolution stays FP32"}
                shutil.copy2(source.with_suffix(".json"), path.with_suffix(".json"))
                candidate = load_model(kind, path)
                row["evaluation"] = evaluate(kind, reference, candidate, images)
                row["size"] = path.stat().st_size
                row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                row["size_ratio_to_fp32"] = row["size"] / row["fp32_size"]
                row["status"] = "preliminary_pass" if row["evaluation"]["preliminary_gate_passed"] else "quality_gate_failed"
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
            (output / f"{mode}_report.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
            all_results.append(row)
            (args.output / "summary.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in ("model", "mode", "status")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
