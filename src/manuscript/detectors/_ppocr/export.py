"""Export a Transformers PP-OCR DB detector to standalone FP32/FP16 ONNX files.

Export dependencies: torch, transformers, safetensors, huggingface_hub, onnx,
onnxruntime. These are not required by the manuscript PPOCR inference class.
"""

import hashlib
import json
from pathlib import Path
import shutil


def export(model_dir, output_dir, model_id=None, revision=None):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import transformers
    import yaml
    from onnxruntime.transformers.float16 import convert_float_to_float16
    from transformers import AutoModelForObjectDetection

    model_dir, output_dir = Path(model_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inference = yaml.safe_load((model_dir / "inference.yml").read_text())
    model_name = inference["Global"]["model_name"]
    model = AutoModelForObjectDetection.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    ).float().eval()

    class ProbabilityMap(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images):
            return self.model(pixel_values=images).last_hidden_state

    wrapper = ProbabilityMap(model).eval()
    torch.set_num_threads(2)
    stem = model_name.replace("-", "").lower()
    fp32_path = output_dir / (stem + ".fp32.onnx")
    fp16_path = output_dir / (stem + ".fp16.onnx")
    sample = torch.zeros(1, 3, 256, 320)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper, sample, str(fp32_path), opset_version=17, dynamo=False,
            input_names=["images"], output_names=["probability_map"],
            dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"},
                          "probability_map": {0: "batch", 2: "height", 3: "width"}},
        )
    graph = onnx.load(fp32_path)
    onnx.checker.check_model(graph)
    half_graph = convert_float_to_float16(graph, keep_io_types=True)
    # ORT's converter can append I/O Cast nodes after their consumers.
    # Restore a valid dependency order before checking/saving the graph.
    pending = list(half_graph.graph.node)
    available = {item.name for item in half_graph.graph.input}
    available.update(item.name for item in half_graph.graph.initializer)
    ordered = []
    while pending:
        ready = [node for node in pending if all(not name or name in available for name in node.input)]
        if not ready:
            raise ValueError("FP16 conversion produced unresolved graph dependencies")
        for node in ready:
            ordered.append(node)
            available.update(node.output)
            pending.remove(node)
    del half_graph.graph.node[:]
    half_graph.graph.node.extend(ordered)
    onnx.checker.check_model(half_graph)
    onnx.save(half_graph, fp16_path)

    processor = json.loads((model_dir / "preprocessor_config.json").read_text())
    db = inference["PostProcess"]
    # Transformers normalizes RGB then reverses channels. Normalize BGR with
    # reversed mean/std to get the same channel convention without torch.
    config = {
        "schema_version": 1, "task": "text_detection", "algorithm": "DB",
        "model_name": model_name, "input_name": "images",
        "output_name": "probability_map",
        "preprocess": {
            "color_order": "BGR", "scale": processor["rescale_factor"],
            "mean": processor["image_mean"][::-1],
            "std": processor["image_std"][::-1],
            "limit_side_len": processor["limit_side_len"],
            "limit_type": processor["limit_type"],
            "max_side_limit": processor["max_side_limit"],
        },
        "postprocess": {
            "threshold": db["thresh"], "box_threshold": db["box_thresh"],
            "unclip_ratio": db["unclip_ratio"],
            "min_size": 3, "max_candidates": db["max_candidates"],
            "unclip_join": "round", "output_is_logits": False, "output_channel": 0,
        },
    }
    for path in [fp32_path, fp16_path]:
        path.with_suffix(".json").write_text(json.dumps(config, indent=2) + "\n")

    sessions = [ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                for p in [fp32_path, fp16_path]]
    rng = np.random.default_rng(42)
    comparisons = []
    for shape in [(1, 3, 256, 320), (1, 3, 320, 256), (2, 3, 128, 192)]:
        data = rng.normal(size=shape).astype(np.float32)
        with torch.inference_mode():
            reference = wrapper(torch.from_numpy(data)).numpy()
        results = [s.run(None, {"images": data})[0] for s in sessions]
        np.testing.assert_allclose(results[0], reference, atol=5e-4, rtol=1e-3)
        # DB probabilities can change locally near steep sigmoid boundaries.
        # Bound both worst-case and average drift; real-image region parity is
        # checked separately before distributing a model bundle.
        half_error = np.abs(results[1] - reference)
        if half_error.max() > 0.1 or half_error.mean() > 0.001:
            raise ValueError("FP16 probability drift exceeds the export verification bounds")
        comparisons.append({"input_shape": shape, "output_shape": list(reference.shape),
                            "fp32_max_abs_error": float(np.max(np.abs(results[0] - reference))),
                            "fp16_max_abs_error": float(half_error.max()),
                            "fp16_mean_abs_error": float(half_error.mean())})
    # Keep original configuration and model card for traceability, without
    # requiring safetensors or transformers to use the resulting bundle.
    sources = output_dir / "source"
    sources.mkdir(exist_ok=True)
    for filename in ["config.json", "preprocessor_config.json", "inference.yml", "README.md"]:
        source = model_dir / filename
        if source.exists():
            shutil.copy2(source, sources / filename)
    info = {
        "source_model": model_id, "source_revision": revision,
        "license": "Apache-2.0", "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "onnxruntime_version": ort.__version__, "opset": 17,
        "precision": "FP16 weights/operators with FP32 input/output and converter-blocked ops",
        "verification_bounds": {"fp32_atol": 0.0005, "fp32_rtol": 0.001,
                                "fp16_max_abs_error": 0.1, "fp16_mean_abs_error": 0.001},
        "comparisons": comparisons,
    }
    info["files"] = {p.name: {"size": p.stat().st_size,
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                     for p in [fp32_path, fp16_path]}
    (output_dir / "export_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
