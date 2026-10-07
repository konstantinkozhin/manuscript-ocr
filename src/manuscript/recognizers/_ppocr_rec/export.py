"""Export local Kraken v6/Paddle v5 CTC recognition bundles to ONNX.

Export-only dependencies: kraken>=7.1, torch, safetensors, paddlepaddle,
paddle2onnx, onnx and onnxruntime. Inference needs none of the training stacks.
"""

from manuscript.utils._onnx_export import half_model

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types

import numpy as np




def kraken_export(source, path):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    # Import the standalone upstream network without Kraken's training/model
    # wrapper (which otherwise imports Lightning and its training dependencies).
    package_path = Path(importlib.util.find_spec("kraken").submodule_search_locations[0]) / "lib/ppocr"
    package = types.ModuleType("kraken.lib.ppocr")
    package.__path__ = [str(package_path)]
    sys.modules["kraken.lib.ppocr"] = package
    from kraken.lib.ppocr.network import build_recognizer

    weights = next(source.glob("*.safetensors"))
    with safe_open(weights, framework="pt") as stream:
        metadata = json.loads(stream.metadata()["kraken_meta"])
    if len(metadata) != 1:
        raise ValueError("Expected one Kraken recognizer")
    prefix, meta = next(iter(metadata.items()))
    model = build_recognizer(meta["variant"], meta["num_classes"]).float().eval()
    state = {key[len(prefix + '.nn.'):]: value for key, value in load_file(weights).items()
             if key.startswith(prefix + '.nn.')}
    model.load_state_dict(state, strict=True)
    chars = [None] * (meta["num_classes"] - 1)
    for char, labels in meta["codec"].items():
        if len(labels) != 1 or not 1 <= labels[0] < meta["num_classes"]:
            raise ValueError("This exporter requires a single-label CTC codec")
        chars[labels[0] - 1] = char
    if any(char is None for char in chars):
        raise ValueError("Kraken codec contains missing labels")

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = model

        def forward(self, images, lengths):
            backbone = self.network.backbone
            feat = backbone.conv1(images)
            for stage in (backbone.blocks2, backbone.blocks3, backbone.blocks4, backbone.blocks5, backbone.blocks6):
                feat = stage(feat)
            # Input height is fixed. ReduceMean over height followed by width
            # pooling exactly represents the upstream (height, 2) AvgPool.
            feat = torch.nn.functional.avg_pool2d(feat.mean(dim=2, keepdim=True), (1, 2))
            # Shape tensors keep this ratio dynamic in the legacy exporter.
            w_in = torch._shape_as_tensor(images)[3]
            w_out = torch._shape_as_tensor(feat)[3]
            out_lengths = (lengths.float() * w_out / w_in).floor().long().clamp(min=1)
            positions = torch.arange(feat.shape[3], device=feat.device)
            mask = positions[None, :] < out_lengths[:, None]
            return self.network.head(self.network.neck(feat, mask)), out_lengths

    wrapper = Wrapper().eval()
    torch.set_num_threads(2)
    height = meta["height"]
    with torch.inference_mode():
        torch.onnx.export(wrapper, (torch.zeros(2, 3, height, 320), torch.tensor([320, 240])),
                          str(path), opset_version=17, dynamo=False,
                          input_names=["images", "lengths"], output_names=["logits", "output_lengths"],
                          dynamic_axes={"images": {0: "batch", 3: "width"}, "lengths": {0: "batch"},
                                        "logits": {0: "batch", 1: "time"}, "output_lengths": {0: "batch"}})

    def reference(images, lengths):
        with torch.inference_mode():
            # Verify against the unmodified upstream forward as well.
            tensor, lens = model(torch.from_numpy(images), torch.from_numpy(lengths))
            return tensor.squeeze(2).permute(0, 2, 1).numpy(), lens.numpy()

    config = {
        "schema_version": 1, "task": "text_recognition", "algorithm": "CTC",
        "rec_image_shape": [3, height, 320], "input_name": "images", "output_name": "logits",
        "length_input_name": "lengths", "length_input_units": "pixels", "length_output_name": "output_lengths",
        "preprocess": {"color_order": "RGB", "normalization": "invert", "dynamic_width": True,
                       "min_width": 0, "padding": 16, "interpolation": "lanczos"},
        "postprocess": {"output_is_logits": True}, "PostProcess": {"character_dict": chars},
    }
    from .config import normalize_config
    return normalize_config(config), reference


def paddle_export(source, path):
    import paddle.inference as inference

    subprocess.run([str(Path(sys.executable).with_name("paddle2onnx")), "--model_dir", str(source),
                    "--model_filename", "inference.json", "--params_filename", "inference.pdiparams",
                    "--save_file", str(path), "--opset_version", "17", "--optimize_tool", "None"], check=True)
    options = inference.Config(str(source / "inference.json"), str(source / "inference.pdiparams"))
    options.disable_gpu()
    options.set_cpu_math_library_num_threads(2)
    options.switch_ir_optim(False)
    predictor = inference.create_predictor(options)

    def reference(images, lengths):
        handle = predictor.get_input_handle(predictor.get_input_names()[0])
        handle.reshape(images.shape)
        handle.copy_from_cpu(images)
        predictor.run()
        return predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu(), None

    config = json.loads((source / "config.json").read_text())
    # Paddle's CTCLabelDecode appends the space token after its dictionary.
    if " " not in config["PostProcess"]["character_dict"]:
        config["PostProcess"]["character_dict"].append(" ")
    config.update({"schema_version": 1, "task": "text_recognition", "algorithm": "CTC",
                   "preprocess": {"dynamic_width": True}, "postprocess": {"output_is_logits": False}})
    from .config import normalize_config
    return normalize_config(config), reference


def export(source, output, crops=()):
    import onnx
    import onnxruntime as ort
    from manuscript.recognizers import PPOCRRec

    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = source.name
    fp32 = output / (name + ".fp32.onnx")
    config, reference = (kraken_export if name.startswith("kraken-") else paddle_export)(source, fp32)
    onnx.checker.check_model(onnx.load(fp32))
    fp16 = half_model(fp32)
    for path in (fp32, fp16):
        path.with_suffix(".json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    models = [PPOCRRec(str(path), device="cpu", rotate_threshold=None) for path in (fp32, fp16)]
    for model in models:
        model._initialize_session()
    sessions = [model.onnx_session for model in models]
    rng = np.random.default_rng(42)
    checks = []
    for batch, width in [(1, 160), (2, 320), (2, 640), (1, 321)]:
        data = rng.uniform(0, 1, (batch, 3, models[0].img_h, width)).astype(np.float32)
        lengths = np.array([width] + [width * 3 // 4] * (batch - 1), np.int64)
        expected, expected_lengths = reference(data, lengths)
        feed = {models[0].input_name: data}
        if models[0].length_input_name:
            feed[models[0].length_input_name] = lengths
        actual = [session.run(None, feed) for session in sessions]
        np.testing.assert_allclose(actual[0][0], expected, atol=0.002, rtol=0.002)
        if expected_lengths is not None:
            for outputs in actual:
                np.testing.assert_array_equal(outputs[1], expected_lengths)
        error = np.abs(actual[1][0] - expected)
        def probabilities(value):
            if not config["postprocess"]["output_is_logits"]:
                return value
            value = np.exp(value - value.max(axis=2, keepdims=True))
            return value / value.sum(axis=2, keepdims=True)
        probability_error = np.abs(probabilities(actual[1][0]) - probabilities(expected))
        checks.append({"shape": list(data.shape), "fp32_max_abs": float(np.abs(actual[0][0] - expected).max()),
                       "fp16_max_abs": float(error.max()), "fp16_mean_abs": float(error.mean()),
                       "fp16_probability_max_abs": float(probability_error.max()),
                       "fp16_probability_mean_abs": float(probability_error.mean())})
        if not np.isfinite(error).all() or probability_error.max() > 0.2 or probability_error.mean() > 0.001:
            raise ValueError(f"FP16 probability drift exceeds bounds: {checks[-1]}")
    real_checks = []
    for crop in crops:
        data = models[0]._preprocess_image(crop)
        expected, lengths = reference(data, np.array([data.shape[3]], np.int64))
        if lengths is not None:
            expected = expected[:, :int(lengths[0])]
        source_prediction = models[0]._decode_recognition_logits(expected)[0]
        predictions = [model._predict_word_images([crop], batch_size=1)[0] for model in models]
        if predictions[0]["text"] != source_prediction.text:
            raise ValueError(f"FP32/source text mismatch on {crop}")
        real_checks.append({"image": str(crop), "source_text": source_prediction.text,
                            "fp32": predictions[0], "fp16": predictions[1],
                            "fp16_text_equal": predictions[1]["text"] == source_prediction.text})
    if crops:
        # Exercise mixed-width batching, including the exported attention masks.
        batched = models[0]._predict_word_images(crops, batch_size=2)
        for entry, prediction in zip(real_checks, batched):
            entry["fp32_batch_text"] = prediction["text"]
            entry["fp32_batch_matches_single"] = prediction["text"] == entry["fp32"]["text"]
    source_output = output / "source"
    source_output.mkdir(exist_ok=True)
    for filename in ("README.md", "config.json", "inference.yml"):
        if (source / filename).exists():
            shutil.copy2(source / filename, source_output / filename)
    info = {"source_model": ("small-models-for-glam/" if name.startswith("kraken-") else "PaddlePaddle/") + name,
            "source_revision": (source / ".cache/huggingface/download/README.md.metadata").read_text().splitlines()[0],
            "verification": checks, "real_images": real_checks,
            "files": {path.name: {"size": path.stat().st_size,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (fp32, fp16)}}
    (output / "export_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"model": name, "output": str(output), "checks": checks,
                      "text_parity": [entry["fp16_text_equal"] for entry in real_checks]}, ensure_ascii=False), flush=True)
