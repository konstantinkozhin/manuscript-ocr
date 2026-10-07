"""Export the Kansallisarkisto RF-DETR Seg 2XL boxes, classes and instance masks.

The resulting ONNX preserves segmentation outputs. Inference requires only manuscript's normal ONNX dependencies.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def load_source(checkpoint):
    import torch
    from rfdetr import RFDETRSeg2XLarge

    with torch.serialization.safe_globals([argparse.Namespace]):
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = RFDETRSeg2XLarge(pretrain_weights=None, device="cpu", num_classes=90)
    network = model.model.model
    state = dict(data["model"])
    # New RF-DETR versions add a non-learned keypoint buffer, unused by this
    # segmentation architecture. All checkpoint parameters load strictly.
    if "_kp_active_mask" not in state and "_kp_active_mask" in network.state_dict():
        state["_kp_active_mask"] = network.state_dict()["_kp_active_mask"]
    network.load_state_dict(state, strict=True)
    network.eval()
    metadata = {
        "model_config": model.model_config.model_dump(mode="json"),
        "class_names": data["args"]["class_names"],
        "state_source": "model",
        "checkpoint_weights_only": True,
        "added_nonlearned_buffers": ["_kp_active_mask"],
    }
    return model, network, metadata


def export(source, output, images):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import torchvision.transforms.functional as F
    from PIL import Image
    from rfdetr.models.postprocess import PostProcess
    from manuscript.utils._onnx_export import half_model
    from manuscript.detectors._rfdetr import RFDETR
    from manuscript.utils import read_image

    torch.set_num_threads(2)
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = source / "checkpoint_best_total.pth"
    public_model, network, metadata = load_source(checkpoint)
    resolution = public_model.model_config.resolution
    config = {
        "schema_version": 1,
        "task": "object_detection",
        "target_size": resolution,
        "input_name": "images",
        "boxes_output_name": "pred_boxes",
        "logits_output_name": "pred_logits",
        "masks_output_name": "pred_masks",
        "mask_threshold": 0.5,
        "activation": "sigmoid",
        "score_thresh": 0.5,
        "num_select": 300,
        "text_class_ids": [2, 3, 4, 5],
        "class_names": {str(i): name for i, name in enumerate(metadata["class_names"])},
        "mean": list(public_model.means),
        "std": list(public_model.stds),
    }
    preprocessor = RFDETR.__new__(RFDETR)
    preprocessor._input_hw = (resolution, resolution)
    preprocessor._input_dtype = np.float32
    preprocessor.mean = np.asarray(config["mean"], np.float32)
    preprocessor.std = np.asarray(config["std"], np.float32)
    data_list, references = [], []
    reference_decoder = RFDETR(checkpoint, config=config)
    native_decoder = PostProcess(num_select=config["num_select"])
    for path in images:
        array = read_image(path)
        data = preprocessor._preprocess(array)
        original_preprocess = F.normalize(
            F.resize(
                F.to_tensor(Image.fromarray(array)),
                [resolution, resolution],
                antialias=False,
            ),
            config["mean"],
            config["std"],
        )[None].numpy()
        # OpenCV and torch round bilinear source coordinates differently on
        # large pages. Bound this small normalized-pixel rounding difference.
        np.testing.assert_allclose(data, original_preprocess, atol=0.001, rtol=0.0001)
        with torch.inference_mode():
            original = network(torch.from_numpy(data))
            native = native_decoder(
                {key: original[key] for key in ("pred_boxes", "pred_logits")},
                torch.tensor([array.shape[:2]]),
            )[0]
        keep = native["scores"].numpy() >= config["score_thresh"]
        native_boxes = native["boxes"].numpy()[keep]
        native_boxes[:, [0, 2]] = np.clip(native_boxes[:, [0, 2]], 0, array.shape[1])
        native_boxes[:, [1, 3]] = np.clip(native_boxes[:, [1, 3]], 0, array.shape[0])
        decoded = reference_decoder._postprocess(
            original["pred_boxes"].numpy(),
            original["pred_logits"].numpy(),
            array.shape[:2],
            return_raw=True,
        )["detections"]
        np.testing.assert_array_equal(
            [row["class_id"] for row in decoded], native["labels"].numpy()[keep]
        )
        if decoded:
            np.testing.assert_allclose(
                [row["bbox"] for row in decoded], native_boxes, atol=0.002
            )
            np.testing.assert_allclose(
                [row["confidence"] for row in decoded],
                native["scores"].numpy()[keep],
                atol=1e-6,
            )
        data_list.append(data)
        references.append(
            (original["pred_boxes"].numpy(), original["pred_logits"].numpy())
        )
        np.savez_compressed(
            output / ("source_" + Path(path).stem + ".npz"),
            pred_boxes=references[-1][0],
            pred_logits=references[-1][1],
            pred_masks=original["pred_masks"].numpy(),
        )
    network.export()

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = network

        def forward(self, images):
            outputs = self.network(images)
            return outputs[0], outputs[1], outputs[2]

    wrapper = Wrapper().eval()
    stem = source.name
    fp32 = output / (stem + ".fp32.onnx")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            torch.zeros(1, 3, resolution, resolution),
            str(fp32),
            opset_version=17,
            dynamo=False,
            input_names=["images"],
            output_names=["pred_boxes", "pred_logits", "pred_masks"],
        )
    onnx.checker.check_model(str(fp32))
    fp16 = half_model(fp32)
    for path in (fp32, fp16):
        path.with_suffix(".json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n"
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    sessions = [
        ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        for path in (fp32, fp16)
    ]
    comparisons = []
    for path, data, reference in zip(images, data_list, references):
        outputs = [
            session.run(["pred_boxes", "pred_logits"], {"images": data})
            for session in sessions
        ]
        row = {"image": str(path), "preprocessing_verified": True}
        for label, values in zip(("fp32", "fp16"), outputs):
            errors = [
                np.abs(actual - expected) for actual, expected in zip(values, reference)
            ]
            row[label] = {
                key: {"max_abs": float(error.max()), "mean_abs": float(error.mean())}
                for key, error in zip(("boxes", "logits"), errors)
            }
            if not all(np.isfinite(actual).all() for actual in values):
                raise ValueError("Non-finite ONNX outputs")
        # Encoder top-k query selection can permute queries after small FP32
        # roundoff. Verify objects by class and IoU in validate_rfdetr.py;
        # per-query raw errors remain recorded for diagnosis.
        comparisons.append(row)
        print(json.dumps(row), flush=True)
    shutil.copy2(source / "README.md", output / "source_model_card.md")
    (output / "source_model_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    info = {
        "source_model": "Kansallisarkisto/" + stem,
        "source_revision": "2f33c96baaf4ffa5009d7bd3f2b3af336c7eeb75",
        "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "rfdetr_version": importlib.metadata.version("rfdetr"),
        "exported_outputs": ["pred_boxes", "pred_logits", "pred_masks"],
        "instance_masks_exported": True,
        "preprocessing": "RGB float bilinear antialias=False + ImageNet normalization",
        "native_postprocess_verified": True,
        "verification": comparisons,
        "files": {
            path.name: {
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (fp32, fp16)
        },
    }
    (output / "export_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n"
    )
    print("Export completed.", flush=True)
