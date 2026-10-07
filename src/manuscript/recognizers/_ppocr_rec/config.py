"""Normalize native Paddle recognition configs into the explicit CTC schema."""
from copy import deepcopy


def normalize_config(config):
    config = deepcopy(config)
    native = "PreProcess" in config or "PostProcess" in config
    if native:
        shape = config.get("rec_image_shape")
        for operation in config.get("PreProcess", {}).get("transform_ops", []):
            if shape is None:
                shape = operation.get("RecResizeImg", {}).get("image_shape")
        shape = shape or config.get("Global", {}).get("d2s_train_image_shape")
        if shape is None:
            raise ValueError("Native Paddle config requires RecResizeImg image_shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 3:
            raise ValueError("rec_image_shape must contain three dimensions")
        config["rec_image_shape"] = shape
        config.setdefault("schema_version", 1)
        config.setdefault("task", "text_recognition")
        config.setdefault("algorithm", "CTC")
        config["preprocess"] = {
            "color_order": "BGR", "normalization": "paddle", "dynamic_width": True,
            "min_width": shape[2], "padding": 0, "interpolation": "paddle",
            **config.get("preprocess", {}),
        }
        config["postprocess"] = {"output_is_logits": False, **config.get("postprocess", {})}
    required = {
        "preprocess": ("color_order", "normalization", "dynamic_width", "min_width", "padding", "interpolation"),
        "postprocess": ("output_is_logits",),
    }
    if not all(key in config for key in ("schema_version", "task", "algorithm", "rec_image_shape")):
        raise ValueError("CTC config requires schema_version, task, algorithm and rec_image_shape")
    for section, keys in required.items():
        if not isinstance(config.get(section), dict):
            raise ValueError(f"CTC config requires {section}")
        missing = [key for key in keys if key not in config[section]]
        if missing:
            raise ValueError(f"CTC config missing {section}: {', '.join(missing)}")
    return config
