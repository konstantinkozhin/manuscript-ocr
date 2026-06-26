from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

from manuscript.api.detector import BaseDetector

from ...data import Block, Line, Page, TextSpan
from ...utils import read_image
from .utils import labels_to_polygons, pse

torch = None
ConcatDataset = None
PSEDataset = None
PSEModel = None
_run_training = None


def _ensure_torch_dependencies(*, training: bool = False):
    global torch, ConcatDataset, PSEDataset, PSEModel, _run_training
    if torch is not None and (not training or _run_training is not None):
        return
    try:
        import torch as torch_module
        from .model import PSEModel as model_cls
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for PSE export/training utilities. "
            "Install with: pip install manuscript-ocr[dev]"
        ) from exc
    torch = torch_module
    PSEModel = model_cls
    if not training:
        return
    try:
        from torch.utils.data import ConcatDataset as concat_dataset_cls
        from .dataset import PSEDataset as dataset_cls
        from .train_utils import _run_training as run_training_fn
    except ImportError as exc:
        raise ImportError(
            "PyTorch training dependencies are required for PSE training. "
            "Install with: pip install manuscript-ocr[dev]"
        ) from exc
    ConcatDataset = concat_dataset_cls
    PSEDataset = dataset_cls
    _run_training = run_training_fn


class PSE(BaseDetector):
    """PSENet/PSE detector with EAST-compatible public API."""

    def __init__(
        self,
        weights: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        *,
        target_size: int = 1280,
        map_scale: float = 0.25,
        kernel_num: int = 7,
        min_score: float = 0.85,
        min_area: int = 16,
        bbox_type: str = "poly",
    ):
        self.weights = str(Path(weights).expanduser().absolute()) if weights else None
        self.device = device or "cpu"
        self.target_size = int(target_size)
        self.map_scale = float(map_scale)
        self.kernel_num = int(kernel_num)
        self.min_score = float(min_score)
        self.min_area = int(min_area)
        self.bbox_type = str(bbox_type)
        self.onnx_session = None
        if self.weights is not None and not Path(self.weights).exists():
            raise ValueError(f"Weights file not found: {self.weights}")

    def runtime_providers(self):
        if self.device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.device == "coreml":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _initialize_session(self):
        if self.onnx_session is not None:
            return
        if self.weights is None:
            raise ValueError("PSE inference requires ONNX weights.")
        self.onnx_session = ort.InferenceSession(
            self.weights,
            providers=self.runtime_providers(),
        )

    def _run_inference_on_image(self, img: np.ndarray):
        self._initialize_session()
        resized = cv2.resize(img, (self.target_size, self.target_size))
        img_norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        img_input = img_norm.transpose(2, 0, 1)[np.newaxis, :, :, :]
        input_name = self.onnx_session.get_inputs()[0].name
        output_names = [out.name for out in self.onnx_session.get_outputs()]
        outputs = self.onnx_session.run(output_names, {input_name: img_input})
        maps = outputs[0].squeeze(0)
        kernels = (maps > 0.5).astype(np.uint8)
        kernels[1:] = kernels[1:] * kernels[:1]
        labels = pse(kernels, min_area=self.min_area)
        return labels, maps[0]

    def predict(self, img_or_path: Union[str, Path, np.ndarray]) -> Page:
        img = read_image(img_or_path)
        orig_h, orig_w = img.shape[:2]
        labels, score = self._run_inference_on_image(img)
        scale_x = orig_w / labels.shape[1]
        scale_y = orig_h / labels.shape[0]
        polygons = labels_to_polygons(
            labels,
            score,
            scale_x=scale_x,
            scale_y=scale_y,
            min_area=self.min_area,
            min_score=self.min_score,
            bbox_type=self.bbox_type,
        )
        spans: List[TextSpan] = []
        for order, (polygon, confidence) in enumerate(polygons):
            spans.append(
                TextSpan(
                    polygon=polygon,
                    detection_confidence=float(np.clip(confidence, 0.0, 1.0)),
                    order=order,
                )
            )
        return Page(blocks=[Block(lines=[Line(text_spans=spans, order=0)], order=0)])

    @staticmethod
    def train(
        train_images: Union[str, Path, Sequence[Union[str, Path]]],
        train_anns: Union[str, Path, Sequence[Union[str, Path]]],
        val_images: Union[str, Path, Sequence[Union[str, Path]]],
        val_anns: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        experiment_root: str = "./experiments",
        model_name: str = "resnet_pse",
        backbone_name: str = "resnet50",
        pretrained_backbone: bool = True,
        freeze_first: bool = False,
        target_size: int = 1024,
        score_geo_scale: Optional[float] = None,
        score_map_shrink_ratio: float = 0.3,
        epochs: int = 500,
        batch_size: int = 3,
        accumulation_steps: int = 1,
        lr: float = 1e-4,
        lr_scheduler: str = "none",
        lr_scheduler_params: Optional[Dict[str, Any]] = None,
        augmentation_config: Optional[Dict[str, Any]] = None,
        grad_clip: float = 5.0,
        early_stop: int = 100,
        use_sam: bool = False,
        sam_type: str = "asam",
        use_lookahead: bool = False,
        use_ema: bool = False,
        use_multiscale: bool = True,
        use_ohem: bool = False,
        ohem_ratio: float = 0.5,
        use_focal_geo: bool = True,
        focal_gamma: float = 2.0,
        resume_from: Optional[Union[str, Path]] = None,
        val_interval: int = 1,
        num_workers: int = 0,
        log_collage: bool = True,
        device: Optional["torch.device"] = None,
    ) -> "torch.nn.Module":
        _ensure_torch_dependencies(training=True)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        kernel_num = 7
        model = PSEModel(
            backbone_name=backbone_name,
            pretrained_backbone=pretrained_backbone,
            freeze_first=freeze_first,
            kernel_num=kernel_num,
        ).to(device)
        map_scale = model.map_scale if score_geo_scale is None else score_geo_scale

        aug_cfg = {
            "kernel_num": kernel_num,
            "min_scale": 0.5,
            "max_shrink": 20,
            "flip_prob": 0.01,
            "vflip_prob": 0.0,
            "color_jitter": (0.1, 0.1, 0.1, 0.05),
            "small_rotate_prob": 0.2,
            "small_rotate_deg": 2.0,
            "shear_prob": 0.15,
            "shear_deg": 5.0,
            "random_crop_prob": 0.2,
            "random_crop_scale": (0.7, 1.0),
            "perspective_prob": 0.1,
            "perspective_scale": 0.015,
            "blur_prob": 0.1,
            "blur_ksize_range": (3, 5),
            "motion_blur_prob": 0.1,
            "motion_blur_ksize_range": (3, 9),
            "noise_prob": 0.1,
            "noise_std": 0.008,
            "salt_pepper_prob": 0.0005,
            "jpeg_prob": 0.1,
            "jpeg_quality_range": (75, 95),
            "shading_prob": 0.1,
            "shading_strength": 0.1,
            "gamma_prob": 0.2,
            "gamma_range": (0.95, 1.05),
            "downscale_prob": 0.1,
            "downscale_range": (0.7, 0.95),
            "negative_prob": 0.05,
            "hsv_prob": 0.15,
            "hsv_h": 0.015,
            "hsv_s": 0.3,
            "hsv_v": 0.2,
            "cutout_prob": 0.15,
            "cutout_num_holes": 2,
            "cutout_hole_size_range": (0.05, 0.15),
            "elastic_prob": 0.1,
            "elastic_alpha": 20.0,
            "elastic_sigma": 4.0,
            "fog_prob": 0.1,
            "fog_strength_range": (0.1, 0.4),
            "fog_direction": "random",
        }
        if augmentation_config:
            aug_cfg.update(augmentation_config)

        def make_dataset(imgs, anns, name: Optional[str] = None):
            return PSEDataset(
                images_folder=imgs,
                coco_annotation_file=anns,
                target_size=target_size,
                map_scale=map_scale,
                dataset_name=name,
                **aug_cfg,
            )

        def _as_list(value):
            return value if isinstance(value, (list, tuple)) else [value]

        train_imgs_list, train_anns_list = _as_list(train_images), _as_list(train_anns)
        val_imgs_list, val_anns_list = _as_list(val_images), _as_list(val_anns)
        assert len(train_imgs_list) == len(train_anns_list), "train_images and train_anns must have the same length"
        assert len(val_imgs_list) == len(val_anns_list), "val_images and val_anns must have the same length"

        def _name(img_path, ann_path, counts, idx, kind):
            ann = Path(os.fspath(ann_path))
            base = "/".join(part for part in (ann.parent.name, ann.stem) if part) or f"{kind}_{idx}"
            count = counts.get(base, 0)
            counts[base] = count + 1
            return base if count == 0 else f"{base}_{count + 1}"

        counts: Dict[str, int] = {}
        train_datasets = [
            make_dataset(imgs, anns, _name(imgs, anns, counts, idx, "train"))
            for idx, (imgs, anns) in enumerate(zip(train_imgs_list, train_anns_list), start=1)
        ]
        counts = {}
        val_datasets = [
            make_dataset(imgs, anns, _name(imgs, anns, counts, idx, "val"))
            for idx, (imgs, anns) in enumerate(zip(val_imgs_list, val_anns_list), start=1)
        ]
        train_ds = ConcatDataset(train_datasets)
        val_ds = ConcatDataset(val_datasets)
        val_dataset_names = [ds.dataset_name for ds in val_datasets]

        def _resolve_path(path: Union[str, Path]) -> Path:
            p = Path(path)
            if p.is_absolute():
                return p
            candidate = (Path(__file__).resolve().parents[4] / p).resolve()
            return candidate if candidate.exists() else (Path.cwd() / p).resolve()

        def _is_experiment_checkpoint(path: Path) -> bool:
            return path.name in {"last_state.pt", "checkpoint_last.pth", "checkpoint_last.pt"} and (
                (path.parent.parent / "training_config.json").exists()
                if path.parent.name == "checkpoints"
                else (path.parent / "training_config.json").exists()
            )

        default_experiment_dir = os.path.abspath(os.path.join(experiment_root, model_name))
        resume_state_path = None
        if resume_from is None:
            experiment_dir, resume_flag = default_experiment_dir, False
        else:
            resolved = _resolve_path(resume_from)
            if not resolved.exists():
                raise FileNotFoundError(f"resume_from target does not exist: {resolved}")
            resume_flag = True
            if resolved.is_file():
                resume_state_path = resolved
                experiment_dir = (
                    resolved.parent.parent
                    if _is_experiment_checkpoint(resolved) and resolved.parent.name == "checkpoints"
                    else default_experiment_dir
                )
            else:
                experiment_dir = resolved
                candidate = (resolved if resolved.name == "checkpoints" else resolved / "checkpoints") / "last_state.pt"
                resume_state_path = candidate if candidate.exists() else None

        return _run_training(
            experiment_dir=os.path.abspath(os.fspath(experiment_dir)),
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            device=device,
            num_epochs=epochs,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            grad_clip=grad_clip,
            early_stop=early_stop,
            use_sam=use_sam,
            sam_type=sam_type,
            use_lookahead=use_lookahead,
            use_ema=use_ema,
            use_multiscale=use_multiscale,
            use_ohem=use_ohem,
            ohem_ratio=ohem_ratio,
            use_focal_geo=use_focal_geo,
            focal_gamma=focal_gamma,
            val_interval=val_interval,
            num_workers=num_workers,
            backbone_name=backbone_name,
            target_size=target_size,
            pretrained_backbone=pretrained_backbone,
            val_datasets=val_datasets,
            val_dataset_names=val_dataset_names,
            resume=resume_flag,
            resume_state_path=os.fspath(resume_state_path) if resume_state_path else None,
            score_map_shrink_ratio=score_map_shrink_ratio,
            augmentation_config=aug_cfg,
            log_collage=log_collage,
        )

    @staticmethod
    def export(
        weights_path: Union[str, Path],
        output_path: Union[str, Path],
        backbone_name: str = "resnet50",
        input_size: int = 1280,
        opset_version: int = 14,
        simplify: bool = False,
        kernel_num: int = 7,
    ) -> None:
        _ensure_torch_dependencies()

        class PSEWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model(x)["maps"]

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        model = PSEModel(
            backbone_name=backbone_name,
            pretrained_backbone=False,
            pretrained_model_path=str(weights_path),
            kernel_num=kernel_num,
        ).eval()
        wrapped = PSEWrapper(model).eval()
        dummy_input = torch.randn(1, 3, input_size, input_size)
        torch.onnx.export(
            wrapped,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["pse_maps"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "pse_maps": {0: "batch_size", 2: "height", 3: "width"},
            },
            verbose=False,
        )
        if simplify:
            import onnx
            import onnxsim

            onnx_model = onnx.load(str(output_path))
            model_simplified, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(model_simplified, str(output_path))


__all__ = ["PSE"]
