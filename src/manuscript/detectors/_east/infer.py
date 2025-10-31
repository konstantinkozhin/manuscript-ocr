import os
import time
from pathlib import Path
from typing import Union, Optional, List, Tuple, Sequence, Dict, Any

import cv2
import gdown
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from torchvision import transforms

from .._types import Word, Block, Page
from .dataset import EASTDataset
from .east import EAST
from .lanms import locality_aware_nms
from .train_utils import _run_training
from .utils import decode_quads_from_maps, draw_quads, read_image, expand_boxes


class EASTInfer:
    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        target_size: int = 1280,
        expand_ratio_w: float = 0.9,
        expand_ratio_h: float = 0.9,
        score_thresh: float = 0.6,
        iou_threshold: float = 0.2,
        score_geo_scale: float = 0.25,
        quantization: int = 2,
        axis_aligned_output: bool = True,
        remove_area_anomalies: bool = True,
        anomaly_sigma_threshold: float = 5.0,
        anomaly_min_box_count: int = 30,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if weights_path is None:
            url = (
                "https://github.com/konstantinkozhin/manuscript-ocr"
                "/releases/download/v0.1.0/east_quad_23_05.pth"
            )
            out = os.path.expanduser("~/.east_quad_23_05.pth")
            if not os.path.exists(out):
                print(f"Downloading EAST weights from {url} …")
                gdown.download(url, out, quiet=False)
            weights_path = out

        self.model = EAST(
            pretrained_backbone=False,
            pretrained_model_path=str(weights_path),
        ).to(self.device)
        self.model.eval()

        self.target_size = target_size
        self.score_geo_scale = score_geo_scale
        self.expand_ratio_w = expand_ratio_w
        self.expand_ratio_h = expand_ratio_h
        self.score_thresh = score_thresh
        self.iou_threshold = iou_threshold
        self.quantization = quantization
        self.axis_aligned_output = axis_aligned_output
        self.remove_area_anomalies = remove_area_anomalies
        self.anomaly_sigma_threshold = anomaly_sigma_threshold
        self.anomaly_min_box_count = anomaly_min_box_count

        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def _scale_boxes_to_original(
        self, boxes: np.ndarray, orig_size: Tuple[int, int]
    ) -> np.ndarray:
        if len(boxes) == 0:
            return boxes

        orig_h, orig_w = orig_size
        scale_x = orig_w / self.target_size
        scale_y = orig_h / self.target_size

        scaled = boxes.copy()
        scaled[:, 0:8:2] *= scale_x
        scaled[:, 1:8:2] *= scale_y
        return scaled

    def _convert_to_axis_aligned(self, quads: np.ndarray) -> np.ndarray:
        if len(quads) == 0:
            return quads
        aligned = quads.copy()
        coords = aligned[:, :8].reshape(-1, 4, 2)
        x_min = coords[:, :, 0].min(axis=1)
        x_max = coords[:, :, 0].max(axis=1)
        y_min = coords[:, :, 1].min(axis=1)
        y_max = coords[:, :, 1].max(axis=1)
        rects = np.stack(
            [
                x_min,
                y_min,
                x_max,
                y_min,
                x_max,
                y_max,
                x_min,
                y_max,
            ],
            axis=1,
        )
        aligned[:, :8] = rects.reshape(-1, 8)
        return aligned

    @staticmethod
    def _polygon_area_batch(polys: np.ndarray) -> np.ndarray:
        if polys.size == 0:
            return np.zeros((0,), dtype=np.float32)
        x = polys[:, :, 0]
        y = polys[:, :, 1]
        return 0.5 * np.abs(
            np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)
        )

    def _is_quad_inside(self, inner: np.ndarray, outer: np.ndarray) -> bool:
        contour = outer.reshape(-1, 1, 2).astype(np.float32)
        for point in inner.astype(np.float32):
            if (
                cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
                < 0
            ):
                return False
        return True

    def _remove_fully_contained_boxes(self, quads: np.ndarray) -> np.ndarray:
        if len(quads) <= 1:
            return quads
        coords = quads[:, :8].reshape(-1, 4, 2)
        areas = self._polygon_area_batch(coords)
        keep = np.ones(len(quads), dtype=bool)
        order = np.argsort(areas)
        for idx in order:
            if not keep[idx]:
                continue
            inner = coords[idx]
            inner_area = areas[idx]
            for jdx in range(len(quads)):
                if idx == jdx or not keep[jdx]:
                    continue
                if areas[jdx] + 1e-6 < inner_area:
                    continue
                if self._is_quad_inside(inner, coords[jdx]):
                    keep[idx] = False
                    break
        return quads[keep]

    def _remove_area_anomalies(self, quads: np.ndarray) -> np.ndarray:
        if (
            not self.remove_area_anomalies
            or len(quads) == 0
            or len(quads) <= self.anomaly_min_box_count
        ):
            return quads
        coords = quads[:, :8].reshape(-1, 4, 2)
        areas = self._polygon_area_batch(coords).astype(np.float32)
        mean = float(np.mean(areas))
        std = float(np.std(areas))
        if std == 0.0:
            return quads
        threshold = mean + self.anomaly_sigma_threshold * std
        keep = areas <= threshold
        if not np.any(keep):
            return quads
        return quads[keep]

    def predict(
        self,
        img_or_path: Union[str, Path, np.ndarray],
        vis: bool = False,
        profile: bool = False,
        return_maps: bool = False,
    ) -> Dict[str, Any]:
        """
        Run EAST inference and return detection results.

        Parameters
        ----------
        img_or_path : str or pathlib.Path or numpy.ndarray
            Path to an image file or an RGB image provided as a NumPy array
            with shape ``(H, W, 3)`` in ``uint8`` format.
        vis : bool, optional
            If True, a visualization image with rendered detections will be
            returned under the key ``"vis_image"``. Default is False.
        profile : bool, optional
            If True, prints timing information for the main inference stages.
            Default is False.
        return_maps : bool, optional
            If True, returns raw model score and geometry maps under keys
            ``"score_map"`` and ``"geo_map"``. Default is False.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - ``"page"`` : Page
                Parsed detection result containing detected ``Word`` objects
                with polygon coordinates and confidence scores.
            - ``"vis_image"`` : PIL.Image.Image or None
                Visualization image with drawn bounding boxes if ``vis=True``,
                otherwise ``None``.
            - ``"score_map"`` : numpy.ndarray or None
                Raw score map produced by the network if ``return_maps=True``.
            - ``"geo_map"`` : numpy.ndarray or None
                Raw geometry map if ``return_maps=True``.

        Notes
        -----
        The method performs:
        (1) image loading, (2) resizing and normalization, (3) model inference,
        (4) quad decoding, (5) NMS, (6) box expansion, (7) scaling coordinates
        back to original size, and (8) optional visualization.

        Examples
        --------
        Perform inference with visualization:

        >>> from manuscript.detectors import EASTInfer
        >>> model = EASTInfer()
        >>> img_path = r"example/ocr_example_image.jpg"
        >>> result = model.predict(img_path, vis=True)
        >>> page = result["page"]
        >>> vis_img = result["vis_image"]
        >>> vis_img.show()

        """
        # 1) Read & RGB
        img = read_image(img_or_path)

        # 2) Resize + ToTensor + Normalize
        resized = cv2.resize(img, (self.target_size, self.target_size))
        img_t = self.tf(resized).to(self.device)

        # 3) Forward
        t0 = time.time()
        with torch.no_grad():
            out = self.model(img_t.unsqueeze(0))

            score_map = out["score"][0].cpu().numpy().squeeze(0)
            geo_map = out["geometry"][0].cpu().numpy()
        if profile:
            print(f"  Model inference: {time.time() - t0:.3f}s")

        # 4) Decode raw quads (с квантизацией на уровне точек)
        t0 = time.time()
        final_quads = decode_quads_from_maps(
            score_map=score_map,
            geo_map=geo_map.transpose(1, 2, 0),
            score_thresh=self.score_thresh,
            scale=1.0 / self.score_geo_scale,
            quantization=self.quantization,
            profile=profile,
        )
        if profile:
            print(f"  Decode boxes: {time.time() - t0:.3f}s")

        # 5) Apply NMS
        t0 = time.time()
        final_quads_nms = locality_aware_nms(
            final_quads, iou_threshold=self.iou_threshold
        )
        if profile:
            print(f"  NMS: {time.time() - t0:.3f}s")
            print(f"    Boxes after NMS: {len(final_quads_nms)}")

        # 6) Expand (inverse shrink)
        final_quads_nms_expanded = expand_boxes(
            final_quads_nms, expand_w=self.expand_ratio_w, expand_h=self.expand_ratio_h
        )

        # 7) Scale coordinates back to original image size
        orig_h, orig_w = img.shape[:2]
        scaled_quads = self._scale_boxes_to_original(
            final_quads_nms_expanded, (orig_h, orig_w)
        )

        processed_quads = self._remove_fully_contained_boxes(scaled_quads)
        processed_quads = self._remove_area_anomalies(processed_quads)
        output_quads = (
            self._convert_to_axis_aligned(processed_quads)
            if self.axis_aligned_output
            else processed_quads
        )

        # 8) Build Page with scaled coordinates (after NMS & expand)
        words: List[Word] = []
        for quad in output_quads:
            pts = quad[:8].reshape(4, 2)
            score = float(quad[8])
            words.append(Word(polygon=pts.tolist(), detection_confidence=score))
        page = Page(blocks=[Block(words=words)])

        # 10) Optional visualization
        vis_img = (
            draw_quads(
                img, output_quads if self.axis_aligned_output else processed_quads
            )
            if vis
            else None
        )

        result: Dict[str, Any] = {
            "page": page,
            "vis_image": vis_img,
            "score_map": score_map if return_maps else None,
            "geo_map": geo_map if return_maps else None,
        }

        return result

    @staticmethod
    def train(
        train_images: Union[str, Path, Sequence[Union[str, Path]]],
        train_anns: Union[str, Path, Sequence[Union[str, Path]]],
        val_images: Union[str, Path, Sequence[Union[str, Path]]],
        val_anns: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        experiment_root: str = "./experiments",
        model_name: str = "resnet_quad",
        pretrained_backbone: bool = True,
        freeze_first: bool = True,
        target_size: int = 1024,
        score_geo_scale: Optional[float] = None,
        epochs: int = 500,
        batch_size: int = 3,
        lr: float = 1e-3,
        grad_clip: float = 5.0,
        early_stop: int = 100,
        use_sam: bool = True,
        sam_type: str = "asam",
        use_lookahead: bool = True,
        use_ema: bool = False,
        use_multiscale: bool = True,
        use_ohem: bool = True,
        ohem_ratio: float = 0.5,
        use_focal_geo: bool = True,
        focal_gamma: float = 2.0,
        resume_from: Optional[Union[str, Path]] = None,
        val_interval: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.nn.Module:
        """
        Train EAST model on custom datasets.

        Parameters
        ----------
        train_images : str, Path or sequence of paths
            Path(s) to training image folders.
        train_anns : str, Path or sequence of paths
            Path(s) to COCO-format JSON annotation files corresponding to
            ``train_images``.
        val_images : str, Path or sequence of paths
            Path(s) to validation image folders.
        val_anns : str, Path or sequence of paths
            Path(s) to COCO-format JSON annotation files corresponding to
            ``val_images``.
        experiment_root : str, optional
            Base directory where experiment folders will be created.
            Default is ``"./experiments"``.
        model_name : str, optional
            Folder name inside ``experiment_root`` for logs and checkpoints.
            Default is ``"resnet_quad"``.
        pretrained_backbone : bool, optional
            Use ImageNet-pretrained backbone weights. Default ``True``.
        freeze_first : bool, optional
            Freeze lowest layers of the backbone. Default ``True``.
        target_size : int, optional
            Resize shorter side of images to this size. Default ``1024``.
        score_geo_scale : float, optional
            Multiplier to recover original coordinates from score/geo maps.
            If None, automatically taken from the model. Default ``None``.
        epochs : int, optional
            Number of training epochs. Default ``500``.
        batch_size : int, optional
            Batch size. Default ``3``.
        lr : float, optional
            Learning rate. Default ``1e-3``.
        grad_clip : float, optional
            Gradient clipping value (L2 norm). Default ``5.0``.
        early_stop : int, optional
            Patience (epochs without improvement) for early stopping.
            Default ``100``.
        use_sam : bool, optional
            Enable SAM optimizer. Default ``True``.
        sam_type : {"sam", "asam"}, optional
            Variant of SAM to use. Default ``"asam"``.
        use_lookahead : bool, optional
            Wrap optimizer with Lookahead. Default ``True``.
        use_ema : bool, optional
            Maintain EMA version of model weights. Default ``False``.
        use_multiscale : bool, optional
            Random multi-scale training. Default ``True``.
        use_ohem : bool, optional
            Online Hard Example Mining. Default ``True``.
        ohem_ratio : float, optional
            Ratio of hard negatives for OHEM. Default ``0.5``.
        use_focal_geo : bool, optional
            Apply focal loss to geometry channels. Default ``True``.
        focal_gamma : float, optional
            Gamma for focal geometry loss. Default ``2.0``.
        resume_from : str or Path, optional
            Resume training from a previous experiment:
            a) experiment directory,
            b) `.../checkpoints/`,
            c) direct path to `last_state.pt`.
            Default ``None``.
        val_interval : int, optional
            Run validation every N epochs. Default ``1``.
        device : torch.device, optional
            CUDA or CPU device. Auto-selects if None.

        Returns
        -------
        torch.nn.Module
            Best model weights (EMA if enabled, otherwise base model).

        Examples
        --------
        Train on two datasets with validation:

        >>> from manuscript.detectors import EASTInfer
        >>>
        >>> train_images = [
        ...     "/data/archive/train_images",
        ...     "/data/ddi/train_images"
        ... ]
        >>> train_anns = [
        ...     "/data/archive/train.json",
        ...     "/data/ddi/train.json"
        ... ]
        >>> val_images = [
        ...     "/data/archive/test_images",
        ...     "/data/ddi/test_images"
        ... ]
        >>> val_anns = [
        ...     "/data/archive/test.json",
        ...     "/data/ddi/test.json"
        ... ]
        >>>
        >>> best_model = EASTInfer.train(
        ...     train_images=train_images,
        ...     train_anns=train_anns,
        ...     val_images=val_images,
        ...     val_anns=val_anns,
        ...     target_size=256,
        ...     epochs=20,
        ...     batch_size=4,
        ...     use_sam=False,
        ...     freeze_first=False,
        ...     val_interval=3,
        ... )
        >>> print("Best checkpoint loaded:", best_model)
        """

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = EAST(
            backbone_name="resnet101",
            pretrained_backbone=pretrained_backbone,
            freeze_first=freeze_first,
        ).to(device)

        if score_geo_scale is None:
            score_geo_scale = model.score_scale

        def make_dataset(imgs, anns, name: Optional[str] = None):
            return EASTDataset(
                images_folder=imgs,
                coco_annotation_file=anns,
                target_size=target_size,
                score_geo_scale=score_geo_scale,
                dataset_name=name,
            )

        def _dataset_base_name(
            img_path: Union[str, Path], ann_path: Union[str, Path]
        ) -> str:
            ann = Path(os.fspath(ann_path))
            parts: List[str] = []
            if ann.parent.name:
                parts.append(ann.parent.name)
            if ann.stem:
                parts.append(ann.stem)
            if not parts:
                img = Path(os.fspath(img_path))
                if img.parent.name:
                    parts.append(img.parent.name)
                stem = img.stem or img.name
                if stem:
                    parts.append(stem)
            return "/".join(parts)

        def _unique_dataset_name(
            img_path: Union[str, Path],
            ann_path: Union[str, Path],
            counts: Dict[str, int],
            idx: int,
            kind: str,
        ) -> str:
            base = _dataset_base_name(img_path, ann_path)
            if not base:
                base = f"{kind}_{idx}"
            count = counts.get(base, 0)
            counts[base] = count + 1
            if count == 0:
                return base
            return f"{base}_{count + 1}"

        train_imgs_list = (
            train_images if isinstance(train_images, (list, tuple)) else [train_images]
        )
        train_anns_list = (
            train_anns if isinstance(train_anns, (list, tuple)) else [train_anns]
        )
        val_imgs_list = (
            val_images if isinstance(val_images, (list, tuple)) else [val_images]
        )
        val_anns_list = val_anns if isinstance(val_anns, (list, tuple)) else [val_anns]

        assert len(train_imgs_list) == len(
            train_anns_list
        ), "train_images и train_anns должны иметь одинаковую длину"
        assert len(val_imgs_list) == len(
            val_anns_list
        ), "val_images и val_anns должны иметь одинаковую длину"

        train_datasets = []
        train_name_counts: Dict[str, int] = {}
        for idx, (imgs, anns) in enumerate(
            zip(train_imgs_list, train_anns_list), start=1
        ):
            dataset_name = _unique_dataset_name(
                imgs, anns, train_name_counts, idx=idx, kind="train"
            )
            train_datasets.append(make_dataset(imgs, anns, name=dataset_name))

        val_datasets = []
        val_name_counts: Dict[str, int] = {}
        for idx, (imgs, anns) in enumerate(zip(val_imgs_list, val_anns_list), start=1):
            dataset_name = _unique_dataset_name(
                imgs, anns, val_name_counts, idx=idx, kind="val"
            )
            val_datasets.append(make_dataset(imgs, anns, name=dataset_name))

        train_ds = ConcatDataset(train_datasets)
        val_ds = ConcatDataset(val_datasets)
        val_dataset_names = [ds.dataset_name for ds in val_datasets]

        def _resolve_path(path: Union[str, Path]) -> Path:
            p = Path(path)
            if p.is_absolute():
                return p
            project_root = Path(__file__).resolve().parents[4]
            candidate = (project_root / p).resolve()
            if candidate.exists():
                return candidate
            return (Path.cwd() / p).resolve()

        def _resolve_resume_target(
            target: Union[str, Path],
        ) -> Tuple[str, Optional[Path]]:
            resolved = _resolve_path(target)
            if not resolved.exists():
                raise FileNotFoundError(
                    f"resume_from target does not exist: {resolved}"
                )

            if resolved.is_file():
                resume_state = resolved
                checkpoints_dir = resolved.parent
                if checkpoints_dir.name == "checkpoints":
                    experiment_dir = checkpoints_dir.parent
                else:
                    experiment_dir = checkpoints_dir
                return os.path.abspath(os.fspath(experiment_dir)), resume_state

            experiment_dir = resolved
            checkpoints_dir = (
                resolved if resolved.name == "checkpoints" else resolved / "checkpoints"
            )
            default_state = checkpoints_dir / "last_state.pt"
            resume_state = default_state if default_state.exists() else None
            return os.path.abspath(os.fspath(experiment_dir)), resume_state

        resume_state_path: Optional[Path] = None
        if resume_from is None:
            experiment_dir = os.path.abspath(os.path.join(experiment_root, model_name))
            resume_flag = False
        else:
            experiment_dir, resume_state_path = _resolve_resume_target(resume_from)
            resume_flag = True

        best_model = _run_training(
            experiment_dir=experiment_dir,
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            device=device,
            num_epochs=epochs,
            batch_size=batch_size,
            lr=lr,
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
            val_datasets=val_datasets,
            val_dataset_names=val_dataset_names,
            resume=resume_flag,
            resume_state_path=(
                os.fspath(resume_state_path) if resume_state_path else None
            ),
        )
        return best_model
