import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image

from manuscript.api.recognizer import BaseRecognizer
from manuscript.data import Page
from manuscript.utils import crop_axis_aligned, read_image

from .._common.debug import save_debug_regions
from .._common.region_preparers import (
    call_region_preparer,
    prepare_bbox_regions,
    prepare_polygon_mask_regions,
    prepare_quad_warp_regions,
    prepare_text_regions,
)
from .._common.region_types import (
    REGION_PREPARER_PRESETS,
    PreparedRegion,
    RecognitionPrediction,
    normalize_prepared_regions,
    normalize_recognition_predictions,
)


def _build_local_custom_preset_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[4]
    paddleocr_root = Path(
        os.environ.get("MANUSCRIPT_PPOCRV5_CUSTOM_ROOT", repo_root.parent / "PaddleOCR")
    ).expanduser()
    return {
        "weights": str(
            (paddleocr_root / "custom_ppocrv5_rec_g2" / "model_g1dict.onnx").absolute()
        ),
        "config": str(
            (
                paddleocr_root
                / "custom_ppocrv5_rec_g2"
                / "inference_g1dict"
                / "inference.yml"
            ).absolute()
        ),
        "charset": str(
            (
                paddleocr_root
                / "train_data"
                / "custom_ppocrv5_rec_g1_restored"
                / "custom_dict.txt"
            ).absolute()
        ),
    }


_LOCAL_CUSTOM_PRESET_PATHS = _build_local_custom_preset_paths()


class PPOCRv5Rec(BaseRecognizer):
    """
    Inference-only recognizer for PaddleOCR PP-OCRv5 recognition ONNX models.

    The public API intentionally mirrors ``TRBA``:
    - artifact-backed initialization
    - ``predict(page, image=None) -> Page``
    - region preparer presets and debug crop export
    """

    default_weights_name = None
    pretrained_registry: Dict[str, str] = {
        "custom_ppocrv5_rec_g2": _LOCAL_CUSTOM_PRESET_PATHS["weights"]
    }
    config_registry: Dict[str, str] = {
        "custom_ppocrv5_rec_g2": _LOCAL_CUSTOM_PRESET_PATHS["config"]
    }
    charset_registry: Dict[str, str] = {
        "custom_ppocrv5_rec_g2": _LOCAL_CUSTOM_PRESET_PATHS["charset"]
    }

    def _resolve_weights(self, weights: Optional[str]) -> str:
        if weights in self.pretrained_registry:
            return super()._resolve_weights(self.pretrained_registry[str(weights)])
        return super()._resolve_weights(weights)

    def __init__(
        self,
        weights: Optional[str] = None,
        config: Optional[str] = None,
        charset: Optional[str] = None,
        device: Optional[str] = None,
        force_download: bool = False,
        rotate_threshold: Optional[float] = 1.5,
        region_preparer: Union[str, Callable[..., Sequence[Any]]] = "bbox",
        region_preparer_options: Optional[Dict[str, Any]] = None,
        min_text_size: int = 5,
        batch_size: int = 16,
        use_space_char: Optional[bool] = None,
        rec_image_shape: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ):
        if "region_predictor" in kwargs:
            raise TypeError(
                "region_predictor has been removed from PPOCRv5Rec. "
                "Pass a custom recognizer to Pipeline instead."
            )
        if "recognizer_debug_dir" in kwargs:
            raise TypeError(
                "recognizer_debug_dir has been removed from PPOCRv5Rec. "
                "Use debug_save_dir instead."
            )

        default_debug_save_dir = kwargs.pop("debug_save_dir", None)
        self._weights_preset = (
            str(weights) if weights is not None and str(weights) in self.pretrained_registry else None
        )

        super().__init__(
            weights=weights,
            device=device,
            force_download=force_download,
            **kwargs,
        )

        if not Path(self.weights).exists():
            raise FileNotFoundError(f"Model file not found: {self.weights}")
        if Path(self.weights).suffix.lower() != ".onnx":
            raise ValueError(f"Expected .onnx file, got: {self.weights}")

        self.config_path = self._resolve_config(config)
        config_data = self._load_config_data(self.config_path) if self.config_path else {}

        self.rec_image_shape = self._resolve_rec_image_shape(
            rec_image_shape=rec_image_shape,
            config_data=config_data,
        )
        self.img_c, self.img_h, self.img_w = self.rec_image_shape

        self.use_space_char = self._resolve_use_space_char(
            use_space_char=use_space_char,
            config_data=config_data,
        )
        self.charset_path = self._resolve_charset(charset)
        self.characters = self._load_characters(
            charset_path=self.charset_path,
            config_data=config_data,
            use_space_char=self.use_space_char,
        )

        self.onnx_session = None
        self.rotate_threshold = rotate_threshold
        self.region_preparer = self._validate_region_preparer(region_preparer)
        self.region_preparer_options = dict(region_preparer_options or {})
        self.min_text_size = min_text_size
        self.batch_size = max(1, int(batch_size))
        self.default_debug_save_dir = (
            None
            if default_debug_save_dir is None
            else Path(default_debug_save_dir).expanduser()
        )
        self._supports_multi_batch_inference: Optional[bool] = None
        self._single_batch_warning_emitted = False
        self._input_width_override: Optional[int] = None

    def _resolve_config(self, config: Optional[str]) -> Optional[str]:
        if config is not None:
            return self._resolve_extra_artifact(
                config,
                default_name=None,
                registry=self.config_registry,
                description="config",
            )

        if self._weights_preset and self._weights_preset in self.config_registry:
            return self._resolve_extra_artifact(
                self.config_registry[self._weights_preset],
                default_name=None,
                registry=self.config_registry,
                description="config",
            )

        weights_path = Path(self.weights)
        candidates = [
            weights_path.with_suffix(".json"),
            weights_path.with_suffix(".yml"),
            weights_path.with_suffix(".yaml"),
            weights_path.parent / "inference.json",
            weights_path.parent / "inference.yml",
            weights_path.parent / "inference.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.absolute())

        recursive_candidates = sorted(
            [
                p
                for pattern in ("inference*.yml", "inference*.yaml", "inference*.json")
                for p in weights_path.parent.rglob(pattern)
            ]
        )
        if len(recursive_candidates) == 1:
            return str(recursive_candidates[0].absolute())

        return None

    def _resolve_charset(self, charset: Optional[str]) -> Optional[str]:
        if charset is not None:
            return self._resolve_extra_artifact(
                charset,
                default_name=None,
                registry=self.charset_registry,
                description="charset",
            )

        if self._weights_preset and self._weights_preset in self.charset_registry:
            return self._resolve_extra_artifact(
                self.charset_registry[self._weights_preset],
                default_name=None,
                registry=self.charset_registry,
                description="charset",
            )

        weights_path = Path(self.weights)
        candidates = [
            weights_path.with_suffix(".txt"),
            weights_path.parent / "dict.txt",
            weights_path.parent / "custom_dict.txt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.absolute())
        return None

    @staticmethod
    def _load_config_data(config_path: Optional[str]) -> Dict[str, Any]:
        if not config_path:
            return {}

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError(f"Expected config dict in {config_path}")
        return data

    @staticmethod
    def _resolve_rec_image_shape(
        rec_image_shape: Optional[Sequence[int]],
        config_data: Dict[str, Any],
    ) -> List[int]:
        if rec_image_shape is not None:
            values = [int(v) for v in rec_image_shape]
            if len(values) != 3:
                raise ValueError("rec_image_shape must contain exactly 3 integers")
            return values

        image_shape = (
            config_data.get("PreProcess", {})
            .get("transform_ops", [{}, {}, {}])[2]
            .get("RecResizeImg", {})
            .get("image_shape")
        )
        if image_shape:
            return [int(v) for v in image_shape]

        global_shape = config_data.get("Global", {}).get("d2s_train_image_shape")
        if global_shape:
            return [int(v) for v in global_shape]

        return [3, 48, 320]

    @staticmethod
    def _resolve_use_space_char(
        use_space_char: Optional[bool],
        config_data: Dict[str, Any],
    ) -> bool:
        if use_space_char is not None:
            return bool(use_space_char)
        return bool(config_data.get("Global", {}).get("use_space_char", False))

    @staticmethod
    def _read_charset_file(charset_path: str) -> List[str]:
        chars: List[str] = []
        with open(charset_path, "r", encoding="utf-8") as f:
            for line in f:
                chars.append(line.rstrip("\r\n"))
        return chars

    def _load_characters(
        self,
        *,
        charset_path: Optional[str],
        config_data: Dict[str, Any],
        use_space_char: bool,
    ) -> List[str]:
        if charset_path is not None:
            chars = self._read_charset_file(charset_path)
            if use_space_char and " " not in chars:
                chars.append(" ")
            return ["blank"] + chars

        config_chars = config_data.get("PostProcess", {}).get("character_dict")
        if config_chars:
            return ["blank"] + [str(ch) for ch in config_chars]

        raise FileNotFoundError(
            "Could not resolve character dictionary. "
            "Provide charset explicitly or place a compatible inference.yml next to the model."
        )

    @staticmethod
    def _validate_region_preparer(
        region_preparer: Union[str, Callable[..., Sequence[Any]]]
    ) -> Union[str, Callable[..., Sequence[Any]]]:
        if isinstance(region_preparer, str):
            if region_preparer not in REGION_PREPARER_PRESETS:
                raise ValueError(
                    f"region_preparer must be one of {REGION_PREPARER_PRESETS}, "
                    f"got: {region_preparer}"
                )
            return region_preparer
        if not callable(region_preparer):
            raise TypeError("region_preparer must be a preset name or callable")
        return region_preparer

    def _initialize_session(self) -> None:
        if self.onnx_session is not None:
            return

        self._prepare_runtime_dependencies()
        providers = self.runtime_providers()
        self.onnx_session = ort.InferenceSession(str(self.weights), providers=providers)
        self._log_device_info(self.onnx_session)

        input_shape = self.onnx_session.get_inputs()[0].shape
        if len(input_shape) >= 4 and isinstance(input_shape[3], int):
            self._input_width_override = int(input_shape[3])

    def _preprocess_image(
        self, image: Union[np.ndarray, str, Path, Image.Image]
    ) -> np.ndarray:
        img = read_image(image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Expected RGB image with shape [H, W, 3]")

        # PaddleOCR decoding pipeline uses BGR images.
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_h = self.img_h
        img_w = self._input_width_override or self.img_w
        h, w = img.shape[:2]
        ratio = w / float(max(h, 1))
        resized_w = img_w if int(np.ceil(img_h * ratio)) > img_w else int(np.ceil(img_h * ratio))
        resized_w = max(1, resized_w)

        interpolation = cv2.INTER_AREA if resized_w < w else cv2.INTER_LINEAR
        resized_image = cv2.resize(img, (resized_w, img_h), interpolation=interpolation)
        resized_image = resized_image.astype("float32").transpose((2, 0, 1)) / 255.0
        resized_image -= 0.5
        resized_image /= 0.5

        padding_im = np.zeros((self.img_c, img_h, img_w), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return np.expand_dims(padding_im, axis=0)

    def _apply_region_rotation(self, crop: np.ndarray) -> np.ndarray:
        if not self.rotate_threshold:
            return crop

        height, width = crop.shape[:2]
        if height > width * self.rotate_threshold:
            return np.rot90(crop, k=-1)
        return crop

    def _prepare_crop(self, crop: np.ndarray) -> np.ndarray:
        return self._apply_region_rotation(crop)

    def _extract_word_image(
        self, image: np.ndarray, polygon: np.ndarray
    ) -> Optional[np.ndarray]:
        return crop_axis_aligned(image, polygon, pad=0)

    @staticmethod
    def _normalize_text_regions(regions: Sequence[Any]) -> List[PreparedRegion]:
        return normalize_prepared_regions(regions)

    @staticmethod
    def _normalize_text_predictions(
        predictions: Sequence[Any],
    ) -> List[RecognitionPrediction]:
        return normalize_recognition_predictions(predictions)

    def _prepare_bbox_regions(
        self, page: Page, image: np.ndarray, options: Optional[Dict[str, Any]] = None
    ) -> List[PreparedRegion]:
        return prepare_bbox_regions(
            page,
            image,
            min_text_size=self.min_text_size,
            rotate_region=self._apply_region_rotation,
            options=options,
        )

    def _prepare_polygon_mask_regions(
        self, page: Page, image: np.ndarray, options: Optional[Dict[str, Any]] = None
    ) -> List[PreparedRegion]:
        return prepare_polygon_mask_regions(
            page,
            image,
            min_text_size=self.min_text_size,
            rotate_region=self._apply_region_rotation,
            options=options,
        )

    def _prepare_quad_warp_regions(
        self, page: Page, image: np.ndarray, options: Optional[Dict[str, Any]] = None
    ) -> List[PreparedRegion]:
        return prepare_quad_warp_regions(
            page,
            image,
            min_text_size=self.min_text_size,
            rotate_region=self._apply_region_rotation,
            options=options,
        )

    def _prepare_text_regions(
        self,
        page: Page,
        image: np.ndarray,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[PreparedRegion]:
        preset = self.region_preparer
        if not isinstance(preset, str):
            raise TypeError("_prepare_text_regions is available only for preset preparers")

        return prepare_text_regions(
            page=page,
            image=image,
            preset=preset,
            min_text_size=self.min_text_size,
            rotate_region=self._apply_region_rotation,
            options=options,
        )

    def _call_region_preparer(self, page: Page, image: np.ndarray) -> List[PreparedRegion]:
        return call_region_preparer(
            page=page,
            image=image,
            preparer=self.region_preparer,
            options=self.region_preparer_options,
            recognizer=self,
            min_text_size=self.min_text_size,
            rotate_region=self._apply_region_rotation,
        )

    def _warn_single_batch_only(self) -> None:
        if self._single_batch_warning_emitted:
            return
        print(
            "[WARN] ONNX model does not support dynamic batch inference; "
            "falling back to batch_size=1"
        )
        self._single_batch_warning_emitted = True

    def _effective_inference_batch_size(self, batch_size: int) -> int:
        if self.onnx_session is None:
            self._initialize_session()

        requested_batch_size = max(1, int(batch_size))
        if self._supports_multi_batch_inference is False:
            if requested_batch_size > 1:
                self._warn_single_batch_only()
            return 1

        input_shape = self.onnx_session.get_inputs()[0].shape
        output_shape = self.onnx_session.get_outputs()[0].shape
        input_batch_dim = input_shape[0] if input_shape else None
        output_batch_dim = output_shape[0] if output_shape else None

        if (
            isinstance(input_batch_dim, int)
            and input_batch_dim == 1
        ) or (
            isinstance(output_batch_dim, int)
            and output_batch_dim == 1
        ):
            self._supports_multi_batch_inference = False
            if requested_batch_size > 1:
                self._warn_single_batch_only()
            return 1

        return requested_batch_size

    @staticmethod
    def _is_batch_shape_error(exc: Exception) -> bool:
        message = str(exc)
        return any(
            marker in message
            for marker in (
                "ReshapeHelper",
                "requested shape",
                "cannot be reshaped to the requested shape",
                "running Reshape node",
            )
        )

    def _decode_recognition_logits(
        self,
        logits: np.ndarray,
    ) -> List[RecognitionPrediction]:
        preds_idx = logits.argmax(axis=2)
        preds_prob = logits.max(axis=2)

        results: List[RecognitionPrediction] = []
        for pred_row, prob_row in zip(preds_idx, preds_prob):
            char_list: List[str] = []
            conf_list: List[float] = []
            last_idx: Optional[int] = None

            for idx, prob in zip(pred_row, prob_row):
                idx = int(idx)
                if idx == 0 or idx == last_idx:
                    last_idx = idx
                    continue
                if idx < len(self.characters):
                    char_list.append(self.characters[idx])
                    conf_list.append(float(prob))
                last_idx = idx

            results.append(
                RecognitionPrediction(
                    text="".join(char_list),
                    confidence=float(np.mean(conf_list)) if conf_list else 0.0,
                )
            )

        return results

    def _predict_text_images(
        self,
        regions: Sequence[PreparedRegion],
        batch_size: Optional[int] = None,
    ) -> List[RecognitionPrediction]:
        return self._run_inference_batches(
            regions=regions,
            batch_size=batch_size,
        )

    def _run_inference_batches(
        self,
        regions: Sequence[PreparedRegion],
        batch_size: Optional[int] = None,
    ) -> List[RecognitionPrediction]:
        if self.onnx_session is None:
            self._initialize_session()

        if batch_size is None:
            batch_size = self.batch_size
        requested_batch_size = max(1, int(batch_size))
        effective_batch_size = self._effective_inference_batch_size(requested_batch_size)

        if not regions:
            return []

        results: List[RecognitionPrediction] = []
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name

        for i in range(0, len(regions), effective_batch_size):
            batch_regions = regions[i : i + effective_batch_size]
            original_batch_size = len(batch_regions)

            batch_tensors = [self._preprocess_image(region.image)[0] for region in batch_regions]
            inference_tensors = batch_tensors
            if 0 < original_batch_size < effective_batch_size:
                pad_tensor = batch_tensors[-1]
                inference_tensors = batch_tensors + [pad_tensor] * (
                    effective_batch_size - original_batch_size
                )

            batch_input = np.stack(inference_tensors, axis=0)

            try:
                logits = self.onnx_session.run([output_name], {input_name: batch_input})[0]
            except Exception as exc:
                if batch_input.shape[0] > 1 and self._is_batch_shape_error(exc):
                    self._supports_multi_batch_inference = False
                    self._warn_single_batch_only()
                    for batch_tensor in batch_tensors:
                        single_input = np.expand_dims(batch_tensor, axis=0)
                        single_logits = self.onnx_session.run(
                            [output_name],
                            {input_name: single_input},
                        )[0]
                        results.extend(self._decode_recognition_logits(single_logits))
                    continue
                raise

            if len(batch_regions) > 1 and self._supports_multi_batch_inference is None:
                self._supports_multi_batch_inference = True

            if len(logits) != original_batch_size:
                logits = logits[:original_batch_size]

            results.extend(self._decode_recognition_logits(logits))

        return results

    def _predict_word_images(
        self,
        images: List[Union[np.ndarray, str, Path, Image.Image]],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        regions = [
            PreparedRegion(
                text_span=None,
                image=read_image(image),
                polygon=np.empty((0, 2), dtype=np.float32),
                meta={"region_preparer": "legacy_raw_images"},
            )
            for image in images
        ]
        if batch_size is None:
            batch_size = self.batch_size
        predictions = self._predict_text_images(regions, batch_size=batch_size)
        return [
            {
                "text": prediction.text,
                "confidence": prediction.confidence,
                "meta": dict(prediction.meta),
            }
            for prediction in predictions
        ]

    def _save_debug_regions(
        self,
        regions: Sequence[PreparedRegion],
        debug_save_dir: Union[str, Path],
        predictions: Optional[Sequence[RecognitionPrediction]] = None,
        *,
        write_images: bool = True,
    ) -> None:
        save_debug_regions(
            regions=regions,
            debug_save_dir=debug_save_dir,
            predictions=predictions,
            write_images=write_images,
        )

    @staticmethod
    def _apply_text_predictions(
        regions: Sequence[PreparedRegion],
        predictions: Sequence[RecognitionPrediction],
    ) -> None:
        if len(regions) != len(predictions):
            raise ValueError(
                "predictor must return the same number of predictions as regions"
            )

        for region, prediction in zip(regions, predictions):
            region.text_span.text = prediction.text
            region.text_span.recognition_confidence = prediction.confidence

    def predict(
        self,
        page: Page,
        image: Optional[Union[np.ndarray, str, Path, Image.Image]] = None,
        batch_size: Optional[int] = None,
        debug_save_dir: Optional[Union[str, Path]] = None,
        profile: bool = False,
    ) -> Page:
        result_page = page.model_copy(deep=True)
        if image is None:
            return result_page

        if debug_save_dir is None:
            debug_save_dir = self.default_debug_save_dir
        if batch_size is None:
            batch_size = self.batch_size

        image_array = read_image(image)
        regions = self._call_region_preparer(result_page, image_array)
        if not regions:
            return result_page

        if debug_save_dir is not None:
            self._save_debug_regions(
                regions=regions,
                debug_save_dir=debug_save_dir,
                write_images=True,
            )

        predictions = self._normalize_text_predictions(
            self._predict_text_images(
                regions=regions,
                batch_size=batch_size,
            )
        )
        if debug_save_dir is not None:
            self._save_debug_regions(
                regions=regions,
                debug_save_dir=debug_save_dir,
                predictions=predictions,
                write_images=False,
            )

        self._apply_text_predictions(regions, predictions)
        return result_page


__all__ = ["PPOCRv5Rec"]
