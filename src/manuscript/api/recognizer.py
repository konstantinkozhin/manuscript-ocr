import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

from manuscript.data import Page
from manuscript.utils.geometry import crop_axis_aligned
from manuscript.utils.io import read_image
from manuscript.recognizers._common.debug import save_debug_regions
from manuscript.recognizers._common.region_preparers import (
    call_region_preparer, prepare_bbox_regions, prepare_polygon_mask_regions,
    prepare_quad_warp_regions, prepare_text_regions,
)
from manuscript.recognizers._common.region_types import (
    REGION_PREPARER_PRESETS, PreparedRegion, RecognitionPrediction,
    normalize_prepared_regions, normalize_recognition_predictions,
)
from .base import BaseArtifactModel


class BaseRecognizer(BaseArtifactModel):
    """Artifact-backed recognizer with a shared Page and text-region workflow.

    Models implement ``_predict_text_images`` to recognize prepared crops.
    Custom recognizers may instead override ``predict`` directly.
    """

    config_registry: Dict[str, str] = {}

    def __init__(
        self,
        weights: Optional[str] = None,
        device: Optional[str] = None,
        force_download: bool = False,
        rotate_threshold: Optional[float] = 1.5,
        region_preparer: Union[str, Callable[..., Sequence[Any]]] = "bbox",
        region_preparer_options: Optional[Dict[str, Any]] = None,
        min_text_size: int = 5,
        batch_size: int = 16,
        debug_save_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ):
        for removed, replacement in (
            ("region_predictor", "Pass a custom recognizer to Pipeline instead."),
            ("recognizer_debug_dir", "Use debug_save_dir instead."),
        ):
            if removed in kwargs:
                raise TypeError(f"{removed} has been removed from {type(self).__name__}. {replacement}")
        self._positive_integer(batch_size, "batch_size")
        if isinstance(min_text_size, bool) or not isinstance(min_text_size, int) or min_text_size < 0:
            raise ValueError("min_text_size must be a nonnegative integer")
        if rotate_threshold is not None and (isinstance(rotate_threshold, bool)
                or not np.isfinite(rotate_threshold) or rotate_threshold < 0):
            raise ValueError("rotate_threshold must be nonnegative and finite or None")
        super().__init__(weights=weights, device=device, force_download=force_download, **kwargs)
        self.rotate_threshold = rotate_threshold
        self.region_preparer = self._validate_region_preparer(region_preparer)
        self.region_preparer_options = dict(region_preparer_options or {})
        self.min_text_size = min_text_size
        self.batch_size = max(1, int(batch_size))
        self.default_debug_save_dir = (
            Path(debug_save_dir).expanduser() if debug_save_dir is not None else None
        )

    @staticmethod
    def _positive_integer(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    def _decode_predictions(self, logits, return_raw=False):
        predictions = self._decode_recognition_logits(logits)
        if return_raw:
            for prediction, row in zip(predictions, logits):
                prediction.meta["model_output"] = row.copy()
        return predictions

    def _predict_text_images(
        self, regions: Sequence[PreparedRegion], batch_size: Optional[int] = None,
    ) -> List[RecognitionPrediction]:
        """Recognize crops; required when using the default ``predict`` workflow."""
        raise NotImplementedError("Implement _predict_text_images or override predict.")

    def _resolve_config(self, config: Optional[str]) -> Optional[str]:
        if config is not None:
            return self._resolve_extra_artifact(
                config,
                default_name=None,
                registry=self.config_registry,
                description="config",
            )

        if getattr(self, '_resolved_model_artifacts', None):
            value = self._resolved_model_artifacts.get('config')
            return str(value) if value else None

        if getattr(self, "_weights_preset", None) in self.config_registry:
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
                import yaml

                data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError(f"Expected config dict in {config_path}")
        return data

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

    def _apply_region_rotation(self, crop: np.ndarray) -> np.ndarray:
        if not self.rotate_threshold:
            return crop

        height, width = crop.shape[:2]
        if height > width * self.rotate_threshold:
            return np.rot90(crop, k=-1)
        return crop

    def _prepare_crop(self, crop: np.ndarray) -> np.ndarray:
        """Compatibility alias for crop orientation."""
        return self._apply_region_rotation(crop)

    def _extract_word_image(
        self, image: np.ndarray, polygon: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compatibility helper for axis-aligned crops."""
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
        return_raw: bool = False,
    ) -> Union[Page, Dict[str, Any]]:
        """Return a Page copy with recognized text and confidence for its regions.

        Without an image, return an unchanged copy. Debug crops are saved
        after region preparation and rotation, before model preprocessing.
        With return_raw=True, return a dictionary containing the Page and
        per-region predictions with model-specific metadata.
        """
        result_page = page.model_copy(deep=True)
        def result(regions=(), predictions=()):
            if not return_raw:
                return result_page
            return {"page": result_page, "predictions": [
                {"text": prediction.text, "confidence": prediction.confidence,
                 "polygon": region.polygon.copy(), "metadata": dict(prediction.meta)}
                for region, prediction in zip(regions, predictions)]}

        if image is None:
            return result()

        if debug_save_dir is None:
            debug_save_dir = self.default_debug_save_dir
        if batch_size is None:
            batch_size = self.batch_size
        self._positive_integer(batch_size, "batch_size")

        image_array = read_image(image)
        regions = self._call_region_preparer(result_page, image_array)
        if not regions:
            return result()

        if debug_save_dir is not None:
            self._save_debug_regions(
                regions=regions,
                debug_save_dir=debug_save_dir,
                write_images=True,
            )

        from ._page_helpers import filter_callable_kwargs
        options = {"regions": regions, "batch_size": batch_size}
        if return_raw:
            options["return_raw"] = True
        predictions = self._normalize_text_predictions(self._predict_text_images(
            **filter_callable_kwargs(self._predict_text_images, options)))
        if debug_save_dir is not None:
            self._save_debug_regions(
                regions=regions,
                debug_save_dir=debug_save_dir,
                predictions=predictions,
                write_images=False,
            )

        self._apply_text_predictions(regions, predictions)
        return result(regions, predictions)


__all__ = ["BaseRecognizer"]
