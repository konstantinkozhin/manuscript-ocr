import os
import json
import traceback
from pathlib import Path
from typing import Callable, List, Union, Optional, Sequence, Dict, Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from manuscript.api.recognizer import BaseRecognizer
from manuscript.data import Page
from manuscript.utils import (
    crop_axis_aligned,
    read_image,
)

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
from .data.transforms import load_charset

# Optional imports for training (not needed for inference)
try:
    from .training.train import Config, run_training

    _TRAINING_AVAILABLE = True
    _TRAINING_IMPORT_ERROR = None
    _TRAINING_IMPORT_TRACEBACK = None
except ImportError as exc:
    Config = None
    run_training = None
    _TRAINING_AVAILABLE = False
    _TRAINING_IMPORT_ERROR = exc
    _TRAINING_IMPORT_TRACEBACK = traceback.format_exc()

class TRBA(BaseRecognizer):
    """
    Инициализация модели распознавания текста TRBA с использованием ONNX Runtime.

    Параметры
    ----------
    weights : str or Path, optional
        Путь или идентификатор для весов ONNX-модели. Поддерживаются:

        - Локальный путь к файлу: ``"path/to/model.onnx"``
        - HTTP/HTTPS URL: ``"https://example.com/model.onnx"``
        - GitHub-релиз: ``"github://owner/repo/tag/file.onnx"``
        - Google Drive: ``"gdrive:FILE_ID"``
        - Предустановленное имя: ``"trba_lite_g1"`` или ``"trba_base_g1"`` (из pretrained_registry)
        - ``None``: автоматически загружает пресет по умолчанию (trba_lite_g1)

    config : str or Path, optional
        Путь или идентификатор для конфигурационного файла JSON модели. Поддерживаются
        те же схемы URL, что и для ``weights``. Если ``None``, конфигурация определяется
        автоматически по расположению весов или используется конфигурация пресета по умолчанию.
    charset : str or Path, optional
        Путь или идентификатор для файла набора символов. Если ``None``, выполняется поиск
        набора символов рядом с весами или используется набор символов пакета по умолчанию.
    device : {"cuda", "coreml", "cpu"}, optional
        Устройство для вычислений. Если ``None``, автоматически выбирается CPU.
        Для ускорения на GPU/CoreML:

        - CUDA (NVIDIA): ``pip install onnxruntime-gpu``
        - CoreML (Apple Silicon M1/M2/M3): ``pip install onnxruntime-silicon``

        По умолчанию ``None`` (CPU).
    rotate_threshold : float or None, optional
        Порог соотношения сторон для поворота вертикальных кропов текстовых областей
        перед распознаванием. Если ``height > width * rotate_threshold``, кроп
        поворачивается на 90 градусов по часовой стрелке. Установите ``0`` или ``None``
        для отключения. По умолчанию ``1.5``.
    region_preparer : {"bbox", "polygon_mask", "quad_warp"} or callable, optional
        Стратегия преобразования полигонов ``Page`` в кропы для распознавания.
        ``"bbox"`` извлекает выровненные по осям ограничивающие прямоугольники для
        произвольных полигонов. ``"polygon_mask"`` маскирует пиксели за пределами полигона
        внутри плотного кропа и также поддерживает произвольные полигоны. ``"quad_warp"``
        выпрямляет только 4-точечные полигоны с помощью перспективного преобразования перед
        распознаванием. Можно также передать пользовательский callable, который должен
        возвращать список подготовленных текстовых областей. По умолчанию ``"bbox"``.
    region_preparer_options : dict or None, optional
        Опциональная конфигурация для встроенных preparers. По умолчанию ``None``.
        Типичные параметры: ``pad`` для ``"bbox"`` и ``"polygon_mask"``, или
        ``output_size=(width, height)`` для ``"quad_warp"``. Полигоны без 4-х точек,
        переданные в ``"quad_warp"``, по умолчанию возвращаются к bbox-кропам.
    min_text_size : int, optional
        Минимальная ширина/высота кропа в пикселях для запуска распознавания текстовой
        области. Текстовые области ниже этого порога пропускаются. По умолчанию ``5``.
    batch_size : int, optional
        Размер батча для инференса по умолчанию, используемый когда
        ``predict(..., batch_size=...)`` не задан. По умолчанию ``128``.
    **kwargs
        Дополнительные параметры конфигурации (зарезервированы для будущего использования).

    Raises
    ------
    FileNotFoundError
        Если указанные файлы не существуют.
    ValueError
        Если формат весов недопустим.

    Notes
    -----
    Класс предоставляет три основных публичных метода:

    - ``predict`` — запуск распознавания текстовых областей в объекте ``Page``.
    - ``train`` — высокоуровневая точка входа для обучения модели TRBA на пользовательских наборах данных.
    - ``export`` — статический метод для экспорта модели PyTorch в формат ONNX.

    Модель использует ONNX Runtime для быстрого инференса на CPU и GPU.
    Для ускорения на GPU установите: ``pip install onnxruntime-gpu``

    Examples
    --------
    Создание распознавателя с пресетом по умолчанию (автозагрузка):

    >>> from manuscript.recognizers import TRBA
    >>> recognizer = TRBA()

    Загрузка из локального ONNX-файла:

    >>> recognizer = TRBA(weights="path/to/model.onnx")

    Загрузка из GitHub-релиза:

    >>> recognizer = TRBA(
    ...     weights="github://owner/repo/v1.0/model.onnx",
    ...     config="github://owner/repo/v1.0/config.json"
    ... )

    Принудительное использование CPU:

    >>> recognizer = TRBA(weights="model.onnx", device="cpu")
    """

    default_weights_name = "trba_lite_g1"

    pretrained_registry = {
        "trba_lite_g1": "github://konstantinkozhin/manuscript-ocr/v0.1.0/trba_lite_g1.onnx",
        "trba_lite_g2": "github://konstantinkozhin/manuscript-ocr/v0.1.0/trba_lite_g2.onnx",
        "trba_base_g1": "github://konstantinkozhin/manuscript-ocr/v0.1.0/trba_base_g1.onnx",
    }

    config_registry = {
        "trba_lite_g1": "github://konstantinkozhin/manuscript-ocr/v0.1.0/trba_lite_g1.json",
        "trba_lite_g2": "github://konstantinkozhin/manuscript-ocr/v0.1.0/trba_lite_g2.json",
        "trba_base_g1": "github://konstantinkozhin/manuscript-ocr/v0.1.0/trba_base_g1.json",
    }

    charset_registry = {
        "trba_lite_g1": "github://konstantinkozhin/manuscript-ocr/v0.1.0/trba_lite_g1.txt",
        "trba_lite_g2": "github://konstantinkozhin/manuscript-ocr/v0.1.0/trba_lite_g2.txt",
        "trba_base_g1": "github://konstantinkozhin/manuscript-ocr/v0.1.0/trba_base_g1.txt",
    }

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
        batch_size: int = 128,
        **kwargs,
    ):
        if "region_predictor" in kwargs:
            raise TypeError(
                "region_predictor has been removed from TRBA. "
                "Pass a custom recognizer to Pipeline instead."
            )
        if "recognizer_debug_dir" in kwargs:
            raise TypeError(
                "recognizer_debug_dir has been removed from TRBA. "
                "Use debug_save_dir instead."
            )

        default_debug_save_dir = kwargs.pop("debug_save_dir", None)

        # Initialize artifact-backed recognizer infrastructure.
        super().__init__(
            weights=weights,
            device=device,
            force_download=force_download,
            **kwargs,
        )

        # Resolve config
        self.config_path = self._resolve_config(config)

        # Resolve charset
        self.charset_path = self._resolve_charset(charset)

        # Load config
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        self.max_length = config_dict.get("max_len", 25)
        self.hidden_size = config_dict.get("hidden_size", 256)
        self.num_encoder_layers = config_dict.get("num_encoder_layers", 2)
        self.img_h = config_dict.get("img_h", 32)
        self.img_w = config_dict.get("img_w", 256)
        self.cnn_in_channels = config_dict.get("cnn_in_channels", 3)
        self.cnn_out_channels = config_dict.get("cnn_out_channels", 512)
        self.cnn_backbone = config_dict.get("cnn_backbone", "seresnet31")

        # Load charset
        if not Path(self.charset_path).exists():
            raise FileNotFoundError(f"Charset file not found: {self.charset_path}")

        self.itos, self.stoi = load_charset(self.charset_path)
        self.pad_id = self.stoi["<PAD>"]
        self.sos_id = self.stoi["<SOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.blank_id = self.stoi.get("<BLANK>", None)

        # Verify ONNX file exists
        if not Path(self.weights).exists():
            raise FileNotFoundError(f"Model file not found: {self.weights}")

        if Path(self.weights).suffix.lower() != ".onnx":
            raise ValueError(f"Expected .onnx file, got: {self.weights}")

        # Initialize ONNX session
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

    def _resolve_config(self, config: Optional[str]) -> str:
        """
        Resolve config path using shared artifact resolution.
        Falls back to inferring from weights location.

        Search order:
        1. Explicit config parameter (if provided)
        2. Preset name from config_registry (if weights stem matches)
        3. Same filename as weights but with .json extension
        4. Default preset config
        """
        if config is not None:
            # Use shared artifact resolution.
            return self._resolve_extra_artifact(
                config,
                default_name=None,
                registry=self.config_registry,
                description="config",
            )

        # Try to infer from weights location
        weights_path = Path(self.weights)
        weights_stem = weights_path.stem

        # 1. Try preset name in config registry
        if weights_stem in self.config_registry:
            return self._resolve_extra_artifact(
                weights_stem,
                default_name=None,
                registry=self.config_registry,
                description="config",
            )

        # 2. Try same filename with .json extension (e.g., model.onnx -> model.json)
        config_candidate = weights_path.with_suffix(".json")
        if config_candidate.exists():
            return str(config_candidate.absolute())

        # 3. Use default preset config
        if (
            self.default_weights_name
            and self.default_weights_name in self.config_registry
        ):
            return self._resolve_extra_artifact(
                self.default_weights_name,
                default_name=None,
                registry=self.config_registry,
                description="config",
            )

        raise FileNotFoundError(
            f"Could not find config file for weights: {self.weights}. "
            f"Expected config at: {config_candidate}. "
            f"Please specify config explicitly or ensure config file has same name as weights."
        )

    def _resolve_charset(self, charset: Optional[str]) -> str:
        """
        Resolve charset path using shared artifact resolution.
        Falls back to inferring from weights location or package default.

        Search order:
        1. Explicit charset parameter (if provided)
        2. Preset name from charset_registry (if weights stem matches)
        3. Same filename as weights but with .txt extension
        4. Default preset charset
        5. Package default charset (configs/charset.txt)
        """
        if charset is not None:
            # Use shared artifact resolution.
            return self._resolve_extra_artifact(
                charset,
                default_name=None,
                registry=self.charset_registry,
                description="charset",
            )

        # Try to infer from weights location
        weights_path = Path(self.weights)
        weights_stem = weights_path.stem

        # 1. Try preset name in charset registry
        if weights_stem in self.charset_registry:
            return self._resolve_extra_artifact(
                weights_stem,
                default_name=None,
                registry=self.charset_registry,
                description="charset",
            )

        # 2. Try same filename with .txt extension (e.g., model.onnx -> model.txt)
        charset_candidate = weights_path.with_suffix(".txt")
        if charset_candidate.exists():
            return str(charset_candidate.absolute())

        # 3. Try default preset charset
        if (
            self.default_weights_name
            and self.default_weights_name in self.charset_registry
        ):
            return self._resolve_extra_artifact(
                self.default_weights_name,
                default_name=None,
                registry=self.charset_registry,
                description="charset",
            )

        # 4. Fallback to package default charset
        current_dir = Path(__file__).parent
        package_charset = current_dir / "configs" / "charset.txt"
        if package_charset.exists():
            return str(package_charset.absolute())

        raise FileNotFoundError(
            f"Could not find charset file. "
            f"Expected charset at: {charset_candidate} or {package_charset}. "
            f"Please specify charset explicitly or ensure charset file has same name as weights."
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

    def _initialize_session(self):
        """Initialize ONNX Runtime session (lazy loading)."""
        if self.onnx_session is not None:
            return

        self._prepare_runtime_dependencies()
        providers = self.runtime_providers()
        self.onnx_session = ort.InferenceSession(str(self.weights), providers=providers)
        self._log_device_info(self.onnx_session)

    def _preprocess_image(
        self, image: Union[np.ndarray, str, Path, Image.Image]
    ) -> np.ndarray:
        """
        Preprocess a prepared crop for ONNX inference and return ``[1, 3, H, W]``.
        """
        img = read_image(image)  # Returns RGB uint8 [H, W, 3]

        h, w = img.shape[:2]
        scale = min(self.img_h / max(h, 1), self.img_w / max(w, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        if new_h < h or new_w < w:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_LINEAR

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        canvas = np.full((self.img_h, self.img_w, 3), 255, dtype=np.uint8)

        y_offset = (self.img_h - new_h) // 2
        x_offset = 0
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = img_resized

        img_normalized = (canvas.astype(np.float32) - 127.5) / 127.5
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        return np.expand_dims(img_chw, axis=0)

    def _apply_region_rotation(self, crop: np.ndarray) -> np.ndarray:
        """
        Rotate tall crops to horizontal orientation when auto-rotation is enabled.
        """
        if not self.rotate_threshold:
            return crop

        height, width = crop.shape[:2]
        if height > width * self.rotate_threshold:
            return np.rot90(crop, k=-1)
        return crop

    def _prepare_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Backward-compatible alias for crop orientation logic.
        """
        return self._apply_region_rotation(crop)

    def _extract_word_image(
        self, image: np.ndarray, polygon: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Backward-compatible axis-aligned text-span crop helper.
        """
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
        preds = np.argmax(logits, axis=-1)
        probs = self._softmax(logits, axis=-1)

        results: List[RecognitionPrediction] = []
        for pred_row, prob_row in zip(preds, probs):
            decoded_chars = []
            for token_id in pred_row:
                if token_id == self.eos_id:
                    break
                if token_id not in [self.pad_id, self.sos_id] and token_id < len(self.itos):
                    decoded_chars.append(self.itos[token_id])

            seq_probs = []
            for t, token_id in enumerate(pred_row):
                if token_id == self.eos_id:
                    break
                if token_id not in [self.pad_id, self.sos_id]:
                    seq_probs.append(prob_row[t, token_id])

            results.append(
                RecognitionPrediction(
                    text="".join(decoded_chars),
                    confidence=float(np.mean(seq_probs)) if seq_probs else 0.0,
                )
            )

        return results

    def _predict_text_images(
        self,
        regions: Sequence[PreparedRegion],
        batch_size: Optional[int] = None,
    ) -> List[RecognitionPrediction]:
        """
        Run ONNX inference on prepared text regions.
        """
        if self.onnx_session is None:
            self._initialize_session()

        if not regions:
            return []

        results: List[RecognitionPrediction] = []
        if batch_size is None:
            batch_size = self.batch_size
        effective_batch_size = self._effective_inference_batch_size(batch_size)
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name

        for i in range(0, len(regions), effective_batch_size):
            batch_regions = regions[i : i + effective_batch_size]
            batch_tensors = [self._preprocess_image(region.image)[0] for region in batch_regions]
            batch_input = np.stack(batch_tensors, axis=0)
            try:
                logits = self.onnx_session.run([output_name], {input_name: batch_input})[0]
            except Exception as exc:
                if len(batch_regions) > 1 and self._is_batch_shape_error(exc):
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

            results.extend(self._decode_recognition_logits(logits))

        return results

    def _predict_word_images(
        self,
        images: List[Union[np.ndarray, str, Path, Image.Image]],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Backward-compatible wrapper for raw image list inference.
        """
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
            {"text": prediction.text, "confidence": prediction.confidence, "meta": dict(prediction.meta)}
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
    ) -> Page:
        """
        Распознаёт текст для текстовых областей на ``Page`` и возвращает обновлённый ``Page``.

        Параметры
        ----------
        page : Page
            Объект страницы с обнаруженными полигонами текстовых областей.
        image : str, Path, numpy.ndarray, or PIL.Image, optional
            Исходное изображение страницы для извлечения текстовых регионов. Если ``None``,
            распознавание пропускается и возвращается глубокая копия ``page``.
        batch_size : int or None, optional
            Количество подготовленных текстовых регионов для одновременной обработки. Если
            ``None``, используется ``batch_size``, переданный в конструктор.
        debug_save_dir : str or Path, optional
            Если указан, сохраняет подготовленные кропы для распознавания в эту директорию
            в виде файлов ``*.png`` вместе с ``index.json``. Кропы сохраняются после
            ``region_preparer`` и авторотации, то есть в той же ориентации, в которой
            поступают на вход инференса распознавателя.

        Возвращает
        -------
        Page
            Новый объект ``Page`` с заполненными ``text`` и ``recognition_confidence``
            для обработанных текстовых областей.
        """
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

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax along axis."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    def train(
        train_csvs: Union[str, Sequence[str]],
        train_roots: Union[str, Sequence[str]],
        val_csvs: Optional[Union[str, Sequence[str]]] = None,
        val_roots: Optional[Union[str, Sequence[str]]] = None,
        *,
        exp_dir: Optional[str] = None,
        charset_path: Optional[str] = None,
        encoding: str = "utf-8",
        img_h: int = 64,
        img_w: int = 256,
        max_len: int = 25,
        hidden_size: int = 256,
        num_encoder_layers: int = 3,
        cnn_in_channels: int = 3,
        cnn_out_channels: int = 512,
        cnn_backbone: str = "seresnet31",
        ctc_weight: float = 0.3,
        ctc_weight_decay_epochs: int = 50,
        ctc_weight_min: float = 0.0,
        max_grad_norm: float = 5.0,
        batch_size: int = 32,
        epochs: int = 20,
        lr: float = 1e-3,
        optimizer: str = "AdamW",
        scheduler: str = "OneCycleLR",
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        val_interval: int = 1,
        val_size: int = 3000,
        train_proportions: Optional[Sequence[float]] = None,
        num_workers: int = 0,
        seed: int = 42,
        resume_from: Optional[str] = None,
        save_interval: Optional[int] = None,
        device: str = "cuda",
        freeze_cnn: str = "none",
        freeze_enc_rnn: str = "none",
        freeze_attention: str = "none",
        pretrain_weights: Optional[object] = "default",
        **extra_config: Any,
    ):
        """
        Обучение модели распознавания текста TRBA на пользовательских наборах данных.

        Параметры
        ----------
        train_csvs : str, Path or sequence of paths
            Путь(и) к обучающим CSV-файлам. Каждый CSV должен содержать столбцы:
            ``image_path`` (относительно ``train_roots``) и ``text`` (эталонная транскрипция).
        train_roots : str, Path or sequence of paths
            Корневая директория/директории с обучающими изображениями. Должны совпадать по
            длине с ``train_csvs``.
        val_csvs : str, Path, sequence of paths, or None, optional
            Путь(и) к валидационным CSV-файлам в том же формате, что и ``train_csvs``.
            Если ``None``, валидация не выполняется. По умолчанию ``None``.
        val_roots : str, Path, sequence of paths, or None, optional
            Корневая директория/директории для валидационных изображений. Должны совпадать
            по длине с ``val_csvs``, если указаны. По умолчанию ``None``.
        exp_dir : str or Path, optional
            Директория эксперимента для сохранения чекпоинтов и логов.
            Если ``None``, генерируется автоматически на основе метки времени.
            По умолчанию ``None``.
        charset_path : str or Path, optional
            Путь к файлу набора символов. Если ``None``, используется набор символов
            пакета по умолчанию. По умолчанию ``None``.
        encoding : str, optional
            Кодировка текста для чтения CSV-файлов. По умолчанию ``"utf-8"``.
        img_h : int, optional
            Целевая высота входных изображений (пиксели). По умолчанию 64.
        img_w : int, optional
            Целевая ширина входных изображений (пиксели). По умолчанию 256.
        max_len : int, optional
            Максимальная длина последовательности для распознавания текста. По умолчанию 25.
        hidden_size : int, optional
            Размер скрытого измерения для RNN-энкодера/декодера. По умолчанию 256.
        num_encoder_layers : int, optional
            Количество двунаправленных LSTM-слоёв в энкодере. По умолчанию 2.
        cnn_in_channels : int, optional
            Количество входных каналов CNN-бэкбона (3 для RGB, 1 для оттенков серого).
            По умолчанию 3.
        cnn_out_channels : int, optional
            Количество выходных каналов CNN-бэкбона. По умолчанию 512.
        cnn_backbone : {"seresnet31", "seresnet31-lite"}, optional
            Вариант CNN-бэкбона. ``"seresnet31"`` — стандартный SE-ResNet-31,
            ``"seresnet31-lite"`` — облегчённая версия с depthwise-свёртками.
            По умолчанию ``"seresnet31"``.
        ctc_weight : float, optional
            Начальный вес CTC-лосса при обучении (CTC всегда используется для стабильности):
            ``loss = attn_loss * (1 - ctc_weight) + ctc_loss * ctc_weight``.
            Вес CTC убывает с эпохами. По умолчанию 0.3.
        ctc_weight_decay_epochs : int, optional
            Число эпох, за которое вес CTC убывает до минимального значения.
            По умолчанию 50.
        ctc_weight_min : float, optional
            Минимальное значение веса CTC после затухания. По умолчанию 0.0.
        max_grad_norm : float, optional
            Максимальная норма градиента для клиппинга (предотвращает взрывной
            рост градиентов/NaN). По умолчанию 5.0.
        batch_size : int, optional
            Размер батча при обучении. По умолчанию 32.
        epochs : int, optional
            Количество эпох обучения. По умолчанию 20.
        lr : float, optional
            Скорость обучения. По умолчанию 1e-3.
        optimizer : {"Adam", "SGD", "AdamW"}, optional
            Тип оптимизатора. По умолчанию ``"AdamW"``.
        scheduler : {"ReduceLROnPlateau", "CosineAnnealingLR", "OneCycleLR", "None"}, optional
            Тип планировщика скорости обучения:

            - ``"OneCycleLR"`` — one-cycle политика с косинусным отжигом (по умолчанию, рекомендуется)
            - ``"ReduceLROnPlateau"`` — снижение LR при плато валидационного лосса
            - ``"CosineAnnealingLR"`` — косинусный отжиг по эпохам
            - ``"None"`` или ``None`` — постоянная скорость обучения

            По умолчанию ``"OneCycleLR"``.
        weight_decay : float, optional
            Коэффициент L2-регуляризации весов. По умолчанию 0.0.
        momentum : float, optional
            Моментум для оптимизатора SGD. По умолчанию 0.9.
        val_interval : int, optional
            Выполнять валидацию каждые N эпох. По умолчанию 1.
        val_size : int, optional
            Максимальное количество валидационных примеров. По умолчанию 3000.
        train_proportions : sequence of float, optional
            Пропорции выборки для нескольких обучающих наборов данных. Должны давать
            сумму 1.0 и совпадать по длине с ``train_csvs``. Если ``None``, наборы данных
            конкатенируются равномерно. По умолчанию ``None``.
        num_workers : int, optional
            Количество воркеров для загрузки данных. По умолчанию 0.
        seed : int, optional
            Случайное зерно для воспроизводимости. По умолчанию 42.
        resume_from : str or Path, optional
            Путь к файлу чекпоинта для возобновления обучения. По умолчанию ``None``.
        save_interval : int, optional
            Сохранять чекпоинт каждые N эпох. Если ``None``, сохраняется только лучшая
            модель. По умолчанию ``None``.
        device : {"cuda", "cpu"}, optional
            Устройство для обучения. По умолчанию ``"cuda"``.
        freeze_cnn : {"none", "all", "first", "last"}, optional
            Политика заморозки CNN. По умолчанию ``"none"``.
        freeze_enc_rnn : {"none", "all", "first", "last"}, optional
            Политика заморозки энкодерной RNN. По умолчанию ``"none"``.
        freeze_attention : {"none", "all"}, optional
            Политика заморозки модуля внимания. По умолчанию ``"none"``.
        pretrain_weights : str, Path, bool, or None, optional
            Предобученные веса для инициализации:

            - ``"default"`` или ``True`` — использовать веса релиза
            - ``None`` или ``False`` — обучение с нуля
            - str/Path — путь или URL к пользовательскому файлу весов

            По умолчанию ``"default"``.
        **extra_config : dict, optional
            Дополнительные параметры конфигурации, передаваемые в конфиг обучения.

        Возвращает
        -------
        str
            Путь к чекпоинту лучшей модели, сохранённой во время обучения.

        Examples
        --------
        Обучение на одном наборе данных с валидацией:

        >>> from manuscript.recognizers import TRBA
        >>>
        >>> best_model = TRBA.train(
        ...     train_csvs="data/train.csv",
        ...     train_roots="data/train_images",
        ...     val_csvs="data/val.csv",
        ...     val_roots="data/val_images",
        ...     exp_dir="./experiments/trba_exp1",
        ...     epochs=50,
        ...     batch_size=64,
        ...     img_h=64,
        ...     img_w=256,
        ... )
        >>> print(f"Best model saved at: {best_model}")

        Обучение на нескольких наборах данных с пользовательскими пропорциями:

        >>> train_csvs = ["data/dataset1/train.csv", "data/dataset2/train.csv"]
        >>> train_roots = ["data/dataset1/images", "data/dataset2/images"]
        >>> train_proportions = [0.7, 0.3]  # 70% из dataset1, 30% из dataset2
        >>>
        >>> best_model = TRBA.train(
        ...     train_csvs=train_csvs,
        ...     train_roots=train_roots,
        ...     train_proportions=train_proportions,
        ...     val_csvs="data/val.csv",
        ...     val_roots="data/val_images",
        ...     epochs=100,
        ...     lr=5e-4,
        ...     optimizer="AdamW",
        ...     weight_decay=1e-4,
        ... )

        Возобновление обучения с чекпоинта:

        >>> best_model = TRBA.train(
        ...     train_csvs="data/train.csv",
        ...     train_roots="data/train_images",
        ...     resume_from="experiments/trba_exp1/checkpoints/last.pth",
        ...     epochs=100,
        ... )

        Дообучение на предобученных весах с заморозкой CNN:

        >>> best_model = TRBA.train(
        ...     train_csvs="data/finetune.csv",
        ...     train_roots="data/finetune_images",
        ...     pretrain_weights="default",
        ...     freeze_cnn="all",
        ...     epochs=20,
        ...     lr=1e-4,
        ... )

        Обучение с CTC для стабильности (всегда включён):

        >>> best_model = TRBA.train(
        ...     train_csvs="data/train.csv",
        ...     train_roots="data/train_images",
        ...     optimizer="AdamW",
        ...     scheduler="OneCycleLR",
        ...     lr=1e-3,
        ...     ctc_weight=0.3,
        ...     ctc_weight_decay_epochs=50,
        ...     max_grad_norm=5.0,
        ...     epochs=100,
        ... )
        """
        if not _TRAINING_AVAILABLE:
            details = ""
            if _TRAINING_IMPORT_TRACEBACK:
                details = (
                    "\nOriginal import error traceback:\n"
                    f"{_TRAINING_IMPORT_TRACEBACK.rstrip()}"
                )
            elif _TRAINING_IMPORT_ERROR is not None:
                details = f"\nOriginal import error: {_TRAINING_IMPORT_ERROR!r}"
            raise ImportError(
                "Training dependencies not available. "
                "Install with: pip install manuscript-ocr[dev]"
                f"{details}"
            ) from _TRAINING_IMPORT_ERROR

        def _ensure_path_list(
            value: Optional[Union[str, Sequence[Optional[str]]]],
            field_name: str,
            allow_none: bool = False,
            allow_item_none: bool = False,
        ) -> Optional[List[Optional[str]]]:
            if value is None:
                if allow_none:
                    return None
                raise ValueError(f"{field_name} must be provided")

            if isinstance(value, (list, tuple)):
                raw_items = list(value)
            else:
                raw_items = [value]

            if not raw_items:
                raise ValueError(f"{field_name} must not be empty")

            result: List[Optional[str]] = []
            for item in raw_items:
                if item is None:
                    if allow_item_none:
                        result.append(None)
                    else:
                        raise ValueError(
                            f"{field_name} contains None but allow_item_none is False"
                        )
                else:
                    result.append(os.fspath(item))
            return result

        train_csvs_list = _ensure_path_list(train_csvs, "train_csvs")
        train_roots_list = _ensure_path_list(train_roots, "train_roots")

        if len(train_csvs_list) != len(train_roots_list):
            raise ValueError(
                "train_csvs and train_roots must contain the same number of items"
            )

        val_csvs_list = _ensure_path_list(
            val_csvs, "val_csvs", allow_none=True, allow_item_none=True
        )
        val_roots_list = _ensure_path_list(
            val_roots, "val_roots", allow_none=True, allow_item_none=True
        )

        if (val_csvs_list is None) ^ (val_roots_list is None):
            raise ValueError(
                "val_csvs and val_roots must both be provided or both be None"
            )
        if val_csvs_list is not None and len(val_csvs_list) != len(val_roots_list):
            raise ValueError(
                "val_csvs and val_roots must contain the same number of items"
            )

        resolved_charset = charset_path
        if resolved_charset is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            resolved_charset = os.path.join(current_dir, "configs", "charset.txt")

        config_payload: Dict[str, Any] = {
            "train_csvs": train_csvs_list,
            "train_roots": train_roots_list,
            "charset_path": resolved_charset,
            "encoding": encoding,
            "img_h": img_h,
            "img_w": img_w,
            "max_len": max_len,
            "hidden_size": hidden_size,
            "num_encoder_layers": num_encoder_layers,
            "cnn_in_channels": cnn_in_channels,
            "cnn_out_channels": cnn_out_channels,
            "cnn_backbone": cnn_backbone,
            "ctc_weight": ctc_weight,
            "ctc_weight_decay_epochs": ctc_weight_decay_epochs,
            "ctc_weight_min": ctc_weight_min,
            "max_grad_norm": max_grad_norm,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "val_interval": val_interval,
            "val_size": val_size,
            "num_workers": num_workers,
            "seed": seed,
        }

        if exp_dir is not None:
            config_payload["exp_dir"] = exp_dir
        if val_csvs_list is not None:
            config_payload["val_csvs"] = val_csvs_list
            config_payload["val_roots"] = val_roots_list
        if train_proportions is not None:
            config_payload["train_proportions"] = list(train_proportions)
        if resume_from is not None:
            config_payload["resume_from"] = resume_from
        if save_interval is not None:
            config_payload["save_interval"] = save_interval
        # Pretrained weights option:
        # - None/False/"none": skip
        # - "default"/True: use release weights
        # - str: path/URL to .pth/.pt/.ckpt
        config_payload["pretrain_weights"] = pretrain_weights

        if extra_config:
            config_payload.update(extra_config)

        # Freeze policies for model submodules
        config_payload["freeze_cnn"] = freeze_cnn
        config_payload["freeze_enc_rnn"] = freeze_enc_rnn
        config_payload["freeze_attention"] = freeze_attention

        config = Config(config_payload)
        return run_training(config, device=device)

    @staticmethod
    def export(
        weights_path: Union[str, Path],
        config_path: Union[str, Path],
        charset_path: Union[str, Path],
        output_path: Union[str, Path],
        opset_version: int = 14,
        simplify: bool = True,
    ) -> None:
        """
        Экспорт модели TRBA PyTorch в формат ONNX.

        Метод конвертирует обученную модель TRBA из PyTorch в формат ONNX,
        который может использоваться для более быстрого инференса с ONNX Runtime.
        Экспортированную модель можно загрузить через ``TRBA(weights="model.onnx")``.

        Параметры
        ----------
        weights_path : str or Path
            Путь к файлу весов модели PyTorch (.pth).
        config_path : str or Path
            Путь к конфигурационному JSON-файлу модели. Используется для определения
            архитектуры модели (img_h, img_w, max_len, hidden_size и др.).
        charset_path : str or Path
            Путь к файлу набора символов (charset.txt). Используется для определения
            num_classes модели.
        output_path : str or Path
            Путь, по которому будет сохранена ONNX-модель (.onnx).
        opset_version : int, optional
            Версия ONNX opset для экспорта. По умолчанию 14.
        simplify : bool, optional
            Если ``True``, применяет упрощение графа ONNX с помощью onnx-simplifier
            для оптимизации модели. Требует пакет ``onnx-simplifier``.
            По умолчанию ``True``.

        Возвращает
        -------
        None
            ONNX-модель сохраняется по пути ``output_path``.

        Raises
        ------
        ImportError
            Если необходимые пакеты (torch, onnx) не установлены.
        FileNotFoundError
            Если ``weights_path`` или ``config_path`` не существуют.

        Notes
        -----
        Экспортированная ONNX-модель имеет один выход:

        - ``logits``: предсказания символов с формой ``(batch, max_length+1, num_classes)``

        Модель использует жадное декодирование (argmax) и поддерживает динамический размер батча.
        Длина последовательности фиксирована равной ``max_length + 1`` из конфига (аналогично
        режиму инференса PyTorch для совместимости).

        Экспортируемая архитектура:
        - CNN-бэкбон (SE-ResNet-31 или SE-ResNet-31-Lite)
        - Двунаправленный LSTM-энкодер
        - Attention-декодер (жадное декодирование)

        Примечание: экспортируется только attention-декодер. CTC-голова используется
        только при обучении и не включается в ONNX-модель.

        Examples
        --------
        Экспорт модели TRBA в ONNX:

        >>> from manuscript.recognizers import TRBA
        >>> TRBA.export(
        ...     weights_path="experiments/best_model/best_acc_weights.pth",
        ...     config_path="experiments/best_model/config.json",
        ...     charset_path="configs/charset.txt",
        ...     output_path="trba_model.onnx"
        ... )
        Loading TRBA model...
        === TRBA ONNX Export ===
        Max decoding length: 40
        Input size: 64x256
        [OK] ONNX model saved to: trba_model.onnx

        Экспорт с пользовательским opset:

        >>> TRBA.export(
        ...     weights_path="model.pth",
        ...     config_path="config.json",
        ...     charset_path="charset.txt",
        ...     output_path="model.onnx",
        ...     opset_version=16,
        ...     simplify=False
        ... )

        Использование экспортированной модели для инференса:

        >>> from manuscript.detectors import EAST
        >>> recognizer = TRBA(weights="trba_model.onnx")
        >>> detector = EAST()
        >>> det = detector.predict("page.jpg")
        >>> result = recognizer.predict(det["page"], image="page.jpg")

        See Also
        --------
        TRBA.__init__ : Инициализация распознавателя TRBA с ONNX-моделью.
        """
        import torch
        from .model.model import TRBAModel, TRBAONNXWrapper

        weights_path = Path(weights_path)
        config_path = Path(config_path)
        charset_path = Path(charset_path)
        output_path = Path(output_path)

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not charset_path.exists():
            raise FileNotFoundError(f"Charset file not found: {charset_path}")

        # Load config
        print(f"Loading config from {config_path}...")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Extract model parameters
        max_length = config.get("max_len", 40)
        img_h = config.get("img_h", 64)
        img_w = config.get("img_w", 256)
        hidden_size = config.get("hidden_size", 256)
        num_encoder_layers = config.get("num_encoder_layers", 2)
        cnn_in_channels = config.get("cnn_in_channels", 3)
        cnn_out_channels = config.get("cnn_out_channels", 512)
        cnn_backbone = config.get("cnn_backbone", "seresnet31")

        # Load charset to determine num_classes
        print(f"Loading charset from {charset_path}...")
        itos, stoi = load_charset(str(charset_path))
        num_classes = len(
            itos
        )  # itos already includes special tokens (PAD, SOS, EOS, BLANK, ...)
        print(f"Charset loaded: {len(itos)} total classes (including special tokens)")
        print(f"  First 4 tokens (special): {itos[:4]}")
        print(f"  Regular characters: {len(itos) - 4}")

        # Load weights
        print(f"\nLoading checkpoint from {weights_path}...")
        checkpoint = torch.load(str(weights_path), map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        print(f"\n=== TRBA ONNX Export ===")
        print(f"Max decoding length: {max_length}")
        print(f"Input size: {img_h}x{img_w}")
        print(f"Architecture: {cnn_backbone}")
        print(f"Hidden size: {hidden_size}")
        print(f"Num classes: {num_classes}")

        # Create PyTorch model with correct num_classes and token IDs
        print(f"\nCreating model architecture...")
        model = TRBAModel(
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_encoder_layers=num_encoder_layers,
            img_h=img_h,
            img_w=img_w,
            cnn_in_channels=cnn_in_channels,
            cnn_out_channels=cnn_out_channels,
            cnn_backbone=cnn_backbone,
            sos_id=stoi["<SOS>"],
            eos_id=stoi["<EOS>"],
            pad_id=stoi["<PAD>"],
            blank_id=stoi.get("<BLANK>", None),
            use_ctc_head=False,
        )

        print(f"   Token IDs:")
        print(f"      SOS:   {stoi['<SOS>']}")
        print(f"      EOS:   {stoi['<EOS>']}")
        print(f"      PAD:   {stoi['<PAD>']}")
        print(f"      BLANK: {stoi.get('<BLANK>', None)}")
        print(f"      SPACE: {stoi.get(' ', 'NOT FOUND')}")

        # Load weights
        print(f"Loading weights into model...")
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        print("[OK] Model loaded")

        # Create ONNX wrapper
        print(f"\nCreating ONNX wrapper...")
        print(f"   max_length from config: {max_length}")
        print(
            f"   ONNX will use: {max_length + 1} steps (max_length + 1 for compatibility)"
        )
        onnx_model = TRBAONNXWrapper(model, max_length=max_length + 1)
        onnx_model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, img_h, img_w)

        print(f"Input shape: {dummy_input.shape}")

        # Test model before export
        print(f"\nTesting model before export...")
        with torch.no_grad():
            output = onnx_model(dummy_input)

        print(f"Output shape: {output.shape}")
        print(f"Expected: [1, {max_length + 1}, {num_classes}] (max_length + 1 steps)")

        torch_output = output.numpy()

        def _validate_exported_onnx(path: Path, label: str) -> None:
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            input_name = "input"
            get_inputs = getattr(session, "get_inputs", None)
            supports_input_introspection = callable(get_inputs)
            if supports_input_introspection:
                input_name = get_inputs()[0].name
            ort_outputs = session.run(None, {input_name: dummy_input.numpy()})
            onnx_output = ort_outputs[0]

            print(f"[OK] {label} inference works!")
            print(f"  Output shape: {onnx_output.shape}")

            max_diff = abs(torch_output - onnx_output).max()
            print(f"  Max difference vs PyTorch: {max_diff:.6f}")
            if max_diff < 1e-4:
                print("  [OK] Outputs match!")
            else:
                print("  [WARNING] Outputs differ slightly")

            if supports_input_introspection:
                batch_probe = np.random.randn(2, 3, img_h, img_w).astype(np.float32)
                batch_probe_output = session.run(None, {input_name: batch_probe})[0]
                if batch_probe_output.shape[0] != 2:
                    raise RuntimeError(
                        "dynamic batch validation failed: "
                        f"expected batch dimension 2, got {batch_probe_output.shape[0]}"
                    )
                print("  [OK] Dynamic batch inference works for batch_size=2")
            else:
                print("  [WARNING] Dynamic batch validation skipped for test session")

        # Export to ONNX
        print(f"\nExporting to ONNX (opset {opset_version})...")
        torch.onnx.export(
            onnx_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            verbose=False,
            dynamo=False,
        )

        print(f"[OK] ONNX model saved to: {output_path}")

        # Verify ONNX model
        import onnx

        print("\nVerifying ONNX model...")
        onnx_model_proto = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model_proto)
        print("[OK] ONNX model is valid")

        # Test ONNX inference
        print(f"\nTesting ONNX inference...")
        _validate_exported_onnx(output_path, "Exported ONNX model")

        # Simplify if requested
        if simplify:
            try:
                import onnxsim

                print("\nSimplifying ONNX model...")
                model_simplified, check = onnxsim.simplify(onnx_model_proto)
                if check:
                    onnx.save(model_simplified, str(output_path))
                    print("[OK] ONNX model simplified")
                    try:
                        print("\nValidating simplified ONNX model...")
                        _validate_exported_onnx(output_path, "Simplified ONNX model")
                    except Exception as exc:
                        onnx.save(onnx_model_proto, str(output_path))
                        print(
                            "[WARNING] Simplified ONNX model failed validation, "
                            f"restored original export: {exc}"
                        )
                else:
                    print("[WARNING] Simplification failed, using original model")
            except ImportError:
                print(
                    "[WARNING] onnx-simplifier not installed, skipping simplification"
                )
                print("  Install with: pip install onnx-simplifier")

        # Print summary
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n[OK] Export complete! Model size: {file_size_mb:.1f} MB")
        print(f"\nInput shape: [batch_size, 3, {img_h}, {img_w}]")
        print(f"Output shape: [batch_size, {max_length + 1}, {num_classes}]")
        print(f"Decoding: Greedy (argmax over last dimension)")
