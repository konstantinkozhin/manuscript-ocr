import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
from manuscript.utils.numerics import softmax
from PIL import Image

from manuscript.api.base import managed_session_initialization
from manuscript.api.recognizer import BaseRecognizer
from manuscript.utils import read_image

from .._common.region_types import PreparedRegion, RecognitionPrediction


class PPOCRRec(BaseRecognizer):
    """Recognize Page text regions with configurable CTC ONNX models.

    With ``region_preparer="bbox"``, ``region_preparer_options`` accepts
    ``padding_x``, ``padding_y``, ``inset_x`` and ``inset_y`` as fractions of
    the original bbox size per side. For example, ``{"inset_y": 0.1}``
    removes 10% from the top and bottom before orientation compensation.
    Insets must be below 0.5. Padding and inset on the same axis, or relative
    margins and nonzero pixel ``pad``, cannot be combined. Page geometry is
    preserved; other built-in crop presets reject relative margins.

    ``sort_by_width=True`` batches crops of similar widths and restores the
    original result order. It is opt-in because changing right padding can
    affect CTC predictions. ``max_batch_pixels`` optionally limits padded input
    pixels (batch * height * width, excluding channels), without resizing crops.
    A single input, including required fixed ONNX batch padding, must fit the
    budget; otherwise prediction raises ValueError. This is not a limit on
    total runtime memory. ``batch_size`` remains the maximum dynamic batch size.

    """

    default_weights_name = None
    registry_model_class = "PPOCRRec"
    pretrained_registry: Dict[str, str] = {}
    config_registry: Dict[str, str] = {}
    charset_registry: Dict[str, str] = {}

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
        recognition_levels: Optional[Sequence[str]] = None,
        use_space_char: Optional[bool] = None,
        rec_image_shape: Optional[Sequence[int]] = None,
        sort_by_width: bool = False,
        max_batch_pixels: Optional[int] = None,
        **kwargs: Any,
    ):
        if not isinstance(sort_by_width, bool):
            raise ValueError("sort_by_width must be a boolean")
        self.sort_by_width = sort_by_width
        self.max_batch_pixels = (None if max_batch_pixels is None else
                                 self._positive_integer(max_batch_pixels, "max_batch_pixels"))
        self._weights_preset = (
            str(weights) if weights is not None and str(weights) in self.pretrained_registry else None
        )

        super().__init__(
            weights=weights,
            device=device,
            force_download=force_download,
            rotate_threshold=rotate_threshold,
            region_preparer=region_preparer,
            region_preparer_options=region_preparer_options,
            min_text_size=min_text_size,
            batch_size=batch_size,
            recognition_levels=recognition_levels,
            **kwargs,
        )

        if not Path(self.weights).exists():
            raise FileNotFoundError(f"Model file not found: {self.weights}")
        if Path(self.weights).suffix.lower() != ".onnx":
            raise ValueError(f"Expected .onnx file, got: {self.weights}")

        self.config_path = self._resolve_config(config)
        if not self.config_path:
            raise ValueError("PPOCRRec requires a config artifact or explicit config")
        from .config import normalize_config
        config_data = normalize_config(self._load_config_data(self.config_path))
        self._configure_recognition_levels(config_data)

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
        self._input_width_override: Optional[int] = None

        if config_data.get("schema_version", 1) != 1:
            raise ValueError("Unsupported PPOCRRec config schema_version")
        if config_data.get("task", "text_recognition") != "text_recognition" or config_data.get("algorithm", "CTC") != "CTC":
            raise ValueError("PPOCRRec requires a CTC text-recognition model")
        self.preprocess = dict(config_data["preprocess"])
        self.postprocess = dict(config_data["postprocess"])
        self.input_name = config_data.get("input_name")
        self.output_name = config_data.get("output_name")
        self.length_input_name = config_data.get("length_input_name")
        self.length_output_name = config_data.get("length_output_name")
        self.length_input_units = config_data.get("length_input_units")
        self.length_input_stride = config_data.get("length_input_stride", 1)
        if self.length_input_name:
            if self.length_input_units not in ("pixels", "timesteps"):
                raise ValueError("length_input_name requires length_input_units: pixels or timesteps")
            self._positive_integer(self.length_input_stride, "length_input_stride")
            if self.length_input_units == "pixels" and self.length_input_stride != 1:
                raise ValueError("Pixel lengths require length_input_stride=1")
        if not isinstance(self.preprocess["dynamic_width"], bool) or not isinstance(self.postprocess["output_is_logits"], bool):
            raise ValueError("dynamic_width and output_is_logits must be boolean")
        self._input_dtype = np.float32
        if self.img_c != 3 or self.img_h <= 0 or self.img_w <= 0:
            raise ValueError("rec_image_shape must contain positive dimensions with 3 channels")
        if self.preprocess["color_order"] not in ("RGB", "BGR"):
            raise ValueError("color_order must be RGB or BGR")
        if self.preprocess["normalization"] not in ("paddle", "invert"):
            raise ValueError("normalization must be paddle or invert")
        if self.preprocess["interpolation"] not in ("paddle", "lanczos"):
            raise ValueError("interpolation must be paddle or lanczos")
        for key in ("padding", "min_width"):
            value = self.preprocess[key]
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if not all(isinstance(ch, str) and ch for ch in self.characters[1:]):
            raise ValueError("CTC dictionary contains an empty or invalid token")


    def _resolve_charset(self, charset: Optional[str]) -> Optional[str]:
        if charset is not None:
            return self._resolve_extra_artifact(
                charset,
                default_name=None,
                registry=self.charset_registry,
                description="charset",
            )

        if getattr(self, '_resolved_model_artifacts', None):
            value = self._resolved_model_artifacts.get('charset')
            return str(value) if value else None

        return None


    @staticmethod
    def _resolve_rec_image_shape(rec_image_shape, config_data):
        shape = config_data["rec_image_shape"] if rec_image_shape is None else rec_image_shape
        if len(shape) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape):
            raise ValueError("rec_image_shape must contain exactly 3 positive integers")
        return list(shape)

    @staticmethod
    def _resolve_use_space_char(use_space_char, config_data):
        if use_space_char is None and "use_space_char" not in config_data.get("Global", {}):
            use_space_char = config_data.get("PostProcess", {}).get("name") == "CTCLabelDecode"
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
            chars = [str(ch) for ch in config_chars]
            if use_space_char and " " not in chars:
                chars.append(" ")
            return ["blank"] + chars

        raise FileNotFoundError(
            "Could not resolve character dictionary. "
            "Provide charset explicitly or include character_dict in the explicit config."
        )


    @managed_session_initialization
    def _initialize_session(self) -> None:
        if self.onnx_session is not None:
            return

        session = self._create_onnx_session()

        inputs = {item.name: item for item in session.get_inputs()}
        self.input_name = self.input_name or next(iter(inputs))
        if self.input_name not in inputs:
            raise ValueError("Configured image input is absent from ONNX graph")
        item = inputs[self.input_name]
        if item.type not in ("tensor(float)", "tensor(float16)") or len(item.shape) != 4:
            raise ValueError("PPOCRRec expects a float32/float16 NCHW image input")
        self._input_dtype = np.float16 if item.type == "tensor(float16)" else np.float32
        for actual, expected in zip(item.shape[1:3], [self.img_c, self.img_h]):
            if isinstance(actual, int) and actual != expected:
                raise ValueError("ONNX image shape disagrees with recognition config")
        self._input_width_override = item.shape[3] if isinstance(item.shape[3], int) else None
        expected_inputs = {self.input_name}
        if self.length_input_name:
            expected_inputs.add(self.length_input_name)
            if self.length_input_name not in inputs or inputs[self.length_input_name].type != "tensor(int64)":
                raise ValueError("CTC length input must be int64")
        if set(inputs) != expected_inputs:
            raise ValueError("Unsupported extra ONNX inputs; configure length_input_name")
        outputs = {item.name: item for item in session.get_outputs()}
        self.output_name = self.output_name or next(iter(outputs))
        if self.output_name not in outputs or (self.length_output_name and self.length_output_name not in outputs):
            raise ValueError("Configured recognition output is absent from ONNX graph")
        shape = outputs[self.output_name].shape
        if len(shape) != 3 or (isinstance(shape[2], int) and shape[2] != len(self.characters)):
            raise ValueError("CTC output must be [batch, time, dictionary_size + 1]")
        self.onnx_session = session

    def _preprocess_image(self, image):
        return self._prepare_image(image)[0]

    def _prepare_image(self, image):
        img = read_image(image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Expected RGB image with shape [H, W, 3]")
        if self.preprocess["color_order"] == "BGR":
            img = img[:, :, ::-1]
        pad = int(self.preprocess["padding"])
        lanczos = self.preprocess["interpolation"] == "lanczos"
        width, target = self._image_width(img)
        if lanczos:
            resized = np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize(
                (width, self.img_h), Image.Resampling.LANCZOS))
        else:
            resized = cv2.resize(img, (width, self.img_h), interpolation=cv2.INTER_LINEAR)
        if self.preprocess["normalization"] == "invert":
            # Kraken inverts after adding white edge padding.
            pixels = np.full((self.img_h, width + 2 * pad, 3), 255, np.uint8)
            pixels[:, pad:pad + width] = resized
            tensor = (float(pixels.max()) - pixels.astype(np.float32)) / 255.0
        else:
            tensor = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
            tensor = np.pad(tensor, ((0, 0), (pad, pad), (0, 0)))
        result = np.zeros((3, self.img_h, target), self._input_dtype)
        result[:, :, :tensor.shape[1]] = tensor.transpose(2, 0, 1)
        return result[None], width + 2 * pad


    def _image_width(self, image):
        """Return resized content width and padded input width without allocating tensors."""
        pad = int(self.preprocess["padding"])
        lanczos = self.preprocess["interpolation"] == "lanczos"
        ratio_width = self.img_h * image.shape[1] / image.shape[0]
        width = max(1, int(ratio_width) if lanczos else math.ceil(ratio_width))
        fixed = self._input_width_override
        if fixed or not self.preprocess["dynamic_width"]:
            target = fixed or self.img_w
            if target <= 2 * pad:
                raise ValueError("ONNX width is too small for configured padding")
            width = min(width, target - 2 * pad)
        else:
            target = max(int(self.preprocess["min_width"]), width + 2 * pad)
        return width, target

    def _decode_recognition_logits(self, logits):
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 3 or logits.shape[2] != len(self.characters) or logits.shape[1] == 0 or not np.isfinite(logits).all():
            raise ValueError("Invalid CTC output or dictionary size mismatch")
        if self.postprocess["output_is_logits"]:
            logits = softmax(logits, axis=2)
        elif np.any(logits < -1e-5) or np.any(logits > 1.00001):
            raise ValueError("CTC probabilities outside [0, 1]; configure output_is_logits")
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
        return_raw: bool = False,
    ) -> List[RecognitionPrediction]:
        return self._run_inference_batches(
            regions=regions,
            batch_size=batch_size,
            return_raw=return_raw,
        )

    def _region_preparation_batch_size(self, batch_size):
        # These options plan batches across the whole page, preserving that contract.
        if self.sort_by_width or self.max_batch_pixels is not None:
            return None
        self._initialize_session()
        shape = next(item.shape for item in self.onnx_session.get_inputs() if item.name == self.input_name)
        return shape[0] if isinstance(shape[0], int) and shape[0] > 0 else batch_size

    def _run_inference_batches(self, regions, batch_size=None, return_raw=False):
        if not regions:
            return []
        self._initialize_session()
        shape = next(item.shape for item in self.onnx_session.get_inputs() if item.name == self.input_name)
        fixed_batch = shape[0] if isinstance(shape[0], int) else None
        batch_size = fixed_batch or max(1, int(batch_size or self.batch_size))
        widths = [self._image_width(region.image)[1] for region in regions]
        order = list(range(len(regions)))
        if self.sort_by_width:
            order.sort(key=widths.__getitem__)
        # Validate before inference: neither resize nor silently exceed the budget.
        if self.max_batch_pixels is not None:
            required = (fixed_batch or 1) * self.img_h * max(widths)
            if required > self.max_batch_pixels:
                raise ValueError(
                    f"A recognition input requires {required} batch pixels, exceeding "
                    f"max_batch_pixels={self.max_batch_pixels}; increase the budget"
                )
        results = [None] * len(regions)
        start = 0
        while start < len(order):
            end = start
            width = 0
            while end < min(start + batch_size, len(order)):
                candidate_width = max(width, widths[order[end]])
                count = fixed_batch or (end - start + 1)
                if (self.max_batch_pixels is not None and
                        count * self.img_h * candidate_width > self.max_batch_pixels):
                    break
                width = candidate_width
                end += 1
            indices = order[start:end]
            predictions = self._infer_batch(regions, indices, width, fixed_batch, return_raw)
            for index, prediction in zip(indices, predictions):
                results[index] = prediction
            start = end
        return results

    def _infer_batch(self, regions, indices, width, fixed_batch, return_raw):
        # Only this input buffer and one prepared crop coexist. Locals (including
        # ONNX outputs) leave scope before the next batch is allocated.
        count = len(indices)
        size = fixed_batch or count
        images = np.zeros((size, self.img_c, self.img_h, width), self._input_dtype)
        lengths = np.empty(size, np.int64)
        for row, index in enumerate(indices):
            tensor, length = self._prepare_image(regions[index].image)
            images[row, :, :, :tensor.shape[3]] = tensor[0]
            lengths[row] = length
            del tensor
        if size > count:
            images[count:] = images[count - 1]
            lengths[count:] = lengths[count - 1]
        feed = {self.input_name: images}
        if self.length_input_name:
            feed[self.length_input_name] = (lengths + self.length_input_stride - 1) // self.length_input_stride
        names = [self.output_name] + ([self.length_output_name] if self.length_output_name else [])
        outputs = self.onnx_session.run(names, feed)
        if len(outputs[0]) != size:
            raise ValueError("ONNX recognition batch dimension mismatch")
        results = []
        for index in range(count):
            sequence = outputs[0][index:index + 1]
            if self.length_output_name:
                length = int(outputs[1][index])
                if not 0 < length <= sequence.shape[1]:
                    raise ValueError("Invalid ONNX CTC output length")
                sequence = sequence[:, :length]
            results.extend(self._decode_predictions(sequence, return_raw))
        return results

    @staticmethod
    def create_config(source, *, recognition_levels=None, output_path=None):
        """Create an inference JSON config; new bundles recognize lines by default."""
        from .config import normalize_config
        import json

        if isinstance(source, dict):
            config = source
        else:
            path = Path(source)
            if path.suffix.lower() == ".json":
                config = json.loads(path.read_text(encoding="utf-8"))
            else:
                import yaml
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = normalize_config(config)
        selected = recognition_levels if recognition_levels is not None else config.get("recognition_levels", ["line"])
        config["recognition_levels"] = list(BaseRecognizer._validate_levels(selected))
        if output_path is not None:
            Path(output_path).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return config



__all__ = ["PPOCRRec"]
