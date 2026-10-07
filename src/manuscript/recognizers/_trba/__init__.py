import os
import json
from pathlib import Path
from typing import Callable, List, Union, Optional, Sequence, Dict, Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from manuscript.api.recognizer import BaseRecognizer
from manuscript.utils import read_image

from .._common.region_types import PreparedRegion, RecognitionPrediction
from .data.charset import load_charset

Config = None
run_training = None
_TRAINING_AVAILABLE = None
_TRAINING_IMPORT_ERROR = None


def _ensure_training_dependencies() -> None:
    global Config, run_training, _TRAINING_AVAILABLE, _TRAINING_IMPORT_ERROR

    if _TRAINING_AVAILABLE is True:
        return

    try:
        from .training.train import Config as config_cls
        from .training.train import run_training as run_training_fn
    except ImportError as exc:
        Config = None
        run_training = None
        _TRAINING_AVAILABLE = False
        _TRAINING_IMPORT_ERROR = exc
        raise ImportError(
            "Training dependencies not available. "
            "Install with: pip install manuscript-ocr[dev]"
        ) from exc

    Config = config_cls
    run_training = run_training_fn
    _TRAINING_AVAILABLE = True
    _TRAINING_IMPORT_ERROR = None

class TRBA(BaseRecognizer):
    """
    Initialize TRBA text recognition model with ONNX Runtime.

    Parameters
    ----------
    weights : str or Path, optional
        Path or identifier for ONNX model weights. Supports:

        - Local file path: ``"path/to/model.onnx"``
        - HTTP/HTTPS URL: ``"https://example.com/model.onnx"``
        - GitHub release: ``"github://owner/repo/tag/file.onnx"``
        - Google Drive: ``"gdrive:FILE_ID"``
        - Preset name: ``"trba_lite_g1"`` or ``"trba_base_g1"`` (from pretrained_registry)
        - ``None``: auto-downloads default preset (trba_lite_g1)

    config : str or Path, optional
        Path or identifier for model configuration JSON. Same URL schemes
        as ``weights``. If ``None``, attempts to infer from weights location
        or uses default config for preset models.
    charset : str or Path, optional
        Path or identifier for character set file. If ``None``, attempts to
        find charset near weights or falls back to package default.
    device : {"cuda", "coreml", "cpu"}, optional
        Compute device. If ``None``, automatically selects CPU.
        For GPU/CoreML acceleration:

        - CUDA (NVIDIA): ``pip install onnxruntime-gpu``
        - CoreML (Apple Silicon M1/M2/M3): ``pip install onnxruntime-silicon``

        Default is ``None`` (CPU).
    rotate_threshold : float or None, optional
        Aspect-ratio threshold for rotating vertical text-span crops before
        recognition. If ``height > width * rotate_threshold``, crop is
        rotated 90 degrees clockwise. Set to ``0`` or ``None`` to disable.
        Default is ``1.5``.
    region_preparer : {"bbox", "polygon_mask", "quad_warp"} or callable, optional
        Strategy used to convert ``Page`` polygons into recognition crops.
        ``"bbox"`` extracts axis-aligned bounding boxes for arbitrary polygons.
        ``"polygon_mask"`` masks pixels outside the polygon inside a tight crop
        and also supports arbitrary polygons. ``"quad_warp"`` rectifies only
        4-point polygons with a perspective transform before recognition. A
        custom callable may also be provided and should return a list of
        prepared text regions. Default is ``"bbox"``.
    region_preparer_options : dict or None, optional
        Optional configuration for built-in region preparers. Defaults to
        ``None``. Typical options are ``pad`` for ``"bbox"`` and
        ``"polygon_mask"``, or ``output_size=(width, height)`` for
        ``"quad_warp"``. Non-quad polygons passed to ``"quad_warp"`` fall back
        to bbox crops by default.
    min_text_size : int, optional
        Minimum crop width/height in pixels to run recognition for a text span.
        Text spans below this threshold are skipped. Default is ``5``.
    batch_size : int, optional
        Default inference batch size used when ``predict(..., batch_size=...)``
        is not provided. Default is ``128``.
    **kwargs
        Additional configuration options (reserved for future use).

    Raises
    ------
    FileNotFoundError
        If specified files do not exist.
    ValueError
        If weights format is invalid.

    Notes
    -----
    The class provides three main public methods:

    - ``predict`` - run recognition over text spans in a ``Page`` object.
    - ``train`` - high-level training entrypoint to train a TRBA model
      on custom datasets.
    - ``export`` - static method to export PyTorch model to ONNX format.

    Model uses ONNX Runtime for fast inference on CPU and GPU.
    For GPU acceleration, install: ``pip install onnxruntime-gpu``

    Examples
    --------
    Create recognizer with default preset (auto-downloads):

    >>> from manuscript.recognizers import TRBA
    >>> recognizer = TRBA()

    Load from local ONNX file:

    >>> recognizer = TRBA(weights="path/to/model.onnx")

    Load from GitHub release:

    >>> recognizer = TRBA(
    ...     weights="github://owner/repo/v1.0/model.onnx",
    ...     config="github://owner/repo/v1.0/config.json"
    ... )

    Force CPU execution:

    >>> recognizer = TRBA(weights="model.onnx", device="cpu")
    """

    default_weights_name = "trba_lite_g1"
    registry_model_class = "TRBA"

    config_registry = {}
    charset_registry = {}

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
        super().__init__(
            weights=weights,
            device=device,
            force_download=force_download,
            rotate_threshold=rotate_threshold,
            region_preparer=region_preparer,
            region_preparer_options=region_preparer_options,
            min_text_size=min_text_size,
            batch_size=batch_size,
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
        for name in ("max_length", "img_h", "img_w"):
            self._positive_integer(getattr(self, name), name)
        if self.cnn_in_channels != 3:
            raise ValueError("TRBA preprocessing requires a three-channel RGB model")

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
        self._supports_multi_batch_inference: Optional[bool] = None
        self._single_batch_warning_emitted = False

    def _resolve_config(self, config):
        path = super()._resolve_config(config)
        if path is None:
            raise FileNotFoundError("TRBA requires a config artifact or explicit config")
        return path

    def _resolve_charset(self, charset):
        if charset is not None:
            return self._resolve_extra_artifact(charset, default_name=None, registry={}, description="charset")
        resolved = getattr(self, "_resolved_model_artifacts", {}).get("charset")
        if resolved:
            return str(resolved)
        candidate = Path(self.weights).with_suffix(".txt")
        if candidate.is_file():
            return str(candidate.absolute())
        raise FileNotFoundError("TRBA requires a charset artifact or explicit charset")

    def _initialize_session(self):
        """Initialize ONNX Runtime session (lazy loading)."""
        if self.onnx_session is not None:
            return

        session = self._create_onnx_session()
        inputs = session.get_inputs()
        if len(inputs) != 1 or inputs[0].type not in ("tensor(float)", "tensor(float16)"):
            raise ValueError("TRBA requires one float32/float16 image input")
        self._input_dtype = np.float16 if inputs[0].type == "tensor(float16)" else np.float32
        self.onnx_session = session

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
        return np.expand_dims(img_chw, axis=0).astype(getattr(self, "_input_dtype", np.float32), copy=False)


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

        return int(input_batch_dim) if isinstance(input_batch_dim, int) and input_batch_dim > 0 else requested_batch_size

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
        if (logits.ndim != 3 or logits.shape[2] != len(self.itos)
                or logits.shape[1] == 0 or not np.isfinite(logits).all()):
            raise ValueError("Invalid TRBA logits or charset mismatch")
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
        return_raw: bool = False,
    ) -> List[RecognitionPrediction]:
        return self._run_inference_batches(
            regions=regions,
            batch_size=batch_size,
            return_raw=return_raw,
        )

    def _run_inference_batches(
        self,
        regions: Sequence[PreparedRegion],
        batch_size: Optional[int] = None,
        return_raw: bool = False,
    ) -> List[RecognitionPrediction]:
        """
        Run ONNX inference on prepared text regions.
        """
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
                        single_logits = self.onnx_session.run([output_name], {input_name: single_input})[0]
                        results.extend(self._decode_predictions(single_logits, return_raw))
                    continue
                raise

            if len(batch_regions) > 1 and self._supports_multi_batch_inference is None:
                self._supports_multi_batch_inference = True

            if len(logits) != original_batch_size:
                logits = logits[:original_batch_size]

            results.extend(self._decode_predictions(logits, return_raw))

        return results


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
        Train TRBA text recognition model on custom datasets.

        Parameters
        ----------
        train_csvs : str, Path or sequence of paths
            Path(s) to training CSV files. Each CSV should have columns:
            ``image_path`` (relative to ``train_roots``) and ``text`` (ground
            truth transcription).
        train_roots : str, Path or sequence of paths
            Root directory/directories containing training images. Must have
            same length as ``train_csvs``.
        val_csvs : str, Path, sequence of paths, or None, optional
            Path(s) to validation CSV files with same format as ``train_csvs``.
            If ``None``, no validation is performed. Default is ``None``.
        val_roots : str, Path, sequence of paths, or None, optional
            Root directory/directories for validation images. Must match length
            of ``val_csvs`` if provided. Default is ``None``.
        exp_dir : str or Path, optional
            Experiment directory where checkpoints and logs will be saved.
            If ``None``, auto-generated based on timestamp. Default is ``None``.
        charset_path : str or Path, optional
            Path to character set file. If ``None``, uses default charset from
            package. Default is ``None``.
        encoding : str, optional
            Text encoding for reading CSV files. Default is ``"utf-8"``.
        img_h : int, optional
            Target height for input images (pixels). Default is 64.
        img_w : int, optional
            Target width for input images (pixels). Default is 256.
        max_len : int, optional
            Maximum sequence length for text recognition. Default is 25.
        hidden_size : int, optional
            Hidden dimension size for RNN encoder/decoder. Default is 256.
        num_encoder_layers : int, optional
            Number of Bidirectional LSTM layers in the encoder. Default is 2.
        cnn_in_channels : int, optional
            Number of input channels for CNN backbone (3 for RGB, 1 for grayscale). Default is 3.
        cnn_out_channels : int, optional
            Number of output channels from CNN backbone. Default is 512.
        cnn_backbone : {"seresnet31", "seresnet31-lite"}, optional
            CNN backbone variant. ``"seresnet31"`` keeps the standard SE-ResNet-31,
            while ``"seresnet31-lite"`` enables a depthwise-lite version. Default is ``"seresnet31"``.
        ctc_weight : float, optional
            Initial weight for CTC loss during training (CTC always used for stability):
            ``loss = attn_loss * (1 - ctc_weight) + ctc_loss * ctc_weight``.
            CTC weight decays over epochs. Default is 0.3.
        ctc_weight_decay_epochs : int, optional
            Number of epochs for CTC weight to decay to minimum. Default is 50.
        ctc_weight_min : float, optional
            Minimum value for CTC weight after decay. Default is 0.0.
        max_grad_norm : float, optional
            Maximum gradient norm for clipping (prevents gradient explosion/NaN).
            Default is 5.0.
        batch_size : int, optional
            Training batch size. Default is 32.
        epochs : int, optional
            Number of training epochs. Default is 20.
        lr : float, optional
            Learning rate. Default is 1e-3.
        optimizer : {"Adam", "SGD", "AdamW"}, optional
            Optimizer type. Default is ``"AdamW"``.
        scheduler : {"ReduceLROnPlateau", "CosineAnnealingLR", "OneCycleLR", "None"}, optional
            Learning rate scheduler type:

            - ``"OneCycleLR"`` - one-cycle policy with cosine annealing (default, recommended)
            - ``"ReduceLROnPlateau"`` - reduce LR on validation loss plateau
            - ``"CosineAnnealingLR"`` - cosine annealing over epochs
            - ``"None"`` or ``None`` - constant learning rate

            Default is ``"OneCycleLR"``.
        weight_decay : float, optional
            L2 weight decay coefficient. Default is 0.0.
        momentum : float, optional
            Momentum for SGD optimizer. Default is 0.9.
        val_interval : int, optional
            Perform validation every N epochs. Default is 1.
        val_size : int, optional
            Maximum number of validation samples to use. Default is 3000.
        train_proportions : sequence of float, optional
            Sampling proportions for multiple training datasets. Must sum to 1.0
            and match length of ``train_csvs``. If ``None``, datasets are
            concatenated equally. Default is ``None``.
        num_workers : int, optional
            Number of data loading workers. Default is 0.
        seed : int, optional
            Random seed for reproducibility. Default is 42.
        resume_from : str or Path, optional
            Path to checkpoint file to resume training from. Default is ``None``.
        save_interval : int, optional
            Save checkpoint every N epochs. If ``None``, only saves best model.
            Default is ``None``.
        device : {"cuda", "cpu"}, optional
            Training device. Default is ``"cuda"``.
        freeze_cnn : {"none", "all", "first", "last"}, optional
            CNN freezing policy. Default is ``"none"``.
        freeze_enc_rnn : {"none", "all", "first", "last"}, optional
            Encoder RNN freezing policy. Default is ``"none"``.
        freeze_attention : {"none", "all"}, optional
            Attention module freezing policy. Default is ``"none"``.
        pretrain_weights : str, Path, bool, or None, optional
            Pretrained weights to initialize from:

            - ``"default"`` or ``True`` - use the registry checkpoint for ``trba_lite_g1``
            - ``None`` or ``False`` - train from scratch
            - str/Path - path or URL to custom weights file

            Default is ``"default"``.
        **extra_config : dict, optional
            Additional configuration parameters passed to training config.

        Returns
        -------
        str
            Path to the best model checkpoint saved during training.

        Examples
        --------
        Train on single dataset with validation:

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

        Train on multiple datasets with custom proportions:

        >>> train_csvs = ["data/dataset1/train.csv", "data/dataset2/train.csv"]
        >>> train_roots = ["data/dataset1/images", "data/dataset2/images"]
        >>> train_proportions = [0.7, 0.3]  # 70% from dataset1, 30% from dataset2
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

        Resume training from checkpoint:

        >>> best_model = TRBA.train(
        ...     train_csvs="data/train.csv",
        ...     train_roots="data/train_images",
        ...     resume_from="experiments/trba_exp1/checkpoints/last.pth",
        ...     epochs=100,
        ... )

        Fine-tune from pretrained weights with frozen CNN:

        >>> best_model = TRBA.train(
        ...     train_csvs="data/finetune.csv",
        ...     train_roots="data/finetune_images",
        ...     pretrain_weights="default",
        ...     freeze_cnn="all",
        ...     epochs=20,
        ...     lr=1e-4,
        ... )

        Train with CTC for stability (always enabled):

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
        _ensure_training_dependencies()

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
        Export TRBA PyTorch model to ONNX format.

        This method converts a trained TRBA model from PyTorch to ONNX format,
        which can be used for faster inference with ONNX Runtime. The exported
        model can be loaded using ``TRBA(weights="model.onnx")``.

        Parameters
        ----------
        weights_path : str or Path
            Path to the PyTorch model weights file (.pth).
        config_path : str or Path
            Path to the model configuration JSON file. Used to determine
            model architecture (img_h, img_w, max_len, hidden_size, etc.).
        charset_path : str or Path
            Path to the charset file (charset.txt). Used to determine
            num_classes for the model.
        output_path : str or Path
            Path where the ONNX model will be saved (.onnx).
        opset_version : int, optional
            ONNX opset version to use for export. Default is 14.
        simplify : bool, optional
            If True, applies ONNX graph simplification using onnx-simplifier
            to optimize the model. Requires ``onnx-simplifier`` package.
            Default is True.

        Returns
        -------
        None
            The ONNX model is saved to ``output_path``.

        Raises
        ------
        ImportError
            If required packages (torch, onnx) are not installed.
        FileNotFoundError
            If ``weights_path`` or ``config_path`` do not exist.

        Notes
        -----
        The exported ONNX model has one output:

        - ``logits``: Character predictions with shape ``(batch, max_length+1, num_classes)``

        The model uses greedy decoding (argmax) and supports dynamic batch size.
        The sequence length is fixed to ``max_length + 1`` from the config (same as PyTorch
        inference mode for compatibility).

        Architecture exported:
        - CNN backbone (SE-ResNet-31 or SE-ResNet-31-Lite)
        - Bidirectional LSTM encoder
        - Attention decoder (greedy decoding)

        Note: Only the attention decoder is exported. CTC head is used only
        during training and is not included in the ONNX model.

        Examples
        --------
        Export TRBA model to ONNX:

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

        Export with custom opset:

        >>> TRBA.export(
        ...     weights_path="model.pth",
        ...     config_path="config.json",
        ...     charset_path="charset.txt",
        ...     output_path="model.onnx",
        ...     opset_version=16,
        ...     simplify=False
        ... )

        Use the exported model for inference:

        >>> from manuscript.detectors import EAST
        >>> recognizer = TRBA(weights="trba_model.onnx")
        >>> detector = EAST()
        >>>         >>> page = detector.predict(Page(), image="page.jpg")
        >>> result = recognizer.predict(page, image="page.jpg")

        See Also
        --------
        TRBA.__init__ : Initialize TRBA recognizer with ONNX model.
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
        transformation = config.get("transformation", "none")
        tps_num_fiducial = config.get("tps_num_fiducial", 20)
        from .model.decoder_parseq import decoder_config
        if transformation.lower() == "tps":
            # ONNX GridSample was introduced in opset 16.
            opset_version = max(opset_version, 16)

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
        elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
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
            transformation=transformation,
            tps_num_fiducial=tps_num_fiducial,
            **decoder_config(config),
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
        # A wrong transformation config must not silently drop the rectifier or
        # export an untrained identity TPS. Only the training CTC head is omitted.
        inference_state = {k: v for k, v in state_dict.items() if not k.startswith("ctc_head.")}
        model.load_state_dict(inference_state, strict=True)
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
