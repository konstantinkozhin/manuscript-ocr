import os
import json
from pathlib import Path
from typing import List, Union, Tuple, Optional, Sequence, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import gdown
except ImportError:  # pragma: no cover - optional dependency for default weights
    gdown = None  # type: ignore[assignment]

from .model.model import TRBAModel
from .data.transforms import load_charset, get_val_transform, decode_tokens
from .training.utils import load_checkpoint
from .training.train import Config, run_training


class TRBA:
    _DEFAULT_PRESET_NAME = "exp_1_baseline"
    _DEFAULT_RELEASE_WEIGHTS_URL = (
        "https://github.com/konstantinkozhin/manuscript-ocr/"
        "releases/download/v0.1.0/trba_exp_1_64.pth"
    )
    _DEFAULT_RELEASE_CONFIG_URL = (
        "https://github.com/konstantinkozhin/manuscript-ocr/"
        "releases/download/v0.1.0/trba_exp_1_64.json"
    )
    _DEFAULT_STORAGE_ROOT = Path.home() / ".manuscript" / "trba"
    _DEFAULT_WEIGHTS_FILENAME = "weights.pth"
    _DEFAULT_CONFIG_FILENAME = "config.json"

    def __init__(
        self,
        model_path: Optional[str] = None,
        charset_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "auto",
        **kwargs: Any,
    ):
        """
        Initialize TRBA text recognition model.

        Parameters
        ----------
        model_path : str or Path, optional
            Path to the trained model checkpoint (.pth file). If ``None``,
            default pretrained weights will be automatically downloaded to
            ``~/.manuscript/trba/exp_1_baseline/weights.pth``.
        charset_path : str or Path, optional
            Path to character set file. If ``None``, uses the default charset
            located at ``configs/charset.txt`` within the package.
        config_path : str or Path, optional
            Path to model configuration JSON file. If ``None``, attempts to
            infer config path from model checkpoint location or downloads
            default config when using default weights.
        device : {"auto", "cuda", "cpu"}, optional
            Compute device. If ``"auto"``, automatically selects CUDA if
            available, otherwise CPU. Default is ``"auto"``.
        **kwargs : dict, optional
            Additional keyword arguments. Currently accepts ``weights_path``
            as an alias for ``model_path`` for backward compatibility.

        Raises
        ------
        FileNotFoundError
            If specified model, config, or charset files do not exist.
        ValueError
            If both ``model_path`` and ``weights_path`` are provided with
            different values.
        RuntimeError
            If gdown is not installed and default weights need to be
            downloaded.

        Notes
        -----
        The class provides two main public methods:

        - ``predict`` — run text recognition inference on cropped word images.
        - ``train`` — high-level training entrypoint to train a TRBA model
          on custom datasets.

        The model architecture is based on TRBA (Transformation + ResNet +
        BiLSTM + Attention) for scene text recognition, adapted for historical
        manuscript recognition.

        Examples
        --------
        Create recognizer with default pretrained weights:

        >>> from manuscript.recognizers import TRBA
        >>> recognizer = TRBA()

        Use custom trained model:

        >>> recognizer = TRBA(
        ...     model_path="path/to/model.pth",
        ...     config_path="path/to/config.json"
        ... )

        Force CPU execution:

        >>> recognizer = TRBA(device="cpu")
        """
        weights_path = kwargs.pop("weights_path", None)
        if kwargs:
            unexpected = ", ".join(kwargs.keys())
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        if weights_path is not None and model_path is not None:
            if os.path.abspath(os.fspath(weights_path)) != os.path.abspath(
                os.fspath(model_path)
            ):
                raise ValueError(
                    "Provide either model_path or weights_path, but not both with "
                    "different values."
                )

        resolved_model_path, resolved_config_path = (
            self._resolve_model_and_config_paths(
                weights_path if weights_path is not None else model_path,
                config_path,
            )
        )

        self.model_path = resolved_model_path

        if charset_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            charset_path = os.path.join(current_dir, "configs", "charset.txt")

        self.charset_path = os.fspath(charset_path)
        self.config_path = resolved_config_path

        if not os.path.exists(self.charset_path):
            raise FileNotFoundError(f"Charset file not found: {self.charset_path}")

        if self.config_path is not None:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}

        self.max_length = config.get("max_len", 25)
        self.hidden_size = config.get("hidden_size", 256)
        self.img_h = config.get("img_h", 64)
        self.img_w = config.get("img_w", 256)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.itos, self.stoi = load_charset(charset_path)
        self.pad_id = self.stoi["<PAD>"]
        self.sos_id = self.stoi["<SOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.blank_id = self.stoi.get("<BLANK>", None)

        self.transform = get_val_transform(self.img_h, self.img_w)

        self.model = self._load_model()

    def _resolve_model_and_config_paths(
        self,
        model_path: Optional[str],
        config_path: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        if model_path is None:
            return self._ensure_default_artifacts(config_path)

        resolved_model_path = os.fspath(model_path)
        if not os.path.exists(resolved_model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {resolved_model_path}"
            )

        if config_path is not None:
            resolved_config_path = os.fspath(config_path)
            if not os.path.exists(resolved_config_path):
                raise FileNotFoundError(
                    f"Config file not found: {resolved_config_path}"
                )
        else:
            resolved_config_path = self._infer_config_path_from_weights(
                resolved_model_path
            )

        return resolved_model_path, resolved_config_path

    def _infer_config_path_from_weights(self, weights_path: str) -> Optional[str]:
        weights_file = Path(weights_path)
        candidates = [
            weights_file.with_suffix(".json"),
            weights_file.parent / "config.json",
        ]

        for candidate in candidates:
            if candidate.exists():
                return os.fspath(candidate)
        return None

    def _ensure_default_artifacts(
        self,
        config_path: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        storage_root = self._DEFAULT_STORAGE_ROOT.expanduser()
        target_dir = storage_root / self._DEFAULT_PRESET_NAME
        target_dir.mkdir(parents=True, exist_ok=True)

        weights_dest = target_dir / self._DEFAULT_WEIGHTS_FILENAME
        if not weights_dest.exists():
            self._download_file(self._DEFAULT_RELEASE_WEIGHTS_URL, weights_dest)

        if config_path is not None:
            resolved_config_path = os.fspath(config_path)
            if not os.path.exists(resolved_config_path):
                raise FileNotFoundError(
                    f"Config file not found: {resolved_config_path}"
                )
        else:
            config_dest = target_dir / self._DEFAULT_CONFIG_FILENAME
            if not config_dest.exists():
                self._download_file(self._DEFAULT_RELEASE_CONFIG_URL, config_dest)
            resolved_config_path = os.fspath(config_dest)

        return os.fspath(weights_dest), resolved_config_path

    def _download_file(self, url: str, destination: Path) -> None:
        if gdown is None:
            raise RuntimeError(
                "gdown is required to download default TRBA weights. "
                "Install gdown or pass weights_path/model_path explicitly."
            )

        print(f"Downloading TRBA artifact from {url} -> {destination}")
        gdown.download(url, os.fspath(destination), quiet=False)
        if not destination.exists():
            raise RuntimeError(f"Failed to download artifact from {url}")

    def _load_model(self) -> TRBAModel:
        model = TRBAModel(
            num_classes=len(self.itos),
            hidden_size=self.hidden_size,
            sos_id=self.sos_id,
            eos_id=self.eos_id,
            pad_id=self.pad_id,
            blank_id=self.blank_id,
        ).to(self.device)

        load_checkpoint(
            path=self.model_path,
            model=model,
            map_location=self.device,
        )

        model.eval()
        return model

    def _preprocess_image(
        self, image: Union[np.ndarray, str, Image.Image]
    ) -> torch.Tensor:
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Cannot read image: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            img = image.copy()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        transformed = self.transform(image=img)
        tensor = transformed["image"].unsqueeze(0)

        return tensor.to(self.device)

    def predict(
        self,
        images: Union[
            np.ndarray, str, Image.Image, List[Union[np.ndarray, str, Image.Image]]
        ],
        batch_size: int = 32,
        mode: str = "beam",
        beam_size: int = 8,
        temperature: float = 1.7,
        alpha: float = 0.9,
    ) -> List[Tuple[str, float]]:
        """
        Run text recognition on one or more word images.

        Parameters
        ----------
        images : str, Path, numpy.ndarray, PIL.Image, or list thereof
            Single image or list of images to recognize. Each image can be:

            - Path to image file (str or Path)
            - RGB numpy array with shape ``(H, W, 3)`` in ``uint8``
            - PIL Image object

        batch_size : int, optional
            Number of images to process simultaneously. Larger batches are
            faster but require more memory. Default is 32.
        mode : {"beam", "greedy"}, optional
            Decoding strategy:

            - ``"beam"`` — beam search decoding (slower, more accurate)
            - ``"greedy"`` — greedy decoding (faster, less accurate)

            Default is ``"beam"``.
        beam_size : int, optional
            Beam width for beam search decoding. Only used when
            ``mode="beam"``. Larger values explore more hypotheses but are
            slower. Default is 8.
        temperature : float, optional
            Temperature for probability scaling in beam search. Higher values
            increase diversity. Only used when ``mode="beam"``. Default is 1.7.
        alpha : float, optional
            Length penalty coefficient for beam search. Penalizes shorter
            sequences when < 1.0, favors longer when > 1.0. Only used when
            ``mode="beam"``. Default is 0.9.

        Returns
        -------
        list of (str, float) tuples
            Recognition results as list of ``(text, confidence)`` pairs where:

            - ``text`` is the recognized string
            - ``confidence`` is recognition confidence score in [0, 1]

            If input is a single image, returns a list with one element.

        Examples
        --------
        Recognize single image with beam search:

        >>> from manuscript.recognizers import TRBA
        >>> recognizer = TRBA()
        >>> results = recognizer.predict("word_image.jpg")
        >>> text, confidence = results[0]
        >>> print(f"Text: '{text}' (confidence: {confidence:.3f})")

        Batch processing with greedy decoding:

        >>> image_paths = ["word1.jpg", "word2.jpg", "word3.jpg"]
        >>> results = recognizer.predict(
        ...     image_paths,
        ...     batch_size=16,
        ...     mode="greedy"
        ... )
        >>> for img_path, (text, conf) in zip(image_paths, results):
        ...     print(f"{img_path}: '{text}' ({conf:.3f})")

        Process numpy arrays:

        >>> import cv2
        >>> img = cv2.imread("word.jpg")
        >>> img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        >>> results = recognizer.predict(img_rgb)
        """

        if not isinstance(images, list):
            images_list = [images]
        else:
            images_list = images

        results = []

        with torch.no_grad():
            for i in range(0, len(images_list), batch_size):
                batch_images = images_list[i : i + batch_size]

                # --- подготовка батча ---
                batch_tensors = []
                for img in batch_images:
                    tensor = self._preprocess_image(img)
                    batch_tensors.append(tensor.squeeze(0))
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                # --- инференс ---
                if mode == "greedy":
                    probs, pred_ids = self.model(
                        batch_tensor,
                        is_train=False,
                        batch_max_length=self.max_length,
                        mode=mode,
                    )
                elif mode == "beam":
                    probs, pred_ids = self.model(
                        batch_tensor,
                        is_train=False,
                        batch_max_length=self.max_length,
                        mode=mode,
                        beam_size=beam_size,
                        alpha=alpha,
                        temperature=temperature,
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # --- вычисление вероятностей ---
                probs = F.log_softmax(probs, dim=-1)  # [B, T, V]

                for j, pred_row in enumerate(pred_ids):
                    text = decode_tokens(
                        pred_row,
                        self.itos,
                        pad_id=self.pad_id,
                        eos_id=self.eos_id,
                        blank_id=self.blank_id,
                    )

                    # Confidence = среднее log-prob
                    pred_row = pred_row.tolist()
                    if len(pred_row) > 0:
                        token_probs = probs[j, torch.arange(len(pred_row)), pred_row]
                        confidence = token_probs.exp().mean().item()
                    else:
                        confidence = 0.0

                    results.append((text, confidence))

        return results

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
        batch_size: int = 32,
        epochs: int = 20,
        lr: float = 1e-3,
        optimizer: str = "Adam",
        scheduler: str = "ReduceLROnPlateau",
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        eval_every: int = 1,
        val_size: int = 3000,
        train_proportions: Optional[Sequence[float]] = None,
        num_workers: int = 0,
        seed: int = 42,
        resume_path: Optional[str] = None,
        save_every: Optional[int] = None,
        dual_validate: bool = False,
        beam_size: int = 8,
        beam_alpha: float = 0.9,
        beam_temperature: float = 1.7,
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
        batch_size : int, optional
            Training batch size. Default is 32.
        epochs : int, optional
            Number of training epochs. Default is 20.
        lr : float, optional
            Learning rate. Default is 1e-3.
        optimizer : {"Adam", "SGD", "AdamW"}, optional
            Optimizer type. Default is ``"Adam"``.
        scheduler : {"ReduceLROnPlateau", "StepLR", "CosineAnnealingLR"}, optional
            Learning rate scheduler type. Default is ``"ReduceLROnPlateau"``.
        weight_decay : float, optional
            L2 weight decay coefficient. Default is 0.0.
        momentum : float, optional
            Momentum for SGD optimizer. Default is 0.9.
        eval_every : int, optional
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
        resume_path : str or Path, optional
            Path to checkpoint file to resume training from. Default is ``None``.
        save_every : int, optional
            Save checkpoint every N epochs. If ``None``, only saves best model.
            Default is ``None``.
        dual_validate : bool, optional
            If ``True``, validates with both greedy and beam search decoding.
            Default is ``False``.
        beam_size : int, optional
            Beam width for beam search validation. Default is 8.
        beam_alpha : float, optional
            Length penalty for beam search. Default is 0.9.
        beam_temperature : float, optional
            Temperature for beam search. Default is 1.7.
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

            - ``"default"`` or ``True`` — use release weights
            - ``None`` or ``False`` — train from scratch
            - str/Path — path or URL to custom weights file

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
        ...     resume_path="experiments/trba_exp1/checkpoints/last.pth",
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
        """

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
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "eval_every": eval_every,
            "val_size": val_size,
            "num_workers": num_workers,
            "seed": seed,
            "dual_validate": bool(dual_validate),
            "beam_size": beam_size,
            "beam_alpha": beam_alpha,
            "beam_temperature": beam_temperature,
        }

        if exp_dir is not None:
            config_payload["exp_dir"] = exp_dir
        if val_csvs_list is not None:
            config_payload["val_csvs"] = val_csvs_list
            config_payload["val_roots"] = val_roots_list
        if train_proportions is not None:
            config_payload["train_proportions"] = list(train_proportions)
        if resume_path is not None:
            config_payload["resume_path"] = resume_path
        if save_every is not None:
            config_payload["save_every"] = save_every
        # Pretrained weights option:
        # - None/False/"none": skip
        # - "default"/True: use release weights
        # - str: path/URL to .pth/.pt/.ckpt
        if pretrain_weights is not None:
            config_payload["pretrain_weights"] = pretrain_weights

        if extra_config:
            config_payload.update(extra_config)

        # Freeze policies for model submodules
        config_payload["freeze_cnn"] = freeze_cnn
        config_payload["freeze_enc_rnn"] = freeze_enc_rnn
        config_payload["freeze_attention"] = freeze_attention

        config = Config(config_payload)
        return run_training(config, device=device)
