import os
import json
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Optional, Sequence, Dict, Any
import cv2

from .model.model import RCNN
from .data.transforms import load_charset, get_val_transform, decode_tokens
from .training.utils import load_checkpoint
from .training.train import Config, run_training
import torch.nn.functional as F


class TRBAInfer:
    def __init__(
        self,
        model_path: str,
        charset_path: str = None,
        config_path: str = None,
        device: str = "auto",
    ):
        self.model_path = model_path

        if charset_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            charset_path = os.path.join(current_dir, "configs", "charset.txt")

        self.charset_path = charset_path
        self.config_path = config_path

        if not os.path.exists(self.charset_path):
            raise FileNotFoundError(f"Charset file not found: {self.charset_path}")

        if config_path is not None:
            with open(config_path, "r", encoding="utf-8") as f:
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

    def _load_model(self) -> RCNN:
        model = RCNN(
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
        mode: str = "greedy",
        beam_size: int = 8,
        temperature: float = 1.7,
        alpha: float = 0.9,
    ) -> List[Tuple[str, float]]:

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
        def _ensure_path_list(
            value: Optional[Union[str, Sequence[str]]],
            field_name: str,
            allow_none: bool = False,
        ) -> Optional[List[str]]:
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

            return [os.fspath(item) for item in raw_items]

        train_csvs_list = _ensure_path_list(train_csvs, "train_csvs")
        train_roots_list = _ensure_path_list(train_roots, "train_roots")

        if len(train_csvs_list) != len(train_roots_list):
            raise ValueError(
                "train_csvs and train_roots must contain the same number of items"
            )

        val_csvs_list = _ensure_path_list(val_csvs, "val_csvs", allow_none=True)
        val_roots_list = _ensure_path_list(val_roots, "val_roots", allow_none=True)

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
