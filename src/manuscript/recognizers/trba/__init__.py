import os
import json
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple
import cv2

from .model.model import RCNN
from .data.transforms import load_charset, get_val_transform, decode_tokens
from .training.utils import (
    load_checkpoint,
    decode_predictions,
)


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
    ) -> List[Tuple[str, float]]:

        if not isinstance(images, list):
            images_list = [images]
        else:
            images_list = images

        results = []

        with torch.no_grad():
            for i in range(0, len(images_list), batch_size):
                batch_images = images_list[i : i + batch_size]

                batch_tensors = []
                for img in batch_images:
                    tensor = self._preprocess_image(img)
                    batch_tensors.append(tensor.squeeze(0))

                batch_tensor = torch.stack(batch_tensors).to(self.device)

                output = self.model(
                    batch_tensor, is_train=False, batch_max_length=self.max_length
                )
                pred_ids = decode_predictions(
                    output,
                    mode="beam",
                    eos_id=self.eos_id,
                    pad_id=self.pad_id,
                    vis=True,
                    itos=self.itos,
                ).cpu()

                probs = torch.softmax(output, dim=-1)

                for j, pred_row in enumerate(pred_ids):
                    text = decode_tokens(
                        pred_row,
                        self.itos,
                        pad_id=self.pad_id,
                        eos_id=self.eos_id,
                        blank_id=self.blank_id,
                    )

                    pred_row = pred_row.tolist()
                    if len(pred_row) > 0:
                        token_probs = probs[j, torch.arange(len(pred_row)), pred_row]
                        confidence = token_probs.mean().item()
                    else:
                        confidence = 0.0

                    results.append((text, confidence))

        return results
