import os
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional, Tuple
import cv2

from .model.model import RCNN
from .data.transforms import load_charset, get_val_transform, decode_tokens


class TRBAInfer:    
    def __init__(
        self,
        model_path: str,
        charset_path: str,
        device: str = "auto",
        img_h: int = 64,
        img_w: int = 256,
    ):
        self.model_path = model_path
        self.charset_path = charset_path
        self.img_h = img_h
        self.img_w = img_w

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.itos, self.stoi = load_charset(charset_path)
        self.pad_id = self.stoi["<PAD>"]
        self.sos_id = self.stoi["<SOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.blank_id = self.stoi.get("<BLANK>", None)

        self.transform = get_val_transform(img_h, img_w)

        self.model = self._load_model()
        
        print(f"OCR model loaded on {self.device}")
        print(f"Charset size: {len(self.itos)} symbols")
        print(f"Input image size: {img_h}x{img_w}")
    
    def _load_model(self) -> RCNN:
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        num_classes = len(self.itos)

        if "config" in checkpoint:
            hidden_size = checkpoint["config"].get("hidden_size", 256)
            state_dict = checkpoint["model_state"]
        elif "model_state_dict" in checkpoint:
            hidden_size = checkpoint.get("hidden_size", 256)
            state_dict = checkpoint["model_state_dict"]
        else:
            hidden_size = 256
            state_dict = checkpoint

        model = RCNN(
            num_classes=num_classes,
            hidden_size=hidden_size,
            sos_id=self.sos_id,
            eos_id=self.eos_id,
            pad_id=self.pad_id,
            blank_id=self.blank_id,
        )
        
        model.load_state_dict(state_dict)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _preprocess_image(self, image: Union[np.ndarray, str, Image.Image]) -> torch.Tensor:
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
        images: Union[np.ndarray, str, Image.Image, List[Union[np.ndarray, str, Image.Image]]], 
        max_length: int = 25,
        batch_size: int = 32,
        return_confidence: bool = False
    ) -> Union[str, List[str], Tuple[str, float], List[Tuple[str, float]]]:
        
        is_single = not isinstance(images, list)
        
        if is_single:
            images_list = [images]
        else:
            images_list = images
        
        results = []
        
        with torch.no_grad():
            for i in range(0, len(images_list), batch_size):
                batch_images = images_list[i:i + batch_size]
                
                batch_tensors = []
                for img in batch_images:
                    tensor = self._preprocess_image(img)
                    batch_tensors.append(tensor.squeeze(0))
                
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                output = self.model(batch_tensor, is_train=False, batch_max_length=max_length)
                pred_ids = output.argmax(dim=-1).cpu()
                
                if return_confidence:
                    probs = torch.softmax(output, dim=-1)
                    max_probs = probs.max(dim=-1)[0].cpu()
                
                for j, pred_row in enumerate(pred_ids):
                    text = decode_tokens(
                        pred_row,
                        self.itos,
                        pad_id=self.pad_id,
                        eos_id=self.eos_id,
                        blank_id=self.blank_id
                    )
                    
                    if return_confidence:
                        valid_mask = (pred_row != self.pad_id) & (pred_row != self.eos_id)
                        if valid_mask.sum() > 0:
                            confidence = max_probs[j][valid_mask].mean().item()
                        else:
                            confidence = 0.0
                        results.append((text, confidence))
                    else:
                        results.append(text)
        
        if is_single:
            return results[0]
        else:
            return results