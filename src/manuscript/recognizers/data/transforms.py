import os
from collections import defaultdict
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


def build_file_index(roots, exts={".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}):
    """Построение индекса файлов для быстрого поиска."""
    if isinstance(roots, str):
        roots = [roots]
    index = defaultdict(list)
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if exts and ext not in exts:
                    continue
                index[fn.lower()].append(os.path.join(dirpath, fn))
    return index


def imread_cv2(path: str):
    """Чтение изображения через OpenCV с поддержкой Unicode путей."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_charset(charset_path: str):
    """
    Загрузка символьного словаря из файла формата:
        <PAD>
        <SOS>
        <EOS>
        <BLANK>
        a
        b
        ...
    Возвращает (itos, stoi).
    """
    itos = []
    with open(charset_path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.rstrip("\n")
            if tok == "":
                continue
            itos.append(tok)
    stoi = {s: i for i, s in enumerate(itos)}
    return itos, stoi


class ResizeAndPadA(A.ImageOnlyTransform):
    """Кастомная трансформация для изменения размера и добавления отступов."""
    
    def __init__(
        self,
        img_h=32,
        img_w=256,
        align_h="left",
        align_v="center",
        always_apply=True,
        p=1.0,
    ):
        super().__init__(always_apply, p)
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.align_h = align_h
        self.align_v = align_v

    def _interp(self, src_h, src_w, dst_h, dst_w):
        if dst_h < src_h or dst_w < src_w:
            return cv2.INTER_AREA
        return cv2.INTER_LINEAR

    def apply(self, img, **params):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        h, w = img.shape[:2]

        scale = min(self.img_h / max(h, 1), self.img_w / max(w, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        interp = self._interp(h, w, new_h, new_w)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        canvas = np.full((self.img_h, self.img_w, 3), 255, dtype=img.dtype)

        if self.align_h == "left":
            x0 = 0
        elif self.align_h == "right":
            x0 = self.img_w - new_w
        else:
            x0 = (self.img_w - new_w) // 2

        if self.align_v == "top":
            y0 = 0
        elif self.align_v == "bottom":
            y0 = self.img_h - new_h
        else:
            y0 = (self.img_h - new_h) // 2
            
        x0 = max(0, min(x0, self.img_w - new_w))
        y0 = max(0, min(y0, self.img_h - new_h))

        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = img_resized
        return canvas


def pack_attention_targets(texts, stoi, max_len, drop_blank=True):
    """Упаковка текстовых целей для attention модели."""
    PAD = stoi["<PAD>"]
    SOS = stoi["<SOS>"]
    EOS = stoi["<EOS>"]
    BLANK = stoi.get("<BLANK>", None)

    B = len(texts)
    T = max_len + 1

    text_in = torch.full((B, T), PAD, dtype=torch.long)
    text_in[:, 0] = SOS

    target_y = torch.full((B, T), PAD, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(texts):
        ids = []
        for ch in s:
            if ch not in stoi:
                continue
            idx = stoi[ch]
            if drop_blank and BLANK is not None and idx == BLANK:
                continue
            ids.append(idx)

        L = min(len(ids), max_len)
        if L > 0:
            text_in[i, 1 : 1 + L] = torch.tensor(ids[:L], dtype=torch.long)
            target_y[i, :L] = torch.tensor(ids[:L], dtype=torch.long)

        target_y[i, L] = EOS
        lengths[i] = L + 1

    return text_in, target_y, lengths


def get_train_transform(params, img_h, img_w):
    """Создание трансформаций для тренировочных данных."""
    return A.Compose(
        [
            ResizeAndPadA(img_h=img_h, img_w=img_w),
            A.ShiftScaleRotate(
                shift_limit=round(params.get("shift_limit", 0.03), 4),
                scale_limit=round(params.get("scale_limit", 0.08), 4),
                rotate_limit=int(params.get("rotate_limit", 3)),
                border_mode=0,
                value=(255, 255, 255),
                p=round(params.get("p_ShiftScaleRotate", 0.3), 4),
            ),
            A.RandomBrightnessContrast(
                brightness_limit=round(params.get("brightness_limit", 0.2), 4),
                contrast_limit=round(params.get("contrast_limit", 0.2), 4),
                p=round(params.get("p_BrightnessContrast", 0.3), 4),
            ),
            A.InvertImg(p=round(params.get("invert_p", 0.0), 4)),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )


def get_val_transform(img_h, img_w):
    """Создание трансформаций для валидационных данных."""
    return A.Compose(
        [
            ResizeAndPadA(img_h=img_h, img_w=img_w),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )


def decode_tokens(ids, itos, pad_id, eos_id, blank_id=None):
    """Декодирование токенов в текст."""
    out = []
    for t in ids:
        t = int(t)
        if t == eos_id:
            break
        if t == pad_id or (blank_id is not None and t == blank_id):
            continue
        out.append(itos[t])
    return "".join(out)