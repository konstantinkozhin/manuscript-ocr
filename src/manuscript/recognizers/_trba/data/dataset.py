import csv
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ....utils import read_image
from .transforms import (
    build_file_index,
    get_train_transform,
    get_val_transform,
    make_text_mosaic,
    pack_attention_targets,
)


class OCRDatasetAttn(Dataset):
    def __init__(
        self,
        csv_path: str,
        images_dir: Union[str, list],
        stoi: dict,
        img_height: int = 32,
        img_max_width: int = 128,
        encoding: str = "utf-8",
        transform: Optional[callable] = None,
        num_workers: int = -1,
        delimiter: Optional[str] = None,
        has_header: Optional[bool] = None,
        strict_charset: bool = True,
        validate_image: bool = True,
        max_len: Optional[int] = None,
        strict_max_len: bool = True,
        text_mosaic_prob: float = 0.0,
        text_mosaic_n_words: int = 2,
        text_mosaic_gap_ratio: Optional[float] = None,
        return_raw_image: bool = False,
    ):
        self.images_dir = images_dir
        self.img_h = img_height
        self.img_w = img_max_width
        self.stoi = stoi
        self.transform = transform
        self.samples: List[Tuple[str, str]] = []
        self._file_index = build_file_index(images_dir)
        self._encoding = encoding
        # Keep the caller-supplied hint (or None); actual delimiter resolved in _read_rows via sniffing.
        self._delimiter = delimiter  # may be None → auto-detect
        self._has_header = has_header
        self._strict_charset = strict_charset
        self._validate_image = validate_image
        self._max_len = max_len
        self._strict_max_len = strict_max_len
        self._text_mosaic_prob = float(text_mosaic_prob)
        self._text_mosaic_n_words = max(2, int(text_mosaic_n_words))
        self._return_raw_image = bool(return_raw_image)
        # None → make_text_mosaic рандомит gap 3–5% на каждый вызов
        self._text_mosaic_gap_ratio = float(text_mosaic_gap_ratio) if text_mosaic_gap_ratio is not None else None

        self._reasons = {
            "bad_row": 0, "empty_fname": 0, "empty_label": 0,
            "charset": 0, "too_long": 0,
            "missing_path": 0, "ambiguous": 0, "readfail": 0,
        }
        self._examples = {k: [] for k in self._reasons}
        self._EX_MAX = 8
        self._missing_chars = Counter()

        rows = self._read_rows(csv_path)
        self._maybe_detect_header(rows)
        self._build_samples(rows, num_workers)

        self._invalid_mask = [False] * len(self.samples)
        self._checked_mask = [not self._validate_image] * len(self.samples)
        self._lazy_warned = False
        self._lazy_skipped = 0
        self._max_getitem_retries = 8

        self._print_summary(csv_path)
        if self._validate_image:
            print("[OCRDatasetAttn] Lazy image validation is enabled; unreadable images will be skipped during the first access.")

        if not self.samples:
            raise RuntimeError(f"No valid samples remain in the dataset {csv_path}!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self.samples)):
            raise IndexError(idx)

        if not self._validate_image:
            abs_path, label = self.samples[idx]
            try:
                img = read_image(abs_path)
            except Exception as e:
                raise IndexError(f"Error reading image {abs_path}: {e}") from e

            img, label = self._apply_text_mosaic(img, label)

            if self.transform:
                augmented = self.transform(image=img)
                image_or_tensor = augmented["image"]
            else:
                image_or_tensor = img

            if self._return_raw_image:
                return image_or_tensor, label

            if torch.is_tensor(image_or_tensor):
                return image_or_tensor, label

            tensor = torch.from_numpy(image_or_tensor).permute(2, 0, 1).float() / 255.0
            return tensor, label

        attempts = self._max_getitem_retries
        current_idx = idx
        while attempts > 0:
            abs_path, label = self.samples[current_idx]

            if self._invalid_mask[current_idx]:
                current_idx = self._choose_alternative_index(current_idx)
                attempts -= 1
                continue

            try:
                img = read_image(abs_path)
                self._checked_mask[current_idx] = True
            except Exception as e:
                self._mark_sample_invalid(current_idx, abs_path, e)
                current_idx = self._choose_alternative_index(current_idx)
                attempts -= 1
                continue

            img, label = self._apply_text_mosaic(img, label)

            if self.transform:
                augmented = self.transform(image=img)
                image_or_tensor = augmented["image"]
            else:
                image_or_tensor = img

            if self._return_raw_image:
                return image_or_tensor, label

            if torch.is_tensor(image_or_tensor):
                return image_or_tensor, label

            tensor = torch.from_numpy(image_or_tensor).permute(2, 0, 1).float() / 255.0
            return tensor, label

        raise RuntimeError("Failed to fetch a valid sample after lazy validation retries.")

    def _apply_text_mosaic(self, img, label: str):
        """Склеить текущее слово с N-1 случайными словами из датасета горизонтально.

        Возвращает (img_mosaic, label_concat) или исходные (img, label), если
        мозаика не применяется (вероятность не выпала или датасет слишком мал).
        """
        import numpy as np

        if self._text_mosaic_prob <= 0.0 or not self.samples:
            return img, label
        if random.random() > self._text_mosaic_prob:
            return img, label
        if len(self.samples) < 2:
            return img, label

        images = [img]
        labels = [label]

        for _ in range(self._text_mosaic_n_words - 1):
            other_idx = random.randrange(len(self.samples))
            other_path, other_label = self.samples[other_idx]
            try:
                other_img = read_image(other_path)
            except Exception:
                continue
            # Проверяем, что объединённая метка не превысит max_len
            combined = label + " " + " ".join(labels[1:] + [other_label]) if len(labels) > 1 \
                else label + " " + other_label
            if self._max_len is not None and len(combined) > self._max_len:
                continue
            images.append(other_img)
            labels.append(other_label)

        if len(images) < 2:
            return img, label

        mosaic_img = make_text_mosaic(images, gap_ratio=self._text_mosaic_gap_ratio)
        mosaic_label = " ".join(labels)
        return mosaic_img, mosaic_label

    def _mark_sample_invalid(self, idx: int, abs_path: str, error: Exception):
        self._invalid_mask[idx] = True
        self._reasons["readfail"] += 1
        if len(self._examples["readfail"]) < self._EX_MAX:
            self._examples["readfail"].append(f"{abs_path} :: {type(error).__name__}")
        self._lazy_skipped += 1
        if not self._lazy_warned:
            print("[OCRDatasetAttn] Lazy validation detected unreadable images; they will be skipped during iteration.")
            self._lazy_warned = True

    def _choose_alternative_index(self, bad_idx: int) -> int:
        candidates = [i for i, invalid in enumerate(self._invalid_mask) if not invalid and i != bad_idx]
        if candidates:
            return random.choice(candidates)
        raise RuntimeError("No valid samples remain after filtering unreadable images.")

    @staticmethod
    def sample_batch_input_size(
        img_height: int,
        img_width: int,
        resolution_jitter: float = 0.0,
        min_img_height: int = 24,
        min_img_width: int = 132,
    ) -> Tuple[int, int]:
        """Sample a shared train-time input size for the whole batch."""
        if resolution_jitter <= 0:
            return int(img_height), int(img_width)

        scale = random.uniform(1.0 - resolution_jitter, 1.0 + resolution_jitter)
        target_h = max(int(min_img_height), int(round(img_height * scale)))
        target_w = max(int(min_img_width), int(round(img_width * scale)))
        return target_h, target_w

    @staticmethod
    def make_collate_attn(
        stoi,
        max_len: int,
        drop_blank: bool = True,
        batch_img_size: Optional[Tuple[int, int]] = None,
        resolution_jitter: float = 0.0,
        min_img_height: int = 24,
        min_img_width: int = 132,
        image_transform_params: Optional[dict] = None,
        is_train: bool = False,
    ):
        def collate(batch):
            imgs, labels_text = zip(*batch)
            first_img = imgs[0]
            if torch.is_tensor(first_img):
                imgs = torch.stack(imgs)
            else:
                if batch_img_size is None:
                    raise ValueError("batch_img_size is required when collating raw images.")

                target_h, target_w = int(batch_img_size[0]), int(batch_img_size[1])
                if is_train:
                    target_h, target_w = OCRDatasetAttn.sample_batch_input_size(
                        img_height=batch_img_size[0],
                        img_width=batch_img_size[1],
                        resolution_jitter=resolution_jitter,
                        min_img_height=min_img_height,
                        min_img_width=min_img_width,
                    )
                    image_transform = get_train_transform(
                        image_transform_params or {},
                        img_h=target_h,
                        img_w=target_w,
                    )
                else:
                    image_transform = get_val_transform(target_h, target_w)

                imgs = torch.stack(
                    [image_transform(image=img)["image"] for img in imgs]
                )
            text_in, target_y, lengths = pack_attention_targets(
                labels_text, stoi=stoi, max_len=max_len, drop_blank=drop_blank
            )
            return imgs, text_in, target_y, lengths
        return collate

    @staticmethod
    def _sniff_delimiter(csv_path: str, encoding: str, hint_delimiter: Optional[str]) -> str:
        """Return the delimiter to use for *csv_path*.

        Priority:
        1. If the caller supplied an explicit *hint_delimiter*, use it.
        2. If the file extension is ``.tsv``, assume tab.
        3. Try ``csv.Sniffer`` on the first 4 KiB of the file.
        4. Fall back to comma.
        """
        if hint_delimiter is not None:
            return hint_delimiter
        if csv_path.lower().endswith(".tsv"):
            return "\t"
        try:
            with open(csv_path, newline="", encoding=encoding) as fh:
                sample = fh.read(4096)
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            return dialect.delimiter
        except csv.Error:
            return ","

    def _read_rows(self, csv_path: str):
        delimiter = self._sniff_delimiter(csv_path, self._encoding, self._delimiter)
        # Store the detected delimiter so callers can inspect it if needed.
        self._delimiter = delimiter
        with open(csv_path, newline="", encoding=self._encoding) as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)
        return rows

    def _maybe_detect_header(self, rows: List[List[str]]):
        if self._has_header is not None or not rows:
            return
        head0 = str(rows[0][0]).strip().lower()
        self._has_header = head0 in {"file", "filename", "image", "path", "img", "name"}
        if self._has_header:
            rows.pop(0)
        self._rows = rows

        if not hasattr(self, "_rows"):
            self._rows = rows

    @staticmethod
    def _norm_label(s: str) -> str:
        return s.replace("\u00A0", " ").strip().replace("\ufeff", "")

    @staticmethod
    def _norm_fname(s: str) -> str:
        return s.strip().replace("\ufeff", "").replace("\\", "/")

    def _resolve_path(self, fname: str) -> Optional[str]:
        if os.path.isabs(fname) and os.path.exists(fname):
            return fname

        if isinstance(self.images_dir, str):
            p = os.path.join(self.images_dir, fname)
            if os.path.exists(p):
                return p
        else:
            for root in self.images_dir:
                p = os.path.join(root, fname)
                if os.path.exists(p):
                    return p

        base = os.path.basename(fname).lower()
        candidates = self._file_index.get(base, [])
        if not candidates:
            return None
        if len(candidates) > 1:
            self._reasons["ambiguous"] += 1
            if len(self._examples["ambiguous"]) < self._EX_MAX:
                self._examples["ambiguous"].append((fname, candidates[:3]))
        return candidates[0]

    def _effective_len(self, label: str) -> int:
        if not self._strict_charset:
            return len(label)
        return sum(1 for c in label if c in self.stoi)

    def _validate_row(self, row: List[str]) -> Optional[Tuple[str, str]]:
        if len(row) < 2:
            self._reasons["bad_row"] += 1
            if len(self._examples["bad_row"]) < self._EX_MAX:
                self._examples["bad_row"].append(row)
            return None

        fname = self._norm_fname(row[0])
        label = self._norm_label(row[1])

        if not fname:
            self._reasons["empty_fname"] += 1
            if len(self._examples["empty_fname"]) < self._EX_MAX:
                self._examples["empty_fname"].append(row)
            return None

        if label == "":
            self._reasons["empty_label"] += 1
            if len(self._examples["empty_label"]) < self._EX_MAX:
                self._examples["empty_label"].append(fname)
            return None

        if self._strict_charset:
            missing = [c for c in label if c not in self.stoi]
            if missing:
                self._reasons["charset"] += 1
                self._missing_chars.update(missing)
                if len(self._examples["charset"]) < self._EX_MAX:
                    uniq = "".join(sorted(set(missing)))[:20]
                    self._examples["charset"].append((fname, label[:50], uniq))
                return None

        if self._strict_max_len and self._max_len is not None:
            if self._effective_len(label) > self._max_len:
                self._reasons["too_long"] += 1
                if len(self._examples["too_long"]) < self._EX_MAX:
                    self._examples["too_long"].append((fname, len(label), f"eff>{self._max_len}"))
                return None

        abs_path = self._resolve_path(fname)
        if not abs_path or not os.path.exists(abs_path):
            self._reasons["missing_path"] += 1
            if len(self._examples["missing_path"]) < self._EX_MAX:
                self._examples["missing_path"].append(fname)
            return None

        return abs_path, label

    def _build_samples(self, rows: List[List[str]], num_workers: int):
        if num_workers == -1:
            workers = os.cpu_count() or 4
        elif num_workers is None:
            workers = 8
        else:
            workers = max(1, num_workers)

        results, skipped = [], 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self._validate_row, row) for row in self._rows]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="Проверка датасета", leave=False):
                res = fut.result()
                if res is not None:
                    results.append(res)
                else:
                    skipped += 1
        self.samples = results
        self._skipped = skipped

    def _print_summary(self, csv_path: str):
        if self._skipped > 0:
            print(f"[OCRDatasetAttn] {csv_path}: пропущено {self._skipped} записей.")
            order = ["bad_row","empty_fname","empty_label","charset","too_long","missing_path","ambiguous","readfail"]
            for k in order:
                cnt = self._reasons[k]
                if cnt > 0:
                    print(f"  - {k}: {cnt}")
                    ex = self._examples[k]
                    if ex:
                        print(f"    examples: {ex[:self._EX_MAX]}")
            if self._reasons["charset"] > 0 and self._missing_chars:
                print("  Missing characters (TOP 30):")
                for ch, cnt in self._missing_chars.most_common(30):
                    print(f"    '{ch}' (U+{ord(ch):04X}, repr={repr(ch)}): {cnt} times")

class ProportionalBatchSampler:
    def __init__(self, datasets, batch_size, proportions):
        assert abs(sum(proportions) - 1.0) < 1e-6, "Proportions must sum to 1"
        self.datasets = datasets
        self.batch_size = batch_size
        self.proportions = proportions
        self.idxs = [list(range(len(ds))) for ds in datasets]
        for idxs in self.idxs:
            random.shuffle(idxs)

    def __iter__(self):
        n_batches = len(self)
        for _ in range(n_batches):
            batch = []
            for ds_idx, prop in enumerate(self.proportions):
                n = int(round(self.batch_size * prop))
                if n == 0:
                    continue

                if len(self.idxs[ds_idx]) < n:
                    self.idxs[ds_idx] = list(range(len(self.datasets[ds_idx])))
                    random.shuffle(self.idxs[ds_idx])

                chosen = [self.idxs[ds_idx].pop() for _ in range(n)]
                batch.extend([(ds_idx, c) for c in chosen])

            random.shuffle(batch)
            yield batch

    def __len__(self):
        min_batches = min(
            len(ds) // max(1, int(round(self.batch_size * prop)))
            for ds, prop in zip(self.datasets, self.proportions)
            if prop > 0
        )
        return min_batches


class MultiDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        ds_idx, sample_idx = index
        return self.datasets[ds_idx][sample_idx]

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)



