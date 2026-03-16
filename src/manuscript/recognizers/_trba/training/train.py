import json
import os
import logging
import csv
import random
import shutil
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data.dataset import (
    OCRDatasetAttn,
    MultiDataset,
    ProportionalBatchSampler,
)
from ..data.transforms import (
    decode_tokens,
    get_train_transform,
    get_val_transform,
    load_charset,
    list_augmentations,
    preview_augmentation,
    make_text_mosaic,
    DEFAULT_AUG_PARAMS,
)
from .metrics import compute_cer, compute_wer, compute_accuracy
from ..model.model import TRBAModel
from .utils import (
    load_checkpoint,
    save_checkpoint,
    save_weights,
    set_seed,
    load_pretrained_weights,
)


# -------------------------
# CTC Decoding
# -------------------------
def ctc_greedy_decode(logits: torch.Tensor, blank_id: int = 0) -> torch.Tensor:
    """
    CTC greedy декодирование с удалением повторов и blank токенов.

    Args:
        logits: [B, W, num_classes] - CTC логиты
        blank_id: ID blank токена (обычно 0)

    Returns:
        decoded: [B, T] - декодированные последовательности (с паддингом -1)
    """
    # Greedy decode: берем argmax
    preds = logits.argmax(dim=-1)  # [B, W]

    batch_size = preds.size(0)
    decoded_batch = []

    for b in range(batch_size):
        pred_seq = preds[b].tolist()  # [W]

        # CTC постобработка: удаляем повторы и blank
        decoded = []
        prev_token = None
        for token in pred_seq:
            if token != blank_id and token != prev_token:
                decoded.append(token)
            prev_token = token

        decoded_batch.append(decoded)

    # Паддинг до одинаковой длины
    max_len = max(len(seq) for seq in decoded_batch) if decoded_batch else 1
    padded = []
    for seq in decoded_batch:
        padded_seq = seq + [-1] * (max_len - len(seq))
        padded.append(padded_seq)

    return torch.tensor(padded, dtype=torch.long, device=logits.device)


def get_ctc_weight_for_epoch(
    epoch: int,
    initial_weight: float = 0.3,
    decay_epochs: int = 50,
    min_weight: float = 0.0,
) -> float:
    """
    Вычисляет вес CTC для текущей эпохи с затуханием.

    CTC помогает на ранних стадиях обучения, затем его влияние уменьшается,
    чтобы attention decoder доминировал для лучшего качества.

    Args:
        epoch: Текущая эпоха (1-indexed)
        initial_weight: Начальный вес CTC (например 0.3)
        decay_epochs: За сколько эпох затухает до min_weight
        min_weight: Минимальный вес (обычно 0.0 - полное затухание)

    Returns:
        Текущий вес CTC для использования в loss

    Example:
        epoch 1:  0.30
        epoch 25: 0.15
        epoch 50: 0.00 (полностью затух)
    """
    if decay_epochs <= 0:
        return initial_weight

    # Линейное затухание
    progress = min(1.0, (epoch - 1) / decay_epochs)
    current_weight = initial_weight * (1 - progress) + min_weight * progress

    return max(min_weight, current_weight)


# -------------------------
# logging
# -------------------------
def setup_logger(exp_dir: str) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # формат
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # консоль
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # файл
    os.makedirs(exp_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(exp_dir, "train.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class Config:
    _RESUME_CKPT_CANDIDATES = [
        "last_ckpt.pth",
        "best_loss_ckpt.pth",
        "best_acc_ckpt.pth",
    ]

    def __init__(self, source: Union[str, Dict[str, Any]]):
        if isinstance(source, str):
            with open(source, "r", encoding="utf-8") as f:
                user_data = json.load(f)
        elif isinstance(source, dict):
            user_data = dict(source)
        else:
            raise TypeError("Config source must be either a path to JSON or a dict.")

        merged = self._maybe_apply_resume(user_data)
        for k, v in merged.items():
            setattr(self, k, v)

        if not getattr(self, "exp_dir", None):
            exp_idx = 1
            while os.path.exists(f"exp{exp_idx}"):
                exp_idx += 1
            self.exp_dir = f"exp{exp_idx}"

    def save(self, out_path: Optional[str] = None):
        if out_path is None:
            out_path = os.path.join(self.exp_dir, "config.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def __getitem__(self, key):
        return getattr(self, key)

    def _maybe_apply_resume(self, user_data: dict) -> dict:
        resume_path = user_data.get("resume_from")

        if not resume_path:
            return dict(user_data)

        resume_path = Path(resume_path).expanduser().resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Путь для резюме не найден: {resume_path}")

        resume_dir: Path
        resume_ckpt: Optional[Path] = None

        if resume_path.is_dir():
            resume_dir = resume_path
            for name in self._RESUME_CKPT_CANDIDATES:
                candidate = resume_dir / name
                if candidate.is_file():
                    resume_ckpt = candidate
                    break
            if resume_ckpt is None:
                raise FileNotFoundError(
                    f"В каталоге {resume_dir} не найдено чекпоинтов из списка {self._RESUME_CKPT_CANDIDATES}"
                )
        else:
            resume_ckpt = resume_path
            if not resume_ckpt.is_file():
                raise FileNotFoundError(f"Чекпоинт для резюме не найден: {resume_ckpt}")
            resume_dir = resume_ckpt.parent

        resume_config_path = resume_dir / "config.json"
        resume_config = {}
        if resume_config_path.is_file():
            try:
                with open(resume_config_path, "r", encoding="utf-8") as f:
                    resume_config = json.load(f)
            except Exception as e:
                print(
                    f"[Config] Не удалось прочитать конфиг эксперимента {resume_config_path}: {e}"
                )
        else:
            print(
                f"[Config] В каталоге резюме нет config.json, используется текущий конфиг"
            )

        merged = dict(resume_config)
        for key, value in user_data.items():
            if value is not None:
                merged[key] = value

        # Store checkpoint path
        merged["resume_from"] = str(resume_ckpt)
        merged["exp_dir"] = str(resume_dir)
        return merged


def split_train_val(
    csvs,
    roots,
    stoi,
    img_h,
    img_w,
    train_transform,
    val_transform,
    encoding="utf-8",
    val_size=3000,
):
    train_sets, val_sets = [], []
    for c, r in zip(csvs, roots):
        full_ds = OCRDatasetAttn(
            c,
            r,
            stoi,
            img_height=img_h,
            img_max_width=img_w,
            transform=None,
            encoding=encoding,
        )
        n_val = min(val_size, len(full_ds))
        n_train = len(full_ds) - n_val
        if n_train <= 0:
            raise ValueError(
                f"В датасете {c} всего {len(full_ds)} примеров, меньше чем {val_size}"
            )

        train_ds, val_ds = random_split(full_ds, [n_train, n_val])

        train_ds.dataset.transform = train_transform
        val_ds.dataset.transform = val_transform

        train_sets.append(train_ds)
        val_sets.append(val_ds)
    return train_sets, val_sets


def visualize_predictions_tensorboard(
    model: nn.Module,
    val_loader: DataLoader,
    itos: List[str],
    pad_id: int,
    eos_id: int,
    blank_id: Optional[int],
    device: torch.device,
    writer: SummaryWriter,
    num_samples: int = 10,
    max_len: int = 25,
    mode: str = "greedy",
    epoch: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    **decode_kwargs,
) -> None:
    """
    Визуализирует случайные примеры распознавания в TensorBoard.

    Отображает изображения с наложенным текстом (ground truth и prediction).

    Parameters
    ----------
    model : nn.Module
        Модель для инференса.
    val_loader : DataLoader
        DataLoader с валидационными данными.
    itos : List[str]
        Список символов (индекс -> токен).
    pad_id : int
        ID токена паддинга.
    eos_id : int
        ID токена конца последовательности.
    blank_id : Optional[int]
        ID токена blank (если используется).
    device : torch.device
        Устройство для вычислений.
    writer : SummaryWriter
        TensorBoard writer для логирования изображений.
    num_samples : int, optional
        Количество случайных примеров. По умолчанию 10.
    max_len : int, optional
        Максимальная длина последовательности. По умолчанию 25.
    mode : str, optional
        Режим декодирования ("greedy" или "beam"). По умолчанию "greedy".
    epoch : Optional[int], optional
        Номер текущей эпохи.
    logger : Optional[logging.Logger], optional
        Logger для вывода информации.
    **decode_kwargs
        Параметры декодирования (beam_size, alpha, temperature).
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import torchvision

    model.eval()

    # Собираем все батчи
    all_batches = []
    with torch.no_grad():
        for batch in val_loader:
            all_batches.append(batch)

    if not all_batches:
        return

    # Выбираем случайные примеры
    num_batches = len(all_batches)
    samples_collected = []

    attempts = 0
    max_attempts = num_batches * 2

    while len(samples_collected) < num_samples and attempts < max_attempts:
        batch_idx = random.randint(0, num_batches - 1)
        imgs, text_in, target_y, lengths = all_batches[batch_idx]

        batch_size = imgs.size(0)
        if batch_size == 0:
            attempts += 1
            continue

        sample_idx = random.randint(0, batch_size - 1)
        sample_key = (batch_idx, sample_idx)

        if sample_key not in [s[0] for s in samples_collected]:
            samples_collected.append(
                (sample_key, imgs[sample_idx : sample_idx + 1], target_y[sample_idx])
            )

        attempts += 1

    if not samples_collected:
        return

    # Настройка декодирования
    forward_kwargs = {
        "is_train": False,
        "batch_max_length": max_len,
    }

    # Обрабатываем примеры
    images_with_text = []

    with torch.no_grad():
        for idx, (sample_key, img, target) in enumerate(samples_collected, 1):
            img_device = img.to(device)

            # Ground truth
            gt_text = decode_tokens(
                target.cpu(), itos, pad_id=pad_id, eos_id=eos_id, blank_id=blank_id
            )

            # Prediction
            result = model(img_device, **forward_kwargs)

            # Get predictions based on mode
            if mode == "attention":
                pred_ids = result["attention_preds"][0]
            else:  # ctc
                ctc_logits = result["ctc_logits"]
                # CTC декодирование с постобработкой
                pred_ids = ctc_greedy_decode(ctc_logits, blank_id=blank_id)[0]

            pred_text = decode_tokens(
                pred_ids.cpu(), itos, pad_id=pad_id, eos_id=eos_id, blank_id=blank_id
            )

            # Конвертируем изображение
            img_np = img[0].cpu().numpy()  # [C, H, W]

            # Денормализация
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)

            # HWC uint8
            img_np = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            # Добавляем область для текста
            img_h, img_w = img_pil.size[1], img_pil.size[0]
            text_height = 60
            new_img = Image.new(
                "RGB", (img_w, img_h + text_height), color=(255, 255, 255)
            )
            new_img.paste(img_pil, (0, 0))

            # Рисуем текст
            draw = ImageDraw.Draw(new_img)

            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()

            # Цвет в зависимости от корректности
            is_correct = gt_text == pred_text
            color_gt = (0, 128, 0) if is_correct else (255, 0, 0)

            draw.text((5, img_h + 5), f"GT:   {gt_text}", fill=color_gt, font=font)
            draw.text(
                (5, img_h + 30), f"Pred: {pred_text}", fill=(0, 0, 255), font=font
            )

            # Обратно в тензор
            img_with_text = np.array(new_img).transpose(2, 0, 1)  # CHW
            img_with_text = torch.from_numpy(img_with_text).float() / 255.0

            images_with_text.append(img_with_text)

    if not images_with_text:
        return

    # Создаем сетку
    grid = torchvision.utils.make_grid(
        images_with_text,
        nrow=min(5, len(images_with_text)),
        padding=10,
        normalize=False,
        pad_value=1.0,
    )

    # Логируем в TensorBoard
    tag = f"Predictions/{mode}"
    if epoch is not None:
        writer.add_image(tag, grid, epoch)


def log_augmentation_previews_tensorboard(
    val_loader: DataLoader,
    writer: SummaryWriter,
    aug_params: dict,
    img_h: int,
    img_w: int,
    num_samples: int = 4,
    epoch: int = 0,
    logger=None,
) -> None:
    """Log a grid of augmentation previews to TensorBoard.

    For each augmentation in ``list_augmentations()``, takes ``num_samples``
    raw images from ``val_loader`` (before any transform), applies the
    augmentation + ResizeAndPadA, and writes a grid image to TensorBoard
    under ``Augmentations/<name>``.

    Parameters
    ----------
    val_loader : DataLoader
        DataLoader; raw images are extracted from first few batches.
    writer : SummaryWriter
        TensorBoard writer.
    aug_params : dict
        Augmentation parameters (same keys as DEFAULT_AUG_PARAMS).
    img_h, img_w : int
        Target image size for ResizeAndPadA.
    num_samples : int
        Number of example images per augmentation.
    epoch : int
        Step tag written to TensorBoard.
    logger : optional Logger
    """
    import numpy as np
    import torchvision

    try:
        from ..data.transforms import (
            ResizeAndPadA,
            preview_augmentation,
            list_augmentations,
        )
    except Exception as e:
        if logger:
            logger.warning(f"log_augmentation_previews_tensorboard: import failed: {e}")
        return

    if ResizeAndPadA is None:
        return

    resize_pad = ResizeAndPadA(img_h=img_h, img_w=img_w)

    # Collect raw numpy images from the loader (images before transform)
    raw_images = []
    for batch in val_loader:
        imgs_t = batch[0]  # (B, C, H, W) tensors already transformed
        # Denormalize back to uint8 for preview
        imgs_np = imgs_t.cpu().numpy()  # (B, C, H, W) float in [-1, 1]
        imgs_np = (imgs_np * 0.5 + 0.5)  # → [0, 1]
        imgs_np = np.clip(imgs_np * 255, 0, 255).astype(np.uint8)
        imgs_np = imgs_np.transpose(0, 2, 3, 1)  # → (B, H, W, C)
        for img in imgs_np:
            raw_images.append(img)
            if len(raw_images) >= num_samples:
                break
        if len(raw_images) >= num_samples:
            break

    if not raw_images:
        return

    aug_names = list_augmentations()
    logged = 0
    for aug_name in aug_names:
        try:
            aug_imgs = []
            for img in raw_images[:num_samples]:
                try:
                    aug_img = preview_augmentation(img, aug_name, aug_params)
                    # Resize/pad for uniform grid size
                    aug_img = resize_pad.apply(aug_img)
                except Exception:
                    aug_img = resize_pad.apply(img.copy())
                # To tensor CHW float [0, 1]
                t = np.ascontiguousarray(aug_img).transpose(2, 0, 1).astype(np.float32) / 255.0
                aug_imgs.append(t)

            if not aug_imgs:
                continue

            import torch as _torch
            grid_tensors = [_torch.from_numpy(t) for t in aug_imgs]
            grid = torchvision.utils.make_grid(
                grid_tensors,
                nrow=len(grid_tensors),
                padding=4,
                normalize=False,
                pad_value=1.0,
            )
            writer.add_image(f"Augmentations/{aug_name}", grid, epoch)
            logged += 1
        except Exception as ex:
            if logger:
                logger.warning(f"log_augmentation_previews: '{aug_name}' failed: {ex}")

    if logger:
        logger.info(f"Logged augmentation previews to TensorBoard: {logged} augmentations")


def run_training(cfg: Config, device: str = "cuda"):
    seed = getattr(cfg, "seed", 42)
    set_seed(seed)
    seed = getattr(cfg, "seed", 42)
    set_seed(seed)

    # --- базовые настройки и пути ---
    exp_dir = getattr(cfg, "exp_dir", None)
    os.makedirs(exp_dir, exist_ok=True)
    logger = setup_logger(exp_dir)

    logger.info("Start training")
    logger.info(f"Experiment dir: {exp_dir}")
    logger.info(f"Seed: {seed}")

    try:
        cfg.save()
        logger.info("Saved config to exp_dir/config.json")
    except Exception as e:
        logger.info(f"Config save skipped: {e}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # пути/данные
    train_csvs = cfg.train_csvs
    train_roots = cfg.train_roots
    val_csvs = getattr(cfg, "val_csvs", None)
    val_roots = getattr(cfg, "val_roots", None)
    charset_path = cfg.charset_path
    encoding = getattr(cfg, "encoding", "utf-8")

    # модель/данные
    img_h = getattr(cfg, "img_h", 64)
    img_w = getattr(cfg, "img_w", 256)
    max_len = getattr(cfg, "max_len", 25)
    hidden_size = getattr(cfg, "hidden_size", 256)

    # text mosaic
    text_mosaic_prob = float(getattr(cfg, "text_mosaic_prob", 0.0))
    text_mosaic_n_words = int(getattr(cfg, "text_mosaic_n_words", 2))
    text_mosaic_gap_ratio = float(getattr(cfg, "text_mosaic_gap_ratio", 0.15))

    # оптимизация
    batch_size = getattr(cfg, "batch_size", 32)
    epochs = getattr(cfg, "epochs", 20)
    lr = getattr(cfg, "lr", 1e-3)
    optimizer_name = getattr(cfg, "optimizer", "AdamW")
    scheduler_name = getattr(cfg, "scheduler", "OneCycleLR")
    weight_decay = getattr(cfg, "weight_decay", 0.0)
    momentum = getattr(cfg, "momentum", 0.9)

    # Resume checkpoint handling
    resume_from = getattr(cfg, "resume_from", None)

    # Автоматический поиск последнего чекпоинта в exp_dir
    if not resume_from:
        possible_checkpoints = []
        if os.path.exists(exp_dir):
            for fname in ["checkpoint_last.pth", "checkpoint_best.pth"]:
                ckpt_path = os.path.join(exp_dir, fname)
                if os.path.isfile(ckpt_path):
                    possible_checkpoints.append(ckpt_path)

        if possible_checkpoints:
            # Предпочитаем checkpoint_last.pth для продолжения обучения
            if os.path.join(exp_dir, "checkpoint_last.pth") in possible_checkpoints:
                resume_from = os.path.join(exp_dir, "checkpoint_last.pth")
            else:
                resume_from = possible_checkpoints[0]
            logger.info(f"Автоматически найден чекпоинт для продолжения: {resume_from}")

    # Validation interval
    val_interval = getattr(cfg, "val_interval", 1)
    try:
        val_interval = int(val_interval)
    except (TypeError, ValueError):
        raise ValueError("val_interval must be a positive integer")
    if val_interval < 1:
        raise ValueError("val_interval must be >= 1")

    # Save interval (не используется, всегда сохраняем при валидации)
    save_interval = getattr(cfg, "save_interval", None)

    train_proportions = getattr(cfg, "train_proportions", None)
    val_size = getattr(cfg, "val_size", 3000)
    num_workers = getattr(cfg, "num_workers", 0)

    # CTC loss weight для стабилизации начального обучения
    ctc_weight_initial = getattr(cfg, "ctc_weight", 0.3)
    ctc_weight_decay_epochs = getattr(
        cfg, "ctc_weight_decay_epochs", 15
    )  # Затухание за 15 эпох
    ctc_weight_min = getattr(
        cfg, "ctc_weight_min", 0.03
    )  # Полное затухание после decay_epochs

    # Gradient clipping для защиты от взрыва градиентов
    max_grad_norm = getattr(cfg, "max_grad_norm", 5.0)

    # --- директории и TensorBoard ---
    if resume_from:
        exp_dir = os.path.dirname(resume_from)
        os.makedirs(exp_dir, exist_ok=True)
        logger = setup_logger(exp_dir)

    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    pin_memory = torch.cuda.is_available()

    metrics_csv_path = os.path.join(exp_dir, "metrics_epoch.csv")
    if not os.path.exists(metrics_csv_path):
        header = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_acc",
            "val_cer",
            "val_wer",
            "lr",
        ]
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)

    best_loss_path = os.path.join(exp_dir, "best_loss_ckpt.pth")
    best_acc_path = os.path.join(exp_dir, "best_acc_ckpt.pth")
    last_path = os.path.join(exp_dir, "last_ckpt.pth")
    best_loss_weights_path = os.path.join(exp_dir, "best_loss_weights.pth")
    best_acc_weights_path = os.path.join(exp_dir, "best_acc_weights.pth")
    last_weights_path = os.path.join(exp_dir, "last_weights.pth")

    # --- charset ---
    itos, stoi = load_charset(charset_path)
    PAD = stoi["<PAD>"]
    SOS = stoi["<SOS>"]
    EOS = stoi["<EOS>"]
    BLANK = stoi.get("<BLANK>", None)
    num_classes = len(itos)
    logger.info(f"Charset loaded: {num_classes} tokens")
    
    # Копируем charset в папку эксперимента
    charset_dest = os.path.join(exp_dir, "charset.txt")
    if not os.path.exists(charset_dest) or os.path.abspath(charset_path) != os.path.abspath(charset_dest):
        shutil.copy2(charset_path, charset_dest)
        logger.info(f"Charset copied to experiment dir: {charset_dest}")

    # --- модель ---
    num_encoder_layers = getattr(cfg, "num_encoder_layers", 2)
    cnn_in_channels = getattr(cfg, "cnn_in_channels", 3)
    cnn_out_channels = getattr(cfg, "cnn_out_channels", 512)
    cnn_backbone = getattr(cfg, "cnn_backbone", "seresnet31")

    # CTC всегда используется при обучении для стабилизации
    use_ctc_head = True

    model = TRBAModel(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_encoder_layers=num_encoder_layers,
        img_h=img_h,
        img_w=img_w,
        cnn_in_channels=cnn_in_channels,
        cnn_out_channels=cnn_out_channels,
        cnn_backbone=cnn_backbone,
        sos_id=SOS,
        eos_id=EOS,
        pad_id=PAD,
        blank_id=BLANK,
        use_ctc_head=use_ctc_head,
    ).to(device)

    pretrain_src = getattr(cfg, "pretrain_weights", "default")

    if not resume_from:  # resume_from может быть найден автоматически

        def _normalize_pretrain(v) -> str:
            if v is True:
                return "default"
            if v is False or v is None:
                return "none"
            return str(v)

        pretrain_src = _normalize_pretrain(pretrain_src)
        if pretrain_src.lower() not in ("none", ""):
            if pretrain_src.lower() == "default":
                pretrain_src = (
                    "https://github.com/konstantinkozhin/manuscript-ocr/releases/download/"
                    "v0.1.0/trba_lite_g1.pth"
                )
                logger.info(
                    "Using default pretrained weights: trba_lite_g1.pth (GitHub release)"
                )
                logger.info(
                    "Default pretrain config: "
                    "https://github.com/konstantinkozhin/manuscript-ocr/releases/download/"
                    "v0.1.0/trba_lite_g1.json"
                )

            stats = load_pretrained_weights(
                model,
                src=pretrain_src,
                map_location=str(device),
                logger=logger,
            )
            if not stats.get("ok", False):
                logger.warning(
                    f"Pretrained load failed from {pretrain_src}. Proceeding with random init."
                )
    else:
        logger.info(
            f"Пропуск загрузки pretrain_weights - найден чекпоинт для продолжения: {resume_from}"
        )

    # --- политика заморозки весов ---
    def _normalize_policy(v: Optional[str]) -> str:
        if v is None:
            return "none"
        v = str(v).strip().lower()
        mapping = {
            # english
            "full": "full",
            "all": "full",
            "freeze": "full",
            "frozen": "full",
            "partial": "partial",
            "smart": "partial",
            "best": "partial",
            "none": "none",
            "no": "none",
            "off": "none",
            "false": "none",
            # russian
            "полностью": "full",
            "частично": "partial",
            "нет": "none",
            "не": "none",
        }
        return mapping.get(v, v)

    def _freeze_module(m: torch.nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    # we keep BN in eval for frozen CNN parts to avoid stats drift
    always_eval_modules = []

    def _collect_bn(mod: torch.nn.Module):
        for sub in mod.modules():
            if isinstance(sub, torch.nn.BatchNorm2d):
                always_eval_modules.append(sub)

    def _wrap_forward_no_grad(module: torch.nn.Module):
        if module is None:
            return
        if getattr(module, "_wrapped_no_grad", False):
            return
        orig_forward = module.forward

        def _no_grad_forward(*args, **kwargs):
            with torch.no_grad():
                return orig_forward(*args, **kwargs)

        module.forward = _no_grad_forward  # type: ignore[attr-defined]
        module._wrapped_no_grad = True  # type: ignore[attr-defined]

    def _apply_cnn_policy(policy: str):
        if policy == "full":
            _freeze_module(model.cnn)
            _collect_bn(model.cnn)
            _wrap_forward_no_grad(model.cnn)
            return "cnn: FULL (all layers frozen)"
        if policy == "partial":
            # freeze early/mid layers, unfreeze the last stage + conv_out
            to_freeze = []
            for name in ("conv0", "layer1", "layer2", "layer3"):
                if hasattr(model.cnn, name):
                    to_freeze.append(getattr(model.cnn, name))
            for part in to_freeze:
                _freeze_module(part)
                _collect_bn(part)
                _wrap_forward_no_grad(part)
            return "cnn: PARTIAL (unfrozen layer4 + conv_out)"
        return "cnn: NONE (no freezing)"

    def _apply_enc_rnn_policy(policy: str):
        if policy == "full":
            _freeze_module(model.enc_rnn)
            _wrap_forward_no_grad(model.enc_rnn)
            return "enc_rnn: FULL (all layers frozen)"
        if policy == "partial":
            # enc_rnn is Sequential of two BiLSTMs; freeze first, unfreeze last
            try:
                first = model.enc_rnn[0]
                _freeze_module(first)
                _wrap_forward_no_grad(first)
            except Exception:
                pass
            return "enc_rnn: PARTIAL (unfrozen last BiLSTM)"
        return "enc_rnn: NONE (no freezing)"

    def _apply_attention_policy(policy: str):
        if policy == "full":
            _freeze_module(model.attn)
            _wrap_forward_no_grad(model.attn)
            return "attention: FULL (all layers frozen)"
        if policy == "partial":
            # keep generator trainable (most beneficial for vocab adaptation), freeze attention_cell
            if hasattr(model.attn, "attention_cell"):
                _freeze_module(model.attn.attention_cell)
                _wrap_forward_no_grad(model.attn.attention_cell)
            return "attention: PARTIAL (unfrozen generator, frozen attention_cell)"
        return "attention: NONE (no freezing)"

    freeze_cnn = _normalize_policy(getattr(cfg, "freeze_cnn", "none"))
    freeze_enc = _normalize_policy(getattr(cfg, "freeze_enc_rnn", "none"))
    freeze_attn = _normalize_policy(getattr(cfg, "freeze_attention", "none"))

    msgs = []
    msgs.append(_apply_cnn_policy(freeze_cnn))
    msgs.append(_apply_enc_rnn_policy(freeze_enc))
    msgs.append(_apply_attention_policy(freeze_attn))

    # register a pre-forward hook to keep frozen BN layers in eval mode
    if always_eval_modules:

        def _set_bn_eval(module, inputs):
            for bn in always_eval_modules:
                bn.eval()

        model.register_forward_pre_hook(_set_bn_eval)
    for m in msgs:
        logger.info(f"Freeze policy applied: {m}")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    n_frozen = n_total - n_trainable
    logger.info(
        f"Parameters: trainable={n_trainable:,} | frozen={n_frozen:,} | total={n_total:,}"
    )

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    # --- optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "Adam":
        optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(
            trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # --- scheduler ---
    # Для OneCycleLR нужно знать total_steps, поэтому создадим scheduler после создания loader
    scheduler = None
    scheduler_name_for_later = scheduler_name

    scaler = amp.GradScaler()

    # --- трансформации ---
    train_transform = get_train_transform(cfg.__dict__, img_h=img_h, img_w=img_w)
    val_transform = get_val_transform(img_h, img_w)

    # --- датасеты и лоадеры ---
    train_sets = []
    val_sets = []

    if val_csvs and val_roots:
        for i, (train_csv, train_root) in enumerate(zip(train_csvs, train_roots)):
            has_separate_val = (
                i < len(val_csvs)
                and i < len(val_roots)
                and val_csvs[i] is not None
                and val_roots[i] is not None
            )

            if has_separate_val:
                train_ds = OCRDatasetAttn(
                    train_csv,
                    train_root,
                    stoi,
                    img_height=img_h,
                    img_max_width=img_w,
                    transform=train_transform,
                    encoding=encoding,
                    max_len=max_len,
                    strict_max_len=True,
                    text_mosaic_prob=text_mosaic_prob,
                    text_mosaic_n_words=text_mosaic_n_words,
                    text_mosaic_gap_ratio=text_mosaic_gap_ratio,
                )
                val_ds = OCRDatasetAttn(
                    val_csvs[i],
                    val_roots[i],
                    stoi,
                    img_height=img_h,
                    img_max_width=img_w,
                    transform=val_transform,
                    encoding=encoding,
                    max_len=max_len,
                    strict_max_len=True,
                )
                train_sets.append(train_ds)
                val_sets.append(val_ds)
            else:
                full_ds = OCRDatasetAttn(
                    train_csv,
                    train_root,
                    stoi,
                    img_height=img_h,
                    img_max_width=img_w,
                    transform=None,
                    encoding=encoding,
                    max_len=max_len,
                    strict_max_len=True,
                    text_mosaic_prob=text_mosaic_prob,
                    text_mosaic_n_words=text_mosaic_n_words,
                    text_mosaic_gap_ratio=text_mosaic_gap_ratio,
                )
                n_val = min(3000 if val_size is None else val_size, len(full_ds))
                n_train = len(full_ds) - n_val
                if n_train <= 0:
                    raise ValueError(
                        f"В датасете {train_csv} всего {len(full_ds)} примеров, меньше чем {n_val}"
                    )

                train_ds, val_ds = random_split(full_ds, [n_train, n_val])
                train_ds.dataset.transform = train_transform
                val_ds.dataset.transform = val_transform

                train_sets.append(train_ds)
                val_sets.append(val_ds)
    else:
        train_sets, val_sets = split_train_val(
            train_csvs,
            train_roots,
            stoi,
            img_h,
            img_w,
            train_transform,
            val_transform,
            encoding=encoding,
            val_size=val_size,
        )

    collate_train = OCRDatasetAttn.make_collate_attn(
        stoi, max_len=max_len, drop_blank=True
    )
    collate_val = OCRDatasetAttn.make_collate_attn(
        stoi, max_len=max_len, drop_blank=True
    )

    if train_proportions is not None:
        total = sum(train_proportions)
        proportions = [p / total for p in train_proportions]
        assert len(proportions) == len(
            train_sets
        ), "train_proportions != num train_sets"
        train_dataset = MultiDataset(train_sets)
        batch_sampler = ProportionalBatchSampler(train_sets, batch_size, proportions)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_train,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            ConcatDataset(train_sets),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_train,
            pin_memory=pin_memory,
        )

    val_loaders_individual = [
        DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_val,
            pin_memory=pin_memory,
        )
        for val_set in val_sets
    ]

    # --- stats about dataset sizes ---
    def _total_len(ds_list):
        total = 0
        for ds in ds_list:
            try:
                total += len(ds)
            except Exception:
                pass
        return total

    n_train_samples = _total_len(train_sets)
    n_val_samples = _total_len(val_sets)

    # Логгирование информации о валидационной стратегии
    logger.info(f"Validation strategy:")
    for i, (train_csv, train_root) in enumerate(zip(train_csvs, train_roots)):
        has_separate_val = (
            val_csvs
            and val_roots
            and i < len(val_csvs)
            and i < len(val_roots)
            and val_csvs[i] is not None
            and val_roots[i] is not None
        )
        if has_separate_val:
            logger.info(
                f"  Dataset {i}: using separate validation set from {val_roots[i]}"
            )
        else:
            logger.info(
                f"  Dataset {i}: using split from training set (val_size={val_size})"
            )

    msg_ds = (
        f"Datasets: train={n_train_samples} samples across {len(train_sets)} set(s); "
        f"val={n_val_samples} samples across {len(val_sets)} set(s)"
    )
    total_val_batches = sum(len(loader) for loader in val_loaders_individual)
    msg_ld = (
        f"Loaders: train_batches/epoch={len(train_loader)}; "
        f"val_batches={total_val_batches}; batch_size={batch_size}"
    )

    print(msg_ds)
    logger.info(msg_ds)
    print(msg_ld)
    logger.info(msg_ld)

    # --- создаем scheduler после train_loader ---
    if scheduler_name_for_later == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
        )
    elif scheduler_name_for_later == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name_for_later == "OneCycleLR":
        total_steps = len(train_loader) * epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr * 2,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25,
        )
        logger.info(
            f"OneCycleLR: max_lr={lr * 2:.6f}, total_steps={total_steps}, pct_start=0.1"
        )
    elif scheduler_name_for_later in ("None", None, ""):
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name_for_later}")

    # --- resume ---
    start_epoch = 1
    global_step = 0
    best_val_loss, best_val_acc = float("inf"), -1.0

    if resume_from and os.path.isfile(resume_from):
        try:
            ckpt = load_checkpoint(
                resume_from,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
        except Exception as e:
            logger.warning(
                f"Failed to load optimizer/scheduler state from resume due to: {e}.\n"
                f"Will load model weights only and continue."
            )
            ckpt = load_checkpoint(
                resume_from, model, optimizer=None, scheduler=None, scaler=None
            )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        best_val_acc = float(ckpt.get("best_val_acc", best_val_acc))
        logger.info(
            f"Resumed from: {resume_from} (epoch={start_epoch-1}, step={global_step})"
        )

    # ── Augmentation previews (logged once before training) ──────────────────
    if start_epoch == 1 and val_loaders_individual:
        try:
            logger.info("Logging augmentation previews to TensorBoard...")
            log_augmentation_previews_tensorboard(
                val_loader=val_loaders_individual[0],
                writer=writer,
                aug_params=cfg.__dict__,
                img_h=img_h,
                img_w=img_w,
                num_samples=4,
                epoch=0,
                logger=logger,
            )
        except Exception as _aug_prev_err:
            logger.warning(f"Augmentation preview logging failed: {_aug_prev_err}")

    # ── Epoch-0 baseline validation (before any weight updates) ─────────────
    if start_epoch == 1 and val_loaders_individual:
        logger.info("Running epoch-0 baseline validation (before training)...")
        try:
            model.eval()
            torch.cuda.empty_cache()
            _ctc_w0 = get_ctc_weight_for_epoch(
                0, ctc_weight_initial, ctc_weight_decay_epochs, ctc_weight_min
            )
            _total_loss0 = 0.0
            _total_samples0 = 0
            _all_refs0, _all_hyps0 = [], []
            with torch.no_grad():
                for _vl in val_loaders_individual:
                    for _imgs, _text_in, _target_y, _lengths in _vl:
                        _imgs = _imgs.to(device, non_blocking=pin_memory)
                        _text_in = _text_in.to(device, non_blocking=pin_memory)
                        _target_y = _target_y.to(device, non_blocking=pin_memory)
                        with amp.autocast():
                            _res = model(_imgs, text=_text_in, is_train=True, batch_max_length=max_len)
                            _al = criterion(_res["attention_logits"].reshape(-1, _res["attention_logits"].size(-1)), _target_y.reshape(-1))
                            _cl = model.compute_ctc_loss(_res["ctc_logits"], _target_y, _lengths)
                            _loss = (1 - _ctc_w0) * _al + _ctc_w0 * _cl
                        _total_loss0 += float(_loss.item())
                        _total_samples0 += 1
                        # decode
                        _res2 = model(_imgs, is_train=False, batch_max_length=max_len)
                        for _ti, _pi in zip(_target_y.cpu(), _res2["attention_preds"].cpu()):
                            _all_refs0.append(decode_tokens(_ti, itos, PAD, EOS, BLANK))
                            _all_hyps0.append(decode_tokens(_pi, itos, PAD, EOS, BLANK))
                        del _imgs, _text_in, _target_y

            _avg_loss0 = _total_loss0 / max(1, _total_samples0)
            _acc0 = sum(r == h for r, h in zip(_all_refs0, _all_hyps0)) / max(1, len(_all_refs0))
            _cer0 = compute_cer(_all_refs0, _all_hyps0) if _all_refs0 else 0.0
            _wer0 = compute_wer(_all_refs0, _all_hyps0) if _all_refs0 else 0.0

            writer.add_scalar("Loss/val_epoch", _avg_loss0, 0)
            writer.add_scalar("Accuracy/val_attention", _acc0, 0)
            writer.add_scalar("CER/val_attention", _cer0, 0)
            writer.add_scalar("WER/val_attention", _wer0, 0)

            logger.info(
                f"Epoch 000 baseline | val_loss={_avg_loss0:.4f} | "
                f"acc={_acc0:.4f} | CER={_cer0:.4f} | WER={_wer0:.4f}"
            )
            torch.cuda.empty_cache()
        except Exception as _e0:
            logger.warning(f"Epoch-0 baseline validation failed: {_e0}")

    # --- training loop ---
    for epoch in range(start_epoch, epochs + 1):
        # Вычисляем текущий CTC weight с затуханием
        if epoch < ctc_weight_decay_epochs:
            ctc_weight = get_ctc_weight_for_epoch(
                epoch,
                initial_weight=ctc_weight_initial,
                decay_epochs=ctc_weight_decay_epochs,
                min_weight=ctc_weight_min,
            )
        else:
            ctc_weight = ctc_weight_initial  # Не используется, но для совместимости

        # train
        model.train()
        total_train_loss = 0.0
        total_attn_loss = 0.0
        total_ctc_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False)
        for imgs, text_in, target_y, lengths in pbar:
            imgs = imgs.to(device, non_blocking=pin_memory)
            text_in = text_in.to(device, non_blocking=pin_memory)
            target_y = target_y.to(device, non_blocking=pin_memory)
            lengths = lengths.to(device, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                # New unified API
                result = model(
                    imgs,
                    text=text_in,
                    is_train=True,
                    batch_max_length=max_len,
                )

                # Compute dual loss (attention + CTC)
                loss = torch.tensor(0.0, device=device)
                attn_loss_val = torch.tensor(0.0, device=device)
                ctc_loss_val = torch.tensor(0.0, device=device)

                attn_logits = result["attention_logits"]
                ctc_logits = result["ctc_logits"]

                attn_loss_val = criterion(
                    attn_logits.reshape(-1, attn_logits.size(-1)),
                    target_y.reshape(-1),
                )
                ctc_loss_val = model.compute_ctc_loss(ctc_logits, target_y, lengths)

                # Weighted combination
                loss = (1.0 - ctc_weight) * attn_loss_val + ctc_weight * ctc_loss_val

            scaler.scale(loss).backward()

            # Gradient clipping для защиты от взрыва градиентов (NaN)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            loss_val = float(loss.item())
            attn_loss_scalar = float(attn_loss_val.item())
            ctc_loss_scalar = float(ctc_loss_val.item())

            total_train_loss += loss_val
            total_attn_loss += attn_loss_scalar
            total_ctc_loss += ctc_loss_scalar

            writer.add_scalar("Loss/train_step", loss_val, global_step)
            writer.add_scalar("Loss/train_attn_step", attn_loss_scalar, global_step)
            writer.add_scalar("Loss/train_ctc_step", ctc_loss_scalar, global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss_val:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

            # OneCycleLR требует step после каждого батча
            if scheduler is not None and isinstance(
                scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                scheduler.step()

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        avg_attn_loss = total_attn_loss / max(1, len(train_loader))
        avg_ctc_loss = total_ctc_loss / max(1, len(train_loader))

        should_eval = ((epoch - start_epoch) % val_interval == 0) or (epoch == epochs)

        avg_val_loss = None
        val_acc = None
        val_cer = None
        val_wer = None

        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/train_attn_epoch", avg_attn_loss, epoch)
        writer.add_scalar("Loss/train_ctc_epoch", avg_ctc_loss, epoch)

        if should_eval:
            model.eval()
            torch.cuda.empty_cache()

            total_val_loss = 0.0
            total_samples = 0

            # Валидация только с attention decoder (CTC используется только при обучении)
            eval_modes = {"attention": {"mode": "attention", "decoder": "attention"}}

            aggregate_mode_stats = {
                mode_name: {
                    "total_correct": 0,
                    "total_predictions": 0,
                    "all_refs": [],
                    "all_hyps": [],
                }
                for mode_name in eval_modes
            }

            for i, val_loader_single in enumerate(val_loaders_individual):
                total_val_loss_single = 0.0
                refs_single: List[str] = []
                hyps_single = {mode_name: [] for mode_name in eval_modes}

                pbar_val = tqdm(
                    val_loader_single,
                    desc=f"Valid Set {i} {epoch}/{epochs}",
                    leave=False,
                )
                with torch.no_grad():
                    for imgs, text_in, target_y, lengths in pbar_val:
                        imgs = imgs.to(device, non_blocking=pin_memory)
                        text_in = text_in.to(device, non_blocking=pin_memory)
                        target_y = target_y.to(device, non_blocking=pin_memory)

                        with amp.autocast():
                            result = model(
                                imgs,
                                text=text_in,
                                is_train=True,
                                batch_max_length=max_len,
                            )
                            attn_logits = result["attention_logits"]
                            ctc_logits = result["ctc_logits"]

                            attn_loss_val = criterion(
                                attn_logits.reshape(-1, attn_logits.size(-1)),
                                target_y.reshape(-1),
                            )
                            ctc_loss_val = model.compute_ctc_loss(
                                ctc_logits, target_y, lengths
                            )
                            val_loss = (
                                attn_loss_val * (1 - ctc_weight)
                                + ctc_loss_val * ctc_weight
                            )

                        total_val_loss_single += float(val_loss.item())

                        preds_batch = {}
                        for mode_name, mode_cfg in eval_modes.items():
                            result = model(
                                imgs,
                                is_train=False,
                                batch_max_length=max_len,
                            )
                            decoder_type = mode_cfg["decoder"]
                            if decoder_type == "attention":
                                pred_ids = result["attention_preds"]
                            else:
                                ctc_logits = result["ctc_logits"]
                                pred_ids = ctc_greedy_decode(ctc_logits, blank_id=BLANK)
                            preds_batch[mode_name] = pred_ids.cpu()

                        tgt_ids = target_y.cpu()
                        refs_batch = []
                        for t_row in tgt_ids:
                            ref = decode_tokens(
                                t_row, itos, pad_id=PAD, eos_id=EOS, blank_id=BLANK
                            )
                            refs_batch.append(ref)
                            refs_single.append(ref)

                        for mode_name, pred_tensor in preds_batch.items():
                            for p_row, ref in zip(pred_tensor, refs_batch):
                                hyp = decode_tokens(
                                    p_row, itos, pad_id=PAD, eos_id=EOS, blank_id=BLANK
                                )
                                hyps_single[mode_name].append(hyp)

                        pbar_val.set_postfix(val_loss=f"{float(val_loss.item()):.4f}")
                        del imgs, text_in, target_y, preds_batch, tgt_ids

                avg_val_loss_single = total_val_loss_single / max(
                    1, len(val_loader_single)
                )

                writer.add_scalar(f"Loss/val_set_{i}", avg_val_loss_single, epoch)

                for mode_name, hyps in hyps_single.items():
                    val_acc_single = compute_accuracy(refs_single, hyps)
                    val_cer_single = compute_cer(refs_single, hyps)
                    val_wer_single = compute_wer(refs_single, hyps)

                    metric_suffix = (
                        f"/val_set_{i}"
                        if mode_name == "greedy"
                        else f"/val_set_{i}_{mode_name}"
                    )
                    writer.add_scalar(f"Accuracy{metric_suffix}", val_acc_single, epoch)
                    writer.add_scalar(f"CER{metric_suffix}", val_cer_single, epoch)
                    writer.add_scalar(f"WER{metric_suffix}", val_wer_single, epoch)

                    stats = aggregate_mode_stats[mode_name]
                    correct_single = sum(1 for r, h in zip(refs_single, hyps) if r == h)
                    stats["total_correct"] += correct_single
                    stats["total_predictions"] += len(refs_single)
                    
                    # Accumulate for aggregate metrics
                    stats["all_refs"].extend(refs_single)
                    stats["all_hyps"].extend(hyps)

                total_val_loss += total_val_loss_single
                total_samples += len(val_loader_single)

                del refs_single, hyps_single
                torch.cuda.empty_cache()

            avg_val_loss = total_val_loss / max(1, total_samples)

            def _finalize(mode_name: str):
                stats = aggregate_mode_stats[mode_name]
                total_pred = max(1, stats["total_predictions"])
                acc = stats["total_correct"] / total_pred
                
                # Compute CER and WER on all accumulated predictions
                all_refs = stats["all_refs"]
                all_hyps = stats["all_hyps"]
                cer = compute_cer(all_refs, all_hyps) if all_refs else 0.0
                wer = compute_wer(all_refs, all_hyps) if all_refs else 0.0
                
                return acc, cer, wer

            # Метрики только для attention decoder
            primary_mode_name = "attention"
            val_acc, val_cer, val_wer = _finalize(primary_mode_name)

            writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/val_attention", val_acc, epoch)
            writer.add_scalar("CER/val_attention", val_cer, epoch)
            writer.add_scalar("WER/val_attention", val_wer, epoch)

            # Визуализация случайных примеров в TensorBoard
            if val_loaders_individual:
                # Выбираем случайный датасет для визуализации
                random_val_loader = random.choice(val_loaders_individual)

                # Визуализация только attention decoder
                visualize_predictions_tensorboard(
                    model=model,
                    val_loader=random_val_loader,
                    itos=itos,
                    pad_id=PAD,
                    eos_id=EOS,
                    blank_id=BLANK,
                    device=device,
                    writer=writer,
                    num_samples=10,
                    max_len=max_len,
                    mode="attention",
                    epoch=epoch,
                    logger=logger,
                )
        else:
            logger.info(
                f"Epoch {epoch:03d}: skipping validation (val_interval={val_interval})"
            )

        with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if should_eval:
                row = [
                    epoch,
                    f"{avg_train_loss:.6f}",
                    f"{avg_val_loss:.6f}",
                    f"{val_acc:.6f}",
                    f"{val_cer:.6f}",
                    f"{val_wer:.6f}",
                ]
                row.append(f"{optimizer.param_groups[0]['lr']:.6e}")
                w.writerow(row)
            else:
                row = [
                    epoch,
                    f"{avg_train_loss:.6f}",
                    "skipped",
                    "skipped",
                    "skipped",
                    "skipped",
                ]
                row.append(f"{optimizer.param_groups[0]['lr']:.6e}")
                w.writerow(row)

        msg_parts = [
            f"Epoch {epoch:03d}/{epochs}",
            f"train_loss={avg_train_loss:.4f}",
        ]
        if should_eval:
            msg_parts.extend(
                [
                    f"val_loss={avg_val_loss:.4f}",
                    f"acc={val_acc:.4f}",
                    f"CER={val_cer:.4f}",
                    f"WER={val_wer:.4f}",
                ]
            )
        else:
            msg_parts.append(f"val=skipped (val_interval={val_interval})")
        msg_parts.append(f"lr={optimizer.param_groups[0]['lr']:.2e}")
        msg = " | ".join(msg_parts)
        print(msg)
        logger.info(msg)

        if should_eval:
            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                avg_val_loss,
                val_acc,
                itos,
                stoi,
                {
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lr": lr,
                    "optimizer": optimizer_name,
                    "scheduler": scheduler_name,
                    "weight_decay": weight_decay,
                    "momentum": momentum,
                    "img_h": img_h,
                    "img_w": img_w,
                    "encoding": encoding,
                    "max_len": max_len,
                    "hidden_size": hidden_size,
                    "num_encoder_layers": num_encoder_layers,
                    "cnn_in_channels": cnn_in_channels,
                    "cnn_out_channels": cnn_out_channels,
                    "cnn_backbone": cnn_backbone,
                    "ctc_weight": ctc_weight,
                    "charset_path": charset_path,
                    "train_csvs": train_csvs,
                    "train_roots": train_roots,
                    "val_csvs": val_csvs,
                    "val_roots": val_roots,
                },
                log_dir,
            )
            save_weights(last_weights_path, model)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(
                    best_loss_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_val_loss,
                    val_acc,
                    itos,
                    stoi,
                    {
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "lr": lr,
                        "optimizer": optimizer_name,
                        "scheduler": scheduler_name,
                        "weight_decay": weight_decay,
                        "momentum": momentum,
                        "img_h": img_h,
                        "img_w": img_w,
                        "encoding": encoding,
                        "max_len": max_len,
                        "hidden_size": hidden_size,
                        "num_encoder_layers": num_encoder_layers,
                        "cnn_in_channels": cnn_in_channels,
                        "cnn_out_channels": cnn_out_channels,
                        "cnn_backbone": cnn_backbone,
                        "ctc_weight": ctc_weight,
                        "charset_path": charset_path,
                        "train_csvs": train_csvs,
                        "train_roots": train_roots,
                        "val_csvs": val_csvs,
                        "val_roots": val_roots,
                    },
                    log_dir,
                )
                save_weights(best_loss_weights_path, model)
                logger.info(f"New best val_loss: {best_val_loss:.4f} (epoch {epoch})")

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    best_acc_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_val_loss,
                    best_val_acc,
                    itos,
                    stoi,
                    {
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "lr": lr,
                        "optimizer": optimizer_name,
                        "scheduler": scheduler_name,
                        "weight_decay": weight_decay,
                        "momentum": momentum,
                        "img_h": img_h,
                        "img_w": img_w,
                        "encoding": encoding,
                        "max_len": max_len,
                        "hidden_size": hidden_size,
                        "num_encoder_layers": num_encoder_layers,
                        "cnn_in_channels": cnn_in_channels,
                        "cnn_out_channels": cnn_out_channels,
                        "cnn_backbone": cnn_backbone,
                        "ctc_weight": ctc_weight,
                        "charset_path": charset_path,
                        "train_csvs": train_csvs,
                        "train_roots": train_roots,
                        "val_csvs": val_csvs,
                        "val_roots": val_roots,
                    },
                    log_dir,
                )
                save_weights(best_acc_weights_path, model)
                logger.info(f"New best acc: {best_val_acc:.4f} (epoch {epoch})")

        if scheduler is not None:
            # OneCycleLR вызывается после каждого батча, не здесь
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                pass
            elif isinstance(scheduler, ReduceLROnPlateau):
                if should_eval and avg_val_loss is not None:
                    scheduler.step(avg_val_loss)
            else:
                scheduler.step()

    writer.close()
    logger.info("Training finished.")
    
    try:
        logger.info("Attempting to export best model to ONNX...")
        from manuscript.recognizers import TRBA
        
        onnx_path = os.path.join(exp_dir, "best_acc_model.onnx")
        config_path = os.path.join(exp_dir, "config.json")
        
        TRBA.export(
            weights_path=best_acc_weights_path,
            config_path=config_path,
            charset_path=charset_dest,
            output_path=onnx_path,
            opset_version=14,
            simplify=True,
        )
        logger.info(f"ONNX model exported successfully: {onnx_path}")
    except Exception as e:
        logger.warning(f"Failed to export ONNX model: {e}")
    
    return {"val_acc": best_val_acc, "val_loss": best_val_loss, "exp_dir": exp_dir}
