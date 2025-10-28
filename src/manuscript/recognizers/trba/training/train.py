import json
import os
import logging
import csv
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
)
from .metrics import character_error_rate, compute_accuracy, word_error_rate
from ..model.model import RCNN
from .utils import load_checkpoint, save_checkpoint, save_weights, set_seed


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
        resume_path = user_data.get("resume_path")
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

        merged["resume_path"] = str(resume_ckpt)
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


def run_training(cfg: Config, device: str = "cuda"):
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

    # оптимизация
    batch_size = getattr(cfg, "batch_size", 32)
    epochs = getattr(cfg, "epochs", 20)
    lr = getattr(cfg, "lr", 1e-3)
    optimizer_name = getattr(cfg, "optimizer", "Adam")
    scheduler_name = getattr(cfg, "scheduler", "ReduceLROnPlateau")
    weight_decay = getattr(cfg, "weight_decay", 0.0)
    momentum = getattr(cfg, "momentum", 0.9)

    # прочее
    resume_path = getattr(cfg, "resume_path", None)
    eval_every = getattr(cfg, "eval_every", getattr(cfg, "save_every", 1))
    try:
        eval_every = int(eval_every)
    except (TypeError, ValueError):
        raise ValueError("eval_every must be a positive integer")
    if eval_every < 1:
        raise ValueError("eval_every must be >= 1")
    train_proportions = getattr(cfg, "train_proportions", None)
    val_size = getattr(cfg, "val_size", 3000)
    num_workers = getattr(cfg, "num_workers", 0)
    dual_validate = bool(getattr(cfg, "dual_validate", False))
    beam_size = getattr(cfg, "beam_size", 8)
    beam_alpha = getattr(cfg, "beam_alpha", 0.9)
    beam_temperature = getattr(cfg, "beam_temperature", 1.7)

    # --- директории и TensorBoard ---
    if resume_path:
        exp_dir = os.path.dirname(resume_path)
        os.makedirs(exp_dir, exist_ok=True)
        logger = setup_logger(exp_dir)

    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    metrics_csv_path = os.path.join(exp_dir, "metrics_epoch.csv")
    if not os.path.exists(metrics_csv_path):
        header = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_acc",
            "val_cer",
            "val_wer",
        ]
        if dual_validate:
            header.extend(["val_acc_beam", "val_cer_beam", "val_wer_beam"])
        header.append("lr")
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

    # --- модель ---
    model = RCNN(
        num_classes=num_classes,
        hidden_size=hidden_size,
        sos_id=SOS,
        eos_id=EOS,
        pad_id=PAD,
        blank_id=BLANK,
    ).to(device)

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

    def _apply_cnn_policy(policy: str):
        if policy == "full":
            _freeze_module(model.cnn)
            _collect_bn(model.cnn)
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
            return "cnn: PARTIAL (unfrozen layer4 + conv_out)"
        return "cnn: NONE (no freezing)"

    def _apply_enc_rnn_policy(policy: str):
        if policy == "full":
            _freeze_module(model.enc_rnn)
            return "enc_rnn: FULL (all layers frozen)"
        if policy == "partial":
            # enc_rnn is Sequential of two BiLSTMs; freeze first, unfreeze last
            try:
                first = model.enc_rnn[0]
                _freeze_module(first)
            except Exception:
                pass
            return "enc_rnn: PARTIAL (unfrozen last BiLSTM)"
        return "enc_rnn: NONE (no freezing)"

    def _apply_attention_policy(policy: str):
        if policy == "full":
            _freeze_module(model.attn)
            return "attention: FULL (all layers frozen)"
        if policy == "partial":
            # keep generator trainable (most beneficial for vocab adaptation), freeze attention_cell
            if hasattr(model.attn, "attention_cell"):
                _freeze_module(model.attn.attention_cell)
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
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name in ("None", None):
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

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
                )
                n_val = min(val_size if val_size else 3000, len(full_ds))
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
        )
    else:
        train_loader = DataLoader(
            ConcatDataset(train_sets),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_train,
        )

    val_loaders_individual = [
        DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_val,
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

    # --- resume ---
    start_epoch = 1
    global_step = 0
    best_val_loss, best_val_acc = float("inf"), -1.0

    if resume_path and os.path.isfile(resume_path):
        try:
            ckpt = load_checkpoint(
                resume_path,
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
            ckpt = load_checkpoint(resume_path, model, optimizer=None, scheduler=None, scaler=None)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        best_val_acc = float(ckpt.get("best_val_acc", best_val_acc))
        logger.info(
            f"Resumed from: {resume_path} (epoch={start_epoch-1}, step={global_step})"
        )

    # --- training loop ---
    for epoch in range(start_epoch, epochs + 1):
        # train
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False)
        for imgs, text_in, target_y, lengths in pbar:
            imgs = imgs.to(device)
            text_in = text_in.to(device)
            target_y = target_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                logits = model(
                    imgs, text=text_in, is_train=True, batch_max_length=max_len
                )  # [B,T,V]
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), target_y.reshape(-1)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_val = float(loss.item())
            total_train_loss += loss_val
            writer.add_scalar("Loss/train_step", loss_val, global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss_val:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        should_eval = ((epoch - start_epoch) % eval_every == 0) or (epoch == epochs)

        avg_val_loss = None
        val_acc = None
        val_cer = None
        val_wer = None

        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)

        if should_eval:
            model.eval()
            torch.cuda.empty_cache()

            total_val_loss = 0.0
            total_samples = 0

            eval_modes = {
                "greedy": {
                    "forward_kwargs": {
                        "mode": "greedy",
                    }
                }
            }
            if dual_validate:
                eval_modes["beam"] = {
                    "forward_kwargs": {
                        "mode": "beam",
                        "beam_size": beam_size,
                        "alpha": beam_alpha,
                        "temperature": beam_temperature,
                    }
                }

            aggregate_mode_stats = {
                mode_name: {
                    "total_correct": 0,
                    "total_predictions": 0,
                    "total_cer_sum": 0.0,
                    "total_wer_sum": 0.0,
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
                        imgs = imgs.to(device)
                        text_in = text_in.to(device)
                        target_y = target_y.to(device)

                        with amp.autocast():
                            logits_tf = model(
                                imgs,
                                text=text_in,
                                is_train=True,
                                batch_max_length=max_len,
                            )
                            val_loss = criterion(
                                logits_tf.reshape(-1, logits_tf.size(-1)),
                                target_y.reshape(-1),
                            )
                        total_val_loss_single += float(val_loss.item())

                        preds_batch = {}
                        for mode_name, mode_cfg in eval_modes.items():
                            forward_kwargs = dict(mode_cfg["forward_kwargs"])
                            _, pred_ids = model(
                                imgs,
                                is_train=False,
                                batch_max_length=max_len,
                                **forward_kwargs,
                            )
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
                        del imgs, text_in, target_y, logits_tf, preds_batch, tgt_ids

                avg_val_loss_single = total_val_loss_single / max(
                    1, len(val_loader_single)
                )

                writer.add_scalar(f"Loss/val_set_{i}", avg_val_loss_single, epoch)

                for mode_name, hyps in hyps_single.items():
                    val_acc_single = compute_accuracy(refs_single, hyps)
                    val_cer_single = sum(
                        character_error_rate(r, h)
                        for r, h in zip(refs_single, hyps)
                    ) / max(1, len(refs_single))
                    val_wer_single = sum(
                        word_error_rate(r, h) for r, h in zip(refs_single, hyps)
                    ) / max(1, len(refs_single))

                    metric_suffix = (
                        f"/val_set_{i}"
                        if mode_name == "greedy"
                        else f"/val_set_{i}_{mode_name}"
                    )
                    writer.add_scalar(f"Accuracy{metric_suffix}", val_acc_single, epoch)
                    writer.add_scalar(f"CER{metric_suffix}", val_cer_single, epoch)
                    writer.add_scalar(f"WER{metric_suffix}", val_wer_single, epoch)

                    stats = aggregate_mode_stats[mode_name]
                    correct_single = sum(
                        1 for r, h in zip(refs_single, hyps) if r == h
                    )
                    stats["total_correct"] += correct_single
                    stats["total_predictions"] += len(refs_single)
                    stats["total_cer_sum"] += sum(
                        character_error_rate(r, h)
                        for r, h in zip(refs_single, hyps)
                    )
                    stats["total_wer_sum"] += sum(
                        word_error_rate(r, h) for r, h in zip(refs_single, hyps)
                    )

                total_val_loss += total_val_loss_single
                total_samples += len(val_loader_single)

                del refs_single, hyps_single
                torch.cuda.empty_cache()

            avg_val_loss = total_val_loss / max(1, total_samples)

            def _finalize(mode_name: str):
                stats = aggregate_mode_stats[mode_name]
                total_pred = max(1, stats["total_predictions"])
                return (
                    stats["total_correct"] / total_pred,
                    stats["total_cer_sum"] / total_pred,
                    stats["total_wer_sum"] / total_pred,
                )

            val_acc, val_cer, val_wer = _finalize("greedy")
            val_acc_beam = val_cer_beam = val_wer_beam = None
            if dual_validate:
                val_acc_beam, val_cer_beam, val_wer_beam = _finalize("beam")

            writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("CER/val", val_cer, epoch)
            writer.add_scalar("WER/val", val_wer, epoch)
            if dual_validate:
                writer.add_scalar("Accuracy/val_beam", val_acc_beam, epoch)
                writer.add_scalar("CER/val_beam", val_cer_beam, epoch)
                writer.add_scalar("WER/val_beam", val_wer_beam, epoch)
        else:
            logger.info(
                f"Epoch {epoch:03d}: skipping validation (eval_every={eval_every})"
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
                if dual_validate:
                    row.extend(
                        [
                            f"{val_acc_beam:.6f}",
                            f"{val_cer_beam:.6f}",
                            f"{val_wer_beam:.6f}",
                        ]
                    )
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
                if dual_validate:
                    row.extend(["skipped", "skipped", "skipped"])
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
            if dual_validate:
                msg_parts.extend(
                    [
                        f"acc_beam={val_acc_beam:.4f}",
                        f"CER_beam={val_cer_beam:.4f}",
                        f"WER_beam={val_wer_beam:.4f}",
                    ]
                )
        else:
            msg_parts.append(f"val=skipped (eval_every={eval_every})")
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
            if isinstance(scheduler, ReduceLROnPlateau):
                if should_eval and avg_val_loss is not None:
                    scheduler.step(avg_val_loss)
            else:
                scheduler.step()

    writer.close()
    logger.info("Training finished.")
    return {"val_acc": best_val_acc, "val_loss": best_val_loss, "exp_dir": exp_dir}
