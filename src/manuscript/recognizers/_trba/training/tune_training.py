"""
Ray Tune integration for TRBA OCR hyperparameter search.

Этот модуль — экспериментальная обёртка вокруг training loop из train.py.
Не модифицирует основной train.py, работает параллельно.

Два режима:
  1) ASHA  — быстрый поиск: запускает N trial-ов с разными гиперпараметрами,
             рано останавливает плохие. Каждый trial стартует с одних и тех же
             pretrained весов.
  2) PBT   — Population Based Training: trial-ы обмениваются весами "на лету",
             мутируют lr/ctc_weight/weight_decay между эпохами. Лучше для
             fine-tuning когда уже есть хорошие веса.

Использование:
    from manuscript.recognizers._trba.training.tune_training import run_tune

    results = run_tune(
        base_config="config.json",   # или dict
        mode="asha",                 # "asha" | "pbt"
        num_samples=8,
        max_epochs=20,
        gpus_per_trial=0.5,
    )

Тумблер (в config.json):
    {
        "use_ray_tune": true,        // включает tune вместо обычного run_training
        "tune_mode": "asha",         // "asha" или "pbt"
        "tune_num_samples": 8,
        "tune_gpus_per_trial": 0.5,
        "pretrain_weights": "path/to/weights.pth"  // или "default"
    }
"""

import json
import os
import tempfile
import logging
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import torch

logger = logging.getLogger("tune_training")


# ---------------------------------------------------------------------------
# Lazy imports — ray may not be installed
# ---------------------------------------------------------------------------
def _check_ray_installed():
    try:
        import ray  # noqa: F401
        from ray import tune  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Default search spaces
# ---------------------------------------------------------------------------

def get_default_search_space_asha() -> dict:
    """Пространство поиска для ASHA (быстрый wide search)."""
    from ray import tune

    return {
        "lr": tune.loguniform(1e-5, 5e-3),
        "batch_size": tune.choice([16, 32, 64]),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "ctc_weight": tune.uniform(0.05, 0.5),
        "ctc_weight_decay_epochs": tune.choice([10, 15, 25, 50]),
        "optimizer": tune.choice(["Adam", "AdamW"]),
        "max_grad_norm": tune.choice([1.0, 5.0, 10.0]),
    }


def get_default_search_space_pbt() -> dict:
    """Пространство мутаций для PBT (fine-tuning на лету)."""
    from ray import tune

    return {
        "lr": tune.loguniform(1e-5, 5e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "ctc_weight": tune.uniform(0.03, 0.4),
        "max_grad_norm": tune.choice([1.0, 5.0, 10.0]),
    }


# ---------------------------------------------------------------------------
# Trainable function for Ray Tune
# ---------------------------------------------------------------------------

def _tune_trainable(config: dict):
    """
    Ray Tune trainable function.

    Внутри — облегчённая версия run_training, которая:
    - На каждой эпохе делает tune.report() с метриками
    - Сохраняет/загружает чекпоинты через Ray Tune API
    - Поддерживает resume через tune.get_checkpoint()
    """
    import torch
    import torch.cuda.amp as amp
    from torch.amp import autocast as _amp_autocast
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

    from ray import tune as ray_tune
    from ray.tune import Checkpoint

    # Импорты из основного проекта
    from manuscript.recognizers._trba.data.dataset import (
        OCRDatasetAttn,
        MultiDataset,
        ProportionalBatchSampler,
    )
    from manuscript.recognizers._trba.data.transforms import (
        decode_tokens,
        get_train_transform,
        get_val_transform,
        load_charset,
        DEFAULT_AUG_PARAMS,
    )
    from manuscript.recognizers._trba.training.metrics import compute_cer, compute_wer
    from manuscript.recognizers._trba.model.model import TRBAModel
    from manuscript.recognizers._trba.training.utils import (
        set_seed,
        load_pretrained_weights,
    )
    from manuscript.recognizers._trba.training.train import (
        get_ctc_weight_for_epoch,
        _autocast,
    )

    # ── Extract config ────────────────────────────────────────────────────────
    # Ray Tune передаёт объединённый config: base_config + search_space overrides.
    # base_config содержит все "фиксированные" параметры (train_csvs, charset_path, ...)

    seed = config.get("seed", 42)
    set_seed(seed)

    # Ray Tune контролирует CUDA_VISIBLE_DEVICES — всегда cuda:0 если GPU доступна
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Пути данных (фиксированные, не тюнятся)
    train_csvs = config["train_csvs"]
    train_roots = config["train_roots"]
    val_csvs = config.get("val_csvs", None)
    val_roots = config.get("val_roots", None)
    charset_path = config["charset_path"]
    encoding = config.get("encoding", "utf-8")

    # Модель (фиксированные)
    img_h = config.get("img_h", 64)
    img_w = config.get("img_w", 256)
    max_len = config.get("max_len", 25)
    hidden_size = config.get("hidden_size", 256)
    num_encoder_layers = config.get("num_encoder_layers", 2)
    cnn_in_channels = config.get("cnn_in_channels", 3)
    cnn_out_channels = config.get("cnn_out_channels", 512)
    cnn_backbone = config.get("cnn_backbone", "seresnet31")

    # Гиперпараметры (тюнятся Ray Tune)
    lr = config.get("lr", 1e-3)
    batch_size = config.get("batch_size", 32)
    weight_decay = config.get("weight_decay", 0.0)
    optimizer_name = config.get("optimizer", "AdamW")
    scheduler_name = config.get("scheduler", "CosineAnnealingLR")  # CosineAnnealingLR безопаснее с ASHA
    momentum = config.get("momentum", 0.9)
    max_grad_norm = config.get("max_grad_norm", 5.0)
    epochs = config.get("epochs", 20)
    val_size = config.get("val_size", 3000)
    num_workers = config.get("num_workers", 0)

    # CTC
    ctc_weight_initial = config.get("ctc_weight", 0.3)
    ctc_weight_decay_epochs = config.get("ctc_weight_decay_epochs", 15)
    ctc_weight_min = config.get("ctc_weight_min", 0.03)

    # Pretrained weights
    pretrain_weights = config.get("pretrain_weights", None)

    # Text mosaic
    text_mosaic_prob = float(config.get("text_mosaic_prob", 0.03))
    text_mosaic_n_words = int(config.get("text_mosaic_n_words", 2))
    text_mosaic_gap_ratio = config.get("text_mosaic_gap_ratio", None)
    if text_mosaic_gap_ratio is not None:
        text_mosaic_gap_ratio = float(text_mosaic_gap_ratio)

    # Aug params
    aug_params = {}
    for k, v in DEFAULT_AUG_PARAMS.items():
        aug_params[k] = config.get(k, v)

    # ── Charset ───────────────────────────────────────────────────────────────
    itos, stoi = load_charset(charset_path)
    PAD = stoi["<PAD>"]
    SOS = stoi["<SOS>"]
    EOS = stoi["<EOS>"]
    BLANK = stoi.get("<BLANK>", None)
    num_classes = len(itos)

    # ── Model ─────────────────────────────────────────────────────────────────
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
        use_ctc_head=True,
    ).to(device)

    # Load pretrained weights (каждый trial стартует с одних и тех же весов)
    if pretrain_weights:
        _src = pretrain_weights
        if isinstance(_src, str) and _src.lower() == "default":
            _src = (
                "https://github.com/konstantinkozhin/manuscript-ocr/releases/download/"
                "v0.1.0/trba_lite_g1.pth"
            )
        if isinstance(_src, str) and _src.lower() not in ("none", ""):
            load_pretrained_weights(model, src=_src, map_location=str(device))

    # ── Freeze policies (фиксированные) ──────────────────────────────────────
    def _freeze_module(m):
        for p in m.parameters():
            p.requires_grad = False

    freeze_cnn = config.get("freeze_cnn", "none")
    freeze_enc = config.get("freeze_enc_rnn", "none")
    freeze_attn = config.get("freeze_attention", "none")

    if freeze_cnn and str(freeze_cnn).lower() in ("full", "all", "freeze"):
        _freeze_module(model.cnn)
    if freeze_enc and str(freeze_enc).lower() in ("full", "all", "freeze"):
        _freeze_module(model.enc_rnn)
    if freeze_attn and str(freeze_attn).lower() in ("full", "all", "freeze"):
        _freeze_module(model.attn)

    # ── Criterion ─────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "Adam":
        optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    scaler = amp.GradScaler(enabled=torch.cuda.is_available())

    # ── Transforms ────────────────────────────────────────────────────────────
    merged_cfg = dict(aug_params)
    merged_cfg.update(config)
    train_transform = get_train_transform(merged_cfg, img_h=img_h, img_w=img_w)
    val_transform = get_val_transform(img_h, img_w)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_sets = []
    val_sets = []

    if val_csvs and val_roots:
        for i, (tc, tr) in enumerate(zip(train_csvs, train_roots)):
            has_sep = (
                i < len(val_csvs) and i < len(val_roots)
                and val_csvs[i] is not None and val_roots[i] is not None
            )
            if has_sep:
                train_ds = OCRDatasetAttn(
                    tc, tr, stoi,
                    img_height=img_h, img_max_width=img_w,
                    transform=train_transform, encoding=encoding,
                    max_len=max_len, strict_max_len=True,
                    text_mosaic_prob=text_mosaic_prob,
                    text_mosaic_n_words=text_mosaic_n_words,
                    text_mosaic_gap_ratio=text_mosaic_gap_ratio,
                )
                val_ds = OCRDatasetAttn(
                    val_csvs[i], val_roots[i], stoi,
                    img_height=img_h, img_max_width=img_w,
                    transform=val_transform, encoding=encoding,
                    max_len=max_len, strict_max_len=True,
                )
                train_sets.append(train_ds)
                val_sets.append(val_ds)
            else:
                full_ds = OCRDatasetAttn(
                    tc, tr, stoi,
                    img_height=img_h, img_max_width=img_w,
                    transform=None, encoding=encoding,
                    max_len=max_len, strict_max_len=True,
                )
                n_val = min(val_size, len(full_ds))
                n_train = len(full_ds) - n_val
                if n_train <= 0:
                    raise ValueError(f"Dataset {tc}: only {len(full_ds)} samples < val_size={n_val}")
                t_split, v_split = random_split(full_ds, [n_train, n_val])

                train_base = OCRDatasetAttn(
                    tc, tr, stoi,
                    img_height=img_h, img_max_width=img_w,
                    transform=train_transform, encoding=encoding,
                    max_len=max_len, strict_max_len=True,
                    text_mosaic_prob=text_mosaic_prob,
                    text_mosaic_n_words=text_mosaic_n_words,
                    text_mosaic_gap_ratio=text_mosaic_gap_ratio,
                )
                val_base = OCRDatasetAttn(
                    tc, tr, stoi,
                    img_height=img_h, img_max_width=img_w,
                    transform=val_transform, encoding=encoding,
                    max_len=max_len, strict_max_len=True,
                )
                train_sets.append(Subset(train_base, t_split.indices))
                val_sets.append(Subset(val_base, v_split.indices))
    else:
        from manuscript.recognizers._trba.training.train import split_train_val
        train_sets, val_sets = split_train_val(
            train_csvs, train_roots, stoi, img_h, img_w,
            train_transform, val_transform,
            encoding=encoding, val_size=val_size,
        )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    pin_memory = use_cuda
    collate_fn = OCRDatasetAttn.make_collate_attn(stoi, max_len=max_len, drop_blank=True)

    train_proportions = config.get("train_proportions", None)
    if train_proportions is not None:
        total = sum(train_proportions)
        proportions = [p / total for p in train_proportions]
        train_dataset = MultiDataset(train_sets)
        batch_sampler = ProportionalBatchSampler(train_sets, batch_size, proportions)
        train_loader = DataLoader(
            train_dataset, batch_sampler=batch_sampler,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            ConcatDataset(train_sets),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
        )

    val_loader = DataLoader(
        ConcatDataset(val_sets),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    # Для ASHA безопаснее CosineAnnealingLR (OneCycleLR плохо совмещается с ранней остановкой)
    scheduler = None
    if scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7)
    elif scheduler_name == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr * 2,
            steps_per_epoch=len(train_loader), epochs=epochs,
            pct_start=0.1, anneal_strategy="cos", div_factor=25,
        )
    # None / "" → no scheduler

    # ── Resume from Ray Tune checkpoint ───────────────────────────────────────
    start_epoch = 0
    loaded_checkpoint = ray_tune.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as ckpt_dir:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
            ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt_data["model_state"])
            optimizer.load_state_dict(ckpt_data["optimizer_state"])
            if scheduler is not None and ckpt_data.get("scheduler_state"):
                try:
                    scheduler.load_state_dict(ckpt_data["scheduler_state"])
                except Exception:
                    pass  # scheduler mismatch is OK during PBT mutation
            if ckpt_data.get("scaler_state"):
                scaler.load_state_dict(ckpt_data["scaler_state"])
            start_epoch = ckpt_data.get("epoch", 0) + 1

            # PBT может мутировать гиперпараметры — обновляем optimizer
            for pg in optimizer.param_groups:
                pg["lr"] = lr
                pg["weight_decay"] = weight_decay

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        # CTC weight decay
        ctc_weight = get_ctc_weight_for_epoch(
            epoch + 1,  # 1-indexed internally
            initial_weight=ctc_weight_initial,
            decay_epochs=ctc_weight_decay_epochs,
            min_weight=ctc_weight_min,
        )

        # --- Train ---
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        for imgs, text_in, target_y, lengths in train_loader:
            imgs = imgs.to(device, non_blocking=pin_memory)
            text_in = text_in.to(device, non_blocking=pin_memory)
            target_y = target_y.to(device, non_blocking=pin_memory)
            lengths = lengths.to(device, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)
            with _autocast():
                result = model(imgs, text=text_in, is_train=True, batch_max_length=max_len)

                attn_logits = result["attention_logits"]
                ctc_logits = result["ctc_logits"]

                attn_loss = criterion(
                    attn_logits.reshape(-1, attn_logits.size(-1)),
                    target_y.reshape(-1),
                )
                ctc_loss = model.compute_ctc_loss(ctc_logits, target_y, lengths)
                loss = (1.0 - ctc_weight) * attn_loss + ctc_weight * ctc_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += float(loss.item())
            n_batches += 1

            # OneCycleLR step per batch
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

        avg_train_loss = total_train_loss / max(1, n_batches)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        refs, hyps = [], []

        with torch.no_grad():
            for imgs, text_in, target_y, lengths in val_loader:
                imgs = imgs.to(device, non_blocking=pin_memory)
                text_in = text_in.to(device, non_blocking=pin_memory)
                target_y = target_y.to(device, non_blocking=pin_memory)
                lengths = lengths.to(device, non_blocking=pin_memory)

                with _autocast():
                    result = model(imgs, text=text_in, is_train=True, batch_max_length=max_len)
                    attn_loss = criterion(
                        result["attention_logits"].reshape(-1, result["attention_logits"].size(-1)),
                        target_y.reshape(-1),
                    )
                    ctc_loss = model.compute_ctc_loss(result["ctc_logits"], target_y, lengths)
                    batch_loss = (1.0 - ctc_weight) * attn_loss + ctc_weight * ctc_loss

                val_loss += float(batch_loss.item())

                # Decode (attention)
                result_inf = model(imgs, is_train=False, batch_max_length=max_len)
                pred_ids = result_inf["attention_preds"].cpu()
                tgt_ids = target_y.cpu()

                for t_row, p_row in zip(tgt_ids, pred_ids):
                    refs.append(decode_tokens(t_row, itos, PAD, EOS, BLANK))
                    hyps.append(decode_tokens(p_row, itos, PAD, EOS, BLANK))

        n_val = len(refs)
        avg_val_loss = val_loss / max(1, len(val_loader))
        accuracy = sum(r == h for r, h in zip(refs, hyps)) / max(1, n_val)
        cer = compute_cer(refs, hyps) if refs else 0.0
        wer = compute_wer(refs, hyps) if refs else 0.0

        # --- Scheduler step (epoch-level) ---
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                pass  # already stepped per batch
            elif isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # --- Save checkpoint for Ray Tune ---
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "checkpoint.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "scaler_state": scaler.state_dict(),
            }, ckpt_path)
            checkpoint = Checkpoint.from_directory(tmp_dir)

            # --- Report to Ray Tune ---
            ray_tune.report(
                {
                    "loss": avg_val_loss,
                    "accuracy": accuracy,
                    "cer": cer,
                    "wer": wer,
                    "train_loss": avg_train_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "ctc_weight": ctc_weight,
                    "epoch": epoch,
                },
                checkpoint=checkpoint,
            )

        if use_cuda:
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_tune(
    base_config: Union[str, Dict[str, Any]],
    mode: str = "asha",
    num_samples: int = 8,
    max_epochs: Optional[int] = None,
    gpus_per_trial: float = 1.0,
    cpus_per_trial: int = 2,
    search_space: Optional[Dict[str, Any]] = None,
    storage_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
    resume: bool = False,
) -> Any:
    """
    Запуск Ray Tune hyperparameter search для OCR модели.

    Parameters
    ----------
    base_config : str or dict
        Путь к config.json или dict с базовыми параметрами обучения.
        Должен содержать: train_csvs, train_roots, charset_path и т.д.
    mode : str
        "asha" — быстрый поиск с ранней остановкой.
        "pbt"  — Population Based Training (мутация гиперпараметров на лету).
    num_samples : int
        Количество trial-ов (комбинаций гиперпараметров).
    max_epochs : int, optional
        Макс. количество эпох на trial. По умолчанию берётся из config["epochs"].
    gpus_per_trial : float
        GPU на каждый trial. 0.5 = 2 trial-а на 1 GPU.
    cpus_per_trial : int
        CPU на каждый trial.
    search_space : dict, optional
        Кастомное пространство поиска. Если None — используется default.
    storage_path : str, optional
        Путь для сохранения результатов. Default: ~/ray_results.
    experiment_name : str, optional
        Имя эксперимента. Default: "ocr_tune_{mode}".
    resume : bool
        Продолжить прерванный эксперимент.

    Returns
    -------
    ray.tune.ResultGrid — результаты всех trial-ов.
    """
    if not _check_ray_installed():
        raise ImportError(
            "Ray Tune не установлен. Установите: pip install 'ray[tune]'\n"
            "Или: pip install ray[tune] torch  (если ray ещё не установлен)"
        )

    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

    # ── Load base config ──────────────────────────────────────────────────────
    if isinstance(base_config, str):
        with open(base_config, "r", encoding="utf-8") as f:
            base_cfg = json.load(f)
    elif isinstance(base_config, dict):
        base_cfg = deepcopy(base_config)
    else:
        raise TypeError("base_config must be a path to JSON or a dict")

    if max_epochs is None:
        max_epochs = base_cfg.get("epochs", 20)
    base_cfg["epochs"] = max_epochs

    # ASHA лучше работает с CosineAnnealingLR (OneCycleLR плохо совместим с ранней остановкой)
    if mode == "asha" and base_cfg.get("scheduler") == "OneCycleLR":
        print("[tune] Switching scheduler from OneCycleLR to CosineAnnealingLR (better ASHA compat)")
        base_cfg["scheduler"] = "CosineAnnealingLR"

    # Убираем resume_from из base_config — в tune каждый trial стартует чисто
    # (pretrained weights загружаются через pretrain_weights)
    base_cfg.pop("resume_from", None)

    # ── Search space ──────────────────────────────────────────────────────────
    if search_space is None:
        if mode == "asha":
            search_space = get_default_search_space_asha()
        else:
            search_space = {}  # PBT uses hyperparam_mutations instead

    # Merge: base_cfg + search_space (search_space overrides)
    param_space = {**base_cfg, **search_space}

    # ── Experiment name ───────────────────────────────────────────────────────
    if experiment_name is None:
        experiment_name = f"ocr_tune_{mode}"

    # ── Init Ray ──────────────────────────────────────────────────────────────
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # ── Build scheduler ───────────────────────────────────────────────────────
    if mode == "asha":
        tune_scheduler = ASHAScheduler(
            time_attr="training_iteration",
            max_t=max_epochs,
            grace_period=max(1, max_epochs // 5),  # Минимум 20% эпох до остановки
            reduction_factor=2,
        )
    elif mode == "pbt":
        # Mutations for PBT
        pbt_mutations = get_default_search_space_pbt()
        if search_space:
            pbt_mutations.update(search_space)

        tune_scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=1,  # Мутация каждую эпоху
            hyperparam_mutations=pbt_mutations,
        )
    else:
        raise ValueError(f"Unknown tune mode: {mode}. Use 'asha' or 'pbt'.")

    # ── Run config ────────────────────────────────────────────────────────────
    run_config_kwargs = {"name": experiment_name}
    if storage_path:
        run_config_kwargs["storage_path"] = storage_path

    run_cfg = tune.RunConfig(**run_config_kwargs)

    # ── Tuner ─────────────────────────────────────────────────────────────────
    if resume:
        restore_path = storage_path or os.path.join(os.path.expanduser("~/ray_results"), experiment_name)
        print(f"[tune] Restoring experiment from: {restore_path}")
        tuner = tune.Tuner.restore(
            restore_path,
            trainable=tune.with_resources(
                tune.with_parameters(_tune_trainable),
                resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
            ),
            resume_errored=True,
            resume_unfinished=True,
        )
    else:
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(_tune_trainable),
                resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
            ),
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                scheduler=tune_scheduler,
                num_samples=num_samples,
            ),
            param_space=param_space,
            run_config=run_cfg,
        )

    # ── Fit ───────────────────────────────────────────────────────────────────
    results = tuner.fit()

    # ── Print summary ─────────────────────────────────────────────────────────
    best = results.get_best_result("loss", "min")
    print("\n" + "=" * 70)
    print("RAY TUNE — BEST TRIAL")
    print("=" * 70)
    print(f"  loss:     {best.metrics['loss']:.6f}")
    print(f"  accuracy: {best.metrics['accuracy']:.4f}")
    print(f"  CER:      {best.metrics['cer']:.4f}")
    print(f"  WER:      {best.metrics['wer']:.4f}")
    print(f"\n  Config:")
    for k in ("lr", "batch_size", "weight_decay", "optimizer", "ctc_weight", "max_grad_norm"):
        if k in best.config:
            v = best.config[k]
            if isinstance(v, float):
                print(f"    {k}: {v:.6g}")
            else:
                print(f"    {k}: {v}")
    print("=" * 70)

    return results


def extract_best_model(
    results: Any,
    output_dir: str,
    base_config: Union[str, Dict[str, Any]],
) -> str:
    """
    Извлекает лучшую модель из результатов Ray Tune и сохраняет в output_dir.

    Parameters
    ----------
    results : ray.tune.ResultGrid
        Результаты от run_tune().
    output_dir : str
        Куда сохранить лучшие веса и конфиг.
    base_config : str or dict
        Базовый конфиг (для пересборки полного конфига).

    Returns
    -------
    str — путь к сохранённым весам.
    """
    import torch
    from ray.tune import Checkpoint

    best = results.get_best_result("loss", "min")

    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    ckpt_dir = best.checkpoint.to_directory()
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Save weights
    weights_path = os.path.join(output_dir, "best_tune_weights.pth")
    torch.save(ckpt_data["model_state"], weights_path)

    # Save merged config
    if isinstance(base_config, str):
        with open(base_config, "r", encoding="utf-8") as f:
            base_cfg = json.load(f)
    else:
        base_cfg = deepcopy(base_config)

    # Merge best hyperparameters into config
    merged = {**base_cfg}
    for k in ("lr", "batch_size", "weight_decay", "optimizer", "ctc_weight",
              "ctc_weight_decay_epochs", "max_grad_norm", "scheduler"):
        if k in best.config:
            merged[k] = best.config[k]

    merged["tune_best_metrics"] = {
        "loss": best.metrics["loss"],
        "accuracy": best.metrics["accuracy"],
        "cer": best.metrics["cer"],
        "wer": best.metrics["wer"],
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)

    print(f"\n[tune] Best model saved to: {output_dir}")
    print(f"  weights: {weights_path}")
    print(f"  config:  {config_path}")
    print(f"\nTo continue training with found hyperparameters:")
    print(f'  from manuscript.recognizers._trba.training.train import run_training, Config')
    print(f'  run_training(Config("{config_path}"))')

    return weights_path


# ---------------------------------------------------------------------------
# Convenience: unified entry point (тумблер в конфиге)
# ---------------------------------------------------------------------------

def maybe_run_tune(cfg_source: Union[str, Dict[str, Any]], device: str = "cuda"):
    """
    Единая точка входа: проверяет use_ray_tune в конфиге.

    Если use_ray_tune=true  → запускает run_tune()
    Если use_ray_tune=false → запускает обычный run_training()

    Parameters
    ----------
    cfg_source : str or dict
        Путь к JSON конфигу или dict.
    device : str
        "cuda" или "cpu".

    Returns
    -------
    Результаты (dict из run_training или ResultGrid из run_tune).
    """
    if isinstance(cfg_source, str):
        with open(cfg_source, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
    else:
        cfg_dict = dict(cfg_source)

    use_tune = cfg_dict.get("use_ray_tune", False)

    if use_tune:
        if not _check_ray_installed():
            print(
                "[WARNING] use_ray_tune=true, but ray not installed. "
                "Falling back to normal training.\n"
                "Install: pip install 'ray[tune]'"
            )
            from .train import run_training, Config
            return run_training(Config(cfg_dict), device=device)

        mode = cfg_dict.get("tune_mode", "asha")
        num_samples = cfg_dict.get("tune_num_samples", 8)
        gpus_per_trial = cfg_dict.get("tune_gpus_per_trial", 1.0)
        cpus_per_trial = cfg_dict.get("tune_cpus_per_trial", 2)
        max_epochs = cfg_dict.get("tune_max_epochs", cfg_dict.get("epochs", 20))
        storage_path = cfg_dict.get("tune_storage_path", None)
        experiment_name = cfg_dict.get("tune_experiment_name", None)
        search_space_override = cfg_dict.get("tune_search_space", None)

        return run_tune(
            base_config=cfg_dict,
            mode=mode,
            num_samples=num_samples,
            max_epochs=max_epochs,
            gpus_per_trial=gpus_per_trial,
            cpus_per_trial=cpus_per_trial,
            search_space=search_space_override,
            storage_path=storage_path,
            experiment_name=experiment_name,
        )
    else:
        from .train import run_training, Config
        return run_training(Config(cfg_dict), device=device)
