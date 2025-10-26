import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit

from .vis import visualize_decoding


def save_checkpoint(
    path,
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
    config,
    log_dir,
):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "itos": itos,
        "stoi": stoi,
        "config": config,
        "log_dir": log_dir,
    }
    torch.save(ckpt, path)


def save_weights(path, model):
    torch.save(model.state_dict(), path)


def load_checkpoint(
    path,
    model=None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="auto",
    strict: bool = True,
):
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_obj = torch.load(path, map_location=map_location)

    if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
        model_state = ckpt_obj["model_state"]
        metadata = ckpt_obj
    else:
        model_state = ckpt_obj
        metadata = {"model_state": model_state}

    if model is not None:
        model.load_state_dict(model_state, strict=strict)

    if optimizer is not None and metadata.get("optimizer_state") is not None:
        optimizer.load_state_dict(metadata["optimizer_state"])
    if scheduler is not None and metadata.get("scheduler_state") is not None:
        scheduler.load_state_dict(metadata["scheduler_state"])
    if scaler is not None and metadata.get("scaler_state") is not None:
        scaler.load_state_dict(metadata["scaler_state"])
    return metadata


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
