import random

import torch
import torch.nn.functional as F

from ..model.model import RCNN


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
    path, model, optimizer=None, scheduler=None, scaler=None, map_location="auto"
):
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_crnn(
    checkpoint_path: str,
    itos: list[str] | None = None,
    stoi: dict | None = None,
    hidden_size: int = 256,
    sos_token: str = "<SOS>",
    eos_token: str = "<EOS>",
    pad_token: str = "<PAD>",
    blank_token: str | None = "<BLANK>",
    device: torch.device | None = None,
    eval_mode: bool = True,
) -> RCNN:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, dict) and "model_state" in state:
        ckpt = state
        model_state = ckpt["model_state"]
        if itos is None:
            itos = ckpt.get("itos")
        if stoi is None:
            stoi = ckpt.get("stoi")
    else:
        model_state = state

    assert (
        itos is not None and stoi is not None
    ), "Нужны itos/stoi: передай явно или используй полный чекпойнт, внутри которого они сохранены."

    num_classes = len(itos)
    PAD = stoi[pad_token]
    SOS = stoi[sos_token]
    EOS = stoi[eos_token]
    BLANK = stoi.get(blank_token, None) if blank_token is not None else None

    model = RCNN(
        num_classes=num_classes,
        hidden_size=hidden_size,
        sos_id=SOS,
        eos_id=EOS,
        pad_id=PAD,
        blank_id=BLANK,
    ).to(device)

    model.load_state_dict(model_state, strict=True)
    if eval_mode:
        model.eval()
    return model


def ctc_greedy_decoder(logits: torch.Tensor, alphabet: str, blank: int = 0):
    """
    Greedy decoder для CTC.
    Args:
        logits: (T, B, C) или (B, T, C) — выход модели
        alphabet: строка символов
        blank: индекс blank (обычно 0)
    Returns:
        список предсказанных строк, список индексов
    """
    if logits.dim() == 3 and logits.shape[0] < logits.shape[1]:
        logits = logits.permute(1, 0, 2)

    B, T, C = logits.shape
    preds = logits.argmax(dim=2)

    texts, seqs = [], []
    for b in range(B):
        prev = blank
        seq, chars = [], []
        for t in range(T):
            p = preds[b, t].item()
            if p != blank and p != prev:
                seq.append(p)
                chars.append(alphabet[p - 1])
            prev = p
        texts.append("".join(chars))
        seqs.append(seq)
    return texts, seqs


def decode(ctc_out, alphabet: str, method: str = "greedy"):
    if isinstance(ctc_out, tuple):
        ctc_out = ctc_out[0]

    log_probs = F.log_softmax(ctc_out, dim=-1)

    if method == "greedy":
        return ctc_greedy_decoder(log_probs, alphabet)
    else:
        raise ValueError(f"Unsupported decode method: {method}")
