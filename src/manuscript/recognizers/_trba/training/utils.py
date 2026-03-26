import random
from typing import Dict, Any, Tuple, Optional

import torch


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
    ckpt_obj = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
        model_state = ckpt_obj["model_state"]
        metadata = ckpt_obj
    else:
        model_state = ckpt_obj
        metadata = {"model_state": model_state}

    if model is not None:
        current_state = model.state_dict()
        filtered_state = {}
        skipped_keys = []
        shape_mismatches = []

        for k, v in model_state.items():
            if k in current_state:
                if current_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    shape_mismatches.append(
                        f"{k}: checkpoint {v.shape} vs model {current_state[k].shape}"
                    )
            else:
                skipped_keys.append(k)

        missing_keys = set(current_state.keys()) - set(filtered_state.keys())

        result = model.load_state_dict(filtered_state, strict=False)

        if shape_mismatches or skipped_keys or missing_keys:
            if shape_mismatches:
                print(f"⚠️  Shape mismatches (skipped {len(shape_mismatches)} layers):")
                for msg in shape_mismatches[:5]:
                    print(f"   - {msg}")
                if len(shape_mismatches) > 5:
                    print(f"   ... and {len(shape_mismatches) - 5} more")

            if missing_keys:
                print(
                    f"ℹ️  Missing keys in checkpoint (initialized randomly): {len(missing_keys)}"
                )
                for key in list(missing_keys)[:3]:
                    print(f"   - {key}")
                if len(missing_keys) > 3:
                    print(f"   ... and {len(missing_keys) - 3} more")

        loaded_params = sum(p.numel() for p in filtered_state.values())
        total_params = sum(p.numel() for p in current_state.values())
        print(
            f" Loaded {len(filtered_state)}/{len(current_state)} layers "
            f"({loaded_params/1e6:.1f}M/{total_params/1e6:.1f}M parameters)"
        )

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


# -------------------------
# Pretrained weights helpers
# -------------------------


def _is_url(path_or_url: str) -> bool:
    return isinstance(path_or_url, str) and (
        path_or_url.startswith("http://") or path_or_url.startswith("https://")
    )


def _extract_model_state(obj: Any) -> Dict[str, torch.Tensor]:
    # Accept multiple common layouts from different toolchains
    if isinstance(obj, dict):
        for key in ("model_state", "state_dict", "model"):  # common variants
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        # If dict itself looks like a state_dict
        if all(isinstance(k, str) for k in obj.keys()):
            return obj  # assume already a state_dict
    # Fallback: return as-is if it quacks like a state_dict
    return obj


def _maybe_strip_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(state.keys())
    if not keys:
        return state
    prefixes = ["module.", "model."]
    for pref in prefixes:
        if all(k.startswith(pref) for k in keys):
            return {k[len(pref) :]: v for k, v in state.items()}
    return state


def build_compatible_state_dict(
    model: torch.nn.Module, raw_state: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Filter and adapt a raw state_dict to only include keys present in `model`
    and with matching shapes. Returns (filtered_state, stats).
    """
    model_state = model.state_dict()
    raw_state = _maybe_strip_prefix(raw_state)

    loaded, skipped_missing, skipped_shape = {}, [], []
    for k, v in raw_state.items():
        if k not in model_state:
            skipped_missing.append(k)
            continue
        if model_state[k].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        loaded[k] = v

    stats = {
        "num_in_ckpt": len(raw_state),
        "num_loaded": len(loaded),
        "num_missing": len(skipped_missing),
        "num_shape_mismatch": len(skipped_shape),
        "missing_keys": skipped_missing,
        "shape_mismatch": skipped_shape,
    }
    return loaded, stats


def load_pretrained_weights(
    model: torch.nn.Module,
    src: str,
    map_location: Optional[str] = "auto",
    logger: Optional[Any] = None,
    legacy_migration: bool = True,
) -> Dict[str, Any]:
    """
    Load pretrained weights into `model` from local path or URL.
    Robust to different checkpoint layouts and tensor prefixes.
    Only matching keys with identical shapes are loaded.

    Args:
        model: Target model
        src: Path or URL to weights
        map_location: Device for loading
        logger: Logger instance
        legacy_migration: If True, automatically migrates legacy TRBA weights (attn.* -> attention_decoder.*)

    Returns stats dict.
    """
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

    # Load raw object
    try:
        if _is_url(src):
            # Use torch.hub helper for URLs (handles caching and TLS)
            raw_obj = torch.hub.load_state_dict_from_url(
                src, map_location=map_location, progress=True
            )
        else:
            raw_obj = torch.load(src, map_location=map_location)
    except Exception as e:
        if logger:
            logger.warning(f"Failed to load pretrained from {src}: {e}")
        else:
            print(f"[pretrain] Failed to load {src}: {e}")
        return {"ok": False, "error": str(e), "src": src}

    raw_state = _extract_model_state(raw_obj)

    # Legacy migration: attn.* -> attention_decoder.*
    if legacy_migration and any(k.startswith("attn.") for k in raw_state.keys()):
        if logger:
            logger.info(
                "Detected legacy TRBA weights (attn.*), applying migration to attention_decoder.*"
            )
        else:
            print("[pretrain] Migrating legacy TRBA weights...")

        migrated_state = {}
        for k, v in raw_state.items():
            if k.startswith("attn."):
                new_key = k.replace("attn.", "attention_decoder.", 1)
                migrated_state[new_key] = v
            else:
                migrated_state[k] = v
        raw_state = migrated_state

    filt_state, stats = build_compatible_state_dict(model, raw_state)

    missing_after = set(model.state_dict().keys()) - set(filt_state.keys())

    missing_after_list = list(sorted(missing_after))
    try:
        model.load_state_dict(filt_state, strict=False)
        ok = True
    except Exception as e:
        ok = False
        if logger:
            logger.warning(f"Failed to load filtered weights: {e}")
        else:
            print(f"[pretrain] Failed to load filtered weights: {e}")

    # Summarize
    if logger:
        logger.info(
            "Pretrain load summary: "
            f"loaded={stats['num_loaded']}/{stats['num_in_ckpt']} keys; "
            f"missing={stats['num_missing']}; shape_mismatch={stats['num_shape_mismatch']}"
        )
    else:
        print(
            f"[pretrain] loaded={stats['num_loaded']}/{stats['num_in_ckpt']} | "
            f"missing={stats['num_missing']} | shape_mismatch={stats['num_shape_mismatch']}"
        )

    return {
        "ok": ok,
        "src": src,
        **stats,
        "missing_after_load": missing_after_list,
    }
