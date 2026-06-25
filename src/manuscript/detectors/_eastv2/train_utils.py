import copy
import json
import os
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch_optimizer as toptim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .._east.sam import SAMSolver
from .._east.train_utils import (
    _build_scheduler,
    _check_architecture_compatibility,
    _extract_model_state,
    _is_full_state_checkpoint,
    dice_coefficient,
)
from .loss import EASTV2Loss


def _custom_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, {
        "score_map": torch.stack([t["score_map"] for t in targets], dim=0),
        "boundary_map": torch.stack([t["boundary_map"] for t in targets], dim=0),
        "center_map": torch.stack([t["center_map"] for t in targets], dim=0),
        "instance_map": torch.stack([t["instance_map"] for t in targets], dim=0),
        "polygons": [t.get("polygons", []) for t in targets],
    }


def _resize_outputs(out: Dict[str, torch.Tensor], target_hw: Tuple[int, int]):
    return {
        "score": F.interpolate(
            out["score"], size=target_hw, mode="bilinear", align_corners=False
        ),
        "boundary": F.interpolate(
            out["boundary"], size=target_hw, mode="bilinear", align_corners=False
        ),
        "center": F.interpolate(
            out["center"], size=target_hw, mode="bilinear", align_corners=False
        ),
    }


def _loss_from_batch(criterion, target, pred):
    return criterion(
        target["score_map"],
        pred["score"],
        target["boundary_map"],
        pred["boundary"],
        target["center_map"],
        pred["center"],
    )


def _run_training(
    experiment_dir: str,
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    device: torch.device,
    num_epochs: int,
    batch_size: int,
    accumulation_steps: int,
    lr: float,
    lr_scheduler: str,
    lr_scheduler_params: Optional[Dict[str, Any]],
    grad_clip: float,
    early_stop: int,
    use_sam: bool,
    sam_type: str,
    use_lookahead: bool,
    use_ema: bool,
    use_multiscale: bool,
    use_ohem: bool,
    ohem_ratio: float,
    use_focal_geo: bool,
    focal_gamma: float,
    val_interval: int = 1,
    num_workers: int = 0,
    *,
    backbone_name: Optional[str] = None,
    target_size: Optional[int] = None,
    pretrained_backbone: Optional[bool] = None,
    val_datasets: Optional[Sequence[torch.utils.data.Dataset]] = None,
    val_dataset_names: Optional[Sequence[str]] = None,
    resume: bool = False,
    resume_state_path: Optional[str] = None,
    score_map_shrink_ratio: Optional[float] = None,
    augmentation_config: Optional[Dict[str, Any]] = None,
    log_collage: bool = True,
):
    del use_ohem, ohem_ratio, use_focal_geo, focal_gamma, log_collage

    experiment_dir = os.path.abspath(os.fspath(experiment_dir))
    log_dir = os.path.join(experiment_dir, "logs")
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    config = {
        "detector": "EASTV2",
        "backbone_name": backbone_name,
        "pretrained_backbone": pretrained_backbone,
        "target_size": target_size,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "lr": lr,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_params": lr_scheduler_params,
        "grad_clip": grad_clip,
        "early_stop": early_stop,
        "use_sam": use_sam,
        "sam_type": sam_type if use_sam else None,
        "use_lookahead": use_lookahead,
        "use_ema": use_ema,
        "use_multiscale": use_multiscale,
        "val_interval": val_interval,
        "scheduler": lr_scheduler,
        "optimizer": "SAM" if use_sam else ("Lookahead(RAdam)" if use_lookahead else "RAdam"),
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "score_map_shrink_ratio": score_map_shrink_ratio,
        "augmentation_config": augmentation_config,
    }
    with open(os.path.join(experiment_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    if val_interval < 1:
        raise ValueError("val_interval must be >= 1")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_custom_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    if val_datasets:
        if val_dataset_names is None:
            val_dataset_names = [f"val_{idx}" for idx in range(len(val_datasets))]
        elif len(val_dataset_names) != len(val_datasets):
            raise ValueError("val_dataset_names length must match val_datasets.")
        val_eval_loaders = [
            (
                name,
                DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=_custom_collate_fn,
                    pin_memory=False,
                    persistent_workers=num_workers > 0,
                ),
            )
            for name, dataset in zip(val_dataset_names, val_datasets)
        ]
    else:
        val_eval_loaders = [
            (
                "val",
                DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=_custom_collate_fn,
                    pin_memory=False,
                    persistent_workers=num_workers > 0,
                ),
            )
        ]

    if use_sam:
        optimizer = SAMSolver(
            model.parameters(),
            torch.optim.SGD,
            rho=0.05,
            lr=lr,
            use_adaptive=(sam_type == "asam"),
        )
    else:
        base_opt = toptim.RAdam(model.parameters(), lr=lr)
        optimizer = (
            toptim.Lookahead(base_opt, k=5, alpha=0.5) if use_lookahead else base_opt
        )

    for attr in (
        "_optimizer_state_dict_pre_hooks",
        "_optimizer_state_dict_post_hooks",
        "_optimizer_load_state_dict_pre_hooks",
        "_optimizer_load_state_dict_post_hooks",
    ):
        if not hasattr(optimizer, attr):
            setattr(optimizer, attr, OrderedDict())

    scheduler, scheduler_step = _build_scheduler(
        optimizer=optimizer,
        scheduler_type=lr_scheduler,
        lr=lr,
        num_epochs=num_epochs,
        params=lr_scheduler_params,
    )

    try:
        scaler = torch.amp.GradScaler(device_type="cuda")

        def autocast_ctx():
            return torch.amp.autocast(device_type="cuda")

    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()

        def autocast_ctx():
            return torch.cuda.amp.autocast()

    criterion = EASTV2Loss()
    ema_model = model if not use_ema else copy.deepcopy(model)
    ema_decay = 0.9999
    if use_ema:
        for param in ema_model.parameters():
            param.requires_grad = False

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_dice = -float("inf")
    patience_loss = 0
    patience_dice = 0

    if resume:
        state_path = (
            os.path.normpath(resume_state_path)
            if resume_state_path is not None
            else os.path.normpath(os.path.join(ckpt_dir, "last_state.pt"))
        )
        if not os.path.isfile(state_path):
            raise FileNotFoundError(
                f"Resume requested, but state file not found: {state_path}"
            )
        checkpoint = torch.load(state_path, map_location=device, weights_only=False)
        if _is_full_state_checkpoint(checkpoint):
            model.load_state_dict(checkpoint["model_state"])
            if use_ema and checkpoint.get("ema_state") is not None:
                ema_model.load_state_dict(checkpoint["ema_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if scheduler is not None and checkpoint.get("scheduler_state") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            scaler_state = checkpoint.get("scaler_state")
            if scaler_state is not None:
                scaler.load_state_dict(scaler_state)
            best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
            best_val_dice = checkpoint.get("best_val_dice", best_val_dice)
            patience_loss = checkpoint.get("patience_loss", checkpoint.get("patience", patience_loss))
            patience_dice = checkpoint.get("patience_dice", patience_dice)
            start_epoch = checkpoint.get("epoch", 0) + 1
        else:
            state_dict = _extract_model_state(checkpoint)
            is_compatible, error_msg = _check_architecture_compatibility(model, state_dict)
            if not is_compatible:
                raise ValueError(f"Architecture mismatch when loading weights: {error_msg}")
            model.load_state_dict(state_dict, strict=False)
            if use_ema:
                ema_model.load_state_dict(state_dict, strict=False)

    writer = SummaryWriter(log_dir, purge_step=start_epoch if resume else None)

    if start_epoch > num_epochs:
        writer.close()
        return ema_model if use_ema else model

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        global_step = (epoch - 1) * len(train_loader)

        for batch_idx, (imgs, tgt) in enumerate(tqdm(train_loader, desc=f"Train {epoch}"), 1):
            imgs = imgs.to(device)
            tgt = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in tgt.items()
            }
            gt_hw = tgt["score_map"].shape[-2:]

            if use_multiscale:
                sf = torch.empty(1).uniform_(0.8, 1.2).item()
                h, w = imgs.shape[-2:]
                nh = max(32, int(h * sf) // 32 * 32)
                nw = max(32, int(w * sf) // 32 * 32)
                imgs_in = F.interpolate(
                    imgs, size=(nh, nw), mode="bilinear", align_corners=False
                )
            else:
                imgs_in = imgs

            if use_sam:

                def closure():
                    out = model(imgs_in)
                    pred = _resize_outputs(out, gt_hw)
                    return _loss_from_batch(criterion, tgt, pred) / accumulation_steps

                loss = optimizer.step(closure) * accumulation_steps
            else:
                with autocast_ctx():
                    out = model(imgs_in)
                    pred = _resize_outputs(out, gt_hw)
                    loss = _loss_from_batch(criterion, tgt, pred)
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()
                if batch_idx % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                loss = loss * accumulation_steps

            if scheduler is not None and scheduler_step == "batch":
                scheduler.step(epoch + imgs.size(0) / len(train_loader))

            train_loss += float(loss.item())
            current_step = global_step + batch_idx
            writer.add_scalar("Loss/Train_Step", float(loss.item()), current_step)
            current_lr = (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else optimizer.param_groups[0]["lr"]
            )
            writer.add_scalar("LearningRate/Step", current_lr, current_step)

            if use_ema:
                with torch.no_grad():
                    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

        avg_train = train_loss / max(len(train_loader), 1)
        writer.add_scalar("Loss/Train", avg_train, epoch)

        if scheduler is not None and scheduler_step == "epoch":
            scheduler.step()
        current_lr = (
            scheduler.get_last_lr()[0]
            if scheduler is not None
            else optimizer.param_groups[0]["lr"]
        )
        writer.add_scalar("LearningRate/Epoch", current_lr, epoch)

        do_validate = (epoch % val_interval) == 0
        should_stop = False

        if do_validate:
            model.eval()
            if use_ema:
                ema_model.eval()
            eval_model = ema_model if use_ema else model
            total_val_loss = 0.0
            total_val_batches = 0
            overall_dice_sum = 0.0
            overall_dice_count = 0
            per_dataset_metrics = {}

            with torch.no_grad():
                for ds_name, ds_loader in val_eval_loaders:
                    dataset_loss = 0.0
                    dataset_batches = 0
                    dataset_dice_sum = 0.0
                    dataset_dice_count = 0

                    for imgs, tgt in ds_loader:
                        imgs = imgs.to(device)
                        tgt = {
                            key: value.to(device) if torch.is_tensor(value) else value
                            for key, value in tgt.items()
                        }
                        out = eval_model(imgs)
                        pred = _resize_outputs(out, tgt["score_map"].shape[-2:])
                        batch_loss = _loss_from_batch(criterion, tgt, pred).item()
                        dataset_loss += batch_loss
                        total_val_loss += batch_loss
                        dataset_batches += 1
                        total_val_batches += 1

                        dice_vals = dice_coefficient(pred["score"], tgt["score_map"])
                        dataset_dice_sum += dice_vals.sum().item()
                        dataset_dice_count += dice_vals.numel()
                        overall_dice_sum += dice_vals.sum().item()
                        overall_dice_count += dice_vals.numel()

                    avg_dataset_loss = dataset_loss / max(dataset_batches, 1)
                    avg_dataset_dice = (
                        dataset_dice_sum / dataset_dice_count
                        if dataset_dice_count > 0
                        else 0.0
                    )
                    per_dataset_metrics[ds_name] = {
                        "loss": avg_dataset_loss,
                        "dice": avg_dataset_dice,
                    }

            avg_val = total_val_loss / max(total_val_batches, 1)
            overall_dice = (
                overall_dice_sum / overall_dice_count if overall_dice_count > 0 else 0.0
            )
            writer.add_scalar("Loss/Val", avg_val, epoch)
            writer.add_scalar("Dice/Val", overall_dice, epoch)

            if scheduler is not None and scheduler_step == "val":
                scheduler.step(avg_val)

            for ds_name, metrics in per_dataset_metrics.items():
                tag = ds_name.replace("\\", "_").replace("/", "_").replace(" ", "_")
                writer.add_scalar(f"Loss/Val/{tag}", metrics["loss"], epoch)
                writer.add_scalar(f"Dice/Val/{tag}", metrics["dice"], epoch)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_loss = 0
                torch.save(
                    (ema_model if use_ema else model).state_dict(),
                    os.path.join(ckpt_dir, "best_loss.pth"),
                )
            else:
                patience_loss += 1

            if overall_dice > best_val_dice:
                best_val_dice = overall_dice
                patience_dice = 0
                torch.save(
                    (ema_model if use_ema else model).state_dict(),
                    os.path.join(ckpt_dir, "best_dice.pth"),
                )
            else:
                patience_dice += 1

            if patience_loss >= early_stop and patience_dice >= early_stop:
                should_stop = True

        torch.save(
            (ema_model if use_ema else model).state_dict(),
            os.path.join(ckpt_dir, "last.pth"),
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_state": ema_model.state_dict() if use_ema else None,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_dice": best_val_dice,
                "patience_loss": patience_loss,
                "patience_dice": patience_dice,
            },
            os.path.join(ckpt_dir, "last_state.pt"),
        )
        if should_stop:
            break

    writer.close()

    try:
        from manuscript.detectors import EASTV2

        onnx_path = os.path.join(ckpt_dir, "best_model.onnx")
        best_weights_path = os.path.join(ckpt_dir, "best_loss.pth")
        if not os.path.isfile(best_weights_path):
            best_weights_path = os.path.join(ckpt_dir, "best_dice.pth")
        EASTV2.export(
            weights_path=best_weights_path,
            output_path=onnx_path,
            backbone_name=backbone_name or "resnet50",
            input_size=target_size if target_size is not None else 1280,
            opset_version=14,
            simplify=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to export EASTV2 ONNX model: {exc}")

    return ema_model if use_ema else model
