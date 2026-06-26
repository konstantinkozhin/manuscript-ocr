import copy
import json
import os
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
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
from ...utils.io import _tensor_to_image
from .loss import EASTV2Loss
from .utils import decode_instance_maps, labels_to_polygons


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
    resized = {
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
    for key in ("score_logits", "boundary_logits", "center_logits"):
        if key in out:
            resized[key] = F.interpolate(
                out[key], size=target_hw, mode="bilinear", align_corners=False
            )
    return resized


def _loss_from_batch(criterion, target, pred):
    return criterion(
        target["score_map"],
        pred["score"],
        target["boundary_map"],
        pred["boundary"],
        target["center_map"],
        pred["center"],
        pred.get("score_logits"),
        pred.get("boundary_logits"),
    )


def _sanitize_tag(name: str) -> str:
    return name.replace("\\", "_").replace("/", "_").replace(" ", "_")


def _to_heatmap(prob_map: np.ndarray, cell_size: int) -> np.ndarray:
    arr = np.asarray(prob_map, dtype=np.float32).squeeze()
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    vis = cv2.applyColorMap((arr * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return cv2.resize(vis, (cell_size, cell_size), interpolation=cv2.INTER_NEAREST)


def _labels_to_vis(labels: np.ndarray, cell_size: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    vis = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label_id in range(1, int(labels.max()) + 1):
        color = np.array(
            [
                (37 * label_id) % 255,
                (97 * label_id) % 255,
                (173 * label_id) % 255,
            ],
            dtype=np.uint8,
        )
        vis[labels == label_id] = color
    return cv2.resize(vis, (cell_size, cell_size), interpolation=cv2.INTER_NEAREST)


def _draw_polygons(
    image: np.ndarray,
    polygons,
    *,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    vis = image.copy()
    for polygon in polygons:
        if isinstance(polygon, tuple) and len(polygon) == 2:
            polygon = polygon[0]
        if torch.is_tensor(polygon):
            polygon = polygon.detach().cpu().numpy()
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 2:
            continue
        cv2.polylines(
            vis,
            [pts.astype(np.int32)],
            isClosed=True,
            color=color,
            thickness=thickness,
        )
    return vis


def create_collage(
    img_tensor: torch.Tensor,
    gt_score_map: torch.Tensor,
    gt_boundary_map: torch.Tensor,
    gt_center_map: torch.Tensor,
    gt_instance_map: torch.Tensor,
    gt_polygons,
    pred_score_map: np.ndarray,
    pred_boundary_map: np.ndarray,
    pred_center_map: np.ndarray,
    pred_instance_map: np.ndarray,
    pred_polygons,
    *,
    cell_size: int = 480,
) -> np.ndarray:
    image = _tensor_to_image(
        img_tensor,
        denormalize={"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    )
    image_cell = cv2.resize(image, (cell_size, cell_size), interpolation=cv2.INTER_AREA)

    gt_image = _draw_polygons(image, gt_polygons, color=(0, 255, 0))
    gt_image = cv2.resize(gt_image, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
    pred_image = _draw_polygons(image, pred_polygons, color=(255, 0, 0))
    pred_image = cv2.resize(
        pred_image, (cell_size, cell_size), interpolation=cv2.INTER_AREA
    )

    gt_score = gt_score_map.detach().cpu().numpy()
    gt_boundary = gt_boundary_map.detach().cpu().numpy()
    gt_center = gt_center_map.detach().cpu().numpy()
    gt_instance = gt_instance_map.detach().cpu().numpy()

    gt_cells = [
        gt_image,
        _to_heatmap(gt_score, cell_size),
        _to_heatmap(gt_boundary, cell_size),
        _to_heatmap(gt_center, cell_size),
        _labels_to_vis(gt_instance, cell_size),
    ]
    pred_cells = [
        pred_image,
        _to_heatmap(pred_score_map, cell_size),
        _to_heatmap(pred_boundary_map, cell_size),
        _to_heatmap(pred_center_map, cell_size),
        _labels_to_vis(pred_instance_map, cell_size),
    ]

    collage = np.full((cell_size * 2, cell_size * 5, 3), 255, dtype=np.uint8)
    for col, cell in enumerate(gt_cells):
        collage[0:cell_size, col * cell_size : (col + 1) * cell_size] = cell
    for col, cell in enumerate(pred_cells):
        collage[cell_size : 2 * cell_size, col * cell_size : (col + 1) * cell_size] = cell

    # Small visual anchor: original image in top-left corner of both image cells.
    thumb = cv2.resize(image_cell, (cell_size // 4, cell_size // 4))
    collage[0 : thumb.shape[0], 0 : thumb.shape[1]] = thumb
    collage[cell_size : cell_size + thumb.shape[0], 0 : thumb.shape[1]] = thumb
    return collage


def _collage_batch(
    model,
    dataset,
    device,
    num: int = 4,
    cell_size: int = 480,
) -> Optional[np.ndarray]:
    if len(dataset) == 0:
        return None

    model.eval()
    collages = []
    sample_count = min(num, len(dataset))
    indices = np.random.choice(len(dataset), size=sample_count, replace=False)
    for idx in indices:
        img_t, target = dataset[int(idx)]
        gt_hw = target["score_map"].shape[-2:]
        with torch.no_grad():
            out = model(img_t.unsqueeze(0).to(device))
            pred = _resize_outputs(out, gt_hw)

        pred_score = pred["score"][0, 0].detach().cpu().numpy()
        pred_boundary = pred["boundary"][0, 0].detach().cpu().numpy()
        pred_center = pred["center"][0, 0].detach().cpu().numpy()
        pred_instances = decode_instance_maps(
            pred_score,
            pred_boundary,
            pred_center,
            score_thresh=0.5,
            boundary_thresh=0.5,
            center_thresh=0.45,
            min_area=4,
        )

        image_h, image_w = img_t.shape[-2:]
        scale_x = image_w / pred_instances.shape[1]
        scale_y = image_h / pred_instances.shape[0]
        pred_polygons = labels_to_polygons(
            pred_instances,
            score_map=pred_score,
            scale_x=scale_x,
            scale_y=scale_y,
        )

        collages.append(
            create_collage(
                img_tensor=img_t,
                gt_score_map=target["score_map"],
                gt_boundary_map=target["boundary_map"],
                gt_center_map=target["center_map"],
                gt_instance_map=target["instance_map"],
                gt_polygons=target.get("polygons", []),
                pred_score_map=pred_score,
                pred_boundary_map=pred_boundary,
                pred_center_map=pred_center,
                pred_instance_map=pred_instances,
                pred_polygons=pred_polygons,
                cell_size=cell_size,
            )
        )

    if len(collages) == 1:
        return collages[0]
    top = np.hstack(collages[:2])
    if len(collages) > 2:
        bottom = np.hstack(collages[2:4])
        if bottom.shape[1] < top.shape[1]:
            pad = np.zeros((bottom.shape[0], top.shape[1] - bottom.shape[1], 3), dtype=np.uint8)
            bottom = np.hstack([bottom, pad])
    else:
        bottom = np.zeros_like(top)
    return np.vstack([top, bottom])


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
    del use_ohem, ohem_ratio, use_focal_geo, focal_gamma

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
        scaler = torch.amp.GradScaler("cuda")

        def autocast_ctx():
            return torch.amp.autocast("cuda")

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

    collage_cell_size = 480
    collage_samples = 4

    def make_collage(epoch: int):
        if not log_collage:
            return
        device_type = getattr(device, "type", str(device))
        if device_type == "cuda":
            torch.cuda.empty_cache()
        vis_model = ema_model if use_ema else model
        sources = (
            list(zip(val_dataset_names, val_datasets))
            if val_datasets
            else [("val", val_dataset)]
        )
        for ds_name, dataset in sources:
            if len(dataset) == 0:
                continue
            collage = _collage_batch(
                vis_model,
                dataset,
                device,
                num=collage_samples,
                cell_size=collage_cell_size,
            )
            if collage is None:
                continue
            writer.add_image(
                f"Validation/{_sanitize_tag(ds_name)}",
                collage,
                epoch,
                dataformats="HWC",
            )

    try:
        make_collage(max(start_epoch - 1, 0))
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] EASTV2 collage creation failed (start): {exc}")

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
            for part_name, part_value in getattr(criterion, "last_losses", {}).items():
                writer.add_scalar(f"LossParts/Train/{part_name}", part_value, current_step)
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
                        for part_name, part_value in getattr(
                            criterion, "last_losses", {}
                        ).items():
                            writer.add_scalar(
                                f"LossParts/Val/{part_name}",
                                part_value,
                                epoch,
                            )
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

            try:
                make_collage(epoch)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARNING] EASTV2 collage creation failed at epoch {epoch}: {exc}")

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
