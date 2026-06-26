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
from .loss import PSELoss
from .utils import labels_to_polygons, pse


def _custom_collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), {
        "text_map": torch.stack([t["text_map"] for t in targets], dim=0),
        "kernel_maps": torch.stack([t["kernel_maps"] for t in targets], dim=0),
        "training_mask": torch.stack([t["training_mask"] for t in targets], dim=0),
        "instance_map": torch.stack([t["instance_map"] for t in targets], dim=0),
        "polygons": [t.get("polygons", []) for t in targets],
    }


def _resize_outputs(out: Dict[str, torch.Tensor], target_hw: Tuple[int, int]):
    return {
        "logits": F.interpolate(
            out["logits"], size=target_hw, mode="bilinear", align_corners=False
        ),
        "maps": F.interpolate(
            out["maps"], size=target_hw, mode="bilinear", align_corners=False
        ),
    }


def _loss_from_batch(criterion, target, pred):
    return criterion(
        pred["logits"],
        target["text_map"],
        target["kernel_maps"],
        target["training_mask"],
    )


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
            [(37 * label_id) % 255, (97 * label_id) % 255, (173 * label_id) % 255],
            dtype=np.uint8,
        )
        vis[labels == label_id] = color
    return cv2.resize(vis, (cell_size, cell_size), interpolation=cv2.INTER_NEAREST)


def _draw_polygons(image: np.ndarray, polygons, color: Tuple[int, int, int]) -> np.ndarray:
    vis = image.copy()
    for polygon in polygons:
        if isinstance(polygon, tuple) and len(polygon) == 2:
            polygon = polygon[0]
        if torch.is_tensor(polygon):
            polygon = polygon.detach().cpu().numpy()
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 2:
            continue
        cv2.polylines(vis, [pts.astype(np.int32)], True, color=color, thickness=2)
    return vis


def _collage_batch(model, dataset, device, num: int = 4, cell_size: int = 480):
    if len(dataset) == 0:
        return None
    model.eval()
    collages = []
    for idx in np.random.choice(len(dataset), size=min(num, len(dataset)), replace=False):
        img_t, target = dataset[int(idx)]
        gt_hw = target["text_map"].shape[-2:]
        with torch.no_grad():
            out = model(img_t.unsqueeze(0).to(device))
            pred = _resize_outputs(out, gt_hw)

        image = _tensor_to_image(
            img_t,
            denormalize={"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        )
        pred_maps = pred["maps"][0].detach().cpu().numpy()
        pred_kernels = (pred_maps > 0.5).astype(np.uint8)
        pred_kernels[1:] = pred_kernels[1:] * pred_kernels[:1]
        pred_labels = pse(pred_kernels, min_area=16)
        scale_x = img_t.shape[-1] / pred_labels.shape[1]
        scale_y = img_t.shape[-2] / pred_labels.shape[0]
        pred_polygons = labels_to_polygons(
            pred_labels,
            pred_maps[0],
            scale_x=scale_x,
            scale_y=scale_y,
            min_score=0.0,
        )

        gt_img = _draw_polygons(image, target.get("polygons", []), color=(0, 255, 0))
        pred_img = _draw_polygons(image, pred_polygons, color=(255, 0, 0))
        gt_instance = target["instance_map"].detach().cpu().numpy()
        gt_text = target["text_map"].detach().cpu().numpy()
        gt_kernel = target["kernel_maps"][-1].detach().cpu().numpy()

        cells = [
            cv2.resize(gt_img, (cell_size, cell_size), interpolation=cv2.INTER_AREA),
            _to_heatmap(gt_text, cell_size),
            _to_heatmap(gt_kernel, cell_size),
            _labels_to_vis(gt_instance, cell_size),
            cv2.resize(pred_img, (cell_size, cell_size), interpolation=cv2.INTER_AREA),
            _to_heatmap(pred_maps[0], cell_size),
            _to_heatmap(pred_maps[-1], cell_size),
            _labels_to_vis(pred_labels, cell_size),
        ]
        collage = np.full((cell_size * 2, cell_size * 4, 3), 255, dtype=np.uint8)
        for col, cell in enumerate(cells[:4]):
            collage[:cell_size, col * cell_size : (col + 1) * cell_size] = cell
        for col, cell in enumerate(cells[4:]):
            collage[cell_size:, col * cell_size : (col + 1) * cell_size] = cell
        collages.append(collage)

    if len(collages) == 1:
        return collages[0]
    top = np.hstack(collages[:2])
    bottom = np.hstack(collages[2:4]) if len(collages) > 2 else np.zeros_like(top)
    if bottom.shape[1] < top.shape[1]:
        pad = np.zeros((bottom.shape[0], top.shape[1] - bottom.shape[1], 3), dtype=np.uint8)
        bottom = np.hstack([bottom, pad])
    return np.vstack([top, bottom])


def _sanitize_tag(name: str) -> str:
    return name.replace("\\", "_").replace("/", "_").replace(" ", "_")


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
    del use_ohem, ohem_ratio, use_focal_geo, focal_gamma, score_map_shrink_ratio

    experiment_dir = os.path.abspath(os.fspath(experiment_dir))
    log_dir = os.path.join(experiment_dir, "logs")
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(experiment_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "detector": "PSE",
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
                "optimizer": "SAM" if use_sam else ("Lookahead(RAdam)" if use_lookahead else "RAdam"),
                "train_dataset_size": len(train_dataset),
                "val_dataset_size": len(val_dataset),
                "augmentation_config": augmentation_config,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

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
        names = val_dataset_names or [f"val_{idx}" for idx in range(len(val_datasets))]
        val_eval_loaders = [
            (
                name,
                DataLoader(
                    ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=_custom_collate_fn,
                    pin_memory=False,
                    persistent_workers=num_workers > 0,
                ),
            )
            for name, ds in zip(names, val_datasets)
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
        optimizer = toptim.Lookahead(base_opt, k=5, alpha=0.5) if use_lookahead else base_opt

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

    criterion = PSELoss()
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
            raise FileNotFoundError(f"Resume requested, but state file not found: {state_path}")
        checkpoint = torch.load(state_path, map_location=device, weights_only=False)
        if _is_full_state_checkpoint(checkpoint):
            model.load_state_dict(checkpoint["model_state"])
            if use_ema and checkpoint.get("ema_state") is not None:
                ema_model.load_state_dict(checkpoint["ema_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if scheduler is not None and checkpoint.get("scheduler_state") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            if checkpoint.get("scaler_state") is not None:
                scaler.load_state_dict(checkpoint["scaler_state"])
            best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
            best_val_dice = checkpoint.get("best_val_dice", best_val_dice)
            patience_loss = checkpoint.get("patience_loss", patience_loss)
            patience_dice = checkpoint.get("patience_dice", patience_dice)
            start_epoch = checkpoint.get("epoch", 0) + 1
        else:
            state_dict = _extract_model_state(checkpoint)
            ok, error_msg = _check_architecture_compatibility(model, state_dict)
            if not ok:
                raise ValueError(f"Architecture mismatch when loading weights: {error_msg}")
            model.load_state_dict(state_dict, strict=False)
            if use_ema:
                ema_model.load_state_dict(state_dict, strict=False)

    writer = SummaryWriter(log_dir, purge_step=start_epoch if resume else None)

    def make_collage(epoch_idx: int) -> None:
        if not log_collage:
            return
        eval_model = ema_model if use_ema else model
        sources = list(zip(val_dataset_names or [], val_datasets or []))
        if not sources:
            sources = [("val", val_dataset)]
        for ds_name, dataset in sources:
            collage = _collage_batch(eval_model, dataset, device)
            if collage is not None:
                writer.add_image(
                    f"Validation/{_sanitize_tag(str(ds_name))}",
                    collage,
                    epoch_idx,
                    dataformats="HWC",
                )

    try:
        make_collage(max(start_epoch - 1, 0))
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to write initial PSE collage: {exc}")

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
            tgt = {k: v.to(device) if torch.is_tensor(v) else v for k, v in tgt.items()}
            gt_hw = tgt["text_map"].shape[-2:]

            if use_multiscale:
                sf = torch.empty(1).uniform_(0.8, 1.2).item()
                h, w = imgs.shape[-2:]
                nh = max(32, int(h * sf) // 32 * 32)
                nw = max(32, int(w * sf) // 32 * 32)
                imgs_in = F.interpolate(imgs, size=(nh, nw), mode="bilinear", align_corners=False)
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
                    loss = _loss_from_batch(criterion, tgt, pred) / accumulation_steps
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
            step = global_step + batch_idx
            writer.add_scalar("Loss/Train_Step", float(loss.item()), step)
            for name, value in getattr(criterion, "last_losses", {}).items():
                writer.add_scalar(f"LossParts/Train/{name}", value, step)
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
            writer.add_scalar("LearningRate/Step", current_lr, step)

            if use_ema:
                with torch.no_grad():
                    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

        avg_train = train_loss / max(len(train_loader), 1)
        writer.add_scalar("Loss/Train", avg_train, epoch)
        if scheduler is not None and scheduler_step == "epoch":
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
        writer.add_scalar("LearningRate/Epoch", current_lr, epoch)

        should_stop = False
        if epoch % val_interval == 0:
            model.eval()
            if use_ema:
                ema_model.eval()
            eval_model = ema_model if use_ema else model
            total_val_loss = 0.0
            total_batches = 0
            dice_sum = 0.0
            dice_count = 0
            with torch.no_grad():
                for ds_name, loader in val_eval_loaders:
                    ds_loss = 0.0
                    ds_batches = 0
                    ds_dice_sum = 0.0
                    ds_dice_count = 0
                    for imgs, tgt in loader:
                        imgs = imgs.to(device)
                        tgt = {k: v.to(device) if torch.is_tensor(v) else v for k, v in tgt.items()}
                        out = eval_model(imgs)
                        pred = _resize_outputs(out, tgt["text_map"].shape[-2:])
                        batch_loss = _loss_from_batch(criterion, tgt, pred).item()
                        ds_loss += batch_loss
                        total_val_loss += batch_loss
                        ds_batches += 1
                        total_batches += 1
                        dice_vals = dice_coefficient(pred["maps"][:, :1], tgt["text_map"].unsqueeze(1))
                        ds_dice_sum += dice_vals.sum().item()
                        ds_dice_count += dice_vals.numel()
                        dice_sum += dice_vals.sum().item()
                        dice_count += dice_vals.numel()
                    tag = ds_name.replace("\\", "_").replace("/", "_").replace(" ", "_")
                    writer.add_scalar(f"Loss/Val/{tag}", ds_loss / max(ds_batches, 1), epoch)
                    writer.add_scalar(
                        f"Dice/Val/{tag}",
                        ds_dice_sum / ds_dice_count if ds_dice_count else 0.0,
                        epoch,
                    )

            avg_val = total_val_loss / max(total_batches, 1)
            val_dice = dice_sum / dice_count if dice_count else 0.0
            writer.add_scalar("Loss/Val", avg_val, epoch)
            writer.add_scalar("Dice/Val", val_dice, epoch)
            try:
                make_collage(epoch)
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to write PSE collage for epoch {epoch}: {exc}")
            if scheduler is not None and scheduler_step == "val":
                scheduler.step(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_loss = 0
                torch.save((ema_model if use_ema else model).state_dict(), os.path.join(ckpt_dir, "best_loss.pth"))
            else:
                patience_loss += 1
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                patience_dice = 0
                torch.save((ema_model if use_ema else model).state_dict(), os.path.join(ckpt_dir, "best_dice.pth"))
            else:
                patience_dice += 1
            should_stop = patience_loss >= early_stop and patience_dice >= early_stop

        torch.save((ema_model if use_ema else model).state_dict(), os.path.join(ckpt_dir, "last.pth"))
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
        from manuscript.detectors import PSE

        best_weights_path = os.path.join(ckpt_dir, "best_loss.pth")
        if not os.path.isfile(best_weights_path):
            best_weights_path = os.path.join(ckpt_dir, "best_dice.pth")
        PSE.export(
            weights_path=best_weights_path,
            output_path=os.path.join(ckpt_dir, "best_model.onnx"),
            backbone_name=backbone_name or "resnet50",
            input_size=target_size if target_size is not None else 1280,
            kernel_num=getattr(model, "kernel_num", 7),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to export PSE ONNX model: {exc}")

    return ema_model if use_ema else model
