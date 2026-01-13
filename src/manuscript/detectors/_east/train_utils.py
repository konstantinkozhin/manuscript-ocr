import json
import os
import random
from collections import OrderedDict
from typing import Optional, Sequence, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch_optimizer as toptim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .lanms import locality_aware_nms
from .loss import EASTLoss
from .sam import SAMSolver
from .utils import create_collage, decode_quads_from_maps


def _is_full_state_checkpoint(checkpoint: Dict[str, Any]) -> bool:
    required_keys = {"model_state", "optimizer_state", "scheduler_state"}
    return isinstance(checkpoint, dict) and required_keys.issubset(checkpoint.keys())


def _extract_model_state(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint
    return checkpoint


def _check_architecture_compatibility(
    model: torch.nn.Module,
    loaded_state: Dict[str, torch.Tensor],
) -> Tuple[bool, str]:
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    loaded_keys = set(loaded_state.keys())
    
    common_keys = model_keys & loaded_keys
    if not common_keys:
        return False, "No common keys between model and loaded weights"
    
    shape_mismatches = []
    for key in common_keys:
        if model_state[key].shape != loaded_state[key].shape:
            shape_mismatches.append(
                f"{key}: model {model_state[key].shape} vs loaded {loaded_state[key].shape}"
            )
    
    if shape_mismatches:
        return False, f"Shape mismatches in keys: {shape_mismatches}"
    
    return True, ""


def _load_weights_only(
    model: torch.nn.Module,
    weights_path: str,
    device: torch.device,
    strict: bool = False,
) -> None:
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = _extract_model_state(checkpoint)
    
    is_compatible, error_msg = _check_architecture_compatibility(model, state_dict)
    if not is_compatible:
        raise ValueError(f"Architecture mismatch: {error_msg}")
    
    model.load_state_dict(state_dict, strict=strict)


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    numerator = 2.0 * torch.sum(pred_flat * target_flat, dim=1)
    denominator = torch.sum(pred_flat, dim=1) + torch.sum(target_flat, dim=1)
    return (numerator + eps) / (denominator + eps)


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
):
    experiment_dir = os.path.abspath(os.fspath(experiment_dir))
    log_dir = os.path.join(experiment_dir, "logs")
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    config = {
        "backbone_name": backbone_name,
        "pretrained_backbone": pretrained_backbone,
        "target_size": target_size,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "lr": lr,
        "grad_clip": grad_clip,
        "early_stop": early_stop,
        "use_sam": use_sam,
        "sam_type": sam_type if use_sam else None,
        "use_lookahead": use_lookahead,
        "use_ema": use_ema,
        "use_multiscale": use_multiscale,
        "use_ohem": use_ohem,
        "ohem_ratio": ohem_ratio if use_ohem else None,
        "use_focal_geo": use_focal_geo,
        "focal_gamma": focal_gamma if use_focal_geo else None,
        "val_interval": val_interval,
        "scheduler": "CosineAnnealingWarmRestarts",
        "optimizer": "SAM" if use_sam else ("Lookahead(RAdam)" if use_lookahead else "RAdam"),
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
    }
    
    config_path = os.path.join(experiment_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Training configuration saved to: {config_path}")

    start_epoch = 1
    best_val_loss = float("inf")
    patience = 0

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

        collage_sources = list(zip(val_dataset_names, val_datasets))
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
            for name, dataset in collage_sources
        ]
    else:
        collage_sources = [("val", val_dataset)]
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

    hook_attrs = [
        "_optimizer_state_dict_pre_hooks",
        "_optimizer_state_dict_post_hooks",
        "_optimizer_load_state_dict_pre_hooks",
        "_optimizer_load_state_dict_post_hooks",
    ]
    for attr in hook_attrs:
        if not hasattr(optimizer, attr):
            setattr(optimizer, attr, OrderedDict())

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=1,
        eta_min=lr / 100,
    )
    try:
        scaler = torch.amp.GradScaler(device_type="cuda")

        def autocast_ctx():
            return torch.amp.autocast(device_type="cuda")

    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()

        def autocast_ctx():
            return torch.cuda.amp.autocast()

    criterion = EASTLoss(
        use_ohem=use_ohem,
        ohem_ratio=ohem_ratio,
        use_focal_geo=use_focal_geo,
        focal_gamma=focal_gamma,
    )


    ema_model = model if not use_ema else torch.deepcopy(model)
    ema_decay = 0.9999
    if use_ema:
        for p in ema_model.parameters():
            p.requires_grad = False

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
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            scaler_state = checkpoint.get("scaler_state")
            if scaler_state is not None:
                scaler.load_state_dict(scaler_state)
            best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
            patience = checkpoint.get("patience", patience)
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

    def _sanitize_tag(name: str) -> str:
        return name.replace("\\", "_").replace("/", "_").replace(" ", "_")

    collage_cell_size = 480
    collage_samples = 4

    def make_collage(tag: str, epoch: int):
        """Create validation collage with predictions from ONNX export."""
        vis_model = ema_model if use_ema else model
        
        for ds_name, dataset in collage_sources:
            if len(dataset) == 0:
                continue
            coll = _collage_batch(
                vis_model,
                dataset,
                device,
                num=collage_samples,
                cell_size=collage_cell_size,
                experiment_dir=experiment_dir,
                epoch=epoch,
            )
            # Use fixed tag name to avoid creating millions of cards
            writer.add_image(
                f"Validation/{_sanitize_tag(ds_name)}",  # Fixed tag without epoch
                coll,
                epoch,  # epoch as step
                dataformats="HWC",
            )

    make_collage("start", max(start_epoch - 1, 0))

    if start_epoch > num_epochs:
        print(
            f"Resume epoch {start_epoch} exceeds configured num_epochs={num_epochs}. Nothing to train."
        )
        writer.close()
        return ema_model if use_ema else model

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        global_step = (epoch - 1) * len(train_loader)
        
        for batch_idx, (imgs, tgt) in enumerate(tqdm(train_loader, desc=f"Train {epoch}"), 1):
            imgs = imgs.to(device)
            gt_s = tgt["score_map"].to(device)
            gt_g = tgt["geo_map"].to(device)

            if use_multiscale:
                sf = random.uniform(0.8, 1.2)
                H, W = imgs.shape[-2:]
                nh = max(32, int(H * sf) // 32 * 32)
                nw = max(32, int(W * sf) // 32 * 32)
                imgs_in = F.interpolate(
                    imgs, size=(nh, nw), mode="bilinear", align_corners=False
                )
            else:
                imgs_in = imgs

            if use_sam:

                def closure():
                    out = model(imgs_in)
                    ps = F.interpolate(
                        out["score"],
                        size=gt_s.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    pg = F.interpolate(
                        out["geometry"],
                        size=gt_s.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    loss = criterion(gt_s, ps, gt_g, pg)
                    return loss / accumulation_steps

                loss = optimizer.step(closure) * accumulation_steps
            else:
                with autocast_ctx():
                    out = model(imgs_in)
                    ps = F.interpolate(
                        out["score"],
                        size=gt_s.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    pg = F.interpolate(
                        out["geometry"],
                        size=gt_s.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    loss = criterion(gt_s, ps, gt_g, pg)

                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                

                if batch_idx % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                

                loss = loss * accumulation_steps

            scheduler.step(epoch + imgs.size(0) / len(train_loader))
            train_loss += loss.item()
            
            current_step = global_step + batch_idx
            writer.add_scalar("Loss/Train_Step", loss.item(), current_step)
            
            # Log learning rate at each step
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar("LearningRate/Step", current_lr, current_step)

            if use_ema:
                with torch.no_grad():
                    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

        avg_train = train_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train, epoch)
        
        # Log learning rate at each epoch
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("LearningRate/Epoch", current_lr, epoch)
        
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
                        gt_s = tgt["score_map"].to(device)
                        gt_g = tgt["geo_map"].to(device)

                        out = eval_model(imgs)
                        ps = F.interpolate(
                            out["score"],
                            size=gt_s.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        pg = F.interpolate(
                            out["geometry"],
                            size=gt_s.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                        batch_loss = criterion(gt_s, ps, gt_g, pg).item()
                        dataset_loss += batch_loss
                        dataset_batches += 1
                        total_val_loss += batch_loss
                        total_val_batches += 1

                        dice_vals = dice_coefficient(ps, gt_s)
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

            for ds_name, metrics in per_dataset_metrics.items():
                tag = _sanitize_tag(ds_name)
                writer.add_scalar(f"Loss/Val/{tag}", metrics["loss"], epoch)
                writer.add_scalar(f"Dice/Val/{tag}", metrics["dice"], epoch)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience = 0
                torch.save(
                    (ema_model if use_ema else model).state_dict(),
                    os.path.join(ckpt_dir, "best.pth"),
                )
            else:
                patience += 1
                if patience >= early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    should_stop = True

            make_collage("Predictions", epoch)
            
            if device.type == "cuda":
                torch.cuda.empty_cache()

        torch.save(
            (ema_model if use_ema else model).state_dict(),
            os.path.join(ckpt_dir, "last.pth"),
        )

        state_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ema_state": ema_model.state_dict() if use_ema else None,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience": patience,
        }
        torch.save(state_payload, os.path.join(ckpt_dir, "last_state.pt"))
        if should_stop:
            break

    writer.close()
    
    try:
        print("Attempting to export best model to ONNX...")
        from manuscript.detectors import EAST
        
        onnx_path = os.path.join(ckpt_dir, "best_model.onnx")
        best_weights_path = os.path.join(ckpt_dir, "best.pth")
        
        export_size = target_size if target_size is not None else 1280
        
        EAST.export_to_onnx(
            weights_path=best_weights_path,
            output_path=onnx_path,
            input_size=export_size,
            opset_version=14,
            simplify=True,
        )
        print(f"ONNX model exported successfully: {onnx_path}")
    except Exception as e:
        print(f"Failed to export ONNX model: {e}")
    
    return ema_model


def _custom_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    score_maps = torch.stack([t["score_map"] for t in targets], dim=0)
    geo_maps = torch.stack([t["geo_map"] for t in targets], dim=0)
    quads_list = [t["quads"] for t in targets]
    return images, {"score_map": score_maps, "geo_map": geo_maps, "quads": quads_list}


def _collage_batch(
    model, 
    dataset, 
    device, 
    num: int = 4, 
    cell_size: int = 640,
    experiment_dir: str = None,
    epoch: int = 0,
):
    """Create collage with model predictions for visualization."""
    coll_imgs = []
    for i in range(min(num, len(dataset))):
        img_t, tgt = dataset[i]
        gt_s = tgt["score_map"].squeeze(0).cpu().numpy()
        gt_g = tgt["geo_map"].cpu().numpy().transpose(1, 2, 0)
        gt_quads = tgt["quads"].cpu().numpy()

        with torch.no_grad():
            out = model(img_t.unsqueeze(0).to(device))
        ps = out["score"][0].cpu().numpy().squeeze(0)
        pg = out["geometry"][0].cpu().numpy().transpose(1, 2, 0)

        pred_quads = decode_quads_from_maps(
            ps, pg, score_thresh=0.7, scale=1 / model.score_scale, quantization=1
        )
        
        if len(pred_quads) > 0:
            pred_quads = locality_aware_nms(
                pred_quads.astype(np.float32), 
                iou_threshold=0.2,
                iou_threshold_standard=0.05
            )

        coll = create_collage(
            img_tensor=img_t,
            gt_score_map=gt_s,
            gt_geo_map=gt_g,
            gt_quads=gt_quads,
            pred_score_map=ps,
            pred_geo_map=pg,
            pred_quads=pred_quads,
            cell_size=cell_size,
        )
        coll_imgs.append(coll)
    
    top = np.hstack(coll_imgs[:2])
    bot = np.hstack(coll_imgs[2:4]) if len(coll_imgs) > 2 else np.zeros_like(top)
    return np.vstack([top, bot])
