import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch_optimizer as toptim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .dataset import EASTDataset
from .sam import SAMSolver
from .loss import EASTLoss
from .utils import create_collage, decode_boxes_from_maps
from .east import TextDetectionFCN
from torch.utils.data import ConcatDataset


def _run_training(
    experiment_dir: str,
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    device: torch.device,
    num_epochs: int,
    batch_size: int,
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
):
    """
    Core training loop. Saves logs and checkpoints under experiment_dir.
    """
    # Setup directories
    log_dir = os.path.join(experiment_dir, "logs")
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=_custom_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=_custom_collate_fn,
        pin_memory=False,
    )

    # Optimizer & Scheduler
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=1,
        eta_min=lr / 100,
    )
    scaler = torch.cuda.amp.GradScaler()

    criterion = EASTLoss(
        use_ohem=use_ohem,
        ohem_ratio=ohem_ratio,
        use_focal_geo=use_focal_geo,
        focal_gamma=focal_gamma,
    )

    # EMA model copy
    ema_model = model if not use_ema else torch.deepcopy(model)
    if use_ema:
        for p in ema_model.parameters():
            p.requires_grad = False

    best_val_loss = float("inf")
    patience = 0

    def make_collage(tag: str, epoch: int):
        coll = _collage_batch(ema_model if use_ema else model, val_dataset, device)
        writer.add_image(f"Val/{tag}", coll, epoch, dataformats="HWC")

    make_collage("start", 0)

    # Training epochs
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, tgt in tqdm(train_loader, desc=f"Train {epoch}"):
            imgs = imgs.to(device)
            gt_s = tgt["score_map"].to(device)
            gt_g = tgt["geo_map"].to(device)

            # Optional multiscale
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

            optimizer.zero_grad()
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
                    return criterion(gt_s, ps, gt_g, pg)

                loss = optimizer.step(closure)
            else:
                with torch.cuda.amp.autocast():
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
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

            scheduler.step(epoch + imgs.size(0) / len(train_loader))
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, tgt in val_loader:
                imgs = imgs.to(device)
                gt_s = tgt["score_map"].to(device)
                gt_g = tgt["geo_map"].to(device)
                eval_model = ema_model if use_ema else model
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
                val_loss += criterion(gt_s, ps, gt_g, pg).item()

        avg_val = val_loss / len(val_loader)
        writer.add_scalar("Loss/Val", avg_val, epoch)

        # Save checkpoints
        torch.save(
            (ema_model if use_ema else model).state_dict(),
            os.path.join(ckpt_dir, f"epoch{epoch:03d}.pth"),
        )
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
                break

        make_collage(f"epoch{epoch}", epoch)

    writer.close()
    return ema_model


def _custom_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    score_maps = torch.stack([t["score_map"] for t in targets], dim=0)
    geo_maps = torch.stack([t["geo_map"] for t in targets], dim=0)
    rboxes_list = [t["rboxes"] for t in targets]
    return images, {"score_map": score_maps, "geo_map": geo_maps, "rboxes": rboxes_list}


def _collage_batch(model, dataset, device, num: int = 4):
    coll_imgs = []
    for i in range(min(num, len(dataset))):
        img_t, tgt = dataset[i]
        gt_s = tgt["score_map"].squeeze(0).cpu().numpy()
        gt_g = tgt["geo_map"].cpu().numpy().transpose(1, 2, 0)
        gt_r = tgt["rboxes"].cpu().numpy()

        with torch.no_grad():
            out = model(img_t.unsqueeze(0).to(device))
        ps = out["score"][0].cpu().numpy().squeeze(0)
        pg = out["geometry"][0].cpu().numpy().transpose(1, 2, 0)

        pred_r = decode_boxes_from_maps(
            ps, pg, score_thresh=0.9, scale=1 / model.score_scale
        )

        coll = create_collage(
            img_tensor=img_t,
            gt_score_map=gt_s,
            gt_geo_map=gt_g,
            gt_rboxes=gt_r,
            pred_score_map=ps,
            pred_geo_map=pg,
            pred_rboxes=pred_r,
            cell_size=640,
        )
        coll_imgs.append(coll)
    top = np.hstack(coll_imgs[:2])
    bot = np.hstack(coll_imgs[2:4]) if len(coll_imgs) > 2 else np.zeros_like(top)
    return np.vstack([top, bot])


def train(
    train_images: str,
    train_anns: str,
    val_images: str,
    val_anns: str,
    *,
    experiment_root: str = "./experiments",
    model_name: str = "resnet_quad",
    pretrained_backbone: bool = True,
    freeze_first: bool = True,
    target_size: int = 1024,
    score_geo_scale: float = None,
    epochs: int = 500,
    batch_size: int = 3,
    lr: float = 1e-3,
    grad_clip: float = 5.0,
    early_stop: int = 100,
    use_sam: bool = True,
    sam_type: str = "asam",
    use_lookahead: bool = True,
    use_ema: bool = False,
    use_multiscale: bool = True,
    use_ohem: bool = True,
    ohem_ratio: float = 0.5,
    use_focal_geo: bool = True,
    focal_gamma: float = 2.0,
    device: torch.device = None,
) -> torch.nn.Module:
    """
    High-level training entrypoint.

    Creates model, datasets, and runs the training. Logs and checkpoints
    are stored under `experiment_root/model_name`.

    Parameters
    ----------
    train_images : str
        Path to training images directory.
    train_anns : str
        Path to COCO JSON for training annotations.
    val_images : str
        Path to validation images directory.
    val_anns : str
        Path to COCO JSON for validation annotations.
    experiment_root : str, optional
        Root directory for experiments (default "./experiments").
    model_name : str, optional
        Subfolder under `experiment_root` for logs/checkpoints.
        Default "resnet_quad".
    pretrained_backbone : bool, optional
        Use pretrained backbone weights. Default True.
    freeze_first : bool, optional
        Freeze first layers of backbone. Default True.
    target_size : int, optional
        Resize shortest side of images to this size. Default 1024.
    score_geo_scale : float, optional
        Scale factor for score/geometry maps. If None, uses model.score_scale.
    epochs : int, optional
        Number of training epochs. Default 500.
    batch_size : int, optional
        Samples per batch. Default 3.
    lr : float, optional
        Initial learning rate. Default 1e-3.
    grad_clip : float, optional
        Gradient clipping norm. Default 5.0.
    early_stop : int, optional
        Patience for early stopping. Default 100.
    use_sam : bool, optional
        Use SAM optimizer. Default True.
    sam_type : str, optional
        "sam" or "asam" variant. Default "asam".
    use_lookahead : bool, optional
        Wrap optimizer with Lookahead. Default True.
    use_ema : bool, optional
        Keep EMA of model weights. Default False.
    use_multiscale : bool, optional
        Randomly scale inputs on train. Default True.
    use_ohem : bool, optional
        Use Online Hard Example Mining. Default True.
    ohem_ratio : float, optional
        Ratio for hard negatives. Default 0.5.
    use_focal_geo : bool, optional
        Apply focal loss to geometry. Default True.
    focal_gamma : float, optional
        Gamma for focal geometry loss. Default 2.0.
    device : torch.device, optional
        Compute device. If None, auto-select. Default None.

    Returns
    -------
    torch.nn.Module
        Trained model (EMA if use_ema else base).
    """
    # Device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Создаём модель
    model = TextDetectionFCN(
        pretrained_backbone=pretrained_backbone,
        freeze_first=freeze_first
    ).to(device)

    # Определяем scale (если не задан)
    if score_geo_scale is None:
        score_geo_scale = model.score_scale

    # Вспомогательная фабрика для востановления списка путей в EASTDataset
    def make_dataset(imgs, anns):
        return EASTDataset(
            images_folder=imgs,
            coco_annotation_file=anns,
            target_size=target_size,
            score_geo_scale=score_geo_scale,
        )

    # Собираем списки (если один путь, оборачиваем в список)
    train_imgs_list = train_images if isinstance(train_images, (list, tuple)) else [train_images]
    train_anns_list = train_anns   if isinstance(train_anns,   (list, tuple)) else [train_anns]
    val_imgs_list   = val_images   if isinstance(val_images,   (list, tuple)) else [val_images]
    val_anns_list   = val_anns     if isinstance(val_anns,     (list, tuple)) else [val_anns]

    # Проверяем что длины совпадают
    assert len(train_imgs_list) == len(train_anns_list), "train_images и train_anns должны иметь одинаковую длину"
    assert len(val_imgs_list)   == len(val_anns_list),   "val_images и val_anns должны иметь одинаковую длину"

    # Строим ConcatDataset
    train_datasets = [make_dataset(imgs, anns) for imgs, anns in zip(train_imgs_list, train_anns_list)]
    val_datasets   = [make_dataset(imgs, anns) for imgs, anns in zip(val_imgs_list,   val_anns_list)]
    train_ds = ConcatDataset(train_datasets)
    val_ds   = ConcatDataset(val_datasets)

    # Путь для логов и чекпоинтов
    experiment_dir = os.path.join(experiment_root, model_name)

    # Запускаем прежнюю функцию обучения
    best_model = _run_training(
        experiment_dir=experiment_dir,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        device=device,
        num_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        grad_clip=grad_clip,
        early_stop=early_stop,
        use_sam=use_sam,
        sam_type=sam_type,
        use_lookahead=use_lookahead,
        use_ema=use_ema,
        use_multiscale=use_multiscale,
        use_ohem=use_ohem,
        ohem_ratio=ohem_ratio,
        use_focal_geo=use_focal_geo,
        focal_gamma=focal_gamma,
    )
    return best_model
