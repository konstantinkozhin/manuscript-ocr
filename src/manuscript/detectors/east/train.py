import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch_optimizer as toptim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataset import EASTDataset
from sam import SAMSolver
from loss import EASTLoss
from utils import create_collage, decode_boxes_from_maps
from east import TextDetectionFCN


def run_training(
    stage_name: str,
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    device: torch.device,
    num_epochs: int,
    batch_size: int,
    lr: float,
    grad_clip: float,
    ema_decay: float,
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
    log_root: str = "./logs",
    ckpt_root: str = "./checkpoints",
):
    # dirs & writer
    log_dir = os.path.join(log_root, stage_name)
    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(ckpt_root, stage_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=False,
    )

    # optimizer / scheduler
    if use_sam:
        optimizer = SAMSolver(
            model.parameters(),
            torch.optim.SGD,
            rho=0.05,
            lr=lr,
            use_adaptive=(sam_type == "asam"),
        )
    else:
        base = toptim.RAdam(model.parameters(), lr=lr)
        optimizer = toptim.Lookahead(base, k=5, alpha=0.5) if use_lookahead else base
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=lr / 100
    )
    scaler = torch.cuda.amp.GradScaler()

    criterion = EASTLoss(
        angle_weight=0.0,
        use_ohem=use_ohem,
        ohem_ratio=ohem_ratio,
        use_focal_geo=use_focal_geo,
        focal_gamma=focal_gamma,
    )

    # EMA model
    ema_model = model if not use_ema else torch.deepcopy(model)
    if use_ema:
        for p in ema_model.parameters():
            p.requires_grad = False

    best_loss = float("inf")
    patience = 0

    def make_collage(tag, epoch):
        coll = collage_batch(ema_model if use_ema else model, val_dataset, device)
        writer.add_image(f"Val/{tag}", coll, epoch, dataformats="HWC")

    make_collage("start", 0)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for imgs, tgt in tqdm(train_loader, desc=f"Train {epoch}"):
            imgs = imgs.to(device)
            gt_s = tgt["score_map"].to(device)
            gt_g = tgt["geo_map"].to(device)
            # multiscale
            if use_multiscale:
                sf = random.uniform(0.8, 1.2)
                H, W = imgs.shape[-2:]
                nh, nw = max(32, int(H * sf) // 32 * 32), max(
                    32, int(W * sf) // 32 * 32
                )
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
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train, epoch)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, tgt in val_loader:
                imgs = imgs.to(device)
                gt_s = tgt["score_map"].to(device)
                gt_g = tgt["geo_map"].to(device)
                out_mod = ema_model if use_ema else model
                res = out_mod(imgs)
                ps = F.interpolate(
                    res["score"],
                    size=gt_s.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                pg = F.interpolate(
                    res["geometry"],
                    size=gt_s.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                val_loss += criterion(gt_s, ps, gt_g, pg).item()
        avg_val = val_loss / len(val_loader)
        writer.add_scalar("Loss/Val", avg_val, epoch)

        # checkpoints & early stop
        torch.save(
            (ema_model if use_ema else model).state_dict(),
            os.path.join(ckpt_dir, f"epoch{epoch:02d}.pth"),
        )
        if avg_val < best_loss:
            best_loss = avg_val
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

def custom_collate_fn(batch):
    images, targets = zip(*batch)
    # 1) стекаем изображения и карты
    images = torch.stack(images, dim=0)
    score_maps = torch.stack([t["score_map"] for t in targets], dim=0)
    geo_maps   = torch.stack([t["geo_map"]   for t in targets], dim=0)
    # 2) собираем rboxes в список
    rboxes_list = [t["rboxes"] for t in targets]
    return images, {
        "score_map": score_maps,
        "geo_map":   geo_maps,
        "rboxes":    rboxes_list
    }

def collage_batch(model, dataset, device, num=4):
    coll_imgs = []
    for i in range(min(num, len(dataset))):
        img_t, tgt = dataset[i]
        gt_s = tgt["score_map"].squeeze(0).cpu().numpy()
        gt_g = tgt["geo_map"].cpu().numpy().transpose(1, 2, 0)
        gt_r = tgt["rboxes"].cpu().numpy()  # <- берем из таргета

        with torch.no_grad():
            out = model(img_t.unsqueeze(0).to(device))
        ps = out["score"][0].cpu().numpy().squeeze(0)
        pg = out["geometry"][0].cpu().numpy().transpose(1, 2, 0)

        # декодим предикшены в rboxes
        pred_r = decode_boxes_from_maps(
            ps, pg, score_thresh=0.9, scale=1 / model.score_scale
        )

        coll = create_collage(
            img_tensor=img_t,
            gt_score_map=gt_s,
            gt_geo_map=gt_g,
            gt_rboxes=gt_r,  # <- подставляем истинные
            pred_score_map=ps,
            pred_geo_map=pg,
            pred_rboxes=pred_r,  # <- подставляем предикты
            cell_size=640,
        )
        coll_imgs.append(coll)
    top = np.hstack(coll_imgs[:2])
    bot = np.hstack(coll_imgs[2:4]) if len(coll_imgs) > 2 else np.zeros_like(top)
    return np.vstack([top, bot])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = TextDetectionFCN(pretrained_backbone=True, freeze_first=True).to(device)
    train_ds = EASTDataset(
        r"C:\data0205\Archives020525\train_images",
        r"C:\data0205\Archives020525\train.json",
        target_size=1024,
        score_geo_scale=model.score_scale,
    )
    val_ds = EASTDataset(
        r"C:\data0205\Archives020525\test_images",
        r"C:\data0205\Archives020525\test.json",
        target_size=1024,
        score_geo_scale=model.score_scale,
    )
    best = run_training(
        "resnet_quad",
        model,
        train_ds,
        val_ds,
        device,
        num_epochs=500,
        batch_size=3,
        lr=1e-3,
        grad_clip=5.0,
        ema_decay=0.999,
        early_stop=100,
        use_sam=True,
        sam_type="asam",
        use_lookahead=True,
        use_ema=False,
        use_multiscale=True,
        use_ohem=True,
        ohem_ratio=0.5,
        use_focal_geo=True,
        focal_gamma=2.0,
    )
