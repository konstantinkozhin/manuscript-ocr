import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


def _autocast_disabled(device_type: str):
    if hasattr(torch, "amp"):
        try:
            return torch.amp.autocast(device_type, enabled=False)
        except TypeError:
            return torch.amp.autocast(device_type=device_type, enabled=False)
    if device_type == "cuda" and hasattr(torch, "cuda"):
        return torch.cuda.amp.autocast(enabled=False)
    return contextlib.nullcontext()


def dice_loss(gt: torch.Tensor, pred: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    pred = pred.reshape(pred.shape[0], -1)
    gt = gt.reshape(gt.shape[0], -1)
    inter = torch.sum(gt * pred, dim=1)
    union = torch.sum(gt, dim=1) + torch.sum(pred, dim=1)
    return torch.mean(1.0 - (2.0 * inter + eps) / (union + eps))


def balanced_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    max_pos_weight: float = 20.0,
) -> torch.Tensor:
    positives = torch.sum(target)
    negatives = target.numel() - positives
    if positives < 1:
        pos_weight = torch.ones((), device=logits.device, dtype=logits.dtype)
    else:
        pos_weight = torch.clamp(
            negatives / (positives + 1e-6),
            min=1.0,
            max=max_pos_weight,
        ).to(device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


class EASTV2Loss(nn.Module):
    """Loss for score, boundary and center maps."""

    def __init__(
        self,
        score_weight: float = 1.0,
        boundary_weight: float = 1.0,
        center_weight: float = 1.0,
        boundary_dice_weight: float = 1.0,
        center_dice_weight: float = 0.5,
    ):
        super().__init__()
        self.score_weight = float(score_weight)
        self.boundary_weight = float(boundary_weight)
        self.center_weight = float(center_weight)
        self.boundary_dice_weight = float(boundary_dice_weight)
        self.center_dice_weight = float(center_dice_weight)

    def forward(
        self,
        gt_score: torch.Tensor,
        pred_score: torch.Tensor,
        gt_boundary: torch.Tensor,
        pred_boundary: torch.Tensor,
        gt_center: torch.Tensor,
        pred_center: torch.Tensor,
        pred_score_logits: torch.Tensor = None,
        pred_boundary_logits: torch.Tensor = None,
    ) -> torch.Tensor:
        device_type = pred_score.device.type
        with _autocast_disabled(device_type):
            gt_score = gt_score.float()
            pred_score = pred_score.float()
            gt_boundary = gt_boundary.float()
            pred_boundary = pred_boundary.float()
            gt_center = gt_center.float()
            pred_center = pred_center.float()

            if pred_score_logits is not None:
                pred_score_logits = pred_score_logits.float()
                score_loss = balanced_bce_with_logits(pred_score_logits, gt_score)
            else:
                score_loss = F.binary_cross_entropy(pred_score, gt_score)

            if pred_boundary_logits is not None:
                pred_boundary_logits = pred_boundary_logits.float()
                boundary_bce = balanced_bce_with_logits(
                    pred_boundary_logits,
                    gt_boundary,
                    max_pos_weight=10.0,
                )
            else:
                boundary_bce = F.binary_cross_entropy(pred_boundary, gt_boundary)

            boundary_dice = dice_loss(gt_boundary, pred_boundary)
            center_mse = F.mse_loss(pred_center, gt_center)
            center_dice = dice_loss(gt_center, pred_center)
            center_loss = center_mse + self.center_dice_weight * center_dice

        total = (
            self.score_weight * score_loss
            + self.boundary_weight
            * (boundary_bce + self.boundary_dice_weight * boundary_dice)
            + self.center_weight * center_loss
        )
        self.last_losses = {
            "score": float(score_loss.detach().cpu()),
            "boundary_bce": float(boundary_bce.detach().cpu()),
            "boundary_dice": float(boundary_dice.detach().cpu()),
            "center_mse": float(center_mse.detach().cpu()),
            "center_dice": float(center_dice.detach().cpu()),
            "center": float(center_loss.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
        return total
