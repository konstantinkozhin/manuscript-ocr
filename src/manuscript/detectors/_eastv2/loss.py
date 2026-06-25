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


class EASTV2Loss(nn.Module):
    """Loss for score, boundary and center maps."""

    def __init__(
        self,
        score_weight: float = 1.0,
        boundary_weight: float = 1.0,
        center_weight: float = 1.0,
        boundary_dice_weight: float = 1.0,
    ):
        super().__init__()
        self.score_weight = float(score_weight)
        self.boundary_weight = float(boundary_weight)
        self.center_weight = float(center_weight)
        self.boundary_dice_weight = float(boundary_dice_weight)

    def forward(
        self,
        gt_score: torch.Tensor,
        pred_score: torch.Tensor,
        gt_boundary: torch.Tensor,
        pred_boundary: torch.Tensor,
        gt_center: torch.Tensor,
        pred_center: torch.Tensor,
    ) -> torch.Tensor:
        device_type = pred_score.device.type
        with _autocast_disabled(device_type):
            gt_score = gt_score.float()
            pred_score = pred_score.float()
            gt_boundary = gt_boundary.float()
            pred_boundary = pred_boundary.float()
            gt_center = gt_center.float()
            pred_center = pred_center.float()

            score_loss = F.binary_cross_entropy(pred_score, gt_score)
            boundary_bce = F.binary_cross_entropy(pred_boundary, gt_boundary)
            boundary_dice = dice_loss(gt_boundary, pred_boundary)
            center_loss = F.mse_loss(pred_center, gt_center)

        return (
            self.score_weight * score_loss
            + self.boundary_weight
            * (boundary_bce + self.boundary_dice_weight * boundary_dice)
            + self.center_weight * center_loss
        )
