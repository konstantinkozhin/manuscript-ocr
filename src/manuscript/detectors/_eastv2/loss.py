import torch
import torch.nn as nn
import torch.nn.functional as F


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
