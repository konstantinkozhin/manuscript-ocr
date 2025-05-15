import torch
import torch.nn as nn


def compute_dice_loss(gt, pred):
    inter = torch.sum(gt * pred)
    union = torch.sum(gt) + torch.sum(pred) + 1e-5
    return 1.0 - (2 * inter / union)


class EASTLoss(nn.Module):
    def __init__(
        self,
        angle_weight=0.0,  # unused for QUAD
        use_ohem=False,
        ohem_ratio=0.5,
        use_focal_geo=False,
        focal_gamma=2.0,
    ):
        super(EASTLoss, self).__init__()
        self.use_ohem = use_ohem
        self.ohem_ratio = ohem_ratio
        self.use_focal_geo = use_focal_geo
        self.focal_gamma = focal_gamma

    def forward(self, gt_score, pred_score, gt_geo, pred_geo):
        # If no positive pixels, return zero
        if torch.sum(gt_score) < 1:
            return torch.tensor(0.0, device=pred_score.device)

        # classification loss
        dice = compute_dice_loss(gt_score, pred_score)

        # geometry loss (QUAD offsets, 8 channels)
        # Compute L1 loss per channel
        diff = torch.abs(gt_geo - pred_geo)  # (B,8,H,W)
        # Sum over channels to get per-pixel loss
        geo_loss_map = torch.sum(diff, dim=1)  # (B,H,W)

        # Focal geometry if needed
        if self.use_focal_geo:
            p_t = torch.exp(-geo_loss_map)
            focal_weight = (1 - p_t) ** self.focal_gamma
            geo_loss_map = geo_loss_map * focal_weight

        # Mask by gt_score
        geo_loss_map = geo_loss_map * gt_score.squeeze(1)

        # OHEM or average
        if self.use_ohem:
            B = geo_loss_map.shape[0]
            geo_loss = 0.0
            for b in range(B):
                flat = geo_loss_map[b].view(-1)
                k = max(int(self.ohem_ratio * flat.numel()), 1)
                topk, _ = torch.topk(flat, k=k, largest=True)
                geo_loss += torch.mean(topk)
            geo_loss = geo_loss / B
        else:
            geo_loss = torch.sum(geo_loss_map) / (torch.sum(gt_score) + 1e-5)

        return dice + geo_loss
