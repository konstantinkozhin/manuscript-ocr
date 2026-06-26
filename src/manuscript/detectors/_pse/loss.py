import torch
import torch.nn as nn


def dice_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    b = pred.shape[0]
    pred = (pred * mask).contiguous().view(b, -1)
    target = (target.float() * mask).contiguous().view(b, -1)
    a = torch.sum(pred * target, dim=1)
    b_sum = torch.sum(pred * pred, dim=1) + eps
    c_sum = torch.sum(target * target, dim=1) + eps
    return torch.mean(1.0 - (2.0 * a) / (b_sum + c_sum))


def ohem_batch(scores: torch.Tensor, gt_texts: torch.Tensor, training_masks: torch.Tensor) -> torch.Tensor:
    selected_masks = []
    for score, gt_text, training_mask in zip(scores, gt_texts, training_masks):
        pos_num = int(torch.sum(gt_text > 0.5)) - int(
            torch.sum((gt_text > 0.5) & (training_mask <= 0.5))
        )
        if pos_num <= 0:
            selected_masks.append(training_mask.float())
            continue
        neg_num = int(torch.sum(gt_text <= 0.5))
        neg_num = min(pos_num * 3, neg_num)
        if neg_num <= 0:
            selected_masks.append(training_mask.float())
            continue
        neg_score = score[gt_text <= 0.5]
        neg_score_sorted, _ = torch.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]
        selected = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
        selected_masks.append(selected.float())
    return torch.stack(selected_masks, dim=0)


class PSELoss(nn.Module):
    """PSENet loss: OHEM Dice for text, Dice over all kernel maps."""

    def __init__(self, text_weight: float = 0.7, kernel_weight: float = 0.3):
        super().__init__()
        self.text_weight = float(text_weight)
        self.kernel_weight = float(kernel_weight)
        self.last_losses = {}

    def forward(
        self,
        pred_logits: torch.Tensor,
        gt_text: torch.Tensor,
        gt_kernels: torch.Tensor,
        training_mask: torch.Tensor,
    ) -> torch.Tensor:
        texts = pred_logits[:, 0]
        kernels = pred_logits[:, 1:]
        selected_masks = ohem_batch(texts.detach(), gt_text, training_mask)
        loss_text = dice_loss_with_logits(texts, gt_text, selected_masks)

        kernel_mask = gt_text.float() * training_mask.float()
        kernel_losses = []
        for idx in range(kernels.shape[1]):
            kernel_losses.append(
                dice_loss_with_logits(kernels[:, idx], gt_kernels[:, idx], kernel_mask)
            )
        loss_kernel = torch.mean(torch.stack(kernel_losses))
        total = self.text_weight * loss_text + self.kernel_weight * loss_kernel
        self.last_losses = {
            "text": float(loss_text.detach().cpu()),
            "kernel": float(loss_kernel.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
        return total
