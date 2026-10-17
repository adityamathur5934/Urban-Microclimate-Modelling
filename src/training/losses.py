"""Physics-informed and outdoor-masked regression loss functions for neural surrogates.

Ensures gradients are computed exclusively over the physical fluid domain (outdoor space),
preventing artificial loss damping from solid zero-flow building interiors.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedL1Loss(nn.Module):
    """Mean Absolute Error computed strictly over outdoor fluid cells.

    Loss = (1 / N_outdoor) * sum(|pred - target| * fluid_mask)
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fluid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        pred: (B, C, H, W)
        target: (B, C, H, W)
        fluid_mask: Optional (B, 1, H, W) or (B, C, H, W) where 1=fluid, 0=building
        """
        diff = torch.abs(pred - target)
        if fluid_mask is None:
            return diff.mean()

        masked_diff = diff * fluid_mask
        denominator = fluid_mask.sum().clamp(min=self.eps)
        return masked_diff.sum() / denominator


class MaskedMSELoss(nn.Module):
    """Mean Squared Error computed strictly over outdoor fluid cells.

    Loss = (1 / N_outdoor) * sum((pred - target)^2 * fluid_mask)
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fluid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff_sq = (pred - target) ** 2
        if fluid_mask is None:
            return diff_sq.mean()

        masked_diff_sq = diff_sq * fluid_mask
        denominator = fluid_mask.sum().clamp(min=self.eps)
        return masked_diff_sq.sum() / denominator


class CombinedSurrogateLoss(nn.Module):
    """Composite loss balancing L1 sharpness, MSE peak penalty, and spatial gradient continuity.

    Total Loss = alpha * MaskedL1 + (1 - alpha) * MaskedMSE + grad_weight * GradientDifference
    """

    def __init__(
        self,
        alpha: float = 0.7,
        grad_weight: float = 0.1,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.alpha = alpha
        self.grad_weight = grad_weight
        self.l1_loss = MaskedL1Loss(eps=eps)
        self.mse_loss = MaskedMSELoss(eps=eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fluid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss_l1 = self.l1_loss(pred, target, fluid_mask)
        loss_mse = self.mse_loss(pred, target, fluid_mask)
        total_loss = self.alpha * loss_l1 + (1.0 - self.alpha) * loss_mse

        if self.grad_weight > 0.0 and pred.shape[-1] > 2 and pred.shape[-2] > 2:
            # Finite difference gradient penalty along x and y
            pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            targ_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
            pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            targ_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

            mask_dx = fluid_mask[:, :, :, 1:] * fluid_mask[:, :, :, :-1] if fluid_mask is not None else None
            mask_dy = fluid_mask[:, :, 1:, :] * fluid_mask[:, :, :-1, :] if fluid_mask is not None else None

            grad_loss_x = self.l1_loss(pred_dx, targ_dx, mask_dx)
            grad_loss_y = self.l1_loss(pred_dy, targ_dy, mask_dy)
            total_loss = total_loss + self.grad_weight * (grad_loss_x + grad_loss_y)

        return total_loss
