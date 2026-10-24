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
    """Mean Absolute Error computed strictly over outdoor fluid cells."""

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        fluid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Accepts either `fluid_mask` (1 = fluid, 0 = building)
        or `mask` (boolean building footprint where True = solid building).
        """
        diff = torch.abs(pred - target)
        if fluid_mask is None and mask is None:
            return diff.mean()

        if fluid_mask is not None:
            f_weight = fluid_mask.float()
        else:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            f_weight = (~mask).float()

        if f_weight.shape != pred.shape:
            f_weight = f_weight.expand_as(pred)

        masked_diff = diff * f_weight
        denominator = f_weight.sum().clamp(min=self.eps)
        return masked_diff.sum() / denominator


class MaskedMSELoss(nn.Module):
    """Mean Squared Error computed strictly over outdoor fluid cells."""

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        fluid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff_sq = (pred - target) ** 2
        if fluid_mask is None and mask is None:
            return diff_sq.mean()

        if fluid_mask is not None:
            f_weight = fluid_mask.float()
        else:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            f_weight = (~mask).float()

        if f_weight.shape != pred.shape:
            f_weight = f_weight.expand_as(pred)

        masked_diff_sq = diff_sq * f_weight
        denominator = f_weight.sum().clamp(min=self.eps)
        return masked_diff_sq.sum() / denominator


class MaskedGradientLoss(nn.Module):
    """First-order spatial finite difference gradient loss to penalize blurry shear layers."""

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        fluid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if fluid_mask is not None:
            solid_mask = (fluid_mask <= 0.5)
        elif mask is not None:
            solid_mask = mask.bool()
        else:
            solid_mask = torch.zeros_like(pred, dtype=torch.bool)

        if solid_mask.ndim == 3:
            solid_mask = solid_mask.unsqueeze(1)

        pred_grad_x = pred[..., :, 1:] - pred[..., :, :-1]
        target_grad_x = target[..., :, 1:] - target[..., :, :-1]
        mask_x = solid_mask[..., :, 1:] | solid_mask[..., :, :-1]
        fluid_mask_x = (~mask_x).expand_as(pred_grad_x).float()

        pred_grad_y = pred[..., 1:, :] - pred[..., :-1, :]
        target_grad_y = target[..., 1:, :] - target[..., :-1, :]
        mask_y = solid_mask[..., 1:, :] | solid_mask[..., :-1, :]
        fluid_mask_y = (~mask_y).expand_as(pred_grad_y).float()

        loss_x = (torch.abs(pred_grad_x - target_grad_x) * fluid_mask_x).sum() / fluid_mask_x.sum().clamp(min=self.eps)
        loss_y = (torch.abs(pred_grad_y - target_grad_y) * fluid_mask_y).sum() / fluid_mask_y.sum().clamp(min=self.eps)
        return 0.5 * (loss_x + loss_y)


class CombinedSurrogateLoss(nn.Module):
    """Composite loss balancing L1 sharpness, MSE peak penalty, and spatial gradient continuity."""

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
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if fluid_mask is None and mask is not None:
            fluid_mask = (~mask.bool()).float()

        loss_l1 = self.l1_loss(pred, target, fluid_mask=fluid_mask)
        loss_mse = self.mse_loss(pred, target, fluid_mask=fluid_mask)
        total_loss = self.alpha * loss_l1 + (1.0 - self.alpha) * loss_mse

        if self.grad_weight > 0.0 and pred.shape[-1] > 2 and pred.shape[-2] > 2:
            pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            targ_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
            pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            targ_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

            mask_dx = fluid_mask[:, :, :, 1:] * fluid_mask[:, :, :, :-1] if fluid_mask is not None else None
            mask_dy = fluid_mask[:, :, 1:, :] * fluid_mask[:, :, :-1, :] if fluid_mask is not None else None

            grad_loss_x = self.l1_loss(pred_dx, targ_dx, fluid_mask=mask_dx)
            grad_loss_y = self.l1_loss(pred_dy, targ_dy, fluid_mask=mask_dy)
            total_loss = total_loss + self.grad_weight * (grad_loss_x + grad_loss_y)

        return total_loss


class CompositeSurrogateLoss(nn.Module):
    """Composite objective combining Masked L1, optional MSE, and Spatial Gradient losses."""

    def __init__(
        self,
        alpha_l1: float = 1.0,
        alpha_mse: float = 0.0,
        alpha_grad: float = 0.2,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.alpha_l1 = alpha_l1
        self.alpha_mse = alpha_mse
        self.alpha_grad = alpha_grad

        self.l1_loss = MaskedL1Loss(eps=eps)
        self.mse_loss = MaskedMSELoss(eps=eps)
        self.grad_loss = MaskedGradientLoss(eps=eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        fluid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = 0.0
        if self.alpha_l1 > 0.0:
            loss = loss + self.alpha_l1 * self.l1_loss(pred, target, mask=mask, fluid_mask=fluid_mask)
        if self.alpha_mse > 0.0:
            loss = loss + self.alpha_mse * self.mse_loss(pred, target, mask=mask, fluid_mask=fluid_mask)
        if self.alpha_grad > 0.0:
            loss = loss + self.alpha_grad * self.grad_loss(pred, target, mask=mask, fluid_mask=fluid_mask)
        return loss
