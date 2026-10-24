"""Physics-aware masked loss functions for urban aerodynamic surrogate training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedL1Loss(nn.Module):
    """Mean Absolute Error computed exclusively over outdoor fluid cells."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Parameters

        ----------
        pred : torch.Tensor
            Predicted field (B, C, Ny, Nx).
        target : torch.Tensor
            Ground truth field (B, C, Ny, Nx).
        mask : torch.Tensor
            Building footprint mask (B, 1, Ny, Nx) or (B, Ny, Nx), True where solid.
        """
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        fluid_mask = (~mask).expand_as(pred)

        diff = torch.abs(pred - target)
        fluid_diff = diff * fluid_mask.float()
        num_fluid = fluid_mask.float().sum()
        return fluid_diff.sum() / (num_fluid + self.eps)


class MaskedMSELoss(nn.Module):
    """Mean Squared Error computed exclusively over outdoor fluid cells."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        fluid_mask = (~mask).expand_as(pred)

        diff_sq = (pred - target) ** 2
        fluid_diff_sq = diff_sq * fluid_mask.float()
        num_fluid = fluid_mask.float().sum()
        return fluid_diff_sq.sum() / (num_fluid + self.eps)


class MaskedGradientLoss(nn.Module):
    """First-order spatial finite difference gradient loss to penalize blurry shear layers."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        # Compute finite differences along X and Y axes
        pred_grad_x = pred[..., :, 1:] - pred[..., :, :-1]
        target_grad_x = target[..., :, 1:] - target[..., :, :-1]
        mask_x = mask[..., :, 1:] | mask[..., :, :-1]
        fluid_mask_x = (~mask_x).expand_as(pred_grad_x).float()

        pred_grad_y = pred[..., 1:, :] - pred[..., :-1, :]
        target_grad_y = target[..., 1:, :] - target[..., :-1, :]
        mask_y = mask[..., 1:, :] | mask[..., :-1, :]
        fluid_mask_y = (~mask_y).expand_as(pred_grad_y).float()

        loss_x = (torch.abs(pred_grad_x - target_grad_x) * fluid_mask_x).sum() / (fluid_mask_x.sum() + self.eps)
        loss_y = (torch.abs(pred_grad_y - target_grad_y) * fluid_mask_y).sum() / (fluid_mask_y.sum() + self.eps)
        return 0.5 * (loss_x + loss_y)


class CompositeSurrogateLoss(nn.Module):
    """Composite objective combining Masked L1, optional MSE, and Spatial Gradient losses."""

    def __init__(
        self,
        alpha_l1: float = 1.0,
        alpha_mse: float = 0.0,
        alpha_grad: float = 0.2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.alpha_l1 = alpha_l1
        self.alpha_mse = alpha_mse
        self.alpha_grad = alpha_grad

        self.l1_loss = MaskedL1Loss(eps=eps)
        self.mse_loss = MaskedMSELoss(eps=eps)
        self.grad_loss = MaskedGradientLoss(eps=eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.alpha_l1 > 0.0:
            loss = loss + self.alpha_l1 * self.l1_loss(pred, target, mask)
        if self.alpha_mse > 0.0:
            loss = loss + self.alpha_mse * self.mse_loss(pred, target, mask)
        if self.alpha_grad > 0.0:
            loss = loss + self.alpha_grad * self.grad_loss(pred, target, mask)
        return loss
