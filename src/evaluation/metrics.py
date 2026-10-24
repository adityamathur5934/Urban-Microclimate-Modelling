"""Quantitative spatial field evaluation metrics strictly evaluated over fluid cells."""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np


def _extract_fluid_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Helper to extract 1D fluid-only arrays from 2D or 3D fields."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if y_true.ndim == 3 and y_true.shape[0] == 1:
        y_true = y_true[0]
        y_pred = y_pred[0]

    fluid_mask = ~mask.astype(bool)
    if not np.any(fluid_mask):
        raise ValueError("Cannot compute metrics: domain contains 0 fluid cells.")

    return y_true[fluid_mask].flatten(), y_pred[fluid_mask].flatten()


def compute_field_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """Mean Absolute Error over fluid cells [m/s]."""
    true_f, pred_f = _extract_fluid_arrays(y_true, y_pred, mask)
    return float(np.mean(np.abs(true_f - pred_f)))


def compute_field_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """Root Mean Squared Error over fluid cells [m/s]."""
    true_f, pred_f = _extract_fluid_arrays(y_true, y_pred, mask)
    return float(np.sqrt(np.mean((true_f - pred_f) ** 2)))


def compute_normalized_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    u_ref: Optional[float] = None,
) -> float:
    """Normalized RMSE expressed as a percentage [%].

    Normalized by inflow reference wind speed u_ref if provided,
    otherwise by the fluid range max(y_true) - min(y_true).
    """
    rmse = compute_field_rmse(y_true, y_pred, mask)
    if u_ref is not None and u_ref > 1e-4:
        return float((rmse / u_ref) * 100.0)

    true_f, _ = _extract_fluid_arrays(y_true, y_pred, mask)
    val_range = float(np.max(true_f) - np.min(true_f))
    if val_range < 1e-4:
        val_range = 1.0
    return float((rmse / val_range) * 100.0)


def compute_relative_l2(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """Relative L2 error norm over fluid cells: ||y_true - y_pred||_2 / ||y_true||_2."""
    true_f, pred_f = _extract_fluid_arrays(y_true, y_pred, mask)
    norm_true = np.linalg.norm(true_f)
    if norm_true < 1e-6:
        norm_true = 1.0
    return float(np.linalg.norm(true_f - pred_f) / norm_true)


def compute_r2_score(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """Coefficient of Determination (R^2) over fluid cells."""
    true_f, pred_f = _extract_fluid_arrays(y_true, y_pred, mask)
    ss_res = float(np.sum((true_f - pred_f) ** 2))
    ss_tot = float(np.sum((true_f - np.mean(true_f)) ** 2))
    if ss_tot < 1e-6:
        return 1.0 if ss_res < 1e-6 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_spatial_error_map(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Computes 2D absolute error map |y_true - y_pred| with solid cells set to 0.0."""
    if y_true.ndim == 3:
        y_true = y_true[0]
        y_pred = y_pred[0]
    if mask.ndim == 3:
        mask = mask[0]

    err_map = np.abs(y_true - y_pred).astype(np.float32)
    err_map[mask] = 0.0
    return err_map
