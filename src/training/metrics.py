"""Evaluation metrics for spatial microclimate fields and tabular AML baselines.

Provides rigorously documented formulas for:
  - Masked Field MAE
  - Masked Field RMSE
  - Normalized RMSE (NRMSE) under multiple documented normalization conventions
  - Spatial field area-averaging bridge for comparing DL and AML outputs
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch


def field_mae(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """Computes Mean Absolute Error over outdoor fluid cells.

    Formula:
        MAE = (1 / N_outdoor) * sum_{i in outdoor} |pred_i - target_i|
    """
    p = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
    t = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else np.asarray(target)

    if fluid_mask is not None:
        m = fluid_mask.detach().cpu().numpy() if isinstance(fluid_mask, torch.Tensor) else np.asarray(fluid_mask)
        m = m.astype(bool)
        if not np.any(m):
            return 0.0
        return float(np.mean(np.abs(p[m] - t[m])))

    return float(np.mean(np.abs(p - t)))


def field_rmse(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """Computes Root Mean Squared Error over outdoor fluid cells.

    Formula:
        RMSE = sqrt((1 / N_outdoor) * sum_{i in outdoor} (pred_i - target_i)^2)
    """
    p = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
    t = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else np.asarray(target)

    if fluid_mask is not None:
        m = fluid_mask.detach().cpu().numpy() if isinstance(fluid_mask, torch.Tensor) else np.asarray(fluid_mask)
        m = m.astype(bool)
        if not np.any(m):
            return 0.0
        return float(np.sqrt(np.mean((p[m] - t[m]) ** 2)))

    return float(np.sqrt(np.mean((p - t) ** 2)))


def field_nrmse(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    normalization: str = "range",
    inflow_speed: Optional[float] = None,
) -> float:
    """Computes Normalized Root Mean Squared Error (NRMSE).

    Different literature uses different normalizing denominators. This function supports
    and makes explicit the three standard definitions:

    1. normalization == "range" (Default):
       NRMSE_range = RMSE / (y_max - y_min)
       Measures error as a fraction of the dynamic range observed in outdoor flow.

    2. normalization == "inflow":
       NRMSE_inflow = RMSE / U_ref
       Standard in wind engineering and aerodynamics: normalizes error by the approach wind speed.
       Requires `inflow_speed` to be specified.

    3. normalization == "std":
       NRMSE_std = RMSE / std(y)
       Statistical normalization: equivalent to sqrt(1 - R^2) when predictions are unbiased.

    4. normalization == "mean":
       NRMSE_mean = RMSE / mean(y)
    """
    rmse = field_rmse(pred, target, fluid_mask=fluid_mask)
    t = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else np.asarray(target)

    if fluid_mask is not None:
        m = fluid_mask.detach().cpu().numpy() if isinstance(fluid_mask, torch.Tensor) else np.asarray(fluid_mask)
        m = m.astype(bool)
        t_valid = t[m] if np.any(m) else t
    else:
        t_valid = t

    if normalization == "range":
        denom = float(np.ptp(t_valid))  # max - min
    elif normalization == "inflow":
        if inflow_speed is None or inflow_speed <= 0.0:
            raise ValueError("inflow_speed must be provided and > 0 when normalization='inflow'")
        denom = float(inflow_speed)
    elif normalization == "std":
        denom = float(np.std(t_valid))
    elif normalization == "mean":
        denom = float(np.abs(np.mean(t_valid)))
    else:
        raise ValueError(
            f"Unknown normalization '{normalization}'. Supported: 'range', 'inflow', 'std', 'mean'"
        )

    if denom <= 1e-8:
        return 0.0
    return float(rmse / denom)


def field_max_abs_error(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """Computes maximum absolute point error over outdoor fluid cells."""
    p = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
    t = target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else np.asarray(target)

    if fluid_mask is not None:
        m = fluid_mask.detach().cpu().numpy() if isinstance(fluid_mask, torch.Tensor) else np.asarray(fluid_mask)
        m = m.astype(bool)
        if not np.any(m):
            return 0.0
        return float(np.max(np.abs(p[m] - t[m])))

    return float(np.max(np.abs(p - t)))


def aggregate_spatial_field_to_scalar(
    field_2d: Union[np.ndarray, torch.Tensor],
    building_mask: Union[np.ndarray, torch.Tensor],
) -> float:
    """Aggregates a 2D spatial field to an area-averaged scalar over outdoor fluid cells.

    This function provides the mathematically compatible representation bridge between
    pixel-level neural surrogate predictions and layout-level AML baseline predictions.

    Formula:
        area_avg = (1 / |Omega_outdoor|) * sum_{(y, x) in Omega_outdoor} field(y, x)
    """
    f = field_2d.detach().cpu().numpy() if isinstance(field_2d, torch.Tensor) else np.asarray(field_2d)
    m = building_mask.detach().cpu().numpy() if isinstance(building_mask, torch.Tensor) else np.asarray(building_mask)

    if f.ndim == 3 and f.shape[0] == 1:
        f = f[0]
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]

    outdoor_mask = (~m.astype(bool))
    if not np.any(outdoor_mask):
        return 0.0
    return float(np.mean(f[outdoor_mask]))


def compute_surrogate_metrics_summary(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    inflow_speed: Optional[float] = None,
) -> Dict[str, float]:
    """Computes full suite of field-level regression metrics."""
    metrics = {
        "field_mae": field_mae(pred, target, fluid_mask=fluid_mask),
        "field_rmse": field_rmse(pred, target, fluid_mask=fluid_mask),
        "field_nrmse_range": field_nrmse(pred, target, fluid_mask=fluid_mask, normalization="range"),
        "field_nrmse_std": field_nrmse(pred, target, fluid_mask=fluid_mask, normalization="std"),
        "field_max_error": field_max_abs_error(pred, target, fluid_mask=fluid_mask),
    }
    if inflow_speed is not None and inflow_speed > 0:
        metrics["field_nrmse_inflow"] = field_nrmse(
            pred, target, fluid_mask=fluid_mask, normalization="inflow", inflow_speed=inflow_speed
        )
    return metrics
