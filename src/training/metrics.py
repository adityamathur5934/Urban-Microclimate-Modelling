"""Evaluation metrics for spatial microclimate fields and tabular AML baselines.

Provides rigorously documented formulas for:
  - Masked Field MAE
  - Masked Field RMSE
  - Normalized RMSE (NRMSE) under multiple documented normalization conventions
  - Spatial field area-averaging bridge for comparing DL and AML outputs

Reuses the core evaluation logic in `src.evaluation.metrics`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch

from src.evaluation.metrics import (
    compute_field_mae,
    compute_field_rmse,
    compute_normalized_rmse,
    compute_max_error,
    _to_numpy,
)


def field_mae(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """Computes Mean Absolute Error over outdoor fluid cells.

    Formula:
        MAE = (1 / N_outdoor) * sum_{i in outdoor} |pred_i - target_i|

    Units:
        Same as target field [e.g. m/s].
    """
    return compute_field_mae(
        y_true=target,
        y_pred=pred,
        fluid_mask=fluid_mask,
        check_finite=True,
    )


def field_rmse(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """Computes Root Mean Squared Error over outdoor fluid cells.

    Formula:
        RMSE = sqrt((1 / N_outdoor) * sum_{i in outdoor} (pred_i - target_i)^2)

    Units:
        Same as target field [e.g. m/s].
    """
    return compute_field_rmse(
        y_true=target,
        y_pred=pred,
        fluid_mask=fluid_mask,
        check_finite=True,
    )


def field_nrmse(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    normalization: str = "range",
    inflow_speed: Optional[float] = None,
) -> float:
    """Computes Normalized Root Mean Squared Error (NRMSE).

    Different literature uses different normalizing denominators. This function supports
    and makes explicit the standard definitions:

    1. normalization == "range" (Default):
       NRMSE_range = RMSE / (y_max - y_min)
       Measures error as a fraction of the dynamic range observed in outdoor flow.

    2. normalization == "inflow":
       NRMSE_inflow = RMSE / U_ref
       Standard in wind engineering and aerodynamics: normalizes error by the approach wind speed.
       Requires `inflow_speed` to be specified (> 0).

    3. normalization == "std":
       NRMSE_std = RMSE / std(y)
       Statistical normalization: equivalent to sqrt(1 - R^2) when predictions are unbiased.

    4. normalization == "mean":
       NRMSE_mean = RMSE / |mean(y)|

    Returns:
        NRMSE as a dimensionless ratio [0, 1].
    """
    return compute_normalized_rmse(
        y_true=target,
        y_pred=pred,
        fluid_mask=fluid_mask,
        normalization=normalization,
        u_ref=inflow_speed,
        as_percent=False,
        check_finite=True,
    )


def field_max_abs_error(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """Computes maximum absolute point error over outdoor fluid cells."""
    return compute_max_error(
        y_true=target,
        y_pred=pred,
        fluid_mask=fluid_mask,
        check_finite=True,
    )


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
    f = _to_numpy(field_2d)
    m = _to_numpy(building_mask)

    if f.ndim == 3 and f.shape[0] == 1:
        f = f[0]
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]

    outdoor_mask = (~m.astype(bool))
    if not np.any(outdoor_mask):
        raise ValueError("Cannot aggregate over empty outdoor domain: building_mask covers 100% of cells.")
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
