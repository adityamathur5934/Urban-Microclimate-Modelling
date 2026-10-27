"""Quantitative spatial field evaluation metrics strictly evaluated over fluid cells.

This module provides rigorous, physics-aware evaluation metrics for urban microclimate
surrogates and CFD flow fields, enforcing:
  1. Strict spatial alignment between predictions, targets, and masks.
  2. Explicit, unambiguous mask semantics (fluid vs solid building cells).
  3. Strict rejection of non-finite values (NaN / Inf) in targets and predictions.
  4. Explicit error handling for degenerate / empty fluid masks.
  5. Documented denominators for all Normalized RMSE (NRMSE) conventions.
  6. Transparent units (e.g. m/s for velocity fields, dimensionless for ratios/R^2).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _to_numpy(arr: Any) -> np.ndarray:
    """Converts torch.Tensor or array-like to np.ndarray."""
    if _HAS_TORCH and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _align_and_extract_fluid_arrays(
    y_true: Union[np.ndarray, "torch.Tensor"],
    y_pred: Union[np.ndarray, "torch.Tensor"],
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    building_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    fluid_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    mask_is_fluid: bool = False,
    check_finite: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aligns spatial fields, validates shapes, handles masks, and enforces finite values.

    Args:
        y_true: Ground truth target field (e.g. CFD velocity magnitude in m/s).
        y_pred: Predicted target field (e.g. surrogate output in m/s).
        mask: Generic spatial mask. By default (mask_is_fluid=False), treated as
            building_mask (True for solid buildings, False for outdoor fluid cells).
            If mask_is_fluid=True, treated as fluid_mask (True for outdoor fluid).
        building_mask: Explicit building mask (True for solid, False for fluid).
        fluid_mask: Explicit fluid mask (True for fluid, False for solid).
        mask_is_fluid: If True and `mask` is provided, interprets `mask` as fluid_mask.
        check_finite: If True, raises ValueError if NaN or Inf are detected in evaluated cells.

    Returns:
        (true_fluid_1d, pred_fluid_1d): 1D float64 arrays of values over valid fluid cells.

    Raises:
        ValueError: If spatial dimensions mismatch, non-finite values exist,
            mask contains NaNs, or mask has 0 valid fluid cells.
    """
    yt = _to_numpy(y_true).astype(np.float64)
    yp = _to_numpy(y_pred).astype(np.float64)

    # Squeeze leading singleton batch/channel dimensions to align spatial shapes
    while yt.ndim > 2 and yt.shape[0] == 1:
        yt = yt[0]
    while yp.ndim > 2 and yp.shape[0] == 1:
        yp = yp[0]

    # Verify spatial alignment between target and prediction
    if yt.shape != yp.shape:
        raise ValueError(
            f"Spatial alignment error: y_true shape {yt.shape} does not match y_pred shape {yp.shape}. "
            f"Fields must share identical spatial grid dimensions before metric evaluation."
        )

    # Resolve mask semantics
    resolved_fluid_mask: Optional[np.ndarray] = None

    if fluid_mask is not None:
        fm = _to_numpy(fluid_mask)
        while fm.ndim > 2 and fm.shape[0] == 1:
            fm = fm[0]
        if fm.shape != yt.shape:
            raise ValueError(
                f"Spatial alignment error: fluid_mask shape {fm.shape} does not match target shape {yt.shape}."
            )
        resolved_fluid_mask = fm.astype(bool)

    elif building_mask is not None:
        bm = _to_numpy(building_mask)
        while bm.ndim > 2 and bm.shape[0] == 1:
            bm = bm[0]
        if bm.shape != yt.shape:
            raise ValueError(
                f"Spatial alignment error: building_mask shape {bm.shape} does not match target shape {yt.shape}."
            )
        resolved_fluid_mask = ~bm.astype(bool)

    elif mask is not None:
        m = _to_numpy(mask)
        while m.ndim > 2 and m.shape[0] == 1:
            m = m[0]
        if m.shape != yt.shape:
            raise ValueError(
                f"Spatial alignment error: mask shape {m.shape} does not match target shape {yt.shape}."
            )
        # Check if mask itself contains NaNs
        if np.issubdtype(m.dtype, np.floating) and np.isnan(m).any():
            raise ValueError("Mask contains NaN values; spatial mask must be boolean or finite.")
        resolved_fluid_mask = m.astype(bool) if mask_is_fluid else ~m.astype(bool)

    else:
        # No mask provided: all cells are valid fluid cells
        resolved_fluid_mask = np.ones(yt.shape, dtype=bool)

    # Empty mask check
    num_valid = int(np.count_nonzero(resolved_fluid_mask))
    if num_valid == 0:
        raise ValueError(
            "Empty mask error: domain contains 0 valid fluid cells. "
            "Cannot compute evaluation metrics over an empty spatial domain."
        )

    true_f = yt[resolved_fluid_mask].flatten()
    pred_f = yp[resolved_fluid_mask].flatten()

    # Non-finite verification over evaluated valid cells
    if check_finite:
        nan_true = int(np.count_nonzero(np.isnan(true_f)))
        inf_true = int(np.count_nonzero(np.isinf(true_f)))
        if nan_true > 0 or inf_true > 0:
            raise ValueError(
                f"Non-finite values detected in ground truth target: {nan_true} NaNs, {inf_true} Infs "
                f"in evaluated fluid cells. Targets must be finite real numbers."
            )

        nan_pred = int(np.count_nonzero(np.isnan(pred_f)))
        inf_pred = int(np.count_nonzero(np.isinf(pred_f)))
        if nan_pred > 0 or inf_pred > 0:
            raise ValueError(
                f"Non-finite values detected in predictions: {nan_pred} NaNs, {inf_pred} Infs "
                f"in evaluated fluid cells. Model predictions must be finite real numbers."
            )

    return true_f, pred_f


# ============================================================================
# Quantitative Regression Metrics
# ============================================================================

def compute_field_mae(
    y_true: Union[np.ndarray, "torch.Tensor"],
    y_pred: Union[np.ndarray, "torch.Tensor"],
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    building_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    fluid_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    mask_is_fluid: bool = False,
    check_finite: bool = True,
) -> float:
    """Mean Absolute Error (MAE) evaluated strictly over outdoor fluid cells.

    Equation:
        MAE = (1 / N_fluid) * sum_{i in fluid} |y_true_i - y_pred_i|

    Units:
        Same units as target field [e.g. m/s for velocity, m^2/s^2 for TKE].
    """
    true_f, pred_f = _align_and_extract_fluid_arrays(
        y_true=y_true,
        y_pred=y_pred,
        mask=mask,
        building_mask=building_mask,
        fluid_mask=fluid_mask,
        mask_is_fluid=mask_is_fluid,
        check_finite=check_finite,
    )
    return float(np.mean(np.abs(true_f - pred_f)))


def compute_field_rmse(
    y_true: Union[np.ndarray, "torch.Tensor"],
    y_pred: Union[np.ndarray, "torch.Tensor"],
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    building_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    fluid_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    mask_is_fluid: bool = False,
    check_finite: bool = True,
) -> float:
    """Root Mean Squared Error (RMSE) evaluated strictly over outdoor fluid cells.

    Equation:
        RMSE = sqrt((1 / N_fluid) * sum_{i in fluid} (y_true_i - y_pred_i)^2)

    Units:
        Same units as target field [e.g. m/s for velocity, m^2/s^2 for TKE].
    """
    true_f, pred_f = _align_and_extract_fluid_arrays(
        y_true=y_true,
        y_pred=y_pred,
        mask=mask,
        building_mask=building_mask,
        fluid_mask=fluid_mask,
        mask_is_fluid=mask_is_fluid,
        check_finite=check_finite,
    )
    return float(np.sqrt(np.mean((true_f - pred_f) ** 2)))


def compute_normalized_rmse(
    y_true: Union[np.ndarray, "torch.Tensor"],
    y_pred: Union[np.ndarray, "torch.Tensor"],
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    normalization: Optional[str] = None,
    u_ref: Optional[float] = None,
    reference_scale: Optional[float] = None,
    building_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    fluid_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    mask_is_fluid: bool = False,
    as_percent: bool = True,
    check_finite: bool = True,
) -> float:
    """Normalized Root Mean Squared Error (NRMSE) with an explicitly documented denominator.

    To eliminate silent ambiguity across different literature conventions, the denominator
    must be explicitly chosen from one of the following documented definitions:

    1. 'range' (Target Dynamic Range):
       Denominator: D = max(y_true) - min(y_true) over valid fluid cells.
       Equation: NRMSE_range = RMSE / (max(y_true) - min(y_true))
       Measures error as a fraction of the observed velocity dynamic range.

    2. 'inflow' or 'reference_scale' (Aerodynamic Standard):
       Denominator: D = u_ref (or reference_scale) > 0.
       Equation: NRMSE_inflow = RMSE / U_ref
       Standard in wind engineering: normalizes error by the ambient approach wind speed.
       Requires `u_ref` or `reference_scale` to be positive.

    3. 'std' (Statistical Dispersion):
       Denominator: D = std(y_true) over valid fluid cells.
       Equation: NRMSE_std = RMSE / sigma_true
       Equivalent to sqrt(1 - R^2) when predictions are unbiased.

    4. 'mean' (Target Central Value):
       Denominator: D = |mean(y_true)| over valid fluid cells.
       Equation: NRMSE_mean = RMSE / |mu_true|

    Args:
        y_true: Ground truth target field.
        y_pred: Predicted target field.
        mask: Optional mask (building_mask by default).
        normalization: One of 'range', 'inflow', 'reference_scale', 'std', 'mean'.
            If None and u_ref/reference_scale is provided, defaults to 'inflow'.
            If None and no reference scale is provided, defaults to 'range'.
        u_ref: Ambient inflow reference wind speed [m/s].
        reference_scale: Generic positive reference normalization denominator.
        building_mask: Explicit building mask (True for solid).
        fluid_mask: Explicit fluid mask (True for fluid).
        mask_is_fluid: If True, interprets `mask` as fluid mask.
        as_percent: If True (default), returns percentage [%]. If False, returns fraction [0, 1].
        check_finite: If True, enforces finite real values.

    Returns:
        NRMSE as a float (percentage if as_percent=True, dimensionless ratio otherwise).

    Raises:
        ValueError: If normalization convention is unknown, or reference scale is non-positive.
    """
    true_f, pred_f = _align_and_extract_fluid_arrays(
        y_true=y_true,
        y_pred=y_pred,
        mask=mask,
        building_mask=building_mask,
        fluid_mask=fluid_mask,
        mask_is_fluid=mask_is_fluid,
        check_finite=check_finite,
    )
    rmse = float(np.sqrt(np.mean((true_f - pred_f) ** 2)))

    # Determine normalization convention
    ref_val = reference_scale if reference_scale is not None else u_ref

    if normalization is None:
        norm_method = "inflow" if ref_val is not None and ref_val > 0 else "range"
    else:
        norm_method = normalization.lower()

    if norm_method == "range":
        denom = float(np.ptp(true_f))  # max - min
    elif norm_method in ("inflow", "reference_scale"):
        if ref_val is None or ref_val <= 0.0:
            raise ValueError(
                f"NRMSE normalization '{norm_method}' requires a positive 'u_ref' or "
                f"'reference_scale' (> 0), received: {ref_val}."
            )
        denom = float(ref_val)
    elif norm_method == "std":
        denom = float(np.std(true_f))
    elif norm_method == "mean":
        denom = float(np.abs(np.mean(true_f)))
    else:
        raise ValueError(
            f"Unknown NRMSE normalization '{normalization}'. Supported options: "
            f"'range', 'inflow', 'reference_scale', 'std', 'mean'."
        )

    # Protect against division by zero in degenerate target fields
    if denom <= 1e-8:
        if rmse <= 1e-8:
            return 0.0
        raise ValueError(
            f"Zero denominator encountered in NRMSE (normalization='{norm_method}', denom={denom:.3e}). "
            f"Target field has near-zero variation."
        )

    ratio = rmse / denom
    return float(ratio * 100.0 if as_percent else ratio)


def compute_relative_l2(
    y_true: Union[np.ndarray, "torch.Tensor"],
    y_pred: Union[np.ndarray, "torch.Tensor"],
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    building_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    fluid_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    mask_is_fluid: bool = False,
    check_finite: bool = True,
) -> float:
    """Relative L2 error norm over fluid cells: ||y_true - y_pred||_2 / ||y_true||_2.

    Units:
        Dimensionless [fraction].
    """
    true_f, pred_f = _align_and_extract_fluid_arrays(
        y_true=y_true,
        y_pred=y_pred,
        mask=mask,
        building_mask=building_mask,
        fluid_mask=fluid_mask,
        mask_is_fluid=mask_is_fluid,
        check_finite=check_finite,
    )
    norm_true = float(np.linalg.norm(true_f))
    if norm_true < 1e-8:
        norm_pred = float(np.linalg.norm(true_f - pred_f))
        return 0.0 if norm_pred < 1e-8 else 1.0
    return float(np.linalg.norm(true_f - pred_f) / norm_true)


def compute_r2_score(
    y_true: Union[np.ndarray, "torch.Tensor"],
    y_pred: Union[np.ndarray, "torch.Tensor"],
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    building_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    fluid_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    mask_is_fluid: bool = False,
    check_finite: bool = True,
) -> float:
    """Coefficient of Determination (R^2) over fluid cells.

    Equation:
        R^2 = 1 - (sum_{i in fluid} (y_true_i - y_pred_i)^2) / (sum_{i in fluid} (y_true_i - y_mean)^2)

    Units:
        Dimensionless (1.0 = perfect prediction, <= 0.0 indicates worse than mean predictor).
    """
    true_f, pred_f = _align_and_extract_fluid_arrays(
        y_true=y_true,
        y_pred=y_pred,
        mask=mask,
        building_mask=building_mask,
        fluid_mask=fluid_mask,
        mask_is_fluid=mask_is_fluid,
        check_finite=check_finite,
    )
    ss_res = float(np.sum((true_f - pred_f) ** 2))
    ss_tot = float(np.sum((true_f - np.mean(true_f)) ** 2))
    if ss_tot < 1e-8:
        return 1.0 if ss_res < 1e-8 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_max_error(
    y_true: Union[np.ndarray, "torch.Tensor"],
    y_pred: Union[np.ndarray, "torch.Tensor"],
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    building_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    fluid_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    mask_is_fluid: bool = False,
    check_finite: bool = True,
) -> float:
    """Peak absolute point error (L_infinity) over outdoor fluid cells.

    Units:
        Same units as target field [e.g. m/s].
    """
    true_f, pred_f = _align_and_extract_fluid_arrays(
        y_true=y_true,
        y_pred=y_pred,
        mask=mask,
        building_mask=building_mask,
        fluid_mask=fluid_mask,
        mask_is_fluid=mask_is_fluid,
        check_finite=check_finite,
    )
    return float(np.max(np.abs(true_f - pred_f)))


def compute_spatial_error_map(
    y_true: Union[np.ndarray, "torch.Tensor"],
    y_pred: Union[np.ndarray, "torch.Tensor"],
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    building_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    fluid_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    mask_is_fluid: bool = False,
) -> np.ndarray:
    """Computes 2D absolute error map |y_true - y_pred| with solid building cells set to 0.0.

    Units:
        Same units as target field [e.g. m/s].
    """
    yt = _to_numpy(y_true).astype(np.float32)
    yp = _to_numpy(y_pred).astype(np.float32)

    while yt.ndim > 2 and yt.shape[0] == 1:
        yt = yt[0]
    while yp.ndim > 2 and yp.shape[0] == 1:
        yp = yp[0]

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")

    err_map = np.abs(yt - yp).astype(np.float32)

    # Determine solid building mask where error should be zeroed
    solid_mask: Optional[np.ndarray] = None
    if building_mask is not None:
        bm = _to_numpy(building_mask)
        while bm.ndim > 2 and bm.shape[0] == 1:
            bm = bm[0]
        solid_mask = bm.astype(bool)
    elif fluid_mask is not None:
        fm = _to_numpy(fluid_mask)
        while fm.ndim > 2 and fm.shape[0] == 1:
            fm = fm[0]
        solid_mask = ~fm.astype(bool)
    elif mask is not None:
        m = _to_numpy(mask)
        while m.ndim > 2 and m.shape[0] == 1:
            m = m[0]
        solid_mask = ~m.astype(bool) if mask_is_fluid else m.astype(bool)

    if solid_mask is not None:
        if solid_mask.shape != err_map.shape:
            raise ValueError(f"Mask shape {solid_mask.shape} does not match field shape {err_map.shape}")
        err_map[solid_mask] = 0.0

    return err_map


def compute_regression_metrics(
    y_true: Union[np.ndarray, "torch.Tensor"],
    y_pred: Union[np.ndarray, "torch.Tensor"],
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    building_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    fluid_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    mask_is_fluid: bool = False,
    u_ref: Optional[float] = None,
    check_finite: bool = True,
) -> Dict[str, float]:
    """Computes standard suite of regression metrics over valid fluid cells.

    Returns:
        Dict with keys:
          - mae_mps: Mean Absolute Error [m/s]
          - rmse_mps: Root Mean Squared Error [m/s]
          - nrmse_range_percent: NRMSE normalized by target range [%]
          - relative_l2: Relative L2 norm [dimensionless]
          - r2_score: Coefficient of Determination [dimensionless]
          - max_error_mps: Peak point error [m/s]
          - (optional) nrmse_inflow_percent: NRMSE normalized by u_ref [%]
    """
    metrics = {
        "mae_mps": compute_field_mae(
            y_true, y_pred, mask=mask, building_mask=building_mask,
            fluid_mask=fluid_mask, mask_is_fluid=mask_is_fluid, check_finite=check_finite,
        ),
        "rmse_mps": compute_field_rmse(
            y_true, y_pred, mask=mask, building_mask=building_mask,
            fluid_mask=fluid_mask, mask_is_fluid=mask_is_fluid, check_finite=check_finite,
        ),
        "nrmse_range_percent": compute_normalized_rmse(
            y_true, y_pred, mask=mask, building_mask=building_mask,
            fluid_mask=fluid_mask, mask_is_fluid=mask_is_fluid,
            normalization="range", as_percent=True, check_finite=check_finite,
        ),
        "relative_l2": compute_relative_l2(
            y_true, y_pred, mask=mask, building_mask=building_mask,
            fluid_mask=fluid_mask, mask_is_fluid=mask_is_fluid, check_finite=check_finite,
        ),
        "r2_score": compute_r2_score(
            y_true, y_pred, mask=mask, building_mask=building_mask,
            fluid_mask=fluid_mask, mask_is_fluid=mask_is_fluid, check_finite=check_finite,
        ),
        "max_error_mps": compute_max_error(
            y_true, y_pred, mask=mask, building_mask=building_mask,
            fluid_mask=fluid_mask, mask_is_fluid=mask_is_fluid, check_finite=check_finite,
        ),
    }

    if u_ref is not None and u_ref > 0:
        metrics["nrmse_inflow_percent"] = compute_normalized_rmse(
            y_true, y_pred, mask=mask, building_mask=building_mask,
            fluid_mask=fluid_mask, mask_is_fluid=mask_is_fluid,
            normalization="inflow", u_ref=u_ref, as_percent=True, check_finite=check_finite,
        )

    return metrics


# Aliases for backward compatibility and cross-module consistency
field_mae = compute_field_mae
field_rmse = compute_field_rmse
field_max_abs_error = compute_max_error


def field_nrmse(
    pred: Union[np.ndarray, "torch.Tensor"],
    target: Union[np.ndarray, "torch.Tensor"],
    fluid_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    normalization: str = "range",
    inflow_speed: Optional[float] = None,
    as_percent: bool = False,
) -> float:
    """Convenience alias matching training metrics signature (pred, target, fluid_mask).

    Returns dimensionless ratio by default (as_percent=False).
    """
    return compute_normalized_rmse(
        y_true=target,
        y_pred=pred,
        fluid_mask=fluid_mask,
        normalization=normalization,
        u_ref=inflow_speed,
        as_percent=as_percent,
        check_finite=True,
    )
