"""Spatial field to area-averaged scalar target conversion with mask and weight tracking."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def compute_area_averaged_wind_speed(
    field: np.ndarray,
    mask: np.ndarray,
    resolution_m: Optional[Tuple[float, float]] = None,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Computes canopy area-averaged wind speed over outdoor fluid cells.

    Strictly preserves physical units (m/s) and guards against solid-interior bias,
    cell-area distortion, and NaN propagation.

    Parameters
    ----------
    field : np.ndarray
        2D spatial scalar wind field (Ny, Nx) in physical units (m/s).
    mask : np.ndarray
        2D boolean building footprint mask (Ny, Nx), where True indicates solid building.
    resolution_m : Optional[Tuple[float, float]]
        Grid resolution (dy, dx) in meters. If provided with uniform cells,
        cell areas cancel in the ratio; if weights is given, weights take precedence.
    weights : Optional[np.ndarray]
        Optional 2D array (Ny, Nx) representing non-uniform differential cell areas dA_ij in m^2.

    Returns
    -------
    float
        Area-averaged wind speed over fluid space in m/s.

    Raises
    ------
    TypeError
        If field or mask are not numpy arrays.
    ValueError
        If shapes do not match, values contain NaNs/Infs, or all cells are masked solid.
    """
    if not isinstance(field, np.ndarray):
        raise TypeError(f"field must be a numpy ndarray, got {type(field)}")
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask must be a numpy ndarray, got {type(mask)}")
    if field.shape != mask.shape:
        raise ValueError(f"Shape mismatch: field {field.shape} vs mask {mask.shape}")
    if field.ndim != 2:
        raise ValueError(f"field must be 2D (Ny, Nx), got shape {field.shape}")
    if np.any(np.isnan(field)) or np.any(np.isinf(field)):
        raise ValueError("field contains NaN or Inf values")

    fluid_mask = ~mask
    total_fluid_cells = int(np.count_nonzero(fluid_mask))
    if total_fluid_cells == 0:
        raise ValueError("Cannot compute area-averaged wind speed: domain has 0 fluid cells (100% solid building).")

    if weights is not None:
        if not isinstance(weights, np.ndarray):
            raise TypeError("weights must be a numpy ndarray")
        if weights.shape != field.shape:
            raise ValueError(f"weights shape {weights.shape} != field shape {field.shape}")
        if np.any(weights < 0.0) or np.any(np.isnan(weights)):
            raise ValueError("weights must be non-negative and finite")

        fluid_weights = weights[fluid_mask]
        total_fluid_area = float(np.sum(fluid_weights))
        if total_fluid_area <= 0.0:
            raise ValueError("Total fluid area from weights must be > 0")
        weighted_sum = float(np.sum(field[fluid_mask] * fluid_weights))
        return float(weighted_sum / total_fluid_area)

    if resolution_m is not None:
        dy, dx = resolution_m
        if dy <= 0.0 or dx <= 0.0:
            raise ValueError(f"Grid resolution must be positive: {resolution_m}")
        # Uniform cells: cell area dx * dy cancels out, but explicitly tracking verifies dimensional consistency
        cell_area = dx * dy
        fluid_values = field[fluid_mask]
        total_fluid_area = total_fluid_cells * cell_area
        weighted_integral = float(np.sum(fluid_values) * cell_area)
        return float(weighted_integral / total_fluid_area)

    # Standard unweighted cell average over fluid cells
    return float(np.mean(field[fluid_mask]))
