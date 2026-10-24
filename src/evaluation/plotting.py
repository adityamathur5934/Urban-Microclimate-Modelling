"""Visualization utilities for comparing reference CFD fields, surrogate predictions, and error maps."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt


def plot_prediction_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    layout_id: str,
    output_path: Union[str, Path],
    units: str = "m/s",
    title_suffix: str = "",
) -> Path:
    """Generates a 3-panel figure: [CFD Reference] | [Surrogate Prediction] | [Absolute Error].

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth velocity magnitude field (Ny, Nx) in physical units (m/s).
    y_pred : np.ndarray
        Predicted velocity magnitude field (Ny, Nx) in physical units (m/s).
    mask : np.ndarray
        Building footprint boolean mask (Ny, Nx), True where solid building.
    layout_id : str
        Identifier for the layout.
    output_path : Union[str, Path]
        Destination image path (.png, .pdf, or .svg).
    units : str
        Physical units label for colorbars (default 'm/s').
    title_suffix : str
        Optional subtitle text.

    Returns
    -------
    Path
        Path to saved figure.
    """
    if y_true.ndim == 3:
        y_true = y_true[0]
    if y_pred.ndim == 3:
        y_pred = y_pred[0]
    if mask.ndim == 3:
        mask = mask[0]

    # Mask solid buildings as NaN for distinct colormap background
    true_masked = np.where(mask, np.nan, y_true)
    pred_masked = np.where(mask, np.nan, y_pred)
    abs_err = np.abs(y_true - y_pred)
    err_masked = np.where(mask, np.nan, abs_err)

    # Determine unified color scaling for flow fields
    valid_vals = true_masked[~np.isnan(true_masked)]
    vmin = 0.0
    vmax = float(np.max(valid_vals)) if len(valid_vals) > 0 else 1.0

    err_max = float(np.nanmax(err_masked)) if not np.all(np.isnan(err_masked)) else 1.0
    err_max = max(err_max, 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # 1. Ground Truth CFD Reference
    im0 = axes[0].imshow(true_masked, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    axes[0].set_title(f"CFD Reference ({layout_id})\n$z = 1.75$ m", fontsize=12, fontweight="bold")
    axes[0].set_facecolor("#808080")  # Dark gray for solid buildings
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label(f"Wind Speed [{units}]", fontsize=10)

    # 2. Neural Surrogate Prediction
    im1 = axes[1].imshow(pred_masked, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    axes[1].set_title(f"Neural Surrogate (U-Net)\n$z = 1.75$ m", fontsize=12, fontweight="bold")
    axes[1].set_facecolor("#808080")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label(f"Wind Speed [{units}]", fontsize=10)

    # 3. Absolute Error Field
    im2 = axes[2].imshow(err_masked, cmap="magma", vmin=0.0, vmax=err_max, origin="lower")
    axes[2].set_title(f"Absolute Error Field $|U_{{true}} - U_{{pred}}|$\nMAE = {float(np.nanmean(err_masked)):.3f} {units}", fontsize=12, fontweight="bold")
    axes[2].set_facecolor("#808080")
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label(f"Absolute Error [{units}]", fontsize=10)

    for ax in axes:
        ax.set_xlabel("X coordinate [grid cells]", fontsize=9)
        ax.set_ylabel("Y coordinate [grid cells]", fontsize=9)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out
