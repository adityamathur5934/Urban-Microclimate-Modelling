"""Visualization utilities for microclimate fields, training curves, and model comparison.

Produces clean, publication-ready figures displaying CFD ground truth, neural surrogate
predictions, error contours, and tabular AML baseline diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_field_comparison(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    building_mask: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    layout_id: str = "layout",
    target_name: str = "Wind Speed (m/s)",
    colormap: str = "viridis",
    figsize: Tuple[int, int] = (14, 4),
) -> plt.Figure:
    """Generates a 3-panel comparison: Ground Truth, Surrogate Prediction, and Absolute Error Map.

    Parameters:
      ground_truth: 2D array (Ny, Nx) of CFD ground truth.
      prediction: 2D array (Ny, Nx) of neural surrogate prediction.
      building_mask: 2D boolean array (Ny, Nx) where True indicates building interior.
      save_path: Optional path to save figure.
      layout_id: Layout identifier string for title.
      target_name: Display label for the colorbar.
    """
    gt = ground_truth.copy().squeeze()
    pred = prediction.copy().squeeze()
    mask = building_mask.copy().squeeze().astype(bool)

    abs_error = np.abs(gt - pred)

    # Mask solid buildings with NaN for clean visualization
    gt_plot = np.where(mask, np.nan, gt)
    pred_plot = np.where(mask, np.nan, pred)
    err_plot = np.where(mask, np.nan, abs_error)

    # Consistent color limits across Ground Truth and Prediction
    vmin = min(np.nanmin(gt_plot), np.nanmin(pred_plot))
    vmax = max(np.nanmax(gt_plot), np.nanmax(pred_plot))
    err_max = np.nanmax(err_plot)

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # 1. Ground Truth
    im0 = axes[0].imshow(gt_plot, origin="lower", cmap=colormap, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"CFD Ground Truth ({layout_id})", fontsize=11, fontweight="bold")
    axes[0].set_facecolor("#222222")  # dark gray for solid buildings
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label(target_name, fontsize=9)

    # 2. Neural Surrogate Prediction
    im1 = axes[1].imshow(pred_plot, origin="lower", cmap=colormap, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Neural Surrogate Prediction", fontsize=11, fontweight="bold")
    axes[1].set_facecolor("#222222")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label(target_name, fontsize=9)

    # 3. Absolute Error Map
    im2 = axes[2].imshow(err_plot, origin="lower", cmap="magma", vmin=0, vmax=max(err_max, 1e-3))
    axes[2].set_title(f"Absolute Error |GT - Pred|", fontsize=11, fontweight="bold")
    axes[2].set_facecolor("#222222")
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label(f"Absolute Error ({target_name})", fontsize=9)

    for ax in axes:
        ax.set_xlabel("x (grid cells)", fontsize=9)
        ax.set_ylabel("y (grid cells)", fontsize=9)

    if save_path:
        dest = Path(save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=200, bbox_inches="tight")

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    """Plots training and validation loss and validation NRMSE curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Panel 1: Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    if "val_loss" in history and history["val_loss"]:
        val_epochs = range(1, len(history["val_loss"]) + 1)
        ax1.plot(val_epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss (Masked Composite)", fontsize=10)
    ax1.set_title("Training & Validation Loss", fontsize=11, fontweight="bold")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Validation Metrics
    if "val_nrmse" in history and history["val_nrmse"]:
        val_epochs = range(1, len(history["val_nrmse"]) + 1)
        ax2.plot(val_epochs, history["val_nrmse"], label="Val NRMSE (Range)", color="#2ca02c", lw=2)
    if "val_mae" in history and history["val_mae"]:
        val_epochs = range(1, len(history["val_mae"]) + 1)
        ax2.plot(val_epochs, history["val_mae"], label="Val MAE (m/s)", color="#d62728", lw=1.5, linestyle=":")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Validation Metric", fontsize=10)
    ax2.set_title("Validation Model Selection Metrics", fontsize=11, fontweight="bold")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    if save_path:
        dest = Path(save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=200, bbox_inches="tight")

    return fig


def plot_baseline_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    target_name: str = "Area-Averaged Wind Speed (m/s)",
    figsize: Tuple[int, int] = (6, 6),
) -> plt.Figure:
    """Scatter plot of AML baseline predictions against CFD ground truth with 1:1 reference line."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()

    ax.scatter(yt, yp, alpha=0.8, color="#2b5c8f", edgecolors="black", linewidth=0.5, s=40, label="Layouts")

    # 1:1 Identity reference line
    min_val = min(float(np.min(yt)), float(np.min(yp))) * 0.95
    max_val = max(float(np.max(yt)), float(np.max(yp))) * 1.05
    ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label="1:1 Perfect Agreement")

    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    ax.text(
        0.05,
        0.90,
        f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nN = {len(yt)}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel(f"CFD Ground Truth {target_name}", fontsize=10)
    ax.set_ylabel(f"Predicted {target_name}", fontsize=10)
    ax.set_title("AML Baseline: Predicted vs. CFD Ground Truth", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    if save_path:
        dest = Path(save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=200, bbox_inches="tight")

    return fig
