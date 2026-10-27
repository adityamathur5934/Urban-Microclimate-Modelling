"""Visualization utilities for comparing reference CFD fields, surrogate predictions, and error maps.

Provides publication-quality, physics-aware plotting utilities for:
  1. Ground-truth CFD reference fields.
  2. Neural surrogate / ML baseline predictions.
  3. Absolute error fields (|y_true - y_pred|).
  4. Signed error fields (y_pred - y_true) for directional bias analysis.
  5. Single-field spatial maps.

Guarantees:
  - Fair visual comparisons via identical colorbar limits (vmin, vmax) across true and pred.
  - Strict preservation of coordinate orientation (origin='lower', no unverified rotation/flipping).
  - Consistent building mask overlays rendering solid structures as visually distinct.
  - Transparent physical units, metadata labels, and data provenance tagging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for batch and CI runs
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


# Standard variable configurations: units, colormaps, and display labels
VARIABLE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "wind_speed": {
        "display_name": "Pedestrian Wind Speed",
        "units": "m/s",
        "field_cmap": "viridis",
        "error_cmap": "magma",
        "signed_cmap": "coolwarm",
    },
    "velocity_u": {
        "display_name": "Streamwise Velocity (U)",
        "units": "m/s",
        "field_cmap": "viridis",
        "error_cmap": "magma",
        "signed_cmap": "coolwarm",
    },
    "velocity_v": {
        "display_name": "Spanwise Velocity (V)",
        "units": "m/s",
        "field_cmap": "viridis",
        "error_cmap": "magma",
        "signed_cmap": "coolwarm",
    },
    "turbulent_kinetic_energy": {
        "display_name": "Turbulent Kinetic Energy (TKE)",
        "units": "m^2/s^2",
        "field_cmap": "cividis",
        "error_cmap": "magma",
        "signed_cmap": "coolwarm",
    },
    "temperature": {
        "display_name": "Ambient Temperature",
        "units": "K",
        "field_cmap": "inferno",
        "error_cmap": "magma",
        "signed_cmap": "coolwarm",
    },
}


def _prepare_spatial_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
    building_mask: Optional[np.ndarray] = None,
    fluid_mask: Optional[np.ndarray] = None,
    mask_is_fluid: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validates shapes, squeezes singleton dims, and produces masked 2D arrays.

    Returns:
        (true_masked, pred_masked, solid_mask)
        where solid cells in true_masked and pred_masked are set to np.nan.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    # Squeeze leading singleton dims (e.g. (1, 1, H, W) or (1, H, W))
    while yt.ndim > 2 and yt.shape[0] == 1:
        yt = yt[0]
    while yp.ndim > 2 and yp.shape[0] == 1:
        yp = yp[0]

    if yt.ndim != 2 or yp.ndim != 2:
        raise ValueError(f"Fields must be 2D horizontal slices, got y_true ndim={yt.ndim}, y_pred ndim={yp.ndim}")

    if yt.shape != yp.shape:
        raise ValueError(
            f"Spatial alignment error: y_true shape {yt.shape} does not match y_pred shape {yp.shape}. "
            f"Fields must share identical grid dimensions."
        )

    # Determine boolean solid building mask (True for solid buildings, False for fluid)
    ny, nx = yt.shape
    solid = np.zeros((ny, nx), dtype=bool)

    if building_mask is not None:
        bm = np.asarray(building_mask)
        while bm.ndim > 2 and bm.shape[0] == 1:
            bm = bm[0]
        if bm.shape != (ny, nx):
            raise ValueError(f"building_mask shape {bm.shape} != field shape {(ny, nx)}")
        solid = bm.astype(bool)
    elif fluid_mask is not None:
        fm = np.asarray(fluid_mask)
        while fm.ndim > 2 and fm.shape[0] == 1:
            fm = fm[0]
        if fm.shape != (ny, nx):
            raise ValueError(f"fluid_mask shape {fm.shape} != field shape {(ny, nx)}")
        solid = ~fm.astype(bool)
    elif mask is not None:
        m = np.asarray(mask)
        while m.ndim > 2 and m.shape[0] == 1:
            m = m[0]
        if m.shape != (ny, nx):
            raise ValueError(f"mask shape {m.shape} != field shape {(ny, nx)}")
        solid = ~m.astype(bool) if mask_is_fluid else m.astype(bool)

    true_masked = np.where(solid, np.nan, yt)
    pred_masked = np.where(solid, np.nan, yp)

    return true_masked, pred_masked, solid


def plot_field_triplet(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
    layout_id: str = "layout",
    output_path: Optional[Union[str, Path]] = None,
    field_type: str = "wind_speed",
    units: Optional[str] = None,
    extent: Optional[Tuple[float, float, float, float]] = None,
    resolution_m: Optional[Tuple[float, float, float]] = None,
    inflow_speed: Optional[float] = None,
    wind_direction_deg: Optional[float] = None,
    provenance_tag: str = "SYNTHETIC_TEST_FIXTURE",
    figsize: Tuple[float, float] = (16.0, 5.0),
    building_facecolor: str = "#2b2b2b",
) -> Tuple[plt.Figure, Path]:
    """Generates a 3-panel comparison: [CFD Reference] | [Surrogate Prediction] | [Absolute Error].

    Parameters:
        y_true: Ground truth 2D field (Ny, Nx).
        y_pred: Predicted 2D field (Ny, Nx).
        mask: Building footprint mask (True for solid buildings).
        layout_id: Identifier of the layout.
        output_path: Filepath where the image should be saved.
        field_type: Type of variable ('wind_speed', 'temperature', 'velocity_u', etc.).
        units: Explicit unit string override.
        extent: Optional (xmin, xmax, ymin, ymax) coordinate bounds in meters.
        resolution_m: Optional (dx, dy, dz) cell sizes in meters to auto-compute extent.
        inflow_speed: Inflow wind speed in m/s (metadata).
        wind_direction_deg: Wind inflow angle in degrees (metadata).
        provenance_tag: Provenance label (e.g. 'VERIFIED_CFD_DATA', 'CANONICAL_BENCHMARK', 'SYNTHETIC_TEST_FIXTURE').
        figsize: Matplotlib figure size tuple.
        building_facecolor: Background color for building interior NaN cells.

    Returns:
        (fig, saved_path): Tuple containing Matplotlib figure and Path to saved image.
    """
    cfg = VARIABLE_CONFIGS.get(field_type, VARIABLE_CONFIGS["wind_speed"])
    unit_str = units if units is not None else cfg["units"]
    var_label = cfg["display_name"]

    true_masked, pred_masked, solid = _prepare_spatial_arrays(y_true, y_pred, mask=mask)
    abs_error = np.where(solid, np.nan, np.abs(true_masked - pred_masked))

    # Shared unified color scaling across reference and prediction for fair visual comparison
    valid_true = true_masked[~np.isnan(true_masked)]
    valid_pred = pred_masked[~np.isnan(pred_masked)]

    if len(valid_true) == 0 and len(valid_pred) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(min(np.min(valid_true) if len(valid_true) > 0 else 0.0,
                         np.min(valid_pred) if len(valid_pred) > 0 else 0.0))
        vmax = float(max(np.max(valid_true) if len(valid_true) > 0 else 1.0,
                         np.max(valid_pred) if len(valid_pred) > 0 else 1.0))
        if vmax - vmin < 1e-4:
            vmax = vmin + 1.0

    valid_err = abs_error[~np.isnan(abs_error)]
    err_max = float(np.max(valid_err)) if len(valid_err) > 0 else 1.0
    err_max = max(err_max, 0.1)
    mae_val = float(np.mean(valid_err)) if len(valid_err) > 0 else 0.0

    # Coordinate extent
    ny, nx = true_masked.shape
    if extent is None and resolution_m is not None:
        dx, dy, _ = resolution_m
        extent = (0.0, nx * dx, 0.0, ny * dy)

    x_label = "X [m]" if extent is not None else "X [grid cells]"
    y_label = "Y [m]" if extent is not None else "Y [grid cells]"

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # Panel 1: Ground Truth CFD Reference
    im0 = axes[0].imshow(
        true_masked, cmap=cfg["field_cmap"], vmin=vmin, vmax=vmax,
        origin="lower", extent=extent, interpolation="nearest",
    )
    axes[0].set_facecolor(building_facecolor)
    axes[0].set_title(f"CFD Reference\n({layout_id})", fontsize=11, fontweight="bold")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label(f"{var_label} [{unit_str}]", fontsize=9)

    # Panel 2: Model Prediction
    im1 = axes[1].imshow(
        pred_masked, cmap=cfg["field_cmap"], vmin=vmin, vmax=vmax,
        origin="lower", extent=extent, interpolation="nearest",
    )
    axes[1].set_facecolor(building_facecolor)
    axes[1].set_title("Model Prediction\n(Surrogate)", fontsize=11, fontweight="bold")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label(f"{var_label} [{unit_str}]", fontsize=9)

    # Panel 3: Absolute Error Map
    im2 = axes[2].imshow(
        abs_error, cmap=cfg["error_cmap"], vmin=0.0, vmax=err_max,
        origin="lower", extent=extent, interpolation="nearest",
    )
    axes[2].set_facecolor(building_facecolor)
    axes[2].set_title(f"Absolute Error |True - Pred|\nMAE = {mae_val:.3f} {unit_str}", fontsize=11, fontweight="bold")
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label(f"Absolute Error [{unit_str}]", fontsize=9)

    # Axis labeling and metadata
    for ax in axes:
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)

    # Header title with condition metadata and provenance tag
    meta_parts = [f"Layout: {layout_id}"]
    if inflow_speed is not None:
        meta_parts.append(f"$U_{{ref}} = {inflow_speed:.1f}$ m/s")
    if wind_direction_deg is not None:
        meta_parts.append(f"$\\theta = {wind_direction_deg:.0f}^\circ$")
    meta_parts.append(f"[{provenance_tag}]")
    fig.suptitle(" • ".join(meta_parts), fontsize=12, fontweight="bold", y=1.04)

    # Destination handling
    if output_path is None:
        output_path = Path("outputs") / "plots" / f"{layout_id}_triplet.png"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return fig, out


def plot_field_quad(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
    layout_id: str = "layout",
    output_path: Optional[Union[str, Path]] = None,
    field_type: str = "wind_speed",
    units: Optional[str] = None,
    extent: Optional[Tuple[float, float, float, float]] = None,
    resolution_m: Optional[Tuple[float, float, float]] = None,
    inflow_speed: Optional[float] = None,
    wind_direction_deg: Optional[float] = None,
    provenance_tag: str = "SYNTHETIC_TEST_FIXTURE",
    figsize: Tuple[float, float] = (20.0, 5.0),
    building_facecolor: str = "#2b2b2b",
) -> Tuple[plt.Figure, Path]:
    """Generates a 4-panel comparison: [CFD Reference] | [Prediction] | [Signed Error] | [Absolute Error].

    Signed Error = (y_pred - y_true):
      - Positive (red) denotes model overprediction.
      - Negative (blue) denotes model underprediction.
      - Uses a diverging colormap centered strictly at 0.0.
    """
    cfg = VARIABLE_CONFIGS.get(field_type, VARIABLE_CONFIGS["wind_speed"])
    unit_str = units if units is not None else cfg["units"]
    var_label = cfg["display_name"]

    true_masked, pred_masked, solid = _prepare_spatial_arrays(y_true, y_pred, mask=mask)
    abs_error = np.where(solid, np.nan, np.abs(true_masked - pred_masked))
    signed_error = np.where(solid, np.nan, pred_masked - true_masked)

    # Shared unified color scaling for flow fields
    valid_true = true_masked[~np.isnan(true_masked)]
    valid_pred = pred_masked[~np.isnan(pred_masked)]

    if len(valid_true) == 0 and len(valid_pred) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(min(np.min(valid_true) if len(valid_true) > 0 else 0.0,
                         np.min(valid_pred) if len(valid_pred) > 0 else 0.0))
        vmax = float(max(np.max(valid_true) if len(valid_true) > 0 else 1.0,
                         np.max(valid_pred) if len(valid_pred) > 0 else 1.0))
        if vmax - vmin < 1e-4:
            vmax = vmin + 1.0

    valid_abs = abs_error[~np.isnan(abs_error)]
    err_max = float(np.max(valid_abs)) if len(valid_abs) > 0 else 1.0
    err_max = max(err_max, 0.1)
    mae_val = float(np.mean(valid_abs)) if len(valid_abs) > 0 else 0.0

    # Symmetric scale for signed error centered at 0.0
    valid_signed = signed_error[~np.isnan(signed_error)]
    signed_bound = float(np.max(np.abs(valid_signed))) if len(valid_signed) > 0 else 1.0
    signed_bound = max(signed_bound, 0.1)

    # Coordinate extent
    ny, nx = true_masked.shape
    if extent is None and resolution_m is not None:
        dx, dy, _ = resolution_m
        extent = (0.0, nx * dx, 0.0, ny * dy)

    x_label = "X [m]" if extent is not None else "X [grid cells]"
    y_label = "Y [m]" if extent is not None else "Y [grid cells]"

    fig, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)

    # Panel 1: CFD Reference
    im0 = axes[0].imshow(
        true_masked, cmap=cfg["field_cmap"], vmin=vmin, vmax=vmax,
        origin="lower", extent=extent, interpolation="nearest",
    )
    axes[0].set_facecolor(building_facecolor)
    axes[0].set_title(f"CFD Reference\n({layout_id})", fontsize=11, fontweight="bold")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label(f"{var_label} [{unit_str}]", fontsize=9)

    # Panel 2: Model Prediction
    im1 = axes[1].imshow(
        pred_masked, cmap=cfg["field_cmap"], vmin=vmin, vmax=vmax,
        origin="lower", extent=extent, interpolation="nearest",
    )
    axes[1].set_facecolor(building_facecolor)
    axes[1].set_title("Model Prediction\n(Surrogate)", fontsize=11, fontweight="bold")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label(f"{var_label} [{unit_str}]", fontsize=9)

    # Panel 3: Signed Error Map (Pred - True)
    im2 = axes[2].imshow(
        signed_error, cmap=cfg["signed_cmap"], vmin=-signed_bound, vmax=signed_bound,
        origin="lower", extent=extent, interpolation="nearest",
    )
    axes[2].set_facecolor(building_facecolor)
    axes[2].set_title(f"Signed Error (Pred - True)\nMax Over: +{np.nanmax(signed_error):.2f} {unit_str}", fontsize=11, fontweight="bold")
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label(f"Signed Error [{unit_str}]", fontsize=9)

    # Panel 4: Absolute Error Map |True - Pred|
    im3 = axes[3].imshow(
        abs_error, cmap=cfg["error_cmap"], vmin=0.0, vmax=err_max,
        origin="lower", extent=extent, interpolation="nearest",
    )
    axes[3].set_facecolor(building_facecolor)
    axes[3].set_title(f"Absolute Error |True - Pred|\nMAE = {mae_val:.3f} {unit_str}", fontsize=11, fontweight="bold")
    cbar3 = fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    cbar3.set_label(f"Absolute Error [{unit_str}]", fontsize=9)

    for ax in axes:
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)

    # Header title
    meta_parts = [f"Layout: {layout_id}"]
    if inflow_speed is not None:
        meta_parts.append(f"$U_{{ref}} = {inflow_speed:.1f}$ m/s")
    if wind_direction_deg is not None:
        meta_parts.append(f"$\\theta = {wind_direction_deg:.0f}^\circ$")
    meta_parts.append(f"[{provenance_tag}]")
    fig.suptitle(" • ".join(meta_parts), fontsize=12, fontweight="bold", y=1.04)

    if output_path is None:
        output_path = Path("outputs") / "plots" / f"{layout_id}_quad.png"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return fig, out


def plot_single_field(
    field_2d: np.ndarray,
    mask: Optional[np.ndarray] = None,
    title: str = "Spatial Field",
    output_path: Optional[Union[str, Path]] = None,
    field_type: str = "wind_speed",
    units: Optional[str] = None,
    extent: Optional[Tuple[float, float, float, float]] = None,
    colormap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (6.5, 5.0),
    building_facecolor: str = "#2b2b2b",
) -> Tuple[plt.Figure, Path]:
    """Plots a single 2D spatial field with building mask overlay and dedicated colorbar."""
    cfg = VARIABLE_CONFIGS.get(field_type, VARIABLE_CONFIGS["wind_speed"])
    unit_str = units if units is not None else cfg["units"]
    cmap_str = colormap if colormap is not None else cfg["field_cmap"]

    f = np.asarray(field_2d, dtype=np.float64)
    while f.ndim > 2 and f.shape[0] == 1:
        f = f[0]
    if f.ndim != 2:
        raise ValueError(f"field_2d must be 2D, got shape {field_2d.shape}")

    ny, nx = f.shape
    solid = np.zeros((ny, nx), dtype=bool)
    if mask is not None:
        m = np.asarray(mask)
        while m.ndim > 2 and m.shape[0] == 1:
            m = m[0]
        if m.shape != (ny, nx):
            raise ValueError(f"mask shape {m.shape} != field shape {(ny, nx)}")
        solid = m.astype(bool)

    f_masked = np.where(solid, np.nan, f)

    if vmin is None:
        valid_vals = f_masked[~np.isnan(f_masked)]
        vmin = float(np.min(valid_vals)) if len(valid_vals) > 0 else 0.0
    if vmax is None:
        valid_vals = f_masked[~np.isnan(f_masked)]
        vmax = float(np.max(valid_vals)) if len(valid_vals) > 0 else 1.0

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(
        f_masked, cmap=cmap_str, vmin=vmin, vmax=vmax,
        origin="lower", extent=extent, interpolation="nearest",
    )
    ax.set_facecolor(building_facecolor)
    ax.set_title(title, fontsize=11, fontweight="bold")

    x_label = "X [m]" if extent is not None else "X [grid cells]"
    y_label = "Y [m]" if extent is not None else "Y [grid cells]"
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"[{unit_str}]", fontsize=9)

    if output_path is None:
        output_path = Path("outputs") / "plots" / "single_field.png"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return fig, out


# Backward compatibility alias
def plot_prediction_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    layout_id: str,
    output_path: Union[str, Path],
    units: str = "m/s",
    title_suffix: str = "",
    include_signed_error: bool = False,
    provenance_tag: str = "SYNTHETIC_TEST_FIXTURE",
) -> Path:
    """Generates comparison figure: 3-panel by default or 4-panel if include_signed_error=True."""
    if include_signed_error:
        _, path = plot_field_quad(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
            layout_id=layout_id,
            output_path=output_path,
            units=units,
            provenance_tag=provenance_tag,
        )
    else:
        _, path = plot_field_triplet(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
            layout_id=layout_id,
            output_path=output_path,
            units=units,
            provenance_tag=provenance_tag,
        )
    return path
