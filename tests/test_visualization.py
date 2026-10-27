"""Unit tests for spatial field visualization, colormap scaling, coordinate orientation, and error maps."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for testing
import matplotlib.pyplot as plt
import pytest

from src.training.visualize import (
    plot_baseline_scatter,
    plot_field_comparison,
    plot_training_curves,
)
from src.evaluation.plotting import (
    plot_field_triplet,
    plot_field_quad,
    plot_single_field,
    plot_prediction_comparison,
)
from tests.fixtures.synthetic_data import create_synthetic_metric_benchmark_fixture


# ============================================================================
# 1. Existing Training Visualization Tests (Preserved)
# ============================================================================

def test_plot_field_comparison(tmp_path: Path):
    """Verify legacy 3-panel field comparison figure creates and saves."""
    ny, nx = 32, 32
    gt = np.ones((ny, nx), dtype=np.float32) * 4.0
    pred = np.ones((ny, nx), dtype=np.float32) * 3.8
    mask = np.zeros((ny, nx), dtype=bool)
    mask[10:20, 10:20] = True

    save_file = tmp_path / "field_comparison.png"
    fig = plot_field_comparison(gt, pred, mask, save_path=save_file)
    assert save_file.exists()
    assert save_file.stat().st_size > 0
    plt.close(fig)


def test_plot_training_curves(tmp_path: Path):
    """Verify training and validation loss curves create and save."""
    history = {
        "train_loss": [0.5, 0.4, 0.3],
        "val_loss": [0.55, 0.42, 0.35],
        "val_nrmse": [0.15, 0.12, 0.09],
        "val_mae": [0.4, 0.3, 0.25],
    }
    save_file = tmp_path / "training_curves.png"
    fig = plot_training_curves(history, save_path=save_file)
    assert save_file.exists()
    assert save_file.stat().st_size > 0
    plt.close(fig)


def test_plot_baseline_scatter(tmp_path: Path):
    """Verify scatter plot of predicted vs actual creates and saves."""
    y_true = np.array([2.5, 3.0, 4.2, 5.1], dtype=np.float32)
    y_pred = np.array([2.4, 3.1, 4.0, 5.3], dtype=np.float32)

    save_file = tmp_path / "baseline_scatter.png"
    fig = plot_baseline_scatter(y_true, y_pred, save_path=save_file)
    assert save_file.exists()
    assert save_file.stat().st_size > 0
    plt.close(fig)


# ============================================================================
# 2. Evaluation Field Visualization & Scale Fairness Tests
# ============================================================================

def test_plot_field_triplet_common_scales_and_output(tmp_path: Path):
    """Verify 3-panel figure saves, sets identical color limits on reference/pred, and masks buildings.

    USES SYNTHETIC TEST ARRAYS ONLY FOR SOFTWARE VERIFICATION.
    """
    bench = create_synthetic_metric_benchmark_fixture()
    save_file = tmp_path / "triplet_test.png"

    fig, out_path = plot_field_triplet(
        y_true=bench.y_true,
        y_pred=bench.y_pred,
        mask=bench.building_mask,
        layout_id="synthetic_bench_001",
        output_path=save_file,
        inflow_speed=10.0,
        provenance_tag="SYNTHETIC_TEST_FIXTURE",
    )

    # 1. Output file creation
    assert out_path.exists()
    assert out_path.stat().st_size > 1000

    # 2. Verify shared color limits across panel 0 (True) and panel 1 (Pred)
    axes = fig.axes  # Note: colorbars also create axes
    im_ref = axes[0].images[0]
    im_pred = axes[1].images[0]
    ref_clim = im_ref.get_clim()
    pred_clim = im_pred.get_clim()
    assert ref_clim == pred_clim, f"Reference clim {ref_clim} != Prediction clim {pred_clim}"

    # 3. Verify origin='lower' is strictly preserved (no silent coordinate flips)
    assert im_ref.origin == "lower"
    assert im_pred.origin == "lower"

    # 4. Verify solid buildings are masked in image data (matplotlib converts NaNs to MaskedArray)
    ref_data = im_ref.get_array()
    if isinstance(ref_data, np.ma.MaskedArray):
        assert ref_data.mask[3, :].all()
    else:
        assert np.isnan(ref_data[3, :]).all()

    plt.close(fig)


def test_plot_field_quad_signed_error_symmetry(tmp_path: Path):
    """Verify 4-panel figure creates signed error with symmetric zero-centered color limits.

    USES SYNTHETIC TEST ARRAYS ONLY.
    """
    bench = create_synthetic_metric_benchmark_fixture()
    save_file = tmp_path / "quad_test.png"

    fig, out_path = plot_field_quad(
        y_true=bench.y_true,
        y_pred=bench.y_pred,
        mask=bench.building_mask,
        layout_id="synthetic_bench_quad",
        output_path=save_file,
        field_type="wind_speed",
        provenance_tag="SYNTHETIC_TEST_FIXTURE",
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 1000

    # Panel 2 is signed error (im2)
    im_signed = fig.axes[2].images[0]
    s_vmin, s_vmax = im_signed.get_clim()
    # Signed error limits must be symmetric: vmin == -vmax
    assert np.isclose(s_vmin, -s_vmax, atol=1e-5), f"Signed error not symmetric: {s_vmin} vs {s_vmax}"

    plt.close(fig)


def test_plot_single_field(tmp_path: Path):
    """Verify single field plotting function saves and sets title and units."""
    ny, nx = 16, 16
    f = np.linspace(1.0, 5.0, ny * nx, dtype=np.float32).reshape(ny, nx)
    mask = np.zeros((ny, nx), dtype=bool)
    mask[0:4, 0:4] = True

    save_file = tmp_path / "single_field.png"
    fig, out_path = plot_single_field(
        field_2d=f,
        mask=mask,
        title="Reference Flow Slices",
        output_path=save_file,
        field_type="wind_speed",
        units="m/s",
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 500
    plt.close(fig)


def test_plot_prediction_comparison_wrapper(tmp_path: Path):
    """Verify backward-compatible plot_prediction_comparison function."""
    bench = create_synthetic_metric_benchmark_fixture()
    out_file = tmp_path / "wrapper_test.png"

    saved = plot_prediction_comparison(
        y_true=bench.y_true,
        y_pred=bench.y_pred,
        mask=bench.building_mask,
        layout_id="compat_test",
        output_path=out_file,
        include_signed_error=True,
    )
    assert saved.exists()
    assert saved.stat().st_size > 1000


def test_temperature_field_visualization(tmp_path: Path):
    """Verify temperature fields use thermal units [K] and distinct colormap."""
    ny, nx = 8, 8
    temp_true = np.full((ny, nx), 295.15, dtype=np.float32)
    temp_pred = temp_true + 1.2
    mask = np.zeros((ny, nx), dtype=bool)

    save_file = tmp_path / "temperature_test.png"
    fig, out_path = plot_field_triplet(
        y_true=temp_true,
        y_pred=temp_pred,
        mask=mask,
        layout_id="thermal_block",
        output_path=save_file,
        field_type="temperature",
    )

    assert out_path.exists()
    # Check that colorbar label indicates Kelvin [K]
    cbar_label = fig.axes[0].figure.axes[-3].get_ylabel()  # Colorbar axis label
    assert "K" in cbar_label or "Temperature" in cbar_label
    plt.close(fig)


def test_spatial_shape_mismatch_raises_error():
    """Mismatched spatial grid shapes must raise ValueError."""
    y_true = np.zeros((16, 16), dtype=np.float32)
    y_pred = np.zeros((16, 32), dtype=np.float32)

    with pytest.raises(ValueError, match="Spatial alignment error"):
        plot_field_triplet(y_true, y_pred)


def test_physical_extent_labels():
    """Passing resolution_m must calibrate axis extent in meters [m]."""
    ny, nx = 10, 20
    f = np.ones((ny, nx), dtype=np.float32)
    fig, _ = plot_field_triplet(
        y_true=f,
        y_pred=f,
        resolution_m=(2.0, 2.0, 1.0),
        layout_id="extent_test",
        output_path=None,
    )

    ax0 = fig.axes[0]
    assert ax0.get_xlabel() == "X [m]"
    assert ax0.get_ylabel() == "Y [m]"
    # Extent should be [0, 40, 0, 20]
    im = ax0.images[0]
    extent = tuple(im.get_extent())
    assert extent == (0.0, 40.0, 0.0, 20.0)

    plt.close(fig)
