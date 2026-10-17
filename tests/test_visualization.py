"""Unit tests for visualization functions to ensure plotting and saving work without errors."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for testing

from src.training.visualize import (
    plot_baseline_scatter,
    plot_field_comparison,
    plot_training_curves,
)


def test_plot_field_comparison(tmp_path: Path):
    """Verify 3-panel field comparison figure creates and saves."""
    ny, nx = 32, 32
    gt = np.ones((ny, nx), dtype=np.float32) * 4.0
    pred = np.ones((ny, nx), dtype=np.float32) * 3.8
    mask = np.zeros((ny, nx), dtype=bool)
    mask[10:20, 10:20] = True

    save_file = tmp_path / "field_comparison.png"
    fig = plot_field_comparison(gt, pred, mask, save_path=save_file)
    assert save_file.exists()
    assert save_file.stat().st_size > 0


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


def test_plot_baseline_scatter(tmp_path: Path):
    """Verify scatter plot of predicted vs actual creates and saves."""
    y_true = np.array([2.5, 3.0, 4.2, 5.1], dtype=np.float32)
    y_pred = np.array([2.4, 3.1, 4.0, 5.3], dtype=np.float32)

    save_file = tmp_path / "baseline_scatter.png"
    fig = plot_baseline_scatter(y_true, y_pred, save_path=save_file)
    assert save_file.exists()
    assert save_file.stat().st_size > 0
