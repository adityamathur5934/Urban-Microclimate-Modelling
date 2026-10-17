"""Unit tests for spatial field metrics, NRMSE conventions, and the DL-AML aggregation bridge."""

import numpy as np
import pytest
import torch

from src.training.metrics import (
    aggregate_spatial_field_to_scalar,
    compute_surrogate_metrics_summary,
    field_mae,
    field_max_abs_error,
    field_nrmse,
    field_rmse,
)


def test_field_mae_and_rmse():
    """Verify MAE and RMSE values match analytical calculations."""
    pred = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    target = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    # Errors are uniformly +1.0

    mae = field_mae(pred, target)
    rmse = field_rmse(pred, target)
    assert np.isclose(mae, 1.0)
    assert np.isclose(rmse, 1.0)

    # Test with fluid mask excluding (0, 0)
    fluid_mask = np.array([[False, True], [True, True]])
    # Change pred[0, 0] to error of 100, which should be ignored because mask is False
    pred_bad = pred.copy()
    pred_bad[0, 0] = 100.0

    masked_mae = field_mae(pred_bad, target, fluid_mask=fluid_mask)
    masked_rmse = field_rmse(pred_bad, target, fluid_mask=fluid_mask)
    assert np.isclose(masked_mae, 1.0)
    assert np.isclose(masked_rmse, 1.0)


def test_field_nrmse_conventions():
    """Verify each documented normalization formula computes expected ratios."""
    target = np.array([[2.0, 4.0], [6.0, 10.0]], dtype=np.float32)
    # Target range is 10.0 - 2.0 = 8.0
    pred = target + 1.0  # RMSE = 1.0

    nrmse_range = field_nrmse(pred, target, normalization="range")
    assert np.isclose(nrmse_range, 1.0 / 8.0)

    nrmse_inflow = field_nrmse(pred, target, normalization="inflow", inflow_speed=5.0)
    assert np.isclose(nrmse_inflow, 1.0 / 5.0)

    nrmse_std = field_nrmse(pred, target, normalization="std")
    expected_std = np.std(target)
    assert np.isclose(nrmse_std, 1.0 / expected_std)


def test_aggregate_spatial_field_to_scalar_bridge():
    """Verify 2D neural surrogate prediction aggregates correctly to layout-level scalar."""
    field_2d = np.array([
        [10.0, 20.0, 0.0],
        [30.0, 40.0, 0.0],
    ], dtype=np.float32)
    building_mask = np.array([
        [False, False, True],
        [False, False, True],
    ], dtype=bool)

    # Outdoor cells are [10, 20, 30, 40]. Average = 25.0
    scalar_avg = aggregate_spatial_field_to_scalar(field_2d, building_mask)
    assert np.isclose(scalar_avg, 25.0)


def test_metrics_summary():
    """Verify full metrics dictionary generation."""
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.2, 1.9], [3.1, 3.8]])
    summary = compute_surrogate_metrics_summary(pred, target, inflow_speed=5.0)

    assert "field_mae" in summary
    assert "field_rmse" in summary
    assert "field_nrmse_range" in summary
    assert "field_nrmse_inflow" in summary
    assert "field_max_error" in summary
