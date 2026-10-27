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
from src.evaluation.metrics import (
    compute_field_mae,
    compute_field_rmse,
    compute_normalized_rmse,
    compute_relative_l2,
    compute_r2_score,
    compute_max_error,
    compute_spatial_error_map,
    compute_regression_metrics,
)
from tests.fixtures.synthetic_data import create_synthetic_metric_benchmark_fixture


# ============================================================================
# 1. Hand-Calculated Analytical Verification via Synthetic Benchmark Fixture
# ============================================================================

def test_hand_calculated_metric_answers():
    """Verify all metrics against hand-calculated analytical answers on synthetic fixture."""
    fixture = create_synthetic_metric_benchmark_fixture()

    # 1. MAE
    mae = compute_field_mae(fixture.y_true, fixture.y_pred, mask=fixture.building_mask)
    assert np.isclose(mae, fixture.expected_mae, atol=1e-6)

    # 2. RMSE
    rmse = compute_field_rmse(fixture.y_true, fixture.y_pred, mask=fixture.building_mask)
    assert np.isclose(rmse, fixture.expected_rmse, atol=1e-6)

    # 3. Max Error
    max_err = compute_max_error(fixture.y_true, fixture.y_pred, mask=fixture.building_mask)
    assert np.isclose(max_err, fixture.expected_max_error, atol=1e-6)

    # 4. NRMSE (range, ratio & percent)
    nrmse_range_ratio = compute_normalized_rmse(
        fixture.y_true, fixture.y_pred, mask=fixture.building_mask,
        normalization="range", as_percent=False,
    )
    assert np.isclose(nrmse_range_ratio, fixture.expected_nrmse_range_ratio, atol=1e-6)

    nrmse_range_pct = compute_normalized_rmse(
        fixture.y_true, fixture.y_pred, mask=fixture.building_mask,
        normalization="range", as_percent=True,
    )
    assert np.isclose(nrmse_range_pct, fixture.expected_nrmse_range_percent, atol=1e-6)

    # 5. NRMSE (inflow, ratio & percent)
    nrmse_inflow_ratio = compute_normalized_rmse(
        fixture.y_true, fixture.y_pred, mask=fixture.building_mask,
        normalization="inflow", u_ref=fixture.u_ref, as_percent=False,
    )
    assert np.isclose(nrmse_inflow_ratio, fixture.expected_nrmse_inflow_ratio, atol=1e-6)

    nrmse_inflow_pct = compute_normalized_rmse(
        fixture.y_true, fixture.y_pred, mask=fixture.building_mask,
        normalization="inflow", u_ref=fixture.u_ref, as_percent=True,
    )
    assert np.isclose(nrmse_inflow_pct, fixture.expected_nrmse_inflow_percent, atol=1e-6)

    # 6. Relative L2
    rel_l2 = compute_relative_l2(fixture.y_true, fixture.y_pred, mask=fixture.building_mask)
    assert np.isclose(rel_l2, fixture.expected_relative_l2, atol=1e-6)

    # 7. R2 Score
    r2 = compute_r2_score(fixture.y_true, fixture.y_pred, mask=fixture.building_mask)
    assert np.isclose(r2, fixture.expected_r2, atol=1e-6)


def test_building_mask_ignores_interior_corruption():
    """Verify that corrupt or massive errors inside solid buildings are strictly ignored."""
    y_true = np.array([[5.0, 5.0], [0.0, 0.0]], dtype=np.float64)
    # Bottom row is solid building. Place 1e6 error inside building.
    y_pred = np.array([[6.0, 6.0], [1e6, 1e6]], dtype=np.float64)
    b_mask = np.array([[False, False], [True, True]], dtype=bool)

    mae = compute_field_mae(y_true, y_pred, building_mask=b_mask)
    rmse = compute_field_rmse(y_true, y_pred, building_mask=b_mask)
    assert np.isclose(mae, 1.0)
    assert np.isclose(rmse, 1.0)


def test_fluid_mask_semantic_support():
    """Verify metrics work with fluid_mask (True=fluid) or mask_is_fluid=True."""
    y_true = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float64)
    y_pred = y_true + 2.0
    f_mask = np.array([[True, True], [False, False]], dtype=bool)

    # Using explicit fluid_mask kwarg
    mae_fm = compute_field_mae(y_true, y_pred, fluid_mask=f_mask)
    assert np.isclose(mae_fm, 2.0)

    # Using generic mask with mask_is_fluid=True
    mae_m = compute_field_mae(y_true, y_pred, mask=f_mask, mask_is_fluid=True)
    assert np.isclose(mae_m, 2.0)


# ============================================================================
# 2. Non-Finite Values and Empty Mask Error Handling
# ============================================================================

def test_non_finite_predictions_raise_error():
    """Non-finite values (NaN / Inf) in predictions must raise ValueError."""
    y_true = np.ones((4, 4), dtype=np.float64)
    y_pred_nan = np.ones((4, 4), dtype=np.float64)
    y_pred_nan[0, 0] = np.nan

    with pytest.raises(ValueError, match="Non-finite values detected in predictions.*NaN"):
        compute_field_mae(y_true, y_pred_nan)

    y_pred_inf = np.ones((4, 4), dtype=np.float64)
    y_pred_inf[1, 1] = np.inf

    with pytest.raises(ValueError, match="Non-finite values detected in predictions.*Inf"):
        compute_field_rmse(y_true, y_pred_inf)


def test_non_finite_targets_raise_error():
    """Non-finite values (NaN / Inf) in targets must raise ValueError."""
    y_true_nan = np.ones((4, 4), dtype=np.float64)
    y_true_nan[0, 0] = np.nan
    y_pred = np.ones((4, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="Non-finite values detected in ground truth target"):
        compute_field_mae(y_true_nan, y_pred)


def test_nan_in_mask_raises_error():
    """Float mask containing NaNs must raise ValueError."""
    y_true = np.ones((4, 4), dtype=np.float64)
    y_pred = np.ones((4, 4), dtype=np.float64)
    bad_mask = np.zeros((4, 4), dtype=np.float32)
    bad_mask[0, 0] = np.nan

    with pytest.raises(ValueError, match="Mask contains NaN values"):
        compute_field_mae(y_true, y_pred, mask=bad_mask)


def test_empty_mask_raises_clear_error():
    """Mask with 0 valid fluid cells must raise ValueError with clear message."""
    y_true = np.ones((4, 4), dtype=np.float64)
    y_pred = np.ones((4, 4), dtype=np.float64)
    all_building_mask = np.ones((4, 4), dtype=bool)  # All building, 0 fluid

    with pytest.raises(ValueError, match="Empty mask error: domain contains 0 valid fluid cells"):
        compute_field_mae(y_true, y_pred, building_mask=all_building_mask)

    all_false_fluid_mask = np.zeros((4, 4), dtype=bool)  # 0 fluid
    with pytest.raises(ValueError, match="Empty mask error"):
        compute_field_rmse(y_true, y_pred, fluid_mask=all_false_fluid_mask)


# ============================================================================
# 3. Spatial Field Alignment and Shape Mismatch Checks
# ============================================================================

def test_spatial_shape_mismatch_raises_error():
    """Mismatched spatial grid dimensions between pred and target must raise ValueError."""
    y_true = np.zeros((32, 32), dtype=np.float64)
    y_pred = np.zeros((32, 64), dtype=np.float64)

    with pytest.raises(ValueError, match="Spatial alignment error: y_true shape.*does not match y_pred shape"):
        compute_field_mae(y_true, y_pred)


def test_mask_shape_mismatch_raises_error():
    """Mismatched mask dimensions relative to target field must raise ValueError."""
    y_true = np.zeros((32, 32), dtype=np.float64)
    y_pred = np.zeros((32, 32), dtype=np.float64)
    bad_mask = np.zeros((16, 16), dtype=bool)

    with pytest.raises(ValueError, match="Spatial alignment error:.*shape.*does not match"):
        compute_field_mae(y_true, y_pred, mask=bad_mask)


def test_singleton_dimension_squeezing_alignment():
    """Fields with leading singleton dimensions (1, H, W) must align seamlessly with (H, W)."""
    y_true_3d = np.full((1, 16, 16), 5.0, dtype=np.float64)
    y_pred_2d = np.full((16, 16), 7.0, dtype=np.float64)
    mask_3d = np.zeros((1, 16, 16), dtype=bool)

    mae = compute_field_mae(y_true_3d, y_pred_2d, mask=mask_3d)
    assert np.isclose(mae, 2.0)


# ============================================================================
# 4. NRMSE Normalization Denominators
# ============================================================================

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

    nrmse_mean = compute_normalized_rmse(
        target, pred, normalization="mean", as_percent=False
    )
    expected_mean = np.mean(target)
    assert np.isclose(nrmse_mean, 1.0 / expected_mean)


def test_nrmse_inflow_missing_uref_raises_error():
    """normalization='inflow' without positive u_ref must raise ValueError."""
    y_true = np.ones((4, 4), dtype=np.float64)
    y_pred = y_true + 0.5
    with pytest.raises(ValueError, match="requires a positive 'u_ref'"):
        compute_normalized_rmse(y_true, y_pred, normalization="inflow", u_ref=None)

    with pytest.raises(ValueError, match="requires a positive 'u_ref'"):
        compute_normalized_rmse(y_true, y_pred, normalization="inflow", u_ref=-2.0)


def test_nrmse_unknown_convention_raises_error():
    """Unknown normalization scheme must raise ValueError with supported list."""
    y_true = np.ones((4, 4), dtype=np.float64)
    y_pred = y_true + 0.5
    with pytest.raises(ValueError, match="Unknown NRMSE normalization"):
        compute_normalized_rmse(y_true, y_pred, normalization="unsupported_norm")


def test_nrmse_zero_denominator_in_flat_field():
    """Zero dynamic range field with non-zero error raises ValueError."""
    y_true = np.full((4, 4), 5.0, dtype=np.float64)  # max - min = 0
    y_pred = y_true + 1.0  # RMSE = 1.0
    with pytest.raises(ValueError, match="Zero denominator encountered in NRMSE"):
        compute_normalized_rmse(y_true, y_pred, normalization="range")

    # If error is also 0, NRMSE should return 0.0
    zero_err = compute_normalized_rmse(y_true, y_true, normalization="range")
    assert zero_err == 0.0


# ============================================================================
# 5. DL-AML Aggregation Bridge and Tensor Interoperability
# ============================================================================

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


def test_aggregate_spatial_field_all_building_raises_error():
    """Empty outdoor domain in aggregation bridge must raise ValueError."""
    field_2d = np.ones((4, 4), dtype=np.float32)
    all_building = np.ones((4, 4), dtype=bool)
    with pytest.raises(ValueError, match="Cannot aggregate over empty outdoor domain"):
        aggregate_spatial_field_to_scalar(field_2d, all_building)


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


def test_compute_regression_metrics_dictionary():
    """Verify compute_regression_metrics returns complete dictionary with m/s and % units."""
    fixture = create_synthetic_metric_benchmark_fixture()
    metrics = compute_regression_metrics(
        fixture.y_true, fixture.y_pred,
        building_mask=fixture.building_mask,
        u_ref=fixture.u_ref,
    )

    assert "mae_mps" in metrics
    assert "rmse_mps" in metrics
    assert "nrmse_range_percent" in metrics
    assert "nrmse_inflow_percent" in metrics
    assert "relative_l2" in metrics
    assert "r2_score" in metrics
    assert "max_error_mps" in metrics
    assert np.isclose(metrics["mae_mps"], 1.5)
    assert np.isclose(metrics["rmse_mps"], 1.5)
    assert np.isclose(metrics["nrmse_inflow_percent"], 15.0)


def test_spatial_error_map_zeroes_buildings():
    """Spatial error map must contain absolute errors in fluid and 0.0 inside buildings."""
    fixture = create_synthetic_metric_benchmark_fixture()
    err_map = compute_spatial_error_map(
        fixture.y_true, fixture.y_pred,
        building_mask=fixture.building_mask,
    )

    assert err_map.shape == (4, 4)
    # Fluid cells (rows 0..2) should all have error 1.5
    assert np.allclose(err_map[0:3, :], 1.5)
    # Solid building cells (row 3) must be 0.0 despite y_pred having 999.0
    assert np.allclose(err_map[3, :], 0.0)
