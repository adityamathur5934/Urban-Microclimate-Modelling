"""Tests for data contract validation, shapes, and boundary conditions."""

import pytest
import numpy as np

from src.data.contract import (
    UrbanLayoutSample,
    LayoutMetadata,
    GeometryData,
    EnvironmentalConditions,
    CFDSolverMetadata,
    TargetFields,
    LayoutFamily,
)
from tests.fixtures.synthetic_data import create_synthetic_sample


def test_valid_synthetic_sample_passes():
    """Verify standard synthetic sample conforms cleanly to the contract."""
    sample = create_synthetic_sample("valid_sample_01")
    sample.validate()
    assert sample.layout_id == "valid_sample_01"
    assert sample.layout_family == LayoutFamily.IDEALIZED_ALIGNED


def test_height_map_shape_mismatch():
    """Shape mismatch between metadata grid_shape and geometry array must fail."""
    sample = create_synthetic_sample("mismatch_shape")
    sample.geometry.height_map = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="shape .* != expected"):
        sample.validate()


def test_target_shape_mismatch():
    """Shape mismatch between geometry and target field must fail."""
    sample = create_synthetic_sample("target_mismatch")
    sample.targets.pedestrian_wind_speed = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="shape .* != expected"):
        sample.validate()


def test_negative_building_heights_rejected():
    """Negative building heights violate physical geometry."""
    sample = create_synthetic_sample("negative_height")
    sample.geometry.height_map[0, 0] = -5.0
    with pytest.raises(ValueError, match="building heights must be >= 0"):
        sample.validate()


def test_negative_wind_speed_rejected():
    """Negative pedestrian wind speed magnitude violates physics."""
    sample = create_synthetic_sample("negative_wind")
    sample.targets.pedestrian_wind_speed[0, 0] = -1.5
    with pytest.raises(ValueError, match="cannot be negative"):
        sample.validate()


def test_mask_height_inconsistency_rejected():
    """Building height > 0 without corresponding building_mask=True must fail."""
    sample = create_synthetic_sample("mask_inconsistency")
    sample.geometry.height_map[0, 0] = 15.0
    sample.geometry.building_mask[0, 0] = False
    with pytest.raises(ValueError, match="Inconsistency: building_mask is False where height_map > 0"):
        sample.validate()


def test_invalid_wind_direction_rejected():
    """Wind direction must be in [0, 360)."""
    sample = create_synthetic_sample("invalid_angle")
    sample.environment = EnvironmentalConditions(
        inflow_speed_mps=5.0,
        wind_direction_deg=370.0,
        reference_height_m=10.0,
        roughness_length_z0_m=0.05,
    )
    with pytest.raises(ValueError, match="wind_direction_deg must be in"):
        sample.validate()


def test_nan_detection():
    """NaNs in spatial target fields must be caught immediately."""
    sample = create_synthetic_sample("nan_test")
    sample.targets.pedestrian_wind_speed[5, 5] = np.nan
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        sample.validate()
