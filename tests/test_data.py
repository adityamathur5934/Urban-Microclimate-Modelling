"""Comprehensive test suite for data pipeline, schema contracts, area-averaging, and normalization.

DISCLAIMER:
All tests in this module utilize synthetic geometric fixtures for software verification.
NO DATA IN THESE TESTS REPRESENT REAL CFD OBSERVATIONS.
"""

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
from src.data.area_averaging import compute_area_averaged_wind_speed
from src.data.features import GeometricFeatureExtractor
from src.data.rasterizer import LayoutRasterizer
from src.data.normalization import FieldNormalizer, FeatureNormalizer
from src.data.split import LayoutSplitter, DatasetSplit
from src.data.loader import LayoutDataLoader
from tests.fixtures.synthetic_data import create_synthetic_sample


# ============================================================================
# 1. Schema and Contract Validation Tests
# ============================================================================

def test_schema_full_contract_validation():
    """Verify that a valid synthetic sample passes all contract validation rules."""
    sample = create_synthetic_sample("valid_layout_01", layout_family=LayoutFamily.IDEALIZED_ALIGNED)
    sample.validate()
    assert sample.metadata.layout_id == "valid_layout_01"
    assert sample.metadata.layout_family == LayoutFamily.IDEALIZED_ALIGNED
    assert sample.geometry.height_map.shape == sample.metadata.grid_shape
    assert sample.targets.pedestrian_wind_speed.shape == sample.metadata.grid_shape


def test_schema_rejects_negative_inflow_speed():
    """Inflow speed must be strictly positive."""
    sample = create_synthetic_sample("test_inflow")
    sample.environment = EnvironmentalConditions(
        inflow_speed_mps=-2.0,
        wind_direction_deg=0.0,
        reference_height_m=10.0,
        roughness_length_z0_m=0.05,
    )
    with pytest.raises(ValueError, match="inflow_speed_mps must be positive"):
        sample.validate()


def test_schema_rejects_invalid_wind_angle():
    """Wind direction angle must be in [0, 360)."""
    sample = create_synthetic_sample("test_angle")
    sample.environment = EnvironmentalConditions(
        inflow_speed_mps=5.0,
        wind_direction_deg=365.0,
        reference_height_m=10.0,
        roughness_length_z0_m=0.05,
    )
    with pytest.raises(ValueError, match="wind_direction_deg must be in"):
        sample.validate()


def test_schema_rejects_geometry_nan():
    """Geometry height map containing NaNs must be rejected."""
    sample = create_synthetic_sample("test_geom_nan")
    sample.geometry.height_map[5, 5] = np.nan
    with pytest.raises(ValueError, match="height_map contains NaN or Inf"):
        sample.validate()


# ============================================================================
# 2. Area-Averaging Conversion & Mask Tracking Tests
# ============================================================================

def test_area_averaging_preserves_fluid_values():
    """Area-averaging must average over fluid cells only, ignoring zero-padded building cells."""
    # 4x4 grid: top half fluid (speed = 4.0 m/s), bottom half solid building (mask = True, speed = 0.0)
    field = np.array([
        [4.0, 4.0, 4.0, 4.0],
        [4.0, 4.0, 4.0, 4.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    mask = np.array([
        [False, False, False, False],
        [False, False, False, False],
        [True, True, True, True],
        [True, True, True, True],
    ], dtype=bool)

    avg_speed = compute_area_averaged_wind_speed(field, mask)
    # Fluid cells are all 4.0 m/s, so mean over fluid cells must be exactly 4.0 m/s
    # (An unmasked mean would incorrectly yield 2.0 m/s)
    assert avg_speed == pytest.approx(4.0, abs=1e-5)


def test_area_averaging_with_differential_cell_weights():
    """Verify area averaging with non-uniform differential cell areas (dA)."""
    # 2x2 grid, all fluid
    field = np.array([[2.0, 6.0], [0.0, 0.0]], dtype=np.float32)
    mask = np.zeros((2, 2), dtype=bool)
    # Cell areas: top-left has area 3.0 m^2, top-right has area 1.0 m^2, bottom row masked
    mask[1, :] = True
    weights = np.array([[3.0, 1.0], [2.0, 2.0]], dtype=np.float32)

    # Expected: (2.0 * 3.0 + 6.0 * 1.0) / (3.0 + 1.0) = (6.0 + 6.0) / 4.0 = 3.0 m/s
    avg_speed = compute_area_averaged_wind_speed(field, mask, weights=weights)
    assert avg_speed == pytest.approx(3.0, abs=1e-5)


def test_area_averaging_rejects_100_percent_solid():
    """Domain with 100% solid building cells must raise ValueError."""
    field = np.zeros((4, 4), dtype=np.float32)
    mask = np.ones((4, 4), dtype=bool)
    with pytest.raises(ValueError, match="domain has 0 fluid cells"):
        compute_area_averaged_wind_speed(field, mask)


def test_area_averaging_rejects_shape_mismatch():
    """Field and mask shape mismatch must be caught."""
    field = np.zeros((4, 4), dtype=np.float32)
    mask = np.zeros((4, 5), dtype=bool)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_area_averaged_wind_speed(field, mask)


# ============================================================================
# 3. Spatial Raster Alignment & Signed Distance Transform Tests
# ============================================================================

def test_raster_alignment_channels_and_target():
    """Spatial inputs and targets must be strictly aligned with 7 input channels."""
    sample = create_synthetic_sample("rast_align", grid_shape=(32, 32))
    rasterizer = LayoutRasterizer()
    raster_data = rasterizer.rasterize(sample)

    assert raster_data.channels.ndim == 3
    assert raster_data.channels.shape == (7, 32, 32)
    assert raster_data.target_field.ndim == 3
    assert raster_data.target_field.shape[1:] == (32, 32)
    assert len(raster_data.channel_names) == 7
    assert raster_data.dx_dy_m == (1.0, 1.0)


def test_signed_distance_transform_continuity():
    """SDF must be smooth and signed: >= 0 in fluid, <= 0 in building."""
    sample = create_synthetic_sample("sdf_continuity", grid_shape=(32, 32))
    rasterizer = LayoutRasterizer()
    raster_data = rasterizer.rasterize(sample)

    sdf_idx = raster_data.channel_names.index("signed_distance_m")
    sdf = raster_data.channels[sdf_idx]
    mask = raster_data.building_mask

    assert np.all(sdf[~mask] >= 0.0)
    assert np.all(sdf[mask] <= 0.0)


# ============================================================================
# 4. Normalization & Exact Mathematical Inversion Tests
# ============================================================================

def test_field_normalizer_train_fitted_inversion():
    """Normalization must fit on train only and invert back to exact original physical units."""
    s_train = create_synthetic_sample("train_s1", inflow_speed=5.0)
    s_test = create_synthetic_sample("test_s1", inflow_speed=8.0)
    rasterizer = LayoutRasterizer()

    r_train = rasterizer.rasterize(s_train)
    r_test = rasterizer.rasterize(s_test)

    # Fit strictly on training samples
    normalizer = FieldNormalizer(method="standard", mask_buildings=True)
    normalizer.fit([r_train.target_field], masks=[r_train.building_mask])

    assert normalizer.is_fitted

    # Transform test sample using train stats
    norm_test = normalizer.transform(r_test.target_field, mask=r_test.building_mask)
    recovered_test = normalizer.inverse_transform(norm_test, mask=r_test.building_mask)

    fluid_mask = ~r_test.building_mask
    np.testing.assert_allclose(
        recovered_test[0][fluid_mask],
        r_test.target_field[0][fluid_mask],
        rtol=1e-5,
        atol=1e-5,
    )
    # Masked building cells must remain exactly zero
    assert np.all(recovered_test[0][r_test.building_mask] == 0.0)


def test_feature_normalizer_inversion():
    """Tabular feature normalizer must invert exactly."""
    X = np.array([
        [0.15, 12.0, 5.0],
        [0.35, 25.0, 8.0],
        [0.22, 18.0, 4.0],
    ], dtype=np.float32)

    norm = FeatureNormalizer(method="standard")
    norm.fit(X)
    X_norm = norm.transform(X)
    X_inv = norm.inverse_transform(X_norm)

    np.testing.assert_allclose(X, X_inv, rtol=1e-5, atol=1e-5)


# ============================================================================
# 5. Zero Data Leakage Across Splits Tests
# ============================================================================

def test_split_prevents_layout_overlap():
    """Verify ratio splitter produces disjoint sets with zero overlap."""
    layout_ids = [f"layout_{i:03d}" for i in range(15)]
    split = LayoutSplitter.split_by_ratio(layout_ids, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
    split.validate()

    train_set = set(split.train_layout_ids)
    val_set = set(split.val_layout_ids)
    test_set = set(split.test_layout_ids)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(train_set | val_set | test_set) == 15


# ============================================================================
# 6. End-to-End Pipeline Serialization Roundtrip
# ============================================================================

def test_npz_serialization_and_feature_extraction(tmp_path):
    """Save sample to NPZ, reload, extract geometric features, and verify complete flow."""
    sample = create_synthetic_sample("pipeline_e2e_01", building_height=22.0, inflow_speed=6.0)
    sample_file = tmp_path / "sample_e2e.npz"

    # Save to disk via LayoutDataLoader
    LayoutDataLoader.save_sample_npz(sample, sample_file)
    assert sample_file.exists()

    # Reload from disk
    loaded = LayoutDataLoader.load_sample_npz(sample_file)
    assert loaded.layout_id == sample.layout_id

    # Feature extraction for Model 1 (AML)
    extractor = GeometricFeatureExtractor()
    features = extractor.extract(loaded)
    vec = features.to_feature_vector()
    assert len(vec) == 26
    assert features.target_area_averaged_wind_speed > 0.0

    # Rasterization for Model 2 (Neural Surrogate)
    rasterizer = LayoutRasterizer()
    raster_data = rasterizer.rasterize(loaded)
    assert raster_data.channels.shape == (7, 32, 32)
    assert raster_data.target_field.shape[0] >= 1
