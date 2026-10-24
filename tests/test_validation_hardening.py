"""Unit tests for hardened data validation, actionable error hierarchy, and leakage-safe preprocessing.

All test samples and arrays in this file are small, deterministically generated
synthetic fixtures used strictly for automated testing and verification. None of these
fixtures are claimed to be real CFD or physical simulation outputs.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.data.contract import (
    UrbanLayoutSample,
    LayoutMetadata,
    GeometryData,
    EnvironmentalConditions,
    CFDSolverMetadata,
    TargetFields,
    LayoutFamily,
    DataValidationError,
    MissingFieldValidationError,
    ShapeMismatchValidationError,
    NonFiniteValueError,
    PhysicalConstraintValidationError,
    CoordinateGridValidationError,
)
from src.data.normalization import (
    FieldNormalizer,
    FeatureNormalizer,
    DataLeakageError,
)
from src.data.loader import LayoutDataLoader
from src.data.rasterizer import (
    LayoutRasterizer,
    verify_spatial_alignment,
)
from tests.fixtures.synthetic_data import create_synthetic_sample


# ============================================================================
# 1. Actionable Exception Hierarchy & Input Validation Tests
# ============================================================================

def test_missing_field_validation_error_on_empty_layout_id():
    """Verify that an empty or non-string layout_id raises MissingFieldValidationError."""
    meta = LayoutMetadata(
        layout_id="",
        layout_family=LayoutFamily.IDEALIZED_ALIGNED,
        city_or_source="synthetic_test_case",
        domain_size_m=(64.0, 64.0, 30.0),
        resolution_m=(2.0, 2.0, 2.0),
        grid_shape=(32, 32),
    )
    with pytest.raises(MissingFieldValidationError) as exc_info:
        meta.validate()
    assert "Invalid layout_id" in str(exc_info.value)
    assert issubclass(MissingFieldValidationError, DataValidationError)


def test_coordinate_grid_validation_error_on_inconsistent_domain():
    """Verify that domain bounds inconsistent with resolution * grid_shape raise CoordinateGridValidationError."""
    meta = LayoutMetadata(
        layout_id="inconsistent_grid_01",
        layout_family=LayoutFamily.IDEALIZED_ALIGNED,
        city_or_source="synthetic_test_case",
        domain_size_m=(100.0, 100.0, 30.0),  # Inconsistent: 32 * 2.0 = 64.0 != 100.0
        resolution_m=(2.0, 2.0, 2.0),
        grid_shape=(32, 32),
    )
    with pytest.raises(CoordinateGridValidationError) as exc_info:
        meta.validate()
    assert "Domain dimensions" in str(exc_info.value)
    assert "inconsistent with grid_shape" in str(exc_info.value)


def test_shape_mismatch_validation_error_on_geometry():
    """Verify that geometry arrays not matching grid_shape raise ShapeMismatchValidationError."""
    geom = GeometryData(
        height_map=np.zeros((16, 16), dtype=np.float32),
        building_mask=np.zeros((32, 32), dtype=bool),
    )
    with pytest.raises(ShapeMismatchValidationError) as exc_info:
        geom.validate(expected_shape=(32, 32))
    assert "height_map shape (16, 16) != expected (32, 32)" in str(exc_info.value)


def test_non_finite_value_error_on_nan_inf_geometry():
    """Verify that NaNs or Infs in height_map raise NonFiniteValueError with count diagnostics."""
    hm = np.zeros((32, 32), dtype=np.float32)
    hm[0, 0] = np.nan
    hm[1, 1] = np.inf
    geom = GeometryData(height_map=hm, building_mask=np.zeros((32, 32), dtype=bool))
    with pytest.raises(NonFiniteValueError) as exc_info:
        geom.validate(expected_shape=(32, 32))
    assert "height_map contains NaN or Inf values" in str(exc_info.value)
    assert "1 NaNs" in str(exc_info.value)
    assert "1 Infs" in str(exc_info.value)


def test_physical_constraint_validation_error_on_negative_values():
    """Verify that negative building heights and speeds raise PhysicalConstraintValidationError."""
    # Negative height
    hm = np.zeros((32, 32), dtype=np.float32)
    hm[5, 5] = -5.0
    geom = GeometryData(height_map=hm, building_mask=np.zeros((32, 32), dtype=bool))
    with pytest.raises(PhysicalConstraintValidationError) as exc_info:
        geom.validate(expected_shape=(32, 32))
    assert "height_map contains negative values" in str(exc_info.value)

    # Negative target wind speed
    targets = TargetFields(pedestrian_wind_speed=np.full((32, 32), -1.5, dtype=np.float32))
    with pytest.raises(PhysicalConstraintValidationError) as exc_info:
        targets.validate(expected_shape=(32, 32), building_mask=np.zeros((32, 32), dtype=bool))
    assert "cannot be negative" in str(exc_info.value)

    # Negative inflow speed
    with pytest.raises(PhysicalConstraintValidationError) as exc_info:
        env = EnvironmentalConditions(
            inflow_speed_mps=-4.0,
            wind_direction_deg=0.0,
            reference_height_m=10.0,
            roughness_length_z0_m=0.1,
        )
        env.validate()
    assert "inflow_speed_mps must be positive" in str(exc_info.value)


def test_mask_and_height_mismatch_validation():
    """Verify that building_mask=False where height > 0 raises PhysicalConstraintValidationError."""
    hm = np.zeros((32, 32), dtype=np.float32)
    hm[10, 10] = 20.0
    mask = np.zeros((32, 32), dtype=bool)  # Mismatch: height is positive but mask is False
    geom = GeometryData(height_map=hm, building_mask=mask)
    with pytest.raises(PhysicalConstraintValidationError) as exc_info:
        geom.validate(expected_shape=(32, 32))
    assert "building_mask is False where height_map > 0" in str(exc_info.value)


# ============================================================================
# 2. Leakage-Safe Normalization Tests
# ============================================================================

def test_field_normalizer_leakage_detection():
    """Verify that FieldNormalizer detects overlap between training and holdout layouts."""
    train_layouts = ["layout_01", "layout_02", "layout_03"]
    holdout_clean = ["layout_04", "layout_05"]
    holdout_leaked = ["layout_02", "layout_06"]  # layout_02 leaked!

    tensors = [np.random.RandomState(i).randn(2, 16, 16).astype(np.float32) for i in range(3)]
    masks = [np.zeros((16, 16), dtype=bool) for _ in range(3)]

    norm = FieldNormalizer(method="standard")
    norm.fit(tensors, masks=masks, layout_ids=train_layouts)

    # Clean holdout should pass without exception
    norm.verify_no_leakage(holdout_clean)

    # Overlapping holdout must raise DataLeakageError
    with pytest.raises(DataLeakageError) as exc_info:
        norm.verify_no_leakage(holdout_leaked)
    assert "Data leakage detected!" in str(exc_info.value)
    assert "layout_02" in str(exc_info.value)


def test_feature_normalizer_leakage_detection():
    """Verify that FeatureNormalizer detects overlap between training and holdout layouts."""
    train_layouts = ["layout_A", "layout_B"]
    holdout_clean = ["layout_C"]
    holdout_leaked = ["layout_B", "layout_D"]

    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    norm = FeatureNormalizer(method="standard")
    norm.fit(X, layout_ids=train_layouts)

    norm.verify_no_leakage(holdout_clean)

    with pytest.raises(DataLeakageError) as exc_info:
        norm.verify_no_leakage(holdout_leaked)
    assert "Data leakage detected!" in str(exc_info.value)
    assert "layout_B" in str(exc_info.value)


def test_normalizer_serialization_preserves_leakage_checks():
    """Verify that to_dict and from_dict preserve fitted_layout_ids and leakage detection."""
    train_layouts = ["train_101", "train_102"]
    tensors = [np.ones((1, 8, 8), dtype=np.float32) * 2.0, np.ones((1, 8, 8), dtype=np.float32) * 4.0]

    norm = FieldNormalizer(method="standard")
    norm.fit(tensors, layout_ids=train_layouts)
    serialized = norm.to_dict()

    restored = FieldNormalizer.from_dict(serialized)
    assert restored.fitted_layout_ids_ == train_layouts

    with pytest.raises(DataLeakageError):
        restored.verify_no_leakage(["train_101", "test_201"])


def test_holdout_transform_does_not_mutate_statistics():
    """Verify that normalizing holdout tensors leaves fitted statistics strictly unchanged."""
    train_tensors = [np.full((1, 8, 8), 10.0, dtype=np.float32)]
    norm = FieldNormalizer(method="standard")
    norm.fit(train_tensors, layout_ids=["train_1"])

    mean_before = float(norm.mean_[0])
    std_before = float(norm.std_[0])

    holdout_tensor = np.full((1, 8, 8), 9999.0, dtype=np.float32)
    _ = norm.transform(holdout_tensor)

    assert float(norm.mean_[0]) == mean_before
    assert float(norm.std_[0]) == std_before


# ============================================================================
# 3. Archive Validation & Pre-flight Inspection Tests
# ============================================================================

def test_validate_npz_archive_valid_file(tmp_path):
    """Verify pre-flight validation succeeds on a conforming sample."""
    sample = create_synthetic_sample("valid_archive_01")
    path = tmp_path / "valid.npz"
    LayoutDataLoader.save_sample_npz(sample, path)

    is_valid, issues = LayoutDataLoader.validate_npz_archive(path)
    assert is_valid is True
    assert len(issues) == 0


def test_validate_npz_archive_missing_metadata(tmp_path):
    """Verify pre-flight validation detects missing _metadata_json."""
    path = tmp_path / "no_meta.npz"
    np.savez(path, height_map=np.zeros((8, 8)), building_mask=np.zeros((8, 8)), pedestrian_wind_speed=np.zeros((8, 8)))

    is_valid, issues = LayoutDataLoader.validate_npz_archive(path)
    assert is_valid is False
    assert any("Corrupt or missing metadata" in msg for msg in issues)


def test_validate_npz_archive_missing_arrays(tmp_path):
    """Verify pre-flight validation detects missing required arrays."""
    sample = create_synthetic_sample("missing_arr_01")
    path = tmp_path / "partial.npz"
    # Save metadata only, no height_map
    meta_dict = {
        "metadata": {
            "layout_id": sample.layout_id,
            "layout_family": sample.layout_family.value,
            "city_or_source": sample.metadata.city_or_source,
            "domain_size_m": list(sample.metadata.domain_size_m),
            "resolution_m": list(sample.metadata.resolution_m),
            "grid_shape": list(sample.metadata.grid_shape),
        },
        "environment": {"inflow_speed_mps": 5.0, "wind_direction_deg": 0.0, "reference_height_m": 10.0, "roughness_length_z0_m": 0.1},
        "cfd_solver": {"solver_name": "uDALES", "turbulence_model": "LES-Deardorff"},
    }
    np.savez(path, _metadata_json=np.array([json.dumps(meta_dict)]))

    is_valid, issues = LayoutDataLoader.validate_npz_archive(path)
    assert is_valid is False
    assert any("Missing required array 'height_map'" in msg for msg in issues)


def test_validate_npz_archive_detects_non_finite_values(tmp_path):
    """Verify pre-flight validation catches NaNs or Infs inside archive arrays."""
    sample = create_synthetic_sample("nan_archive_01")
    path = tmp_path / "nan_archive.npz"
    LayoutDataLoader.save_sample_npz(sample, path)

    # Tamper with file by injecting NaN into pedestrian_wind_speed
    with np.load(path) as data:
        arrays = dict(data)
    arrays["pedestrian_wind_speed"][0, 0] = np.nan
    np.savez_compressed(path, **arrays)

    is_valid, issues = LayoutDataLoader.validate_npz_archive(path)
    assert is_valid is False
    assert any("contains non-finite values" in msg for msg in issues)


# ============================================================================
# 4. Spatial Alignment Verification Tests
# ============================================================================

def test_verify_spatial_alignment_success():
    """Verify that a freshly rasterized synthetic sample passes spatial alignment."""
    sample = create_synthetic_sample("align_ok_01")
    rasterizer = LayoutRasterizer()
    rasterized = rasterizer.rasterize(sample)

    assert verify_spatial_alignment(rasterized, sample) is True


def test_verify_spatial_alignment_rejects_identity_mismatch():
    """Verify spatial alignment fails if layout_id does not match."""
    sample = create_synthetic_sample("id_orig")
    rasterizer = LayoutRasterizer()
    rasterized = rasterizer.rasterize(sample)
    rasterized.layout_id = "id_mismatch"

    with pytest.raises(PhysicalConstraintValidationError) as exc_info:
        verify_spatial_alignment(rasterized, sample)
    assert "Layout ID mismatch" in str(exc_info.value)


def test_verify_spatial_alignment_rejects_mask_mismatch():
    """Verify spatial alignment fails if building_mask differs from sample geometry."""
    sample = create_synthetic_sample("mask_mismatch_01")
    rasterizer = LayoutRasterizer()
    rasterized = rasterizer.rasterize(sample)
    # Tamper with building_mask
    rasterized.building_mask[0, 0] = not rasterized.building_mask[0, 0]

    with pytest.raises(PhysicalConstraintValidationError) as exc_info:
        verify_spatial_alignment(rasterized, sample)
    assert "does not match sample.geometry.building_mask" in str(exc_info.value)


def test_verify_spatial_alignment_rejects_non_zero_interior_target():
    """Verify spatial alignment fails if target values are non-zero inside buildings."""
    sample = create_synthetic_sample("interior_nonzero_01", building_height=25.0)
    rasterizer = LayoutRasterizer(zero_inside_buildings=True)
    rasterized = rasterizer.rasterize(sample)

    # Find a building cell and contaminate target
    b_cells = np.argwhere(sample.geometry.building_mask)
    assert len(b_cells) > 0
    by, bx = b_cells[0]
    rasterized.target_field[0, by, bx] = 3.5

    with pytest.raises(PhysicalConstraintValidationError) as exc_info:
        verify_spatial_alignment(rasterized, sample, check_zero_interiors=True)
    assert "contains non-zero values inside building interior" in str(exc_info.value)


def test_verify_spatial_alignment_rejects_non_finite_channels():
    """Verify spatial alignment fails if input channels contain NaN."""
    sample = create_synthetic_sample("nan_channel_01")
    rasterizer = LayoutRasterizer()
    rasterized = rasterizer.rasterize(sample)
    rasterized.channels[0, 0, 0] = np.nan

    with pytest.raises(NonFiniteValueError) as exc_info:
        verify_spatial_alignment(rasterized, sample)
    assert "rasterized.channels contains non-finite values" in str(exc_info.value)
