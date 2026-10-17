"""Tests for training-fitted normalizers and exact mathematical inversion."""

import pytest
import numpy as np

from src.data.normalization import FieldNormalizer, FeatureNormalizer
from src.data.rasterizer import LayoutRasterizer
from tests.fixtures.synthetic_data import create_synthetic_sample


def test_field_normalizer_standard_inversion():
    """Verify transform followed by inverse_transform returns original field exactly."""
    s1 = create_synthetic_sample("s1", inflow_speed=4.0)
    s2 = create_synthetic_sample("s2", inflow_speed=7.0)
    rasterizer = LayoutRasterizer()

    r1 = rasterizer.rasterize(s1)
    r2 = rasterizer.rasterize(s2)

    norm = FieldNormalizer(method="standard", mask_buildings=True)
    norm.fit([r1.target_field, r2.target_field], masks=[r1.building_mask, r2.building_mask])

    assert norm.is_fitted

    normalized = norm.transform(r1.target_field, mask=r1.building_mask)
    recovered = norm.inverse_transform(normalized, mask=r1.building_mask)

    # For fluid cells, recovered should match original within float32 tolerance
    fluid_mask = ~r1.building_mask
    np.testing.assert_allclose(
        recovered[0][fluid_mask],
        r1.target_field[0][fluid_mask],
        rtol=1e-5,
        atol=1e-5,
    )
    # Masked cells should remain zero
    assert np.all(recovered[0][r1.building_mask] == 0.0)


def test_field_normalizer_minmax():
    """Verify minmax normalization maps fluid values to [0, 1]."""
    s1 = create_synthetic_sample("s1", inflow_speed=3.0)
    s2 = create_synthetic_sample("s2", inflow_speed=8.0)
    rasterizer = LayoutRasterizer()

    r1 = rasterizer.rasterize(s1)
    r2 = rasterizer.rasterize(s2)

    norm = FieldNormalizer(method="minmax", mask_buildings=True)
    norm.fit([r1.target_field, r2.target_field], masks=[r1.building_mask, r2.building_mask])

    normalized = norm.transform(r1.target_field, mask=r1.building_mask)
    fluid_mask = ~r1.building_mask

    assert np.all(normalized[0][fluid_mask] >= -1e-5)
    assert np.all(normalized[0][fluid_mask] <= 1.0 + 1e-5)


def test_feature_normalizer_roundtrip():
    """Verify FeatureNormalizer fit, transform, inverse_transform, and serialization."""
    X = np.array([
        [0.2, 10.0, 5.0],
        [0.4, 25.0, 7.0],
        [0.1, 15.0, 3.0],
    ], dtype=np.float32)

    norm = FeatureNormalizer(method="standard")
    norm.fit(X)

    X_norm = norm.transform(X)
    X_rec = norm.inverse_transform(X_norm)

    np.testing.assert_allclose(X, X_rec, rtol=1e-5, atol=1e-5)

    # Test serialization to dict and reload
    d = norm.to_dict()
    reloaded_norm = FeatureNormalizer.from_dict(d)
    X_rec2 = reloaded_norm.inverse_transform(X_norm)
    np.testing.assert_allclose(X, X_rec2, rtol=1e-5, atol=1e-5)
