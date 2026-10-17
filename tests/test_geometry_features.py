"""Tests for geometric feature extraction used by traditional ML baselines."""

import pytest
import numpy as np

from src.data.features import GeometricFeatureExtractor, GeometricFeatures
from src.data.contract import LayoutFamily
from tests.fixtures.synthetic_data import create_synthetic_sample


def test_analytical_plan_area_density():
    """Verify plan area density matches exact analytical count of building cells."""
    sample = create_synthetic_sample(
        "geom_test_01",
        layout_family=LayoutFamily.IDEALIZED_ALIGNED,
        grid_shape=(32, 32),
        building_height=20.0,
    )
    extractor = GeometricFeatureExtractor()
    features = extractor.extract(sample)

    expected_cells = np.count_nonzero(sample.geometry.building_mask)
    expected_lambda_p = expected_cells / (32 * 32)

    assert pytest.approx(features.plan_area_density_lambda_p, abs=1e-4) == expected_lambda_p
    assert pytest.approx(features.mean_building_height_m, abs=1e-4) == 20.0
    assert pytest.approx(features.max_building_height_m, abs=1e-4) == 20.0


def test_feature_vector_conversion():
    """Verify feature vector matches names and contains no NaNs."""
    sample = create_synthetic_sample("geom_test_02")
    extractor = GeometricFeatureExtractor()
    features = extractor.extract(sample)

    vec = features.to_feature_vector()
    names = GeometricFeatures.feature_names()

    assert len(vec) == len(names)
    assert not np.isnan(vec).any()
    assert features.target_area_averaged_wind_speed > 0.0


def test_empty_layout_handles_gracefully():
    """An empty domain (no buildings) should yield 0 for building stats without division by zero."""
    sample = create_synthetic_sample("empty_layout")
    sample.geometry.height_map[:] = 0.0
    sample.geometry.building_mask[:] = False
    sample.targets.pedestrian_wind_speed[:] = 4.0
    sample.targets.area_averaged_wind_speed = 4.0
    sample.validate()

    extractor = GeometricFeatureExtractor()
    features = extractor.extract(sample)

    assert features.plan_area_density_lambda_p == 0.0
    assert features.mean_building_height_m == 0.0
    assert features.frontal_area_density_lambda_f == 0.0
    assert features.displacement_height_d_m == 0.0
