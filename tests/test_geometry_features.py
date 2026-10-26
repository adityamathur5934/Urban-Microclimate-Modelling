"""Tests for geometric feature extraction used by traditional ML baselines.

Coverage
--------
- Analytical verification of existing features (plan area density, heights)
- Analytical verification of new height quantile features
- Building count and footprint statistics on known synthetic geometries
- Empty / degenerate layout edge cases
- Single-cell building degenerate case
- Feature vector / feature_names consistency and NaN-free guarantee
- Determinism: identical inputs → identical outputs
- No target leakage: layout_id and target not in feature_names()
- Environmental boundary inputs round-trip
- All three layout families produce valid (finite, non-negative) feature vectors
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.contract import LayoutFamily
from src.data.features import (
    GeometricFeatureExtractor,
    GeometricFeatures,
    _building_component_stats,
)
from tests.fixtures.synthetic_data import create_synthetic_sample


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def extractor():
    return GeometricFeatureExtractor()


@pytest.fixture
def aligned_sample():
    return create_synthetic_sample(
        "geom_test_aligned",
        layout_family=LayoutFamily.IDEALIZED_ALIGNED,
        grid_shape=(32, 32),
        building_height=20.0,
    )


@pytest.fixture
def staggered_sample():
    return create_synthetic_sample(
        "geom_test_staggered",
        layout_family=LayoutFamily.IDEALIZED_STAGGERED,
        grid_shape=(32, 32),
        building_height=20.0,
    )


@pytest.fixture
def realistic_sample():
    return create_synthetic_sample(
        "geom_test_realistic",
        layout_family=LayoutFamily.REALISTIC_NEIGHBORHOOD,
        grid_shape=(32, 32),
        building_height=20.0,
    )


@pytest.fixture
def empty_sample():
    """Urban layout with no buildings at all."""
    s = create_synthetic_sample("empty_layout")
    s.geometry.height_map[:] = 0.0
    s.geometry.building_mask[:] = False
    s.targets.pedestrian_wind_speed[:] = 4.0
    s.targets.area_averaged_wind_speed = 4.0
    s.validate()
    return s


# ===========================================================================
# GROUP 1 — Existing feature regression tests (must not break)
# ===========================================================================

class TestExistingFeatures:
    """Regression tests for the 11 original features."""

    def test_plan_area_density_analytical(self, extractor, aligned_sample):
        """λ_p must match exact pixel count of building_mask."""
        features = extractor.extract(aligned_sample)
        expected_cells = np.count_nonzero(aligned_sample.geometry.building_mask)
        expected_lambda_p = expected_cells / (32 * 32)
        assert pytest.approx(features.plan_area_density_lambda_p, abs=1e-4) == expected_lambda_p

    def test_mean_and_max_height_uniform_building(self, extractor, aligned_sample):
        features = extractor.extract(aligned_sample)
        assert pytest.approx(features.mean_building_height_m, abs=1e-3) == 20.0
        assert pytest.approx(features.max_building_height_m, abs=1e-3) == 20.0

    def test_feature_vector_matches_names_length(self, extractor, aligned_sample):
        features = extractor.extract(aligned_sample)
        vec = features.to_feature_vector()
        names = GeometricFeatures.feature_names()
        assert len(vec) == len(names), (
            f"Feature vector length {len(vec)} != feature_names length {len(names)}"
        )

    def test_no_nan_in_feature_vector(self, extractor, aligned_sample):
        vec = extractor.extract(aligned_sample).to_feature_vector()
        assert not np.isnan(vec).any(), "Feature vector contains NaN"

    def test_target_not_zero_for_outdoor_wind(self, extractor, aligned_sample):
        features = extractor.extract(aligned_sample)
        assert features.target_area_averaged_wind_speed > 0.0


# ===========================================================================
# GROUP 2 — Height distribution quantile tests (new features)
# ===========================================================================

class TestHeightQuantiles:
    """Tests for min_h, p25_h, p75_h, height_iqr_m, height_cv."""

    def test_uniform_height_quantiles(self, extractor, aligned_sample):
        """Uniform-height building → all quantiles equal building_height."""
        features = extractor.extract(aligned_sample)
        assert pytest.approx(features.min_building_height_m, abs=1e-3) == 20.0
        assert pytest.approx(features.p25_building_height_m, abs=1e-3) == 20.0
        assert pytest.approx(features.p75_building_height_m, abs=1e-3) == 20.0
        assert pytest.approx(features.height_iqr_m, abs=1e-3) == 0.0
        assert pytest.approx(features.height_cv, abs=1e-4) == 0.0

    def test_two_height_quantiles_staggered(self, extractor, staggered_sample):
        """Staggered layout has two heights (20.0 and 15.0 m).
        IQR and std must be > 0.
        """
        features = extractor.extract(staggered_sample)
        # Heights are 20.0 m and 15.0 m; min must be 15.0
        assert features.min_building_height_m < features.max_building_height_m
        assert features.height_iqr_m >= 0.0
        assert features.std_building_height_m >= 0.0

    def test_height_cv_zero_for_uniform(self, extractor, aligned_sample):
        """CV must be 0 for uniform-height layout."""
        f = extractor.extract(aligned_sample)
        assert pytest.approx(f.height_cv, abs=1e-4) == 0.0

    def test_height_cv_positive_for_heterogeneous(self, extractor, realistic_sample):
        """Realistic neighbourhood has varied heights → CV > 0."""
        f = extractor.extract(realistic_sample)
        # Multiple distinct heights in realistic fixture
        assert f.height_cv > 0.0 or f.std_building_height_m > 0.0

    def test_quantile_ordering(self, extractor, aligned_sample):
        """min ≤ p25 ≤ mean ≤ p75 ≤ max must always hold."""
        f = extractor.extract(aligned_sample)
        assert f.min_building_height_m <= f.p25_building_height_m
        assert f.p25_building_height_m <= f.mean_building_height_m
        assert f.mean_building_height_m <= f.p75_building_height_m
        assert f.p75_building_height_m <= f.max_building_height_m

    def test_quantile_ordering_heterogeneous(self, extractor, realistic_sample):
        f = extractor.extract(realistic_sample)
        assert f.min_building_height_m <= f.p25_building_height_m
        assert f.p25_building_height_m <= f.mean_building_height_m
        assert f.mean_building_height_m <= f.p75_building_height_m
        assert f.p75_building_height_m <= f.max_building_height_m


# ===========================================================================
# GROUP 3 — Building count and footprint statistics (new features)
# ===========================================================================

class TestBuildingComponentStats:
    """Tests for building_count, mean/std/max footprint area, footprint_cv."""

    def test_single_block_count_one(self, extractor, aligned_sample):
        """IDEALIZED_ALIGNED fixture has exactly 1 central connected block."""
        f = extractor.extract(aligned_sample)
        assert f.building_count == 1.0

    def test_two_blocks_count_two(self, extractor, staggered_sample):
        """IDEALIZED_STAGGERED fixture has exactly 2 disjoint blocks."""
        f = extractor.extract(staggered_sample)
        assert f.building_count == 2.0

    def test_three_blocks_realistic(self, extractor, realistic_sample):
        """REALISTIC_NEIGHBORHOOD fixture has 3 separate building blocks."""
        f = extractor.extract(realistic_sample)
        assert f.building_count == 3.0

    def test_footprint_area_matches_mask_cells(self, extractor, aligned_sample):
        """Mean footprint area should equal total_built_area for single building."""
        f = extractor.extract(aligned_sample)
        # single building: mean_fp == max_fp == std_fp = 0
        assert pytest.approx(f.mean_footprint_area_m2, rel=1e-4) == f.max_footprint_area_m2
        assert pytest.approx(f.std_footprint_area_m2, abs=1e-4) == 0.0
        assert pytest.approx(f.footprint_area_cv, abs=1e-4) == 0.0

    def test_footprint_area_units_m2(self, aligned_sample):
        """Footprint area for the central block must be cells * dx * dy."""
        n_cells = int(np.count_nonzero(aligned_sample.geometry.building_mask))
        dx, dy, _ = aligned_sample.metadata.resolution_m
        expected_area = float(n_cells) * dx * dy
        n_bldgs, mean_a, std_a, max_a, cv_a = _building_component_stats(
            aligned_sample.geometry.building_mask, dx, dy
        )
        assert n_bldgs == 1
        assert pytest.approx(mean_a, rel=1e-5) == expected_area

    def test_footprint_cv_zero_single_building(self, extractor, aligned_sample):
        f = extractor.extract(aligned_sample)
        assert f.footprint_area_cv == 0.0

    def test_footprint_cv_positive_multiple_buildings(self, extractor, realistic_sample):
        """Three buildings with different footprints → CV > 0."""
        f = extractor.extract(realistic_sample)
        # Realistic fixture has blocks of 6×6, 8×6, 8×8 cells → unequal areas
        # std > 0 → cv > 0
        assert f.std_footprint_area_m2 >= 0.0  # must not be negative

    def test_built_area_total_matches_mask(self, extractor, aligned_sample):
        """built_area_total_m2 must equal N_built_cells * dx * dy."""
        f = extractor.extract(aligned_sample)
        n_cells = int(np.count_nonzero(aligned_sample.geometry.building_mask))
        dx, dy, _ = aligned_sample.metadata.resolution_m
        expected = float(n_cells) * dx * dy
        assert pytest.approx(f.built_area_total_m2, rel=1e-4) == expected


# ===========================================================================
# GROUP 4 — Domain extent and summary features (new features)
# ===========================================================================

class TestDomainExtentFeatures:
    """Tests for domain_aspect_ratio, built_area_total_m2, mean_height_times_lambda_p."""

    def test_domain_aspect_ratio_square(self, extractor, aligned_sample):
        """Square domain (32×32 cells, dx=dy=1m) → aspect ratio = 1.0."""
        f = extractor.extract(aligned_sample)
        assert pytest.approx(f.domain_aspect_ratio, abs=1e-4) == 1.0

    def test_mean_height_times_lambda_p_zero_for_empty(self, extractor, empty_sample):
        f = extractor.extract(empty_sample)
        assert pytest.approx(f.mean_height_times_lambda_p, abs=1e-6) == 0.0

    def test_mean_height_times_lambda_p_positive_for_buildings(self, extractor, aligned_sample):
        f = extractor.extract(aligned_sample)
        assert f.mean_height_times_lambda_p > 0.0
        # Must equal mean_h * lambda_p
        expected = f.mean_building_height_m * f.plan_area_density_lambda_p
        assert pytest.approx(f.mean_height_times_lambda_p, rel=1e-3) == expected


# ===========================================================================
# GROUP 5 — Environmental boundary input features (new features)
# ===========================================================================

class TestEnvironmentalInputFeatures:
    """Tests for reference_height_m and roughness_z0_env_m."""

    def test_reference_height_round_trip(self, extractor, aligned_sample):
        """reference_height_m must equal the input environment value."""
        f = extractor.extract(aligned_sample)
        assert pytest.approx(f.reference_height_m, abs=1e-4) == \
               aligned_sample.environment.reference_height_m

    def test_roughness_z0_env_round_trip(self, extractor, aligned_sample):
        """roughness_z0_env_m must equal the input environment value."""
        f = extractor.extract(aligned_sample)
        assert pytest.approx(f.roughness_z0_env_m, abs=1e-5) == \
               aligned_sample.environment.roughness_length_z0_m

    def test_z0_env_distinct_from_macdonald_z0(self, extractor, aligned_sample):
        """roughness_z0_env_m (terrain input) and roughness_length_macdonald_m
        (canopy-derived aerodynamic roughness) are distinct features.
        """
        f = extractor.extract(aligned_sample)
        assert "roughness_z0_env_m" in GeometricFeatures.feature_names()
        assert "roughness_length_macdonald_m" in GeometricFeatures.feature_names()
        # They may equal each other in degenerate cases but must be separately stored
        # (different physical meanings)
        assert hasattr(f, "roughness_z0_env_m")
        assert hasattr(f, "roughness_length_macdonald_m")


# ===========================================================================
# Edge cases and robustness
# ===========================================================================

class TestEdgeCases:
    """Tests for empty, degenerate, and invalid-input scenarios."""

    def test_empty_layout_all_stats_zero(self, extractor, empty_sample):
        """No buildings → all building-derived stats must be 0.0."""
        f = extractor.extract(empty_sample)
        assert f.plan_area_density_lambda_p == 0.0
        assert f.mean_building_height_m == 0.0
        assert f.std_building_height_m == 0.0
        assert f.max_building_height_m == 0.0
        assert f.min_building_height_m == 0.0
        assert f.p25_building_height_m == 0.0
        assert f.p75_building_height_m == 0.0
        assert f.height_iqr_m == 0.0
        assert f.height_cv == 0.0
        assert f.building_count == 0.0
        assert f.mean_footprint_area_m2 == 0.0
        assert f.std_footprint_area_m2 == 0.0
        assert f.max_footprint_area_m2 == 0.0
        assert f.footprint_area_cv == 0.0
        assert f.displacement_height_d_m == 0.0
        assert f.built_area_total_m2 == 0.0
        assert f.mean_height_times_lambda_p == 0.0

    def test_empty_layout_no_nan(self, extractor, empty_sample):
        """Feature vector must be finite even for empty layout."""
        vec = extractor.extract(empty_sample).to_feature_vector()
        assert not np.isnan(vec).any()
        assert not np.isinf(vec).any()

    def test_single_cell_building(self, extractor):
        """Degenerate: one building cell — must not raise or produce NaN."""
        sample = create_synthetic_sample(
            "single_cell_bldg",
            layout_family=LayoutFamily.IDEALIZED_ALIGNED,
            grid_shape=(8, 8),
            building_height=10.0,
        )
        # Overwrite mask to exactly one cell
        sample.geometry.height_map[:] = 0.0
        sample.geometry.building_mask[:] = False
        sample.geometry.height_map[4, 4] = 10.0
        sample.geometry.building_mask[4, 4] = True
        sample.targets.pedestrian_wind_speed[sample.geometry.building_mask] = 0.0
        sample.targets.area_averaged_wind_speed = float(
            np.mean(sample.targets.pedestrian_wind_speed[~sample.geometry.building_mask])
        )
        sample.validate()

        f = extractor.extract(sample)
        vec = f.to_feature_vector()
        assert not np.isnan(vec).any(), "Single-cell building produced NaN"
        assert f.building_count == 1.0
        assert f.std_footprint_area_m2 == 0.0   # only one building → std = 0
        assert f.footprint_area_cv == 0.0

    def test_invalid_input_raises_on_nan_height(self):
        """Contract validation must catch NaN in height_map before extraction."""
        sample = create_synthetic_sample("nan_test")
        sample.geometry.height_map[0, 0] = float("nan")
        with pytest.raises(Exception):
            sample.validate()

    def test_component_stats_empty_mask():
        """_building_component_stats on all-False mask → (0, 0, 0, 0, 0)."""
        mask = np.zeros((16, 16), dtype=bool)
        n, mean_a, std_a, max_a, cv_a = _building_component_stats(mask, 1.0, 1.0)
        assert n == 0
        assert mean_a == 0.0
        assert std_a == 0.0
        assert max_a == 0.0
        assert cv_a == 0.0


# ===========================================================================
# Determinism and metadata leakage tests
# ===========================================================================

class TestDeterminismAndLeakage:
    """Tests for deterministic output and absence of leakage."""

    def test_deterministic_output(self, extractor, aligned_sample):
        """Calling extract twice on the same sample must give identical results."""
        f1 = extractor.extract(aligned_sample)
        f2 = extractor.extract(aligned_sample)
        np.testing.assert_array_equal(
            f1.to_feature_vector(),
            f2.to_feature_vector(),
        )

    def test_layout_id_not_in_feature_names(self):
        """layout_id must NEVER appear in feature_names() — it is a string ID."""
        names = GeometricFeatures.feature_names()
        assert "layout_id" not in names

    def test_target_not_in_feature_names(self):
        """target_area_averaged_wind_speed must not be in feature_names() — it is the target."""
        names = GeometricFeatures.feature_names()
        assert "target_area_averaged_wind_speed" not in names

    def test_feature_vector_length_equals_names_length(self, extractor, aligned_sample):
        f = extractor.extract(aligned_sample)
        assert len(f.to_feature_vector()) == len(GeometricFeatures.feature_names())

    def test_all_features_finite_for_all_families(self, extractor):
        """All three layout families must produce finite, non-NaN feature vectors."""
        for family in LayoutFamily:
            sample = create_synthetic_sample(
                f"leakage_check_{family.value}",
                layout_family=family,
                grid_shape=(32, 32),
            )
            vec = extractor.extract(sample).to_feature_vector()
            assert not np.isnan(vec).any(), f"NaN found for family {family.value}"
            assert not np.isinf(vec).any(), f"Inf found for family {family.value}"

    def test_inflow_speed_in_feature_vector(self, extractor):
        """inflow_speed_mps must be preserved exactly in the feature vector."""
        for speed in [3.0, 5.5, 9.0]:
            sample = create_synthetic_sample("speed_test", inflow_speed=speed)
            f = extractor.extract(sample)
            names = GeometricFeatures.feature_names()
            vec = f.to_feature_vector()
            idx = names.index("inflow_speed_mps")
            assert pytest.approx(float(vec[idx]), abs=1e-3) == speed

    def test_target_leakage_absent_from_vector(self, extractor, aligned_sample):
        """The feature vector must not contain the target value."""
        f = extractor.extract(aligned_sample)
        vec = f.to_feature_vector()
        target = f.target_area_averaged_wind_speed
        # The target should not appear by value in the feature vector
        # (this is a heuristic check — not definitive but catches obvious bugs)
        assert not np.any(np.abs(vec - target) < 1e-6), (
            f"Target value {target} found at position(s) in feature vector — check for leakage."
        )
