"""Unit tests for the Traditional ML Baseline (Random Forest & Naive benchmark).

Validates:
  1. Target construction & area-averaging physical consistency
  2. Feature/target alignment and leakage isolation
  3. Layout split isolation (zero cross-partition leakage)
  4. Deterministic training and reproducible seeds
  5. Regression metric calculations & Naive Baseline comparisons
  6. Pipeline artifact serialization and round-trip persistence
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.data.area_averaging import compute_area_averaged_wind_speed
from src.data.contract import LayoutFamily, UrbanLayoutSample
from src.data.features import GeometricFeatureExtractor, GeometricFeatures
from src.data.split import DatasetSplit, LayoutSplitter
from src.models.baseline_rf import (
    BaselineConfig,
    NaiveMeanBaseline,
    RandomForestBaseline,
    run_baseline_experiment,
)
from tests.fixtures.synthetic_data import create_synthetic_sample


@pytest.fixture
def synthetic_sample_cohort() -> List[UrbanLayoutSample]:
    """Generates a small cohort of synthetic samples across families for testing."""
    samples = []
    families = [
        LayoutFamily.IDEALIZED_ALIGNED,
        LayoutFamily.IDEALIZED_STAGGERED,
        LayoutFamily.REALISTIC_NEIGHBORHOOD,
    ]
    speeds = [3.0, 5.0, 8.0]
    heights = [15.0, 25.0, 35.0]

    for i, fam in enumerate(families):
        for speed in speeds:
            for h in heights:
                sid = f"sample_{fam.value}_u{int(speed)}_h{int(h)}_{i}"
                sample = create_synthetic_sample(
                    layout_id=sid,
                    layout_family=fam,
                    grid_shape=(32, 32),
                    inflow_speed=speed,
                    building_height=h,
                )
                samples.append(sample)
    return samples


# ===========================================================================
# 1. Target Construction & Area Averaging Tests
# ===========================================================================

class TestTargetConstruction:
    """Verifies canopy area-averaged target calculation adheres to physical fluid masking."""

    def test_area_averaging_fluid_cells_only(self):
        """Verifies building interior cells are excluded from area averaging."""
        ny, nx = 10, 10
        field = np.ones((ny, nx), dtype=np.float32) * 5.0
        mask = np.zeros((ny, nx), dtype=bool)

        # 25% of cells are building solid with wind speed = 0.0
        mask[:5, :5] = True
        field[:5, :5] = 0.0

        # Mean over entire grid would be 5.0 * 0.75 = 3.75
        # Canopy average over outdoor fluid cells must be exactly 5.0
        canopy_avg = compute_area_averaged_wind_speed(field, mask)
        assert np.isclose(canopy_avg, 5.0)

    def test_area_averaging_with_differential_weights(self):
        """Verifies cell-area weighting behaves identically for uniform cells."""
        field = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        mask = np.zeros((2, 2), dtype=bool)
        weights = np.ones((2, 2), dtype=np.float32) * 4.0  # 4 m^2 cells

        avg_unweighted = compute_area_averaged_wind_speed(field, mask)
        avg_weighted = compute_area_averaged_wind_speed(field, mask, weights=weights)
        assert np.isclose(avg_unweighted, avg_weighted)
        assert np.isclose(avg_weighted, 5.0)

    def test_area_averaging_solid_domain_raises(self):
        """100% solid domain (0 fluid cells) must raise ValueError."""
        field = np.ones((8, 8), dtype=np.float32)
        mask = np.ones((8, 8), dtype=bool)
        with pytest.raises(ValueError, match="0 fluid cells"):
            compute_area_averaged_wind_speed(field, mask)

    def test_area_averaging_nan_raises(self):
        """NaN values in flow field must raise ValueError."""
        field = np.ones((8, 8), dtype=np.float32)
        field[2, 2] = np.nan
        mask = np.zeros((8, 8), dtype=bool)
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_area_averaged_wind_speed(field, mask)


# ===========================================================================
# 2. Feature / Target Alignment and Leakage Tests
# ===========================================================================

class TestFeatureTargetAlignment:
    """Verifies feature matrix aligns 1:1 with targets and isolates leakage."""

    def test_feature_matrix_and_target_alignment(self, synthetic_sample_cohort):
        """Verifies X and y shapes align exactly with sample count and feature dimension."""
        extractor = GeometricFeatureExtractor()
        X_list = []
        y_list = []
        for s in synthetic_sample_cohort:
            feat = extractor.extract(s)
            X_list.append(feat.to_feature_vector())
            y_list.append(s.targets.area_averaged_wind_speed)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        assert X.shape[0] == len(synthetic_sample_cohort)
        assert y.shape[0] == len(synthetic_sample_cohort)
        assert X.shape[1] == len(GeometricFeatures.feature_names())
        assert X.shape[1] == 26

    def test_target_leakage_strictly_excluded(self, synthetic_sample_cohort):
        """Verifies layout_id and target values never appear in the predictor matrix."""
        extractor = GeometricFeatureExtractor()
        feature_names = GeometricFeatures.feature_names()

        assert "target_area_averaged_wind_speed" not in feature_names
        assert "area_averaged_wind_speed" not in feature_names
        assert "layout_id" not in feature_names

        for s in synthetic_sample_cohort[:5]:
            feat = extractor.extract(s)
            vec = feat.to_feature_vector()
            target_val = feat.target_area_averaged_wind_speed
            # Target speed should not be directly present as any feature
            assert not any(np.isclose(v, target_val, atol=1e-7) for v in vec if v != s.environment.inflow_speed_mps)


# ===========================================================================
# 3. Layout Split Isolation Tests
# ===========================================================================

class TestLayoutSplitIsolation:
    """Verifies layout-aware partitioning without spatial leakage."""

    def test_split_by_ratio_no_overlap(self, synthetic_sample_cohort):
        """Verifies train, val, and test layout sets have empty pairwise intersections."""
        layout_ids = [s.layout_id for s in synthetic_sample_cohort]
        split = LayoutSplitter.split_by_ratio(
            layout_ids=layout_ids,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )

        train_set = set(split.train_layout_ids)
        val_set = set(split.val_layout_ids)
        test_set = set(split.test_layout_ids)

        assert len(train_set.intersection(val_set)) == 0
        assert len(train_set.intersection(test_set)) == 0
        assert len(val_set.intersection(test_set)) == 0
        assert len(train_set) + len(val_set) + len(test_set) == len(set(layout_ids))

    def test_split_by_family_unseen_holdout(self, synthetic_sample_cohort):
        """Verifies family-level split holds out entire family from train set."""
        test_fam = [LayoutFamily.REALISTIC_NEIGHBORHOOD]
        split = LayoutSplitter.split_by_family(
            samples=synthetic_sample_cohort,
            test_families=test_fam,
            val_ratio=0.20,
            seed=42,
        )

        sample_map = {s.layout_id: s for s in synthetic_sample_cohort}
        for tid in split.test_layout_ids:
            assert sample_map[tid].layout_family == LayoutFamily.REALISTIC_NEIGHBORHOOD

        for tr_id in split.train_layout_ids:
            assert sample_map[tr_id].layout_family != LayoutFamily.REALISTIC_NEIGHBORHOOD

    def test_preprocessing_fitted_only_on_train(self, synthetic_sample_cohort):
        """Verifies StandardScaler statistics are computed strictly on training data."""
        samples_by_id = {s.layout_id: s for s in synthetic_sample_cohort}
        layout_ids = list(samples_by_id.keys())
        split = LayoutSplitter.split_by_ratio(layout_ids, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

        model = RandomForestBaseline(n_estimators=10, random_state=42)
        model.evaluate_dataset_split(samples_by_id, split)

        scaler = model.pipeline.named_steps["scaler"]
        train_samples = [samples_by_id[lid] for lid in split.train_layout_ids]
        extractor = GeometricFeatureExtractor()
        X_train = np.array([extractor.extract(s).to_feature_vector() for s in train_samples], dtype=np.float32)

        # Scaler mean must match X_train mean exactly
        expected_mean = np.mean(X_train, axis=0)
        np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-5)


# ===========================================================================
# 4. Deterministic Setup Tests
# ===========================================================================

class TestDeterministicSetup:
    """Verifies reproducible pipeline behavior given a fixed seed."""

    def test_reproducible_predictions(self, synthetic_sample_cohort):
        """Identical random seeds yield bitwise identical model predictions."""
        extractor = GeometricFeatureExtractor()
        model_a = RandomForestBaseline(n_estimators=25, max_depth=6, random_state=123)
        model_b = RandomForestBaseline(n_estimators=25, max_depth=6, random_state=123)

        model_a.fit_from_samples(synthetic_sample_cohort, extractor=extractor)
        model_b.fit_from_samples(synthetic_sample_cohort, extractor=extractor)

        preds_a = model_a.predict_from_samples(synthetic_sample_cohort, extractor=extractor)
        preds_b = model_b.predict_from_samples(synthetic_sample_cohort, extractor=extractor)

        np.testing.assert_array_equal(preds_a, preds_b)

    def test_reproducible_split(self, synthetic_sample_cohort):
        """Identical random seeds yield identical layout split assignments."""
        layout_ids = [s.layout_id for s in synthetic_sample_cohort]
        split1 = LayoutSplitter.split_by_ratio(layout_ids, seed=99)
        split2 = LayoutSplitter.split_by_ratio(layout_ids, seed=99)

        assert split1.train_layout_ids == split2.train_layout_ids
        assert split1.val_layout_ids == split2.val_layout_ids
        assert split1.test_layout_ids == split2.test_layout_ids


# ===========================================================================
# 5. Metric Calculations and Naive Baseline Tests
# ===========================================================================

class TestMetricCalculations:
    """Verifies regression metrics (MAE, RMSE, R2, NRMSE) and naive comparison."""

    def test_hand_crafted_metric_calculations(self):
        """Tests exact formulas for MAE, RMSE, R2, and NRMSE."""
        model = RandomForestBaseline()
        # Mock fitted state
        model._is_fitted = True

        y_true = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
        y_pred = np.array([2.5, 3.5, 6.5, 7.5], dtype=np.float32)

        # Manually compute:
        # errors = [+0.5, -0.5, +0.5, -0.5]
        # abs_errors = [0.5, 0.5, 0.5, 0.5] -> MAE = 0.5
        # squared errors = [0.25, 0.25, 0.25, 0.25] -> MSE = 0.25, RMSE = 0.5
        # y_true mean = 5.0, var = (( -3)^2 + (-1)^2 + 1^2 + 3^2) / 4 = (9 + 1 + 1 + 9) / 4 = 5.0
        # R2 = 1 - (0.25 / 5.0) = 0.95
        # y_range = 8.0 - 2.0 = 6.0 -> NRMSE_range = 0.5 / 6.0 * 100 = 8.333%
        # y_mean = 5.0 -> NRMSE_mean = 0.5 / 5.0 * 100 = 10.0%
        # max_error = 0.5

        # Patch pipeline.predict to return y_pred
        model.predict = lambda X: y_pred
        metrics = model.evaluate(np.zeros((4, 2)), y_true)

        assert np.isclose(metrics["mae"], 0.5)
        assert np.isclose(metrics["rmse"], 0.5)
        assert np.isclose(metrics["r2"], 0.95)
        assert np.isclose(metrics["nrmse_range"], 8.333333, atol=1e-3)
        assert np.isclose(metrics["nrmse_mean"], 10.0)
        assert np.isclose(metrics["max_error"], 0.5)

    def test_naive_mean_baseline_properties(self):
        """Verifies NaiveMeanBaseline predicts training mean and achieves R^2 = 0 on train."""
        naive = NaiveMeanBaseline()
        y_train = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float32)
        X_train = np.zeros((5, 2))

        naive.fit(X_train, y_train)
        assert np.isclose(naive.mean_val, 6.0)

        preds = naive.predict(X_train)
        assert np.all(preds == 6.0)

        train_metrics = naive.evaluate(X_train, y_train)
        assert np.isclose(train_metrics["r2"], 0.0)
        assert train_metrics["mae"] > 0.0

    def test_rf_outperforms_naive_on_test(self, synthetic_sample_cohort):
        """Verifies RandomForest achieves lower test MAE and higher R^2 than naive mean."""
        samples_by_id = {s.layout_id: s for s in synthetic_sample_cohort}
        layout_ids = list(samples_by_id.keys())
        split = LayoutSplitter.split_by_ratio(layout_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

        extractor = GeometricFeatureExtractor()
        rf = RandomForestBaseline(n_estimators=30, random_state=42)
        rf_results = rf.evaluate_dataset_split(samples_by_id, split, extractor=extractor)

        train_samples = [samples_by_id[lid] for lid in split.train_layout_ids]
        test_samples = [samples_by_id[lid] for lid in split.test_layout_ids]

        naive = NaiveMeanBaseline()
        naive.fit_from_samples(train_samples, extractor=extractor)
        naive_test = naive.evaluate_from_samples(test_samples, extractor=extractor)

        # Random Forest must outperform naive mean baseline
        assert rf_results["test"]["mae"] < naive_test["mae"]
        assert rf_results["test"]["rmse"] < naive_test["rmse"]
        assert rf_results["test"]["r2"] > naive_test["r2"]


# ===========================================================================
# 6. Pipeline Artifacts and Persistence Tests
# ===========================================================================

class TestPipelineArtifactsAndPersistence:
    """Verifies model and experiment persistence to disk."""

    def test_save_and_load_roundtrip(self, synthetic_sample_cohort, tmp_path: Path):
        """Verifies RandomForest model pipeline serialization and deserialization."""
        model = RandomForestBaseline(n_estimators=10, max_depth=4, random_state=42)
        model.fit_from_samples(synthetic_sample_cohort)

        save_file = tmp_path / "rf_baseline.joblib"
        model.save(save_file)
        assert save_file.exists()

        loaded = RandomForestBaseline.load(save_file)
        assert loaded._is_fitted
        assert loaded.n_estimators == 10

        orig_preds = model.predict_from_samples(synthetic_sample_cohort)
        loaded_preds = loaded.predict_from_samples(synthetic_sample_cohort)
        np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5)

    def test_save_and_load_naive_roundtrip(self, tmp_path: Path):
        """Verifies NaiveMeanBaseline serialization and deserialization."""
        naive = NaiveMeanBaseline()
        naive.fit(np.zeros((3, 1)), np.array([3.0, 5.0, 7.0], dtype=np.float32))

        save_file = tmp_path / "naive_baseline.joblib"
        naive.save(save_file)
        assert save_file.exists()

        loaded = NaiveMeanBaseline.load(save_file)
        assert loaded._is_fitted
        assert np.isclose(loaded.mean_val, 5.0)

    def test_run_baseline_experiment_creates_artifacts(self, synthetic_sample_cohort, tmp_path: Path):
        """Verifies run_baseline_experiment generates split, model, and metrics JSON files."""
        out_dir = tmp_path / "baseline_out"
        config = BaselineConfig(n_estimators=10, random_state=42, output_dir=str(out_dir))

        result = run_baseline_experiment(synthetic_sample_cohort, config=config, output_dir=out_dir)

        assert (out_dir / "split_assignments.json").exists()
        assert (out_dir / "rf_baseline.joblib").exists()
        assert (out_dir / "baseline_metrics.json").exists()
        assert result["experiment_status"] == "SUCCESS"
        assert "data_provenance" in result

        # Verify split assignments JSON content
        with open(out_dir / "split_assignments.json") as f:
            split_data = json.load(f)
        assert "train_layout_ids" in split_data
        assert "test_layout_ids" in split_data
        assert split_data["num_train"] > 0
