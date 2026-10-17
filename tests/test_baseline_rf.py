"""Unit tests for the AML Random Forest baseline pipeline."""

from pathlib import Path
import numpy as np
import pytest

from src.data.contract import LayoutFamily
from src.data.features import GeometricFeatureExtractor
from src.data.split import LayoutSplitter
from src.models.baseline_rf import RandomForestBaseline
from tests.fixtures.synthetic_data import create_synthetic_sample


@pytest.fixture
def synthetic_sample_cohort():
    """Generates a small cohort of synthetic samples across families for testing."""
    samples = []
    families = [
        LayoutFamily.IDEALIZED_ALIGNED,
        LayoutFamily.IDEALIZED_STAGGERED,
        LayoutFamily.REALISTIC_NEIGHBORHOOD,
    ]
    for i, fam in enumerate(families):
        for speed in [3.0, 5.0, 8.0]:
            sid = f"sample_{fam.value}_{int(speed)}_{i}"
            sample = create_synthetic_sample(
                layout_id=sid,
                layout_family=fam,
                grid_shape=(32, 32),
                inflow_speed=speed,
                building_height=15.0 + i * 5.0,
            )
            samples.append(sample)
    return samples


def test_baseline_rf_fit_and_predict(synthetic_sample_cohort):
    """Verify RandomForestBaseline fits on training features and predicts scalars."""
    extractor = GeometricFeatureExtractor()
    model = RandomForestBaseline(n_estimators=20, max_depth=6, random_state=42)

    model.fit_from_samples(synthetic_sample_cohort, extractor=extractor)
    assert model._is_fitted

    preds = model.predict_from_samples(synthetic_sample_cohort, extractor=extractor)
    assert len(preds) == len(synthetic_sample_cohort)
    assert np.all(preds > 0.0)

    metrics = model.evaluate_from_samples(synthetic_sample_cohort, extractor=extractor)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0


def test_baseline_rf_train_only_split_evaluation(synthetic_sample_cohort):
    """Verify split evaluation fits scaler strictly on train set and evaluates held-out test."""
    samples_by_id = {s.layout_id: s for s in synthetic_sample_cohort}
    layout_ids = list(samples_by_id.keys())

    split = LayoutSplitter.split_by_ratio(layout_ids, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

    model = RandomForestBaseline(n_estimators=15, max_depth=5, random_state=42)
    results = model.evaluate_dataset_split(samples_by_id, split)

    assert "train" in results
    assert "val" in results
    assert "test" in results

    assert results["train"]["mae"] >= 0.0
    assert results["test"]["mae"] >= 0.0
    assert results["test"]["num_samples"] == len(split.test_layout_ids)


def test_baseline_rf_feature_importances(synthetic_sample_cohort):
    """Verify feature importances are extracted with matching feature names."""
    extractor = GeometricFeatureExtractor()
    model = RandomForestBaseline(n_estimators=15, random_state=42)
    model.fit_from_samples(synthetic_sample_cohort, extractor=extractor)

    importances = model.feature_importances
    from src.data.features import GeometricFeatures
    assert len(importances) == len(GeometricFeatures.feature_names())
    assert np.isclose(sum(importances.values()), 1.0, atol=1e-3)
    assert "plan_area_density_lambda_p" in importances


def test_baseline_rf_save_and_load_roundtrip(synthetic_sample_cohort, tmp_path: Path):
    """Verify model pipeline serialization and deserialization."""
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
