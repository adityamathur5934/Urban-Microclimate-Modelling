"""Traditional Machine Learning Baseline using Scikit-Learn Random Forest regression.

Predicts scalar area-averaged physical targets (such as outdoor pedestrian wind speed)
from 1D tabular geometric summary features and atmospheric inflow boundary conditions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.contract import LayoutFamily, UrbanLayoutSample
from src.data.features import GeometricFeatureExtractor, GeometricFeatures
from src.data.split import DatasetSplit, LayoutSplitter
from src.models.interfaces import (
    AMLBaselineInterface,
    DEFAULT_SCALAR_WIND_TARGET,
    TargetConfig,
)


class NaiveMeanBaseline(AMLBaselineInterface):
    """Zero-skill naive baseline predicting the constant scalar mean of the training target.

    Serves as the foundational regression benchmark:
      y_hat = mean(y_train)

    Any scientifically meaningful ML model must achieve significantly lower MAE and RMSE,
    and a strictly positive R^2 relative to this baseline on held-out test splits.
    """

    def __init__(
        self,
        target_config: TargetConfig = DEFAULT_SCALAR_WIND_TARGET,
    ):
        self.target_config = target_config
        self.mean_val: Optional[float] = None
        self._is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> NaiveMeanBaseline:
        """Stores the training-set scalar mean."""
        y_arr = np.asarray(y, dtype=np.float32)
        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1D array of shape (N_samples,), got shape {y_arr.shape}")
        if len(y_arr) == 0:
            raise ValueError("Cannot fit on empty dataset")
        self.mean_val = float(np.mean(y_arr))
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the constant training mean for every sample."""
        if not self._is_fitted or self.mean_val is None:
            raise RuntimeError("Model must be fitted before calling predict()")
        X_arr = np.asarray(X)
        return np.full((len(X_arr),), self.mean_val, dtype=np.float32)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculates regression metrics for the naive baseline.

        Metric definitions:
          - mae: Mean Absolute Error [m/s]
          - rmse: Root Mean Squared Error [m/s]
          - r2: Coefficient of Determination (0.0 on train, <= 0.0 on test)
          - nrmse_range: RMSE normalized by target range (max(y) - min(y)) [%]
          - nrmse_mean: RMSE normalized by target mean |mean(y)| [%]
          - max_error: Maximum absolute residual [m/s]
        """
        y_true = np.asarray(y, dtype=np.float32)
        y_pred = self.predict(X)

        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        mae = float(np.mean(abs_errors))
        mse = float(np.mean(errors ** 2))
        rmse = float(np.sqrt(mse))
        max_error = float(np.max(abs_errors))

        y_var = float(np.var(y_true))
        r2 = float(1.0 - (mse / y_var)) if y_var > 1e-8 else 0.0

        y_range = float(np.max(y_true) - np.min(y_true))
        nrmse_range = float(rmse / y_range * 100.0) if y_range > 1e-8 else 0.0

        y_mean = float(np.mean(y_true))
        nrmse_mean = float(rmse / abs(y_mean) * 100.0) if abs(y_mean) > 1e-8 else 0.0

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "nrmse_range": nrmse_range,
            "nrmse_mean": nrmse_mean,
            "max_error": max_error,
            "num_samples": len(y_true),
        }

    def fit_from_samples(
        self,
        samples: List[UrbanLayoutSample],
        extractor: Optional[GeometricFeatureExtractor] = None,
        target_field: str = "area_averaged_wind_speed",
    ) -> NaiveMeanBaseline:
        """Extracts targets from sample objects and computes training mean."""
        y_list = []
        for s in samples:
            val = getattr(s.targets, target_field, None)
            if val is None:
                if extractor is None:
                    extractor = GeometricFeatureExtractor()
                val = extractor.extract(s).target_area_averaged_wind_speed
            y_list.append(val)
        return self.fit(np.zeros((len(y_list), 1)), np.array(y_list, dtype=np.float32))

    def predict_from_samples(
        self,
        samples: List[UrbanLayoutSample],
        extractor: Optional[GeometricFeatureExtractor] = None,
    ) -> np.ndarray:
        """Predicts constant mean for list of samples."""
        return self.predict(np.zeros((len(samples), 1)))

    def evaluate_from_samples(
        self,
        samples: List[UrbanLayoutSample],
        extractor: Optional[GeometricFeatureExtractor] = None,
        target_field: str = "area_averaged_wind_speed",
    ) -> Dict[str, float]:
        """Evaluates naive baseline on list of samples."""
        y_list = []
        for s in samples:
            val = getattr(s.targets, target_field, None)
            if val is None:
                if extractor is None:
                    extractor = GeometricFeatureExtractor()
                val = extractor.extract(s).target_area_averaged_wind_speed
            y_list.append(val)
        return self.evaluate(np.zeros((len(y_list), 1)), np.array(y_list, dtype=np.float32))

    @property
    def feature_importances(self) -> Dict[str, float]:
        """Naive baseline does not utilize features; returns empty mapping."""
        return {}

    def save(self, path: Union[str, Path]) -> None:
        """Serializes naive baseline parameters to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"mean_val": self.mean_val, "_is_fitted": self._is_fitted}, target_path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> NaiveMeanBaseline:
        """Loads serialized naive baseline from disk."""
        target_path = Path(path)
        payload = joblib.load(target_path)
        model = cls()
        model.mean_val = payload.get("mean_val")
        model._is_fitted = payload.get("_is_fitted", False)
        return model


class RandomForestBaseline(AMLBaselineInterface):
    """Scikit-Learn Random Forest Pipeline for layout-level scalar prediction.

    Pipeline:
      1. StandardScaler (fitted strictly on training split to prevent data leakage)
      2. RandomForestRegressor (ensemble of decision trees with reproducible seed)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        target_config: TargetConfig = DEFAULT_SCALAR_WIND_TARGET,
        feature_names: Optional[List[str]] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.target_config = target_config
        self.feature_names = feature_names or list(GeometricFeatures.feature_names())

        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self._is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestBaseline:
        """Fits scaler and Random Forest regressor strictly on training data."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D array of shape (N_samples, N_features), got shape {X_arr.shape}")
        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1D array of shape (N_samples,), got shape {y_arr.shape}")
        if len(X_arr) != len(y_arr):
            raise ValueError(f"Length mismatch: len(X)={len(X_arr)} != len(y)={len(y_arr)}")
        if len(X_arr) == 0:
            raise ValueError("Cannot fit on empty dataset")

        self.pipeline.fit(X_arr, y_arr)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts scalar target for input feature matrix."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict()")
        X_arr = np.asarray(X, dtype=np.float32)
        return self.pipeline.predict(X_arr).astype(np.float32)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculates MAE, RMSE, R2, NRMSE, and Max Error on given dataset.

        Metric definitions and normalization denominators:
          - mae: Mean Absolute Error, 1/N * sum(|y_true - y_pred|) [m/s]
          - rmse: Root Mean Squared Error, sqrt(1/N * sum((y_true - y_pred)^2)) [m/s]
          - r2: Coefficient of Determination, 1 - (SS_res / SS_tot) [-]
          - nrmse_range: RMSE divided by target range (max(y_true) - min(y_true)) [%]
              Denominator: range = max(y_true) - min(y_true)
          - nrmse_mean: RMSE divided by ground-truth mean (|mean(y_true)|) [%]
              Denominator: |mean(y_true)|
          - max_error: Maximum absolute residual, max(|y_true - y_pred|) [m/s]
        """
        y_true = np.asarray(y, dtype=np.float32)
        y_pred = self.predict(X)

        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        mae = float(np.mean(abs_errors))
        mse = float(np.mean(errors ** 2))
        rmse = float(np.sqrt(mse))
        max_error = float(np.max(abs_errors))

        # Variance of ground truth
        y_var = float(np.var(y_true))
        r2 = float(1.0 - (mse / y_var)) if y_var > 1e-8 else 0.0

        # Normalized RMSE relative to target range [%]
        y_range = float(np.max(y_true) - np.min(y_true))
        nrmse_range = float(rmse / y_range * 100.0) if y_range > 1e-8 else 0.0

        # Normalized RMSE relative to target mean [%]
        y_mean = float(np.mean(y_true))
        nrmse_mean = float(rmse / abs(y_mean) * 100.0) if abs(y_mean) > 1e-8 else 0.0

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "nrmse_range": nrmse_range,
            "nrmse_mean": nrmse_mean,
            "max_error": max_error,
            "num_samples": len(y_true),
        }

    @property
    def feature_importances(self) -> Dict[str, float]:
        """Returns feature importances mapped to feature names."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted to access feature importances")
        rf: RandomForestRegressor = self.pipeline.named_steps["regressor"]
        importances = rf.feature_importances_

        names = self.feature_names
        if len(names) != len(importances):
            names = [f"feature_{i}" for i in range(len(importances))]

        return {name: float(imp) for name, imp in zip(names, importances)}

    # High-level helpers accepting contract data samples
    def fit_from_samples(
        self,
        samples: List[UrbanLayoutSample],
        extractor: Optional[GeometricFeatureExtractor] = None,
        target_field: str = "area_averaged_wind_speed",
    ) -> RandomForestBaseline:
        """Extracts features and fits pipeline on training UrbanLayoutSample objects."""
        if extractor is None:
            extractor = GeometricFeatureExtractor()

        X_list = []
        y_list = []
        for s in samples:
            feat = extractor.extract(s)
            X_list.append(feat.to_feature_vector())
            val = getattr(s.targets, target_field, None)
            if val is None:
                # Fall back to computed area-average from pedestrian speed
                val = feat.target_area_averaged_wind_speed
            y_list.append(val)

        return self.fit(np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32))

    def predict_from_samples(
        self,
        samples: List[UrbanLayoutSample],
        extractor: Optional[GeometricFeatureExtractor] = None,
    ) -> np.ndarray:
        """Extracts features and predicts scalar target for list of UrbanLayoutSample objects."""
        if extractor is None:
            extractor = GeometricFeatureExtractor()
        X_list = [extractor.extract(s).to_feature_vector() for s in samples]
        return self.predict(np.array(X_list, dtype=np.float32))

    def evaluate_from_samples(
        self,
        samples: List[UrbanLayoutSample],
        extractor: Optional[GeometricFeatureExtractor] = None,
        target_field: str = "area_averaged_wind_speed",
    ) -> Dict[str, float]:
        """Evaluates pipeline on a list of held-out UrbanLayoutSample objects."""
        if extractor is None:
            extractor = GeometricFeatureExtractor()

        X_list = []
        y_list = []
        for s in samples:
            feat = extractor.extract(s)
            X_list.append(feat.to_feature_vector())
            val = getattr(s.targets, target_field, None)
            if val is None:
                val = feat.target_area_averaged_wind_speed
            y_list.append(val)

        return self.evaluate(np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32))

    def evaluate_dataset_split(
        self,
        samples_by_id: Dict[str, UrbanLayoutSample],
        split: DatasetSplit,
        extractor: Optional[GeometricFeatureExtractor] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Fits strictly on train split and evaluates on train, val, and held-out test splits."""
        if extractor is None:
            extractor = GeometricFeatureExtractor()

        train_samples = [samples_by_id[lid] for lid in split.train_layout_ids if lid in samples_by_id]
        val_samples = [samples_by_id[lid] for lid in split.val_layout_ids if lid in samples_by_id]
        test_samples = [samples_by_id[lid] for lid in split.test_layout_ids if lid in samples_by_id]

        if not train_samples:
            raise ValueError("No training samples found in samples_by_id matching split.train_layout_ids")

        # Fit strictly on train
        self.fit_from_samples(train_samples, extractor=extractor)

        results = {
            "train": self.evaluate_from_samples(train_samples, extractor=extractor),
        }
        if val_samples:
            results["val"] = self.evaluate_from_samples(val_samples, extractor=extractor)
        if test_samples:
            results["test"] = self.evaluate_from_samples(test_samples, extractor=extractor)

        return results

    def save(self, path: Union[str, Path]) -> None:
        """Saves model pipeline and configuration to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "pipeline": self.pipeline,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
            "target_config": {
                "target_name": self.target_config.target_name,
                "target_type": self.target_config.target_type.value,
                "unit": self.target_config.unit,
                "description": self.target_config.description,
            },
            "_is_fitted": self._is_fitted,
        }
        joblib.dump(payload, target_path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> RandomForestBaseline:
        """Loads model pipeline and configuration from disk."""
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found at {target_path}")

        payload = joblib.load(target_path)
        target_cfg_dict = payload.get("target_config", {})
        target_cfg = TargetConfig(
            target_name=target_cfg_dict.get("target_name", DEFAULT_SCALAR_WIND_TARGET.target_name),
            target_type=DEFAULT_SCALAR_WIND_TARGET.target_type,
            unit=target_cfg_dict.get("unit", "m/s"),
            description=target_cfg_dict.get("description", ""),
        )

        model = cls(
            n_estimators=payload.get("n_estimators", 100),
            max_depth=payload.get("max_depth", 12),
            min_samples_split=payload.get("min_samples_split", 2),
            min_samples_leaf=payload.get("min_samples_leaf", 1),
            random_state=payload.get("random_state", 42),
            target_config=target_cfg,
            feature_names=payload.get("feature_names"),
        )
        model.pipeline = payload["pipeline"]
        model._is_fitted = payload.get("_is_fitted", True)
        return model


@dataclass
class BaselineConfig:
    """Configuration for reproducible AML baseline training and evaluation."""
    n_estimators: int = 100
    max_depth: Optional[int] = 12
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42
    split_method: str = "ratio"  # "ratio" or "family"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    test_families: Optional[List[str]] = None
    target_field: str = "area_averaged_wind_speed"
    output_dir: str = "outputs/baseline"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BaselineConfig:
        return cls(**data)


def run_baseline_experiment(
    samples: List[UrbanLayoutSample],
    config: Optional[BaselineConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Executes a full, reproducible AML baseline training and evaluation run.

    Workflow:
      1. Validates input dataset and partitions layouts strictly by layout ID
         (zero data leakage across splits).
      2. Persists split assignments and configuration to output directory.
      3. Fits StandardScaler and RandomForestRegressor strictly on training data.
      4. Fits NaiveMeanBaseline on training data for benchmark comparison.
      5. Evaluates both models across train, val, and test splits.
      6. Computes feature importances and relative performance improvements.
      7. Persists trained model artifact and full evaluation report.

    Returns
    -------
    Dict[str, Any]
        Dictionary of experiment results, metrics, and artifact paths.
    """
    if config is None:
        config = BaselineConfig()

    out_path = Path(output_dir or config.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not samples:
        raise ValueError("Cannot run baseline experiment on empty sample list")

    samples_by_id = {s.layout_id: s for s in samples}
    layout_ids = list(samples_by_id.keys())

    # 1. Dataset Partitioning
    if config.split_method == "family" and config.test_families:
        family_enums = [LayoutFamily(f) for f in config.test_families]
        split = LayoutSplitter.split_by_family(
            samples=samples,
            test_families=family_enums,
            val_ratio=config.val_ratio,
            seed=config.random_state,
        )
    else:
        split = LayoutSplitter.split_by_ratio(
            layout_ids=layout_ids,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.random_state,
        )

    # Strict leakage validation
    split.validate()

    # 2. Persist Split Assignments
    split_payload = {
        "random_state": config.random_state,
        "split_method": config.split_method,
        "train_layout_ids": split.train_layout_ids,
        "val_layout_ids": split.val_layout_ids,
        "test_layout_ids": split.test_layout_ids,
        "num_train": len(split.train_layout_ids),
        "num_val": len(split.val_layout_ids),
        "num_test": len(split.test_layout_ids),
    }
    split_file = out_path / "split_assignments.json"
    with open(split_file, "w") as f:
        json.dump(split_payload, f, indent=2)

    # 3. Fit and Evaluate Random Forest Baseline
    extractor = GeometricFeatureExtractor()
    rf_model = RandomForestBaseline(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
    )
    rf_results = rf_model.evaluate_dataset_split(
        samples_by_id=samples_by_id,
        split=split,
        extractor=extractor,
    )

    # 4. Fit and Evaluate Naive Mean Baseline
    train_samples = [samples_by_id[lid] for lid in split.train_layout_ids]
    val_samples = [samples_by_id[lid] for lid in split.val_layout_ids if lid in samples_by_id]
    test_samples = [samples_by_id[lid] for lid in split.test_layout_ids if lid in samples_by_id]

    naive_model = NaiveMeanBaseline()
    naive_model.fit_from_samples(train_samples, extractor=extractor, target_field=config.target_field)

    naive_results = {
        "train": naive_model.evaluate_from_samples(train_samples, extractor=extractor, target_field=config.target_field),
    }
    if val_samples:
        naive_results["val"] = naive_model.evaluate_from_samples(val_samples, extractor=extractor, target_field=config.target_field)
    if test_samples:
        naive_results["test"] = naive_model.evaluate_from_samples(test_samples, extractor=extractor, target_field=config.target_field)

    # 5. Relative Improvement over Naive Baseline on Test
    improvement = {}
    if test_samples:
        rf_test_mae = rf_results["test"]["mae"]
        naive_test_mae = naive_results["test"]["mae"]
        rf_test_rmse = rf_results["test"]["rmse"]
        naive_test_rmse = naive_results["test"]["rmse"]

        mae_reduct = ((naive_test_mae - rf_test_mae) / naive_test_mae * 100.0) if naive_test_mae > 1e-6 else 0.0
        rmse_reduct = ((naive_test_rmse - rf_test_rmse) / naive_test_rmse * 100.0) if naive_test_rmse > 1e-6 else 0.0

        improvement = {
            "mae_reduction_percent": round(mae_reduct, 2),
            "rmse_reduction_percent": round(rmse_reduct, 2),
            "r2_delta": round(rf_results["test"]["r2"] - naive_results["test"]["r2"], 4),
        }

    # 6. Feature Importances
    importances = rf_model.feature_importances
    sorted_importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

    # 7. Persist Model and Metrics
    model_file = out_path / "rf_baseline.joblib"
    rf_model.save(model_file)

    # Determine if dataset is synthetic fixture
    is_synthetic = any("synthetic" in s.metadata.city_or_source.lower() or "synthetic" in s.layout_id.lower() for s in samples)
    data_label = (
        "SOFTWARE_SMOKE_TEST_SYNTHETIC_FIXTURE_NOT_CFD_PERFORMANCE"
        if is_synthetic
        else "VERIFIED_CFD_DATASET"
    )

    metrics_payload = {
        "experiment_status": "SUCCESS",
        "data_provenance": data_label,
        "config": config.to_dict(),
        "split_summary": {
            "train_samples": len(split.train_layout_ids),
            "val_samples": len(split.val_layout_ids),
            "test_samples": len(split.test_layout_ids),
        },
        "rf_metrics": rf_results,
        "naive_metrics": naive_results,
        "improvement_over_naive_test": improvement,
        "top_5_features": dict(list(sorted_importances.items())[:5]),
        "artifacts": {
            "model_path": str(model_file),
            "split_path": str(split_file),
        },
    }

    metrics_file = out_path / "baseline_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    return metrics_payload
