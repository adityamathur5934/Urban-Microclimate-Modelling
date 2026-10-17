"""Traditional Machine Learning Baseline using Scikit-Learn Random Forest regression.

Predicts scalar area-averaged physical targets (such as outdoor pedestrian wind speed)
from 1D tabular geometric summary features and atmospheric inflow boundary conditions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.contract import UrbanLayoutSample
from src.data.features import GeometricFeatureExtractor, GeometricFeatures
from src.data.split import DatasetSplit
from src.models.interfaces import (
    AMLBaselineInterface,
    DEFAULT_SCALAR_WIND_TARGET,
    TargetConfig,
)


class RandomForestBaseline(AMLBaselineInterface):
    """Scikit-Learn Random Forest Pipeline for layout-level scalar prediction.

    Pipeline:
      1. StandardScaler (fitted strictly on training split to prevent leakage)
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
        """Calculates MAE, RMSE, R2, NRMSE, and Max Error on given dataset."""
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

        # Normalized RMSE relative to target range
        y_range = float(np.max(y_true) - np.min(y_true))
        nrmse_range = float(rmse / y_range) if y_range > 1e-8 else 0.0

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "nrmse_range": nrmse_range,
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
