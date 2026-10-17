"""Interfaces and configuration schemas for ML baseline and neural surrogate models.

This module establishes the explicit data contracts and configuration specifications
between data ingestion, traditional machine learning models, and deep neural surrogates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class TargetType(str, Enum):
    """Specifies the dimensionality and structure of the prediction target."""
    SCALAR = "scalar"        # e.g., area-averaged pedestrian wind speed, total drag
    FIELD_2D = "field_2d"    # e.g., 2D horizontal slice of wind speed or temperature


@dataclass(frozen=True)
class TargetConfig:
    """Explicit configuration defining the target variable and representation.
    
    Prevents hardcoded assumptions about target physical variables (e.g., wind speed
    vs. temperature vs. turbulent kinetic energy).
    """
    target_name: str = "pedestrian_wind_speed"
    target_type: TargetType = TargetType.FIELD_2D
    unit: str = "m/s"
    pedestrian_height_m: float = 1.75
    description: str = "Pedestrian-level horizontal wind speed magnitude at z=1.75m"
    zero_inside_buildings: bool = True
    inflow_normalized: bool = False

    def validate(self) -> None:
        if not self.target_name:
            raise ValueError("target_name cannot be empty")
        if not self.unit:
            raise ValueError("unit cannot be empty")
        if self.pedestrian_height_m <= 0.0:
            raise ValueError(f"pedestrian_height_m must be positive: {self.pedestrian_height_m}")


# Standard target configurations for microclimate modelling
DEFAULT_WIND_FIELD_TARGET = TargetConfig(
    target_name="pedestrian_wind_speed",
    target_type=TargetType.FIELD_2D,
    unit="m/s",
    pedestrian_height_m=1.75,
    description="2D pedestrian-level wind speed field at z=1.75m",
    zero_inside_buildings=True,
)

DEFAULT_SCALAR_WIND_TARGET = TargetConfig(
    target_name="area_averaged_wind_speed",
    target_type=TargetType.SCALAR,
    unit="m/s",
    pedestrian_height_m=1.75,
    description="Area-averaged outdoor pedestrian wind speed for AML baseline",
    zero_inside_buildings=True,
)

OPTIONAL_TEMPERATURE_FIELD_TARGET = TargetConfig(
    target_name="temperature_field",
    target_type=TargetType.FIELD_2D,
    unit="K",
    pedestrian_height_m=1.75,
    description="2D pedestrian-level thermal air temperature field",
    zero_inside_buildings=False,
)

OPTIONAL_TKE_FIELD_TARGET = TargetConfig(
    target_name="turbulent_kinetic_energy",
    target_type=TargetType.FIELD_2D,
    unit="m^2/s^2",
    pedestrian_height_m=1.75,
    description="2D pedestrian-level turbulent kinetic energy field",
    zero_inside_buildings=True,
)


class AMLBaselineInterface(ABC):
    """Abstract base class for tabular Machine Learning baselines."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> AMLBaselineInterface:
        """Fits preprocessing scalers and regression model on training data only."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts scalar target for given tabular features."""
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculates regression metrics (MAE, RMSE, R2, NRMSE) on given partition."""
        pass

    @property
    @abstractmethod
    def feature_importances(self) -> Dict[str, float]:
        """Returns feature importances mapped to feature names."""
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Serializes model pipeline and metadata to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> AMLBaselineInterface:
        """Loads serialized model pipeline from disk."""
        pass


class NeuralSurrogateInterface(ABC):
    """Abstract interface defining the requirements for spatial neural surrogates."""

    @abstractmethod
    def get_param_count(self) -> int:
        """Returns total number of trainable model parameters."""
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        val_metrics: Dict[str, float],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves model weights, optimizer state, and training metadata."""
        pass

    @classmethod
    @abstractmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: str = "cpu",
    ) -> Tuple[NeuralSurrogateInterface, Dict[str, Any]]:
        """Loads model weights and checkpoint metadata from disk."""
        pass
