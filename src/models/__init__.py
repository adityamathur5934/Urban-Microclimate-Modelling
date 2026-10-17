"""Models package for urban microclimate modelling (AML Baseline & Neural Surrogates)."""

from src.models.interfaces import (
    AMLBaselineInterface,
    DEFAULT_SCALAR_WIND_TARGET,
    DEFAULT_WIND_FIELD_TARGET,
    NeuralSurrogateInterface,
    OPTIONAL_TEMPERATURE_FIELD_TARGET,
    OPTIONAL_TKE_FIELD_TARGET,
    TargetConfig,
    TargetType,
)
from src.models.baseline_rf import RandomForestBaseline
from src.models.neural_unet import CompactCNN, CompactUNet

__all__ = [
    "AMLBaselineInterface",
    "NeuralSurrogateInterface",
    "TargetConfig",
    "TargetType",
    "DEFAULT_SCALAR_WIND_TARGET",
    "DEFAULT_WIND_FIELD_TARGET",
    "OPTIONAL_TEMPERATURE_FIELD_TARGET",
    "OPTIONAL_TKE_FIELD_TARGET",
    "RandomForestBaseline",
    "CompactUNet",
    "CompactCNN",
]
