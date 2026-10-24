"""Training infrastructure, dataset adapters, and loss functions for neural surrogates."""

from src.training.losses import (
    MaskedL1Loss,
    MaskedMSELoss,
    MaskedGradientLoss,
    CompositeSurrogateLoss,
)
from src.training.dataset import UrbanSurrogateDataset
from src.training.trainer import SurrogateTrainer, TrainingConfig

__all__ = [
    "MaskedL1Loss",
    "MaskedMSELoss",
    "MaskedGradientLoss",
    "CompositeSurrogateLoss",
    "UrbanSurrogateDataset",
    "SurrogateTrainer",
    "TrainingConfig",
]
