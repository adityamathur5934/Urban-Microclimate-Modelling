"""Neural surrogate model architectures for spatial microclimate prediction."""

from src.models.unet import CompactUNet, SimpleCNNBaseline

__all__ = ["CompactUNet", "SimpleCNNBaseline"]
