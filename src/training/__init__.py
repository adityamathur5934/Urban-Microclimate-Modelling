"""Training, evaluation metrics, and visualization package for microclimate models."""

from src.training.losses import CombinedSurrogateLoss, MaskedL1Loss, MaskedMSELoss
from src.training.metrics import (
    aggregate_spatial_field_to_scalar,
    compute_surrogate_metrics_summary,
    field_mae,
    field_max_abs_error,
    field_nrmse,
    field_rmse,
)
from src.training.trainer import (
    NeuralSurrogateTrainer,
    SurrogateDataset,
    TrainerConfig,
    get_default_device,
    set_seed,
)
from src.training.visualize import (
    plot_baseline_scatter,
    plot_field_comparison,
    plot_training_curves,
)

__all__ = [
    "MaskedL1Loss",
    "MaskedMSELoss",
    "CombinedSurrogateLoss",
    "field_mae",
    "field_rmse",
    "field_nrmse",
    "field_max_abs_error",
    "aggregate_spatial_field_to_scalar",
    "compute_surrogate_metrics_summary",
    "set_seed",
    "get_default_device",
    "SurrogateDataset",
    "TrainerConfig",
    "NeuralSurrogateTrainer",
    "plot_field_comparison",
    "plot_training_curves",
    "plot_baseline_scatter",
]
