"""Training infrastructure, loss functions, metrics, and visualization for neural surrogates."""

from src.training.dataset import UrbanSurrogateDataset
from src.training.losses import (
    CombinedSurrogateLoss,
    CompositeSurrogateLoss,
    MaskedGradientLoss,
    MaskedL1Loss,
    MaskedMSELoss,
)
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
    SurrogateTrainer,
    TrainerConfig,
    TrainingConfig,
    get_default_device,
    set_seed,
)
from src.training.visualize import (
    plot_baseline_scatter,
    plot_field_comparison,
    plot_training_curves,
)

__all__ = [
    # Losses
    "MaskedL1Loss",
    "MaskedMSELoss",
    "MaskedGradientLoss",
    "CombinedSurrogateLoss",
    "CompositeSurrogateLoss",
    # Datasets
    "UrbanSurrogateDataset",
    "SurrogateDataset",
    # Trainers & Configs
    "NeuralSurrogateTrainer",
    "TrainerConfig",
    "SurrogateTrainer",
    "TrainingConfig",
    "get_default_device",
    "set_seed",
    # Metrics
    "field_mae",
    "field_rmse",
    "field_nrmse",
    "field_max_abs_error",
    "aggregate_spatial_field_to_scalar",
    "compute_surrogate_metrics_summary",
    # Visualizations
    "plot_field_comparison",
    "plot_training_curves",
    "plot_baseline_scatter",
]
