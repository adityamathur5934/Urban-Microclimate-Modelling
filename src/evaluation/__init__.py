"""Evaluation metrics, benchmarking, and spatial field plotting for neural surrogates."""

from src.evaluation.metrics import (
    compute_field_mae,
    compute_field_rmse,
    compute_normalized_rmse,
    compute_relative_l2,
    compute_r2_score,
    compute_max_error,
    compute_spatial_error_map,
    compute_regression_metrics,
    field_mae,
    field_rmse,
    field_nrmse,
    field_max_abs_error,
)
from src.evaluation.evaluator import SurrogateEvaluator, EvaluationSummary
from src.evaluation.plotting import (
    plot_prediction_comparison,
    plot_field_triplet,
    plot_field_quad,
    plot_single_field,
)

__all__ = [
    "compute_field_mae",
    "compute_field_rmse",
    "compute_normalized_rmse",
    "compute_relative_l2",
    "compute_r2_score",
    "compute_max_error",
    "compute_spatial_error_map",
    "compute_regression_metrics",
    "field_mae",
    "field_rmse",
    "field_nrmse",
    "field_max_abs_error",
    "SurrogateEvaluator",
    "EvaluationSummary",
    "plot_prediction_comparison",
    "plot_field_triplet",
    "plot_field_quad",
    "plot_single_field",
]
