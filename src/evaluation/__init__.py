"""Evaluation metrics, benchmarking, and spatial field plotting for neural surrogates."""

from src.evaluation.metrics import (
    compute_field_mae,
    compute_field_rmse,
    compute_normalized_rmse,
    compute_relative_l2,
    compute_r2_score,
    compute_spatial_error_map,
)
from src.evaluation.evaluator import SurrogateEvaluator, EvaluationSummary
from src.evaluation.plotting import plot_prediction_comparison

__all__ = [
    "compute_field_mae",
    "compute_field_rmse",
    "compute_normalized_rmse",
    "compute_relative_l2",
    "compute_r2_score",
    "compute_spatial_error_map",
    "SurrogateEvaluator",
    "EvaluationSummary",
    "plot_prediction_comparison",
]
