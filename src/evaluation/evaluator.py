"""Surrogate evaluation pipeline converting normalized outputs back to physical units."""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn

from src.training.dataset import UrbanSurrogateDataset
from src.data.normalization import FieldNormalizer
from src.evaluation.metrics import (
    compute_field_mae,
    compute_field_rmse,
    compute_normalized_rmse,
    compute_relative_l2,
    compute_r2_score,
    compute_spatial_error_map,
)


@dataclass
class LayoutEvaluationResult:
    """Evaluation metrics for a single held-out layout in physical units."""
    layout_id: str
    mae_mps: float
    rmse_mps: float
    nrmse_percent: float
    relative_l2: float
    r2_score: float
    u_ref_mps: float
    inference_time_ms: float


@dataclass
class EvaluationSummary:
    """Aggregated evaluation benchmark across a held-out test split."""
    mean_mae_mps: float
    mean_rmse_mps: float
    mean_nrmse_percent: float
    mean_relative_l2: float
    mean_r2_score: float
    mean_latency_ms: float
    layout_results: List[LayoutEvaluationResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_mae_mps": round(self.mean_mae_mps, 4),
            "mean_rmse_mps": round(self.mean_rmse_mps, 4),
            "mean_nrmse_percent": round(self.mean_nrmse_percent, 2),
            "mean_relative_l2": round(self.mean_relative_l2, 4),
            "mean_r2_score": round(self.mean_r2_score, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "num_layouts": len(self.layout_results),
            "layout_details": [asdict(r) for r in self.layout_results],
        }


class SurrogateEvaluator:
    """Evaluates neural surrogate models on held-out datasets in true physical units."""

    def __init__(self, target_normalizer: Optional[FieldNormalizer] = None):
        self.target_normalizer = target_normalizer

    def evaluate(
        self,
        model: nn.Module,
        dataset: UrbanSurrogateDataset,
        device: torch.device | str = "cpu",
    ) -> EvaluationSummary:
        """Evaluates model over dataset, performing exact inverse normalization to m/s."""
        if isinstance(device, str):
            device = torch.device(device)

        model.eval()
        model.to(device)

        results: List[LayoutEvaluationResult] = []

        with torch.no_grad():
            for i in range(len(dataset)):
                x_tensor, y_tensor, mask_tensor, layout_id = dataset[i]
                sample = dataset.samples[i]
                u_ref = float(sample.environment.inflow_speed_mps)

                # Add batch dimension
                x_in = x_tensor.unsqueeze(0).to(device)

                # Time inference
                start_t = time.perf_counter()
                pred_norm = model(x_in)
                if device.type == "mps":
                    torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0

                pred_norm_np = pred_norm.squeeze(0).cpu().numpy()
                y_norm_np = y_tensor.cpu().numpy()
                mask_np = mask_tensor.squeeze(0).cpu().numpy().astype(bool)

                # Inverse transform predictions back to physical units (m/s)
                if self.target_normalizer is not None and self.target_normalizer.is_fitted:
                    pred_phys = self.target_normalizer.inverse_transform(pred_norm_np, mask=mask_np)
                    # Use unnormalized ground truth from sample directly
                    true_phys = sample.targets.pedestrian_wind_speed.copy()
                    if true_phys.ndim == 2:
                        true_phys = true_phys[None, ...]
                else:
                    pred_phys = pred_norm_np
                    true_phys = y_norm_np

                # Zero out building interiors
                pred_phys[..., mask_np] = 0.0

                mae = compute_field_mae(true_phys, pred_phys, mask_np)
                rmse = compute_field_rmse(true_phys, pred_phys, mask_np)
                nrmse = compute_normalized_rmse(true_phys, pred_phys, mask_np, u_ref=u_ref)
                rel_l2 = compute_relative_l2(true_phys, pred_phys, mask_np)
                r2 = compute_r2_score(true_phys, pred_phys, mask_np)

                results.append(
                    LayoutEvaluationResult(
                        layout_id=layout_id,
                        mae_mps=float(mae),
                        rmse_mps=float(rmse),
                        nrmse_percent=float(nrmse),
                        relative_l2=float(rel_l2),
                        r2_score=float(r2),
                        u_ref_mps=float(u_ref),
                        inference_time_ms=float(elapsed_ms),
                    )
                )

        if not results:
            raise ValueError("Dataset is empty; no evaluation results generated.")

        summary = EvaluationSummary(
            mean_mae_mps=float(np.mean([r.mae_mps for r in results])),
            mean_rmse_mps=float(np.mean([r.rmse_mps for r in results])),
            mean_nrmse_percent=float(np.mean([r.nrmse_percent for r in results])),
            mean_relative_l2=float(np.mean([r.relative_l2 for r in results])),
            mean_r2_score=float(np.mean([r.r2_score for r in results])),
            mean_latency_ms=float(np.mean([r.inference_time_ms for r in results])),
            layout_results=results,
        )
        return summary
