"""Practical predictive-uncertainty estimation via independently trained ensembles.

The uncertainty reported here is **ensemble predictive disagreement** (the
per-pixel standard deviation across independently initialised, trained
models). It quantifies model epistemic spread — where the ensemble members
disagree — and is intentionally **not** claimed to be a calibrated probability
or a confidence interval. Calibration against CFD data has not been evaluated
(because no CFD solver ran here); this limitation is stated explicitly in the
docs and on the figures.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from src.data.contract import UrbanLayoutSample
from src.data.normalization import FieldNormalizer
from src.data.rasterizer import LayoutRasterizer
from src.experiment.dataset import ExperimentDataset


@torch.no_grad()
def ensemble_predict(
    models: List[torch.nn.Module],
    samples: List[UrbanLayoutSample],
    rasterizer: LayoutRasterizer,
    input_normalizer: FieldNormalizer,
    target_normalizer: FieldNormalizer,
    device: torch.device | str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run an ensemble over a set of layouts.

    Returns ``(mean, std, preds, building_mask)`` where:
      * ``mean``  : (M, Ny, Nx) float32 — ensemble-mean prediction in m/s;
      * ``std``   : (M, Ny, Nx) float32 — per-pixel ensemble std (m/s), 0 in solid;
      * ``preds`` : (K, M, Ny, Nx) float32 — each member's prediction in m/s;
      * ``building_mask`` : (Ny, Nx) bool (shared, since all layouts use one grid).
    """
    if isinstance(device, str):
        device = torch.device(device)
    dataset = ExperimentDataset(
        samples=samples, rasterizer=rasterizer,
        input_normalizer=input_normalizer, target_normalizer=target_normalizer,
    )
    M = len(samples)
    K = len(models)
    ny, nx = samples[0].metadata.grid_shape
    preds = np.zeros((K, M, ny, nx), dtype=np.float32)
    bmask = np.zeros((M, ny, nx), dtype=bool)

    for m_idx in range(M):
        x, _, _, _ = dataset[m_idx]
        x = x.unsqueeze(0).to(device)
        bmask[m_idx] = dataset.building_mask(m_idx)
        for k, model in enumerate(models):
            model.eval()
            model.to(device)
            out = model(x).squeeze(0).cpu().numpy()  # (1, Ny, Nx)
            # Inverse-transform to physical units; field_normalizer expects (C, Ny, Nx)
            if target_normalizer is not None and target_normalizer.is_fitted:
                out = target_normalizer.inverse_transform(out, mask=bmask[m_idx])
            out = np.clip(out, 0.0, None)
            out[..., bmask[m_idx]] = 0.0
            preds[k, m_idx] = out[0]

    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    std[..., bmask] = 0.0
    return mean.astype(np.float32), std.astype(np.float32), preds, bmask


def predictive_disagreement(preds: np.ndarray, building_mask: np.ndarray) -> dict:
    """Summarise ensemble disagreement (pixel std) over the fluid domain.

    Parameters
    ----------
    preds : np.ndarray
        (K, Ny, Nx) per-member physical predictions for a single layout.
    building_mask : np.ndarray
        (Ny, Nx) bool, True = solid building.
    """
    fluid = ~building_mask
    std = preds.std(axis=0)
    std[building_mask] = 0.0
    fluid_std = std[fluid]
    return {
        "mean_pixel_std_mps": float(np.mean(fluid_std)) if fluid_std.size else 0.0,
        "max_pixel_std_mps": float(np.max(fluid_std)) if fluid_std.size else 0.0,
        "p95_pixel_std_mps": float(np.percentile(fluid_std, 95)) if fluid_std.size else 0.0,
        "std_field": std.astype(np.float32),
    }
