"""Training/evaluation dataset with correct train-only normalization and masking.

This wraps the existing :class:`~src.data.rasterizer.LayoutRasterizer` and
:class:`~src.data.normalization.FieldNormalizer` (using their correct
``transform``/``inverse_transform`` API) and yields a *fluid* mask (1 = fluid,
0 = building) as the third tensor, which is what the masked training loss
(:class:`~src.training.losses.CombinedSurrogateLoss`) expects.

Why a dedicated wrapper
-----------------------
The project ships two datasets. ``SurrogateDataset`` (training/trainer.py)
returns a correct fluid mask but applies normalization via a
``transform_field`` method that does **not** exist on ``FieldNormalizer``
(it would raise at runtime if a normalizer were passed). ``UrbanSurrogateDataset``
(dataset.py) uses the correct normalization API but returns a *building* mask
(True = building), which the trainer's loss would misinterpret as a fluid
mask. This wrapper combines the correct halves of both: correct
``transform``/``inverse_transform`` normalization **and** a correct fluid
mask. Normalizers are fitted strictly on the training split and verified for
zero leakage against the held-out splits.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.contract import UrbanLayoutSample
from src.data.normalization import FieldNormalizer
from src.data.rasterizer import LayoutRasterizer, RasterizedInput


def fit_field_normalizers(
    train_samples: List[UrbanLayoutSample],
    rasterizer: Optional[LayoutRasterizer] = None,
    method: str = "standard",
) -> Tuple[FieldNormalizer, FieldNormalizer, LayoutRasterizer]:
    """Fit input and target field normalizers strictly on the training split.

    Returns ``(input_normalizer, target_normalizer, rasterizer)``. The
    normalizers record the training layout IDs so leakage can be verified later
    via :meth:`FieldNormalizer.verify_no_leakage`.
    """
    rasterizer = rasterizer or LayoutRasterizer()
    rasters = [rasterizer.rasterize(s) for s in train_samples]

    input_tensors = [r.channels for r in rasters]
    target_tensors = [r.target_field[0:1] for r in rasters]  # pedestrian wind speed only
    masks = [r.building_mask for r in rasters]
    train_ids = [s.layout_id for s in train_samples]

    input_norm = FieldNormalizer(method=method, mask_buildings=True).fit(
        input_tensors, masks=masks, channel_names=rasters[0].channel_names, layout_ids=train_ids
    )
    target_norm = FieldNormalizer(method=method, mask_buildings=True).fit(
        target_tensors, masks=masks, channel_names=(rasters[0].target_names[0],), layout_ids=train_ids
    )
    return input_norm, target_norm, rasterizer


class ExperimentDataset(Dataset):
    """PyTorch dataset yielding normalized inputs, normalized target, and fluid mask.

    ``__getitem__`` returns ``(x, y, fluid_mask, layout_id)`` where:
      * ``x`` is ``(C_in, Ny, Nx)`` float32 — normalized if ``input_normalizer``
        is fitted (building cells kept at 0.0);
      * ``y`` is ``(1, Ny, Nx)`` float32 — pedestrian wind speed, normalized if
        ``target_normalizer`` is fitted (building cells kept at 0.0);
      * ``fluid_mask`` is ``(1, Ny, Nx)`` float32 with 1.0 = fluid, 0.0 = solid
        building (the convention required by the masked training loss).
    """

    def __init__(
        self,
        samples: List[UrbanLayoutSample],
        rasterizer: Optional[LayoutRasterizer] = None,
        input_normalizer: Optional[FieldNormalizer] = None,
        target_normalizer: Optional[FieldNormalizer] = None,
    ):
        self.samples = samples
        self.rasterizer = rasterizer or LayoutRasterizer()
        self.input_normalizer = input_normalizer
        self.target_normalizer = target_normalizer
        self._rasters: List[RasterizedInput] = [self.rasterizer.rasterize(s) for s in samples]

    def __len__(self) -> int:
        return len(self._rasters)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        r = self._rasters[idx]
        mask = r.building_mask
        channels = r.channels.copy()
        target = r.target_field[0:1].copy()  # pedestrian wind speed

        if self.input_normalizer is not None and self.input_normalizer.is_fitted:
            channels = self.input_normalizer.transform(channels, mask=mask)
        if self.target_normalizer is not None and self.target_normalizer.is_fitted:
            target = self.target_normalizer.transform(target, mask=mask)

        x = torch.from_numpy(channels).float()
        y = torch.from_numpy(target).float()
        fluid_mask = (~mask).astype(np.float32)[None, :, :]  # 1 = fluid, 0 = solid
        return x, y, fluid_mask, r.layout_id

    def raw_target(self, idx: int) -> np.ndarray:
        """Unnormalized pedestrian wind speed ground truth (Ny, Nx) for a layout."""
        return self._rasters[idx].target_field[0].copy()

    def building_mask(self, idx: int) -> np.ndarray:
        return self._rasters[idx].building_mask.copy()
