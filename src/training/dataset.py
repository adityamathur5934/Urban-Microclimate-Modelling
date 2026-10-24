"""PyTorch Dataset adapter for rasterized urban microclimate samples."""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np

from src.data.contract import UrbanLayoutSample
from src.data.rasterizer import LayoutRasterizer, RasterizedInput
from src.data.normalization import FieldNormalizer


class UrbanSurrogateDataset(Dataset):
    """PyTorch Dataset yielding normalized multi-channel spatial tensors and aligned targets.

    Parameters
    ----------
    samples : List[UrbanLayoutSample]
        Validated layout samples conforming to the data contract.
    rasterizer : Optional[LayoutRasterizer]
        Rasterizer instance for generating multi-channel inputs and SDFs.
    input_normalizer : Optional[FieldNormalizer]
        Pre-fitted FieldNormalizer for input channels (fitted strictly on training split).
    target_normalizer : Optional[FieldNormalizer]
        Pre-fitted FieldNormalizer for target fields (fitted strictly on training split).
    target_channel_idx : Optional[int]
        Index of the target channel to predict (default 0 for pedestrian wind speed magnitude).
        Set to None to yield all available target channels (e.g. speed, u, v, TKE).
    """

    def __init__(
        self,
        samples: List[UrbanLayoutSample],
        rasterizer: Optional[LayoutRasterizer] = None,
        input_normalizer: Optional[FieldNormalizer] = None,
        target_normalizer: Optional[FieldNormalizer] = None,
        target_channel_idx: Optional[int] = 0,
    ):
        self.samples = samples
        self.rasterizer = rasterizer if rasterizer is not None else LayoutRasterizer()
        self.input_normalizer = input_normalizer
        self.target_normalizer = target_normalizer
        self.target_channel_idx = target_channel_idx

        # Pre-rasterize and cache for rapid training iteration
        self._raster_cache: List[RasterizedInput] = [
            self.rasterizer.rasterize(s) for s in self.samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        raster_data = self._raster_cache[idx]
        mask = raster_data.building_mask  # (Ny, Nx) bool

        channels = raster_data.channels.copy()  # (C_in, Ny, Nx)
        target = raster_data.target_field.copy()  # (C_out, Ny, Nx)

        # Apply train-fitted normalization if provided
        if self.input_normalizer is not None and self.input_normalizer.is_fitted:
            channels = self.input_normalizer.transform(channels, mask=mask)
        if self.target_normalizer is not None and self.target_normalizer.is_fitted:
            target = self.target_normalizer.transform(target, mask=mask)

        # Slice target channel if requested
        if self.target_channel_idx is not None:
            target = target[self.target_channel_idx : self.target_channel_idx + 1]

        # Convert to PyTorch float32 tensors
        x_tensor = torch.from_numpy(channels).float()
        y_tensor = torch.from_numpy(target).float()
        mask_tensor = torch.from_numpy(mask).bool().unsqueeze(0)  # (1, Ny, Nx)

        return x_tensor, y_tensor, mask_tensor, raster_data.layout_id
