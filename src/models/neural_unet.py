"""Compact PyTorch Neural Surrogate architectures for spatial microclimate field prediction.

Includes:
  1. CompactUNet: Multi-scale encoder-decoder architecture with skip connections.
  2. CompactCNN: Dilated residual convolutional network architecture.

Both models support configurable input and output channels, arbitrary spatial grid sizes
(divisible by 2^depth for U-Net), and physical outdoor building masking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.interfaces import (
    DEFAULT_WIND_FIELD_TARGET,
    NeuralSurrogateInterface,
    TargetConfig,
)


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm2D -> LeakyReLU(0.2)) * 2 block."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool2D then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling with bilinear interpolation / ConvTranspose2d then DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 if shapes differ slightly due to odd grid dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CompactUNet(nn.Module, NeuralSurrogateInterface):
    """Compact U-Net Neural Surrogate for 2D urban microclimate spatial fields.

    Parameters:
      in_channels: Number of rasterized input channels (default 7: height, mask, sdf,
                   inflow_u, inflow_v, coord_x, coord_y).
      out_channels: Number of predicted spatial target channels (default 1: pedestrian wind speed).
      base_channels: Number of initial feature map filters in stage 1 (default 32).
      depth: Number of downsampling/upsampling resolution stages (default 3, 8x downsampling).
      bilinear: If True, uses bilinear upsampling; otherwise transposed convolutions.
      enforce_zero_inside_buildings: If True, zeros out predictions in solid building cells.
      building_mask_channel_idx: Channel index containing the binary building mask (default 1).
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        bilinear: bool = True,
        enforce_zero_inside_buildings: bool = True,
        building_mask_channel_idx: int = 1,
        target_config: TargetConfig = DEFAULT_WIND_FIELD_TARGET,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth
        self.bilinear = bilinear
        self.enforce_zero_inside_buildings = enforce_zero_inside_buildings
        self.building_mask_channel_idx = building_mask_channel_idx
        self.target_config = target_config

        # Encoder stages
        self.inc = DoubleConv(in_channels, base_channels)
        self.downs = nn.ModuleList()
        current_ch = base_channels
        down_channels = [base_channels]
        for i in range(depth):
            next_ch = current_ch * 2
            self.downs.append(Down(current_ch, next_ch))
            current_ch = next_ch
            down_channels.append(current_ch)

        # Decoder stages: current_ch starts at down_channels[-1]
        self.ups = nn.ModuleList()
        for i in range(depth):
            skip_ch = down_channels[-(i + 2)]
            out_ch = skip_ch
            self.ups.append(Up(current_ch, skip_ch, out_ch, bilinear=bilinear))
            current_ch = out_ch

        # Final prediction head (1x1 conv)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Input:
          x: Tensor of shape (B, C_in, Ny, Nx)
        Output:
          out: Tensor of shape (B, C_out, Ny, Nx)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (B, C, H, W), got shape {x.shape}")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Input channel count mismatch: expected {self.in_channels}, got {x.size(1)}"
            )

        # Retain building mask for physical enforcement if requested
        building_mask = None
        if self.enforce_zero_inside_buildings and x.size(1) > self.building_mask_channel_idx:
            # Building mask is 1 inside buildings, 0 in fluid domain
            building_mask = x[:, self.building_mask_channel_idx : self.building_mask_channel_idx + 1]

        # Encoder path
        skips = [self.inc(x)]
        curr = skips[0]
        for down in self.downs:
            curr = down(curr)
            skips.append(curr)

        # Decoder path with skip connections
        curr = skips[-1]
        for idx, up in enumerate(self.ups):
            skip = skips[-(idx + 2)]
            curr = up(curr, skip)

        out = self.outc(curr)

        # Enforce physical constraint: zero wind speed inside solid building footprint
        if building_mask is not None:
            fluid_mask = (1.0 - building_mask).clamp(0.0, 1.0)
            out = out * fluid_mask

        return out

    def get_param_count(self) -> int:
        """Returns total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        val_metrics: Dict[str, float],
        optimizer_state: Optional[Dict[str, Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves weights, architecture config, and training progress."""
        dest_path = Path(path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_type": "CompactUNet",
            "state_dict": self.state_dict(),
            "epoch": epoch,
            "val_metrics": val_metrics,
            "config": {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "base_channels": self.base_channels,
                "depth": self.depth,
                "bilinear": self.bilinear,
                "enforce_zero_inside_buildings": self.enforce_zero_inside_buildings,
                "building_mask_channel_idx": self.building_mask_channel_idx,
            },
            "target_config": {
                "target_name": self.target_config.target_name,
                "target_type": self.target_config.target_type.value,
                "unit": self.target_config.unit,
                "description": self.target_config.description,
            },
            "param_count": self.get_param_count(),
            "optimizer_state": optimizer_state,
            "extra_metadata": extra_metadata or {},
        }
        torch.save(checkpoint, dest_path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: str = "cpu",
    ) -> Tuple[CompactUNet, Dict[str, Any]]:
        """Loads model weights and checkpoint metadata from disk."""
        src_path = Path(path)
        if not src_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {src_path}")

        checkpoint = torch.load(src_path, map_location=device)
        cfg = checkpoint.get("config", {})
        target_cfg_dict = checkpoint.get("target_config", {})
        target_cfg = TargetConfig(
            target_name=target_cfg_dict.get("target_name", DEFAULT_WIND_FIELD_TARGET.target_name),
            target_type=DEFAULT_WIND_FIELD_TARGET.target_type,
            unit=target_cfg_dict.get("unit", "m/s"),
            description=target_cfg_dict.get("description", ""),
        )

        model = cls(
            in_channels=cfg.get("in_channels", 7),
            out_channels=cfg.get("out_channels", 1),
            base_channels=cfg.get("base_channels", 32),
            depth=cfg.get("depth", 3),
            bilinear=cfg.get("bilinear", True),
            enforce_zero_inside_buildings=cfg.get("enforce_zero_inside_buildings", True),
            building_mask_channel_idx=cfg.get("building_mask_channel_idx", 1),
            target_config=target_cfg,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model, checkpoint


class CompactCNN(nn.Module, NeuralSurrogateInterface):
    """Compact Multi-Scale Residual CNN surrogate.

    An alternative, lightweight baseline without pooling downsamplers, using dilated
    convolutions to expand the receptive field across urban wake structures.
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 4,
        enforce_zero_inside_buildings: bool = True,
        building_mask_channel_idx: int = 1,
        target_config: TargetConfig = DEFAULT_WIND_FIELD_TARGET,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.enforce_zero_inside_buildings = enforce_zero_inside_buildings
        self.building_mask_channel_idx = building_mask_channel_idx
        self.target_config = target_config

        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Intermediate dilated layers for receptive field expansion
        for i in range(num_layers - 2):
            dilation = 2 ** (i % 3)
            layers.extend([
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (B, C, H, W), got shape {x.shape}")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Input channel count mismatch: expected {self.in_channels}, got {x.size(1)}")

        out = self.net(x)

        if self.enforce_zero_inside_buildings and x.size(1) > self.building_mask_channel_idx:
            building_mask = x[:, self.building_mask_channel_idx : self.building_mask_channel_idx + 1]
            fluid_mask = (1.0 - building_mask).clamp(0.0, 1.0)
            out = out * fluid_mask

        return out

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        val_metrics: Dict[str, float],
        optimizer_state: Optional[Dict[str, Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        dest_path = Path(path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_type": "CompactCNN",
            "state_dict": self.state_dict(),
            "epoch": epoch,
            "val_metrics": val_metrics,
            "config": {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "hidden_channels": self.hidden_channels,
                "num_layers": self.num_layers,
                "enforce_zero_inside_buildings": self.enforce_zero_inside_buildings,
                "building_mask_channel_idx": self.building_mask_channel_idx,
            },
            "param_count": self.get_param_count(),
            "optimizer_state": optimizer_state,
            "extra_metadata": extra_metadata or {},
        }
        torch.save(checkpoint, dest_path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: str = "cpu",
    ) -> Tuple[CompactCNN, Dict[str, Any]]:
        src_path = Path(path)
        checkpoint = torch.load(src_path, map_location=device)
        cfg = checkpoint.get("config", {})
        model = cls(
            in_channels=cfg.get("in_channels", 7),
            out_channels=cfg.get("out_channels", 1),
            hidden_channels=cfg.get("hidden_channels", 32),
            num_layers=cfg.get("num_layers", 4),
            enforce_zero_inside_buildings=cfg.get("enforce_zero_inside_buildings", True),
            building_mask_channel_idx=cfg.get("building_mask_channel_idx", 1),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model, checkpoint
