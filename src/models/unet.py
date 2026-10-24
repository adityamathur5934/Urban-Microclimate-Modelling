"""Compact 2D U-Net and baseline CNN architectures for urban flow field prediction."""

from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two successive Convolution-Normalization-Activation blocks with optional Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=not use_batch_norm),
            nn.BatchNorm2d(mid_channels) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(dropout_rate))

        layers.extend([
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_norm),
            nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(0.1, inplace=True),
        ])

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block using MaxPool2d followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block followed by DoubleConv with skip connection concatenation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Handle spatial dimension padding if odd dimensions occur
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        # Concatenate along channel dimension (skip connection)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CompactUNet(nn.Module):
    """Compact 2D U-Net for mapping multi-channel geometry rasters to spatial microclimate fields.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 7: height, mask, signed distance, inflow_u, inflow_v, coord_x, coord_y).
    out_channels : int
        Number of output target channels (default 1: pedestrian wind speed magnitude).
    base_channels : int
        Feature depth at the first convolutional layer (default 32).
    bilinear : bool
        Whether to use bilinear interpolation for upsampling (default True).
    dropout_rate : float
        Dropout probability for regularization and Monte Carlo Dropout uncertainty estimation.
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 1,
        base_channels: int = 32,
        bilinear: bool = True,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear
        self.dropout_rate = dropout_rate

        # Encoder path
        c = base_channels
        self.inc = DoubleConv(in_channels, c, dropout_rate=dropout_rate)
        self.down1 = Down(c, c * 2, dropout_rate=dropout_rate)
        self.down2 = Down(c * 2, c * 4, dropout_rate=dropout_rate)
        self.down3 = Down(c * 4, c * 8, dropout_rate=dropout_rate)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(c * 8, (c * 16) // factor, dropout_rate=dropout_rate)

        # Decoder path with skip connections
        self.up1 = Up(c * 16, (c * 8) // factor, bilinear=bilinear, dropout_rate=dropout_rate)
        self.up2 = Up(c * 8, (c * 4) // factor, bilinear=bilinear, dropout_rate=dropout_rate)
        self.up3 = Up(c * 4, (c * 2) // factor, bilinear=bilinear, dropout_rate=dropout_rate)
        self.up4 = Up(c * 2, c, bilinear=bilinear, dropout_rate=dropout_rate)

        # Output projection (1x1 conv)
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class SimpleCNNBaseline(nn.Module):
    """Direct convolutional baseline without skip connections for ablation comparison."""

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 5,
    ):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.LeakyReLU(0.1, inplace=True),
            ])
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
