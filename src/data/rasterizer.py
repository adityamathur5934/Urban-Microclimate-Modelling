"""Spatial rasterizer preparing multi-channel 2D inputs and target tensors for neural surrogates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt

from src.data.contract import (
    UrbanLayoutSample,
    ShapeMismatchValidationError,
    NonFiniteValueError,
    PhysicalConstraintValidationError,
)


@dataclass
class RasterizedInput:
    """Multi-channel spatial tensor and aligned target tensor for neural surrogates."""
    layout_id: str
    channels: np.ndarray        # (C_in, Ny, Nx) float32
    channel_names: Tuple[str, ...]
    target_field: np.ndarray    # (C_out, Ny, Nx) float32
    target_names: Tuple[str, ...]
    building_mask: np.ndarray   # (Ny, Nx) bool
    dx_dy_m: Tuple[float, float]


class LayoutRasterizer:
    """Converts UrbanLayoutSample geometry and environment into aligned multi-channel tensors."""

    def __init__(
        self,
        include_distance_transform: bool = True,
        include_inflow_channels: bool = True,
        include_coordinates: bool = True,
        zero_inside_buildings: bool = True,
    ):
        self.include_distance_transform = include_distance_transform
        self.include_inflow_channels = include_inflow_channels
        self.include_coordinates = include_coordinates
        self.zero_inside_buildings = zero_inside_buildings

    def compute_signed_distance_transform(
        self, mask: np.ndarray, dx: float, dy: float
    ) -> np.ndarray:
        """Computes signed distance field (SDF) in meters.

        Positive distance outside buildings (fluid domain),
        negative distance inside buildings (solid domain).
        """
        sampling = (dy, dx)
        # Distance to nearest building pixel for fluid cells (outside)
        dist_out = distance_transform_edt(~mask, sampling=sampling)
        # Distance to nearest fluid pixel for building cells (inside)
        dist_in = distance_transform_edt(mask, sampling=sampling)
        sdf = dist_out - dist_in
        return sdf.astype(np.float32)

    def rasterize(self, sample: UrbanLayoutSample) -> RasterizedInput:
        """Transforms an UrbanLayoutSample into multi-channel input and target arrays."""
        height_map = sample.geometry.height_map.astype(np.float32)
        mask = sample.geometry.building_mask
        dx, dy, _ = sample.metadata.resolution_m
        ny, nx = sample.metadata.grid_shape

        channel_list = []
        name_list = []

        # Channel 1: Building height map (m)
        channel_list.append(height_map)
        name_list.append("height_map_m")

        # Channel 2: Binary building footprint mask (1.0 = building, 0.0 = fluid)
        channel_list.append(mask.astype(np.float32))
        name_list.append("building_mask")

        # Channel 3: Signed distance field (SDF) (m)
        if self.include_distance_transform:
            sdf = self.compute_signed_distance_transform(mask, dx=dx, dy=dy)
            channel_list.append(sdf)
            name_list.append("signed_distance_m")

        # Channels 4 & 5: Inflow velocity vector components (m/s)
        if self.include_inflow_channels:
            u_ref = sample.environment.inflow_speed_mps
            rad = np.deg2rad(sample.environment.wind_direction_deg)
            u_inflow = float(u_ref * np.cos(rad))
            v_inflow = float(u_ref * np.sin(rad))

            u_inflow_grid = np.full((ny, nx), u_inflow, dtype=np.float32)
            v_inflow_grid = np.full((ny, nx), v_inflow, dtype=np.float32)

            channel_list.append(u_inflow_grid)
            name_list.append("inflow_u_mps")
            channel_list.append(v_inflow_grid)
            name_list.append("inflow_v_mps")

        # Channels 6 & 7: Normalized spatial coordinates [0, 1]
        if self.include_coordinates:
            y_coords = np.linspace(0.0, 1.0, ny, dtype=np.float32)[:, None]
            x_coords = np.linspace(0.0, 1.0, nx, dtype=np.float32)[None, :]
            y_grid = np.tile(y_coords, (1, nx))
            x_grid = np.tile(x_coords, (ny, 1))

            channel_list.append(x_grid)
            name_list.append("coord_x_norm")
            channel_list.append(y_grid)
            name_list.append("coord_y_norm")

        stacked_channels = np.stack(channel_list, axis=0)

        # Prepare target fields
        target_list = []
        target_names = []

        # Primary target: pedestrian wind speed (m/s)
        ped_speed = sample.targets.pedestrian_wind_speed.copy().astype(np.float32)
        if self.zero_inside_buildings:
            ped_speed[mask] = 0.0

        target_list.append(ped_speed)
        target_names.append("pedestrian_wind_speed_mps")

        if sample.targets.velocity_u is not None:
            u_field = sample.targets.velocity_u.copy().astype(np.float32)
            if self.zero_inside_buildings:
                u_field[mask] = 0.0
            target_list.append(u_field)
            target_names.append("velocity_u_mps")

        if sample.targets.velocity_v is not None:
            v_field = sample.targets.velocity_v.copy().astype(np.float32)
            if self.zero_inside_buildings:
                v_field[mask] = 0.0
            target_list.append(v_field)
            target_names.append("velocity_v_mps")

        if sample.targets.turbulent_kinetic_energy is not None:
            tke_field = sample.targets.turbulent_kinetic_energy.copy().astype(np.float32)
            if self.zero_inside_buildings:
                tke_field[mask] = 0.0
            target_list.append(tke_field)
            target_names.append("turbulent_kinetic_energy_m2ps2")

        stacked_targets = np.stack(target_list, axis=0)

        return RasterizedInput(
            layout_id=sample.metadata.layout_id,
            channels=stacked_channels,
            channel_names=tuple(name_list),
            target_field=stacked_targets,
            target_names=tuple(target_names),
            building_mask=mask.copy(),
            dx_dy_m=(dx, dy),
        )


def verify_spatial_alignment(
    rasterized: RasterizedInput,
    sample: UrbanLayoutSample,
    check_zero_interiors: bool = True,
) -> bool:
    """Verifies complete spatial and physical alignment between rasterized tensors and layout sample.

    Parameters
    ----------
    rasterized : RasterizedInput
        The output from LayoutRasterizer.rasterize().
    sample : UrbanLayoutSample
        The source urban layout sample.
    check_zero_interiors : bool
        If True, validates that all target channels are strictly zero within building interiors.

    Returns
    -------
    bool
        True if spatial alignment and physical integrity checks pass.

    Raises
    ------
    ShapeMismatchValidationError
        If spatial dimensions (Ny, Nx) do not match metadata grid_shape.
    NonFiniteValueError
        If channels or target fields contain NaN or Inf values.
    PhysicalConstraintValidationError
        If mask mismatches, coordinates are out of bounds, or building interiors are non-zero.
    """
    ny, nx = sample.metadata.grid_shape

    # 1. Identity match
    if rasterized.layout_id != sample.metadata.layout_id:
        raise PhysicalConstraintValidationError(
            f"Layout ID mismatch: rasterized has '{rasterized.layout_id}', sample has '{sample.metadata.layout_id}'"
        )

    # 2. Dimensions and shape checks
    if rasterized.channels.ndim != 3:
        raise ShapeMismatchValidationError(
            f"rasterized.channels must be 3D (C, Ny, Nx), got shape {rasterized.channels.shape}"
        )
    if rasterized.channels.shape[1:] != (ny, nx):
        raise ShapeMismatchValidationError(
            f"rasterized.channels spatial dimensions {rasterized.channels.shape[1:]} != expected grid_shape {(ny, nx)}"
        )
    if rasterized.target_field.ndim != 3:
        raise ShapeMismatchValidationError(
            f"rasterized.target_field must be 3D (C, Ny, Nx), got shape {rasterized.target_field.shape}"
        )
    if rasterized.target_field.shape[1:] != (ny, nx):
        raise ShapeMismatchValidationError(
            f"rasterized.target_field spatial dimensions {rasterized.target_field.shape[1:]} != expected grid_shape {(ny, nx)}"
        )
    if rasterized.building_mask.shape != (ny, nx):
        raise ShapeMismatchValidationError(
            f"rasterized.building_mask shape {rasterized.building_mask.shape} != expected grid_shape {(ny, nx)}"
        )

    # 3. Mask agreement
    if not np.array_equal(rasterized.building_mask, sample.geometry.building_mask):
        mismatches = int(np.count_nonzero(rasterized.building_mask != sample.geometry.building_mask))
        raise PhysicalConstraintValidationError(
            f"rasterized.building_mask does not match sample.geometry.building_mask ({mismatches} mismatched cells)."
        )

    # 4. Non-finite value checks
    if np.any(np.isnan(rasterized.channels)) or np.any(np.isinf(rasterized.channels)):
        nan_c = int(np.count_nonzero(np.isnan(rasterized.channels)))
        inf_c = int(np.count_nonzero(np.isinf(rasterized.channels)))
        raise NonFiniteValueError(f"rasterized.channels contains non-finite values ({nan_c} NaNs, {inf_c} Infs).")

    if np.any(np.isnan(rasterized.target_field)) or np.any(np.isinf(rasterized.target_field)):
        nan_t = int(np.count_nonzero(np.isnan(rasterized.target_field)))
        inf_t = int(np.count_nonzero(np.isinf(rasterized.target_field)))
        raise NonFiniteValueError(f"rasterized.target_field contains non-finite values ({nan_t} NaNs, {inf_t} Infs).")

    # 5. Normalized coordinate bounds
    if "coord_x_norm" in rasterized.channel_names:
        idx_x = rasterized.channel_names.index("coord_x_norm")
        x_grid = rasterized.channels[idx_x]
        if np.min(x_grid) < -1e-5 or np.max(x_grid) > 1.0 + 1e-5:
            raise PhysicalConstraintValidationError(
                f"coord_x_norm values must be in [0, 1], found min={np.min(x_grid):.3f}, max={np.max(x_grid):.3f}"
            )
    if "coord_y_norm" in rasterized.channel_names:
        idx_y = rasterized.channel_names.index("coord_y_norm")
        y_grid = rasterized.channels[idx_y]
        if np.min(y_grid) < -1e-5 or np.max(y_grid) > 1.0 + 1e-5:
            raise PhysicalConstraintValidationError(
                f"coord_y_norm values must be in [0, 1], found min={np.min(y_grid):.3f}, max={np.max(y_grid):.3f}"
            )

    # 6. Building interior zeroing
    if check_zero_interiors and np.any(rasterized.building_mask):
        for c_idx, t_name in enumerate(rasterized.target_names):
            vals_inside = rasterized.target_field[c_idx][rasterized.building_mask]
            if np.any(np.abs(vals_inside) > 1e-6):
                max_inside = float(np.max(np.abs(vals_inside)))
                raise PhysicalConstraintValidationError(
                    f"Target field '{t_name}' contains non-zero values inside building interior (max: {max_inside:.4f})."
                )

    return True
