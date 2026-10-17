"""Data contracts and schemas for Urban Microclimate CFD and Neural Surrogates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple
import numpy as np


class LayoutFamily(str, Enum):
    """Categorization of urban building morphology."""
    IDEALIZED_ALIGNED = "idealized_aligned"
    IDEALIZED_STAGGERED = "idealized_staggered"
    REALISTIC_NEIGHBORHOOD = "realistic_neighborhood"


class TargetVariableType(str, Enum):
    """Target variable for microclimate prediction."""
    PEDESTRIAN_WIND_SPEED = "pedestrian_wind_speed"
    TURBULENT_KINETIC_ENERGY = "turbulent_kinetic_energy"
    TEMPERATURE = "temperature"


@dataclass(frozen=True)
class LayoutMetadata:
    """Metadata describing the spatial domain and origin of a layout."""
    layout_id: str
    layout_family: LayoutFamily
    city_or_source: str
    domain_size_m: Tuple[float, float, float]  # (Lx, Ly, Lz) in meters
    resolution_m: Tuple[float, float, float]   # (dx, dy, dz) in meters
    grid_shape: Tuple[int, int]                # (Ny, Nx) -> height, width in pixels

    def validate(self) -> None:
        if not self.layout_id or not isinstance(self.layout_id, str):
            raise ValueError(f"Invalid layout_id: {self.layout_id!r}")
        if not isinstance(self.layout_family, LayoutFamily):
            raise ValueError(f"Invalid layout_family: {self.layout_family!r}")
        if len(self.domain_size_m) != 3 or any(d <= 0 for d in self.domain_size_m):
            raise ValueError(f"domain_size_m must contain 3 positive numbers: {self.domain_size_m}")
        if len(self.resolution_m) != 3 or any(r <= 0 for r in self.resolution_m):
            raise ValueError(f"resolution_m must contain 3 positive numbers: {self.resolution_m}")
        if len(self.grid_shape) != 2 or any(g <= 0 for g in self.grid_shape):
            raise ValueError(f"grid_shape must contain 2 positive integers: {self.grid_shape}")


@dataclass
class GeometryData:
    """Building geometry representations."""
    height_map: np.ndarray  # (Ny, Nx) building heights in meters (0 for ground)
    building_mask: np.ndarray  # (Ny, Nx) bool mask: True where building is present

    def validate(self, expected_shape: Tuple[int, int]) -> None:
        if not isinstance(self.height_map, np.ndarray):
            raise TypeError("height_map must be a numpy ndarray")
        if not isinstance(self.building_mask, np.ndarray):
            raise TypeError("building_mask must be a numpy ndarray")
        if self.height_map.shape != expected_shape:
            raise ValueError(f"height_map shape {self.height_map.shape} != expected {expected_shape}")
        if self.building_mask.shape != expected_shape:
            raise ValueError(f"building_mask shape {self.building_mask.shape} != expected {expected_shape}")
        if np.any(np.isnan(self.height_map)) or np.any(np.isinf(self.height_map)):
            raise ValueError("height_map contains NaN or Inf values")
        if np.any(self.height_map < 0.0):
            raise ValueError("height_map contains negative values; building heights must be >= 0")
        # Mask consistency: where height > 0, mask should be True
        positive_height = self.height_map > 1e-4
        if np.any(positive_height & (~self.building_mask)):
            raise ValueError("Inconsistency: building_mask is False where height_map > 0")


@dataclass(frozen=True)
class EnvironmentalConditions:
    """Atmospheric and inflow boundary conditions."""
    inflow_speed_mps: float       # Reference wind speed in m/s
    wind_direction_deg: float     # Inflow angle: 0 deg = streamwise (+x), 90 deg = spanwise (+y)
    reference_height_m: float     # Height at which inflow_speed is defined (meters)
    roughness_length_z0_m: float  # Aerodynamic roughness length z0 in meters
    ambient_temperature_k: Optional[float] = 293.15  # Kelvin (for thermal extensions)

    def validate(self) -> None:
        if self.inflow_speed_mps <= 0.0:
            raise ValueError(f"inflow_speed_mps must be positive: {self.inflow_speed_mps}")
        if not (0.0 <= self.wind_direction_deg < 360.0):
            raise ValueError(f"wind_direction_deg must be in [0, 360): {self.wind_direction_deg}")
        if self.reference_height_m <= 0.0:
            raise ValueError(f"reference_height_m must be positive: {self.reference_height_m}")
        if self.roughness_length_z0_m <= 0.0:
            raise ValueError(f"roughness_length_z0_m must be positive: {self.roughness_length_z0_m}")
        if self.ambient_temperature_k is not None and self.ambient_temperature_k <= 0.0:
            raise ValueError(f"ambient_temperature_k must be positive: {self.ambient_temperature_k}")


@dataclass(frozen=True)
class CFDSolverMetadata:
    """Metadata regarding the CFD / LES solver setup."""
    solver_name: str  # e.g., "uDALES", "PALM", "OpenFOAM"
    turbulence_model: str  # e.g., "LES-Smagorinsky", "LES-Deardorff", "RANS-k-epsilon"
    boundary_conditions: Dict[str, str] = field(default_factory=dict)
    time_averaging_window_s: Optional[float] = None
    convergence_criteria: Optional[str] = None

    def validate(self) -> None:
        if not self.solver_name:
            raise ValueError("solver_name must not be empty")
        if not self.turbulence_model:
            raise ValueError("turbulence_model must not be empty")


@dataclass
class TargetFields:
    """Target quantities to be predicted by ML baseline or neural surrogate."""
    pedestrian_wind_speed: np.ndarray  # (Ny, Nx) at pedestrian height z=1.75m (m/s)
    velocity_u: Optional[np.ndarray] = None  # (Ny, Nx) streamwise velocity (m/s)
    velocity_v: Optional[np.ndarray] = None  # (Ny, Nx) spanwise velocity (m/s)
    velocity_w: Optional[np.ndarray] = None  # (Ny, Nx) vertical velocity (m/s)
    turbulent_kinetic_energy: Optional[np.ndarray] = None  # (Ny, Nx) TKE in m^2/s^2
    area_averaged_wind_speed: Optional[float] = None  # Scalar target for baseline ML (m/s)

    def validate(self, expected_shape: Tuple[int, int], building_mask: np.ndarray) -> None:
        if not isinstance(self.pedestrian_wind_speed, np.ndarray):
            raise TypeError("pedestrian_wind_speed must be a numpy ndarray")
        if self.pedestrian_wind_speed.shape != expected_shape:
            raise ValueError(f"pedestrian_wind_speed shape {self.pedestrian_wind_speed.shape} != expected {expected_shape}")
        if np.any(np.isnan(self.pedestrian_wind_speed)) or np.any(np.isinf(self.pedestrian_wind_speed)):
            raise ValueError("pedestrian_wind_speed contains NaN or Inf values")
        if np.any(self.pedestrian_wind_speed < -1e-4):
            raise ValueError("pedestrian_wind_speed (scalar magnitude) cannot be negative")

        for name, field_arr in [
            ("velocity_u", self.velocity_u),
            ("velocity_v", self.velocity_v),
            ("velocity_w", self.velocity_w),
            ("turbulent_kinetic_energy", self.turbulent_kinetic_energy),
        ]:
            if field_arr is not None:
                if not isinstance(field_arr, np.ndarray):
                    raise TypeError(f"{name} must be a numpy ndarray")
                if field_arr.shape != expected_shape:
                    raise ValueError(f"{name} shape {field_arr.shape} != expected {expected_shape}")
                if np.any(np.isnan(field_arr)) or np.any(np.isinf(field_arr)):
                    raise ValueError(f"{name} contains NaN or Inf values")

        if self.area_averaged_wind_speed is not None:
            if not isinstance(self.area_averaged_wind_speed, (float, int, np.floating)):
                raise TypeError("area_averaged_wind_speed must be a float")
            if self.area_averaged_wind_speed < 0.0:
                raise ValueError(f"area_averaged_wind_speed cannot be negative: {self.area_averaged_wind_speed}")


@dataclass
class UrbanLayoutSample:
    """Unified container for an urban microclimate sample conforming to the contract."""
    metadata: LayoutMetadata
    geometry: GeometryData
    environment: EnvironmentalConditions
    cfd_solver: CFDSolverMetadata
    targets: TargetFields

    def validate(self) -> None:
        """Thoroughly validate all sub-components and their spatial alignment."""
        self.metadata.validate()
        self.geometry.validate(expected_shape=self.metadata.grid_shape)
        self.environment.validate()
        self.cfd_solver.validate()
        self.targets.validate(
            expected_shape=self.metadata.grid_shape,
            building_mask=self.geometry.building_mask,
        )

    @property
    def layout_id(self) -> str:
        return self.metadata.layout_id

    @property
    def layout_family(self) -> LayoutFamily:
        return self.metadata.layout_family
