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


# ============================================================================
# Actionable Data Validation Exception Hierarchy
# ============================================================================

class DataValidationError(ValueError):
    """Base class for all dataset contract and validation errors."""
    pass


class MissingFieldValidationError(DataValidationError, KeyError):
    """Raised when a required metadata, geometry, or target field is missing."""
    pass


class ShapeMismatchValidationError(DataValidationError):
    """Raised when array dimensions do not match the expected grid shape."""
    pass


class NonFiniteValueError(DataValidationError):
    """Raised when NaN or Inf values are encountered in physical fields."""
    pass


class PhysicalConstraintValidationError(DataValidationError):
    """Raised when physical invariants (positive heights, positive speeds, etc.) are violated."""
    pass


class CoordinateGridValidationError(DataValidationError):
    """Raised when spatial domain bounds and resolution do not match grid shape."""
    pass


class IncompatibleUnitsError(DataValidationError):
    """Raised when non-SI or unconvertible units are encountered."""
    pass


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
            raise MissingFieldValidationError(
                f"Invalid layout_id: {self.layout_id!r}. layout_id must be a non-empty string identifier."
            )
        if not isinstance(self.layout_family, LayoutFamily):
            raise PhysicalConstraintValidationError(
                f"Invalid layout_family: {self.layout_family!r}. Must be an instance of LayoutFamily "
                f"({[f.value for f in LayoutFamily]})."
            )
        if len(self.domain_size_m) != 3 or any(d <= 0 for d in self.domain_size_m):
            raise PhysicalConstraintValidationError(
                f"domain_size_m must contain 3 positive numbers (Lx, Ly, Lz in meters), received: {self.domain_size_m}"
            )
        if len(self.resolution_m) != 3 or any(r <= 0 for r in self.resolution_m):
            raise PhysicalConstraintValidationError(
                f"resolution_m must contain 3 positive numbers (dx, dy, dz in meters), received: {self.resolution_m}"
            )
        if len(self.grid_shape) != 2 or any(g <= 0 for g in self.grid_shape):
            raise ShapeMismatchValidationError(
                f"grid_shape must contain 2 positive integers (Ny, Nx in cells), received: {self.grid_shape}"
            )

        # Coordinate grid consistency check: domain size should be consistent with grid_shape * resolution
        ny, nx = self.grid_shape
        dx, dy, _ = self.resolution_m
        lx, ly, _ = self.domain_size_m
        if abs(lx - nx * dx) > 0.05 * lx or abs(ly - ny * dy) > 0.05 * ly:
            raise CoordinateGridValidationError(
                f"Domain dimensions ({lx:.1f}m, {ly:.1f}m) are inconsistent with grid_shape "
                f"{self.grid_shape} and resolution ({dx:.2f}m, {dy:.2f}m). "
                f"Expected Lx ≈ Nx*dx = {nx*dx:.1f}m and Ly ≈ Ny*dy = {ny*dy:.1f}m."
            )


@dataclass
class GeometryData:
    """Building geometry representations."""
    height_map: np.ndarray  # (Ny, Nx) building heights in meters (0 for ground)
    building_mask: np.ndarray  # (Ny, Nx) bool mask: True where building is present

    def validate(self, expected_shape: Tuple[int, int]) -> None:
        if not isinstance(self.height_map, np.ndarray):
            raise TypeError(f"height_map must be a numpy ndarray, got {type(self.height_map).__name__}")
        if not isinstance(self.building_mask, np.ndarray):
            raise TypeError(f"building_mask must be a numpy ndarray, got {type(self.building_mask).__name__}")
        if self.height_map.shape != expected_shape:
            raise ShapeMismatchValidationError(
                f"height_map shape {self.height_map.shape} != expected {expected_shape}. "
                f"Ensure height_map matches grid_shape (Ny, Nx)."
            )
        if self.building_mask.shape != expected_shape:
            raise ShapeMismatchValidationError(
                f"building_mask shape {self.building_mask.shape} != expected {expected_shape}. "
                f"Ensure building_mask matches grid_shape (Ny, Nx)."
            )
        if np.any(np.isnan(self.height_map)) or np.any(np.isinf(self.height_map)):
            nan_count = int(np.count_nonzero(np.isnan(self.height_map)))
            inf_count = int(np.count_nonzero(np.isinf(self.height_map)))
            raise NonFiniteValueError(
                f"height_map contains NaN or Inf values ({nan_count} NaNs, {inf_count} Infs). "
                f"Building heights must be finite real numbers."
            )
        if np.any(self.height_map < 0.0):
            min_val = float(np.min(self.height_map))
            raise PhysicalConstraintValidationError(
                f"height_map contains negative values; building heights must be >= 0 (minimum found: {min_val:.3f} m)."
            )
        # Mask consistency: where height > 0, mask should be True
        positive_height = self.height_map > 1e-4
        if np.any(positive_height & (~self.building_mask)):
            mismatch_count = int(np.count_nonzero(positive_height & (~self.building_mask)))
            raise PhysicalConstraintValidationError(
                f"Inconsistency: building_mask is False where height_map > 0 "
                f"({mismatch_count} cell(s) have positive height but building_mask=False)."
            )


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
            raise PhysicalConstraintValidationError(
                f"inflow_speed_mps must be positive: {self.inflow_speed_mps}. Inflow speed must be > 0.0 m/s."
            )
        if not (0.0 <= self.wind_direction_deg < 360.0):
            raise PhysicalConstraintValidationError(
                f"wind_direction_deg must be in [0, 360): {self.wind_direction_deg}. Angle must be between 0 and 360 degrees."
            )
        if self.reference_height_m <= 0.0:
            raise PhysicalConstraintValidationError(
                f"reference_height_m must be positive: {self.reference_height_m}. Reference height must be > 0.0 m."
            )
        if self.roughness_length_z0_m <= 0.0:
            raise PhysicalConstraintValidationError(
                f"roughness_length_z0_m must be positive: {self.roughness_length_z0_m}. Surface roughness z0 must be > 0.0 m."
            )
        if self.ambient_temperature_k is not None and self.ambient_temperature_k <= 0.0:
            raise PhysicalConstraintValidationError(
                f"ambient_temperature_k must be positive: {self.ambient_temperature_k}. Ambient temperature must be > 0.0 Kelvin."
            )


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
            raise MissingFieldValidationError("solver_name must not be empty. Specify CFD code (e.g. 'uDALES', 'OpenFOAM').")
        if not self.turbulence_model:
            raise MissingFieldValidationError("turbulence_model must not be empty. Specify turbulence model (e.g. 'RANS-k-omega-SST', 'LES-Deardorff').")


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
            raise TypeError(f"pedestrian_wind_speed must be a numpy ndarray, got {type(self.pedestrian_wind_speed).__name__}")
        if self.pedestrian_wind_speed.shape != expected_shape:
            raise ShapeMismatchValidationError(
                f"pedestrian_wind_speed shape {self.pedestrian_wind_speed.shape} != expected {expected_shape}"
            )
        if np.any(np.isnan(self.pedestrian_wind_speed)) or np.any(np.isinf(self.pedestrian_wind_speed)):
            nan_count = int(np.count_nonzero(np.isnan(self.pedestrian_wind_speed)))
            inf_count = int(np.count_nonzero(np.isinf(self.pedestrian_wind_speed)))
            raise NonFiniteValueError(
                f"pedestrian_wind_speed contains NaN or Inf values ({nan_count} NaNs, {inf_count} Infs)."
            )
        if np.any(self.pedestrian_wind_speed < -1e-4):
            min_val = float(np.min(self.pedestrian_wind_speed))
            raise PhysicalConstraintValidationError(
                f"pedestrian_wind_speed (scalar magnitude) cannot be negative (min found: {min_val:.3f} m/s)"
            )

        for name, field_arr in [
            ("velocity_u", self.velocity_u),
            ("velocity_v", self.velocity_v),
            ("velocity_w", self.velocity_w),
            ("turbulent_kinetic_energy", self.turbulent_kinetic_energy),
        ]:
            if field_arr is not None:
                if not isinstance(field_arr, np.ndarray):
                    raise TypeError(f"{name} must be a numpy ndarray, got {type(field_arr).__name__}")
                if field_arr.shape != expected_shape:
                    raise ShapeMismatchValidationError(f"{name} shape {field_arr.shape} != expected {expected_shape}")
                if np.any(np.isnan(field_arr)) or np.any(np.isinf(field_arr)):
                    raise NonFiniteValueError(f"{name} contains NaN or Inf values")

        if self.area_averaged_wind_speed is not None:
            if not isinstance(self.area_averaged_wind_speed, (float, int, np.floating)):
                raise TypeError(f"area_averaged_wind_speed must be a float, got {type(self.area_averaged_wind_speed).__name__}")
            if self.area_averaged_wind_speed < 0.0:
                raise PhysicalConstraintValidationError(
                    f"area_averaged_wind_speed cannot be negative: {self.area_averaged_wind_speed}"
                )


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
