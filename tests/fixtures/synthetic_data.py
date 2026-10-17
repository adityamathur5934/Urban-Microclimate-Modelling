"""Synthetic data generator strictly for software unit tests and pipeline validation.

DISCLAIMER:
All data generated in this module are synthetic mathematical fixtures designed
exclusively for software testing (shape validation, normalization, and data loading).
THEY DO NOT REPRESENT REAL CFD OBSERVATIONS OR VALIDATED PHYSICAL FLOW MEASUREMENTS.
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np

from src.data.contract import (
    UrbanLayoutSample,
    LayoutMetadata,
    GeometryData,
    EnvironmentalConditions,
    CFDSolverMetadata,
    TargetFields,
    LayoutFamily,
)


def create_synthetic_sample(
    layout_id: str = "synthetic_toy_001",
    layout_family: LayoutFamily = LayoutFamily.IDEALIZED_ALIGNED,
    grid_shape: Tuple[int, int] = (32, 32),
    resolution_m: Tuple[float, float, float] = (1.0, 1.0, 0.5),
    inflow_speed: float = 5.0,
    wind_direction_deg: float = 0.0,
    building_height: float = 20.0,
) -> UrbanLayoutSample:
    """Creates a deterministic synthetic test fixture.

    THIS IS STRICTLY A TEST FIXTURE FOR PIPELINE VERIFICATION, NOT A CFD RESULT.
    """
    ny, nx = grid_shape
    dx, dy, dz = resolution_m
    lx = nx * dx
    ly = ny * dy
    lz = 100.0

    height_map = np.zeros((ny, nx), dtype=np.float32)
    building_mask = np.zeros((ny, nx), dtype=bool)

    if layout_family == LayoutFamily.IDEALIZED_ALIGNED:
        # Place a central rectangular building
        cy, cx = ny // 2, nx // 2
        hy, hx = ny // 8, nx // 8
        height_map[cy - hy : cy + hy, cx - hx : cx + hx] = building_height
        building_mask[cy - hy : cy + hy, cx - hx : cx + hx] = True

    elif layout_family == LayoutFamily.IDEALIZED_STAGGERED:
        # Two staggered blocks
        height_map[4:10, 4:10] = building_height
        building_mask[4:10, 4:10] = True
        height_map[18:24, 18:24] = building_height * 0.75
        building_mask[18:24, 18:24] = True

    elif layout_family == LayoutFamily.REALISTIC_NEIGHBORHOOD:
        # Multiple varied building blocks
        height_map[2:8, 2:8] = 12.0
        building_mask[2:8, 2:8] = True
        height_map[12:20, 4:10] = 28.0
        building_mask[12:20, 4:10] = True
        height_map[6:14, 18:26] = 18.0
        building_mask[6:14, 18:26] = True

    # Synthetic surrogate target field:
    # A simple analytical flow pattern around obstacles (potential flow-like / wake cavity)
    # Outside building: inflow_speed with a downwind wake reduction; inside: 0.0
    ped_wind = np.full((ny, nx), inflow_speed * 0.8, dtype=np.float32)
    # Add simple gradient
    ped_wind += np.linspace(-0.5, 0.5, nx, dtype=np.float32)[None, :]
    # Set inside building to zero
    ped_wind[building_mask] = 0.0

    u_field = ped_wind * np.cos(np.deg2rad(wind_direction_deg))
    v_field = ped_wind * np.sin(np.deg2rad(wind_direction_deg))
    w_field = np.zeros((ny, nx), dtype=np.float32)
    tke_field = np.where(building_mask, 0.0, 0.5 * (inflow_speed * 0.1) ** 2).astype(np.float32)

    outdoor_cells = ~building_mask
    area_avg_speed = float(np.mean(ped_wind[outdoor_cells])) if np.any(outdoor_cells) else 0.0

    metadata = LayoutMetadata(
        layout_id=layout_id,
        layout_family=layout_family,
        city_or_source="SYNTHETIC_FIXTURE_DO_NOT_USE_AS_CFD",
        domain_size_m=(lx, ly, lz),
        resolution_m=resolution_m,
        grid_shape=grid_shape,
    )

    geometry = GeometryData(
        height_map=height_map,
        building_mask=building_mask,
    )

    environment = EnvironmentalConditions(
        inflow_speed_mps=inflow_speed,
        wind_direction_deg=wind_direction_deg,
        reference_height_m=10.0,
        roughness_length_z0_m=0.05,
        ambient_temperature_k=293.15,
    )

    cfd_solver = CFDSolverMetadata(
        solver_name="SYNTHETIC_SOLVER_MOCK",
        turbulence_model="SYNTHETIC_LES_MOCK",
        boundary_conditions={"inflow": "fixed_velocity", "outflow": "zero_gradient"},
        time_averaging_window_s=3600.0,
        convergence_criteria="synthetic_tol_1e-4",
    )

    targets = TargetFields(
        pedestrian_wind_speed=ped_wind,
        velocity_u=u_field,
        velocity_v=v_field,
        velocity_w=w_field,
        turbulent_kinetic_energy=tke_field,
        area_averaged_wind_speed=area_avg_speed,
    )

    sample = UrbanLayoutSample(
        metadata=metadata,
        geometry=geometry,
        environment=environment,
        cfd_solver=cfd_solver,
        targets=targets,
    )
    sample.validate()
    return sample
