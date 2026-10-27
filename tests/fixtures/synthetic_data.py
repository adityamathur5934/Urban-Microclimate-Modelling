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


from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticMetricBenchmark:
    """SYNTHETIC TEST FIXTURE: Known analytical answers for metric validation.

    DISCLAIMER:
    This is a purely synthetic mathematical fixture used to unit-test numerical
    metric formulas, spatial field alignments, and mask logic against exact hand calculations.
    IT IS NOT EVIDENCE OF SURROGATE MODEL ACCURACY OR REAL CFD MEASUREMENTS.
    """
    y_true: np.ndarray
    y_pred: np.ndarray
    building_mask: np.ndarray
    fluid_mask: np.ndarray
    u_ref: float
    expected_mae: float
    expected_rmse: float
    expected_max_error: float
    expected_nrmse_range_ratio: float
    expected_nrmse_range_percent: float
    expected_nrmse_inflow_ratio: float
    expected_nrmse_inflow_percent: float
    expected_relative_l2: float
    expected_r2: float


def create_synthetic_metric_benchmark_fixture() -> SyntheticMetricBenchmark:
    """Returns a deterministic 4x4 synthetic test fixture with exact hand-calculated metric answers.

    THIS FIXTURE IS STRICTLY FOR SOFTWARE UNIT TESTS AND MUST NEVER BE USED AS AN ACCURACY CLAIM.
    """
    # 4x4 grid: top 3 rows (12 cells) are outdoor fluid, bottom row (4 cells) is solid building
    building_mask = np.zeros((4, 4), dtype=bool)
    building_mask[3, :] = True
    fluid_mask = ~building_mask

    # Ground truth values in fluid cells: 2.0 to 13.0 m/s
    y_true = np.zeros((4, 4), dtype=np.float64)
    y_true[0, :] = [2.0, 3.0, 4.0, 5.0]
    y_true[1, :] = [6.0, 7.0, 8.0, 9.0]
    y_true[2, :] = [10.0, 11.0, 12.0, 13.0]
    y_true[3, :] = [0.0, 0.0, 0.0, 0.0]  # Solid interior

    # Predictions: exactly +1.5 error on fluid cells, +999.0 on solid cells (must be masked out)
    y_pred = np.zeros((4, 4), dtype=np.float64)
    y_pred[0:3, :] = y_true[0:3, :] + 1.5
    y_pred[3, :] = [999.0, 999.0, 999.0, 999.0]  # Corrupt interior to verify masking

    u_ref = 10.0  # Inflow speed in m/s

    # Exact hand-calculated answers:
    # Error on all 12 fluid cells = +1.5
    expected_mae = 1.5
    expected_rmse = 1.5
    expected_max_error = 1.5
    expected_nrmse_range_ratio = 1.5 / 11.0  # 3 / 22 ≈ 0.13636363636
    expected_nrmse_range_percent = (1.5 / 11.0) * 100.0
    expected_nrmse_inflow_ratio = 1.5 / 10.0  # 0.15
    expected_nrmse_inflow_percent = 15.0
    expected_relative_l2 = float(np.sqrt(27.0 / 818.0))
    expected_r2 = 116.0 / 143.0  # ≈ 0.811188811

    return SyntheticMetricBenchmark(
        y_true=y_true,
        y_pred=y_pred,
        building_mask=building_mask,
        fluid_mask=fluid_mask,
        u_ref=u_ref,
        expected_mae=expected_mae,
        expected_rmse=expected_rmse,
        expected_max_error=expected_max_error,
        expected_nrmse_range_ratio=expected_nrmse_range_ratio,
        expected_nrmse_range_percent=expected_nrmse_range_percent,
        expected_nrmse_inflow_ratio=expected_nrmse_inflow_ratio,
        expected_nrmse_inflow_percent=expected_nrmse_inflow_percent,
        expected_relative_l2=expected_relative_l2,
        expected_r2=expected_r2,
    )


def create_synthetic_split_dataset_fixture() -> Dict[str, Any]:
    """Generates a synthetic layout dataset with known layout-family and layout-group tags.

    THIS FIXTURE IS STRICTLY FOR TESTING DATA SPLIT ISOLATION AND PREVENTING LEAKAGE.
    """
    layout_ids = [f"synth_layout_{i:03d}" for i in range(18)]
    families = [
        "idealized_aligned", "idealized_aligned", "idealized_aligned",
        "idealized_aligned", "idealized_aligned", "idealized_aligned",
        "idealized_staggered", "idealized_staggered", "idealized_staggered",
        "idealized_staggered", "idealized_staggered", "idealized_staggered",
        "realistic_neighborhood", "realistic_neighborhood", "realistic_neighborhood",
        "realistic_neighborhood", "realistic_neighborhood", "realistic_neighborhood",
    ]
    # 6 groups of 3 related layouts each (e.g. multi-angle runs of same base layout)
    groups = [f"group_{i // 3:02d}" for i in range(18)]

    layout_to_family = {lid: fam for lid, fam in zip(layout_ids, families)}
    layout_to_group = {lid: grp for lid, grp in zip(layout_ids, groups)}

    return {
        "layout_ids": layout_ids,
        "layout_to_family": layout_to_family,
        "layout_to_group": layout_to_group,
    }
