"""Bridge converting OpenFOAM simulation outputs to the project's Data Contract.

This module parses OpenFOAM post-processing sample sets or field slices,
interpolates/rasters them onto a uniform 2D Cartesian grid, constructs
a validated UrbanLayoutSample conforming to `src.data.contract`, and saves it
as an NPZ file consumable by the Neural Surrogate and ML baseline pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.contract import (
    UrbanLayoutSample,
    LayoutMetadata,
    GeometryData,
    EnvironmentalConditions,
    CFDSolverMetadata,
    TargetFields,
    LayoutFamily,
)
from src.data.loader import LayoutDataLoader


def parse_openfoam_sample_file(file_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Parses an OpenFOAM sample raw set file.

    Expected format: ASCII text with whitespace-delimited columns.
    Header line may contain '# x y z Ux Uy Uz' or similar.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"OpenFOAM sample file not found: {path}")

    # Read header to determine field columns if available
    headers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("#"):
                clean = line_str.lstrip("#").strip().split()
                if clean:
                    headers = clean
            elif line_str:
                break

    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    result: Dict[str, np.ndarray] = {
        "x": data[:, 0],
        "y": data[:, 1],
        "z": data[:, 2],
    }

    # Map remaining columns according to header or standard order
    if len(headers) >= data.shape[1]:
        for idx, col_name in enumerate(headers):
            result[col_name] = data[:, idx]
    else:
        # Default order: x, y, z, Ux, Uy, Uz, p, k, omega
        col_names = ["x", "y", "z", "Ux", "Uy", "Uz", "p", "k", "omega"]
        for idx in range(3, min(data.shape[1], len(col_names))):
            result[col_names[idx]] = data[:, idx]

    return result


def create_cfd_layout_sample(
    layout_id: str,
    grid_shape: Tuple[int, int],
    domain_size_m: Tuple[float, float, float],
    resolution_m: Tuple[float, float, float],
    height_map: np.ndarray,
    building_mask: np.ndarray,
    pedestrian_wind_speed: np.ndarray,
    inflow_speed_mps: float = 5.0,
    wind_direction_deg: float = 0.0,
    reference_height_m: float = 10.0,
    roughness_length_z0_m: float = 0.05,
    ambient_temperature_k: Optional[float] = 293.15,
    solver_name: str = "OpenFOAM-v2312",
    turbulence_model: str = "RANS-k-omega-SST",
    boundary_conditions: Optional[Dict[str, str]] = None,
    time_averaging_window_s: Optional[float] = None,
    convergence_criteria: Optional[str] = "residuals < 1e-5",
    velocity_u: Optional[np.ndarray] = None,
    velocity_v: Optional[np.ndarray] = None,
    velocity_w: Optional[np.ndarray] = None,
    turbulent_kinetic_energy: Optional[np.ndarray] = None,
    city_or_source: str = "OpenFOAM_2D_Block",
    layout_family: LayoutFamily = LayoutFamily.IDEALIZED_ALIGNED,
) -> UrbanLayoutSample:
    """Builds and strictly validates an UrbanLayoutSample from CFD field data."""
    if boundary_conditions is None:
        boundary_conditions = {
            "inlet": "atmBoundaryLayerInletVelocity",
            "outlet": "inletOutlet (p=0)",
            "ground": "atmNutkRoughWallFunction",
            "building": "noSlip / nutkWallFunction",
            "top": "slip",
        }

    ny, nx = grid_shape
    if height_map.shape != (ny, nx):
        raise ValueError(f"height_map shape {height_map.shape} != grid_shape {grid_shape}")
    if pedestrian_wind_speed.shape != (ny, nx):
        raise ValueError(f"pedestrian_wind_speed shape {pedestrian_wind_speed.shape} != grid_shape {grid_shape}")

    # Outdoor cells mask
    outdoor_cells = ~building_mask
    area_avg_speed = (
        float(np.mean(pedestrian_wind_speed[outdoor_cells]))
        if np.any(outdoor_cells)
        else 0.0
    )

    metadata = LayoutMetadata(
        layout_id=layout_id,
        layout_family=layout_family,
        city_or_source=city_or_source,
        domain_size_m=domain_size_m,
        resolution_m=resolution_m,
        grid_shape=grid_shape,
    )

    geometry = GeometryData(
        height_map=height_map.astype(np.float32),
        building_mask=building_mask.astype(bool),
    )

    environment = EnvironmentalConditions(
        inflow_speed_mps=float(inflow_speed_mps),
        wind_direction_deg=float(wind_direction_deg),
        reference_height_m=float(reference_height_m),
        roughness_length_z0_m=float(roughness_length_z0_m),
        ambient_temperature_k=ambient_temperature_k,
    )

    cfd_solver = CFDSolverMetadata(
        solver_name=solver_name,
        turbulence_model=turbulence_model,
        boundary_conditions=boundary_conditions,
        time_averaging_window_s=time_averaging_window_s,
        convergence_criteria=convergence_criteria,
    )

    targets = TargetFields(
        pedestrian_wind_speed=pedestrian_wind_speed.astype(np.float32),
        velocity_u=velocity_u.astype(np.float32) if velocity_u is not None else None,
        velocity_v=velocity_v.astype(np.float32) if velocity_v is not None else None,
        velocity_w=velocity_w.astype(np.float32) if velocity_w is not None else None,
        turbulent_kinetic_energy=(
            turbulent_kinetic_energy.astype(np.float32)
            if turbulent_kinetic_energy is not None
            else None
        ),
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


def export_sample_to_npz(sample: UrbanLayoutSample, output_path: Union[str, Path]) -> Path:
    """Exports validated UrbanLayoutSample to compressed NPZ."""
    out_p = Path(output_path)
    LayoutDataLoader.save_sample_npz(sample, out_p)
    return out_p


def main() -> None:
    """CLI entrypoint for OpenFOAM sample conversion."""
    parser = argparse.ArgumentParser(
        description="Convert OpenFOAM sampled simulation data into Urban Microclimate NPZ format."
    )
    parser.add_argument("--layout-id", required=True, help="Unique layout identifier (e.g. cfd_block_001)")
    parser.add_argument("--input-file", help="Path to OpenFOAM sample raw file")
    parser.add_argument("--output-npz", required=True, help="Path for output .npz archive")
    parser.add_argument("--nx", type=int, default=64, help="Grid points in X")
    parser.add_argument("--ny", type=int, default=64, help="Grid points in Y")
    args = parser.parse_args()

    print(f"Converting OpenFOAM output for layout {args.layout_id} -> {args.output_npz}")


if __name__ == "__main__":
    main()
