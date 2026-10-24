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
from scipy.interpolate import griddata

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


def interpolate_samples_to_cartesian_grid(
    sample_data: Dict[str, np.ndarray],
    grid_shape: Tuple[int, int],
    domain_size_m: Tuple[float, float, float],
    building_bounds: Optional[Tuple[float, float, float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Interpolates scattered OpenFOAM sample points onto a regular 2D Cartesian grid.

    Args:
        sample_data: Dictionary containing coordinate arrays ('x', 'y' or 'z') and field arrays.
        grid_shape: (ny, nx) output matrix shape.
        domain_size_m: (lx, ly, lz) in meters.
        building_bounds: Optional tuple (x_min, x_max, y_min, y_max, height) in meters.

    Returns:
        height_map: (ny, nx) float32 building height array.
        building_mask: (ny, nx) boolean mask (True where solid building).
        interpolated_fields: Dictionary of (ny, nx) float32 arrays.
    """
    ny, nx = grid_shape
    lx, ly, lz = domain_size_m

    # Coordinate mapping: in 2D OpenFOAM simulations, coordinates are typically (x, z) or (x, y)
    pts_x = sample_data["x"]
    if "y" in sample_data and np.ptp(sample_data["y"]) > 1e-4:
        pts_y = sample_data["y"]
    elif "z" in sample_data and np.ptp(sample_data["z"]) > 1e-4:
        pts_y = sample_data["z"]
    else:
        pts_y = sample_data.get("y", np.zeros_like(pts_x))

    points = np.column_stack([pts_x, pts_y])

    # Construct target uniform Cartesian lattice
    grid_x = np.linspace(0, lx, nx)
    grid_y = np.linspace(0, ly, ny)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)

    # Building mask and height map
    height_map = np.zeros((ny, nx), dtype=np.float32)
    building_mask = np.zeros((ny, nx), dtype=bool)

    if building_bounds is not None:
        bx0, bx1, by0, by1, b_height = building_bounds
        in_b = (mesh_x >= bx0) & (mesh_x <= bx1) & (mesh_y >= by0) & (mesh_y <= by1)
        building_mask[in_b] = True
        height_map[in_b] = float(b_height)

    interp_fields: Dict[str, np.ndarray] = {}

    # Identify fields to interpolate
    field_keys = [k for k in sample_data.keys() if k not in ("x", "y", "z")]
    for key in field_keys:
        vals = sample_data[key]
        if len(points) >= 4:
            grid_val = griddata(points, vals, (mesh_x, mesh_y), method="linear")
            # Fill NaNs with nearest neighbor
            nan_mask = np.isnan(grid_val)
            if np.any(nan_mask):
                nearest_val = griddata(points, vals, (mesh_x, mesh_y), method="nearest")
                grid_val[nan_mask] = nearest_val[nan_mask]
        else:
            grid_val = np.full((ny, nx), float(np.mean(vals)), dtype=np.float32)

        # Inside solid building, clamp velocities to 0.0
        if "U" in key or key in ("velocity_u", "velocity_v", "velocity_w"):
            grid_val[building_mask] = 0.0

        interp_fields[key] = grid_val.astype(np.float32)

    return height_map, building_mask, interp_fields


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
    parser.add_argument("--layout-id", default="cfd_2d_block_001", help="Unique layout identifier")
    parser.add_argument("--input-file", help="Path to OpenFOAM sample raw file")
    parser.add_argument("--output-npz", required=True, help="Path for output .npz archive")
    parser.add_argument("--nx", type=int, default=128, help="Grid points in X")
    parser.add_argument("--ny", type=int, default=64, help="Grid points in Y")
    parser.add_argument("--lx", type=float, default=420.0, help="Domain length X [m]")
    parser.add_argument("--ly", type=float, default=120.0, help="Domain width/height Y [m]")
    parser.add_argument("--generate-demo", action="store_true", help="Generate demonstration sample if raw file is not yet available")
    args = parser.parse_args()

    out_p = Path(args.output_npz)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if args.input_file and Path(args.input_file).is_file():
        print(f"Parsing OpenFOAM sample file: {args.input_file}")
        raw_data = parse_openfoam_sample_file(args.input_file)
        building_bounds = (100.0, 120.0, 0.0, 20.0, 20.0)
        h_map, b_mask, fields = interpolate_samples_to_cartesian_grid(
            sample_data=raw_data,
            grid_shape=(args.ny, args.nx),
            domain_size_m=(args.lx, args.ly, 60.0),
            building_bounds=building_bounds,
        )
        u_mag = fields.get("pedestrian_wind_speed", None)
        if u_mag is None:
            ux = fields.get("Ux", np.zeros((args.ny, args.nx), dtype=np.float32))
            uy = fields.get("Uy", np.zeros((args.ny, args.nx), dtype=np.float32))
            u_mag = np.sqrt(ux**2 + uy**2).astype(np.float32)
        sample = create_cfd_layout_sample(
            layout_id=args.layout_id,
            grid_shape=(args.ny, args.nx),
            domain_size_m=(args.lx, args.ly, 60.0),
            resolution_m=(args.lx / args.nx, args.ly / args.ny, 1.0),
            height_map=h_map,
            building_mask=b_mask,
            pedestrian_wind_speed=u_mag,
            velocity_u=fields.get("Ux"),
            velocity_v=fields.get("Uy"),
            turbulent_kinetic_energy=fields.get("k"),
        )
    elif args.generate_demo:
        print(f"Generating demonstration reference sample: {args.layout_id}")
        h_map = np.zeros((args.ny, args.nx), dtype=np.float32)
        b_mask = np.zeros((args.ny, args.nx), dtype=bool)
        # 2D block obstacle: 100m to 120m in x, 0 to 20m in y
        bx0 = int((100.0 / args.lx) * args.nx)
        bx1 = int((120.0 / args.lx) * args.nx)
        by0 = 0
        by1 = int((20.0 / args.ly) * args.ny)
        b_mask[by0:by1, bx0:bx1] = True
        h_map[by0:by1, bx0:bx1] = 20.0

        # Flow field: baseline 5.0 m/s with leeward wake cavity and corner acceleration
        u_mag = np.full((args.ny, args.nx), 5.0, dtype=np.float32)
        # Recirculation wake behind building (x from 120m to 250m)
        wake_x1 = int((250.0 / args.lx) * args.nx)
        u_mag[by0:by1, bx1:wake_x1] = 1.2
        # Rooftop acceleration
        u_mag[by1 : by1 + 5, bx0:bx1] = 6.8
        u_mag[b_mask] = 0.0

        sample = create_cfd_layout_sample(
            layout_id=args.layout_id,
            grid_shape=(args.ny, args.nx),
            domain_size_m=(args.lx, args.ly, 60.0),
            resolution_m=(args.lx / args.nx, args.ly / args.ny, 1.0),
            height_map=h_map,
            building_mask=b_mask,
            pedestrian_wind_speed=u_mag,
            city_or_source="OpenFOAM_2D_Block_Demo",
        )
    else:
        raise ValueError(
            "Either --input-file pointing to an existing file or --generate-demo must be specified."
        )

    export_sample_to_npz(sample, out_p)
    print(f"Successfully exported validated NPZ reference to: {out_p}")


if __name__ == "__main__":
    main()
