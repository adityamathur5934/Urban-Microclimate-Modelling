"""Dataset loader and serializer supporting validated multi-format disk storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
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


class LayoutDataLoader:
    """Loads, validates, and serializes UrbanLayoutSamples."""

    @staticmethod
    def save_sample_npz(sample: UrbanLayoutSample, output_path: Union[str, Path]) -> None:
        """Serializes an UrbanLayoutSample into a compact, validated NPZ file."""
        sample.validate()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays_to_save: Dict[str, Any] = {
            "height_map": sample.geometry.height_map.astype(np.float32),
            "building_mask": sample.geometry.building_mask.astype(bool),
            "pedestrian_wind_speed": sample.targets.pedestrian_wind_speed.astype(np.float32),
        }

        if sample.targets.velocity_u is not None:
            arrays_to_save["velocity_u"] = sample.targets.velocity_u.astype(np.float32)
        if sample.targets.velocity_v is not None:
            arrays_to_save["velocity_v"] = sample.targets.velocity_v.astype(np.float32)
        if sample.targets.velocity_w is not None:
            arrays_to_save["velocity_w"] = sample.targets.velocity_w.astype(np.float32)
        if sample.targets.turbulent_kinetic_energy is not None:
            arrays_to_save["turbulent_kinetic_energy"] = sample.targets.turbulent_kinetic_energy.astype(np.float32)

        meta_dict = {
            "metadata": {
                "layout_id": sample.metadata.layout_id,
                "layout_family": sample.metadata.layout_family.value,
                "city_or_source": sample.metadata.city_or_source,
                "domain_size_m": list(sample.metadata.domain_size_m),
                "resolution_m": list(sample.metadata.resolution_m),
                "grid_shape": list(sample.metadata.grid_shape),
            },
            "environment": {
                "inflow_speed_mps": sample.environment.inflow_speed_mps,
                "wind_direction_deg": sample.environment.wind_direction_deg,
                "reference_height_m": sample.environment.reference_height_m,
                "roughness_length_z0_m": sample.environment.roughness_length_z0_m,
                "ambient_temperature_k": sample.environment.ambient_temperature_k,
            },
            "cfd_solver": {
                "solver_name": sample.cfd_solver.solver_name,
                "turbulence_model": sample.cfd_solver.turbulence_model,
                "boundary_conditions": sample.cfd_solver.boundary_conditions,
                "time_averaging_window_s": sample.cfd_solver.time_averaging_window_s,
                "convergence_criteria": sample.cfd_solver.convergence_criteria,
            },
            "target_scalars": {
                "area_averaged_wind_speed": sample.targets.area_averaged_wind_speed,
            },
        }

        # Store JSON metadata as a string array inside the NPZ
        arrays_to_save["_metadata_json"] = np.array([json.dumps(meta_dict)])
        np.savez_compressed(path, **arrays_to_save)

    @staticmethod
    def load_sample_npz(file_path: Union[str, Path]) -> UrbanLayoutSample:
        """Loads and strictly validates an UrbanLayoutSample from an NPZ file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with np.load(path) as data:
            if "_metadata_json" not in data:
                raise KeyError(f"Corrupt or missing metadata in {path}")

            meta_dict = json.loads(str(data["_metadata_json"][0]))
            m = meta_dict["metadata"]
            e = meta_dict["environment"]
            c = meta_dict["cfd_solver"]
            t_scalars = meta_dict.get("target_scalars", {})

            metadata = LayoutMetadata(
                layout_id=m["layout_id"],
                layout_family=LayoutFamily(m["layout_family"]),
                city_or_source=m["city_or_source"],
                domain_size_m=tuple(m["domain_size_m"]),
                resolution_m=tuple(m["resolution_m"]),
                grid_shape=tuple(m["grid_shape"]),
            )

            environment = EnvironmentalConditions(
                inflow_speed_mps=float(e["inflow_speed_mps"]),
                wind_direction_deg=float(e["wind_direction_deg"]),
                reference_height_m=float(e["reference_height_m"]),
                roughness_length_z0_m=float(e["roughness_length_z0_m"]),
                ambient_temperature_k=e.get("ambient_temperature_k"),
            )

            cfd_solver = CFDSolverMetadata(
                solver_name=c["solver_name"],
                turbulence_model=c["turbulence_model"],
                boundary_conditions=c.get("boundary_conditions", {}),
                time_averaging_window_s=c.get("time_averaging_window_s"),
                convergence_criteria=c.get("convergence_criteria"),
            )

            if "height_map" not in data or "building_mask" not in data:
                raise KeyError(f"Missing geometry arrays in {path}")
            if "pedestrian_wind_speed" not in data:
                raise KeyError(f"Missing pedestrian_wind_speed target in {path}")

            geometry = GeometryData(
                height_map=data["height_map"],
                building_mask=data["building_mask"].astype(bool),
            )

            targets = TargetFields(
                pedestrian_wind_speed=data["pedestrian_wind_speed"],
                velocity_u=data["velocity_u"] if "velocity_u" in data else None,
                velocity_v=data["velocity_v"] if "velocity_v" in data else None,
                velocity_w=data["velocity_w"] if "velocity_w" in data else None,
                turbulent_kinetic_energy=(
                    data["turbulent_kinetic_energy"]
                    if "turbulent_kinetic_energy" in data
                    else None
                ),
                area_averaged_wind_speed=t_scalars.get("area_averaged_wind_speed"),
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

    @classmethod
    def load_directory(
        cls,
        directory_path: Union[str, Path],
        layout_ids: Optional[List[str]] = None,
    ) -> List[UrbanLayoutSample]:
        """Loads all NPZ samples from a directory, optionally filtered by layout_ids."""
        p = Path(directory_path)
        if not p.is_dir():
            raise NotADirectoryError(f"Directory not found: {p}")

        id_filter = set(layout_ids) if layout_ids is not None else None
        samples: List[UrbanLayoutSample] = []

        for f in sorted(p.glob("*.npz")):
            sample = cls.load_sample_npz(f)
            if id_filter is None or sample.layout_id in id_filter:
                samples.append(sample)

        return samples
