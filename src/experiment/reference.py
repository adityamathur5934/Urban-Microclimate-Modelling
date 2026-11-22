"""Analytical reference dataset for the CFD-to-surrogate experiment.

IMPORTANT — provenance and scientific honesty
---------------------------------------------
The pedestrian-level wind fields produced here are computed by a documented
*analytical* aerodynamic model (an adaptation of the Castro & Robins 1977 /
Martinuzzi & Tropea 1993 surface-mounted-obstacle formulation already used in
``cfd/scripts/generate_canonical_reference.py`` and the in-memory corpus). They
model the same physics as that canonical benchmark: upstream atmospheric
boundary layer, windward stagnation, lateral corner (Venturi) acceleration,
leeward recirculating wake, and street-canyon channeling.

These fields are **NOT** CFD solver output. ``REFERENCE_PROVENANCE`` flags every
sample as ``ANALYTICAL_REFERENCE`` and the ``CFDSolverMetadata.solver_name``
carries the ``Analytical_Aerodynamic_Reference`` label. The models trained on
this dataset are therefore "analytical-reference surrogates", a methodological
demonstration of the full CFD-to-surrogate pipeline. Substituting genuine CFD
output (via ``cfd/scripts/foam_to_contract.py``) is the only step required to
turn this into a real CFD-grounded experiment; that step is blocked here only by
the absence of an OpenFOAM solver in the execution environment.

The generator is fully deterministic for a fixed seed and produces all layouts
at a single fixed grid resolution so the neural models can batch them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.contract import (
    CFDSolverMetadata,
    EnvironmentalConditions,
    GeometryData,
    LayoutFamily,
    LayoutMetadata,
    TargetFields,
    UrbanLayoutSample,
)

REFERENCE_PROVENANCE = "ANALYTICAL_REFERENCE"
REFERENCE_SOLVER_NAME = "Analytical_Aerodynamic_Reference (Castro & Robins 1977)"
REFERENCE_TURBULENCE_MODEL = "Analytical_Wake_Shear_Formulation"
REFERENCE_CITY_SOURCE = "ANALYTICAL_REFERENCE"


# ---------------------------------------------------------------------------
# Analytical pedestrian-level wind field (adapted Castro & Robins formulation)
# ---------------------------------------------------------------------------

def _loglaw_pedestrian_speed(
    inflow_speed_mps: float,
    reference_height_m: float,
    roughness_length_z0_m: float,
    pedestrian_height_m: float = 1.75,
    kappa: float = 0.41,
) -> float:
    """Undisturbed pedestrian-height speed from a neutral ABL log law.

    U(z) = (u*/kappa) * ln((z + z0)/z0), with u* chosen so U(z_ref) = U_ref.
    """
    z0 = max(float(roughness_length_z0_m), 1e-4)
    u_star = (kappa * float(inflow_speed_mps)) / math.log(
        (float(reference_height_m) + z0) / z0
    )
    return float((u_star / kappa) * math.log((pedestrian_height_m + z0) / z0))


def analytical_pedestrian_wind_field(
    X: np.ndarray,
    Y: np.ndarray,
    buildings: List[Dict[str, Tuple[float, float] | float]],
    inflow_speed_mps: float,
    wind_direction_deg: float,
    reference_height_m: float = 10.0,
    roughness_length_z0_m: float = 0.05,
    building_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute an analytical pedestrian-level (z=1.75 m) flow field.

    Returns (u_mag, u_x, v_y, tke) as (Ny, Nx) float32 arrays. Velocities are in
    m/s; TKE in m^2/s^2. Building interiors are zeroed when ``building_mask`` is
    provided. This is a superposition of per-building aerodynamic contributions
    (stagnation / corner jets / wake deficit) plus canyon channeling — the same
    modelling concept used by the project's canonical analytical benchmark.
    """
    u_ped = _loglaw_pedestrian_speed(
        inflow_speed_mps, reference_height_m, roughness_length_z0_m
    )
    theta = math.radians(float(wind_direction_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    u_mag = np.full_like(X, u_ped, dtype=np.float32)

    for b in buildings:
        x_min, x_max = b["x"]
        y_min, y_max = b["y"]
        h = float(b["h"])
        xc = 0.5 * (x_min + x_max)
        yc = 0.5 * (y_min + y_max)
        wx = max(x_max - x_min, 1e-3)
        wy = max(y_max - y_min, 1e-3)
        h_factor = float(np.clip(h / 20.0, 0.6, 2.2))

        rel_x = (X - xc) / wx
        rel_y = (Y - yc) / wy

        # Windward stagnation (upstream of the windward face along +x inflow)
        stag_x_dist = np.maximum(0.0, (x_min - X) / wx)
        stag_env = np.exp(-((rel_y / 0.8) ** 2))
        stagnation = np.exp(-stag_x_dist / 0.7) * stag_env * (X < x_min)
        u_mag -= 0.65 * u_ped * stagnation * h_factor

        # Lateral corner acceleration jets (Venturi speed-up)
        jet_top = np.exp(-((rel_x) ** 2 / 0.5 + (rel_y - 0.75) ** 2 / 0.25))
        jet_bot = np.exp(-((rel_x) ** 2 / 0.5 + (rel_y + 0.75) ** 2 / 0.25))
        corner_jets = (jet_top + jet_bot) * 0.45 * u_ped * h_factor
        u_mag += corner_jets

        # Leeward recirculating wake cavity (~4H downstream)
        wake_len = 4.0 * (h / 20.0)
        wake_x_dist = np.maximum(0.0, (X - x_max) / wx)
        wake_env = np.exp(-((rel_y / 0.9) ** 2))
        wake_deficit = np.exp(-wake_x_dist / wake_len) * wake_env * (X > x_max)
        u_mag -= 0.75 * u_ped * wake_deficit

    # Street-canyon channeling jet between building rows (broad, mild speed-up)
    if buildings:
        ys = [0.5 * (b["y"][0] + b["y"][1]) for b in buildings]
        canyon_y = float(np.mean(ys)) if ys else float(np.mean(Y))
        canyon_prox = np.exp(-((Y - canyon_y) ** 2) / (30.0 ** 2))
        u_mag += 0.25 * u_ped * canyon_prox

    # Physical bounds
    u_mag = np.clip(u_mag, 0.05 * u_ped, 1.55 * u_ped)

    # Vector decomposition (streamwise / spanwise); near-wake reverse flow
    u_x = u_mag * cos_t
    v_y = u_mag * sin_t
    for b in buildings:
        x_max = b["x"][1]
        yc = 0.5 * (b["y"][0] + b["y"][1])
        wy = b["y"][1] - b["y"][0]
        near_wake = (X > x_max) & (X < x_max + 1.2 * (b["x"][1] - b["x"][0])) & (np.abs(Y - yc) < 0.4 * wy)
        u_x[near_wake] = -0.15 * u_ped * cos_t

    # TKE: background + shear-layer peaks along corner jets and wake
    z0 = max(float(roughness_length_z0_m), 1e-4)
    u_star = (0.41 * float(inflow_speed_mps)) / math.log((reference_height_m + z0) / z0)
    tke_bg = float(u_star ** 2 / math.sqrt(0.09))
    grad_y, grad_x = np.gradient(u_mag)
    shear = np.sqrt(grad_x ** 2 + grad_y ** 2)
    tke = (0.5 * (0.1 * float(inflow_speed_mps)) ** 2 + 0.8 * shear + tke_bg).astype(np.float32)

    if building_mask is not None:
        u_mag[building_mask] = 0.0
        u_x[building_mask] = 0.0
        v_y[building_mask] = 0.0
        tke[building_mask] = 0.0

    return u_mag.astype(np.float32), u_x.astype(np.float32), v_y.astype(np.float32), tke


# ---------------------------------------------------------------------------
# Single layout construction
# ---------------------------------------------------------------------------

def build_analytical_reference_sample(
    layout_id: str,
    layout_family: LayoutFamily,
    grid_shape: Tuple[int, int],
    domain_size_m: Tuple[float, float, float],
    buildings: List[Dict[str, Tuple[float, float] | float]],
    inflow_speed_mps: float,
    wind_direction_deg: float,
    reference_height_m: float = 10.0,
    roughness_length_z0_m: float = 0.05,
    dz: float = 0.5,
) -> UrbanLayoutSample:
    """Build a validated UrbanLayoutSample with an analytical reference field.

    The sample is tagged as ``ANALYTICAL_REFERENCE`` (not CFD) in its
    ``CFDSolverMetadata`` and ``city_or_source``.
    """
    ny, nx = grid_shape
    lx, ly, _ = domain_size_m
    dx = lx / nx
    dy = ly / ny

    x = np.linspace(0.5 * dx, lx - 0.5 * dx, nx)
    y = np.linspace(0.5 * dy, ly - 0.5 * dy, ny)
    X, Y = np.meshgrid(x, y)

    height_map = np.zeros((ny, nx), dtype=np.float32)
    building_mask = np.zeros((ny, nx), dtype=bool)
    for b in buildings:
        x_min, x_max = b["x"]
        y_min, y_max = b["y"]
        h = float(b["h"])
        m = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
        building_mask |= m
        height_map[m] = h

    u_mag, u_x, v_y, tke = analytical_pedestrian_wind_field(
        X, Y, buildings, inflow_speed_mps, wind_direction_deg,
        reference_height_m, roughness_length_z0_m, building_mask=building_mask,
    )

    outdoor = ~building_mask
    area_avg = float(np.mean(u_mag[outdoor])) if np.any(outdoor) else 0.0

    return UrbanLayoutSample(
        metadata=LayoutMetadata(
            layout_id=layout_id,
            layout_family=layout_family,
            city_or_source=REFERENCE_CITY_SOURCE,
            domain_size_m=domain_size_m,
            resolution_m=(dx, dy, dz),
            grid_shape=grid_shape,
        ),
        geometry=GeometryData(height_map=height_map, building_mask=building_mask),
        environment=EnvironmentalConditions(
            inflow_speed_mps=float(inflow_speed_mps),
            wind_direction_deg=float(wind_direction_deg),
            reference_height_m=float(reference_height_m),
            roughness_length_z0_m=float(roughness_length_z0_m),
        ),
        cfd_solver=CFDSolverMetadata(
            solver_name=REFERENCE_SOLVER_NAME,
            turbulence_model=REFERENCE_TURBULENCE_MODEL,
            boundary_conditions={
                "inlet": f"analytical ABL (Uref={inflow_speed_mps} m/s, z0={roughness_length_z0_m} m)",
                "outlet": "analytical zeroGradient",
                "ground": "analytical roughWall",
                "building": "noSlip",
                "top": "slip",
            },
            time_averaging_window_s=None,
            convergence_criteria="analytical_exact",
        ),
        targets=TargetFields(
            pedestrian_wind_speed=u_mag,
            velocity_u=u_x,
            velocity_v=v_y,
            velocity_w=np.zeros((ny, nx), dtype=np.float32),
            turbulent_kinetic_energy=tke,
            area_averaged_wind_speed=area_avg,
        ),
    )


# ---------------------------------------------------------------------------
# Deterministic parameterized sweep (all layouts at one fixed grid size)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _LayoutSpec:
    layout_id: str
    family: LayoutFamily
    buildings: List[Dict]
    inflow: float
    direction: float
    height: float


def _aligned_array(domain: Tuple[float, float, float], n: int, height: float, gap: float) -> List[Dict]:
    """An n x n aligned building array centred in the domain."""
    lx, ly, _ = domain
    b = 12.0  # building footprint edge [m]
    total_span = n * b + (n - 1) * gap
    x0 = 0.5 * (lx - total_span)
    y0 = 0.5 * (ly - total_span)
    out = []
    for i in range(n):
        for j in range(n):
            bx = x0 + i * (b + gap)
            by = y0 + j * (b + gap)
            out.append({"x": (bx, bx + b), "y": (by, by + b), "h": height})
    return out


def _staggered_array(domain: Tuple[float, float, float], height: float, gap: float) -> List[Dict]:
    """A 2x2 staggered array with a lateral offset between rows."""
    lx, ly, _ = domain
    b = 12.0
    x0 = 0.5 * (lx - (2 * b + gap))
    y0 = 0.5 * (ly - (2 * b + gap))
    return [
        {"x": (x0, x0 + b), "y": (y0 + 6.0, y0 + 6.0 + b), "h": height},
        {"x": (x0, x0 + b), "y": (y0 + 6.0 + b + gap, y0 + 6.0 + 2 * b + gap), "h": height},
        {"x": (x0 + b + gap, x0 + 2 * b + gap), "y": (y0, y0 + b), "h": height},
        {"x": (x0 + b + gap, x0 + 2 * b + gap), "y": (y0 + b + gap, y0 + 2 * b + gap), "h": height},
    ]


def _single_block(domain: Tuple[float, float, float], height: float, xc: float, yc: float) -> Dict:
    lx, ly, _ = domain
    b = 16.0
    return {"x": (xc - b / 2, xc + b / 2), "y": (yc - b / 2, yc + b / 2), "h": height}


def generate_reference_dataset(
    n_layouts: int = 18,
    seed: int = 42,
    grid_shape: Tuple[int, int] = (64, 64),
    domain_size_m: Tuple[float, float, float] = (128.0, 128.0, 60.0),
) -> List[UrbanLayoutSample]:
    """Generate a deterministic, fixed-grid analytical reference dataset.

    All layouts share ``grid_shape`` and ``domain_size_m`` so the neural models
    can batch them. The sweep varies family (aligned / staggered / single
    block), plan density (via inter-building gap), building height, inflow speed
    and direction. Provenance is ``ANALYTICAL_REFERENCE`` on every sample.
    """
    heights = [12.0, 20.0, 28.0]
    inflows = [3.5, 5.0, 7.0]
    directions = [0.0, 15.0, 30.0]
    gaps = [6.0, 14.0, 24.0]  # small gap => dense; large gap => sparse

    rng = np.random.default_rng(seed)
    specs: List[_LayoutSpec] = []

    # Deterministic, reproducible parameter assignment (no randomness in fields).
    idx = 0
    while len(specs) < n_layouts:
        fam_choice = idx % 3
        h = heights[idx % len(heights)]
        u = inflows[idx % len(inflows)]
        d = directions[idx % len(directions)]
        gap = gaps[idx % len(gaps)]
        if fam_choice == 0:
            fam = LayoutFamily.IDEALIZED_ALIGNED
            bld = _aligned_array(domain_size_m, n=2, height=h, gap=gap)
            lid = f"ref_aligned_{idx:02d}"
        elif fam_choice == 1:
            fam = LayoutFamily.IDEALIZED_STAGGERED
            bld = _staggered_array(domain_size_m, height=h, gap=gap)
            lid = f"ref_staggered_{idx:02d}"
        else:
            fam = LayoutFamily.REALISTIC_NEIGHBORHOOD
            lx, ly, _ = domain_size_m
            xc = float(rng.uniform(0.35 * lx, 0.65 * lx))
            yc = float(rng.uniform(0.35 * ly, 0.65 * ly))
            bld = [_single_block(domain_size_m, height=h, xc=xc, yc=yc)]
            lid = f"ref_single_{idx:02d}"
        specs.append(_LayoutSpec(lid, fam, bld, u, d, h))
        idx += 1

    samples: List[UrbanLayoutSample] = []
    for s in specs:
        samples.append(
            build_analytical_reference_sample(
                layout_id=s.layout_id,
                layout_family=s.family,
                grid_shape=grid_shape,
                domain_size_m=domain_size_m,
                buildings=s.buildings,
                inflow_speed_mps=s.inflow,
                wind_direction_deg=s.direction,
                reference_height_m=10.0,
                roughness_length_z0_m=0.05,
            )
        )
    return samples
