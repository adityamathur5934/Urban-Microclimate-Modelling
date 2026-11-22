#!/usr/bin/env python3
"""Reproducible OpenFOAM case generator for simplified urban building-block CFD.

Generates a small collection of quasi-2D (x-z plane, 1 cell thick) steady
incompressible RANS cases around a single surface-mounted building block,
ready to run with ``simpleFoam`` (k-omega SST). The mesh, boundary conditions,
solver settings, and turbulence model are documented per case.

IMPORTANT
---------
* This script **does not run** any solver. It only writes case directories.
  ``simpleFoam`` / ``blockMesh`` are NOT available in this execution
  environment (macOS Apple Silicon, no OpenFOAM installation / image), so no
  CFD simulation is executed or claimed here. See ``docs/cfd_surrogate_experiment.md``.
* The cases are quasi-2D vertical-plane cases. Their converged solution is a 2D
  (x, z) field; a z = 1.75 m pedestrian slice is a 1D profile over x, **not** a
  2D (x, y) horizontal field. Producing the 2D horizontal pedestrian target that
  the neural surrogate predicts requires 3D cases (e.g. snappyHexMesh over a
  building array); that is the remaining CFD step documented in the report.
* When a solver is available, run ``./Allrun`` in each case, then convert the
  sampled pedestrian slice to the data contract with
  ``cfd/scripts/foam_to_contract.py`` and feed it into the experiment pipeline.

Usage:
    python cfd/scripts/generate_cfd_cases.py
    python cfd/scripts/generate_cfd_cases.py --out cfd/cases/cfd_experiment --n-cases 5
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATE = PROJECT_ROOT / "cfd" / "cases" / "simple2d_building_block"

HEADER = """/*--------------------------------*- OpenFOAM -*-----------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312 / 10 / 11                       |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
"""


@dataclass
class CaseSpec:
    case_id: str
    building_height_m: float
    building_width_m: float
    inflow_speed_mps: float
    reference_height_m: float
    roughness_z0_m: float


def _abl_k_omega(u_ref: float, z_ref: float, z0: float, kappa: float = 0.41, cmu: float = 0.09):
    """Richards & Hoxey (1993) inlet turbulence: k = u*^2/sqrt(Cmu), omega = u*/(sqrt(Cmu)*kappa*(Zref+z0))."""
    z0e = max(z0, 1e-4)
    u_star = (kappa * u_ref) / math.log((z_ref + z0e) / z0e)
    k = (u_star ** 2) / math.sqrt(cmu)
    omega = u_star / (math.sqrt(cmu) * kappa * (z_ref + z0e))
    return k, omega


def block_mesh_dict(spec: CaseSpec) -> str:
    H = spec.building_height_m
    W = spec.building_width_m
    Lu = 5.0 * W            # upstream length
    Lw = 15.0 * W           # wake length
    Lz = 6.0 * H            # domain height
    x_front = Lu
    x_back = Lu + W
    Lx = x_back + Lw
    nx_up, nx_b, nx_w = 50, 20, 100
    nz_h = max(20, int(round(H / (Lz / 50))))
    nz_top = 30
    s = HEADER + f"""    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale   1;

// Quasi-2D (1 cell thick in y). Domain: Lx={Lx:.1f}m (upstream {Lu:.1f}, building {W:.1f}, wake {Lw:.1f}), Lz={Lz:.1f}m (H={H:.1f}m)

vertices
(
    // Plane y = -0.5
    (  0.0 -0.5   0.0) // 0 inlet bottom
    ({x_front:.1f} -0.5   0.0) // 1 building front ground
    ({x_back:.1f} -0.5   0.0) // 2 building back ground
    ({Lx:.1f} -0.5   0.0) // 3 outlet bottom
    (  0.0 -0.5  {H:.1f}) // 4 inlet roof height
    ({x_front:.1f} -0.5  {H:.1f}) // 5 building windward top
    ({x_back:.1f} -0.5  {H:.1f}) // 6 building leeward top
    ({Lx:.1f} -0.5  {H:.1f}) // 7 outlet roof height
    (  0.0 -0.5  {Lz:.1f}) // 8 inlet top
    ({x_front:.1f} -0.5  {Lz:.1f}) // 9 above building windward top
    ({x_back:.1f} -0.5  {Lz:.1f}) // 10 above building leeward top
    ({Lx:.1f} -0.5  {Lz:.1f}) // 11 outlet top
    // Plane y = +0.5
    (  0.0  0.5   0.0) // 12
    ({x_front:.1f}  0.5   0.0) // 13
    ({x_back:.1f}  0.5   0.0) // 14
    ({Lx:.1f}  0.5   0.0) // 15
    (  0.0  0.5  {H:.1f}) // 16
    ({x_front:.1f}  0.5  {H:.1f}) // 17
    ({x_back:.1f}  0.5  {H:.1f}) // 18
    ({Lx:.1f}  0.5  {H:.1f}) // 19
    (  0.0  0.5  {Lz:.1f}) // 20
    ({x_front:.1f}  0.5  {Lz:.1f}) // 21
    ({x_back:.1f}  0.5  {Lz:.1f}) // 22
    ({Lx:.1f}  0.5  {Lz:.1f}) // 23
);

blocks
(
    hex (0 1 13 12 4 5 17 16) ({nx_up} 1 {nz_h}) simpleGrading (1 1 1)
    hex (4 5 17 16 8 9 21 20) ({nx_up} 1 {nz_top}) simpleGrading (1 1 1)
    hex (5 6 18 17 9 10 22 21) ({nx_b} 1 {nz_top}) simpleGrading (1 1 1)
    hex (2 3 15 14 6 7 19 18) ({nx_w} 1 {nz_h}) simpleGrading (1 1 1)
    hex (6 7 19 18 10 11 23 22) ({nx_w} 1 {nz_top}) simpleGrading (1 1 1)
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 12 16 4)
            (4 16 20 8)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            (3 7 19 15)
            (7 11 23 19)
        );
    }}
    ground
    {{
        type wall;
        faces
        (
            (0 1 13 12)
            (2 3 15 14)
        );
    }}
    building
    {{
        type wall;
        faces
        (
            (1 5 17 13)
            (5 6 18 17)
            (2 14 18 6)
        );
    }}
    top
    {{
        type patch;
        faces
        (
            (8 9 21 20)
            (9 10 22 21)
            (10 11 23 22)
        );
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
            (0 4 5 1) (4 8 9 5) (5 9 10 6) (2 6 7 3) (6 10 11 7)
            (12 13 17 16) (16 17 21 20) (17 18 22 21) (14 15 19 18) (18 19 23 22)
        );
    }}
);

mergePatchPairs
(
);
// ************************************************************************* //
"""
    return s


def control_dict(spec: CaseSpec, end_time: int = 2000) -> str:
    cx = 5.0 * spec.building_width_m + 0.5 * spec.building_width_m
    cz = 0.5 * spec.building_height_m
    return HEADER + f"""    class       dictionary;
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      3;
writeFormat     ascii;
writePrecision  7;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    residuals
    {{
        type            residuals;
        libs            ("libutilityFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        fields          (p U k omega);
    }}
    buildingForces
    {{
        type            forces;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   10;
        patches         (building);
        rho             rhoInf;
        rhoInf          1.225;
        CofG            ({cx:.1f} 0 {cz:.1f});
    }}
}}
// ************************************************************************* //
"""


def u_field(spec: CaseSpec) -> str:
    return HEADER + f"""    class       volVectorField;
    location    "0";
    object      U;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({spec.inflow_speed_mps:.2f} 0 0);

boundaryField
{{
    inlet
    {{
        type            atmBoundaryLayerInletVelocity;
        flowDir         (1 0 0);
        zDir            (0 0 1);
        Uref            {spec.inflow_speed_mps:.2f};
        Zref            {spec.reference_height_m:.2f};
        z0              uniform {spec.roughness_z0_m:.4f};
        d               uniform 0.0;
        value           uniform ({spec.inflow_speed_mps:.2f} 0 0);
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform (0 0 0);
        value           uniform ({spec.inflow_speed_mps:.2f} 0 0);
    }}
    ground
    {{
        type            noSlip;
    }}
    building
    {{
        type            noSlip;
    }}
    top
    {{
        type            slip;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
// ************************************************************************* //
"""


def k_field(spec: CaseSpec) -> str:
    k, _ = _abl_k_omega(spec.inflow_speed_mps, spec.reference_height_m, spec.roughness_z0_m)
    return HEADER + f"""    class       volScalarField;
    location    "0";
    object      k;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];
// Richards & Hoxey (1993): k = u*^2 / sqrt(Cmu) = {k:.4f} m2/s2
internalField   uniform {k:.4f};

boundaryField
{{
    inlet
    {{
        type            atmBoundaryLayerInletK;
        Uref            {spec.inflow_speed_mps:.2f};
        Zref            {spec.reference_height_m:.2f};
        z0              uniform {spec.roughness_z0_m:.4f};
        d               uniform 0.0;
        value           uniform {k:.4f};
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform {k:.4f};
        value           uniform {k:.4f};
    }}
    ground
    {{
        type            kqRWallFunction;
        value           uniform {k:.4f};
    }}
    building
    {{
        type            kqRWallFunction;
        value           uniform {k:.4f};
    }}
    top
    {{
        type            slip;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
// ************************************************************************* //
"""


def omega_field(spec: CaseSpec) -> str:
    _, omega = _abl_k_omega(spec.inflow_speed_mps, spec.reference_height_m, spec.roughness_z0_m)
    return HEADER + f"""    class       volScalarField;
    location    "0";
    object      omega;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 -1 0 0 0 0];
// Specific dissipation rate at zref: omega = {omega:.4f} s-1
internalField   uniform {omega:.4f};

boundaryField
{{
    inlet
    {{
        type            atmBoundaryLayerInletOmega;
        Uref            {spec.inflow_speed_mps:.2f};
        Zref            {spec.reference_height_m:.2f};
        z0              uniform {spec.roughness_z0_m:.4f};
        d               uniform 0.0;
        value           uniform {omega:.4f};
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform {omega:.4f};
        value           uniform {omega:.4f};
    }}
    ground
    {{
        type            omegaWallFunction;
        value           uniform {omega:.4f};
    }}
    building
    {{
        type            omegaWallFunction;
        value           uniform {omega:.4f};
    }}
    top
    {{
        type            slip;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
// ************************************************************************* //
"""


ALLRUN = """#!/bin/bash
# Reproducible runner for a generated CFD experiment case.
# Requires OpenFOAM (v2312/10/11) on PATH or sourced, OR an OpenFOAM container.
set -e
if ! command -v blockMesh >/dev/null 2>&1 || ! command -v simpleFoam >/dev/null 2>&1; then
    echo "ERROR: OpenFOAM not available (blockMesh/simpleFoam not on PATH)."
    echo "Source your OpenFOAM etc/bashrc, or run inside an OpenFOAM docker image, then re-run."
    exit 1
fi
echo "[1/3] blockMesh"; blockMesh > log.blockMesh 2>&1
echo "[2/3] checkMesh"; checkMesh > log.checkMesh 2>&1
echo "[3/3] simpleFoam"; simpleFoam > log.simpleFoam 2>&1
echo "Done. Inspect log.simpleFoam for residuals and convergence."
"""


def case_readme(spec: CaseSpec) -> str:
    k, omega = _abl_k_omega(spec.inflow_speed_mps, spec.reference_height_m, spec.roughness_z0_m)
    return f"""# CFD case: {spec.case_id}

Quasi-2D (x-z plane, 1 cell thick) steady incompressible RANS case around a
single surface-mounted building block. **Generated, not executed** — see the
top-level README and `docs/cfd_surrogate_experiment.md`.

## Geometry & domain
- Building height H = {spec.building_height_m} m, width W = {spec.building_width_m} m
- Upstream = 5W, wake = 15W, domain height = 6H (quasi-2D, 1 cell in y)

## Inflow / boundary conditions
- Inlet: atmospheric boundary layer `atmBoundaryLayerInletVelocity`, U_ref = {spec.inflow_speed_mps} m/s @ Z_ref = {spec.reference_height_m} m, z0 = {spec.roughness_z0_m} m
- Outlet: `inletOutlet` (p = 0)
- Ground: `noSlip`; Building: `noSlip`; Top: `slip`; frontAndBack: `empty`
- Inlet turbulence (Richards & Hoxey 1993): k = {k:.4f} m^2/s^2, omega = {omega:.4f} s^-1

## Solver & numerics
- Solver: `simpleFoam` (steady, semi-implicit SIMPLE)
- Turbulence: RANS k-omega SST (`kOmegaSST`)
- Schemes: linearUpwind (U), upwind (k, omega), Gauss linear corrected (Laplacian)
- SIMPLE relaxation: p 0.3, U/k/omega 0.7; residualControl 1e-5
- endTime = 2000 iterations (steady state)

## How to run (when OpenFOAM is available)
```bash
cd cfd/cases/cfd_experiment/{spec.case_id}
./Allrun
# then sample the z = 1.75 m slice and convert:
# python cfd/scripts/foam_to_contract.py --input-file <sampleFile> --output-npz ../../outputs/{spec.case_id}.npz
```

## Limitation
This is a 2D vertical-plane case; its pedestrian (z = 1.75 m) slice is a 1D
profile over x, **not** a 2D (x, y) horizontal field. A 3D case (snappyHexMesh
over a building array) is required to produce the 2D horizontal target the
neural surrogate predicts; see the experiment report.
"""


def write_case(spec: CaseSpec, out_root: Path) -> Path:
    case_dir = out_root / spec.case_id
    # Copy static files from the validated template (fvSchemes, fvSolution, 0/p, 0/nut, constant/*)
    if TEMPLATE.exists():
        shutil.copytree(TEMPLATE, case_dir, dirs_exist_ok=True)
        # Remove template-specific sampleDict/Allclean/Allrun/README we will replace
        for f in ("Allrun", "Allclean", "README.md", "system/sampleDict"):
            p = case_dir / f
            if p.exists():
                p.unlink()
    else:
        (case_dir / "system").mkdir(parents=True, exist_ok=True)
        (case_dir / "0").mkdir(parents=True, exist_ok=True)
        (case_dir / "constant").mkdir(parents=True, exist_ok=True)

    (case_dir / "system" / "blockMeshDict").write_text(block_mesh_dict(spec))
    (case_dir / "system" / "controlDict").write_text(control_dict(spec))
    (case_dir / "0" / "U").write_text(u_field(spec))
    (case_dir / "0" / "k").write_text(k_field(spec))
    (case_dir / "0" / "omega").write_text(omega_field(spec))
    (case_dir / "Allrun").write_text(ALLRUN)
    (case_dir / "Allrun").chmod(0o755)
    (case_dir / "README.md").write_text(case_readme(spec))
    return case_dir


def default_specs() -> List[CaseSpec]:
    return [
        CaseSpec("case_01_h20_u5", 20.0, 20.0, 5.0, 10.0, 0.05),
        CaseSpec("case_02_h20_u7", 20.0, 20.0, 7.0, 10.0, 0.05),
        CaseSpec("case_03_h12_u5", 12.0, 20.0, 5.0, 10.0, 0.05),
        CaseSpec("case_04_h28_u5", 28.0, 20.0, 5.0, 10.0, 0.05),
        CaseSpec("case_05_h20_u3", 20.0, 20.0, 3.0, 10.0, 0.05),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate reproducible OpenFOAM building-block CFD cases (no solver execution).")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "cfd" / "cases" / "cfd_experiment"))
    parser.add_argument("--n-cases", type=int, default=0, help="Override number of cases (0 = use default set)")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    specs = default_specs()
    if args.n_cases > 0:
        specs = specs[: args.n_cases]

    written = [write_case(s, out_root) for s in specs]

    top_readme = out_root / "README.md"
    top_readme.write_text(
        "# CFD experiment cases (generated, NOT executed)\n\n"
        "Reproducible quasi-2D `simpleFoam` RANS k-omega SST cases around a single\n"
        "building block, generated by `cfd/scripts/generate_cfd_cases.py`.\n\n"
        "## Status\n"
        "**No simulation was run.** OpenFOAM (`blockMesh`, `simpleFoam`) is not\n"
        "available in this execution environment (macOS Apple Silicon, no OpenFOAM\n"
        "installation or container image). The cases below are ready to run when a\n"
        "solver is available.\n\n"
        "## Cases\n"
        + "\n".join(f"- `{p.name}`" for p in written) + "\n\n"
        "## How to run\n"
        "Source OpenFOAM (or start an OpenFOAM docker container), then in each case:\n"
        "`./Allrun`. Convert the sampled pedestrian slice with\n"
        "`cfd/scripts/foam_to_contract.py` and feed the resulting NPZ into the\n"
        "experiment pipeline (`src.experiment.runner.run_experiment`).\n\n"
        "## Limitation\n"
        "These are 2D vertical-plane cases; the z = 1.75 m slice is a 1D profile over x.\n"
        "Producing the 2D (x, y) horizontal pedestrian target the surrogate predicts\n"
        "requires 3D cases (snappyHexMesh over a building array) — the remaining CFD step.\n"
    )
    print(f"Generated {len(written)} CFD cases under {out_root}")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
