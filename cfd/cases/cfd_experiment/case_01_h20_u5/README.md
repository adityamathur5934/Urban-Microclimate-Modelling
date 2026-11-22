# CFD case: case_01_h20_u5

Quasi-2D (x-z plane, 1 cell thick) steady incompressible RANS case around a
single surface-mounted building block. **Generated, not executed** — see the
top-level README and `docs/cfd_surrogate_experiment.md`.

## Geometry & domain
- Building height H = 20.0 m, width W = 20.0 m
- Upstream = 5W, wake = 15W, domain height = 6H (quasi-2D, 1 cell in y)

## Inflow / boundary conditions
- Inlet: atmospheric boundary layer `atmBoundaryLayerInletVelocity`, U_ref = 5.0 m/s @ Z_ref = 10.0 m, z0 = 0.05 m
- Outlet: `inletOutlet` (p = 0)
- Ground: `noSlip`; Building: `noSlip`; Top: `slip`; frontAndBack: `empty`
- Inlet turbulence (Richards & Hoxey 1993): k = 0.4981 m^2/s^2, omega = 0.3127 s^-1

## Solver & numerics
- Solver: `simpleFoam` (steady, semi-implicit SIMPLE)
- Turbulence: RANS k-omega SST (`kOmegaSST`)
- Schemes: linearUpwind (U), upwind (k, omega), Gauss linear corrected (Laplacian)
- SIMPLE relaxation: p 0.3, U/k/omega 0.7; residualControl 1e-5
- endTime = 2000 iterations (steady state)

## How to run (when OpenFOAM is available)
```bash
cd cfd/cases/cfd_experiment/case_01_h20_u5
./Allrun
# then sample the z = 1.75 m slice and convert:
# python cfd/scripts/foam_to_contract.py --input-file <sampleFile> --output-npz ../../outputs/case_01_h20_u5.npz
```

## Limitation
This is a 2D vertical-plane case; its pedestrian (z = 1.75 m) slice is a 1D
profile over x, **not** a 2D (x, y) horizontal field. A 3D case (snappyHexMesh
over a building array) is required to produce the 2D horizontal target the
neural surrogate predicts; see the experiment report.
