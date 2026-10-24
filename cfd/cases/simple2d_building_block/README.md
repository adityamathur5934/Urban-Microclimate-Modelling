# OpenFOAM 2D Building-Block Case Template

**Case Name:** `simple2d_building_block`  
**Application:** `simpleFoam` (Steady-state incompressible RANS)  
**Turbulence Closure:** Menter Shear Stress Transport ($k$-$\omega$ SST)  
**Status:** **Proposed Case Template** (Configured, verified dictionary syntax; not yet executed locally pending container runtime approval)

---

## 1. Case Overview & Physical Setup

This directory contains a complete, production-grade OpenFOAM case template designed to compute steady-state aerodynamic flow around a surface-mounted bluff building block under neutral atmospheric boundary layer conditions.

```
                               Top Boundary: Slip (z = 120m, 6H)
    +-----------------------------------------------------------------------------------+
    |                                                                                   |
    |  Inlet (x = 0)                                                    Outlet (x=420m) |
    |  ABL Log Profile U(z)                                             p = 0, zeroGrad |
    |  Uref = 5 m/s @ 10m                      +--------+               ======>         |
    |  =======>                                |Building|   Wake Cavity                 |
    |  =======>                                | H = 20m|   (u < 0)                     |
    |  =======>                                +--------+                               |
    |  ========================================+        +=============================  |
    |  Ground: noSlip, wall functions (z = 0)   W = 20m                                 |
    +-----------------------------------------------------------------------------------+
    |<-------------- 100m (5H) --------------->|<--20m->|<------------ 300m (15H) ------>|
```

### 1.1 Geometry & Domain Dimensions
- **Domain Length ($L_x$):** $420\text{ m}$
  - Upstream fetch: $100\text{ m}$ ($5 \times H$)
  - Building obstacle: $20\text{ m}$ ($1 \times H$)
  - Downstream wake fetch: $300\text{ m}$ ($15 \times H$)
- **Domain Height ($L_z$):** $120\text{ m}$ ($6 \times H$)
- **Domain Width ($L_y$):** $1.0\text{ m}$ (Quasi-2D representation, 1 cell thick in $y$)
- **Blockage Ratio:** $\frac{A_{\text{building}}}{A_{\text{domain}}} = \frac{20 \times 1}{120 \times 1} \approx 16.7\%$ (Note: In 2D, domain height of $6H$ follows standard benchmark setups such as Castro & Robins 1977; 3D setups require blockage $< 3\%$).

---

## 2. Mesh Architecture (`system/blockMeshDict`)

The computational domain is discretized into 5 structured hexahedral blocks:
- **Block 0 (Upstream lower):** $x \in [0, 100]$, $z \in [0, 20]$ $\rightarrow$ $50 \times 1 \times 20$ cells ($\Delta x = 2.0\text{ m}, \Delta z = 1.0\text{ m}$)
- **Block 1 (Upstream upper):** $x \in [0, 100]$, $z \in [20, 120]$ $\rightarrow$ $50 \times 1 \times 30$ cells
- **Block 2 (Rooftop upper):** $x \in [100, 120]$, $z \in [20, 120]$ $\rightarrow$ $20 \times 1 \times 30$ cells ($\Delta x = 1.0\text{ m}$)
- **Block 3 (Downstream wake lower):** $x \in [120, 420]$, $z \in [0, 20]$ $\rightarrow$ $100 \times 1 \times 20$ cells
- **Block 4 (Downstream wake upper):** $x \in [120, 420]$, $z \in [20, 120]$ $\rightarrow$ $100 \times 1 \times 30$ cells
- **Total Cell Count:** $1,000 + 1,500 + 600 + 2,000 + 3,000 = \mathbf{8,100\text{ cells}}$

---

## 3. Boundary Conditions & Physical Specifications

| Patch Name | Patch Type | Velocity ($\mathbf{U}$) | Kinematic Pressure ($p$) | Turbulent Kinetic Energy ($k$) | Specific Dissipation ($\omega$) | Eddy Viscosity ($\nu_t$) |
|---|---|---|---|---|---|---|
| **inlet** | `patch` | `atmBoundaryLayerInletVelocity` ($U_{\text{ref}}=5\text{ m/s}, z_{\text{ref}}=10\text{ m}, z_0=0.05\text{ m}$) | `zeroGradient` | `atmBoundaryLayerInletK` | `atmBoundaryLayerInletOmega` | `calculated` |
| **outlet** | `patch` | `inletOutlet` (inletValue: `(0 0 0)`) | `fixedValue` ($p = 0\text{ m}^2/\text{s}^2$) | `zeroGradient` | `zeroGradient` | `calculated` |
| **ground** | `wall` | `noSlip` | `zeroGradient` | `kqRwallFunction` | `omegaWallFunction` | `nutkWallFunction` |
| **building** | `wall` | `noSlip` | `zeroGradient` | `kqRwallFunction` | `omegaWallFunction` | `nutkWallFunction` |
| **top** | `patch` | `slip` | `zeroGradient` | `slip` | `slip` | `calculated` |
| **frontAndBack** | `empty` | `empty` | `empty` | `empty` | `empty` | `empty` |

---

## 4. Physical Quantities Produced vs. Excluded

### 4.1 Quantities Physically Produced
- **Velocity Vector Field ($\mathbf{u}$):** $u_x(x, z)$ (streamwise) and $u_z(x, z)$ (vertical).
- **Wind Speed Magnitude ($U$):** $U(x, z) = \sqrt{u_x^2 + u_z^2}$ [$\text{m/s}$].
- **Kinematic Pressure ($p$):** $p/\rho$ [$\text{m}^2/\text{s}^2$].
- **Turbulent Kinetic Energy ($k$):** $k(x, z) = \frac{1}{2}\overline{u_i' u_i'}$ [$\text{m}^2/\text{s}^2$].
- **Specific Dissipation Rate ($\omega$):** $\omega(x, z)$ [$1/\text{s}$].
- **Kinematic Eddy Viscosity ($\nu_t$):** $\nu_t = \frac{a_1 k}{\max(a_1 \omega, S F_2)}$ [$\text{m}^2/\text{s}$].
- **Aerodynamic Drag Force & Coefficient:** Integrated surface pressure and friction forces on `building` patch via `forces` function object.

### 4.2 Quantities Excluded (Physical Impossibility in this Setup)
- **Temperature ($T$):** **NOT PRODUCED.** The solver `simpleFoam` is strictly an isothermal Navier–Stokes solver. No thermal energy equation, solar radiation, or thermal boundary conditions are solved.
- **Spanwise Flow ($u_y$):** Identically zero by virtue of the `empty` 2D boundary constraint.
- **3D Horseshoe Vortex Wrapping & Rooftop Divergence:** 2D flow cannot divert around flanks or realistically ventilate over roofs.

---

## 5. Execution Instructions (When Container Runtime is Provisioned)

### Method A: Containerized Execution via Docker (Recommended)
```bash
# From workspace root
docker run -it --rm \
  -v "$(pwd)":/home/foam/project \
  -w /home/foam/project/cfd/cases/simple2d_building_block \
  opencfd/openfoam-default:latest \
  bash -c "source /usr/lib/openfoam/openfoam2312/etc/bashrc && ./Allrun"
```

### Method B: Native OpenFOAM Installation (Linux / macOS sparse image)
```bash
cd cfd/cases/simple2d_building_block
source /usr/lib/openfoam/openfoam2312/etc/bashrc  # Adjust path as appropriate
./Allrun
```

---

## 6. Numerical Validation & Convergence Checks

When executed, the simulation must be validated against the following three objective criteria:
1. **Residual Convergence:** Scaled residuals for $p, U_x, U_z, k, \omega$ must drop below **$10^{-5}$** (enforced in `system/fvSolution`).
2. **Integral Quantity Stabilization:** Total drag force on the building patch (`postProcessing/forces/0/forces.dat`) must oscillate by less than **$0.1\%$** over the final 200 iterations.
3. **Field Plausibility Verification:**
   - Stagnation point on the windward facade at $z \approx 0.65 - 0.75 H$.
   - Flow separation at the windward corner ($x = 100\text{ m}, z = 20\text{ m}$).
   - Leeward wake reverse flow ($u_x < 0$).
   - Ground wake reattachment length $L_r$ within expected 2D bounds ($L_r \sim 10 - 15 H$).

---

## 7. Data Pipeline Export Bridge

Once the simulation completes and sampling data is extracted via `sampleDict`, execute:
```bash
python cfd/scripts/foam_to_contract.py \
  --layout-id cfd_2d_block_001 \
  --input-file postProcessing/sets/1000/pedestrian_wake_U_p_k_omega.xy \
  --output-npz cfd/outputs/cfd_2d_block_001.npz
```
This serializes the CFD solution into a validated `UrbanLayoutSample` conforming to the project's data contract.
