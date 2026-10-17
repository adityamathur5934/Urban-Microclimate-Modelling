# Model Design Specification: AML Baseline and Neural Surrogates

**Course Project:** Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)  
**Role:** Machine Learning & Deep Learning Lead  
**Status:** Implemented, Modularized, and Tested  

---

## 1. Executive Summary

This document specifies the technical design, mathematical interfaces, and architectural choices for the two predictive modeling approaches evaluated in this project:

1. **Applied Machine Learning (AML) Baseline:** A tabular tree-ensemble regressor predicting area-averaged physical quantities ($\bar{U}_{\text{ped}} \in \mathbb{R}$) from layout summary morphometrics and environmental inflow boundary conditions.
2. **Deep Learning (DL) Neural Surrogate:** A compact convolutional encoder-decoder network predicting high-resolution 2D spatial physical fields ($U_{\text{ped}}(y, x) \in \mathbb{R}^{N_y \times N_x}$) from multi-channel rasterized geometry and boundary condition grids.

```
                           +------------------------------------------+
                           |   Raw Urban Layout & Inflow Conditions   |
                           +------------------------------------------+
                                       /                  \
                                      /                    \
           [Tabular Feature Extraction]                    [Spatial 2D Rasterization]
                        |                                              |
                        v                                              v
          Feature Vector x in R^11                      Multi-Channel Tensor X in R^(7 x H x W)
       (lambda_p, lambda_f, H_stats, z0, U_inflow)    (Height, Mask, SDF, Inflow U/V, Coords X/Y)
                        |                                              |
                        v                                              v
           +-------------------------+                    +-------------------------+
           |  Random Forest Pipeline |                    |   Compact U-Net / CNN   |
           |  (Train-fitted Scaler)  |                    | (Outdoor-masked Output) |
           +-------------------------+                    +-------------------------+
                        |                                              |
                        v                                              v
           Predicted Scalar Mean y_hat                    Predicted Spatial Field Y_hat
            [Area-Averaged Wind Speed]                     [2D Wind Velocity Field]
                        \                                              /
                         \-----> [Spatial Aggregation Bridge] <-------/
                                (Fair Apples-to-Apples Comparison)
```

---

## 2. Explicit Target Variable Configurations

To prevent hardcoded assumptions regarding the physical target variable, the modeling pipeline defines explicit, immutable contracts via `TargetConfig` ([src/models/interfaces.py](file:///Users/vaidehisahu/Downloads/amlproj/Urban-Microclimate-Modelling/src/models/interfaces.py)):

| Configuration Name | Physical Target Variable | Output Dimension | Physical Unit | Description |
|---|---|---|---|---|
| `DEFAULT_SCALAR_WIND_TARGET` | `area_averaged_wind_speed` | Scalar ($\mathbb{R}$) | $\text{m/s}$ | Mean wind velocity magnitude over outdoor pedestrian cells at $z = 1.75\text{ m}$. Primary target for AML baseline. |
| `DEFAULT_WIND_FIELD_TARGET` | `pedestrian_wind_speed` | 2D Field ($\mathbb{R}^{N_y \times N_x}$) | $\text{m/s}$ | Horizontal wind speed field slice at pedestrian height ($z = 1.75\text{ m}$). Primary target for Neural Surrogate. |
| `OPTIONAL_TEMPERATURE_FIELD_TARGET` | `temperature_field` | 2D Field ($\mathbb{R}^{N_y \times N_x}$) | $\text{K}$ | Pedestrian-level thermal air temperature field (for thermal microclimate extensions). |
| `OPTIONAL_TKE_FIELD_TARGET` | `turbulent_kinetic_energy` | 2D Field ($\mathbb{R}^{N_y \times N_x}$) | $\text{m}^2/\text{s}^2$ | Turbulent kinetic energy $k = \frac{1}{2}(\overline{u'^2} + \overline{v'^2} + \overline{w'^2})$ indicating local gustiness. |

---

## 3. AML Baseline Architecture: Random Forest Pipeline

The AML baseline is implemented in [src/models/baseline_rf.py](file:///Users/vaidehisahu/Downloads/amlproj/Urban-Microclimate-Modelling/src/models/baseline_rf.py) using Scikit-Learn.

### 3.1 Input Morphometric & Environmental Features ($D = 11$)
The feature vector $\mathbf{x} \in \mathbb{R}^{11}$ encapsulates domain-standard urban aerodynamics parameters:
1. `plan_area_density_lambda_p`: $\lambda_p = A_{\text{built}} / A_{\text{total}}$ (ratio of built footprint to lot area).
2. `frontal_area_density_lambda_f`: $\lambda_f = A_{\text{frontal}} / A_{\text{total}}$ (projected facade area orthogonal to wind direction $\delta$).
3. `mean_building_height_m`: $H_{\text{mean}}$ over built structures.
4. `std_building_height_m`: $H_{\text{std}}$ (height variance driving vertical momentum exchange).
5. `max_building_height_m`: $H_{\text{max}}$ (tallest building causing downwash and corner vortices).
6. `wall_area_density_lambda_w`: Total vertical facade surface area / total lot area.
7. `roughness_length_macdonald_m`: Aerodynamic roughness length $z_0$ computed via MacDonald et al. (1998).
8. `displacement_height_d_m`: Zero-plane displacement height $d$ (MacDonald et al., 1998).
9. `inflow_speed_mps`: Reference approach wind velocity $U_{\text{ref}}$ at height $z_{\text{ref}}$.
10. `wind_direction_cos`: $\cos(\delta)$, capturing directional inflow orientation.
11. `wind_direction_sin`: $\sin(\delta)$, capturing orthogonal inflow orientation.

### 3.2 Pipeline Composition & Train-Only Normalization
```python
Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    ))
])
```
- **Strict Isolation:** `StandardScaler` computes sample mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$ solely from the training partition. Validation and test layouts are transformed using $\mu_{\text{train}}$ and $\sigma_{\text{train}}$.
- **Interpretability:** Random Forest enables direct extraction of Gini / variance-reduction feature importances, revealing which morphometric parameters dominate outdoor ventilation.

---

## 4. Neural Surrogate Architecture: Compact U-Net & CNN

The deep learning surrogate is implemented in PyTorch in [src/models/neural_unet.py](file:///Users/vaidehisahu/Downloads/amlproj/Urban-Microclimate-Modelling/src/models/neural_unet.py).

### 4.1 Input Tensor Representation ($C_{\text{in}} = 7$)
The input tensor $\mathbf{X} \in \mathbb{R}^{7 \times N_y \times N_x}$ combines topographical morphology and boundary forcing:
- Channel 0 (`height_map_m`): Topographical building elevation in meters.
- Channel 1 (`building_mask`): Binary indicator ($1.0 = \text{building interior}$, $0.0 = \text{fluid space}$).
- Channel 2 (`signed_distance_m`): Euclidean signed distance field (SDF) in meters ($>0$ outside building, $<0$ inside). Provides continuous proximity gradients to building facades.
- Channel 3 (`inflow_u_mps`): Approach velocity streamwise component $U_{\text{ref}} \cos(\delta)$ spatially tiled.
- Channel 4 (`inflow_v_mps`): Approach velocity spanwise component $U_{\text{ref}} \sin(\delta)$ spatially tiled.
- Channel 5 (`coord_x_norm`): Normalized coordinate $x / L_x \in [0, 1]$.
- Channel 6 (`coord_y_norm`): Normalized coordinate $y / L_y \in [0, 1]$.

### 4.2 Compact U-Net Specification
A symmetric multi-scale encoder-decoder with skip connections preserving fine-scale boundary details:
- **Encoder:** Stage 0 (`DoubleConv` with 32 filters) followed by `depth=3` downsampling stages (`MaxPool2D(2)` + `DoubleConv`), expanding feature maps from $32 \to 64 \to 128 \to 256$.
- **Decoder:** 3 upsampling stages (`Bilinear Upsampling(2)` + concatenated skip connections + `DoubleConv`), contracting feature maps from $256 \to 128 \to 64 \to 32$.
- **Prediction Head:** $1 \times 1$ Conv2D mapping 32 filters to `out_channels` ($C_{\text{out}} = 1$ for wind speed).
- **Physical Outdoor Masking:** Solid building interiors cannot have wind velocity. When `enforce_zero_inside_buildings=True`, the network output is multiplied by $(1 - M_{\text{building}})$, enforcing $U = 0.0$ inside building footprints by construction.
- **Parameter Count:** ~480,000 parameters, ensuring rapid training and inference without overfitting on small CFD sweeps.

### 4.3 Alternative: Compact Residual CNN
A 4-to-6 layer convolutional network using dilated convolutions (dilation rates $1, 2, 4$) to expand the receptive field across building wakes without downsampling feature maps. Parameter count: ~75,000 parameters.

---

## 5. Loss Formulation & Optimization

Implemented in [src/training/losses.py](file:///Users/vaidehisahu/Downloads/amlproj/Urban-Microclimate-Modelling/src/training/losses.py):

### 5.1 Masked Regression Losses
Because fluid flow only occurs in outdoor space, unmasked loss calculations over solid building footprints artificially damp errors (since building interiors are trivial constant zeros). The loss is computed strictly over outdoor fluid cells:

$$\mathcal{L}_{\text{L1}} = \frac{1}{\sum_{(y, x)} M_{\text{fluid}}(y, x)} \sum_{(y, x)} |Y(y, x) - \hat{Y}(y, x)| \cdot M_{\text{fluid}}(y, x)$$

$$\mathcal{L}_{\text{MSE}} = \frac{1}{\sum_{(y, x)} M_{\text{fluid}}(y, x)} \sum_{(y, x)} (Y(y, x) - \hat{Y}(y, x))^2 \cdot M_{\text{fluid}}(y, x)$$

### 5.2 Composite Surrogate Loss
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{L1}} + (1 - \alpha) \mathcal{L}_{\text{MSE}} + \lambda_{\text{grad}} \mathcal{L}_{\text{grad}}$$
- $\alpha = 0.7$: Prioritizes L1 robustness against sharp gradients at building corners while preserving MSE penalty on peak velocity gusts.
- $\lambda_{\text{grad}} = 0.05$: Spatial finite-difference gradient penalty encouraging smooth shear layers across wake boundaries.
- **Optimization:** AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$), Cosine Annealing learning rate schedule, and early stopping on validation NRMSE with checkpoint preservation.
