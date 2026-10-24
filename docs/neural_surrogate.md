# Neural Surrogate Architecture & Technical Theory

**Project:** Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)  
**Task:** Task 5 — Neural-Surrogate Prototype  
**Owner:** Project Lead  
**Document Version:** 1.0.0  
**Branch:** `feature/neural-surrogate-prototype`  
**Status:** Approved Reference Documentation  

---

## 1. Executive Summary & Purpose

This document provides the theoretical foundations, mathematical formulations, and engineering architecture for the **2D Neural Surrogate Prototype** implemented in `src/models/`, `src/training/`, and `src/evaluation/`.

The neural surrogate replaces the iterative, computationally expensive numerical solution of the Reynolds-Averaged Navier-Stokes (RANS) or Large-Eddy Simulation (LES) equations with a direct, data-driven non-linear operator mapping:

$$\hat{\mathbf{Y}} = \mathcal{M}_\theta(\mathbf{X})$$

where $\mathbf{X} \in \mathbb{R}^{7 \times N_y \times N_x}$ is a multi-channel spatial input raster (building heights, footprint mask, signed distance field, inflow velocity vectors, and normalized coordinates), and $\hat{\mathbf{Y}} \in \mathbb{R}^{1 \times N_y \times N_x}$ is the predicted pedestrian wind speed field ($z = 1.75\text{ m}$) in physical units [$\text{m/s}$].

---

## 2. Deep Learning Fundamentals for Fluid Dynamics

### 2.1 2D Convolution & Spatial Inductive Bias
In fluid mechanics, the flow state at a given spatial coordinate $(x, y)$ is governed locally by viscous shear stresses and pressure gradients. Convolutional neural networks (CNNs) encode this spatial locality through the discrete 2D cross-correlation operator:

$$S(i, j) = (I * K)(i, j) = \sum_{m=-k}^{k} \sum_{n=-k}^{k} I(i + m, j + n) K(m, n) + b$$

where $I$ is the input feature map, $K \in \mathbb{R}^{(2k+1) \times (2k+1)}$ is the learnable kernel matrix (typically $3 \times 3$), and $b$ is a learnable scalar bias.

#### Simple Explanation:
> *Think of a convolution filter as a small sliding magnifying glass (3x3 pixels) moving across the blueprint of a city. As it slides, it looks for specific local patterns—such as a sharp building corner, a narrow street gap, or an open plaza. By stacking these filters, the network combines simple edges into complex aerodynamic concepts like recirculation eddies and wind jets.*

#### Aerodynamic Benefits of Convolutions:
1. **Translation Equivariance:** If an isolated building is shifted 20 meters downwind, the local stagnation pressure and separation bubble translate accordingly:
   $$f(T_v(x)) = T_v(f(x))$$
2. **Parameter Sharing:** A $3 \times 3$ kernel has only 9 weights regardless of whether the grid is $32 \times 32$ or $1024 \times 1024$, preventing the parameter explosion of fully connected multi-layer perceptrons (MLPs).

---

### 2.2 Feature Maps & Receptive Field Hierarchy
A **feature map** is a 2D spatial channel representing the activation response of a specific learned convolutional kernel:
- **Shallow Layers (Resolution $N_y \times N_x$):** Extract high-frequency geometric primitives—sharp building corners, facade orientations, and solid-fluid boundary edges.
- **Intermediate Layers (Resolution $N_y/2 \times N_x/2$ to $N_y/4 \times N_x/4$):** Combine edge primitives into contextual aerodynamic entities—street canyon corridors, windward stagnation zones, and adjacent building spacing.
- **Deep Bottleneck Layers (Resolution $N_y/16 \times N_x/16$):** Large effective receptive fields capture macro-scale domain flow: whole-neighborhood wake recovery, city-scale pressure drops, and cumulative drag effects.

---

### 2.3 The U-Net Architecture & Skip Connections

Standard convolutional autoencoders downsample inputs through successive pooling layers ($2 \times 2$ MaxPool) to expand the receptive field. While this captures macro-scale flow, downsampling discards fine spatial resolution. In bluff-body aerodynamics, boundary-layer detachment and corner separation occur along razor-thin wall boundaries; reconstructing these from a compressed bottleneck causes severe boundary blurring.

Ronneberger et al. (2015) resolved this bottleneck with the **U-Net** architecture via **Skip Connections**:

```
INPUT (7, H, W)
      |
   [IncConv] -----------------------------------------------------> [UpConv 4] ---> OUTPUT (1, H, W)
      |                                                            ^
  (MaxPool)                                                    (Concat)
      v                                                            |
   [Down 1] -----------------------------------------> [UpConv 3] -+
      |                                                ^
  (MaxPool)                                        (Concat)
      v                                                |
   [Down 2] ----------------------------> [UpConv 2] -+
      |                                   ^
  (MaxPool)                           (Concat)
      v                                   |
   [Down 3] ---------------> [UpConv 1] -+
      |                      ^
  (MaxPool)              (Concat)
      v                      |
   [Down 4 (Bottleneck)] ----+
```

#### Mathematical Formulation of Skip Connections:
At decoder level $l$, the upsampled feature map $\mathbf{x}_{\text{up}}^{(l)}$ is concatenated with the corresponding encoder feature map $\mathbf{x}_{\text{enc}}^{(l)}$ along the channel dimension before convolution:

$$\mathbf{x}_{\text{dec}}^{(l)} = \text{DoubleConv}\left( \left[ \text{Upsample}(\mathbf{x}_{\text{dec}}^{(l-1)}), \, \mathbf{x}_{\text{enc}}^{(l)} \right] \right)$$

#### Why Skip Connections are Vital for Urban Aerodynamics:
1. **Preserving Exact Building Boundaries:** High-resolution wall coordinates from the shallow encoder bypass the bottleneck, allowing the decoder to place zero-velocity boundary conditions precisely at building facades.
2. **Gradient Highway:** Skip connections provide direct backpropagation paths from the output layer to the shallow layers, mitigating vanishing gradients during training.

---

## 3. Physics-Aware Masked Loss Functions

In standard computer vision, Mean Squared Error (MSE) evaluates every pixel equally. In urban microclimate modeling, solid building interiors have undefined or zero fluid velocity. Treating building interiors as valid zero-valued pixels creates two critical pathologies:
1. **Loss Dilution:** As building packing density ($\lambda_p = A_{\text{built}} / A_{\text{total}}$) increases, unmasked loss artificially decreases simply because the network easily predicts $0.0$ inside solid walls.
2. **Gradient Waste:** The network expends capacity memorizing interior zeroes rather than optimizing complex turbulent wakes.

To solve this, our training pipeline implements **Masked Loss Formulations** evaluated strictly over the fluid domain:

```
+-----------------------------------------------------------------------------------------------+
| Let M(x, y) in {0, 1} be the building footprint mask (1 = solid building, 0 = fluid).         |
| Fluid mask: F(x, y) = 1 - M(x, y). Total fluid cells: N_fluid = sum(F).                       |
+-----------------------------------------------------------------------------------------------+
```

### 3.1 Masked L1 Loss ($\mathcal{L}_1$):
$$\mathcal{L}_{1, \text{fluid}} = \frac{1}{N_{\text{fluid}} + \epsilon} \sum_{i=1}^{N_y} \sum_{j=1}^{N_x} (1 - M_{i,j}) |Y_{i,j} - \hat{Y}_{i,j}|$$
*Advantage:* More robust to turbulent peak outliers than MSE; produces sharper wake boundaries without severe over-smoothing.

### 3.2 Masked Spatial Gradient Loss ($\mathcal{L}_{\text{grad}}$):
Pure pixel-wise losses ignore spatial derivatives ($\nabla U$). In urban flows, wind hazards are characterized by steep shear gradients ($\partial U / \partial y$ at building corners). We enforce spatial gradient consistency using discrete finite differences:

$$\nabla_x U_{i,j} = U_{i, j+1} - U_{i, j}, \quad \nabla_y U_{i,j} = U_{i+1, j} - U_{i, j}$$

$$\mathcal{L}_{\text{grad}} = \frac{1}{2} \left( \frac{\sum (1 - M_x) |\nabla_x Y - \nabla_x \hat{Y}|}{\sum (1 - M_x) + \epsilon} + \frac{\sum (1 - M_y) |\nabla_y Y - \nabla_y \hat{Y}|}{\sum (1 - M_y) + \epsilon} \right)$$

### 3.3 Composite Training Loss:
$$\mathcal{L}_{\text{composite}} = \alpha_{\text{L1}} \mathcal{L}_{1, \text{fluid}} + \alpha_{\text{grad}} \mathcal{L}_{\text{grad}}$$
*(Default hyperparameters in `src/training/losses.py`: $\alpha_{\text{L1}} = 1.0, \alpha_{\text{grad}} = 0.2$).*

---

## 4. Overfitting, Regularization, and Generalization

Because deep convolutional networks possess millions of parameters, they are susceptible to overfitting on small CFD training cohorts:

```
+-----------------------------------------------------------------------------------------------+
| OVERFITTING RISKS                          COUNTERMEASURES IMPLEMENTED                        |
+--------------------------------------------+--------------------------------------------------+
| Memorizing specific building layouts       | Strict Layout-Level Holdout (DatasetSplit)       |
| Distorting physical scales                 | Train-Only Normalization (FieldNormalizer)       |
| Weight explosion / sharp local minima      | Weight Decay (AdamW, lambda = 1e-4)              |
| Internal covariate shift                   | Batch Normalization (BatchNorm2d)                |
| Co-adaptation of feature detectors         | Spatial Dropout (Dropout2d, p = 0.10)            |
| Training past the generalization peak      | Early Stopping (patience = 6 epochs)             |
+--------------------------------------------+--------------------------------------------------+
```

---

## 5. Field Evaluation Metrics in Physical Units

All model evaluations are conducted in **physical units [$\text{m/s}$]** after applying exact inverse transformation:

$$Y_{\text{phys}} = \tilde{Y}_{\text{norm}} \cdot \sigma_{\text{train}} + \mu_{\text{train}}$$

1. **Field Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N_{\text{fluid}}} \sum_{(i,j) \in \text{fluid}} |U_{i,j} - \hat{U}_{i,j}| \quad [\text{m/s}]$$
2. **Normalized Root Mean Squared Error (nRMSE):**
   $$\text{nRMSE} = \frac{\sqrt{\frac{1}{N_{\text{fluid}}} \sum_{(i,j) \in \text{fluid}} (U_{i,j} - \hat{U}_{i,j})^2}}{U_{\text{ref}}} \times 100\% \quad [\%]$$
   *(Normalizing by approach inflow speed $U_{\text{ref}}$ ensures aerodynamic consistency across diverse wind speeds).*
3. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{(i,j) \in \text{fluid}} (U_{i,j} - \hat{U}_{i,j})^2}{\sum_{(i,j) \in \text{fluid}} (U_{i,j} - \bar{U}_{\text{fluid}})^2}$$

---

## 6. Scientific Disclaimers & Fair Comparison Principles

In accordance with strict scientific integrity guidelines:

> [!IMPORTANT]
> **Disclaimer 1 — No Claim of Superiority Over AML Baseline:**  
> The neural surrogate prototype (`src/models/unet.py`) has been verified for architectural soundness, backward gradient propagation, and software execution. However, **no claim is made that the neural model outperforms the traditional tabular AML baseline** until both models are trained and benchmarked side-by-side on identical, verified CFD ground truth data.

> [!WARNING]
> **Disclaimer 2 — Deterministic Point Predictor (No Calibrated Uncertainty):**  
> The standard U-Net is a deterministic regression model. While Monte Carlo Dropout ($p=0.10$) provides an initial heuristic for epistemic uncertainty, a deterministic neural network **does not produce calibrated physical uncertainty**. True probabilistic uncertainty quantification requires calibrated Bayesian neural networks, conformal prediction, or deep ensembles.
