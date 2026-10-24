# Experiment Design & Evaluation Plan

**Project:** Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)  
**Task:** Task 1 — Research and Experiment Design  
**Owner:** Project Lead  
**Document Version:** 1.0.0  
**Branch:** `feature/project-research`  
**Status:** Approved Reference Plan  

---

## 1. Research Questions & Formal Hypotheses

This project evaluates the predictive capability, spatial fidelity, and computational efficiency of neural surrogates compared to traditional automated machine learning (AML) baselines for urban canopy aerodynamics.

### Research Questions (RQs):

- **RQ1 (Surrogate Predictive Accuracy):** Can a deep convolutional 2D U-Net surrogate trained on multi-channel rasterized urban geometries accurately predict pedestrian-level wind speed fields ($z = 1.75\text{ m}$) on completely unseen building layouts with a normalized RMSE $< 15\%$ and field $R^2 > 0.85$?
- **RQ2 (Spatial Information Gap):** How much actionable microclimatic information is lost when relying exclusively on a traditional tabular ML baseline (Random Forest / XGBoost) predicting an area-averaged wind speed compared to a full spatial neural surrogate?
- **RQ3 (Morphological Error Correlation):** In which urban zones and under what geometric conditions does the neural surrogate fail or exhibit peak errors? Specifically, does surrogate error correlate with plan area density ($\lambda_p$), building height variance ($H_{\text{std}}$), or proximity to solid building walls?
- **RQ4 (Computational Acceleration vs. Physical Fidelity):** What wall-clock inference speedup does the neural surrogate achieve compared to steady-state RANS CFD ($k$-$\omega$ SST) and high-fidelity LES (UrbanTALES), and what is the trade-off in accuracy?

### Testable Hypotheses:

- **$H_1$ (Spatial Surrogate Superiority):** While Model 1 (Random Forest / XGBoost) will achieve competitive $R^2 > 0.80$ for bulk area-averaged wind speed, it will fail to identify localized corner-stream wind hazard zones ($U > 2.0 \times U_{\text{ref}}$) and stagnant wake zones ($U < 0.2 \times U_{\text{ref}}$), resulting in high classification error ($> 30\%$) on localized pedestrian comfort criteria. Model 2 (2D U-Net) will resolve these localized features, achieving comfort classification accuracy $> 85\%$.
- **$H_2$ (Inductive Bias of Signed Distance Fields):** Augmenting binary building footprint masks with an explicit Signed Distance Field (SDF) input channel will reduce near-wall boundary layer prediction error by at least $20\%$ compared to a model trained purely on raw binary masks.
- **$H_3$ (Out-of-Distribution Generalization Drop):** When trained exclusively on regular idealized building arrays and tested on realistic, heterogeneous city neighborhoods, the surrogate’s normalized RMSE will degrade by no more than $10 - 15\%$, demonstrating that localized bluff-body aerodynamic patterns learned from canonical arrays partially transfer to complex geometries.
- **$H_4$ (Real-Time Inference Speedup):** Forward inference of the 2D U-Net surrogate will execute in $< 50\text{ ms}$ per layout on a standard multi-core CPU, delivering a runtime speedup exceeding $1,000\times$ over steady RANS and $> 100,000\times$ over unsteady LES.

---

## 2. Experimental Protocol & Comparison Workflow

To ensure scientific rigor, Model 1 and Model 2 are trained and evaluated on **identical, strictly partitioned layout splits**:

```
+---------------------------------------------------------------------------------------------------+
|                                     DATASET: 538 URBAN LAYOUTS                                    |
|                               (Idealized Arrays & Realistic Cities)                               |
+---------------------------------------------------------------------------------------------------+
                                                  |
                    +-----------------------------+-----------------------------+
                    |                                                           |
                    v                                                           v
       [70% TRAINING SPLIT] (N = 376)                              [15% TEST SPLIT] (N = 81)
                    |                                             (Completely Held-Out Layouts)
                    |                                                           |
    +---------------+---------------+                           +---------------+---------------+
    |                               |                           |                               |
    v                               v                           v                               v
[MODEL 1: TABULAR AML]    [MODEL 2: NEURAL SURROGATE]   [MODEL 1 EVALUATION]    [MODEL 2 EVALUATION]
Feature Extractor         Layout Rasterizer             Predict Scalar y_hat    Predict Field Y_hat
11 Summary Features       7-Channel Input Tensor        Compare vs true mean    Compare vs true field
Random Forest / XGBoost   2D U-Net (MSE + Sobel)                                Spatial Error Maps
```

### Direct Comparative Protocol:
1. **Scalar Target Comparison:**
   - Model 1 directly predicts scalar $\hat{y}_{\text{AML}} = \bar{U}_{\text{canopy}}$ [$\text{m/s}$].
   - Model 2 predicts continuous 2D field $\hat{\mathbf{Y}}_{\text{Surrogate}}(x, y)$. We compute its spatially integrated canopy average:
     $$\hat{y}_{\text{Surrogate}} = \frac{1}{N_{\text{outdoor}}} \sum_{(x, y) \in \text{outdoor}} \hat{\mathbf{Y}}_{\text{Surrogate}}(x, y)$$
   - Both scalar estimates are benchmarked against the ground truth CFD canopy average $y_{\text{CFD}}$.
2. **Spatial Field Comparison:**
   - Model 2's spatial field $\hat{\mathbf{Y}}_{\text{Surrogate}}(x, y)$ is directly benchmarked pixel-by-pixel against the high-fidelity CFD reference slice $\mathbf{Y}_{\text{CFD}}(x, y)$ at $z = 1.75\text{ m}$.
   - We construct a *Spatial Baseline Benchmark* by broadcasting Model 1's scalar prediction uniformly across all outdoor fluid pixels, illustrating the exact information gain provided by spatial surrogates.

---

## 3. Quantitative Evaluation Metrics

All metrics are evaluated **strictly over outdoor fluid cells** ($\text{building\_mask} = \text{False}$). Cells inside solid building walls are masked out to prevent zero-padding from artificially deflating error values.

Let $N$ be the number of outdoor fluid cells in a layout, $U_i$ be the ground truth CFD wind speed at cell $i$, and $\hat{U}_i$ be the predicted wind speed.

### 3.1 Primary Regression Metrics:

1. **Field Mean Absolute Error (MAE) [$\text{m/s}$]:**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |U_i - \hat{U}_i|$$
2. **Root Mean Squared Error (RMSE) [$\text{m/s}$]:**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (U_i - \hat{U}_i)^2}$$
3. **Normalized Root Mean Squared Error (nRMSE) [$\%$]:**
   $$\text{nRMSE} = \frac{\text{RMSE}}{U_{\text{ref}}} \times 100\% \quad \text{or} \quad \frac{\text{RMSE}}{\max(U) - \min(U)} \times 100\%$$
   *(Using reference inflow speed $U_{\text{ref}}$ ensures physically standardized normalization across varying wind regimes).*
4. **Relative $L_2$ Error Norm:**
   $$\epsilon_{L_2} = \frac{\|\mathbf{U} - \hat{\mathbf{U}}\|_2}{\|\mathbf{U}\|_2} = \sqrt{\frac{\sum_{i=1}^N (U_i - \hat{U}_i)^2}{\sum_{i=1}^N U_i^2}}$$
5. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{i=1}^N (U_i - \hat{U}_i)^2}{\sum_{i=1}^N (U_i - \bar{U})^2}$$

### 3.2 Aerodynamic & Zonal Metrics:

1. **Spatial Absolute Error Maps:**
   $$\Delta U(x, y) = |\mathbf{Y}_{\text{CFD}}(x, y) - \hat{\mathbf{Y}}_{\text{Surrogate}}(x, y)|$$
2. **Peak Corner Jet Error ($\Delta U_{\text{corner}}$):**
   Maximum error evaluated exclusively within the corner separation streams ($U > 1.5 \times U_{\text{ref}}$).
3. **Wake Cavity Recirculation Error ($\Delta U_{\text{wake}}$):**
   Mean error evaluated within the leeward separation bubble ($U < 0.3 \times U_{\text{ref}}$).
4. **Pedestrian Comfort Criterion Accuracy:**
   Wind speeds are discretized into standard Lawson Comfort Criteria bins:
   - Sitting ($U \le 2.5\text{ m/s}$)
   - Standing / Walking ($2.5 < U \le 5.0\text{ m/s}$)
   - Strolling ($5.0 < U \le 7.5\text{ m/s}$)
   - Uncomfortable / Hazardous ($U > 7.5\text{ m/s}$)
   We compute the **Confusion Matrix**, **Classification Accuracy**, and **Cohen's Kappa ($\kappa$)** between CFD and surrogate comfort classifications.

---

## 4. Layout Hold-Out & Data Partitioning Strategy

To guarantee that models are tested on genuine spatial generalization rather than interpolating between neighboring pixels, the project implements two rigorous partitioning protocols:

```
+-----------------------------------------------------------------------------------------------+
|                            SPLIT PROTOCOL 1: STANDARD RATIO HOLDOUT                           |
|       - 70% Train (N = 376 layouts) | 15% Val (N = 81 layouts) | 15% Test (N = 81 layouts)     |
|       - Random seed fixed (seed=42) | Stratified across plan area density bins (lambda_p)     |
|       - ZERO layout ID overlap | Asserts disjoint sets in code via DatasetSplit.validate()    |
+-----------------------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------------------+
|                       SPLIT PROTOCOL 2: OUT-OF-DISTRIBUTION FAMILY HOLDOUT                    |
|       - Train & Val: 100% Idealized Arrays (Aligned & Staggered, uniform/var height, N = 224) |
|       - Test Set: 100% Realistic City Neighborhoods (Sydney, London, Tokyo, etc., N = 314)    |
|       - Purpose: Evaluates zero-shot transfer from idealized canonical blocks to real cities  |
+-----------------------------------------------------------------------------------------------+
```

---

## 5. Runtime & Computational Efficiency Evaluation

To evaluate operational speedup, we establish a standardized benchmarking suite:
- **Hardware Environments:** Benchmarked on both Apple Silicon CPU (`arm64`) and Nvidia GPU (CUDA).
- **Batch Sizes:** Evaluated for batch sizes $B = 1$ (real-time interactive CAD tool mode) and $B = 16$ (batch screening mode).

### Benchmark Comparison Matrix:

| Metric | Traditional AML Baseline (Random Forest) | Neural Surrogate (2D U-Net) | 2D Steady RANS CFD (`simpleFoam`) | 3D Unsteady LES CFD (`uDALES`) |
|---|---|---|---|---|
| **Input Generation** | $1.2\text{ ms}$ (feature extraction) | $8.5\text{ ms}$ (SDF + rasterization) | $15 - 45\text{ s}$ (`blockMesh` generation) | $2 - 5\text{ min}$ (mesh & initial field) |
| **Solve / Inference Time** | **$0.05\text{ ms}$** | **$12 - 35\text{ ms}$** | $90 - 240\text{ s}$ | $24 - 72\text{ hours}$ |
| **Total Turnaround Time** | $\approx 1.3\text{ ms}$ | $\approx 20 - 45\text{ ms}$ | $\approx 2 - 5\text{ min}$ | $\approx 1 - 3\text{ days}$ |
| **Estimated Speedup Factor** | **$\sim 100,000\times$ vs RANS** | **$\sim 5,000 - 15,000\times$ vs RANS** | $1\times$ (Baseline RANS) | — |
| **Memory Consumption** | $< 10\text{ MB}$ | $\approx 250\text{ MB}$ (VRAM) | $\approx 1 - 2\text{ GB}$ (RAM) | $\approx 16 - 64\text{ GB}$ (RAM/HPC) |

---

## 6. Uncertainty Quantification (UQ) Plan

Neural networks are deterministic point predictors by default, which can lead to overconfidence in safety-critical aerodynamic applications. We incorporate a multi-faceted uncertainty analysis:

```
                                  INPUT RASTER X
                                        |
                 +----------------------+----------------------+
                 |                      |                      |
                 v                      v                      v
          Forward Pass 1         Forward Pass 2         Forward Pass T (T=50)
         (Active Dropout)       (Active Dropout)       (Active Dropout)
                 |                      |                      |
                 +----------------------+----------------------+
                                        |
                                        v
                 +---------------------------------------------+
                 |  Predictive Mean:     mu(x, y) = E[Y_t]     |
                 |  Epistemic Variance:  sigma^2(x, y) = Var   |
                 +---------------------------------------------+
                                        |
                    +-------------------+-------------------+
                    |                                       |
                    v                                       v
        SPATIAL UNCERTAINTY MAP                 ERROR-UNCERTAINTY CALIBRATION
        sigma(x, y) across domain               Plot: |True Error| vs sigma(x, y)
        (High in wakes & corners)               Evaluate Spearman rank correlation
```

### 6.1 Epistemic Uncertainty via Monte Carlo Dropout (Gal & Ghahramani, 2016):
- Dropout layers ($p = 0.15$) are kept active at inference time.
- For each test layout, we perform $T = 50$ stochastic forward passes:
  $$\hat{\mathbf{Y}}_t = \mathcal{M}_{\theta, \mathbf{w}_t}(\mathbf{X}), \quad t = 1, \dots, T$$
- **Predictive Mean Field:**
  $$\mu(x, y) = \frac{1}{T} \sum_{t=1}^T \hat{\mathbf{Y}}_t(x, y)$$
- **Epistemic Uncertainty Field:**
  $$\sigma^2(x, y) = \frac{1}{T} \sum_{t=1}^T (\hat{\mathbf{Y}}_t(x, y) - \mu(x, y))^2$$

### 6.2 Spatial Error Correlation & Calibration:
- **Spatial Proximity Correlation:** Plot surrogate absolute error $\Delta U$ as a function of the Signed Distance Field (SDF). We test whether uncertainty spikes near solid facades where shear stresses are steepest.
- **Morphological Density Correlation:** Plot mean layout error against plan area density ($\lambda_p$). We identify whether the model experiences elevated uncertainty in the transitional skimming flow regime ($\lambda_p \approx 0.20 - 0.35$).
- **Calibration Check:** Evaluate the Spearman rank correlation between predictive standard deviation $\sigma(x, y)$ and empirical absolute error $|U_{\text{true}}(x, y) - \mu(x, y)|$. A well-calibrated surrogate will exhibit high uncertainty exactly where empirical prediction error is largest.

---

## 7. Unresolved Decisions Dependent on Dataset & CFD Feasibility

The following architectural and workflow decisions are cataloged as **blocked** until specific hardware or dataset dependencies are resolved:

| Decision Item | Nature of Dependency | Current Working Status | Resolution Trigger |
|---|---|---|---|
| **1. Ground Truth Source** | Feasibility of downloading real UrbanTALES slices vs running local RANS OpenFOAM. | Currently utilizing validated synthetic test fixtures conforming to contract. | Final team confirmation of UrbanTALES slice download availability via portal. |
| **2. Target Variable Finalization** | Whether the college course brief strictly mandates temperature. | Wind-speed magnitude ($U$ at $z=1.75\text{ m}$) established as primary target; temperature excluded. | Final confirmation from course instructors regarding thermal CFD requirements. |
| **3. Sample Volume Sizing** | Workstation disk space and training time constraints. | Pipeline tested on $N = 20 - 50$ layout prototypes; contract supports $N = 538$. | Selection of target layout subset (recommended: $80 - 120$ cases across diverse densities). |
| **4. Local OpenFOAM Execution** | macOS lacks native container runtime (Docker/OrbStack) or case-sensitive APFS volume. | Production-grade 2D OpenFOAM case and bridge script ready in `cfd/`. | Provisioning of Docker/OrbStack or access to a Linux computing server. |
| **5. Model 2 Architecture Tuning** | GPU availability for surrogate training. | Standard 2D U-Net configured; cGAN discriminator deferred to optional exploratory extension. | Hardware verification of GPU access for training runs. |

---

## 8. Pedagogical Guide & Viva Voce Q&A

### Simple Explanation of the Experiment Plan
> *To test if our AI model is actually learning physics rather than just memorizing answers, we test it on buildings it has never seen before. We test two models side-by-side: Model 1 takes a quick summary of the neighborhood (like the average building height) and guesses the average wind speed. Model 2 looks at an overhead blueprint and draws the entire wind map. We measure how close their predictions are to real wind simulations, measure how much faster the AI is than traditional supercomputers, and make the AI highlight where it feels 'uncertain' about its own predictions.*

### Viva Voce Questions & Model Answers

**Q1: Why is pixel-level random splitting forbidden in spatial microclimate modeling?**  
*Answer:* Fluid flow is governed by elliptic and hyperbolic spatial differential operators, meaning neighboring grid cells are strongly spatially autocorrelated. If pixels from the same building layout are randomly assigned across training and test sets, the neural network can simply interpolate between known adjacent pixels, causing severe data leakage. This yields artificially near-zero test errors that fail completely when deployed on a new, unseen architectural layout. Strict layout-level holdout is mandatory.

**Q2: How does Normalized RMSE (nRMSE) normalized by inflow velocity ($U_{\text{ref}}$) prevent misleading comparisons across varying wind directions or speeds?**  
*Answer:* If layout A has an inflow speed of $2\text{ m/s}$ and layout B has an inflow speed of $15\text{ m/s}$, an absolute RMSE of $1\text{ m/s}$ represents a massive $50\%$ relative error for layout A, but a negligible $6.7\%$ relative error for layout B. Normalizing RMSE by $U_{\text{ref}}$ expresses error as a dimensionless aerodynamic percentage, allowing consistent aggregation and fair benchmarking across arbitrary wind scenarios.

**Q3: What is the difference between epistemic and aleatoric uncertainty in fluid flow surrogates?**  
*Answer:* Epistemic uncertainty represents model ignorance—lack of knowledge due to sparse training data in certain geometry regimes (e.g. an unusual skyscraper shape). It can be reduced by collecting more simulation data and is quantified via Monte Carlo Dropout or Deep Ensembles. Aleatoric uncertainty represents inherent, irreducible physical randomness or noise in the system (e.g. chaotic turbulent vortex shedding and flow unsteadiness in bluff-body wakes).
