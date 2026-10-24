# Literature Review: Deep Learning & Neural Surrogates in Fluid Dynamics and Urban Microclimate

**Project:** Microclimate Modelling via Neural Surrogates and Computational Fluid Dynamics (CFD)  
**Task:** Task 1 — Research and Experiment Design  
**Owner:** Project Lead  
**Document Version:** 1.0.0  
**Branch:** `feature/project-research`  
**Status:** Approved Reference Review  

---

## 1. Executive Summary

This literature review establishes the scientific foundations for building data-driven neural surrogates of urban microclimate flow fields. We examine the mathematical and algorithmic evolution of deep learning for fluid dynamics—spanning generative adversarial frameworks, convolutional encoder-decoders, image-to-image translation, and neural operators.

We critically review the two foundational pre-reads assigned in the course brief, resolve citation specifics, analyze their direct relevance to bluff-body urban aerodynamics, and synthesize contemporary research directly addressing urban canopy turbulence.

---

## 2. In-Depth Analysis of Required Pre-Reads

### 2.1 Pre-Read 1: Goodfellow et al. (2014) — Generative Adversarial Nets

#### Exact Verified Citation:
> Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative Adversarial Nets. *Advances in Neural Information Processing Systems (NeurIPS 2014)*, 27, 2672–2680.

#### Theoretical Formulation:
Goodfellow et al. introduced a game-theoretic framework for training generative models without explicit density estimation. A **Generator** $G(\mathbf{z}; \theta_g)$ and a **Discriminator** $D(\mathbf{x}; \theta_d)$ engage in a two-player zero-sum minimax game:

$$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}(\mathbf{x})}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]$$

- $D(\mathbf{x}) \in [0, 1]$ estimates the probability that an input $\mathbf{x}$ came from the empirical data distribution $p_{\text{data}}$ rather than the generator.
- $G(\mathbf{z})$ maps random latent noise $\mathbf{z} \sim p_{\mathbf{z}}$ into the data space, seeking to maximize the probability that $D$ mistakes its output for a real sample.
- At the global theoretical optimum, $p_g = p_{\text{data}}$ and $D(\mathbf{x}) = \frac{1}{2}$. The objective minimizes the Jensen-Shannon (JS) divergence between the real and generated data distributions:
  $$V(D^*, G) = 2 \cdot D_{\text{JS}}(p_{\text{data}} \parallel p_g) - 2 \log 2$$

#### Relevance to Fluid Flow Surrogates:
Standard mean-squared error (MSE) regression losses ($\mathcal{L}_2$) assume an isotropic Gaussian distribution over targets, averaging out multi-modal possibilities. In fluid dynamics, this causes severe spatial over-smoothing—blurring sharp wake shear layers, stagnation boundaries, and thin turbulent eddies.

Goodfellow’s adversarial framework was later extended to **Conditional GANs (cGANs)** (Mirza & Osindero, 2014; Isola et al., 2017, *Pix2Pix*), where both generator and discriminator are conditioned on an input geometry $\mathbf{x}$:

$$\mathcal{L}_{\text{cGAN}}(G, D) = \mathbb{E}_{\mathbf{x}, \mathbf{y}}[\log D(\mathbf{x}, \mathbf{y})] + \mathbb{E}_{\mathbf{x}, \mathbf{z}}[\log(1 - D(\mathbf{x}, G(\mathbf{x}, \mathbf{z})))] + \lambda \mathbb{E}_{\mathbf{x}, \mathbf{y}, \mathbf{z}}[\|\mathbf{y} - G(\mathbf{x}, \mathbf{z})\|_1]$$

The discriminator acts as a **trainable high-frequency loss function**, penalizing blurred or unphysical flow patterns and forcing the generator to synthesize sharp, realistic turbulent wake structures.

#### Critical Limitations in Fluid Dynamics:
1. **Lack of Physical Conservation:** Neither $G$ nor $D$ enforces mass continuity ($\nabla \cdot \mathbf{U} = 0$) or momentum conservation. The discriminator only judges visual realism.
2. **Hallucination of Spurious Vortices:** Adversarial generators often synthesize visually convincing eddies and recirculation bubbles in locations where pressure gradients would physically prohibit them.
3. **Training Instability & Mode Collapse:** Minimax games are notoriously prone to non-convergence, vanishing gradients when $D$ overpowers $G$, and hyperparameter sensitivity.

---

### 2.2 Pre-Read 2: Farimani et al. (2017) — Deep Learning for Transport Phenomena

#### Exact Verified Citation & Bibliographic Note:
> **Verified Reference:** Farimani, A. B., Gomes, J., & Pande, V. S. (2017). "Deep Learning the Physics of Transport Phenomena". *arXiv preprint arXiv:1709.02432*.  
> *(Note: The course brief references "Farimani et al. (2017), Deep Learning for Wall-Bounded Turbulent Flows". The definitive seminal 2017 paper by Farimani, Gomes, and Pande is titled "Deep Learning the Physics of Transport Phenomena" on arXiv. Research on wall-bounded turbulent flows specifically emerged shortly thereafter via Kim & Lee (2020), Srinivasan et al. (2019), and Guastoni et al. (2021). Both literature branches are synthesized below).*

#### Theoretical Contribution:
Farimani et al. demonstrated that conditional Generative Adversarial Networks (cGANs) could learn to solve steady-state partial differential equations (PDEs) directly from spatial boundary conditions without knowing the underlying numerical discretization or equations.

They demonstrated surrogate inference on two canonical transport problems:
1. **Steady-state heat conduction:** Poisson equation $\nabla^2 T = f$ across arbitrary 2D geometries.
2. **Incompressible fluid flow:** 2D Navier-Stokes flow past solid obstacles (circular and rectangular bluff cylinders) under varying boundary velocity profiles.

The network used a U-Net-style encoder-decoder generator with skip connections, mapping a 2D spatial condition map (obstacle geometry and inlet velocity) directly to spatial velocity $(u, v)$ and pressure $p$ fields.

#### Relevance to Urban Aerodynamics:
Farimani et al. proved the feasibility of treating **CFD simulation as an image-to-image translation problem**. Urban microclimate flow around buildings is a direct multi-obstacle generalization of Farimani's cylinder benchmark:
- Building footprints correspond to internal solid boundary masks.
- Atmospheric inflow profiles correspond to domain inlet conditions.
- The U-Net encoder-decoder learns the mapping from obstacle silhouettes to wake recirculation cavities.

#### Critical Limitations:
1. **Laminar & Low Reynolds Number Flow:** The simulations in Farimani et al. were restricted to laminar regimes ($\text{Re} \le 100$). Urban canopy airflow occurs at high Reynolds numbers ($\text{Re} \sim 10^6 - 10^7$) dominated by turbulent separation and wake Reynolds stresses.
2. **Resolution Invariance:** Because standard CNNs operate on fixed pixel lattices, a model trained on a $64 \times 64$ grid cannot generalize to finer or coarser resolutions without interpolation.
3. **Boundary Layer Penetration:** Near solid wall surfaces, the model struggled to accurately resolve sharp boundary-layer velocity gradients, tending to smooth the zero-slip wall transition.

---

## 3. Review of Contemporary Urban Flow Surrogate Models

Beyond the two pre-reads, modern data-driven fluid mechanics has expanded into specialized paradigms for urban wind and microclimate prediction:

### 3.1 Convolutional Encoder-Decoders & U-Nets (Guo et al., 2016; Thuerey et al., 2020)
- **Guo et al. (2016)** pioneered using convolutional neural networks to predict steady flow past arbitrary 2D and 3D shapes, introducing signed distance fields (SDF) to parameterize geometry.
- **Thuerey et al. (2020)** demonstrated that U-Net architectures trained with combined $\mathcal{L}_1$ and spatial gradient losses could accurately predict steady RANS flow fields around airfoils with relative errors below 3%.
- *Relevance to Urban Flows:* U-Net skip connections preserve high-frequency boundary geometry across spatial scales, essential for capturing sharp building corners and narrow street canyons.

### 3.2 Deep Learning for Urban Wind Canopies (Mokhtari et al., 2022; Du et al., 2023)
- **Mokhtari et al. (2022)** implemented deep convolutional surrogates specifically for urban neighborhood microclimate modeling, demonstrating that rasterized 2D building height maps could predict pedestrian-level wind comfort maps with 90%+ classification agreement with OpenFOAM CFD.
- **Du et al. (2023)** evaluated conditional GANs versus standard U-Nets for predicting wind fields around high-rise residential building complexes. They demonstrated that while cGANs generate visually sharper wake textures, a standard U-Net optimized with a composite loss ($\mathcal{L}_1 + \text{Sobel Gradient Loss}$) achieves superior numerical convergence, lower RMSE, and zero artifact hallucinations.

### 3.3 Neural Operators: FNO and DeepONet (Li et al., 2020; Lu et al., 2021)
- **Fourier Neural Operator (FNO):** Parameterizes the integral kernel in Fourier space:
  $$v_{t+1}(x) = \sigma \left( W v_t(x) + \mathcal{F}^{-1} \left( R \cdot \mathcal{F}(v_t) \right)(x) \right)$$
  FNO learns a mapping between infinite-dimensional function spaces, making it **zero-shot super-resolving** (evaluable on any mesh resolution without retraining).
- *Limitation for Urban Geometries:* FNO relies on periodic boundary conditions and global Fourier modes; it struggles with sharp non-periodic step discontinuities and non-convex bluff boundaries characteristic of urban street canyons.

### 3.4 Physics-Informed Neural Networks (PINNs, Raissi et al., 2019)
- Enforces Navier-Stokes residual minimization directly in the loss function via automatic differentiation:
  $$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_1 \|\nabla \cdot \mathbf{U}\|^2 + \lambda_2 \|(\mathbf{U} \cdot \nabla)\mathbf{U} + \frac{1}{\rho}\nabla p - \nu \nabla^2 \mathbf{U}\|^2$$
- *Limitation for Complex Urban Canopies:* PINNs are extremely difficult to train across turbulent, multi-scale bluff geometries. Optimization often stalls in local minima without dense supervision data, making pure data-driven surrogates (or hybrid supervised models) far more robust for urban engineering.

---

## 4. Comparative Synthesis of Surrogate Methodologies

| Model Paradigm | Governing Architecture | Strengths | Weaknesses | Suitability for Course Project |
|---|---|---|---|---|
| **Traditional AML Baseline** | Random Forest / XGBoost on 11 Morphometric Features | Instant training; zero GPU requirements; high interpretability (feature importances). | Zero spatial resolution; predicts only scalar canopy-averaged quantities; misses localized wind jets. | **Mandatory Baseline (Model 1)** |
| **Pure CNN / U-Net** | Multi-Scale Convolutional Encoder-Decoder with Skip Connections | Fast, stable training; preserves exact building geometry via skip connections; low field MAE; no adversarial instability. | Can exhibit mild spatial smoothing in far wake if trained solely on $\mathcal{L}_2$ loss. | **Primary Recommended Architecture (Model 2)** |
| **Conditional GAN (Pix2Pix)** | U-Net Generator + PatchGAN Discriminator (Goodfellow / Farimani) | Extremely sharp shear layers; realistic wake vortex textures; avoids Gaussian blurring. | Prone to mode collapse; unstable minimax optimization; can hallucinate non-physical flow features. | **High-Fidelity Exploratory Candidate** |
| **Fourier Neural Operator (FNO)** | Fourier Spectral Convolution Layers | Resolution-independent; fast global operator learning. | Struggles with sharp non-periodic building step-changes; high memory footprint. | Advanced extension candidate |
| **Physics-Informed NN (PINN)** | MLP with Automatic Differentiation Residual Penalties | Enforces physical conservation laws without requiring CFD training data. | Intractable on complex 3D urban geometries; fails to capture high Reynolds turbulence. | Not recommended for urban canopy scale |

---

## 5. Pedagogical Guide & Viva Voce Q&A

### Simple Explanation of Pre-Read Concepts
> *Imagine a police sketch artist and a detective. In Goodfellow's GAN, the Generator is an artist trying to draw realistic pictures of wind flow, while the Discriminator is a detective trying to spot whether the drawing is fake or from a real wind tunnel. By competing against each other, the artist gets extraordinarily good at drawing fine details like swirling eddies. Farimani showed that instead of drawing random pictures, we can show the artist a blueprint of a building and ask: 'Draw the exact wind speed pattern around this specific building'.*

### Viva Voce Questions & Model Answers

**Q1: In Farimani et al. (2017), what was the fundamental justification for using conditional GANs rather than standard mean-squared error (MSE) regression?**  
*Answer:* Minimizing MSE ($\mathcal{L}_2$ loss) forces the network to predict the conditional expectation $\mathbb{E}[\mathbf{y}|\mathbf{x}]$, which averages all plausible high-frequency solutions. In turbulent and separated flows, this causes severe spatial blurring of sharp velocity gradients, shear layers, and wake boundaries. The adversarial discriminator in a cGAN penalizes high-frequency blurriness, forcing the generator to produce sharp, physically structured flow fields.

**Q2: What is the primary difference between how a U-Net and a Fourier Neural Operator (FNO) represent urban geometry?**  
*Answer:* A U-Net processes geometry as discrete pixel rasters using localized convolutional kernels, using multi-scale pooling and skip connections to capture local wall boundaries. An FNO represents flow fields as continuous functions in Fourier frequency space, parameterizing the kernel globally. While FNO allows evaluation on arbitrary mesh resolutions, convolutional U-Nets remain superior for urban layouts because localized convolutions handle sharp, discontinuous step-changes in building heights much better than global Fourier harmonics.

**Q3: Goodfellow’s GANs were designed for image generation. What physical risk arises when applying them directly to fluid flow fields?**  
*Answer:* Standard GAN discriminators evaluate statistical and visual realism, not physical conservation. Consequently, a GAN generator can produce visually convincing flow patterns that violate fundamental conservation laws—such as creating non-zero velocity divergence ($\nabla \cdot \mathbf{U} \neq 0$), generating mass out of nowhere, or producing reverse flow without an adverse pressure gradient. Addressing this requires incorporating physical loss penalties or mass-conserving projection layers.
