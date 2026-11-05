# Radial Basis Function Networks and Kolmogorov-Arnold Networks: A Technical Guide

**Scope:** classical RBF networks, their deep and modern variants, and their relationship to Kolmogorov-Arnold Networks (KANs). Literature coverage runs through mid-2026.

## Contents

1. [Introduction](#1-introduction)
2. [Classical RBF Networks](#2-classical-rbf-networks)
3. [Why RBF Networks Fell Out of Mainstream Use](#3-why-rbf-networks-fell-out-of-mainstream-use)
4. [Kolmogorov-Arnold Networks](#4-kolmogorov-arnold-networks)
5. [The Gaussian RBF View of KANs](#5-the-gaussian-rbf-view-of-kans)
6. [KANs at Scale](#6-kans-at-scale)
7. [Deep RBF Networks](#7-deep-rbf-networks)
8. [RBF Layers on Deep Backbones](#8-rbf-layers-on-deep-backbones)
9. [Mathematical Foundations](#9-mathematical-foundations)
10. [Implementation](#10-implementation)
11. [State of the Evidence](#11-state-of-the-evidence)
12. [Applications](#12-applications)
13. [Open Problems](#13-open-problems)
14. [Practical Recommendations](#14-practical-recommendations)
15. [References](#15-references)
16. [Notation](#16-notation)

---

## 1. Introduction

Radial Basis Function (RBF) networks compute outputs from the distances between an input and a set of centers. They were a standard tool in the late 1980s and 1990s for interpolation, regression and classification. Over the following two decades they were largely displaced by deep MLPs, CNNs and later transformers.

Interest returned from 2024 onward, through four lines of work:

- **KANs and their RBF form.** KANs (Liu et al., 2024) proposed learnable univariate functions on network edges. FastKAN (Li, 2024) showed that the B-spline bases in KANs can be replaced by Gaussian RBFs with similar accuracy and faster evaluation. Since then, RBF-based KANs with trainable centers and shape parameters (Free-RBF-KAN, adaptive RBF-KAN, hierarchical RBF-KAN; 2026) have closed most of the accuracy gap to B-spline KANs while keeping the speed advantage.
- **Theory.** KANs are at least as expressive as MLPs of comparable size. They also show a reduced spectral bias, which helps on high-frequency scientific targets and makes them more prone to overfitting noise (Wang et al., ICLR 2025).
- **Scale.** Naive KAN substitution into vision transformers trains poorly at ImageNet scale. The Kolmogorov-Arnold Transformer (KAT; ICLR 2025) makes it work with rational bases, group-shared functions and variance-preserving initialization. Language-model evidence remains small-scale.
- **Distance-based layers outside KANs.** Deep RBF networks (Wurzberger & Schwenker, 2024) train stably with data-driven initialization. RBF output layers on CNN backbones (Amirian & Schwenker, 2020) add prototype-based interpretability. Gaussian kernel attention (2026) removes query/key/value projections entirely.

Evidence remains strongest on scientific function fitting, PDEs and small or structured problems. General-purpose benchmarks at matched compute mostly favor MLPs (Sections 11 and 14).

### Timeline

| Date | Development | Significance |
|---|---|---|
| 1988 | Broomhead & Lowe | RBF interpolation framed as a neural network |
| 1989 | Moody & Darken | Hybrid training: clustering for centers, least squares for weights |
| 1991 | Park & Sandberg | Universal approximation for RBF networks |
| 2020 | RBF classifiers on CNNs (Amirian & Schwenker) | End-to-end training with modern backbones |
| Apr 2024 | KAN (Liu et al.) | Learnable spline activations on edges |
| Apr 2024 | Deep RBF networks (Wurzberger & Schwenker) | Multi-layer RBF with Mahalanobis distance and partial connectivity |
| May 2024 | FastKAN (Li) | Gaussian RBF bases replace B-splines in KANs |
| Jun 2024 | KAGNNs (Bresson et al.) | KAN layers (B-spline and RBF) inside graph neural networks |
| Jun 2024 | BSRBF-KAN (Ta) | Additive combination of B-spline and Gaussian RBF bases |
| Jul 2024 | Controlled KAN vs MLP comparison (Yu et al.) | MLPs outperform KANs at matched parameters/FLOPs on most non-symbolic tasks |
| ICLR 2025 | Kolmogorov-Arnold Transformer (Yang & Wang) | Group-rational KAN layers scale ViT-style models to ImageNet |
| ICLR 2025 | Expressiveness and spectral bias of KANs (Wang et al.) | KANs at least as expressive as MLPs; reduced spectral bias |
| 2025 | Free-knot and preconditioning analyses (Zheng et al.; Actor et al.) | Learned knot placement; spline KANs as preconditioned multichannel MLPs |
| 2025-2026 | Critical and practitioner surveys (Ji et al.; Noorizadegan et al.) | Consolidated evidence; basis function as the central design axis |
| Jan 2026 | Free-RBF-KAN (Chiu et al.) | Trainable RBF centers and shape parameters in KANs |
| Feb 2026 | Unified PINN vs PIKAN benchmark (Dzimah et al.) | KAN-based PINNs more accurate at matched (small) parameter budgets |
| May 2026 | Adaptive RBF-KAN (Cavoretto et al.) | LOOCV-initialized shape parameters; Matérn and Wendland kernels |
| May 2026 | Gaussian Kernel Attention (Kundu et al.) | Projection-free attention from Gaussian RBF affinities |
| Jun 2026 | Hierarchical RBF-KAN / RBF-SKAN | Hierarchical RBF-KANs for multivariate functions and random fields |
| Jul 2026 | KANs for small language models (Alves & Vicente) | KAN feed-forward blocks benchmark-equivalent to MLPs at 10M parameters |

---

## 2. Classical RBF Networks

### 2.1 Architecture

```
Input (D) -> Hidden RBF layer (N units) -> Linear output (K)
```

- **Input layer:** passes the input through unchanged.
- **Hidden layer:** unit $i$ responds according to the distance between the input and its center $\mathbf{c}_i$.
- **Output layer:** forms a linear combination of the hidden activations.

### 2.2 Gaussian kernel and output

$$
\phi_i(\mathbf{x}) = \exp\left(-\frac{\|\mathbf{x}-\mathbf{c}_i\|^2}{2\sigma_i^2}\right),
\qquad
y_k(\mathbf{x}) = \sum_{i=1}^{N} w_{ik}\,\phi_i(\mathbf{x}) + b_k
$$

### 2.3 Hybrid training

The classical procedure trains the network in three stages.

```
Stage 1: Centers (unsupervised)
  Option A: random subset of N training inputs
  Option B: k-means with k = N; use the centroids
  Option C: every training point as a center (exact interpolation, N = M)

Stage 2: Widths
  Shared width heuristic (Haykin):  sigma = d_max / sqrt(2N)
    d_max = maximum distance between chosen centers
  Per-center heuristic: sigma_i = mean distance from c_i to its p nearest centers
  Data-driven: leave-one-out cross-validation over a shared shape parameter (Rippa's method)

Stage 3: Output weights (supervised, closed form)
  Phi[m, i] = phi_i(x_m)            (M x N design matrix)
  W = (Phi^T Phi + lambda I)^(-1) Phi^T Y
```

The ridge term $\lambda$ matters in practice. The design matrix $\Phi$ is frequently ill-conditioned, especially with wide kernels or nearby centers. Because the output layer is linear, the weights have a closed-form solution once the centers and widths are fixed. Gradient descent can then refine all parameters jointly.

### 2.4 Universal approximation

Park and Sandberg (1991) proved that single-hidden-layer RBF networks are universal approximators under mild conditions on the kernel.

This is an existence result. It does not say how many units are needed, and that number can grow exponentially with input dimension for local kernels. Depth can therefore still be valuable for efficiency, exactly as with MLPs.

### 2.5 Relation to kernel methods

There are two direct links to kernel methods:

- If every training point is used as a center ($N = M$) and the output weights are fitted with ridge regression, the network is identical to kernel ridge regression with a Gaussian kernel.
- For strictly positive definite kernels such as the Gaussian, the interpolation matrix is nonsingular for distinct points (Micchelli, 1986).

RBF networks with $N \ll M$ centers can therefore be viewed as sparse, learnable-center approximations to kernel machines.

---

## 3. Why RBF Networks Fell Out of Mainstream Use

### 3.1 Scaling with dimension and data

- **Exact interpolation scales with data.** It places one center per training sample, which is $O(MD)$ memory and $O(M^3)$ to solve.
- **Local kernels suffer from the curse of dimensionality.** Covering a $D$-dimensional region at fixed resolution requires a number of centers exponential in $D$.
- **Euclidean distances concentrate in high dimensions.** This makes isotropic Gaussian responses less discriminative unless the input is first mapped to a lower-dimensional feature space.

### 3.2 Training difficulty in deep stacks

- **Gradients are local.** Each Gaussian's gradient is proportional to its activation, which is near zero far from the center. Poorly initialized centers therefore receive almost no learning signal.
- **The problem compounds with depth.** Stacking distance-based layers multiplies this effect and makes networks sensitive to initialization and width settings.
- **Early attempts struggled.** Multi-layer RBF networks were widely reported as unstable.

### 3.3 Ecosystem

- **Easier alternatives.** ReLU MLPs, CNNs and later transformers trained reliably at depth and scaled with data and compute.
- **Tooling and hardware.** Both co-evolved around dense matrix multiplication and those architectures.

Euclidean distance computation is not itself hardware-unfriendly. It reduces to a matrix multiplication (Section 10.4). The larger overheads in modern RBF-style layers come from per-edge basis expansion (KANs) and from full covariance parameterizations (Mahalanobis RBFs).

---

## 4. Kolmogorov-Arnold Networks

### 4.1 The representation theorem

Kolmogorov (1957) showed that every continuous $f:[0,1]^n \to \mathbb{R}$ can be written as:

$$
f(\mathbf{x}) = \sum_{q=0}^{2n} \Phi_q\!\left(\sum_{p=1}^{n} \phi_{q,p}(x_p)\right)
$$

with continuous univariate $\Phi_q$ and $\phi_{q,p}$.

The theorem has an important caveat. The inner functions guaranteed by the theorem can be highly non-smooth and impractical to learn (Girosi & Poggio, 1989). KAN design is inspired by the theorem rather than dictated by it. KANs do not rely on the exact two-layer construction. Instead, they stack multiple layers of smooth learnable univariate functions.

### 4.2 MLP vs KAN layers

| | MLP | KAN |
|---|---|---|
| Learnable object | Scalar weight per edge | Univariate function per edge |
| Nonlinearity | Fixed, on nodes | Learnable, on edges |
| Node operation | $\sigma(W\mathbf{x} + \mathbf{b})$ | Sum of incoming edge outputs |

A KAN layer maps $\mathbf{x}^{(l)} \in \mathbb{R}^{d_{in}}$ to $\mathbf{x}^{(l+1)} \in \mathbb{R}^{d_{out}}$:

$$
x^{(l+1)}_j = \sum_{i=1}^{d_{in}} \varphi_{l,j,i}\!\left(x^{(l)}_i\right)
$$

### 4.3 Edge function in the original KAN

$$
\varphi(x) = w_b\,\mathrm{silu}(x) + w_s \sum_{k} c_k B_k(x)
$$

where $B_k$ are B-spline basis functions (typically cubic) on a grid of $G$ intervals. Each edge carries $G + k$ spline coefficients plus the scalars $w_b, w_s$. A KAN layer therefore has roughly $(G+k)$ times the parameters of an MLP layer of the same width.

The original paper also introduced grid extension (refining the spline grid during training), sparsity regularization, pruning, and symbolification (snapping learned edge functions to symbolic forms). KAN 2.0 (Liu et al., 2024) extended this toolkit for scientific discovery, including multiplication nodes and ways to embed prior knowledge into the network structure.

### 4.4 Reported strengths and costs

**Strengths reported by Liu et al.:**

- Accurate fits on synthetic functions with smooth compositional structure.
- A theoretical error scaling of $\ell \propto N^{-(k+1)}$, i.e. $N^{-4}$ for cubic splines, under the assumption that the target admits a smooth KAN representation. This is observed empirically on such synthetic targets.
- Interpretable edge functions that support symbolic regression on scientific problems.

**Costs reported by the authors:**

- Training is about 10x slower than an MLP with the same number of parameters. The cause is per-edge basis evaluation that maps poorly to dense matrix multiplication.

### 4.5 Controlled comparisons

Yu, Yu and Wang (2024) compared KANs and MLPs at matched parameter counts and FLOPs across machine learning, vision, NLP, audio and symbolic formula tasks.

- **Results:** MLPs outperformed KANs on all task families except symbolic formula representation.
- **Explanation:** the KAN advantage there was largely attributable to the B-spline activation itself. MLPs given B-spline activations closed most of the gap.

Later critical surveys (Ji et al., 2025 revision) reach the same conclusion across the broader literature. KANs underperform MLPs on standard machine learning, vision and NLP benchmarks at matched compute, and their claimed advantages stem mostly from the basis functions rather than from the edge-function architecture.

The parameter-efficiency advantage of KANs is therefore task-specific. It is strongest for smooth, low-dimensional, compositional or high-frequency scientific targets, and it does not hold as a general property.

### 4.6 Theory

**Expressiveness (Wang et al., ICLR 2025):**

- **KANs can represent MLPs.** Any MLP with ReLU$^k$ activations can be reparameterized as a KAN of comparable size, so KANs are at least as expressive as MLPs.
- **MLPs can represent KANs, at a cost.** Representing a KAN as an MLP increases the parameter count by a factor of the grid size $G$. KANs with large grids may therefore be more efficient than MLPs for certain functions.
- **Rates.** Combining these reductions with known MLP results gives approximation rates for KANs on Sobolev spaces.

**Spectral bias (Wang et al., ICLR 2025):**

- **MLPs learn low frequencies first.** This acts as a useful regularizer for typical machine learning data.
- **KANs suffer less from this bias.** A single-layer analysis and experiments on 1D waves, Gaussian random fields and high-frequency Poisson problems show KANs fitting many frequencies simultaneously. Grid extension improves high-frequency learning further.
- **Overfitting is the flip side.** Reduced spectral bias explains why KANs are more sensitive to noise and overfitting. Grid coarsening restores some bias and reduces overfitting.
- **Grid bounds frequency.** The highest frequency a single KAN layer can represent is limited by its number of grid points.

**Optimization:**

- **Preconditioning view.** Actor et al. (2025) show that spline-based KANs act as preconditioners for multichannel MLPs. This yields better-conditioned optimization landscapes and faster convergence.
- **Free knots.** Learning the knot or center positions (Zheng et al., 2025) improves accuracy and stability over fixed uniform grids.

**Practical reading:** the KAN-vs-MLP question is largely a question of inductive bias. KANs favor localized, high-frequency, low-dimensional structure. MLPs favor smooth, low-frequency structure in high dimensions.

---

## 5. The Gaussian RBF View of KANs

### 5.1 Core observation

Li (2024) observed that cubic B-spline bases on a uniform grid are closely approximated by Gaussians:

$$
B_k(u) \approx \exp\!\left(-\left(\frac{u - u_k}{h}\right)^2\right)
$$

with $u_k$ the grid points and $h$ proportional to the grid spacing. Replacing the B-splines with Gaussians gives **FastKAN**.

FastKAN matches the accuracy of B-spline KANs on MNIST-scale benchmarks. Its reference implementation reports a forward pass about 3.3x faster than the efficient-kan B-spline implementation. The speedup comes from evaluating Gaussians directly instead of using the recursive Cox-de Boor B-spline evaluation.

### 5.2 FastKAN layer

```
Input: X (B x D_in)
Parameters:
  grid g_1..g_G   (fixed, uniform over a bounded interval)
  h               (width, tied to grid spacing)
  W  (D_out x D_in x G)   basis coefficients
  V  (D_out x D_in)       optional base branch weights
  b  (D_out)

1. Xn = LayerNorm(X)                          # keeps inputs inside grid range
2. R[n, i, k] = exp(-((Xn[n, i] - g_k) / h)^2)  # B x D_in x G
3. Y[n, j] = sum_{i,k} W[j, i, k] * R[n, i, k]
           + sum_i V[j, i] * silu(X[n, i]) + b[j]
```

Step 3 is a single matrix multiplication after reshaping $R$ to $B \times (D_{in} G)$.

### 5.3 What "KANs are RBF networks" means precisely

A FastKAN layer is an RBF network with specific restrictions:

- **Univariate bases.** Each basis function depends on a single input coordinate, not on the full input vector.
- **Fixed centers.** Centers sit on a fixed shared grid and are not learned.
- **Learned weights only.** Only the output coefficients are learned, and widths are typically fixed.

Each layer is therefore an additive model of one-dimensional Gaussian expansions. Classical multivariate RBF networks, with learned centers in $\mathbb{R}^D$ and isotropic or Mahalanobis widths, are a different and more expressive per-unit construction. The two families share the Gaussian kernel and the linear readout over basis activations, but they are not interchangeable.

The univariate structure is also what makes RBF-KANs tractable in high dimensions. Classical RBF networks need a number of multivariate centers exponential in $D$ for uniform coverage. RBF-KANs place one-dimensional grids per coordinate and build multivariate structure through superposition and depth.

### 5.4 Adaptive RBF-KANs

Fixed-grid, fixed-width RBF-KANs are fast but can lose accuracy relative to B-spline KANs. Work in 2026 restores classical RBF degrees of freedom inside the KAN structure.

**Free-RBF-KAN (Chiu et al., 2026):**

- **Trainable centers.** Grid points move during training (adaptive meshing) while staying within a fixed domain.
- **Trainable smoothness.** Per-basis shape parameters adapt the sharpness of each Gaussian.
- **Results.** Accuracy comparable to the original B-spline KAN with faster training and inference. Tasks include multiscale function approximation, physics-informed learning and PDE operator learning.

**Adaptive RBF-KAN (Cavoretto et al., 2026):**

- **LOOCV initialization.** The kernel shape parameter is initialized by leave-one-out cross-validation, a classical RBF interpolation technique, and then refined by training.
- **Broader kernel family.** Matérn and Wendland kernels join the Gaussian.
- **Finding.** Kernel choice matters by target type: different kernels suit smooth functions, discontinuities and oscillatory patterns.

**Hierarchical RBF-KAN and RBF-SKAN (2026):**

- **Hierarchical RBF-KAN** stacks RBF-KAN levels with trainable Gaussian centers and scales for multivariate deterministic function approximation.
- **RBF-SKAN** is a stochastic counterpart for learning random field models.

### 5.5 Basis-function variants

| Variant | Basis | Notes |
|---|---|---|
| KAN (original) | Cubic B-splines | Grid extension, symbolification; slowest |
| EfficientKAN | Cubic B-splines | Reformulated computation; lower memory |
| FreeKnots-KAN | B-splines with learned knots | Better accuracy and stability than fixed grids |
| FastKAN | Gaussian RBF on fixed grid | Similar accuracy to B-splines; faster evaluation |
| FasterKAN | Reflectional switch activation (RSWAF) | Approximates Gaussians without exponentials; further speedup |
| Free-RBF-KAN | Gaussian RBF, trainable centers and shapes | Matches B-spline KAN accuracy; faster |
| Adaptive RBF-KAN | Gaussian, Matérn, Wendland; LOOCV-initialized shape | Kernel choice tuned to target regularity |
| BSRBF-KAN | B-spline + Gaussian RBF, summed | Reported 97.55% (MNIST) and 89.33% (Fashion-MNIST), averaged over 5 runs |
| ChebyKAN / Jacobi KANs | Orthogonal polynomials | Global basis; requires input normalization (e.g. tanh); common in PIKANs |
| FourierKAN | Fourier series | Suited to periodic structure |
| GR-KAN (KAT) | Rational functions, shared across channel groups | GPU-friendly; scales to ImageNet (Section 6) |

In the BSRBF-KAN ablations, a variant with both bases removed (effectively an MLP) performed comparably on MNIST and worse on Fashion-MNIST. MNIST alone does not discriminate between these designs.

The basis function is the main design axis. Practitioner reviews (Noorizadegan et al., 2026) organize choices by target properties:

- **Local bases** (B-splines, RBFs): localized and multiscale structure.
- **Global polynomials** (Chebyshev, Jacobi): smooth targets on bounded domains.
- **Fourier bases:** periodic structure.
- **Rational bases:** GPU efficiency at scale.

---

## 6. KANs at Scale

### 6.1 Kolmogorov-Arnold Transformer

Replacing the MLP blocks of a ViT with standard KAN layers ("ViT+KAN") trains poorly at ImageNet scale. Yang and Wang (ICLR 2025) identify three causes and a fix for each:

| Problem | Fix |
|---|---|
| B-spline evaluation is slow on GPUs | Rational-function bases with fused CUDA/Triton kernels |
| One function per edge makes compute and parameters explode | Group KAN: share function coefficients across groups of channels |
| Learnable activations destabilize standard initialization | Variance-preserving initialization |

The resulting **Group-Rational KAN (GR-KAN)** replaces the transformer MLP. The resulting **KAT** models outperform ViT and DeiT baselines of matching size at similar compute on ImageNet classification, with gains carried over to detection and segmentation. KAT can also be initialized from pretrained ViT weights.

**Interpretation:** with functions shared per channel group, a GR-KAN layer is close to a learnable group-wise activation followed by a linear map. Its success at scale shows that learnable activations can pay off once their cost matches dense matmul. It does not show that unshared per-edge functions scale.

### 6.2 Language models

- **Small scale only.** Published evaluations are at tens of millions of parameters or less, often on BabyLM-style data.
- **Benchmark-equivalent.** At 10M parameters on BabyLM Strict-Small, with ten seeds, KAN-family and gated feed-forward blocks were benchmark-equivalent to the MLP on BLiMP and EWoK (Alves & Vicente, 2026). The only robust effect was a sub-one-point GR-KAN edge on the main benchmark, which reversed on the supplementary set.
- **Interpretability holds.** In the same study, nearly all reconstructed feed-forward edge functions were active and many were nonlinear, which supports using KANs to inspect learned scalar transformations.
- **Throughput.** Implementations that replace every linear layer with per-edge KANs scale poorly in memory and time.

No evidence yet shows KAN feed-forward blocks improving language models at production scale.

---

## 7. Deep RBF Networks

Wurzberger and Schwenker (2024) build multi-layer networks of RBF units with a linear output layer. They report results on image classification and speech emotion recognition comparable to simple CNNs. Three components make training stable.

### 7.1 Layer-wise data-driven initialization

```
For each RBF layer l = 1..L:
  1. Propagate training data through initialized layers 1..l-1
  2. Run k-means on layer (l-1) outputs; centroids -> initial centers C_l
  3. For each center c_i: estimate covariance Sigma_i from its cluster members
     (or its p nearest samples), add a small ridge (Sigma_i + eps I)
  4. Factor Sigma_i^(-1) = L_i L_i^T (Cholesky) to initialize the distance parameters
  5. Isotropic fallback: sigma_i = sqrt(trace(Sigma_i) / D)
```

Initializing each layer from the distribution of its actual inputs keeps every unit inside a region where it receives gradient signal.

### 7.2 Mahalanobis distance

$$
d_M(\mathbf{x}, \mathbf{c}_i) = \sqrt{(\mathbf{x}-\mathbf{c}_i)^\top \Sigma_i^{-1} (\mathbf{x}-\mathbf{c}_i)} = \left\|L_i^\top(\mathbf{x}-\mathbf{c}_i)\right\|_2,
\qquad \Sigma_i^{-1} = L_i L_i^\top
$$

Learning a lower-triangular $L_i$ guarantees a valid metric, provided its diagonal stays strictly positive. Two parameterizations achieve this:

- **Exponential:** $L_{i,jj} = \exp(\theta_{jj})$
- **Softplus:** $L_{i,jj} = \mathrm{softplus}(\theta_{jj}) + \epsilon$

The off-diagonal entries are unconstrained.

**Benefits:**

- Captures feature correlations.
- Gives anisotropic, adaptive receptive fields.
- Makes the response invariant to invertible linear reparameterizations of the input, provided the metric is learned or re-estimated in the new coordinates.

**Cost:** a full $L_i$ has $D(D+1)/2$ parameters per unit. At $D = 784$ this is 307,720 per unit, so full covariance is impractical on raw high-dimensional inputs. The standard ways around this are:

- **Diagonal metric:** $D$ parameters per unit.
- **Low-rank metric:** $L_i \in \mathbb{R}^{D \times r}$.
- **Partial connectivity:** restrict each unit to a small input patch (Section 7.3).

### 7.3 Partially connected (convolution-like) RBF layers

```
Input feature map: H x W x C
Patch size K x K, stride S

For each output position (u, v):
  p = flatten(input[uS : uS+K, vS : vS+K, :])      # dimension P = K*K*C
  For each center n = 1..N (shared across positions):
    out[u, v, n] = exp(-d_M(p, c_n)^2 / 2)

Output: H' x W' x N,  H' = floor((H-K)/S) + 1,  W' = floor((W-K)/S) + 1
```

**Parameters per center:**

| Metric | Fully connected unit | Patch unit |
|---|---|---|
| Center | $HWC$ | $P = K^2 C$ |
| Diagonal metric | $HWC$ | $P$ |
| Full metric | $\tfrac{1}{2}(HWC)^2$ | $\tfrac{1}{2}P^2$ |

**Compute:**

- Diagonal metric: $O(H'W' N P)$
- Full metric: $O(H'W' N P^2)$

Weight sharing across positions also gives translation equivariance, as in CNNs.

### 7.4 Training loop

```
Initialize centers, Cholesky factors and output weights (Section 7.1)
Optionally solve output weights in closed form (ridge) before joint training

For each mini-batch:
  Forward: distances -> RBF activations -> ... -> linear output
  Backward: autodiff through distances, centers, metric factors
  Update with Adam; use a lower learning rate for centers and metrics
  Re-apply constraints (lower-triangular mask; positive diagonal via parameterization)

Schedule: linear warmup, then cosine decay
```

---

## 8. RBF Layers on Deep Backbones

### 8.1 RBF classifier heads

Amirian and Schwenker (2020) replace the softmax head of a CNN with an RBF layer trained end-to-end:

```
Image -> CNN backbone -> feature f in R^D -> RBF layer (centers = learned prototypes) -> class scores
```

Their reported results:

- Accuracy comparable to standard MLP or softmax heads on benchmark datasets.
- A learned similarity metric over the feature space.
- Interpretability through nearest prototypes and training examples.

### 8.2 Related distance-based heads

- **Prototypical Networks** (Snell et al., 2017): classify by distance to class-mean embeddings. Few-shot classes are added by computing a new prototype, with no retraining of the backbone.
- **DUQ** (van Amersfoort et al., 2020): an RBF output layer with per-class centroids, used for out-of-distribution detection.

DUQ requires a gradient penalty on the backbone. Without it, feature collapse can map out-of-distribution inputs close to class centroids. An RBF head alone does not guarantee reliable uncertainty estimates.

---

## 9. Mathematical Foundations

### 9.1 Distance-based vs projection-based units

| Unit | Response | Geometry |
|---|---|---|
| MLP neuron | $\sigma(\mathbf{w}^\top\mathbf{x} + b)$ | Constant along hyperplanes; global |
| RBF neuron | $\phi(\|\mathbf{x}-\mathbf{c}\|)$ | Constant on spheres (Euclidean) or ellipsoids (Mahalanobis) around $\mathbf{c}$; local for decaying $\phi$ |

An isotropic RBF unit's response is radially symmetric about its center. It is invariant to rotations of the input about $\mathbf{c}$. This is a property of the unit, not a rotation invariance of the overall network.

### 9.2 Common radial functions

With $r = \|\mathbf{x} - \mathbf{c}\|$:

| Name | $\phi(r)$ | Properties |
|---|---|---|
| Gaussian | $\exp(-r^2/(2\sigma^2))$ | Smooth, strictly positive definite, local |
| Multiquadric | $\sqrt{1 + (r/\sigma)^2}$ | Unbounded, global; strong for scattered-data interpolation |
| Inverse multiquadric | $1/\sqrt{1 + (r/\sigma)^2}$ | Bounded, slow polynomial decay |
| Matérn ($\nu = 3/2$) | $(1 + \sqrt{3}\,r/\ell)\exp(-\sqrt{3}\,r/\ell)$ | Finite smoothness controlled by $\nu$; better for rough targets than Gaussian |
| Thin plate spline | $r^2 \log r$ | Minimizes bending energy in 2D; conditionally positive definite (needs a polynomial term) |
| Wendland $C^2$ | $(1 - r)_+^4 (4r + 1)$ | Compact support (zero for $r \ge 1$, with $r$ scaled); sparse matrices; positive definite for $D \le 3$ |

### 9.3 Normalized RBF networks

$$
u_i(\mathbf{x}) = \frac{\phi_i(\mathbf{x})}{\sum_{j=1}^{N} \phi_j(\mathbf{x})}
$$

The normalized activations sum to 1, forming a partition of unity. They behave as soft cluster assignments, closely related to the responsibilities in a Gaussian mixture model.

### 9.4 Relation to attention

For Gaussian kernels with a shared temperature $\tau$, normalized RBF weights are a softmax over negative squared distances:

$$
u_i = \mathrm{softmax}_i\!\left(-\frac{\|\mathbf{q} - \mathbf{k}_i\|^2}{\tau}\right)
$$

Expanding the square:

$$
-\frac{\|\mathbf{q} - \mathbf{k}_i\|^2}{\tau} = \frac{2\,\mathbf{q}^\top\mathbf{k}_i}{\tau} - \frac{\|\mathbf{k}_i\|^2}{\tau} - \frac{\|\mathbf{q}\|^2}{\tau}
$$

The last term is constant across $i$ and cancels in the softmax. Gaussian attention is therefore dot-product attention with a per-key bias of $-\|\mathbf{k}_i\|^2/\tau$. With normalized keys the two are identical up to temperature. The practical distinction is that attention learns the query/key projections, while a classical RBF layer measures distance in a fixed or separately learned metric.

**Gaussian Kernel Attention (Kundu et al., 2026)** takes the RBF side of this equivalence literally:

- **No projections.** It removes the query, key and value projections. Token affinities come from a Gaussian kernel on per-head token features, with a learned bandwidth $\sigma_h$ per head and a single output projection.
- **Kernel smoothing.** Row normalization makes each head a normalized kernel regression over tokens, i.e. classical non-local filtering.
- **Results.** Reported vision and small-scale language-modeling results are competitive with roughly half the parameters and FLOPs of the attention baseline. At the tested language-model scale there is a modest loss penalty.

### 9.5 Gradients of the Gaussian RBF

With $d_i = \|\mathbf{x} - \mathbf{c}_i\|$:

$$
\frac{\partial \phi_i}{\partial \mathbf{x}} = -\frac{\mathbf{x} - \mathbf{c}_i}{\sigma_i^2}\,\phi_i,
\qquad
\frac{\partial \phi_i}{\partial \mathbf{c}_i} = \frac{\mathbf{x} - \mathbf{c}_i}{\sigma_i^2}\,\phi_i,
\qquad
\frac{\partial \phi_i}{\partial \sigma_i} = \frac{d_i^2}{\sigma_i^3}\,\phi_i
$$

With the log-width parameterization $\sigma_i = e^{s_i}$:

$$
\frac{\partial \phi_i}{\partial s_i} = \frac{d_i^2}{\sigma_i^2}\,\phi_i
$$

Every gradient carries the factor $\phi_i$, so parameters of units far from the data receive almost no update. This locality produces stable local updates, and it also makes "dead" units likely after poor initialization.

### 9.6 Gradient flow in deep RBF networks

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{c}^{(l)}_i} = \sum_j \frac{\partial \mathcal{L}}{\partial h^{(l)}_j}\,\frac{\partial h^{(l)}_j}{\partial \mathbf{c}^{(l)}_i}
$$

The local derivative contains $\phi_i$, and backpropagating through earlier RBF layers multiplies further such factors. Mitigations:

1. Data-driven initialization of each layer (k-means plus covariance estimation).
2. Normalization between layers so that inputs stay in the region covered by centers.
3. Lower learning rates for centers and metric parameters than for output weights.
4. Residual connections around RBF blocks.
5. Width floors (a minimum $\sigma$) to prevent collapse to near-delta responses.
6. Variance-preserving initialization of learnable activations (as in KAT) when RBF or basis-expansion layers sit inside deep residual stacks.

---

## 10. Implementation

### 10.1 Number of centers

- **Too few:** underfitting. **Too many:** overfitting, ill-conditioning and cost.
- **Shallow networks:** start from a modest $N$ (for example on the order of $\sqrt{M}$ for $M$ samples) and select by cross-validation. Treat this as a starting heuristic, not a rule.
- **KAN-style layers:** 5 to 10 grid points per edge is typical. FastKAN defaults to 8. Larger grids raise the highest representable frequency and the risk of overfitting (Section 4.6).

### 10.2 Center initialization

| Strategy | Pros | Cons |
|---|---|---|
| Random subset of data | Simple, fast | Uneven coverage; duplicates in dense regions |
| k-means | Data-adaptive coverage | Cost grows with $N$, $M$, $D$; local optima |
| Uniform grid (KANs) | Reproducible; cheap for univariate bases | Infeasible for multivariate centers in high $D$ |
| Uniform grid, then trainable (free knots / Free-RBF-KAN) | Starts reproducible, adapts to activation patterns | Needs domain constraints to keep centers ordered and in range |
| Supervised (e.g. LVQ) | Class-aware placement | Requires labels; more complex |

### 10.3 Widths

- **Shared:** $\sigma = d_{max}/\sqrt{2N}$.
- **Per-center:** mean distance to the $p$ nearest centers.
- **LOOCV:** choose the shared shape parameter by leave-one-out error, computed in closed form for interpolation (Rippa's method). Use it as an initialization, then refine by training.
- **Learnable:** parameterize as $\sigma = e^{s}$ (or softplus) and optimize jointly. Add a floor to avoid collapse.

### 10.4 Efficient distance and basis computation

```
Squared Euclidean distances for a batch X (B x D) and centers C (N x D):
  D2 = rowsq(X) 1_N^T + 1_B rowsq(C)^T - 2 X C^T
  D2 = max(D2, 0)          # guard against negative values from round-off

Diagonal Mahalanobis: rescale X and C by the per-dimension scales, then apply the same identity.
Full Mahalanobis per center: transform (X - c_i) by L_i^T; batch across centers with einsum.
```

Euclidean RBF layers therefore run at matrix-multiplication speed.

**Further options:**

- **Sparse evaluation:** evaluate only the $k$ nearest centers per input. Tree indices (KD-tree, ball tree) degrade toward linear scan in high dimensions, so use approximate nearest-neighbor methods (e.g. graph-based indices) or compactly supported kernels instead. Sparse, dynamic computation is harder to batch efficiently on GPUs.
- **KAN-style layers:** cost is dominated by the $B \times D_{in} \times G$ basis tensor. Gaussian bases are cheaper to evaluate than recursive B-splines. Exponential-free approximations (RSWAF) cut cost further. A single matmul over the flattened basis tensor is the efficient contraction.
- **Fused kernels and sharing:** fused CUDA/Triton kernels for rational bases, plus group sharing of function coefficients, bring KAN layers close to MLP throughput (Section 6.1).

### 10.5 Parameter counting

| Layer | Parameters |
|---|---|
| Linear (MLP) | $d_{in} d_{out} + d_{out}$ |
| FastKAN-style | $d_{in} d_{out} G$ (basis) $+\; d_{in} d_{out}$ (base branch, optional) $+\; d_{out}$ |
| Group-shared (GR-KAN-style), $g$ groups, $m$ coefficients per function | $d_{in} d_{out} + g\,m + d_{out}$ |
| RBF, isotropic | $N D + N$ (centers, widths) $+\; N K + K$ (output) |
| RBF, diagonal metric | $2 N D + N K + K$ |
| RBF, full metric | $N D + N \tfrac{D(D+1)}{2} + N K + K$ |

Worked examples for MNIST input ($D = 784$, 10 classes):

| Model | Parameters |
|---|---|
| MLP [784-256-256-10] | 269,322 |
| FastKAN [784-64-10], $G = 8$, with base branch | 406,528 basis + 50,816 base + 74 bias = 457,418 |
| Shallow RBF, 128 centers, isotropic | 100,352 + 128 + 1,290 = 101,770 |
| Shallow RBF, 128 centers, diagonal metric | 201,994 |
| Shallow RBF, 128 centers, full metric | 100,352 + 39,388,160 + 1,290 = 39,489,802 |

At equal width, per-edge KAN layers carry roughly $G$ times the parameters of a linear layer. Any parameter-efficiency advantage must come from using much narrower or shallower networks, which is task-dependent. Group-shared variants avoid this overhead by sharing functions across channels.

### 10.6 Reference algorithm: shallow Gaussian RBF network

```
Hyperparameters: D, N, K, learning rate alpha, ridge lambda

Initialize:
  C = kmeans(X_train, N)                       # N x D
  s = log(d_max / sqrt(2N)) * ones(N)          # log-widths (or LOOCV-selected)
  W, b = ridge_solution(Phi(X_train), Y_train, lambda)   # warm start

Forward(X):                                    # X: B x D
  D2 = max(rowsq(X) + rowsq(C)^T - 2 X C^T, 0) # B x N
  S2 = exp(2 s)                                # sigma^2, N
  H  = exp(-D2 / (2 S2))                       # B x N
  Y  = H W + b                                 # B x K
  return Y, H, D2

Loss: L = (1 / 2B) ||Y - T||^2 + (lambda / 2) ||W||^2

Backward:
  G   = (Y - T) / B                            # B x K
  dW  = H^T G + lambda W
  db  = colsum(G)
  GH  = (G W^T) * H                            # B x N, elementwise
  dC[i] = sum_n GH[n, i] (x_n - c_i) / S2[i]
  ds[i] = sum_n GH[n, i] D2[n, i] / S2[i]

Update (Adam recommended):
  W, b at rate alpha; C, s at rate 0.1 * alpha
  clip gradient norm; clamp s >= log(sigma_min)
```

---

## 11. State of the Evidence

| Claim | Status (through mid-2026) |
|---|---|
| Single-layer RBF networks are universal approximators | Established (Park & Sandberg, 1991) |
| KANs are at least as expressive as MLPs of comparable size | Established theoretically (Wang et al., 2025) |
| KANs have reduced spectral bias compared with MLPs | Supported by analysis and experiments (Wang et al., 2025); comes with higher overfitting risk |
| KANs fit smooth compositional/scientific functions accurately and support symbolic recovery | Reported by Liu et al. and many follow-ups on synthetic and physics tasks |
| KANs are more parameter-efficient than MLPs in general | Not supported; controlled comparisons and critical surveys favor MLPs outside symbolic and scientific tasks (Yu et al., 2024; Ji et al., 2025) |
| Gaussian RBF bases match B-spline bases in KANs with faster evaluation | Reported (Li, 2024); adaptive RBF-KANs close remaining accuracy gaps (Chiu et al., 2026) |
| KAN-based PINNs outperform MLP-based PINNs | Mixed; comparable in Shukla et al. (2024), consistently better at small matched budgets in Dzimah et al. (2026) |
| Deep RBF networks train stably with data-driven initialization | Reported (Wurzberger & Schwenker, 2024); performance comparable to simple CNNs |
| RBF heads on CNNs match softmax heads and add prototype interpretability | Reported (Amirian & Schwenker, 2020) |
| KAN layers in GNNs are competitive with MLP layers | Reported (Bresson et al., 2025 revision): on par in classification, advantage in graph regression, higher compute |
| KAN-style layers work at ImageNet scale | Yes for group-shared rational KANs in transformers (KAT, 2025); plain per-edge KANs train poorly at that scale |
| KAN feed-forward blocks improve language models | Not shown; benchmark-equivalent to MLPs at small scale (Alves & Vicente, 2026) |
| KANs train about as fast as MLPs | Original per-edge KANs about 10x slower at equal parameters; RBF, rational and group-shared variants narrow or close the gap |

---

## 12. Applications

### 12.1 Scientific computing and PDEs

RBFs have a long history in meshfree PDE solvers. Kansa's (1990) multiquadric collocation method is the classical example. Neural variants train an RBF or KAN model $u_\theta(x, t)$ with a physics-informed loss.

```
Problem: u_t = alpha u_xx on [0,1] x [0,T]

L = L_initial + L_boundary + L_residual
  L_initial  = mean (u_theta(x, 0) - u0(x))^2
  L_boundary = mean of boundary-condition errors
  L_residual = mean (d/dt u_theta - alpha d2/dx2 u_theta)^2 at collocation points
```

**Advantages:**

- Gaussian derivatives are analytic and smooth.
- Local bases help represent sharp but localized features.
- Reduced spectral bias helps with oscillatory and multiscale solutions.
- KAN edge functions can be inspected for symbolic structure.

**Evidence on physics-informed KANs (PIKANs):**

- **Comparable to PINNs.** Shukla et al. (2024) found PIKANs broadly comparable to MLP-based PINNs. Chebyshev and Jacobi bases were more efficient than B-splines but sensitive to polynomial degree and depth.
- **Better at small budgets.** A unified 2026 benchmark under matched parameter budgets (about 700 parameters) and identical losses found PIKANs consistently more accurate, faster to converge and better at gradient estimation across ODEs and PDEs (Dzimah et al., 2026).
- **Adaptive RBF-KANs.** These have been applied to physics-informed learning and operator learning with accuracy comparable to B-spline KANs at lower cost (Chiu et al., 2026).

**Limits:**

- Most benchmarks use small networks and low-dimensional domains.
- Kernel width selection still governs conditioning for multiquadric and Gaussian kernels.
- Reduced spectral bias increases sensitivity to noisy data.

### 12.2 Time series

Casdagli (1989) used RBF networks for nonlinear prediction of chaotic time series:

- Embed a window $[x_{t-w}, \dots, x_{t-1}]$ in delay coordinates.
- Place centers on recurring patterns (k-means).
- Predict $x_t$ as a weighted combination of pattern responses.

**Strengths:** fast training through the linear readout and interpretable similarity to past regimes. **Limits:** weak extrapolation outside observed regimes, and degradation with long windows due to dimensionality.

KAN-based forecasters such as TimeKAN (ICLR 2025) combine frequency decomposition with KAN layers for long-horizon forecasting. Gains are benchmark-dependent and should be checked against strong linear and transformer baselines.

### 12.3 Graph neural networks

KAGNNs replace the MLP update functions in message-passing GNNs (GCN, GIN) with KAN layers built on either B-spline or RBF bases:

$$
\text{GCN: } H' = \sigma(\hat{A} H W),
\qquad
\text{KAN-GCN: } H' = \mathrm{KAN}(\hat{A} H),
\qquad
\hat{A} = \tilde{D}^{-1/2}(A + I)\tilde{D}^{-1/2}
$$

Reported results are on par with or better than MLP-based counterparts. They are roughly equal on node and graph classification, show a clear advantage on graph regression, and carry additional computational cost.

### 12.4 Computer vision

- **RBF heads** on CNN backbones give prototype-based explanations, add new classes through new centers, and support distance-based confidence (Section 8).
- **Deep partially connected RBF networks** are an alternative to small CNNs (Section 7).
- **KAT** brings group-rational KAN layers into vision transformers at ImageNet scale, with transfer to detection and segmentation (Section 6.1).
- **Gaussian kernel attention** gives a projection-free, distance-based attention operator with an explicit locality scale (Section 9.4).
- **Medical imaging** is a motivating domain because prototype-based explanations are valued there. Systematic evidence at scale is limited.

### 12.5 Symbolic regression

```
Goal: recover f(x, y) = x^2 + y^3 from samples

1. Train a small KAN [2, 1, 1] with sparsity regularization
2. Plot each edge function; prune weak edges
3. Fit candidate symbolic forms per edge (linear, polynomial, exp, sin, ...)
4. Compose: phi_1(x) ~ x^2, phi_2(y) ~ y^3, outer function ~ identity
5. Refit constants; validate on held-out data
```

This works best when the target is low-dimensional and has a sparse compositional structure.

### 12.6 Anomaly detection

```
Fit on normal data only:
  Centers via k-means; per-center widths or covariances from cluster statistics
  Score(x) = max_i phi_i(x)        (or a mixture density estimate)
  Threshold tau = low percentile of scores on held-out normal data

Detect: anomaly if Score(x) < tau
```

The approach needs no anomaly labels and gives interpretable distance-based scores. Performance depends on the feature space. In raw high-dimensional input spaces, distance concentration weakens the signal.

### 12.7 Reinforcement learning

RBF features are a standard choice for linear value-function approximation in continuous state spaces:

$$
\hat{Q}(s, a) = \sum_i w_{i,a}\,\phi_i(s)
$$

The semi-gradient Q-learning update is:

$$
w_{i,a} \leftarrow w_{i,a} + \alpha\left[r + \gamma \max_{a'} \hat{Q}(s', a') - \hat{Q}(s, a)\right]\phi_i(s)
$$

**Strengths:** updates are local and learning is sample-efficient in low-dimensional state spaces.

**Caveat:** off-policy learning combined with function approximation and bootstrapping can diverge even with linear features (the "deadly triad"). On-policy methods (SARSA) or gradient-TD variants are safer.

---

## 13. Open Problems

### 13.1 Theory

- **Function classes.** Characterize which classes yield practical KAN efficiency advantages beyond the grid-size factor in the KAN-to-MLP reduction.
- **Bias and generalization.** Develop generalization theory that accounts for reduced spectral bias and the resulting overfitting risk.
- **Deep RBF networks.** Establish approximation and generalization rates for deep RBF networks with learned metrics.
- **Kernel connections.** Formalize how deep RBF networks and RBF-KANs relate to kernel methods and their infinite-width limits.

### 13.2 Efficiency

- **Speed.** Fused kernels for Gaussian and other RBF bases with trainable centers and shapes, matching what rational bases achieved in KAT.
- **Scalable metrics.** Mahalanobis parameterizations (diagonal-plus-low-rank, shared factors) that preserve the benefits of full covariance.
- **Sparse evaluation.** Efficient sparse center evaluation on GPUs.

### 13.3 Architecture

- **Function sharing.** Determine how much per-edge function diversity is needed. Group sharing scales, but it moves KAN layers toward MLPs with learnable activations.
- **Hybrids.** Evaluate hybrids (KAN feed-forward blocks, RBF heads, Gaussian kernel attention) at matched parameters and FLOPs against strong baselines.

### 13.4 Scale

- **Language models.** Language-model evidence is limited to tens of millions of parameters.
- **Unshared KANs.** Per-edge KANs without sharing have not been shown to scale.
- **Prerequisites.** Progress depends on efficient implementations and on controlled comparisons at scale.

### 13.5 Recurrent RBF models

- **Construction.** RBF units over concatenated state and input, $\mathbf{z}_t = [\mathbf{h}_{t-1}, \mathbf{x}_t]$.
- **Open questions:** gradient behavior through time, center initialization for sequential data, and competitiveness with gated RNNs, state-space models and transformers.

### 13.6 Adoption barriers

- **Tooling.** Libraries are maturing (rational KAN kernels, curated implementation collections), but still trail MLP tooling.
- **Speed.** Per-parameter training cost remains higher for unshared, basis-expanded layers.
- **Evidence.** Gains remain concentrated in scientific and specialized domains, with limited general-purpose large-scale evidence.

---

## 14. Practical Recommendations

### 14.1 When to use

**Consider RBF or KAN models when:**

- The target is a smooth or multiscale, low- to moderate-dimensional function, as in scientific fitting, surrogate modeling, PDE solving or operator learning.
- High-frequency components matter and spectral bias is hurting an MLP.
- Interpretability of learned functions or prototypes is a primary requirement.
- You need distance-based confidence, prototype explanations or few-shot class addition. Use them as a head on a learned feature space.
- Data are limited and a closed-form or near-closed-form readout is useful.

**Prefer standard architectures when:**

- The task is large-scale vision, language or audio with established baselines. Group-shared KAN layers (KAT) are the exception worth testing in vision transformers.
- Throughput and latency are critical and no fused KAN kernels are available.
- Inputs are raw and high-dimensional, with no feature extractor in front of the RBF layer.
- Data are noisy and spectral bias is acting as useful regularization.

### 14.2 Workflow

```
1. Problem analysis
   Samples M, dimension D, task type, noise level, constraints (speed, interpretability, accuracy)

2. Architecture
   Interpretable function fitting  -> KAN (B-spline or adaptive RBF basis)
   Speed-sensitive KAN             -> Gaussian-basis or Free-RBF-style layers
   Physics-informed / operators    -> PIKAN with RBF or Chebyshev basis vs PINN baseline
   Vision transformer              -> GR-KAN feed-forward blocks (KAT)
   Classification on raw images    -> CNN/ViT backbone + RBF head
   Small tabular / low-D problems  -> shallow RBF with ridge readout

3. Starting configuration
   KAN: [D, 16-64, K], G = 5-8, LayerNorm before basis expansion
   RBF-KAN: uniform grid init, trainable centers/shapes with domain constraints
   RBF: k-means centers, log-parameterized widths (LOOCV init), ridge warm start

4. Training
   Adam, lr 1e-3 for weights, about 0.1x for centers and widths
   Batch 32-256; warmup + cosine decay; gradient-norm clipping
   Grid extension when underfitting high frequencies; grid coarsening when overfitting

5. Evaluation (always against a matched MLP baseline)
   Accuracy/loss, parameters, FLOPs, wall-clock training and inference time, memory
   Multiple seeds; report variance

6. Interpretation (KANs)
   Plot edge functions, prune, symbolify, refit
```

### 14.3 Debugging

| Symptom | Likely causes | Remedies |
|---|---|---|
| Loss does not decrease | Centers far from data; widths too small (dead units) or too large (flat responses) | Re-initialize with k-means on current layer inputs; reset widths with the heuristic or LOOCV; add width floor; normalize inputs |
| Overfitting or fitting noise | Too many centers; grid too fine; reduced spectral bias; weak regularization | Fewer centers/grid points; grid coarsening; ridge or L2 on readout; early stopping; dropout between layers |
| Underfitting | Too few centers; grid too coarse; inputs outside grid range | More centers or grid points; grid extension; trainable centers; LayerNorm before basis |
| Unstable training | Learning rate too high for centers/metrics; non-positive widths or metric diagonals; poor initialization of learnable activations | Separate lower learning rate; log/softplus parameterizations; variance-preserving init; gradient clipping |
| Centers drift out of range or cross | Unconstrained trainable centers | Constrain to a fixed domain; enforce ordering via cumulative positive increments |
| Ill-conditioned readout | Overlapping or duplicate centers; wide kernels | Increase ridge $\lambda$; remove near-duplicate centers; narrow widths |
| Slow training | Recursive B-spline evaluation; unshared per-edge functions; full covariance metrics | Gaussian or rational bases; group sharing; single-matmul contraction; fused kernels; diagonal or low-rank metrics |

---

## 15. References

### Classical

1. Broomhead, D. S., & Lowe, D. (1988). Multivariable functional interpolation and adaptive networks. *Complex Systems*, 2, 321-355. (Also RSRE Memorandum 4148.)
2. Moody, J., & Darken, C. J. (1989). Fast learning in networks of locally-tuned processing units. *Neural Computation*, 1(2), 281-294.
3. Park, J., & Sandberg, I. W. (1991). Universal approximation using radial-basis-function networks. *Neural Computation*, 3(2), 246-257.
4. Micchelli, C. A. (1986). Interpolation of scattered data: distance matrices and conditionally positive definite functions. *Constructive Approximation*, 2, 11-22.
5. Rippa, S. (1999). An algorithm for selecting a good value for the parameter c in radial basis function interpolation. *Advances in Computational Mathematics*, 11, 193-210.
6. Powell, M. J. D. (1992). The theory of radial basis function approximation in 1990. In *Advances in Numerical Analysis, Vol. II*, Oxford University Press.
7. Wendland, H. (1995). Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree. *Advances in Computational Mathematics*, 4, 389-396.
8. Buhmann, M. D. (2003). *Radial Basis Functions: Theory and Implementations*. Cambridge University Press.
9. Haykin, S. (1999). *Neural Networks: A Comprehensive Foundation* (2nd ed.). Prentice Hall.
10. Dash, C. S. K., Behera, A. K., Dehuri, S., & Cho, S.-B. (2016). Radial basis function neural networks: a topical state-of-the-art survey. *Open Computer Science*, 6(1).

### Kolmogorov-Arnold theory

11. Kolmogorov, A. N. (1957). On the representation of continuous functions of several variables by superposition of continuous functions of one variable and addition. *Doklady Akademii Nauk SSSR*, 114, 953-956.
12. Girosi, F., & Poggio, T. (1989). Representation properties of networks: Kolmogorov's theorem is irrelevant. *Neural Computation*, 1(4), 465-469.

### KANs: foundations, theory and surveys

13. Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M. (2024). KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756.
14. Liu, Z., Ma, P., Wang, Y., Matusik, W., & Tegmark, M. (2024). KAN 2.0: Kolmogorov-Arnold Networks Meet Science. arXiv:2408.10205.
15. Yu, R., Yu, W., & Wang, X. (2024). KAN or MLP: A Fairer Comparison. arXiv:2407.16674.
16. Wang, Y., Siegel, J. W., Liu, Z., & Hou, T. Y. (2025). On the expressiveness and spectral bias of KANs. *ICLR 2025*. arXiv:2410.01803.
17. Actor, J. A., Harper, G., Southworth, B., & Cyr, E. C. (2025). Leveraging KANs for expedient training of multichannel MLPs via preconditioning and geometric refinement.
18. Zheng, L. N., et al. (2025). Free-knots Kolmogorov-Arnold Network: on the analysis of spline knots and advancing stability.
19. Ji, T., Hou, Y., & Zhang, D. (2025). A Comprehensive Survey on Kolmogorov Arnold Networks (KAN). arXiv:2407.11075 (v8, critical assessment revision).
20. Noorizadegan, A., Wang, S., Ling, L., & Dominguez-Morales, J. P. (2026). A Practitioner's Guide to Kolmogorov-Arnold Networks. *Computer Science Review*. doi:10.1016/j.cosrev.2026.100991. arXiv:2510.25781.

### RBF-based and alternative-basis KANs

21. Li, Z. (2024). Kolmogorov-Arnold Networks are Radial Basis Function Networks. arXiv:2405.06721.
22. Ta, H.-T. (2024). BSRBF-KAN: A combination of B-splines and Radial Basis Functions in Kolmogorov-Arnold Networks. arXiv:2406.11173.
23. Chiu, S.-T., Cheung, S. W., Braga-Neto, U., Lee, C. S., & Li, R. P. (2026). Free-RBF-KAN: Kolmogorov-Arnold Networks with Adaptive Radial Basis Functions for Efficient Function Learning. arXiv:2601.07760.
24. Cavoretto, R., De Rossi, A., Haider, A., & Noorizadegan, A. (2026). Adaptive RBF-KAN: A Comparative Evaluation of Dynamic Shape Parameters in Kolmogorov-Arnold Networks. arXiv:2605.21534.
25. Hierarchical RBF-KAN and RBF-SKAN Architectures for Multidimensional Function Approximation and Random Field Learning (2026). arXiv:2606.02936.

### KANs at scale and in other architectures

26. Yang, X., & Wang, X. (2025). Kolmogorov-Arnold Transformer. *ICLR 2025*. arXiv:2409.10594.
27. Alves, F., & Vicente, R. (2026). Kolmogorov-Arnold Networks for Small Language Models. arXiv:2607.15525.
28. Bresson, R., Nikolentzos, G., Panagopoulos, G., Chatzianastasis, M., Pang, J., & Vazirgiannis, M. (2025). KAGNNs: Kolmogorov-Arnold Networks meet Graph Learning. arXiv:2406.18380 (v4).
29. Huang, S., Zhao, Z., Li, C., & Bai, L. (2025). TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting. *ICLR 2025*.

### Physics-informed learning

30. Shukla, K., Toscano, J. D., Wang, Z., Zou, Z., & Karniadakis, G. E. (2024). A comprehensive and FAIR comparison between MLP and KAN representations for differential equations and operator networks. arXiv:2406.02917.
31. Dzimah, S. K., Rubio Herranz, S., Lopez Hernandez, F. C., & López Montes, A. (2026). A Unified Benchmark of Physics-Informed Neural Networks and Kolmogorov-Arnold Networks for Ordinary and Partial Differential Equations. arXiv:2602.15068.
32. Kansa, E. J. (1990). Multiquadrics: a scattered data approximation scheme with applications to computational fluid dynamics II. *Computers & Mathematics with Applications*, 19(8-9), 147-161.

### Deep and hybrid RBF networks

33. Wurzberger, F., & Schwenker, F. (2024). Learning in Deep Radial Basis Function Networks. *Entropy*, 26(5), 368.
34. Amirian, M., & Schwenker, F. (2020). Radial Basis Function Networks for Convolutional Neural Networks to Learn Similarity Distance Metric and Improve Interpretability. *IEEE Access*, 8, 123087-123097. (arXiv:2208.11401.)
35. van Amersfoort, J., Smith, L., Teh, Y. W., & Gal, Y. (2020). Uncertainty Estimation Using a Single Deep Deterministic Neural Network. *ICML*.
36. Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical Networks for Few-shot Learning. *NeurIPS*.
37. Kundu, D., Ghosh, A., Ghosh, S., & Honavar, V. (2026). Projection-Free Transformers via Gaussian Kernel Attention. arXiv:2605.02144.

### Other

38. Casdagli, M. (1989). Nonlinear prediction of chaotic time series. *Physica D*, 35(3), 335-356.
39. Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.
40. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

### Implementations

- PyKAN: github.com/KindXiaoming/pykan
- EfficientKAN: github.com/Blealtan/efficient-kan
- FastKAN: github.com/ZiyaoLi/fast-kan
- BSRBF-KAN: github.com/hoangthangta/BSRBF_KAN
- KAT and rational KAN kernels: github.com/Adamdad/kat, github.com/Adamdad/rational_kat_cu
- KAGNN: github.com/RomanBresson/KAGNN
- KAN literature review repository: github.com/AmirNoori68/kan-review
- Awesome KAN (curated list): github.com/mintisan/awesome-kan

---

## 16. Notation

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | Input vector |
| $\mathbf{c}_i$ | Center of the $i$-th RBF unit |
| $\sigma_i$, $s_i = \log \sigma_i$ | Width of the $i$-th unit and its log-parameterization |
| $\phi_i(\cdot)$ | Activation of the $i$-th RBF unit |
| $\varphi_{l,j,i}(\cdot)$ | KAN edge function from input $i$ to output $j$ in layer $l$ |
| $w_{ik}$, $b_k$ | Output weights and biases |
| $N$ | Number of RBF centers |
| $M$ | Number of training samples |
| $D$ | Input dimension |
| $K$ | Output dimension |
| $G$ | Number of grid points / basis functions per KAN edge |
| $g$, $m$ | Number of function-sharing groups; coefficients per shared function |
| $B$ | Batch size |
| $\Phi$ | Design matrix, $\Phi_{mi} = \phi_i(\mathbf{x}_m)$ |
| $\Sigma_i$ | Covariance associated with center $i$ |
| $L_i$ | Cholesky factor, $\Sigma_i^{-1} = L_i L_i^\top$ |
| $d_M(\cdot,\cdot)$ | Mahalanobis distance |
| $\hat{A}$ | Normalized adjacency matrix |
| $\tau$, $\sigma_h$ | Attention temperature; per-head Gaussian bandwidth |
| $\alpha$ | Learning rate |
| $\lambda$ | Ridge / L2 regularization strength |