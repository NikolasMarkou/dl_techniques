# CliffordNet: All You Need is Geometric Algebra
> A Technical Guide to arXiv:2601.06793v2 — Zhongping Ji (Feb 2026)

---

## Table of Contents

1. [Core Motivation](#1-core-motivation)
2. [Theoretical Foundation: Geometric Algebra](#2-theoretical-foundation-geometric-algebra)
3. [The Three Core Products](#3-the-three-core-products)
4. [The Clifford Interaction Ansatz](#4-the-clifford-interaction-ansatz)
5. [Context Instantiation](#5-context-instantiation)
6. [Efficient Realization: Sparse Rolling](#6-efficient-realization-sparse-rolling)
7. [Gated Geometric Residual (GGR)](#7-gated-geometric-residual-ggr)
8. [The gFFN Family](#8-the-gffn-family)
9. [Full Algorithm: CliffordNet Block](#9-full-algorithm-cliffordnet-block)
10. [Architecture Design](#10-architecture-design)
11. [Experimental Results](#11-experimental-results)
12. [Ablation Studies](#12-ablation-studies)
13. [Theoretical Interpretation](#13-theoretical-interpretation)
14. [Complexity Analysis](#14-complexity-analysis)
15. [Implementation Reference](#15-implementation-reference)

---

## 1. Core Motivation

### The MetaFormer Problem

Modern vision backbones universally follow the **MetaFormer** pattern: every block stacks a **spatial mixer** (captures *where*) on top of a **channel mixer** (captures *what*):

```
Input
  └─> Norm ─> Spatial Mixer (Attention / Conv)  ─> Residual
  └─> Norm ─> Channel Mixer (FFN / MLP)          ─> Residual
Output
```

This two-module decomposition is an **engineering convention**, not a mathematical necessity. It arose historically because:
- Dot-product attention only captures scalar similarity — it discards structural information
- FFNs compensate for this information loss via brute-force channel mixing

CliffordNet's central claim: **if the interaction operator is algebraically complete, the FFN becomes redundant.**

### The Information Bottleneck in Standard Attention

Standard scaled dot-product attention compresses the entire relationship between two feature vectors $q, k \in \mathbb{R}^D$ into a single scalar:

$$\text{score}(q, k) = \frac{q \cdot k}{\sqrt{D}}$$

This is **geometrically lossy**. It retains only the symmetric, magnitude-based alignment (the inner product) while discarding the **anti-symmetric structural component** — the oriented plane that $q$ and $k$ span together.

CliffordNet restores this missing component via Geometric Algebra.

---

## 2. Theoretical Foundation: Geometric Algebra

### The Clifford Algebra $\text{Cl}(p, q)$

For a vector space $\mathbb{R}^n$, the Clifford algebra is constructed from basis vectors $e_1, \ldots, e_n$ satisfying the fundamental relation:

$$e_i e_j + e_j e_i = 2\eta_{ij}$$

where $\eta$ is a diagonal metric with $p$ positive and $q$ negative entries. For standard vision applications, we use $\mathbb{R}^D$ with Euclidean metric ($p = D$, $q = 0$).

### Multivectors: The Grade Hierarchy

The fundamental object in GA is the **multivector** — a sum over all grades:

| Grade | Name      | Dimension             | Geometric Meaning              |
|-------|-----------|-----------------------|--------------------------------|
| 0     | Scalar    | 1                     | Magnitude / Energy             |
| 1     | Vector    | $D$                   | Directed line segment          |
| 2     | Bivector  | $\binom{D}{2}$        | Oriented plane (area element)  |
| 3     | Trivector | $\binom{D}{3}$        | Oriented volume                |
| $k$   | $k$-blade | $\binom{D}{k}$        | Oriented $k$-subspace          |

A general multivector in $\mathbb{R}^D$ has $2^D$ components total.

**For CliffordNet's 2-vector interaction**, only Grades 0 and 2 appear:

$$uv = \underbrace{u \cdot v}_{\text{Grade 0: scalar}} + \underbrace{u \wedge v}_{\text{Grade 2: bivector}}$$

### Algebraic Completeness — Formal Definition

> **Definition.** An architecture is *algebraically complete under the Geometric Product* if it explicitly models both the **coherence** (scalar, symmetric) and **structural** (bivector, anti-symmetric) terms of the interaction between feature vectors.

Standard neural primitives (dot-product attention, cosine similarity, Hadamard gating) are all **algebraically incomplete** — they use only the symmetric component.

---

## 3. The Three Core Products

### 3.1 Inner Product (Dot Product): $u \cdot v$

$$u \cdot v = \frac{1}{2}(uv + vu) = \sum_{i=1}^{D} u_i v_i$$

| Property | Value |
|----------|-------|
| Result grade | 0 (scalar) |
| Symmetry | Symmetric: $u \cdot v = v \cdot u$ |
| Zero condition | $u \perp v$ |
| Max condition | $u \parallel v$ |
| Invertible alone? | No |
| Neural analogue | Dot-product attention, cosine similarity |

**In CliffordNet:** Acts as a **coherence gate** — measures how much a feature aligns with its local context, implementing anisotropic diffusion.

### 3.2 Exterior Product (Wedge Product): $u \wedge v$

$$u \wedge v = \frac{1}{2}(uv - vu) = \sum_{1 \le i < j \le D} (u_i v_j - v_i u_j)(e_i \wedge e_j)$$

| Property | Value |
|----------|-------|
| Result grade | 2 (bivector) |
| Symmetry | Anti-symmetric: $u \wedge v = -v \wedge u$ |
| Zero condition | $u \parallel v$ (self-wedge: $u \wedge u = 0$) |
| Max condition | $u \perp v$ |
| Invertible alone? | No |
| Neural analogue | **None in standard DL** — this is the missing piece |

**In CliffordNet:** Acts as a **structural variation detector** — captures orthogonality between a feature and its context, implementing geometric vorticity at edges and texture boundaries.

### 3.3 Geometric Product: $uv$

$$\boxed{uv = u \cdot v + u \wedge v}$$

| Property | Value |
|----------|-------|
| Result | Multivector (scalar + bivector) |
| Associativity | Yes |
| Distributivity | Yes |
| Invertibility | **Yes** — uniquely invertible |
| Encodes inner product | Yes |
| Encodes wedge product | Yes |
| Information loss | **None** |

The geometric product is the **only** standard algebraic product over vectors that is fully invertible and information-preserving. It is strictly more expressive than either the dot product or the cross product alone.

---

## 4. The Clifford Interaction Ansatz

### Feature Evolution as a Differential Equation

CliffordNet models layer-wise feature updates as a continuous dynamical system:

$$\frac{\partial H}{\partial t} = \mathcal{F}\bigl(H,\; C(H)\bigr)$$

where:
- $H \in \mathbb{R}^{h \times w \times D}$ is the feature field at "time" $t$ (layer index)
- $C(H)$ is the **context field** — a spatially aggregated version of $H$
- $\mathcal{F}$ is the interaction function

### The Ansatz: Geometric Product as $\mathcal{F}$

$$\mathcal{F}(H, C) := \mathcal{P}\!\left(\underbrace{H \cdot C}_{\text{Coherence}} \oplus \underbrace{H \wedge C}_{\text{Structure}}\right)$$

where:
- $\mathcal{P} : \mathbb{R}^{2D} \to \mathbb{R}^D$ is a **learnable linear projection** that contracts the concatenated scalar+bivector representation back to the feature dimension
- $\oplus$ denotes channel-wise concatenation (embedding the heterogeneous multivector into a unified vector space)

### Why This Replaces the FFN

The standard justification for FFNs in transformers is that post-attention representations are "too linear." The geometric product resolves this:

1. **Internalized non-linearity:** The bivector $H \wedge C$ contains second-order multiplicative terms. Combined with SiLU gating in GGR, the interaction layer functions as a powerful non-linear approximator.
2. **Structured channel mixing:** The shifted geometric product mixes channels along a ring topology with logarithmic path length — more sample-efficient than dense MLP mixing.
3. **Dual information:** The layer simultaneously propagates energy (inner product) and structural contrast (wedge product), eliminating the need for a separate channel-mixing stage.

---

## 5. Context Instantiation

The context $C(H)$ is the key design variable. CliffordNet defines three instantiations:

### 5.1 Local Context via Factorized Laplacian

The local context approximates the discrete Laplacian via factorized depth-wise convolutions:

$$C_{\text{loc}}(H) = \text{DWConv}_{3\times3}\!\left(\text{DWConv}_{3\times3}(H)\right)$$

This achieves an effective $7 \times 7$ receptive field using highly optimized $3 \times 3$ cuDNN primitives.

### 5.2 The Self-Energy Switch $\lambda$

A binary parameter $\lambda \in \{0, 1\}$ controls the **context mode**:

$$C = C_{\text{loc}}(H) - \lambda \cdot H$$

| Mode | $\lambda$ | Context $C$ | Behavior |
|------|-----------|-------------|----------|
| **Absolute** | 0 | $C_{\text{loc}}(H)$ | Energy-preserving flow; retains feature intensity |
| **Differential** | 1 | $C_{\text{loc}}(H) - H \approx \Delta H$ | High-pass filter; suppresses static self-energy; focuses on structural variation |

**Empirical finding:** Differential mode ($\lambda = 1$) consistently outperforms Absolute mode by ~1.4% on CIFAR-100 — filtering static self-energy improves the signal-to-noise ratio for structural variations.

### 5.3 Global Context

$$C_{\text{glo}} = \text{GlobalAvgPool}(H) \in \mathbb{R}^D$$

The global context interacts with each local token $H_i$ via the full geometric product. This is termed **gFFN-G**:
- **Wedge term** ($H \wedge C_{\text{glo}}$): highlights features that deviate from the global scene — saliency detection
- **Inner term** ($H \cdot C_{\text{glo}}$): reinforces features consistent with the global scene — semantic coherence

### 5.4 Dual-Scale Superposition (gFFN-H)

$$\frac{\partial H}{\partial t} = \mathcal{P}_{\text{loc}}\!\left(H \cdot C_{\text{loc}}\right) + \beta \cdot \mathcal{P}_{\text{glo}}\!\left(H \cdot C_{\text{glo}}\right)$$

where $\beta \in \{0, 1\}$ is a structural switch:
- $\beta = 0$: Local only — extreme parameter efficiency (Nano/Lite variants)
- $\beta = 1$: Local + Global — maximum performance (+0.4–0.5% accuracy)

---

## 6. Efficient Realization: Sparse Rolling

### The Quadratic Problem

Computing the full geometric product $H \cdot C$ as an all-to-all channel interaction requires an outer product matrix $M \in \mathbb{R}^{D \times D}$, yielding $O(D^2)$ complexity — reintroducing the quadratic cost of standard FFNs.

### The Rolling Solution

Instead of computing all $D^2$ channel pairs, CliffordNet samples **specific diagonals** of $M$ using **cyclic channel shifts**.

Define the shift operator $T_s$ that cyclically shifts a tensor along the channel dimension by offset $s$:

$$[T_s(X)]_{i,c} = X_{i,\, (c+s) \bmod D}$$

For a set of sparse shift offsets $S = \{s_1, s_2, \ldots\}$ (e.g., $\{1, 2, 4, 8, 16\}$), the scalar and bivector components at offset $s$ are:

$$\text{Dot}^{(s)}_{i,c} = \text{SiLU}\!\left(H_{i,c} \cdot C_{i,\,(c+s)\%D}\right)$$

$$\text{Wedge}^{(s)}_{i,c} = H_{i,c} \cdot C_{i,\,(c+s)\%D} \;-\; C_{i,c} \cdot H_{i,\,(c+s)\%D}$$

### Geometric Interpretation of Each Term

**Dot term:** A generalized inner product $u^\top P_s v$ where $P_s$ is the cyclic permutation matrix for shift $s$. The SiLU activation makes this a non-linear coherence gate — it measures alignment between a feature and its channel-shifted context.

**Wedge term:** Computes the bivector coefficient for the oriented plane spanned by basis vectors $e_c$ and $e_{c+s}$. This is the oriented area of the parallelogram spanned by $H$ and $C$ in the $(c, c+s)$ channel plane. Anti-symmetry is preserved: when $H = C$, the wedge term evaluates to zero.

### Shift Set Design

| Shift set $S$ | Variant | Params | Accuracy |
|---------------|---------|--------|----------|
| $\{1, 2\}$ | Nano | 1.43M | 77.82% |
| $\{1, 2, 4, 8, 16\}$ | Lite | 2.61M | 79.05% |

**Exponential shifts** ($s \in \{1, 2, 4, \ldots, 2^k\}$) give logarithmic path length across the channel dimension — global channel mixing with $O(\log D)$ hops, analogous to the $O(\log N)$ path length in dilated convolutions.

### Circulant Sparsity Interpretation

Conceptually, the full channel interaction matrix $M$ is dense. The rolling operator extracts specific diagonals:
- **All shifts** $S = \{0, \ldots, D-1\}$: equivalent to a full Circulant Matrix multiplication
- **Sparse shifts** (CliffordNet): structured circulant sparsity — retains the most significant frequency bands of the channel interaction topology

Complexity: from $O(D^2)$ dense to $O(|S| \cdot D)$ sparse.

### Interaction Modes

Three modes control which components are concatenated before projection:

| Mode | Concatenation | Output dim before proj | Use case |
|------|--------------|------------------------|----------|
| `inner` | Dot terms only | $\|S\| \cdot D$ | Ablation — scalar only |
| `wedge` | Wedge terms only | $\|S\| \cdot D$ | Ablation — structure only |
| `full` | Dot + Wedge | $2\|S\| \cdot D$ | Default — algebraically complete |

---

## 7. Gated Geometric Residual (GGR)

### Euler Discretization of the ODE

The continuous evolution $\frac{\partial H}{\partial t} = \mathcal{F}(H, C)$ is discretized via first-order Euler stepping:

$$H^{(l)} = H^{(l-1)} + \Delta t \cdot \mathcal{F}(H^{(l-1)}, C^{(l-1)})$$

where $\Delta t$ corresponds to a learnable LayerScale parameter $\gamma$.

### The Refined GGR Update

Directly adding raw geometric interaction to the identity path causes noise in the semantic stream. The GGR introduces a non-linear pre-filtering and gated injection:

$$H^{(l)} = H^{(l-1)} + \gamma \odot \Bigl(\underbrace{\text{SiLU}(H^{(l-1)})}_{\text{noise suppression}} + \underbrace{\alpha \odot G_{\text{feat}}}_{\text{gated geometric force}}\Bigr)$$

where:
- $G_{\text{feat}}$ is the output of the Clifford interaction (after projection $\mathcal{P}$)
- $\alpha = \sigma\!\left(W_{\text{gate}} \cdot [H_{\text{norm}}; G_{\text{feat}}]\right)$ is a learned gate
- $\gamma$ is a learnable LayerScale scalar (initialized near zero for training stability)
- DropPath is applied before the residual addition

### Role of Each Component

| Component | Role |
|-----------|------|
| $\text{SiLU}(H^{(l-1)})$ | Suppresses negative (background noise) values; activates salient features before geometric integration |
| $\alpha$ (sigmoid gate) | Soft selector that modulates how much geometric force to inject at each spatial/channel position |
| $\gamma$ (LayerScale) | Ensures early training stability — small initial step size prevents disruption of pretrained or random-init features |
| DropPath | Stochastic depth regularization; drops entire blocks during training |

---

## 8. The gFFN Family

CliffordNet unifies all context variants under the **geometric Feed-Forward Network (gFFN)** abstraction:

| Variant | Context $C$ | $\beta$ | $\lambda$ | Best for |
|---------|------------|---------|-----------|----------|
| **gFFN-L** (Local) | $\Delta H = C_{\text{loc}} - H$ | 0 | 1 | Extreme efficiency (Nano) |
| **gFFN-G** (Global) | $\text{GlobalAvgPool}(H)$ | 1 | — | Semantic coherence supplement |
| **gFFN-H** (Hybrid) | $\Delta H + \beta \cdot C_{\text{glo}}$ | 1 | 1 | Maximum performance (SOTA) |

The gFFN-G is the **geometrically principled replacement for the standard FFN**: instead of blind channel mixing, it aligns local features with the global scene context via the full geometric product.

---

## 9. Full Algorithm: CliffordNet Block

```
Algorithm: CliffordNet Block (No-gFFN-G Variant)
Input:  X_prev  shape (B, H, W, C)
Output: X_out   shape (B, H, W, C)

Step 1 — Input Normalization
  X_norm = LayerNorm(X_prev)

Step 2 — Dual-Stream Generation
  Z_det = Linear_det(X_norm)           # "detail" stream — high-frequency features
  Z_ctx = SiLU(BN(DWConv(X_norm)))     # "context" stream — local neighborhood
  if ctx_mode == 'diff':
    Z_ctx = Z_ctx - Z_det              # Differential mode: context ≈ ΔH

Step 3 — Sparse Rolling Interaction
  F_list = []
  for s in S:                          # e.g., S = {1, 2, 4, 8, 16}
    Z_det_s = Roll(Z_det, shift=s, axis=channels)
    Z_ctx_s = Roll(Z_ctx, shift=s, axis=channels)
    
    W_s = Z_det ⊙ Z_ctx_s - Z_ctx ⊙ Z_det_s    # Wedge component
    D_s = SiLU(Z_det ⊙ Z_ctx_s)                  # Dot component
    
    if cli_mode == 'inner':   F_list.append(D_s)
    if cli_mode == 'wedge':   F_list.append(W_s)
    if cli_mode == 'full':    F_list.append(W_s, D_s)
  
  G_raw  = Concat(F_list)              # shape: (B, H, W, |S|·C) or (B, H, W, 2|S|·C)
  G_feat = Linear_proj(G_raw)          # shape: (B, H, W, C)

Step 4 — Gated Geometric Residual (GGR)
  M      = Concat([X_norm, G_feat])    # shape: (B, H, W, 2C)
  alpha  = Sigmoid(Linear_gate(M))     # shape: (B, H, W, C) — learned gate
  H_mix  = SiLU(X_norm) + alpha ⊙ G_feat

Step 5 — Output Update
  X_out = X_prev + DropPath(gamma ⊙ H_mix)
  return X_out
```

### Key Design Decisions

1. **No FFN block** — the geometric interaction replaces it entirely
2. **Two linear projections** — `Linear_det` and `Linear_proj` are the primary learnable parameters
3. **BN before DWConv activation** — BatchNorm stabilizes the context stream before non-linearity
4. **LayerNorm on input** — standard pre-norm pattern for training stability
5. **LayerScale $\gamma$** — prevents disruption of feature evolution in early training

---

## 10. Architecture Design

### Isotropic (Columnar) Layout

Unlike hierarchical backbones (ResNet, Swin) that downsample spatial resolution while expanding channels, CliffordNet uses a **constant feature map throughout all layers**:

```
Input Image (H×W×3)
       │
  Patch Embed (Conv, stride P)
       │
  Feature Map (h×w×D)  ← fixed dimensions
       │
  [CliffordNet Block] × L layers
       │
  Global AvgPool
       │
  Classifier Head
```

This aligns with the theoretical formulation: learning is a geometric flow within a **fixed phase space**, not a progressive dimensionality change.

### Model Variants

| Model | Depth $L$ | Channels $D$ | Shifts $|S|$ | CLI mode | $\beta$ | Params | CIFAR-100 |
|-------|-----------|--------------|--------------|----------|---------|--------|-----------|
| CliffordNet-Nano | — | — | 2 ($\{1,2\}$) | full | 0 | 1.4M | 77.82% |
| CliffordNet-Lite | — | — | 5 ($\{1..16\}$) | full | 0 | 2.6M | 79.05% |
| CliffordNet-Lite+gFFN-G | — | — | 5 | full | 1 | 3.40M | 79.57% |
| CliffordNet-32 | — | 32 | 3 | full | — | 4.8M | 81.42% |
| CliffordNet-64 | — | 64 | 5 | inner | — | 8.6M | 82.46% |

### Patch Embedding

- Patch size $P = 2$ (empirically optimal for CIFAR-100)
- Implemented as a strided convolution: $h = H/P$, $w = W/P$
- Feature dimension $D$ fixed for all blocks

---

## 11. Experimental Results

### Training Protocol (CIFAR-100)

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 200 |
| Optimizer | AdamW |
| LR schedule | Cosine annealing |
| Augmentation | AutoAugment + Random Erasing |
| Regularization (CliffordNet) | DropPath |
| Regularization (CNN baselines) | Weight decay only |
| Pre-training | None — trained from scratch |
| Warm-up | None |

### Main Results

| Model | Params | FFN? | Top-1 Acc | Notes |
|-------|--------|------|-----------|-------|
| ShuffleNetV2 1.0× | 1.4M | Yes | 74.60% | Baseline CNN |
| MobileNetV2 | 2.3M | Yes | 70.90% | Depthwise CNN |
| ViT-Tiny (ratio=1.0) | 2.7M | Yes | 65.87% | Global attention — collapses without heavy FFN |
| ResNet-18 | 11.2M | — | 76.75% | Standard deep CNN |
| ResNet-50 | 23.7M | — | 79.14% | |
| DenseNet-121 | 7.0M | — | 80.20% | |
| **CliffordNet-Nano** | **1.4M** | **No** | **77.82%** | +4.3% vs ShuffleNetV2, same params |
| **CliffordNet-Lite** | **2.6M** | **No** | **79.05%** | SOTA <3M params; +2.3% vs ResNet-18 (4× fewer params) |
| **CliffordNet-32** | **4.8M** | **No** | **81.42%** | Surpasses ResNet-50 |
| **CliffordNet-64** | **8.6M** | **No** | **82.46%** | Surpasses DenseNet-121 |

### Key Takeaways

- CliffordNet-Nano matches ResNet-18 with **8× fewer parameters**
- ViT-Tiny collapses to 65.87% without heavy FFNs — confirms that standard attention lacks geometric density
- All CliffordNet results are **No-FFN** — the geometric interaction is sufficient
- Training from scratch, no pretraining, no warmup

---

## 12. Ablation Studies

### Context Mode (Absolute vs. Differential)

| Variant | $\lambda$ | Shifts | Params | Acc |
|---------|-----------|--------|--------|-----|
| Nano (abs) | 0 | 2 | 1.43M | 76.41% |
| Nano (diff) | 1 | 2 | 1.43M | **77.82%** |
| Lite (abs) | 0 | 5 | 2.61M | 77.63% |
| Lite (diff) | 1 | 5 | 2.61M | **79.05%** |

Differential mode wins by ~1.4% — suppressing static self-energy focuses the interaction on structural variation.

### Inner vs. Wedge vs. Full (all use Differential mode, $S = \{1,2,4,8,16\}$, 1.63M params)

| Variant | Algebraic type | Has diagonal energy | Focus | Acc |
|---------|---------------|---------------------|-------|-----|
| Inner-Only | Symmetric | Yes | Coherence | 78.17% |
| Wedge-Only | Anti-symmetric | No ($u \wedge u = 0$) | Structure | 77.76% |
| **Full (CliffordNet)** | Geometric | Yes | Complete | **79.05%** |

**Critical insight:** Wedge-Only (77.76%) nearly matches Inner-Only (78.17%) despite having **zero diagonal energy terms**. This demonstrates that structural/topological information (bivectors) is almost as discriminative as feature magnitude — validating the core CliffordNet thesis.

### Effect of Global Context (gFFN-G)

| Variant | $\beta$ | Params | Acc |
|---------|---------|--------|-----|
| Nano (diff) | 0 | 1.43M | 77.82% |
| Nano (diff) + gFFN-G | 1 | 2.22M | 78.22% |
| Lite (diff) | 0 | 2.61M | 79.05% |
| Lite (diff) + gFFN-G | 1 | 3.40M | 79.57% |

Adding global context provides ~+0.4% at the cost of ~+0.8M parameters.

### Shift Density

Increasing shifts from 2 (Nano) to 5 (Lite) gives +1.2% accuracy with +56% training time (unoptimized). With custom CUDA kernels: training times converge (57.64 vs 58.27 min).

---

## 13. Theoretical Interpretation

### Geometric Reaction-Diffusion System

CliffordNet can be interpreted as a high-dimensional **Turing reaction-diffusion system**:

| Component | Mathematical role | Physical analogy |
|-----------|------------------|------------------|
| $H \cdot C$ (Inner) | Anisotropic diffusion | Smoothing in homogeneous regions; coherence-based noise reduction |
| $H \wedge C$ (Wedge) | Geometric reaction / vorticity | Edge sharpening; preserves structural boundaries where $H \perp C$ |

This explains the No-FFN success: the architecture **emerges complex semantic representations from a cascade of geometric pattern formation steps**, analogous to how Turing patterns emerge from simple local rules.

### Why FFNs Are Redundant

Standard ViT reasoning:
1. Attention is primarily linear aggregation (post-softmax weighted sum)
2. Therefore, a heavy FFN (ratio = 4) is needed for non-linear transformation

CliffordNet reasoning:
1. The geometric product contains **second-order multiplicative terms** (both $H \odot T_s(C)$ and $T_s(H) \odot C$)
2. Combined with SiLU in the GGR, the block is already a powerful non-linear approximator
3. The structured ring-topology mixing of shifted products is more **sample-efficient** than dense MLP mixing

### Circulant Matrix Perspective

The sparse rolling interaction can be formalized as structured matrix sparsity:

- **Full dense mixing** (FFN): $D \times D$ learnable weight matrix — $O(D^2)$ params and FLOPs
- **Full circulant** (all shifts): multiplication by a circulant matrix — $O(D \log D)$ via FFT
- **Sparse circulant** (CliffordNet): $|S|$ selected diagonals — $O(|S| \cdot D)$ params, retains key frequency bands

---

## 14. Complexity Analysis

| Dimension | Standard Self-Attention | CliffordNet |
|-----------|------------------------|-------------|
| Sequence length $N$ | $O(N^2)$ | $O(N)$ |
| Channel dimension $D$ | $O(D)$ | $O(\|S\| \cdot D)$ |
| Parameters (mixer) | $O(D^2)$ per head | $O(\|S\| \cdot D)$ rolling + $O(D^2)$ projection |
| Global context | Explicit (QKV attention) | Emergent through depth / optional gFFN-G |
| Image topology | Serialized 1D | Native 2D |

**Total block complexity:** $O(N \cdot D \cdot |S|)$ — strictly linear in sequence length $N$, making CliffordNet naturally suited for high-resolution dense prediction tasks.

---

## 15. Implementation Reference

### Core Tensor Operations

```python
import keras

def shifted_geometric_product(
    Z_det: keras.KerasTensor,   # (B, H, W, D)
    Z_ctx: keras.KerasTensor,   # (B, H, W, D)
    shift: int,
    cli_mode: str = 'full',     # 'inner' | 'wedge' | 'full'
) -> keras.KerasTensor:
    """
    Compute one slice of the sparse geometric product at channel offset `shift`.
    
    Returns tensor of shape (B, H, W, D) or (B, H, W, 2D) depending on cli_mode.
    """
    # Cyclic channel shift
    Z_det_s = keras.ops.roll(Z_det, shift=shift, axis=-1)
    Z_ctx_s = keras.ops.roll(Z_ctx, shift=shift, axis=-1)
    
    # Wedge (bivector) component: anti-symmetric cross-term
    # W[i,c] = H[i,c] * C[i,(c+s)%D] - C[i,c] * H[i,(c+s)%D]
    wedge = Z_det * Z_ctx_s - Z_ctx * Z_det_s
    
    # Dot (scalar) component: generalized inner product with SiLU gate
    # D[i,c] = SiLU(H[i,c] * C[i,(c+s)%D])
    dot = keras.activations.silu(Z_det * Z_ctx_s)
    
    if cli_mode == 'inner':
        return dot
    elif cli_mode == 'wedge':
        return wedge
    else:  # 'full'
        return keras.ops.concatenate([wedge, dot], axis=-1)
```

### Shift Set Conventions

```python
# Nano variant: 2 shifts
S_nano = [1, 2]

# Lite variant: 5 shifts (powers of 2 up to 16)
S_lite = [1, 2, 4, 8, 16]

# General: exponential shifts up to D//2
def make_shift_set(num_shifts: int) -> list[int]:
    return [2**i for i in range(num_shifts)]
```

### Differential Context Generation

```python
def make_context(
    X_norm: keras.KerasTensor,    # (B, H, W, D)
    ctx_mode: str = 'diff',       # 'diff' | 'abs'
) -> tuple[keras.KerasTensor, keras.KerasTensor]:
    """
    Returns (Z_det, Z_ctx) dual streams.
    
    Z_det: detail stream (high-frequency linear projection)
    Z_ctx: context stream (local neighborhood via DWConv)
    """
    Z_det = keras.layers.Dense(D)(X_norm)
    Z_ctx = keras.activations.silu(
        keras.layers.BatchNormalization()(
            keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(X_norm)
        )
    )
    if ctx_mode == 'diff':
        Z_ctx = Z_ctx - Z_det    # Differential mode: C ≈ ΔH
    return Z_det, Z_ctx
```

### Layer Architecture Parameters

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `channels D` | int | — | Feature dimension (constant throughout) |
| `shifts S` | list[int] | `[1,2]` | Which channel offsets to compute |
| `cli_mode` | str | `'full'` | `'inner'`, `'wedge'`, or `'full'` |
| `ctx_mode` | str | `'diff'` | `'diff'` (Laplacian) or `'abs'` (raw conv) |
| `beta` | int | 0 | 0: local only; 1: local + global (gFFN-G) |
| `layer_scale_init` | float | 1e-5 | LayerScale $\gamma$ initialization |
| `drop_path_rate` | float | 0.0 | DropPath probability |

---

## Summary: Design Principles

| Principle | Decision | Rationale |
|-----------|----------|-----------|
| Feature interaction | Geometric Product | Algebraically complete — captures both coherence and structure |
| Spatial mixing | Local depth-wise conv | $O(N)$, native 2D, cuDNN optimized |
| Channel mixing | Sparse rolling ($|S|$ shifts) | $O(|S| \cdot D)$ structured circulant; more sample-efficient than FFNs |
| Context mode | Differential ($\lambda=1$) | Suppresses static self-energy; maximizes structural variation signal |
| Residual update | GGR with SiLU + gating | Stabilized numerical solver; prevents noisy geometric force injection |
| Global context | Optional gFFN-G | +0.4% accuracy; replaces FFN semantically rather than blindly |
| Architecture shape | Isotropic | Consistent with fixed-phase-space geometric flow formulation |
| FFN | Removed entirely | Bivector term internalizes the non-linearity and channel mixing |
