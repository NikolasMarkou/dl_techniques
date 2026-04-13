# CliffordNet Family

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

Geometric-algebra-based neural network architectures. The family includes a vision classifier, a causal language model, an image denoiser, and a field-theory-augmented vision model.

Based on: **"CliffordNet: All You Need is Geometric Algebra"** (arXiv:2601.06793v2)

---

## Table of Contents

1. [Models](#1-models)
2. [CliffordNet (Vision)](#2-cliffordnet-vision)
3. [CliffordFieldNet (Vision + Field Theory)](#3-cliffordfieldnet-vision--field-theory)
4. [CliffordNetLM (Language)](#4-cliffordnetlm-language)
5. [CliffordNetDenoiser](#5-cliffordnetdenoiser)
6. [Core Primitives](#6-core-primitives)
7. [Quick Start](#7-quick-start)

---

## 1. Models

| Model | Domain | File | Key Idea |
|:------|:-------|:-----|:---------|
| `CliffordNet` | Vision | `model.py` | Attention-free backbone: geometric product replaces both attention and FFN |
| `CliffordFieldNet` | Vision | `field_net.py` | CliffordNet + gauge field theory (curvature, connections, holonomy, transport) |
| `CliffordNetLM` | NLP | `lm.py` | Autoregressive LM with causal Clifford blocks |
| `CliffordNetDenoiser` | Vision | `denoiser.py` | Bias-free denoiser satisfying Miyasawa's theorem |

All models share the same algebraic core: `SparseRollingGeometricProduct` and `GatedGeometricResidual` from `layers/geometric/clifford_block.py`.

---

## 2. CliffordNet (Vision)

The standard isotropic backbone. Replaces both attention and FFN with a single Clifford geometric product pathway.

### Architecture

```
Input (B, H, W, C)
  --> GeometricStem (Conv + BN)
  --> L x CliffordNetBlock
  --> GlobalAvgPool --> LayerNorm --> Dense
  --> Logits (B, num_classes)
```

### CliffordNetBlock Data Flow

```
X_prev (B, H, W, D)
  |
  v
LayerNorm --> X_norm
  |
  +--> Dense(D) ---------> Z_det        (detail stream, 1x1 pointwise)
  |
  +--> DWConv3x3 --> DWConv3x3 --> BN --> SiLU --> Z_ctx   (context, 7x7 RF)
       |
       (if diff: Z_ctx -= Z_det)         discrete Laplacian
       |
       v
SparseRollingGeometricProduct(Z_det, Z_ctx) --> G_feat
       |
       (+ optional GAP global branch)
       |
       v
GatedGeometricResidual(X_norm, G_feat) --> H_mix
       |
       v
X_out = X_prev + H_mix
```

No FFN, no attention -- the geometric product replaces both.

### Variants

| Variant | Channels | Depth | Shifts | Params |
|:--------|:--------:|:-----:|:-------|:------:|
| `nano` | 128 | 12 | [1, 2] | ~1.4M |
| `lite` | 128 | 12 | [1, 2, 4, 8, 16] | ~2.6M |
| `lite_g` | 128 | 12 | [1, 2, 4, 8, 16] + global | ~3.4M |

---

## 3. CliffordFieldNet (Vision + Field Theory)

CliffordFieldNet integrates six improvements from the gauge field theory layers (`layers/geometric/fields/`) into the CliffordNet block, giving the network explicit geometric structure -- curvature, connections, holonomy, and parallel transport -- while preserving the Clifford algebraic core.

### CliffordFieldBlock vs CliffordNetBlock

The core dual-stream + geometric product pipeline is identical. Everything below is **added around it**:

#### Improvement 1: Curvature-Aware Normalization

```
CliffordNetBlock:   x_norm = LayerNorm(x)

CliffordFieldBlock: curv = Dense(x, tanh) * 0.1
                    x_norm = FieldNorm(x, curv)
                           = LayerNorm(x) * 1/(1 + alpha * ||curv||)
```

**What changes**: A learned curvature estimator modulates normalization strength per position. High-curvature positions (edges, texture boundaries, object contours) receive *less* normalization, preserving geometric detail. Smooth regions get full normalization. The scale factor `alpha` is learnable.

**Why it matters**: Standard LayerNorm uniformly normalizes all positions, which washes out fine structure at edges and boundaries -- precisely where geometric information is most important for classification.

#### Improvement 2: Connection-Guided Context Stream

```
CliffordNetBlock:   z_ctx = DWConv(x)          fixed, isotropic, content-independent

CliffordFieldBlock: z_ctx = DWConv(x)           same conv
                    Gamma = ConnectionLayer(z_det, curvature)   learned gauge connection
                    z_ctx = z_ctx + 0.1 * einsum('bsij,bsj->bsi', Gamma, z_ctx)
```

**What changes**: After the standard DWConv context, a `ConnectionLayer` computes a `(B, S, D, D)` gauge connection matrix from `(z_det, curvature)`. This matrix transforms the context features through a content-dependent, position-specific linear rotation before they enter the geometric product.

The connection type is Yang-Mills by default: `Gamma = sum_g c_g T_g` where `T_g` are antisymmetric Lie algebra generators, ensuring the transport is rotation-like (preserves vector norms, no scaling).

**Why it matters**: DWConv applies the same fixed kernel everywhere -- it gathers the same local neighborhood regardless of content. The connection adapts *how* context features are mixed based on what the network sees. Think of it as the differential geometry equivalent of deformable convolution: instead of shifting *where* you sample, you rotate *what* you sampled into the correct local frame.

#### Improvement 3: Holonomy Global Context

```
CliffordNetBlock:   c_glo = mean(x, spatial)    global average pooling
                    g_glo = SRGP(z_det, c_glo - z_det)

CliffordFieldBlock: holonomy = HolonomyLayer(x, Gamma)
                    g_feat += 0.1 * holonomy_proj(holonomy)
```

**What changes**: Instead of (or in addition to) simple mean pooling, `HolonomyLayer` computes Wilson loop traces `Tr([Gamma_s, Gamma_{s+offset}]^2)` for multiple loop sizes `[2, 4, 8]` and orientations.

The commutator `[Gamma_s, Gamma_{s+offset}]` measures the mismatch between transporting a vector one way around a loop vs another -- this is the field strength (curvature enclosed by the loop). `Tr([A,B]^2)` is gauge-invariant: it doesn't depend on the choice of local coordinate frame. Different loop sizes capture structure at different scales.

**Why it matters**: GAP tells you "the average feature value is X". Holonomy tells you "there's a strong geometric twist at scale 4 here but the geometry is flat at scale 8" -- richer, gauge-invariant non-local information that cannot be captured by mean statistics.

#### Improvement 4: Parallel-Transport Residual

```
CliffordNetBlock:   x_out = x_prev + h_mix         plain addition

CliffordFieldBlock: x_out = Transport(x_prev, Gamma) + h_mix
```

**What changes**: `ParallelTransportLayer` maps the skip connection through the learned gauge connection before addition: `x' = x - 0.1 * Gamma @ x + correction`.

**Why it matters**: In curved representation space, vectors at block input and output live in different tangent spaces -- you can't just add them. Parallel transport maps `x_prev` into the frame where `h_mix` lives. Without it, the skip connection carries a vector in the "wrong frame" -- a small but systematic error that compounds over depth. The transport uses the same connection `Gamma` already computed for the context stream, so overhead is minimal.

#### Improvement 5: Manifold Stress Anomaly Detection (model-level)

```
CliffordNet:        output = logits

CliffordFieldNet:   output = {"logits": ..., "stress": ..., "anomaly_mask": ...}
                    (when use_anomaly_detection=True)
```

**What changes**: Optional `ManifoldStressLayer` at the model head computes per-position stress from curvature deviation, connection variation, and holonomy magnitude. Stress is pooled to a per-image scalar; an adaptive threshold produces an anomaly mask.

**Why it matters**: Inputs that violate learned manifold structure (adversarial perturbations, OOD samples, corrupted data) produce high stress because their local geometry doesn't match training distribution. This is a geometry-based anomaly score -- no separate OOD detector network needed. Pure addition, no architectural changes to blocks.

#### Improvement 6: Gauge-Invariant Attention (optional, off by default)

```
CliffordNetBlock:   (no attention -- by design)

CliffordFieldBlock: attn = GaugeInvariantAttention(x, curvature, Gamma)
                    g_feat += sigmoid_gate * attn
                    (when use_gauge_attention=True)
```

**What changes**: `GaugeInvariantAttention` computes attention scores using gauge-invariant quantities: connection differences (holonomy-based), curvature similarity, and geodesic distance -- not raw Q*K dot products. A sigmoid gate controls how much attention contributes.

**Why it matters**: The standard block is purely local (7x7 RF from DWConv + holonomy at various scales). Gauge attention adds O(S^2) long-range dependencies where the attention pattern is geometrically meaningful. Off by default because it (a) breaks the attention-free philosophy and (b) adds O(S^2) cost. Enable for tasks where long-range geometric dependencies matter.

### Full CliffordFieldBlock Data Flow

```
X_prev (B, H, W, D)
  |
  v
curvature = Dense(x, tanh) * 0.1               <-- learned curvature estimator
  |
  v
FieldNormalization(x, curvature) --> X_norm     <-- [1] curvature-aware norm
  |
  +--> Dense(D) ---------> Z_det                    detail stream
  |
  +--> DWConv --> DWConv --> BN --> SiLU --> Z_ctx   context stream
       |
       (if diff: Z_ctx -= Z_det)
       |
       v
  Gamma = ConnectionLayer(Z_det, curvature)     <-- [2] gauge connection
  Z_ctx = Z_ctx + 0.1 * Gamma @ Z_ctx               connection-guided transport
       |
       v
  SRGP(Z_det, Z_ctx) --> G_feat                     Clifford geometric product
       |
       + 0.1 * HolonomyLayer(X_norm, Gamma)     <-- [3] holonomy global context
       |
       (+ gate * GaugeAttention(X, curv, Gamma)) <-- [6] optional attention
       |
       v
  GGR(X_norm, G_feat) --> H_mix
       |
       v
  X_out = Transport(X_prev, Gamma) + H_mix      <-- [4] parallel-transport residual
```

Model head optionally adds ManifoldStressLayer [5] returning `{"logits", "stress", "anomaly_mask"}`.

### Key Design Principle: Shared Connection

The gauge connection `Gamma` is computed **once** per block and reused by four components: context transport, holonomy, residual transport, and (optionally) attention. This makes the overhead relatively efficient -- the `ConnectionLayer` computation is the main cost, and everything else shares it.

### Variants

| Variant | Channels | Depth | Shifts | Gauge Attention | Notes |
|:--------|:--------:|:-----:|:-------|:---------------:|:------|
| `nano` | 128 | 12 | [1, 2] | No | Smallest, all field improvements except attention |
| `lite` | 128 | 12 | [1, 2, 4, 8, 16] | No | More shifts, still attention-free |
| `base` | 192 | 16 | [1, 2, 4, 8, 16] | Yes (8 heads) | Full model with gauge attention |

All variants use Yang-Mills connections with 4 generators (nano/lite) or 8 generators (base), holonomy loop sizes `[2, 4, 8]`, and parallel-transport residuals.

### Configuration Options

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `use_holonomy_context` | `True` | Holonomy global context (Wilson loop features) |
| `use_parallel_transport_residual` | `True` | Transport skip connections through gauge connection |
| `use_gauge_attention` | `False` | Add gauge-invariant attention (O(S^2) cost) |
| `use_anomaly_detection` | `False` | Return stress/anomaly alongside logits |
| `connection_type` | `"yang_mills"` | Gauge connection type (`yang_mills`, `levi_civita`, `affine`) |
| `num_generators` | `4` | Lie algebra generators for Yang-Mills connection |
| `num_attention_heads` | `4` | Heads for gauge-invariant attention |
| `holonomy_loop_sizes` | `[2, 4, 8]` | Loop sizes for holonomy computation |
| `stress_types` | `["curvature", "connection", "combined"]` | Stress components for anomaly detection |

---

## 4. CliffordNetLM (Language)

Autoregressive language model using causal Clifford blocks.

### Architecture

```
Token IDs (B, seq_len)
  --> Embedding + Positional Embedding
  --> Reshape to (B, 1, seq_len, D)      H=1 for 2D conv compatibility
  --> L x CausalCliffordNetBlock          left-only padding, causal cumulative mean
  --> Squeeze to (B, seq_len, D)
  --> LayerNorm --> Dense
  --> {"logits": (B, seq_len, vocab_size)}
```

The causal block uses `padding="valid"` with explicit left-only zero-padding to enforce autoregressive causality. The global branch uses causal cumulative mean instead of GAP.

### Variants

| Variant | Channels | Depth | Shifts |
|:--------|:--------:|:-----:|:-------|
| `nano` | 128 | 12 | [1, 2] |
| `mini` | 192 | 12 | [1, 2, 4] |
| `base` | 384 | 18 | [1, 2, 4, 8, 16] |
| `large` | 512 | 20 | [1, 2, 4, 8, 16] |
| `xl` | 768 | 28 | [1, 2, 4, 8, 16] |

---

## 5. CliffordNetDenoiser

Bias-free image denoiser satisfying Miyasawa's theorem (1961).

All Dense/Conv layers use `use_bias=False`, and all normalizations use `center=False` (scale-only, no additive shift). This ensures the network acts as a score function estimator, suitable for diffusion models.

### Architecture

```
Input (B, H, W, C) in [-1, 1]
  --> BN(center=False) --> Conv2D(bias=False)
  --> L x BiasFreeClifordNetBlock
  --> LayerNorm(center=False)
  --> Conv2D(C, 1x1, bias=False) --> residual
  --> Output = Input + residual
```

### Variants

| Variant | Channels | Depth | Shifts | Global Context |
|:--------|:--------:|:-----:|:-------|:--------------:|
| `tiny` | 64 | 6 | [1, 2] | No |
| `small` | 96 | 8 | [1, 2, 4] | No |
| `base` | 128 | 12 | [1, 2, 4, 8] | No |
| `large` | 128 | 16 | [1, 2, 4, 8, 16] | Yes |

---

## 6. Core Primitives

All models share these building blocks from `layers/geometric/clifford_block.py`:

### SparseRollingGeometricProduct

Approximates the Clifford geometric product `AB = A . B + A ^ B` via cyclic channel shifts:

- **Wedge** (bivector): `Z_det * roll(Z_ctx, s) - Z_ctx * roll(Z_det, s)` -- antisymmetric outer product
- **Inner** (scalar): `SiLU(Z_det * roll(Z_ctx, s))` -- gated symmetric product

For each shift `s`, both components are computed and concatenated, then projected back to `D` dimensions. The `cli_mode` parameter selects `"inner"`, `"wedge"`, or `"full"` (both).

This is a *sparse approximation* -- not a genuine Cl(p,q) multivector representation with grade projections. Shifts are channel-space cyclic rolls, not algebraic basis elements.

### GatedGeometricResidual

Euler-discretized ODE update step:

```
gate = sigmoid(Dense(concat(h_norm, g_feat)))
h_mix = SiLU(h_norm) + gate * g_feat
h_mix = gamma * h_mix                        LayerScale, init 1e-5
h_mix = DropPath(h_mix)                       optional stochastic depth
```

The `gamma` (LayerScale) starts near zero so blocks contribute almost nothing initially, enabling stable deep training.

---

## 7. Quick Start

### CliffordNet (Vision)

```python
from dl_techniques.models.cliffordnet import CliffordNet

model = CliffordNet.nano(num_classes=100)
# or: CliffordNet.lite(num_classes=100)
# or: CliffordNet.lite_g(num_classes=100)  # with global context
```

### CliffordFieldNet (Vision + Field Theory)

```python
from dl_techniques.models.cliffordnet import CliffordFieldNet

# All field improvements, no attention
model = CliffordFieldNet.nano(num_classes=100)

# With gauge-invariant attention
model = CliffordFieldNet.base(num_classes=100)

# With anomaly detection
model = CliffordFieldNet.nano(num_classes=100, use_anomaly_detection=True)
result = model(images)  # {"logits": ..., "stress": ..., "anomaly_mask": ...}
```

### CliffordNetLM (Language)

```python
from dl_techniques.models.cliffordnet import CliffordNetLM

model = CliffordNetLM.nano(vocab_size=32000, max_seq_length=512)
result = model(token_ids)  # {"logits": (B, seq_len, vocab_size)}
```

### Custom Configuration

```python
model = CliffordFieldNet(
    num_classes=10,
    channels=192,
    depth=16,
    shifts=[1, 2, 4, 8, 16],
    use_holonomy_context=True,
    use_parallel_transport_residual=True,
    use_gauge_attention=True,
    num_attention_heads=8,
    connection_type="yang_mills",
    num_generators=8,
    use_anomaly_detection=True,
    stochastic_depth_rate=0.15,
    dropout_rate=0.1,
)
```
