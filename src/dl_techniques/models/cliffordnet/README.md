# CliffordNet Family

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

Geometric-algebra-based neural network architectures. The family includes a vision classifier, a causal language model, an image denoiser, and a dual-encoder contrastive vision-language model.

Based on: **"CliffordNet: All You Need is Geometric Algebra"** (arXiv:2601.06793v2)

---

## Table of Contents

1. [Models](#1-models)
2. [CliffordNet (Vision)](#2-cliffordnet-vision)
3. [CliffordNetLM (Language)](#3-cliffordnetlm-language)
4. [CliffordNetDenoiser](#4-cliffordnetdenoiser)
5. [CliffordCLIP (Vision-Language)](#5-cliffordclip-vision-language)
6. [Core Primitives](#6-core-primitives)
7. [Quick Start](#7-quick-start)

---

## 1. Models

| Model | Domain | File | Key Idea |
|:------|:-------|:-----|:---------|
| `CliffordNet` | Vision | `model.py` | Attention-free backbone: geometric product replaces both attention and FFN |
| `CliffordNetLM` | NLP | `lm.py` | Autoregressive LM with causal Clifford blocks |
| `CliffordNetDenoiser` | Vision | `denoiser.py` | Bias-free denoiser satisfying Miyasawa's theorem |
| `CliffordCLIP` | Vision-Language | `clip.py` | Dual-encoder contrastive model with Clifford-aware projection head |

All models share the same algebraic core: `SparseRollingGeometricProduct` and `GatedGeometricResidual` from `layers/geometric/clifford_block.py`.

---

## 2. CliffordNet (Vision)

The standard isotropic backbone. Replaces both attention and FFN with a single Clifford geometric product pathway.

### Architecture

```
Input (B, H, W, C)
  --> Patch stem Conv2D + BN
  --> L x CliffordNetBlock     isotropic; channels constant throughout
  --> GlobalAveragePool
  --> LayerNorm --> Dense(num_classes)
```

Each `CliffordNetBlock` contains no FFN. The dual-stream detail/context pipeline followed by a sparse Clifford geometric product is the entire non-linear interaction.

### Variants

| Variant | Channels | Depth | Shifts | Params |
|:--------|:--------:|:-----:|:-------|:-------|
| `nano` | 128 | 12 | [1, 2] | ~1.4M |
| `lite` | 128 | 12 | [1, 2, 4, 8, 16] | ~2.6M |
| `lite_g` | 128 | 12 | [1, 2, 4, 8, 16] + global | ~3.4M |

---

## 3. CliffordNetLM (Language)

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

## 4. CliffordNetDenoiser

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

## 5. CliffordCLIP (Vision-Language)

Dual-encoder CLIP-style contrastive model. Both towers are built from Clifford blocks, and the projection head itself is Clifford-aware so the contrastive loss sees explicit bivector (structural) content -- not just the scalar coherence term.

### Architecture (default `head_kind="learned_query_residual"`)

```
Image (B, H, W, 3)                      Tokens (B, seq_len)
  |                                           |
  v                                           v
Conv2D patch stem + BN                  Token + Position Embedding
  |                                           |
  v                                           v
L x CliffordNetBlock                    L x CausalCliffordNetBlock
  |                                           |
  v                                           v
z_det   = GAP(x)               (B,D)    z_anchor = last-non-pad(x)   (B,D)
z_ctx   = LearnedQueryPool(x)  (B,D)    z_det    = masked-mean(x)    (B,D)
                                        z_ctx    = LearnedQueryPool(x,mask)
  |                                           |
  v                                           v
geo = SparseRollingGeometricProduct(z_det, z_ctx)       # wedge + inner
  |                                           |
  v                                           v
mixed = z_det   + gamma_v * geo         mixed = z_anchor + gamma_t * geo
                   (init 1e-5)                             (init 1e-5)
  |                                           |
  v                                           v
LayerNorm --> Dense(embed_dim)          Dense(embed_dim)
  |                                           |
  v                                           v
L2 Normalize                            L2 Normalize
  |                                           |
  +----------------> cos_sim * exp(logit_scale) <--------+
                          |
                          v
                symmetric contrastive CE
```

### Why the Clifford projection head?

Plain CLIP pools the backbone output to a single vector and uses cosine similarity. Cosine captures only the *scalar* (coherence) term of the geometric product. The *bivector* (structural) term, which is half of the algebraic signal the Clifford blocks compute, is thrown away.

The default Clifford projection head (`head_kind="learned_query_residual"`) runs the canonical CLIP anchor (GAP for vision, last-non-pad-token for text) through a LayerScale-gated residual path that adds a Clifford geometric product of (anchor, learned-query-pool) on top. LayerScale γ initialises to 1e-5, so the head starts out behaving like plain CLIP and only introduces wedge/inner content where it measurably helps — mirroring the GGR pattern used inside the Clifford backbone itself.

Three other head variants are kept for A/B comparisons (`plain`, `mean_max`, `learned_query`, plus `learned_query_residual` with `cli_mode=wedge`). See `src/train/cliffordnet/README.md` for the full sweep table on CC3M-smoke at 12,500 steps; the residual variant is the empirical winner.

This design preserves:
- O(|S| · D) parameter cost (no O(D²) full bivector tensor),
- The existing `SparseRollingGeometricProduct` primitive (no new math),
- Standard cosine-similarity contrastive loss (no loss changes),
- Backwards compatibility with plain-CLIP behaviour at initialisation.

### Variants

| Variant | Vision ch/depth/shifts | Text ch/depth/shifts | embed_dim |
|:--------|:-----------------------|:---------------------|:---------:|
| `nano`  | 128 / 8 / [1,2]        | 128 / 8 / [1,2]      | 256       |
| `mini`  | 192 / 12 / [1,2,4]     | 192 / 12 / [1,2,4]   | 384       |
| `base`  | 256 / 16 / [1,2,4,8]   | 256 / 12 / [1,2,4]   | 512       |
| `large` | 384 / 20 / [1,2,4,8,16]| 384 / 16 / [1,2,4,8] | 768       |

---

## 6. Core Primitives

All models share these building blocks from `layers/geometric/clifford_block.py`:

### SparseRollingGeometricProduct

Approximates the Clifford geometric product `AB = A . B + A ^ B` via cyclic channel shifts:

- **Wedge** (bivector): `Z_det * roll(Z_ctx, s) - Z_ctx * roll(Z_det, s)` -- antisymmetric outer product
- **Inner** (scalar): `SiLU(Z_det * roll(Z_ctx, s))` -- gated symmetric product

For each shift `s`, both components are computed and concatenated, then projected back to `D` dimensions. The `cli_mode` parameter selects `"inner"`, `"wedge"`, or `"full"` (both).

This is a *sparse approximation* -- not a genuine Cl(p,q) multivector representation with grade projections. Shifts are channel-space cyclic rolls, not algebraic basis elements. The signature is hardcoded Euclidean (Cl(D, 0)).

Shifts `s >= channels` are filtered out at construction (a full cyclic roll carries no new information). If all supplied shifts are filtered out, the constructor raises -- you cannot silently degrade to a no-op block.

### GatedGeometricResidual

Euler-discretized ODE update step:

```
gate = sigmoid(Dense(concat(h_norm, g_feat)))
h_mix = SiLU(h_norm) + gate * g_feat
h_mix = gamma * h_mix                        LayerScale, init 1e-5
h_mix = DropPath(h_mix)                       optional stochastic depth
```

The `gamma` (LayerScale) starts near zero so blocks contribute almost nothing initially, enabling stable deep training.

### CliffordNetBlock / CausalCliffordNetBlock

Full isotropic / causal vision-and-sequence blocks composed of the primitives above with a dual-stream (detail via 1x1 Dense; context via stacked DWConv + BN + SiLU), optional differential context subtraction, optional global-context branch, and a `GatedGeometricResidual` at the residual junction.

**Note on the global-context branch:** when `use_global_context=True`, the global `SparseRollingGeometricProduct` uses fixed `shifts=[1, 2]`, `cli_mode='full'` and differential context regardless of the caller's `shifts` / `cli_mode` / `ctx_mode` settings. This is intentional (the global branch only needs to summarise whole-image or whole-sequence statistics), but it does mean block-level settings do not propagate to the global branch.

---

## 7. Quick Start

### CliffordNet (Vision)

```python
from dl_techniques.models.cliffordnet import CliffordNet

model = CliffordNet.nano(num_classes=100)
# or: CliffordNet.lite(num_classes=100)
# or: CliffordNet.lite_g(num_classes=100)  # with global context
```

### CliffordNetLM (Language)

```python
from dl_techniques.models.cliffordnet import CliffordNetLM

model = CliffordNetLM.nano(vocab_size=32000, max_seq_length=512)
result = model(token_ids)  # {"logits": (B, seq_len, vocab_size)}
```

### CliffordNetDenoiser

```python
from dl_techniques.models.cliffordnet import CliffordNetDenoiser

model = CliffordNetDenoiser.base(image_channels=3)
noise_pred = model(noisy_images)  # (B, H, W, C)
```

### CliffordCLIP (Vision-Language)

```python
from dl_techniques.models.cliffordnet import CliffordCLIP

model = CliffordCLIP.from_variant(
    "nano", vocab_size=100352, image_size=96, context_length=64,
)
out = model({"image": images, "text": tokens})
# out keys: image_features, text_features, logits_per_image,
#           logits_per_text, logit_scale
```
