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
7. [Architectural Compliance](#7-architectural-compliance)
8. [Quick Start](#8-quick-start)

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

| Variant | Vision ch/depth/shifts | Text ch/depth/shifts | embed_dim | Params |
|:--------|:-----------------------|:---------------------|:---------:|-------:|
| `nano`  | 128 / 8 / [1,2]        | 128 / 8 / [1,2]      | 256       | ~8.5M  |
| `mini`  | 192 / 12 / [1,2,4]     | 192 / 12 / [1,2,4]   | 384       | ~18M   |
| `small` | 192 / 15 / [1,2,4]     | 192 / 15 / [1,2,4]   | 384       | ~20M   |
| `base`  | 256 / 16 / [1,2,4,8]   | 256 / 12 / [1,2,4]   | 512       | ~33M   |
| `large` | 384 / 20 / [1,2,4,8,16]| 384 / 16 / [1,2,4,8] | 768       | ~120M  |

The `small` variant is parameter-matched to ViT-CLIP at the same channel width (192) and vocabulary (GPT-2 BPE, 50,257 tokens). ViT-CLIP at 192ch/12 layers has ~20.4M params; CliffordCLIP-small reaches the same count with 15+15 depth instead of 12+12 -- 25% more layers at the same budget because each Clifford block carries no FFN and no QKV projections.

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

## 7. Architectural Compliance

Every model in the family was audited against the core properties of the Clifford geometric product. The audit verifies that each model fully exploits the algebra's strengths and does not introduce components (FFN, attention, biased operations in the denoiser) that would undermine those properties.

### What the Clifford block provides

| Property | Mechanism | Why it matters |
|:---------|:----------|:---------------|
| **Algebraic completeness** | Wedge (antisymmetric, bivector) + inner (symmetric, scalar) computed simultaneously | Captures both coherence and structural variation in a single operator; standard dot-product attention sees only the scalar part |
| **Information preservation** | The geometric product is the only standard algebraic product over vectors that is fully invertible | No information-lossy bottleneck (unlike attention's softmax or FFN's ReLU/GeLU truncation) |
| **FFN redundancy** | Second-order multiplicative terms + SiLU in the GGR gate already provide non-linear channel mixing | No separate FFN needed; the entire FFN parameter budget (typically 4x channel expansion) is eliminated |
| **Identity-start training** | GGR LayerScale gamma initialises to 1e-5; blocks start as near-identity residuals | Deep stacking (20+ blocks) is stable from step 1 without warmup hacks |
| **Linear sequence cost** | O(N) via depthwise convolution, not O(N^2) via attention | Dense prediction (denoising, segmentation) and long sequences (LM) don't hit a quadratic wall |
| **Dual-stream structure** | Detail (1x1 Dense) and context (DWConv x2, effective 7x7 RF) generate complementary views | The geometric product needs two distinct inputs; feeding the same signal to both would collapse the wedge to zero |

### Per-model compliance

| Check | CliffordNet | CliffordNetLM | CliffordNetDenoiser | CliffordCLIP |
|:------|:-----------:|:--------------:|:-------------------:|:------------:|
| Uses `SparseRollingGeometricProduct` | Yes (backbone) | Yes (backbone) | Yes (backbone, bias-free) | Yes (backbone + head) |
| Dual-stream (detail + context) | Yes | Yes (causal DWConv) | Yes (bias-free DWConv) | Yes (both towers) |
| GGR + LayerScale (gamma=1e-5) | Yes | Yes | Yes (bias-free variant) | Yes (backbone + head residual) |
| No FFN | Yes | Yes | Yes | Yes |
| No attention | Yes | Yes | Yes | Yes |
| Residual = X_prev + GGR output | Yes | Yes | Yes (X_in + f(X_in)) | Yes |
| Shifts filtered with warning | Yes | Yes | Yes | Yes (+ `_head_shifts_for` validation) |
| Causality preserved | N/A | Left-only DWConv padding + causal cumulative mean | N/A | Text tower: causal; vision: bidirectional |
| Bias-free constraint | N/A | N/A | All Dense/Conv `use_bias=False`; all norms `center=False` | N/A |
| Clifford algebra in projection head | N/A | N/A | N/A | `SparseRollingGeometricProduct` + `_LayerScale1D` residual |
| Serialization round-trip | `get_config` / `from_config` | `get_config` / `from_config` | `get_config` / `from_config` | `get_config` / `from_config` (+ `_LayerScale1D`, `_LearnedQueryPool1D`) |

### Design decisions and their rationale

1. **No FFN anywhere.** The geometric product's second-order multiplicative terms combined with SiLU gating in the GGR already provide the non-linear channel mixing that FFNs exist to supply. Adding an FFN would double the parameter count per block without increasing algebraic expressivity. All four models honour this.

2. **Denoiser enforces bias-free end-to-end.** Miyasawa's theorem (1961) requires that a least-squares denoiser have zero-mean output, implying no additive bias in the network. `BiasFreeClifordNetBlock` passes `use_bias=False` through every Dense, Conv2D, and GGR gate layer; all norms use `center=False` (scale-only, no shift). LayerScale gamma is multiplicative, not additive, so it is compatible with the constraint.

3. **CliffordCLIP uses the geometric product in the projection head, not just the backbone.** A plain CLIP head (GAP -> Dense -> cosine) collapses the bivector content the backbone computed. The `learned_query_residual` head runs two pooled views (GAP + learned-query attention pool) through a `SparseRollingGeometricProduct` and injects the result as a LayerScale-gated residual on top of the canonical CLIP anchor. This keeps the contrastive loss pathway Clifford-algebra-aware end-to-end. An A/B sweep on CC3M (12,500 steps, 5 variants) confirmed this head matches or beats the plain baseline on 5/6 retrieval metrics.

4. **Global-context branch uses hardcoded shifts=[1, 2] and cli_mode='full'.** The global branch summarises whole-image or whole-sequence statistics via GAP (vision) or causal cumulative mean (text). It deliberately decouples its hyperparameters from the local branch because the global view operates at a different spatial scale and does not need the same shift set.

5. **CliffordCLIP-small (15+15 depth) is parameter-matched to ViT-CLIP at 192ch/12L.** At the same 20.4M parameter budget, CliffordCLIP fits 25% more layers (15 vs 12 per tower) because each block has no FFN and no QKV projections. This trades single-layer receptive field (7x7 DWConv vs global attention) for deeper compositional feature extraction, which is the natural scaling axis of the Clifford architecture.

---

## 8. Quick Start

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
    "small", vocab_size=50257, image_size=112, context_length=64,
)
out = model({"image": images, "text": tokens})
# out keys: image_features, text_features, logits_per_image,
#           logits_per_text, logit_scale
# Default head_kind="learned_query_residual" -- Clifford-aware end-to-end
```
