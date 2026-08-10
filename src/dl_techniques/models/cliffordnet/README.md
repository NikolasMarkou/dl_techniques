# CliffordNet Family

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

Geometric-algebra-based neural network architectures. The surviving family is a vision classifier, a causal language model, and a dual-encoder contrastive vision-language model.

Based on: **"CliffordNet: All You Need is Geometric Algebra"** (arXiv:2601.06793v2)

---

## Table of Contents

1. [Models](#1-models)
2. [CliffordNet (Vision)](#2-cliffordnet-vision)
3. [CliffordNetLM (Language)](#3-cliffordnetlm-language)
4. [CliffordCLIP (Vision-Language)](#4-cliffordclip-vision-language)
5. [Core Primitives](#5-core-primitives)
6. [Architectural Compliance](#6-architectural-compliance)
7. [Quick Start](#7-quick-start)

---

## 1. Models

This package contains exactly two modules plus its `__init__.py`:

| Model | Domain | File | Key Idea |
|:------|:-------|:-----|:---------|
| `CliffordNet` | Vision | `src/dl_techniques/models/cliffordnet/model.py` | Attention-free backbone: geometric product replaces both attention and FFN |
| `CliffordNetLM` | NLP | `src/dl_techniques/models/cliffordnet/lm.py` | Autoregressive LM with causal Clifford blocks |

A third family member lives in a sibling package and is documented here because it
shares the same algebraic core:

| Model | Domain | File | Key Idea |
|:------|:-------|:-----|:---------|
| `CliffordCLIP` | Vision-Language | `src/dl_techniques/models/clip/clifford_clip.py` | Dual-encoder contrastive model with Clifford-aware projection head |

**Export surface.** `src/dl_techniques/models/cliffordnet/__init__.py` exports exactly
two names — `CliffordNet` and `create_cliffordnet`. `CliffordNetLM` is **not**
re-exported; import it from `dl_techniques.models.cliffordnet.lm` directly. That
`__all__` is pinned by `tests/test_models/test_cliffordnet/test_model.py`.

**Removed 2026-08-10.** The denoiser (`CliffordNetDenoiser`), the conditional and
confidence-interval denoisers, the Laplacian autoencoder (`CliffordLaplacianUNet`),
the routing LM (`CliffordNetLMRouting`), the U-Net depth/segmentation model
(`CliffordNetUNet`, `create_cliffordnet_depth`), the bidirectional embedding U-Net
(`CliffordNetEmbedding`) and the LM U-Net (`CliffordNetLMUNet`) were all deleted
together with the strided `CliffordNetBlockDSv2` / `CausalCliffordNetBlockDSv2`
blocks they were built on. Do not re-document them; recover them from git history
if they are ever needed.

All models share the same algebraic core: `SparseRollingGeometricProduct` and `GatedGeometricResidual` from `src/dl_techniques/layers/geometric/clifford_block.py`.

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

## 4. CliffordCLIP (Vision-Language)

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

| Variant  | Vision ch/depth/shifts | Text ch/depth/shifts | embed_dim | Params |
|:---------|:-----------------------|:---------------------|:---------:|-------:|
| `nano`   | 128 / 12 / [1,2]       | 128 / 12 / [1,2]     | 256       | ~9.5M  |
| `nano_g` | 128 / 12 / [1,2] +gFFN-G | 128 / 12 / [1,2]   | 256       | ~10.3M |
| `mini`   | 192 / 12 / [1,2,4]     | 192 / 12 / [1,2,4]   | 384       | ~18M   |
| `small`  | 192 / 15 / [1,2,4]     | 192 / 15 / [1,2,4]   | 384       | ~20M   |
| `base`   | 256 / 16 / [1,2,4,8]   | 256 / 12 / [1,2,4]   | 512       | ~33M   |
| `large`  | 384 / 20 / [1,2,4,8,16]| 384 / 16 / [1,2,4,8] | 768       | ~120M  |

`nano` and `nano_g` are depth/shift-aligned with `CliffordNet.nano` and `CliffordNetLM.from_variant("nano", ...)` on every shared axis (channels=128, depth=12, shifts=[1,2]) so the CLIP backbone can be ablated against the vanilla classifier / LM at matched capacity. `nano_g` adds a global-context branch on the vision tower only, mirroring `CliffordNet.lite_g`; the text side stays `use_global_context=False` because `CliffordNetLM` has no global-context variant in its ladder.

Both towers also expose an optional pre-projection `head_dropout` gated on `dropout_rate > 0`, placed after `*_head_norm` and before the projection Dense -- the same hook `CliffordNet.head_dropout` (on the pooled vector) and `CliffordNetLM.head_dropout` (on the `(B, L, D)` sequence) provide. `dropout_rate=0.0` skips the sublayer entirely.

The `small` variant is parameter-matched to ViT-CLIP at the same channel width (192) and vocabulary (GPT-2 BPE, 50,257 tokens). ViT-CLIP at 192ch/12 layers has ~20.4M params; CliffordCLIP-small reaches the same count with 15+15 depth instead of 12+12 -- 25% more layers at the same budget because each Clifford block carries no FFN and no QKV projections.

---

## 5. Core Primitives

All models share these building blocks from `src/dl_techniques/layers/geometric/clifford_block.py`:

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

## 6. Architectural Compliance

Every model in the family was audited against the core properties of the Clifford geometric product. The audit verifies that each model fully exploits the algebra's strengths and does not introduce components (FFN, attention) that would undermine those properties.

### What the Clifford block provides

| Property | Mechanism | Why it matters |
|:---------|:----------|:---------------|
| **Algebraic completeness** | Wedge (antisymmetric, bivector) + inner (symmetric, scalar) computed simultaneously | Captures both coherence and structural variation in a single operator; standard dot-product attention sees only the scalar part |
| **Information preservation** | The geometric product is the only standard algebraic product over vectors that is fully invertible | No information-lossy bottleneck (unlike attention's softmax or FFN's ReLU/GeLU truncation) |
| **FFN redundancy** | Second-order multiplicative terms + SiLU in the GGR gate already provide non-linear channel mixing | No separate FFN needed; the entire FFN parameter budget (typically 4x channel expansion) is eliminated |
| **Identity-start training** | GGR LayerScale gamma initialises to 1e-5; blocks start as near-identity residuals | Deep stacking (20+ blocks) is stable from step 1 without warmup hacks |
| **Linear sequence cost** | O(N) via depthwise convolution, not O(N^2) via attention | Long sequences (LM) and high-resolution images don't hit a quadratic wall |
| **Dual-stream structure** | Detail (1x1 Dense) and context (DWConv x2, effective 7x7 RF) generate complementary views | The geometric product needs two distinct inputs; feeding the same signal to both would collapse the wedge to zero |

### Per-model compliance

| Check | CliffordNet | CliffordNetLM | CliffordCLIP |
|:------|:-----------:|:--------------:|:------------:|
| Uses `SparseRollingGeometricProduct` | Yes (backbone) | Yes (backbone) | Yes (backbone + head) |
| Dual-stream (detail + context) | Yes | Yes (causal DWConv) | Yes (both towers) |
| GGR + LayerScale (gamma=1e-5) | Yes | Yes | Yes (backbone + head residual) |
| No FFN | Yes | Yes | Yes |
| No attention | Yes | Yes | Yes |
| Residual = X_prev + GGR output | Yes | Yes | Yes |
| Shifts filtered with warning | Yes | Yes | Yes (+ `_head_shifts_for` validation) |
| Causality preserved | N/A | Left-only DWConv padding + causal cumulative mean | Text tower: causal; vision: bidirectional |
| Clifford algebra in projection head | N/A | N/A | `SparseRollingGeometricProduct` + `LearnableMultiplier` (CHANNEL) residual |
| Serialization round-trip | `get_config` / `from_config` | `get_config` / `from_config` | `get_config` / `from_config` (+ `LearnableMultiplier`, `AttentionPooling`) |

### Design decisions and their rationale

1. **No FFN anywhere.** The geometric product's second-order multiplicative terms combined with SiLU gating in the GGR already provide the non-linear channel mixing that FFNs exist to supply. Adding an FFN would double the parameter count per block without increasing algebraic expressivity. All three models honour this.

2. **CliffordCLIP uses the geometric product in the projection head, not just the backbone.** A plain CLIP head (GAP -> Dense -> cosine) collapses the bivector content the backbone computed. The `learned_query_residual` head runs two pooled views (GAP + learned-query attention pool) through a `SparseRollingGeometricProduct` and injects the result as a LayerScale-gated residual on top of the canonical CLIP anchor. This keeps the contrastive loss pathway Clifford-algebra-aware end-to-end. An A/B sweep on CC3M (12,500 steps, 5 variants) confirmed this head matches or beats the plain baseline on 5/6 retrieval metrics.

3. **Global-context branch uses hardcoded shifts=[1, 2] and cli_mode='full'.** The global branch summarises whole-image or whole-sequence statistics via GAP (vision) or causal cumulative mean (text). It deliberately decouples its hyperparameters from the local branch because the global view operates at a different spatial scale and does not need the same shift set.

4. **CliffordCLIP-small (15+15 depth) is parameter-matched to ViT-CLIP at 192ch/12L.** At the same 20.4M parameter budget, CliffordCLIP fits 25% more layers (15 vs 12 per tower) because each block has no FFN and no QKV projections. This trades single-layer receptive field (7x7 DWConv vs global attention) for deeper compositional feature extraction, which is the natural scaling axis of the Clifford architecture.

---

## 7. Quick Start

### CliffordNet (Vision)

```python
from dl_techniques.models.cliffordnet import CliffordNet, create_cliffordnet

model = CliffordNet.nano(num_classes=100)
# or: CliffordNet.lite(num_classes=100)
# or: CliffordNet.lite_g(num_classes=100)  # with global context
# or the module-level factory, which delegates to CliffordNet.from_variant:
model = create_cliffordnet("lite", num_classes=100)
```

### CliffordNetLM (Language)

`CliffordNetLM` is not re-exported from the package `__init__`, and it has no
per-variant classmethods — use `from_variant`:

```python
from dl_techniques.models.cliffordnet.lm import CliffordNetLM

model = CliffordNetLM.from_variant("nano", vocab_size=32000, max_seq_length=512)
result = model(token_ids)  # {"logits": (B, seq_len, vocab_size)}
```

### CliffordCLIP (Vision-Language)

```python
from dl_techniques.models.clip.clifford_clip import CliffordCLIP

model = CliffordCLIP.from_variant(
    "small", vocab_size=50257, image_size=112, context_length=64,
)
out = model({"image": images, "text": tokens})
# out keys: image_features, text_features, logits_per_image,
#           logits_per_text, logit_scale
# Default head_kind="learned_query_residual" -- Clifford-aware end-to-end
```

### Training

No public pretrained weights are distributed for any model in this family. The
training and inference entry points are:

| Task | Script |
|:-----|:-------|
| CIFAR-10/100 classification | `src/train/cliffordnet/train_cliffordnet.py` |
| Causal LM pre-training | `src/train/cliffordnet/train_cliffordnet_nlp.py` |
| Text generation / power sampling | `src/train/cliffordnet/infer_cliffordnet_nlp.py` |
| CLIP contrastive pre-training | `src/train/cliffordnet/train_clip.py` |
| Downsampling-variant ablation | `src/train/cliffordnet/train_downsampling_techniques.py` |

See `src/train/cliffordnet/README.md` for the full protocol, flags and results.
