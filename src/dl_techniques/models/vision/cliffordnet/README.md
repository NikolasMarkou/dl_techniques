# CliffordNet Family

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

Geometric-algebra-based neural network architectures. This package holds the vision classifier; the family also has a dual-encoder contrastive vision-language model in a sibling package.

Based on: **"CliffordNet: All You Need is Geometric Algebra"** (arXiv:2601.06793v2)

---

## Table of Contents

1. [Models](#1-models)
2. [CliffordNet (Vision)](#2-cliffordnet-vision)
3. [CliffordCLIP (Vision-Language)](#3-cliffordclip-vision-language)
4. [Core Primitives](#4-core-primitives)
5. [Architectural Compliance](#5-architectural-compliance)
6. [Quick Start](#6-quick-start)

---

## 1. Models

This package holds `model.py`, `hierarchical.py` and `__init__.py`:

| Model | Domain | File | Key Idea |
|:------|:-------|:-----|:---------|
| `CliffordNet` | Vision | `src/dl_techniques/models/vision/cliffordnet/model.py` | Attention-free backbone: geometric product replaces both attention and FFN |
| `HierarchicalCliffordNet` | Vision | `src/dl_techniques/models/vision/cliffordnet/hierarchical.py` | The same block in stages at falling resolution, a stride-2 downsampler between them |

`model.py` also holds what the two models share and `hierarchical.py` imports: the stem
(`build_clifford_stem`), the head configuration (`clifford_head_config`), the rate and
`include_top` checks (`validate_rates`, `coerce_include_top`), and the constants
`DEFAULT_KERNEL_INITIALIZER`, `LAYERNORM_EPSILON`, `STEM_BN_MOMENTUM`, `STEM_BN_EPSILON` and
`DEFAULT_INPUT_SHAPE`.

A second family member lives in a sibling package and is documented here because it
shares the same algebraic core:

| Model | Domain | File | Key Idea |
|:------|:-------|:-----|:---------|
| `CliffordCLIP` | Vision-Language | `src/dl_techniques/models/vision_language/clifford_clip/model.py` | Dual-encoder contrastive model with Clifford-aware projection head |

**Export surface.** `src/dl_techniques/models/vision/cliffordnet/__init__.py` exports exactly
four names — `CliffordNet`, `create_cliffordnet`, `HierarchicalCliffordNet` and
`create_hierarchical_cliffordnet`. That `__all__` is pinned by
`tests/test_models/test_cliffordnet/test_model.py`.

The causal language model `CliffordNetLM` is documented in [`language/cliffordlm/README.md`](../../language/cliffordlm/README.md).

All models share the same algebraic core: `SparseRollingGeometricProduct` and `GatedGeometricResidual` from `src/dl_techniques/layers/geometric/clifford_block.py`.

---

## 2. CliffordNet (Vision)

The standard isotropic backbone. Replaces both attention and FFN with a single Clifford geometric product pathway.

### Architecture

```
Input (B, H, W, C)
  --> Patch stem (ConvStem: Conv2D + BN per conv)
  --> L x CliffordNetBlock     isotropic; channels constant throughout
  --> GlobalAveragePool
  --> LayerNorm --> [Dropout] --> Dense(num_classes)
  --> logits (B, num_classes)
```

Each `CliffordNetBlock` contains no FFN. The dual-stream detail/context pipeline followed by a sparse Clifford geometric product is the entire non-linear interaction.

**Patch stem.** One `ConvStem` (`dl_techniques/layers/conv_blocks/stem.py`) named
`stem`, chosen by `patch_size`, with every convolution followed by its own batch
norm (momentum 0.9, epsilon 1e-3; the layers are `stem_conv1`, `stem_norm1`,
`stem_act1`, `stem_conv2`, `stem_norm2`):

| `patch_size` | Layers | Bias |
|:------------:|:-------|:-----|
| 1 | conv3 `D/2` stride 1, BN, SiLU, conv3 `D`, BN | none |
| 2 | conv3 `D` stride 2, BN | `use_bias` |
| 4 | conv3 `D/2` stride 2, BN, SiLU, conv3 `D` stride 2, BN | none |
| any other `p` | conv `p` `D` stride `p`, BN | `use_bias` |

The model's `kernel_initializer`, `bias_initializer`, `kernel_regularizer` and
`bias_regularizer` reach the stem convolutions (a bias-free convolution has no
bias to initialise or regularise). The epsilon 1e-3 is the Keras default the
stem had before it moved onto `ConvStem`, kept as that pre-migration value; no
paper value for it was found (the norm factory's own default is 1e-6). In the
two-conv stems the second convolution draws its initial kernel independently of
the first. The stem's weights live under `stem/...`, so `.keras` archives saved
before the stem moved onto `ConvStem` (`stem_conv/...`) do not load. The
hierarchical model's stem is the same layer for patch 1 and 2.

The head is the reference implementation's (`ParaMind2025/CAN` `model.py`:
`x.mean(dim=[-2, -1])`, `nn.LayerNorm(embed_dim)`, `nn.Linear(embed_dim,
num_classes)`), built as the shared `ClassificationHead` with no hidden block: a
`pooled_norm` LayerNorm at torch's default epsilon `1e-5`
(`LAYERNORM_EPSILON`, a module constant of `cliffordnet/model.py`), the dropout when `dropout_rate > 0`, and the
linear `classifier`. The model's `kernel_initializer`, `bias_initializer`,
`kernel_regularizer` and `bias_regularizer` reach the classifier, and
`head_config_overrides` is merged into the head config (`name`, `num_classes` and
`task_type` raise `ValueError`).

**The model returns logits, one `(B, num_classes)` tensor, with no softmax.**
Compile with `keras.losses.SparseCategoricalCrossentropy(from_logits=True)`.
Until 2026-10-14 `call` returned the head's `{'logits', 'probabilities'}` dict,
whose `'logits'` held a softmax, and the head carried two hidden blocks. That
change breaks archives: a `.keras` file saved before it does not load, and is
not supported. `head_config_preset`, which nothing read, is gone; passing it
raises `ValueError`.

**`include_top=False` returns the feature map.** `CliffordNet(include_top=True)`
is the default and is the model described above. With `include_top=False` the
head is not built, `model.classification_head` is `None`, and `call` returns the
block stack's output `(B, ceil(H / p), ceil(W / p), channels)` with `p =
patch_size`: the map itself, not a pooled vector (pool it yourself for an
embedding). `num_classes` is then ignored, so `0` is accepted; it defaults to
`1000` as for ResNet. A headed checkpoint loads into a headless model through
`load_pretrained_weights` (on a built model): the trunk layers (`stem`, `block_*`) are
copied and the checkpoint's `classification_head` is left unused.
`HierarchicalCliffordNet` has the same switch (see below).

**`include_top` is the last named constructor parameter.** It follows
`head_config_overrides`, so `CliffordNet(10, 64)` binds `64` to `channels` and
`HierarchicalCliffordNet(10, [1, 1])` binds the list to `stage_depths`. It must
be a real `bool` or a `numpy.bool_` (stored as a Python `bool`); anything else
(`64`, `[1, 1]`, `1`, `None`) raises `TypeError`, naming the value and its
module-qualified type.

**`from_variant(variant, num_classes=1000, input_shape=None, pretrained=False, **kwargs)`
is the one constructor path** (the same as ResNet's), for both models and for
`create_cliffordnet` and `create_hierarchical_cliffordnet`, whose defaults are the same.
An unknown variant raises `ValueError` listing the variants; `pretrained=True` raises
`NotImplementedError` naming the variant (no weights are distributed); `pretrained=<path>`
loads a local `.keras` file through the shared loader, which skips a layer whose shapes differ
(a checkpoint with another `num_classes` loads its trunk and leaves the new head at its
initialization), and raises when it restores no layer. There are no `nano()`, `lite()` or
`lite_g()` shortcuts: use `from_variant("nano", ...)`.

**`from_variant(..., input_shape=...)` builds the model.** The constructors take no
`input_shape`, so when one `(H, W, C_in)` (unbatched) is given, `from_variant` returns the
model built at it, with or without `pretrained`; ResNet, by contrast, only stores its
`input_shape`. With `pretrained=<path>` and no `input_shape` the loader builds at
`DEFAULT_INPUT_SHAPE`, `(32, 32, 3)` (a public constant of `cliffordnet/model.py`, also used by
`hierarchical.py`; only `C_in` matters, weight shapes do not depend on `H` and `W`) and then
transfers the weights. With neither, the model stays unbuilt until its first call or
`build((None, H, W, C_in))`; `build` takes the batched shape only.

**Initializers are stored as specs.** `kernel_initializer` and `bias_initializer` may be a
name, a config dict or an `Initializer` instance; the model stores a name or a config dict (an
instance is serialized, so `get_config()` is JSON-safe) and hands that spec to the stem, to
every block and to the head, so each layer draws its own kernel. Before, one instance reached
all of them and replayed one random draw (666 of 1891 kernel pairs at `nano`, 820 of 2211 at
`hier_345`), and the presets' `TruncatedNormal(0.02)` was created at import time with a seed
fixed then. Trained weights compute the same function; only the random starting weights
differ. `dropout_rate` and `stochastic_depth_rate` must lie in `[0, 1)` and the model says so
itself, naming its own argument.

**Layer names and old archives.** Blocks are `block_{i}` and `block_{i}_drop_path` (0-based),
hierarchical blocks are `stage{S}_block{B}` and `stage{S}_block{B}_drop_path` (both 1-based) and
its downsamplers `downsample_{i}_depthwise`, `downsample_{i}_norm` and `downsample_{i}_pointwise`
(`i` is 0-based). Before they were `clifford_block_{i}` and `clifford_drop_path_{i}`,
`stage{s}_block_{j}` and `stage{s}_drop_path_{j}` (0-based, underscore) and `downsample_{i}_bn`.
Only the `w.path` strings change; `.keras` archives load by the order of the tracked layers,
which is unchanged. Measured by saving archives in a separate process from a checkout of
`be92a74eb` and loading them into the current code: a `.keras` archive of every CliffordNet patch size 1 to 4 and the nano preset, and
of every `HierarchicalCliffordNet` patch size 1 and 2 and `hier_345`, headed and headless (14
archives), loads into the current code through `keras.models.load_model` and through
`load_weights`, with outputs equal at `atol=0` in inference and in `training=True`. Archives
saved before the stem moved onto `ConvStem` (`stem_conv/...`) still do not load.

### Variants

| Variant | Channels | Depth | Patch | Shifts | cli_mode | Params (paper) | Params (100 classes) |
|:--------|:--------:|:-----:|:-----:|:-------|:--------:|:------:|-------:|
| `nano` | 128 | 12 | 2 | [1, 2] | full | 1.43M | 1,436,516 |
| `lite` | 128 | 12 | 2 | [1, 2, 4, 8, 16] | full | 2.61M | 2,616,164 |
| `lite_g` | 128 | 12 | 2 | [1, 2, 4, 8, 16] + global | full | 3.40M | 3,404,132 |
| `nano_g` | 128 | 12 | 2 | [1, 2] + global | full | 2.22M | 2,224,484 |
| `cn_12_3` | 160 | 12 | 1 | [1, 2, 4] | full | not reported | 2,953,300 |
| `cn_18_5` | 128 | 18 | 2 | [1, 2, 4, 8, 16] | full | not reported | 3,915,620 |
| `cn_32_3` | 128 | 32 | 2 | [1, 2, 4] | full | 4.8M (CliffordNet-32) | 4,850,532 |
| `cn_32_5` | 128 | 32 | 2 | [1, 2, 4, 8, 16] | inner | not reported | 4,326,244 |
| `cn_64_5` | 128 | 64 | 2 | [1, 2, 4, 8, 16] | inner | 8.6M (CliffordNet-64) | 8,635,236 |

The paper column is Ji 2026, arXiv:2601.06793v2, tables 2 and 3; the paper
truncates its figures (1,436,516 is printed 1.43M). `cn_<depth>_<shifts>` are the
factories of the authors' `networks.py` (github.com/ParaMind2025/CAN), read
through a fetch summary, not byte-checked; the authors train `cn_64_5` with drop
path 0.4 and the others with 0.3. `cn_12_3` has `patch_size=1`, whose stem was
not compared with the authors'. All counts are pinned by
`tests/test_models/test_cliffordnet/test_paper_v2_variants.py`. The last column
is `count_params()`, re-derived with:

```python
from dl_techniques.models.vision.cliffordnet import CliffordNet
for v in CliffordNet.MODEL_VARIANTS:
    m = CliffordNet.from_variant(v, num_classes=100)
    m.build((None, 32, 32, 3))
    print(v, m.count_params())
```

The global branch of `lite_g` and `nano_g` was checked against the authors'
`model.py`: the spatial mean of the normalised input goes through the same
interaction (full, `diff`, shifts [1, 2]) and is added with weight 1, as here. The
paper's text writes `C_glo = GlobalAvg(H)` and leaves the `diff` subtraction
implicit.

### Hierarchical variant (`hierarchical.py`)

`HierarchicalCliffordNet` arranges the same `CliffordNetBlock` in stages instead of
one isotropic stack: stage `i` runs `stage_depths[i]` blocks at width
`stage_dims[i]`, and a downsampler between stages halves the resolution and moves
to the next width (depthwise 3x3 stride 2, batch norm, SiLU, 1x1 convolution). It
follows `model_hier.py` of the authors' repository (github.com/ParaMind2025/CAN),
read through a fetch summary and not byte-checked.

| Preset | Stages (blocks) | Widths | Patch | Shifts | cli_mode | Params (100 classes) |
|:-------|:---------------:|:------:|:-----:|:-------|:--------:|-------:|
| `hier_345` | 3, 4, 5 | 32, 64, 128 | 1 | [1, 2] | full | 767,476 |

`hier_345` is the authors' constructor default. Stage 1 runs at full input
resolution (patch size 1), so a 32x32 image goes 32, 16, 8. The drop-path ramp runs
linearly over all 12 blocks, not per stage; the authors' default rate is 0.1. The
count is a regression pin (`tests/test_models/test_cliffordnet/test_hierarchical.py`),
not a paper value: no accuracy and no parameter count of this model is published,
and the paper v2 text has no hierarchical results.

Ported: the patch-1 and patch-2 stems, the `conv` downsampler, the pooled-LayerNorm
head, `get_config()`. Left out on purpose: `embed_dim` other than `stage_dims[0]`,
the `dim_policy` switch, the authors' `patch` and `avgpool` downsample modes, patch
sizes other than 1 and 2, `enable_cuda`, and a global branch (the authors'
hierarchical model has none).

```python
from dl_techniques.models.vision.cliffordnet import HierarchicalCliffordNet
model = HierarchicalCliffordNet.from_variant("hier_345", num_classes=100)
features = model.extract_stage_features(images)   # one feature map per stage

# Backbone only: the last stage's map (B, h, w, stage_dims[-1]), no head.
backbone = HierarchicalCliffordNet.from_variant("hier_345", include_top=False)
last_stage = backbone(images)

# A headed checkpoint into a headless model; builds at input_shape first.
backbone = HierarchicalCliffordNet.from_variant(
    "hier_345", include_top=False, pretrained="path/to/weights.keras",
    input_shape=(32, 32, 3),
)
```

`include_top=False` builds no head, sets `model.classification_head` to `None`
and returns the last stage's feature map, `ceil(size / 2)` per downsampler after
the stem, not a pooled vector. `num_classes` is then ignored (`0` is accepted) and
defaults to `1000`, as does `create_hierarchical_cliffordnet`.
The headed default is unchanged (the 767,476 pin holds). A headed checkpoint
loads into a headless model: the trunk (`stem`, `stage*_block*`,
`downsample_*`) is copied and the checkpoint's `classification_head` is unused.

The two-hidden-block head this replaced read 1,502,948 / 2,682,596 / 3,470,564,
which round to 1.5M and 2.7M, not the paper's figures.

---

## 3. CliffordCLIP (Vision-Language)

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

| Variant  | Vision stage channels / depths | Text ch / depth / shifts | embed_dim | Params |
|:---------|:-------------------------------|:-------------------------|:---------:|-------:|
| `nano`   | 128,128,256,256 / 3,3,3,3      | 128 / 12 / [1,2]         | 256       | 13,671,361 |
| `nano_g` | as `nano`, + gFFN-G            | 128 / 12 / [1,2]         | 256       | 15,639,745 |
| `mini`   | 192,192,384,384 / 3,3,3,3      | 192 / 12 / [1,2,4]       | 384       | 28,955,041 |
| `small`  | 192,192,384,384 / 4,4,4,3      | 192 / 15 / [1,2,4]       | 384       | 32,264,929 |
| `base`   | 256,256,512,512 / 4,4,4,4      | 256 / 12 / [1,2,4]       | 512       | 54,632,321 |
| `large`  | 384,384,768,768 / 5,5,5,5      | 384 / 16 / [1,2,4,8]     | 768       | 152,394,049 |

Params is `count_params()` of `CliffordCLIP.from_variant(v, vocab_size=50257,
image_size=112, context_length=64, vision_patch_size=4)` after one call (the
trainer's defaults; the GPT-2 token embedding alone is `50257 * text_channels`,
6.4M at 128 channels). Measured 2026-10-20, after the two final projections
became bias-free. The vision shifts per stage are in `CliffordCLIP.MODEL_VARIANTS`.

`nano` and `nano_g` are depth/shift-aligned with the `nano` variant of `CliffordNet` and with the `nano` variant of the Clifford language model in `language/cliffordlm/` on every shared axis (channels=128, depth=12, shifts=[1,2]) so the CLIP backbone can be ablated against the vanilla classifier / LM at matched capacity. `nano_g` adds a global-context branch on the vision tower only, mirroring the `lite_g` variant of `CliffordNet`; the text side stays `use_global_context=False` because the language model's variant ladder has no global-context entry.

Both towers also expose an optional pre-projection `head_dropout` gated on `dropout_rate > 0`, placed after `*_head_norm` and before the projection Dense -- the same hook `CliffordNet.head_dropout` (on the pooled vector) and the language model's `head_dropout` (on the `(B, L, D)` sequence) provide. `dropout_rate=0.0` skips the sublayer entirely.

`small` is not parameter-matched to a ViT-CLIP of the same width. Measured with the same vocabulary (GPT-2 BPE, 50,257 tokens), 112 px and context 64: CliffordCLIP-small has 32,264,929 parameters with 15+15 blocks; this repo's `CLIP` at width 192, 12+12 layers, 3 heads, `embed_dim=384`, patch 16 has 23,734,657. Small has 25% more blocks per tower and 36% more parameters.

---

## 4. Core Primitives

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

## 5. Architectural Compliance

Every model in the family exploits the core properties of the Clifford geometric product and introduces no component (FFN, attention) that would undermine them.

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

| Check | CliffordNet | CliffordCLIP |
|:------|:-----------:|:------------:|
| Uses `SparseRollingGeometricProduct` | Yes (backbone) | Yes (backbone + head) |
| Dual-stream (detail + context) | Yes | Yes (both towers) |
| GGR + LayerScale (gamma=1e-5) | Yes | Yes (backbone + head residual) |
| No FFN | Yes | Yes |
| No attention | Yes | Yes |
| Residual = X_prev + GGR output | Yes | Yes |
| Shifts filtered with warning | Yes | Yes (+ `_head_shifts_for` validation) |
| Causality preserved | N/A | Text tower: causal; vision: bidirectional |
| Clifford algebra in projection head | N/A | `SparseRollingGeometricProduct` + `LayerScale` (CHANNEL) residual |
| Serialization round-trip | `get_config` / `from_config` | `get_config` / `from_config` (+ `LayerScale`, `AttentionPooling`) |

The language model's own compliance row set is in [`language/cliffordlm/README.md`](../../language/cliffordlm/README.md).

### Design decisions and their rationale

1. **No FFN anywhere.** The geometric product's second-order multiplicative terms combined with SiLU gating in the GGR already provide the non-linear channel mixing that FFNs exist to supply. Adding an FFN would double the parameter count per block without increasing algebraic expressivity. Both models here honour this.

2. **CliffordCLIP uses the geometric product in the projection head, not just the backbone.** A plain CLIP head (GAP -> Dense -> cosine) collapses the bivector content the backbone computed. The `learned_query_residual` head runs two pooled views (GAP + learned-query attention pool) through a `SparseRollingGeometricProduct` and injects the result as a LayerScale-gated residual on top of the canonical CLIP anchor. This keeps the contrastive loss pathway Clifford-algebra-aware end-to-end. An A/B sweep on CC3M (12,500 steps, 5 variants) confirmed this head matches or beats the plain baseline on 5/6 retrieval metrics.

3. **Global-context branch uses hardcoded shifts=[1, 2] and cli_mode='full'.** The global branch summarises whole-image or whole-sequence statistics via GAP (vision) or causal cumulative mean (text). It deliberately decouples its hyperparameters from the local branch because the global view operates at a different spatial scale and does not need the same shift set.

4. **CliffordCLIP-small (15+15 depth) is deeper than a ViT-CLIP at 192ch/12L, not parameter-matched to it.** Measured at the trainer's defaults it has 32.3M parameters against 23.7M (see Variants above): 25% more blocks per tower and 36% more parameters. Each Clifford block has no FFN and no QKV projections. This trades single-layer receptive field (7x7 DWConv vs global attention) for deeper compositional feature extraction.

---

## 6. Quick Start

### CliffordNet (Vision)

```python
import keras
from dl_techniques.models.vision.cliffordnet import CliffordNet, create_cliffordnet

model = CliffordNet.from_variant("nano", num_classes=100)
# or: CliffordNet.from_variant("lite", num_classes=100)
# or: CliffordNet.from_variant("lite_g", num_classes=100)  # with global context
# or the module-level factory, which delegates to CliffordNet.from_variant:
model = create_cliffordnet("lite", num_classes=100)

# The model returns logits: the loss applies the softmax.
model.compile(
    optimizer="adamw",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Backbone only: the final feature map (B, H/p, W/p, channels), no head.
backbone = CliffordNet.from_variant("nano", include_top=False)
features = backbone(images)
```

### CliffordCLIP (Vision-Language)

```python
from dl_techniques.models.vision_language.clifford_clip.model import CliffordCLIP

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
| CLIP contrastive pre-training | `src/train/cliffordnet/train_clip.py` |
| Downsampling-variant ablation | `src/train/cliffordnet/train_downsampling_techniques.py` |

See `src/train/cliffordnet/README.md` for the full protocol, flags and results.
