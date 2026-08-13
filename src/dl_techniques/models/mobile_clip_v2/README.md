# MobileCLIP2: A Faithful FastViT (MCi) Port

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A channels-last Keras 3 transcription of **MobileCLIP2** (Faghri et al., 2025,
arXiv:2508.20691) — Apple's dual-encoder vision-language model pairing a FastViT
**MCi** image tower with an OpenCLIP text transformer.

This package is **architecture only**. No pretrained weights are ported, and it
makes no accuracy claim. See [§15](#15-deviations-from-the-reference-implementation)
before quoting it against any published number.

---

## Table of Contents

1. [Overview: What MobileCLIP2 Is and Why It Matters](#1-overview-what-mobileclip2-is-and-why-it-matters)
2. [The Problem MobileCLIP2 Solves](#2-the-problem-mobileclip2-solves)
3. [How It Works: Core Concepts](#3-how-it-works-core-concepts)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Component Reference](#6-component-reference)
7. [Configuration & Model Variants](#7-configuration--model-variants)
8. [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9. [Advanced Usage Patterns](#9-advanced-usage-patterns)
10. [Performance Optimization](#10-performance-optimization)
11. [Training and Best Practices](#11-training-and-best-practices)
12. [Serialization & Deployment](#12-serialization--deployment)
13. [Testing & Validation](#13-testing--validation)
14. [Troubleshooting & FAQs](#14-troubleshooting--faqs)
15. [Deviations from the reference implementation](#15-deviations-from-the-reference-implementation)
16. [Relationship to `models/mobile_clip/` (v1)](#16-relationship-to-modelsmobile_clip-v1)
17. [Technical Details](#17-technical-details)
18. [Citation](#18-citation)

---

## 1. Overview: What MobileCLIP2 Is and Why It Matters

MobileCLIP2 is a CLIP-style dual encoder built for on-device inference. Its
image branch is not a scaled-down ViT: it is **FastViT MCi**, a hybrid tower
whose shallow stages mix tokens with a *convolutional* RepMixer and whose deepest
one or two stages use global self-attention, at a resolution where the token
count has already collapsed to 64 (or 16).

Three facts distinguish this port from a generic CLIP implementation:

* **The image tower's terminal `Dense` IS the CLIP image projection.** MobileCLIP's
  open_clip configs set `"timm_pool": "avg"` with `"timm_proj": null`. In
  open_clip's `TimmModel`, a non-attention pool asserts that the trunk itself does
  the projecting and instantiates the trunk with `num_classes=embed_dim`. There is
  no separate projection layer, and stacking one on top would be a second,
  unfaithful projection.
* **The stochastic-depth ramp is global.** One linear ramp is computed across
  every block of every stage and then *sliced* stagewise. A fresh per-stage ramp
  produces an identically-shaped, identically-parameterized, subtly-wrong model.
* **The variant table deliberately holds two families.** The four `mobileclip2_s*`
  rows are **non-causal** (their JSON configs say `"no_causal_mask": true`); the
  two earlier `mobileclip_s3` / `mobileclip_s4` rows are causal. Same image
  backbones, different text-tower attention.

## 2. The Problem MobileCLIP2 Solves

Standard CLIP's ViT image encoder dominates on-device cost: it must run on every
frame, at full resolution, with attention over hundreds of tokens. The text
encoder, by contrast, usually runs once over a small fixed prompt set and can be
comparatively heavy.

MobileCLIP exploits that asymmetry. The image tower is convolutional where tokens
are numerous (stride 4 through stride 16) and only becomes attentional once the
feature map is 8x8 or 4x4. The text tower stays a conventional transformer.

```
                     tokens at 256 px input
  stage 0 (64x64)  ->  4096   RepMixer   (depthwise conv, O(N))
  stage 1 (32x32)  ->  1024   RepMixer
  stage 2 (16x16)  ->   256   RepMixer
  stage 3 ( 8x8 )  ->    64   Attention  (O(N^2) is affordable here)
  stage 4 ( 4x4 )  ->    16   Attention  (mci3 / mci4 only)
```

Global self-attention never sees more than 64 tokens.

## 3. How It Works: Core Concepts

### Dual encoder + contrastive objective

Both towers project into a shared `embed_dim` space. Features are L2-normalized,
scaled by a learnable temperature, and compared with a symmetric similarity
matrix.

```
{'image': (B,256,256,3)}          {'text': (B,77)}
        |                                |
 FastVitImageEncoder            MobileClipTextEncoder
        |                                |
   (B, embed_dim)                  (B, embed_dim)
        |                                |
   L2 normalize                     L2 normalize
        \________________  _______________/
                         \/
       scale = clip(exp(logit_scale), 0, 100)
                         |
   {'image_features', 'text_features',
    'logits_per_image', 'logits_per_text', 'logit_scale'}
```

The clip on `exp(logit_scale)` is load-bearing, not cosmetic: an unbounded
temperature turns a diverging run into `inf` logits and a `nan` loss with no
other observable symptom.

### RepMixer — the convolutional token mixer

```
x -> x + gamma * (mixer(x) - norm(x))
```

`mixer` is a depthwise `MobileOneBlock` with its activation removed; `norm`
degenerates to a *single* BatchNorm (a `MobileOneBlock` with zero conv branches,
no scale branch, and only its identity branch surviving). `gamma` is a per-channel
LayerScale initialized at `1e-5`. At `gamma == 0` the mixer is the exact identity.

### Structural reparameterization is *not* implemented

FastViT's "Rep" prefix refers to fusing the multi-branch train-time blocks into a
single convolution at inference. This port ships the train-time multi-branch form
only. See deviation **X-1**.

## 4. Architecture Deep Dive

### Image tower (`FastVitImageEncoder`)

```
image (B, H, W, 3)
    |
stem: 3 x MobileOneBlock       k3/s2 dense, k3/s2 depthwise, k1/s1     -> /4
    |
stage_0 .. stage_{N-1}         FastVitStage
    |                          (downsample? -> RepCPE? -> depth x token mixer)
final_conv: MobileOneBlock     k3, depthwise, SE, -> embed_dims[-1] * cls_ratio
    |
GlobalAveragePooling2D
    |
Dropout(head_dropout_rate)
    |
Dense(projection_dim)          <- THIS IS THE CLIP IMAGE PROJECTION
    |
(B, projection_dim)
```

Per-stage geometry at 256 px (the MobileCLIP fastvit input resolution):

| Variant family | Spatial size after stem, then per stage |
|---|---|
| 4-stage (mci0 / mci1 / mci2) | `64 -> 64 -> 32 -> 16 -> 8` |
| 5-stage (mci3 / mci4)        | `64 -> 64 -> 32 -> 16 -> 8 -> 4` |

Stage 0 never downsamples — the stem has already done the /4.

### MCi image-tower table

Fields common to every row and therefore not tabulated: `cls_ratio=2.0`,
`down_patch_size=7`, `down_stride=2`, `repmixer_kernel_size=3`,
`layer_scale_init_value=1e-5`, `head_dim=32`, activation GELU. RepCPE is present
in a stage iff that stage's `pos_embs` entry is not `None`.

| Backbone | `layers` | `embed_dims` | `mlp_ratios` | token mixers | `se_downsamples` | `pos_embs` | stem scale branch | attention norm |
|---|---|---|---|---|---|---|---|---|
| `mci0` | `(2, 6, 10, 2)`     | `(64, 128, 256, 512)`        | 3.0 | rm, rm, rm, attn      | `(F, F, T, T)`    | `(-, -, -, 7x7)`     | yes | `batch_norm` |
| `mci1` | `(4, 12, 20, 4)`    | `(64, 128, 256, 512)`        | 3.0 | rm, rm, rm, attn      | `(F, F, T, T)`    | `(-, -, -, 7x7)`     | yes | `batch_norm` |
| `mci2` | `(4, 12, 24, 4)`    | `(80, 160, 320, 640)`        | 3.0 | rm, rm, rm, attn      | `(F, F, T, T)`    | `(-, -, -, 7x7)`     | yes | `batch_norm` |
| `mci3` | `(2, 12, 24, 4, 2)` | `(96, 192, 384, 768, 1536)`  | 4.0 | rm, rm, rm, attn, attn | all `False`      | `(-, -, -, 7x7, 7x7)` | no  | `layer_norm` |
| `mci4` | `(2, 12, 24, 4, 4)` | `(128, 256, 512, 1024, 2048)`| 4.0 | rm, rm, rm, attn, attn | all `False`      | `(-, -, -, 7x7, 7x7)` | no  | `layer_norm` |

`rm` = `repmixer`. **`mci0` / `mci1` / `mci2` have no local oracle** — see
deviation **X-3**.

### Text tower

`MobileClipTextEncoder`, imported unchanged from `models/mobile_clip/components.py`
(see [§16](#16-relationship-to-modelsmobile_clip-v1)): token embedding +
learned positional embedding -> `text_layers` x `TransformerLayer` ->
`LayerNormalization` -> EOT-token extraction -> projection to `embed_dim`.
`intermediate_size` is `4 * text_width` in every shipped variant.

## 5. Quick Start Guide

```python
import numpy as np
from dl_techniques.models.mobile_clip_v2 import MobileClipV2Model

model = MobileClipV2Model.from_variant('mobileclip2_s0')

out = model(
    {
        'image': np.zeros((2, 256, 256, 3), dtype='float32'),
        'text': np.zeros((2, 77), dtype='int32'),
    },
    training=False,           # explicit: see the FAQ on training=None
)

sorted(out)
# ['image_features', 'logit_scale', 'logits_per_image',
#  'logits_per_text', 'text_features']
out['logits_per_image'].shape   # (2, 2)
```

> **Always pass `training=False` explicitly for a deterministic forward.**
> `StochasticDepth` short-circuits to the identity only when `training is False`;
> at `training=None` it runs its stochastic path, and every BatchNormalization in
> the tower uses batch statistics and updates its moving averages.

## 6. Component Reference

Public surface (`from dl_techniques.models.mobile_clip_v2 import ...`):

| Symbol | Kind | What it is |
|---|---|---|
| `MobileClipV2Model` | `keras.Model` | The dual encoder. |
| `create_mobile_clip_v2` | function | Factory over `MODEL_VARIANTS`. |
| `MODEL_VARIANTS` | dict | The six variant rows (§7). |
| `FastVitImageEncoder` | `keras.Model` | The image tower, usable standalone. |
| `create_fastvit_image_encoder` | function | Factory over `MCI_VARIANTS`. |
| `MCI_VARIANTS` | dict | The five MCi backbone rows (§4). |

`MobileClipV2Model` methods beyond the Keras surface:

| Method | Purpose |
|---|---|
| `from_variant(variant, **kwargs)` | Build from a `MODEL_VARIANTS` key; kwargs override the row. |
| `encode_image(image, normalize=True, training=None)` | Image branch only. |
| `encode_text(text, normalize=True, training=None)` | Text branch only. |
| `compute_logit_scale()` | `clip(exp(logit_scale), 0, logit_scale_max)`. |

The eight FastViT primitives the image tower is built from live in
`src/dl_techniques/layers/fastvit/` and are documented in that package's own
README, including its `RepMixerBlock` name-collision warning (deviation **X-5**).

## 7. Configuration & Model Variants

Every row shares `vocab_size=49408` (OpenAI BPE), `context_length=77`,
`image_size=256`, and `text_intermediate = 4 * text_width`. `use_causal_mask` is
the **negation** of the JSON config's `no_causal_mask` field.

| Variant | `embed_dim` | `image_backbone` | `text_width` | `text_heads` | `text_layers` | `use_causal_mask` |
|---|---|---|---|---|---|---|
| `mobileclip2_s0` | 512 | `mci0` | 512 | 8  | 12 | `False` |
| `mobileclip2_s2` | 512 | `mci2` | 512 | 8  | 12 | `False` |
| `mobileclip2_s3` | 768 | `mci3` | 768 | 12 | 12 | `False` |
| `mobileclip2_s4` | 768 | `mci4` | 768 | 12 | 12 | `False` |
| `mobileclip_s3`  | 768 | `mci3` | 768 | 12 | 12 | `True`  |
| `mobileclip_s4`  | 768 | `mci4` | 768 | 12 | 12 | `True`  |

Each row is keyed by the name of the supplied JSON config file it transcribes,
and `test_model_variants_match_supplied_json_configs` checks it field by field
against the committed copy of that file at
`research/mobileclip2_reference/model_configs/` — so this table is verified, not
re-derived.

To count parameters for a variant, derive the number rather than trusting a
number written here:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -c "
from dl_techniques.models.mobile_clip_v2 import MobileClipV2Model
m = MobileClipV2Model.from_variant('mobileclip2_s0')
m.build({'image': (None,256,256,3), 'text': (None,77)})
print(m.count_params())
"
```

## 8. Comprehensive Usage Examples

### Zero-shot classification

```python
import numpy as np
from keras import ops
from dl_techniques.models.mobile_clip_v2 import MobileClipV2Model

model = MobileClipV2Model.from_variant('mobileclip2_s0')

# `prompts` is (num_classes, 77) int32 — tokenize with the OpenAI BPE
# tokenizer of your choice (this package ships no tokenizer).
class_embeddings = model.encode_text(prompts, training=False)      # (C, D)
image_embedding = model.encode_image(images, training=False)       # (B, D)

logits = model.compute_logit_scale() * ops.matmul(
    image_embedding, ops.transpose(class_embeddings)
)                                                                  # (B, C)
predictions = ops.argmax(logits, axis=-1)
```

Both `encode_*` methods L2-normalize by default. Leave `normalize=True` for
anything that feeds a similarity or a contrastive loss.

### The image tower on its own

```python
from dl_techniques.models.mobile_clip_v2 import create_fastvit_image_encoder

# CLIP embedding (the terminal Dense IS the projection)
tower = create_fastvit_image_encoder('mci0', projection_dim=512)

# Pooled backbone features instead, for dense / transfer use — NOT for CLIP
backbone = create_fastvit_image_encoder('mci0', projection_dim=None)
```

### A reduced-depth tower for tests

Full `mci4` at 256 px does not fit comfortably alongside a training job on a
12 GB card. `image_encoder_kwargs` is forwarded verbatim to
`FastVitImageEncoder`:

```python
model = MobileClipV2Model.from_variant(
    'mobileclip2_s3',
    image_encoder_kwargs={'layers': (1, 1, 1, 1, 1)},
)
```

## 9. Advanced Usage Patterns

### Naming and warm-starting

`IMAGE_TOWER_NAME` and `TEXT_TOWER_NAME` are module constants
(`"image_encoder"` / `"text_encoder"`). `load_weights_from_checkpoint` matches
layers **by name**, so a tower that is ever warm-started independently must be
named identically in every model that holds it. Do not rename the towers.

### Substituting a tower

Pass an already-constructed `image_encoder=` / `text_encoder=` to the
constructor. This is the route `from_config` uses, and it is why a
reduced-depth or otherwise-overridden tower round-trips as itself rather than
being re-derived from the scalar fields.

Towers are **never** substituted after construction: Keras refuses a post-build
sub-layer swap, and a pre-build one leaves the discarded tower's variables
reachable through tracking.

### Overriding the temperature

```python
model = MobileClipV2Model.from_variant(
    'mobileclip2_s0',
    logit_scale_init=0.0,     # raw log temperature; default is log(1 / 0.07)
    logit_scale_max=50.0,
)
```

## 10. Performance Optimization

* **Attention cost is already bounded** by the architecture — at most 64 tokens.
  The convolutional stages dominate runtime; reduce `layers` / `embed_dims`
  before reaching for attention tricks.
* **No fused inference path exists** (deviation **X-1**). If you need one,
  implement it as an explicit, separately tested conversion pass over a trained
  model; do not silently change the blocks' `call()` paths.
* **Text embeddings for a fixed prompt set are constant.** Compute
  `encode_text` once and cache it; only `encode_image` needs to run per frame.
* **`head_dropout_rate` is inert at the reference settings** (`timm_drop` is
  `0.0` in all four MobileCLIP configs).

## 11. Training and Best Practices

Use stock `fit()` with `CLIPContrastiveLoss` from `dl_techniques.losses` — this
repo deliberately avoids custom `train_step` implementations.

```python
from dl_techniques.losses import CLIPContrastiveLoss

model.compile(optimizer='adamw', loss=CLIPContrastiveLoss())
```

Notes:

* The model returns a **dict**, not a tensor. Wire the loss against the key your
  training script actually consumes.
* `drop_path_rate` is the **maximum** of one global linear ramp over every block
  of every stage. Setting it per stage is not expressible and would not
  reproduce the reference.
* `dropout_rate` is threaded to both towers; `attention_dropout_rate` reaches
  the text tower only.

## 12. Serialization & Deployment

Every class carries `@keras.saving.register_keras_serializable()`, a complete
`get_config`, and a `compute_output_shape`, so the standard round trip works:

```python
model.save('mobileclip2_s0.keras')
restored = keras.models.load_model('mobileclip2_s0.keras')
```

Two implementation details matter if you subclass or extend:

* `get_config` serializes **both towers explicitly** (not merely their variant
  names), so a checkpoint keeps describing the network it was trained with even
  if a variant table is later corrected. `from_config` re-materializes them with
  `deserialize_keras_object`.
* `get_build_config` / `build_from_config` are overridden because Keras' generic
  implementation cannot round-trip a **dict** input-shape spec — it warns that
  the model cannot be built automatically and leaves the restored model unbuilt.

## 13. Testing & Validation

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
    tests/test_models/test_mobile_clip_v2/ tests/test_layers/test_fastvit/ -q
```

The suites cover initialization, invalid-config raises, forward shape,
`compute_output_shape` pre- and post-build, `.keras` save->load compared **by
value** (`atol=1e-6, rtol=0`), an **elementwise** weight check on the list-held
per-block sub-layers (counts, variable paths and parameter totals can all match
while the restored kernels are fresh), gradient flow to every trainable weight,
and per-class behavioural pins that were each proven RED against a deliberately
broken variant.

## 14. Troubleshooting & FAQs

**My forward pass is not reproducible.** Pass `training=False` explicitly.
`StochasticDepth` short-circuits only on `training is False`, and `training=None`
also puts every BatchNormalization into batch-statistics mode.

**Should I add a projection head on top of the image tower?** No. The tower's
terminal `Dense` already is the CLIP image projection. Use `projection_dim=None`
only when you want pooled backbone features for a non-CLIP purpose.

**`ValueError: All per-stage tuples must have one entry per stage`.** A 4-stage
vs 5-stage mixup. `mci0`/`mci1`/`mci2` need 4 entries in every per-stage tuple;
`mci3`/`mci4` need 5.

**Can I load an official MobileCLIP2 checkpoint?** No. No weights are ported
(deviation **X-4**), and no conversion script exists here.

**Why does `FastVitRepMixerBlock` not match `RepMixerBlock`?** They are different
architectures that share a name. See deviation **X-5**.

## 15. Deviations from the reference implementation

Every known divergence from the reference is listed here with a stable id. The
same list, with its full rationale, is in this plan's decision log; a deviation
that is silently absorbed makes the port unauditable.

### X-1 — No structural reparameterization

Train-time multi-branch form only. There is no `reparameterize()` / conv-fusion
path, and none is tested. The reference release passes `inference_mode=False`
everywhere, i.e. it too runs the multi-branch form, so this port matches what the
reference actually executes.

**No claim of numerical identity to a fused form is made or tested here.** An
earlier version of this section claimed the fused form is "mathematically
identical", making X-1 "a speed deviation, not an accuracy one". That was wrong
as shipped: fusing parallel branches requires them to sample the same input
pixels, and under Keras' asymmetric `padding='same'` the strided `k x k` and
`1 x 1` branches did not — MEASURED, a one-pixel offset at stride 2. The padding
convention has since been fixed to the reference's symmetric `kernel_size // 2`
(see the `layers/fastvit/README.md` deviation list), so the branches **are** now
fusible in principle; but no fusion code exists, so "the fused form is identical"
remains an unexecuted claim and is not asserted anywhere.

### X-2 — One `use_bias` for both qkv and the output projection

The shared `MultiHeadAttention` exposes a single `use_bias` covering **both** the
qkv projection and the output projection. timm's `Attention` is `qkv_bias=False`
with a **biased** output projection.

This port sets `use_bias=False`, so **each attention block is short exactly one
bias vector of length `dim`**. `use_bias=True` would instead add a spurious
`3 * dim` qkv bias the reference does not have, so `False` is the closer of the
two available settings; a third option would require editing the shared
`MultiHeadCrossAttention`, which is out of scope.

MEASURED: the attention sub-layer has exactly **2 weights, both kernels, zero
biases**. Pinned by `test_attention_has_no_bias_weights` so it cannot drift
silently.

### X-3 — `mci0` / `mci1` / `mci2` have no local oracle

> **Narrowed on 2026-08-14.** `mci3` / `mci4` now have a REAL committed oracle
> (see below). X-3 applies to `mci0` / `mci1` / `mci2` only.

Those three stage tables are transcribed from **timm upstream**
(`timm/models/fastvit.py`, fetched 2026-08-13), **not** from the supplied
`mobileclip2.py`, which defines only `fastvit_mci3` and `fastvit_mci4`. `timm` is
**not installed** in this environment, so nothing here can check those three rows
against timm itself — only against a second hand transcription of the same fetch.

`mci3` / `mci4` **do** have a real oracle. The upstream `mobileclip2.py` and the
six open_clip JSON configs are committed verbatim at
[`research/mobileclip2_reference/`](../../../../research/mobileclip2_reference/),
and two tests read those files directly rather than restating them:

| Test | Oracle | Covers |
| --- | --- | --- |
| `test_image_encoder.py::test_mci3_mci4_match_supplied_source` | `mobileclip2.py`, parsed with `ast` | `MCI_VARIANTS['mci3']`, `['mci4']`, every field |
| `test_model.py::test_model_variants_match_supplied_json_configs` | the six `model_configs/*.json`, via `json.load` | all six `MODEL_VARIANTS` rows + the shared constants |

The source is PARSED, not imported: it is PyTorch/`timm` code and this
environment has neither. Both tests were RED-proven by perturbing a value in the
committed reference files and confirming the failure names the field.

This does **not** extend to `mci0` / `mci1` / `mci2`. The supplied source does
not define them, and the JSON configs merely NAME `fastvit_mci0` /
`fastvit_mci2` without giving their architecture — so those three rows remain
transcription-only, exactly as stated above. Vendoring `timm` is the only thing
that would change that.

Anyone changing an `mci0`/`mci1`/`mci2`
row must re-derive it from timm upstream and say so — and must not reason from
the `mci3`/`mci4` rows, which differ structurally (5 stages, no SE, LayerNorm,
`mlp_ratio` 4).

### X-4 — No pretrained weights

The model is architecture-only and makes **no accuracy claim**. Any comparison
against published MobileCLIP2 numbers would be invalid by construction.

### X-5 — `RepMixerBlock` is a different architecture that shares the name

`src/dl_techniques/layers/repmixer_block.py::RepMixerBlock` is **not** FastViT's
RepMixer block. It has no LayerScale, no stochastic depth, a different token
mixer, and a 1x1 ConvFFN instead of a depthwise 7x7 ConvMlp. It is consumed by
`models/fastvlm/` and was deliberately left untouched — substituting the FastViT
block would change a shipped model's semantics and its checkpoint layout.

**Which one do you want?**

* Building a faithful FastViT / MobileCLIP2 tower, or anything that must
  correspond block-for-block to timm's `fastvit.py` — use
  `dl_techniques.layers.fastvit.FastVitRepMixerBlock`.
* Touching `models/fastvlm/` or loading one of its checkpoints — that model
  consumes `layers/repmixer_block.py::RepMixerBlock`. Leave it alone.

The same disambiguation applies to `layers/repmixer_block.py::ConvolutionalStem`,
which is likewise not the FastViT stem.

## 16. Relationship to `models/mobile_clip/` (v1)

`models/mobile_clip/` is **deliberately non-faithful**. Its own in-file decision
D-001 records that it substitutes `keras.applications` MobileNetV2 / MobileNetV3
backbones for the real MCi towers. This package is the faithful port.

**v1 is not deprecated by this work, and its semantics are unchanged.** The two
packages coexist on purpose:

| | `models/mobile_clip/` (v1) | `models/mobile_clip_v2/` (this package) |
|---|---|---|
| Image tower | `keras.applications` MobileNetV2 / V3 substitute | faithful FastViT MCi |
| Text tower | `MobileClipTextEncoder` | the same class, **imported**, unchanged |
| Faithful to the reference? | no, by its own recorded decision | yes, modulo X-1..X-5 |
| Status | shipped and tested; untouched by this work | new |

**v2 imports v1's `MobileClipTextEncoder` rather than duplicating it.** That is
not merely a DRY preference. `MobileClipTextEncoder` owns one of exactly **two**
block->keep causal-mask adapter sites in `src/` (the other is
`layers/heads/vlm/factory.py`). A **third** site triggers a mandatory promotion of
that adapter into a keep-polarity `MaskFactory` variant — an unrelated refactor
with its own blast radius. Re-implementing the text tower here would create that
third site for no architectural gain: the layer is already dimension-generic
(measured at 768/12/3072 and 512/8/2048, causal on and off) and already carries
the graph-safe `MaskFactory.create_causal_mask` path.

The cost of that import is a coupling worth naming: `components.py` is guarded by
no `__all__` and no factory, so this is an import of a non-public module. If v1's
text tower semantics ever change, v2's change with them.

## 17. Technical Details

| Detail | Value / rule |
|---|---|
| Input resolution | 256 x 256 x 3, channels last |
| Context length | 77 tokens |
| Vocabulary | 49408 (OpenAI BPE) — **no tokenizer ships with this package** |
| `logit_scale` init | `log(1 / 0.07)`, created in `build()` |
| `logit_scale` use | `clip(exp(logit_scale), 0, 100)` |
| LayerScale | `LearnableMultiplier(CHANNEL, Constant(1e-5), constraint=None)` |
| Normalization epsilon | `1e-5`, passed **explicitly** (the factory `setdefault`s `1e-6`) |
| Attention heads | `dim // 32` |
| Drop-path schedule | one global linear ramp over `sum(layers)` blocks, sliced stagewise |
| Docstring style | Google (`Args:`) in `models/`; Sphinx (`:param:`) in `layers/` |

`LearnableMultiplier` defaults to `constraint='non_neg'`. Every LayerScale here
passes `constraint=None` — a gamma clamped at zero would silently halve the
parameterization, and nothing but a negative-assignment pin would notice.

## 18. Citation

```bibtex
@inproceedings{vasu2023fastvit,
  title     = {FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization},
  author    = {Vasu, Pavan Kumar Anasosalu and Gabriel, James and Zhu, Jeff and
               Tuzel, Oncel and Ranjan, Anurag},
  booktitle = {ICCV},
  year      = {2023}
}

@inproceedings{vasu2024mobileclip,
  title     = {MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training},
  author    = {Vasu, Pavan Kumar Anasosalu and Pouransari, Hadi and Faghri, Fartash and
               Vemulapalli, Raviteja and Tuzel, Oncel},
  booktitle = {CVPR},
  year      = {2024}
}

@article{faghri2025mobileclip2,
  title   = {MobileCLIP2: Improving Multi-Modal Reinforced Training},
  author  = {Faghri, Fartash and Vasu, Pavan Kumar Anasosalu and Pouransari, Hadi and
             Vemulapalli, Raviteja and Tuzel, Oncel},
  journal = {arXiv preprint arXiv:2508.20691},
  year    = {2025}
}

@inproceedings{radford2021clip,
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  author    = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle = {ICML},
  year      = {2021}
}
```
