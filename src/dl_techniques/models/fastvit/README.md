# FastViT (MCi): A Faithful Keras 3 Image Backbone

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A channels-last Keras 3 transcription of timm's `FastVit` class, restricted to
the five **MCi** configurations (`mci0`..`mci4`) — the image backbone of Apple's
MobileCLIP and MobileCLIP2 (Vasu et al., 2023/2024; Faghri et al., 2025).

FastViT is a *hybrid* tower: its shallow stages mix tokens with a
**convolutional** RepMixer, and only its deepest one or two stages use global
self-attention, at a resolution where the token count has already collapsed to
64 (or 16).

This package is **architecture only**. No pretrained weights are ported and it
makes no accuracy claim. See [§9](#9-deviations-from-the-reference-implementation)
before quoting it against any published number.

**Where things live**

| | |
|---|---|
| The eight block primitives | [`layers/fastvit/`](../../layers/fastvit/README.md) |
| The assembled tower (here) | `models/fastvit/` |
| The CLIP model that consumes it | [`models/mobile_clip/`](../mobile_clip/README.md) |

---

## Table of Contents

1. [What this is, and when to use it](#1-what-this-is-and-when-to-use-it)
2. [Core concepts](#2-core-concepts)
3. [Architecture deep dive](#3-architecture-deep-dive)
4. [Quick start](#4-quick-start)
5. [Component reference](#5-component-reference)
6. [The MCi variant table](#6-the-mci-variant-table)
7. [Usage patterns](#7-usage-patterns)
8. [Performance, serialization, testing](#8-performance-serialization-testing)
9. [Deviations from the reference implementation](#9-deviations-from-the-reference-implementation)
10. [Troubleshooting & FAQs](#10-troubleshooting--faqs)
11. [Technical details](#11-technical-details)
12. [Citation](#12-citation)

---

## 1. What this is, and when to use it

`FastVitImageEncoder` is a standalone `keras.Model` that turns an image into
either a projected embedding or pooled backbone features. It has two audiences:

* **As a CLIP image tower** — this is what MobileCLIP2 uses it for. In that role
  its terminal `Dense(projection_dim)` **is** the CLIP image projection (see
  §3), and `models/mobile_clip/mobile_clip_v2.py` imports it from here.
* **As a general image backbone** — build it with `projection_dim=None` to get
  pooled features for classification, transfer, or dense heads.

The tower lives in `models/` rather than `layers/` because it is a complete
network with its own variant table and factory. Its constituent blocks stay in
`layers/fastvit/` so they can be reused independently.

## 2. Core concepts

### RepMixer — the convolutional token mixer

```
x -> x + gamma * (mixer(x) - norm(x))
```

`mixer` is a depthwise `MobileOneBlock` with its activation removed; `norm`
degenerates to a *single* BatchNorm (a `MobileOneBlock` with zero conv branches,
no scale branch, and only its identity branch surviving). `gamma` is a
per-channel LayerScale initialized at `1e-5`. At `gamma == 0` the mixer is the
exact identity.

Replacing self-attention with this in the shallow, high-resolution stages is the
whole efficiency argument of FastViT: attention only ever runs where the token
count is already small.

### Structural reparameterization is *not* implemented

FastViT's "Rep" prefix refers to fusing the multi-branch train-time blocks into
a single convolution at inference. This port ships the train-time multi-branch
form only. See deviation **X-1**.

### The stochastic-depth ramp is GLOBAL

The reference computes **one** linear ramp across `sum(layers)` blocks — every
block of every stage — and hands each stage its contiguous slice. Computing a
fresh `0 -> drop_path_rate` ramp per stage would be a different function (stage 1
of a `(2, 12, 24, 4)` model must start where stage 0 ended, not at zero) and
produces an identically-shaped, identically-parameterized, subtly-wrong model.

## 3. Architecture deep dive

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

### The head `Dense` is a projection, not a classifier

All four MobileCLIP / MobileCLIP2 fastvit configs set `"timm_proj": null` with
`"timm_pool": "avg"`. In open_clip's `TimmModel` a non-attention pool asserts
that the trunk itself does the projecting and instantiates the trunk with
`num_classes=embed_dim`; the timm `ClassifierHead`'s linear layer therefore *is*
the image-side projection into the joint embedding space.

**There is no separate projection layer to add, and adding one would be a
second, unfaithful projection.** The constructor argument is spelled
`projection_dim` rather than `num_classes` for exactly this reason. Pass
`projection_dim=None` when you want pooled features instead.

`timm_drop` is `0.0` in all four configs, so `head_dropout_rate` is inert at the
reference settings.

## 4. Quick start

```python
import numpy as np
from dl_techniques.models.fastvit import FastVitImageEncoder

tower = FastVitImageEncoder.from_variant('mci0', projection_dim=512)

out = tower(
    np.zeros((2, 256, 256, 3), dtype='float32'),
    training=False,           # explicit: see the FAQ
)
out.shape                     # (2, 512)
```

> **Always pass `training=False` explicitly for a deterministic forward.**
> `StochasticDepth` short-circuits to the identity only when `training is
> False`; at `training=None` it runs its stochastic path, and every
> BatchNormalization in the tower uses batch statistics and updates its moving
> averages.

## 5. Component reference

Public surface (`from dl_techniques.models.fastvit import ...`):

| Symbol | Kind | What it is |
|---|---|---|
| `FastVitImageEncoder` | `keras.Model` | The assembled MCi tower. |
| `create_fastvit_image_encoder` | function | Factory over `MCI_VARIANTS`. |
| `MCI_VARIANTS` | dict | The five MCi backbone rows (§6). |

Methods beyond the Keras surface:

| Method | Purpose |
|---|---|
| `from_variant(variant, input_shape=(256,256,3), projection_dim=512, **kwargs)` | Build from an `MCI_VARIANTS` key; kwargs override the row. |
| `stem_output_shape(input_shape)` | Spatial shape after the /4 stem. |
| `stage_output_shapes(input_shape)` | Per-stage output shapes, for dense heads. |

The eight primitives the tower is built from live in
[`src/dl_techniques/layers/fastvit/`](../../layers/fastvit/README.md) and are
documented there, including the `RepMixerBlock` name-collision warning
(deviation **X-5**).

## 6. The MCi variant table

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
deviation **X-3**. The `mci3` / `mci4` rows *are* checked field by field against
a committed copy of the reference source.

To count parameters for a variant, derive the number rather than trusting one
written here:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -c "
from dl_techniques.models.fastvit import FastVitImageEncoder
m = FastVitImageEncoder.from_variant('mci0')
m.build((None, 256, 256, 3))
print(m.count_params())
"
```

## 7. Usage patterns

### Projection vs. pooled features

```python
from dl_techniques.models.fastvit import create_fastvit_image_encoder

# CLIP embedding — the terminal Dense IS the projection
tower = create_fastvit_image_encoder('mci0', projection_dim=512)

# Pooled backbone features instead, for dense / transfer use — NOT for CLIP
backbone = create_fastvit_image_encoder('mci0', projection_dim=None)
```

### A reduced-depth tower for tests

Full `mci4` at 256 px does not fit comfortably alongside a training job on a
12 GB card. Every per-stage tuple is an ordinary constructor argument:

```python
tower = FastVitImageEncoder.from_variant('mci3', layers=(1, 1, 1, 1, 1))
```

Inside MobileCLIP2 the same override is reached through
`image_encoder_kwargs={'layers': (1, 1, 1, 1, 1)}`, which is forwarded verbatim.

### Naming and warm-starting

`load_weights_from_checkpoint` matches layers **by name**. A tower that is ever
warm-started independently must be named identically in every model that holds
it — inside MobileCLIP2 that name is fixed by the module constant
`IMAGE_TOWER_NAME` (`"image_encoder"`). Do not rename it.

## 8. Performance, serialization, testing

* **Attention cost is already bounded** by the architecture — at most 64 tokens.
  The convolutional stages dominate runtime; reduce `layers` / `embed_dims`
  before reaching for attention tricks.
* **No fused inference path exists** (deviation **X-1**). If you need one,
  implement it as an explicit, separately tested conversion pass over a trained
  model; do not silently change the blocks' `call()` paths.
* Every class carries `@keras.saving.register_keras_serializable()`, a complete
  `get_config`, and a `compute_output_shape`, so `model.save(...)` /
  `keras.models.load_model(...)` round-trips.

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
    tests/test_models/test_fastvit/ tests/test_layers/test_fastvit/ -q
```

The suites cover initialization, invalid-config raises, forward shape,
`compute_output_shape` pre- and post-build, `.keras` save->load compared **by
value** (`atol=1e-6, rtol=0`), an **elementwise** weight check on the list-held
per-block sub-layers (counts, variable paths and parameter totals can all match
while the restored kernels are fresh), gradient flow to every trainable weight,
and per-class behavioural pins that were each proven RED against a deliberately
broken variant.

## 9. Deviations from the reference implementation

Every known divergence from the reference is listed here with a stable id. A
deviation that is silently absorbed makes the port unauditable.

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
| `test_models/test_fastvit/test_model.py::test_mci3_mci4_match_supplied_source` | `mobileclip2.py`, parsed with `ast` | `MCI_VARIANTS['mci3']`, `['mci4']`, every field |
| `test_models/test_mobile_clip/test_mobile_clip_v2.py::test_model_variants_match_supplied_json_configs` | the six `model_configs/*.json`, via `json.load` | all six `MODEL_VARIANTS` rows + the shared constants |

The source is PARSED, not imported: it is PyTorch/`timm` code and this
environment has neither. Both tests were RED-proven by perturbing a value in the
committed reference files and confirming the failure names the field.

This does **not** extend to `mci0` / `mci1` / `mci2`. The supplied source does
not define them, and the JSON configs merely NAME `fastvit_mci0` /
`fastvit_mci2` without giving their architecture — so those three rows remain
transcription-only, exactly as stated above. Vendoring `timm` is the only thing
that would change that.

Anyone changing an `mci0`/`mci1`/`mci2` row must re-derive it from timm upstream
and say so — and must not reason from the `mci3`/`mci4` rows, which differ
structurally (5 stages, no SE, LayerNorm, `mlp_ratio` 4).

### X-4 — No pretrained weights

The tower is architecture-only and makes **no accuracy claim**. Any comparison
against published FastViT or MobileCLIP2 numbers would be invalid by
construction, and no conversion script exists here.

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

## 10. Troubleshooting & FAQs

**My forward pass is not reproducible.** Pass `training=False` explicitly.
`StochasticDepth` short-circuits only on `training is False`, and `training=None`
also puts every BatchNormalization into batch-statistics mode.

**Should I add a projection head on top of the tower?** No. The tower's terminal
`Dense` already is the CLIP image projection. Use `projection_dim=None` only
when you want pooled backbone features for a non-CLIP purpose.

**`ValueError: All per-stage tuples must have one entry per stage`.** A 4-stage
vs 5-stage mixup. `mci0`/`mci1`/`mci2` need 4 entries in every per-stage tuple;
`mci3`/`mci4` need 5.

**Can I load an official checkpoint?** No — deviation **X-4**.

**Why does `FastVitRepMixerBlock` not match `RepMixerBlock`?** They are different
architectures that share a name. See deviation **X-5**.

**Where is the CLIP model?** [`models/mobile_clip/`](../mobile_clip/README.md).
This package is the image branch only.

## 11. Technical details

| Detail | Value / rule |
|---|---|
| Input resolution | 256 x 256 x 3, channels last (the MobileCLIP setting) |
| LayerScale | `LearnableMultiplier(CHANNEL, Constant(1e-5), constraint=None)` |
| Normalization epsilon | `1e-5`, passed **explicitly** (the factory `setdefault`s `1e-6`) |
| Attention heads | `dim // 32` |
| Drop-path schedule | one global linear ramp over `sum(layers)` blocks, sliced stagewise |
| SE ratios | **two in one network** — `1/16` inside `MobileOneBlock`, `0.25` inside the large-kernel conv |
| SE ordering | SE runs BEFORE the activation: `act(se(x))` |

`LearnableMultiplier` defaults to `constraint='non_neg'`. Every LayerScale here
passes `constraint=None` — a gamma clamped at zero would silently halve the
parameterization, and nothing but a negative-assignment pin would notice.

## 12. Citation

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
```
