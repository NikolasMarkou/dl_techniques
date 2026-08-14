# MobileCLIP: Fast and Efficient On-Device CLIP

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

Keras 3 implementations of Apple's **MobileCLIP** family — efficient
vision-language models that adapt CLIP's zero-shot capabilities to the latency,
memory, and power limits of mobile and edge hardware.

**This package ships two models. They are not interchangeable.**

| | `MobileClipModel` (v1) | `MobileClipV2Model` (v2) |
|---|---|---|
| Module | `mobile_clip_v1.py` | `mobile_clip_v2.py` |
| Paper | MobileCLIP (Vasu et al., 2024) | MobileCLIP2 (Faghri et al., 2025) |
| Image tower | `keras.applications` MobileNetV2 / V3 **substitute** | faithful FastViT **MCi**, from [`models/fastvit/`](../fastvit/README.md) |
| Text tower | `MobileClipTextEncoder` | the same class, shared |
| Faithful to the reference? | **No**, by its own recorded decision D-001 | Yes, modulo [X-1..X-5](#16-deviations-v2) |
| Variant keys | `b`, `s0`, `s1`, `s2` | `mobileclip2_s0/s2/s3/s4`, `mobileclip_s3/s4` |
| Status | shipped and tested; not deprecated | the faithful port; architecture only, no weights |

Neither model ships pretrained weights, and **v2 makes no accuracy claim** — read
[§16](#16-deviations-v2) before comparing it against any published number. See
[§17](#17-v1-vs-v2-which-should-i-use) for how to choose.

---

## Table of Contents

1. [Overview: What is MobileCLIP and Why It Matters](#1-overview-what-is-mobileclip-and-why-it-matters)
2. [The Problem MobileCLIP Solves](#2-the-problem-mobileclip-solves)
3. [How MobileCLIP Works: Core Concepts](#3-how-mobileclip-works-core-concepts)
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
15. [Technical Details](#15-technical-details)
16. [Deviations from the reference implementation (v2)](#16-deviations-v2)
17. [v1 vs v2: which should I use?](#17-v1-vs-v2-which-should-i-use)
18. [Citation](#18-citation)

---

## 1. Overview: What is MobileCLIP and Why It Matters

**MobileCLIP** is a highly efficient multimodal model family from Apple that
brings large-scale vision-language understanding to on-device applications. It
follows the same principle as the original CLIP — learning a shared embedding
space between images and text from web-scale data — but with a co-designed,
asymmetric architecture that prioritizes mobile performance.

### Key Innovations

1. **Efficient image backbones**: instead of the large Vision Transformers used
   in the original CLIP, MobileCLIP employs custom lightweight hybrid backbones
   (`mci0`, `mci1`, ...) built on FastViT.
2. **Asymmetric design**: the model is intentionally lopsided. The image encoder
   is made extremely fast, as it must run on every input frame in real-time
   applications. The text encoder, which often runs only once over a small set of
   prompts for zero-shot classification, can be comparatively more powerful.
3. **Optimized text encoder**: a standard Transformer whose depth and masking are
   tailored per variant to balance accuracy against on-device cost.

### Why It Matters

```
Problem: Perform zero-shot classification on a mobile phone.

Standard CLIP:
  1. A massive ViT image encoder.
  2. Too large for memory, too slow for real time.
  3. Forces a cloud round trip -> latency and privacy costs.

MobileCLIP:
  1. Replace the image encoder with a hyper-efficient hybrid CNN.
  2. Optimize the whole architecture for on-device constraints.
  3. Runs locally, enabling real-time recognition, visual search and
     content filtering with no network connection.
```

---

## 2. The Problem MobileCLIP Solves

### The "Last Mile" Problem for Vision-Language Models

Models like CLIP demonstrated a revolutionary leap in zero-shot learning, but
their size and computational cost made them largely inaccessible on edge
devices.

```
┌──────────────────────────────────────────────────────────────┐
│  The Dilemma of Deploying Large VLMs                         │
│                                                              │
│  Large Cloud Models (e.g., CLIP ViT-L/14):                   │
│    - State-of-the-art accuracy.                              │
│    - Prohibitively large: hundreds of millions of parameters.│
│    - High latency: requires a round trip to a server.        │
│    - Privacy concerns: user data must leave the device.      │
│                                                              │
│  The Need for On-Device AI:                                  │
│    - A model that retains the flexible, zero-shot            │
│      capabilities of CLIP but is small, fast, and            │
│      power-efficient enough to run locally.                  │
└──────────────────────────────────────────────────────────────┘
```

MobileCLIP starts from on-device performance as a primary constraint, making the
model "efficient by design" rather than a scaled-down version of a larger one.

---

## 3. How MobileCLIP Works: Core Concepts

### The Dual-Encoder Contrastive Architecture

Two networks — one for images, one for text — map their inputs into a shared
space. Features are L2-normalized, scaled by a learnable temperature, and
compared with a symmetric similarity matrix.

```
{'image': (B,256,256,3)}          {'text': (B,77)}
        |                                |
   Image encoder                   Text encoder
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

Training shows the model a batch of N (image, text) pairs and computes an N x N
similarity matrix. The contrastive loss maximizes the diagonal (correct pairs)
and minimizes everything else.

**The clip on `exp(logit_scale)` is load-bearing, not cosmetic**: an unbounded
temperature turns a diverging run into `inf` logits and a `nan` loss with no
other observable symptom. Both v1 and v2 cap it at 100.

---

## 4. Architecture Deep Dive

### 4.1 The image tower

**v2 — faithful.** A FastViT MCi hybrid: shallow stages mix tokens with a
*convolutional* RepMixer, and only the deepest one or two stages use global
self-attention, at a resolution where the token count has already collapsed to
64 (or 16). It lives in its own package,
[`models/fastvit/`](../fastvit/README.md), and is a standalone `keras.Model`.

> **There is no separate image projection in v2.** The tower's terminal `Dense`
> **is** the CLIP image projection: MobileCLIP's open_clip configs set
> `"timm_pool": "avg"` with `"timm_proj": null`, and in open_clip's `TimmModel` a
> non-attention pool asserts that the trunk itself does the projecting and
> instantiates it with `num_classes=embed_dim`. So `projection_dim` is passed
> straight through to `FastVitImageEncoder` and its output *is* the image
> embedding. Stacking another projection on top would be a second, unfaithful
> one.

**v1 — deliberately non-faithful.** `MobileClipImageEncoder` pairs a
`keras.applications` backbone with an `ImageProjectionHead` (global average pool
+ `Dense` into `embed_dim`). Its in-file decision D-001 (`components.py:18-32`)
records the substitution: the real MCi backbones did not exist in
`keras.applications`, so `mci0 -> MobileNetV3Small`, `mci1 -> MobileNetV2`,
`mci2 -> MobileNetV3Large`, `vit_b16 -> MobileNetV3Large`.

> **The strings `mci0`/`mci1`/`mci2` mean opposite things in the two models.** In
> v2 they name real MCi rows (`models.fastvit.MCI_VARIANTS`); in v1 they are keys
> of `components._BACKBONE_ALIASES` resolving to MobileNet stand-ins.

### 4.2 `MobileClipTextEncoder` — shared by both

Token embedding + learned positional embedding -> `text_layers` x
`TransformerLayer` -> `LayerNormalization` -> EOT-token extraction -> projection
to `embed_dim`. `intermediate_size` is `4 * text_width` in every shipped variant.

This tower is **faithful for both models**: v1's substitution is confined to the
image branch. v2 shares the class rather than re-implementing it — see
[§17](#17-v1-vs-v2-which-should-i-use) for why that is a hard constraint, not a
DRY preference.

### 4.3 Contrastive head and `logit_scale`

Both features are L2-normalized (constraining them to a hypersphere, so the dot
product is cosine similarity), then multiplied by a learnable scalar
`logit_scale` initialized to `log(1/0.07)`. This temperature controls the
sharpness of the similarity distribution and is critical for stable training.

In v2 the towers return **raw, un-normalized** features from their own `call`;
normalization happens in `encode_image` / `encode_text`, because
`dl_techniques.utils.clip_utils.compute_clip_logits` documents that it expects
already-normalized inputs and does not normalize internally.

---

## 5. Quick Start Guide

### v2 (the faithful port)

```python
import numpy as np
from dl_techniques.models.mobile_clip import MobileClipV2Model

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
> `StochasticDepth` short-circuits to the identity only when `training is
> False`; at `training=None` it runs its stochastic path, and every
> BatchNormalization in the image tower uses batch statistics and updates its
> moving averages.

### v1

```python
import numpy as np
from dl_techniques.models.mobile_clip import MobileClipModel

model = MobileClipModel.from_variant('s0')
model.build({'image': (None, 256, 256, 3), 'text': (None, 77)})

outputs = model({
    'image': np.random.rand(8, 256, 256, 3).astype('float32'),
    'text': np.random.randint(0, 49408, (8, 77)),
})
outputs['image_features'].shape   # (8, 512)
```

Neither model ships a tokenizer. Both expect CLIP's byte-pair encoding with a
49,408-token vocabulary — pre-trained versions are available in libraries such as
Hugging Face `transformers` (e.g. `openai/clip-vit-base-patch32`).

---

## 6. Component Reference

Package surface (`from dl_techniques.models.mobile_clip import ...`):

| Symbol | Kind | What it is |
|---|---|---|
| `MobileClipV2Model` | `keras.Model` | The faithful MobileCLIP2 dual encoder. |
| `create_mobile_clip_v2` | function | Factory over v2's `MODEL_VARIANTS`. |
| `MobileClipModel` | `keras.Model` | The v1 dual encoder. |
| `create_mobile_clip_model` | function | Factory over v1's variants. |

> **`MODEL_VARIANTS` is deliberately NOT re-exported at package level** — the two
> tables are different objects with disjoint keys. v1's is the class attribute
> `MobileClipModel.MODEL_VARIANTS`; v2's is the module-level
> `mobile_clip_v2.MODEL_VARIANTS`. Import each from its own module, explicitly.

Submodule components:

| Component | Location | Purpose |
| :--- | :--- | :--- |
| `MobileClipTextEncoder` | `.components` | The Transformer text encoder — used by **both** models. |
| `MobileClipImageEncoder` | `.components` | v1's image encoder: backbone + projection head. |
| `ImageProjectionHead` | `.components` | v1's pooling + projection layer. |
| `FastVitImageEncoder` | [`models.fastvit`](../fastvit/README.md) | v2's image tower. |

`MobileClipV2Model` methods beyond the Keras surface:

| Method | Purpose |
|---|---|
| `from_variant(variant, **kwargs)` | Build from a `MODEL_VARIANTS` key; kwargs override the row. |
| `encode_image(image, normalize=True, training=None)` | Image branch only. |
| `encode_text(text, normalize=True, training=None)` | Text branch only. |
| `compute_logit_scale()` | `clip(exp(logit_scale), 0, logit_scale_max)`. |

---

## 7. Configuration & Model Variants

### v2 — `mobile_clip_v2.MODEL_VARIANTS`

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

**The table deliberately holds two families.** The four `mobileclip2_s*` rows are
non-causal (their JSON configs say `"no_causal_mask": true`); the two earlier
`mobileclip_s3`/`s4` rows are causal. Same image backbones, different text-tower
attention — that single flag is the only reason both families appear. Do not
"simplify" it away.

Each row is keyed by the name of the supplied JSON config file it transcribes,
and `test_model_variants_match_supplied_json_configs` checks it field by field
against the committed copy at `research/mobileclip2_reference/model_configs/` —
so this table is verified, not re-derived.

### v1 — `MobileClipModel.MODEL_VARIANTS`

| Variant | Embed Dim | Image Backbone | Image Size | Text Layers | Causal Mask |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **`b`** | 512 | `vit_b16` | 224 | 12 | **True** |
| **`s0`**| 512 | `mci0` | 256 | **4** | False |
| **`s1`** | 512 | `mci1` | 256 | 12 | False |
| **`s2`** | 512 | `mci2` | 256 | 12 | False |

*The backbone names are the paper's; this model resolves them to
`keras.applications` stand-ins (§4.1).*

### Counting parameters

Derive the number rather than trusting one written here:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -c "
from dl_techniques.models.mobile_clip import MobileClipV2Model
m = MobileClipV2Model.from_variant('mobileclip2_s0')
m.build({'image': (None,256,256,3), 'text': (None,77)})
print(m.count_params())
"
```

---

## 8. Comprehensive Usage Examples

### Zero-shot classification

```python
import numpy as np
from keras import ops
from dl_techniques.models.mobile_clip import MobileClipV2Model

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

### Embeddings for downstream tasks

The encoders give efficient embeddings for visual search, clustering, or
retrieval:

```python
image_embeddings = model.encode_image(images, training=False)   # (B, D)
text_embeddings = model.encode_text(tokens, training=False)     # (B, D)
```

### The image tower on its own

```python
from dl_techniques.models.fastvit import create_fastvit_image_encoder

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

---

## 9. Advanced Usage Patterns

### Naming and warm-starting

`IMAGE_TOWER_NAME` and `TEXT_TOWER_NAME` are module constants in
`mobile_clip_v2.py` (`"image_encoder"` / `"text_encoder"`).
`load_weights_from_checkpoint` matches layers **by name**, so a tower that is
ever warm-started independently must be named identically in every model that
holds it. Do not rename the towers.

### Substituting a tower

Pass an already-constructed `image_encoder=` / `text_encoder=` to the v2
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

---

## 10. Performance Optimization

* **Attention cost is already bounded** in v2's image tower — at most 64 tokens.
  The convolutional stages dominate runtime; reduce `layers` / `embed_dims`
  before reaching for attention tricks.
* **No fused inference path exists** (deviation **X-1**). If you need one,
  implement it as an explicit, separately tested conversion pass over a trained
  model; do not silently change the blocks' `call()` paths.
* **Text embeddings for a fixed prompt set are constant.** Compute `encode_text`
  once and cache it; only `encode_image` needs to run per frame. This is the
  asymmetry the whole architecture is designed around (§15).
* **Mixed precision** suits both the Transformer text encoder and the
  convolutional image tower:

  ```python
  keras.mixed_precision.set_global_policy('mixed_float16')
  ```

---

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
* **Batch size is the dominant hyperparameter.** Contrastive learning scales
  with the number of negatives per positive; the original CLIP used batches
  above 32,000. On consumer hardware, gradient accumulation is essential.
* **AdamW with cosine decay and linear warmup** is the standard recipe. Weight
  decay matters for the Transformer branch.
* `drop_path_rate` (v2) is the **maximum** of one global linear ramp over every
  block of every stage. Setting it per stage is not expressible and would not
  reproduce the reference.
* `dropout_rate` is threaded to both towers; `attention_dropout_rate` reaches
  the text tower only.

---

## 12. Serialization & Deployment

Every class carries `@keras.saving.register_keras_serializable()`, a complete
`get_config`, and a `compute_output_shape`, so the standard round trip works:

```python
model.save('mobileclip2_s0.keras')
restored = keras.models.load_model('mobileclip2_s0.keras')
```

Two v2 implementation details matter if you subclass or extend:

* `get_config` serializes **both towers explicitly** (not merely their variant
  names), so a checkpoint keeps describing the network it was trained with even
  if a variant table is later corrected. `from_config` re-materializes them with
  `deserialize_keras_object`.
* `get_build_config` / `build_from_config` are overridden because Keras' generic
  implementation cannot round-trip a **dict** input-shape spec — it warns that
  the model cannot be built automatically and leaves the restored model unbuilt.

---

## 13. Testing & Validation

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
    tests/test_models/test_mobile_clip/ tests/test_models/test_fastvit/ \
    tests/test_layers/test_fastvit/ -q
```

The suites cover initialization, invalid-config raises, forward shape,
`compute_output_shape` pre- and post-build, `.keras` save->load compared **by
value** (`atol=1e-6, rtol=0`), an **elementwise** weight check on the list-held
per-block sub-layers (counts, variable paths and parameter totals can all match
while the restored kernels are fresh), gradient flow to every trainable weight,
and per-class behavioural pins that were each proven RED against a deliberately
broken variant.

---

## 14. Troubleshooting & FAQs

**My forward pass is not reproducible.** Pass `training=False` explicitly.
`StochasticDepth` short-circuits only on `training is False`, and `training=None`
also puts every BatchNormalization into batch-statistics mode.

**Training is unstable and the loss becomes `NaN`.** Usually the learning rate,
or a diverging `logit_scale`. Use a warmup schedule and a smaller LR; the
temperature is already clipped at 100, which prevents the `inf`-logit failure
mode but not a badly-scaled optimizer.

**Should I add a projection head on top of v2's image tower?** No — its terminal
`Dense` already *is* the CLIP image projection. Use `projection_dim=None` only
when you want pooled backbone features for a non-CLIP purpose.

**`ValueError: All per-stage tuples must have one entry per stage`.** A 4-stage
vs 5-stage mixup in the image tower. `mci0`/`mci1`/`mci2` need 4 entries in every
per-stage tuple; `mci3`/`mci4` need 5.

**Can I load an official MobileCLIP or MobileCLIP2 checkpoint?** No. No weights
are ported (deviation **X-4**) and no conversion script exists here.

**What is the difference between v1's `b` and `s` variants?** `b` is larger and
uses a ViT-style image config with a deeper, causally-masked text encoder; the
`s` variants target on-device use with convolutional backbones and no causal
mask. `s0` is the smallest, with only 4 text Transformer layers.

**Where can I find the tokenizer?** Neither model ships one. Use a CLIP-compatible
BPE tokenizer with a 49,408 vocabulary (e.g. Hugging Face
`openai/clip-vit-base-patch32`).

---

## 15. Technical Details

### Asymmetric design philosophy

The key insight is the asymmetry of the zero-shot use case:

* **Image encoder** — must be extremely fast; it processes a new image on every
  inference.
* **Text encoder** — can be more expensive. For `K` classes it runs `K` times to
  build class embeddings, which are then reused across many images.

Hence a hyper-efficient hybrid CNN for images and a more powerful Transformer for
text.

### Causal masking in the text encoder

A causal mask prevents a token from attending to future tokens — standard for
autoregressive language models. Its use is per-variant, not per-generation: v1's
`b` sets it, v1's `s*` do not; v2's `mobileclip_s3`/`s4` set it, the four
`mobileclip2_s*` rows do not (§7).

### v2 reference values

| Detail | Value / rule |
|---|---|
| Input resolution | 256 x 256 x 3, channels last |
| Context length | 77 tokens |
| Vocabulary | 49408 (OpenAI BPE) — **no tokenizer ships here** |
| `logit_scale` init | `log(1 / 0.07)`, created in `build()` |
| `logit_scale` use | `clip(exp(logit_scale), 0, 100)` |
| Attention heads (image tower) | `dim // 32` |
| Drop-path schedule | one global linear ramp over `sum(layers)` blocks, sliced stagewise |

More image-tower internals — LayerScale, normalization epsilon, the two SE
ratios — are in [`models/fastvit/README.md`](../fastvit/README.md) §11.

---

## 16. Deviations (v2)

**These apply to `MobileClipV2Model` only.** v1's single, larger deviation is its
backbone substitution (§4.1); it is not part of this numbered list.

Every known divergence from the reference carries a stable id. A deviation that
is silently absorbed makes the port unauditable. The image-tower deviations
**X-1**, **X-2**, **X-3** and **X-5** are stated in full, with their measurements
and RED-proofs, in
[`models/fastvit/README.md` §9](../fastvit/README.md#9-deviations-from-the-reference-implementation);
summarized here:

| Id | Deviation |
|---|---|
| **X-1** | **No structural reparameterization.** Train-time multi-branch form only; no fusion path exists or is tested. The reference release also runs the multi-branch form (`inference_mode=False`), so this matches what it actually executes. |
| **X-2** | **One `use_bias` for both qkv and the output projection.** The shared `MultiHeadAttention` cannot express timm's unbiased-qkv + biased-proj split, so each attention block is short exactly one bias vector of length `dim`. MEASURED and pinned. |
| **X-3** | **`mci0`/`mci1`/`mci2` have no local oracle.** They are transcribed from timm upstream, which is not installed here. `mci3`/`mci4` *do* have a real committed oracle. |
| **X-4** | **No pretrained weights.** Architecture only; **no accuracy claim**. Any comparison against published MobileCLIP2 numbers is invalid by construction. |
| **X-5** | **`RepMixerBlock` is a different architecture that shares the name.** `layers/repmixer_block.py::RepMixerBlock` is not FastViT's block; it is consumed by `models/fastvlm/` and deliberately untouched. Use `layers/fastvit/FastVitRepMixerBlock` for anything that must match timm block-for-block. |

---

## 17. v1 vs v2: which should I use?

**Use v2** for anything that should correspond to the published architecture, or
that will consume a real MCi tower. **Use v1** only if you already depend on it —
its checkpoints, its variant keys, or its `keras.applications` backbones.

**v1 is not deprecated by v2 and its semantics are unchanged.** The two coexist
on purpose: v1's substitution is a recorded, tested decision, not a bug awaiting
repair.

### Why the text tower is shared rather than duplicated

`MobileClipTextEncoder` owns one of exactly **two** block->keep causal-mask
adapter sites in `src/` (the other is `layers/heads/vlm/factory.py`). A **third**
site triggers a mandatory promotion of that adapter into a keep-polarity
`MaskFactory` variant — an unrelated refactor with its own blast radius.

Re-implementing the text tower for v2 would create that third site for no
architectural gain: the layer is already dimension-generic (measured at
768/12/3072 and 512/8/2048, causal on and off) and already carries the
graph-safe `MaskFactory.create_causal_mask` path that `ops.tril` cannot provide
on this stack.

The coupling worth naming: if v1's text tower semantics ever change, v2's change
with them. Both models are covered by `tests/test_models/test_mobile_clip/`, so
such a change cannot land silently.

---

## 18. Citation

Please cite the original works:

```bibtex
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

@inproceedings{vasu2023fastvit,
  title     = {FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization},
  author    = {Vasu, Pavan Kumar Anasosalu and Gabriel, James and Zhu, Jeff and
               Tuzel, Oncel and Ranjan, Anurag},
  booktitle = {ICCV},
  year      = {2023}
}

@inproceedings{radford2021clip,
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  author    = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle = {ICML},
  year      = {2021}
}
```
