# MobileCLIP: Fast and Efficient On-Device CLIP

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready Keras 3 implementation of Apple's **MobileCLIP** family — efficient vision-language models that adapt CLIP's zero-shot capabilities to the latency, memory and power limits of mobile and edge hardware. This package ships **two** models that share a text tower and a class structure but differ on the image side: `MobileClipModel` (v1) and `MobileClipV2Model` (v2, the faithful port).

Neither model ships pretrained weights, and v2 makes **no accuracy claim** — it is architecture only. Read [§16](#16-deviations-from-the-reference-implementation-v2) before comparing it against any published number, and [§17](#17-v1-vs-v2-which-should-i-use) to choose between the two.

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
16. [Deviations from the Reference Implementation (v2)](#16-deviations-from-the-reference-implementation-v2)
17. [v1 vs v2: Which Should I Use?](#17-v1-vs-v2-which-should-i-use)
18. [Citation](#18-citation)

---

## 1. Overview: What is MobileCLIP and Why It Matters

### What is MobileCLIP?

**MobileCLIP** is a family of efficient multimodal models from Apple that brings large-scale vision-language understanding to on-device applications. It follows the same principle as the original CLIP — learning a shared embedding space between images and text from web-scale data — but with a co-designed, asymmetric architecture that treats on-device performance as a primary constraint rather than an afterthought.

**MobileCLIP2** (Faghri et al., 2025) keeps that architecture and improves the multi-modal reinforced training recipe behind it. Its image branch is not a scaled-down ViT: it is **FastViT MCi**, a hybrid tower whose shallow stages mix tokens with a *convolutional* RepMixer and whose deepest one or two stages use global self-attention, at a resolution where the token count has already collapsed to 64 (or 16).

### Key Innovations

1.  **Efficient image backbones**: instead of the large Vision Transformers used in the original CLIP, MobileCLIP employs custom lightweight hybrid backbones (`mci0` … `mci4`) built on FastViT.
2.  **Asymmetric design**: the model is intentionally lopsided. The image encoder is made extremely fast, because it must run on every input frame in real-time applications. The text encoder, which often runs only once over a small set of prompts for zero-shot classification, can be comparatively more powerful.
3.  **Optimized text encoder**: a standard Transformer whose depth and *masking* are tailored per variant. The MobileCLIP2 series runs a **bidirectional** text tower; the earlier MobileCLIP-S3/S4 configs run the classic causal one.
4.  **No separate image projection**: the image tower's own terminal `Dense` *is* the CLIP image projection, so there is one fewer layer than a naive port would have.

### Why MobileCLIP Matters

```
Problem: Perform zero-shot classification on a mobile phone.

Standard CLIP:
  1. A massive ViT image encoder.
  2. Too large for memory, too slow for real time.
  3. Forces a cloud round trip -> latency and privacy costs.

MobileCLIP:
  1. Replace the image encoder with a hyper-efficient hybrid CNN.
  2. Optimize the whole architecture for on-device constraints.
  3. Runs locally, with no network connection.
```

### Real-World Impact

-   📱 **On-device visual search**: match a camera frame against a set of natural-language prompts with no server round trip and no user data leaving the device.
-   🔍 **Open-vocabulary recognition**: classify against categories chosen at runtime, without retraining, by tokenizing new prompts.
-   **Content filtering and moderation**: score frames against textual policies locally, where latency and privacy both matter.
-   **Retrieval and clustering**: use either tower alone as a frozen embedding extractor for a downstream index.

---

## 2. The Problem MobileCLIP Solves

### The "Last Mile" Problem for Vision-Language Models

Models like CLIP demonstrated a revolutionary leap in zero-shot learning, but their size and computational cost made them largely inaccessible on edge devices.

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

### How MobileCLIP Changes the Game

The insight is that attention is expensive exactly where the token count is high, and that a convolutional token mixer is a perfectly good substitute in the shallow, high-resolution stages. FastViT therefore spends attention only in the last one or two stages, by which point the grid has collapsed from 64x64 to 8x8 or 4x4.

```
┌──────────────────────────────────────────────────────────────┐
│  Where the cost goes, per stage (256px input)                │
│                                                              │
│  stem   64x64   -> convolutional, no attention               │
│  stage0 64x64   -> RepMixer  (depthwise conv token mixing)   │
│  stage1 32x32   -> RepMixer                                  │
│  stage2 16x16   -> RepMixer                                  │
│  stage3  8x8    -> ATTENTION, 64 tokens                      │
│  stage4  4x4    -> ATTENTION, 16 tokens   (5-stage variants) │
│                                                              │
│  Attention never sees more than 64 tokens. The convolutional │
│  stages dominate runtime, and they are the cheap ones.       │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. How MobileCLIP Works: Core Concepts

### The Dual-Encoder Contrastive Architecture

Two networks — one for images, one for text — map their inputs into a shared space. Features are L2-normalized, scaled by a learnable temperature, and compared with a symmetric similarity matrix.

```
┌───────────────────────────────────────────────────────────────────┐
│                      MobileCLIP Architecture                      │
│                                                                   │
│  Image Input (Batch) ───►┌─────────────────┐                      │
│                          │  Image Encoder  │                      │
│                          │ (FastViT MCi)   │───► Image Embeddings │
│                          └─────────────────┘      (Batch, D)      │
│                                                        │          │
│  Text Input (Batch) ───► ┌─────────────────┐           ▼          │
│                          │  Text Encoder   │                ┌───────────┐
│                          │  (Transformer)  │───► Text Emb.  │ Compute   │
│                          └─────────────────┘    (Batch, D)  │ NxN       │
│                                                        ▲    │ Similarity│
│                                                        │    └───────────┘
│                                                        └──────────▼
│                                                    Contrastive Loss
│              (Maximize similarity of correct pairs, minimize others)
└───────────────────────────────────────────────────────────────────┘
```

Training shows the model a batch of N (image, text) pairs and computes an N x N similarity matrix. The contrastive loss maximizes the diagonal (correct pairs) and minimizes everything else.

The clip on `exp(logit_scale)` is load-bearing, not cosmetic: an unbounded temperature turns a diverging run into `inf` logits and a `nan` loss with no other observable symptom. Both models cap it at 100.

### The Complete Data Flow

```
Input dict {'image': (B, 256, 256, 3), 'text': (B, 77)}
    │
    ├─► image ─► FastVitImageEncoder
    │              ├─► stem: 3 x MobileOneBlock            -> (B, 64, 64, C0)
    │              ├─► stage_0 .. stage_{N-1}              -> (B, 8, 8, C_last)
    │              ├─► final_conv + GlobalAveragePooling2D -> (B, C_last * 2)
    │              └─► Dense(embed_dim)  ◄── IS the CLIP projection
    │                                                      -> (B, D)
    │
    ├─► text ──► MobileClipTextEncoder
    │              ├─► token embedding + positional embedding
    │              ├─► num_layers x TransformerLayer  (causal or not)
    │              ├─► LayerNormalization
    │              ├─► EOT-token extraction (argmax over token ids)
    │              └─► projection to embed_dim           -> (B, D)
    │
    ├─► ops.normalize(..., axis=-1) on BOTH               -> unit vectors
    │
    ├─► scale = clip(exp(logit_scale), 0, logit_scale_max)
    │
    └─► compute_clip_logits(image, text, scale)
           └─► logits_per_image (B, B), logits_per_text (B, B)
```

Both towers return **raw, un-normalized** features from their own `call`. Normalization happens in `encode_image` / `encode_text`, because `compute_clip_logits` expects already-normalized inputs and does not normalize internally.

---

## 4. Architecture Deep Dive

### 4.1 `FastVitImageEncoder` — v2's image tower

-   **Purpose**: to convert an image into a joint-space embedding at minimum cost, and to do so *faithfully* with respect to timm's `FastVit`.
-   **Location**: its own package, [`models/fastvit/`](../fastvit/README.md). It is a standalone `keras.Model` and is usable without any CLIP model around it.
-   **Architecture**: a `MobileOneBlock` stem doing a /4 downsample, then 4 or 5 `FastVitStage`s (RepMixer in the shallow ones, global self-attention in the deepest), then a depthwise `final_conv` with squeeze-excitation, global average pooling, and a terminal `Dense`.
-   **The terminal `Dense` IS the CLIP image projection**: MobileCLIP's open_clip configs set `"timm_pool": "avg"` with `"timm_proj": null`, and in open_clip's `TimmModel` a non-attention pool asserts that the trunk itself does the projecting and instantiates it with `num_classes=embed_dim`. So `embed_dim` is injected as `projection_dim` and the tower's output *is* the image embedding. **Stacking another projection on top would be a second, unfaithful one** — the model rejects a `projection_dim` written into `image_config` for exactly this reason.

### 4.2 `MobileClipImageEncoder` — v1's image tower

-   **Purpose**: the same role, but **deliberately non-faithful**.
-   **Architecture**: a `keras.applications` backbone plus an `ImageProjectionHead` (global average pool + `Dense` into `embed_dim`).
-   **Why**: its in-file decision D-001 (`components.py`) records that the real MCi backbones did not exist in `keras.applications`, so `mci0 -> MobileNetV3Small`, `mci1 -> MobileNetV2`, `mci2 -> MobileNetV3Large`, `vit_b16 -> MobileNetV3Large`. It is a functional substitute, kept working and tested, not a bug awaiting repair.
-   **Name collision to watch**: the strings `mci0`/`mci1`/`mci2` mean **opposite things** in the two models. In v2 they name real MCi rows (`models.fastvit.MCI_VARIANTS`); in v1 they are keys of `components._BACKBONE_ALIASES` resolving to MobileNet stand-ins.

### 4.3 `MobileClipTextEncoder` — shared by both

-   **Purpose**: to convert a sequence of text tokens into a vector that aligns with the image embeddings.
-   **Architecture**:
    1.  **Token and positional embeddings**: token ids become dense vectors, and a learnable positional embedding encodes word order.
    2.  **Transformer stack**: `num_layers` x `TransformerLayer`, applying self-attention that is causal or bidirectional depending on the variant.
    3.  **Feature extraction**: the embedding at the end-of-text (EOT) token — located by `argmax` over the token ids — represents the whole sequence.
    4.  **Projection**: that vector is projected into the shared `embed_dim`.
-   **Faithful for both models**: v1's substitution is confined to the image branch, so v2 shares this class rather than re-implementing it. See [§17](#17-v1-vs-v2-which-should-i-use) for why that is a hard constraint and not a DRY preference.

### 4.4 Contrastive head and `logit_scale`

-   **Purpose**: to turn two sets of embeddings into the similarity scores a contrastive loss consumes.
-   **Functionality**:
    1.  **L2 normalization**: both feature sets are normalized, constraining them to a hypersphere so the dot product *is* cosine similarity.
    2.  **Temperature scaling**: scores are multiplied by `clip(exp(logit_scale), 0, logit_scale_max)`, where `logit_scale` is a learnable scalar initialized to `log(1 / 0.07)` following the CLIP paper.
    3.  **Weight creation**: in v2 that scalar is created in `build()`, not `__init__`, following the `models/clip` precedent.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First MobileCLIP2 Model (30 seconds)

```python
import numpy as np
from dl_techniques.models.mobile_clip import MobileClipV2Model

# 1. Create the smallest faithful variant
model = MobileClipV2Model.from_variant('mobileclip2_s0')

# 2. Run a forward pass. Pass training=False EXPLICITLY: StochasticDepth
#    short-circuits to the identity only when `training is False`, and at
#    training=None every BatchNormalization uses batch statistics and updates
#    its moving averages, so the forward is not reproducible.
outputs = model(
    {
        'image': np.zeros((2, 256, 256, 3), dtype='float32'),
        'text': np.zeros((2, 77), dtype='int32'),
    },
    training=False,
)

# 3. Inspect the outputs
print(sorted(outputs))
# ['image_features', 'logit_scale', 'logits_per_image',
#  'logits_per_text', 'text_features']
print(outputs['image_features'].shape)   # (2, 512)
print(outputs['logits_per_image'].shape) # (2, 2)

# 4. Or build v1, whose API has the same shape
from dl_techniques.models.mobile_clip import MobileClipModel
v1 = MobileClipModel.from_variant('s0')
v1.build({'image': (None, 256, 256, 3), 'text': (None, 77)})
```

Neither model ships a tokenizer. Both expect CLIP's byte-pair encoding with a 49,408-token vocabulary — pre-trained versions are available in libraries such as Hugging Face `transformers` (e.g. `openai/clip-vit-base-patch32`).

---

## 6. Component Reference

### 6.1 Model Classes and Creation Functions

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`MobileClipV2Model`** | `...mobile_clip.mobile_clip_v2.MobileClipV2Model` | The faithful MobileCLIP2 dual encoder. |
| **`create_mobile_clip_v2`** | `...mobile_clip.mobile_clip_v2.create_mobile_clip_v2` | Recommended convenience function for v2 variants. |
| **`MobileClipModel`** | `...mobile_clip.mobile_clip_v1.MobileClipModel` | The v1 dual encoder. |
| **`create_mobile_clip_model`** | `...mobile_clip.mobile_clip_v1.create_mobile_clip_model` | Recommended convenience function for v1 variants. |

All four are re-exported from the package: `from dl_techniques.models.mobile_clip import ...`.

`MODEL_VARIANTS` is deliberately **not** re-exported at package level. Each model owns its own table as a **class attribute** — `MobileClipModel.MODEL_VARIANTS` and `MobileClipV2Model.MODEL_VARIANTS` — and their key sets are disjoint and not interchangeable. Import each from its own class, explicitly.

### 6.2 Core Building Blocks

| Layer | Location | Purpose |
| :--- | :--- | :--- |
| **`MobileClipTextEncoder`** | `...mobile_clip.components.MobileClipTextEncoder` | The Transformer text encoder — used by **both** models. |
| **`FastVitImageEncoder`** | `...models.fastvit.FastVitImageEncoder` | v2's faithful FastViT MCi image tower. |
| **`MobileClipImageEncoder`** | `...mobile_clip.components.MobileClipImageEncoder` | v1's image encoder: `keras.applications` backbone + projection head. |
| **`ImageProjectionHead`** | `...mobile_clip.components.ImageProjectionHead` | v1's pooling and projection layer. |

Methods both models expose beyond the Keras surface:

| Method | Purpose |
| :--- | :--- |
| `from_variant(variant, **kwargs)` | Build from a `MODEL_VARIANTS` key; kwargs override the row at the TOP level. |
| `encode_image(image, normalize=True, training=None)` | Image branch only. |
| `encode_text(text, normalize=True, training=None)` | Text branch only. |
| `summary(**kwargs)` | Keras summary plus the resolved configuration. |
| `compute_logit_scale()` | v2 only: `clip(exp(logit_scale), 0, logit_scale_max)`. |

---

## 7. Configuration & Model Variants

Both tables are nested the same way. A row is `{'embed_dim': int, 'image_config': {...}, 'text_config': {...}}`, where the two sub-dicts are the **literal constructor keywords** of the respective encoder. `embed_dim` is the joint image-text space and is injected into both towers as `projection_dim`; it must never appear inside a sub-dict.

Two naming hazards, both one nesting level apart:

-   `text_config['embed_dim']` is the **text width**, not the joint space.
-   v2's `image_config['variant']` (`'mci0'`) is FastViT's own kwarg name, a different "variant" from the row key (`'mobileclip2_s0'`).

### MobileCLIP2 Variants (v2)

`MobileClipV2Model.MODEL_VARIANTS`, keyed by the name of the supplied JSON config file each row transcribes. `use_causal_mask` is the **negation** of the JSON's `no_causal_mask` field.

| Variant | `embed_dim` | image `variant` | text width | text heads | text layers | `use_causal_mask` |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`mobileclip2_s0`** | `512` | `mci0` | `512` | `8` | `12` | `False` |
| **`mobileclip2_s2`** | `512` | `mci2` | `512` | `8` | `12` | `False` |
| **`mobileclip2_s3`** | `768` | `mci3` | `768` | `12` | `12` | `False` |
| **`mobileclip2_s4`** | `768` | `mci4` | `768` | `12` | `12` | `False` |
| **`mobileclip_s3`** | `768` | `mci3` | `768` | `12` | `12` | `True` |
| **`mobileclip_s4`** | `768` | `mci4` | `768` | `12` | `12` | `True` |

Every row also carries `vocab_size=49408`, `max_seq_len=77`, `input_shape=(256, 256, 3)` and `intermediate_size = 4 * width`. The table **deliberately holds two families**: the four `mobileclip2_s*` rows are bidirectional, the two `mobileclip_s*` rows causal, over the same image backbones. That single flag is the only reason both appear — do not "simplify" it away.

Each row is checked field by field against the committed copy of its config at `research/mobileclip2_reference/model_configs/` by `test_model_variants_match_supplied_json_configs`, so this table is verified, not re-derived.

A full row looks like this:

```python
"mobileclip2_s0": {
    "embed_dim": 512,                # the joint image-text space
    "image_config": {
        "variant": "mci0",           # FastVit's kwarg name
        "input_shape": (256, 256, 3),
    },
    "text_config": {
        "vocab_size": 49408,
        "max_seq_len": 77,           # JSON: context_length
        "embed_dim": 512,            # the TEXT width, not the joint space
        "num_layers": 12,
        "num_heads": 8,
        "intermediate_size": 2048,   # 4 * width, stated as a literal
        "use_causal_mask": False,    # NOT no_causal_mask
    },
},
```

### MobileCLIP Variants (v1)

`MobileClipModel.MODEL_VARIANTS`. The backbone names are the paper's; this model resolves them to `keras.applications` stand-ins (§4.2).

| Variant | `embed_dim` | image `backbone_name` | image size | text layers | `use_causal_mask` |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **`b`** | `512` | `vit_b16` | `224` | `12` | `True` |
| **`s0`** | `512` | `mci0` | `256` | `4` | `False` |
| **`s1`** | `512` | `mci1` | `256` | `12` | `False` |
| **`s2`** | `512` | `mci2` | `256` | `12` | `False` |

### Counting Parameters

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

### Example 1: Zero-Shot Image Classification

Classify an image using natural-language prompts, with no fine-tuning.

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

Both `encode_*` methods L2-normalize by default. Leave `normalize=True` for anything that feeds a similarity or a contrastive loss.

### Example 2: Embeddings for Downstream Tasks

```python
image_embeddings = model.encode_image(images, training=False)   # (B, D)
text_embeddings = model.encode_text(tokens, training=False)     # (B, D)
```

Use them for visual search, clustering or retrieval. Text embeddings for a fixed prompt set are constant — compute them once and cache.

### Example 3: Overriding One Sub-Config Field

`from_variant` overrides at the **top level**, so passing `text_config=` replaces the row's sub-dict wholesale. To change a single field, merge:

```python
row = MobileClipV2Model.MODEL_VARIANTS['mobileclip2_s0']

model = MobileClipV2Model.from_variant(
    'mobileclip2_s0',
    text_config={**row['text_config'], 'num_layers': 2},
)
```

This is the same convention v1 uses, and it keeps "which fields did the caller actually set" recoverable from `get_config()`.

### Example 4: Using the Image Tower on Its Own

```python
from dl_techniques.models.fastvit import create_fastvit_image_encoder

# CLIP embedding — the terminal Dense IS the projection
tower = create_fastvit_image_encoder('mci0', projection_dim=512)

# Pooled backbone features instead, for dense / transfer use — NOT for CLIP
backbone = create_fastvit_image_encoder('mci0', projection_dim=None)
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Building a Reduced-Depth Model for Tests

A full `mci4` at 256 px does not fit comfortably alongside a training job on a 12 GB card. `image_config` *is* the image tower's keyword set, so any override goes straight in:

```python
row = MobileClipV2Model.MODEL_VARIANTS['mobileclip2_s3']

model = MobileClipV2Model.from_variant(
    'mobileclip2_s3',
    image_config={
        **row['image_config'],
        'layers': (1, 1, 1, 1, 1),           # one block per stage
        'embed_dims': (8, 16, 32, 64, 128),  # narrower channels
        'input_shape': (128, 128, 3),
    },
)
```

Reduce **width and depth**, not the number of stages or the grid: at a single token, softmax is identically 1.0 and the deepest attention stage degenerates.

### Pattern 2: Substituting a Pre-Built Tower

Pass an already-constructed `image_encoder=` / `text_encoder=` to the constructor. This is the route `from_config` uses, and it is why a reduced-depth or otherwise-overridden tower round-trips as itself rather than being re-derived from the configs.

Towers are **never** substituted after construction: Keras refuses a post-build sub-layer swap, and a pre-build one leaves the discarded tower's variables reachable through tracking.

Tower names are fixed by the module constants `IMAGE_TOWER_NAME` / `TEXT_TOWER_NAME` (`"image_encoder"` / `"text_encoder"`). `load_weights_from_checkpoint` matches layers **by name**, so a tower that is ever warm-started independently must be named identically in every model that holds it. Do not rename them.

### Pattern 3: Tuple Outputs and Temperature Control

```python
# Tuple outputs, for a training loop that unpacks positionally.
model = MobileClipV2Model.from_variant('mobileclip2_s0', output_dict=False)
image_features, text_features, per_image, per_text, scale = model(inputs, training=False)

# A custom temperature range.
model = MobileClipV2Model.from_variant(
    'mobileclip2_s0',
    logit_scale_init=0.0,   # raw LOG temperature; default is log(1 / 0.07)
    logit_scale_max=50.0,
)
```

The tuple is a **5-tuple** in the documented key order, with `None` for an absent modality. It is not v1's 3-tuple: dropping to `(image, text, logit_scale)` would silently discard both logits matrices, which v1 never computes and v2 always does.

---

## 10. Performance Optimization

### Mixed Precision Training

Both the Transformer text encoder and the convolutional image tower are good candidates for mixed precision.

```python
import keras
keras.mixed_precision.set_global_policy('mixed_float16')

model = MobileClipV2Model.from_variant('mobileclip2_s0')
```

### Caching Text Embeddings

Attention cost in the image tower is already bounded by the architecture — at most 64 tokens — so the convolutional stages dominate runtime. Reduce `layers` / `embed_dims` before reaching for attention tricks.

The bigger win is architectural. In zero-shot classification with `K` classes, the text tower runs `K` times to build class embeddings and then never again, while the image tower runs on every frame. Compute `encode_text` once, cache it, and run only `encode_image` in the loop.

There is **no fused inference path** (deviation X-1). If you need one, implement it as an explicit, separately tested conversion pass over a trained model; do not silently change the blocks' `call()` paths.

---

## 11. Training and Best Practices

### Optimizer and Schedule

-   **Optimizer**: **AdamW** is the standard choice — effective weight-decay handling matters for the Transformer branch.
-   **Schedule**: **cosine decay with linear warmup** is standard practice for CLIP-style models and generally yields the best results.
-   **Loss**: use stock `fit()` with `CLIPContrastiveLoss` from `dl_techniques.losses`; this repo deliberately avoids custom `train_step` implementations.

```python
from dl_techniques.losses import CLIPContrastiveLoss

model.compile(optimizer='adamw', loss=CLIPContrastiveLoss())
```

### Batch Size is Key

-   Contrastive learning scales directly with **batch size**: a larger batch provides more negatives per positive, and a stronger learning signal. The original CLIP trained above 32,000. On consumer hardware, **gradient accumulation** is essential to simulate that.
-   The model returns a **dict** by default, not a tensor — wire the loss against the key your training script actually consumes, or build with `output_dict=False`.
-   `drop_path_rate` in the image tower is the **maximum** of one global linear ramp over every block of every stage. Setting it per stage is not expressible and would not reproduce the reference.
-   Dropout is now **per tower**: `image_config['dropout_rate']` and `text_config['dropout_rate']` are separate knobs, and `attention_dropout_rate` reaches the text tower only.

---

## 12. Serialization & Deployment

### Saving and Loading

Every class carries `@keras.saving.register_keras_serializable()`, a complete `get_config`, and a `compute_output_shape`, so the standard round trip works.

```python
model.save('mobileclip2_s0.keras')
restored = keras.models.load_model('mobileclip2_s0.keras')
```

### What `get_config` Carries

Two v2 implementation details matter if you subclass or extend:

-   `get_config` serializes **both towers explicitly** (not merely their variant names), so a checkpoint keeps describing the network it was trained with even if a variant table is later corrected. `from_config` re-materializes them with `deserialize_keras_object` and hands them to `__init__` as objects.
-   `get_build_config` / `build_from_config` are overridden because Keras' generic implementation cannot round-trip a **dict** input-shape spec — it warns that the model cannot be built automatically and leaves the restored model unbuilt.

`get_config()` is a **fixed point**: sequence fields such as `input_shape` and `layers` are written as tuples but return from JSON as lists, so they are coerced back on the way in. A restored model's config compares equal to the one it was saved from.

---

## 13. Testing & Validation

### Unit Tests

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
    tests/test_models/test_mobile_clip/ tests/test_models/test_fastvit/ \
    tests/test_layers/test_fastvit/ -q
```

The suites cover initialization, invalid-config raises, forward shape, `compute_output_shape` pre- and post-build, `.keras` save->load compared **by value** (`atol=1e-6, rtol=0`), an **elementwise** weight check on the list-held per-block sub-layers, gradient flow to every trainable weight, and per-class behavioural pins each proven RED against a deliberately broken variant.

A minimal smoke test of your own:

```python
import numpy as np
from dl_techniques.models.mobile_clip import MobileClipV2Model

def test_creation_all_variants():
    """Every variant is constructible (config only — nothing is built)."""
    for variant in MobileClipV2Model.MODEL_VARIANTS:
        model = MobileClipV2Model.from_variant(variant)
        assert model.variant == variant
        print(f"✓ {variant} created successfully")

def test_forward_pass_shapes():
    """A reduced tower produces the documented output contract."""
    row = MobileClipV2Model.MODEL_VARIANTS['mobileclip2_s0']
    model = MobileClipV2Model.from_variant(
        'mobileclip2_s0',
        image_config={**row['image_config'], 'layers': (1, 1, 1, 1),
                      'embed_dims': (8, 16, 32, 64), 'input_shape': (64, 64, 3)},
        text_config={**row['text_config'], 'num_layers': 1, 'max_seq_len': 8,
                     'vocab_size': 64},
    )
    outputs = model(
        {'image': np.zeros((4, 64, 64, 3), 'float32'),
         'text': np.zeros((4, 8), 'int32')},
        training=False,
    )
    assert outputs['image_features'].shape == (4, 512)
    assert outputs['logits_per_image'].shape == (4, 4)
    print("✓ Forward pass shapes are correct")

if __name__ == '__main__':
    test_creation_all_variants()
    test_forward_pass_shapes()
    print("\n✅ All tests passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: My forward pass is not reproducible.**

-   **Cause**: `training=None`. `StochasticDepth` short-circuits to the identity only when `training is False`, and every BatchNormalization in the image tower otherwise uses batch statistics and updates its moving averages.
-   **Solution**: pass `training=False` explicitly.

**Issue 2: Training is unstable and the loss becomes `NaN`.**

-   **Cause 1**: the learning rate is too high.
-   **Cause 2**: the learnable temperature `logit_scale` is diverging.
-   **Solution**: use a smaller learning rate with a warmup schedule. The temperature is already clipped at `logit_scale_max`, which prevents the `inf`-logit failure mode but not a badly scaled optimizer.

**Issue 3: `ValueError: All per-stage tuples must have one entry per stage`.**

-   **Cause**: a 4-stage vs 5-stage mixup in the image tower.
-   **Solution**: `mci0`/`mci1`/`mci2` need 4 entries in every per-stage tuple; `mci3`/`mci4` need 5.

**Issue 4: `ValueError: image_config is missing required key(s)`.**

-   **Cause**: a hand-written sub-config. Both dicts are the encoders' literal constructor keywords, and `image_config` needs at least `variant` and `input_shape`.
-   **Solution**: start from a variant row (`MobileClipV2Model.MODEL_VARIANTS[...]`) and merge your overrides onto it, as in [§8 Example 3](#example-3-overriding-one-sub-config-field).

**Issue 5: My override of one text field wiped out the rest of the row.**

-   **Cause**: `from_variant` overrides at the top level, so `text_config=` replaces the sub-dict wholesale.
-   **Solution**: merge — `text_config={**row['text_config'], 'num_layers': 2}`.

### Frequently Asked Questions

**Q: Should I add a projection head on top of v2's image tower?**

A: No. Its terminal `Dense` already *is* the CLIP image projection, and the model raises if `projection_dim` appears in `image_config`. Use `projection_dim=None` on a standalone `FastVitImageEncoder` only when you want pooled backbone features for a non-CLIP purpose.

**Q: What are the `mci0`, `mci1` and `mci2` backbones?**

A: Custom efficient hybrid architectures designed by Apple, part of the FastViT family. In v2 they are faithfully ported and live in `models/fastvit/`. In v1 the same strings resolve to `keras.applications` MobileNet substitutes (§4.2) — same name, different network.

**Q: Can I load an official MobileCLIP or MobileCLIP2 checkpoint?**

A: No. No weights are ported (deviation X-4) and no conversion script exists here.

**Q: What is the difference between v1's `b` and `s` variants?**

A: `b` is larger and uses a ViT-style image config with a deeper, causally-masked text encoder; the `s` variants target on-device use with convolutional backbones and no causal mask. `s0` is the smallest, with only 4 text Transformer layers.

**Q: Where can I find the correct tokenizer?**

A: Neither model ships one. Use a CLIP-compatible byte-pair-encoding tokenizer with a 49,408 vocabulary — for example Hugging Face's `openai/clip-vit-base-patch32`.

---

## 15. Technical Details

### Asymmetric Design Philosophy

The key insight is the asymmetry of the zero-shot use case. The **image encoder** must be extremely fast, because it processes a new image on every inference. The **text encoder** can be more expensive: for `K` classes it runs `K` times to build class embeddings, which are then reused across many images. Hence a hyper-efficient hybrid CNN for images and a more powerful Transformer for text.

### Causal Masking in the Text Encoder

A causal mask prevents a token from attending to future tokens — standard for autoregressive language models. Its use here is **per variant, not per generation**: v1's `b` sets it and v1's `s*` do not; v2's `mobileclip_s3`/`s4` set it while the four `mobileclip2_s*` rows do not. In the MobileCLIP2 series the text tower is deliberately bidirectional.

### v2 Reference Values

| Detail | Value / rule |
|---|---|
| Input resolution | `256 x 256 x 3`, channels last |
| Context length | `77` tokens |
| Vocabulary | `49408` (OpenAI BPE) — **no tokenizer ships here** |
| `logit_scale` init | `log(1 / 0.07)`, created in `build()` |
| `logit_scale` use | `clip(exp(logit_scale), 0, logit_scale_max)` |
| Attention heads (image tower) | `dim // 32` |
| Drop-path schedule | one global linear ramp over `sum(layers)` blocks, sliced stagewise |
| Text FFN width | `4 * text_config['embed_dim']`, tabulated as a literal |

More image-tower internals — LayerScale, normalization epsilon, the two squeeze-excitation ratios — are in [`models/fastvit/README.md`](../fastvit/README.md) §11.

---

## 16. Deviations from the Reference Implementation (v2)

These apply to `MobileClipV2Model` only. v1's single, larger deviation is its backbone substitution (§4.2); it is not part of this numbered list.

Every known divergence carries a stable id — a deviation that is silently absorbed makes the port unauditable. The image-tower deviations **X-1**, **X-2**, **X-3** and **X-5** are stated in full, with their measurements and RED proofs, in [`models/fastvit/README.md` §9](../fastvit/README.md#9-deviations-from-the-reference-implementation); summarized here:

| Id | Deviation |
|---|---|
| **X-1** | **No structural reparameterization.** Train-time multi-branch form only; no fusion path exists or is tested. The reference release also runs the multi-branch form (`inference_mode=False`), so this matches what it actually executes. |
| **X-2** | **One `use_bias` for both qkv and the output projection.** The shared `MultiHeadAttention` cannot express timm's unbiased-qkv + biased-proj split, so each attention block is short exactly one bias vector of length `dim`. MEASURED and pinned. |
| **X-3** | **`mci0`/`mci1`/`mci2` have no local oracle.** They are transcribed from timm upstream, which is not installed here. `mci3`/`mci4` *do* have a real committed oracle. |
| **X-4** | **No pretrained weights.** Architecture only; **no accuracy claim**. Any comparison against published MobileCLIP2 numbers is invalid by construction. |
| **X-5** | **`RepMixerBlock` is a different architecture that shares the name.** `layers/repmixer_block.py::RepMixerBlock` is not FastViT's block; it is consumed by `models/fastvlm/` and deliberately untouched. Use `layers/fastvit/FastVitRepMixerBlock` for anything that must match timm block-for-block. |

---

## 17. v1 vs v2: Which Should I Use?

Use **v2** for anything that should correspond to the published architecture, or that will consume a real MCi tower. Use **v1** only if you already depend on it — its checkpoints, its variant keys, or its `keras.applications` backbones.

**v1 is not deprecated by v2 and its semantics are unchanged.** The two coexist on purpose: v1's substitution is a recorded, tested decision, not a bug awaiting repair.

| | `MobileClipModel` (v1) | `MobileClipV2Model` (v2) |
|---|---|---|
| Module | `mobile_clip_v1.py` | `mobile_clip_v2.py` |
| Paper | MobileCLIP (Vasu et al., 2024) | MobileCLIP2 (Faghri et al., 2025) |
| Image tower | `keras.applications` MobileNetV2 / V3 **substitute** | faithful FastViT **MCi**, from [`models/fastvit/`](../fastvit/README.md) |
| Text tower | `MobileClipTextEncoder` | the same class, shared |
| Faithful to the reference? | **No**, by its own recorded decision D-001 | Yes, modulo X-1..X-5 |
| Variant keys | `b`, `s0`, `s1`, `s2` | `mobileclip2_s0/s2/s3/s4`, `mobileclip_s3/s4` |
| Image projection | a separate `ImageProjectionHead` | the tower's own terminal `Dense` |
| Structure | nested `MODEL_VARIANTS` class attribute; `(embed_dim, image_config, text_config, ...)` constructor | **the same** |
| Status | shipped and tested; not deprecated | architecture only, no weights, no accuracy claim |

### Why the Text Tower is Shared Rather Than Duplicated

`MobileClipTextEncoder` owns one of exactly **two** block-to-keep causal-mask adapter sites in `src/` (the other is `layers/heads/vlm/factory.py`). A **third** site triggers a mandatory promotion of that adapter into a keep-polarity `MaskFactory` variant — an unrelated refactor with its own blast radius.

Re-implementing the text tower for v2 would create that third site for no architectural gain: the layer is already dimension-generic (measured at 768/12/3072 and 512/8/2048, causal on and off) and already carries the graph-safe `MaskFactory.create_causal_mask` path that `ops.tril` cannot provide on this stack.

The coupling worth naming: if v1's text tower semantics ever change, v2's change with them. Both models are covered by `tests/test_models/test_mobile_clip/`, so such a change cannot land silently.

---

## 18. Citation

This implementation is based on the official MobileCLIP papers from Apple. If you use these models in your research, please cite the original works:

-   **MobileCLIP**:
    ```bibtex
    @inproceedings{vasu2024mobileclip,
      title={MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training},
      author={Vasu, Pavan Kumar Anasosalu and Pouransari, Hadi and Faghri, Fartash and Vemulapalli, Raviteja and Tuzel, Oncel},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2024}
    }
    ```
-   **MobileCLIP2**:
    ```bibtex
    @article{faghri2025mobileclip2,
      title={MobileCLIP2: Improving Multi-Modal Reinforced Training},
      author={Faghri, Fartash and Vasu, Pavan Kumar Anasosalu and Pouransari, Hadi and Vemulapalli, Raviteja and Tuzel, Oncel},
      journal={arXiv preprint arXiv:2508.20691},
      year={2025}
    }
    ```
-   **FastViT** (the image backbone):
    ```bibtex
    @inproceedings{vasu2023fastvit,
      title={FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization},
      author={Vasu, Pavan Kumar Anasosalu and Gabriel, James and Zhu, Jeff and Tuzel, Oncel and Ranjan, Anurag},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
      year={2023}
    }
    ```
-   **CLIP** (the contrastive framework):
    ```bibtex
    @inproceedings{radford2021clip,
      title={Learning Transferable Visual Models From Natural Language Supervision},
      author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
      booktitle={Proceedings of the 38th International Conference on Machine Learning (ICML)},
      year={2021}
    }
    ```
