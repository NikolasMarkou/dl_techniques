# BEiT: BERT Pre-Training of Image Transformers

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of **BEiT** (Bao, Dong, Piao & Wei, *BEiT: BERT Pre-Training of
Image Transformers*, ICLR 2022, [arXiv:2106.08254](https://arxiv.org/abs/2106.08254)) — a
pre-norm Vision Transformer trained with a **masked image modeling** (MIM) objective that
predicts *discrete visual token ids* rather than raw pixels.

The package ships one shared trunk (`BeitModel`) and two consumers that compose it under the
same layer name with **disjoint head prefixes**, so a classifier warm-starts from an MIM
checkpoint layer-for-layer:

| Class | Head prefix | Output |
|:---|:---:|:---|
| `BeitModel` | — | `(B, N+1, D)` token sequence, cls first |
| `BeitForMaskedImageModeling` | `decoder_` | `(B, N, vocab_size)` logits, cls excluded |
| `BeitForImageClassification` | `head_` | `(B, num_classes)` logits |

> ⚠️ **No pretrained weights are distributed with this library, and there is no
> `pretrained=` argument anywhere in this package.** `BeitModel.from_variant(...)` and the
> `create_beit_*` factories always return a randomly-initialized model. To obtain a
> pre-trained trunk you must run the three-stage pipeline in `src/train/beit/` yourself
> (tokenizer → MIM → classification) and transfer the resulting `.keras` checkpoint with
> `load_weights_from_checkpoint` (§9.1). Every code block in this README runs against the
> shipped API exactly as written.

> ⚠️ **This implementation deviates from the reference in four recorded ways** — most
> importantly, its discrete visual tokenizer is a **VQ-VAE**, not BEiT v1's Gumbel-softmax
> DALL·E dVAE. Comparison against published BEiT numbers is therefore **invalid by
> construction**. Read [§15 Deviations from the reference implementation](#15-deviations-from-the-reference-implementation)
> before you cite anything this package produces.

---

## Table of Contents

1. [Overview: What is BEiT and Why It Matters](#1-overview-what-is-beit-and-why-it-matters)
2. [The Problem BEiT Solves](#2-the-problem-beit-solves)
3. [How BEiT Works: Core Concepts](#3-how-beit-works-core-concepts)
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
16. [Technical Details](#16-technical-details)
17. [Citation](#17-citation)

---

## 1. Overview: What is BEiT and Why It Matters

### What is BEiT?

BEiT transplants BERT's masked-token pre-training recipe into vision. The obstacle is that
an image patch is a continuous vector, not a symbol from a vocabulary, so there is nothing
to "predict the id of". BEiT resolves this by giving every image **two views**:

- an **image view** — the usual grid of pixel patches, fed to the ViT encoder, with ~40% of
  the positions replaced by a learnable `[MASK]` token;
- a **token view** — the same image passed through a *frozen* discrete visual tokenizer,
  producing one integer code id per patch position.

The encoder sees the corrupted image view and must predict the token view's code id at each
masked position. That is a plain cross-entropy classification over a codebook — no pixel
regression, no contrastive pairs, no momentum encoder, no negatives.

### Key properties of this implementation

1. **Maximum layer reuse.** Everything except the attention is an existing, tested layer
   from `dl_techniques.layers` — see §4. Exactly **one** new layer was authored.
2. **A shared, separately-checkpointable trunk.** The MIM model and the classifier compose
   the *same* `BeitModel` under the *same* name with disjoint head prefixes, which is what
   makes the warm start a 1:1 transfer rather than a hopeful name match.
3. **No custom `train_step`.** The mask reaches the loss through `sample_weight` in the
   `tf.data` element. Training is stock `model.compile(...)` + `model.fit(ds)`.
4. **Logits everywhere.** Both heads emit logits. Compile with `from_logits=True`.

### Why it matters

**Supervised pre-training** needs labels for every image, which caps the usable corpus at
the size of the labelling budget:

```
   1.3M labelled images  ->  encoder
   (ImageNet-1k)             (bounded by annotation cost)
```

**BEiT** needs none:

```
   unlimited unlabelled images -> tokenizer (self-supervised)
                               -> MIM pre-training (self-supervised)
                               -> small labelled set for fine-tuning
```

---

## 2. The Problem BEiT Solves

Pre-BEiT self-supervised vision fell into two camps, each with a structural cost:

```
 ┌──────────────────────────────────────────────────────────────────┐
 │ Pixel regression (e.g. denoising / inpainting autoencoders)      │
 │   target = raw pixels                                            │
 │   -> the loss is dominated by high-frequency texture the model    │
 │      does not need, and short-range detail is easy to cheat      │
 ├──────────────────────────────────────────────────────────────────┤
 │ Contrastive / instance discrimination                            │
 │   target = "is this the same image?"                             │
 │   -> needs large batches, negative sampling, careful augmentation │
 │      recipes; the pretext task is a training-systems problem     │
 └──────────────────────────────────────────────────────────────────┘
```

BEiT's move is to insert a **discretization step** between the image and the target:

```
   image ──► frozen visual tokenizer ──► integer code ids ──► cross-entropy
                    (fixed)                (a vocabulary)     (BERT's loss)
```

The tokenizer's quantization throws away exactly the low-level detail that made pixel
regression a poor objective, while the resulting symbolic target restores the simple,
batch-size-insensitive classification loss that made BERT practical.

---

## 3. How BEiT Works: Core Concepts

### 3.1 The three stages

```
 stage 0  ┌────────────────────────────────────────────┐
 tokenizer│ train a discrete visual tokenizer on images│  (self-supervised)
          │ -> frozen `.keras` artifact                │
          └───────────────────┬────────────────────────┘
                              │  encode_to_indices(image) -> (gh, gw) int ids
 stage 1  ┌───────────────────▼────────────────────────┐
 MIM      │ block-mask ~40% of patch positions          │  (self-supervised)
          │ predict the code id at the masked positions │
          │ -> `BeitForMaskedImageModeling` checkpoint  │
          └───────────────────┬────────────────────────┘
                              │  warm start (trunk only)
 stage 2  ┌───────────────────▼────────────────────────┐
 finetune │ `BeitForImageClassification` on labels      │  (supervised)
          └────────────────────────────────────────────┘
```

Stage 0 and stage 1 are wired end-to-end in `src/train/beit/`.

### 3.2 Block-wise masking, not i.i.d. masking

BEiT does not mask patches independently. It stamps **rectangular blocks** with a
log-uniform aspect ratio into the patch grid until a budget is reached:

```
   i.i.d. masking (easy)          block-wise masking (BEiT)
   . X . . X . . X . . . X        . . . . . . . . . . . .
   X . . X . . X . . X . .        . . X X X X . . . . . .
   . . X . . X . . X . . X        . . X X X X . . X X . .
   X . . X . . X . . X . .        . . X X X X . . X X . .
   ^ every masked patch has        ^ a masked patch is often surrounded
     unmasked neighbours             by masked patches -> long-range
     -> local interpolation          reasoning is required
       suffices
```

`BeitMaskingGenerator` (`dl_techniques/datasets/vision/beit_masking.py`) is a transcription
of the official `microsoft/unilm/beit/masking_generator.py`, quirks included: the 10-attempt
retry, the strict `h < H` / `w < W` rejection (a block can never span a full grid
dimension), and the **early-termination under-fill** — if 10 consecutive attempts place
nothing, the generator gives up and returns a mask with *fewer* than `num_masking_patches`
cells set, without raising. That is reference behaviour and it is preserved on purpose.

### 3.3 The mask token replaces, it does not drop

Unlike MAE, BEiT never removes tokens from the sequence. At a masked position the patch
embedding is *substituted* by a shared learnable `[MASK]` vector, and the transformer then
processes the **full** `N+1` token sequence. Both masked and unmasked positions attend to
each other in every block.

### 3.4 The loss is restricted by `sample_weight`, not by code

The `tf.data` element is `((image, bool_mask), target_ids, sample_weight)` where
`sample_weight = cast(bool_mask, float32)` exactly — 1.0 at masked positions, 0.0 elsewhere.
Keras multiplies the per-token losses by these weights, so unmasked positions contribute
exactly zero. **No `train_step`, `test_step` or `compute_loss` override exists in this
package, and none may be added.**

---

## 4. Architecture Deep Dive

```
   image (B, H, W, 3)                    bool_mask (B, N)   [MIM only]
         │                                      │
         ▼                                      │
 ┌──────────────────────┐                       │
 │ PatchEmbedding2D     │  [REUSED]             │
 │ Conv2d(k=p, s=p)     │                       │
 └──────────┬───────────┘                       │
       (B, N, D)                                │
            │◄─────────────────────────────────-┘
 ┌──────────▼───────────┐
 │ MaskTokenApply       │  [REUSED]  always CREATED and BUILT,
 │ x[mask] = mask_token │            called only when a mask is passed
 └──────────┬───────────┘
 ┌──────────▼───────────┐
 │ ClassTokenPrepend    │  [REUSED]
 └──────────┬───────────┘
      (B, N+1, D)
            │
      [ optional absolute position embedding — OFF by default ]
            │
 ┌──────────▼─────────────────────────────────────────────────────┐
 │ num_layers ×  TransformerLayer(attention_type='beit')  [REUSED]│
 │   ┌──────────────────────────────────────────────────────────┐ │
 │   │ LayerNorm(eps=1e-12) ─► BeitAttention   ◄── THE ONLY NEW  │ │
 │   │                          (rel. pos. bias, q/v-only bias)  │ │
 │   │ ─► LearnableMultiplier (LayerScale γ₁) ─► StochasticDepth │ │
 │   │ ─► + residual                                             │ │
 │   │ LayerNorm(eps=1e-12) ─► MLP(GELU)                         │ │
 │   │ ─► LearnableMultiplier (LayerScale γ₂) ─► StochasticDepth │ │
 │   │ ─► + residual                                             │ │
 │   └──────────────────────────────────────────────────────────┘ │
 └──────────┬─────────────────────────────────────────────────────┘
            │
      [ final LayerNorm — present ONLY when use_mean_pooling=False, see §15.4 ]
            │
      (B, N+1, D)
        ╱          ╲
 decoder_norm       head_pool  (SequencePooling, cls excluded)  [REUSED]
 decoder_head       head_norm ─► head_dropout ─► head_classifier
 (B, N, vocab)      (B, num_classes)
```

### 4.1 What is reused (everything but one layer)

| Stage | Layer | Module | New? |
|:---|:---|:---|:---:|
| Patch embedding | `PatchEmbedding2D` | `layers/embedding/` via `create_embedding_layer('patch_2d', ...)` | reused |
| Mask substitution | `MaskTokenApply` | `layers/embedding/mask_token.py` | reused |
| CLS token | `ClassTokenPrepend` | `layers/embedding/class_token.py` | reused |
| Absolute pos. emb. (optional) | `PositionalEmbedding` | `create_embedding_layer('positional_learned', ...)` | reused |
| Transformer block | `TransformerLayer` | `layers/transformers/transformer.py` | reused |
| LayerScale (γ₁, γ₂) | `LearnableMultiplier` | inside `TransformerLayer` (`use_layer_scale=True`) | reused |
| Stochastic depth | `StochasticDepth` | inside `TransformerLayer` (`use_stochastic_depth=True`) | reused |
| DropPath ramp | `linear_drop_path_rates` | `utils/drop_path.py` | reused |
| Norms / FFN | `LayerNormalization`, `MLPBlock` | via `TransformerLayer`'s factories | reused |
| Classifier pooling | `SequencePooling(strategy='mean', exclude_positions=[0])` | `layers/sequence_pooling.py` | reused |
| **Self-attention** | **`BeitAttention`** | **`layers/attention/beit_attention.py`** | **NEW** |

`TransformerLayer`'s signature was **not** changed for BEiT. It gained one `'beit'` case in
its per-type attention-parameter table, and the attention factory gained one `'beit'`
registry entry — both purely additive.

### 4.2 `BeitAttention` — why one new layer was unavoidable

Two structural properties of BEiT's attention have no existing implementation in this
repository, and neither is reachable by subclassing:

1. **Asymmetric QKV bias (q and v only).** BEiT's key projection has *no bias parameter at
   all* — structurally absent, not zero-initialized and not frozen. Every existing attention
   class here (`MultiHeadAttention`, `MultiHeadCrossAttention`, `WindowAttention`,
   `SingleWindowAttention`) exposes a single `use_bias` / `qkv_bias` flag governing Q, K and
   V together, and `MultiHeadCrossAttention`'s self-attention path fuses the three into one
   `Dense(dim * 3)`, which puts the asymmetry structurally out of reach.

2. **A cls-augmented relative position bias.** The bias table has
   `(2·Wh − 1)·(2·Ww − 1) + 3` rows over a `(Wh, Ww)` patch grid. The `+3` rows are
   dedicated to the cls→token, token→cls and cls→cls relations, which have no well-defined
   2D displacement. This repository had **no standalone relative-position-bias layer at
   all**: the Swin-family tables are inlined inside `WindowAttention` /
   `SingleWindowAttention`, are window-scoped, use Swin's square `(2W − 1)²` form, and have
   no cls slots — so the index arithmetic is a *different function of the window size* and
   cannot be shared.

The bias is added to the attention logits **before the softmax**:

```
   B_h[i, j] = T[ R[i, j], h ]                 T: (M, num_heads) learnable table
   A_h       = softmax( Q_h K_hᵀ / √d_h + B_h )

   R[i, j] = ((y_i − y_j) + Wh − 1)·(2·Ww − 1) + ((x_i − x_j) + Ww − 1)   patch↔patch
   R[0, j] = M − 3   (cls attends to a patch)
   R[i, 0] = M − 2   (a patch attends to cls)
   R[0, 0] = M − 1   (cls attends to cls)      M = (2·Wh − 1)(2·Ww − 1) + 3
```

`R` is a static, non-trainable integer buffer derived from the patch grid; only `T` is
learned. Non-square grids (`Wh ≠ Ww`) are supported and tested.

### 4.3 The `window_size` of `BeitAttention` is the PATCH GRID

This is the single most confusable parameter in the package. For `'window'`,
`'window_zigzag'` and `'single_window'` attention, `window_size` is a scalar edge length.
For `'beit'` it is the **`(Wh, Ww)` patch grid** of the whole image, and the layer expects
exactly `Wh·Ww + 1` tokens. `BeitModel` computes it as `(H // patch_h, W // patch_w)` and
passes it down for you; you only supply it when constructing `BeitAttention` directly.

### 4.4 `layer_norm_eps = 1e-12`

HF's `BeitConfig` uses `layer_norm_eps=1e-12`, six orders of magnitude tighter than a
generic ViT's `1e-6`. This package passes it **explicitly at every normalization site**
(`attention_norm_args`, `ffn_norm_args`, `decoder_norm`, `head_norm`, `final_norm`) rather
than letting any factory default apply. Copy-pasting a generic ViT block here would silently
change the architecture with no error.

---

## 5. Quick Start Guide

### 5.1 A classifier in six lines

```python
import numpy as np
from dl_techniques.models.beit import create_beit_classifier

model = create_beit_classifier(
    variant="tiny", input_shape=(224, 224, 3), patch_size=16, num_classes=10
)
model.build((None, 224, 224, 3))

logits = model(np.random.rand(2, 224, 224, 3).astype("float32"), training=False)
print(logits.shape)          # (2, 10)   -- LOGITS, not probabilities
```

### 5.2 The MIM model

```python
import numpy as np
from dl_techniques.models.beit import create_beit_mim

mim = create_beit_mim(
    variant="tiny", input_shape=(64, 64, 3), patch_size=16, vocab_size=512
)
mim.build((None, 64, 64, 3))

images = np.random.rand(2, 64, 64, 3).astype("float32")
mask = np.zeros((2, 16), dtype=bool)     # 4x4 patch grid -> N = 16
mask[:, :6] = True                       # mask 6 of the 16 positions

logits = mim((images, mask), training=False)
print(logits.shape)          # (2, 16, 512)  -- cls position already excluded
```

### 5.3 Compiling for training

```python
import keras
from dl_techniques.models.beit import create_beit_mim, create_beit_classifier

mim = create_beit_mim("tiny", (64, 64, 3), 16, vocab_size=512)
mim.build((None, 64, 64, 3))     # ALWAYS build before fit() -- see Issue 6 in section 14
mim.compile(
    optimizer="adamw",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
# The mask is delivered as `sample_weight` in the tf.data element -- see 8.2.

clf = create_beit_classifier("tiny", (64, 64, 3), 16, num_classes=10)
clf.build((None, 64, 64, 3))
clf.compile(
    optimizer="adamw",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
print(mim.name, clf.name)
```

---

## 6. Component Reference

### 6.1 Classes

#### `BeitModel(input_shape=(224,224,3), patch_size=16, scale='base', ...)`
The trunk. Emits the **full** `(B, N+1, D)` sequence, cls token first. Accepts
`image`, `(image, bool_mask)`, or `{'images': ..., 'mask': ...}`. Named `beit_backbone`
(`BACKBONE_NAME`) by default — **do not rename it** on any model that must participate in
the warm start, because `load_weights_from_checkpoint` matches layers *by name*.

Notable constructor arguments (all serialized by `get_config`):

| Argument | Default | Notes |
|:---|:---:|:---|
| `patch_size` | `16` | `int` or `(h, w)` |
| `scale` | `'base'` | key of `SCALE_CONFIGS`, or a `'beit_*'` variant spelling |
| `layer_norm_eps` | `1e-12` | §4.4 |
| `drop_path_rate` | `0.1` | maximum of the linear ramp over `num_layers` |
| `use_absolute_position_embeddings` | `False` | BEiT uses relative bias instead |
| `use_relative_position_bias` | `True` | per-layer tables |
| `use_shared_relative_position_bias` | `False` | `True` **raises** — see §14 |
| `use_mean_pooling` | `True` | also controls the trunk's final norm, §15.4 |
| `hidden_size` / `num_layers` / `num_heads` / `intermediate_size` / `layer_scale_init_value` | `None` | `None` means "take it from `scale`" |

#### `BeitForMaskedImageModeling(backbone, vocab_size=8192)`
Trunk → `decoder_norm` → `decoder_head` → `(B, N, vocab_size)` logits. The cls position is
sliced off **before** the head, so output index `i` is patch `i`; emitting `N+1` logits
would put every target off by one with no error anywhere.

#### `BeitForImageClassification(backbone, num_classes, dropout_rate=0.0)`
Trunk → pooling → `head_norm` → `head_dropout` → `head_classifier` → `(B, num_classes)`
logits. Pooling follows the backbone's `use_mean_pooling` (§15.4).

### 6.2 Factory functions

- `create_beit_backbone(variant='base', input_shape=(224,224,3), patch_size=16, **overrides)`
- `create_beit_mim(variant, input_shape, patch_size, vocab_size=8192, **overrides)`
- `create_beit_classifier(variant, input_shape, patch_size, num_classes=1000, dropout_rate=0.0, **overrides)`

`**overrides` are forwarded verbatim to the `BeitModel` constructor. Prefer these over the
classes: they name the backbone `BACKBONE_NAME` for you, which is a warm-start
precondition.

### 6.3 Module constants

| Name | Value | Meaning |
|:---|:---|:---|
| `BACKBONE_NAME` | `"beit_backbone"` | the stable trunk layer name |
| `DEFAULT_VOCAB_SIZE` | `8192` | DALL·E dVAE codebook size (HF `BeitConfig.vocab_size`) |
| `SCALE_CONFIGS` | dict | width/depth/heads/FFN/layer-scale per scale, §7 |
| `MODEL_VARIANTS` | dict | `'beit_tiny'` … `'beit_large'` → scale |

### 6.4 Related modules outside this package

| Need | Where |
|:---|:---|
| The attention layer | `dl_techniques.layers.attention.BeitAttention`, factory key `'beit'` |
| Block-wise masking + the `tf.data` map fn | `dl_techniques.datasets.vision.beit_masking` |
| Warm-start transfer | `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint` |
| The three-stage trainers | `src/train/beit/` |

---

## 7. Configuration & Model Variants

Parameter counts below are **measured** on the shipped code (backbone only, `224×224×3`
input, `patch_size=16`, i.e. a 14×14 = 196-patch grid).

| Scale | `hidden_size` | Layers | Heads | FFN | `layer_scale_init_value` | Backbone params | Upstream? |
|:---:|:---:|:---:|:---:|:---:|:---:|---:|:---|
| **`tiny`** | 192 | 12 | 3 | 768 | 0.1 | 5,515,056 | ❌ **repo invention** — no BEiT of this size exists in the paper, in HF, or in timm |
| **`small`** | 384 | 12 | 6 | 1536 | 0.1 | 21,646,944 | ❌ **repo invention** — same |
| **`base`** | 768 | 12 | 12 | 3072 | 0.1 | 85,761,216 | ✅ `microsoft/beit-base-patch16-224` |
| **`large`** | 1024 | 24 | 16 | 4096 | **1e-5** | 303,404,544 | ✅ `microsoft/beit-large-patch16-224` (but see §15.2 for the init value) |

`tiny` and `small` exist for cheap CI and smoke runs (deviation **X-3**, §15.3). They are
not reproductions of anything.

`MODEL_VARIANTS` maps the variant spellings onto those scales:

```python
from dl_techniques.models.beit import MODEL_VARIANTS, SCALE_CONFIGS

print(sorted(MODEL_VARIANTS))   # ['beit_base', 'beit_large', 'beit_small', 'beit_tiny']
print(sorted(SCALE_CONFIGS))    # ['base', 'large', 'small', 'tiny']
print(SCALE_CONFIGS["large"]["layer_scale_init_value"])   # 1e-05
```

Both spellings are accepted everywhere a variant is taken:

```python
from dl_techniques.models.beit import BeitModel, create_beit_backbone

a = BeitModel.from_variant("beit_tiny", input_shape=(64, 64, 3), patch_size=16)
b = create_beit_backbone("tiny", (64, 64, 3), 16)
print(a.scale, b.scale, a.grid_size, b.num_patches)   # tiny tiny (4, 4) 16
```

### 7.1 Overriding a scale

Any scale field can be overridden individually; `None` means "inherit from the scale".

```python
from dl_techniques.models.beit import create_beit_backbone

model = create_beit_backbone(
    "base", (224, 224, 3), 16,
    drop_path_rate=0.2,          # heavier stochastic depth for a long run
    hidden_dropout_prob=0.1,
    num_layers=6,                # a half-depth base
)
print(model.num_layers, model.hidden_size, len(model.drop_path_rates))   # 6 768 6
```

---

## 8. Comprehensive Usage Examples

### 8.1 Non-square images and non-square patch grids

The relative-position index is built for a general `(Wh, Ww)` grid, so rectangular inputs
work — the only requirement is divisibility.

```python
import numpy as np
from dl_techniques.models.beit import create_beit_backbone

model = create_beit_backbone("tiny", input_shape=(64, 96, 3), patch_size=16)
model.build((None, 64, 96, 3))
print(model.grid_size, model.num_patches)          # (4, 6) 24

out = model(np.random.rand(2, 64, 96, 3).astype("float32"), training=False)
print(out.shape)                                    # (2, 25, 192)  -- 24 patches + cls
```

### 8.2 A complete MIM `tf.data` pipeline

The masking module knows nothing about tokenizers — you supply `tokenizer_fn`, a callable
mapping one **unbatched** image to per-patch code ids, built from TensorFlow ops (it runs
inside the `tf.data` graph). The stub below stands in for a trained VQ-VAE; `src/train/beit/`
supplies the real one.

```python
import keras
import numpy as np
import tensorflow as tf

from dl_techniques.datasets.vision.beit_masking import make_beit_mim_map_fn
from dl_techniques.models.beit import create_beit_mim

GRID, VOCAB = (4, 4), 512

def fake_tokenizer_fn(image):
    """Stand-in for a frozen VQ-VAE: (H, W, C) -> (gh, gw) int code ids."""
    pooled = tf.nn.avg_pool2d(image[None], ksize=16, strides=16, padding="VALID")[0]
    return tf.cast(tf.reduce_mean(pooled, axis=-1) * (VOCAB - 1), tf.int32)

map_fn = make_beit_mim_map_fn(
    tokenizer_fn=fake_tokenizer_fn,
    grid_size=GRID,
    num_masking_patches=6,      # 6 of 16 positions
    min_num_patches=2,
)

images = np.random.rand(8, 64, 64, 3).astype("float32")
ds = tf.data.Dataset.from_tensor_slices(images).map(map_fn).batch(4)

(img_b, mask_b), targets_b, weights_b = next(iter(ds))
print(img_b.shape, mask_b.shape, targets_b.shape, weights_b.shape)
# (4, 64, 64, 3) (4, 16) (4, 16) (4, 16)
# sample_weight is EXACTLY the mask -- no rescaling
assert np.array_equal(weights_b.numpy(), mask_b.numpy().astype("float32"))

mim = create_beit_mim("tiny", (64, 64, 3), 16, vocab_size=VOCAB)
mim.build((None, 64, 64, 3))     # MANDATORY before fit() -- see Issue 6 in section 14
mim.compile(
    optimizer="adamw",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
mim.fit(ds, epochs=1, verbose=0)
print("one MIM epoch done")
```

### 8.3 Block-wise masks on their own

```python
import random
from dl_techniques.datasets.vision.beit_masking import (
    BeitMaskingGenerator, BEIT_NUM_MASK_PATCHES, BEIT_MIN_MASK_PATCHES_PER_BLOCK,
)

print(BEIT_NUM_MASK_PATCHES, BEIT_MIN_MASK_PATCHES_PER_BLOCK)   # 75 16  (BEiT v1 CLI defaults)

gen = BeitMaskingGenerator(
    input_size=(14, 14),
    num_masking_patches=BEIT_NUM_MASK_PATCHES,
    min_num_patches=BEIT_MIN_MASK_PATCHES_PER_BLOCK,
    rng=random.Random(0),        # optional: the reference uses the GLOBAL random module
)
mask = gen()
print(mask.shape, mask.dtype, int(mask.sum()) <= BEIT_NUM_MASK_PATCHES)
# (14, 14) int64 True   -- '<=' , not '==' : under-fill is reference behaviour
```

### 8.4 Using `BeitAttention` on its own

```python
import numpy as np
from dl_techniques.layers.attention import BeitAttention
from dl_techniques.layers.attention.factory import create_attention_layer

attn = BeitAttention(dim=192, num_heads=3, window_size=(4, 4))
x = np.random.rand(2, 17, 192).astype("float32")     # 4*4 patches + 1 cls
print(attn(x, training=False).shape)                  # (2, 17, 192)

# The same layer through the factory (registry key 'beit').
attn2 = create_attention_layer("beit", dim=192, window_size=(4, 4), num_heads=3)
attn2.build((None, 17, 192))
print(type(attn2).__name__, attn2.k_dense.use_bias, attn2.q_dense.use_bias)
# BeitAttention False True     <- K has NO bias parameter at all
```

### 8.5 A BEiT block through `TransformerLayer`

```python
import numpy as np
from dl_techniques.layers.transformers import TransformerLayer

block = TransformerLayer(
    hidden_size=192, num_heads=3, intermediate_size=768,
    attention_type="beit",
    window_size=(4, 4),                       # the PATCH GRID, not an edge length
    attention_args={"use_relative_position_bias": True},
    normalization_type="layer_norm", normalization_position="pre",
    attention_norm_args={"epsilon": 1e-12},
    ffn_norm_args={"epsilon": 1e-12},
    ffn_type="mlp", activation="gelu",
    use_layer_scale=True, layer_scale_init_value=0.1,
    use_stochastic_depth=True, stochastic_depth_rate=0.05,
)
block.build((None, 17, 192))
print(type(block.attention).__name__)                     # BeitAttention
print(block(np.random.rand(2, 17, 192).astype("float32"), training=False).shape)
```

---

## 9. Advanced Usage Patterns

### 9.1 Warm-starting a classifier from an MIM checkpoint

This is the pattern the whole prefix discipline exists to serve. **Three preconditions**,
each of which fails silently if violated:

1. the target must be **built before** the transfer;
2. the two backbones must share the identical **name** (`BACKBONE_NAME`) and config;
3. the transfer must be **asserted**, not logged — `load_weights_from_checkpoint` does *not*
   raise on a zero-layer trunk transfer when `skip_prefixes` is non-empty, so a model that
   trained entirely from scratch will happily report success.

```python
import os, tempfile
import numpy as np
from dl_techniques.models.beit import (
    BACKBONE_NAME, create_beit_mim, create_beit_classifier,
)
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint

CFG = dict(variant="tiny", input_shape=(64, 64, 3), patch_size=16)

mim = create_beit_mim(vocab_size=512, **CFG)
mim.build((None, 64, 64, 3))

clf = create_beit_classifier(num_classes=10, **CFG)
clf.build((None, 64, 64, 3))          # (1) BEFORE the transfer

with tempfile.TemporaryDirectory() as tmp:
    ckpt = os.path.join(tmp, "beit_mim.keras")
    mim.save(ckpt)
    report = load_weights_from_checkpoint(
        target=clf, ckpt_path=ckpt, skip_prefixes=("decoder_", "head_"),
    )

# (3) ASSERT the transfer happened -- never just log the report.
assert BACKBONE_NAME in report.loaded, report.summary_string()
assert BACKBONE_NAME not in [n for n, _, _ in report.shape_mismatch]
print(sorted(report.skipped_by_prefix))       # ['decoder_head', 'decoder_norm']

src = [np.asarray(w) for w in mim.backbone.get_weights()]
dst = [np.asarray(w) for w in clf.backbone.get_weights()]
print(all(np.array_equal(a, b) for a, b in zip(src, dst)))   # True -- trunk moved 1:1
```

`src/train/beit/train_classification.py` wraps exactly this in a `warm_start_encoder`
helper that **raises** on both failure modes.

### 9.2 Freezing the trunk for linear probing

```python
import numpy as np
from dl_techniques.models.beit import create_beit_classifier

clf = create_beit_classifier("tiny", (64, 64, 3), 16, num_classes=10)
clf.build((None, 64, 64, 3))

clf.backbone.trainable = False
trainable = [w.path for w in clf.trainable_weights]
print(len(clf.trainable_weights), len(clf.weights))      # only the head remains trainable
assert all("backbone" not in p for p in trainable)
print(clf(np.random.rand(2, 64, 64, 3).astype("float32"), training=False).shape)  # (2, 10)
```

### 9.3 Extracting features instead of predictions

```python
import numpy as np
from dl_techniques.models.beit import create_beit_backbone

trunk = create_beit_backbone("tiny", (64, 64, 3), 16)
trunk.build((None, 64, 64, 3))

tokens = trunk(np.random.rand(2, 64, 64, 3).astype("float32"), training=False)
cls_feature = tokens[:, 0, :]        # (2, 192)
patch_mean = np.mean(np.asarray(tokens)[:, 1:, :], axis=1)   # BEiT's own pooling
print(tokens.shape, cls_feature.shape, patch_mean.shape)
```

### 9.4 Switching to the cls-pooling convention

```python
from dl_techniques.models.beit import create_beit_classifier

clf = create_beit_classifier(
    "tiny", (64, 64, 3), 16, num_classes=10, use_mean_pooling=False,
)
clf.build((None, 64, 64, 3))
print(clf.backbone.final_norm is not None, clf.head_pool, clf.head_norm)
# True None None   <- the norm moved INTO the trunk; see 15.4
```

Note that this changes the **layer set**, so an MIM checkpoint trained at
`use_mean_pooling=True` will not transfer cleanly into a `use_mean_pooling=False`
classifier. Keep the flag identical across both stages.

---

## 10. Performance Optimization

### 10.1 Cost drivers

- **Sequence length is quadratic.** `N = (H/p)·(W/p)`. At `224/16` that is 196 tokens; at
  `384/16` it is 576, i.e. ~8.6× the attention FLOPs. Halving the patch size quadruples `N`.
- **The relative-position bias adds a `(num_heads, N+1, N+1)` tensor per block.** It is
  gathered from a table each forward pass; the table itself is tiny
  (`((2Wh−1)(2Ww−1)+3) × num_heads`), but the materialized bias is the same size as the
  attention logits.
- **Stochastic depth costs nothing at inference** (`training=False`), but note that
  `training=None` is **not** inference — pass `training=False` explicitly for a
  deterministic forward.

### 10.2 Mixed precision

```python
import keras, numpy as np
from dl_techniques.models.beit import create_beit_classifier

original = keras.mixed_precision.global_policy()
keras.mixed_precision.set_global_policy("mixed_float16")
try:
    clf = create_beit_classifier("tiny", (64, 64, 3), 16, num_classes=10)
    clf.build((None, 64, 64, 3))
    out = clf(np.random.rand(2, 64, 64, 3).astype("float32"), training=False)
    print(out.dtype, np.isfinite(np.asarray(out, dtype="float32")).all())
finally:
    keras.mixed_precision.set_global_policy(original)
```

Note `layer_norm_eps=1e-12` is far below float16's smallest normal (~6.1e-5). It is added to
a variance inside a `LayerNormalization` that Keras runs in its variable dtype (float32
under `mixed_float16`), so this is safe as shipped — but if you force a norm into float16
compute, raise the epsilon.

### 10.3 Practical levers

| Lever | Effect |
|:---|:---|
| smaller `variant` | linear in params, roughly quadratic-ish in time via width |
| larger `patch_size` | quadratically fewer tokens, coarser targets |
| `drop_path_rate` ↑ | better regularization on long runs, slower convergence |
| `use_absolute_position_embeddings=False` (default) | one fewer `(1, N+1, D)` weight |
| freeze the trunk (§9.2) | linear probing: no backbone gradients |

---

## 11. Training and Best Practices

### 11.1 The reference hyperparameters

From the paper's appendix (see the caveat in §16.3):

| Stage | Optimizer | Peak LR | Weight decay | Warmup | Epochs | Batch | Stoch. depth |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| MIM pre-training | Adam (β=0.9/0.999) | 1.5e-3 | 0.05 | 10 ep | 800 | 2048 | 0.1 |
| Fine-tuning (base) | AdamW + layer-wise decay 0.65 | swept 2e-3…5e-3 | — | — | 100 | 1024 | — |
| Fine-tuning (large) | AdamW + layer-wise decay 0.75 | — | — | — | 50 | 1024 | — |

Dropout is **disabled** during pre-training (`hidden_dropout_prob=0.0`,
`attention_probs_dropout_prob=0.0` — the shipped defaults); regularization comes from
stochastic depth.

### 11.2 Rules this package enforces

- **Weight decay comes from the optimizer only.** Do not also set a `kernel_regularizer` on
  the layers — that double-counts it.
- **`from_logits=True`, always.** Neither head applies a softmax.
- **No custom `train_step`.** The mask is `sample_weight` (§3.4).
- **Keep the backbone config byte-identical across stages 1 and 2**, or the warm start
  transfers a subset of the trunk and the rest stays random.
- **Call `model.build((None, H, W, C))` before `compile()`/`fit()`.** Lazy building inside
  the traced training step raises `InaccessibleTensorError` — see Issue 6 in §14.

### 11.3 Masking budget

BEiT v1's CLI defaults are `--num_mask_patches 75` and `--min_mask_patches_per_block 16` on
a 14×14 grid — 75/196 = 38.3%, which the paper rounds to "roughly 40%". Both are exported as
`BEIT_NUM_MASK_PATCHES` and `BEIT_MIN_MASK_PATCHES_PER_BLOCK`. On a smaller grid you must
scale the budget down yourself: `BeitMaskingGenerator` **raises** if
`num_masking_patches > grid area`.

---

## 12. Serialization & Deployment

All three classes are `@keras.saving.register_keras_serializable()` and round-trip through
the `.keras` format with **value** equality (asserted in the test suite at `atol=1e-6`).
The two head classes serialize their nested backbone via
`serialize_keras_object` / `deserialize_keras_object`.

```python
import os, tempfile
import keras, numpy as np
from dl_techniques.models.beit import create_beit_classifier

clf = create_beit_classifier("tiny", (64, 64, 3), 16, num_classes=10)
clf.build((None, 64, 64, 3))
x = np.random.rand(2, 64, 64, 3).astype("float32")
before = np.asarray(clf(x, training=False))

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "beit_clf.keras")
    clf.save(path)
    restored = keras.models.load_model(path)

after = np.asarray(restored(x, training=False))
np.testing.assert_allclose(before, after, atol=1e-6, rtol=0)
print(type(restored).__name__, restored.backbone.name)   # BeitForImageClassification beit_backbone
```

Config round-trips are also exact:

```python
from dl_techniques.models.beit import BeitModel

a = BeitModel.from_variant("tiny", input_shape=(64, 64, 3), patch_size=16)
b = BeitModel.from_config(a.get_config())
print(b.scale, b.grid_size, b.layer_norm_eps, b.layer_scale_init_value)
# tiny (4, 4) 1e-12 0.1
```

---

## 13. Testing & Validation

The suites below are the real ones, and the counts are what they produce at this commit —
`passed` quoted **with** `collected`, because a suite that runs almost nothing also reports
green.

| Suite | Command | Result |
|:---|:---|:---|
| Model package | `pytest tests/test_models/test_beit/ -q` | **88 passed / 88 collected** |
| Attention layer | `pytest tests/test_layers/test_attention/test_beit_attention.py -q` | **92 passed / 92 collected** |
| Masking + map fn | `pytest tests/test_datasets/test_beit_masking.py -q` | **32 passed / 32 collected** |
| `TransformerLayer` wiring | `pytest tests/test_layers/test_transformers/test_transformer_beit_integration.py -q` | **31 passed / 31 collected** |

All commands are run from the repo root with the `.venv` interpreter and
`CUDA_VISIBLE_DEVICES=1`.

`tests/test_models/test_beit/test_model.py` is organized as nine classes:
`TestBeitScaleConfigs`, `TestBeitModelInitialization`, `TestBeitModelBuild`,
`TestBeitModelForward`, `TestBeitModelSerialization`, `TestBeitArchitectureValidation`,
`TestBeitForMaskedImageModeling`, `TestBeitForImageClassification`, `TestBeitWarmStart`.

Guards worth knowing about, because they encode facts rather than shapes:

- `TestBeitScaleConfigs::test_layer_scale_init_value_split_is_timms` pins the 0.1 / 1e-5
  split (§15.2) so nobody "corrects" it back to HF's uniform 0.1.
- `TestBeitArchitectureValidation::test_final_norm_follows_the_mean_pooling_fork` asserts
  **both** branches of the `use_mean_pooling` fork (§15.4), including that the
  `use_mean_pooling=False` output really is normed (per-token mean ≈ 0, std ≈ 1). A
  dead-component mutation that always creates the norm turns it RED by name.
- `TestBeitWarmStart::test_mim_to_classifier_transfers_the_trunk_values` fails on a
  **zero-layer** transfer rather than passing it, and asserts the trunks start *different*
  so the post-transfer equality is not vacuous.
- `TestBeitAttentionRelativePositionIndex` compares the index matrix against an oracle
  transcribed from the reference pseudocode, **not** from this implementation, at both a
  square and a non-square grid.
- `TestBeitAttentionBiasIsLive` has a documented dead-component mutation (zero the bias
  table) that must turn the named liveness assertion RED.

---

## 14. Troubleshooting & FAQs

**Issue 1: `ValueError: use_shared_relative_position_bias=True is not implemented`.**

- **Cause**: You asked for BEiT's pre-training-only shared-table mode. Only per-layer tables
  are implemented (`False`), which is what every shipped BEiT/BEiTv2 variant in HF and timm
  actually uses.
- **Solution**: Leave it at `False`. Supporting `True` would require threading a
  per-forward-pass bias tensor through `TransformerLayer.call()` — a shared-block signature
  change affecting ~33 unrelated consumers, deliberately out of scope.

**Issue 2: The MIM logits and the target ids are off by one position.**

- **Cause**: A head that does not slice the cls token off. `BeitForMaskedImageModeling`
  outputs `(B, N, vocab)` — `N`, not `N+1` — precisely so output index `i` is patch `i`.
- **Solution**: If you write your own head, slice `tokens[:, 1:, :]` before projecting. This
  failure produces a finite, plausible loss curve and no error.

**Issue 3: `TypeError` or a "parameter compatibility" error mentioning `window_size`.**

- **Cause**: `window_size` is a scalar edge length for the Swin-family attention types but a
  `(Wh, Ww)` **patch grid** for `'beit'` (§4.3).
- **Solution**: Pass a 2-tuple. `BeitModel` does this for you; only direct `BeitAttention` /
  `TransformerLayer` construction requires it.

**Issue 4: The warm start "succeeded" but the classifier trains as if from scratch.**

- **Cause**: `load_weights_from_checkpoint` with a non-empty `skip_prefixes` does **not**
  raise on a zero-layer trunk transfer. A renamed backbone, or a target that was not built
  first, produces a report that looks fine.
- **Solution**: Assert `BACKBONE_NAME in report.loaded` and that it is not in
  `report.shape_mismatch` (§9.1). Never just log the report.

**Issue 5: `ValueError: Image height (H) must be divisible by patch height (p)`.**

- **Cause**: The patch grid must be exact; there is no padding path.
- **Solution**: Resize the input, or pick a `patch_size` that divides both dimensions.
  Non-square grids are fine (§8.1).

**Issue 6: `InaccessibleTensorError: ... Cast:0 ... is out of scope` on the first `fit()` step.**

- **Cause**: You called `fit()` on a model that was never built. Keras then builds it lazily
  *inside* the traced training step, and `BeitAttention.build()` materializes the
  relative-position index with `ops.convert_to_tensor`. That tensor is created in the inner
  `one_step_on_data` FuncGraph and is unreachable from the outer `multi_step_on_iterator`
  graph, so the trace fails. Measured on both `BeitForMaskedImageModeling` and
  `BeitForImageClassification`; the message names
  `<model>/beit_backbone/encoder_layer_0/attention/Cast:0`.
- **Solution**: **Always call `model.build((None, H, W, C))` before `compile()`/`fit()`.**
  Every example in this README does. The trainers in `src/train/beit/` do it as an explicit
  probe, and the warm start requires it anyway (§9.1). Eager forward passes
  (`model(x)`) are unaffected, because they build eagerly.

### Frequently Asked Questions

**Q: Can I load `microsoft/beit-base-patch16-224` weights into this?**

A: Not without writing a converter, and the result would not be faithful anyway — see §15.
No pretrained weights ship with this library and there is no `pretrained=` argument.

**Q: Why is the trunk's final LayerNorm sometimes missing?**

A: Because BEiT's own reference makes it an `Identity` at the default configuration. See
§15.4 — this is a fork, not an omission.

**Q: Why does the key projection have no bias?**

A: Because BEiT's does not. The reference concatenates `[q_bias, zeros_like(v_bias, requires_grad=False), v_bias]`
into the fused QKV bias; HF's independent-projection port expresses the same invariant as
`bias=False` on the K linear. It is a genuine architectural property, not a simplification
made here.

**Q: `training=None` gave me different outputs on two calls. Bug?**

A: No. `training=None` is not inference — stochastic depth and dropout are live. Pass
`training=False` explicitly.

**Q: Is mean pooling or cls pooling correct for classification?**

A: BEiT's own default is mean pooling over the patch tokens with the cls excluded
(`use_mean_pooling=True`), which differs from plain ViT's cls-only convention. Both are
available here; keep the flag identical across pre-training and fine-tuning.

---

## 15. Deviations from the reference implementation

**Read this section before comparing anything from this package to published BEiT
results.** Four deviations are recorded, each decided deliberately and each pinned by a
test.

### 15.1 The visual tokenizer is a VQ-VAE, not a Gumbel-softmax DALL·E dVAE (X-1)

BEiT v1's MIM target comes from a **frozen, externally pre-trained DALL·E dVAE** — a
Gumbel-softmax categorical relaxation over an 8192-entry codebook, trained by OpenAI on data
this repository does not have. This repository has no Gumbel-softmax codebook mechanism
anywhere and no dVAE.

`src/train/beit/` therefore trains a **`VQVAERotationTrickModel`** (hard nearest-neighbour
vector quantization) as stage 0 and uses its `encode_to_indices` output as the MIM target.

**Consequences, stated plainly:**

- The pre-training recipe differs from the paper's — no temperature annealing, no soft
  relaxation, a different codebook geometry, and a tokenizer trained on your data rather
  than OpenAI's.
- **Any comparison to published BEiT accuracy numbers is invalid by construction.** This
  package makes no claim about reproducing them.
- The precedent is real but does not erase the deviation: **BEiT v2** (arXiv:2208.06366)
  itself replaced the dVAE with a VQ-style target (VQ-KD), so a VQ tokenizer is a recognised
  variant of the recipe rather than an unprincipled shortcut. It is still a deviation from
  BEiT **v1**, which is what this package's architecture reproduces.

Note that the *target-construction* step is faithful: BEiT uses `get_codebook_indices`, a
**hard argmax**, for the MIM targets, and `encode_to_indices` is likewise hard.

*(Decision D-002.)*

### 15.2 `layer_scale_init_value` follows timm's split, not HF's uniform value (X-2)

The two primary sources genuinely disagree about the same official checkpoints — this is not
a citation error, both were fetched verbatim:

| Source | base | large |
|:---|:---:|:---:|
| HF `microsoft/beit-{base,large}-patch16-224/config.json` | **0.1** | **0.1** |
| timm `timm/models/beit.py` model entrypoints | **0.1** | **1e-5** |

This package adopts **timm's split**: `0.1` for `tiny`/`small`/`base`, `1e-5` for `large`.
Consequently `SCALE_CONFIGS['large']` does **not** match HF's shipped `config.json`
field-for-field, and anyone diffing against a HF checkpoint config will see that difference.
It is deliberate.

Layer-scale init is a training-time-only hyperparameter — it sets a value the model moves
away from during training and does not constrain a converged checkpoint — so neither number
is "wrong" for its own port. The point of recording it is that an *unrecorded* pick gets
re-litigated every time someone reads a different source.

*(Decision D-003. Pinned by `TestBeitScaleConfigs::test_layer_scale_init_value_split_is_timms`.)*

### 15.3 `tiny` and `small` are repo inventions (X-3)

No BEiT of either size exists in the paper, in HF `transformers`, or in timm — timm's
`beit.py` defines only base and large (plus resolution variants and BEiTv2 base/large).
These two scales exist here so the test suite and CPU smoke runs are cheap. Their widths
follow the usual DeiT-style tiny/small proportions, but they reproduce nothing and should
never be cited as "BEiT-tiny".

*(Decision D-003.)*

### 15.4 The trunk's final LayerNorm follows the `use_mean_pooling` fork (D-007)

`BeitModel.final_norm` is **created and applied only when `use_mean_pooling is False`**.

This mirrors the reference: in HF's port `BeitModel.layernorm` is `nn.Identity()` whenever
`use_mean_pooling=True`, and `BeitPooler` then owns the only LayerNorm on that path (applied
to the mean of the patch tokens); when `use_mean_pooling=False` the trunk applies the
LayerNorm and the pooled output is the raw cls hidden state with no further norm.

| `use_mean_pooling` | trunk `final_norm` | classifier pooling | classifier `head_norm` |
|:---:|:---:|:---|:---:|
| `True` (default) | **absent** | mean over patch tokens, cls excluded | present |
| `False` | present | the cls token | absent |

So there is exactly one normalization on every path. **Do not "clean this up" by always
applying a final norm in the trunk**: at the default configuration that inserts a
normalization the reference does not have, in front of *both* heads. It raises no error,
changes no shape, and produces a perfectly plausible loss curve. The layer is also not
created-then-skipped, because an unused-but-built sub-layer is dead weight in every
checkpoint and the two heads share one backbone config, so there is no warm-start asymmetry
to protect against here.

The visible consequence: `final_norm` is **absent from every default-config checkpoint**,
and this trunk's layer set therefore depends on a config flag — a divergence from the
always-norm shape of this repository's other vision backbones (`ViT.norm`,
`EnergyTransformerBackbone`).

*(Decision D-007. Pinned by `TestBeitArchitectureValidation::test_final_norm_follows_the_mean_pooling_fork`.)*

### 15.5 A single-resolution pipeline (X-4)

BEiT v1 feeds the same image at **two** resolutions: 224 to the ViT encoder (patch 16 →
14×14) and 112 to the dVAE tokenizer (downsample 8 → 14×14), with two different
interpolation filters (bicubic and lanczos). The 112 was chosen *only* so a fixed,
externally-trained /8 dVAE would land on 14×14.

`src/train/beit/` trains its own tokenizer, so that constraint does not bind: the **same**
image tensor feeds both the encoder (`patch_size=16`) and the tokenizer
(`downsample_factor=16`), landing on the identical 14×14 grid. One transform, one `tf.data`
branch, no second interpolation filter to get wrong — at the cost of the tokenizer seeing 4×
the pixels per sample. The grid, which is the thing MIM actually depends on, is identical.

*(Decision D-004.)*

### 15.6 What is *not* a deviation

For the avoidance of doubt, these are faithful: the `(2Wh−1)(2Ww−1)+3` relative-position
table with the exact three cls-interaction slots; the structurally-absent K bias;
`layer_norm_eps=1e-12`; mask-token substitution with full-sequence processing (no token
dropping); pre-norm blocks with LayerScale on both branches and a linear stochastic-depth
ramp; GELU MLP at 4× width; hard-argmax MIM targets; the block-wise masking algorithm
including its under-fill behaviour; `use_absolute_position_embeddings=False`; and mean
pooling with the cls token excluded.

---

## 16. Technical Details

### 16.1 Why the relative position bias needs three extra rows

A coordinate-derived index answers "what is the displacement from token `j` to token `i`?"
For two patches that is a well-defined `(Δy, Δx)` pair, and over a `(Wh, Ww)` grid there are
exactly `(2Wh−1)(2Ww−1)` distinct displacements. The cls token has no grid position, so
three relations — cls→patch, patch→cls, cls→cls — have no displacement to derive an index
from. BEiT gives each its own learned row rather than folding them into a spatial bucket,
which is why the table is `(2Wh−1)(2Ww−1)+3` rows and not a clean square.

### 16.2 Why the mask token must exist in the classifier too

`MaskTokenApply` is created **and built** by every backbone, including the classifier's,
which never calls it. That is deliberate. `load_weights_from_checkpoint` matches layers by
name and compares shapes; if the classifier's trunk lacked the mask token, the trunks would
no longer be weight-identical and the transfer would quietly move a different set of layers.
Deleting the "dead" mask token from the classifier would silently break the warm start
without raising anything. (The absolute position embedding, by contrast, *is* conditional —
it is static backbone config that both stages agree on by construction, so creating it
unconditionally would only add dead weight to every checkpoint.)

### 16.3 Confidence in the reference numbers

The architecture facts in this README (the relative-position index construction, the
no-k-bias QKV asymmetry, `layer_norm_eps=1e-12`, `vocab_size=8192`, the masking algorithm,
the `use_mean_pooling` semantics, and both variant tables) come from directly-fetched
primary sources: `microsoft/unilm/beit/{modeling_finetune,masking_generator,run_beit_pretraining}.py`,
`timm/models/beit.py`, and the raw HF `config.json` files. They are high confidence.

The pre-training/fine-tuning **hyperparameter table in §11.1** is a summary of an
ar5iv-rendered read of the paper's appendix, not a verbatim table transcription. It is
internally consistent with widely-cited BEiT numbers, but if exact reproduction fidelity
matters, read the arXiv PDF's appendix tables directly.

### 16.4 Relationship to BEiT v2 and BEiT-3

**BEiT v2** (arXiv:2208.06366) keeps this backbone but replaces the dVAE target with a
self-trained semantic codebook learned by vector-quantized knowledge distillation (VQ-KD),
where the decoder must reconstruct a frozen teacher's *features* rather than pixels; it also
adds a cls-level patch-aggregation pretext task. **BEiT-3** (arXiv:2208.10442) generalizes
the recipe into a Multiway Transformer with modality-specific expert FFNs behind shared
self-attention, pre-trained with one masked-modelling objective over images, text and
image-text pairs. Neither is implemented here.

---

## 17. Citation

```bibtex
@inproceedings{bao2022beit,
  title={{BEiT}: {BERT} Pre-Training of Image Transformers},
  author={Bao, Hangbo and Dong, Li and Piao, Songhao and Wei, Furu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022},
  eprint={2106.08254},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2106.08254}
}
```

Related work referenced by this implementation:

```bibtex
@article{raffel2020t5,
  title={Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  author={Raffel, Colin and Shazeer, Noam and Roberts, Adam and Lee, Katherine and
          Narang, Sharan and Matena, Michael and Zhou, Yanqi and Li, Wei and Liu, Peter J.},
  journal={Journal of Machine Learning Research},
  volume={21},
  number={140},
  pages={1--67},
  year={2020}
}

@article{peng2022beitv2,
  title={{BEiT} v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers},
  author={Peng, Zhiliang and Dong, Li and Bao, Hangbo and Ye, Qixiang and Wei, Furu},
  journal={arXiv preprint arXiv:2208.06366},
  year={2022}
}
```
