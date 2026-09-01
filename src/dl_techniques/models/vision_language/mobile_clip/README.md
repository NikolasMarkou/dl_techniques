# MobileCLIP: Fast and Efficient On-Device CLIP

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of Apple's **MobileCLIP** family — dual encoders that adapt CLIP's zero-shot capability to the latency, memory and power limits of edge hardware.

This package ships **two** models that share a text tower and a class structure but differ on the image side:

- **`MobileClipV2Model`** (`mobile_clip_v2.py`) — MobileCLIP2, **the faithful port**, over a real FastViT MCi tower.
- **`MobileClipModel`** (`mobile_clip_v1.py`) — MobileCLIP, **deliberately non-faithful on the image branch**: `keras.applications` CNNs stand in for the MCi backbones.

Neither ships pretrained weights; `pretrained=True` raises `NotImplementedError` on both factories. Neither makes an accuracy claim. Read [§14](#14-deviations-from-the-reference-implementation-v2) before comparing against any published number, and [§15](#15-v1-vs-v2-which-should-i-use) to choose.

---

## 1. Overview: What is MobileCLIP and Why It Matters

MobileCLIP's contribution is **efficiency, not a new training objective**. The contrastive premise is CLIP's: two towers map an image and a caption into one L2-normalized space, and the only supervision is which pairing in the batch is correct. What MobileCLIP changes is the cost of the image side — a hybrid convolution/transformer trunk whose training-time branches reparameterize into a single convolution at inference — and the data side, via multi-modal reinforced training against a captioner-and-ensemble teacher.

**MobileCLIP2** (Faghri et al., 2025) keeps that architecture and improves the training recipe. That is precisely why an architecture-only port can be structurally faithful and still make no accuracy claim.

### Key ideas

1. **Efficient image backbones** (`mci0` … `mci4`) built on FastViT, instead of a large ViT.
2. **Asymmetric design.** The image encoder must be extremely fast, because it runs on every frame. The text encoder, which in zero-shot classification runs once per class and never again, can afford to be comparatively heavier.
3. **Per-variant text masking.** The MobileCLIP2 series runs a **bidirectional** text tower; the earlier MobileCLIP-S3/S4 configs run the classic causal one.
4. **No separate image projection.** The image tower's own terminal `Dense` *is* the CLIP projection, so there is one fewer layer than a naive port would have.

---

## 2. The Problem MobileCLIP Solves

Attention is expensive exactly where the token count is high, and a convolutional token mixer is a perfectly good substitute in the shallow, high-resolution stages. FastViT therefore spends attention only in the last one or two stages:

```
Where the cost goes, per stage (256px input)

  stem    64x64  -> convolutional, no attention
  stage0  64x64  -> RepMixer  (depthwise conv token mixing)
  stage1  32x32  -> RepMixer
  stage2  16x16  -> RepMixer
  stage3   8x8   -> ATTENTION, 64 tokens
  stage4   4x4   -> ATTENTION, 16 tokens   (5-stage variants)

Attention never sees more than 64 tokens. The convolutional stages
dominate runtime, and they are the cheap ones.
```

The second lever is **structural reparameterization**: a FastViT block is trained with several parallel branches — an over-parameterized convolution, a scale branch, an implicit skip — each an affine map over the same input, so at inference they collapse algebraically into a single convolution of the same kernel size. Wide and easy to optimize during training, narrow and cheap afterwards, with no approximation in between. (This port does **not** implement the fusion pass — deviation X-1, §14.)

---

## 3. How MobileCLIP Works: Core Concepts

```
Input dict {'image': (B, 256, 256, 3), 'text': (B, 77)}
    │
    ├─► image ─► FastVitImageEncoder
    │              ├─► stem: MobileOne blocks            -> (B, 64, 64, C0)
    │              ├─► stage_0 .. stage_{N-1}            -> (B, 8, 8, C_last)
    │              ├─► final_conv + GlobalAveragePooling -> (B, C_last * 2)
    │              └─► Dense(embed_dim)  ◄── IS the CLIP projection  -> (B, D)
    │
    ├─► text ──► MobileClipTextEncoder
    │              ├─► token embedding (scaled by embed_dim ** -0.5) + positional
    │              ├─► num_layers x TransformerLayer  (causal or not, per variant)
    │              ├─► LayerNormalization
    │              ├─► EOT-token extraction (argmax over raw token ids)
    │              └─► projection to embed_dim          -> (B, D)
    │
    ├─► ops.normalize(..., axis=-1) on BOTH             -> unit vectors
    ├─► scale = clip(exp(logit_scale), 0, logit_scale_max)
    └─► compute_clip_logits(...)  -> logits_per_image (B, B), logits_per_text (B, B)
```

Both towers return **raw, un-normalized** features from their own `call`. Normalization happens in `encode_image` / `encode_text`, because `compute_clip_logits` expects already-normalized inputs and does not normalize internally.

**The clip on `exp(logit_scale)` is load-bearing.** An unbounded temperature turns a diverging run into `inf` logits and a `nan` loss with no other observable symptom. Both models cap it at 100.

---

## 4. Architecture Deep Dive

### 4.1 `FastVitImageEncoder` — v2's image tower

- **Location**: its own package, [`models/vision/fastvit/`](../../vision/fastvit/README.md). A standalone `keras.Model`, usable without a CLIP model around it.
- **Architecture**: a `MobileOneBlock` stem doing a /4 downsample, then 4 or 5 `FastVitStage`s (RepMixer in the shallow ones, global self-attention in the deepest), a depthwise `final_conv` with squeeze-excitation, global average pooling, and a terminal `Dense`.
- **The terminal `Dense` IS the CLIP image projection.** MobileCLIP's open_clip configs set `"timm_pool": "avg"` with `"timm_proj": null`, so the trunk is instantiated at `num_classes=embed_dim` and does the projecting itself. `embed_dim` is injected as `projection_dim`, and the model **rejects** a `projection_dim` written into `image_config` — stacking a second projection on top would be an unfaithful one.

### 4.2 `MobileClipImageEncoder` — v1's image tower

Deliberately non-faithful: a `keras.applications` backbone plus an `ImageProjectionHead` (global average pool + `Dense` into `embed_dim`).

The official backbones have no equivalent in `keras.applications` — there is no MCi port and no ViT — and since `ImageProjectionHead` opens with `GlobalAveragePooling2D`, the backbone must emit a 4-D `[B, H, W, C]` map, which independently rules out a token-sequence ViT. `components._BACKBONE_ALIASES` therefore resolves:

| Config name | Actually built |
|:---|:---|
| `mci0` | `MobileNetV3Small` |
| `mci1` | `MobileNetV2` |
| `mci2` | `MobileNetV3Large` |
| `vit_b16` | `MobileNetV3Large` |

This is functional buildability chosen over weights fidelity, recorded as the package's D-001. Every tabulated variant also sets `backbone_weights=None`, so not even the substitute's ImageNet weights are loaded. **Nothing here reproduces published MobileCLIP numbers.**

> **Name collision to watch.** `mci0`/`mci1`/`mci2` mean **opposite things** in the two models. In v2 they name real MCi rows (`models.fastvit.MCI_VARIANTS`); in v1 they are keys of `_BACKBONE_ALIASES` resolving to MobileNet stand-ins.

### 4.3 `MobileClipTextEncoder` — shared by both, faithful for both

A plain CLIP transformer. Token embeddings are scaled by `embed_dim ** -0.5` before positional embeddings are added, the stack ends in a `LayerNormalization`, and a raw `(embed_dim, projection_dim)` weight — not a `Dense` — performs the projection into the shared space.

**Pooling is CLIP's end-of-text convention, implemented as `argmax` over the raw token ids.** That is correct only because CLIP's BPE vocabulary assigns EOT the numerically largest id in a well-formed sequence; a tokenizer that breaks that property, or a sequence containing an id above EOT, pools the wrong position silently. The gather is a one-hot matmul so it stays differentiable and backend-agnostic.

**The causal mask is built from `MaskFactory.create_causal_mask` and inverted, not from `ops.tril`.** `ops.tril` routes through a `tf.cond` that rejects a Python-bool predicate the moment it is traced, so it works eagerly and fails on every graph path — `tf.function`, `predict`, `.keras` save/load, XLA. The `logical_not` and cast produce the complementary keep polarity the attention layers want. v1's substitution is confined to the image branch, so v2 shares this class rather than re-implementing it — see [§15](#15-v1-vs-v2-which-should-i-use) for why that is a hard constraint, not a DRY preference.

### 4.4 Contrastive head

Both feature sets are L2-normalized (so the dot product *is* cosine similarity) and scaled by `clip(exp(logit_scale), 0, logit_scale_max)`, where `logit_scale` is a learnable scalar initialized to `log(1 / 0.07)` — `exp` of which is `14.2857`. In v2 that scalar is created in `build()`, not `__init__`.

**v1 and v2 differ in what `call` returns.** v1 returns the two feature tensors and the scale, never the similarity matrices, and emits `None` for a missing modality rather than omitting the key — so v1 consumers must check for `None`, not for key presence. v2 computes and returns both logits matrices, and omits absent keys.

---

## 5. Quick Start Guide

```python
import numpy as np
from dl_techniques.models.vision_language.mobile_clip import MobileClipV2Model

model = MobileClipV2Model.from_variant('mobileclip2_s0')

# Pass training=False EXPLICITLY. StochasticDepth short-circuits to the identity
# only when `training is False`, and at training=None every BatchNormalization
# uses batch statistics and updates its moving averages.
outputs = model(
    {'image': np.zeros((2, 256, 256, 3), dtype='float32'),
     'text':  np.zeros((2, 77), dtype='int32')},
    training=False,
)
print(sorted(outputs))
# ['image_features', 'logit_scale', 'logits_per_image',
#  'logits_per_text', 'text_features']
print(outputs['image_features'].shape)    # (2, 512)
print(outputs['logits_per_image'].shape)  # (2, 2)
print(float(model.compute_logit_scale())) # 14.285715

# v1 has the same API shape.
from dl_techniques.models.vision_language.mobile_clip import MobileClipModel
v1 = MobileClipModel.from_variant('s0')
v1.build({'image': (None, 256, 256, 3), 'text': (None, 77)})
print(sorted(v1({'image': ..., 'text': ...}, training=False)))
# ['image_features', 'logit_scale', 'text_features']   <- no logits matrices
```

**Neither model ships a tokenizer.** Both expect CLIP's byte-pair encoding with a 49,408-token vocabulary — available in libraries such as Hugging Face `transformers` (`openai/clip-vit-base-patch32`).

---

## 6. Component Reference

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`MobileClipV2Model`** | `...mobile_clip.mobile_clip_v2` | The faithful MobileCLIP2 dual encoder. |
| **`create_mobile_clip_v2`** | `...mobile_clip.mobile_clip_v2` | Factory for v2 variants; `pretrained=True` raises. |
| **`MobileClipModel`** | `...mobile_clip.mobile_clip_v1` | The v1 dual encoder. |
| **`create_mobile_clip_model`** | `...mobile_clip.mobile_clip_v1` | Factory for v1 variants; `pretrained=True` raises. |
| **`MobileClipTextEncoder`** | `...mobile_clip.components` | The text encoder — used by **both** models. |
| **`FastVitImageEncoder`** | `...models.vision.fastvit` | v2's faithful FastViT MCi image tower. |
| **`MobileClipImageEncoder`** | `...mobile_clip.components` | v1's `keras.applications` backbone + projection head. |
| **`ImageProjectionHead`** | `...mobile_clip.components` | v1's pooling and projection layer. |

The four model-level names are re-exported from the package. **`MODEL_VARIANTS` is deliberately not** — each model owns its own table as a class attribute and their key sets are disjoint. Reach them explicitly: `MobileClipModel.MODEL_VARIANTS`, `MobileClipV2Model.MODEL_VARIANTS`.

Methods beyond the Keras surface:

| Method | Purpose |
| :--- | :--- |
| `from_variant(variant, **kwargs)` | Build from a table key; kwargs override the row at the **top** level. |
| `encode_image(image, normalize=True, training=None)` | Image branch only. |
| `encode_text(text, normalize=True, training=None)` | Text branch only. |
| `compute_logit_scale()` | v2 only: `clip(exp(logit_scale), 0, logit_scale_max)`. |
| `summary(**kwargs)` | Keras summary plus the resolved configuration. |

---

## 7. Configuration & Model Variants

A row is `{'embed_dim': int, 'image_config': {...}, 'text_config': {...}}`, where the two sub-dicts are the **literal constructor keywords** of the respective encoder. `embed_dim` is the joint image-text space and is injected into both towers as `projection_dim`; it must never appear inside a sub-dict.

> **Two naming hazards, one nesting level apart.**
> `text_config['embed_dim']` is the **text width**, not the joint space.
> v2's `image_config['variant']` (`'mci0'`) is FastViT's own kwarg, a different "variant" from the row key (`'mobileclip2_s0'`).

### MobileCLIP2 variants (v2)

`MobileClipV2Model.MODEL_VARIANTS`, keyed by the name of the supplied JSON config each row transcribes. `use_causal_mask` is the **negation** of the JSON's `no_causal_mask` field.

| Variant | `embed_dim` | image `variant` | text width | heads | layers | `use_causal_mask` |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`mobileclip2_s0`** | 512 | `mci0` | 512 | 8 | 12 | `False` |
| **`mobileclip2_s2`** | 512 | `mci2` | 512 | 8 | 12 | `False` |
| **`mobileclip2_s3`** | 768 | `mci3` | 768 | 12 | 12 | `False` |
| **`mobileclip2_s4`** | 768 | `mci4` | 768 | 12 | 12 | `False` |
| **`mobileclip_s3`** | 768 | `mci3` | 768 | 12 | 12 | `True` |
| **`mobileclip_s4`** | 768 | `mci4` | 768 | 12 | 12 | `True` |

Every row also carries `vocab_size=49408`, `max_seq_len=77`, `input_shape=(256, 256, 3)` and `intermediate_size = 4 * width` (stated as a literal, so the oracle checks a transcription rather than agreeing with a re-derivation of itself).

**The table deliberately holds two families.** The four `mobileclip2_s*` rows are bidirectional, the two `mobileclip_s*` rows causal, over the same image backbones. That single flag is the only reason both appear — do not "simplify" it away. Each row is checked field by field against the committed config at `research/mobileclip2_reference/model_configs/` by `test_model_variants_match_supplied_json_configs`.

### MobileCLIP variants (v1)

`MobileClipModel.MODEL_VARIANTS`. The backbone names are the paper's; this model resolves them to stand-ins (§4.2).

| Variant | `embed_dim` | image `backbone_name` | image size | text layers | `use_causal_mask` |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`b`** | 512 | `vit_b16` | 224 | 12 | `True` |
| **`s0`** | 512 | `mci0` | 256 | 4 | `False` |
| **`s1`** | 512 | `mci1` | 256 | 12 | `False` |
| **`s2`** | 512 | `mci2` | 256 | 12 | `False` |

### Counting parameters

Derive the number rather than trusting one written here:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -c "
from dl_techniques.models.vision_language.mobile_clip import MobileClipV2Model
m = MobileClipV2Model.from_variant('mobileclip2_s0')
m.build({'image': (None,256,256,3), 'text': (None,77)})
print(m.count_params())
"
```

---

## 8. Comprehensive Usage Examples

### Example 1: Zero-shot classification

```python
from keras import ops
from dl_techniques.models.vision_language.mobile_clip import MobileClipV2Model

model = MobileClipV2Model.from_variant('mobileclip2_s0')

# `prompts` is (num_classes, 77) int32 — tokenize with a CLIP BPE tokenizer.
class_embeddings = model.encode_text(prompts, training=False)   # (C, D)
image_embedding = model.encode_image(images, training=False)    # (B, D)

logits = model.compute_logit_scale() * ops.matmul(
    image_embedding, ops.transpose(class_embeddings))           # (B, C)
predictions = ops.argmax(logits, axis=-1)
```

Both `encode_*` methods L2-normalize by default. Leave `normalize=True` for anything feeding a similarity or a contrastive loss.

Text embeddings for a fixed prompt set are constant — cache them; only `encode_image` belongs in the loop.

### Example 2: Overriding one sub-config field

`from_variant` overrides at the **top level**, so passing `text_config=` replaces the row's sub-dict wholesale. To change a single field, merge:

```python
row = MobileClipV2Model.MODEL_VARIANTS['mobileclip2_s0']
model = MobileClipV2Model.from_variant(
    'mobileclip2_s0',
    text_config={**row['text_config'], 'num_layers': 2},
)
```

### Example 3: A reduced-depth model for tests

A full `mci4` at 256 px does not fit comfortably beside a training job on a 12 GB card. `image_config` *is* the tower's keyword set, so any override goes straight in:

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

### Example 4: The image tower alone

```python
from dl_techniques.models.vision.fastvit import create_fastvit_image_encoder

tower = create_fastvit_image_encoder('mci0', projection_dim=512)      # CLIP embedding
backbone = create_fastvit_image_encoder('mci0', projection_dim=None)  # pooled features, NOT CLIP
```

---

## 9. Advanced Usage Patterns

### Substituting a pre-built tower

Pass an already-constructed `image_encoder=` / `text_encoder=` to the constructor. This is the route `from_config` uses, and it is why a reduced-depth or otherwise-overridden tower round-trips as *itself* rather than being re-derived from the configs.

Towers are **never** substituted after construction: Keras refuses a post-build sub-layer swap, and a pre-build one leaves the discarded tower's variables reachable through tracking.

Tower names are fixed by `IMAGE_TOWER_NAME` / `TEXT_TOWER_NAME` (`"image_encoder"` / `"text_encoder"`). `load_weights_from_checkpoint` matches layers **by name**, so a tower that is ever warm-started independently must be named identically in every model that holds it. Do not rename them.

### Tuple outputs and temperature control

```python
model = MobileClipV2Model.from_variant('mobileclip2_s0', output_dict=False)
image_features, text_features, per_image, per_text, scale = model(inputs, training=False)

model = MobileClipV2Model.from_variant(
    'mobileclip2_s0',
    logit_scale_init=0.0,   # the raw LOG temperature; default is log(1 / 0.07)
    logit_scale_max=50.0,
)
```

v2's tuple is a **5-tuple** with `None` for an absent modality, not v1's 3-tuple: dropping to `(image, text, logit_scale)` would silently discard both logits matrices.

### Mixed precision

`keras.mixed_precision.set_global_policy('mixed_float16')` before construction; both towers are good candidates.

---

## 10. Training and Best Practices

- **Optimizer**: AdamW, cosine decay with linear warmup.
- **Loss**: stock `fit()` with `CLIPContrastiveLoss` from `dl_techniques.losses` — this repo deliberately avoids custom `train_step` implementations. `model.compile(optimizer='adamw', loss=CLIPContrastiveLoss())`.
- **Batch size is the main lever.** Contrastive learning scales directly with negatives per positive; the original CLIP trained above 32,000. Gradient accumulation is how you approximate that on consumer hardware.
- **The model returns a dict by default**, not a tensor — wire the loss against the key your training script consumes, or build with `output_dict=False`.
- `drop_path_rate` in the image tower is the **maximum** of one global linear ramp over every block of every stage. A per-stage setting is not expressible and would not reproduce the reference.
- Dropout is **per tower**: `image_config['dropout_rate']` and `text_config['dropout_rate']` are separate knobs, and `attention_dropout_rate` reaches the text tower only.
- Reduce `layers` / `embed_dims` before reaching for attention tricks: attention cost is already bounded at 64 tokens, so the convolutional stages dominate runtime.

---

## 11. Serialization & Deployment

```python
model.save('mobileclip2_s0.keras')
restored = keras.models.load_model('mobileclip2_s0.keras')
```

Every class carries `@register_dl_technique` (from `dl_techniques.utils.keras_registration`), a complete `get_config` and a `compute_output_shape`. The package string is the defining module's dotted path with the `vision_language/` family directory stripped: `dl_techniques.models.mobile_clip.mobile_clip_v1>MobileClipModel`, `...mobile_clip_v2>MobileClipV2Model`, and `dl_techniques.models.mobile_clip.components><ClassName>` for the three component classes. FastViT blocks register under `dl_techniques.layers.fastvit.<module>`. Pre-2026-08-29 archives still load through the legacy `Custom>ClassName` alias.

Three v2 details matter if you subclass or extend:

- `get_config` serializes **both towers explicitly** (not merely their variant names), so a checkpoint keeps describing the network it was trained with even if a variant table is later corrected. `from_config` re-materializes them with `deserialize_keras_object`.
- `get_build_config` / `build_from_config` are overridden because Keras' generic implementation cannot round-trip a **dict** input-shape spec, and would leave the restored model unbuilt.
- `build` checks `built` **per tower**, not only on `self`. On a `.keras` load the towers arrive already built, and the shared text encoder has no idempotence guard — a second `build` re-enters `LayerNormalization.build` and raises.

`get_config()` is a **fixed point**: sequence fields such as `input_shape` and `layers` are written as tuples but return from JSON as lists, so they are coerced back on the way in and a restored model's config compares equal to the one it was saved from.

---

## 12. Testing & Validation

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
    tests/test_models/test_mobile_clip/ tests/test_models/test_fastvit/ \
    tests/test_layers/test_fastvit/ -q
```

The suites cover initialization, invalid-config raises, forward shape, `compute_output_shape` pre- and post-build, `.keras` save/load compared **by value** (`atol=1e-6, rtol=0`), an **elementwise** weight check on the list-held per-block sub-layers, gradient flow to every trainable weight, and per-class behavioural pins each proven RED against a deliberately broken variant.

A smoke test of your own:

```python
import numpy as np
from dl_techniques.models.vision_language.mobile_clip import MobileClipV2Model

def test_creation_all_variants():
    for variant in MobileClipV2Model.MODEL_VARIANTS:
        assert MobileClipV2Model.from_variant(variant).variant == variant

def test_forward_pass_shapes():
    row = MobileClipV2Model.MODEL_VARIANTS['mobileclip2_s0']
    model = MobileClipV2Model.from_variant(
        'mobileclip2_s0',
        image_config={**row['image_config'], 'layers': (1, 1, 1, 1),
                      'embed_dims': (8, 16, 32, 64), 'input_shape': (64, 64, 3)},
        text_config={**row['text_config'], 'num_layers': 1, 'max_seq_len': 8,
                     'vocab_size': 64},
    )
    outputs = model({'image': np.zeros((4, 64, 64, 3), 'float32'),
                     'text': np.zeros((4, 8), 'int32')}, training=False)
    assert outputs['image_features'].shape == (4, 512)
    assert outputs['logits_per_image'].shape == (4, 4)
```

---

## 13. Troubleshooting & FAQs

**My forward pass is not reproducible.** `training=None`. `StochasticDepth` short-circuits to the identity only at `training is False`, and every BatchNormalization otherwise uses batch statistics and updates its moving averages. Pass `training=False` explicitly.

**The loss becomes `nan`.** Either the learning rate or a diverging `logit_scale`. The temperature is already clipped at `logit_scale_max`, which prevents the `inf`-logit mode but not a badly scaled optimizer.

**`ValueError: All per-stage tuples must have one entry per stage`.** A 4-stage vs 5-stage mixup: `mci0`/`mci1`/`mci2` need 4 entries in every per-stage tuple, `mci3`/`mci4` need 5.

**`ValueError: image_config is missing required key(s)`.** A hand-written sub-config. Both dicts are the encoders' literal constructor keywords, and `image_config` needs at least `variant` and `input_shape`. Start from a variant row and merge (§8 Example 2).

**My override of one text field wiped out the rest of the row.** `from_variant` overrides at the top level. Merge: `text_config={**row['text_config'], 'num_layers': 2}`.

**`NotImplementedError: No pretrained MobileCLIP2 weights are distributed`.** Correct behaviour. Build with `pretrained=False` and warm-start from a local file — prefer `dl_techniques.utils.weight_transfer.load_weights_or_raise(model, path)`, which raises when a load changes zero variables. Raw `load_weights` is silent about a checkpoint that matches nothing.

**Should I add a projection head on top of v2's image tower?** No. Its terminal `Dense` already *is* the CLIP projection, and the model raises if `projection_dim` appears in `image_config`. Use `projection_dim=None` on a standalone `FastVitImageEncoder` only for non-CLIP pooled features.

**Can I load an official MobileCLIP checkpoint?** No. No weights are ported (X-4) and no conversion script exists here.

---

## 14. Deviations from the Reference Implementation (v2)

These apply to `MobileClipV2Model` only. v1's single, larger deviation is its backbone substitution (§4.2) and is not part of this numbered list.

Every known divergence carries a stable id — a deviation that is silently absorbed makes the port unauditable. **X-1**, **X-2**, **X-3** and **X-5** are stated in full, with measurements and RED proofs, in [`models/vision/fastvit/README.md` §9](../../vision/fastvit/README.md#9-deviations-from-the-reference-implementation).

| Id | Deviation |
|---|---|
| **X-1** | **No structural reparameterization.** Train-time multi-branch form only; no fusion path exists or is tested. The reference release also runs the multi-branch form (`inference_mode=False`), so this matches what it actually executes. |
| **X-2** | **One `use_bias` for both qkv and the output projection.** The shared `MultiHeadAttention` cannot express timm's unbiased-qkv + biased-proj split, so each attention block is short exactly one bias vector of length `dim`. MEASURED and pinned. |
| **X-3** | **`mci0`/`mci1`/`mci2` have no local oracle.** Transcribed from timm upstream, which is not installed here. `mci3`/`mci4` *do* have a committed oracle. |
| **X-4** | **No pretrained weights.** Architecture only; **no accuracy claim**. Any comparison against published MobileCLIP2 numbers is invalid by construction. |
| **X-5** | **`RepMixerBlock` is a different architecture sharing the name.** `layers/repmixer_block.py::RepMixerBlock` is not FastViT's block; it is consumed by `models/vision_language/fastvlm/` and deliberately untouched. Use `layers/fastvit/FastVitRepMixerBlock` for anything that must match timm block-for-block. |

---

## 15. v1 vs v2: Which Should I Use?

Use **v2** for anything that should correspond to the published architecture, or that will consume a real MCi tower. Use **v1** only if you already depend on it — its checkpoints, its variant keys, or its `keras.applications` backbones. **v1 is not deprecated by v2**; its substitution is a recorded, tested decision, not a bug awaiting repair.

| | `MobileClipModel` (v1) | `MobileClipV2Model` (v2) |
|---|---|---|
| Module | `mobile_clip_v1.py` | `mobile_clip_v2.py` |
| Paper | MobileCLIP (Vasu et al., 2024) | MobileCLIP2 (Faghri et al., 2025) |
| Image tower | `keras.applications` MobileNet **substitute** | faithful FastViT **MCi** |
| Text tower | `MobileClipTextEncoder` | the same class, shared |
| Faithful? | **No**, by its own D-001 | Yes, modulo X-1..X-5 |
| Variant keys | `b`, `s0`, `s1`, `s2` | `mobileclip2_s0/s2/s3/s4`, `mobileclip_s3/s4` |
| Image projection | a separate `ImageProjectionHead` | the tower's own terminal `Dense` |
| `call` output | features + scale; `None` for a missing modality | features + both logits matrices; absent keys omitted |
| Pretrained weights | none; `pretrained=True` raises | none; `pretrained=True` raises |

### Why the text tower is shared rather than duplicated

`MobileClipTextEncoder` owns one of exactly **two** block-to-keep causal-mask adapter sites in `src/` (the other is `layers/heads/vlm/factory.py`). A **third** site triggers a mandatory promotion of that adapter into a keep-polarity `MaskFactory` variant — an unrelated refactor with its own blast radius.

Re-implementing the tower for v2 would create that third site for no architectural gain: the layer is already dimension-generic (measured at 768/12/3072 and 512/8/2048, causal on and off) and already carries the graph-safe mask path (§4.3).

The coupling worth naming: if v1's text tower semantics ever change, v2's change with them. Both models are covered by `tests/test_models/test_mobile_clip/`, so such a change cannot land silently.

---

## 16. Citation

```bibtex
@inproceedings{vasu2024mobileclip,
  title={MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training},
  author={Vasu, Pavan Kumar Anasosalu and Pouransari, Hadi and Faghri, Fartash
          and Vemulapalli, Raviteja and Tuzel, Oncel},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}

@article{faghri2025mobileclip2,
  title={MobileCLIP2: Improving Multi-Modal Reinforced Training},
  author={Faghri, Fartash and Vasu, Pavan Kumar Anasosalu and Pouransari, Hadi
          and Vemulapalli, Raviteja and Tuzel, Oncel},
  journal={arXiv preprint arXiv:2508.20691},
  year={2025}
}

@inproceedings{vasu2023fastvit,
  title={FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization},
  author={Vasu, Pavan Kumar Anasosalu and Gabriel, James and Zhu, Jeff
          and Tuzel, Oncel and Ranjan, Anurag},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

The contrastive framework is CLIP (Radford et al., 2021, [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)); the reparameterizable block family is RepVGG (Ding et al., 2021, [arXiv:2101.03697](https://arxiv.org/abs/2101.03697)).
