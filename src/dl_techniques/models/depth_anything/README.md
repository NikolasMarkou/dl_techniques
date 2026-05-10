# Depth Anything

Keras 3 reference implementation of the *Depth Anything* monocular depth
estimation architecture (`encoder + DPT-style decoder`). Source files:

```
src/dl_techniques/models/depth_anything/
├── __init__.py        # public API: DepthAnything, create_depth_anything, DPTDecoder
├── components.py      # DPTDecoder layer (linear default + upsample_factor)
└── model.py           # DepthAnything keras.Model + create_depth_anything factory
```

> **Status (post-`plan_2026-05-10_bd098beb`).** The model now uses the
> in-tree `dl_techniques.models.vit.ViT` as its real encoder
> (`encoder_kind='real'`, default), the DPT decoder defaults to `linear`
> output (no longer sigmoid-clamped), and `train_step` supports both the
> labeled-only path and a semi-supervised path keyed on
> `enable_semi_supervised`. Save/load round-trip is verified end-to-end on
> CPU (max-abs-diff = 0.0). The legacy Conv-BN-ReLU encoder is preserved
> behind `encoder_kind='placeholder'` for back-compat.
>
> What is **still open** (deeper, separate-plan work): EMA-trained teacher
> with on-step decay schedule, pseudo-label depth on the unlabeled stream,
> per-sample CutMix factors, and external pretrained-encoder weight
> loading (e.g. DINOv2). See "Known Issues" below.

---

## Overview

`DepthAnything` is a `keras.Model` subclass that takes an RGB image batch
and produces a single-channel depth map at the same spatial resolution:

```
input  : (B, H, W, 3)   float32 RGB
output : (B, H, W, 1)   float32 depth (linear by default)
```

It is composed of three sub-networks:

| Sub-network        | Type                          | Where built       | Trainable |
|--------------------|-------------------------------|-------------------|-----------|
| `encoder`          | `keras.Model` (`ViT` or Conv) | `__init__`/`build()` | yes |
| `decoder`          | `DPTDecoder`                  | `build()`         | yes |
| `frozen_encoder`   | `keras.Model` (clone of encoder, weight-shared at build) | `build()` if `use_feature_alignment=True` | no |
| `augmentation`     | `StrongAugmentation` layer    | `build()`         | n/a |

Forward path (`call`):

```
x → [augmentation if training] → encoder → [_features_to_spatial if ViT] → decoder → depth
```

`train_step` accepts two input shapes:

* `(x, y)` — labeled-only path (default).
* `((x_lab, x_unlab), y_lab)` — semi-supervised path. Active only when
  `enable_semi_supervised=True` AND `use_feature_alignment=True`. Adds a
  Feature-Alignment Loss term on unlabeled features against the
  weight-shared frozen teacher.

---

## Components

### `DepthAnything` (in `model.py`)

A `@keras.saving.register_keras_serializable()`-decorated `keras.Model` with
full `get_config()` / `from_config()` round-trip.

**Constructor (relevant args)**:

```python
DepthAnything(
    encoder_type:           str = 'vit_l',                # {'vit_s','vit_b','vit_l'}
    image_shape:            Tuple[int,int,int] = (384, 384, 3),
    decoder_dims:           Optional[List[int]] = [256, 128, 64, 32],
    output_channels:        int  = 1,
    kernel_initializer:     Union[str, Initializer] = 'he_normal',
    kernel_regularizer:     Optional[Regularizer]   = None,
    loss_weights:           Optional[Dict[str,float]] = {'labeled':1.0,
                                                         'unlabeled':0.5,
                                                         'feature':0.1},
    cutmix_prob:            float = 0.5,
    color_jitter_strength:  float = 0.2,
    use_feature_alignment:  bool  = True,
    encoder_kind:           str   = 'real',           # 'real' | 'placeholder'
    enable_semi_supervised: bool  = False,
    encoder:                Optional[keras.Model] = None,   # for from_config
    input_shape:            Optional[Tuple[int,int,int]] = None,  # legacy alias
    **kwargs,
)
```

> **`input_shape` → `image_shape` rename.** `image_shape` is the canonical
> kwarg. `input_shape` is accepted as a deprecated alias for one cycle so
> previously-saved configs continue to load.

**Save/load — D-004 override.** `DepthAnything` overrides Keras 3's
`save_own_variables` / `load_own_variables` to persist `self.weights` as a
flat numeric-keyed store at the model level. This bypasses the framework's
path-walking, which (with a wrapped `ViT` sub-Model) was dropping 55/172
kernel arrays during load. The override force-builds nested sub-Models on
load when needed so `self.weights` matches the saved store. See
`# DECISION plan_2026-05-10_bd098beb/D-004` in `model.py`.

**EMA teacher.** `update_teacher_ema(decay=0.999)` advances the frozen
teacher's weights toward the student via in-place EMA. Call this from a
custom on-step callback when training semi-supervised. Default decay is
0.999.

### `create_depth_anything(...)` (in `model.py`)

Convenience factory that returns a *built* `DepthAnything`:

```python
from dl_techniques.models.depth_anything import create_depth_anything

model = create_depth_anything(
    encoder_kind='real',
    encoder_type='vit_l',
    image_shape=(384, 384, 3),
    decoder_dims=[256, 128, 64, 32],
    use_feature_alignment=False,
)
```

The factory ends with a dummy forward pass so the model is fully built
when returned.

### `DPTDecoder` (in `components.py`)

`@keras.saving.register_keras_serializable()` `keras.layers.Layer`.
Convolutional decoder head with optional bilinear upsampling.

**Architecture**:

```
features (B, h, w, C_in)
   │
   ├── for dim in dims[:-1]:
   │       Conv3x3(dim) → BN → ReLU → [UpSample2D(2x bilinear) if upsample_factor>1]
   │
   ├── Conv3x3(output_channels) with output_activation        # default 'linear'
   │
   └── depth (B, h*upsample_factor, w*upsample_factor, output_channels)
```

`output_activation` defaults to `'linear'`. `upsample_factor` defaults to
`1`; `DepthAnything` passes `upsample_factor=encoder_stride` so the decoder
lifts features back to input resolution.

### `StrongAugmentation` (in `dl_techniques.layers.strong_augmentation`)

CutMix + color jitter. Used by `DepthAnything` during `training=True` forward
passes. Two recent fixes (D-005 + follow-up):

* `keras.random.uniform/shuffle` now used in place of nonexistent
  `keras.ops.random.*`.
* Cutmix gating uses a symbolic mask multiplier (no Python `if`), so the
  layer is fully graph-traceable inside `model.fit`.

---

## Usage

### Forward pass / inference

```python
import keras
from dl_techniques.models.depth_anything import create_depth_anything

model = create_depth_anything(encoder_kind='real', encoder_type='vit_l',
                              image_shape=(384, 384, 3),
                              use_feature_alignment=False)
x = keras.random.normal([2, 384, 384, 3])
depth = model(x, training=False)
print(depth.shape)  # (2, 384, 384, 1)
```

### Compile / train (labeled-only)

```python
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=5e-6, weight_decay=1e-5),
    loss=keras.losses.MeanSquaredError(),
)
# model.fit(x_train, y_train, epochs=...)
```

### Semi-supervised usage

```python
model = create_depth_anything(
    encoder_kind='real', encoder_type='vit_l', image_shape=(384, 384, 3),
    use_feature_alignment=True, enable_semi_supervised=True,
)
model.compile(optimizer=keras.optimizers.AdamW(1e-4))
# data shape per batch: ((x_lab, x_unlab), y_lab)
# model.fit(ds, epochs=...)
# Optionally advance the EMA teacher on each step:
# class TeacherEMACallback(keras.callbacks.Callback):
#     def on_train_batch_end(self, *_):
#         self.model.update_teacher_ema(decay=0.999)
```

The infrastructure is wired (frozen weight-shared teacher, FAL term inside
`train_step`); the open work is feeding `((x_lab, x_unlab), y_lab)` from a
real dataset (the in-tree `MegaDepthDataset` only yields labeled batches).

### Serialization

```python
model.save('depth_anything.keras')
loaded = keras.models.load_model('depth_anything.keras')
```

Verified end-to-end on CPU (max-abs-diff = 0.0; SC-6 in
`plan_2026-05-10_bd098beb`).

---

## Configuration

| Argument                 | Type                       | Default                                | Notes |
|--------------------------|----------------------------|----------------------------------------|-------|
| `encoder_type`           | `str`                      | `'vit_l'`                              | One of `{vit_s, vit_b, vit_l}`. Picks ViT scale. |
| `encoder_kind`           | `str`                      | `'real'`                               | `'real'` builds in-tree `ViT`; `'placeholder'` builds the legacy Conv-BN-ReLU. |
| `image_shape`            | `Tuple[int,int,int]`       | `(384, 384, 3)`                        | Canonical kwarg. `input_shape` accepted as deprecated alias. |
| `decoder_dims`           | `List[int]`                | `[256, 128, 64, 32]`                   | Stage widths; last entry is the penultimate decoder width. |
| `output_channels`        | `int`                      | `1`                                    | Final depth-map channel count. |
| `kernel_initializer`     | `str` \| `Initializer`     | `'he_normal'`                          | |
| `kernel_regularizer`     | `Regularizer` \| `None`    | `None`                                 | Do **not** combine with `AdamW(weight_decay=...)`. |
| `loss_weights`           | `Dict[str,float]`          | `{labeled:1.0, unlabeled:0.5, feature:0.1}` | Consumed by the semi-sup `train_step` path. |
| `cutmix_prob`            | `float`                    | `0.5`                                  | Forwarded to `StrongAugmentation`. |
| `color_jitter_strength`  | `float`                    | `0.2`                                  | Forwarded to `StrongAugmentation`. |
| `use_feature_alignment`  | `bool`                     | `True`                                 | Builds `frozen_encoder` (weight-shared at build). |
| `enable_semi_supervised` | `bool`                     | `False`                                | Switches `train_step` to `((x_lab, x_unlab), y_lab)` mode. |

`DPTDecoder` (used internally):

| Argument            | Default       | Notes |
|---------------------|---------------|-------|
| `dims`              | required      | Forwarded from `DepthAnything.decoder_dims`. |
| `output_channels`   | `1`           | Forwarded. |
| `output_activation` | `'linear'`    | Linear is the canonical depth-estimation output. |
| `upsample_factor`   | `1`           | Forwarded as `encoder_stride` (16 for both real and placeholder). Power of 2 only. |
| `kernel_initializer`| `'he_normal'` | |
| `kernel_regularizer`| `None`        | |

---

## Known Issues

The 14-item review from `plan_2026-05-10_44694bc9` plus D-005 has been
folded into the work below. Item numbers are kept stable for traceability.

**FIXED in `plan_2026-05-10_bd098beb`** (this plan):

* **#1** — Encoder is now a real `ViT` backbone (`encoder_kind='real'`,
  default). Placeholder Conv-BN-ReLU preserved behind `'placeholder'`.
* **#2** — Frozen teacher is now weight-shared at build via
  `keras.models.clone_model(student) + set_weights(student.get_weights())`.
  EMA advance via `update_teacher_ema(decay=...)`. *Open at deeper layer*:
  on-step EMA decay schedule + integration with a real pretrained student.
* **#3** — Semi-supervised infrastructure wired: `enable_semi_supervised`
  flag, `train_step` accepts `((x_lab, x_unlab), y_lab)`,
  `FeatureAlignmentLoss` term on unlabeled features. *Open at deeper layer*:
  pseudo-label depth on unlabeled stream + dataset-side
  `((x_lab, x_unlab), y_lab)` pairing.
* **#4** — `train_step` Keras-3 API (was fixed in `plan_44694bc9`); the
  `# DECISION plan_2026-05-10_44694bc9/D-003` anchor is preserved.
* **#6** — `DPTDecoder.output_activation` defaults to `'linear'`. Compatible
  with `AffineInvariantLoss` and the masked-L1 + gradient loss.
* **#7** — Real ViT encoder is constructed in `__init__`, not `build()`,
  for the real path; serialization is verified by SC-6 (max-abs-diff = 0.0)
  via the D-004 `save_own_variables` / `load_own_variables` override.
* **#8** — Dead `if X is not None` guards and `_create_fallback_decoder`
  removed.
* **#10** — `encoder_type` now actually selects ViT scale
  (`small`/`base`/`large`) — no longer cosmetic.
* **#11** — `input_shape` renamed to `image_shape` (legacy kwarg accepted
  for one cycle).
* **#12** — Test module added at `tests/test_models/test_depth_anything/`
  covering build, forward, save/load, train_step (labeled and semi-sup),
  decoder default, StrongAugmentation forward.
* **#13** — `frozen_encoder` weight-share issue subsumed by #2.
* **#14** — `compile()` override no longer stashes dead `self.depth_loss` /
  `self.feature_loss`.
* **D-005** — `StrongAugmentation` uses `keras.random.uniform/shuffle`
  (and a graph-mode cutmix gate; see #9 below for what's still pending).

**STILL OPEN**:

* **#5** — Custom `train_step` uses `tf.GradientTape` rather than
  default Keras `train_step`. Acceptable for now (semi-sup path needs the
  custom tape). *(LOW.)*
* **#2-deeper** — On-step EMA decay schedule + integration with a real
  pretrained student.
* **#3-deeper** — Pseudo-label depth on unlabeled stream + dataset-side
  `((x_lab, x_unlab), y_lab)` pairing.
* **#9** — `StrongAugmentation._apply_cutmix` still hard-codes 3 channels
  and uses batch-scalar (not per-sample) brightness/contrast factors.
  Cutmix gating is now graph-mode safe (D-005 follow-up). *(LOW.)*

**REMOVED** (issue no longer applicable):

* **#12** — there are now tests.

---

## References

- Yang, Lihe et al. **"Depth Anything: Unleashing the Power of Large-Scale
  Unlabeled Data."** CVPR 2024.
- Ranftl, René et al. **"Vision Transformers for Dense Prediction"** (DPT
  decoder). ICCV 2021.
- Oquab, Maxime et al. **"DINOv2: Learning Robust Visual Features without
  Supervision."** 2023.
- In-tree canonical Keras-3 `train_step` pattern:
  `src/dl_techniques/models/masked_language_model/mlm.py:309-343`.
- D-004 save/load override:
  `src/dl_techniques/models/depth_anything/model.py` (search for
  `# DECISION plan_2026-05-10_bd098beb/D-004`).

---

## See also

- `src/train/depth_anything/` — Pattern-5 training scaffold for this model
  (MegaDepth + masked depth loss + visualization callbacks).
- `src/train/cliffordnet/train_depth_estimation.py` — reference Pattern-5
  trainer that the depth_anything trainer mirrors.
- `src/dl_techniques/models/depth_anything/components.py` — `DPTDecoder` source.
- `src/dl_techniques/models/depth_anything/model.py` — `DepthAnything` source.
- `tests/test_models/test_depth_anything/` — pytest coverage.
