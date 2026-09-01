# Depth Anything

Keras 3 reference implementation of the *Depth Anything* monocular depth
estimation architecture (`encoder + DPT-style decoder`). Source files:

```
src/dl_techniques/models/vision/depth_anything/
├── __init__.py        # public API: DepthAnything, create_depth_anything, DPTDecoder
├── components.py      # DPTDecoder layer (linear default + upsample_factor)
├── model.py           # DepthAnything keras.Model + create_depth_anything factory
└── teacher_ema.py     # TeacherEMACallback, cosine_ema_schedule, linear_ema_schedule
```

The encoder is the in-tree `dl_techniques.models.vision.vit.ViT` (`encoder_kind='real'`,
the default); the DPT decoder defaults to a `linear` output. `train_step` dispatches to a
clean labeled-only path or a semi-supervised path that adds FAL on pooled features **and**
an L1 pseudo-label consistency term on student-vs-teacher depth over `x_unlab`. On-step EMA
decay is provided by `TeacherEMACallback` with cosine/linear schedules, and
`DepthAnything.from_pretrained_encoder(path)` loads encoder weights from a saved `.keras`
checkpoint and re-syncs the teacher. `StrongAugmentation` supports any number of channels,
applies per-sample brightness/contrast factors, and mixes the depth target by the same
CutMix box as the image (the mixing happens in `train_step`, not in `call`). The legacy
Conv-BN-ReLU encoder is preserved behind `encoder_kind='placeholder'` for back-compat.

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
| `frozen_encoder`   | `keras.Model` (clone of encoder, weight-shared at build) | `build()` if `use_feature_alignment=True` **or** `enable_semi_supervised=True` | no |
| `augmentation`     | `StrongAugmentation` layer    | `build()`         | n/a |

Forward path (`call`):

```
x → encoder → [_features_to_spatial if ViT] → decoder → depth
```

`call` does **not** augment. CutMix mixes across batch rows, so the depth
target has to be mixed by the same rectangle, and only the training path holds
a target: `train_step` calls `_augment_with_targets(x, [y, ...])`, which returns
the augmented image together with every identically-mixed target. Both training
paths go through it — the labeled path for `y`, the semi-supervised path a
second time for the teacher's pseudo-depth. `model(x, training=True)` is a plain
un-augmented forward pass.

`train_step` accepts two input shapes:

* `(x, y)` — labeled-only path (default).
* `((x_lab, x_unlab), y_lab)` — semi-supervised path. Active when
  `enable_semi_supervised=True`. Adds a stop-gradient pseudo-label
  L1-consistency term over `x_unlab` against the weight-shared frozen
  teacher, plus a Feature-Alignment Loss term on unlabeled features when
  `use_feature_alignment=True` as well.

---

## Components

### `DepthAnything` (in `model.py`)

A `keras.Model` registered as `dl_techniques.models.depth_anything.model>DepthAnything`
via `@register_dl_technique(...)` (from `dl_techniques.utils.keras_registration`), with
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
    input_value_range:      Optional[Tuple[float,float]] = (0.0, 1.0),
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

**Save/load.** `DepthAnything` overrides Keras 3's `load_own_variables` to
force-build the nested sub-Models with a dummy forward before the framework
restores into them; without it the path-based restore lands on a tree whose
sub-layers do not exist yet and (with a wrapped `ViT` sub-Model) 55 of 172 kernel
arrays come back re-initialized. The save side is the stock Keras recursive save;
the model must not add a second, flat dump of `self.weights` on top of it, because
that runs alongside the framework's path-walking rather than replacing it and
doubles every archive (`vit_l`/384: 610 weights, 1220 HDF5 datasets, 4.88 GB
instead of 2.44 GB). The guard is
`tests/test_models/test_depth_anything/test_the_archive_holds_each_weight_once.py`.

**EMA teacher.** `update_teacher_ema(decay=0.999)` advances the frozen
teacher's weights toward the student via in-place EMA. Call this from a
custom on-step callback when training semi-supervised. Default decay is
0.999.

### `create_depth_anything(...)` (in `model.py`)

Convenience factory that returns a *built* `DepthAnything`:

```python
from dl_techniques.models.vision.depth_anything import create_depth_anything

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

`keras.layers.Layer` registered as
`dl_techniques.models.depth_anything.components>DPTDecoder` via
`@register_dl_technique(...)`. Convolutional decoder head with optional bilinear
upsampling.

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

CutMix + color jitter. Used by `DepthAnything` from inside `train_step`, not
from `call`.

* `call(x, training=True)` returns the mixed image alone and is correct only
  for a target-free consumer. A supervised consumer uses
  `augment_with_mix(x, training=True) -> (x_aug, (mix_mask, perm_indices))` and
  feeds that descriptor to `apply_mix_to_target(target, mix)` for every target
  sharing the batch and spatial axes — the channel count is free, so a
  `(B,H,W,2)` depth+validity target works unchanged.
* `input_value_range` is the caller's **declared** input range; colour jitter
  clips its result back into it. `None` disables clipping, which is what
  standardized or `[-1, +1]` images need — `src/train/depth_anything/` sets it,
  since `src/train/common/megadepth.py` emits RGB in `[-1, +1]` and clipping
  those to `[0, 1]` would zero every negative pixel on the training path only.
* `keras.random.uniform/shuffle` is used in place of the nonexistent
  `keras.ops.random.*`.
* Cutmix gating uses a symbolic mask multiplier (no Python `if`), so the
  layer is fully graph-traceable inside `model.fit`.

---

## Usage

### Forward pass / inference

```python
import keras
from dl_techniques.models.vision.depth_anything import create_depth_anything

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
import keras
from dl_techniques.models.vision.depth_anything import (
    create_depth_anything, TeacherEMACallback, cosine_ema_schedule,
)
from train.common.megadepth import (
    MegaDepthDataset, UnlabeledImageDataset, pair_labeled_unlabeled,
)

model = create_depth_anything(
    encoder_kind='real', encoder_type='vit_l', image_shape=(384, 384, 3),
    use_feature_alignment=True, enable_semi_supervised=True,
)

# Optional: load encoder weights from a saved .keras checkpoint
# (re-syncs the EMA teacher automatically).
# model.from_pretrained_encoder('/path/to/encoder.keras')

model.compile(optimizer=keras.optimizers.AdamW(1e-4))

# Build a paired ((x_lab, x_unlab), y_lab) tf.data.Dataset.
labeled_ds   = MegaDepthDataset(rgb_paths, depth_paths, batch_size=8, patch_size=384)
unlabeled_ds = UnlabeledImageDataset(unlab_paths, batch_size=8, patch_size=384)
paired_ds    = pair_labeled_unlabeled(labeled_ds, unlabeled_ds, patch_size=384, batch_size=8)

# On-step EMA decay (cosine 0.5 → 0.999 over the run).
total_steps = len(labeled_ds) * 100
ema_cb = TeacherEMACallback(
    schedule=cosine_ema_schedule(0.5, 0.999, total_steps),
    warmup_steps=0,
)

model.fit(paired_ds, epochs=100, steps_per_epoch=len(labeled_ds),
          callbacks=[ema_cb])
```

The semi-supervised `train_step` adds two losses on top of the labeled loss:
**FAL** between pooled student/teacher features, and **L1 pseudo-label
consistency** between the student's depth and the teacher's stop-gradient depth
pseudo-labels. In both terms the student sees the *augmented* unlabeled batch
while the teacher reads the *clean* one — that asymmetry is the recipe — and the
teacher's pseudo-depth is then mixed by the same CutMix box the student's input
received, so the consistency term never asks the student to reproduce another
scene's depth over the cut rectangle.

### Serialization

```python
model.save('depth_anything.keras')
loaded = keras.models.load_model('depth_anything.keras')
```

The round-trip is exact on CPU (max-abs-diff = 0.0).

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
| `input_value_range`      | `Tuple[float,float]` \| `None` | `(0.0, 1.0)`                       | Declared input range for the colour-jitter clip. `None` for standardized / `[-1,+1]` inputs. |
| `use_feature_alignment`  | `bool`                     | `True`                                 | Adds the FAL term in the semi-supervised step. Also builds `frozen_encoder` — but so does `enable_semi_supervised`, which needs a teacher for its consistency term. Setting this True with `enable_semi_supervised=False` builds a teacher nothing reads, and warns. |
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

* Custom `train_step` uses `tf.GradientTape` rather than the default Keras
  `train_step`. This is a *sanctioned* exception, not an open defect: the
  semi-supervised path is dual-batch with asymmetric augmentation and reads a
  teacher outside the loss graph, none of which
  `compute_loss(x, y, y_pred, sample_weight, training)` can carry. The rationale
  is written into `model.py`'s module docstring so it ships with the code. Do not
  cite this file as precedent for an ordinary single-batch model. *(LOW.)*

Both `train_step` paths route their metrics through `_finalize_train_step`, which
feeds `self._loss_tracker`; Keras 3 updates that tracker inside the default
`train_step` rather than inside `compute_loss`, so a custom step that skips
`"loss"` in its metrics loop makes `history.history["loss"]` all zeros and kills
every `ModelCheckpoint` / `EarlyStopping` / `ReduceLROnPlateau` that monitors it.
The returned logs dict is flat, not nested under `"compile_metrics"`.
Guards: `tests/test_models/test_depth_anything/test_train_step.py`.

---

## References

- Yang, Lihe et al. **"Depth Anything: Unleashing the Power of Large-Scale
  Unlabeled Data."** CVPR 2024.
- Ranftl, René et al. **"Vision Transformers for Dense Prediction"** (DPT
  decoder). ICCV 2021.
- Oquab, Maxime et al. **"DINOv2: Learning Robust Visual Features without
  Supervision."** 2023.
- In-tree canonical Keras-3 `train_step` pattern:
  `src/dl_techniques/models/language/masked_language_model/mlm.py`.

---

## See also

- `src/train/depth_anything/` — Pattern-5 training scaffold for this model
  (MegaDepth + masked depth loss + visualization callbacks).
- `src/train/CLAUDE.md` § "Pattern 5: Depth Estimation (MegaDepth)" — the pattern
  this trainer implements. `train_depth_anything.py` is the only Pattern-5 trainer.
- `src/dl_techniques/models/vision/depth_anything/components.py` — `DPTDecoder` source.
- `src/dl_techniques/models/vision/depth_anything/model.py` — `DepthAnything` source.
- `tests/test_models/test_depth_anything/` — pytest coverage.
