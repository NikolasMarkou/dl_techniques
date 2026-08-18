# Depth Anything

Keras 3 reference implementation of the *Depth Anything* monocular depth
estimation architecture (`encoder + DPT-style decoder`). Source files:

```
src/dl_techniques/models/depth_anything/
├── __init__.py        # public API: DepthAnything, create_depth_anything, DPTDecoder
├── components.py      # DPTDecoder layer (linear default + upsample_factor)
└── model.py           # DepthAnything keras.Model + create_depth_anything factory
```

> **Status (post-`plan_2026-05-10_54e6e303`).** The model uses the in-tree
> `dl_techniques.models.vit.ViT` as its real encoder (`encoder_kind='real'`,
> default); the DPT decoder defaults to `linear` output;  `train_step`
> dispatches to a clean labeled-only path or a clearly-delimited
> semi-supervised path that adds FAL on pooled features **and** an L1
> pseudo-label consistency term on student-vs-teacher depth on `x_unlab`.
> On-step EMA decay is provided by `TeacherEMACallback` with cosine/linear
> schedules, and `DepthAnything.from_pretrained_encoder(path)` loads encoder
> weights from a saved `.keras` checkpoint and re-syncs the teacher.
> `StrongAugmentation` supports any number of channels, applies per-sample
> brightness/contrast factors, and mixes the depth target by the same CutMix box
> as the image (the mixing now happens in `train_step`, not in `call`). Save/load round-trip is verified
> end-to-end on CPU (max-abs-diff = 0.0). The legacy Conv-BN-ReLU encoder
> is preserved behind `encoder_kind='placeholder'` for back-compat.
>
> Only one residual note remains (`tf.GradientTape` in the custom
> `train_step` — required by the semi-sup path). See "Known Issues" below.

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
```

`train_step` accepts two input shapes:

* `(x, y)` — labeled-only path (default).
* `((x_lab, x_unlab), y_lab)` — semi-supervised path. Active when
  `enable_semi_supervised=True`. Adds a stop-gradient pseudo-label
  L1-consistency term over `x_unlab` against the weight-shared frozen
  teacher, plus a Feature-Alignment Loss term on unlabeled features when
  `use_feature_alignment=True` as well.

  This used to read "active only when `enable_semi_supervised=True` **AND**
  `use_feature_alignment=True`", and that was an accurate description of a
  bug rather than of a design: both terms sat under the FAL flag while
  `train_step` routed on `enable_semi_supervised` alone, so
  `enable_semi_supervised=True, use_feature_alignment=False` unpacked the
  unlabeled batch and used it for nothing. Fixed in
  `plan-2026-08-17T183311-79c63e38` (D-033).

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
import keras
from dl_techniques.models.depth_anything import (
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

The 14-item review from `plan_2026-05-10_44694bc9` plus D-005 has been
folded into the work below. Item numbers are kept stable for traceability.

**FIXED in `plan_2026-05-10_bd098beb`**:

* **#1** — Encoder is now a real `ViT` backbone (`encoder_kind='real'`,
  default). Placeholder Conv-BN-ReLU preserved behind `'placeholder'`.
* **#2** — Frozen teacher is now weight-shared at build via
  `keras.models.clone_model(student) + set_weights(student.get_weights())`.
  EMA advance via `update_teacher_ema(decay=...)`.
* **#3** — Semi-supervised infrastructure wired: `enable_semi_supervised`
  flag, `train_step` accepts `((x_lab, x_unlab), y_lab)`,
  `FeatureAlignmentLoss` term on unlabeled features.
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
* **#13** — `frozen_encoder` weight-share issue subsumed by #2.
* **#14** — `compile()` override no longer stashes dead `self.depth_loss` /
  `self.feature_loss`.
* **D-005** — `StrongAugmentation` uses `keras.random.uniform/shuffle`
  (and a graph-mode cutmix gate).

**FIXED in `plan_2026-05-10_54e6e303`** (this plan):

* **#2-deeper** — On-step EMA decay schedule + integration. New module
  `dl_techniques/models/depth_anything/teacher_ema.py` provides
  `cosine_ema_schedule(start, end, total_steps)`,
  `linear_ema_schedule(...)`, and `TeacherEMACallback(schedule, warmup_steps)`.
  `DepthAnything.from_pretrained_encoder(weights_path)` loads encoder
  weights from a `.keras` checkpoint and re-syncs the teacher.
* **#3-deeper** — Pseudo-label depth on the unlabeled stream
  (`DepthAnything._pseudo_label_depth`, stop-gradient L1 consistency in
  the semi-sup path) + dataset-side pairing utilities
  `train.common.megadepth.UnlabeledImageDataset` and
  `pair_labeled_unlabeled` yielding `((x_lab, x_unlab), y_lab)` via
  `tf.data.Dataset.from_generator` + zip.
* **#5** — `train_step` refactored into a clean labeled-only path
  (`_train_step_labeled`: forward + `compute_loss` + apply grads) and a
  clearly delimited semi-supervised branch
  (`_train_step_semi_supervised`: labeled + FAL + pseudo-label
  consistency). The D-003 anchor is preserved at the labeled
  `compute_loss` call.
* **#9** — `_apply_cutmix`'s mask keeps its trailing singleton axis and
  broadcasts over any channel count (1/3/4, and a target's); brightness/contrast
  factors are per-sample with shape `(B, 1, 1, 1)` so each image gets distinct
  jitter.
* **C-2 / C-19** — `_apply_cutmix` returns the paste box and the donor
  permutation, `train_step` mixes image and target together, the
  feature-alignment term no longer bypasses `self.augmentation`, and the
  colour-jitter clip follows `input_value_range`. RED-proven in
  `tests/test_layers/test_strong_augmentation.py::TestTargetAwareCutMix` /
  `::TestDeclaredInputValueRange` and
  `tests/test_models/test_depth_anything/test_depth_anything.py::TestTrainPathAugmentation`.
* **Multi-epoch FAL stability test** — added at
  `tests/test_models/test_depth_anything/test_depth_anything.py`
  (`TestMultiEpochFALStability`): 3 epochs * 2 steps semi-sup synthetic;
  asserts losses finite, last <= 1.5 * first, and teacher weights moved.

**FIXED in `plan-2026-08-17T183311-79c63e38`** (D-033):

* **The consistency term was gated on the wrong flag.** Both the FAL term
  and the pseudo-label consistency term sat under `use_feature_alignment`,
  while `train_step` routes to `_train_step_semi_supervised` on
  `enable_semi_supervised` alone. `enable_semi_supervised=True,
  use_feature_alignment=False` — two independent CLI flags in
  `src/train/depth_anything/` — therefore unpacked `x_unlab`, used it for
  nothing, and silently trained labeled-only at half the throughput, with
  `update_teacher_ema` a no-op too. The teacher is now built under either
  flag, consistency is gated on the teacher's existence, and
  `use_feature_alignment` governs only the FAL term. The trainer's EMA
  callback condition was widened to match.
* **`self._loss_tracker` was never fed.** Keras 3 updates it inside the
  DEFAULT `train_step`, not inside `compute_loss`, and both step methods
  explicitly skipped `"loss"` in their metrics loop. MEASURED:
  `history.history["loss"] == [0.0, 0.0]` across two epochs, so every
  `ModelCheckpoint`/`EarlyStopping`/`ReduceLROnPlateau` monitoring `"loss"`
  was dead. `test_step` is not overridden, which is why `val_loss` was real
  and the shipped trainer (which monitors `val_loss`) never exposed it.
  Both paths now route through `_finalize_train_step`.
* **The returned logs dict nested under `"compile_metrics"`.** Now flat.
  Scope correction: this was *never* visible in `history.history`, because
  Keras' `pythonify_logs` recursively splices nested dicts into the parent
  and discards the outer key. It was visible only to a direct
  `model.train_step(batch)` caller.

Guards: `tests/test_models/test_depth_anything/test_train_step.py`
(8 tests, 6 RED-proven at `b0914e836`).

**STILL OPEN**:

* **#5 (base note)** — Custom `train_step` uses `tf.GradientTape` rather
  than default Keras `train_step`. This is now a *sanctioned* exception
  rather than an open issue: the semi-supervised path is dual-batch with
  asymmetric augmentation and reads a teacher outside the loss graph, none
  of which `compute_loss(x, y, y_pred, sample_weight, training)` can carry.
  The rationale is written into `model.py`'s module docstring so it ships
  with the code. Do not cite this file as precedent for an ordinary
  single-batch model. *(LOW.)*

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
- `src/train/CLAUDE.md` § "Pattern 5: Depth Estimation (MegaDepth)" — the pattern
  this trainer implements. (Its original exemplar,
  `src/train/cliffordnet/train_depth_estimation.py`, was deleted on 2026-08-10;
  `train_depth_anything.py` is now the only surviving Pattern-5 trainer.)
- `src/dl_techniques/models/depth_anything/components.py` — `DPTDecoder` source.
- `src/dl_techniques/models/depth_anything/model.py` — `DepthAnything` source.
- `tests/test_models/test_depth_anything/` — pytest coverage.
