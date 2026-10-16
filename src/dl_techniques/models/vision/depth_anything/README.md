# Depth Anything

Keras 3 reference implementation of the *Depth Anything* monocular depth
estimation architecture (`encoder + DPT-style decoder`). Source files:

```
src/dl_techniques/models/vision/depth_anything/
├── __init__.py        # public API: DepthAnything, create_depth_anything, DPTDecoder
├── components.py      # DPTDecoder layer (exported; used by OmniPoint, not by DepthAnything)
├── model.py           # DepthAnything keras.Model + create_depth_anything factory
└── teacher_ema.py     # TeacherEMACallback, cosine_ema_schedule, linear_ema_schedule
```

The encoder is the in-tree `dl_techniques.models.vision.vit.ViT` (`encoder_kind='real'`,
the default); the decoder is the shared `DepthEstimationHead(decoder_style='dpt')` and ends in a
ReLU, as the reference heads do (see "Output range" below). `train_step` dispatches to a
clean labeled-only path or a semi-supervised path that adds FAL on pooled features **and**
a pseudo-label consistency term on student-vs-teacher depth over `x_unlab` (the paper's
affine-invariant loss per CutMix region, eq. 6-8). On-step EMA
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
output : (B, H, W, 1)   float32 relative depth, non-negative (final ReLU)
```

### Output range

The output is the ReLU of the decoder's last conv, so it is non-negative with no upper bound and no
range parameters. This is what the reference does, re-derived from the paper and the official code
(fetched 2026-10-15):

* The Depth Anything paper (arXiv 2401.10891) trains relative depth: depth is turned into disparity
  `d = 1/t`, normalized per map, and supervised with an affine-invariant loss. There is no metric
  range in the model.
* V1 `depth_anything/dpt.py` (LiheYoung/Depth-Anything): the head's `output_conv2` ends
  `Conv2d(32, 1, 1) -> nn.ReLU(True) -> nn.Identity()`, and `DPT_DINOv2.forward` applies `F.relu`
  again after the bilinear resize.
* V2 `depth_anything_v2/dpt.py` (DepthAnything/Depth-Anything-V2): the same ReLU tail and `F.relu`
  after the resize.
* The only range maps are in the metric variants, which this package does not implement: V2's
  `metric_depth/depth_anything_v2/dpt.py` ends in `nn.Sigmoid()` and multiplies by `max_depth`
  (20 indoor, 80 outdoor), and V1's metric models are ZoeDepth with metric bins.

The decoder used to be linear and received `min_depth=0.1`, `max_depth=100.0` and
`use_log_depth=True`, which the DPT decoder stored and never applied; the decoder now refuses them.
A `.keras` archive saved before 2026-10-15 still loads, with every weight restored, and its output
is then the ReLU of what it returned before (measured: max difference 0.0 against the old output
clamped at zero). Nothing marks such an archive, so the change is silent.

The trainer (`src/train/depth_anything/`) supervises the paper's target and loss: per-map normalized
MegaDepth disparity in `[0, 1]` and the masked affine-invariant MAE (`AffineInvariantLoss(masked=True)`).
The decoder is a plain conv stack rather than the reference's fusion blocks.

It is composed of three sub-networks:

| Sub-network        | Type                          | Where built       | Trainable |
|--------------------|-------------------------------|-------------------|-----------|
| `encoder`          | `keras.Model` (`ViT` or Conv) | `__init__`/`build()` | yes |
| `decoder`          | `DepthEstimationHead` (`decoder_style='dpt'`), named `depth_estimation_head` | `build()`, only when `include_top=True` | yes |
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
  consistency term over `x_unlab` against the weight-shared frozen
  teacher (affine-invariant, per CutMix region, eq. 6-8 of the paper), plus a Feature-Alignment Loss term on unlabeled features when
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
    include_top:            bool  = True,
    **kwargs,
)
```

`input_shape`, a deprecated alias for `image_shape`, was removed on 2026-10-15 with no replacement:
pass `image_shape`. A saved config that still carries `input_shape` no longer loads.

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
upsampling. `DepthAnything` does not use it (its decoder is the shared
`DepthEstimationHead` in `'dpt'` mode, the same stage structure); OmniPoint's dense heads do.

**Architecture**:

```
features (B, h, w, C_in)
   │
   ├── for i, dim in enumerate(dims):
   │       Conv3x3(dim) → BN → activation → [UpSample2D(2x bilinear) if i < log2(upsample_factor)]
   │
   ├── Conv3x3(output_channels) with output_activation        # default 'linear'
   │
   └── output (B, h*upsample_factor, w*upsample_factor, output_channels)
```

`output_activation` defaults to `'linear'`. `upsample_factor` defaults to `1` and must be a power of
two at most `2 ** len(dims)`.

### `StrongAugmentation` (in `dl_techniques.layers.signal_processing.strong_augmentation`)

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
* Colour jitter is dtype-preserving: the per-sample factors are cast to the
  batch dtype, so a float16 batch stays float16 on the training path.
* `cutmix_ratio_range` is the fraction of each image *side* that is cut, so the
  pasted area is the ratio squared.
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
**FAL** between pooled student/teacher features, and **pseudo-label
consistency** between the student's depth and the teacher's stop-gradient depth
pseudo-labels. The consistency term is the paper's (arXiv 2401.10891, eq. 6-8): the
affine-invariant loss of eq. 1-3 (median shift, mean-absolute-deviation scale, MAE)
computed separately over the pixels outside the CutMix box and inside it, averaged by
their pixel shares (`DepthAnything._unlabeled_consistency`); without a mix it is that
loss over the whole map. It was a plain L1 until plan-2026-10-14T062616-81938e81 D-113. In both terms the student sees the *augmented* unlabeled batch
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
| `image_shape`            | `Tuple[int,int,int]`       | `(384, 384, 3)`                        | Input image shape. |
| `decoder_dims`           | `List[int]`                | `[256, 128, 64, 32]`                   | Stage widths; last entry is the penultimate decoder width. |
| `output_channels`        | `int`                      | `1`                                    | Final depth-map channel count. |
| `kernel_initializer`     | `str` \| `Initializer`     | `'he_normal'`                          | Reaches the encoder and every decoder conv (one clone per conv). |
| `kernel_regularizer`     | `Regularizer` \| `None`    | `None`                                 | Reaches the encoder and every decoder conv. Do **not** combine with `AdamW(weight_decay=...)`. |
| `loss_weights`           | `Dict[str,float]`          | `{labeled:1.0, unlabeled:0.5, feature:0.1}` | Consumed by the semi-sup `train_step` path. |
| `cutmix_prob`            | `float`                    | `0.5`                                  | Forwarded to `StrongAugmentation`. |
| `color_jitter_strength`  | `float`                    | `0.2`                                  | Forwarded to `StrongAugmentation`. |
| `input_value_range`      | `Tuple[float,float]` \| `None` | `(0.0, 1.0)`                       | Declared input range for the colour-jitter clip. `None` for standardized / `[-1,+1]` inputs. |
| `use_feature_alignment`  | `bool`                     | `True`                                 | Adds the FAL term in the semi-supervised step. Also builds `frozen_encoder` — but so does `enable_semi_supervised`, which needs a teacher for its consistency term. Setting this True with `enable_semi_supervised=False` builds a teacher nothing reads, and warns. |
| `enable_semi_supervised` | `bool`                     | `False`                                | Switches `train_step` to `((x_lab, x_unlab), y_lab)` mode. |

The decoder, `DepthEstimationHead(decoder_style='dpt')`, built in `build()` when `include_top=True`:

| Argument                 | Value DepthAnything passes | Notes |
|--------------------------|----------------------------|-------|
| `dpt_dims`               | `decoder_dims`             | One `Conv3x3 → BN → ReLU` stage per entry. |
| `output_channels`        | `output_channels`          | |
| `dpt_use_bias`           | `False`                    | Stage convs; the output conv has a bias. |
| `dpt_activation`         | `'relu'`                   | Stage activation. |
| `dpt_output_activation`  | `'relu'`                   | The reference's final ReLU; see "Output range". |
| `dpt_upsample_factor`    | `encoder_stride` (16)      | Same for the real and the placeholder encoder. Power of 2 only. |
| `dpt_kernel_initializer` | `kernel_initializer`       | Cloned per conv. |
| `dpt_kernel_regularizer` | `kernel_regularizer`       | |

`min_depth`, `max_depth` and `use_log_depth` are not passed: they configure the head's progressive
decoder, and the DPT mode refuses them.

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
  `src/dl_techniques/models/common/masked_language_model/mlm.py`.

---

## See also

- `src/train/depth_anything/` — Pattern-5 training scaffold for this model
  (MegaDepth + masked depth loss + visualization callbacks).
- `src/train/AGENTS.md` § "Pattern 5: Depth Estimation (MegaDepth)" — the pattern
  this trainer implements. `train_depth_anything.py` is the only Pattern-5 trainer.
- `src/dl_techniques/models/vision/depth_anything/components.py` — `DPTDecoder` source.
- `src/dl_techniques/models/vision/depth_anything/model.py` — `DepthAnything` source.
- `tests/test_models/test_depth_anything/` — pytest coverage.
