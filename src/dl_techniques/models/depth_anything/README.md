# Depth Anything

Keras 3 reference implementation of the *Depth Anything* monocular depth estimation
architecture (`encoder + DPT-style decoder`). Source files:

```
src/dl_techniques/models/depth_anything/
├── __init__.py        # currently empty
├── components.py      # DPTDecoder layer
└── model.py           # DepthAnything keras.Model + create_depth_anything factory
```

> **READ THIS BEFORE USING.** This module has known gaps between its name/docstring
> and what it actually does. Most notably: **the encoder is a randomly-initialized
> Conv-BN-ReLU stack**, *not* DINOv2; the semi-supervised pipeline implied by the
> module docstring is **not implemented**. See "Known Issues" at the bottom.

---

## Overview

`DepthAnything` is a `keras.Model` subclass that takes an RGB image batch and
produces a single-channel depth map at the same spatial resolution:

```
input  : (B, H, W, 3)   float32 RGB
output : (B, H, W, 1)   float32 depth
```

It is composed of three sub-networks:

| Sub-network        | Type                          | Where built       | Trainable |
|--------------------|-------------------------------|-------------------|-----------|
| `encoder`          | `keras.Model` (Functional)    | `build()`         | yes       |
| `decoder`          | `DPTDecoder`                  | `build()`         | yes       |
| `frozen_encoder`   | `keras.Model` (Functional)    | `build()` if `use_feature_alignment=True` | no |
| `augmentation`     | `StrongAugmentation` layer    | `build()`         | n/a       |

Forward path (`call`):

```
x → [augmentation if training] → encoder → decoder → depth
```

If `inputs` is a `tuple(x_labeled, x_unlabeled)`, only the labeled half is used
in the forward pass (see Known Issue #3 — semi-supervised pipeline unimplemented).

---

## Components

### `DepthAnything` (in `model.py`)

A `@keras.saving.register_keras_serializable()`-decorated `keras.Model` with full
`get_config()` / `from_config()` round-trip support.

**Constructor signature**:

```python
DepthAnything(
    encoder_type:           str = 'vit_l',                     # one of {'vit_s','vit_b','vit_l'}
    input_shape:            Tuple[int,int,int] = (384, 384, 3),
    decoder_dims:           Optional[List[int]] = [256, 128, 64, 32],
    output_channels:        int  = 1,
    kernel_initializer:     Union[str, Initializer] = 'he_normal',
    kernel_regularizer:     Optional[Regularizer]   = None,
    loss_weights:           Optional[Dict[str, float]] = {'labeled': 1.0,
                                                          'unlabeled': 0.5,
                                                          'feature': 0.1},
    cutmix_prob:            float = 0.5,
    color_jitter_strength:  float = 0.2,
    use_feature_alignment:  bool  = True,
    **kwargs,
)
```

> **Note on `input_shape`**. The constructor argument shadows
> `keras.layers.Layer.input_shape`; the model stores it internally as
> `self.input_shape_param`. `get_config()` exports it back under the key
> `"input_shape"` so `from_config(**config)` round-trips cleanly.

### `create_depth_anything(...)` (in `model.py`)

Convenience factory that returns a *built* `DepthAnything`:

```python
from dl_techniques.models.depth_anything.model import create_depth_anything

model = create_depth_anything(
    encoder_type='vit_l',
    input_shape=(384, 384, 3),
    decoder_dims=[256, 128, 64, 32],
    use_feature_alignment=False,        # see Known Issue #2
)
```

The factory ends with a dummy forward pass (`model(keras.random.normal([1] + list(input_shape)))`)
so the model is fully built when returned.

### `DPTDecoder` (in `components.py`)

A `@keras.saving.register_keras_serializable()`-decorated `keras.layers.Layer`.
Implements a **simple convolutional decoder head**, *not* a full multi-scale DPT
(Dense Prediction Transformer) decoder.

**Architecture**:

```
features (B, h, w, C_in)
   │
   ├── for dim in dims[:-1]:
   │       Conv3x3(dim) → BN → ReLU
   │
   ├── Conv3x3(output_channels) with output_activation        # default 'sigmoid'
   │
   └── depth (B, h, w, output_channels)
```

**Important**: `output_activation` defaults to `'sigmoid'`, which clamps the
output to `[0, 1]`. This is **incompatible with `AffineInvariantLoss`**, which
expects unbounded scale (see Known Issue #6).

`DPTDecoder` provides full Keras 3 idioms: `build()`, `call()`, `get_config()`,
`get_build_config()`, `build_from_config()`, `compute_output_shape()`.

### `StrongAugmentation` (external — `dl_techniques.layers.strong_augmentation`)

A separate layer that combines CutMix + color jitter. Used by `DepthAnything`
during training-mode forward passes. Its `__init__.py` accepts `cutmix_prob`
and `color_jitter_strength`. See Known Issue #9 below for caveats.

---

## Usage

### Forward pass / inference

```python
import keras
from dl_techniques.models.depth_anything.model import create_depth_anything

model = create_depth_anything(
    encoder_type='vit_l',
    input_shape=(384, 384, 3),
    use_feature_alignment=False,
)
x = keras.random.normal([2, 384, 384, 3])
depth = model(x, training=False)
print(depth.shape)  # (2, 384, 384, 1)
```

### Compile / train

> **The model's `train_step` was rewritten in this plan to use the canonical
> Keras-3 pattern** (`self.compute_loss(...)` + iterate `self.metrics`). See
> Known Issue #4 — fixed in this plan.

```python
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=5e-6, weight_decay=1e-5),
    loss=keras.losses.MeanSquaredError(),    # see Known Issue #6 if you want AffineInvariantLoss
)
# model.fit(x_train, y_train, epochs=...)
```

For a complete training script wired to MegaDepth + depth metrics + visualization
callbacks, see `src/train/depth_anything/`.

### Serialization

```python
model.save('depth_anything.keras')
loaded = keras.models.load_model('depth_anything.keras')
```

`get_config()` exports every constructor argument; `from_config(cls, config)`
calls `cls(**config)` so any extra base-class keys (`name`, `trainable`,
`dtype`) are absorbed by `keras.Model.__init__(**kwargs)`.

---

## Configuration

| Argument                | Type                       | Default                          | Notes |
|-------------------------|----------------------------|----------------------------------|-------|
| `encoder_type`          | `str`                      | `'vit_l'`                        | Validated against `{'vit_s','vit_b','vit_l'}` but the placeholder encoder is identical for all three (see Known Issue #10). |
| `input_shape`           | `Tuple[int,int,int]`       | `(384, 384, 3)`                  | Shadows `Layer.input_shape`; stored as `self.input_shape_param`. |
| `decoder_dims`          | `List[int]`                | `[256, 128, 64, 32]`             | First entry sets the encoder feature-projection dim; last entry is the penultimate decoder width. |
| `output_channels`       | `int`                      | `1`                              | Final depth-map channel count. |
| `kernel_initializer`    | `str` \| `Initializer`     | `'he_normal'`                    | Accepts string names or initializer instances. |
| `kernel_regularizer`    | `Regularizer` \| `None`    | `None`                           | Do **not** combine with `AdamW(weight_decay=...)` (double weight decay). |
| `loss_weights`          | `Dict[str,float]`          | `{labeled:1.0,unlabeled:0.5,feature:0.1}` | Currently dead state — `train_step` does not consume these (Known Issue #14). |
| `cutmix_prob`           | `float`                    | `0.5`                            | Forwarded to `StrongAugmentation`. |
| `color_jitter_strength` | `float`                    | `0.2`                            | Forwarded to `StrongAugmentation`. |
| `use_feature_alignment` | `bool`                     | `True`                           | Builds `frozen_encoder` (which has random weights — Known Issue #2). Set `False` for honest baselines. |

`DPTDecoder` constructor (used internally by `DepthAnything.build()`):

| Argument             | Default       | Notes |
|----------------------|---------------|-------|
| `dims`               | required      | Forwarded from `DepthAnything.decoder_dims`. |
| `output_channels`    | `1`           | Forwarded. |
| `output_activation`  | `'sigmoid'`   | **Not exposed via `DepthAnything`'s constructor.** Override path requires editing the model. |
| `kernel_initializer` | `'he_normal'` | Forwarded. |
| `kernel_regularizer` | `None`        | Forwarded. |

---

## Example — full pipeline

```python
import keras, numpy as np
from dl_techniques.models.depth_anything.model import create_depth_anything

model = create_depth_anything(
    encoder_type='vit_l',
    input_shape=(384, 384, 3),
    decoder_dims=[256, 128, 64, 32],
    use_feature_alignment=False,
)
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=5e-6, weight_decay=1e-5),
    loss=keras.losses.MeanSquaredError(),
)

x = np.random.randn(4, 384, 384, 3).astype('float32')
y = np.random.rand (4, 384, 384, 1).astype('float32')   # in [0,1] because output is sigmoid
hist = model.fit(x, y, epochs=1, batch_size=2, verbose=0)
```

---

## Known Issues / Caveats

The following gaps were surfaced by a code review during plan
`plan_2026-05-10_44694bc9`. They are listed honestly; the README is **not** a
patch — it is a contract with the reader. Items #4 was fixed in this plan;
items #1, #2, #3, #5–#14 are *not* fixed and remain open work.

1. **Placeholder encoder, not DINOv2.** `_create_encoder()` builds a small
   Conv-BN-ReLU stack with a `feature_projection` 1×1 head. The module
   docstring's claim "Inherits semantic priors from pre-trained encoders" is
   **unfulfilled**. Without a real DINOv2 (or other pretrained vision encoder)
   the model has no semantic prior and cannot match published Depth Anything
   numbers. Pass `--init-from <pretrained.keras>` in the train script to
   transfer at least *some* useful initialization. *(HIGH)*
2. **Frozen "teacher" encoder shares no weights with student.** When
   `use_feature_alignment=True`, `self.frozen_encoder = self._create_encoder(trainable=False)`
   builds an *independent* Functional model with **freshly randomly-initialized
   weights**. `FeatureAlignmentLoss` against random features is meaningless.
   Use `use_feature_alignment=False` until this is rewired (e.g., after
   pretraining + EMA copy of the student into the teacher). *(HIGH)*
3. **Semi-supervised pipeline implied by docstring is NOT implemented.**
   `call()` accepts `(x_labeled, x_unlabeled)` but processes only `x_labeled`.
   `train_step()` does not use the unlabeled half, the `feature` loss term, the
   `frozen_encoder`, or any consistency loss. The
   `loss_weights={'labeled','unlabeled','feature'}` dict is dead state.
   The "62M unlabeled images" claim in the module docstring is aspirational. *(HIGH)*
4. **[FIXED in this plan]** *(was: `self.compiled_loss` / `self.compiled_metrics`
   are removed/deprecated in Keras 3.)* The `train_step` previously called
   `self.compiled_loss(y, y_pred)` and `self.compiled_metrics.update_state(...)`,
   both of which raise `AttributeError` on Keras 3.8+. **Status: fixed.**
   `train_step` now uses `self.compute_loss(x=x, y=y, y_pred=y_pred)` and
   iterates `self.metrics` for user-defined metric updates, mirroring
   `dl_techniques/models/masked_language_model/mlm.py`. The semi-supervised
   pipeline (#3) and feature-alignment loss (#2) are still **not** wired in
   `train_step` — only the deprecated-API crash is fixed. *(HIGH — was blocking
   any training; now cleared.)*
5. **`tf.GradientTape` instead of `keras.ops` / default `train_step`.** The
   custom `train_step` couples the model to the TensorFlow backend even though
   the rest of the library is backend-agnostic via `keras.ops`. Minor
   convention violation; consider letting Keras provide the default `train_step`
   once the semi-supervised pipeline is properly designed. *(LOW)*
6. **`DPTDecoder.output_activation='sigmoid'` is incompatible with
   `AffineInvariantLoss`.** Sigmoid clamps prediction to `[0,1]`, making global
   scale ill-defined; AIL specifically expects unbounded scale to median-shift /
   MAD-normalize meaningfully. The standard depth output is `linear` (or
   `softplus`/`exp`). The training script in `src/train/depth_anything/` works
   around this by using a masked L1 + gradient-matching loss (compatible with
   any output range) rather than AIL. To use AIL, the decoder must be patched
   to remove sigmoid. *(HIGH for training quality)*
7. **Functional encoder built inside `build()` is fragile under serialization.**
   `dl_techniques` convention is to declare sub-layers in `__init__`. The
   current pattern (Functional model constructed in `build()`) has not been
   tested under save/load round-trip. *(MEDIUM)*
8. **Dead defensive checks.** `if DPTDecoder is not None`,
   `if StrongAugmentation is not None`, `if AffineInvariantLoss is not None`,
   `if FeatureAlignmentLoss is not None` — all are class objects imported at
   the top of the file and can never be `None`. `_create_fallback_decoder()` is
   dead code. *(LOW)*
9. **`StrongAugmentation._apply_cutmix` hard-codes 3 channels** (`ops.tile(mask, [1, 1, 3])`)
   and applies *batch-scalar* (single value applied to whole batch) brightness
   and contrast factors rather than per-sample. Much weaker than typical CutMix.
   *(LOW; pre-existing layer in `dl_techniques.layers.strong_augmentation`.)*
10. **`encoder_type` validated but unused.** `'vit_s'`, `'vit_b'`, `'vit_l'` all
    produce the same placeholder Conv-BN-ReLU encoder. Misleading API. *(MEDIUM)*
11. **`input_shape` parameter shadows `Layer.input_shape`.** Stored as
    `self.input_shape_param`. `get_config()` exports it as `"input_shape"` so
    `from_config(**config)` round-trips, but the naming is error-prone for
    contributors used to standard Keras attributes. *(LOW)*
12. **No tests.** There is no `tests/test_models/test_depth_anything/` — the
    model has zero pytest coverage. Save/load, gradient flow, and shape
    invariants are unverified. *(MEDIUM)*
13. **`frozen_encoder.trainable = trainable` set after Functional construction.**
    Works in isolation, but combined with #2 above the frozen "teacher" never
    becomes a real teacher. *(LOW)*
14. **`compile()` override silently mutates `self.loss_weights`** and stores
    `self.depth_loss` / `self.feature_loss` that are then never read by
    `train_step`. The override is partially dead state — only `super().compile(...)`
    is doing useful work. *(LOW)*

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

---

## See also

- `src/train/depth_anything/` — Pattern-5 training scaffold for this model
  (MegaDepth + masked depth loss + visualization callbacks).
- `src/train/cliffordnet/train_depth_estimation.py` — reference Pattern-5
  trainer that the depth_anything trainer mirrors.
- `src/dl_techniques/models/depth_anything/components.py` — `DPTDecoder` source.
- `src/dl_techniques/models/depth_anything/model.py` — `DepthAnything` source.
