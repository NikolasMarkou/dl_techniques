# DINO — self-distillation vision transformers (v1 / v2 / v3)

`src/dl_techniques/models/dino/` implements three Vision Transformer backbones from the
DINO paper line, plus the DINO projection head. The matching losses live one package
over, in `src/dl_techniques/losses/dino_loss.py`.

DINO ("**DI**stillation with **NO** labels") trains an image encoder with no labels by
making a **student** network agree with a **teacher** network that is an exponential
moving average of the student itself. Different augmented crops of one image go to the
two networks; the student is asked to predict the teacher's output distribution. Nothing
stops both networks collapsing to a constant, so DINO adds two stabilizers: the teacher's
outputs are **centered** by a running mean and **sharpened** by a low temperature. Those
two together are what make an otherwise-degenerate objective learn something.

## Contents

1. [What is here](#1-what-is-here)
2. [Honest capability table (v1 / v2 / v3)](#2-honest-capability-table-v1--v2--v3)
3. [Factory signatures](#3-factory-signatures)
4. [Runnable examples](#4-runnable-examples)
5. [The loss family, the training model, and the rules that are not optional](#5-the-loss-family-the-training-model-and-the-rules-that-are-not-optional)
6. [Why there is no shared base class](#6-why-there-is-no-shared-base-class)
7. [Named backlog — what is deliberately NOT implemented](#7-named-backlog--what-is-deliberately-not-implemented)
8. [Tests](#8-tests)
9. [References](#9-references)

---

## 1. What is here

| File | Contents |
|---|---|
| `src/dl_techniques/models/dino/__init__.py` | The package's public API: an explicit `__all__` re-exporting every class and factory below. |
| `src/dl_techniques/models/dino/dino_v1.py` | `DINOHead` (the projection head all three versions use for SSL), `DINOv1`, `create_dino_v1`, `create_dino_teacher_student_pair`. |
| `src/dl_techniques/models/dino/dino_v2.py` | `DINOv2Block`, `DINOv2VisionTransformer` (the backbone), `DINOv2` (backbone + classifier), `create_dino_v2`. |
| `src/dl_techniques/models/dino/dino_v3.py` | `DINOv3`, `create_dino_v3`. |
| `src/dl_techniques/models/dino/dino_training.py` | `DINOTrainingModel`, `create_dino_training_model` — student + frozen EMA teacher over a multi-crop batch, trainable under stock `fit()`. See § 5.5. |
| `src/dl_techniques/models/dino/common.py` | `reject_input_shape` — the one piece of the converged factory scheme that is identical in all three files. Not part of the public API. |
| `src/dl_techniques/models/dino/README.md` | This file. |

Related, outside this package:

| Path | Relationship |
|---|---|
| `src/dl_techniques/losses/dino_loss.py` | `DINOLoss`, `iBOTPatchLoss`, `KoLeoLoss`. Re-exported from `dl_techniques.losses`. See § 5. |
| `src/dl_techniques/datasets/vision/multi_crop.py` | `make_multi_crop_map_fn` — the `tf.data` transform producing the multi-crop element `DINOTrainingModel` consumes. See § 5.6. |
| `src/dl_techniques/models/depth_anything/teacher_ema.py` | `TeacherEMACallback` plus `cosine_ema_schedule` / `linear_ema_schedule`. Model-agnostic: it drives the teacher EMA of any model exposing `update_teacher_ema(decay)`. |
| `src/dl_techniques/models/depth_anything/model.py` | Uses a **placeholder** encoder described in prose as "a placeholder for DINOv2". It does **not** import from this package. Wiring it to a real `DINOv2` is a named backlog item (§ 7). |
| `tests/test_models/test_dino/` | This package's tests. See § 8. |

**Variant tables are per-class.** `DINOv1.MODEL_VARIANTS`,
`DINOv2VisionTransformer.MODEL_VARIANTS` and `DINOv3.MODEL_VARIANTS` are three separate
class attributes that share a name and disagree on their contents (v2's `giant` carries
`ffn_type='swiglu'`; v3's carries `patch_size=(14, 14)` and `stochastic_depth_rate=0.4`;
v1's carries neither). The package binds **no** module-level `MODEL_VARIANTS` alias,
because a single name would have to pick one of the three and misdescribe the other two.
Reach them through the classes: `from dl_techniques.models.dino import DINOv3;
DINOv3.MODEL_VARIANTS`.

All five variant key sets agree — `tiny`, `small`, `base`, `large`, `giant` — and the
four shared architecture fields (`embed_dim`, `depth`, `num_heads`, `mlp_ratio`) agree
value-for-value across the three versions. `giant` is **not** a DINOv1-paper variant
(Caron et al. 2021 stop at ViT-B/8); it exists in `dino_v1.py` for key-set parity only.

---

## 2. Honest capability table (v1 / v2 / v3)

Rows marked **NOT IMPLEMENTED** are the point of this table. Each is a mechanism the
corresponding paper has and this code does not.

| Mechanism | `DINOv1` | `DINOv2` | `DINOv3` |
|---|---|---|---|
| Pre-norm ViT trunk, `[CLS]` token, patch embedding | yes | yes | yes |
| Learned absolute positional embedding | yes | yes | yes (default) |
| `DINOHead` projection head | yes (`include_projection_head=True`) | not built in — compose `DINOHead` yourself | not built in — compose `DINOHead` yourself |
| `norm_last_layer` (unit-norm output prototypes) | **yes**, on `DINOHead` — see note below | via `DINOHead` | via `DINOHead` |
| Teacher/student pair factory | `create_dino_teacher_student_pair` | none | none |
| iBOT-style patch masking on the forward path | no | **yes** — the backbone takes `[images, masks]` and applies a learnable mask token (`MaskTokenApply`) | no |
| The iBOT **loss** half | — | **not wired here.** `iBOTPatchLoss` exists in `src/dl_techniques/losses/dino_loss.py`; connecting it to the masked forward pass is the trainer's job, not the model's | — |
| Register tokens (Darcet et al. 2023) | no | **yes** — `num_register_tokens`, defaulting to 4 on `large`/`giant` | **NOT IMPLEMENTED** |
| LayerScale (`LearnableMultiplier`) | no | yes — `init_values` | no |
| SwiGLU FFN | opt-in via `ffn_type` | **yes**, and it is the `giant` variant's default | opt-in via `ffn_type` |
| Stochastic depth with linear decay | yes | yes | yes |
| RoPE positional embedding | no | no | **yes — but 1-D, not the paper's 2-D axial.** See note below |
| Gram anchoring | — | — | **NOT IMPLEMENTED** (needs a third frozen "Gram teacher" network) |
| Sinkhorn-Knopp centering | **NOT IMPLEMENTED** (EMA centering only) | **NOT IMPLEMENTED** | **NOT IMPLEMENTED** |
| High-res adaptation / distillation from a large pretrained teacher | — | — | **NOT IMPLEMENTED** |
| `get_last_selfattention()` (attention-map extraction) | **raises `NotImplementedError`** | not defined | **raises `NotImplementedError`** |
| Pretrained weights | none shipped | `pretrained=True` logs a warning and is ignored | `pretrained=True` logs a warning and is ignored |
| `.keras` round-trip | tested | tested | tested |

### `norm_last_layer` on `DINOHead` — what is and is not reproduced

The reference DINO wraps the final projection in PyTorch's weight-norm
reparameterization `w = g * v / ||v||` and, when `norm_last_layer=True`, pins `g = 1`
and freezes it, so every output prototype keeps unit L2 norm throughout training.

Here the **invariant** is reproduced — `keras.constraints.UnitNorm(axis=0)` on the final
`Dense` kernel, **plus** a one-off application at `build()` time — but not the
**optimization path**. A Keras constraint projects *after* each optimizer step where the
reference reparameterizes *before* it. The build-time application is not redundant: Keras
applies a constraint inside `optimizer.apply`, so a freshly built or inference-only head
would otherwise violate the invariant the flag promises. `norm_last_layer=False` leaves
the kernel unconstrained.

### RoPE on `DINOv3` — real, live, and 1-D

`positional_embedding_type='rope'` is genuine rotary position embedding applied to Q and
K *inside* the attention operator, routed through the registered `group_query` attention
type with `num_kv_heads == num_heads` (which reduces it to plain multi-head attention
with RoPE). When RoPE is selected, the learned absolute table is **omitted entirely**,
not stacked on top.

Two things this is not:

- **It is 1-D**, over the flattened token sequence (position = token index). DINOv3 uses
  a 2-D axial formulation over patch `(row, column)` coordinates with random coordinate
  jittering. The rotation here is real and live; it is not the paper's.
- **A checkpoint is NOT portable between `learned` and `rope`.** The two modes build
  structurally different attention classes (`MultiHeadAttention` vs
  `GroupedQueryAttention`), so their weight sets do not correspond. Choose the mode
  before you train, not after.

`rope_percentage=0.0` is legal but degenerate: it rotates nothing, leaving the model with
no positional information at all. It is kept legal on purpose — it is the control arm the
RoPE-liveness test needs.

### `get_last_selfattention()` raises — and for two different reasons

Both `DINOv1` and `DINOv3` used to log a warning and return a zeros tensor, which renders
as a blank attention map a caller cannot distinguish from a broken model. Both now raise
`NotImplementedError` naming the exact missing capability, and the two reasons differ:

- **`DINOv1`** (and any `learned`-mode `DINOv3`): nothing on the `multi_head` attention
  path accepts a `return_attention_scores` flag at all.
- **`DINOv3` in `rope` mode**: `GroupedQueryAttention.call` *does* accept
  `return_attention_weights` and *does* return a valid attention map — but
  `TransformerLayer.call` does not forward the flag, so the capability is unreachable one
  frame away. If `TransformerLayer` ever forwards it, revisit this.

---

## 3. Factory signatures

All three factories share one parameter scheme:

```text
create_dino_v1(variant, *, image_size, patch_size, num_classes, include_top, ...)
create_dino_v2(variant, *, image_size, patch_size, num_classes, include_top, ...)
create_dino_v3(variant, *, image_size, patch_size, num_classes, include_top, ...)
```

Everything after `variant` is **keyword-only**. `image_size` and `patch_size` each accept
an `int` or an `(height, width)` tuple on all three.

### There is no `input_shape` argument

It was a redundant second spelling of `image_size` that could **disagree** with it, and a
disagreement built a model whose patch grid did not match its input — silently, with the
failure surfacing much later or not at all. All three factories now raise `TypeError`
naming `image_size` if you pass it. (The `DINOv1` and `DINOv2` *constructors* still accept
`input_shape` as a lower-level escape hatch; the factories do not.)

### The `None`-defers-to-the-variant precedence rule

A factory parameter passed as `None` means **"the caller said nothing"** and defers to the
variant's own `MODEL_VARIANTS` entry, or to the version's default if the entry says
nothing. An **explicit non-`None` value always wins.** This is the only formulation that
lets a caller override a variant, and it applies to:

| Parameter | `None` resolves to |
|---|---|
| `create_dino_v1(patch_size=...)` | `16` (v1's variants define no patch size) |
| `create_dino_v2(patch_size=...)` | `14` (v2's variants define no patch size) |
| `create_dino_v3(patch_size=...)` | **the variant's own**: `(14, 14)` on `giant`, `(16, 16)` otherwise |
| `create_dino_v2(ffn_type=...)` | **the variant's own**: `'swiglu'` on `giant`, `'mlp'` otherwise |
| `create_dino_v2(num_register_tokens=...)` | `4` on `large`/`giant`, `0` otherwise |

Before this rule, `ffn_type='mlp'` was both the default *and* the promotion trigger on
`giant`, so a caller who explicitly wanted MLP on `giant` was silently upgraded to SwiGLU
with no way to opt out. Same for `num_register_tokens=0`.

### Full signatures

```text
create_dino_v1(
    variant: ModelVariant = "small",
    *,
    image_size: int | tuple[int, int] = 224,
    patch_size: int | tuple[int, int] | None = None,
    num_classes: int = 0,
    include_top: bool = True,
    include_projection_head: bool = False,
    dino_out_dim: int = 65536,
    **kwargs,
) -> DINOv1

create_dino_v2(
    variant: Literal['tiny','small','base','large','giant'] = 'base',
    *,
    image_size: int | tuple[int, int] = 224,
    patch_size: int | tuple[int, int] | None = None,
    num_classes: int = 1000,
    include_top: bool = True,
    num_register_tokens: int | None = None,
    init_values: float | None = 1e-5,
    stochastic_depth_rate: float = 0.0,
    ffn_type: str | None = None,
    pretrained: bool = False,
    **kwargs,
) -> DINOv2

create_dino_v3(
    variant: str = "base",
    *,
    image_size: int | tuple[int, int] = (224, 224),
    patch_size: int | tuple[int, int] | None = None,
    num_classes: int = 1000,
    include_top: bool = True,
    positional_embedding_type: Literal['learned', 'rope'] = 'learned',
    rope_theta: float = 10000.0,
    rope_percentage: float = 1.0,
    pretrained: bool = False,
    **kwargs,
) -> DINOv3
```

**The `num_classes` defaults genuinely differ** — `0` on v1, `1000` on v2 and v3 — and are
left that way rather than converged, because changing them would be a behaviour change to
existing callers rather than a surface normalization. Pass `num_classes` explicitly.

Version-specific knobs stay per-version on purpose. Convergence here is about the shared
surface, not about erasing real differences between three different papers.

---

## 4. Runnable examples

Every example below is executed by
`tests/test_models/test_dino/test_dino_package.py` — its imports and the paths named in
this file are checked to resolve, so a rename that breaks them fails a test rather than
rotting quietly.

### DINOv1 — classification, and an SSL projection head

```python
from dl_techniques.models.dino import create_dino_v1, create_dino_teacher_student_pair

# Supervised fine-tuning head: 32x32 input, patch 16 -> a 2x2 patch grid.
model = create_dino_v1("small", image_size=32, patch_size=16, num_classes=10)

# Self-supervised: no classifier, a 65536-way DINO projection head instead.
ssl_model = create_dino_v1(
    "small",
    image_size=32,
    patch_size=16,
    num_classes=0,
    include_top=False,
    include_projection_head=True,
    dino_out_dim=4096,
)

# A teacher/student pair with identical architecture and independent weights.
teacher, student = create_dino_teacher_student_pair(
    "small", image_size=32, patch_size=16, dino_out_dim=4096
)
```

### DINOv2 — masked forward pass, register tokens, SwiGLU

`DINOv2` takes a **two-element input** `[images, masks]`, where `masks` is a boolean
`(batch, num_patches)` iBOT mask. A mask of all-`False` is the ordinary unmasked forward
pass.

```python
import numpy as np
from dl_techniques.models.dino import create_dino_v2

model = create_dino_v2("tiny", image_size=28, patch_size=14, num_classes=10)

images = np.random.rand(2, 28, 28, 3).astype("float32")
masks = np.zeros((2, 4), dtype=bool)          # 28 // 14 == 2 -> 2*2 == 4 patches
logits = model([images, masks], training=False)

# `giant` brings SwiGLU and 4 register tokens from its own variant entry.
giant = create_dino_v2(
    "giant", image_size=32, patch_size=16, num_classes=10,
    embed_dim=32, depth=1, num_heads=4,        # shrunk so the example is cheap
)
```

### DINOv3 — learned vs RoPE positional embeddings

```python
import keras
from dl_techniques.models.dino import DINOHead, create_dino_v3

# Default: learned absolute positional embeddings.
learned = create_dino_v3("small", image_size=32, patch_size=16, num_classes=10)

# RoPE: the learned table is omitted entirely, not stacked on top.
rope = create_dino_v3(
    "small",
    image_size=32,
    patch_size=16,
    num_classes=10,
    positional_embedding_type="rope",
    rope_theta=10000.0,
)

# A feature-extraction backbone plus a DINO projection head.
backbone = create_dino_v3("small", image_size=32, patch_size=16, include_top=False)
head = DINOHead(in_dim=384, out_dim=4096, hidden_dim=512, bottleneck_dim=64)
ssl_model = keras.Model(inputs=backbone.input, outputs=head(backbone.output))
```

---

## 5. The loss family, the training model, and the rules that are not optional

`src/dl_techniques/losses/dino_loss.py` provides three losses, all re-exported from
`dl_techniques.losses`:

```python
from dl_techniques.losses import DINOLoss, iBOTPatchLoss, KoLeoLoss
```

| Loss | Job |
|---|---|
| `DINOLoss` | The cross-view consistency objective on `[CLS]` outputs, with the EMA-centered, temperature-sharpened teacher. Owns the centering state. |
| `iBOTPatchLoss` | The same objective at patch level, over the masked patches only. Also owns a centering state. |
| `KoLeoLoss` | A Kozachenko-Leonenko entropic regularizer that pushes embeddings apart on the unit sphere. An anti-collapse term, not a distillation term. |

### Rule 1 — SSL pretraining must run **without** `validation_data`

`DINOLoss` maintains its centering statistic by `.assign()`ing a `keras.Variable` inside
`call()`. That works correctly under stock `fit()` (measured: it reaches the
hand-computed EMA value to `1.6e-08`, bit-identically under `jit_compile` auto / `False` /
`True`) — but `call()` runs on **every** batch, including validation batches. Every
validation batch therefore performs a full, unwanted centering update.

This is not a rounding error. Measured: a 4-sample validation set at `batch_size=2`
doubled an epoch's update count from 2 to 4 and pushed the center **81% past** its correct
value — silently, with a finite loss and a clean exit.

So: **do not pass `validation_data` to a DINO pretraining `fit()`.** Validation belongs in
a separate callback (a k-NN probe on frozen features), which does not invoke the loss.

### Rule 2 — the center's **value** is serialized through `get_config()`

Keras does not checkpoint loss-owned variables. A `.keras` save/load of a model compiled
with `DINOLoss` succeeds, returns the right registered subclass with a live `.center`
Variable, and silently brings the center back at its `Zeros()` initialization — so a
resumed run restarts centering from scratch.

`DINOLoss.get_config()` and `iBOTPatchLoss.get_config()` therefore carry the center's
value as a nested list, and `from_config` assigns it back. A missing `center` key is
tolerated, so configs written before this change still load. The cost is a config blob
that grows with `out_dim`: about 84 KB at `out_dim=4096`, 165 KB at `8192`, and 1.29 MiB
at DINO's paper-scale `65536`. A `get_config()` round-trip test that compares only
hyperparameters is vacuous here — assert the center's value.

### Rule 3 — the structured-dict `y_pred` is **direct-invocation only**; stock `fit()` uses the **packed single tensor**

`DINOLoss` and `iBOTPatchLoss` accept a `Dict[str, Tensor]` `y_pred` (carrying student
logits, teacher logits, and — for iBOT — the patch mask). **This does not work under stock
`compile(loss=...)` / `fit()` on Keras 3.8.** Measured: `CompileLoss.build` broadcasts a
single `Loss` object across every leaf of a nested `y_pred` and then raises
`KeyError: "The path: ('student_logits',) in the 'loss' argument, can't be found in either
the model's output ('y_pred') or in the labels ('y_true')."`

The dict form is a documented API for calling the loss **directly**
(`loss(y_true, y_pred_dict)`), which is how the tests exercise it. A model that must be
compiled with one of these losses has to return a **single tensor**.

The single-tensor form is the **packed** convention: `y_pred` has last dimension
`2 * out_dim`, holding `concatenate([student_logits, teacher_logits], axis=-1)`. Build it
with `pack_student_teacher` from `src/dl_techniques/losses/dino_loss.py` — that function
is the single source of truth for the layout, and `_resolve_student_teacher` in the same
module is the only place that unpacks it. `y_true` is ignored, so feed a dummy label.

Two constraints on the packed form, both measured:

- **It must be rank 2.** The centering EMA reduces over `axis=0` only, so a
  `(batch, n_pairs, 2 * out_dim)` `y_pred` produces a `(1, n_pairs, out_dim)` batch centre
  and `center.assign()` dies mid-`fit()` with
  `NotImplementedError: numpy() is only available when eager execution is enabled` — a
  shape error wearing a backend error's clothes. Flatten the pair axis into the batch axis.
- **`sample_weight` cannot carry the second tensor.** `fit()` sources `sample_weight` from
  the **dataset** tuple, and the teacher's logits are produced by the model from the same
  batch, so there is no point at which they could be handed over. `iBOTPatchLoss`
  additionally refuses a non-`None` `sample_weight` outright (see below).

Relatedly, `iBOTPatchLoss.__call__` **refuses** a non-`None` `sample_weight` with a
`TypeError` naming `y_pred['mask']`, because the pre-fix docstring told callers to write
`loss(teacher, student, mask)` — a call that would otherwise silently apply the mask as a
scalar weighting.

### Rule 4 — `teacher_temp` is a Variable behind a **read-only** property

DINO warms the teacher's temperature up during training. A Python-float temperature cannot
do that: it is constant-folded into the traced training step. Measured, with the centering
EMA frozen so the temperature is the only thing that could move the loss — setting
`loss.teacher_temp = 4.0` (a 100x change) between two epochs moved the reported loss
`9.953619 -> 9.953612`, i.e. by 7e-07. Nothing.

So `teacher_temp` is a non-trainable `keras.Variable`, exposed as a read-only `float`
property. Change it with `loss.set_teacher_temp(value)`; a plain
`loss.teacher_temp = value` raises `AttributeError` naming the setter, which turns the
silent no-op into a loud failure. `get_config()` carries the **current** value, so a
warmup survives a checkpoint.

Drive it with the schedule **functions** already in
`src/dl_techniques/models/depth_anything/teacher_ema.py`
(`linear_ema_schedule` / `cosine_ema_schedule`) plus a `keras.callbacks.LambdaCallback` —
**not** with a new schedule-callback class. `src/dl_techniques/callbacks/temperature_annealing.py`
does **not** fit: measured, `TemperatureAnnealingCallback._iter_target_layers` walks
`self.model.layers` and requires each target to expose a `temperature` **Variable** plus a
`softplus_temperature` flag (the `dl_techniques.layers.logic.*` contract). A `keras.losses.Loss`
is not a layer of the model, so the callback finds `[]` targets on a compiled DINO model
and anneals nothing.

### 5.5 — `DINOTrainingModel`

`src/dl_techniques/models/dino/dino_training.py` packages the two networks:

- Input: one fixed-shape tensor `(batch, n_views, height, width, channels)`, where views
  `0` and `1` are the **global** crops and every view is at the **same** pixel resolution
  (the local-crop limitation of § 7, item 1).
- The student runs on all views; the teacher runs on the global views only, under
  `keras.ops.stop_gradient`, with `trainable=False`.
- Output: the packed tensor of Rule 3, at `(batch * n_pairs, 2 * out_dim)` — one row per
  (teacher global view, student view) pair, same-view pairs excluded.
- It exposes `update_teacher_ema(decay)`, which is the **exact name**
  `TeacherEMACallback` looks up. If that method is absent the callback logs one warning
  and self-disables: the run completes, the loss curve looks plausible, and the teacher is
  never trained. There is no other symptom, which is why the tests assert the teacher
  weights moved *and* moved toward the student.

```python
import numpy as np
from dl_techniques.models.dino import create_dino_training_model
from dl_techniques.losses.dino_loss import DINOLoss
from dl_techniques.models.depth_anything.teacher_ema import (
    TeacherEMACallback, cosine_ema_schedule,
)

model = create_dino_training_model(
    "tiny", image_size=32, patch_size=16, n_local_views=2, dino_out_dim=64,
)
model.compile(optimizer="adamw", loss=DINOLoss(out_dim=64))

views = np.random.rand(2, 4, 32, 32, 3).astype("float32")   # 2 global + 2 local
dummy_labels = np.zeros((2, 1), dtype="float32")            # ignored by the loss
model.fit(                       # NOTE: no validation_data (Rule 1)
    views, dummy_labels, epochs=1, batch_size=2, verbose=0,
    callbacks=[TeacherEMACallback(cosine_ema_schedule(0.99, 0.9999, 100))],
)
```

### 5.6 — The multi-crop dataset element

`src/dl_techniques/datasets/vision/multi_crop.py`'s `make_multi_crop_map_fn` produces
exactly the element § 5.5 consumes. It is a per-sample `tf.data` transform, so it enters
a pipeline through `build_raw_image_dataset(..., element_map_fn=...)` in
`src/train/energy_transformer/common.py` (applied after normalization, before batching).

```python
from dl_techniques.datasets.vision.multi_crop import make_multi_crop_map_fn

map_fn = make_multi_crop_map_fn(global_crop_size=96, n_local_crops=4, seed=42)
# (image, label) -> (views, label), views shape (2 + 4, 96, 96, 3);
# views 0 and 1 are the global crops. Batching gives (batch, 6, 96, 96, 3).
```

Two things about it are worth reading before use:

- **Local views are rendered at the GLOBAL resolution** — a smaller *area* is cropped and
  resized up (§ 7, item 1). `local_crop_size != global_crop_size` raises
  `NotImplementedError` naming positional-embedding interpolation, rather than silently
  mis-shaping the batch. The cost is that local views are exactly as expensive as global
  ones. Measured on GPU 1 (RTX 4070, 12 GB) at the smoke scale — `tiny`, 96 px, 4 local
  crops, batch 32, `dino_out_dim=4096` — one forward+backward peaks at **1.5 GiB**, so
  the trade-off is affordable at this scale.
- **Its augmentation is weaker than the paper's, and says so.** RandomResizedCrop,
  horizontal flip, brightness/contrast jitter, random grayscale and Gaussian blur are
  implemented; **saturation/hue jitter and solarization are not**, because this transform
  runs on mean/std-normalized images and both of those operators are defined against a
  `[0, 1]` value domain. The module docstring lists this; nothing infers the full recipe
  from the file's name.

---

## 6. Why there is no shared base class

`dino_v1.py`, `dino_v2.py` and `dino_v3.py` each build their own patch-embedding /
transformer-block wiring, and each carries its own `MODEL_VARIANTS`, `from_variant`,
`get_config` and `summary`. The obvious "normalize" reading is to unify them onto a shared
ViT trunk. That is a **deliberate non-goal**, and the reason is testability, not taste: a
trunk unification is a large behaviour-preserving refactor over three files whose test
suite is not dense enough to prove it behaviour-preserving. The risk budget is better
spent on the parts that were actually broken.

What *is* normalized: factory signatures, parameter names, validation guards, variant key
sets, the `__init__.py` surface, this README, and the test layout.

---

## 7. Named backlog — what is deliberately NOT implemented

These are gaps by decision, not by oversight. Each is named here so it cannot become a
silent omission again.

1. **Positional-embedding interpolation for smaller local crops.** DINO's multi-crop
   augmentation renders local views at a *smaller pixel resolution* than global views,
   which changes the patch-grid length and therefore requires interpolating the positional
   embedding table. Not implemented. The consequence is that a multi-crop pipeline built on
   these models must render local crops at the **same** pixel resolution as global crops
   (crop a smaller area, resize up), which loses the paper's compute saving on local views.
   `src/dl_techniques/datasets/vision/multi_crop.py` does exactly that, and refuses a
   different `local_crop_size` with a `NotImplementedError` naming this gap.
2. **Gram anchoring** (DINOv3). Needs a third, frozen "Gram teacher" network and an extra
   loss term on patch-feature Gram matrices. None of that machinery exists here.
3. **Sinkhorn-Knopp centering.** `src/dl_techniques/losses/dino_loss.py` provides EMA
   centering only.
4. **2-D axial RoPE with coordinate jittering** (DINOv3). The RoPE implemented here is 1-D
   over the flattened token sequence. See § 2.
5. **Register tokens on `DINOv3`.** `DINOv2VisionTransformer` has `num_register_tokens`;
   `DINOv3` does not.
6. **`src/dl_techniques/models/depth_anything/model.py`'s DINOv2 placeholder.** That model
   describes its encoder in prose as "a placeholder for DINOv2" and does not import from
   this package. Wiring it to a real `DINOv2VisionTransformer` is an open opportunity, not
   a defect in either package.
7. **`get_last_selfattention()`.** Blocked on `TransformerLayer.call`
   (`src/dl_techniques/layers/transformers/transformer.py`) not forwarding a
   `return_attention_weights` / `return_attention_scores` flag to its attention sub-layer.
   `GroupedQueryAttention` already supports it; `MultiHeadAttention` does not. Fixing this
   properly means touching a shared layer used across the repository.
8. **Pretrained weights.** None are shipped for any version. `pretrained=True` logs a
   warning and is otherwise ignored.

> **Training pipeline.** `src/dl_techniques/models/dino/dino_training.py` (§ 5.5) is the
> trainable model, `src/dl_techniques/datasets/vision/multi_crop.py` (§ 5.6) is its data
> side, and `src/train/dino/train_dino.py` is the runnable trainer that joins them under
> stock `fit()`:
>
> ```
> MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.dino.train_dino --smoke
> ```
>
> `--smoke` pins a MEASURED shape-validation scale (`tiny`, 96px globals, 4 local crops,
> `batch_size=32`, `dino_out_dim=4096`, a handful of steps; peak 1518.6 MiB of 10001 MiB
> on an RTX 4070). That is **not** a paper reproduction. The trainer passes **no**
> `validation_data` — see § 5 Rule 1, which is the reason, not a preference.
>
> **Validation is `src/train/dino/knn_eval.py`.** `KNNEvalCallback` extracts FROZEN
> student-backbone CLS features (the tensor feeding `DINOHead`, width `embed_dim` — not
> the projection-head output), runs a temperature-weighted cosine k-NN at k = 10 and 20
> against a TRAIN-split memory bank, and logs top-1 plus the two collapse numbers:
> `dino_feat_mean_cos` (mean pairwise feature cosine) and `dino_teacher_entropy_norm`
> (entropy of the mean teacher softmax, as a fraction of `log(out_dim)`). **A decreasing
> loss does not rule out collapse** — the collapsed solution is a genuine minimum of the
> cross-view objective — so read those columns in `results/<run>/training_log.csv`
> before calling a run good. That module's docstring carries the STOP thresholds.

---

## 8. Tests

```
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_models/test_dino/ -q
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_losses/test_dino_loss.py -q
```

| File | Covers |
|---|---|
| `tests/test_models/test_dino/test_dino_v1.py` | v1 smoke + forward, `qkv_bias` reaching the attention layer, `norm_last_layer`'s kernel-norm invariant, the `get_last_selfattention` raise, the `image_size % patch_size` guard, cross-version variant-table parity, `DINOHead` and `DINOv1` `.keras` round-trips with numeric assertions, and the D-020 fp16 normalization-overflow guard. |
| `tests/test_models/test_dino/test_dino_v2.py` | v2 smoke, the masked forward path, per-position mixed masks, register tokens, `.keras` round-trip, dtype-policy forward. |
| `tests/test_models/test_dino/test_dino_v3.py` | v3 smoke, RoPE liveness (a same-weights token-permutation contrast against a positional-information-free control), `.keras` round-trips at BOTH `positional_embedding_type` values, the `get_last_selfattention` raises, `image_size` int-or-tuple, dtype-policy forward on both positional modes. |
| `tests/test_models/test_dino/conftest.py` | The restore-safe parametrized `dtype_policy` fixture (float32 / mixed_float16 / float64). It is a LOCAL copy: `tests/test_layers/conftest.py`'s fixture is not reachable from `tests/test_models/` (sibling trees). |
| `tests/test_models/test_dino/test_dino_package.py` | The package surface: `__all__` completeness in both directions, factory-signature convergence, the `None`-defers-to-the-variant precedence rule, the `input_shape` refusal, and a checker asserting every path and import named in **this README** resolves. |
| `tests/test_models/test_dino/test_dino_training.py` | `DINOTrainingModel`: the multi-crop input contract, the packed row layout verified pair-by-pair against independent sub-model calls, a gradient-free teacher (with a student control proving the probe can see a gradient), the `update_teacher_ema` contract and its exact EMA arithmetic, a real `fit()` with `TeacherEMACallback` asserting the teacher moved TOWARD the student, `.keras` round-trip with numeric assertions AND an explicit `teacher.trainable is False` assertion. |
| `tests/test_datasets/test_multi_crop.py` | The multi-crop element (§ 5.6): the fixed-shape `(2 + n_local)`-view stack pinned against `dino_training.N_GLOBAL_VIEWS`, a REAL batched-and-iterated `tf.data` pipeline, pairwise proof that the crops are genuinely different tensors, a statistical check that global views cover a larger area than local ones, the `local_crop_size` refusal asserted on its MESSAGE, seeded determinism and unseeded non-determinism, per-augmentation liveness, and an end-to-end forward pass of a batched element through `DINOTrainingModel`. |
| `tests/test_train/test_dino/test_train_dino.py` | The trainer (`src/train/dino/train_dino.py`): a STRUCTURAL CLI-to-config wiring guard (reflection over `dataclasses.fields` and the parser's dests, fail-closed in both directions, so an unwired flag is RED by default), `TrainingConfig.__post_init__` rejections asserted on their messages, model/loss/callback construction, a spy on the real `train_dino()` path asserting `fit()` receives **no** `validation_data`, and a real two-epoch `fit()` asserting the teacher-temperature `LambdaCallback` MOVES the loss's `teacher_temp` Variable. |
| `tests/test_train/test_dino/test_knn_eval.py` | The k-NN probe and the collapse diagnostic (`src/train/dino/knn_eval.py`): perfectly separable synthetic features scoring 1.0 against identical features falling to chance (with a non-vacuity control proving a constant predictor cannot satisfy both), a contrast proving the `exp(sim/T)` weighting is really used rather than a majority vote, the bank/query OVERLAP guard — which fires on self-retrieval but deliberately does NOT fire on collapse — a RED-proven collapse detector fed a deliberately collapsed matrix, the entropy half isolated with spread features and a one-hot teacher, the entropy taken of the MEAN distribution rather than the mean of per-sample entropies, a backbone-vs-projection-head width assertion, and REAL two-epoch `fit()` runs proving the columns reach `CSVLogger`'s CSV — including the executed negative control that placing the callback AFTER `CSVLogger` loses every column. |
| `tests/test_losses/test_dino_loss.py` | The three losses: construction, forward finiteness, the center reaching a hand-computed EMA value under a real 2-step `fit()`, `get_config` round-trip including the center's value, the all-masked / none-masked `iBOTPatchLoss` edges, the packed single-tensor convention under a real `fit()`, the schedulable `teacher_temp`, and the D-023 KoLeo fp16 normalization-overflow guard. |

---

## 9. References

- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO),
  ICCV 2021. arXiv:2104.14294
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*,
  TMLR 2024. arXiv:2304.07193
- Darcet et al., *Vision Transformers Need Registers*, ICLR 2024. arXiv:2309.16588
- Zhou et al., *iBOT: Image BERT Pre-Training with Online Tokenizer*, ICLR 2022.
  arXiv:2111.07832
- Siméoni et al., *DINOv3*, 2025. arXiv:2508.10104
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021.
  arXiv:2104.09864
