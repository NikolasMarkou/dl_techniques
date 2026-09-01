# DINO — self-distillation vision transformers (v1 / v2 / v3)

`src/dl_techniques/models/vision/dino/` implements three Vision Transformer backbones from
the DINO paper line, plus the DINO projection head. The matching losses live one package
over, in `src/dl_techniques/losses/dino_loss.py`.

DINO ("**DI**stillation with **NO** labels") trains an image encoder with no labels by
making a **student** network agree with a **teacher** that is an exponential moving average
of the student itself. Different augmented crops of one image go to the two networks, and
the student is asked to predict the teacher's output distribution. Nothing stops both
collapsing to a constant, so DINO adds two stabilizers: the teacher's outputs are
**centered** by a running mean and **sharpened** by a low temperature. Those two together
are what make an otherwise-degenerate objective learn something. The measurements behind
every number in § 6 live in `research/2026_dino_ssl_measurements.md`, which this README
cites rather than carries.

---

## 1. What is here

| File | Contents |
|---|---|
| `src/dl_techniques/models/vision/dino/dino_v1.py` | `DINOHead` (the projection head all three versions use for SSL), `DINOv1`, `create_dino_v1`, `create_dino_teacher_student_pair`. |
| `src/dl_techniques/models/vision/dino/dino_v2.py` | `DINOv2Block`, `DINOv2VisionTransformer` (the backbone), `DINOv2` (backbone + classifier), `create_dino_v2`. |
| `src/dl_techniques/models/vision/dino/dino_v3.py` | `DINOv3`, `create_dino_v3`. |
| `src/dl_techniques/models/vision/dino/training.py` | `DINOTrainingModel`, `create_dino_training_model` — student + frozen EMA teacher over a multi-crop batch, trainable under stock `fit()`. See § 5.5. |
| `src/dl_techniques/losses/dino_loss.py` | `DINOLoss`, `iBOTPatchLoss`, `KoLeoLoss`, re-exported from `dl_techniques.losses`. See § 5. |
| `src/dl_techniques/datasets/vision/multi_crop.py` | `make_multi_crop_map_fn` — the `tf.data` transform producing the element `DINOTrainingModel` consumes. See § 5.6. |
| `src/dl_techniques/models/vision/depth_anything/teacher_ema.py` | `TeacherEMACallback` plus `cosine_ema_schedule` / `linear_ema_schedule`. Model-agnostic: it drives the teacher EMA of any model exposing `update_teacher_ema(decay)`. |
| `tests/test_models/test_dino/` | This package's tests. See § 9. |

`src/dl_techniques/models/vision/dino/__init__.py` re-exports every class and factory above
through an explicit `__all__`; `src/dl_techniques/models/vision/dino/common.py` holds the
non-public `reject_input_shape` / `sync_teacher_to_student` helpers.

**Variant tables are per-class.** `DINOv1.MODEL_VARIANTS`,
`DINOv2VisionTransformer.MODEL_VARIANTS` and `DINOv3.MODEL_VARIANTS` are three separate class
attributes that share a name and disagree on their contents (v2's `giant` carries
`ffn_type='swiglu'`; v3's carries `patch_size=(14, 14)` and `stochastic_depth_rate=0.4`; v1's
carries neither), so the package binds **no** module-level alias — a single name would have
to pick one and misdescribe the other two; reach them through the classes. All three
key sets agree (`tiny`, `small`, `base`, `large`, `giant`) and the four shared architecture
fields (`embed_dim`, `depth`, `num_heads`, `mlp_ratio`) agree value-for-value. `giant` is
**not** a DINOv1-paper variant (Caron et al. 2021 stop at ViT-B/8); it exists in `dino_v1.py`
for key-set parity only.

---

## 2. Honest capability table (v1 / v2 / v3)

All three share a pre-norm ViT trunk with a `[CLS]` token, patch embedding, stochastic
depth with linear decay, and a tested `.keras` round-trip. The rows below are where they
differ; those marked **NOT IMPLEMENTED** name a mechanism the corresponding paper has and
this code does not.

| Mechanism | `DINOv1` | `DINOv2` | `DINOv3` |
|---|---|---|---|
| Learned absolute positional embedding | yes | yes | yes (default) |
| `DINOHead` projection head | yes (`include_projection_head=True`) | not built in — compose `DINOHead` yourself | not built in — compose `DINOHead` yourself |
| `norm_last_layer` (unit-norm output prototypes) | **yes**, on `DINOHead` — see note below | via `DINOHead` | via `DINOHead` |
| Teacher/student pair factory | `create_dino_teacher_student_pair` | none | none |
| iBOT-style patch masking on the forward path | no | **yes** — the backbone takes `[images, masks]` and applies a learnable mask token (`MaskTokenApply`) | no |
| The iBOT **loss** half | — | **not wired here.** `iBOTPatchLoss` exists in `src/dl_techniques/losses/dino_loss.py`; connecting it to the masked forward pass is the trainer's job, not the model's | — |
| Register tokens (Darcet et al. 2023) | no | **yes** — `num_register_tokens`, defaulting to 4 on `large`/`giant` | **NOT IMPLEMENTED** |
| LayerScale (`layers/layer_scale.py`) | no | yes — `init_values` | no |
| SwiGLU FFN | opt-in via `ffn_type` | **yes**, and it is the `giant` variant's default | opt-in via `ffn_type` |
| RoPE positional embedding | no | no | **yes — but 1-D, not the paper's 2-D axial.** See note below |
| Gram anchoring | — | — | **NOT IMPLEMENTED** (needs a third frozen "Gram teacher" network) |
| Sinkhorn-Knopp centering | **NOT IMPLEMENTED** (EMA centering only) | **NOT IMPLEMENTED** | **NOT IMPLEMENTED** |
| High-res adaptation / distillation from a large pretrained teacher | — | — | **NOT IMPLEMENTED** |
| `get_last_selfattention()` (attention-map extraction) | **raises `NotImplementedError`** | not defined | **raises `NotImplementedError`** |
| Pretrained weights | none shipped | `pretrained=True` raises `NotImplementedError` | `pretrained=True` raises `NotImplementedError` |

### `norm_last_layer` on `DINOHead`

The reference DINO wraps the final projection in PyTorch's weight-norm reparameterization
`w = g * v / ||v||` and, when `norm_last_layer=True`, pins `g = 1` and freezes it so every
output prototype keeps unit L2 norm. Here the **invariant** is reproduced —
`keras.constraints.UnitNorm(axis=0)` on the final `Dense` kernel plus a one-off application
at `build()` time — but not the **optimization path**: a Keras constraint projects *after*
each optimizer step where the reference reparameterizes *before* it. The build-time
application is not redundant, since Keras applies a constraint inside `optimizer.apply` and
a freshly built or inference-only head would otherwise violate the invariant the flag
promises. `norm_last_layer=False` leaves the kernel unconstrained.

### RoPE on `DINOv3` — real, live, and 1-D

`positional_embedding_type='rope'` is genuine rotary position embedding applied to Q and K
*inside* the attention operator, routed through the registered `group_query` attention type
with `num_kv_heads == num_heads` (which reduces it to plain multi-head attention with RoPE),
and the learned absolute table is then **omitted entirely**. Two things it is not: it is
**1-D**, over the flattened token sequence, where DINOv3 uses a 2-D axial formulation over
patch `(row, column)` coordinates with random jittering; and **a checkpoint is not portable
between `learned` and `rope`**, because the two modes build structurally different attention
classes (`MultiHeadAttention` vs `GroupedQueryAttention`) whose weight sets do not
correspond. `rope_percentage=0.0` is legal but degenerate — it rotates nothing, leaving no
positional information at all — and stays legal as the RoPE-liveness test's control arm.

### `get_last_selfattention()` raises — for two different reasons

Both used to log a warning and return a zeros tensor, which renders as a blank attention
map a caller cannot distinguish from a broken model; both now raise `NotImplementedError`
naming the missing capability. On `DINOv1` (and any `learned`-mode `DINOv3`) nothing on the
`multi_head` attention path accepts a `return_attention_scores` flag at all. On `DINOv3` in
`rope` mode, `GroupedQueryAttention.call` *does* accept `return_attention_weights` and
returns a valid map, but `TransformerLayer.call` does not forward it, so the capability is
unreachable one frame away.

---

## 3. Factory signatures

All three factories share one parameter scheme; everything after `variant` is
**keyword-only**, and `image_size` / `patch_size` each accept an `int` or an
`(height, width)` tuple. **There is no `input_shape` argument**: it was a redundant second
spelling of `image_size` that could disagree with it, silently building a model whose patch
grid did not match its input, so all three factories raise `TypeError` naming `image_size`.
(The `DINOv1` and `DINOv2` *constructors* still accept it as a lower-level escape hatch.)

### The `None`-defers-to-the-variant precedence rule

A factory parameter passed as `None` means "the caller said nothing" and defers to the
variant's own `MODEL_VARIANTS` entry, or to the version default if the entry is silent; an
explicit non-`None` value always wins. Before this rule, `ffn_type='mlp'` was both the
default *and* the promotion trigger on `giant`, so a caller asking for MLP got SwiGLU.

| Parameter | `None` resolves to |
|---|---|
| `create_dino_v1(patch_size=...)` | `16` (v1's variants define no patch size) |
| `create_dino_v2(patch_size=...)` | `14` (v2's variants define no patch size) |
| `create_dino_v3(patch_size=...)` | **the variant's own**: `(14, 14)` on `giant`, `(16, 16)` otherwise |
| `create_dino_v2(ffn_type=...)` | **the variant's own**: `'swiglu'` on `giant`, `'mlp'` otherwise |
| `create_dino_v2(num_register_tokens=...)` | `4` on `large`/`giant`, `0` otherwise |

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
**The `num_classes` defaults genuinely differ** — `0` on v1, `1000` on v2 and v3 — and are
left that way rather than converged: changing them would be a behaviour change to existing
callers, not a surface normalization. Pass `num_classes` explicitly. Version-specific
knobs stay per-version on purpose; convergence here is about the shared surface, not about
erasing real differences between three papers.

---

## 4. Runnable examples
Every example below is executed by `tests/test_models/test_dino/test_dino_package.py`, which
also checks that every import and path named in this file resolves.

### DINOv1 — classification, and an SSL projection head

```python
from dl_techniques.models.vision.dino import create_dino_v1, create_dino_teacher_student_pair

# Supervised fine-tuning head: 32x32 input, patch 16 -> a 2x2 patch grid.
model = create_dino_v1("small", image_size=32, patch_size=16, num_classes=10)

# Self-supervised: no classifier, a DINO projection head instead.
ssl_model = create_dino_v1(
    "small", image_size=32, patch_size=16, num_classes=0, include_top=False,
    include_projection_head=True, dino_out_dim=4096,
)

# A teacher/student pair: identical architecture, DISTINCT weight variables, and — as
# DINO requires — identical weight VALUES at construction. A teacher from its own random
# draw is not an EMA of the student and corrupts the first 1-3 epochs.
teacher, student = create_dino_teacher_student_pair(
    "small", image_size=32, patch_size=16, dino_out_dim=4096
)
```

### DINOv2 — masked forward pass, register tokens, SwiGLU

`DINOv2` takes a **two-element input** `[images, masks]` — a boolean
`(batch, num_patches)` iBOT mask, all-`False` for the ordinary unmasked pass.

```python
import numpy as np
from dl_techniques.models.vision.dino import create_dino_v2

model = create_dino_v2("tiny", image_size=28, patch_size=14, num_classes=10)
images = np.random.rand(2, 28, 28, 3).astype("float32")
masks = np.zeros((2, 4), dtype=bool)          # 28 // 14 == 2 -> 2*2 == 4 patches
logits = model([images, masks], training=False)

# `giant` brings SwiGLU and 4 register tokens from its own variant entry. The R register
# tokens are R INDEPENDENT learnable vectors — one `(1, R, embed_dim)` weight owned by
# `RegisterTokens` — inserted after the positional embedding, so position-free by design.
giant = create_dino_v2(
    "giant", image_size=32, patch_size=16, num_classes=10,
    embed_dim=32, depth=1, num_heads=4,        # shrunk so the example is cheap
)
```

### DINOv3 — RoPE, and a backbone plus projection head

```python
import keras
from dl_techniques.models.vision.dino import DINOHead, create_dino_v3

# RoPE replaces the learned absolute table; drop the flag for the default embeddings.
rope = create_dino_v3(
    "small", image_size=32, patch_size=16, num_classes=10,
    positional_embedding_type="rope", rope_theta=10000.0,
)

# A feature-extraction backbone plus a projection head.
backbone = create_dino_v3("small", image_size=32, patch_size=16, include_top=False)
head = DINOHead(in_dim=384, out_dim=4096, hidden_dim=512, bottleneck_dim=64)
ssl_model = keras.Model(inputs=backbone.input, outputs=head(backbone.output))
```

---


`src/dl_techniques/losses/dino_loss.py` provides three losses, all re-exported from
`dl_techniques.losses`:

```python
from dl_techniques.losses import DINOLoss, iBOTPatchLoss, KoLeoLoss
```

| Loss | Job |
|---|---|
| `DINOLoss` | The cross-view consistency objective on `[CLS]` outputs, with the EMA-centered, temperature-sharpened teacher. Owns the centering state. |
| `iBOTPatchLoss` | The same objective at patch level, over the masked patches only. Also owns a centering state. |
| `KoLeoLoss` | A Kozachenko-Leonenko entropic regularizer pushing embeddings apart on the unit sphere. An anti-collapse term, not a distillation term. |

The four rules below are **API contracts**: each names a Keras or TensorFlow behaviour that
fails silently if ignored.

### Rule 1 — SSL pretraining must run **without** `validation_data`

`DINOLoss` maintains its centering statistic by `.assign()`ing a `keras.Variable` inside
`call()`, and `call()` runs on **every** batch, validation batches included. Measured: a
4-sample validation set at `batch_size=2` doubled an epoch's update count from 2 to 4 and
pushed the center 81% past its correct value, with a finite loss and a clean exit.
Validation belongs in a k-NN probe callback on frozen features (§ 6), which never invokes
the loss.

### Rule 2 — the center's **value** is serialized through `get_config()`

Keras does not checkpoint loss-owned variables. A `.keras` round-trip of a model compiled
with `DINOLoss` succeeds, returns the right subclass with a live `.center` Variable, and
silently brings the center back at its `Zeros()` initialization, so a resumed run restarts
centering from scratch. Both `get_config()` implementations therefore carry the center's
value as a nested list and `from_config` assigns it back; a missing `center` key is
tolerated, so older configs still load. The cost is a config blob growing with `out_dim` —
about 84 KB at `4096`, 1.29 MiB at paper-scale `65536`. A round-trip test comparing only
hyperparameters is vacuous here: assert the center's value.

### Rule 3 — the structured-dict `y_pred` is **direct-invocation only**; stock `fit()` uses the **packed single tensor**

`DINOLoss` and `iBOTPatchLoss` accept a `Dict[str, Tensor]` `y_pred` (student logits,
teacher logits, and for iBOT the patch mask). **This does not work under stock
`compile(loss=...)` / `fit()` on Keras 3.8**: `CompileLoss.build` broadcasts a single `Loss`
across every leaf of a nested `y_pred` and raises `KeyError: "The path:
('student_logits',) in the 'loss' argument, can't be found ..."`. The dict form is for
calling the loss **directly**, which is how the tests exercise it.

A model compiled with one of these losses must return a **single tensor** in the **packed**
convention: last dimension `2 * out_dim`, holding
`concatenate([student_logits, teacher_logits], axis=-1)`. Build it with
`pack_student_teacher` from `src/dl_techniques/losses/dino_loss.py`, the single source of
truth for the layout; `y_true` is ignored. It must also be **rank 2**:
the centering EMA reduces over `axis=0` only, so a `(batch, n_pairs, 2 * out_dim)` `y_pred`
gives a `(1, n_pairs, out_dim)` batch centre and `center.assign()` dies mid-`fit()` with
`NotImplementedError: numpy() is only available when eager execution is enabled` — a shape
error wearing a backend error's clothes, fixed by flattening the pair axis into the batch
axis. `sample_weight` cannot carry the teacher tensor either: `fit()` sources it from the
dataset tuple, and `iBOTPatchLoss.__call__` refuses a non-`None` `sample_weight` with a
`TypeError` naming `y_pred['mask']`.

### Rule 4 — `teacher_temp` is a Variable behind a **read-only** property

DINO warms the teacher temperature up during training, and a Python float cannot do that:
it is constant-folded into the traced training step. Measured, with the centering EMA frozen
so temperature was the only free variable, setting `loss.teacher_temp = 4.0` between two
epochs moved the reported loss by 7e-07 — nothing. So `teacher_temp` is a non-trainable `keras.Variable` behind a read-only `float` property:
change it with `loss.set_teacher_temp(value)`, since a plain assignment raises
`AttributeError` naming the setter. `get_config()` carries the current value, so a warmup
survives a checkpoint. Drive it with the schedule **functions** in
`src/dl_techniques/models/vision/depth_anything/teacher_ema.py` plus a
`keras.callbacks.LambdaCallback`, not a new callback class:
`src/dl_techniques/callbacks/temperature_annealing.py` walks `self.model.layers` and needs
each target to expose a `temperature` Variable, and a `keras.losses.Loss` is not a layer of
the model, so it finds `[]` targets and anneals nothing.

### 5.5 — `DINOTrainingModel`

`src/dl_techniques/models/vision/dino/training.py` packages the two networks. Its input is
one fixed-shape tensor `(batch, n_views, height, width, channels)`, where views `0` and `1`
are the **global** crops and every view is at the **same** pixel resolution (§ 8). The
student runs on all views; the teacher runs on the global views only, under
`keras.ops.stop_gradient` and with `trainable=False`. The output is the packed tensor of Rule
3 at `(batch * n_pairs, 2 * out_dim)` — one row per (teacher global, student) view pair.

It exposes `update_teacher_ema(decay)`, the **exact name** `TeacherEMACallback` looks up. If
that method is absent the callback logs one warning and self-disables: the run completes, the
loss curve looks plausible, and the teacher is never trained — no other symptom, which is why
the tests assert the teacher weights moved *and* moved toward the student.

### 5.6 — The multi-crop dataset element

`src/dl_techniques/datasets/vision/multi_crop.py`'s `make_multi_crop_map_fn` produces
exactly the element § 5.5 consumes. It is a per-sample `tf.data` transform, entering a
pipeline through `build_raw_image_dataset(..., element_map_fn=...)` in
`src/train/energy_transformer/common.py`, after normalization and before batching.

```python
from dl_techniques.datasets.vision.multi_crop import make_multi_crop_map_fn

# (image, label) -> (views, label); views 0 and 1 are the global crops, so the element
# is (2 + 4, 96, 96, 3) and a batch is (batch, 6, 96, 96, 3).
map_fn = make_multi_crop_map_fn(global_crop_size=96, n_local_crops=4, seed=42)
```

**Local views are rendered at the GLOBAL resolution** — a smaller *area* is cropped and
resized up (§ 8) — so they cost as much as global ones, and
`local_crop_size != global_crop_size` raises `NotImplementedError` naming
positional-embedding interpolation rather than silently mis-shaping the batch. **The
augmentation is also weaker than the paper's**: RandomResizedCrop, flip, brightness and
contrast jitter, random grayscale and Gaussian blur are implemented, but
**saturation/hue jitter and solarization are not** — the transform runs on
mean/std-normalized images and both operators need a `[0, 1]` domain.

---

## 6. Running a training job: the trainer, its flags, and what was measured

This package is the **library**. The runnable trainer is `src/train/dino/train_dino.py` and
its validation probe is `src/train/dino/knn_eval.py`; read those two module docstrings
before launching anything, and `research/2026_dino_ssl_measurements.md` for the evidence.

```
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.dino.train_dino --smoke
```

The trainer joins `src/dl_techniques/models/vision/dino/training.py` (§ 5.5) and
`src/dl_techniques/datasets/vision/multi_crop.py` (§ 5.6) under stock `fit()` with **no**
`validation_data` — § 5 Rule 1 is the reason, not a preference. Validation is
`KNNEvalCallback`: it extracts FROZEN student-backbone CLS features (the tensor feeding
`DINOHead`, width `embed_dim`, not the projection-head output), runs a temperature-weighted
cosine k-NN at k = 10 and 20 against a TRAIN-split memory bank, and logs top-1 plus
`dino_feat_mean_cos` (mean pairwise feature cosine) and `dino_teacher_entropy_norm`
(entropy of the mean teacher softmax over `log(out_dim)`). **A decreasing loss does not rule
out collapse** — the collapsed solution is a genuine minimum of the objective — so read
those columns in `results/<run>/training_log.csv` first.

### 6.1 What the headline measurement obliges you to pass
**The result itself is not repeated here** — the 2-arm x 2-seed x 60-epoch comparison, its
pre-registered decision rule, the per-arm zero-step controls, endpoints, deltas and verdicts
are `research/2026_dino_ssl_measurements.md` § 1.

**A no-flag run carries the improved arm's two treatment values and NOT its scale.** The
treatment keys are the shipped defaults (`ema_warmup_epochs=1.0`, 295 steps at
imagenette/batch 32, and `teacher_temp_final=0.04`); the scale is not.
`parse_arguments([])` gives `--variant small --global-crop-size 224 --dino-out-dim 65536
--epochs 100 --knn-bank-batches 16 --knn-query-batches 8` and no `--max-steps`, whereas
every measured arm ran at `tiny` / 96 px / `dino_out_dim=4096` / 4 local crops / batch 32 /
60 epochs / bank 64 / query 32 / `--max-steps 100000`, so a bare
`python -m train.dino.train_dino` reproduces nothing in that record and has never been run
at all. **To reproduce an arm, use the `--smoke`-scaled invocations in the `Usage::` block
of `src/train/dino/train_dino.py`**; to restore the OLD treatment keys at any scale, pass
`--ema-warmup-epochs 0 --teacher-temp-final 0.07` — the baseline arm's two values, not the
baseline arm. Note also that `--knn-eval-every` picks which epochs the endpoint averages (`4`
averages 48/52/56, the default `1` averages 57/58/59): a different estimator.


Defaults below are the SHIPPED values, read from `TrainingConfig` and `parse_arguments` in
`src/train/dino/train_dino.py`. "Measured?" is three-valued: **MEASURED** names the endpoint
the effect was seen on, **MEASURED-NO-DIFFERENCE** means it was run and moved nothing, and
**UNMEASURED** means no arm has ever isolated it — an unmeasured flag next to a measured one
inherits none of its credibility.

| flag | what it does | default | measured? | the number, and its caveat |
|---|---|---|---|---|
| `--ema-warmup-epochs` | Freezes the teacher-weight EMA for the first N epochs (teacher stays at its student-synced init). The default-bearing knob. | `1.0` | MEASURED — epoch-0 `dino_knn_top1_k20` vs own control, 8 epochs, 2 seeds, smoke scale, controlled stream | **+0.0498** vs the old no-freeze default (per-seed +0.0625 / +0.0371), on a matrix whose own null was exactly 0.000. Measured as `--ema-warmup-steps 295`, which is one epoch at imagenette/batch 32. It is a **SUPERSET**: it also re-bases the cosine EMA ramp N steps later, and freeze-vs-re-basing has never been separated. The epoch-0 endpoint is also structurally kind to it (it cannot distinguish "fixed the dip" from "learned less"). |
| `--ema-warmup-steps` | Absolute-step override of the same freeze; any value `> 0` wins over `--ema-warmup-epochs`. | `0` (defer to epochs) | MEASURED — this is the flag the 8-epoch and 60-epoch runs actually passed | `295` is `num_train // batch_size` at imagenette/batch 32 **only**; it is roughly two epochs at batch 64 and arbitrary elsewhere. Use it to reproduce a pre-existing run, not as a recipe constant. |
| `--teacher-temp` | Teacher temperature at epoch 0 (start of the warmup). | `0.04` | UNMEASURED as a knob | It was `0.04` in **both** arms of the 60-epoch runs and in every arm of the 8-epoch matrices, so no measurement recorded here varies it. |
| `--teacher-temp-final` | Teacher temperature after the warmup. | `0.04` (EQUAL to `--teacher-temp`, i.e. no warmup) | MEASURED-NO-DIFFERENCE at the pre-registered endpoint; AMBIGUOUS on the exploratory one | Exactly **0.0000** at both seeds on the pre-registered epoch-0 endpoint (8 epochs, controlled stream) — it is structurally inert there. Its exploratory last-3-epochs effect **shrank from +0.0518 to +0.0387** once the stream was controlled. Only the PAIR with the EMA freeze has a 60-epoch IMPROVED verdict; **no temp-only arm was ever run at 60 epochs**. Pass `0.07` for the paper recipe. |
| `--teacher-temp-warmup-epochs` | Epoch horizon of the linear `teacher_temp -> teacher_temp_final` ramp. | `30` | UNMEASURED, and **INERT at the shipped defaults** | Both endpoints are `0.04`, so `linear_ema_schedule` returns a constant and the horizon multiplies a zero delta. It becomes live only when `--teacher-temp-final` differs from `--teacher-temp`. |
| `--stateless-augmentation` / `--no-stateless-augmentation` | Draws the multi-crop augmentation from `tf.random.stateless_uniform` keyed on a per-element counter instead of one shared `tf.random.Generator` stream. | **ON** | MEASURED — sha1 of the first 3 batches, 2 processes, CPU-only probe, real `build_dataset` | This flag ALONE still gives non-identical batches across processes; **both** reproducibility flags together are bit-identical on all 3 batches. The probe never ran a GPU kernel, so it says nothing about cuDNN nondeterminism. `--no-` restores the old shared-`Generator` behaviour and with it comparability to any run in `results/` measured before this default moved. This path has only ever been exercised end-to-end in one plan's matrices. |
| `--seed-training-stream` / `--no-seed-training-stream` | Seeds the TRAINING stream's TFDS file interleave with `--seed`. | **ON** | MEASURED — same 2-process CPU-only sha1 probe | Also not sufficient alone (the augmentation RNG is the residual source). `--seed` on its own does **not** make two runs comparable. `--no-` restores the unpinned file order. |
| `--source-image-size` | Resolution at which records are DECODED, i.e. what the multi-crop transform crops from. `None` means `--global-crop-size`, so local crops come from an already-downsampled thumbnail. | `None` | MEASURED-NO-DIFFERENCE — 8 epochs, 2 seeds, smoke scale, controlled stream | **+0.0024** on the pre-registered endpoint and +0.0028 on the exploratory one, both an order of magnitude inside the ±0.02 no-difference band. Geometry (N=2000 draws): at a 96 px source, 100% of local views are upsampled, mean 2.35x, worst 4.50x; at 224 px, mean 1.006, worst 1.92x, and 39% are still upsamples. So 224 px **mitigates** the defect and does not eliminate it — the null result is not evidence that crop resolution is irrelevant. |
| `--smoke` | Preset pinning the shape-validation scale: `tiny`, 96 px globals, 4 local crops, batch 32, `dino_out_dim=4096`, 2 epochs, `max_steps=5`, `ema_warmup_epochs=0.0`. **Any flag you actually type beats the preset — at any value, including the flag's own parser default.** Provenance is a raw `sys.argv` scan (`explicitly_set_flags` in `src/train/common/args.py`, attached to the Namespace by `parse_arguments`), not a parsed-value-vs-default comparison, so an explicitly-passed default is no longer indistinguishable from an omission. MEASURED at imagenette/batch 32 (`steps_per_epoch=295`): `--smoke --ema-warmup-epochs 1.0` resolves to `warmup_steps=295`, `1.5` gives 442 and `--ema-warmup-steps 295` gives 295. | off | UNMEASURED as an arm — it is a scale, not a treatment | Validates SHAPES and wiring; it is **not** a paper reproduction. It carries two traps that silently change what you measure — see § 6.3. Note it pins `ema_warmup_epochs=0.0` on purpose, so `--smoke` keeps its historical no-freeze semantics rather than silently gaining the new default's teacher freeze. |
| `--knn-bank-batches` / `--knn-query-batches` | Size of the k-NN memory bank (TRAIN split) and query set (VALIDATION split), in batches. | `16` / `8` | Not a treatment — it is the **estimator** | Every k-NN figure in `research/2026_dino_ssl_measurements.md` (which is where they all live — § 6.1 no longer restates them) was measured at **64 / 32** (2048 bank images). At the default 16 / 8 (512 images) a top-1 is a *different estimator*, not a noisier reading of the same one. Pass `--knn-bank-batches 64 --knn-query-batches 32` for anything you intend to compare. |
| `--max-steps` | Caps `steps_per_epoch`. | `None` (full epoch); `--smoke` pins **5** | UNMEASURED | Not a treatment, but a silent scale change: see § 6.3 trap 1. |
| `--random-init-repeats` | Repeats of the ZERO-OPTIMIZER-STEP k-NN control written to `<run_dir>/random_init_control.json` before `fit()` performs a single update. | `2` | MEASURED-NO-DIFFERENCE **by construction** | The bank is `.cache()`d, so repeats replay the identical bank and the within-run range is exactly 0.0. Each run records `repeats_are_independent: false`. **This is not a noise estimate.** The only genuine noise data are the across-seed spread and one 0.0020 cross-process datum. |
| `--center-momentum` | Momentum of the `DINOLoss` centering EMA. | `0.9` | MEASURED — **CLEARED** as a cause of the early k-NN dip, 8 epochs, 2 seeds, unseeded stream | Effect **-0.0063** at `--center-momentum 0.0` in the dip matrix, i.e. removing the centering lag entirely made the dip marginally worse. That matrix ran on an unseeded stream whose own null was 0.0400 wide. |

### 6.3 Two measurement traps in the shipped defaults

Neither is a bug; both silently produce a number that is not the one you think you read.
**`--smoke` sets `max_steps=5`**, so an unqualified `--smoke` "epoch" is 5 of its 295 steps
and `--smoke --epochs 40` trains for 200 steps, not ~11 800; a run meant to *train* must
also pass `--max-steps 100000`. And **`--smoke` leaves `--knn-bank-batches 16
--knn-query-batches 8`** (512 bank images) while every k-NN number in the record, and the
control band in `src/train/dino/knn_eval.py`'s docstring, was measured at **64 / 32** (2048
images) — a different estimator.

---

## 7. Why there is no shared base class

`dino_v1.py`, `dino_v2.py` and `dino_v3.py` each build their own patch-embedding and
transformer-block wiring, and each carries its own `MODEL_VARIANTS`, `from_variant`,
`get_config` and `summary`. Unifying them onto a shared ViT trunk is a **deliberate
non-goal**, for testability rather than taste: it is a large behaviour-preserving refactor
over three files whose test suite is not dense enough to prove it behaviour-preserving.
What *is* normalized: factory signatures, parameter names, validation guards, variant key
sets, the `__init__.py` surface and the test layout.

---

## 8. Named backlog — what is deliberately NOT implemented

Gaps by decision, not oversight, named so none becomes a silent omission again.

| Gap | Note |
|---|---|
| Positional-embedding interpolation for smaller local crops | DINO renders local views at a smaller pixel resolution, which changes the patch-grid length and needs the positional table interpolated. Not implemented, so a pipeline must render local crops at the **same** resolution as global ones (crop a smaller area, resize up), losing the paper's compute saving. `src/dl_techniques/datasets/vision/multi_crop.py` does exactly that and refuses a different `local_crop_size` with a `NotImplementedError` naming this gap. |
| Gram anchoring (DINOv3) | Needs a third frozen "Gram teacher" and a patch-feature Gram loss term. None of that machinery exists here. |
| Sinkhorn-Knopp centering | `src/dl_techniques/losses/dino_loss.py` is EMA-only. |
| 2-D axial RoPE, and register tokens, on `DINOv3` | The RoPE here is 1-D (§ 2), with no coordinate jittering; `DINOv2VisionTransformer` has register tokens and `DINOv3` does not. |
| A real DINOv2 inside `depth_anything` | `src/dl_techniques/models/vision/depth_anything/model.py` uses a placeholder encoder and does not import from this package. An open opportunity, not a defect in either package. |
| `get_last_selfattention()` | Blocked on `TransformerLayer.call` (`src/dl_techniques/layers/transformers/transformer.py`) not forwarding a `return_attention_weights` flag. Fixing it means touching a repo-wide shared layer. |
| Pretrained weights | None are shipped. `pretrained=True` raises `NotImplementedError`; build with `pretrained=False` and warm-start with `model.load_weights(path)`. |
| Attributing the improved 60-epoch configuration | The verdict belongs to the PAIR (`ema_warmup_epochs 0 -> 1.0`, `teacher_temp_final 0.07 -> 0.04`); the two were separated only at 8 epochs, never at 60. `--ema-warmup-epochs` is itself a superset — it freezes the teacher EMA *and* re-bases the cosine EMA ramp — and those have never been separated either. |
| Cropping local views from the ORIGINAL record | `build_raw_image_dataset` resizes each record to `image_size` before the multi-crop transform runs, so a "local crop of the source" is a crop of a downsampled square. The bounded remedy (`--source-image-size`) was run and measured NO DIFFERENCE (§ 6.2). |

---

## 9. Tests

```
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_models/test_dino/ tests/test_losses/test_dino_loss.py tests/test_train/test_dino/ tests/test_datasets/test_multi_crop.py -q
```

`test_dino_package.py` asserts that every path and import named in **this README** resolves;
`test_training.py` verifies the packed row layout pair-by-pair, a gradient-free teacher
against a student control, and a real `fit()` in which the teacher moves TOWARD the student;
`test_knn_eval.py` carries a RED-proven collapse detector and real `fit()` runs proving the
diagnostic columns reach `CSVLogger`.

---

## 10. References

- Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), ICCV 2021. arXiv:2104.14294
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024. arXiv:2304.07193
- Darcet et al., *Vision Transformers Need Registers*, ICLR 2024. arXiv:2309.16588
- Zhou et al., *iBOT: Image BERT Pre-Training with Online Tokenizer*, ICLR 2022. arXiv:2111.07832
- Siméoni et al., *DINOv3*, 2025. arXiv:2508.10104
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021. arXiv:2104.09864
