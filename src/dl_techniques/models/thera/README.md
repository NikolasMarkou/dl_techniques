# THERA in `dl_techniques` — JAX→Keras Port

**What fits, what doesn't, what changed.**

This document is the user-facing record of porting **THERA** (aliasing-free
arbitrary-scale super-resolution with neural heat fields) from its reference
JAX/Flax implementation into `dl_techniques` (Keras 3 / TF 2.18). It is a *port*,
not a wrapper: every math operation was reimplemented. Decision IDs (`D-NNN`)
reference `plans/plan_2026-06-11_f662207d/decisions.md`.

---

## 1. Overview

**THERA** performs arbitrary-scale single-image super-resolution. A feature
backbone encodes the LR image; a hypernetwork emits a *per-pixel* parameter field;
this field parameterizes a SIREN-style **neural heat field** whose Gaussian
heat-kernel envelope is governed by a scale-tied diffusion time `t`. Sampling the
field at arbitrary continuous query coordinates yields an SR image at any scale.
The heat-kernel envelope is what makes the result **aliasing-free**: higher target
scales (`t = scale**-2` smaller) widen the kernel, band-limiting the field. An
exact spatial-Jacobian total-variation term regularizes for aliasing during
training.

### The 6-variant matrix

`{edsr-baseline, rdn} × {air, plus, pro}` — backbone × refiner tail. All six are
genuine architectural configs (not aliases): `air` uses `hidden_dim=32`, `plus`
and `pro` use `hidden_dim=512` (D-009).

| | air (identity tail) | plus (ConvNeXt tail) | pro (SwinIR/RSTB tail) |
|---|---|---|---|
| **edsr-baseline** | `edsr-air` | `edsr-plus` | `edsr-pro` |
| **rdn** | `rdn-air` | `rdn-plus` | `rdn-pro` |

### Package map

| File | Role |
|---|---|
| `initializers/linear_up_initializer.py` | `LinearUpInitializer`: 2D-disk frequency init `r = scale·√U[0,1]` for the heat-field components. |
| `layers/grid_sample.py` | TF-native `make_grid` (pixel-center grid) + `interpolate_grid` (gather+lerp sampler, order 0/1). |
| `layers/thera_heat_field.py` | `ThermalActivation` (`sin(w0·x+φ)·exp(-(w0·‖·‖)²·k·t)`) + `HeatField` (per-pixel einsum field). |
| `models/thera/edsr_backbone.py` | `EDSRBackbone` — no-upsampling EDSR feature extractor. |
| `models/thera/rdn_backbone.py` | `RDNBackbone` — residual dense feature extractor. |
| `models/thera/tails.py` | `air`/`plus`/`pro` tail builders (`build_thera_tail`). |
| `models/thera/hypernetwork.py` | `TheraHypernetwork` — 1×1 phi conv → einsum decode (+ `decode_with_jac`). |
| `models/thera/model.py` | `Thera` model + `build_thera` + `Thera.from_variant`. |
| `losses/thera_jacobian_tv.py` | `thera_tv_penalty` / `thera_total_loss` — exact Jacobian-TV aliasing penalty. |
| `train/thera/data.py` | `build_arbitrary_scale_dataset` — pure `tf.data` arbitrary-scale SR pipeline. |
| `train/thera/train_thera.py` | `TheraTrainingModel` + trainer (nested-tape `train_step`). |
| `train/thera/eval_thera.py` | `super_resolve` + `evaluate_multiscale` — inference + multi-scale benchmark. |

---

## 2. What was REUSED from `dl_techniques`

The port deliberately reuses verified existing components rather than
reimplementing them:

- **`SwinTransformerBlock`** (`layers/transformers/swin_transformer_block.py`) +
  **`WindowAttention`** (`layers/attention/window_attention.py`, relative-position
  bias built in) → assembled into the RSTB body of the `pro` tail.
- **`ConvNextV1Block`** (`layers/convnext_v1_block.py`) → the `plus` tail.
- **`CharbonnierLoss`** (`losses/image_restoration_loss.py`) → optional
  reconstruction loss.
- **`PsnrMetric`** (`metrics/psnr_metric.py`), **`SsimMetric`**
  (`metrics/ssim_metric.py`) → training/eval metrics.
- **`train/common`** scaffold (`setup_gpu`, `set_seeds`, `create_callbacks`,
  `save_config_json`, `EpochMetricsPlotCallback`, `collect_image_paths`).

What was **NOT** reused: `SpatialLayer` (`layers/spatial_layer.py`) was considered
for `make_grid` but rejected — it samples linspace *endpoints* and z-score
normalizes, whereas THERA needs the un-normalized pixel-*center* grid (see §3,
the corrected finding).

---

## 3. What did NOT fit / what needed changing (the core deliverable)

Each JAX→Keras divergence, with the reason.

### G1 — `pmap` / `n_devices` multi-device → single-GPU `model.fit`
THERA replicates over devices with `jax.pmap` and threads `n_devices` through the
data and step. **Dropped entirely.** The port is single-GPU `model.fit` with a
custom `train_step`. (INV-6)

### G2 — `jax.tree_util` hypernetwork param-tree → explicit `ops.split`/reshape
The Flax hypernetwork emits a flat `phi` vector that a `jax.tree_util` param-tree
unflattens into per-pixel field weights. Keras has no param-tree. The 1×1-conv
output is **explicitly split/reshaped**: `phi_phase = phi[..., :hidden]`,
`phi_kernel = reshape(phi[..., hidden:], (..., hidden, out))`. Because this is
train-from-scratch (no weight port), the split layout is *our* internally-consistent
choice — there is no JAX layout to match. Crucially, the per-pixel field weights
(phase, kernel) are **`call` INPUTS** produced by the hypernetwork, NOT module
weights; the `HeatField` owns only the global shared `components` and scalar `k`.
(D-004 / D-008, INV-5)

### G3 — `repeat_vmap` over `(params, coords, t)` → batched `einsum`
THERA `vmap`s the field evaluation over every query pixel and threads a nested
param tree. Keras has no `vmap`. This becomes a single **batched einsum** over the
leading `(B, Hq, Wq)` dims (D-004, INV-5):

```
einsum('...c,ck->...k')   # coord projection (shared components)
einsum('...k,...ko->...o')  # per-pixel output dense (per-pixel kernel)
```

No vmap, no param-tree, no `jax.tree_util`. **Surprise:** `t` arrives with only a
leading batch dim `(B, 1)` and must be rank-aligned (singleton spatial axes
inserted) before the heat envelope can broadcast against `(B, Hq, Wq, hidden)`
(handled in `ThermalActivation.call`).

### G4 — Orbax/pickle `.pkl` checkpoints → `.keras` save (no weight port)
THERA ships Orbax/pickle checkpoints. The port saves `.keras`. **The pretrained-
weight port is deliberately OUT of scope** (Q2): all variants are *train-from-
scratch*. The deployable artifact is the inner `Thera` saved as `thera_model.keras`
(NOT the `TheraTrainingModel` wrapper, D-012).

### G5 — `chunkax` inference tiling → not ported
THERA tiles large-image inference with `chunkax`. **Not ported.** If a future
large-image inference path OOMs, a plain-Python coord-axis tiling loop is the
documented fallback (INV-6 allows it for inference only).

### H4 — no Keras `grid_sample`/`map_coordinates` → TF-native `interpolate_grid`
Keras 3 has no `grid_sample`. `layers/grid_sample.py` provides a **TF-native
gather+lerp sampler** (`interpolate_grid`, order 0 = nearest, order 1 =
differentiable bilinear) and a faithful **`make_grid`** (D-003). The coordinate
convention was source-verified against the reference `model/utils.py`:
pixel-center normalized grid `linspace(-0.5+1/(2n), 0.5-1/(2n), n)`, output
`(h, w, 2)` with channel order `[h, w]`, `indexing='ij'`; coord→pixel map
`pix = coord·size + (size-1)/2`; border `mode='nearest'` = clamp to `[0, size-1]`.

> **Corrected finding:** an earlier finding claimed `SpatialLayer` covers
> `make_grid`. It does **not** — `SpatialLayer` uses linspace *endpoints* and
> z-score normalizes the channels. THERA needs the un-normalized pixel-center
> grid, so `make_grid` is reimplemented faithfully (verified vs scipy).

These are pure functions (not a Layer): they are stateless, and the Jacobian-TV
helper must call `interpolate_grid` *bare* inside a nested tape (D-003).

### H5 — `jax.jacrev` analytic Jacobian → nested `tf.GradientTape.batch_jacobian`
THERA's aliasing-TV regularizer uses `jax.jacrev` for the exact per-pixel spatial
Jacobian `d(field)/d(rel_coords)` evaluated at `t=0`. Keras has no `jacrev`. The
port computes it with a **nested `tf.GradientTape.batch_jacobian`** over a
flattened pixel axis `N = B·Hq·Wq`, with `experimental_use_pfor=False` and
`persistent=True` (D-010). This is **EXACT, not finite-difference** (Q3 forbids
finite-difference). Two subtleties:

- pfor vectorization (`batch_jacobian` default) does **not** compose with a second
  outer tape → `None` weight-grads. `experimental_use_pfor=False` builds an
  unrolled per-output loop the outer tape *can* differentiate — still exact.
- TF requires `persistent=True` when `experimental_use_pfor=False` in eager mode.

STOP-IF #1 (the highest-risk falsifier) was cleared: the outer-tape weight-grad
oracle confirms non-`None`, finite, non-zero grads to both `heat_field.components`
and the hypernetwork `out_conv.kernel`, and the nested tape composes under
`model.fit`'s `tf.function` graph mode with no `run_eagerly` needed (D-012).

### The differentiability nuance
THERA's `interpolate_grid` is **order=0 NEAREST**. The TV coordinate-Jacobian does
**not** flow through the sampler — it flows through the *direct* `coords` term of
`rel = coords − nearest(coords)` (nearest is piecewise-constant, zero-gradient
a.e.) into the heat field. Therefore the sampler need not be differentiable for
THERA's exact-Jacobian deliverable; a fully-differentiable order=1 path exists for
future use but is not on the TV path (corrected/clarified finding, D-003 / D-008).

### EDSR `res_scale` quirk
THERA's reference EDSR block stores but **never applies** `res_scale` (a
jax-enhance quirk): it returns `x + body(x)`. The port implements
`x + res_scale·body(x)` with **default `res_scale=1.0`** — numerically identical to
THERA at 1.0, and a caller may pass `0.1` to recover canonical EDSR (D-005). A
faithful superset, not a deviation.

### SwinIR RSTB assembly
The `pro` tail's RSTB is **assembled from the existing NHWC `SwinTransformerBlock`**
— no PatchEmbed/Unembed needed (the repo block is already NHWC). For non-window-
divisible / non-square inputs, the tail **reflect-pads to the next window-size
multiple then crops back** to the exact original `H, W` (D-007), rather than
asserting divisibility (which would forbid the arbitrary crop sizes the
hypernetwork emits).

### Keras-3 gotcha: nested layer lists break `.keras` reload
A nested `List[List[Layer]]` **is** tracked for `trainable_weights` (Keras
flattens nesting for variable collection) but its inner layers' weights are **NOT
restored** on `.keras` reload → a 100% output mismatch that is masked if you only
spot-check init=ones/zeros weights. All sublayers are stored in **flat** attribute
lists, with stage boundaries recovered by offset-slicing (iter-1 lesson, D-007 /
D-009). The round-trip test compares a *forward output*, not just weight shapes.

### Trainer: standardize / denorm / `+source_nearest` live in the trainer
The `Thera` model emits a **raw residual field** (D-009). THERA's standardize
(`(x−MEAN)/√VAR`), denorm (`out·√VAR+MEAN`), and `+ nearest-upsampled source`
residual add all live in the **trainer/eval**, not the model — the trainer owns
the channel statistics and the upsampled-source term; baking them into the model
would double-apply them (D-012). The trainer uses manual
`tf.clip_by_global_norm` (NOT the optimizer's `global_clipnorm`, to avoid double-
clipping) and `jit_compile=False` (the nested persistent tape + dynamic query
shapes do not trace under XLA).

---

## 4. Variants & usage

### Build
```python
from dl_techniques.models.thera import build_thera, Thera

model = build_thera(out_dim=3, backbone="edsr-baseline", size="pro")
# or by named variant:
model = Thera.from_variant("edsr-pro")
```

### Train (single GPU, headless)
```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.thera.train_thera \
    --data-dir /path/to/DIV2K --backbone edsr-baseline --size pro \
    --epochs 100 --batch-size 16
```
The deployable inner `Thera` is saved as `results/<exp>_<ts>/thera_model.keras`.

### Evaluate (multi-scale ×2/×3/×4)
```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.thera.eval_thera \
    --checkpoint results/<exp>_<ts>/thera_model.keras \
    --data-dir /path/to/benchmark/HR --scales 2,3,4
```
`eval_thera.super_resolve(model, lr_01, target_hw)` and
`eval_thera.evaluate_multiscale(model, paths, scales=(2,3,4))` are the programmatic
entry points; the benchmark crops an `s`-pixel border and reports PSNR/SSIM for
both THERA and a bicubic baseline.

---

## 5. Known limitations

- **Train-from-scratch only.** No JAX→Keras weight port (G4 / Q2). Numbers depend
  on the training corpus and budget, not on porting paper-SOTA weights.
- **Beating bicubic needs real training.** SC-12 (THERA > bicubic baseline) is a
  property of a *trained* model. The shipped unit/smoke tests assert only shape,
  finiteness, and round-trip — an untrained model will not beat bicubic, and the
  eval tests intentionally do **not** assert quality.
- **`pro`/`plus` are memory-heavy** (`hidden_dim=512`). The per-pixel field + the
  nested-tape Jacobian are memory-intensive; use small crops/batch and prefer the
  24GB GPU for these variants. `air` (`hidden_dim=32`) is the light variant.
- **`pro` RSTB shift-mask caveat.** The reused `SwinTransformerBlock` builds its
  shift mask **dynamically at call time**, whereas the reference builds it at
  build-time from a fixed `img_size`. For window-divisible inputs these are
  functionally equivalent; non-divisible inputs are handled by pad-to-multiple +
  crop (D-007), but the dynamic-mask path has not been bit-exactly cross-validated
  against the reference build-time mask.
- **No `chunkax` tiling** (G5): very large single-image inference may OOM; the
  documented fallback is a plain-Python coord-axis tiling loop (not yet
  implemented).
