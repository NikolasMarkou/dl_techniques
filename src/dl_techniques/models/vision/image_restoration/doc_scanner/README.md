# DocScanner: Robust Document Image Rectification with Progressive Learning

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 port of **DocScanner** (Feng et al., arXiv:2110.14968): a *two-stage*
document-unwarping network that localizes the page, then refines a backward map over
twelve recurrent steps.

The two stages are trained **independently** (paper §4.3). That is not a deployment
detail — it determines the shape of this package. There are three model classes, two
trainers, and a composite that exists only to run inference.

---

## Table of Contents

1. [Overview](#1-overview)
2. [The output contract: a BACKWARD map, and in which units](#2-the-output-contract-a-backward-map-and-in-which-units)
3. [The two stages](#3-the-two-stages)
4. [Architecture](#4-architecture)
5. [Variants — and why there is no `-T` or `-B`](#5-variants--and-why-there-is-no--t-or--b)
6. [Quick start](#6-quick-start)
7. [Constraints you must read before using this](#7-constraints-you-must-read-before-using-this)
8. [Component reference](#8-component-reference)
9. [Configuration](#9-configuration)
10. [Serialization](#10-serialization)
11. [Testing](#11-testing)
12. [Citation](#12-citation)

---

## 1. Overview

A photograph of a curled page is a warped sampling of a flat document. DocScanner
predicts, for every output pixel, *where in the input to read from* — a dense **backward
map** — and unwarping is then one bilinear gather.

The pipeline is `inference.py:17-31` of the upstream release, four lines long:

```
msk, _1, _2, _3, _4, _5, _6 = segmenter(x)   # U2NET-P, seven maps; only the first is used
msk = (msk > 0.5).float()                    # BINARY, not soft
x   = msk * x                                # MULTIPLICATIVE, not concatenated
bm  = rectifier(x, iters=12)                 # absolute pixel coordinates
bm  = (2 * (bm / 286.8) - 1) * 0.99          # calibrated to a normalized grid
```

Two things about that listing carry more weight than their line count suggests, and both
have anchored decisions in `model.py`:

* the mask is applied by **multiplication**, not concatenation — concatenating would
  change the rectifier's input width from 3 to 4, which the feature encoder infers from
  the tensor and would accept silently;
* the threshold is a **hard** `> 0.5`, so the composite is **not differentiable end to
  end**. See §7.1.

Despite the RAFT lineage of stage 2, **there is no correlation volume anywhere in this
architecture**. The update block's `corr` input is a bilinear resample of the encoder's
own 320-channel feature map at the current coordinates. Do not add one.

---

## 2. The output contract: a BACKWARD map, and in which units

| Class | Returns | Units |
|---|---|---|
| `DocScannerSegmenter` | a `list` of **seven** `(B, H, W, 1)` sigmoid maps `[d0, …, d6]` | confidence in `[0, 1]` |
| `DocScannerRectifier`, `training=False` | `(B, H, W, 2)` | **absolute full-resolution pixel coordinates** |
| `DocScannerRectifier`, `training=True` | `(B, 12, H, W, 2)` — the whole refinement sequence | same |
| `DocScanner` | `(B, H, W, 2)`, always | **calibrated** — `[-0.99, +0.99]` *only while the rectifier's raw map stays inside `[0, W-1]`* |

Channel 0 is the **x** (column) coordinate, channel 1 is **y** (row) — the order
`coords_grid` emits, matching the reference. Six of the segmenter's seven maps exist for
deep supervision; `d0` is the fusion and the only one the pipeline consumes.

The rectifier's training-mode output rank differs on purpose: this repo forbids a custom
`train_step`, so the twelve-iteration exponentially-weighted sequence loss can only reach
`compile(loss=...)` if the sequence is a model **output**.

That loss is `dl_techniques.losses.DocScannerFlowSequenceLoss` — it lives in `losses/`, the
repo's home for losses, not in this package. It takes the `(B, 12, H, W, 2)` sequence as
`y_pred` and the 4-channel stack `[f_gt(2), g(2)]` as `y_true`, where `g` is the
ground-truth **forward** map the circle-consistency term needs (paper Eq. 12).

The rectifier's map and the composite's map **are not interchangeable**. One is in pixels
and *should* run to 287 at a 288 input; the other is normalized. See §7.3.

**Neither range is enforced, and an untrained model leaves both.** The rectifier's output
is an unconstrained regression — nothing clamps it to the image domain — and the
calibration `(2 * bm / 286.8 - 1) * 0.99` is affine, so it inherits whatever the rectifier
emits. Measured on a freshly initialized `docscanner-l` at 288×288 (2026-09-10): the
rectifier spans `[-77.59, +298.07]` px and the composite `[-1.434, +1.036]`. The nominal
ranges above are a property of a *converged* model, not a contract of the class. The
downstream sampler is edge-clamped, so an out-of-domain map degrades the gather rather
than producing NaN — which is also why nothing crashes to tell you about it.

---

## 3. The two stages

**Stage 1 — `DocScannerSegmenter`.** A pruned U2-Net (`U2NETP(3, 1)`), six nested-U
encoder stages, five decoder stages, six side-supervision heads and a 1×1 fusion. Every
RSU block is built at the uniform interior/output widths 16/64. **1,136,877 parameters**
(1,131,181 trainable plus 5,696 BatchNorm buffers), verified against an independent
arithmetic transcription of `seg.py`'s own construction rather than against this port.

It imposes **no constraint on input size at all**: the `ceil_mode` pooling remedy and the
resize-onto-the-skip decoder absorb a ragged ladder. At height 37 the encoder levels run
37, 19, 10, 5, 3, 2 and every concatenation still meets at the skip's own size.

**Stage 2 — `DocScannerRectifier`.** A RAFT-lineage progressive refiner: a stride-8
feature encoder, then twelve GRU update steps, each emitting a residual to a
1/8-resolution coordinate field that is convex-upsampled 8× to full resolution.
**7,328,752 parameters** (feature encoder 3,080,080 + update block 4,248,672). One update
block, applied twelve times — the recurrence is in the state, not in the parameters.

**They are trained independently.** `src/train/doc_scanner/` therefore has two *training*
entry points, `train_doc_scanner_segmenter.py` and `train_doc_scanner_rectifier.py`, and
neither needs the composite. It carries two further CLIs that train nothing:
`prepare_doc_scanner_data.py` stages the UVDoc archive, and `stage_uvdoc_samples.py`
densifies renders out of it into the sidecar layout the pipeline reads.

---

## 4. Architecture

```
                    (B, H, W, 3)  page photograph, [0, 1]
                          |
        +-----------------+------------------+
        |            DocScannerSegmenter     |    U2NET-P, 1,136,877 params
        |  RSU7 RSU6 RSU5 RSU4 RSU4F RSU4F   |    six encoder stages
        |  RSU4F RSU4 RSU5 RSU6 RSU7         |    five decoder stages
        |  side1..side6 -> resize -> 1x1     |    fusion
        +-----------------+------------------+
                          |  d0  (B, H, W, 1)
                     (> 0.5)  ->  msk
                          |
              masked = msk * page            MULTIPLICATIVE
                          |
        +-----------------+------------------+
        |           DocScannerRectifier      |    7,328,752 params
        |  fnet: 7x7/2 stem @80 -> 80/160/240|    total stride 8
        |        -> 1x1 -> 320               |
        |  split 320 -> tanh(160) | relu(160)|    net | inp
        |  x12:  detach coords1              |
        |        motion encoder -> SepConvGRU|
        |        -> flow head, mask head      |
        |        coords1 += delta            |
        |        convex_upsample(8x) -> bm_up |
        |        warpfea = sample(fmap1, c1)  |    NOT a cost volume
        +-----------------+------------------+
                          |  (B, H, W, 2)  absolute pixels
             (2 * bm / 286.8 - 1) * 0.99
                          |
                    (B, H, W, 2)  calibrated backward map
```

Every resample in the forward path reproduces PyTorch `F.grid_sample(...,
align_corners=True)` exactly. It is implemented as a **coordinate adapter** over the
repo's existing `layers/spatial_layer.py::interpolate_grid` —
`coord = (pix - (S - 1) / 2) / S`, measured against a hand-computed 5×5 reference — never
as a new sampler.

---

## 5. Variants — and why there is no `-T` or `-B`

`MODEL_VARIANTS` has exactly **one** row on all three classes: `"docscanner-l"`.

| Variant | Segmenter | Rectifier | Total | Source |
|---|---|---|---|---|
| `"docscanner-l"` | 1,136,877 | 7,328,752 | **8,465,629** | the released, executable upstream wiring |

**The paper's Table 2 quotes 8.5M for DocScanner-L.** The 8,465,629 above corroborates
this port's width table to about **±1% only**, and is **not a reproduction of a published
number**: 8.5M is quoted to two significant figures, so *any* total in [8.45M, 8.55M]
rounds to it. It is recorded here as an observation. No test asserts it against the paper.

**There is no `-T` row and no `-B` row, and adding one would be an invention.**

* The paper publishes **no architectural split for `-T` at all**. There is nothing to
  port.
* `-B`'s only published trace is **Table 1, which is incomplete**: it gives the encoder
  column and *nothing* for the motion encoder or the GRU. It also does not match the
  released code.
* The released, executable architecture is **`-L`**. Its widths are read from the CALL
  SITES (`model.py:35-39`, `update.py:88`), never from `update.py:18,36`'s dead
  constructor defaults (`hidden_dim=128, input_dim=192+128`) that nothing ever
  constructs. The checkpoint name `./model_pretrained/DocScanner-L.pth`
  (`inference.py:104`) and the upstream demo agree.

A plausible interpolation between a documented model and an undocumented one is not a
variant; it is a number with a paper's name on it.

---

## 6. Quick start

```python
import numpy as np
from dl_techniques.models.vision.image_restoration.doc_scanner import (
    create_doc_scanner,
    create_doc_scanner_rectifier,
    create_doc_scanner_segmenter,
)

# Inference: the assembled pipeline.
scanner = create_doc_scanner("docscanner-l")     # 8,465,629 params
scanner.build((None, 288, 288, 3))               # H, W must be concrete, %8 == 0

page = np.random.rand(1, 288, 288, 3).astype("float32")   # [0, 1], channels-last RGB
bm = scanner(page, training=False)                        # (1, 288, 288, 2), normalized
```

```python
# Training: a STAGE, never the composite (see §7.1).
segmenter = create_doc_scanner_segmenter("docscanner-l")
rectifier = create_doc_scanner_rectifier("docscanner-l")

sequence = rectifier(page, training=True)        # (1, 12, 288, 288, 2) for the loss
maps = segmenter(page, training=True)            # 7 x (1, 288, 288, 1) for deep supervision
```

Overrides are forwarded to the constructor unchanged; an unknown variant raises
`ValueError` listing the valid keys.

---

## 7. Constraints you must read before using this

### 7.1 The composite CANNOT be trained end to end — by design

`(msk > 0.5)` has zero gradient almost everywhere. No gradient reaches
`DocScannerSegmenter` through `DocScanner`, ever.

This matches the paper, which trains the two modules **independently** (§4.3). It is not
a gap in the port and **must not be "fixed"** with a straight-through estimator, a soft
threshold or a temperature — each of those would invent a mechanism the reference does not
have, and each would train, stay finite and keep every shape. The property is *pinned* by
`TestNoGradientReachesTheSegmenter`, adopted through the shared gradient oracle's
two-sided `expect_zero` waiver: every segmenter weight must be dead **and** every
rectifier weight must be live.

Train each stage separately with `src/train/doc_scanner/`, then assemble.

### 7.2 `286.8` is carried for FIDELITY, not because it is calibrated for this model

`bm = (2 * (bm / 286.8) - 1) * 0.99` is `inference.py:29`, transcribed exactly. Neither
number is explained upstream or in the paper.

`286.8` is an empirical constant of the **released DocScanner-L checkpoint** at its 288
training resolution. **No released checkpoint is ever loaded here** — `pretrained=True`
raises on all three classes (§7.4) — so this constant is carried to stay faithful to the
reference pipeline, *not* because it is calibrated for a from-scratch model. **If you
train this from scratch you may well need a different divisor**, and 288 is the obvious
candidate. Change it deliberately and record that you did; do not let it drift silently.

The calibration is applied **exactly once**, in `DocScanner.call` and nowhere else.
`DocScannerRectifier` deliberately emits uncalibrated absolute pixel coordinates.

### 7.3 The rectifier's height and width must be divisible by 8 — and statically known

The refinement loop runs at 1/8 resolution and convex-upsamples by exactly that factor.
`DocScannerRectifier` and `DocScanner` **raise a `ValueError` naming the offending
extent** rather than padding internally: padding is a property of the inference contract,
and whoever pads must also un-pad the prediction.

They additionally require **concrete** extents — unlike the sibling `doc_res`, they cannot
be built at `(None, None, None, 3)`, because they materialize an identity coordinate field
with an `arange` per axis. Build at `(None, 288, 288, 3)`.

`DocScannerSegmenter` has neither constraint (§3).

### 7.4 `pretrained=True` raises `NotImplementedError` on all three classes

No trained weights ship with this package and none are downloadable. The two stages refuse
for two *different* reasons, and neither is repairable here.

* **Segmenter**: the checkpoint **does not exist in the reference checkout**.
  `inference.py:103` loads `./model_pretrained/seg.pth` and no such file is present — only
  an external download link in the upstream README. That absence is also why the one clue
  to its key layout (`inference.py:40` strips a six-character key prefix) cannot be acted
  on: `reload_seg_model` *filters* non-matching keys rather than raising, so a conversion
  script written against a guess would fail silently for every key it mismatched.
* **Rectifier**: even given the file, this port's conventions differ structurally, so a
  transferred checkpoint would be **silently wrong** rather than merely untested. See
  §7.5.

### 7.5 Five measured divergences from the PyTorch reference

Each is anchored at its site in the source with the decision that settled it.

| # | Divergence | Why |
|---|---|---|
| 1 | The feature encoder's `norm1` is built at **80** channels, not the reference's literal 64 (`extractor.py:93`) | The reference declares 64 and applies it to an 80-channel tensor. That is inert under torch's `InstanceNorm2d(affine=False)`, which allocates no parameters — a Keras affine norm allocates by shape and would crash. 80 is the true width; 64 is a latent upstream bug. (`D-011`) |
| 2 | Every **stride-2** convolution pads explicitly (`ZeroPadding2D` then `padding="valid"`); stride-1 3×3 stays on `"same"` | Keras `"same"` is *not* torch's symmetric `padding=k//2` at stride 2 — it biases the sampling grid by one pixel on one side. At stride 1 the two agree exactly. (`D-012`) |
| 3 | Instance norm is a direct `keras.layers.GroupNormalization(groups=C, epsilon=1e-5, center=False, scale=False)` | The repo's normalization factory has neither instance nor group norm among its 18 keys. Keras' own GroupNorm default epsilon is `1e-3` — a silent 100× against torch's `1e-5` — so it is passed explicitly at every site, and `affine=False` is reproduced as `center=False, scale=False`. (`D-011`) |
| 4 | The convex upsample's `extract_patches` is replaced by `keras.ops.conv` against a one-hot gather kernel | That *is* `_extract_patches`' own body, inlined. Keras 3.8's public `extract_patches` mis-handles the symbolic path this model reaches through `build()`; the substitution is bit-identical eagerly. (`D-018`) |
| 5 | `sample_at_pixel_coords` **edge-clamps** out-of-range queries; `F.grid_sample(img, grid, align_corners=True)` (`DocScanner/model.py:18`) takes torch's default `padding_mode='zeros'` | Measured on a 5×5 ramp holding `5y + x`: `(x=-3, y=1) → 5.0`, `(x=7, y=1) → 9.0`, `(x=2, y=-2) → 2.0`, `(x=3, y=9) → 23.0`, `(x=-0.5, y=1) → 5.0` — torch returns `0.0` or a zero-faded value for each. This is **not** a dormant edge case: on a freshly built `docscanner-l` at 288×288, **5.1%–7.3%** of `warpfea` queries land outside the 36×36 feature map at iteration 0 and **18.6%–32.2%** by iteration 11 (seeds 0, 1, 2). Kept because zero-padding an out-of-domain query is not more correct than clamping it — a zero feature is a *confident* statement about content that is simply absent — and because it is the graceful-degradation behaviour the composite's full-resolution warp relies on. (`D-056`) |

Consequence: **train from scratch, or restore your own checkpoint with
`model.load_weights(path)`.** Do not write a weight-transfer script against this port
without first solving all five.

### 7.6 The published DocUNet-benchmark MS-SSIM / LD numbers are NOT reproducible here

**This is a limitation of the port, stated flatly, not a hedge.** The upstream README's
table gives DocScanner-L **MS-SSIM 0.5178 / LD 7.45** on the DocUNet benchmark. Nothing in
this package can produce those numbers, and no future run of it will, for four independent
structural reasons — each of which alone is sufficient:

1. **No released weights are loadable.** `pretrained=True` raises on all three classes
   (§7.4). Every figure would come from a from-scratch run of this re-implementation.
2. **A different training corpus.** The paper trains on **Doc3D** (549 GB,
   registration-gated behind a contact-info agreement, not obtainable by script). This
   port trains on **UVDoc** (27.5 GB, openly downloadable, no auth) plus the synthetic
   generator in `dl_techniques.datasets.document_rectification`.
3. **A different corpus scale.** Doc3D is **100,000 renders**. UVDoc is **20,000 renders**
   — and those 20,000 are drawn from only **4,032 distinct geometries** (five renders per
   geometry, measured over the whole archive), so the *deformation* diversity is smaller
   again than the 5× render-count ratio suggests.
4. **The benchmark itself is not staged on this machine.** There is no DocUNet benchmark
   under the dataset root; `doc_res/dewarping/{doc3d,dir300}/` are README-only stubs. The
   score could not be *measured* here even with a converged model.

And if you do stage the benchmark, one thing to check before scoring anything:

> **Two DocUNet samples are known-bad.** The upstream README records that `64_1.png` and
> `64_2.png` are **rotated by 180 degrees** relative to their ground-truth documents, that
> most published work scores them anyway, and that "the performances in most of the
> existing work are computed with these two ***mistaken*** samples". A number computed with
> them and a number computed without them are not comparable, and the published 0.5178 /
> 7.45 is of the *with* kind.

**What this package does demonstrate** is the architecture, end to end and under load. The
rectifier has been trained for real against densified UVDoc backward maps
(`results/doc_scanner_rectifier_docscanner-l_20260910_203552`): four epochs, training loss
390.96 → 200.17 → 176.28 → 159.92, validation 52.90 → 35.94 → 31.35 → 32.71, read from that
run's `training_log.csv` and not from the progress bar, which under-reports by a smoothed
running mean. That is a *learning curve*, not a benchmark, and the two validation numbers
are not comparable to the training ones (the sequence loss scores all twelve iterations at
training time and only the last at validation time — see the trainer README).

The deliverable of this package is a working, verified architecture, not a paper-matching
score. Any MS-SSIM/LD figure measured from a model trained here is a figure about *that*
training run and must be reported as such.

---

## 8. Component reference

| Name | Module | Role |
|---|---|---|
| `DocScanner` | `.model` | The composite. Segment → threshold → mask → rectify → calibrate. Inference only. **Exported.** |
| `DocScannerSegmenter` | `.model` | Stage 1, U2NET-P. **Exported.** |
| `DocScannerRectifier` | `.model` | Stage 2, the progressive refiner. **Exported.** |
| `create_doc_scanner` / `_segmenter` / `_rectifier` | `.model` | Module-level factories; each delegates to its class's `from_variant`. **Exported.** |
| `coords_grid`, `sample_at_pixel_coords`, `convex_upsample` | `.warp` | The port's one sampling convention: the `align_corners` adapter, the identity field, the RAFT 8× learned-convex upsample. Not exported. |
| `DocScannerFeatureEncoder`, `DocScannerResidualBlock` | `.components` | The stride-8 encoder and its residual unit. Not exported. |
| `SepConvGRU`, `DocScannerMotionEncoder`, `FlowHead`, `DocScannerUpdateBlock` | `.components` | The recurrent update core. Not exported. |
| `REBNCONV`, `RSU7`, `RSU6`, `RSU5`, `RSU4`, `RSU4F` | `.u2net_blocks` | The nested-U building blocks of stage 1. Not exported. |
| `_VARIANT_SPEC` | `.components` | The port's **single** width table. Every channel count in the package is read from it; no width literal lives anywhere else. Not exported. |

The unexported names are assembly parts, not a public surface. Import them from their
module if you are testing them.

Three upstream things are deliberately **not** ported because they are dead code in the
reference: `sobel_net` (`seg.py:8-31`), `BottleneckBlock`, and the plain `ConvGRU`.

---

## 9. Configuration

`DocScanner` takes two dicts, one per stage; they are the stages' own constructor keyword
arguments and are *replaced*, never merged, so a partial override cannot mix two variants'
widths.

Neither stage constructor has width defaults. That is deliberate and structural: upstream
carries dead constructor defaults that no call site reaches, and a signature with no
defaults cannot be misread that way. Use `from_variant` or a `create_*` factory.

| Class | Argument | `docscanner-l` | Notes |
|---|---|---|---|
| Segmenter | `mid_channels` | `16` | Every RSU block's interior width |
| | `out_channels` | `64` | Every RSU block's output, hence every skip |
| | `output_channels` | `1` | One confidence plane |
| Rectifier | `hidden_dim` / `context_dim` | `160` / `160` | The two halves of the encoder's output |
| | `fnet_output_dim` | `320` | Must equal `hidden_dim + context_dim`; asserted |
| | `gru_input_dim` | `320` | Must equal `context_dim + motion_output_dim`; asserted |
| | `encoder_stem_channels` | `80` | **Not 64** — see §7.5 |
| | `encoder_stage_channels` | `(80, 160, 240)` | Three stages, total stride 8 |
| | `motion_corr_hidden` / `_out` | `240` / `160` | The warped-feature branch |
| | `motion_flow_hidden` / `_out` | `160` / `80` | The flow branch |
| | `motion_output_dim` | `160` | `concat(out, flow)` |
| | `flow_head_hidden` | `320` | 160 → 320 → 2, no output activation |
| | `mask_head_hidden` | `288` | 160 → 288 → 576, scaled by 0.25 at the call site |
| | `iters` | `12` | Value knob: one block applied twelve times |

Module constants with an upstream citation each, in `.components`:
`INSTANCE_NORM_EPSILON = 1e-5`, `SPATIAL_DIVISOR = 8`, `REFINE_ITERATIONS = 12`,
`SEQUENCE_LOSS_GAMMA = 0.85` and `LINE_LOSS_WEIGHT = 0.5` (both the paper, not the code —
the release ships no training loop), `SEG_MASK_THRESHOLD = 0.5`, `BM_CALIBRATION_DIVISOR = 286.8`,
`BM_CALIBRATION_SCALE = 0.99`, `MASK_LOGIT_SCALE = 0.25`, `CONVEX_NEIGHBOURS = 9`,
`FLOW_CHANNELS = 2`, `SEG_OUTPUT_CHANNELS = 1`.

---

## 10. Serialization

Standard Keras 3. `get_config()` round-trips every constructor argument, and every class
calls `materialize_sublayers` from `build()` — without it a subclassed model is marked
built while its sub-layers are not, and a `.keras` reload restores nothing.

All three classes register under `dl_techniques.models.doc_scanner.model`; the components
under `dl_techniques.models.doc_scanner.components` and
`dl_techniques.models.doc_scanner.u2net_blocks`. The key strips **both** the `vision`
family and the `image_restoration` subfamily — it is not the import path.

```python
model.save("doc_scanner.keras")
restored = keras.models.load_model("doc_scanner.keras")
```

---

## 11. Testing

```bash
.venv/bin/python -m pytest tests/test_models/test_doc_scanner/ -q
```

The suite is written against the defect class this architecture actually has: **every
plausible error in it is shape-preserving**. A transposed convex-upsample reshape, a
swapped GRU pass order, a chained rather than re-sampled `warpfea`, a dropped
`stop_gradient`, a soft mask, a doubly-applied calibration — each of those still returns
finite tensors of the right shape, still saves, still reloads, still trains its loss down.
So the guards are behavioural and each one has been proven RED against an in-place
mutation of the tracked source:

* the `align_corners` adapter against a hand-computed 5×5 reference, with the half-pixel
  variant proven to fail it;
* the convex upsample's neighbour ordering by delta impulse, plus a uniform-mask control;
* the `ceil_mode` pooling remedy, measured two ways rather than assumed;
* the mask is multiplicative and binary; the calibration is applied exactly once; no
  gradient reaches the segmenter;
* round-trip on **values**, weights-restored-before-first-call, build parity, gradient
  flow after a real optimizer step, a falsifiable smoke contract, and knob sensitivity —
  all through the shared oracles under `tests/test_models/*_oracle.py`, none
  re-implemented locally.

---

## 12. Citation

```bibtex
@article{feng2021docscanner,
  title   = {DocScanner: Robust Document Image Rectification with Progressive Learning},
  author  = {Feng, Hao and Zhou, Wengang and Deng, Jiajun and Tian, Qi and Li, Houqiang},
  journal = {arXiv preprint arXiv:2110.14968},
  year    = {2021}
}

@inproceedings{teed2020raft,
  title     = {{RAFT}: Recurrent All-Pairs Field Transforms for Optical Flow},
  author    = {Teed, Zachary and Deng, Jia},
  booktitle = {ECCV},
  year      = {2020}
}

@article{qin2020u2net,
  title   = {{U2-Net}: Going Deeper with Nested U-Structure for Salient Object Detection},
  author  = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood
             and Zaiane, Osmar R. and Jagersand, Martin},
  journal = {Pattern Recognition},
  year    = {2020}
}
```

Upstream reference implementation: <https://github.com/fh2019ustc/DocScanner>.
