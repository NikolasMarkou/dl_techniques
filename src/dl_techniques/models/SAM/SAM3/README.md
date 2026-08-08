# SAM 3 (Segment Anything with Concepts) — Keras 3 Implementation

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of the **SAM 3** phase-1 image path, after "SAM 3:
Segment Anything with Concepts" (Ravi et al., 2025): a text-prompted,
**open-vocabulary** detector and
segmenter — no class table, no softmax over categories, the prompt *is* the
class.

> **Scope, stated up front.** This package ships the **architecture of SAM 3's
> phase-1 image path and a training wrapper** — and nothing else.
> - **No pretrained checkpoint ships here or is downloaded**, and **no released
>   Meta SAM 3 checkpoint has ever been loaded in this repository.** No
>   key-mapping layer exists.
> - **It makes no learnability claim, no accuracy claim and no
>   segmentation-quality claim.** Nothing here has been trained to any quality.
>   §8 records a measured result that is decisive about how any future accuracy
>   number on this repository's synthetic generator must be read — read it
>   before quoting one.
> - **Phase 1 is not all of SAM 3.** The out-of-scope list in §9 is named
>   rather than left to be rediscovered; the largest single structural
>   divergence is that the reference's vision-language **early-fusion encoder**
>   between neck and decoder is not built here.
> - **This package was reimplemented, never transliterated**, because
>   upstream's licence forbids the alternative. See §11 — that constraint binds
>   any future extension of this package too.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Component Reference](#4-component-reference)
5. [Variants and Measured Parameter Geometries](#5-variants-and-measured-parameter-geometries)
6. [Training](#6-training)
7. [Query Selection — an opt-in, non-reference head](#7-query-selection--an-opt-in-non-reference-head)
8. [The Baseline That Beats Every Trained Arm](#8-the-baseline-that-beats-every-trained-arm)
9. [Out of Scope in Phase 1](#9-out-of-scope-in-phase-1)
10. [Serialization and Checkpoints](#10-serialization-and-checkpoints)
11. [Licensing — read this before extending](#11-licensing--read-this-before-extending)
12. [Testing](#12-testing)
13. [Citation](#13-citation)

---

## 1. Overview

SAM 1 and SAM 2 segment what a *geometric* prompt points at. SAM 3 segments
what a *phrase* names: it takes an image plus tokenized text and returns, per
decoder query, a class score, a box and a mask. There is no vocabulary — swap
the prompt and you have swapped the "class".

Nine independently constructible, serializable layer classes, plus two
`keras.Model`s — the `Sam3Image` assembly, which wires six of the nine together,
and the `Sam3TrainingModel` wrapper:

| Module | Public classes | Role |
|---|---|---|
| [`vitdet.py`](vitdet.py) | `Sam3ViTDetBackbone`, `Sam3ViTDetBlock` | plain-ViT trunk, mostly window-local attention, **one** feature map out |
| [`necks.py`](necks.py) | `Sam3DualViTDetNeck` | SimpleFPN: that one map → four scales, two independently-weighted copies |
| [`text_encoder_ve.py`](text_encoder_ve.py) | `Sam3TextEncoder` | CLIP text tower, causal, **per-token** output (no pooling here) |
| [`decoder.py`](decoder.py) | `Sam3TransformerDecoder`, `Sam3DecoderLayer` | DETR decoder with **three** attention sub-blocks, boxRPB, presence token |
| [`model_misc.py`](model_misc.py) | `Sam3DotProductScoring` | open-vocabulary per-query class logit |
| [`maskformer_segmentation.py`](maskformer_segmentation.py) | `Sam3SegmentationHead` | MaskFormer head, one mask per query |
| [`query_selection.py`](query_selection.py) | `Sam3EncoderQuerySelection` | this package's OWN opt-in proposal head, default OFF (§7) |
| [`sam3_image.py`](sam3_image.py) | `Sam3Image` | the assembly; owns no learned weights of its own |
| [`training_model.py`](training_model.py) | `Sam3TrainingModel`, `compile_sam3_trainer` | one packed supervision tensor for one joint loss |

Every module carries a full docstring with its own measured-caveats block. This
README points at those rather than restating them, so each fact has one home.
The loss and the Hungarian matcher live outside this package, in
`dl_techniques.losses.sam3_detection_loss`, which is also the single home of
the packed tensor's channel layout.

## 2. Architecture

```
image (B, H, W, 3)                          token_ids (B, seq)
  │                                            │
  ▼                                            ▼
Sam3ViTDetBackbone ──── ONE feature map    Sam3TextEncoder ──── causal CLIP tower
  │                     (B, g, g, dim)         │               per-token, no pooling
  ▼                                            ▼  prompt (B, seq, d_model)
Sam3DualViTDetNeck ──── 4x / 2x / 1x / 0.5x + a per-branch sine PE
  │                     two weight sets: one for the detector, one for the
  │                     SAM-2-style tracker — ONE backbone, never two
  ├── the 1x level, flattened ──► image memory (+ its PE)
  │                                            │
  ▼                                            ▼
Sam3TransformerDecoder ◄─────────────────── prompt (text cross-attention)
  │   per layer: self-attn → text cross-attn → image cross-attn (+boxRPB) → FFN
  │   a presence token rides the query sequence and is split back off
  ▼
hidden states, per-layer reference boxes, presence logits
  │
  ├─► Sam3DotProductScoring ──► pred_logits   (open vocabulary)
  ├─► box head (re-applied HERE, not in the decoder) ──► pred_boxes
  └─► Sam3SegmentationHead ──► pred_masks, semantic_seg
```

Two structural notes that are easy to get backwards:

- **The final box is produced by `Sam3Image`, not by the decoder.** The decoder
  returns the anchor each layer *consumed*, so the shared box head is
  re-applied to the stacked hidden states rather than read out of the stack.
  `call_per_layer` is public and does this correctly; calling the decoder
  directly is not the supported route to per-layer boxes.
- **There is exactly ONE presence signal**, the decoder's presence token. The
  segmentation head has no presence mechanism of any kind — not disabled, not
  built and left unused: absent. That matches the shipped reference
  configuration, which switches its own presence head off.

## 3. Quick Start

```python
from dl_techniques.models.SAM.SAM3 import Sam3Image

model = Sam3Image.from_variant("tiny")     # RANDOM weights — no checkpoint ships

outputs = model({"image": images, "token_ids": ids}, training=False)
outputs["pred_logits"]   # (B, num_queries)      open-vocabulary class score
outputs["pred_boxes"]    # (B, num_queries, 4)   cxcywh, normalized
outputs["pred_masks"]    # (B, num_queries, h, w)
```

> **Pass `training=False` explicitly on the `sam3` variant.** It carries the
> reference's `drop_path_rate=0.1`, and the shared `StochasticDepth` layer
> short-circuits on `training is False` **only** — so the `training=None` that
> a plain `model(inputs)` passes down **drops paths**. At `training=None` a
> *correct* `.keras` round trip measures value deltas of 0.2-2.2, which look
> exactly like reinitialized weights. `tiny` uses `drop_path_rate=0.0` and is
> unaffected.

## 4. Component Reference

Several mechanisms here are **silent when ported wrong** — the layer builds,
forward-passes, trains and serializes either way, with no shape error. Each is
stated at its own class, most of them next to a `# DECISION` anchor recording
the measurement, and guarded behaviourally in the matching test module:

| Where | What is silent if wrong |
|---|---|
| [`vitdet.py`](vitdet.py) | the absolute position embedding is TILED, not interpolated; tiling is a literal `tile` + `crop`, and at the settled `72 / 24` geometry four tiles are produced and the fourth is discarded; MLP hidden width is `int(dim * mlp_ratio)`, a truncation |
| [`necks.py`](necks.py) | the branch convs carry **no normalization of any kind** (faithful to the SAM 3.0 dual neck); per-scale encodings are not shared; the transpose check is on the OUTPUT, because on a square grid whose side equals `d_model` a forgotten transpose broadcasts silently |
| [`decoder.py`](decoder.py) | boxRPB is a real-valued additive bias into RAW scores; `k` and `v` are not drawn from the same tensor at two of the three sites; the presence token has a zeroed query position and an all-zero bias row; the reference SHARES one pair of boxRPB MLPs across every layer |
| [`text_encoder_ve.py`](text_encoder_ve.py) | the wrapped encoder is bidirectional by default and builds no causal mask for you |
| [`model_misc.py`](model_misc.py) | two independent projections; a MASKED pool with its divisor floored at one; a clamp bound that is a different number from the decoder's and must NOT be unified |
| [`maskformer_segmentation.py`](maskformer_segmentation.py) | the prompt cross-attend is a residual, not a replacement, and folds back into the COARSEST pyramid level |

### Two accepted divergences that bind any future weight transfer

Phase 1 loads no pretrained weights, so nothing is wrong today. Both are
enumerated with their signed parameter arithmetic in
[`text_encoder_ve.py`](text_encoder_ve.py)'s docstring, and the headline is
that this text tower is the reference's **plus** one unmatched normalization
and **minus** one missing projection.

A third, in [`necks.py`](necks.py): the sine positional encoding **omits the
reference's half-pixel centre offset**, a constant angular shift of `pi / H`.
It is largest at the coarsest level and is measured per level there.

Each of these was measured at the **settled** width, not at a toy width — the
same probe run at this package's `tiny` width understates one of them by 2.2x,
which is exactly why the docstring quotes both readings with the width beside
each.

## 5. Variants and Measured Parameter Geometries

`Sam3Image.from_variant` takes three keys. **Only `sam3` is a published size**;
`small` and `tiny` are this repository's own development geometries and
correspond to nothing upstream.

| Variant | What it is | Trainable parameters, measured from random init | Where the figure is held |
|---|---|---:|---|
| `tiny` | the test/CI geometry | **24,818** | MEASURED, not asserted anywhere: `Sam3Image.from_variant("tiny").build(None)` then `count_params()`, on CPU at this commit |
| `small` | a development geometry | **5,881,614** | `SMALL_TOTAL` in `tests/test_models/test_sam3/test_model.py` |
| `sam3` | the released configuration | **821,708,598** | `SHIPPED_TOTAL` in `tests/test_models/test_sam3/test_model.py`, and matched against the trainer's refusal message in `tests/test_train/test_sam3/test_train_sam3.py` |

`sam3`'s total is asserted two ways, and only one of them runs by default: the
test that actually instantiates the 821 M-parameter model is **opt-in**
(`SAM3_SHIPPED_AUDIT=1`, ~3.3 GiB of device memory). The default gate checks the
closed form against per-component figures instead. Read the row as a closed form
cross-checked per component, not as a routinely-executed end-to-end count.

The trainer **refuses** the `sam3` variant outright, and its refusal message
carries the reason: 821,708,598 parameters at a measured **10,072.9 MiB forward
peak** leaves no room for AdamW moments on a 12 GB card. That refusal message is
the single home of the 10,072.9 MiB figure — a measurement, carried by no
assertion — and `Sam3TrainingConfig(variant="sam3")` raising it is asserted in
`tests/test_train/test_sam3/test_train_sam3.py`. This table is the single prose
home of all three parameter counts; the package `__init__` points here rather
than restating them.

Enabling `query_selection=True` with `prompt_conditioned=True` adds weights on
top of `small` (§7); with the flag off, no weight is created at all, so the
on-disk checkpoints and the exact parameter-count oracle above are untouched.

## 6. Training

`Sam3TrainingModel` (see [`training_model.py`](training_model.py)) wraps
`Sam3Image` and emits **ONE packed supervision tensor**. That is all it does,
and the reason is that SAM 3's detection loss is **joint**: one Hungarian
assignment is shared across the classification, box, presence and mask terms,
so all four must be seen by one `Sam3DetectionLoss` object. Per-output-key dict
losses would compute a different assignment per term — or none — which is
exactly the property the joint matcher provides.

The packed tensor's channel layout is **not** defined in this package. Its
single home is `dl_techniques.losses.sam3_detection_loss`, and the wrapper
imports the channel constants rather than re-spelling any index, so a layout
change moves one file. The data pipeline and CLI live in
[`src/train/sam3/`](../../../../train/sam3/).

```python
from dl_techniques.models.SAM.SAM3 import (
    Sam3Image, Sam3TrainingModel, compile_sam3_trainer)

trainer = Sam3TrainingModel(Sam3Image.from_variant("tiny"), include_masks=True)
compile_sam3_trainer(trainer)      # sets jit_compile=False by setdefault
trainer.fit(dataset)
```

- **`jit_compile=False` is MANDATORY and has exactly one home**,
  `compile_sam3_trainer`, which sets it by `setdefault` so the invariant holds
  by construction. It is doubly forced: this model family already pins it, and
  the matcher crosses an eager `py_function` boundary for which no XLA kernel
  exists, so `jit_compile=True` fails hard.
- **There is no custom `train_step`, and there must never be one.**
- **`training=` is forwarded explicitly at every call site**, for the
  `drop_path_rate` reason in §3.
- **Nothing in this package may route mask supervision through SAM 1's
  mask-loss class or through the shared segmentation focal-loss class it
  calls**, whose probability clip has an exactly-zero derivative outside its
  range. Those two module names are deliberately not spelled anywhere in this
  package, including here: the constraint's gate is a grep over every file in
  it, and a prose mention erodes that instrument as effectively as a real
  import would — a failure already measured four times in this repository on
  the sibling backend-purity grep.

Deep supervision over the decoder's per-layer outputs is available and
optional. What it was measured to do — and, just as importantly, what it was
measured **not** to do — is recorded in [`sam3_image.py`](sam3_image.py)'s
docstring, with the comparator named beside each number, because the answer
depends entirely on the comparator. Read §8 before reading that result as a
capability result.

## 7. Query Selection — an opt-in, non-reference head

`Sam3EncoderQuerySelection` is **not** a reference component. It is this
package's own addition, reached through `Sam3Image(..., query_selection=True)`,
**OFF by default**, and behaviourally inert when off.

It exists for a measured reason: SAM 3's box output was image-independent **by
construction**, not by a training-time collapse. On the shipped synthetic runs
`val_box_std_across_images` read `6.9e-06` **on GPU** against an
across-*query* spread of `0.13`, and it is already that low at epoch 0. Quote
that figure to one significant figure and with the device attached: below
`~1e-5` the statistic is DEVICE-DEPENDENT — the same weights, split and code
read `6.94e-06` on GPU and `1.84e-06` on CPU, a factor of 3.8
(`src/train/sam3/train_sam3.py`'s module docstring is its home). What carries
the argument is the four-order gap to the across-query spread, not the digits.
The mechanism is that the decoder's box chain is
`sigmoid(delta + inverse_sigmoid(reference))` with a zero-initialized last
projection over a learned table broadcast across the batch, so at step 0 the
boxes *cannot* depend on the image at all.

The head reads the flattened finest neck level and emits, per position, an
objectness logit and a `cxcywh` box refined from that position's grid anchor;
the top `num_queries` boxes replace the decoder's learned, image-independent
`reference_points` table. Its own vacuity mode is named and guarded: a
degenerate objectness field selects positions `0 .. k - 1`, because `top_k`
breaks ties by ascending index — an image-independent selection with the right
shapes, dtypes and a plausible spread.

An optional `prompt_conditioned` FiLM modulation makes the top-k *selection*
itself prompt-dependent. The instrument is prompt-swap retention — replace every
image's prompt with another image's, then score the SAME checkpoint against the
SAME ground truth through the SAME IoU expression; it ships as
`train.sam3.baselines.prompt_swap_retention` and is runnable with
`python -m train.sam3.baselines --prompt-swap`. Read per arm, never as a
constant of the generator: **1.0000 on 9 of 9** prompt-blind checkpoints — the
swap changed nothing on any of them — against **0.6219 / 0.5982 / 0.6227** for
the `prompt_conditioned` arm across its three seeds, which is the arm doing
something. It is default-off.

## 8. The Baseline That Beats Every Trained Arm

This is the most important number in the package, and it is a negative one.

On this repository's synthetic SAM 3 generator, a **20-line, zero-parameter,
category-blind connected-components detector** — threshold `100/255`,
`scipy.ndimage.label`, the largest components' bounding boxes, no network, no
gradient, no text input — scores box IoU **0.9413 / 0.9370 / 0.9397** across
three seeds, through the identical IoU evaluation path and against the
identical 93 / 88 / 98 matched-pair denominators as the model arms.

The best trained SAM 3 arm this repository has produced — the `step9_qsel`
checkpoints, `val_box_iou` — scores **0.8450 / 0.8296 / 0.8191** on those same
three seeds.

`src/train/sam3/train_sam3.py`'s module docstring, which is the home of both
readings, forbids quoting that pair without **two** qualifiers. The first is the
paragraph above. The second: **`box_iou` does not read the text prompt at all.**
Replacing every image's prompt with another image's and scoring the same
checkpoint against the same ground truth retains `box_iou` at 100.00% on 3 of 3
seeds, in the arm AND in the deep-supervision control.

It ships as `connected_components_predictor` in
[`src/train/sam3/baselines.py`](../../../../train/sam3/baselines.py) and is
deliberately EXCLUDED from that module's `family_max`, so it can never silently
become the bar a trained arm is auto-compared against.

**Consequences, stated so they are not rediscovered:**

- Any accuracy number produced on this generator is a **wiring / learnability**
  result, not a capability result.
- A non-model baseline family made only of image-*blind* members (a fixed 5x5
  grid, a k-means box prior fitted on the training split's ground truth) is
  **not sufficient**: clearing it cannot distinguish "the model learned
  detection" from "this generator's box task is trivially solvable by any rule
  that reads the pixels". That family must gain an image-reading member before
  any future arm quotes it as a bar.
- An untrained-network floor alone understated the trivial level by roughly
  0.2 IoU.

## 9. Out of Scope in Phase 1

Named rather than left to be rediscovered:

- **The vision-language early-fusion encoder** the reference runs between neck
  and decoder. The neck's image memory and the text tower's prompt go straight
  into the decoder here. This is the largest single structural divergence in
  the package.
- **The exemplar / geometry prompt path**, which needs `grid_sample` /
  `roi_align` primitives `keras.ops` does not provide.
- **DAC-DETR query doubling**, which is gated on `self.training` upstream and
  so is provably inert at inference.
- **`Sam3TriViTDetNeck`** (the SAM 3.1 three-way neck) and the **video /
  tracking path**.
- **The `cxcywh → xyxy` conversion**, and the loss and matcher, which live in
  the losses package rather than here.

## 10. Serialization and Checkpoints

Every serializable class here carries a **bare** (zero-argument)
`@keras.saving.register_keras_serializable()`, so its registry key is
`Custom>ClassName` and contains no module path.

### Loading a checkpoint written before this package moved — registrar-first

This package used to live at a different dotted path. A `.keras` file records
the module path its classes lived in *when it was saved*, and Keras resolves a
class by looking in its registry FIRST and only then falling back to
`importlib.import_module` on that recorded string. That fallback now raises for
every SAM 3 checkpoint written before the move:

```
TypeError: Could not deserialize class 'Sam3TrainingModel' because its parent
module <the pre-move dotted path>.training_model cannot be imported.
```

(The old path is deliberately not spelled here. A close-out grep asserts it
survives nowhere under `src/`, and writing it in prose erodes that instrument
exactly as effectively as a real reference would.)

The fix is one import line — import the module that **defines** the saved
class before calling `load_model`, so the registry answers and the fallback is
never reached. Executed, on this package's own smallest on-disk checkpoint:

```python
import keras
import dl_techniques.models.SAM.SAM3.training_model   # registrar-first: the MODULE

model = keras.models.load_model(path, compile=False)
print(type(model).__name__, len(model.weights))
# Sam3TrainingModel 217
```

Import the **module**, not the package. For SAM 3 the package import happens to
be sufficient too, because `SAM3/__init__.py` imports `training_model` — but
that is not true of every sibling (SAM 2's `__init__` does not, and fails), so
naming the defining module is the rule that holds for all three.

## 11. Licensing — read this before extending

`dl_techniques` is **GPL-3.0**. Meta's released SAM 3 code is under the **SAM
License**, which is not compatible with it.

**This package was therefore reimplemented from published numbers and a re-read
reference — never transliterated.** No upstream source file was copied,
adapted, machine-translated or line-by-line ported into this repository, and no
upstream weights ship here.

**That constraint binds any future work on this package.** If you extend it —
building the early-fusion encoder, adding the video path, writing a
weight-conversion layer — the same rule applies: work from the paper, from
published configuration numbers, and from a read-and-set-aside reading of the
reference. Do not paste. Comparisons against upstream behaviour are fine and
several are recorded in the module docstrings, each pinned to a reference SHA;
a transcription is not.

The architecture is described by Ravi et al. (2025); the original SAM 3 was
developed by Meta AI Research.

## 12. Testing

Gate:

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest \
    tests/test_models/test_sam3/ \
    tests/test_train/test_sam3/ \
    tests/test_losses/test_sam3_detection_loss.py -q
```

Run it as **one** invocation. The model-only path covers neither the trainer
wiring (which is where the `sam3`-variant refusal and the parameter-count
oracle live) nor the joint loss and its matcher. No count is quoted here,
because a count in a README rots the day it is written.

## 13. Citation

```bibtex
@article{ravi2025sam3,
  title={SAM 3: Segment Anything with Concepts},
  author={Ravi, Nikhila and others},
  year={2025}
}
```

```bibtex
@inproceedings{carion2020detr,
  title={End-to-End Object Detection with Transformers},
  author={Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and Usunier, Nicolas and Kirillov, Alexander and Zagoruyko, Sergey},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```

```bibtex
@inproceedings{cheng2021maskformer,
  title={Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author={Cheng, Bowen and Schwing, Alexander G. and Kirillov, Alexander},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

```bibtex
@inproceedings{li2022vitdet,
  title={Exploring Plain Vision Transformer Backbones for Object Detection},
  author={Li, Yanghao and Mao, Hanzi and Girshick, Ross and He, Kaiming},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```
