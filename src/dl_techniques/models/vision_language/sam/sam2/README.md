# SAM 2 (Segment Anything in Images and Videos) — Keras 3 Implementation

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of the **SAM 2** architecture, after
["SAM 2: Segment Anything in Images and Videos"](https://arxiv.org/abs/2408.00714)
(Ravi et al., 2024): a hierarchical Hiera trunk with an FPN neck, a streaming
memory (memory attention, memory encoder, memory bank), and a mask decoder that
additionally emits an object score and an object pointer.

> **Scope, stated up front.** This package ships the **architecture, both of
> its forward paths, and a trainable multi-frame wrapper** — and nothing else.
> - **No pretrained checkpoint ships here or is downloaded**, and **no released
>   Meta SAM 2 checkpoint has ever been loaded in this repository.** No
>   key-mapping layer exists. Every reference-fidelity statement in this
>   package is an architectural argument, not a measurement against real
>   weights.
> - **It makes no accuracy claim and no segmentation-quality claim.** Nothing
>   here has been trained to any quality. What is measured is that the trainer's
>   own command memorises its 2-clip batch and that a 32-clip smoke run beats a
>   constant on 4 held-out synthetic clips (§7); nothing else.
> - **The mask head did not learn under joint training until step 12.8 of
>   plan-2026-10-14T062616-81938e81**: the image encoder was fed `[0, 255]`
>   pixels where upstream feeds standardized ones (§7). Training runs logged
>   before that step are not results.
> - **This package was reimplemented, never transliterated**, because upstream's
>   licence forbids the alternative. See §10 — that constraint binds any future
>   extension of this package too.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Two Entry Points, Deliberately Different in Kind](#4-two-entry-points-deliberately-different-in-kind)
5. [Component Reference](#5-component-reference)
6. [Variants](#6-variants)
7. [Training, and Why the Mask Head Did Not Learn](#7-training-and-why-the-mask-head-did-not-learn)
8. [Serialization and Checkpoints](#8-serialization-and-checkpoints)
9. [Testing](#9-testing)
10. [Licensing — read this before extending](#10-licensing--read-this-before-extending)
11. [Citation](#11-citation)

---

## 1. Overview

SAM 1 segments a single image from a prompt. SAM 2 keeps that and adds a
**streaming memory**: each frame is conditioned on a bank of previously encoded
frames plus a set of object pointers, so a prompt given on frame 0 propagates
forward without being re-given.

The package is fifteen public classes across eight implementation modules (the
ninth `.py` file is the package `__init__`). The exported
surface is deliberately three names — `SAM2`, `SAM2MemoryBank`, `create_sam2` —
mirroring SAM 1's; every component stays behind its own submodule and is
imported from there. Widening `__all__` is a deliberate act, asserted in both
directions (nothing missing, nothing extra) by
`tests/test_models/test_sam2/test_package_surface.py`, so the surface cannot
drift open one re-export at a time.

| Module | Public classes | Role |
|---|---|---|
| [`hiera.py`](hiera.py) | `Hiera`, `HieraBlock`, `HieraMultiScaleAttention`, `HieraPatchEmbed` | hierarchical windowed trunk, four feature levels out |
| [`neck.py`](neck.py) | `SAM2FpnNeck`, `SAM2ImageEncoder` | FPN to `d_model` per level, one sine PE each, `scalp` drop |
| [`memory_attention.py`](memory_attention.py) | `SAM2MemoryAttention`, `SAM2MemoryAttentionLayer` | self- then cross-attention against the memory sequence, 2D axial RoPE |
| [`mask_decoder.py`](mask_decoder.py) | `SAM2MaskDecoder` | masks, IoU, object score, object pointer |
| [`memory_encoder.py`](memory_encoder.py) | `SAM2MemoryEncoder`, `SAM2MaskDownSampler`, `SAM2Fuser` | mask + pixel features → `mem_dim` memory |
| [`memory_bank.py`](memory_bank.py) | `SAM2MemoryBank` | plain-Python per-video state; **not a Keras layer** |
| [`model.py`](model.py) | `SAM2`, `create_sam2` | assembly, the learned tensors belonging to no component, the variant table |
| [`training_model.py`](training_model.py) | `SAM2TrainingModel` | traceable multi-frame wrapper for stock `fit()` |

SAM 1 is **imported from, never edited by**, this package:
`SAM2MaskDecoder` reuses SAM 1's `TwoWayTransformer`, and `SAM2` reuses SAM 1's
`PromptEncoder`. `SAM2MaskDecoder` is a new *sibling* of SAM 1's `MaskDecoder`,
not a subclass or a configured variant of it — SAM 1's decoder bakes its token
layout into positional slices inside method bodies and has no skip-connection
argument at all, so none of the SAM 2 deltas can be expressed as a defaulted
keyword argument.

Every module carries a full docstring with its own measured-caveats block. This
README points at those rather than restating them, so each fact has one home.

## 2. Architecture

```
image (B, H, W, 3), RGB in [0, 255]
  │
  ▼
standardize ────────── (x - pixel_mean) / pixel_std, upstream's ImageNet statistics
  │                    (inside SAM2ImageEncoder, so every entry point gets it)
  ▼
Hiera ──────────────── stem (one strided conv, 4x) → four stages
  │                    at each boundary: width x2, heads x2, grid /2 by
  ▼                    max-pooling the attention QUERIES
  four levels, ASCENDING stage order (outputs[0] finest and narrowest)
  │
  ▼
SAM2FpnNeck ────────── one 1x1 lateral conv per level at d_model, top-down
  │                    addition, one sine PE map per level
  ▼
SAM2ImageEncoder ───── drop `scalp` levels; vision_features = last of the rest
  │
  ├──────────── IMAGE path (SAM2.call) ───────────────────────────┐
  │                                                               │
  └── VIDEO path (SAM2.stream_step) ───► SAM2MemoryAttention      │
                                          (memory as k/v)          │
                                              │                    │
                                              ▼                    ▼
                                 PromptEncoder (SAM 1's) ──► SAM2MaskDecoder
                                                                   │
                    ┌──────────────────────────────────────────────┤
                    ▼                                              ▼
            SAM2MemoryEncoder ──► SAM2MemoryBank          masks, IoU,
            (mask + pixels → mem_dim)                     object score,
                                                          object pointer
```

## 3. Quick Start

```python
from dl_techniques.models.vision_language.sam.sam2 import SAM2, SAM2MemoryBank, create_sam2

model = create_sam2("tiny")            # RANDOM weights — no checkpoint ships
# create_sam2(variant, pretrained=False, **kwargs) delegates to SAM2.from_variant;
# pretrained=True (or a path) raises NotImplementedError.

# Image path: traceable, touches neither the memory bank nor memory attention.
outputs = model({"image": images, "points": (coords, labels)})
```

```python
# Video path: plain Python, never traced, mutates the bank it is given.
model.stream_reset()
for t, frame in enumerate(frames):
    out = model.stream_step(
        frame,
        frame_idx=t,
        points=(coords, labels) if t == 0 else None,
        is_conditioning=(t == 0),   # conditioning frames keep t_pos = 0 forever
    )
```

## 4. Two Entry Points, Deliberately Different in Kind

This is the single most important structural fact about the package.

| | `SAM2.call` | `SAM2.stream_step` |
|---|---|---|
| Traceable under `tf.function` / `fit()` | **Yes** | **No, by design** |
| Touches the memory bank | No | Yes |
| Touches memory attention | No | Yes |
| Calls `self(...)` | — | Never |
| What it is | the image path | the online video tracker |

`stream_step` mutates a Python object, branches on whether that object is
empty, and reads Python integers out of its selection policy. None of that
survives tracing, and none of it is a defect: it follows the repository's
existing `VideoJEPA.stream_reset` / `stream_step` precedent. Training does not
go through it — see §7.

A second constraint runs through the whole package and is a **correctness**
constraint, not a layout choice: **RoPE inside memory attention is SPATIAL
ONLY**, broadcast identically across every memory frame, so a memory frame's
temporal identity is carried *exclusively* by `maskmem_tpos_enc`, the learned
per-slot embedding owned by `SAM2`. The memory bank returns slot INDICES and
never adds the embedding itself. Conflating the two produces a model that runs
and that no test could distinguish from a correct one.

## 5. Component Reference

A number of mechanisms in this package are **silent when ported wrong** — the
model builds, forward-passes, trains and serializes either way, with no shape
error anywhere. Each is stated at its own class and guarded *behaviourally*
(not by a shape assertion) in the matching test module. They are listed here by
location so a reader knows where to look, not restated:

| Where | What is silent if wrong | Guard |
|---|---|---|
| [`hiera.py`](hiera.py) | window size lags one block behind the stage transition; query pooling is asymmetric (`q` only) | `test_hiera.py` |
| [`neck.py`](neck.py) | lateral-conv index orientation `convs[n - i]`; the `scalp` drop happens BEFORE `vision_features` is taken; `SAM2ImageEncoder` standardizes `[0, 255]` pixels with upstream's `NormalizeAPI` statistics before the trunk (`pixel_mean` / `pixel_std`, config). Without it the trainer never got below a constant (D-260) | `test_neck.py`, `test_the_image_encoder_standardizes_its_input.py` |
| [`mask_decoder.py`](mask_decoder.py) | the object-score token is PREPENDED (every index shifts by 1); skips are ADDED not concatenated; the stability score is self-consistency, not IoU-vs-GT; the unstable fallback is PER BATCH ELEMENT. `SAM2.from_variant` builds the decoder with the fallback OFF: when training decoded token 0 only, the fallback answered with untrained tokens (validation mask loss 1.712 with it on against 1.043 off, plan-2026-10-14T062616-81938e81 D-245); with the multimask tokens trained (D-249) and the input standardized (D-260) it still raises the loss, scored on the same 8 fixed draws of frame-0 points for both: 4 held-out clips of the 32-clip run 0.367 on against 0.347 off (D-261) | `test_mask_decoder.py`, `test_inference_decodes_the_mask_token_training_trains.py` |
| [`memory_encoder.py`](memory_encoder.py) | `20 * sigmoid(x) - 10`, affine AFTER the sigmoid; the downsampler's layer COUNT (both plausible readings give total stride 16); the fusion is additive | `test_memory_encoder.py` |
| [`memory_bank.py`](memory_bank.py) | object-pointer tokens sit at the TAIL; conditioning frames always take temporal slot `t_pos = 0` | `test_memory_bank.py` |
| [`memory_attention.py`](memory_attention.py) | four independently configurable positional-encoding sites, asymmetric in the shipped SAM 2.1 setting | `test_memory.py` |

> **`SAM2MemoryBank` is unrelated to a language-modelling memory bank.** It is
> SAM 2's per-video slot store and nothing else; no keyed read/write store for
> language modelling is reused here.

## 6. Variants

`SAM2.MODEL_VARIANTS` holds **only `tiny` and `hiera_l`**. The other published
SAM 2 sizes' numbers were never read by this work, and inventing them would be
fabrication.

That table also deliberately does **not** restate `embed_dim`, `stages`,
`window_spec`, `image_size`, `d_model` or `scalp`: it reads the trunk geometry
from `Hiera.MODEL_VARIANTS` and the neck/scalp geometry from
`SAM2ImageEncoder.MODEL_VARIANTS`, so each geometry has exactly one home. A
geometry restated in two places is a latent defect.

## 7. Training, and Why the Mask Head Did Not Learn

`SAM2TrainingModel` (see [`training_model.py`](training_model.py)) runs the
image encoder ONCE over a flattened `(B * T, ...)` batch and then drives SAM 2's
submodules through an explicitly UNROLLED Python loop over a STATIC
`num_frames`. The whole loop traces under stock graph-mode `fit()`. There is no
custom `train_step`. The data pipeline and CLI live in
[`src/train/sam2/`](../../../../train/sam2/); the loss is
`dl_techniques.losses.sam2_video_loss`.

Four properties of that wrapper are load-bearing and stated at the class:

- **`compile(jit_compile=False)` IS MANDATORY.** Keras 3.8's `fit()` defaults to
  `jit_compile='auto'`, which selects XLA on a GPU, and `Hiera`'s stem
  interpolates its learned positional embedding with a bicubic
  `ops.image.resize` for which no XLA GPU kernel exists. The failure is at the
  first `fit()` step and is loud. The verbatim error and the both-directions
  guard (`TestXLARefusal`) are in the module docstring.
- **The gradient policy is TRUNCATED BPTT, with exactly one truncation point**
  — the memory encoder's inputs — because upstream's alternative
  (backpropagating through the whole T-frame chain, paid for with gradient
  checkpointing) is not in this package's budget. Which tensors are detached
  and which stay live is enumerated in the module docstring.
- **`obj_ptr_proj` (6 variables) still ships FROZEN.** Stated rather than left
  looking fixed.
- **Frame 0 trains the three multimask candidates, as upstream does**
  (plan-2026-10-14T062616-81938e81 D-249; `multimask_output_in_sam=True`, upstream's
  name and training value). On a frame-0 prompt of one point and no box the decoder
  returns three candidates; the mask loss reaches the one of lowest
  `20 * focal + dice` against the ground truth, and the IoU head is supervised on
  all three (`iou_supervision` then has `3 * T` slots). Every other frame decodes
  mask token 0. The memory and the object pointer take the candidate of highest
  predicted IoU. The switch lives on `SAM2` (`multimask_output_in_sam`, default
  `True`, D-252) and the wrapper reads it, so inference answers the frames training
  trains: `SAM2.call` and `SAM2.stream_step(..., is_conditioning=True)` with one point
  and no box decode the three candidates and return the one of highest predicted
  IoU, still one mask (`(B, 1, h, w)`), and its object pointer and memory come from
  that candidate, as upstream's `_forward_sam_heads` does. A tracking frame, a
  non-conditioning click, two points or a box decode token 0 (upstream's
  `_use_multimask` at the `SAM2Base` defaults `multimask_min_pt_num =
  multimask_max_pt_num = 1`, tracking off; the released configs turn tracking on,
  which this wrapper does not train). `multimask_output=True` still returns all three
  candidates. Measured (mask loss, inference mode, every answer scored on the SAME 8
  fixed frame-0 clicks per clip, numpy draws of one foreground cell; D-261): on the
  2-clip memorisation run the best-IoU candidate and the lowest-loss candidate are the
  same mask on every draw (0.042) and token 0 scores 0.060; on 4 held-out clips of the
  32-clip run the best-IoU candidate scores 0.347 and token 0 0.397.

### Why the mask head did not learn, and what fixed it

Until step 12.8 of plan-2026-10-14T062616-81938e81 this section said the mask head
does not learn under joint training and blamed the jointly-trained image encoder.
Two defects were behind it (D-258, D-260):

- **The encoder was fed raw pixels.** Upstream SAM 2 standardizes its input with
  ImageNet statistics before the image encoder (`NormalizeAPI` in the training
  transforms, `SAM2Transforms` in both predictors). This port fed `[0, 255]` pixels
  straight into Hiera's patch embedding: 255 x 0.226, about 58 times upstream's input
  scale, and centred near 115 instead of 0, from all three entry points. `SAM2ImageEncoder` now standardizes them itself
  (`pixel_mean` / `pixel_std`, upstream's values, config). The earlier "unbounded
  activation-scale growth" finding was this input scale, not the neck's missing
  normalization: upstream's neck has none either.
- **The logged loss was half the true loss.** The clip source is a
  `from_generator` dataset; with unknown cardinality Keras 3.8 counts one phantom
  zero-loss batch per epoch, so the logged loss was `n / (n + 1)` of the mean over `n`
  batches. A 2-clip run logged 0.48 for a model whose loss on that batch was 0.95,
  against a best constant of 0.96. The pipeline now states its length. It also
  redraws the frame-0 click on every pass: before, the trainer's global seed made the
  stateful random op repeat the same clicks every epoch (D-259).

Measured after both fixes on the RTX 4070 (scores are the hand mask loss of an
inference-mode call on arrays, 8 fixed clicks per clip, the same clicks for every
answer; D-261):

| Run | Clips scored | Model's answer | Per-draw range | Token 0 | Best constant |
|---|---|---|---|---|---|
| `--smoke --num-clips-train 2 --batch-size 2 --epochs 150` | its 2 training clips | 0.042 | 0.041 to 0.043 | 0.060 | 0.959 |
| same run | 4 held-out clips | 0.956 | 0.954 to 0.957 | 0.996 | 0.921 |
| `--smoke --num-clips-train 32 --epochs 30` | its 32 training clips | 0.261 | 0.224 to 0.331 | 0.249 | 0.947 |
| same run | 4 held-out clips | 0.347 | 0.346 to 0.348 | 0.397 | 0.921 |

Both checkpoints are the epoch `EarlyStopping` restored (lowest `val_loss`: epoch 80 of
150 and epoch 14 of 30). A 2-clip run memorises and generalises to nothing, as
expected. In the 32-clip run the training mask loss jumps from 0.22 to 2.6-22 at epoch
16 and stays high until epoch 28, and the held-out mask loss sits at 5.3 from epoch 16
on (except epoch 19), while the object-score loss keeps falling: those are present
frames the score head calls absent, scored against the `-1024` sentinel. That is the
reference loss's behaviour (D-250), and it is why the restored epoch, not the last
one, is the result. The trainer's own memorisation guard is
`tests/test_train/test_sam2/test_the_trainer_memorises_one_batch.py`.

## 8. Serialization and Checkpoints

Every serializable class here carries `@register_dl_technique(...)`, from
`dl_techniques.utils.keras_registration`, so its registry key is
`dl_techniques.models.sam2.<module>>ClassName` — the defining module's dotted path with
`models/`'s `vision_language/` family directory and `sam/` subfamily container stripped, both
being a filing decision rather than a namespace. `SAM2TrainingModel` resolves to
`dl_techniques.models.sam2.training_model>SAM2TrainingModel` and `SAM2` to
`dl_techniques.models.sam2.model>SAM2` (both readable with
`keras.saving.get_registered_name`). The helper additionally binds the legacy
`Custom>ClassName` as an alias to the **same object**, which is what an older
archive names, so `keras.saving.get_registered_object("Custom>SAM2TrainingModel")` still
returns the class once its module is imported.
`SAM2MemoryBank` is the deliberate exception: it is a plain-Python container with no weights
(§5) and carries no decorator at all. `.keras` round trips are covered by the test gate.

### `SAM2FpnNeck.fpn_interp_model` — narrow at construction, forgiving on load

The neck's 2x top-down step is an owned
`keras.layers.UpSampling2D(size=(2, 2), interpolation=fpn_interp_model,
name="top_down_upsample")` sub-layer, created in `__init__` and built in `build()`. So:

* **`__init__` raises `ValueError`** on anything outside `{"nearest", "bilinear"}` — including
  the legacy spellings `"nn"`, `"NEAREST"` and `" nearest "`.
* **`from_config` normalizes and remaps**, logging a warning: `.lower().strip()`, then
  `"nn" -> "nearest"`. Every archive carrying a legacy spelling still loads.

The sub-layer is weightless, so `count_params()` and the weight-path list are unaffected
(guarded by
`tests/test_models/test_sam2/test_neck.py::TestTopDownUpsampleSubLayer`). The shipped `tiny`
and `hiera_l` variants both default to `"nearest"`.

### Loading a checkpoint written before this package moved — registrar-first

This package used to live at a different dotted path. A `.keras` file records
the module path its classes lived in *when it was saved*, and Keras resolves a
class by looking in its registry FIRST and only then falling back to
`importlib.import_module` on that recorded string. That fallback now raises for
every SAM 2 checkpoint written before the move:

```
TypeError: Could not deserialize class 'SAM2TrainingModel' because its parent
module <the pre-move dotted path>.training_model cannot be imported.
```

(The old path is deliberately not spelled here. A close-out grep asserts it
survives nowhere under `src/`, and writing it in prose erodes that instrument
exactly as effectively as a real reference would.)

> **Importing the SAM 2 *package* is NOT sufficient.** `sam2/__init__.py`
> imports `memory_bank` and `model` only, never `training_model`, so
> `SAM2TrainingModel` is never entered into the registry by a package import under
> **either** of its keys — neither the current
> `dl_techniques.models.sam2.training_model>SAM2TrainingModel` nor the legacy
> `Custom>SAM2TrainingModel` alias an older archive names — and Keras falls
> straight through to the failing fallback. In a fresh process, after
> `import dl_techniques.models.vision_language.sam.sam2`,
> `keras.saving.get_registered_object` returns `None` for both keys and
> `training_model` is absent from `sys.modules`. This is a statement about import SIDE
> EFFECTS, not about how the key is spelled. SAM 1's and
> SAM 3's inits both import theirs, which is why this trips people on SAM 2
> and only on SAM 2. The rule that holds for all three is: **import the module
> that DEFINES the saved class.**

Executed, on this package's own smallest on-disk checkpoint:

```python
import keras
import dl_techniques.models.vision_language.sam.sam2.training_model   # registrar-first: the MODULE

model = keras.models.load_model(path, compile=False)
print(type(model).__name__, len(model.weights))
# SAM2TrainingModel 345
```

Replacing that import with `import dl_techniques.models.vision_language.sam.sam2` and changing
nothing else reproduces the `TypeError` above.

One further caveat for anyone A/B-ing a checkpoint's outputs across a change:
**SAM 2's forward pass is nondeterministic on GPU.** On one RTX 4070, on a
single 796,401-parameter `SAM2TrainingModel` checkpoint, holding commit,
weights, seed and input fixed: 8 runs produced **three** distinct output digests
while all 8 produced one identical weight digest, and 3 runs of the same probe
on CPU produced a single output digest. Compare outputs on CPU, or a
reduction-order difference will read as a model change. The specific
nondeterministic op was not identified.

## 9. Testing

Gate:

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest \
    tests/test_models/test_sam2/ \
    tests/test_train/test_sam2/ -q
```

Run it as **one** invocation; the model-only path covers neither the data
pipeline nor the args→config wiring. No count is quoted here, because a count
in a README rots the day it is written.

The per-module guards are named in §5's table. Two more are worth knowing:
`test_package_surface.py` pins `__all__` in both directions, and
`test_training_model.py` carries the spy proving `SAM2.call` / `SAM2.__call__`
is never invoked by the training wrapper — pinned by execution, not by
inspection.

## 10. Licensing — read this before extending

`dl_techniques` is **Apache-2.0**. Meta's released SAM 2 code is under the **SAM
License**, a different license.

**This package was therefore reimplemented from published numbers and a re-read
reference — never transliterated.** No upstream source file was copied,
adapted, machine-translated or line-by-line ported into this repository, and no
upstream weights ship here.

**That constraint binds any future work on this package.** If you extend it —
adding the video/tracking pieces, a new variant, a weight-conversion path —
the same rule applies: work from the paper, from published configuration
numbers, and from a read-and-set-aside reading of the reference. Do not paste.
A comparison against upstream behaviour is fine and several are recorded in the
module docstrings; a transcription is not.

The architecture is described by Ravi et al. (2024); the original SAM 2 was
developed by Meta AI Research.

## 11. Citation

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv:2408.00714},
  year={2024}
}
```

```bibtex
@inproceedings{ryali2023hiera,
  title={Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles},
  author={Ryali, Chaitanya and Hu, Yuan-Ting and Bolya, Daniel and Wei, Chen and Fan, Haoqi and Huang, Po-Yao and Aggarwal, Vaibhav and Chowdhury, Arkabandhu and Poursaeed, Omid and Hoffman, Judy and Malik, Jitendra and Li, Yanghao and Feichtenhofer, Christoph},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```

```bibtex
@inproceedings{su2021roformer,
  title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and Murtadha, Ahmed and Wen, Bo and Liu, Yunfeng},
  booktitle={arXiv:2104.09864},
  year={2021}
}
```
