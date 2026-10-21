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
>   here has been trained to any quality. `SAM2TrainingModel` is proven only to
>   RUN, with live gradients, under stock graph-mode `fit()`.
> - **The mask head does not learn under joint training** on the one setup that
>   was measured (§7). The cause is known and it is UNFIXED. Read §7 before
>   reading any training run of this package as a result.
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
7. [Training, and the Mask Head That Does Not Learn](#7-training-and-the-mask-head-that-does-not-learn)
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
image (B, H, W, 3)
  │
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
| [`neck.py`](neck.py) | lateral-conv index orientation `convs[n - i]`; the `scalp` drop happens BEFORE `vision_features` is taken | `test_neck.py` |
| [`mask_decoder.py`](mask_decoder.py) | the object-score token is PREPENDED (every index shifts by 1); skips are ADDED not concatenated; the stability score is self-consistency, not IoU-vs-GT; the unstable fallback is PER BATCH ELEMENT. `SAM2.from_variant` builds the decoder with the fallback OFF: when training decoded token 0 only, the fallback answered with untrained tokens (validation mask loss 1.712 with it on against 1.043 off, plan-2026-10-14T062616-81938e81 D-245); with the multimask tokens trained (D-249) it still raised the loss, scored on the same 8 draws of frame-0 points for both (D-252; held-out clips: 1.451 against 1.185 and 2.164 against 1.983 for two runs of the 32-clip command) | `test_mask_decoder.py`, `test_inference_decodes_the_mask_token_training_trains.py` |
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

## 7. Training, and the Mask Head That Does Not Learn

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
  candidates. Measured (mask loss, inference mode, through `evaluate`, every answer
  scored on the SAME frame-0 points, 8 draws of the pipeline's random points; D-252):
  on the smoke models no single-click answer beats the best constant logit, and which
  of the two answers is better changes between two runs of one command. The 32-clip,
  30-epoch run of step 12.6, on its held-out clips: best-IoU candidate 1.185, token 0
  1.925, lowest-loss candidate 1.116, constant 0.921; the same command rerun at step
  12.7: 1.984, 1.519, 1.094, 0.921. The 2-clip memorisation run of step 12.6 on its
  own clips: 1.070, 1.077, 0.958, constant 0.959. The step-12.6 figures that beat the
  constant (0.547 memorised, 0.790 held out) were single draws: the pipeline redraws
  the frame-0 point on every pass, and two consecutive `evaluate` calls of one answer
  scored 0.531 and 1.999. This is section 7's finding, not a new one.

### The mask head does not learn jointly

**SAM 2's mask head does not learn under joint training, the cause is known,
and it is UNFIXED.** Every objective arm fails — plain BCE, the shipped
focal+dice, upstream's `alpha_t` weighting, and dice-only. The binding
constraint is **the jointly-trained image encoder**, not the loss family and
not the step budget: on the trainer's own 8 diverse targets and within the same
step budget, the SAME decoder with a **frozen** encoder reaches mask IoU
**1.0000**, against **0.0091** for the jointly-trained arm.

This is the reason the scope note above says the package makes no accuracy
claim: the training path demonstrably runs, and on the one configuration that
was measured it demonstrably does not learn masks jointly.

Measured since, on the trainer's synthetic clips with the encoder trained jointly
(plan-2026-10-14T062616-81938e81, D-246, D-249, D-252): a 2-clip run's logged mask
loss falls from 7.34 to 0.48 on the frame-0 points it was drawn last, but the
pipeline redraws those points on every pass, and scored on 8 draws its lowest-loss
candidate reaches 0.958 on its own clips against a best constant of 0.959; a 32-clip,
30-epoch run scores 1.094 to 1.116 on 4 held-out clips against 0.921. The step-12.6
figures that beat the constant (0.48, 0.744) were single draws. The statement above
stands for this trainer's configuration.

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
