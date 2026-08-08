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
from dl_techniques.models.SAM.SAM2 import SAM2, SAM2MemoryBank, create_sam2

model = create_sam2("tiny")            # RANDOM weights — no checkpoint ships

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
| [`mask_decoder.py`](mask_decoder.py) | the object-score token is PREPENDED (every index shifts by 1); skips are ADDED not concatenated; the stability score is self-consistency, not IoU-vs-GT; the unstable fallback is PER BATCH ELEMENT | `test_mask_decoder.py` |
| [`memory_encoder.py`](memory_encoder.py) | `20 * sigmoid(x) - 10`, affine AFTER the sigmoid; the downsampler's layer COUNT (both plausible readings give total stride 16); the fusion is additive | `test_memory_encoder.py` |
| [`memory_bank.py`](memory_bank.py) | object-pointer tokens sit at the TAIL; conditioning frames always take temporal slot `t_pos = 0` | `test_memory_bank.py` |
| [`memory_attention.py`](memory_attention.py) | four independently configurable positional-encoding sites, asymmetric in the shipped SAM 2.1 setting | `test_memory.py` |

> **`SAM2MemoryBank` is not to be confused with
> `src/dl_techniques/models/memory_bank/`.** That package is
> `WaveFieldMemoryLLM`'s keyed read/write store for language modelling — a
> different data structure with a colliding name. It was reviewed and REJECTED
> as a reuse target here; nothing in this package derives from it.

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

Three properties of that wrapper are load-bearing and stated at the class:

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

### The measured non-result

**SAM 2's mask head does not learn under joint training, the cause is known,
and it is UNFIXED.** Every objective arm failed — plain BCE, the shipped
focal+dice, upstream's `alpha_t` weighting, and dice-only. The binding
constraint is **the jointly-trained image encoder**, not the loss family and
not the step budget: measured on the trainer's own 8 diverse targets and within
the same step budget, the SAME decoder with a **frozen** encoder reaches mask
IoU **1.0000**, against **0.0091** for the jointly-trained arm.

Two earlier written diagnoses — that the loss composition was at fault, and
that it was "the decoder's convergence rate at `lr=1e-4` / 240 steps" — are
both SUPERSEDED by that pair.

This is the reason the scope note above says the package makes no accuracy
claim: the training path demonstrably runs, and on the one configuration that
was measured it demonstrably does not learn masks jointly.

## 8. Serialization and Checkpoints

Every serializable class here carries a **bare** (zero-argument)
`@keras.saving.register_keras_serializable()`, so its registry key is
`Custom>ClassName` and contains no module path. `SAM2MemoryBank` is the
deliberate exception: it is a plain-Python container with no weights (§5) and
carries no decorator at all. `.keras` round trips are covered by the test gate.

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

> **Importing the SAM 2 *package* is NOT sufficient.** `SAM2/__init__.py`
> imports `memory_bank` and `model` only, never `training_model`, so
> `Custom>SAM2TrainingModel` is never entered into the registry by a package
> import and Keras falls straight through to the failing fallback. SAM 1's and
> SAM 3's inits both import theirs, which is why this trips people on SAM 2
> and only on SAM 2. The rule that holds for all three is: **import the module
> that DEFINES the saved class.**

Executed, on this package's own smallest on-disk checkpoint:

```python
import keras
import dl_techniques.models.SAM.SAM2.training_model   # registrar-first: the MODULE

model = keras.models.load_model(path, compile=False)
print(type(model).__name__, len(model.weights))
# SAM2TrainingModel 345
```

Replacing that import with `import dl_techniques.models.SAM.SAM2` and changing
nothing else reproduces the `TypeError` above.

One further caveat for anyone A/B-ing a checkpoint's outputs across a change:
**SAM 2's forward pass is nondeterministic on GPU.** Measured on one RTX 4070,
on a single 796,401-parameter `SAM2TrainingModel` checkpoint, holding commit,
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

`dl_techniques` is **GPL-3.0**. Meta's released SAM 2 code is under the **SAM
License**, which is not compatible with it.

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
