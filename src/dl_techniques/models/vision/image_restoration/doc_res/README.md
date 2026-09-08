# DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 port of **DocRes** (Zhang et al., CVPR 2024): a *single* Restormer backbone
that performs five document-image restoration tasks — dewarping, deshadowing,
appearance enhancement, deblurring and binarization — with **no task embedding, no task
head and no conditioning module anywhere in the network**.

The task is carried entirely by the input. The caller stacks a 3-channel classical-CV
**DTSPrompt** onto the 3-channel RGB page, and that 6-channel tensor is the only thing
that tells the network which job it is doing.

---

## Table of Contents

1. [Overview](#1-overview)
2. [The input contract: 6 channels, not 3](#2-the-input-contract-6-channels-not-3)
3. [The five tasks](#3-the-five-tasks)
4. [Architecture](#4-architecture)
5. [Variants](#5-variants)
6. [Quick start](#6-quick-start)
7. [Constraints you must read before using this](#7-constraints-you-must-read-before-using-this)
8. [Component reference](#8-component-reference)
9. [Configuration](#9-configuration)
10. [Serialization](#10-serialization)
11. [Testing](#11-testing)
12. [Citation](#12-citation)

---

## 1. Overview

Document restoration is normally five networks: a dewarper, a shadow remover, an
enhancer, a deblurrer, a binarizer. DocRes' claim is that one network is enough, provided
the *degradation-specific* knowledge is moved out of the weights and into a cheap,
deterministic, hand-written visual prompt computed on the CPU.

That prompt is not learned and carries no parameters. It is three extra image channels
produced by classical operations (background estimation, Sobel magnitude, Sauvola
thresholding, a coordinate grid) whose choice depends on the task. The network sees
6 channels in and emits 3 channels out, always, for every task.

Consequences worth internalising before reading the code:

* There is **nothing to inspect in this package** that reveals which task a checkpoint was
  trained for. The architecture is task-blind by construction.
* **One task per training run.** A checkpoint is task-specific even though the class is not.
* The prompt generators are not here — they live in
  `dl_techniques/datasets/document_restoration/` — because they are a *data* concern,
  and they are precomputed to sidecar files rather than run per batch.

---

## 2. The input contract: 6 channels, not 3

```
input tensor  (B, H, W, 6)
              └─ channels 0:3  RGB page, scaled to [0, 1]
              └─ channels 3:6  DTSPrompt — task-dependent, see §3

output tensor (B, H, W, 3)     always 3 channels; how many of them are
                               supervised is task-dependent, see §3
```

Passing a 3-channel image is not a degenerate case that "mostly works" — it is a shape
error, and the model raises. If you want a different width, `input_channels` is a
constructor argument, but the pretrained-DocRes semantics only hold at 6.

---

## 3. The five tasks

| Task | DTSPrompt channels (3) | Supervised output channels | Loss |
|---|---|---|---|
| Dewarping | normalised base-coordinate grid: `x`, `y`, mask | `[..., :2]` — a 2-channel backward-mapping flow; channel 2 is unused | L1 |
| Deshadowing | the estimated background image (`bg_img`), 3 ch | all 3 (RGB) | L1 |
| Appearance enhancement | the background-normalised image (`norm_img`), 3 ch | all 3 (RGB) | L1 |
| Deblurring | Sobel gradient magnitude, replicated to 3 ch | all 3 (RGB) | L1 |
| Binarization | Sauvola threshold map, Sobel magnitude, Sauvola binary @155 | `[..., :2]` — 2-class logits | categorical cross-entropy |

Two rows do **not** use the full output width. Dewarping predicts a flow field, and
binarization predicts a 2-class map; in both cases the third output channel is produced
by the network and then dropped by the loss. That is upstream's design, not an oversight
in this port — the output width is fixed at 3 so a single backbone serves all five rows.

The single source of truth for this table in code is the `TASKS` table in
`dl_techniques/datasets/document_restoration/tasks.py`. Nothing else in the tree may
branch on a task string; if you find code that does, that is the defect.

---

## 4. Architecture

The backbone is an **unmodified Restormer** (Zamir et al., 2022). Restormer's attention
runs over the *channel* axis rather than the pixel axis: for `c` channels and `n = h*w`
pixels the attention matrix is `c x c`, so cost is linear in `n`. That is what makes a
full-resolution document page — routinely 1024 px on a side — affordable at all, and it
is why this architecture rather than a ViT is the base for a document generalist.

```
inputs (B, H, W, 6)
  │  patch_embed 3x3, bias-free
enc1_in (H, W, 48) ──────────────────────────────────────┐
  │  encoder_level1: 2 blocks, 1 head                     │
out_enc1 (H, W, 48) ─────────────────────────────┐        │
  │  down1_2 -> encoder_level2: 3 blocks, 2 heads │        │
out_enc2 (H/2, W/2, 96) ─────────────────┐        │        │
  │  down2_3 -> encoder_level3: 3 blocks, 4 heads │        │
out_enc3 (H/4, W/4, 192) ────────┐        │        │       │
  │  down3_4 -> latent: 4 blocks, 8 heads │        │       │
  │  up4_3, concat ◄──────────────┘        │        │       │
  │  reduce_chan_level3 1x1 (384 -> 192)   │        │       │
  │  decoder_level3: 3 blocks, 4 heads     │        │       │
  │  up3_2, concat ◄───────────────────────┘        │       │
  │  reduce_chan_level2 1x1 (192 -> 96)             │       │
  │  decoder_level2: 3 blocks, 2 heads              │       │
  │  up2_1, concat ◄────────────────────────────────┘       │
  │  *** NO reduce_chan_level1 *** -> stays at 96           │
  │  decoder_level1: 2 blocks, 1 head, at 96 channels       │
  │  refinement: 4 blocks, 1 head, at 96 channels           │
  +  skip_conv 1x1 (48 -> 96) ◄────────────────────────────┘
  │  output 3x3, bias-free
outputs (B, H, W, 3)
```

Six things in that diagram look like bugs and are not. Each is faithful to the reference
implementation, each is enumerated with its upstream line number at
`# DECISION plan-2026-09-08T111844-de235227/D-015` in `model.py`, and each is commented
again at its wiring site. The load-bearing ones for a reader:

* **No `reduce_chan_level1`.** Levels 3 and 2 reduce the concatenated skip back down; level 1
  does not, so decoder level 1 and the refinement stage run at **96** channels, not 48.
* **No global input residual.** DocRes does *not* predict `input + net(input)`. It cannot:
  the input is 6 channels and the output is 3, and for dewarping the output is a
  coordinate flow, not an image.
* **`skip_conv` is unconditional** — the patch-embedding output is always added back after
  refinement.

Both the MDTA attention and the GDFN feed-forward used inside every block are registered
library layers (`multi_dconv_head_transposed` in `ATTENTION_REGISTRY`, `gated_dconv` in
`FFN_REGISTRY`), not private code.

---

## 5. Variants

`DocRes.MODEL_VARIANTS` has exactly **two** rows, and both are real published
configurations. It is a class attribute, not a module-level constant.

| Variant | `num_blocks` | Params | Source |
|---|---|---|---|
| `"docres"` (default) | `[2, 3, 3, 4]` | **15,203,680** | The shipped DocRes configuration, identical at all three upstream call sites — arXiv:2405.04408 |
| `"restormer_base"` | `[4, 6, 6, 8]` | **26,132,548** | The Restormer paper default and the upstream class default, which every DocRes call site overrides — arXiv:2111.09881 |

Both parameter counts were verified against an independent analytic derivation (delta 0),
and 26.13M is the figure the Restormer paper reports for its dual-pixel model.

No third row exists. Do not add one that is a plausible interpolation between these two:
there is no published model at that size, and naming one would imply there is.

---

## 6. Quick start

```python
import numpy as np
from dl_techniques.models.vision.image_restoration.doc_res import DocRes, create_doc_res

model = create_doc_res("docres")          # 15.2M params, the shipped configuration
model.build((None, None, None, 6))
model.summary()

# 6 channels: RGB page ++ DTSPrompt. Sides must be divisible by 8.
page   = np.random.rand(1, 256, 256, 3).astype("float32")
prompt = np.random.rand(1, 256, 256, 3).astype("float32")
restored = model(np.concatenate([page, prompt], axis=-1))   # -> (1, 256, 256, 3)
```

Depth ablation against the Restormer paper default:

```python
big = create_doc_res("restormer_base")    # 26.1M params
```

Overrides are forwarded to the constructor unchanged:

```python
model = create_doc_res("docres", dim=32, num_refinement_blocks=2)
```

An unknown variant raises `ValueError` listing the valid keys.

---

## 7. Constraints you must read before using this

### 7.1 Upstream PyTorch weights CANNOT be loaded into this port

This is the constraint most likely to cost you a wasted experiment, so it is stated
before anything else can go wrong.

`dl_techniques`' `PixelShuffle2D` / `PixelUnshuffle2D` use a **different channel-block
ordering** from `torch.nn.PixelShuffle` (measured — see the `D-007` anchors in
`components.py`). Keras splits the channel axis as `(r, r, C')` so `C'` varies fastest;
PyTorch has `C'` varying slowest. Training from scratch is unaffected, because every
pixel-shuffle site is bracketed by a learnable per-channel-parameterised op that simply
absorbs the fixed permutation — and a test asserts that bracketing rather than trusting
the comment. **Transferring upstream weights is a different matter: the permutation is
then not absorbed, and the result is silently wrong output rather than an error.**

`pretrained=True` therefore raises `NotImplementedError` naming the variant. Train from
scratch; do not write a weight-transfer script against this port without first solving
the ordering.

### 7.2 Height and width must be divisible by 8

The encoder halves resolution three times, so `H % 8 == 0` and `W % 8 == 0`. This model
**raises a `ValueError` naming the offending size rather than padding internally**
(`# DECISION plan-2026-09-08T111844-de235227/D-016`).

Padding is a property of the *inference contract*, not of the network: whoever pads must
also un-pad the prediction, and a network that pads silently cannot tell a caller that it
happened. The reference implementation pads outside the network, in its inference script,
and so does this port: the un-padding lives in the inference shim
`src/train/doc_res/infer_doc_res.py` (added by a later step of the same plan that added
this package; if that file is not there yet, padding is *your* responsibility).

### 7.3 A checkpoint is task-specific; the class is not

Nothing in the saved model records which task it was trained on. Track that yourself.

---

## 8. Component reference

| Name | Module | Role |
|---|---|---|
| `DocRes` | `.model` | The `keras.Model` subclass. **Exported.** |
| `create_doc_res` | `.model` | Module-level factory; delegates to `DocRes.from_variant`. **Exported.** |
| `RestormerTransformerBlock` | `.components` | LN → MDTA → residual; LN → GDFN → residual. Not exported. |
| `RestormerDownsample` | `.components` | `Conv2D(n//2, 3)` → `PixelUnshuffle2D(2)`. Not exported. |
| `RestormerUpsample` | `.components` | `Conv2D(2n, 3)` → `PixelShuffle2D(2)`. Not exported. |

The three components are assembly parts of this one backbone, not a public surface. Import
them from `.components` if you are testing them; nothing outside this package builds a
Restormer block.

---

## 9. Configuration

| Argument | Default | Notes |
|---|---|---|
| `dim` | `48` | Base width. Doubles per level. |
| `num_blocks` | `[2, 3, 3, 4]` | Per-level transformer-block counts. Must have length 4. |
| `num_refinement_blocks` | `4` | Blocks after decoder level 1, at `2*dim` channels. |
| `heads` | `[1, 2, 4, 8]` | MDTA heads per level. Must have length 4. |
| `ffn_expansion_factor` | `2.66` | GDFN hidden width is `int(dim * factor)` — truncated, not rounded. |
| `use_bias` | `False` | Every backbone convolution is bias-free. |
| `input_channels` | `6` | 3 RGB + 3 DTSPrompt. |
| `output_channels` | `3` | Fixed across all five tasks; see §3. |

The `LayerNormalization` inside every block uses `epsilon=1e-5` (Keras' *default* is
`1e-3`, which is not what Restormer uses).

---

## 10. Serialization

Standard Keras 3. `get_config()` round-trips every constructor argument; the class is
registered as `dl_techniques.models.doc_res.model`.

```python
model.save("doc_res.keras")
restored = keras.models.load_model("doc_res.keras")
```

---

## 11. Testing

```bash
.venv/bin/python -m pytest tests/test_models/test_doc_res/ -q
```

Covers the components, the assembled model, an independent analytic parameter-count
oracle (delta 0 against both variants), the six D-015 asymmetries as single-claim guards,
the D-016 divisibility refusal, and a structural assertion that every pixel-shuffle site
is bracketed by a learnable per-channel op — the property §7.1's from-scratch argument
rests on.

---

## 12. Citation

```bibtex
@inproceedings{zhang2024docres,
  title     = {DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks},
  booktitle = {CVPR},
  year      = {2024},
  note      = {author list intentionally omitted here -- take it from the arXiv page below
               rather than from this file}
}

@inproceedings{zamir2022restormer,
  title     = {Restormer: Efficient Transformer for High-Resolution Image Restoration},
  author    = {Zamir, Syed Waqas and Arora, Aditya and Khan, Salman and Hayat, Munawar
               and Khan, Fahad Shahbaz and Yang, Ming-Hsuan},
  booktitle = {CVPR},
  year      = {2022}
}
```

* DocRes — <https://arxiv.org/abs/2405.04408>
* Restormer — <https://arxiv.org/abs/2111.09881>
