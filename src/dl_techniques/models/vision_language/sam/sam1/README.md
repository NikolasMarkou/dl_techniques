# Segment Anything Model (SAM 1) — Keras 3 Implementation

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of the **Segment Anything Model** architecture, after
["Segment Anything"](https://arxiv.org/abs/2304.02643) (Kirillov et al., 2023):
a heavy image encoder run once per image, a cheap prompt encoder, and a mask
decoder light enough to re-run per click.

> **Scope, stated up front.** This package ships the **architecture, its
> forward pass, and a trainable wrapper** — and nothing else.
> - **No pretrained checkpoint ships here or is downloaded.**
>   `SAM.from_variant('vit_b')` builds a randomly initialized model. Every
>   qualitative statement about segmentation quality below describes SAM *as
>   published*, never what these weights produce.
> - **It makes no accuracy claim and no segmentation-quality claim.** Nothing
>   in this repository has trained SAM to any quality. `SAMTrainingModel` is
>   proven only to RUN, with live gradients on an executed `--smoke` run.
> - **No official Meta SAM checkpoint has ever been loaded here.** Two known
>   blockers to ever loading one were removed (the cross-attention internal
>   dim, the MLP head depth), but no key-mapping layer exists and no such load
>   has been demonstrated. Every reference-fidelity statement in this package
>   is an architectural argument, not a measurement against real weights.
> - **`vit_l` and `vit_h` are constructed by tests but forward-passed by none
>   of them.** §7's parameter counts come from a build-only measurement that
>   runs on CPU with no forward pass at all
>   (`test_correctness.py::TestRealVariantForwardPass`, which declares the gap
>   in its own docstring). Do not read that table as evidence they run or train.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Component Reference](#4-component-reference)
5. [The Output Contract](#5-the-output-contract)
6. [Training](#6-training)
7. [Model Variants and Measured Parameter Counts](#7-model-variants-and-measured-parameter-counts)
8. [Serialization, Checkpoints and Deployment](#8-serialization-checkpoints-and-deployment)
9. [Testing](#9-testing)
10. [Known Limitations](#10-known-limitations)
11. [Licensing](#11-licensing)
12. [Citation](#12-citation)

---

## 1. Overview

SAM is a *promptable* segmenter. Instead of predicting a fixed set of classes,
it takes a user prompt — a point, a box, a coarse mask, or any combination —
and returns the mask that prompt refers to. The asymmetry between its three
parts is the whole design: the image encoder is ~96-99% of the parameters and
runs once, so an interactive session pays that cost a single time and then
re-runs the small decoder per click.

| Component | Class | Runs | Role |
|---|---|---|---|
| Image encoder | `ImageEncoderViT` | once per image | ViT + stride-1 channel neck → `(B, 64, 64, 256)` at `img_size=1024` |
| Prompt encoder | `PromptEncoder` | per prompt | points / boxes → sparse embeddings; mask → dense embedding grid |
| Mask decoder | `MaskDecoder` + `TwoWayTransformer` | per prompt | tokens ↔ image, hypernetwork mask head, IoU head |
| Trainable wrapper | `SAMTrainingModel` | training only | traceable path for stock `compile()` / `fit()` |
| Input transform | `resize_longest_side` | before the model | aspect-preserving resize onto `img_size` |

Ambiguity is handled by *proposing*: at `multimask_output=True` the decoder
emits `num_multimask_outputs` masks plus one predicted IoU each, and the caller
picks. That is the published design; §6 records that this repository's training
path supervises those proposals in a way the paper does not.

Every module in this package carries a full docstring with its own measured
caveats block. This README does not duplicate them — where a fact has a home in
a module docstring, the section below points at that module rather than
restating the number in a second place that can drift.

## 2. Architecture

```
image (B, H, W, 3)
  │
  ├─ SAM.preprocess ──────── normalize by pixel_mean / pixel_std, PAD to img_size
  │                          (it pads only — it never resizes; see preprocessing.py)
  ▼
ImageEncoderViT ──────────── patch embed → + absolute PE → depth ViT blocks
  │                          (windowed, except at global_attn_indexes) → neck
  ▼  (B, img_size/16, img_size/16, 256)
  │                       ┌── points / boxes ──► sparse (B, N, 256)
  ├── PromptEncoder ──────┤
  │                       └── mask ────────────► dense  (B, H', W', 256)
  ▼
MaskDecoder ─────────────── [iou_token, mask_tokens, sparse] ↔ image, two-way
  │                          → upscale 4x → hypernetwork dot product
  ▼
low_res_logits (B, N, 4H', 4W')  +  iou_predictions (B, N)
  │
  └─ SAM.postprocess_masks ─ resize to original_size, threshold → masks
```

Per-component detail, including each component's silent-if-ported-wrong
mechanisms, lives in the module docstrings:
[`image_encoder.py`](image_encoder.py), [`prompt_encoder.py`](prompt_encoder.py),
[`transformer.py`](transformer.py), [`mask_decoder.py`](mask_decoder.py),
[`model.py`](model.py), [`preprocessing.py`](preprocessing.py),
[`training_model.py`](training_model.py).

## 3. Quick Start

```python
import keras
from dl_techniques.models.SAM.SAM1 import SAM, resize_longest_side

model = SAM.from_variant('vit_b')          # RANDOM weights — no checkpoint ships

# preprocess() pads only, so pin the longest side yourself first.
image = resize_longest_side(image, model.image_encoder.img_size)

outputs = model({
    'image': image,                                  # (B, H, W, 3)
    'points': (
        keras.ops.convert_to_tensor([[[512.0, 512.0]]]),   # (B, N, 2) x, y
        keras.ops.convert_to_tensor([[1]]),                # (B, N) 1=fg 0=bg -1=pad
    ),
    'original_size': (1024, 1024),
    'multimask_output': True,                        # optional, defaults True
})

best = keras.ops.argmax(outputs['iou_predictions'][0])
mask = outputs['masks'][0, best]                     # uint8 0/1 by default
```

Prompt formats, all optional and combinable in one call:

```python
'points': (coords, labels)   # coords (B, N, 2) float, labels (B, N) int32
'boxes':  boxes              # (B, N, 4) as x1, y1, x2, y2
'masks':  low_res_mask       # (B, 1, 4*H_emb, 4*W_emb) — EXACT size, or it raises
```

## 4. Component Reference

Four construction-time contracts are worth knowing before you configure
anything; each is enforced by a `ValueError` and documented at its class:

- **A windowed-only encoder is refused.** `window_size > 0` with an empty
  `global_attn_indexes` would leave no unit with a global receptive field.
  `use_rel_pos` (default `True`) is a `.keras` **layout** change, not a runtime
  flag — see `image_encoder.py`.
- **`attention_downsample_rate` is also a layout knob** (default `2`,
  matching reference SAM): the three cross-attentions run at
  `embedding_dim // rate` while self-attention stays full width. See
  `transformer.py`.
- **`activation` and `mlp_activation` are two separate knobs whose defaults
  deliberately differ**, and `iou_head_depth` drives both the IoU head and the
  hypernetwork MLPs. Do not collapse either pair. See `mask_decoder.py`.
- **A mask prompt must be exactly `4 * image_embedding_size` in both axes**,
  and a padding point (`label == -1`) has its positional encoding zeroed, not
  merely overwritten. See `prompt_encoder.py`.

`SAM.from_variant` sizes the image encoder only. The prompt encoder and mask
decoder are **variant-independent** — both run at `prompt_embed_dim=256` for
every variant (§7).

## 5. The Output Contract

`SAM.call` returns three keys, and only two of them can be trained.

| Output key | dtype at the default `binarize_masks=True` | Differentiable | Use it for |
|---|---|---|---|
| `masks` | `uint8`, 0/1, at `original_size` | **No** — 0 of N trainable variables receive a gradient | Visualization, evaluation, export |
| `iou_predictions` | float, `(B, N)` | Yes | Mask-quality supervision |
| `low_res_logits` | float, `(B, N, 4H', 4W')` | Yes | **Training** |

A cast to an integer dtype has no gradient, so a trainer that supervises
`outputs['masks']` trains nothing — silently, with no error and a loss that
simply never moves the weights. `binarize_masks=False` makes `masks` carry
float logits instead and is then differentiable; the default `True` matches
reference SAM's thresholded output contract. Either way `low_res_logits` is
what you supervise: it is the resolution reference SAM computes its loss at.

**A dict `y_pred` CAN be trained by stock `fit()`.** Measured on keras 3.8.0,
`CompileLoss.build` raises `KeyError` in exactly one configuration — a
**single** `Loss` object plus a bare-tensor `y_true`. Given
`loss={<output key>: Loss, ...}` (even covering a *subset* of the keys) and a
matching dict `y_true`, `fit()` trains normally and emits per-key metrics.
`y_true`'s keys must match the keys `loss=` covers, not the model's output
keys; an entry for an uncovered key raises
`ValueError: y_true and y_pred have different structures`.

What does block `fit()` through `SAM.call` is different: **`SAM.call` cannot be
traced at all.** `postprocess_masks` runs unconditionally at the end of `call`
and its `ops.image.resize` raises under graph mode, regardless of which output
key you read. That is why the training path drives SAM's submodules directly.

## 6. Training

`SAMTrainingModel` (see [`training_model.py`](training_model.py)) is the
traceable path: `preprocess → image_encoder → prompt_encoder → mask_decoder`,
returning a dict of differentiable tensors. `postprocess_masks` is never
reached. There is no custom `train_step`. The data pipeline and CLI live in
[`src/train/sam/`](../../../../train/sam/).

Four things about the shipped loss and training loop are measured here and
recorded nowhere else:

1. **The repository's segmentation losses are reused, but not as-is, and the
   trap is silent.** `SegmentationLosses.focal_loss` on a 1-channel binary map
   is *bit-identically blind to negative pixels*: setting every negative
   pixel's prediction to maximally wrong leaves the loss unchanged to six
   decimals. `dice_loss` is sound behind a `(B*M, h, w, 1)` reshape but
   **silently accepts the raw `(B, M, h, w)` layout** and returns a plausible
   number while reducing `(num_masks, height)` instead of `(height, width)`.
   `SAMMaskLoss` adapts both (2-channel one-hot for focal, channels-last
   single-channel for dice) and **refuses** the raw layout. Do not route SAM
   masks through `SegmentationWrapperLoss` directly.
2. **The focal:dice mix was re-derived on this code, not pasted from the
   paper.** Unweighted on this repository's implementations dice is ~8x focal;
   at the paper's 20:1 focal leads by ~2.4x in loss value and ~10.5:1 in
   gradient magnitude. 20:1 ships because the two readings agree in order of
   magnitude, and the tests assert those two structural facts rather than the
   constants.
3. **The IoU target is packed next to the prediction, and that is forced.** The
   achieved IoU depends on the prediction *and* the ground truth, so it exists
   only inside `call()`; a `tf.data` pipeline cannot produce it and stock
   `compile(loss={...})` hands each loss only its own key. Hence the
   `iou_supervision` output, shape `(B, M, 2)`: `[..., 0]` predicted,
   `[..., 1]` achieved under `stop_gradient`. It appears only when `gt_mask` is
   in the inputs.
4. **Iterative refinement saturates early.** Measured at the reduced test
   fixture geometry (`img_size=256`, `embed_dim=64`, `depth=4`), variable
   coverage saturates at **2** rounds — 170 of 201 trainable variables moved at
   1 round, 181 of 201 at 2 — while the class default is **1** (refinement off,
   which keeps value-exact equivalence with an eager `SAM.call`) and the
   shipped trainer passes 3.

Initial prompts are DATA and come from the `tf.data` pipeline; refinement
prompts depend on the current prediction and therefore cannot.

## 7. Model Variants and Measured Parameter Counts

The per-variant parameter table is **measured from this package** and has one
home: the `Model Variants:` block of [`model.py`](model.py)'s module docstring,
which also carries the reproduction recipe. It is not restated here.

Two readings of it are worth stating in prose:

- The prompt encoder and mask decoder are **variant-independent** — only the
  image encoder scales, and it is 95.7% / 98.7% / 99.4% of the total at
  `vit_b` / `vit_l` / `vit_h` respectively.
- Those figures are this package's own, and this package's deliberate layout
  deviations move them relative to reference SAM. Do not compare them to
  reference-PyTorch numbers without accounting for the
  `attention_downsample_rate=2` cross-attentions and the 3-layer MLP heads.

**Inference latency and per-variant memory footprint are NOT measured anywhere
in this package.** No inference benchmark ships here and none was ever run, so
no latency or GB-per-variant figure for `vit_b` / `vit_l` / `vit_h` appears
under `models/SAM/`. Treat any such number you find elsewhere for these variants
as unrelated to this implementation. (The two memory figures that DO appear —
the 6,754.5 MiB `vit_h` forward peak in `test_correctness.py` and the COCO
wall-clock in §10 — are a test-cost measurement and a data-pipeline
measurement, not inference benchmarks.)

## 8. Serialization, Checkpoints and Deployment

> **Build the model before you save it.** `SAM.from_variant(...)` returns an
> unbuilt model whose sub-layer variables do not exist yet; saving it produces
> Keras' "you are saving a model that has not yet been built" warning and an
> archive carrying no weights. Call the model once on real inputs first. On
> load, `SAM.build_from_config` runs a full-resolution dummy forward to
> materialize the lazily built variables BEFORE the archive is restored (at
> `vit_h` that is a 1024x1024 forward through 630M+ parameters); a cheaper
> `build()`-only path was measured and rejected because it materializes only
> part of the weight list and leaves the rest silently re-initialized.

```python
model = SAM.from_variant('vit_b')

# BUILD FIRST — an unbuilt save stores no weights.
model({
    'image': keras.random.normal(shape=(1, 1024, 1024, 3)),
    'points': (keras.ops.convert_to_tensor([[[512.0, 512.0]]]),
               keras.ops.convert_to_tensor([[1]])),
    'original_size': (1024, 1024),
})

model.save('sam_model.keras')
```

### Loading a checkpoint written before this package moved — registrar-first

This package used to live at a different dotted path. A `.keras` file records
the module path its classes lived in *when it was saved*, and Keras resolves a
class by looking in its registry FIRST and only then falling back to
`importlib.import_module` on that recorded string. That fallback now raises for
every SAM checkpoint written before the move:

```
TypeError: Could not deserialize class 'SAMTrainingModel' because its parent
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
import dl_techniques.models.SAM.SAM1.training_model   # registrar-first

model = keras.models.load_model(path, compile=False)
print(type(model).__name__, len(model.weights))
# SAMTrainingModel 202
```

Note the import names the **module**, not the package. For SAM 1 the package
import happens to be sufficient (`SAM1/__init__.py` imports `training_model`),
but that is not true of every sibling — SAM 2's `__init__` does not, so
`import ...SAM.SAM2` alone still fails. Importing the defining module is the
rule that holds for all three.

### TensorFlow Serving

```bash
python -c "
import keras
from dl_techniques.models.SAM.SAM1 import SAM

model = SAM.from_variant('vit_b')
# Build model...
model.export('sam_serving/1')
"

# Serve with TensorFlow Serving
docker run -p 8501:8501 \
  --mount type=bind,source=$(pwd)/sam_serving,target=/models/sam \
  -e MODEL_NAME=sam \
  tensorflow/serving

# Query
curl -X POST http://localhost:8501/v1/models/sam:predict \
  -d '{
    "inputs": {
      "image": [...],
      "points": [...],
      "original_size": [1024, 1024]
    }
  }'
```

The two container paths in the `docker run` and `curl` lines above are
TensorFlow Serving's **container-internal** model directory and REST endpoint.
They are not repository paths, they do not track this package's location, and
they must not be rewritten when it moves.

## 9. Testing

Gate:

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest \
    tests/test_models/test_sam/ \
    tests/test_train/test_sam/ \
    tests/test_losses/test_sam_mask_loss.py -q
```

Run it as **one** invocation. The single-path `tests/test_models/test_sam/`
form collects materially fewer tests and reports a green that covers neither
the training-data path nor the loss traps; no count is quoted here because a
count in a README rots the day it is written.

What each path holds: `test_models/test_sam/test_model.py` (shape / dtype / API
coverage), `test_correctness.py` (guards each proven RED against a deliberate
re-break — per-output-key gradient counts, the rel-pos interpolation branch,
padding-point invariance, weight-count plus value-exact `.keras` round-trip,
and the tiling / mask-shape / oversize / degenerate-encoder refusals),
`test_training_model.py` (the `SAM.call` spy, the moved-variable
decomposition, the dead-component probe, refinement liveness),
`test_train/test_sam/` (data sources, prompt-sampling properties, the
args→config wiring), and `test_losses/test_sam_mask_loss.py` (the
focal-blindness and raw-layout traps of §6.1). `tests/test_datasets/test_coco_instances.py`
covers the COCO source separately.

## 10. Known Limitations

None of these is fixed by anything in this package; they are stated so nobody
rediscovers them.

- **No accuracy or segmentation-quality claim.** The executed `--smoke` run
  proves the path runs with live gradients and a falling loss on 32 synthetic
  instances. It says nothing about mask quality.
- **No official Meta SAM checkpoint has ever been loaded**, and no key-mapping
  layer exists.
- **`vit_l` / `vit_h` are never forward-passed by any test**, and the trainer
  has never been run at those widths. Its default and only executed
  configuration is a reduced-WIDTH `tiny` geometry at real SAM patch/window
  geometry.
- **`multimask_output=True` runs, but its objective is an approximation.** All
  proposals are supervised against the same single ground-truth mask, because
  the pipeline emits one GT per instance and `SAMMaskLoss` repeats it across
  the mask axis. The paper's "back-propagate only the minimum loss over the
  masks" rule is **not** implemented. The default is `False`.
- **`SAM.call` cannot be traced**, so no traced pipeline can consume
  `outputs['masks']`. That is a property of `postprocess_masks`, not something
  the wrapper hides.
- **At `num_refinement_rounds > 1` the `.keras` round-trip is value-exact only
  on the round-1 slice.** Later rounds depend on a random draw whose generator
  state advanced differently. Asserted in both directions rather than papered
  over.
- **The shipped COCO path does not preprocess the way §3 does.**
  `resize_longest_side` has **zero production consumers** — under `src/` it
  appears only as its definition, its export, the two error messages naming it
  as the remedy, and documentation. `train.sam.data`'s COCO source instead
  squashes each image to a square, so a non-square COCO image reaches the model
  with a distorted aspect ratio. Mask, box and image are derived from the same
  resized frame, but **no test asserts they still agree after a strongly
  non-square source image**, so an asymmetric resize on one of the three would
  go uncaught.
- **`--data-source coco` is I/O-bound, not GPU-bound.** One epoch at
  `--steps-per-epoch 16` on one machine read ~1 s on the synthetic source
  against ~18 s on COCO. No test pins that ratio and nothing else in the
  repository records it, so treat it as an order-of-magnitude budgeting hint and
  re-derive it on your own hardware — the two runs differ only in
  `--data-source`.

## 11. Licensing

This package is part of `dl_techniques` and is **GPL-3.0**, like the rest of
the repository. It is an independent Keras 3 implementation of the
architecture described by Kirillov et al. (2023); the original SAM was
developed by Meta AI Research, and no code and no weights from that release
are vendored here.

Its two siblings carry a **stricter constraint that binds any future work on
them**: SAM 2's and SAM 3's released code is under the SAM License, which is
incompatible with this repository's GPL-3.0. Both packages were therefore
reimplemented from published numbers and a re-read reference, never
transliterated — see [`../SAM2/README.md`](../SAM2/README.md) and
[`../SAM3/README.md`](../SAM3/README.md) before extending either.

## 12. Citation

```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

```bibtex
@inproceedings{dosovitskiy2021image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

```bibtex
@inproceedings{liu2021swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10012--10022},
  year={2021}
}
```
