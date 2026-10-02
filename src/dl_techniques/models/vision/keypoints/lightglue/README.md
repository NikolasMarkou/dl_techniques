# LightGlue: Local Feature Matching

A Keras 3 implementation of **LightGlue** (Lindenberger, Sarlin & Pollefeys, 2023), a
transformer that matches the keypoints of two images. It consumes keypoint positions and one
descriptor per keypoint from any extractor (SuperPoint, SIFT, DISK, ...) and returns, for
every keypoint, a soft assignment to the keypoints of the other image or to a dustbin.

---

## 1. Overview

Each of `num_layers` layers runs three steps on the pair of images:

1. **Self-attention** inside each image, with a **learned-Fourier rotary position encoding**:
   the normalised keypoint coordinates are projected by a bias-free linear map to
   `head_dim / 2` phases, and the cosine and sine of those phases rotate adjacent channel pairs
   of every query and key. The table is computed once per image and shared by all layers.
2. **Cross-attention** between the images. The similarity matrix is computed once and softmaxed
   along both axes, so one matrix gives both directions of message passing.
3. A **match assignment** head: a double log-softmax of the descriptor similarity plus per-point
   matchability logits, with an extra dustbin row and column. Layers `0 .. L-2` also get a
   **token confidence** head that predicts whether a keypoint's assignment is already final.

### Static path versus adaptive inference

`LightGlue.call()` is the **static, fully masked path**. It always runs every layer, prunes
nothing, and returns the per-layer assignments, so it can be trained, jit-compiled and
batched. Padded keypoint slots are marked by the optional `mask0` / `mask1` inputs and take no
part in the result: padding a batch changes nothing for the real keypoints.

The paper's adaptive behaviour (early exit when the confident fraction exceeds
`depth_confidence`, pruning of unmatchable points with `width_confidence`) changes tensor
shapes at run time. It lives in a separate **eager** method, `LightGlue.match(inputs)`, which
handles one image pair per call (batch size 1) and returns numpy values. The constructor
stores `depth_confidence`, `width_confidence` and `pruning_min_kpts` for it; `call()` ignores
all three. With both knobs disabled (`-1`), `match()` equals the last layer of `call()` (a test
pins that).

| | `call(inputs)` | `match(inputs)` |
| :--- | :--- | :--- |
| Layers run | all `num_layers` | stops early when `depth_confidence` is met |
| Pruning | none | drops unmatchable points with `width_confidence` |
| Batch | any `B`, padding via `mask0` / `mask1` | `B == 1` (raises `ValueError` otherwise) |
| Graph / jit / `fit` | yes, this is the training path | no, eager numpy control flow |
| Returns | tensors, per-layer log assignments and confidences | numpy: matches, scores, `stop`, `matches` pairs, `prune0/1` |

### One size, no variants

The paper publishes a single configuration (9 layers, width 256, 4 heads), and the defaults
equal it. There is therefore no `MODEL_VARIANTS` table and no `from_variant`: a table with one
row would be a placeholder. No pretrained weights are distributed with this repository, so
`create_lightglue` has no `pretrained` argument (see "Pretrained weights" below).

### Data flow

```
keypoints0/1 (B,N,2) pixels, image_size0/1 (B,2) = (w,h)        descriptors0/1 (B,N,D)
      |  normalise per image: (p - size/2) / (max(size)/2)             |  stop_gradient
      v                                                               v
posenc: learned-Fourier rotary table (2,B,1,N,Dh)              input_proj (Dense, or identity
      |                                                         when input_dim == descriptor_dim)
      |       +------------ layer i = 0 .. L-1 ------------+          |
      +------>| self block  on image 0, on image 1        |<---------+
              | cross block between the images            |
              | MatchAssignment -> log_assignments[i]     |
              | MatchTokenConfidence -> confidences[i]    |
              +--------------------------------------------+
                          |
              filter_matches(log_assignments[L-1])
```

The descriptors are detached before `input_proj`, as in the reference implementation:
LightGlue does not train the extractor.

---

## 2. Contents

| File | Contents |
| :--- | :--- |
| `model.py` | `LightGlue` (the `keras.Model`), `create_lightglue` (factory), `normalize_keypoints`, and the pure helpers of `match()`: `confidence_threshold`, `get_pruning_mask`, `check_if_stop`. |
| `__init__.py` | Re-exports `LightGlue` and `create_lightglue` under `__all__`. |

The building blocks are generic and live in `src/dl_techniques/layers/matching/`. That package
exports nothing: import from the module, e.g.
`from dl_techniques.layers.matching.match_assignment import MatchAssignment`.

| File | Contents |
| :--- | :--- |
| `learned_fourier_rotary.py` | `LearnedFourierRotaryEncoding`, `apply_rotary_interleaved` |
| `lightglue_blocks.py` | `LightGlueSelfBlock`, `LightGlueCrossBlock` |
| `match_assignment.py` | `MatchAssignment`, `filter_matches` |
| `token_confidence.py` | `MatchTokenConfidence` |

Training and evaluation code is not in this package. It is `src/train/lightglue/` (see its
`README.md`), which uses the loss `dl_techniques.losses.lightglue_loss.LightGlueLoss`, the
metric `dl_techniques.metrics.keypoint_matching.KeypointMatchMetric`, and the utilities
`dl_techniques.utils.keypoint_extraction` (batched SuperPoint decode) and
`dl_techniques.utils.keypoint_matching` (homography ground-truth labels).

Tests: `tests/test_models/test_lightglue/` (model, numpy-oracle parity, shared-oracle adoption)
and `tests/test_layers/test_matching/`. The parity oracle is `tests/lightglue_reference_numpy.py`,
an explicit-loop transcription of the official reference forward.

---

## 3. Component Reference

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`LightGlue`** | `models.lightglue.model.LightGlue` | The subclassed `keras.Model`, static masked path. |
| **`create_lightglue`** | `models.lightglue.model.create_lightglue` | Factory with the published size as defaults. |
| **`normalize_keypoints`** | `models.lightglue.model.normalize_keypoints` | Pixel to normalised coordinates, in float32 under any dtype policy. |

### Constructor parameters

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `input_dim` | `256` | Width of the incoming descriptors. A `Dense` projection is created when it differs from `descriptor_dim`. |
| `descriptor_dim` | `256` | Internal node width; divisible by `num_heads` with an even head width. |
| `num_layers` | `9` | Number of layers (at least 1). |
| `num_heads` | `4` | Attention heads. |
| `filter_threshold` | `0.1` | Match threshold on `exp(score)` for the final `filter_matches`. |
| `depth_confidence` | `0.95` | Early-exit confidence of the adaptive path; `-1` disables. Ignored by `call()`. |
| `width_confidence` | `0.99` | Pruning confidence of the adaptive path; `-1` disables. Ignored by `call()`. |
| `pruning_min_kpts` | `-1` | `match()` prunes an image only while it has MORE than this many points alive; `-1` always prunes (the reference's CPU setting, the reference uses 1024 on a GPU). Ignored by `call()`. |
| `add_scale_ori` | `False` | Append keypoint scale and orientation to the position encoding (feature width 4). |
| `gamma` | `1.0` | Positional-encoding frequency initialiser scale (kernel std `gamma ** -2`). |

### Inputs

A dict:

| Key | Shape | Meaning |
| :--- | :--- | :--- |
| `keypoints0`, `keypoints1` | `(B, M, 2)`, `(B, N, 2)` | Pixel `(x, y)`. |
| `descriptors0`, `descriptors1` | `(B, M, input_dim)`, `(B, N, input_dim)` | One descriptor per keypoint. |
| `image_size0`, `image_size1` | `(B, 2)` | `(width, height)`; sets the coordinate normalisation. |
| `mask0`, `mask1` (optional) | `(B, M)`, `(B, N)` | 1 = real keypoint, 0 = padding. Default all real. |
| `scales0/1`, `oris0/1` (only with `add_scale_ori`) | `(B, M)`, `(B, N)` | Keypoint scale and orientation. |

### Outputs

A dict. Every per-layer tensor is batch-first so `predict()` can concatenate batches:

| Key | Shape | Meaning |
| :--- | :--- | :--- |
| `log_assignments` | `(B, L, M+1, N+1)` | Per-layer log assignment; the dustbin row and column are the last index. Padded rows and columns are 0. Float32 or wider. |
| `token_confidences0`, `token_confidences1` | `(B, L-1, M)`, `(B, L-1, N)` | Per-layer confidence that a keypoint is already decided. |
| `matches0`, `matches1` | `(B, M)`, `(B, N)` | Final-layer mutual matches: partner index, or `-1`. Int32. |
| `matching_scores0`, `matching_scores1` | `(B, M)`, `(B, N)` | `exp(score)` of mutual pairs, 0 elsewhere. |

`log_assignments` is not a normalised distribution; see the `MatchAssignment` docstring.

### Pretrained weights and conversion from the official PyTorch checkpoint

**No pretrained weights are distributed here**, and no converter script is shipped; the
trainer in `src/train/lightglue/` produces weights for the in-repo SuperPoint. A checkpoint
of the official PyTorch model can be loaded by the following rule, which is exactly what the
test-only helper `tests/test_models/test_lightglue/weight_loading.py::load_torch_weights`
does and what the parity tests exercise against the numpy oracle:

- every `Linear` weight is copied **transposed** (torch `(out, in)` to Keras `(in, out)`);
- every bias, and every LayerNorm `weight` / `bias` (to `gamma` / `beta`), is copied as is;
- there is **no permutation**: the fused `Wqkv` keeps the torch `(heads, head_dim, 3)` column
  layout, and the sublayers are named like the torch state dict (`input_proj`, `posenc.Wr`,
  `transformers.i.self_attn.*`, `transformers.i.cross_attn.*`, `log_assignment.i.*`,
  `token_confidence.i.*`).

The mapping has been exercised against the numpy transcription of the reference code, not
against a real downloaded checkpoint (PyTorch is not installed in this environment).

### Why there is no `MODEL_VARIANTS`

See "One size, no variants" above: a single published configuration does not make a table.

---

## 4. Quick Start

```python
import keras
import numpy as np
from dl_techniques.models.vision.keypoints.lightglue import create_lightglue

rng = np.random.default_rng(0)
B, N = 2, 64
inputs = {
    "keypoints0": rng.uniform(0, 128, (B, N, 2)).astype("float32"),   # pixels (x, y)
    "keypoints1": rng.uniform(0, 128, (B, N, 2)).astype("float32"),
    "descriptors0": rng.normal(size=(B, N, 256)).astype("float32"),
    "descriptors1": rng.normal(size=(B, N, 256)).astype("float32"),
    "image_size0": np.full((B, 2), 128.0, "float32"),                  # (w, h)
    "image_size1": np.full((B, 2), 128.0, "float32"),
    "mask0": np.ones((B, N), "float32"),                               # optional, 1 = real
    "mask1": np.ones((B, N), "float32"),
}

model = create_lightglue(input_dim=256)

# Static masked path: batched, graph-safe, what the trainer calls.
out = model(inputs)
out["matches0"]            # (B, N) int32, -1 = unmatched
out["log_assignments"]     # (B, 9, N+1, N+1)

# Adaptive eager path: one pair, early exit and pruning.
single = {k: v[:1] for k, v in inputs.items()}
res = model.match(single)
res["matches0"], res["stop"]   # (1, N) numpy, 1-based layer whose assignment was used
res["matches"]                 # (S, 2) index pairs
```

With random weights the matches carry no meaning; the snippet only shows the call contract.
Training is described in `src/train/lightglue/README.md`.

---

## 5. Serialization

`LightGlue` is registered with `@register_dl_technique("dl_techniques.models.lightglue.model")`
(the `vision/` family and `keypoints/` subfamily directories are stripped from the package
string) and round-trips through the `.keras` format:

```python
model.save("lightglue.keras")
restored = keras.models.load_model("lightglue.keras")
```

`get_config` emits every constructor parameter. The model sets `autocast=False` so that pixel
keypoints and the position-encoding kernel stay float32 under a mixed dtype policy; each
sublayer casts its own inputs.

---

## 6. References

- Lindenberger, Sarlin, Pollefeys. *LightGlue: Local Feature Matching at Light Speed.*
  ICCV 2023. https://arxiv.org/abs/2306.13643
- Sarlin, DeTone, Malisiewicz, Rabinovich. *SuperGlue: Learning Feature Matching with Graph
  Neural Networks.* CVPR 2020. https://arxiv.org/abs/1911.11763
- Li, Si, Li, Hsieh, Bengio. *Learnable Fourier Features for Multi-Dimensional Spatial
  Positional Encoding.* NeurIPS 2021. https://arxiv.org/abs/2106.02795
- Su, Lu, Pan, Murtadha, Wen, Liu. *RoFormer: Enhanced Transformer with Rotary Position
  Embedding.* 2021. https://arxiv.org/abs/2104.09864 (the rotary rule the encoding applies)
- DeTone, Malisiewicz, Rabinovich. *SuperPoint: Self-Supervised Interest Point Detection and
  Description.* CVPRW 2018. https://arxiv.org/abs/1712.07629 (the detector the trainer uses;
  see `models/vision/keypoints/superpoint/`)
- Official implementation: https://github.com/cvg/LightGlue. Training code and the
  homography benchmark definitions the trainer and evaluation follow:
  https://github.com/cvg/glue-factory.
