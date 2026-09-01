# FractalNet: Ultra-Deep Neural Networks without Residuals

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of **FractalNet**, a deep classifier that reaches great depth
without a single residual connection. Depth comes from a recursive expansion rule that
generates an exponential number of paths of many different lengths, and the paths are
regularized by dropping them at random during training ("drop-path").

The model is a short stack of `FractalBlock` stages over a plain `ConvBlock` base unit,
followed by a global pool and a linear classifier.

---

## 1. Overview: What is FractalNet and Why It Matters

### What is FractalNet?

FractalNet questions whether identity shortcuts are the essential ingredient in training
very deep networks. Its answer is that what matters is having **short paths from input to
loss alongside the long ones**, and that a recursive rule supplies those directly:

```
F_1(x) = block(x)
F_k(x) = join( F_{k-1}(F_{k-1}(x)) , block(x) )
```

The deep branch **composes** two depth-`k-1` fractals — the second consumes the first's
output — while the shallow branch is a single base block on the same input. Composition,
not parallelism, is what makes the longest path `2^(k-1)` blocks long while the shortest
stays exactly 1.

### Key Ideas

1. **Recursive expansion.** One rule, applied `k` times, produces a self-similar block
   holding `2^k - 1` base blocks (`L(1)=1`, `L(k)=2*L(k-1)+1`). No weights are shared;
   every leaf is an independent instance.
2. **Implicit ensembling.** Every route from input to output is a valid, shallower
   network. The number of distinct routes follows `P(1)=1`, `P(k)=P(k-1)^2 + 1`
   (1, 2, 5, 26, 677) — super-exponential in the depth.
3. **Drop-path.** During training each input to a join is dropped by its own Bernoulli
   draw, so a different sub-network is trained at each step. At inference every path is
   live and the joins average them.

### How This Implementation Differs From the Paper

This package implements the paper's expansion rule and its **local** drop-path. Several
things in Larsson et al. (2017) are deliberately not here, and the README describes the
code rather than the paper:

| Paper | This package |
| :--- | :--- |
| Local **and** global drop-path, sampled 50/50 per mini-batch (global picks one column and runs the whole network through it) | Local drop-path only. There is no global mode and no alternation. |
| A drop-path schedule / per-depth rates | A single `drop_path_rate` applies at every join and every depth, with no schedule. |
| CIFAR configuration of 5 blocks x 4 columns with filters 64/128/256/512/512 | Four repo-defined variants (`micro`/`small`/`medium`/`large`), none of which reproduces the paper's configuration. |
| Base unit ordered Conv -> Dropout -> BatchNorm -> ReLU | `ConvBlock` is ordered Conv -> Norm -> Activation -> Dropout, with the normalization and activation configurable. |

Nothing in this repository has been trained or benchmarked, so no accuracy number here is
this implementation's; the paper's results belong to the paper.

---

## 2. The Problem FractalNet Solves

Plain deep networks suffer from vanishing gradients and from *degradation*: past a certain
depth, adding layers increases even the training error. ResNet solved this by making every
block skippable, which creates an implicit short path around each one.

FractalNet takes the same insight — a very deep network is trainable when short paths
coexist with long ones — and supplies the short paths explicitly instead of implicitly:

- The recursion creates paths of length `1, 2, 4, ..., 2^(k-1)` all reaching the same join.
- The shortest path is a single convolution no matter how deep the block is. That path is
  what trains early; the long path is what the short path teaches.
- Drop-path stops the network from collapsing onto any one of them.

---

## 3. How FractalNet Works: Core Concepts

The model is a plain sequence of stages. Each stage is one `FractalBlock` at **constant
resolution**, followed by max-pooling when that stage's stride exceeds 1. There is no stem
and no bottleneck.

```
Input (B, H, W, 3)
  │
  ├─► Stage 0: FractalBlock(depth=D0, filters=F0)  ──► MaxPool(2)  ──► (B, H/2, W/2, F0)
  │
  ├─► Stage 1: FractalBlock(depth=D1, filters=F1)  ──► MaxPool(2)  ──► (B, H/4, W/4, F1)
  │
  ├─► Stage 2: FractalBlock(depth=D2, filters=F2)  ──► MaxPool(2)  ──► (B, H/8, W/8, F2)
  │
  ├─► GlobalAveragePooling2D
  ├─► Dropout(classifier_dropout_rate)
  └─► Dense(num_classes)   ──►  RAW LOGITS (no softmax)
```

Downsampling happens **between** blocks, never inside one. The deep branch applies its
base block `2^(k-1)` times, so a stride inside the block would shrink the deep branch
`2^(k-1)` times against the shallow branch's once and the join would receive mismatched
shapes. `FractalBlock` rejects a strided `block_config` with a `ValueError` for that
reason.

---

## 4. Architecture Deep Dive

### 4.1 `FractalBlock` (the recursive engine)

```
              input
         ┌──────┴───────┐
    DEEP │              │ SHALLOW
         ▼              ▼
   FractalBlock      ONE base
   depth = k-1        block
         │              │
         ▼              │
   FractalBlock         │   (composed: consumes
   depth = k-1          │    the first's OUTPUT)
         └──────┬───────┘
                ▼
   drop-path join: mean over the SURVIVING branches
                ▼
              output
```

- **Base case (`depth=1`)**: a single `ConvBlock`. The recursion bottoms out here.
- **Recursive case**: `deep_second(deep_first(x))` against `shallow(x)`, joined under
  drop-path. The two depth-`k-1` blocks are composed, **not** run in parallel over the
  same input.
- **Cost**: `2^k - 1` leaf convolutions, so parameters and FLOPs grow exponentially in
  `depth`. The shipped variants stay at depth <= 5 for that reason.

The composition is easy to get wrong invisibly: parameter count, layer count and output
shape are identical under the parallel form, and only the receptive field distinguishes
them (with 3x3 `same` convolutions a correct depth-`k` block spans `1 + 2*2^(k-1)` pixels
— 3, 5, 9, 17). That measurement is pinned by
`tests/test_layers/test_fractal_block.py::TestFractalExpansionRule`.

### 4.2 `ConvBlock` (the base unit)

The leaf of the fractal tree: `Conv2D` -> normalization -> activation -> dropout, with the
normalization type (`batch_norm` by default) and activation (`relu`) selected by string.
`FractalNet` constructs one `ConvBlock` per stage purely to harvest its `get_config()`;
that dict is what `FractalBlock` stores and re-instantiates per leaf, which is what makes
the recursive structure serializable and why every leaf in a stage is configured
identically while holding independent weights.

### 4.3 Local drop-path (inside `FractalBlock._join`)

Drop-path is inline in the join, not a `StochasticDepth` layer.

- **Training**: each of the two inputs is dropped by its own per-sample Bernoulli draw with
  probability `drop_path_rate`, and the join averages only the **survivors** — a mean over
  a varying number of paths, not a fixed `0.5` scaling.
- **Both dropped**: one branch is revived by a fair coin, so at least one path is always
  live and the join is never a zero map. Without that rescue a join emits exactly zero for
  the sample, at rate `drop_path_rate ** 2` (about 2.3% at the 0.15 default), and the zero
  then propagates through every remaining stage.
- **Inference** (or `drop_path_rate == 0.0`): the plain mean of the two branches.

Each block owns a `keras.random.SeedGenerator`, so the draws are reproducible under a
seeded run and independent between blocks.

---

## 5. Quick Start Guide

```python
import numpy as np

from dl_techniques.models.vision.fractalnet.model import create_fractal_net

# Build and compile a FractalNet-Small for CIFAR-10.
# The factory compiles with SparseCategoricalCrossentropy(from_logits=True),
# which matches the head's raw-logit output.
model = create_fractal_net(
    variant="small",
    num_classes=10,
    input_shape=(32, 32, 3),
    learning_rate=1e-3,
)
model.summary()  # 1,033,194 parameters

images = np.random.rand(16, 32, 32, 3).astype("float32")
labels = np.random.randint(0, 10, 16)

# Drop-path is active here.
model.fit(images, labels, epochs=1, verbose=1)

# Drop-path is off here: every path is live and the joins average them.
predictions = model.predict(images)
print(predictions.shape)  # (16, 10)
```

---

## 6. Component Reference

| Component | Location | Purpose |
| :--- | :--- | :--- |
| `FractalNet` | `dl_techniques.models.vision.fractalnet.model` | The Keras `Model` that assembles the fractal stages and the head. |
| `create_fractal_net` | `dl_techniques.models.vision.fractalnet.model` | Builds a variant **and compiles it**. Use this unless you want to compile yourself. |
| `FractalBlock` | `dl_techniques.layers.fractal_block` | The recursive block. Drop-path lives in its `_join`. |
| `ConvBlock` | `dl_techniques.layers.standard_blocks` | The base-case unit at the leaves. |

### Key constructor arguments

| Argument | Default | Meaning |
| :--- | :--- | :--- |
| `num_classes` | `10` | Head width. Only used when `include_top=True`. |
| `depths` | `(2, 3, 3)` | Fractal expansion level per stage — **not** a block count. |
| `filters` | `(32, 64, 128)` | Channels per stage. Must match `depths` in length. |
| `strides` | `(2, 2, 2)` | Max-pool stride after each stage. Must match `filters` in length. |
| `drop_path_rate` | `0.15` | Per-branch drop probability at every join. |
| `dropout_rate` | `0.1` | Dropout inside each `ConvBlock`. |
| `normalization_type` | `"batch_norm"` | Any type the normalization factory accepts. |
| `activation_type` | `"relu"` | Activation inside each `ConvBlock`. |
| `global_pool` | `"avg"` | `"avg"` or `"max"`. Anything else raises `ValueError`. |
| `classifier_dropout_rate` | `0.2` | Dropout before the final `Dense`. |
| `include_top` | `True` | `False` returns the final stage's feature map. |
| `input_shape` | `(32, 32, 3)` | Must be 3D. |

`create_fractal_net` adds `variant`, `optimizer` (default `"adam"`), `learning_rate`
(default `0.001`), `loss` and `metrics`, and forwards everything else to the constructor.
There are no pretrained weights in this package.

---

## 7. Configuration & Model Variants

`FractalNet.MODEL_VARIANTS` holds four repo-defined presets. They are **not** the paper's
configurations. Parameter counts below were measured at `input_shape=(32, 32, 3)` with
`num_classes=10`.

| Variant | `depths` | `filters` | Leaf convolutions | Parameters |
| :--- | :--- | :--- | :--- | ---: |
| `micro` | `[1, 2, 2]` | `[16, 32, 64]` | 1 + 3 + 3 | 94,762 |
| `small` | `[2, 3, 3]` | `[32, 64, 128]` | 3 + 7 + 7 | 1,033,194 |
| `medium` | `[3, 4, 4]` | `[64, 128, 256]` | 7 + 15 + 15 | 9,770,890 |
| `large` | `[4, 5, 5]` | `[96, 192, 384]` | 15 + 31 + 31 | 48,301,162 |

A depth-`k` stage holds `2^k - 1` leaf convolutions, which is why the parameter count grows
roughly an order of magnitude per variant.

---

## 8. Usage Examples

### Example 1: FractalNet as a feature-extraction backbone

```python
import numpy as np

from dl_techniques.models.vision.fractalnet.model import FractalNet

backbone = FractalNet.from_variant(
    "medium",
    include_top=False,
    input_shape=(256, 256, 3),
)

features = backbone.predict(np.random.rand(2, 256, 256, 3).astype("float32"))
print(features.shape)  # (2, 32, 32, 256) -- three stride-2 stages downsample by 8
```

### Example 2: A custom four-stage architecture

`depths`, `filters` and `strides` must all have the same length — adding a stage without
extending `strides` raises `ValueError`.

```python
from dl_techniques.models.vision.fractalnet.model import FractalNet

custom = FractalNet(
    num_classes=50,
    depths=[2, 3, 4, 2],
    filters=[32, 64, 128, 256],
    strides=[2, 2, 2, 2],       # one entry per stage
    input_shape=(128, 128, 3),
    drop_path_rate=0.2,
)
print(custom.count_params())  # 3,339,282
```

---

## 9. Advanced Usage Patterns

### Tuning `drop_path_rate`

It is the architecture's main regularizer. Lower it (`0.0 - 0.1`) for small datasets or
shallow variants; raise it (`0.15 - 0.25`) for deep variants, where the long paths
otherwise go undertrained.

```python
from dl_techniques.models.vision.fractalnet.model import create_fractal_net

model = create_fractal_net(
    "large",
    num_classes=1000,
    input_shape=(224, 224, 3),
    drop_path_rate=0.25,
)
```

Remember the two regimes this creates: `fit()` trains a randomly sampled sub-network at
each step, while `predict()` and `evaluate()` run every path and average at the joins.

---

## 10. Performance Optimization

Mixed precision works out of the box — the model is ordinary convolutions and Keras
handles loss scaling inside `fit()`:

```python
import keras

from dl_techniques.models.vision.fractalnet.model import create_fractal_net

keras.mixed_precision.set_global_policy("mixed_float16")
model = create_fractal_net("medium", num_classes=100)
```

Memory scales with `2^depth`, not with `depth`. A stage at `depth=6` costs twice a stage at
`depth=5`; prefer more stages over deeper ones when a run does not fit.

---

## 11. Training and Best Practices

- **Optimizer**: the paper used SGD with Nesterov momentum; AdamW also works. The factory
  defaults to Adam.
- **Schedule**: cosine or step decay, optionally with a few warmup epochs.
- **Loss**: the head emits raw logits. `create_fractal_net` defaults to
  `SparseCategoricalCrossentropy(from_logits=True)`; if you compile the model yourself, do
  not pass the string `"sparse_categorical_crossentropy"`, which is `from_logits=False` and
  mis-trains silently.
- **Augmentation**: standard CNN augmentation — random flips, padded random crops, and
  Cutout or Mixup.

---

## 12. Serialization & Deployment

`FractalNet`, `FractalBlock` and `ConvBlock` are registered and implement `get_config`, so
the `.keras` format round-trips without a `custom_objects` argument. A save/load/predict
round-trip on `create_fractal_net("small")` reproduces the original outputs exactly
(max absolute difference 0.0).

```python
import keras

model.save("fractalnet.keras")
loaded = keras.models.load_model("fractalnet.keras")
```

Saving a compiled model and reloading it emits a Keras warning that the optimizer state was
skipped when the saved optimizer had not yet been built. It is harmless for inference;
re-compile before resuming training.

---

## 13. Testing & Validation

```python
import numpy as np

from dl_techniques.models.vision.fractalnet.model import FractalNet


def test_creation_all_variants():
    for variant in FractalNet.MODEL_VARIANTS:
        model = FractalNet.from_variant(variant, num_classes=10, input_shape=(64, 64, 3))
        assert model is not None


def test_forward_pass_shape():
    model = FractalNet.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
    output = model.predict(np.random.rand(4, 32, 32, 3).astype("float32"))
    assert output.shape == (4, 10)
```

The package's own suites are `tests/test_models/test_fractalnet/` and
`tests/test_layers/test_fractal_block.py`; the latter holds the receptive-field test that
pins the expansion rule.

---

## 14. Troubleshooting

- `Length of strides (N) must equal length of filters (M)` — `strides` defaults to three
  entries. Pass one per stage whenever you change the number of stages.
- `block_config['strides'] must be 1 inside a FractalBlock` — downsample between stages via
  `strides`, not inside the base block.
- **Out of memory at a modest `depth`** — cost is `2^depth`, not `depth`. Drop the stage
  depth by one before touching the batch size.
- **Loss is NaN or unstable** — lower the learning rate and add warmup. The default
  `he_normal` initialization is already the right one for ReLU.
- **Accuracy far below the loss's implied level** — check you did not compile with the
  string `"sparse_categorical_crossentropy"` against the logit head.

---

## 15. Technical Details

**Expansion.** `F_1(x) = block(x)`, `F_k(x) = join(DP(F_{k-1}(F_{k-1}(x))), DP(block(x)))`,
where the two `F_{k-1}` instances are composed and `DP` is the drop-path operator.

**Counts at depth `k`.** Leaf blocks `2^k - 1`; longest path `2^(k-1)`; shortest path 1;
distinct paths `P(k) = P(k-1)^2 + 1`.

**Drop-path.** `DP(y) = y * b` with `b ~ Bernoulli(1 - drop_path_rate)`, drawn per sample
and per branch, followed by renormalization over the survivors. The deepest path is active
only when nothing on it is dropped, so short paths are trained far more often than long
ones — which is the intended asymmetry.

**Inference.** `p` is effectively 0, every join is a plain mean, and the output is a
deterministic average over the whole path ensemble.

---

## 16. Citation

```bibtex
@inproceedings{larsson2017fractalnet,
  title={FractalNet: Ultra-Deep Neural Networks without Residuals},
  author={Larsson, Gustav and Maire, Michael and Shakhnarovich, Gregory},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017},
  note={arXiv:1605.07648}
}
```
