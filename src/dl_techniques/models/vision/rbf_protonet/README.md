# `RBFProtoNet`

A small CIFAR-style CNN backbone paired with an RBF (radial basis function) prototype
classification head, for CIFAR-100. The head replaces the usual `Dense(num_classes)`
softmax-of-logits classifier with one learned, inspectable prototype vector per class in
feature space: `model.rbf_head.centers[i]` IS the class-`i` prototype after training, not a
distributed weight matrix.

This is a small-scale demonstration of the mechanism, not a CIFAR-100 SOTA attempt.

## Overview

| | |
|---|---|
| Backbone | `stem_type='cifar'`-style: one 3x3 stride-1 stem (no max-pool), then a
configurable number of stride-2 `ConvBlock` downsampling stages, then global average pool |
| Head | `create_mixture_layer('rbf', units=num_classes, output_mode='normalized', ...)` — an
`RBFLayer` (`dl_techniques.layers.mixtures.radial_basis_function`) applied to the pooled
feature vector |
| Feature width | `feature_dim=128` (default) |
| Input size | any: global average pooling ends the backbone, so the constructor takes no `input_shape`, and a stale `input_shape=` or `input_dim=` raises `ValueError` naming it (Keras alone would only warn) |
| Registration | `dl_techniques.models.rbf_protonet.model` |
| Pretrained weights | none — `create_rbf_protonet(pretrained=...)` raises `NotImplementedError` for `True` or a path, and the message says to build the model and call `model.load_weights(path)`; the constructor takes no `pretrained` |

## Architecture

```
input (32, 32, 3)
  stem      ConvBlock 3x3 s1                         -> (32, 32, stem_filters)
  stage 1   ConvBlock 3x3 s2                          -> (16, 16, filters_per_stage[0])
  stage 2   ConvBlock 3x3 s2                          -> ( 8,  8, filters_per_stage[1])
  stage 3   ConvBlock 3x3 s2                          -> ( 4,  4, filters_per_stage[2])
  head      GlobalAveragePooling2D (+ Dense proj      -> (feature_dim,)
            only if last stage width != feature_dim)
  rbf_head  RBFLayer(units=num_classes,
            output_mode='normalized')                 -> (num_classes,) probabilities, sum=1.0
```

Every conv+norm+activation triple is a `ConvBlock` (`dl_techniques.layers.conv_blocks.conv_block`),
which itself dispatches to `create_normalization_layer` / `resolve_activation_layer` — no
hand-rolled `Conv2D` + `BatchNormalization` + activation exists in `model.py`. Defaults:
`stem_filters=32`, `filters_per_stage=[64, 128, 128]`, `feature_dim=128`, so the pooled
output already matches `feature_dim` and no projection `Dense` is created.

**No `MODEL_VARIANTS` table** — this package intentionally ships without one. There is exactly
one baseline architecture here; inventing named size multipliers (e.g. a fake `_small` /
`_base` split) with no genuine use case would reproduce the "forcing an ill-fitting template"
anti-pattern `models/AGENTS.md` § "When the shape does not apply" warns against.

## Math

The RBF head computes, for each prototype (class) `i` and pooled feature vector `x`:

```
phi_i(x) = exp(-gamma_i * ||x - c_i||^2)
```

where `c_i` is prototype `i`'s learned center (`RBFLayer.centers`) and `gamma_i` is its
learned, always-positive (softplus-parameterized) width (`RBFLayer.gamma_raw`), per
`radial_basis_function.py`'s docstring. `RBFProtoNet` fixes `output_mode='normalized'`
(never `'basis'`: the comment above `_create_rbf_head` in `model.py` records why), so the head's output is the **normalized RBF (NRBF)**
activation:

```
p_i(x) = phi_i(x) / sum_j phi_j(x)
```

equivalently a softmax over the (unclipped) negative squared distances
`-gamma_i * ||x - c_i||^2`. `p(x)` sums to 1.0 along the last axis by construction, which is
why `SparseCategoricalCrossentropy(from_logits=False)` — not `from_logits=True` — is the
correct loss against it (see the trainer).

`output_mode='normalized'` is the model's only supported mode: the layer's own docstring
documents `'basis'` (the raw, unnormalized `phi_i`) as barely-trainable at realistic feature
dimensions, a footgun this model does not expose as a togglable knob.

An auxiliary center-repulsion loss (`RBFLayer`'s own `repulsion_strength`/`min_distance`
mechanism, added to `model.losses`) discourages prototypes from collapsing onto each other;
see Hyperparameters below for the measured numbers behind this model's defaults.

## Hyperparameters

- **`feature_dim=128`** — a standalone smoke test measured `RBFLayer(units=100,
  output_mode='normalized')` gradients on `centers`/`gamma` at `feature_dim` in
  `{128, 256, 512}` (the comment beside `feature_dim` in `model.py` records it); `128` already fully passed the gradient-liveness bar
  (`MIN_USEFUL_GRADMAX = 1e-4`) with the *strongest* gradients of the three candidates —
  larger `feature_dim` bought nothing measurable, so the smallest passing value was kept for
  parameter/compute economy.
- **`repulsion_strength=0.1`, `min_distance=1.0`** — the RBF factory's own stock defaults,
  kept unchanged because that smoke test already measured them at
  this exact configuration (`units=100, feature_dim=128`): the repulsion loss stayed at
  0.14%-1.14% of total loss, nowhere near a dominating share. Both are exposed as ordinary
  constructor kwargs, not hardcoded, so a caller can override them without a code change.

## Usage

```python
import keras
from dl_techniques.models.vision.rbf_protonet.model import create_rbf_protonet

model = create_rbf_protonet(num_classes=100)
model.build((None, 32, 32, 3))

x = keras.random.normal((4, 32, 32, 3))
probs = model(x, training=False)
print(probs.shape)                       # (4, 100)
print(keras.ops.sum(probs, axis=-1))     # [1.0, 1.0, 1.0, 1.0] (within float tolerance)

# pretrained=True (or a path) raises -- no weights are shipped by this package:
#   create_rbf_protonet(pretrained=True)  # NotImplementedError
```

`get_config()` / `from_config()` round-trip every constructor argument (`kernel_regularizer` goes
in as an instance, a name or a config and is stored as the object), and a model reloaded
from a saved `.keras` file reproduces the original's output within `atol=1e-5` on a fixed
input — see `tests/test_models/test_rbf_protonet/test_round_trip.py`. `num_classes` goes through the
library's one positive-integer rule (`utils.validation.positive_int`): any positive integer type is accepted, a
numpy integer included and stored as an `int`; a bool, a float, a string, `None` and a non-positive value raise
`ValueError: RBFProtoNet: num_classes must be a positive integer, got ...`. `dropout_rate` must be in `[0, 1)`.
Any keyword that is not an argument raises `ValueError` naming it: `input_shape` and `input_dim` by this
model's own message, the rest by Keras (`Unrecognized keyword arguments passed to RBFProtoNet: {...}`).

## Testing

`pytest tests/test_models/test_rbf_protonet/ -vvv` — construction/forward-pass shape and
softmax-sum-to-1 contract, argument validation, the shared `pretrained` refusal,
`get_config`/`from_config` + `.keras` save/load round trip, and oracle adoption
(`gradient_flow_oracle`, `smoke_contract_oracle`, `knob_sensitivity_oracle`) run after one
real optimizer step.

## Archives

**Archive break (2026-10-28).** Measured by saving the four roster subjects (`RBFProtoNet` with 20 and 22
tensors, `CliffordRBFProtoNet` with 38 and 40) in a separate process from the old revision and
restoring them with the current code (`git archive`, `python -I`); inference and `training=True`
outputs are compared at `atol=0`:

- `keras.models.load_model` refuses all four archives, from `be92a74eb` and from `20b48aba6^`:
  `ValueError: RBFProtoNet takes no input_shape (it ends in a global average pool and runs at any spatial
  size); got input_shape=[32, 32, 3]` (`CliffordRBFProtoNet` the same with its own name). The saved config
  carries the removed `input_shape` and `pretrained` keys, and nothing reads them; the constructor refuses
  `input_shape` first (before this refusal existed Keras popped it with a warning and then refused
  `pretrained` with `Unrecognized keyword arguments passed to RBFProtoNet: {'pretrained': False}`). Re-measured
  with the current code: `scratch_iter2/step16_1/archive_mlct_rbf_summary.txt`.
- `model.load_weights(path)` into a freshly built model: from `be92a74eb` all four restore, with equal
  outputs in both modes. `CliffordRBFProtoNet`'s blocks changed spelling (`clifford_block_{i}` to
  `block_{i}`: 30 of 38 and 30 of 40 `w.path` strings), which does not matter to the loader. From
  `20b48aba6^` `RBFProtoNet` restores (equal outputs) and `CliffordRBFProtoNet` refuses:
  `Layer 'stem_conv1' expected 2 variables, but received 0 variables during loading`, because its stem was
  a `Conv2D` and a `BatchNormalization` as two tracked layers and is now one `ConvStem`
  (`clifford_rbf_proto_net/stem/stem_conv1/kernel`).
- Transplanting by `w.path`, with `stem_conv/*` to `stem/stem_conv1/*`, `stem_bn/*` to `stem/stem_norm1/*`
  and `clifford_block_{i}/*` to `block_{i}/*`, restores every tensor of all four subjects from either
  revision (20/20, 22/22, 38/38, 40/40) with equal outputs at `atol=0`.
- A model built with a given seed has the same starting weights as before the change: the four
  subjects are tensor-for-tensor equal to the `be92a74eb` build (after the `block_{i}` rename), outputs included.
- Transcripts: `plans/plan-2026-10-26T125959-7af851e5/scratch_iter2/step16/archive_summary.txt`,
  `scratch_iter2/step16/archive_rbf_values.txt` and `scratch_iter2/archive/transcripts.txt`.

---

## References

- He, Zhang, Ren & Sun, 2016. *Deep Residual Learning for Image Recognition*
  (the `stem_type='cifar'` precedent this backbone's stem mirrors).
  [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Snell, Swersky & Zemel, 2017. *Prototypical Networks for Few-shot Learning*
  (the "one prototype vector per class in feature space" framing this backbone's pooled
  output feeds into once the RBF head is attached).
  [arXiv:1703.05175](https://arxiv.org/abs/1703.05175)

## CliffordRBFProtoNet

An alternative backbone for the same RBF prototype head: instead of `RBFProtoNet`'s
multi-stage stride-2 CNN, `CliffordRBFProtoNet` pairs an isotropic stack of
`CliffordNetBlock`s (`dl_techniques.layers.geometric.clifford_block`) with the identical
`_create_rbf_head(...)` classification head. Both classes live in this package's `model.py`
and share that one helper; the two backbones themselves have no shared code, mirroring
`vision/dino/`'s own "no common trunk" precedent for architecturally distinct backbones.

### Overview

| | |
|---|---|
| Backbone | One strided patch-embedding stem (a `ConvStem` named `stem`: one `Conv2D`, then `BatchNormalization` with momentum 0.9 and epsilon 1e-3, no activation), then `depth` identical `CliffordNetBlock`s (`block_{i}`, each with a `block_{i}_drop_path`) at a fixed `channels` width -- isotropic, no per-stage downsampling |
| Head | The same `_create_rbf_head(...)` / `RBFLayer(output_mode='normalized')` head as `RBFProtoNet` |
| Feature width | `feature_dim=128` (default); a `Dense` projection is only created if `channels != feature_dim` |
| Registration | `dl_techniques.models.rbf_protonet.model` (same module as `RBFProtoNet`) |
| Input size | any: the pooling is global, so the constructor takes no `input_shape`, and a stale `input_shape=` or `input_dim=` raises `ValueError` naming it (Keras alone would only warn) |
| Pretrained weights | none -- `create_clifford_rbf_protonet(pretrained=...)` raises `NotImplementedError` for `True` or a path, and the message says to build the model and call `model.load_weights(path)`; the constructor takes no `pretrained` |

### Architecture

```
Input [B, 32, 32, 3]
  stem      ConvStem: Conv2D 3x3 /patch_size + BatchNorm   -> 32x32 -> 16x16 (patch_size=2)
  blocks    depth x CliffordNetBlock(channels)              -> 16x16 -> 16x16 (isotropic)
            x = x + drop_path(block(x))
  head      GlobalAveragePooling2D (+ Dense projection      -> (feature_dim,)
            only if channels != feature_dim)
  rbf_head  RBFLayer(units=num_classes,
            output_mode='normalized')                       -> (num_classes,) probabilities, sum=1.0
```

Unlike `RBFProtoNet`'s three stride-2 `ConvBlock` stages (32x32 -> 16x16 -> 8x8 -> 4x4), the
Clifford backbone downsamples exactly once, in the stem, and then runs every block at the same
16x16 resolution and `channels` width -- see Hyperparameters below for why.

### Hyperparameters

Defaults: `channels=128, depth=12, shifts=(1, 2), patch_size=2, use_global_context=False,
drop_path_rate=0.1`.

- **`channels=128`, `depth=12`, `shifts=(1, 2)`, `use_global_context=False`** -- the isotropic,
  no-downsampling backbone shape mirrors every existing `CliffordNetBlock` consumer in this
  repo (`CliffordNet`, `CliffordNetLM`, `CliffordCLIP`), none of which interleaves separate
  stride-2 downsampling between blocks, so no interleaved-downsampling variant was attempted.
  The specific channel/depth/shift values are the `nano`-equivalent
  configuration `CliffordNet` and `CliffordCLIP` both already ship (`cliffordnet/README.md`'s
  own `nano` row), chosen to keep this demonstration's parameter count in the same small/fast
  scale class as `RBFProtoNet`'s own ~255K-parameter baseline rather than introducing an
  unvalidated new size.
- **`patch_size=2`** -- a single stride-2 `Conv2D` stem, replicated from `CliffordNet`'s own
  patch-embedding stem construction, reducing resolution once before the isotropic block stack
  runs at a fixed spatial size. The BatchNorm values are the module constants `STEM_BN_MOMENTUM`
  (0.9) and `STEM_BN_EPSILON` (1e-3, stated because the norm factory's own default is 1e-6): the
  stem was validated at exactly these values (the comment above `_build_stem` in `model.py`).
- **`drop_path_rate=0.1`** -- matches `CliffordNet`'s own recorded `nano`-variant training
  practice (`stochastic_depth_rate=0.1` in its `from_variant("nano", ...)` usage example), not
  `CliffordNet`'s unused bare-constructor default of `0.0`; it must be in `[0, 1)`. Per-block rates are ramped
  linearly from `0` to this value via `linear_drop_path_rates(depth, max_rate=drop_path_rate)`,
  the same schedule `CliffordNet.call()` uses.

### Math

Each `CliffordNetBlock` approximates the Clifford geometric product `AB = A . B + A ^ B` via
`SparseRollingGeometricProduct` (`dl_techniques.layers.geometric.clifford_block`): for every
shift `s` in `shifts`, a detail stream and a context stream are combined as a gated symmetric
product (`SiLU(Z_det * roll(Z_ctx, s))`, the scalar/inner part) and an antisymmetric outer
product (`Z_det * roll(Z_ctx, s) - Z_ctx * roll(Z_det, s)`, the bivector/wedge part); both parts
are concatenated across shifts and projected back to `channels` dimensions. This is a sparse,
channel-space-cyclic-roll approximation of a genuine Cl(p,q) multivector product, not an
algebraic grade decomposition (`cliffordnet/README.md` § Core Primitives).

The block's residual junction is a `GatedGeometricResidual`, an Euler-discretized ODE step:
a learned sigmoid gate mixes the geometric-product features into a `SiLU`-activated
normalized hidden state, the result is scaled by a `LayerScale` `gamma` initialized near zero
(`1e-5`), and optionally passed through stochastic depth (`StochasticDepth`,
`drop_paths[i]`). The near-zero `gamma` init is why the stem's standing smoke test
(`TestCliffordRBFProtoNetD005StandingSmoke`) measures small-but-live gradients rather than dead ones at
initialization: every block starts close to an identity residual by design, which is what
lets `depth=12` blocks stack stably from the first training step with no warmup. `call()`
composes each block exactly as `CliffordNet.call()` does: `x = x + drop_path(block(x))`, an
external residual around a block whose own `call()` returns only the transform term.

Once pooled to a flat feature vector, the RBF head's math (the normalized-RBF activation
`p_i(x) = phi_i(x) / sum_j phi_j(x)` over learned prototypes `c_i`, plus the center-repulsion
auxiliary loss) is identical to `RBFProtoNet`'s own -- see the Math section above; nothing
about the head differs between the two backbones.

References: Ji, 2026, [arXiv:2601.06793](https://arxiv.org/abs/2601.06793); Brandstetter et
al., 2023, [arXiv:2209.04934](https://arxiv.org/abs/2209.04934); Ruhe et al., 2023,
[arXiv:2302.06594](https://arxiv.org/abs/2302.06594) -- per `cliffordnet/README.md`'s own
citation set for these primitives, not re-derived independently here.

### Usage

```python
import keras
from dl_techniques.models.vision.rbf_protonet.model import create_clifford_rbf_protonet

model = create_clifford_rbf_protonet(num_classes=100)
model.build((None, 32, 32, 3))

x = keras.random.normal((4, 32, 32, 3))
probs = model(x, training=False)
print(probs.shape)                       # (4, 100)
print(keras.ops.sum(probs, axis=-1))     # [1.0, 1.0, 1.0, 1.0] (within float tolerance)

# pretrained=True (or a path) raises -- no weights are shipped by this package:
#   create_clifford_rbf_protonet(pretrained=True)  # NotImplementedError
```

`get_config()` / `from_config()` round-trip every constructor argument, and a model reloaded
from a saved `.keras` file reproduces the original's output within tolerance.

### Testing

`pytest tests/test_models/test_rbf_protonet/test_clifford_model.py -vvv` -- construction,
argument validation, layer names, the shared `pretrained` refusal, `get_config`/`from_config`
round trip, and a gradient-flow oracle across the full block stack.

### Authoring rules

Conventions: [`models/AGENTS.md`](../../AGENTS.md). Mandatory guide:
`research/2026_keras_custom_models_instructions_v2.md`.
