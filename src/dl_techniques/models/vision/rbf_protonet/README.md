# `RBFProtoNet`

A small CIFAR-style CNN backbone paired with an RBF (radial basis function) prototype
classification head, for CIFAR-100. The head replaces the usual `Dense(num_classes)`
softmax-of-logits classifier with one learned, inspectable prototype vector per class in
feature space: `model.rbf_head.centers[i]` IS the class-`i` prototype after training, not a
distributed weight matrix.

This is a small-scale demonstration of the mechanism, not a CIFAR-100 SOTA attempt — see
`plans/plan-2026-09-16T052349-7dfede94/plan.md` Problem Statement for the full framing.

## Overview

| | |
|---|---|
| Backbone | `stem_type='cifar'`-style: one 3x3 stride-1 stem (no max-pool), then a
configurable number of stride-2 `ConvBlock` downsampling stages, then global average pool |
| Head | `create_mixture_layer('rbf', units=num_classes, output_mode='normalized', ...)` — an
`RBFLayer` (`dl_techniques.layers.mixtures.radial_basis_function`) applied to the pooled
feature vector |
| Feature width | `feature_dim=128` (default) |
| Registration | `dl_techniques.models.rbf_protonet.model` |
| Pretrained weights | none — `pretrained=True` raises `NotImplementedError` |

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
anti-pattern `models/CLAUDE.md` § "When the shape does not apply" warns against, and the
pre-mortem in this plan's `plan.md` names it as a STOP-IF condition. See `decisions.md` D-004.

## Math

The RBF head computes, for each prototype (class) `i` and pooled feature vector `x`:

```
phi_i(x) = exp(-gamma_i * ||x - c_i||^2)
```

where `c_i` is prototype `i`'s learned center (`RBFLayer.centers`) and `gamma_i` is its
learned, always-positive (softplus-parameterized) width (`RBFLayer.gamma_raw`), per
`radial_basis_function.py`'s docstring. `RBFProtoNet` fixes `output_mode='normalized'`
(never `'basis'` — see D-002 below), so the head's output is the **normalized RBF (NRBF)**
activation:

```
p_i(x) = phi_i(x) / sum_j phi_j(x)
```

equivalently a softmax over the (unclipped) negative squared distances
`-gamma_i * ||x - c_i||^2`. `p(x)` sums to 1.0 along the last axis by construction, which is
why `SparseCategoricalCrossentropy(from_logits=False)` — not `from_logits=True` — is the
correct loss against it (see the trainer, Step 6 of this plan, and `decisions.md` D-003).

`output_mode='normalized'` is the model's only supported mode (`# DECISION
plan-2026-09-16-7dfede94/D-002` in `model.py`): the layer's own docstring documents
`'basis'` (the raw, unnormalized `phi_i`) as barely-trainable at realistic feature
dimensions, a footgun this model does not expose as a togglable knob.

An auxiliary center-repulsion loss (`RBFLayer`'s own `repulsion_strength`/`min_distance`
mechanism, added to `model.losses`) discourages prototypes from collapsing onto each other;
see Hyperparameters below for the measured numbers behind this model's defaults.

## Hyperparameters

- **`feature_dim=128`** — `decisions.md` D-006 measured `RBFLayer(units=100,
  output_mode='normalized')` gradients on `centers`/`gamma` at `feature_dim` in
  `{128, 256, 512}`; `128` already fully passed the gradient-liveness bar
  (`MIN_USEFUL_GRADMAX = 1e-4`) with the *strongest* gradients of the three candidates —
  larger `feature_dim` bought nothing measurable, so the smallest passing value was kept for
  parameter/compute economy.
- **`repulsion_strength=0.1`, `min_distance=1.0`** — the RBF factory's own stock defaults,
  kept unchanged (`decisions.md` D-007) because D-006's smoke test already measured them at
  this exact configuration (`units=100, feature_dim=128`): the repulsion loss stayed at
  0.14%-1.14% of total loss, nowhere near a dominating share. Both are exposed as ordinary
  constructor kwargs, not hardcoded, so a caller can override them without a code change.

## Usage

```python
import keras
from dl_techniques.models.vision.rbf_protonet.model import create_rbf_protonet

model = create_rbf_protonet(num_classes=100, input_shape=(32, 32, 3))
model.build((None, 32, 32, 3))

x = keras.random.normal((4, 32, 32, 3))
probs = model(x, training=False)
print(probs.shape)                       # (4, 100)
print(keras.ops.sum(probs, axis=-1))     # [1.0, 1.0, 1.0, 1.0] (within float tolerance)

# pretrained=True raises -- no weights are shipped by this package:
#   create_rbf_protonet(pretrained=True)  # NotImplementedError
```

`get_config()` / `from_config()` round-trip every constructor argument, and a model reloaded
from a saved `.keras` file reproduces the original's output within `atol=1e-5` on a fixed
input — see `tests/test_models/test_rbf_protonet/test_round_trip.py`.

## Testing

`pytest tests/test_models/test_rbf_protonet/ -vvv` — construction/forward-pass shape and
softmax-sum-to-1 contract, `get_config`/`from_config` + `.keras` save/load round trip, and
oracle adoption (`gradient_flow_oracle`, `smoke_contract_oracle`, `knob_sensitivity_oracle`)
run after one real optimizer step.

## References

- He, Zhang, Ren & Sun, 2016. *Deep Residual Learning for Image Recognition*
  (the `stem_type='cifar'` precedent this backbone's stem mirrors).
  [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Snell, Swersky & Zemel, 2017. *Prototypical Networks for Few-shot Learning*
  (the "one prototype vector per class in feature space" framing this backbone's pooled
  output feeds into once the RBF head is attached).
  [arXiv:1703.05175](https://arxiv.org/abs/1703.05175)

### Authoring rules

Conventions: [`models/CLAUDE.md`](../../CLAUDE.md). Mandatory guide:
`research/2026_keras_custom_models_instructions_v2.md`.
