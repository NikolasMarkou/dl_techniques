# OrthogonalButterfly

A learnable, **exactly-orthogonal** `d x d` linear layer built from a log-depth
**butterfly of 2x2 Givens rotations**. It is the operator-valued sibling of the
[Polar Weight Normalization](norms/polar_weight_norm.md) work: the polar
transform arranges a vector's angles into a `log2(d)`-level binary tree of
coordinate pairs; `OrthogonalButterfly` uses the *same hierarchical pairing* to
parameterize an orthogonal transform.

```python
from dl_techniques.layers.orthogonal_butterfly import OrthogonalButterfly
```

## What it computes

For `d = 2^L`, the transform is `num_blocks` butterfly blocks, each `L` stages:

```
for stage s in [0..L):  stride = 2^s
    view x as (.., d/(2*stride), 2, stride)      # partners are `stride` apart
    rotate each pair:  [a; b] -> [a cosθ - b sinθ ; a sinθ + b cosθ]
```

Each stage applies `d/2` independent 2x2 rotations to **disjoint** coordinate
pairs (the Cooley-Tukey / FFT access pattern), so every stage — and their
product — is orthogonal.

- **Exactly orthogonal** for any angle values: `‖layer(x)‖ == ‖x‖` and `WᵀW = I`
  (verified ~1e-7). No matrix inverse, no Cayley/`expm`, no soft penalty.
- **Cheap**: `O(d log d)` compute and `(d/2)·log2(d)` parameters per block, vs
  `O(d^2)` for a dense orthogonal layer.
- **Identity at init**: `angle_initializer='zeros'` (default) => `layer(x) = x`,
  so it is a stable drop-in / residual block.

## Arguments

| Arg | Default | Meaning |
|---|---|---|
| `num_blocks` | `1` | stacked butterfly blocks; one block spans only the FFT-structured subset of `SO(d)`, more blocks => more expressive |
| `use_bias` | `False` | add a bias after the rotation (keeps the rotation orthogonal; the affine map is no longer pure-linear) |
| `angle_initializer` | `'zeros'` | initializer for the rotation angles (`'zeros'` = identity) |
| `angle_regularizer` | `None` | optional regularizer on the angles |
| `bias_initializer` / `bias_regularizer` | `'zeros'` / `None` | bias init / reg |

## Constraints

- **Power-of-two feature dim only.** Non-power-of-two `d` raises `ValueError`.
  Unlike `PolarWeightNorm` (which zero-pads + renormalizes), padding cannot
  preserve orthogonality on the original subspace, so it is rejected.
- Square: output dim == input dim == `d`.

## Invertibility (normalizing flows)

The transform is exactly invertible — `W^{-1} = W^T` — realized by reversing the
block/stage order and transposing each 2x2 rotation (`R(θ)^{-1} = R(-θ)`). With a
bias, the forward map is `y = W x + b` and the inverse is `x = W^T (y - b)`.

```python
layer = OrthogonalButterfly(num_blocks=2)
y = layer(x)                 # forward
x_rec = layer(y, inverse=True)   # exact reconstruction (== layer.inverse(y))
ldj = layer.log_det_jacobian(x)  # zeros, shape x.shape[:-1] — orthogonal map
```

- `call(x, inverse=True)` / `layer.inverse(x)` — the exact inverse.
- `layer.log_det_jacobian(x)` — returns `0` (one scalar per vector): the
  change-of-variables contribution of an orthogonal flow step.

## When to use

- **Normalizing flows** — orthogonal => zero log-det Jacobian and an exact,
  cheap inverse (`inverse=True`); a free, expressive linear/rotation flow step.
- **Orthogonal RNN recurrence** — norm-preserving recurrent maps avoid
  exploding/vanishing hidden-state norms.
- **Lossless / invertible blocks**, structured mixing layers, or a cheap
  learnable replacement for a fixed orthogonal transform (e.g. DCT/FFT-like).

## Provenance

- Structure: butterfly / Givens parameterizations of orthogonal matrices
  (Cooley-Tukey factorization; cf. butterfly / Kaleidoscope matrices, Dao et al.).
- Connection: the recursive coordinate-pairing tree of PolarQuant
  (arXiv:2502.02617); see `norms/polar_weight_norm.md`.
- Distinct from `OrthoBlock` (`layers/orthoblock.py`), which is a *soft*
  orthogonally-regularized Dense, not an exact orthogonal operator.
- Tests: `tests/test_layers/test_orthogonal_butterfly.py`.
