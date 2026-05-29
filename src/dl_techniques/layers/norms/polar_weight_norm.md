# Polar Weight Normalization

A trainable **polar-coordinate reparameterization of weights**, repurposed from
[PolarQuant](https://arxiv.org/abs/2502.02617) (Han et al., 2025). PolarQuant
uses a recursive Cartesian->polar transform to *quantize KV-cache vectors*; here
the same transform is turned into a **differentiable weight parameterization**
and a matching **initializer**.

Two public components:

| Component | Location | Import |
|---|---|---|
| `PolarWeightNorm` (layer) | `layers/norms/polar_weight_norm.py` | `from dl_techniques.layers.norms import PolarWeightNorm` |
| `PolarInitializer` | `initializers/polar_initializer.py` | `from dl_techniques.initializers import PolarInitializer` |
| `polar_encode` / `polar_decode` (helpers) | `layers/norms/polar_weight_norm.py` | `from dl_techniques.layers.norms import polar_encode, polar_decode` |

---

## 1. Idea

A vector `x` of dimension `d = 2^L` has a bijective polar representation: a
single radius `r = ‖x‖` plus `d - 1` angles organized into `log2(d)` hierarchical
levels (paper Definition 1). The transform is computed by a balanced binary tree
that repeatedly pairs adjacent coordinates `(a, b) -> (atan2(b, a), sqrt(a²+b²))`,
and inverted by the symmetric `(r, ψ) -> (r·cos ψ, r·sin ψ)` expansion.

**`PolarWeightNorm`** makes the radius and angles the *trainable parameters* of a
`Dense`-style layer and reconstructs the kernel on every forward pass. This is a
strict generalization of Weight Normalization (`w = g · v/‖v‖`): the direction is
given a full hierarchical angular coordinate system instead of a free unit
vector, and the magnitude is an explicit parameter equal to the exact per-unit
weight norm.

**`PolarInitializer`** "samples in polar coordinates": a uniform-on-sphere
direction with an exact, user-specified radius. By PolarQuant Lemma 2 a Gaussian
vector's direction is exactly uniform on the sphere, so this is realized (for any
shape, power-of-two or not) by normalizing a Gaussian and rescaling to the target
norm.

---

## 2. `PolarWeightNorm`

```python
import keras
from dl_techniques.layers.norms import PolarWeightNorm

inputs = keras.Input(shape=(256,))
h = PolarWeightNorm(128, activation="relu")(inputs)
out = PolarWeightNorm(10)(h)
model = keras.Model(inputs, out)
```

Trainable parameters per layer:

- `radius` — shape `(units,)`, the exact L2 norm of each output unit's weight column.
- `angles` — shape `(units, d-1)` with `d = next_pow2(fan_in)`, the hierarchical direction.
- `bias`   — shape `(units,)` (optional).

### Guarantees / behavior

- **Exact per-unit norm.** After build and after every optimizer step,
  `‖kernel[:, j]‖₂ == |radius[j]|` (verified to ~1e-7). Magnitude and direction
  are therefore independently controllable — e.g. apply a regularizer or a
  different learning rate to `radius` vs `angles`.
- **Drop-in initialization.** `build()` samples a seed kernel from
  `kernel_initializer`, encodes it, and stores the resulting `(radius, angles)`,
  so a freshly built layer reproduces a standard `Dense` kernel exactly. Training
  then moves the polar parameters.
- **Any `fan_in`.** Non-power-of-two `fan_in` is internally zero-padded to the
  next power of two; the reconstructed direction is sliced back and renormalized,
  keeping the exact-norm guarantee. Cost: up to ~2x redundant angle parameters
  when `fan_in` is not already a power of two.
- **Angular prior (optional).** Pass an `angle_regularizer` that pulls level >= 2
  angles toward `pi/4` to impose a Gaussian-direction prior (PolarQuant Lemma 2
  shows higher-level angles of a random Gaussian concentrate at `pi/4`).

### Key arguments

| Arg | Default | Meaning |
|---|---|---|
| `units` | — | output dimensionality |
| `activation` | `None` | activation applied after the matmul |
| `use_bias` | `True` | add a bias vector |
| `kernel_initializer` | `'glorot_uniform'` | initializer for the *seed* kernel that is encoded |
| `radius_regularizer` / `angle_regularizer` / `bias_regularizer` | `None` | per-parameter regularizers |
| `epsilon` | `1e-12` | slice-renormalization stability constant |

### Caveat

The kernel is reconstructed (cos/sin tree) on **every forward pass**. This is
`O(units · d)` extra work — negligible relative to the matmul for research use,
but not tuned for production inference throughput.

---

## 3. `PolarInitializer`

```python
import keras
from dl_techniques.initializers import PolarInitializer

# Every output unit's weight vector starts with L2 norm exactly 1.0 ("equinorm")
layer = keras.layers.Dense(128, kernel_initializer=PolarInitializer(norm=1.0, axis=0))
```

| Arg | Default | Meaning |
|---|---|---|
| `norm` | `None` | exact L2 norm of each vector along `axis`; `None` -> `sqrt(2)` (He-energy) |
| `axis` | `0` | axis along which each weight vector lies (0 = `fan_in` for a Dense kernel) |
| `gain` | `1.0` | multiplicative scale on the target norm |
| `seed` | `None` | reproducibility seed |

Unlike Gaussian/He/Glorot sampling (whose per-vector norms are chi-distributed),
`PolarInitializer` gives every vector an identical, exact norm with a uniform
direction — useful for well-conditioned, magnitude-controlled initialization.

---

## 4. Provenance

- Paper: *PolarQuant: Quantizing KV Caches with Polar Transformation*, Han,
  Kacham, Mirrokni, Karbasi, Zandieh, arXiv:2502.02617 (2025). The paper itself
  notes the transform's principles "extend beyond KV cache compression, offering
  potential applications in LLM weight quantization"; this module realizes a
  training-time variant of that idea.
- Generalizes: Weight Normalization (Salimans & Kingma, 2016).
- Tests: `tests/test_layers/test_norms/test_polar_weight_norm.py`,
  `tests/test_initializers/test_polar_initializer.py`.
