# Structured linear (`dl_techniques.layers.structured_linear`)

Alternative parameterizations to a dense weight matrix: quantized, tensor-train,
orthonormally-regularized, exactly-orthogonal, random-feature, spline-kernel,
distance-scored, and spherical-code heads.

There is **no factory** here and the `__init__.py` is empty (the
"exports-nothing" shape in `layers/AGENTS.md`). Import each layer from its
defining module:

```python
from dl_techniques.layers.structured_linear.bitlinear_layer import BitLinear
from dl_techniques.layers.structured_linear.crystal_layer import CrystalLayer
```

## What is here

| File | Contents |
| :--- | :--- |
| `bitlinear_layer.py` | `BitLinear` — 1.58-bit (ternary) quantization-aware dense |
| `mps_layer.py` | `MPSLayer` — matrix-product-state / tensor-train chain |
| `orthoblock.py` | `OrthoBlock` — decorrelated, gated dense block |
| `orthogonal_butterfly.py` | `OrthogonalButterfly` — exactly-orthogonal butterfly (Givens) map |
| `random_fourier_features.py` | `RFFKernelLayer` — fixed random Fourier features + trainable projection |
| `kanvolution.py` | `KANvolution` — convolution with per-tap learnable B-spline + SiLU |
| `harmonic_dense.py` | `HarmonicDense` — Euclidean-distance classifier head |
| `hierarchical_harmonic.py` | `HierarchicalHarmonicHead` — tree-factored harmonic head for large vocabularies |
| `crystal_frames.py` | Shared geometry: `hyperspherical_simplex`, `haar_orthogonal`, `crystal_frame`, `tiled_simplex_frame` |
| `crystal_layer.py` | `CrystalLayer` — cosine logits against a packed spherical-code frame |
| `hierarchical_crystal_layer.py` | `HierarchicalCrystalLayer` — two-tier crystal classifier over a class tree |
| `crystal_multihead_projection.py` | `CrystalMultiHeadProjection` — per-head simplex prisms + learned projection |

---

## `BitLinear`

```python
y = matmul(round(x / gamma_x), round(W / gamma_w)) * gamma_x * gamma_w
```

with a straight-through estimator through the round. Weights default to ternary
`{-1, 0, 1}` (`weight_bits=1.58`); `weight_bits=1` is a distinct binary path
(`{-1, +1}` via `sign`). Activations default to 8-bit. `gamma` never reduces
over the batch axis, so the layer computes the same function at any batch size.

`use_bias` defaults to **`False`**, matching the BitNet formulation — unlike
`keras.layers.Dense`. `get_config` has always carried it, so saved archives
rebuild as constructed; hand-written code that relied on the old default must
pass `use_bias=True` (see `MIGRATION.md` in the class docstring).

**New in this revision:**
- `seed` parameter for reproducible quantization noise (stochastic rounding)
- Explicit `from_config` for reliable deserialization

---

## `MPSLayer`

Each input feature owns a `(bond_dim, bond_dim)` core; a boundary vector
contracts left-to-right along the chain, then projects to `output_dim`.
Parameters scale as `O(input_dim * bond_dim^2)` instead of `O(n^2)`.

The chain contracts the **last** axis at any rank: leading dims flatten through
it and are restored, so `(B, T, D)` works exactly as `compute_output_shape`
promises (each leading position contracts independently). The two kernel-fed
weights draw from their **own clones** of `kernel_initializer` — one shared
seedless instance replays the same sample at every site, so sharing it tied
the cores to the projection (Pearson r ≈ 1.0; see the `DECISION` block in
`build()`).

**New in this revision:**
- `seed` parameter for reproducible initialization
- Explicit `from_config` for reliable deserialization

---

## `OrthoBlock`

Pipeline: orthonormally-regularized `Dense` → `ZeroCenteredRMSNorm` → learnable
`[0, 1]` gate → activation (applied last, after the gate). The kernel penalty
is `(lambda / sqrt(rank)) * ||G - I||_F^2` over whichever kernel axis is
smaller. `use_bias` defaults to `False`: before a centering norm the bias mean
is unidentifiable.

Two traps, both in the docstrings: `scale_initial_value` defaults to **0.8**,
not 0.5, because 0.5 is the unstable equilibrium of the binary-preference
well; and a ReLU-family activation makes a fully-closed gate **absorbing**
(`relu'(0) = 0`, so a closed gate receives no gradient) — a warning is logged.
`OrthoBlock` is consumed by `OrthoGLUFFN` in `layers/ffn/orthoglu_ffn.py`.

**New in this revision:**
- `seed` parameter for reproducible initialization
- Explicit `from_config` for reliable deserialization

---

## `OrthogonalButterfly`

`log2(d)` stages of `d/2` disjoint 2x2 Givens rotations in the FFT butterfly
pattern: exactly orthogonal for any angles, at `O(d log d)` cost. Exactly
invertible (`W^-1 = W^T` via `call(x, inverse=True)` or `inverse()`), with an
exactly-zero log-det-Jacobian — a cheap normalizing-flow step. The feature dim
must be a power of two; anything else raises. `angle_initializer='zeros'` (the
default) starts at the identity.

**Fixed in this revision:**
- **Dtype preservation**: The layer no longer internally casts to float32.
  It now respects the input dtype (including mixed_float16/bfloat16) throughout
  the forward and inverse passes.

**New in this revision:**
- `seed` parameter for reproducible initialization
- Explicit `from_config` for reliable deserialization

---

## `RFFKernelLayer`

`phi(x) = sqrt(2/D) * cos(x @ omega + b)` with frozen `omega ~ N(0, gamma^2)`,
`b ~ Uniform(0, 2*pi)`, then a trainable `Dense`. By Bochner's theorem
`phi(x)^T phi(y)` approximates the RBF kernel with `O(D)` memory.

`seed` fixes the frozen projection only; the output `Dense` still draws from
the global RNG (same contract as the crystal family's `seed`). The projection
is a saved weight, so `.keras` archives restore it bit-exactly. Two same-seed
layers share `omega`/`b` bit-exactly and differ downstream.

**New in this revision:**
- Explicit `from_config` for reliable deserialization

---

## `KANvolution`

Convolution layer whose per-tap activation is a learnable B-spline combined
with SiLU, instead of a fixed kernel weight and activation.

Each kernel tap computes `K(x) = w_spline * B(x) + w_silu * SiLU(x)`, with
`B(x)` a linear B-spline over a fixed `[-1, 1]` grid and `x` squashed by
`tanh`. `control_points`, `w_spline` and `w_silu` each draw from their **own
clone** of `kernel_initializer` (same replay mechanism as `MPSLayer`; see the
`DECISION` block in `build()`). `strides > 1` together with `dilation_rate > 1`
raises at construction — TF refuses it at `call()` time.

**New in this revision:**
- `seed` parameter for reproducible initialization
- Explicit `from_config` for reliable deserialization

---

## `HarmonicDense`

Scores by distance, not dot product: `logit_i = -n * log ||x - w_i||`. Three
`output_mode`s: `"distances"` (raw `||x - w_i||`, feed into `HarMax`),
`"logits"` (`-0.5 * n * log(d2)`, for `from_logits=True`), `"probs"`
(softmax of those, the default). `n=None` resolves to `sqrt(input_dim)` at
build. `HarMax(distances)` reproduces `"probs"` mode on the same kernel to
~1e-7. No bias: a per-class additive shift has no distance interpretation.

**New in this revision:**
- `seed` parameter for reproducible initialization
- Explicit `from_config` for reliable deserialization

---

## `HierarchicalHarmonicHead`

Factorizes the flat `HarMax` over a fixed balanced tree with a **learned**
class-to-leaf assignment: per-level node prototypes plus one member prototype
per leaf slot, routed top-down, leaf probability as the product of
sibling-group-normalized conditionals. Exact forward sums to 1 by
construction; `beam_call` evaluates only the top-k level-1 subtrees for
retrieval-scale inference. `reassign()` is the explicit E-step (nearest
routing-center assignment with exact per-node capacities — dead clusters are
structurally impossible); member rows permute to follow their classes. Padding
(`prod(branching) > num_classes`) is a slot suffix: padded leaves land on
exactly 0.

`"logits"` here are **normalized** log-probabilities — do not pass them to a
`from_logits=True` loss (that normalizes twice). `set_n_per_level()` is the
annealing hook (see `callbacks/anneal_harmonic_exponent.py`); a pre-`build()`
call is overwritten by `build()`'s resolution. Consumed by
`train/head_comparison/` and `callbacks/periodic_reassign.py`.

**New in this revision:**
- `seed` parameter for reproducible initialization
- Explicit `from_config` for reliable deserialization

---

## The crystal family

Neural Collapse (Papyan et al., 2020) is taken as the *initialization*:
`CrystalLayer` scores cosine logits against a frame of unit vectors packed
from rotated simplexes by a greedy minimax-angular-separation search.
`HierarchicalCrystalLayer` keeps a two-tier (macro centroid + isotropic leaf
bundle) version for hierarchical label spaces, with `cluster_spread` as the
perturbation magnitude. `CrystalMultiHeadProjection` splits features into `H`
sub-spaces with one rotated-simplex prism each, then projects back with a
learned `Dense` (hardcoded GELU between prism readout and projection).

All three route `compute_output_shape` through
`utils/tensors.canonical_binary_input_shape`, and all frames are computed
inside weight initializers (a `StatelessScope` build discards out-of-
initializer assignments). `seed` covers the geometric frames only.

### Known limits (measured)

- **Separation, not distinctness.** Packing yields `width` distinct columns at
  every width measured (up to 50,000 in 2-D). What degrades is the minimum
  angular separation — and a frame is a set of *lines* while the search prices
  only the signed inner product, so antipodal pairs (`cos -> -1`) are never
  penalised: `dim=2, width=64, seed=0` holds a pair at `cos = -1.0000`.
  Minimising `max|.|` instead is uniformly better-or-equal where measured but
  moves every wide frame, so it is a recorded-but-deferred option. Details and
  numbers: `crystal_frames.py` module docstring.
- **`cluster_spread=0` collapses the leaf tier.** Every leaf becomes its
  centroid: `num_macro` distinct columns of `num_macro * num_micro`, identical
  logits per group, uniform conditional softmax by construction. Legal and
  documented; use a strictly positive spread for leaf discrimination.
- **`cluster_spread=1` can zero a leaf** (displacement exactly opposite the
  centroid → all-zero column → permanent zero score). Measure-zero, stays
  legal, pinned by tests.
- All eleven layers' `compute_output_shape` now routes through the shared
  helper `canonical_binary_input_shape`, fixing the list-wrapped shape issue
  that previously corrupted the batch axis (e.g. `OrthoBlock cos([[None, 8]])
  = (4,)` instead of `(None, 4)`).

**New in this revision:**
- All crystal family layers now have explicit `from_config` for reliable deserialization

---

## Tests

`tests/test_layers/test_structured_linear/` — one suite per layer plus
single-claim guards (`test_the_*`): the list-wrapped-shape sweep over all
seven non-crystal layers, RFF reproducibility/dtype honesty, MPS rank
generality, the KANvolution strides guard, and the crystal antipodal
characterisation. `mutation_audit.py` is the crystal mutation harness (its
`_ALLOWED` covers the whole package); `test_the_mutation_suite_found_no_new_holes.py`
records which mutations the suite catches.

---

## References

- Wang et al., 2024. The Era of 1-bit LLMs. (BitNet)
- Stoudenmire & Schwab, 2016. Supervised Learning with Tensor Networks. (MPS)
- Novikov et al., 2015. Tensorizing Neural Networks.
- Rahimi & Recht, 2007. Random Features for Large-Scale Kernel Machines. (RFF)
- Liu et al., 2024. KAN: Kolmogorov-Arnold Networks; Bodner et al., 2024. Convolutional KANs.
- Baek et al., 2025. Harmonic Loss Trains Interpretable AI Models. (HarmonicDense, HarMax)
- Papyan, Han and Donoho, 2020. Prevalence of Neural Collapse. (crystal family)
- Strohmer and Heath, 2003. Grassmannian Frames. (crystal packing)

---

## Common Parameters

### `seed: Optional[int]`

Added to all layers in this revision for reproducible initialization. When
provided and the relevant initializer argument is a string (e.g.
`"glorot_uniform"`), a seeded instance of that initializer is created.
Deterministic initializers (`"zeros"`, `"ones"`, `Constant`, `Identity`)
are unaffected. For full reproducibility, call `keras.utils.set_random_seed()`
and leave the initializer seedless.

### `use_bias: bool`

Default varies by layer to match the cited formulation:
- `False`: `BitLinear`, `OrthoBlock`, `OrthogonalButterfly`, `CrystalLayer`,
  `HierarchicalCrystalLayer`, `CrystalMultiHeadProjection` (projection only)
- `True`: `MPSLayer`, `RFFKernelLayer`, `KANvolution`, `HarmonicDense`,
  `HierarchicalHarmonicHead`

### `from_config` classmethod

All layers now implement an explicit `from_config` classmethod for reliable
deserialization. This handles nested initializer/regularizer/constraint dicts
correctly and is the recommended way to reconstruct a layer from its config.