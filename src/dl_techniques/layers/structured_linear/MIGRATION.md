# Migration notes — `structured_linear/` crystal layers

Checkpoint-affecting changes, recorded here because this file ships with the
code. A note in an untracked planning directory does not reach the next reader.

## DELETED: `RigidSimplexLayer` (2026-10-12)

The layer and its entire suite are gone: `structured_linear/rigid_simplex_layer.py`
and `tests/test_layers/test_structured_linear/test_rigid_simplex_layer.py`. The
MIGRATION sections below marked `[DELETED LAYER]` describe its history and are
kept so the removal, and the reasons for it, stay on the record.

**Why.** The layer shipped a trainable square rotation `R` in front of a frozen
simplex frame `W`, with an orthogonality penalty meant to keep `R` close to
`O(N)`:

* The rotation **destroyed the property the layer was named for**. The frozen
  `W` kept equiangularity at every checkpoint (off-diagonal spread `0.000000`),
  but the map the layer actually computes is `x @ (R·W)` -- and spread grew to
  `0.1063` after 8 epochs and `0.2222` after 40. A trained `RigidSimplexLayer`
  no longer projected onto a simplex frame; it was a general linear projection
  that had started from one.
* The Saxe et al. citation on orthogonal-init stability was **contradicted by
  measurement**. Controlled properly (identical head weights, identical data,
  five seeds, only `R` differing), non-orthogonal Glorot init beat orthogonal
  at every penalty value, including zero: 0.0027/0.1760/0.6931 against the
  orthogonal arm's 0.2031/0.1760 and identity's stuck 0.6931. The
  `orthogonality_penalty` knob made training worse monotonically, and the
  penalty did not protect the simplex property at any value from `0.0` to
  `100.0` -- measured spread `0.3153` down to only `0.1971`.
* The degree-1 `x -> (x@R)@W` linear map is **genuinely distinct** from
  `CrystalLayer`'s degree-0 `x/||x||` cosine map -- no weights make one the
  other -- but that linearity earns no advantage over a plain `Dense` in the
  measurement above, and `trainable_crystal=True` on the surviving
  `CrystalLayer` subsumes the frame-rotation expressivity for the cosine case.

No model, trainer or application consumed the layer, so the deletion has zero
blast radius. `tiled_simplex_frame` survives in `crystal_frames.py` because its
test asserts the degenerate construction, not because anything uses it.

## [DELETED LAYER] `RigidSimplexLayer` — the `frame` argument (2026-10-11)

`frame` was added after the layer shipped. `get_config()` always emits it.

| `frame` | construction | fresh-build values |
|---|---|---|
| `'crystal'` (new default) | `units` distinct directions, packed from rotated simplexes | changed for `units > input_dim + 1` |
| `'tiled'` | one simplex's vertices, repeated | **identical** to every previous release |

**What actually changes.** Only `units > input_dim + 1`. At or below that count
one simplex already supplies every column and the two constructions agree.

- **Before** (`input_dim=16, units=100`): 17 distinct columns of 100, with 490
  column pairs at inner product exactly `1.0`. Eighty-three percent of the
  layer's outputs were exact duplicates of another output, each receiving an
  identical logit — despite the layer documenting itself as a rigid simplex
  frame.
- **After**: 100 distinct columns of 100, maximum pairwise inner product ~0.71
  at the same settings.

**Old `.keras` archives are unaffected.** The frame is a saved weight
(`static_simplex`), so it is restored bit-exactly and never rebuilt. A load does
not re-run the constructor's construction path.

**Hand-built configs are covered.** A config dict that *lacks* the `frame` key
predates the argument and was necessarily built by tiling, so `from_config`
reconstructs it as `frame='tiled'` and logs a warning naming the change. A
config carrying `frame` is honoured as written. `from_config` also no longer
mutates the caller's dict, and no longer re-deserializes
`rotation_initializer` — `__init__` resolves it through
`keras.initializers.get`, which accepts both a string and a serialized dict.

## [DELETED LAYER] `RigidSimplexLayer` — the orthogonality penalty is gated on `training` (2026-10-11)

The penalty was previously added on **every** call, so `layer.losses` was
non-empty after a `training=False` forward and an inference pass carried a term
it was never trained against.

| call | before | after |
|---|---|---|
| `layer(x, training=False)` | 1 loss | **0** |
| `layer(x, training=True)` | 1 loss | 1 loss |
| `layer(x)` (`training=None`) | 1 loss | 1 loss |
| `orthogonality_penalty=0.0` | 1 loss, value 0 | **0** (a zero-weighted term is not added) |

A reported `model.evaluate` loss that included the penalty at inference will now
be lower by that term. Training behaviour is unchanged.

## New: `CrystalLayer`, `HierarchicalCrystalLayer`, `CrystalMultiHeadProjection`

Additive. No existing configuration, key or checkpoint is affected.

Registered as `dl_techniques.layers.structured_linear.<module>><Class>`, each
with its own module-level registration string, following the one-class-per-module
precedent set by `complex/` and `conv_blocks/`.

Two behaviours worth knowing before composing them:

- **`seed` covers the geometric frames only.** On `CrystalMultiHeadProjection`
  it fixes the prisms; the output projection's Glorot kernel still draws from
  the global RNG. Two layers sharing one seed therefore produce equal prisms and
  different output weights (measured at `max|delta| = 3.75` between two eagerly
  built layers).
- **`HierarchicalCrystalLayer`'s seed root is memoized, not re-derived.**
  The macro centroids and the `num_macro` leaf-bundle frames must come from ONE
  root, spawned ONCE. Two failure modes were measured and fixed during
  development; both produced leaf bundles centred on centroids that were not the
  ones stored in `macro_weights`, so the leaf geometry described a different
  classifier from the one the layer ran:
  - Decorrelating groups by `seed + group` fails because `crystal_frame` draws
    its first simplex from the generator: a group whose offset lands on the
    macro seed reproduces the macro simplex exactly, and that leaf sits at
    cosine `1.0` with its own centroid at *every* `cluster_spread`.
  - `np.random.SeedSequence(self.seed).spawn(...)` called once per initializer
    also fails: `spawn` is **stateful** (it advances an internal counter), and
    `SeedSequence(None)` draws fresh entropy per construction. The root and its
    children are therefore built once and cached on the instance.

  Measured leaf-to-stored-centroid cosine, worst group, at `cluster_spread=0.25`:
  `0.31` at `seed=None` before the fix, `0.98+` after, at every seed including
  `None`. An explicit `seed` masked the defect for most settings, which is why
  the regression guard is parametrized over `seed=None`.

## `compute_output_shape` accepts a shape that arrives as a list (2026-10-12)

All five layers in this package. A single shape is a legal Keras argument form
and can arrive wrapped in a list — `[[None, 8]]`, `[(None, 8)]` or
`[None, 8]` — notably after deserialization.

`list(input_shape)` over `[[None, 8]]` iterates the **outer** list, so
rewriting its last element replaces the *batch* axis:

| `input_shape` | before | after |
|---|---|---|
| `[[None, 8]]` | `(units,)` | `(None, units)` |
| `[(None, 8)]` | `(units,)` | `(None, units)` |
| `[None, 8]` | `(units,)` | `(None, units)` |

This affected the tuple return of every layer and the per-key dict of
`HierarchicalCrystalLayer`. Fixed by routing the argument through the shared
`dl_techniques.utils.tensors.canonical_binary_input_shape`, which already
exists for exactly this and whose docstring names `compute_output_shape` as a
caller. No shipped behaviour changes for the tuple form, which was already
correct.

## `cluster_spread=0` is degenerate, and is now documented as such (2026-10-12)

No behaviour change — the configuration stays legal — but it is now stated in
the module docstring, carries a `.. warning::` on the constructor argument, and
is pinned by a test, because it previously read as a working leaf discriminator.

At exactly zero, every leaf of a group IS its centroid: the frame holds
`num_macro_classes` distinct columns out of `num_macro_classes *
num_micro_per_macro`, every leaf of a group receives an identical logit, and
the conditional softmax within that group is uniform **by construction**
(measured within-group logit spread: exactly `0.0` at `spread=0`, `1.19e-05` at
`1e-06`, `2.71` at `0.25`). This is the same duplication `tiled_simplex_frame`
produces, reached through a different door.

The collapse is exact at zero rather than approximate near it, so no value below
which it is avoided exists; a caller wanting leaf discrimination must use a
strictly positive `cluster_spread`.

## Packing's distinctness ceiling is now stated (2026-10-12)

`crystal_frame`'s docstring, and `CrystalLayer`'s, claimed `width` distinct
columns without qualification. That holds up to the sphere's own ceiling of
`2 * dimensions` points, and below that the simplex count binds first at
`dimensions + 1`. Past the ceiling a repeat is unavoidable rather than a
defect: at `dimensions=1` the sphere is `S^0 = {-1, +1}`, so a 3-column frame
has 2 distinct columns and one repeat no matter how it is built. The claim is
now bounded in the docstring and pinned by a parametrized test that checks
both sides of the ceiling.

## `CrystalMultiHeadProjection` — the rank-2 reshape (2026-10-12)

The final reshape built its target from the **input's** shape while flattening
a tensor that had already gained the head axis:

```python
flat_shape = keras.ops.shape(inputs)[:-1] + (total_features,)   # wrong
```

`features` is `(..., H, V)` — the einsum has already split the head axis off
the trailing end — so collapsing to `(..., H * V)` drops **two** trailing axes,
not one. The two spellings coincide by accident at rank 3 and only there:
`inputs` is `(..., D)`, so `shape[:-1]` keeps one leading axis, which by chance
matches `features`' `shape[:-2]`. At every other rank they disagree:

```
Input to reshape has 1024 values, but the requested shape has 1152
```

A rank-2 `(B, D)` input asks for `B*H*V` elements out of `B*H*V/B`. **Every
pre-existing test used rank 3**, so the defect survived three audit rounds. It
is also invisible to a shape assertion, because the reshape raises before any
shape exists to compare — the only probe that found it was a `fit()` call on a
model whose preceding `Dense` collapsed rank 3 to rank 2.

Fixed by slicing `features`' own shape. Verified against a hand-written NumPy
reference at `max|delta| = 4.8e-07`, with ranks 2/3/4 swept. Rank-1 is
correctly rejected: a bare feature vector has no batch axis to split heads over.

## `l2_normalize` — a redundant guard removed (2026-10-12)

Mutation testing found that deleting the trailing `ops.where` in
`dl_techniques.utils.tensors.l2_normalize` changed **nothing** — bit-identical
on zero, ordinary, `1e-30`, `1e30` and mixed zero/huge rows. `peak == 0`
implies `x` is all zeros, so the pre-scaled vector is already `0`, the norm is
`sqrt(eps)`, and the correct answer falls out of the divisor substitution
alone. The trailing `ops.where` was dead code that only looked like a safety
net, and has been removed; the mutation now targets the divisor substitution
that actually does the work, and is caught by 18 tests.

## Removed: a guard that could not fire

`CrystalMultiHeadProjection.call()` re-checked `input_dim % num_heads == 0`.
`_input_dim` is fixed by `build()`, which already validated it, so the check
was structurally unreachable. Reinstatement was verified to leave the line
**uncovered with all 63 module tests passing** — no input can reach it. Removed
rather than left as decoration; the width check beside it, which *is* reachable,
stays.
