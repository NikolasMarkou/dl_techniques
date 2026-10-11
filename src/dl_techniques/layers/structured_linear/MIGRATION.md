# Migration notes — `structured_linear/` crystal layers

Checkpoint-affecting changes, recorded here because this file ships with the
code. A note in an untracked planning directory does not reach the next reader.

## `RigidSimplexLayer` — the `frame` argument (2026-10-11)

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

## `RigidSimplexLayer` — the orthogonality penalty is gated on `training` (2026-10-11)

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
