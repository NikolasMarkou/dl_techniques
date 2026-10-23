# Plan: TopoLM — extending the topographic layout from 2D to 3D

> Status: proposed, not yet implemented. Decisions below were taken with the
> maintainer on 2026-10-23. Everything else is investigation.

TopoLM (Rathi et al. 2025, ICLR; arXiv:2410.11516) gives the units of every
attention and feed-forward branch a distinct cell of an `h x w` grid — with
`h * w == embed_dim` — and adds the sampled pairwise-correlation penalty
`SL = 0.5 * (1 - corr(r, d))` alongside next-token cross-entropy. The layout is
a sheet of tissue. This plan extends it to a volume: `h x w x d` with
`h * w * d == embed_dim`, cubic (L∞) neighbourhoods, and a most-cubic
auto-factoring rule — with 2D behaviour byte-identical by default.

## Decisions taken (with the maintainer, 2026-10-23)

| Question | Decision |
|---|---|
| Neighbourhood geometry in 3D | **Cubic (L∞ ball)** — cells with `max(|dz|,|dy|,|dx|) <= rho`. Natural extension of the current square patches; keeps `patch_units` constant per radius, which is the assumption the whole shared-`d_unit` design rests on. Rejected: spherical (L2) neighbourhoods — `patch_units` then varies per centre and `d_unit` stops being a single shared precomputed vector, breaking the affordability argument in `build_patch_tables`. |
| Grid resolution in 3D | **Most cubic (h ≈ w ≈ d)** — find the factor triplet closest to the cube root, the same way 2D finds the most-square rectangle. |
| Compatibility | **Extend TopoLM in-place** — a `grid_dim: int = 2` parameter threaded through `SpatialSmoothness`, `SpatialLayout`, `build_patch_tables`, `TopoLMBlock`, `TopoLM`, config and validation. Default 2 preserves every existing archive, variant and test. Rejected: a separate `topolm3d/` package (duplicates the block and the plumbing for one parameter) and a shared-base-package refactor (more churn than the feature justifies). |

## Existing 2D machinery (review summary)

The implementation lives in two places, and the split is deliberate:

* `src/dl_techniques/layers/regularization/spatial_smoothness.py` — the
  model-agnostic topographic primitives:
  * `resolve_grid_shape(num_units, grid_shape)` — the ONE place grid
    arithmetic lives; factors `num_units` into the most-square `(h, w)` via
    `_largest_divisor_at_most`, or validates an explicit pair.
  * `SpatialLayout` — the unit-to-cell bijection. `perm[i]` is unit i's cell
    in row-major order; `cell_to_unit` (shape `(h, w)`) is its inverse;
    `positions` holds `(row, col)` per unit. Pure NumPy, no tensors.
  * `build_patch_tables(cell_to_unit, radius, distance)` — enumerates every
    fully-inside square neighbourhood (centres run `radius .. side-1-radius`
    per axis) into `patch_table (num_centers, patch_units)`, builds the strict
    upper-triangle flat index `triu_flat`, and the mean-centred,
    unit-normalised inverse-distance prior `d_unit`, computed ONCE because
    `P = (2rho+1)**2` is constant across centres.
  * `spatial_smoothness_loss(activations, patch_indices, triu_flat, d_unit)` —
    the pure penalty function. Operates on a gathered `(K, P, M)` tensor;
    **already dimension-agnostic**. Needs no change.
  * `SpatialSmoothness` — the identity tap. Builds layout + tables as
    non-trainable int64 weights (`perm`, `patch_table`, `triu_flat`,
    `d_unit`) plus the recorded `last_spatial_loss` scalar; adds
    `alpha * penalty` through `add_loss` when `training is True`.
* `src/dl_techniques/models/language/topolm/{config,components,model}.py` —
  the model that composes them:
  * `config.py` — variant registry (`paper` 784=28x28, `small` 512=16x32,
    `tiny` 256=16x16), `DEFAULT_ALPHA=2.5`, `DEFAULT_RADIUS=5`,
    `DEFAULT_NUM_NEIGHBORHOODS=5`, `validate_variant` (head divisibility +
    `embed_dim >= (2r+1)**2`), `validate_all_variants` run at import.
  * `components.py` — `TopoLMBlock`, a bespoke pre-norm decoder block (the
    shared `TransformerLayer` cannot reach its branch outputs before the
    residual add) holding `attn_out` / `mlp_out` as named locals, passing each
    through a `SpatialSmoothness` tap before the add. Taps are created
    unconditionally and gated in `call`, so `alpha=0` keeps a byte-identical
    weight set. Per-tap seeds via `permutation_seed(base_seed, tap_index)`.
  * `model.py` — `TopoLM` (token + learned position embeddings, blocks,
    final norm, tied LM head or `task_head`), `from_variant` /
    `create_topolm` factory, by-name strict checkpoint loading.

## Why 2D has to become a parameter, not a rewrite

`spatial_smoothness_loss` already collapses any `(K, P, M)` gather, so the
penalty itself is dimension-blind. The dimensionality lives in exactly three
spots:

1. the factoring rule (`resolve_grid_shape`),
2. the layout bookkeeping (`SpatialLayout`'s positions and cell grid),
3. the neighbourhood enumeration and offset arithmetic
   (`build_patch_tables`).

The layer decides `axis=2` / `-2` nowhere except through these three; the
gather, the correlation matmul and the Spearman-vs-distance dot product never
learn how many axes the grid had. So the plan is a `grid_dim` parameter with
branch points in those three spots only — everything downstream is untouched.

## Changes, file by file

### 1. `layers/regularization/spatial_smoothness.py` (the real work)

**`resolve_grid_shape(num_units, grid_shape=None, grid_dim=2)`**

* New `grid_dim` parameter; validated `in (2, 3)`.
* `grid_dim == 2`: current behaviour, byte-identical.
* `grid_dim == 3`: explicit `(h, w, d)` validated as a triplet with
  `h * w * d == num_units`; `None` factors via a new
  `_factor_3d_cubic(num_units)`:

```python
def _factor_3d_cubic(num_units: int) -> Tuple[int, int, int]:
    """The h <= w <= d triplet with h*w*d == n closest to the cube."""
    best = (1, 1, num_units)
    for h in range(1, round(num_units ** (1 / 3)) + 1):
        if num_units % h:
            continue
        rest = num_units // h
        w_lo = h
        # largest divisor of rest that is >= w_lo and <= sqrt(rest)
        for w in range(w_lo, math.isqrt(rest) + 1):
            if rest % w == 0:
                d = rest // w
                if w <= d and (d - h) < (best[2] - best[0]):
                    best = (h, w, d)
    return best
```

  Examples to pin in tests: 343 → (7, 7, 7); 512 → (8, 8, 8); 256 →
  (4, 8, 8) (isqrt(64) = 8, so (4, 8, 8) beats (4, 4, 16) and (2, 8, 16));
  784 → (7, 7, 16) — the paper's width does NOT re-factor to a cube, see
  "Variant table" below; 1000 → (10, 10, 10). Integer arithmetic throughout
  (`isqrt`, cube-root via rounding + bound check), never float `**`.

**`SpatialLayout`**

* Constructor gains `grid_dim`.
* `collapse_dims = grid_dim - 1`: `to_grid` splits the last axis into
  `grid_dim` axes, `from_grid` re-flattens them; both checks change from
  "last two axes" to "last `grid_dim` axes".
* `positions` becomes `(z, y, x)` triplets for 3D (`np.unravel_index` on the
  perm against `(depth, height, width)`).
* `cell_to_unit` is shaped `(d, h, w)` for 3D — `build_patch_tables` cuts
  patches out of it directly, so its shape must match what a 3D row loop
  slices.

**`build_patch_tables(cell_to_unit, radius, distance='linf', grid_dim=2)`**

* `grid_dim` inferred from `cell_to_unit.ndim` and asserted against the
  parameter — cheaper and less confusing than a mismatched pair.
* 3D centre enumeration:

```python
for cz in range(radius, depth - radius):
    for cy in range(radius, height - radius):
        for cx in range(radius, width - radius):
            rows.append(cell_to_unit[cz-r:cz+r+1,
                                     cy-r:cy+r+1,
                                     cx-r:cx+r+1].reshape(-1))
```

* Fit check becomes `all(dim >= side for dim in shape)`.
* Offsets: `np.unravel_index(np.arange(patch_units), (side, side, side))`
  gives `(dz, dy, dx)`; then the SAME three metrics on three offset arrays:

```python
if distance == "linf":  sep = np.maximum(np.maximum(dy_, dx_), dz_)
elif distance == "l1":  sep = dy_ + dx_ + dz_
else:                   sep = np.sqrt(dy_**2 + dx_**2 + dz_**2)
```

* The degeneracy check (`d_norm == 0` → "neighbourhood is degenerate")
  carries over verbatim; under L∞ the separation distribution is a function
  of the side alone in both 2D and 3D, so a radius that was non-degenerate in
  2D cannot become degenerate in 3D.
* `NeighborhoodTables` gains a `grid_dim` field. `side`, `patch_units =
  side**grid_dim`, `num_pairs`, `num_centers` all recompute from it. Docstring
  example updated: a radius-5 cubic patch contains `11**3 = 1331` units and
  has `1331 * 1330 / 2 = 885,015` pairs.

### 2. `models/language/topolm/config.py`

* `grid_shape_of(embed_dim, grid_dim=2)` — forwards `grid_dim`.
* `validate_variant(name, config, radius, grid_dim=2)` — the floor becomes
  `(2 * radius + 1) ** grid_dim`; keep the per-variant error message naming
  the offending value. Head divisibility unchanged.
* `validate_all_variants(radius, grid_dim=2)` — forwarded.
* **Variant table** — `paper` (784), `small` (512), `tiny` (256) stay 2D and
  pinned. 3D needs wider widths: at radius 5 a cubic patch needs
  `11**3 = 1331` units minimum vs. 121 in 2D. Paper width 784 factors as
  (7, 7, 16) for 3D, which cannot host radius 5 (needs ≥ 11 per axis; only
  the 16 axis can); add a 3D-capable repo-authored variant, deliberately NOT
  upstream-pinned, e.g.:

  | variant | embed_dim | 3D grid | usable 3D radius | provenance |
  |---|---|---|---|---|
  | `paper3d` | 1000 | 10 x 10 x 10 | 4 (needs `9^3 = 729` <= 1000) | repo-authored, not in the paper |
  | `tiny3d` | 343 | 7 x 7 x 7 | 2 (`5^3 = 125`) | repo-authored smoke scale |

  Where they live is an open question this plan defers to implementation:
  either new rows in `MODEL_VARIANTS` ignored by the upstream guard, or a
  second `MODEL_VARIANTS_3D` table keyed the same way. The contract to keep:
  nothing repo-authored joins the pinned `paper` row.

### 3. `models/language/topolm/components.py`

* `TopoLMBlock.__init__` gains `grid_dim: int = 2`; validated `in (2, 3)`
  alongside `normalization_position`; stored; passed to `_make_tap`.
* `_make_tap` forwards `grid_dim` to `SpatialSmoothness`.
* `get_config` includes `grid_dim`.
* `build` / `call` / the tap-position bookkeeping are untouched — the taps
  still sit on the branch outputs before the residual add, still created
  unconditionally and built conditionally, and the `alpha = 0` control
  argument (byte-identical weight set + identical permutations) is unchanged
  by dimensionality.

### 4. `models/language/topolm/model.py`

* `TopoLM.__init__` gains `grid_dim: int = 2`; forwarded into
  `_validate_config`, `_build_architecture`, and each `TopoLMBlock`.
* `_validate_config` — the minimum-width message changes with `grid_dim`
  (`(2r+1)^2` vs `(2r+1)^3`); `resolve_grid_shape(embed_dim, grid_shape,
  grid_dim)` call gains the argument.
* `grid_shape` property — forwards `grid_dim` to `resolve_grid_shape`; the
  cached `_resolved_grid_shape` tuple is then a triplet when 3D. Consumers
  that unpack it as `(h, w)` (analysis plots, `to_grid` calls in trainer
  evaluation) must be swept — grep for `grid_shape` under `src/train/topolm/`
  and `dl_techniques.metrics` during implementation.
* `get_config` / `from_config` round-trip `grid_dim` like every other
  constructor argument.
* `from_variant` / `create_topolm` — no signature change; `grid_dim` flows
  through `**kwargs` like `alpha` and `radius` already do. Never a separate
  factory for the 3D case (house rule: factories delegate, variants carry
  defaults).

### 5. Tests

Pinned numbers, not shape-only asserts (repo rule: a tolerance below the
dtype's resolution measures nothing, and a shape assert is how the three
guards in GUIDE.md § 8 went blind):

* `tests/test_layers/test_regularization/`:
  * `_factor_3d_cubic` table: 343 → (7,7,7), 512 → (8,8,8), 256 → (4,8,8),
    1000 → (10,10,10), 784 → (7,7,16), and a prime (e.g. 997 → (1,1,997)).
  * 3D patch tables: `patch_units == side**3`, `num_centers ==
    (h-2r)(w-2r)(d-2r)` for the resolved grid, first row of `patch_table`
    matches a hand-sliced cell_to_unit patch; `d_unit` norm 1 within float32
    roundoff.
  * Sign-discriminating probe in 3D (the 2D suite's two load-bearing rows,
    re-measured): a smooth 3D field scores well below 0.5, the SAME field
    against a negated distance prior scores well above it — 0.174 / 0.826 in
    2D roughly; 3D numbers are measured, not asserted-by-symmetry.
  * `grid_dim=2` byte-identical: 2D outputs of every table and the loss on a
    fixed field are unchanged by the refactor (the backward-compat contract).
* `tests/test_models/test_topolm/`:
  * Round-trip `get_config` with `grid_dim=3`.
  * A 3D model's forward pass, one training step moves variables, `alpha=0`
    keeps the identical weight set and identical permutations in 3D too.
  * Variant validation: `grid_dim=3` with `radius=5` on `embed_dim=784`
    raises naming `(2*5+1)**3 = 1331`; a legit 3D config passes.
* `tests/test_variant_tables_match_upstream_references.py` — stays green; if
  3D variants share `MODEL_VARIANTS`, the pinned-`paper` assertion must
  survive without special-casing the new rows.

## Sweep additions (during implementation, not after)

* `resolve_grid_shape` has other callers than TopoLM — grep `resolve_grid_shape`
  across `src/` and confirm each either pins `grid_dim` explicitly or cannot
  reach it (feature interaction check, not a rewrite).
* `metrics/spatial_autocorrelation.py` (Moran's I) and
  `metrics/topographic_selectivity.py` (contrast t-map, cluster growth) index
  cell coordinates; a 3D cluster-growth path is OPTIONAL follow-up work, not
  a blocker — the trainer's evaluation can keep scoring on 2D slices or be
  extended separately. Record the gap in the trainer README when the model
  ships.
* `GaussianReadout` (`layers/regularization/gaussian_readout.py`) blurs over
  the same grid to simulate a sensor; a 3D readout is the same class of
  optional follow-up.

## Numerical notes (carried over, unchanged in 3D)

* Tables stay int64 weights via `INDEX_WEIGHT_DTYPE` — TF keeps int32
  variables in host memory where an XLA-compiled GPU step cannot read them.
* Correlation block runs at `promote_types(input_dtype, float32)`.
* `patch_units` grows fast in 3D: 121 → 1331 at radius 5, so pairs go
  7,260 → 885,015 and the `K x M x P**2` correlation matmul is ~12x bigger
  per neighbourhood. `max_samples` (the batch×sequence flattening cap) and/or
  a smaller 3D radius are the expected mitigations; the cost model belongs in
  the README when this ships.
* The `eps` floor on the per-unit std and the epsilon-guarded denominator in
  the final dot product are untouched.

## Implementation order

1. `spatial_smoothness.py`: `_factor_3d_cubic`, `resolve_grid_shape`,
   `SpatialLayout`, `build_patch_tables`, `SpatialSmoothness` table plumbing.
2. Tests for 1, RED-to-GREEN, including the 2D-unchanged byte-identity pins.
3. `config.py` validation + 3D variant rows (or parallel table).
4. `components.py` + `model.py` threading; round-trip config test.
5. Model-level 3D tests (forward, one step, `alpha=0` control).
6. Sweep: trainer `grid_shape` consumers, metric callers, README updates in
   the model package documenting the 3D objective story and the cost model.

## Open questions deferred to implementation

* Exact home of the 3D variant rows (`MODEL_VARIANTS` vs parallel table).
* Whether the trainer's plotting wants a 3D slice view or opaque per-layer
  heat maps in this pass (probably the latter; record as follow-up).
* Whether `SpatialLayout.get_config` should carry `grid_dim` explicitly or
  re-derive it from `grid_shape`'s arity (prefer explicit — a config value
  reading back as something derivable is how the alpha=0 lesson happened).

## References

* Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, 2025. TopoLM:
  Brain-like spatio-functional organization in a topographic language model.
  ICLR 2025. (https://arxiv.org/abs/2410.11516)
* Lee, Margalit, Jozwik, Cohen, Kanwisher & DiCarlo, 2020. Topographic deep
  artificial neural networks. (https://arxiv.org/abs/2007.09019)
* Repo: `src/dl_techniques/models/language/topolm/README.md` § 2 (the
  objective), § 8 (`SpatialSmoothness` reference); the layer docstring in
  `spatial_smoothness.py`.
