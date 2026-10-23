# TopoLM: extending the topographic layout from 2D to 3D

> Status: implemented. The implementation is the commit series
> `635747fa5..eea1dbe7f` on `main` (seven commits, in order `635747fa5`,
> `3dd7ee0e0`, `c2cac9ef5`, `246a42b67`, `1e64b58ec`, `85e03cc74`, `eea1dbe7f`),
> built by plan `plan-2026-10-23T141429-390f3815` from the decisions taken with
> the maintainer on 2026-10-23. This document was written before the code as a
> proposal; the commit that follows that series rewrites it into a record of what
> shipped, and records what the first draft got wrong. Every statement below was
> checked against the committed code, and every number is either derived by
> arithmetic (and says so) or a measurement quoted with its configuration.
> The adversarial review that followed produced three completion fixes (`805be4ba2`,
> `4a71059a0` and a documents-only one); the checkpoint-guard row of the decisions table,
> the factoring section and the implementation history below describe the code as those
> fixes left it.

TopoLM (Rathi et al. 2025, ICLR; arXiv:2410.11516) gives the units of every
attention and feed-forward branch a distinct cell of an `h x w` grid, with
`h * w == embed_dim`, and adds the sampled pairwise-correlation penalty
`SL = 0.5 * (1 - corr(r, d))` alongside next-token cross-entropy. The layout is a
sheet of tissue. The paper is 2D throughout. This change, which is repo-authored
and not in the paper, extends it to a volume: `a x b x c` with
`a * b * c == embed_dim`, cubic (L-infinity) neighbourhoods, and a most-cubic
auto-factoring rule, with 2D behaviour byte-identical by default.

## Decisions taken (with the maintainer, 2026-10-23), as shipped

| Question | Decision |
|---|---|
| Neighbourhood geometry in 3D | **Cubic (L-infinity ball)**: cells within `radius` of the centre along every axis, a cube of `(2 * radius + 1) ** 3` units. It keeps `patch_units` constant per radius, which is the assumption the whole shared-`d_unit` design rests on. Rejected: spherical (L2) neighbourhoods, whose `patch_units` would vary per centre and stop `d_unit` being one shared precomputed vector. The `l1` and `l2` metrics still exist in 3D as distance priors over the same cubic patch (`l2` adds the third squared offset), they do not change the patch shape. |
| Grid resolution in 3D | **Most cubic.** `_factor_3d_cubic(n)` returns the triple `(h, w, d)` with `h <= w <= d` and `h * w * d == n` that minimises `d - h`; among equal objectives the **larger `h`** wins. Integer arithmetic only: `h` ascends while `h ** 3 <= n` (that loop bound is the integer cube root, so there is no float cube root and no separate helper), and `w` runs from `h` to `math.isqrt(n // h)`. A prime `n` gives `(1, 1, n)`, which no radius can use. |
| Compatibility | **Extend TopoLM in place** with `grid_dim: int = 2`, values 2 or 3. Default 2 preserves every existing archive, variant and test. Rejected: a separate `topolm3d/` package (duplicates the block and the plumbing for one parameter) and a shared-base refactor (more churn than the feature justifies). |
| Where the rank is stated and where it is inferred | `grid_dim` is **explicit** in `resolve_grid_shape`, `SpatialLayout`, `SpatialSmoothness`, `TopoLMBlock`, `TopoLM`, `grid_shape_of` and `validate_variant`. It is **never** inferred from `len(grid_shape)`. It is **inferred from `cell_to_unit.ndim`** in `build_patch_tables`, which has no `grid_dim` parameter, so the tables cannot disagree with the array they were cut from. `GaussianReadout` relies on the explicit default: it calls `resolve_grid_shape` with the 2D default, so a three-entry `grid_shape` is refused. A config or `.keras` archive without the `grid_dim` key loads as 2D. |
| Axis order | One order everywhere: **`grid_shape == cell_to_unit.shape == (axis0, axis1, axis2)`, row-major, last axis fastest**, and `positions[:, k]` is the coordinate along axis `k` of that same shape. No axis has a name (no depth, height, width, no `(z, y, x)`), and nothing reorders or transposes anywhere. The first draft used three different orderings; see the section on what it got wrong. |
| Validation | Every resolved **axis** must be at least `2 * radius + 1`, not only the total unit count. The total floor `(2 * radius + 1) ** grid_dim` is still checked first (it is necessary, not sufficient). The per-axis check runs in `config.validate_variant` for **both** ranks and in the `TopoLM` constructor for **3D only**: a directly constructed 2D model with an explicit short-axis grid (an `(8, 32)` grid at the default radius 5) has always constructed and reported its grid, failing only when a tap builds, and an existing test pins that, so the 2D constructor is unchanged. Shipped variants cannot reach that hole because `validate_variant` is strict in both ranks. Errors name the short axis, its length and the largest usable radius, `(min_axis - 1) // 2`. |
| Variant home | The 3D variants are rows of `MODEL_VARIANTS` (`paper3d`, `tiny3d`), repo-authored and described as "not in the paper", each carrying its own `"grid_dim": 3` and `"radius"`. No parallel table. The `paper` row is untouched and nothing repo-authored joins it. `validate_all_variants(radius)` takes no `grid_dim`: it judges each row by the row's own `radius` and `grid_dim` and falls back to the argument and to 2. |
| Trainer | The trainer is 2D-only and **refuses a 3D variant by name before any run work**: `train_topolm` calls `_require_two_dimensional_variant` first, before seeding, `os.makedirs`, the tokenizer, the dataset or `prepare_run_dir`, and `build_backbone` also checks the built backbone's rank. There is **no `--grid-dim` flag**: the 3D rows are valid `--variant` choices because the parser reads the table, and choosing one is refused with a message naming the variant, its width and grid, and the 2D variants that work. |
| Checkpoints across ranks | `TopoLM.from_variant(..., pretrained=<path>)` guards two things before any weight is copied, in `TopoLM._refuse_other_grid` (completion fix `iter-1/step-5.1`, commit `805be4ba2`, decision D-017). (a) A bare TopoLM archive recorded at a different `grid_dim` is refused by name (a missing key in a bare TopoLM config counts as 2D). (b) An archive whose resolved unit grid differs from the model's is refused by name, naming both grids: a 3D `(10, 5, 20)` archive into a model on `(20, 5, 10)`, or a transposed 2D `(8, 32)` into `(32, 8)`; a different seed on the same grid loads. The shape check alone is not enough: 729 units give identically shaped `perm`, `patch_table` and `triu_flat` tables in both ranks (one 729-unit patch at radius 13 in 2D and at radius 4 in 3D), and a permuted grid gives identical shapes because the centre count is symmetric in the axes, so such loads would otherwise succeed and leave the model with a foreign layout. The rank and the grid are read only from an archive whose top-level config is a bare TopoLM: for any other archive the guard does not guess and the loader's own error stands (a `CausalLanguageModel`-wrapped TopoLM, in either rank, gets `No overlapping layers between target and source checkpoint`; the first version of the guard defaulted the missing key to 2 for such an archive and reported a 3D archive as `saved with grid_dim=2`, which was false). A direct `load_backbone_from_checkpoint` call stays uncovered, pinned by `test_a_direct_loader_call_does_not_see_the_rank`. Downgrade: an archive written by this code carries a `grid_dim` key (2 as well as 3), so a checkout from before this change cannot load it (`Unrecognized keyword arguments passed to TopoLM: {'grid_dim': 2}`, measured by the review); a config without the key still loads as 2D here. |
| Out of scope | `GaussianReadout`, Moran's I (`metrics/spatial_autocorrelation.py`) and the cluster-growth path (`metrics/topographic_selectivity.py`) stay 2D-only. 3D versions are follow-ups, not built. |

## What the first draft got wrong

The draft was investigated against the code before implementation; seven items
were wrong or missing, and several more surfaced while reading the shipped code.
They are recorded here in words, so the wrong figures do not survive in the text.

Known before implementation:

1. **The 784 factoring.** The draft's example table pinned a triplet for 784 that
   its own function did not return. The function returns the box
   `(7, 8, 14)`, whose gap `d - h` is 7; the triplet the draft pinned has a gap
   of 9 and loses. The paper's width does not re-factor to a cube, and its
   shortest 3D axis, 7, holds a radius of at most 3.
2. **The radius-5 3D pair count** was off by a hundred. A radius-5 cube holds
   1331 units, and `1331 * 1330 / 2` is 885,115.
3. **The cost growth** was understated by an order of magnitude. Pairs and the
   `K * M * P ** 2` correlation matmul both grow with `P ** 2`, so a radius-5 cube
   is about 121x (121 times) the paper's 2D neighbourhood, not about a tenth of that.
4. **"Integer arithmetic throughout, never a float power"** contradicted the
   draft's own sketch, which computed the cube root with a float power.
5. **Axis order.** The draft named the factored triple `(h, w, d)` ascending, the
   cell array `(d, h, w)` and positions `(z, y, x)`: three orderings, any of
   which would be silently wrong against another. A centre count is symmetric
   under an axis permutation, so no shape or count assert can see the mistake.
6. **The floor on total units is not enough.** 784 units at radius 4 clear the 3D
   floor of 729 and still resolve to `(7, 8, 14)`, whose first axis cannot hold a
   9-wide patch, so the failure would have fired deep inside a tap build. The same
   hole exists in 2D today (a prime width resolves to `1 x n`).
7. **Few centres.** `paper3d` leaves 8 admissible centres and `tiny3d` 27, so five
   sampled neighbourhoods overlap heavily. A design caveat, not a defect.

Found while checking the draft against the shipped code:

8. The draft listed `perm`, `patch_table`, `triu_flat` and `d_unit` as int64
   weights. `d_unit` is float32; only the three index tables are int64.
9. The draft called the block "a bespoke pre-norm decoder block" whose taps are
   "gated in `call`". The block supports both `normalization_position` values;
   taps are created unconditionally, but only the sites in `tap_sites` are built
   and called, and the `alpha` gate lives inside the tap (`alpha > 0` and
   `training is True`), which is what keeps an `alpha=0` model's weight set
   identical.
10. The draft gave `validate_all_variants(radius, grid_dim=2)` and
    `build_patch_tables(cell_to_unit, radius, distance, grid_dim=2)` signatures.
    Neither parameter exists as shipped (rows carry their own keys; the rank is
    read from the array).
11. The draft's `SpatialLayout` sketch (a `collapse_dims` bookkeeping value, a
    `(depth, height, width)` unravel, a `(d, h, w)` `cell_to_unit`) and its
    nested-loop centre enumeration with `(dz, dy, dx)` offsets are not what
    shipped; they are replaced below by the shipped code.
12. For `tiny3d` the draft's table said the "usable 3D radius" was 2. Radius 3
    also fits a 7-cell axis (one centre); radius 2 is the shipped choice because
    it leaves 27 centres.
13. The draft cited "GUIDE.md section 8" without a path; it is
    `src/dl_techniques/models/GUIDE.md`, section 8 ("Writing tests that can fail").
14. The draft's claim that a radius non-degenerate in 2D cannot become
    degenerate in 3D was never needed and is not asserted here; the degeneracy
    check in `build_patch_tables` carried over unchanged.

## Existing 2D machinery (review summary)

The implementation lives in two places, and the split is deliberate:

* `src/dl_techniques/layers/regularization/spatial_smoothness.py`: the
  model-agnostic topographic primitives.
  * `resolve_grid_shape(num_units, grid_shape=None, grid_dim=2)`: the ONE place
    grid arithmetic lives. In 2D it factors `num_units` into the most-square
    `(h, w)` through `_largest_divisor_at_most(num_units, math.isqrt(num_units))`
    (that function derives `isqrt` itself and ignores its `bound` argument, which
    is why the 3D factoring does not reuse it). In 3D it calls
    `_factor_3d_cubic`. An explicit `grid_shape` is validated, never used to
    infer the rank.
  * `SpatialLayout`: the unit-to-cell bijection. `perm[i]` is unit `i`'s cell in
    row-major order; `cell_to_unit` (shape `grid_shape`) is its inverse;
    `positions` holds one coordinate per axis per unit. Pure NumPy, no tensors.
  * `build_patch_tables(cell_to_unit, radius, distance)`: enumerates every
    fully-inside neighbourhood (centres run `radius .. n - 1 - radius` per axis)
    into `patch_table (num_centers, patch_units)`, builds the strict upper-triangle
    flat index `triu_flat`, and the mean-centred, unit-normalised inverse-distance
    prior `d_unit`, computed ONCE because `P = (2 * rho + 1) ** grid_dim` is
    constant across centres.
  * `spatial_smoothness_loss(activations, patch_indices, triu_flat, d_unit)`: the
    pure penalty function. It operates on a gathered `(K, P, M)` tensor and is
    dimension-agnostic. Its source is byte-identical before and after this change.
  * `SpatialSmoothness`: the identity tap. It builds the layout and tables as
    non-trainable weights (`perm`, `patch_table` and `triu_flat` as int64 through
    `INDEX_WEIGHT_DTYPE`, `d_unit` as float32) plus the recorded
    `last_spatial_loss` scalar, and adds `alpha * penalty` through `add_loss`
    when `training is True`. Its `call` code is unchanged by this work (one
    comment was reworded).
* `src/dl_techniques/models/language/topolm/{config,components,model}.py`: the
  model that composes them.
  * `config.py`: the variant registry, `DEFAULT_ALPHA = 2.5`, `DEFAULT_RADIUS = 5`,
    `DEFAULT_NUM_NEIGHBORHOODS = 5`, `validate_variant`, and
    `validate_all_variants`, run when `model.py` is imported.
  * `components.py`: `TopoLMBlock`, a bespoke decoder block (the shared
    `TransformerLayer` cannot reach its branch outputs before the residual add)
    holding `attn_out` and `mlp_out` as named locals and passing each through a
    `SpatialSmoothness` tap before the add. Per-tap seeds come from
    `permutation_seed(base_seed, tap_index)`.
  * `model.py`: `TopoLM` (token and learned position embeddings, blocks, final
    norm, a tied LM head or a `task_head`), `from_variant` and the `create_topolm`
    factory, and by-name strict checkpoint loading.

## Why 2D became a parameter, not a rewrite

`spatial_smoothness_loss` collapses any `(K, P, M)` gather, so the penalty itself
is dimension-blind. The dimensionality lives in exactly three places:

1. the factoring rule (`resolve_grid_shape`),
2. the layout bookkeeping (`SpatialLayout`'s positions and cell grid),
3. the neighbourhood enumeration and offset arithmetic (`build_patch_tables`).

The gather, the correlation matmul and the final dot product against `d_unit`
never learn how many axes the grid had. So the change is a `grid_dim`
parameter with branch points in those three places, a rank-aware validation layer
in `config.py` and the model, and plumbing through the block and model to every tap.

## Changes, file by file (as shipped)

Paths that do not start with `src/` or `tests/` are relative to `src/dl_techniques/`.

### 1. `layers/regularization/spatial_smoothness.py`

**`_factor_3d_cubic(num_units) -> (h, w, d)`**, copied from the shipped code
(docstring omitted):

```python
best_gap = None
best = (1, 1, num_units)
h = 1
while h ** 3 <= num_units:
    if num_units % h == 0:
        rest = num_units // h
        for w in range(h, math.isqrt(rest) + 1):
            if rest % w == 0:
                d = rest // w
                if best_gap is None or d - h <= best_gap:
                    best_gap, best = d - h, (h, w, d)
    h += 1
return best
```

The `<=` is what sends ties to the larger `h`, because `h` ascends. The shipped
values are in the next section. A brute-force oracle in the tests, which
enumerates every ordered triplet and shares no code with this function, agrees
for every width from 1 to 2000.

**`validate_grid_dim(grid_dim)`** (public, shared by the resolver, the layer, the
block and the model): returns `int(grid_dim)` if it is 2 or 3, else raises
`ValueError("grid_dim must be 2 or 3, got ...")`. `bool` and non-integers such as
`"3"` or `3.0` are refused, because a stringly rank would otherwise compare
unequal to 3 further down and fall through as 2D.

**`resolve_grid_shape(num_units, grid_shape=None, grid_dim=2)`**: validates
`grid_dim`; an explicit `grid_shape` must have exactly `grid_dim` entries, positive,
with product `num_units`. The 2D messages are byte-identical to before; the 3D
messages name `grid_dim`. A source anchor on the rank rule says why it must not be
simplified to `len(grid_shape)`.

**`SpatialLayout(num_units, grid_shape=None, permute=True, seed=None, grid_dim=2)`**:
the permutation is drawn before and independently of the rank, so one seed gives the
same `perm` in 2D and 3D. `positions` is
`np.stack(np.unravel_index(perm, grid_shape), axis=-1)`, shape `(N, grid_dim)`;
`cell_to_unit` is `argsort(perm).reshape(grid_shape)`. `to_grid` scatters the last
axis into `*grid_shape`; `from_grid` checks and flattens the last `grid_dim` axes.
`get_config` writes `grid_dim` explicitly; `from_config` reads
`config.get("grid_dim", 2)`.

**`NeighborhoodTables`** gains a trailing, defaulted `grid_dim: int = 2`, so
positional construction of a 2D record is unchanged; `grid_shape` is the full tuple
and `patch_units == side ** grid_dim`.

**`build_patch_tables(cell_to_unit, radius, distance="linf")`**: `grid_dim` is
`cell_to_unit.ndim`, which must be 2 or 3. Every axis must be at least `side`; the
2D text (`needs a 11x11 patch, but the grid is HxW`) is verbatim, and the 3D message
names each short axis. Centres are enumerated lexicographically, first axis slowest,
which in 2D is the row-major order of the original nested loop:

```python
rows = [
    cell_to_unit[tuple(slice(o, o + side) for o in offset)].reshape(-1)
    for offset in np.ndindex(*(n - 2 * radius for n in shape))
]
```

Offsets and the three metrics, one code path for both ranks (the 2D expressions are
unchanged):

```python
offsets = np.unravel_index(np.arange(patch_units), (side,) * grid_dim)
deltas = [np.abs(o[:, None] - o[None, :]) for o in offsets]
dy, dx = deltas[0], deltas[1]
if distance == "linf":
    separation = np.maximum(dy, dx)
    for extra in deltas[2:]:
        separation = np.maximum(separation, extra)
elif distance == "l1":
    separation = dy + dx
    for extra in deltas[2:]:
        separation = separation + extra
else:
    squared = dy ** 2 + dx ** 2
    for extra in deltas[2:]:
        squared = squared + extra ** 2
    separation = np.sqrt(squared)
```

(`dy` and `dx` are the first two axes' offset differences, not a naming of the axes.)
The degeneracy check (`d_norm == 0` raises "the neighbourhood is degenerate")
carries over verbatim.

**`SpatialSmoothness(..., eps=1e-6, grid_dim=2, **kwargs)`**: `grid_dim` is a
keyword after `eps`, so no positional caller shifts. It is validated, forwarded to
`resolve_grid_shape` and `SpatialLayout`, stored, written by `get_config`, and
formatted into the build log line as N axes. `spatial_smoothness_loss` is not touched.

### 2. `models/language/topolm/config.py`

* `grid_shape_of(embed_dim, grid_dim=2)` delegates to `resolve_grid_shape`.
* `validate_variant(name, config, radius, grid_dim=2)`: head divisibility, then the
  floor `(2 * radius + 1) ** grid_dim` (2D wording unchanged; the 3D message appends
  the largest usable radius), then the resolved grid with every axis at least
  `2 * radius + 1`. The axis test is exactly the condition `build_patch_tables`
  enforces, and a test sweeps every width below 300, radii 1 to 3 and both ranks
  (1794 cases) to check the validator and the tap build never disagree. For 2D this
  moves where an unbuildable width fails (a prime width, `1 x n`, now fails at
  validation, not at tap build); it does not change what builds.
* `axes_fit_problem(grid, radius)` and `radius_advice(grid)` are the shared wording
  used by `validate_variant` and the model constructor.
* `validate_all_variants(radius)`: described in the decisions table.
* The variant rows: next section.

### 3. `models/language/topolm/components.py`

`TopoLMBlock.__init__` gains `grid_dim: int = 2`, validated through
`validate_grid_dim`, stored, forwarded to every tap by `_make_tap`, and written by
`get_config`. A tap resolves its grid lazily at its own build, so a dropped argument
would leave a silently 2D tap in a 3D block; the forwarding has its own test. `build`
and `call` are untouched: the taps still sit on the branch outputs before the
residual add, are created unconditionally and built only for the sites in `tap_sites`,
and the `alpha = 0` control (identical weight set, identical permutations) is
unaffected by the rank.

### 4. `models/language/topolm/model.py`

* `TopoLM.__init__` gains `grid_dim: int = 2` after `grid_shape`; it is validated,
  forwarded to `_validate_config`, to the `grid_shape` property and to every block,
  and written by `get_config`. `from_config` stays `cls(**config)`, so a config
  without the key loads as 2D.
* `_validate_config`: the floor message uses `(2 * radius + 1) ** grid_dim`; the
  per-axis check follows for `grid_dim == 3` only (see the decisions table).
  `TopoLM(grid_dim=3)` at the default radius 5 therefore fails on every shipped
  width, and the message names the largest radius that does fit.
* The `grid_shape` property is typed `Tuple[int, ...]`: a pair in 2D, three axes in
  3D in the order `cell_to_unit.shape` has. The log lines that indexed two axes are
  rank-agnostic.
* `from_variant` and `create_topolm` have no signature change; the 3D rows carry
  `grid_dim` and `radius`, which flow through to the constructor like any other row
  key, and `create_topolm("paper3d", radius=3)` overrides the row's radius. There is
  no 3D-specific factory. The cross-rank checkpoint refusal is described in the
  decisions table.

### 5. Trainer and metrics (`src/train/topolm/`, `metrics/`)

* `common.py`: `_require_two_dimensional_variant` (the by-name refusal, checked on the
  variant's ROW before anything is built, because the trainer overrides a row's own
  radius with `spatial_radius`, default 5, which a 3D width cannot hold, so building
  first would raise a less helpful axis error), the rank check on the built backbone,
  and rank-agnostic grid label and report lines. The report's `grid_shape` for a 2D
  run is still `[h, w]`. `pretrain.py` has no new `add_argument`; only two help texts changed: `--variant`
  says `paper3d` and `tiny3d` are 3D layouts refused here, and the radius flag says a
  radius-r patch needs `(2r+1)^2` units "in this 2D trainer".
* `metrics/topographic_selectivity.py`: `grow_clusters` raises a named `ValueError`
  on a map that is not 2-D instead of dying in a bare `height, width = t_grid.shape`
  unpack. `plotting.py` and `metrics/spatial_autocorrelation.py` already refused a
  non-2-D map by name and are unchanged.
* `layers/regularization/gaussian_readout.py`: its behaviour is unchanged (a
  three-entry `grid_shape` is refused at build by the shared resolver); only its
  docstring says so.

## Grid factoring, verified

Run against the shipped `_factor_3d_cubic`; "largest usable radius" is
`(min axis - 1) // 2`:

| `n` | shipped triple | `d - h` | largest usable radius |
|---|---|---|---|
| 343 | `(7, 7, 7)` | 0 | 3 |
| 512 | `(8, 8, 8)` | 0 | 3 |
| 256 | `(4, 8, 8)` | 4 | 1 |
| 784 | `(7, 8, 14)` | 7 | 3 |
| 1000 | `(10, 10, 10)` | 0 | 4 |
| 997 | `(1, 1, 997)` | 996 | none |
| 1331 | `(11, 11, 11)` | 0 | 5 |

For 256 the rule picks `(4, 8, 8)` over `(4, 4, 16)` and `(2, 8, 16)` because its gap
is smaller. For 784 the paper's width does not re-factor to a cube: it resolves to
`(7, 8, 14)`, so a 784-wide 3D model at radius 4 is refused (the 3D floor 729 passes,
the 7-cell axis cannot hold a 9-wide patch) while radius 3 builds. 2D, for reference:
784 is `(28, 28)`, 512 is `(16, 32)`, 256 is `(16, 16)`.

**The objective is the spread, not the usable radius.** Minimising `d - h` (ties to the
larger `h`) is not the same as maximising the largest usable radius,
`(min_axis - 1) // 2`, and the two can disagree. Measured by script against a brute-force
search for the triple with the longest shortest axis, over every width from 1 to 20000:
exactly 8 widths get a smaller usable radius from the rule than another triple would host.
The first is 7425, which the rule factors to `(11, 25, 27)` (usable radius 5) while
`(15, 15, 33)` hosts radius 7. The other seven are 8775, 12285, 14280, 15120, 17556, 19800
and 19950. No shipped width is affected, no test pins the disagreement (the brute-force
oracle in `TestCubicFactoring` is an oracle for the rule's own objective, widths 1 to
2000), and the rule is left as decided; an explicit `grid_shape` overrides it.

## Variants

| variant | embed_dim | depth | heads (head dim) | ffn | max_seq_len | grid_dim | grid | radius | centres | provenance |
|---|---|---|---|---|---|---|---|---|---|---|
| `paper` | 784 | 12 | 16 (49) | 3136 | 1024 | 2 | 28 x 28 | 5 (default) | 324 | Rathi et al. 2025, Sec. 3; pinned |
| `small` | 512 | 6 | 8 (64) | 2048 | 512 | 2 | 16 x 32 | 5 (default) | 132 | repo-authored |
| `tiny` | 256 | 4 | 4 (64) | 1024 | 256 | 2 | 16 x 16 | 5 (default) | 36 | repo-authored |
| `paper3d` | 1000 | 12 | 8 (125) | 4000 | 1024 | 3 | 10 x 10 x 10 | 4 | 8 | repo-authored, not in the paper |
| `tiny3d` | 343 | 4 | 7 (49) | 1372 | 256 | 3 | 7 x 7 x 7 | 2 | 27 | repo-authored, not in the paper |

(Centres are `(a - 2r)(b - 2r)(c - 2r)` over the grid, recomputed by script.) 3D needs
wider widths than 2D: a radius-5 cube needs `11 ** 3 = 1331` units and 11 cells on
every axis, against 121 units in 2D. `paper3d` therefore uses 1000, the nearest cube
width to the paper's 784, at radius 4, the largest a 10-cell axis holds (a 9-cell
cube, `9 ** 3 = 729 <= 1000`). `tiny3d` is the smoke scale: 343 is the cube 7 x 7 x 7,
and radius 2 (a 5-cell cube, `5 ** 3 = 125`) leaves 27 centres.

**Few-centres caveat.** Centres are drawn with replacement, so with
`num_neighborhoods = 5` and only 8 or 27 admissible centres the same neighbourhood
can be drawn twice, and any two of `paper3d`'s cubes (9 cells wide on a 10-cell
axis) share at least 8 of the 9 cells along every axis. The sampled neighbourhoods
overlap heavily. This is a property of the shipped geometry, not a defect.

Only `paper` is quoted data and pinned by
`tests/test_variant_tables_match_upstream_references.py`, which is unmodified and
reads only that row; the CLI offers every row because the parser reads the table.

## Cost

Derived by arithmetic (`P = (2 * radius + 1) ** 3`, pairs `P * (P - 1) / 2`; the
correlation matmul is `(K, P, M) x (K, M, P)`, so its work scales as `K * M * P ** 2`).
Not measurements:

| neighbourhood | P | pairs | `P**2` against the paper's 2D | pairs against the paper's 2D | centres |
|---|---|---|---|---|---|
| 2D, 28 x 28, radius 5 (the paper) | 121 | 7,260 | 1 | 1 | 324 |
| 3D, radius 5 (every axis at least 11) | 1331 | 885,115 | 121.0 | 121.92 | 1 on 11 x 11 x 11, 8 on 12 x 12 x 12 |
| 3D, `paper3d`, radius 4 | 729 | 265,356 | 36.30 | 36.55 | 8 |
| 3D, `tiny3d`, radius 2 | 125 | 7,750 | 1.067 | 1.067 | 27 |

So a radius-5 cube costs about 121x (121 times) the paper's 2D neighbourhood in pairs and
in the matmul (`(1331 / 121) ** 2 = 121`, `885,115 / 7,260 = 121.9`), and the
shipped `paper3d` about 36x (36 times). The full 784-unit 2D grid has 306,936 pairs, which
is why the layer samples `K` neighbourhoods per step rather than scoring the grid.
`max_samples` (the cap on the batch-times-sequence rows) and a smaller 3D radius are
the mitigations the code offers.

**Measured** on the committed tree (the scripts and raw rows are in the plan's scratch
directory, which is not tracked, so these are a record, not something the suite
re-derives):

| quantity | configuration | measured |
|---|---|---|
| `build_patch_tables` wall time | CPU, radius 5, `linf`, 3 fresh processes; 11 x 11 x 11, 1 centre | median 0.060 s (0.059 to 0.061), +105 MB resident |
| same | 12 x 12 x 12, 8 centres | median 0.063 s (0.060 to 0.064), +105 MB resident |
| same, for reference | 28 x 28 (2D), 324 centres | median 0.0021 s, +1.5 MB resident |
| jitted training step, taps on | GPU 0 (RTX 4090), depth 2, vocabulary 2048, batch 8, sequence 256, `jit_compile=True`, SGD, median of three run medians of 30 steps | 2D width 784 radius 5: 12.8 ms; `tiny3d` width 343 radius 2: 5.2 ms; `paper3d` width 1000 radius 4: 21.7 ms |
| same, batch 4, sequence 64 | as above | 4.1 ms, 2.3 ms, 6.4 ms |
| tap share (taps on minus taps off, same configuration, batch 8, sequence 256) | as above | 2D 0.93 ms; `tiny3d` 0.39 ms; `paper3d` 6.09 ms |
| process peak GPU memory, taps on (TensorFlow allocator, excludes the CUDA context) | batch 8, sequence 256 | 860 MiB (2D 784); 291 MiB (`tiny3d`); 1357 to 1384 MiB (`paper3d`) |

The whole-step time of `paper3d` against the 2D width-784 step is 1.70 at batch 8 x 256
and 1.56 at batch 4 x 64 (derived from the medians above); the tap share alone is 6.6
and 3.9 times the 2D tap share. Both are far below the derived 36, because a step also
pays for embeddings, attention and the feed-forward, and because the widths differ
(1000 against 784).

**Caveats that travel with every timing.** GPU 0 was shared with another process whose
load during the timed loops is unknown, so these are timings of a shared card, not of an
isolated device. The 4 x 64 tap-share differences are of the order of the 0.1 to 0.5 ms
run-to-run spread and are weak evidence; the 8 x 256 figures are the better ones. The
vocabulary was 2048 and the optimiser plain SGD, so none of this says anything about
the cost of a full-vocabulary head. No radius-5 3D training step was timed: the shipped
3D widths have axes of 10 and 7, which cannot hold radius 5, so the 121 above is a
derived cost of the table and the matmul, and the only measured radius-5 3D number is
the table build.

## Sign probe, measured

The 2D suite's sign-discriminating pair (one smooth field scored against the prior and
against its negation) reads 0.174 and 0.826. In 3D the pair was measured, not asserted
by symmetry: on 512 units laid out `(8, 8, 8)`, radius 2 (64 centres), a smooth 3D field
of 256 samples and five fixed centres, the forward score (`+d_unit`) is **0.2002** and
the backward score (`-d_unit`) **0.7998**, summing to 1.0. Over three centre draws (seeds 1, 2
and 3, the first being the pair above) the forward score reads 0.2002, 0.2066 and
0.1701. On 1000 units `(10, 10, 10)` at
radius 4, which has only 8 centres, the pair reads 0.3311 and 0.6689: still on the right
sides of one half, but close enough to the 0.30 and 0.70 bands used for the assertion
that it is recorded in the test docstring and not asserted.

## Tests, as they exist

Pinned numbers, not shape-only asserts (a tolerance below the dtype's resolution
measures nothing, and a shape assert is how guards went blind;
`src/dl_techniques/models/GUIDE.md` section 8). The commit messages of the model and
trainer steps (`1e64b58ec`, `85e03cc74`) record the mutations of the tracked files that
were seen failing.

* `tests/test_layers/test_regularization/test_spatial_smoothness.py`:
  `TestTwoDimensionalIsByteIdentical` pins the 2D layer byte for byte: sha256 digests
  of `perm`, `positions`, `cell_to_unit`, `patch_table`, `triu_flat` and `d_unit` for
  three configurations (784, 512, 256 units), the `l1` and `l2` `d_unit` digests,
  dtypes and shapes, the pinned error substrings, the config keys, and a config without
  `grid_dim`. It was written green before any source edit and stays green.
* `tests/test_layers/test_regularization/test_spatial_smoothness_3d.py`:
  `TestCubicFactoring` (the pin table above, plus a brute-force oracle for every width
  from 1 to 2000), `TestResolveGridShape3D`, `TestSpatialLayout3D`, `TestPatchTables3D`
  (centre order, block-around-centre, the non-cubic axis-order value pin on a
  `(5, 6, 7)` grid at radius 2 with 6 centres in
  `test_a_non_cubic_grid_pins_the_axis_order_by_value`, all three metrics against an
  independent transcription, per-axis rejection in
  `test_one_short_axis_is_refused_even_when_the_total_is_large`, and the term-for-term
  loss comparison), `TestSpatialSmoothness3DConstruction`,
  `TestSpatialSmoothness3DBuild`, `TestSpatialSmoothness3DConfigAndSign`
  (`test_the_negated_prior_scores_far_above_one_half`).
* `tests/test_layers/test_regularization/spatial_smoothness_oracle.py`: independent 3D
  helpers `reference_patch_units_3d`, `reference_d_unit_3d` and
  `smooth_field_activations_3d`, written as explicit loops with no `np.unravel_index`;
  their mutation self-tests are in `test_spatial_smoothness_oracle.py`
  (`TestTheThreeDimensionalOracleCanDiscriminate`,
  `TestTheThreeDimensionalHelpersAgainstHandDerivedValues`).
* `tests/test_layers/test_regularization/test_gaussian_readout.py`: `TestGridRankRefusal`
  pins that the readout refuses a three-axis grid, both a 3-tuple and the grid of a 3D
  layout.
* `tests/test_models/test_topolm/test_topolm_3d.py`: `TestThreeDimensionalValidation`,
  `TestTwoDimensionalHole`, `TestValidateAllVariants`,
  `TestValidationAgreesWithTheTapBuild`, `TestBlockGridDim`, `TestModelGridDim`,
  `TestTheAlphaControlInThreeDimensions`, `TestThreeDimensionalValidationThroughTheModel`
  (including `test_a_two_dimensional_model_still_constructs_with_a_short_axis`),
  `TestTheAxisOrderReachesTheTaps`, `TestThreeDimensionalSerialization`,
  `TestThreeDimensionalPrecisionAndDevice`, `TestACheckpointOfTheOtherRankIsRefused`,
  `TestTheGuardReadsOnlyATopoLMConfig` (a wrapped archive gets the loader's own error in
  all four rank combinations), `TestACheckpointOnAnotherGridIsRefused` (permuted 3D grids,
  a transposed 2D grid, a same-grid load with another seed),
  `TestADocumentedLimitationOfTheGuard` (a direct `load_backbone_from_checkpoint` call does
  not see the rank), `TestThreeDimensionalVariantRows`. The existing 2D guard
  `test_an_explicit_grid_is_reported_and_used` in `tests/test_models/test_topolm/test_topolm.py`
  is unchanged; that file's only edit adds `grid_dim` to the config-key tuple and makes
  the table loop rank-aware.
* `tests/test_train/test_topolm/test_topolm_trainer.py`:
  `TestThreeDimensionalVariantsAreRefused` (by name, before any directory or data work:
  `test_train_topolm_refuses_before_any_directory_or_data_work`) and
  `TestTheGridIsReportedAtItsOwnRank`.
* `tests/test_metrics/test_topographic_selectivity.py`:
  `test_a_map_that_is_not_two_dimensional_is_named_not_unpacked` pins the `grow_clusters`
  guard.
* GPU coverage: one 3D model arm, `test_a_jit_compiled_fit_runs_on_a_gpu` in
  `TestThreeDimensionalPrecisionAndDevice` (depth 1); the layer-level 3D file has no GPU
  arm. The measured jitted steps above also ran 3D models under XLA on the same card.

## Sweep (what was checked, what remains)

* `resolve_grid_shape` has two callers outside the spatial layer and the topolm package:
  `GaussianReadout` (stays 2D, refusal pinned) and the topolm config and model (which
  state the rank explicitly).
* The trainer's four places that unpacked the grid as `(height, width)` are
  rank-agnostic, with 2D output identical, and a 3D variant never reaches them.
* Follow-ups, deliberately not built: a 3D `GaussianReadout`, 3D Moran's I
  (`metrics/spatial_autocorrelation.py`), 3D cluster growth, and 3D t-map plots. They
  are recorded in the trainer README.

## Numerical notes (carried over, unchanged in 3D)

* The three index tables (`perm`, `patch_table`, `triu_flat`) stay int64 weights via
  `INDEX_WEIGHT_DTYPE`: TensorFlow keeps int32 variables in host memory where an
  XLA-compiled GPU step cannot read them. `d_unit` is float32.
* The correlation block runs at `promote_types(input_dtype, float32)`.
* The `eps` floor on the per-unit standard deviation and the epsilon-guarded
  denominator in the final dot product are untouched.
* `patch_units` grows fast in 3D (121 to 1331 at radius 5), and with it the pair count
  and the matmul: see the cost section for the derived factor and the measured table
  build.

## Implementation history

| commit | step | content |
|---|---|---|
| `635747fa5` | 1 | golden 2D digests, independent 3D oracle helpers and their self-tests, 3D layer tests written failing first; no source change |
| `3dd7ee0e0` | 2 | `_factor_3d_cubic`, `resolve_grid_shape`, `SpatialLayout`, `NeighborhoodTables`, `build_patch_tables` for 2 or 3 axes, `GaussianReadout` refusal tests |
| `c2cac9ef5` | 3 | `SpatialSmoothness` `grid_dim`, the 3D layer tests, the measured sign probe |
| `246a42b67` | 4 | rank-aware `config.py` validation, per-axis, both ranks, no rows yet |
| `1e64b58ec` | 5 | `TopoLMBlock` and `TopoLM` `grid_dim`, 3D model tests, cross-rank checkpoint refusal |
| `85e03cc74` | 6 | `paper3d` and `tiny3d` rows with the trainer refusal and the `grow_clusters` guard |
| `eea1dbe7f` | 8 | documentation with derived and measured numbers (step 7, the measurement run, produced no commit) |
| `805be4ba2` | 5.1 | completion fix after the adversarial review: the checkpoint guard reads the rank only from a bare TopoLM config and also compares the resolved grid (D-017) |
| `4a71059a0` | 6.1 | completion fix: the trainer test's table loop reuses its local `grid_dim` (cosmetic) |
| step 8.1 | 8.1 | completion fix: documentation of the corrected guard, its uncovered direct-loader route, the downgrade note and the factoring-objective caveat (this commit; documents only) |

The rows landed together with the trainer refusal on purpose: earlier, a selectable
3D variant would have reached a trainer that could not evaluate it.

## Open questions, answered

* **Exact home of the 3D variant rows**: rows in `MODEL_VARIANTS`, each carrying its own
  `grid_dim` and `radius`; no second table.
* **Whether the trainer's plotting wants a 3D view**: no. The trainer is 2D-only and
  refuses a 3D variant by name; 3D plots are a recorded follow-up.
* **Whether `SpatialLayout.get_config` carries `grid_dim` explicitly**: yes, explicitly,
  and `from_config` defaults a missing key to 2. A value that reads back as something
  derivable is how the `alpha = 0` lesson happened.

## References

* Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, 2025. TopoLM:
  Brain-like spatio-functional organization in a topographic language model.
  ICLR 2025. (https://arxiv.org/abs/2410.11516)
* Lee, Margalit, Jozwik, Cohen, Kanwisher & DiCarlo, 2020. Topographic deep
  artificial neural networks. (https://arxiv.org/abs/2007.09019)
* Repo: `src/dl_techniques/models/language/topolm/README.md` (the objective, the 3D
  section, the variants and the cost model), `src/train/topolm/README.md` (what the
  trainer does not do), and the layer docstring in `spatial_smoothness.py`.
