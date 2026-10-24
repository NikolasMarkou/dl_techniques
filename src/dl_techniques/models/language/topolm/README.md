# TopoLM: a language model whose units are laid out on a sheet of tissue

> Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, *TopoLM: Brain-like
> spatio-functional organization in a topographic language model*, ICLR 2025.
> ([arXiv:2410.11516](https://arxiv.org/abs/2410.11516))

---

## 1. Overview

A GPT-2 block indexes its units by nothing but their position on the last axis of
a weight matrix. Two units that learn to encode the same thing have no reason to
sit near each other, so the emergent clustering in a trained model is a property
of the initialisation rather than of anything the objective asked for.

TopoLM gives the units of every attention and feed-forward branch a distinct cell
on an `h x w` grid (`h * w == embed_dim`; an `h x w x d` box with `grid_dim=3`) and adds a penalty alongside the
next-token cross-entropy that makes nearby units co-activate. Functional clusters
then emerge as a consequence of the objective rather than as a post-hoc
clustering of whatever the model happened to learn.

```bash
python - <<'PY'
from dl_techniques.models.language.topolm import create_topolm

topographic = create_topolm("paper")                # alpha = 2.5
control     = create_topolm("paper", alpha=0.0)     # identical weights, no loss
ablation    = create_topolm("paper", permute=False) # the paper's Fig. 12
PY
```

**No pretrained weights are distributed.** `pretrained=True` raises
`NotImplementedError` at every entry point, and the message names the alternative.

**Two grid ranks.** `grid_dim=2` (the default) is the paper's layout. `grid_dim=3`
is a repo-authored extension: the units sit in an `h x w x d` box with
`h <= w <= d` and `h * w * d == embed_dim`, and a neighbourhood is a cube of
`(2*rho + 1)**3` units. The paper is 2D throughout, so nothing here about 3D
reproduces a published number. A config saved before the key existed loads as 2D.
The Gaussian readout stays 2D-only; the trainer, the post-hoc analysis and the plots
take both ranks (sections 6, 7 and 8).

```python
volume = create_topolm("tiny3d")   # 343 units on a 7 x 7 x 7 box, radius 2
```

---

## 2. The objective

For one square neighbourhood of radius `rho` centred on the grid (a cube when
`grid_dim=3`) — covering `P = (2*rho + 1)**grid_dim` units — the penalty compares
two things that should agree:

- `r`, the Pearson correlation of every unit pair's activations across the `M`
  samples in the batch;
- `d`, the inverse distance `1 / (dist + 1)` of the same pairs, mean-centred and
  unit-normalised.

```text
SL = 0.5 * (1 - corr(r, d))          in [0, 1], 0 == smooth
```

The `0.5` is what puts the range at `[0, 1]`, since `corr` is in `[-1, 1]`.
Minimising `SL` makes near units co-activate and pushes distant units apart in
correlation, an efficient proxy for short wiring length.

Implemented in
[`layers/regularization/spatial_smoothness.py`](../../../layers/regularization/spatial_smoothness.py).

### The whole grid is never scored at once

`r` for all `N` units is `N*(N-1)/2` numbers — 306,936 of them at the paper's
`N = 784`, a 2D grid — computed from a `K x M x P` gather every step. The loss instead
samples `K` neighbourhoods per step, which is what makes it affordable, and is
why centres are restricted to those whose patch fits entirely inside the grid: a
constant `P` means `d` is identical for every neighbourhood and can be
precomputed once, at build time, and shared.

At the paper's settings: `P = 121`, `Q = 7260` pairs each, `C = 324` admissible
centres.

### The same objective in 3D

`P` and the pair count `Q = P*(P-1)/2` change; nothing else in the loss does. Every
number in the first table is arithmetic on the grid, not a measurement
(`C = (a-2r)(b-2r)(c-2r)` for an `a x b x c` box):

| neighbourhood | units `P` | pairs `Q` | centres `C` |
|---|---|---|---|
| 2D, 28 x 28, radius 5 (the paper) | 121 | 7,260 | 324 |
| 3D, radius 5 (every axis must be at least 11) | 1331 | 885,115 | 1 on 11 x 11 x 11, 8 on 12 x 12 x 12 |
| 3D, `paper3d`, 10 x 10 x 10, radius 4 | 729 | 265,356 | 8 |
| 3D, `tiny3d`, 7 x 7 x 7, radius 2 | 125 | 7,750 | 27 |

The correlation matmul is `(K, P, M) x (K, M, P)`, so its work scales as
`K * M * P**2` (`K` = `num_neighborhoods`, `M` = samples per step). Against the
paper's 2D neighbourhood that is about 121x, a factor of `(1331/121)**2 = 121` for a radius-5
cube (885,115 / 7,260 = 121.9 in pairs) and about 36x for `paper3d`'s radius 4
(`P**2` ratio 36.3, pair ratio 36.55). **Derived, not measured.** No radius-5 3D
model exists to time: a radius-5 patch needs 11 cells on every axis, and the
shipped 3D widths (1000 and 343) have axes of 10 and 7.

What was measured (CPU table build and a jitted training step on GPU 0; the scripts
and raw rows live in the 3D plan's scratch directory, which is not tracked, so these
numbers are a record, not something the test suite re-derives):

| quantity | config | measured |
|---|---|---|
| `build_patch_tables` wall time | CPU, radius 5, `linf`, 3 fresh processes; 11 x 11 x 11 with 1 centre | median 0.060 s (0.059 to 0.061), +105 MB resident |
| same | 12 x 12 x 12 with 8 centres | median 0.063 s (0.060 to 0.064), +105 MB resident |
| same, for reference | 28 x 28 (2D) with 324 centres | median 0.0021 s, +1.5 MB resident |
| jitted training step, taps on | GPU 0 (RTX 4090), depth 2, vocabulary 2048, batch 8, sequence 256, `jit_compile=True`, SGD, median of three run medians of 30 steps | 2D width 784 radius 5: 12.8 ms; `tiny3d` width 343 radius 2: 5.2 ms; `paper3d` width 1000 radius 4: 21.7 ms |
| same, batch 4, sequence 64 | as above | 4.1 ms, 2.3 ms, 6.4 ms |
| tap share (taps on minus taps off, same config, batch 8, sequence 256) | as above | 2D 0.93 ms; `tiny3d` 0.39 ms; `paper3d` 6.09 ms |
| process peak GPU memory, taps on (TensorFlow allocator, excludes the CUDA context) | batch 8, sequence 256 | 860 MiB (2D 784); 291 MiB (`tiny3d`); 1357 to 1384 MiB (`paper3d`) |

The whole-step time of `paper3d` against the 2D width-784 step is 1.70 at batch 8 x
256 and 1.56 at batch 4 x 64 (derived from the medians above); the tap share alone
is 6.6 and 3.9 times the 2D tap share. Both are far below the derived 36 because a
step also pays for embeddings, attention and the feed-forward, and because the
widths differ (1000 against 784).

**Caveats that travel with every timing above.** GPU 0 was shared with another
process whose load during the timed loops is unknown, so these are timings of a
shared card, not of an isolated device; the 4 x 64 tap-share differences are of the
order of the 0.1 to 0.5 ms run-to-run spread and are weak evidence (the 8 x 256
figures are the better ones). The vocabulary was 2048 and the optimiser plain SGD, so
none of this says anything about the cost of a full-vocabulary head. No radius-5 3D
step was timed.

**Few centres.** A 3D box leaves few admissible centres once the radius is large:
`paper3d` has 8 and `tiny3d` 27. Centres are drawn with
replacement, so with `num_neighborhoods = 5` the same neighbourhood can be drawn
twice, and any two of `paper3d`'s cubes (9 cells wide on a 10-cell axis) share at
least 8 of the 9 cells along every axis. The sampled neighbourhoods therefore overlap
heavily. That is a property of the shipped geometry, recorded here, not a defect.

### Measured behaviour

784 units on a 28x28 grid, radius 5, 5 sampled neighbourhoods, 256 samples:

| stimulus | `SL` |
|---|---|
| spatially smooth field (lowest Fourier modes) | 0.172 |
| same field, distance prior negated | 0.826 |
| i.i.d. noise | 0.496 |
| wavelength finer than the neighbourhood (`f=9`) | 0.526 |

Two of those rows are load-bearing. The negation row is the **sign-discriminating**
probe: 0.174 and 0.826 are not reachable by loosening a tolerance, so nothing
about `d`'s direction or scale is unconstrained. And the `f=9` row is the
anti-smooth arm — a field whose wavelength is much finer than the neighbourhood
is anti-smooth *at that neighbourhood's scale*. A checkerboard does **not**
demonstrate this: at l-infinity separation 1 it has three offset classes, two of
them negative, so it averages to zero correlation and scores exactly what noise
scores.

The 3D sign probe, measured on the real implementation before any bound was
written (512 units on an 8 x 8 x 8 box, radius 2, 64 centres, 256 samples of a
smooth 3D field, 5 sampled neighbourhoods): the true prior scores **0.2002** and
the negated prior **0.7998**. Over centre-draw seeds 1, 2 and 3 the true prior reads
0.2002, 0.2066 and 0.1701 and the negated one 0.7998, 0.7934 and 0.8299 (the exact
text is in the docstring of `test_the_negated_prior_scores_far_above_one_half` in
`tests/test_layers/test_regularization/test_spatial_smoothness_3d.py`). On a larger
box (1000 units, 10 x 10 x 10, radius 4, only 8 centres) the same pair is 0.3311 and
0.6689: the right side of one half, but too close to the 0.30 / 0.70 bands to
assert, so it is recorded only.

---

## 3. Architecture

```text
input_ids [B, S]
     │
     ▼
┌─────────────────────────┐
│ token + learned position│
└─────────────────────────┘
     │ [B, S, D]
     ▼
┌─────────────────────────┐
│ TopoLMBlock x depth     │◄── rank-3 causal mask
│  taps on both branches  │
└─────────────────────────┘
     │
     ▼  last_hidden_state [B, S, D]
┌─────────────────────────┐
│ final LayerNorm         │
└─────────────────────────┘
     │
     ├──► {"last_hidden_state"}
     ▼
tied matmul(token emb.T)  |  lm_head Dense
     │
     ▼  logits [B, S, vocab_size]
```

Both leaves come back together, so
[`CausalLanguageModel`](../../common/masked_language_model/) and the standard CLM
data wrapper work unchanged. No `None` is echoed into the output dict, so
`predict({"input_ids": ...})` does not trip Keras' nested-structure check.

### The block owns its taps, and why

Every architectural piece a TopoLM block needs already exists in
`layers/transformers/`, and none of them can be used here.
`TransformerLayer.call` computes the attention and feed-forward branches and then
adds each straight into the residual stream; in the pre-norm path the branch
tensor is a local overwritten by stochastic depth and layer scale before the
residual add, so it is gone by the time `call` returns. The spatial loss has to be
computed on that tensor, at exactly that point. There is no flag, no hook and no
`activity_regularizer` that reaches it — and `TextDecoder` has no per-block config
to hang a tap on, and only `gpt2/` uses it anyway.

So [`TopoLMBlock`](components.py) is written here, and it **composes** the shared
`create_attention_layer` / `create_ffn_layer` / `create_normalization_layer`
factories rather than reimplementing any of them. The reuse order is unchanged:
factories first, a bespoke layer last. Precedent: `hnet`, `wave_field`,
`tree_transformer`, `qwen3_next`.

The taps sit **on** the branch outputs, before the residual add. Their forward
pass is the identity, so the residual arithmetic is byte-identical to a plain
GPT-2 block and any difference in the loss curve is attributable to the added term
alone.

**The taps carry `grid_dim`.** `TopoLMBlock` and `TopoLM` take `grid_dim` (2 or 3),
validate it, forward it to every tap, and write it in `get_config`; a config without
the key loads as 2D. It is a stated value and is never inferred from
`len(grid_shape)`: a tap resolves its grid lazily at its own build, so a dropped
argument would leave a silently 2D layer inside a model that reports `grid_dim=3`.
Each forwarding site has its own test (`TestBlockGridDim` and `TestModelGridDim` in
`tests/test_models/test_topolm/test_topolm_3d.py`).
`TopoLM.grid_shape` reports the resolved grid, `grid_dim` entries long, in the axis
order of the taps' `cell_to_unit` arrays; nothing reorders the axes and there is no
depth/height/width naming in 3D.

---

## 4. Two choices that are not the obvious ones

**Each tap permutes its units independently** (`permute=True`, the default).
Without it the residual stream hands every layer the same spatial pattern, the
loss is satisfied by copying one map forward rather than by each layer learning
its own, and the model ends up with one map instead of `depth` of them. That is
the paper's Fig. 12 ablation; `--no-permute` reproduces it.

**`alpha = 0` is a real control, not an absence of one.** The taps are still
created, still built, and still derive their layouts from the same seeds, so the
control has a byte-identical weight set and differs from the trained arm in its
objective alone. `tests/test_models/test_topolm/` pins the identical weight paths
*and* the identical permutations; without that, "identical hyperparameters" would
be a claim rather than a measured fact.

The control holds in 3D too, pinned by `TestTheAlphaControlInThreeDimensions`
(`test_alpha_zero_keeps_every_weight`,
`test_alpha_zero_keeps_every_permutation_and_table_bit_identical`,
`test_alpha_zero_adds_no_loss`). The 3D aggregate-loss check was re-measured rather
than copied from 2D (216 units on a 6 x 6 x 6 box, radius 2, depth 2,
`alpha = 2.5`, 4 taps, SGD 1e-3, one epoch over 2 batches, CPU, float32): the
reported training loss is 10.377 with `aggregate_backbone_losses` and 5.373
without, a gap of **5.004**, against a derived 4 taps * 2.5 * 0.5 = 5.0; the
evaluation loss is 5.323 either way
(`test_the_reported_training_loss_includes_the_spatial_term`).

---

## 5. Variants

| variant | embed_dim | depth | heads | max_seq_len | grid_dim | grid | radius | provenance |
|---|---|---|---|---|---|---|---|---|
| `paper` | 784 | 12 | 16 | 1024 | 2 | 28 x 28 | 5 (default) | Rathi et al. 2025, Sec. 3 |
| `small` | 512 | 6 | 8 | 512 | 2 | 16 x 32 | 5 (default) | repo-authored |
| `tiny` | 256 | 4 | 4 | 256 | 2 | 16 x 16 | 5 (default) | repo-authored |
| `paper3d` | 1000 | 12 | 8 | 1024 | 3 | 10 x 10 x 10 | 4 | repo-authored, not in the paper |
| `tiny3d` | 343 | 4 | 7 | 256 | 3 | 7 x 7 x 7 | 2 | repo-authored, not in the paper |

Only `paper` is quoted data, and only `paper` is pinned by
`tests/test_variant_tables_match_upstream_references.py`. `small`, `tiny`,
`paper3d` and `tiny3d` are repo-authored: the first two are scales of the paper's
shape for smoke runs and CI, and the 3D pair is a layout the paper never uses. None
of the four exists in the paper and none is pinned, because quoting a derived number
as if it were published is exactly the false citation that guard exists to prevent.
The two 3D rows carry their own `"grid_dim": 3` and `"radius"` keys, which
`from_variant` passes to the constructor (`description` is dropped), so
`create_topolm("paper3d", radius=3)` overrides the row's radius.

784 is not a free parameter — it is the only plausible hidden size near GPT-2
small that factors into a **square** unit grid (784 = 28 x 28), which is what the
spatial loss needs. `small`'s 16 x 32 grid is deliberately non-square so the
orientation probes have a transposed stride to miss.

In 3D 784 is the wrong width: it does **not** factor into a cube. The rule is the
most cubic triple `(h, w, d)` with `h <= w <= d` and `h * w * d == embed_dim`,
minimising `d - h` and breaking ties towards the larger `h`, in integer arithmetic
only. It gives 784 -> (7, 8, 14), 343 -> (7, 7, 7), 1000 -> (10, 10, 10) and
256 -> (4, 8, 8). The shortest axis of (7, 8, 14) holds a radius of at most 3, so a
784-wide 3D model at radius 4 is refused. `paper3d` therefore uses 1000, the nearest
cube width (head dimension 125 with 8 heads), at radius 4, the largest a 10-cell axis
holds; that leaves `(10-8)**3 = 8` centres. `tiny3d` is 343 = 7 x 7 x 7 at radius 2
(27 centres). The grid is `(axis0, axis1, axis2)` and equals the taps'
`cell_to_unit.shape`, row-major.

The rule's objective is the spread `d - h`, which is **not** the same as the largest
usable radius, `(min_axis - 1) // 2`. Minimising the spread can pick a triple whose
shortest axis is shorter than another triple's. Measured against a brute-force search
for the triple with the longest shortest axis, over every width up to 20000: exactly 8
widths get a smaller usable radius than another triple would host. The first is 7425,
which the rule factors to `(11, 25, 27)` (usable radius 5) while `(15, 15, 33)` would
host radius 7. The others are 8775, 12285, 14280, 15120, 17556, 19800 and 19950. No
shipped width is affected (784, 343, 1000 and 256 resolve as stated above), and the rule
is kept as shipped; a model that needs a larger radius at such a width passes an explicit
`grid_shape`.

Every variant is validated at import (`validate_all_variants`): a width that
cannot host a radius-`r` patch, or that does not divide among the heads, fails on
the import that introduced it rather than on whichever run selects that row first.
The check now covers every axis of the resolved grid, not just the unit count: the
floor `embed_dim >= (2r+1)**grid_dim` is necessary but not sufficient (784 units at
radius 4 clear the 3D floor of 729 and still resolve to (7, 8, 14)), so each axis
must be at least `2r+1` and the error names the short axis and the largest usable
radius, `(min_axis - 1) // 2`. Each row is judged by its own `radius` and `grid_dim`
and falls back to radius 5 and 2D when it has none, so the rows that predate the keys
behave as before. In 2D this moves *where* an unbuildable width fails (997 = 1 x 997
now fails at validation, not at tap build); it does not change what builds.

**The model constructor is stricter in 3D than in 2D, on purpose.** `TopoLM`'s own
per-axis check is 3D-only. A directly constructed 2D model with an explicit
`grid_shape` that has a short axis at the default radius (an `(8, 32)` grid at radius
5) constructs exactly as it always has, reports its grid, and fails only when a tap
builds; the existing tests pin that
(`test_a_two_dimensional_model_still_constructs_with_a_short_axis`). The variant table
(`validate_variant`) is strict in both ranks, so no shipped row can reach that hole.

---

## 6. Training

The auxiliary term arrives through `add_loss`, not a custom `train_step`, so stock
`fit` still scales the loss under mixed precision and still reports the **pure
task loss** at validation time. A training head therefore needs:

```python
from dl_techniques.models.common.masked_language_model import CausalLanguageModel

head = CausalLanguageModel(
    backbone=create_topolm("paper"),
    vocab_size=100277,
    skip_head=True,
    output_key="logits",
    aggregate_backbone_losses=True,   # load-bearing
)
```

`aggregate_backbone_losses=True` is not a default worth overriding. The taps reach
the objective only through `backbone.losses`; without the flag the head computes
every penalty, discards it, and the run produces a non-topographic model with a
perfectly healthy loss curve. Measured on the same weights at `alpha = 2.5`: a
reported training loss of **15.31** aggregated against **5.36** not, with an
evaluation loss of 5.31 either way.

[`train/topolm/`](../../../../train/topolm/) is the trainer. Two things about it
are worth knowing before reading its code, both of which would otherwise look like
mistakes: the validation cadence *is* the virtual epoch length, and the head must
be told to aggregate the backbone's losses.

**The trainer trains both ranks.** `--variant paper3d` and `--variant tiny3d` train at
the variant row's own radius (`--radius` unset resolves per variant: 4 and 2; the 2D
variants keep the paper's 5), and the post-hoc evaluation, the figures and the live view
(`--viz-every-steps`) follow the grid's rank. The Gaussian readout is 2D, so a 3D run
scores its raw arm only: the default `--readout-fwhm` resolves to no readout on a 3D
variant, a positive value typed for one is refused by flag and variant, and the report
says `readout_skipped`. There is still no `--grid-dim` flag; the rank comes from the
variant. See `src/train/topolm/README.md` (*Training a 3D variant*). Until the
placement correction recorded there, the trainer put each tap's t-map on the grid with a
plain `reshape`, so 2D Moran's I and cluster numbers produced under `permute=True` should
be recomputed. The 3D path is pinned by `TestThreeDimensionalVariantsTrain` and
`TestTheTrainerTrainsA3DVariant` in `tests/test_train/test_topolm/`.

### A task head in the model (`include_top`)

```python
import numpy as np
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.models.language.topolm import create_topolm

sst2 = NLPTaskConfig(name="sst2", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=2)
clf = create_topolm("tiny", vocab_size=512, include_top=True, task_config=sst2,
                    pad_token_id=0)  # the pad id of THIS data; None = no padding

ids = np.random.randint(1, 512, size=(2, 16)).astype("int32")
ids[1, 10:] = 0                                    # right padding (id 0 is the pad id here)
print(clf.task_head.pooling_type)                  # 'last'
print(clf(ids).shape)                              # (2, 2) -- bare logits, no LM logits
clf(ids, training=True)
print(len(clf.losses))                             # 8 -- the taps still fire

# Warm-start another head from this one: the backbone is copied by layer name, the
# checkpoint's task_head never is.
clf.save("topolm_sst2.keras")
ner = create_topolm(
    "tiny", vocab_size=512, include_top=True, pretrained="topolm_sst2.keras",
    task_config=NLPTaskConfig(name="ner", task_type=NLPTaskType.TOKEN_CLASSIFICATION,
                              num_classes=9),
)
print(ner(ids).shape)                              # (2, 16, 9)
```

With `include_top=True` and a `task_config` the model owns one head from
`layers/heads/nlp`, named `task_head`, and returns only its output (a bare tensor, or a dict for
question answering); no LM logits are computed and an untied `lm_head` is not created. TopoLM is
causal, so the head pools the **last real token** (`pooling_type='last'`) under the input's
`attention_mask`, else `input_ids != pad_token_id` (`pad_token_id=None` by default: no padding, the
head sees every position, because id 0 is a real `cl100k_base` token; pass your pad id or a mask); `head_config_overrides={"pooling_type": ...}`
replaces the default. The blocks never see that derived mask, run unchanged, and keep adding the
smoothness terms through `add_loss`, so a fine-tune under `fit` still pays the spatial penalty
unless `alpha=0.0`. The backbone's weight paths are the same with or without a head.

`pretrained=<path>` must be a `.keras` file, headed or headless. It copies the backbone (and an
untied `lm_head`) by layer name and never the checkpoint's `task_head`; the model's own head keeps
its fresh initialization. Every backbone layer loads, and every checkpoint layer finds its place,
or the call raises `ValueError`, so a checkpoint of another depth (deeper or shallower) or width
is refused, never half-restored or truncated. Two more refusals come from
`from_variant(pretrained=)` itself, before any weight is copied, because a 2D and a 3D
grid (or two grids of the same width) can give the tap tables identical shapes, so the
loader's shape check alone let such a load through:

- **A bare TopoLM archive saved at the other `grid_dim`** is refused by name. A missing
  `grid_dim` key in a bare TopoLM config means 2D, the layout of every archive written
  before the key existed.
- **An archive whose resolved unit grid differs from the model's** is refused by name,
  naming both grids and the `grid_shape` to build with: a 3D `(10, 5, 20)` archive into a
  model on `(20, 5, 10)` (every table has the same shape there, because the centre count is
  symmetric in the axes), or a transposed 2D grid such as `(8, 32)` into `(32, 8)`. Both
  sides are resolved, so a model left on `grid_shape=None` is compared by the grid it
  factors to. A different seed on the same grid still loads. In 2D this tightens one
  thing: a load across two grids of one width, which used to succeed with an inconsistent
  layout, is now refused.

The guard does not guess. It reads the rank and the grid only from an archive whose
top-level config is a bare TopoLM; for any other archive it stays out of the way. A
`CausalLanguageModel` wrapped around a TopoLM has no `grid_dim` key at its top level, and
`from_variant` has never been able to warm-start from one, in either rank: the loader's
own `ValueError` ("No overlapping layers between target and source checkpoint", measured
for all four rank combinations) stands, and it is not replaced by a made-up rank. A width
mismatch likewise keeps the loader's `Shape mismatch` error, which names the real cause.
A direct call to `load_backbone_from_checkpoint` is not covered by the rank or the grid
guard, only `from_variant(pretrained=)` is: a 3D 9 x 9 x 9 radius-4 archive loaded that
way into a 2D 27 x 27 radius-13 model of equal table shapes returns a report normally and
the model keeps reporting `(27, 27)`, pinned by
`TestADocumentedLimitationOfTheGuard::test_a_direct_loader_call_does_not_see_the_rank`
(a change that guards the loader has to update that test and this sentence together).

An archive written by this code carries a `grid_dim` key in the model config, 2 as well as
3. A checkout from before this change cannot load it: it reports
`Unrecognized keyword arguments passed to TopoLM: {'grid_dim': 2}` (measured by the
adversarial review on a 2D archive). Loading goes one way only; a config without the key
still loads as 2D here. The vocabulary tables (`word_embeddings` and an untied
`lm_head`) stay fresh only when the checkpoint's own `vocab_size` differs from the model's. An
untied `lm_head` may otherwise be left out only when one side carries a task head: a tied
headless checkpoint does not fill an untied headless model, and an untied `lm_head` is not
dropped into a tied one. The model is built first: before that, every
`pretrained=<path>` raised "model must be built". To restore a whole model, head included, use
`keras.models.load_model`.

---

## 7. Post-hoc analysis

**The chain takes a 2D or a 3D map.** The grid-based steps take a `(height, width)` or a
`(depth, height, width)` map: `morans_i_grid`, `label_islands` (in both metric modules),
`islands_morans_i`, `morans_i_summary` and `grow_clusters` read the rank from the array
(`grid_weights_nd` builds the weights; queen is 26 neighbours in 3D and rook is 6), and
raise a `ValueError` naming the shape for any other rank. The 2D results are bit-identical
to what the code returned before 3D support. The per-unit steps (`contrast_tmap`,
`fdr_across_layers`) act on `(stimuli, units)` arrays and do not depend on the grid rank.
The Gaussian readout is the part that stays 2D (a `Conv2D`), and a 3D readout is not built.

The trainer scores a trained backbone through four steps, and each has a way to
go quietly wrong:

1. a two-sample t-map between two stimulus conditions, on activations
   **mean-pooled** over token positions;
2. BH-FDR corrected **across all taps**, then split back;
3. clusters grown from the surviving cells, both polarities;
4. Moran's I on the **unthresholded** map, plus the islands variant.

```text
activations        contrast_tmap       fdr_across_layers    grow_clusters
(stimuli x units)  -> (t per unit)     -> (q per unit)      -> islands
                     |                  |                    |
                     |                  +- ONE correction over
                     |                     every tap at once
                     +- independent across
                        stimuli, NOT across units
```

Two rules in that list are load-bearing and not interchangeable:

- **Correction is joint across taps.** A twelve-layer model taps 9,408 units;
  correcting each layer's 784 against 5% in isolation admits roughly 470 false
  positives per layer, and the paper's figures read as one organism with one map
  per layer rather than as twelve independent families.
- **Moran's I is scored on the raw map.** Thresholding produces contiguous
  patches of zero, and a patch of zeros is a patch of agreement, so the statistic
  jumps without any change in the underlying organisation.

A run also **draws** the result: one t-map per tapped layer on a shared diverging
scale with rejected cells hatched and clusters outlined (three orthogonal planes per
tap on a 3D grid), Moran's I against depth, and one categorical cluster map (a slice
montage on a 3D grid). That is `train.topolm.plotting`, called from
`run_topographic_evaluation` on the same arrays the report counts. Its rules are
in the trainer's README (`src/train/topolm/README.md` § *Figures*); the short
version is that a topographic figure has exactly two ways of lying by
construction — a per-panel autoscaled colour map, and a significance overlay on
the wrong half of the cells — and both are pinned by tests that measure the
artists rather than eyeballing a PNG.

Two things the metrics do *not* do, deliberately:

- **A simulator is not a result.** `GaussianReadout` exists so a model can be built
  to emit voxel-like signals, and the evaluation runs a raw arm and a readout arm
  side by side. The readout's default is edge-replicating (`padding_mode='nearest'`)
  rather than Keras' zero-padded `'same'`, which lets a constant field decay at
  the grid border.
- **The islands statistic is not a strict improvement.** On a small island it is
  high-variance and sometimes *negative* — a compact blob carries more same-sign
  diagonal pairs than the queen weighting expects (Anselin's queen-contiguity
  instability). It is reported alongside the standard value, never instead of it.

---

## 8. Component reference

| symbol | location | role |
|---|---|---|
| `TopoLM` | `model.py` | the model; `{"logits", "last_hidden_state"}`, or one task head's output with `include_top=True` |
| `TopoLMBlock` | `components.py` | decoder block carrying the taps |
| `create_topolm` | `model.py` | module-level factory |
| `MODEL_VARIANTS` | `config.py` | the variant table |
| `SpatialSmoothness` | `layers/regularization/spatial_smoothness.py` | the identity tap; `grid_dim` 2 or 3 |
| `SpatialLayout` | same | the unit-to-grid bijection on a 2D or 3D grid |
| `resolve_grid_shape`, `build_patch_tables` | same | the grid rule (`grid_dim` stated) and the neighbourhood tables (rank taken from `cell_to_unit.ndim`) |
| `GaussianReadout` | `layers/regularization/gaussian_readout.py` | fixed-width sensor simulation; 2D only |
| `morans_i_summary` | `metrics/spatial_autocorrelation.py` | standard + islands; 2D maps only |
| `contrast_tmap` / `fdr_across_layers` / `grow_clusters` | `metrics/topographic_selectivity.py` | the measurement chain; `grow_clusters` takes a 2D map |

### `SpatialSmoothness`

An identity forward pass. All the work happens in the training-only loss, so the
tap is invisible to the residual arithmetic and to validation. Place it on a
tensor whose last axis is the model's unit dimension and where spatial neighbours
*should* respond alike — on the residual-path output of an attention or
feed-forward branch, before normalization and before the add.

`grid_dim=3` lays the same units out as a box and cuts cubic neighbourhoods; the loss
function and `call` are the same code in both ranks. The rank is the stated
`grid_dim` keyword (after `eps`), validated to 2 or 3, forwarded to
`resolve_grid_shape` and `SpatialLayout`, and written by `get_config`; it is never
inferred from `len(grid_shape)`, which is what lets `GaussianReadout` (whose blur is
one `Conv2D`) refuse a three-entry `grid_shape`, including the grid of a 3D layout,
at build, naming the `(height, width)` pair it needs, instead of failing inside the
convolution. `build_patch_tables` takes no rank argument: it reads
`cell_to_unit.ndim`, so the rank and the array cannot disagree. Centres are
enumerated lexicographically with the first axis slowest, and a neighbourhood needs
every axis at least `2*radius + 1` across.

Its tables (`perm`, `patch_table`, `triu_flat`, `d_unit`) travel as
**non-trainable weights**, not as configuration. `get_config` carries the seed
that makes a fresh construction reproducible; it does not carry the permutation
itself. That asymmetry is only safe because the permutation is restored
bit-exactly, and a trainable `perm` would let the optimizer silently re-map every
unit while the checkpoint still looked loadable. The three index tables are
**int64**: TensorFlow keeps an int32 variable in host memory, where the
XLA-compiled step that `fit` uses on a GPU cannot read it. A checkpoint saved with
the earlier int32 tables still loads, with identical values.

Two numerical rules, both measured rather than assumed:

- the correlation block runs at `numpy.promote_types(input_dtype, "float32")` —
  a hard cast to `float32` *narrows* the reduction under a `float64` policy;
- `add_loss` is skipped for a symbolic `KerasTensor`, so the penalty is never
  computed from an unrealisable shape.

**The second rule is belt-and-braces, and this README used to overstate it.** An
earlier draft claimed `add_loss` accepts a `KerasTensor`, stores it, and that the
stored tensor never becomes a value — so a training loss would silently lack the
spatial term. That could not be reproduced. `add_loss` on a symbolic tensor here
stores *nothing at all*, so the term is absent either way; and this model's
`build()` materialises the tree without a forward pass, so no tap is ever reached
with a `KerasTensor` in the normal lifecycle (measured: 0 symbolic tap calls after
`build()`, 4 eager calls and 4 losses after the first `training=True` step).

The honest reason the guard stays: the skip is free, and the alternative is a
sampled reduction (7260 terms per neighbourhood at the paper's 2D settings, 265,356
at `paper3d`) whose result is discarded. A reader weighing whether
to delete the line should know it buys no observable behaviour today.

---

## 9. Reproducing the paper's numbers

```bash
MPLBACKEND=Agg python -m train.topolm.pretrain \
    --variant paper --alpha 2.5 --train-steps 100000
```

The paper trained on FineWeb-Edu and evaluated against an fMRI stimulus set.
**Neither is in this repository.** The trainer's data path is Wikipedia (the
shared `ClmPretrainConfig` default) and its evaluation falls back to a small
built-in smoke stimulus set, with every report stamped
`"stimuli_are_smoke_set": true` and a warning logged. Pass `--variant paper` plus
your own `stimuli=` config for a real analysis.

The paper is 2D, so there is nothing to reproduce in 3D: `paper3d` and `tiny3d` are
repo-authored layouts, the trainer trains them (raw arm only), and no number in the
paper applies to them.

---

## 10. Tests

```bash
pytest tests/test_models/test_topolm/ tests/test_train/test_topolm/ \
       tests/test_layers/test_regularization/ tests/test_metrics/test_spatial*.py \
       tests/test_metrics/test_topographic*.py tests/test_callbacks/test_topographic_callbacks.py
```

The 3D work lives in these files, named so a failure points at its layer:
`tests/test_layers/test_regularization/test_spatial_smoothness_3d.py` (factoring
against a brute-force oracle, the 3D tables, per-axis rejection, the sign probe),
`tests/test_layers/test_regularization/spatial_smoothness_oracle.py` with
`test_spatial_smoothness_oracle.py` (an independent nested-loop 3D reference and its
mutation self-tests), `tests/test_models/test_topolm/test_topolm_3d.py` (config
validation, block and model `grid_dim`, the alpha control, the other-rank and
other-grid checkpoint refusals, the wrapped-archive and direct-loader cases, the variant
rows) and `tests/test_train/test_topolm/test_topolm_trainer.py`
(`TestThreeDimensionalVariantsTrain`). The 2D layout is pinned byte for byte by
`TestTwoDimensionalIsByteIdentical` in `test_spatial_smoothness.py`, and the readout's
2D-only refusal by `TestGridRankRefusal` in `test_gaussian_readout.py`. The GPU arms use
the `requires_gpu` marker from `tests/index_weights.py`; one of them runs a 3D model
under XLA.

Do not run the full suite as a routine check — it takes about 1.5 hours and is
also the pre-push hook. Scope pytest to the modules you touched.