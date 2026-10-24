# `train/topolm`: pre-training a topographic language model

> Pipeline for [TopoLM](../../dl_techniques/models/language/topolm/README.md)
> (Rathi et al., ICLR 2025, [arXiv:2410.11516](https://arxiv.org/abs/2410.11516)).

```bash
MPLBACKEND=Agg python -m train.topolm.pretrain --variant tiny --train-steps 20
```

`MPLBACKEND=Agg` is required on a headless machine. The run draws training curves
at the end, and an unset backend reaches for X11 and takes the process down with
it.

---

## 1. Two things that would otherwise look like bugs

### The validation cadence *is* the virtual epoch length

Pretraining is roughly one pass over a text stream, and Keras only validates at
epoch ends. A run that validates once per pass cannot early-stop on the rule the
paper used, so `fit` is given `steps_per_epoch=eval_every_steps` and the
optimization-step count becomes `eval_every_steps * epochs`.

The consequence is stated rather than hidden: **that is not one pass over the
data.** Set `--train-steps` to pin the step count instead; the evaluation cadence
is then derived from it and is unchanged. Both numbers are logged before the
first step, and `resolve_cadence` reports how many steps one real pass would have
taken so the discrepancy is visible in the log rather than inferred later from a
short run.

```python
from train.topolm import TopoLMTrainingConfig, resolve_cadence

config = TopoLMTrainingConfig(eval_every_steps=100, train_steps=250)
resolve_cadence(config, real_steps_per_epoch=9_999)   # -> (3, 100)
```

250 steps at a 100-step cadence rounds *up* to three epochs, so no requested step
is dropped.

### The head must be told to aggregate the backbone's losses

`create_topolm_model` passes `aggregate_backbone_losses=True`, and it is
load-bearing. The taps reach the objective only through `backbone.losses`; without
the flag the head computes every penalty, discards it, and the run produces a
non-topographic model with a healthy curve.

Measured on identical weights at `alpha = 2.5`, same data, same seed, same taps:

| `aggregate_backbone_losses` | reported train loss | eval loss |
|---|---|---|
| `True` | **15.31** | 5.31 |
| `False` | **5.36** | 5.31 |

The evaluation loss is 5.31 either way, because the taps add nothing when
`training` is not `True`. That is the intended behaviour and it is what makes the
paper's reported 3.075 / 2.966 pair comparable.

---

## 2. What a run reports, and which number to read

`loss` alone cannot answer the question this trainer exists to ask. A rising curve
can be a rising task loss with a falling penalty, or the reverse, and the two look
identical in one number. Every run therefore also records, through
`SpatialLossLogger`:

| key | meaning |
|---|---|
| `spatial/<tap_name>` | that tap's penalty for the last batch |
| `spatial/total` | `sum(alpha_k * SL_k)` — the weighted total |
| `spatial/unweighted` | `sum(SL_k)` — with no `alpha` |
| `task_loss` | `loss - spatial/total` |
| `spatial/share` | `spatial/total / loss`, the fraction topography currently owns |
| `spatial/accounted` | `loss - compile_loss` — what the loop actually folded in |
| `spatial/unaccounted` | `spatial/total - accounted` |

**`spatial/unaccounted` is the one to watch.** It is non-zero exactly when the
backbone's losses are being computed and thrown away. Note that `task_loss` does
*not* reveal this: it equals `loss - spatial_total` by construction, whatever the
training loop did with the term.

---

## 3. Why not the shared callback factory

`train.common.create_nlp_callbacks` unconditionally installs
`keras.callbacks.EarlyStopping`. The rule the paper trained under is "**three
CONSECUTIVE INCREASES** on validation loss", which is a different question: the
stock callback counts from the best value seen, so a curve that dips to a new best
and then rises three times is stopped by the paper's rule and tolerated by the
stock one. Two stop signals on one run would also be two answers to one question.

So `build_callbacks` constructs its own list and takes `results_dir` from
`prepare_run_dir`.

```python
from train.topolm import TopoLMTrainingConfig, build_callbacks

callbacks = build_callbacks(config, head, results_dir, initial_step=0)
# [ConsecutiveIncreaseEarlyStopping, SpatialLossLogger,
#  StepCheckpointCallback, GenerationProbeCallback]
```

With `viz_every_steps > 0` the call also takes `backbone=` and `preprocessor=` and a
fifth callback, `TopographyLiveCallback`, is appended last (see *Watching a run*
in section 5). With the default `0` the list is the four above and nothing else
is added.

`ConsecutiveIncreaseEarlyStopping` restores the snapshot taken immediately before
the final streak — the last *good* evaluation, not the best one seen — which is
deliberately different from `EarlyStopping(restore_best_weights=True)`.

---

## 4. The optimizer recipe

The paper's numbers are the defaults, so the shipped configuration *is* the recipe
rather than an approximation of it:

```python
optimizer_builder({
    "type": "adamw",
    "beta_1": 0.9, "beta_2": 0.95,
    "weight_decay": 0.1,
    "gradient_clipping_by_norm": 1.0,          # GLOBAL norm
    "exclude_from_weight_decay": ["bias", "gamma", "beta", "embedding"],
}, schedule)
```

Two details that are not interchangeable:

- `gradient_clipping_by_norm` is the **global** norm. The builder also accepts
  `gradient_clipping_by_norm_local`, and the two keys look alike while clipping
  different quantities. "Gradient clipping at 1.0" means the global one.
- Weight decay is excluded for biases, the two normalization scales and the
  embedding table. Both exclusions matter: decaying a LayerNorm gain toward zero
  fights the very normalization the taps are trying to make spatially smooth, and
  the embedding table is **tied** to the output head, so decaying it decays both.

No `loss=` or `metrics=` are passed to `compile`. `CausalLanguageModel` owns its
loss and its `loss` / `accuracy` / `perplexity` trackers through a hand-rolled
`train_step`, so a compiled loss here would be inert config. `jit_compile` is not
passed either, so it stays at Keras's `"auto"`: XLA whenever a GPU is visible,
eager on a CPU-only host (`compile_model`'s docstring records the measurement:
`model.jit_compile` is `True` after the call on a GPU). That is why the taps'
index tables are int64 weights: an int32 weight is kept in host memory, where an
XLA step on a GPU cannot read it.

---

## 5. Running it

```bash
# The paper's shape.
MPLBACKEND=Agg python -m train.topolm.pretrain --variant paper --train-steps 100000

# The non-topographic control the paper compares against.
MPLBACKEND=Agg python -m train.topolm.pretrain --variant paper --alpha 0

# Both arms, identical seeds and data order, back to back.
MPLBACKEND=Agg python -m train.topolm.pretrain --variant tiny --paired

# The Fig. 12 ablation: one shared unit layout instead of one per tap.
MPLBACKEND=Agg python -m train.topolm.pretrain --no-permute

# A single-tap ablation, and the stopping rule relaxed.
MPLBACKEND=Agg python -m train.topolm.pretrain \
    --tap-sites attention --early-stop-patience 10

# Watch the topography form: a draw at step 0, every 50 optimizer steps and at the end.
MPLBACKEND=Agg python -m train.topolm.pretrain --variant tiny --train-steps 400 \
    --eval-every-steps 100 --viz-every-steps 50 --batch-size 8

# A 3D layout (343 units on a 7 x 7 x 7 box, radius 2), raw arm only.
MPLBACKEND=Agg python -m train.topolm.pretrain --variant tiny3d --train-steps 400 \
    --eval-every-steps 100 --viz-every-steps 50 --batch-size 8
```

`--variant` offers every row of `MODEL_VARIANTS`. The grid's rank comes from the
variant, so there is no grid flag: `paper3d` (1000 units on a 10 x 10 x 10 box,
radius 4) and `tiny3d` (343 units on 7 x 7 x 7, radius 2) train a 3D layout (see
*Training a 3D variant* below). `--radius` means a radius-`r` patch of
`(2r+1)**grid_dim` units. Left unset it resolves per variant: the row's own radius
for `tiny3d` (2) and `paper3d` (4), 5 (the paper's) for the 2D variants, and an
explicit value wins. `config.json` records the integer that was used.

Every flag maps onto exactly one field of `TopoLMTrainingConfig`, so `--help` and
`config.json` describe the same run — and
`tests/test_train/test_topolm/test_cli_contract.py` drives the real parser and the
real config builder to assert that every declared flag arrives, with set-equality
against the parser so a newly added flag fails until it is given a row.

### Two defaults that must match the tokenizer, not the paper

Both of these were wrong and both killed the run on its first batch, so they are
worth stating rather than leaving to `--help`:

| Flag | Value that works | Why the other value fails |
|---|---|---|
| `--vocab-size` | `100277` (default) | The trainer's tokenizer is Tiktoken `gpt2`: `encoding_name` is a field of `ClmPretrainConfig` (default `"gpt2"`) with no flag, and the run log says `encoding 'gpt2'`, `vocab_size=50257`. Its 50,257 base ids are followed by the four special ids the config assigns, 50257 to 50260 (`cls_token_id`, `sep_token_id`, `pad_token_id`, `mask_token_id`), so a table that holds all of them has at least 50,261 rows (derived: 50260 + 1). A 50,257-row table failed with `indices[..] = 50259 is not in [0, 50257)`, which is the pad id of that list. `100277` is larger than that and is the flag's default. |
| `--max-seq-length` | `512` (default) | The data pipeline chunks to this, so the position table must match. `build_backbone` forwards it to `max_seq_len`; before it did, a `tiny` run kept the variant's 256-entry table against 512-token windows and died with `indices[256] = 256 is not in [0, 256)`. |

An earlier version of this table said `create_tokenizer` uses `cl100k_base`, whose
special ids reach `100267`. That does not describe this trainer: `create_tokenizer`'s
own default is `cl100k_base`, but the trainer passes `config.encoding_name`. The
guard `_require_vocab_covers_tokenizer` compares `--vocab-size` with the larger of
the base `vocab_size` the log prints (50257) and the largest special id plus one
(50261), so a 50,257-row or 50,260-row table is refused at start-up by name, before
the dataset is built; `build_backbone` checks the same number. Both failures above
were measured on the pre-audit pipeline; the `--vocab-size 50257` case was not
re-run on the packed dataset path the trainer uses now.

Both are measured, not inferred: each was reproduced on a clean worktree at the
pre-audit commit before being fixed here. `build_backbone` now also raises, rather
than deferring the mismatch to the first forward.

### Run directory naming

```text
<save_dir>/topolm_<variant>_<topo|control>_alpha<a>_<timestamp>/
```

`alpha` is in the name because the control arm writes into the same tree, and the
two must not be confusable when they are read back months apart. Contents:
`config.json`, `training_history.json`, `training_curves.png`,
`topography_report.json`, `checkpoints/`, `generation_probes/`, plus the figures
below when topography ran, and `topography_live/` when `--viz-every-steps` is
positive.

### Figures

New to these pictures? [Reading the figures: a plain-language guide](#reading-the-figures-a-plain-language-guide)
explains every term and every mark, with an example of each figure.

| File | Contents |
|---|---|
| `t_maps_raw.png` | one t-map per tapped layer, shared diverging scale, rejected cells hatched, both conditions' clusters outlined (solid black for A, dashed magenta for B); on a 3D grid, three orthogonal planes per tap |
| `t_maps_readout.png` | the same, after the simulated fMRI Gaussian readout (2D variants only) |
| `morans_i_raw.png`, `morans_i_readout.png` | standard Moran's I against depth |
| `clusters_raw.png`, `clusters_readout.png` | one categorical cluster map per arm, from the deepest tap that found a cluster, drawn from condition A's clusters only and titled with the tap, the arm and the condition (`block_3/mlp_tap (raw), condition a: 1 cluster`; the t-map of the same tap outlines both conditions); on a 3D grid, every slice along axis 0 |

Three rules the figures follow, each of which exists because the obvious
alternative misleads:

- **One shared colour scale per row, diverging at zero.** Per-panel autoscaling
  renders noise and signal at identical contrast, which is the cheapest way to
  manufacture a topographic result out of an untrained model. The limit is a
  pooled 99th percentile, not the maximum, so one diverging cell cannot flatten
  the map.
- **Rejected cells are hatched, and the hatch is measured.** `contourf` fills the
  *unmasked* cells, so writing `masked_where(~sig, ones)` hatches the
  **survivors**. Pinned by
  `test_the_hatch_covers_rejected_cells_and_spares_surviving_ones`, which reads
  the hatched region's extent off the axes rather than off the image.
- **No contour lines around surviving cells.** At realistic BH-FDR densities
  that is a dense mesh over the whole panel that buries the t-values it annotates.
  Cluster edges come from the label grid instead, so a cluster outlined in a
  figure is the cluster counted in the JSON — there is one source for both.
- **The figure says what its marks mean.** The line under the headline reads, for
  example, `positive = a, negative = b; hatched = not significant after BH-FDR;
  outline = cluster of significant units (solid black = a, dashed magenta = b)`.
  Both conditions' clusters are outlined, because a map shows both signs and a
  large blue region without an outline would read as "not a cluster". Only the
  marks actually drawn are described, and if a caller hands over condition A's
  labels alone the line says that B's clusters are not drawn.

The figures are drawn by `train.topolm.plotting` from the arrays
`evaluate_topography` already computed, which it returns alongside the report.
Matplotlib is imported lazily inside `plot_topography`, so importing `common` for
its analysis does not pull in a plotting stack.

> `results/` at the repo root is **gitignored and untracked**. There is no history
> and no backup. Do not clean it up by name; route a test's config through
> `tmp_path`.

### Reading the figures: a plain-language guide

This part is for a reader who has never seen a t-map, a Moran's I or a topographic
model. Each term is explained where it first appears, and the glossary at the end
repeats them one line each. The pictures come from two small runs made to show what
the figures look like: 400 optimizer steps, seed 42, the built-in smoke stimuli.
**They are toy runs. They show that the pipeline works and what its output looks
like; they are not a result.**

#### The idea in one paragraph

Inside a language model, each layer turns every word position into a list of numbers
(256 of them in the `tiny` variant). Each of those numbers is a **unit**. In an
ordinary model the order of the units means nothing: unit 17 and unit 18 are
unrelated. TopoLM gives every unit a cell on a grid (a 16 by 16 square for 256 units,
a 7 by 7 by 7 cube for the 343 units of `tiny3d`) and adds a **penalty** to the
training loss: units that are neighbours on the grid should respond alike, more than
units that are far apart (the formula is in
[section 2 of the model README](../../dl_techniques/models/language/topolm/README.md)).
If that works, the grid becomes a **topographic map**: nearby units respond to
similar things, like the functional patches of the brain's cortex that the paper
sets out to reproduce. The
penalty is applied to the output of the attention part and of the feed-forward part of
every block; each of those places is called a **tap**. `tiny` has 4 blocks and 2 taps
per block, so 8 taps, each with its own grid of 256 units.

Which cell a unit gets comes from a random permutation that is drawn once per tap at
the start and never changed. So at step 0 the units that sit side by side are
unrelated, and any map drawn then is noise. Training has to make units that happen to
sit side by side behave alike. That is why the first picture below is noise.

#### The experiment behind every map

The maps are not computed from the training text. The trainer runs a separate small
test: it feeds the model a fixed set of hand-written sentences in two **conditions**
and compares how each unit reacts to them. The built-in set (`SMOKE_STIMULI` in
`common.py`) has 5 sentences per condition:

| Condition | What it is | Two examples |
|---|---|---|
| `a` | well-formed English sentences | "the quick brown fox jumps over the lazy dog"; "she sells sea shells by the seashore" |
| `b` | unconnected real words, no grammar | "bright table yellow winter quiet glass"; "silver morning pocket corner gentle" |

(The set also holds conditions `c` and `d`, jabberwocky sentences and unconnected
nonwords; the default test does not use them.) For every tap and every unit:

1. Run each sentence through the model and record the unit's activation. The
   unit's **response** to a sentence is that activation averaged over all positions
   of the sentence's input window (the tokenizer pads each sentence to the window
   length and the code does not leave the padding out of the average; see
   `extract_tap_activations`).
2. The unit now has 5 responses to condition `a` and 5 to condition `b`. Compute a
   two-sample **t value** from them: the difference of the two group means (a minus
   b) divided by the standard error of that difference. In words, how many standard
   errors apart the two groups are. A positive t means the unit responds more to the
   sentences (`a`); a negative t means it responds more to the word lists (`b`);
   near 0 means no difference.
3. Put each unit's t value on the grid, in the cell the permutation gave that unit.
   The grid of t values is the **t-map**.

What a t-map is **not**: it is not accuracy and it is not a measure of how well the
model writes. It says, for these 10 sentences, which units react differently to
sentences and to word lists. It says nothing about why.

#### Reading a t-map

<a href="figures/t_maps_2d_untrained.png"><img src="figures/t_maps_2d_untrained.png" width="900" alt="Eight 16 by 16 t-maps at step 0: salt-and-pepper noise, every cell hatched"></a>

*`--variant tiny`, 400 steps, seed 42, smoke stimuli, drawn at step 0 (untrained).
Mean Moran's I 0.005, 0 significant units in all 8 taps, 0 clusters. t values run from
-4.3 to 7.3 and the colour scale reaches about +/-4. Click for the full-size file
(2356 x 856).*

<a href="figures/t_maps_2d_trained.png"><img src="figures/t_maps_2d_trained.png" width="900" alt="Eight 16 by 16 t-maps at step 400: large red and blue regions, outlined clusters"></a>

*The same run at step 400. Mean Moran's I 0.763 (per tap 0.615 to 0.911), 142 to 206
significant units per tap (mean 187.9 of 256), 14 clusters for `a` and 14 for `b`
over the 8 taps. t values run from -17.3 to 19.1 and the colour scale reaches about
+/-14. Click for the full-size file (2365 x 875).*

Each small square is one tap, titled `block_k/attention_tap` or `block_k/mlp_tap`
(the prefix `causal_language_model/topolm/` is the same for all of them, so the
figure drops it and says so under the headline). Each cell is one unit at its grid
position. The colour is its t value: red means the unit prefers sentences, blue means
it prefers word lists, white means no difference. The colour bar on the right shows
the scale.

At step 0 the cells are salt and pepper: a red cell sits next to a blue one for no
reason, because neighbours are unrelated. At step 400 there are large patches of one
colour. That is the topography forming: neighbouring units now tend to share a t
value. That the patches exist follows from the penalty; whether a patch means
anything about language is a separate question the figure cannot answer.

**Do not compare colours between two figures.** Each figure sets its own colour
limit, the 99th percentile of the absolute t values pooled over all its panels, and
the colour bar says what it is. Step 0 is drawn on about +/-4 and step 400 on about
+/-14, so the same shade means a different t in the two pictures. Colours are
comparable between the panels of one figure. A cell above the limit is drawn in the
darkest shade (at step 0 a few cells reach t = 7.3 on a scale of about +/-4).

#### Hatching and outlines

**Hatched cells** (diagonal lines) are units that are *not significant*. With 256
units in each of 8 taps there are 2048 comparisons, and some would look different by
luck alone. So the trainer applies a **Benjamini-Hochberg false discovery rate
(BH-FDR)** correction, which is designed to keep the expected share of false alarms
among the cells it marks significant at or below 5% (`fdr_alpha` 0.05). The
correction is applied once, jointly over all units of all taps (the report says
`fdr_scope: joint across all taps`), so the threshold is a statement about the whole
model and not about each layer on its own. A cell without hatching passed it. At step
0 every cell is hatched: nothing is significant. At step 400 most cells passed, and
the hatched cells that remain sit mostly in the pale regions.

**Outlines** mark **clusters**. A cluster is a group of significant cells that
all have the same sign and touch each other (a diagonal touch counts: 8 neighbours in
2D, 26 in 3D), with at least 10 cells. The trainer finds them by starting from the
strongest unused significant cell (largest |t|) and adding touching significant
cells of the same sign until none is left; a group of fewer than 10 cells is
dropped. A **solid black** outline surrounds a cluster of condition `a` (red,
positive t); a **dashed magenta** outline surrounds a cluster of condition `b` (blue,
negative t).

A large coloured region without an outline is one of three things: its cells are
hatched (not significant), it is smaller than 10 cells, or its significant cells do
not touch. The outline count in the JSON is the same number as the outlines you see:
both come from one label grid.

#### The 3D figures

<a href="figures/t_maps_3d_trained.png"><img src="figures/t_maps_3d_trained.png" width="900" alt="Twenty-four slices through eight 7 by 7 by 7 t-maps, with a dashed green crosshair in each"></a>

*`--variant tiny3d` (343 units per tap on a 7 x 7 x 7 box), 400 steps, seed 42, smoke
stimuli, step 400. Mean Moran's I 0.623 (per tap 0.448 to 0.788), 44 to 102 significant
units per tap (mean 72.9 of 343), 7 clusters for `a` and 9 for `b` over the 8 taps,
t values from -12.3 to 11.3, colour scale about +/-8. The panel titles are small
(about 6 pt) at the width shown here; click for the full-size file (1522 x 1265).*

A cube cannot be shown on a flat page, so every tap is drawn as **three slices** of
the cube. A slice is the flat plane you get by fixing one of the three axes at one
index. The three panels of a tap are the planes through one chosen cell, one for each
axis.

- The **dashed green crosshair** marks that cell: the one with the largest significant
  |t| of the tap (the largest |t| overall when no cell is significant, and the line
  under the headline says which applies).
- A title such as `axis 0 = 3  @(3, 4, 9)` has two parts. `axis 0 = 3` says that this
  panel is the plane at index 3 along axis 0. `@(3, 4, 9)` is the crosshair cell, as
  (axis 0, axis 1, axis 2). It is the same in the three panels of one tap, and the
  three panels are the planes through that cell.
- The crosshair is a place to start looking, not a result. A cluster that does not
  pass through the crosshair cell is not in these three planes.

Hatching, outlines and the colour scale mean what they mean in 2D, applied to each
slice. The figure has no axis ticks, so the title is the only place that says which
slice you are looking at.

<a href="figures/clusters_3d.png"><img src="figures/clusters_3d.png" width="900" alt="Seven slices along axis 0 of one cluster of block_3/mlp_tap"></a>

*The cluster montage of the same `tiny3d` run: tap `block_3/mlp_tap`, condition `a`
only, one cluster of 33 units. Slices 0 to 2 are empty, slice 3 holds one cell, and
the cluster widens to slice 6. Full-size file 1475 x 628.*

The cluster montage shows **every slice along axis 0** of the clusters of one tap, so
a cluster is followed through the cube. It is one 3D object, which is why one colour
runs through several slices. Each cluster gets its own colour; this run has one.

#### The cluster figure

<img src="figures/clusters_2d.png" width="360" alt="One cluster of block_3/mlp_tap, condition a, drawn as a blue shape with a black outline">

*The 2D run at step 400: tap `block_3/mlp_tap`, condition `a`, one cluster of 68
units. Full size, 469 x 496.*

`clusters_raw.png` (2D: one picture; 3D: the montage above) shows **one tap, the
deepest one that has a condition-`a` cluster**, and **one condition: `a` only**, which the title
says (`block_3/mlp_tap (raw), condition a: 1 cluster`). Each cluster is one colour on
a white background. It is the same cluster as the solid black outline in the t-map
of that tap, redrawn alone. The t-map shows both conditions, so the same tap there
also has a dashed magenta cluster of 60 units that this figure does not draw.

#### Moran's I

<img src="figures/morans_i_by_depth_2d.png" width="620" alt="Moran's I per tap in depth order, one line between 0.6 and 0.9, a dashed line at 0">

*The 2D run, after training (`morans_i_raw.png`). Moran's I (standard) per tap in
depth order: 0.797, 0.615, 0.767, 0.730, 0.798, 0.788, 0.698, 0.911. Full size,
782 x 467.*

**Moran's I** is one number per tap that answers: *do neighbouring cells hold similar
values?* It compares each cell's t value with its neighbours' (queen contiguity, the
same 8 or 26 neighbours as for clusters).

- Near 0: no spatial pattern, as in a random map. (For a map with no structure the
  expected value is -1/(N-1), about -0.004 for 256 cells, not exactly 0.)
- Toward 1: a smooth map, neighbours alike.
- Negative: neighbours tend to differ, as on a checkerboard.

The **standard** version is computed on the whole unthresholded t-map of the tap (every
cell, significant or not), because a patch of non-significant cells would otherwise
count as agreement. This is the number in the pictures: one point per tap in the
profile above, and the mean over the 8 taps (`mean_morans_i`) in the curve below. The
profile is not a trend with depth; here the lowest tap is `block_0/mlp_tap` and the
highest `block_3/mlp_tap`.

The **islands** version is computed only inside each connected group (an "island") of
significant cells, using those cells' own values and the adjacency between them, and
the islands are then averaged with equal weight (islands of at least 3 cells; the
sign is not looked at; `nan` when there is none). It exists for the case where few
cells are significant and the standard value mostly describes the rest. It is in
`topography_report.json` (`morans_i.islands`) and `metrics.jsonl`
(`morans_i_islands`), and the figures do not plot it. **It can exceed 1**: in the 2D
run `block_0/attention_tap` has 1.034, so do not read it as a correlation. In the 3D
run it ranges from -0.019 to 0.249.

A third version, a **permutation p-value** (how often a random shuffle of the map
scores at least as high), is off by default (`--permutation-p-value`); the example
reports hold `nan` for it.

#### The live curve

<a href="figures/live_curve_2d.png"><img src="figures/live_curve_2d.png" width="700" alt="Four stacked panels against the optimizer step: Moran's I, clusters, significant units, validation loss"></a>

*`topography_curve.png` of the 2D run (`--variant tiny`, 400 steps, seed 42, a draw
every 50 steps, validation every 100). Full-size file 1360 x 1432.*

With `--viz-every-steps N` the trainer redraws this picture after every draw. Four
panels share the step axis (steps 0, 50, ..., 400 here):

- **Top, Moran's I.** One thin line per tap (each has its own colour, marker and dash
  pattern, named in the legend on the right) and the mean over taps in bold black. The
  dashed horizontal line at 0 is "no pattern". Rising means the map is getting
  smoother. Example: the mean goes from 0.005 at step 0 to 0.62 at step 50 and sits
  between 0.73 and 0.78 from step 100 on; the taps end between 0.615 and 0.911.
- **Second, clusters.** The number of clusters of condition `a` (solid) and of `b`
  (dashed), summed over all 8 taps. Example: 0 and 0 at step 0, 11 and 11 at step 50,
  14 and 14 at step 400 (the most was 18 clusters of `a` at step 250).
- **Third, significant units.** The mean over taps of how many units pass BH-FDR (out
  of 256). Example: 0 at step 0, 166.5 at step 50, 203 at step 150, 188 at step 400.
- **Bottom, validation loss.** The ordinary language-model loss on held-out text,
  without the penalty term. Validation runs once per virtual epoch (every 100 steps
  here), so there is a point at 100, 200, 300 and 400, each at the step where it was
  measured: 7.914, 7.813, 7.818, 7.819. That is a flat curve (a range of 0.1) and it is
  not a trained model: 400 steps only prove the pipeline.

What to expect: Moran's I jumps in the first 50 steps, then moves little, while the
clusters and significant units rise and settle. What would be a warning sign:

- **Moran's I stays near 0.** The penalty is not reaching the objective. Look at
  `spatial/unaccounted` ([section 2](#2-what-a-run-reports-and-which-number-to-read))
  and at the head's `aggregate_backbone_losses`
  ([section 1](#1-two-things-that-would-otherwise-look-like-bugs)).
- **No clusters and no significant units while Moran's I is high.** One possible
  reading is a smooth map whose units react too little to these stimuli to pass the
  test. The curve cannot tell you more than that.
- **The validation loss rises.** The task is getting worse while the map forms.

#### What these figures do and do not tell you

- **The smoothness is what the penalty asks for.** A rising Moran's I shows that the
  penalty is working. It does not show that the model learned anything about
  language.
- **Five sentences per condition is very little.** A live map is noisy, and its
  clusters will change from one draw to the next (the clusters panel above wobbles
  between 14 and 18 for `a` after step 150). A large |t| from 5 against 5 values is a
  weak statement.
- **The runs are toys.** `tiny` has 256 units per tap and 4 blocks, `tiny3d` 343 units;
  both ran 400 steps on a GPU, which is not bit-reproducible, so a re-run gives
  similar and not identical pictures. The numbers in the captions are from one run
  each.
- **No claim is made that 3D is better or worse than 2D.** They were never compared:
  the widths (256 against 343) and the neighbourhood radii differ, so their numbers
  are not comparable. The 3D layout is an extension of the paper, which is 2D only.
- **Earlier 2D numbers need care.** Moran's I and cluster numbers made under the
  default permutation before a placement fix were computed on a scrambled map; see
  [A correction to earlier 2D numbers](#a-correction-to-earlier-2d-numbers).

#### Glossary

| Term | In one line |
|---|---|
| unit | one number in a layer's output vector; one cell of the grid |
| grid, layout | the 2D square or 3D cube of cells; a fixed random permutation decides which unit sits in which cell |
| tap | one place the penalty and the figures look at: the output of the attention part or the feed-forward part of a block (`block_k/attention_tap`, `block_k/mlp_tap`) |
| condition `a`, `b` | the two kinds of test sentence: `a` well-formed sentences, `b` unconnected words; 5 of each |
| response | a unit's activation for one sentence, averaged over the positions of the input window |
| t value, t-map | per unit, (mean response to `a` minus mean response to `b`) divided by its standard error; the t-map is those values on the grid; red positive, blue negative |
| significant, FDR | a unit that passes the BH-FDR correction (5%), applied jointly over all units of all taps |
| hatch | diagonal lines over the units that are not significant |
| cluster | at least 10 touching significant cells of one sign; solid black outline for `a`, dashed magenta for `b` |
| Moran's I (standard) | one number per tap: do neighbouring t values match? about 0 none, toward 1 smooth, negative checkerboard |
| Moran's I (islands) | the same inside each island of significant cells, averaged; can exceed 1 |
| crosshair | the dashed green lines of a 3D figure, through the cell of largest significant \|t\| |
| slice | one flat plane of the 3D cube, with one axis fixed at one index |
| step | one optimizer step; the figures are drawn at step 0 and every N steps |
| draw | one evaluation of the t-map, clusters and Moran's I at a step, with its figures |

Where the pictures come from (paths relative to a run directory; both runs are the
commands in section 5 with `--variant tiny` and `--variant tiny3d`, seed 42):
`t_maps_2d_untrained.png` is `topography_live/step_0000000/t_maps_raw.png`;
`t_maps_2d_trained.png` and `t_maps_3d_trained.png` are
`topography_live/step_0000400/t_maps_raw.png`; `live_curve_2d.png` is
`topography_live/topography_curve.png`; `clusters_2d.png`, `clusters_3d.png` and
`morans_i_by_depth_2d.png` are the run's own `clusters_raw.png` and `morans_i_raw.png`.
The files are unchanged copies, kept in `figures/` next to this README.

### Watching a run: `--viz-every-steps`

```bash
MPLBACKEND=Agg python -m train.topolm.pretrain --variant tiny --train-steps 400 \
    --eval-every-steps 100 --viz-every-steps 50 --batch-size 8
```

`--viz-every-steps N` (default `0`, off) makes the run score its own topography
while it trains (what the pictures mean is in the
[plain-language guide](#reading-the-figures-a-plain-language-guide) above). A draw is taken at step 0 (the untrained map), at every optimizer
step that is a multiple of `N`, and at the end of the run. **`N` counts optimizer
steps and is independent of the virtual epoch**: `--eval-every-steps` sets the
length of the epoch Keras validates at, `N` is a count of steps inside the run
(a resumed run continues from its checkpoint's step), so `--viz-every-steps 500`
in a run that validates every 2000 steps draws four times per virtual epoch.
`0` adds no callback and no directory. A negative value is refused by name.
`--no-topography` does not switch it off: that flag skips the post-hoc report
only, so passing both gives live figures and no final report. With `--paired` each
arm draws into its own run directory.

```text
<run_dir>/topography_live/
    step_0000000/               one figure set per draw (t_maps_raw.png,
    step_0000050/ ...           morans_i_raw.png, clusters_raw.png once a cluster exists)
    latest_t_maps_raw.png       the newest t-maps
    latest_morans_i_raw.png     the newest Moran's I profile
    metrics.jsonl               one JSON row per draw, and a closing metrics-only row
                                when a validation ran after the last draw
    topography_curve.png        Moran's I per tap (one colour, marker and dash
                                pattern each, named in a legend right of the axes),
                                clusters, significant units and validation loss
                                against step
```

**How to watch it live.** Open `latest_t_maps_raw.png` in an image viewer that
reloads a file when it changes; its subtitle names the step and the stimulus set.
Follow the numbers with `tail -f <run_dir>/topography_live/metrics.jsonl`, and open
`topography_curve.png` for the trend. The `latest_*` files and the curve are
written to a temporary name and moved into place, so a viewer that reloads in the
middle of a draw never reads a half-written PNG.

**`metrics.jsonl`.** One strict-JSON row per draw (a value that is not finite is
`null`), flushed to disk after each row. Fields: `step`, `seconds` (wall time of
the draw), `val_loss`, `val_loss_step`, `grid_shape`, `contrast`,
`stimuli_are_smoke_set`, `mean_morans_i`, `mean_num_significant_units`,
`total_clusters_a`, `total_clusters_b`, and `per_tap`: for each tap `morans_i`,
`morans_i_islands`, `num_significant_units`, `clusters_a`, `clusters_b`, `t_min`,
`t_max` and `t_abs_mean`. `val_loss` is the last validation completed when the row
was written and `val_loss_step` is the step it was measured at. A draw at a step
where a validation is also due is taken at the end of that batch, before Keras
validates, so **the last full row lags the final value in `training_history.json`**:
in a 400-step `tiny3d` run (validation every 100 steps, a draw every 50) the step-400
row held 7.5333 and the final history value was 7.4917; in the matching `tiny` run
7.8182 and 7.8185. The curve plots each validation once, at its own `val_loss_step`.

**The closing validation row.** So that the curve ends on the run's last validation,
`on_train_end` appends one metrics-only row when a validation newer than the one the
last row carries ran after it: `{"step": 400, "val_only": true, "val_loss": ...,
"val_loss_step": 400}`, with the `step` of the last row, the newer value and nothing
else (no `per_tap`, no map statistics). It redraws the curve only: no evaluation,
no second set of map figures, the `latest_*` t-maps untouched. A run whose last
validation is already in the last row (a draw made after it, an off-grid final step,
a restore redraw) gets no such row, and a switched-off callback writes none. A
reader merges a repeated step with `last_row_per_step`: a row with map statistics
replaces the row before it, a metrics-only row is merged onto it (its non-null fields
override, a null or a missing field leaves the earlier value alone), and the curve
reads every validation from every row, so a merge never drops an older one.

**The last frame is the model the run ends with.** `fit` draws the last step at
the end of its final batch; `ConsecutiveIncreaseEarlyStopping` restores its
snapshot only afterwards. When the weights differ from the ones the last draw saw
(a fingerprint of the trainable weights decides), the end of training draws that
step again: the step directory and the `latest_*` files are replaced and
`metrics.jsonl` gets a second row with the same `step` and
`"restored_weights": true`. A run without a restore has strictly increasing steps
and no such field (apart from the closing row above, which repeats the last step). A
reader takes one entry per step with `last_row_per_step`, as the curve does. A stop
that falls between two draws gets one final frame labelled with the step it stopped
at, already showing the restored weights. A `metrics.jsonl` with a damaged line (a crash mid-write) does not stop the view:
the line is skipped with one warning and the next row starts on a line of its own.

**What a draw scores.** The two contrast conditions of the built-in smoke stimuli
(five sentences per condition; the command line has no flag for other stimuli),
the raw arm only, no permutation test, in batches of 8 prompts. A live map is
noisy, and the subtitle says so (`smoke stimuli, n=5/5 per condition`): it shows
that a structure is forming, not a result.

**The colour scale is per frame, on purpose.** Each figure uses the pooled 99th
percentile of its own t-values (see the rules above), and the colour bar states
that scale. A 2D `tiny` run was on +/-4 at step 0 and +/-14 at step 400. Two
frames are therefore not comparable by colour across steps; a fixed scale would
either clip the trained map or hide the early structure.

**The frame size is not constant.** The figure is laid out in inches and the line
under the headline wraps at clause boundaries (after a `;` or a `|`), so its height,
and with it the PNG's pixel size, changes by a few pixels between draws when the
subtitle gains or loses a wrapped line (a real 3D run went from 1522x1246 at step 0 to
1522x1265 and 1540x1265 later). A viewer that reloads the file may resize its window
or rescale the picture; padding every frame to one size was not done, because it
would change every figure. The 3D subtitle also says where the crosshair is and is
true at every step: `crosshair at the largest significant |t|`, with `(the largest
|t| where none is significant)` added when only some taps have a significant cell,
and `at the largest |t| (nothing is significant)` at step 0.

**What the periodic pass guarantees.**

- *Eval mode only.* A draw never calls the model with `training=True`. A
  training-mode forward overwrites each tap's non-trainable `last_spatial_loss`
  and advances every `SeedGenerator` (measured on CPU).
- *A tripwire.* Around each draw the callback compares every `SeedGenerator`
  state and the Python and NumPy random states. On any difference it restores
  them, logs an ERROR and switches the live view off; the run continues. At the
  start it warns if it finds fewer generators than taps, because the tripwire
  would then be blind.
- *Failure isolation.* An exception in a draw is logged with its location and
  training goes on. Three consecutive failed draws switch the live view off; a
  good draw resets the count. `KeyboardInterrupt` and `SystemExit` are not
  caught. The post-hoc evaluation still raises on a real failure, so a run cannot
  exit 0 without its topography report.

**Warnings after the last line.** A long run with the live view can print five TensorFlow
warnings, `Error occurred when finalizing GeneratorDataset iterator: FAILED_PRECONDITION:
Python interpreter state is not initialized`, after the final `Wrote ...` line. The exit code
is 0 and every output is written: the data generator's iterators are finalized after the
interpreter has begun to shut down. It is intermittent: on `tiny3d` runs of 400 steps with
`--viz-every-steps 50` it appeared in 9 of 11 runs of the current code and 1 of 3 runs of
an earlier callback, and in none of 6 runs without the live view (nor in 60-step runs with
it). It was traced as far as exceptions retained inside `pyparsing` (matplotlib's math-text
parser) keeping the `fit` frame, and with it the data iterators, alive to the end of the
process; it was not traced to the callback's own code, and nothing was changed for it.

**Does it change the run?** On CPU, no: a same-seed training with the callback
gives bit-equal trainable and non-trainable weights, `SeedGenerator` states and
per-epoch losses (`TestTheABInstrument`; its control run reproduces itself, and a
callback doing a `training=True` forward is seen to fail it). On GPU the
comparison cannot be bit-exact, because two identical runs without the callback
differ from each other. Measured on GPU 0 (`tiny`, seed 42, 200 steps,
`--eval-every-steps 50 --batch-size 8`, four runs without and two with
`--viz-every-steps 25`), the largest absolute difference over the trainable weights
of the final checkpoints, as min / median / max:

| Pairs | n | max abs difference |
|---|---|---|
| without vs without | 6 | 1.145e-2 / 1.331e-2 / 1.466e-2 |
| with vs without | 8 | 1.071e-2 / 1.304e-2 / 1.581e-2 |

The ranges overlap and most of the difference sits in the embedding table in both
rows. The honest statement is: **bit-exact on CPU; on GPU no effect detectable
above the device's own run-to-run noise (n=6 pairs without, n=8 with); the GPU is
not deterministic.** A single-pair criterion set before the runs (with-callback
difference no larger than the self-difference) was not met: 1.549e-2 against
1.291e-2, which one pair cannot resolve at this noise level. Every
`SeedGenerator` state and every non-trainable variable other than
`last_spatial_loss` was bit-identical in all 15 pairs, and the tripwire never
fired. One observation is unexplained: the epoch-3 training loss of both runs with
the callback (8.93612 and 8.93605) is below all four runs without it (8.93624 to
8.93645), by 1.2e-4 to 1.9e-4; no mechanism was found.

**What it costs** (measured 2026-10-24 on an RTX 4090 that another process was
sharing; about 11,065 MiB of the 24,564 MiB were already in use before each job,
and the timings are indicative, although the four runs without the callback agree
in wall time within 2.5%):

- *Per draw*, the `seconds` field over 9 draws each: `tiny` median 1.46 s (0.87 to
  1.97), `tiny3d` median 1.93 s (1.28 to 2.37), and a `paper3d`-width run
  (`--num-layers 2 --max-seq-length 128 --batch-size 4`) median 1.75 s (0.76 to
  2.16). The step-0 draw is the cheapest: nothing is learned yet and no cluster
  outlines are drawn.
- *Against a run without it*: `tiny`, 200 steps, a draw every 25 steps (9 draws).
  The fit phase took a median of 100.15 s over four runs without the callback and
  116.27 s and 117.07 s with it, i.e. 16.1 s and 16.9 s more (+16.1% and +16.9%).
  A draw every 25 steps is much denser than a real run needs; the overhead at a
  sparser cadence was not measured. Peak memory held by the process rose from
  5,028 to 6,052 MiB (the allocator's reserved memory under
  `TF_FORCE_GPU_ALLOW_GROWTH`, an upper bound on use).

### Training a 3D variant

`--variant tiny3d` (343 units, 7 x 7 x 7, radius 2) and `--variant paper3d` (1000
units, 10 x 10 x 10, 8 heads, radius 4) are repo-authored layouts, not the paper's,
and no number in the paper applies to them. The trainer trains, scores and draws
them like a 2D variant, with these differences.

- *Radius.* `--radius` unset resolves to the variant's own radius, so
  `--variant tiny3d` needs no radius flag. An explicit radius the grid cannot hold
  is refused before the run starts: no run directory, tokenizer or dataset is
  touched, and on the command line the GPU is not configured either. It is a parser
  error (the message on stderr under the usage line, exit code 2, no traceback; a head
  count that does not divide the width is refused the same way), and a programmatic
  call of `train_topolm` raises the same text as a `ValueError`. The message names the
  radius, the variant and its unit grid, the unit count a radius-r neighbourhood needs,
  and the largest radius the grid can use; `--variant tiny3d --radius 5` reports the
  7x7x7 grid, the 1331 units needed against 343, and a largest usable radius of 3.
  When the unit count suffices but an axis is short, it names that axis and its length.
- *Readout.* The Gaussian readout is a `Conv2D`, so the readout arm is 2D only.
  `--readout-fwhm` unset means 2.0 on a 2D variant and no readout on a 3D one, `0`
  means no readout on either, and a positive value typed for a 3D variant is
  refused by flag and variant before any work is done. A 3D run therefore scores
  the raw arm only. Its `topography_report.json` has a three-entry `grid_shape`,
  the single arm `raw`, and a top-level `readout_skipped` string giving the reason;
  a 2D report has no such key. (A programmatic config left at the dataclass default
  `readout_fwhm=2.0` on a 3D backbone also runs: the arm is skipped with one INFO
  line.)
- *Analysis.* The t-map, BH-FDR, cluster growth and both Moran statistics follow
  the grid's rank: queen contiguity is 26 neighbours in 3D and rook is 6.
- *Figures.* `t_maps_raw.png` draws three orthogonal planes per tap, two taps per
  row, through a crosshair at the cell of the largest `|t|` among the significant
  cells (the largest `|t|` overall if none is significant, ties to the lowest flat
  index). The subtitle states which of the two applies. The crosshair is a proposal
  for where to look, not a result: a cluster away from it is not in the planes.
  `clusters_raw.png` is a montage of every slice
  along axis 0 of the clusters of the deepest tap that found one, one colour per
  cluster. There are no `*_readout.png` files. The colour scale, hatch and outlines
  are the 2D ones, applied to each plane.
- *Measured, `tiny3d`* (`--train-steps 400 --eval-every-steps 100
  --viz-every-steps 50 --batch-size 8`, smoke stimuli): the t-maps are finite and
  not all zero, and the mean Moran's I over the 8 taps went from -0.0014 at step 0
  to 0.555 at step 50 and 0.625 at step 400, while the significant units per tap
  fell from 191 at step 50 to 70 at step 400.
- *Memory at paper width.* `--variant paper3d --num-layers 2 --train-steps 200
  --eval-every-steps 50 --viz-every-steps 25 --batch-size 4`: the first attempt, at
  the default `--max-seq-length 512`, held 8,608 MiB in its own process when it was
  stopped by an 8,000 MiB cap set for that measurement on a shared card, so its
  peak is not known. With `--max-seq-length 128` it finished, peaking at 6,558 MiB.
  That run shows the pipeline and the figures work at the paper's width; it is not
  what a real `paper3d` run looks like (2 layers, 200 steps).

### A correction to earlier 2D numbers

Before this change `evaluate_topography` put each tap's per-unit t and
significance vectors on the grid with a plain `reshape`. Under the default
`permute=True` unit `i` does not sit at cell `i`: the tap's layout places it at
`layout.positions`. **Earlier 2D Moran's I and cluster numbers produced under the
default `permute=True` were computed on a map whose cells were not at their layout
positions; they should be recomputed.** Runs with `permute=False`
(`--no-permute`) were unaffected, because the two placements agree there. The
t-map and cluster figures from `permute=True` runs were drawn from the same
scrambled map. Per-unit quantities, such as the t value of a unit and the number
of significant units, do not depend on where a unit is placed.

Both arms and the 3D path now place the vectors through the tap's
`layout.to_grid` (the readout also returns unit order). On a field that is smooth
on a 28 x 28 layout, Moran's I came out at 0.7927 through `layout.to_grid` and
0.0005 through `reshape`, with 5 clusters against 0 (measured before the fix, on
synthetic data). Pinned by `TestTheMapPlacementFollowsTheLayout`.

A second, separate figure defect: the cluster outline was drawn displaced from the
cells it enclosed (for the cell at row 3, column 4, the segments sat at x 5 to 6
and 4 to 5, while the cell spans x 3.5 to 4.5). It is drawn on the cell edges now,
for 2D and 3D; the cluster counts in the JSON never depended on it. Pinned by
`TestClusterBoundariesSitOnTheirCells`.

---

## 6. `--paired`

The control is the paper's comparison and the point of running it, so the two arms
share a seed, a data ordering and every other hyperparameter except `alpha`. The
control's taps are still created and still built, which is what makes that true at
the **weight level** rather than only at the config level — pinned in
`test_the_control_arm_is_only_alpha_that_differs` and
`test_an_alpha_zero_control_holds_an_identical_weight_set`.

`train_paired` reports the delta between the two arms' best validation losses and
names the paper's own figure for comparison:

```text
Task cost of topography: control is +0.1090 nats versus the topographic model.
The paper reports +0.109 for its own pair.
```

`--no-topography` reaches **both** arms. The flag is a CLI argument arriving at
`train_paired`, and it was originally dropped on this path, so `--paired
--no-topography` still wrote a report and four figures for the topographic arm
while the user had asked for no topography. A one-armed evaluation is worse than
none: it reads as a result and invites comparison against a control that has
nothing. Pinned by `test_no_topography_reaches_both_arms` and
`test_topography_still_runs_on_both_arms_when_asked`.

---

## 7. Reading activations out of the model

`extract_tap_activations` is not `keras.Model(inputs=..., outputs=layer.output)`:
the backbone is subclassed, so there is no `layer.output`, and wrapping internal
tensors functionally would mean rebuilding the stack under a second model.

Instead `TopoLM.call` accepts a `taps` argument — a sequence of identity layers,
one per entry of `model.tap_layers` — and each capture returns its input
unchanged. The captures are therefore invisible to the arithmetic, which is
asserted directly:

```python
plain   = backbone(tokens, training=False)["last_hidden_state"]
observed = backbone(tokens, training=False, taps=tuple(TapCapture() for _ in
                                                      backbone.tap_layers))
np.testing.assert_array_equal(plain, observed)
```

Pooling is a **mean** over token positions, not a sum, so a longer encoding does
not read as a stronger response — a confound the paper's own footnote 9 reports
for exactly this analysis.

---

## 8. What is *not* here

The paper trained on **FineWeb-Edu** and evaluated against an **fMRI stimulus
set**. Neither is in this repository.

- The data path is Wikipedia, the shared `ClmPretrainConfig` default.
- The evaluation falls back to a small built-in `SMOKE_STIMULI` set. Every report
  it produces is stamped `"stimuli_are_smoke_set": true` and a warning is logged:

  ```text
  These topography numbers came from the BUILT-IN smoke stimuli, not a published
  stimulus set. They demonstrate that the pipeline runs end to end; they are not
  a result.
  ```

  Pass `stimuli=` in the config for a real analysis. The paper's four conditions
  are named in `PAPER_CONDITIONS`; `--contrast c d` selects which pair the t-map
  contrasts.

The encoding probe, RSA and bootstrap CIs are deferred to a later pass.

**Dimension.** The trainer evaluates, plots and live-draws a 2D or a 3D grid; the
rank comes from the variant's row (`grid_dim`), and there is no grid flag. The grid
metrics (`morans_i_grid`, `label_islands`, `islands_morans_i`, `morans_i_summary`
and `grow_clusters`) take a `(height, width)` or a `(depth, height, width)` map and
refuse any other rank by name. Deferred, not built: a 3D Gaussian readout.
`GaussianReadout` is a `Conv2D` and refuses a three-entry grid at build, so a 3D
run scores its raw arm only (see *Training a 3D variant* in section 5).

Moran's I **permutation test** is now implemented and reaches the report via
`--permutation-p-value` (off by default). It is the paper's third Moran
statistic; the standard and islands values are reported either way. The p-value
is reproducible from the run seed and independent per tap, so re-running one arm
regenerates exactly the same numbers.

> On `run_paired`: there is no function by that name in this package. The paired
> run is `train_paired`, reached with `--paired`, and it **is** implemented — see
> §6. If a note elsewhere defers `run_paired`, it means this and should be read as
> a stale reference to the same work, not as a gap.

---

## 9. Tests

```bash
pytest tests/test_train/test_topolm/
```

The training-shaped tests drive the real `train_topolm` with the dataset loader
and tokenizer replaced, because what is worth testing here is the cadence
arithmetic, the objective wiring and the evaluation — none of which need
Wikipedia. The substitution is explicit in `_patch_data`, not hidden in a fixture,
and the run directory is redirected under `tmp_path` by an autouse fixture.

The plotting tests (`test_topolm_plotting.py`) assert on **artists**, not on saved
images. `contourf` is a single artist whose hatch lives on the set rather than on a
per-level collection, so a test that reads `ax.collections[0].get_hatch()` sees
`None` and passes vacuously; `_hatched` reads the set instead. The
significance-direction assertion compares the hatched region's **extent** against
the mask. Reading the pixels would work too, but it would not say *which* half of
the panel was wrong.

One note on the stub tokenizer: it takes a **list of strings**, which is what
`TiktokenPreprocessor.__call__` accepts (`str` or `list`, and it raises `TypeError`
on anything else). It used to take a `{"text": [...]}` mapping — a shape that
matched neither the real preprocessor nor anything else — and that agreement
between two unlike stubs is what hid the live `TypeError` in
`extract_tap_activations` for the whole first pass of this code.

The 3D work and the live view live in these files (all under
`tests/test_train/test_topolm/` unless a path is given):

- `test_topolm_3d_evaluation.py` (new): `TestThreeDimensionalPlacement`,
  `TestThreeDimensionalReadout`, `TestAThreeDimensionalReportAlwaysSaysItsReadoutIsAbsent`,
  `TestThreeDimensionalFigures` and `TestTheTrainerTrainsA3DVariant`, a short CPU
  `tiny3d` run through `train_topolm`.
- `test_topolm_live.py` (new): `TestTheABInstrument` (the same-seed CPU A/B, with a
  control that must reproduce itself and a `training=True` negative control),
  `TestTheTripwire`, `TestEvalModeOnly`, `TestFailureIsolation`, `TestOutputs`,
  `TestLatestFilesAreAtomic`, `TestTheMetricsRows`, `TestBothRanks`,
  `TestTheFlagIsRefusedNegativeAndNeedsItsInputs`, `TestTheFinalFrameShowsTheEndingWeights`,
  `TestADamagedMetricsFileDoesNotKillTheView`, `TestTheLastValidationIsDrawn` (the closing
  metrics-only row) and `TestARepeatedStepIsMerged` (`last_row_per_step`).
- `test_topolm_trainer.py`: `TestThreeDimensionalVariantsTrain` (it replaces
  the class that pinned an earlier refusal of 3D variants, which this trainer no
  longer makes), `TestRadiusResolution`, `TestVizEveryStepsField`,
  `TestTheMapPlacementFollowsTheLayout` (the placement correction above, against an
  oracle that does not use `to_grid`), the golden pins that hold the 2D report, config
  and `permute=False` placement as they were
  (`TestTwoDimensionalUnchangedReport`, `TestTwoDimensionalUnchangedConfig`,
  `TestTwoDimensionalUnchangedPlacementWithoutPermutation`), and
  `TestTheGridIsReportedAtItsOwnRank`.
- `test_cli_contract.py`: `TestRadiusPerVariant` and `TestVizEveryStepsFlag`.
- `test_topolm_trainer.py` also holds `TestTheVariantIsCheckedAgainstTheRadiusFirst` and
  `TestTheRadiusRefusalIsAParserError` (`main([...])` exits 2 with the refusal's text,
  before any run work).
- `test_topolm_plotting.py`: `TestClusterBoundariesSitOnTheirCells`,
  `TestThreeDimensionalFigures`, `TestThreeDimensionalCrosshairAndMontageContent`,
  `TestTheCurveDrawsValidationAtItsOwnStep`,
  `TestTheClusterFigureNamesItsTapAndCountsCorrectly`,
  `TestTheCrosshairIsVisibleOnSaturatedCells`, `TestTheClusterFigureNamesTheConditionItShows`,
  `TestTheTapLinesCanBeToldApart`, `TestTheCrosshairSentenceIsTrueAtEveryStep`,
  `TestTheSubtitleBreaksAtClauses`, and the layout classes
  `TestThreeDimensionalTitlesAreReadableForRealTapPaths`,
  `TestTwoDimensionalTitlesAreReadableForRealTapPaths` and
  `TestTheMoransProfileIsReadableForRealTapPaths`, which render real-length tap paths
  and assert that no title box overlaps another or leaves the figure. The four 2D
  pixel pins in `TestTwoDimensionalUnchangedPixels` were re-pinned once on purpose,
  for layout and not for content, when the 2D figure was laid out in inches.
- `tests/test_metrics/test_spatial_autocorrelation.py` and
  `tests/test_metrics/test_topographic_selectivity.py`: golden pins of the 2D results
  (`TestMoransIGoldenPins`, `TestLabelIslandsGoldenPins`,
  `TestGrowClustersGoldenPins`) and the 3D oracles (`TestMoransIGrid3D`,
  `TestGridWeightsNd`, `TestGrowClusters3D`). The rank guard is
  `test_a_map_that_is_not_two_dimensional_is_named_not_unpacked`, now parametrized
  over ranks 1, 4 and 5.

**Do not run the full suite as a routine check.** It takes about 1.5 hours and is
also the pre-push hook; scope pytest to the modules you touched.