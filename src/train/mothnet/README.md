# mothnet — MothNet Hebbian Trainer

Trains `MothNet` (`dl_techniques.models.general_purpose.mothnet.model`) on a small MNIST
subsample via its hand-rolled `train_hebbian()` method — the model's readout layer is
`trainable=False` and is updated by a direct Hebbian `.assign()`, never by `model.fit()`
or a custom `train_step`. Outputs land under repo-root `results/<run>/`, built from the
same shared primitives (`prepare_run_dir`, `default_experiment_name`,
`save_training_history_json`, `best_checkpoint_path`) the `bfunet`/`kan` trainers use.

---

## No gradient-based training

MothNet has **no optimizer, no loss backprop, and no `model.fit()` path**. Every epoch of
this trainer calls `model.train_hebbian(x_train, y_train, epochs=1, batch_size=...,
verbose=0)` — the ONE weight-mutating step, updating only the `HebbianReadoutLayer`'s
`readout_weights` via a local Hebbian rule (`ΔW = α · (1/N) · Σ(x_i ⊗ y_i)`). The
Antennal Lobe (competitive inhibition) and Mushroom Body (fixed sparse random
projection) layers never train — see "MothNet-specific caveats" below for what that
implies for the periodic MB-sparsity plot.

---

## Quick start

Always run with a non-interactive matplotlib backend and from the repo root:

```bash
# Plain invocation — CLI defaults (3 epochs, full MNIST: 60000 train / 10000 val samples)
MPLBACKEND=Agg .venv/bin/python -m train.mothnet.train_mothnet

# Custom experiment name
MPLBACKEND=Agg .venv/bin/python -m train.mothnet.train_mothnet \
    --experiment-name my_custom_name --epochs 5

# Larger-scale run — more MNIST samples and MB units, longer training
MPLBACKEND=Agg .venv/bin/python -m train.mothnet.train_mothnet \
    --num-train-samples 10000 --num-val-samples 2000 \
    --mb-units 8000 --epochs 10 --viz-freq 5 --gpu 1
```

`--help` exits 0, prints a `usage:` line, and performs no GPU/dataset/model work.

---

## CLI flags

All flags are defined in `parse_arguments()`. This trainer uses a fresh, purpose-built
`argparse.ArgumentParser` rather than `train.common.create_base_argument_parser()` — see
`plans/plan-2026-09-18T045308-c89cdf76/decisions.md` D-003 for why (MothNet has no
optimizer/LR-schedule/patience/dataset-choice surface for that shared parser to cover).

| Flag | Default | Meaning |
|------|---------|---------|
| `--mb-units` | `16000` | Number of Mushroom Body units. Raised from `2000` (MothNet's own constructor default) per an 8-seed paired statistical sweep — see "MothNet-specific caveats" below |
| `--mb-sparsity` | `0.1` | Fraction of MB units allowed to fire per sample |
| `--al-units` | `None` | Number of Antennal Lobe units; `None` lets MothNet infer it from the input dimension |
| `--connection-sparsity` | `0.1` | Mushroom Body projection connection sparsity |
| `--hebbian-learning-rate` | `0.01` | Learning rate used by the Hebbian readout update |
| `--inhibition-strength` | `0.5` | Antennal Lobe inhibition strength |
| `--epochs` | `3` | Number of training epochs; each epoch is one `train_hebbian(epochs=1)` call. Validated `>= 1` at parse time |
| `--batch-size` | `32` | Mini-batch size passed to `train_hebbian` |
| `--viz-freq` | `2` | Render the four periodic visualizations (MB-sparsity, AL/MB-activation distribution + heatmap, confusion matrix — see "Visualizations" below) every N epochs; `<= 0` disables all four |
| `--gpu` | `0` | GPU index to use |
| `--output-dir` | `None` | Output directory for run artifacts; `None` means repo-root `results/`, resolved at run time |
| `--experiment-name` | `None` | Experiment name; `None` means auto-generated via `default_experiment_name` |
| `--seed` | `42` | Seed, routed through `train.common.set_seeds()` in `build_model()` and through `load_mnist_data`'s own `np.random.default_rng(config.seed)` — see "What `--seed` covers, and what it does not" below |
| `--num-train-samples` | `60000` | Number of MNIST training samples to subsample (full MNIST train split size; previously defaulted to `2000` as a fast-iteration toy subsample). Validated `>= 1` at parse time |
| `--num-val-samples` | `10000` | Number of MNIST validation samples to subsample (full MNIST test split size; previously defaulted to `500` as a fast-iteration toy subsample). Validated `>= 1` at parse time |
| `--eval-batch-size` | `5000` | Chunk size for the per-epoch accuracy computation's forward pass (`_predict_in_batches`), decoupled from `--batch-size` — a pure eval-chunking performance/memory knob with no effect on the computed accuracy value, since it does not change `train_hebbian`'s own training mini-batch |

---

## Output layout

Each run writes to `results/<run>/` (same artifact set as `bfunet`/`train_kan.py`):

```
results/<run>/
├── config.json
├── best_model.keras
├── final_model.keras
├── training_log.csv
├── training_history.json
└── visualizations/
    ├── training_dashboard.png                              (per-epoch, overwritten)
    ├── epoch_{NNN}_mb_sparsity.png                          (every --viz-freq epochs)
    ├── epoch_{NNN}_al_mb_activations_distribution.png       (every --viz-freq epochs)
    ├── epoch_{NNN}_al_mb_activations_heatmap.png            (every --viz-freq epochs)
    └── epoch_{NNN}_confusion_matrix.png                     (every --viz-freq epochs)
```

See "Visualizations" below for what each of the five files shows.

- `best_model.keras` — saved via `best_checkpoint_path()` whenever an epoch's
  `val_accuracy` beats the best seen so far.
- `final_model.keras` and `training_history.json` — **attempted on every run,
  including one that raises mid-loop.** The epoch loop body runs inside a
  `try/finally`, so if `train_hebbian`/`model.save()`/`extract_features` raises on
  some epoch, both saves are still attempted on the way out (best-effort: a save
  that itself then raises is caught, logged via `logger.error`, and does not mask
  the original exception — nothing here silently swallows a failure). This was
  RED-then-GREEN proven with a forced mid-loop exception
  (`tests/test_train/test_mothnet/test_train_mothnet.py::
  test_final_save_attempted_on_mid_loop_exception`).
- `training_log.csv` — one row per completed epoch (`epoch, loss, train_accuracy,
  val_accuracy`), flushed immediately after each write (never batched), so a mid-run
  crash leaves exactly N complete rows for N completed epochs. **The `epoch` column
  is 1-based** (`1, 2, 3, ...`, matching `train_hebbian`'s own internal
  `Epoch {epoch+1}/{epochs}` logging convention and the PNG filenames below) — a
  breaking change to the CSV schema versus this trainer's earlier 0-based column for
  anyone parsing `training_log.csv` directly.
- `visualizations/epoch_{NNN:03d}_<name>.png` (four families — `mb_sparsity`,
  `al_mb_activations_distribution`, `al_mb_activations_heatmap`,
  `confusion_matrix`) — only written when `--viz-freq > 0`; filenames are
  epoch-stamped (1-based, matching the CSV `epoch` column) so periodic renders
  never overwrite each other. **A rerun into the same `--experiment-name`
  starts `visualizations/` clean of stale periodic PNGs**: at run start, each of
  the four `epoch_*_<name>.png` glob patterns above is independently swept and
  any file left over from a prior (possibly longer) run at that same name is
  deleted before the epoch loop begins — matching the fresh-open/overwrite
  behavior every other artifact class here already had (`training_log.csv` opened in
  `"w"` mode; `best_model.keras`/`final_model.keras`/`training_dashboard.png`
  overwritten in place by `model.save()`/`fig.savefig()`). `training_dashboard.png`
  is deliberately NOT globbed — it is overwritten in place every epoch, never
  epoch-stamped, so there is nothing stale to clean.

---

## Visualizations

`training_dashboard.png` renders every epoch, overwritten in place. Every
`--viz-freq` epochs (`<= 0` disables all four of the following), four periodic,
epoch-stamped PNGs render, each behind its OWN `try/except Exception:
logger.warning(...)` at the call site — one failing never skips or aborts the
others (a visualization failure must never abort training; RED-then-GREEN
proven in `tests/test_train/test_mothnet/test_train_mothnet.py::
test_al_features_failure_is_independently_fail_soft_from_confusion_matrix`).

- **`training_dashboard.png`** (`render_training_dashboard`) — a 2-panel plot:
  left panel is training loss vs. epoch; right panel is `train_accuracy` and
  `val_accuracy` vs. epoch on shared axes with a legend.
- **`epoch_{NNN}_mb_sparsity.png`** (`render_mb_sparsity`, reworked from a
  `plt.spy()` presence/absence scatter — see "MothNet-specific caveats" below
  for why) — a per-class Mushroom Body activation-magnitude heatmap. Groups a
  500-row `x_val` sample by true class (`argmax(y_sample)`), averages each
  class's MB codes (`extract_mb_features`) into one row, and column-bins the
  result to `min(mb_units, 200)` bins (`np.array_split`, so an `mb_units` not
  evenly divisible by the bin count never raises). Plots the binned
  `(10, num_bins)` matrix via `imshow` with a colorbar labeled "Mean MB
  activation"; y-ticks are plain class digits (`"0"`..`"9"`) and the MB
  sparsity fraction — architecturally constant across classes, since
  `MushroomBodyLayer` enforces a fixed top-k regardless of class — is stated
  exactly once, in the plot title, read directly off `model.mb_sparsity`
  rather than repeated per class. A class absent from the sample gets an
  all-zero row rather than raising or producing NaN.
- **`epoch_{NNN}_al_mb_activations_distribution.png`** and
  **`epoch_{NNN}_al_mb_activations_heatmap.png`** — rendered via the repo's
  `ActivationVisualization` plugin (`dl_techniques.visualization`), reused
  as-is rather than hand-rolled (`decisions.md` D-001), on a fixed 1000-row
  `x_val` sample's `extract_al_features`/`extract_mb_features` output for both
  the `antennal_lobe` and `mushroom_body` layers:
  - `distribution` — one histogram per layer of that layer's flattened
    activation values, each panel annotated with mean (μ), std (σ), and the
    "% dead" fraction of exactly-zero activations.
  - `heatmap` — one `imshow` panel per layer (`cmap="hot"`), rows are samples
    (capped at 100). The `antennal_lobe` panel's columns are raw neurons; the
    `mushroom_body` panel's columns are column-binned (same `_bin_activation_columns`
    helper `render_mb_sparsity` uses, to at most 200 bins) rather than raw
    neurons — the `ActivationVisualization` plugin titles each panel by its
    data-dict key and hard-codes the x-axis label "Neurons" with no
    per-panel override, so the `mushroom_body` panel's dict key itself carries
    the bin-size annotation (e.g. `"mushroom_body (80-unit bin means)"`) to
    disclose that its 200 x-axis positions are bin means, not individual
    neurons.
- **`epoch_{NNN}_confusion_matrix.png`** — rendered via the repo's
  `ConfusionMatrixVisualization` plugin, reused as-is (`decisions.md` D-001),
  on the FULL val set — it reuses the val-accuracy pass's already-computed
  `val_logits`/`y_val` rather than a separate forward pass, with both sides
  `argmax`'d (`y_true`/`y_pred`, never an integer-label path). Row-normalized
  (`normalize="true"`) 10x10 matrix with both percentages and raw counts
  annotated per cell.

The four periodic filename globs (`epoch_*_mb_sparsity.png`,
`epoch_*_al_mb_activations_distribution.png`,
`epoch_*_al_mb_activations_heatmap.png`, `epoch_*_confusion_matrix.png`) are
each independently cleaned of stale files from a prior run at the same
`--experiment-name` — see "Output layout" above.

---

## MothNet-specific caveats

**`mb_units` default is now 16000, backed by a measured A/B comparison, not a guess.**
The MothNet model's own README (`src/dl_techniques/models/general_purpose/mothnet/
README.md` §7/§12) recommends `mb_units` be 20-50x the input dimension (784 for
flattened MNIST, so roughly 16,000-39,000) for the Mushroom Body's sparse
high-dimensional coding to work as intended. This trainer's old default (`2000`,
matching `MothNet`'s own constructor default, ~2.6x the input dimension) sat well
under that range. An 8-seed paired statistical sweep
(`src/train/mothnet/multiseed_sweep.py`, canonical scale — `--epochs 3
--num-train-samples 2000 --num-val-samples 500`) compared `mb_units=2000` (baseline)
against `mb_units=16000` (candidate), paired by seed: baseline `val_accuracy`
mean±std = 0.246 ± 0.031, candidate = 0.410 ± 0.066, `mean_diff=0.163` (95% bootstrap
CI [0.113, 0.209]), two-sided paired permutation test `p=0.0086` — both legs of the
pre-registered decision rule (`p < 0.05` and `|mean_diff| >= 0.07`) cleared
comfortably (`decisions.md` D-009 has the full numbers and the pre-registration
itself). The CLI default was raised to `16000` on this evidence.
`--mb-units` sizing is still not enforced by this trainer — nothing in
`train_mothnet.py` or in `MothNet`/`train_hebbian` validates the ratio, so passing an
undersized `--mb-units` will not raise or warn — but the default itself is no longer
an unmeasured toy-scale guess. Re-run or extend the comparison with
`python -m train.mothnet.multiseed_sweep --help` (see "mb_units A/B sweep driver"
below).

**OUT OF SCOPE, disclosed gap: this sweep has never been re-run at the current
full-MNIST default scale.** The 8-seed comparison above ran exclusively at the
OLD `--num-train-samples 2000 --num-val-samples 500` toy scale — the sweep's
own `multiseed_sweep.py` hard-codes that scale via its `_NUM_TRAIN_SAMPLES`/
`_NUM_VAL_SAMPLES` module constants (see "mb_units A/B sweep driver" below),
independent of `train_mothnet.py`'s own CLI defaults, so raising this trainer's
defaults to `60000`/`10000` did NOT also raise the scale the sweep validates
`mb_units=16000` against. Whether `mb_units=16000` remains the optimal choice
at full-MNIST scale is genuinely unknown — plausible mechanisms exist in both
directions (more training data could favor either a larger or a smaller
expansion ratio) — and re-validating it there is explicitly OUT OF SCOPE for
the plan that raised the dataset-scale defaults
(`plan-2026-09-18T080513-debe8b11/decisions.md` D-004): a full-MNIST re-sweep
is a materially larger, separately-scoped piece of work that would roughly
double that plan's size and mixes two unrelated concerns (dataset scale vs.
hyperparameter re-optimization) into one. This is a flagged, disclosed
follow-up, not a silent assumption — `--mb-units 16000` remains the shipped
default pending that future sweep.

**What `--seed` covers, and what it does not.** `--seed` is routed through
`train.common.set_seeds(args.seed)`, called once in `build_model()` strictly before
`MothNet(...)` is constructed, plus `load_mnist_data`'s own separate
`np.random.default_rng(config.seed)` generator used for MNIST subsample selection.
Together these cover: (1) `train_hebbian`'s internal `np.random.permutation` shuffle
(`model.py`, no `seed=` kwarg exists there — `set_seeds()` also seeds NumPy's global
RNG, which is what that shuffle draws from); (2) Keras/TensorFlow-initialized
weights — `AntennalLobeLayer`/`HebbianReadoutLayer`'s `kernel_initializer=
'glorot_uniform'` draws its per-instance seed from Python's stdlib `random` module at
object-construction time, and `set_seeds()` seeds that stream too, so weight init is
now reproducible across separate process invocations (this was previously the one
gap this section documented; it is now closed — see below); (3) the Mushroom Body's
sparse-connectivity mask, built at `.build()` time, which draws from the same covered
stream. MEASURED (`decisions.md` D-001, `plan-2026-09-18T060057-c1cfc3d3`): two
separate process invocations with the same `--seed` produced bit-identical
(`np.array_equal`) weights for all 6 weight arrays, including the MB connectivity-
masked projection matrix, immediately after `build_model()` returns and before any
`train_hebbian` call. `--seed` does NOT cover anything outside these three RNG
consumers — there is no remaining known gap in what `--seed` reaches for a
single-process run.

**The periodic MB-sparsity visualization looks identical across epochs — this is
expected, not a bug.** `visualizations/epoch_{NNN}_mb_sparsity.png` renders
`model.extract_mb_features(x_sample)`, which reads only the Antennal Lobe and Mushroom
Body layers — both are architecturally frozen (fixed competitive inhibition, fixed
sparse random projection) and never train; only the readout layer's weights change per
epoch. Originally measured against the pre-rework `plt.spy()` design
(`plan-2026-09-18T060057-c1cfc3d3` Step 6: `--epochs 4 --viz-freq 2`, epoch_002 vs.
epoch_004 byte-identical per `cmp`); re-confirmed against the CURRENT per-class
binned-heatmap design (`plan-2026-09-18T080513-debe8b11` Step 6/8) with the same
`--epochs 4 --viz-freq 2` shape — still byte-identical, as expected, since the
rework changed only how frozen AL/MB output is presented, not which layers or
samples it reads. Do not expect this plot to change during a run; it reflects a
real architectural property of MothNet, not a rendering defect.

**Small `--num-train-samples` can produce flat, at-chance results.** Measured at Step 7:
`--num-train-samples 500 --num-val-samples 200` produced degenerate `val_accuracy`
(0.09 → 0.09 → 0.10, at the 0.10 MNIST chance baseline) across 3 epochs, while the
then-current CLI default of `2000` train / `500` val samples (since raised to the full
MNIST split, `60000`/`10000` — see the CLI flags table above) reached `val_accuracy`
0.114 → 0.150 → 0.198 over the same 3 epochs. If you pass a much smaller
`--num-train-samples` than the full-MNIST default and see a flat accuracy curve, this
sample-count sensitivity — not a wiring bug — is the likely explanation.

---

## Testing

`tests/test_train/test_mothnet/` covers this trainer: a CLI-contract test proving all
16 declared flags reach their destination (a config field, or — for `--gpu` — the
`setup_gpu` call), and five `@pytest.mark.integration`-marked end-to-end tests that
actually train a tiny real model — checkpoint round-tripping; `final_model.keras`/
`training_history.json` still landing after a forced mid-loop exception; all five
periodic visualization files landing at the expected epoch-stamped names; stale
periodic-visualization files from a prior run at the same `--experiment-name` being
cleaned up; and the AL/MB-activation render block's failure being independently
fail-soft from the confusion-matrix render block. This repo's pytest config does not
deselect the `integration` marker by default, so the whole suite (44 tests,
including all five integration tests) runs with:

```bash
pytest tests/test_train/test_mothnet/ -v
```

Target only the real end-to-end training tests with `-m integration`:

```bash
pytest tests/test_train/test_mothnet/ -v -m integration
```

## mb_units A/B sweep driver

`src/train/mothnet/multiseed_sweep.py` is a general-purpose paired-seed A/B driver for
comparing two `--mb-units` values — the tool that produced the measured comparison
documented above (see "MothNet-specific caveats"), and reusable for future
hyperparameter comparisons on this trainer, not a one-off script. It launches one
`train.mothnet.train_mothnet` subprocess per `(arm, seed)` pair, collects each run's
final `training_log.csv` `val_accuracy`, and reports a paired permutation test plus
per-arm `mean_std`/`bootstrap_ci` in a written `summary.md`. See its own `--help` for
the full flag set:

```bash
python -m train.mothnet.multiseed_sweep --help
```

---

## Constraints & gotchas

- **Never route MothNet through `model.fit()`.** `train_hebbian` is the only training
  entry point; the readout weights are `trainable=False`.
- **Labels passed to `train_hebbian` must be one-hot**, never class indices —
  `load_mnist_data` already applies `keras.utils.to_categorical`.
- **Outputs go to repo-root `results/`.** Do not point `--output-dir` inside `src/`.
- **Always set `MPLBACKEND=Agg`** to avoid X11 crashes on headless/remote systems.
