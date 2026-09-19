# power_mlp - PowerMLP classification training

Training script for the `PowerMLP` model
(`dl_techniques.models.general_purpose.power_mlp.model`) on flattened **MNIST** or
**CIFAR-10** images. It trains with stock `model.fit`, keeps labels as integers
(`SparseCategoricalCrossentropy`), and writes one self-contained run directory with the
checkpoints, the metrics, a per-epoch training dashboard and the final evaluation figures.

The trainer is a measurement tool as much as a training script. Every default below was
chosen by a rule fixed before the runs, from runs whose numbers are quoted in this file, and
every number in the "Measured results" section was copied from a `results_summary.json`
written by the trainer itself (`results/` is untracked, so the run directories are not part
of the repository).

---

## Data domain and input scaling

`load_dataset` returns pixels scaled to `[0, 1]`. Two scalings are available:

| `--input-scaling` | Input the model sees | Notes |
|---|---|---|
| `unit` (default) | the loader's plain `[0, 1]` pixels, flattened | MNIST `784` wide (one channel, the loader repeats it into 3 identical channels and channel 0 is lossless), CIFAR-10 `3072` wide |
| `standardize` | `(x - mean) / std` (MNIST `0.1307 / 0.3081`; CIFAR-10 per channel), flattened | starts far above the uniform-prediction loss with `glorot_normal` |

The validation split is a seeded shuffle of the training set (`--validation-split`, default
0.1: 54,000 train / 6,000 validation on MNIST) and is never the test set. The test set is
touched only after training.

**Why `unit` is the default.** `ReLU-k` composes to degree `k ** depth`, so any gain above
the fixed point (input scale times initializer scale) blows the logits up doubly
exponentially with depth. The initial loss of an untrained network is the direct readout;
the uniform prediction gives `ln(10) = 2.3026`.

Measured on GPU 1, MNIST, `default` preset, `k=2`, no batch normalization, seed 0, 3 epochs
(`results/powermlp_init_grid/*`):

| initializer | input scaling | initial loss | ratio to `ln(10)` | test accuracy after 3 epochs |
|---|---|---|---|---|
| `glorot_normal` | `standardize` | 218.574 | 94.9 | 0.9644 |
| `lecun_normal` | `standardize` | 20.271 | 8.80 | 0.9672 |
| `glorot_normal` | `unit` | 2.457 | 1.07 | 0.9655 |
| `lecun_normal` | `unit` | 2.312 | 1.00 | 0.9668 |

Measured on CPU before the defaults were changed (2048 train samples, seed 0, untrained
models, initial loss only, no batch normalization; the script was a scratch probe and is not
committed), all on standardized input: `default` preset `k=2` with `glorot_normal` 210.6 (the
GPU run above gave 218.6, so the probe transferred within 4%) and with `lecun_normal` 19.2;
`default` `k=3` with `glorot_normal` 5.7e11; `deep` `k=2` with `glorot_normal` 6.8e25;
`deep` `k=3` NaN. On plain `[0, 1]` input the same probe gave 2.48 (`default` `k=2`
`glorot_normal`) and 2.32 (`lecun_normal`).

The 3-epoch accuracy differences in the table are inside seed noise (the multi-seed grid
below shows a seed standard deviation of 0.001 to 0.004); the table's job is the initial
loss column, which is structural and not noisy.

---

## Directory layout

| File | Role |
|------|------|
| `train_power_mlp.py` | The trainer: `TrainingConfig`, `_build_parser`, `prepare_data`, the initial-loss guard, `train_model` (the whole run) and `main`. Run as a module. |
| `visualization.py` | The per-epoch `TrainingDashboardCallback` and the four figures written after training (confusion matrix, per-class metrics, calibration, confident errors). Matplotlib, numpy and scikit-learn only. |
| `__init__.py` | Package marker. |

Tests: `tests/test_train/test_power_mlp/` (CLI contract, defaults pins, initial-loss guard,
run health and divergence, figures) and `tests/test_train/test_config_fields_are_live.py`
(every `TrainingConfig` field is read by the trainer).

---

## Quick start

Run from the repository root with the project virtualenv and a non-interactive matplotlib
backend (headless-safe). The venv resolves `train` and `dl_techniques` from `src/`, so no
`PYTHONPATH` is needed (`PYTHONPATH=src` in front is harmless and was used for the grids).

```bash
cd <repo root>

# MNIST, the shipped defaults, 100 epochs, on GPU 1
MPLBACKEND=Agg .venv/bin/python -m train.power_mlp.train_power_mlp \
    --dataset mnist --epochs 100 --gpu 1

# The same, forced to train all 100 epochs (early stopping cannot fire)
MPLBACKEND=Agg .venv/bin/python -m train.power_mlp.train_power_mlp \
    --dataset mnist --epochs 100 --patience 100 --gpu 1 --experiment-name my_mnist_100ep

# CIFAR-10 (batch normalization defaults to OFF here, see below)
MPLBACKEND=Agg .venv/bin/python -m train.power_mlp.train_power_mlp \
    --dataset cifar10 --epochs 100 --gpu 1

# Override the per-dataset batch-normalization default
MPLBACKEND=Agg .venv/bin/python -m train.power_mlp.train_power_mlp --dataset mnist --no-batch-normalization
MPLBACKEND=Agg .venv/bin/python -m train.power_mlp.train_power_mlp --dataset cifar10 --batch-normalization
```

`--help` prints the usage and allocates nothing (no GPU, no run directory).

- Everything is written under `<repo root>/results/<experiment_name>/` whatever the current
  directory is (`--output-dir` is anchored at the repo root when relative).
- **A reused `--experiment-name` is refused.** If the resolved run directory already holds
  any of `results_summary.json`, `config.json`, `best_model.keras`, `run.log` or
  `training_log.csv`, `train_model` raises `FileExistsError` naming the path and asking for
  a new `--experiment-name` before anything is written; nothing is overwritten, merged or
  deleted (an empty or missing directory is fine, and `run.log` is opened in write mode).
  Before this check a reused name silently merged two runs into one directory. The default
  name is `powermlp_<dataset>_<architecture>_<timestamp>`, so it never collides.
- One GPU job at a time. A foreign process on the same GPU shows up as an out-of-memory
  error at import.
- `--gpu N` restricts the process to device `N`; without it memory growth is enabled on
  every visible GPU. Training is single-device either way (there is no distribution
  strategy), so on a multi-GPU host pass `--gpu` to keep the other devices untouched.

---

## Command-line flags

Defaults are read off `TrainingConfig` (the parser cannot drift from the config), except
`--batch-normalization` whose CLI default is `None` so the per-dataset table can apply.

| Flag | Default | Meaning |
|------|---------|---------|
| `--dataset` | `mnist` | `mnist` or `cifar10`. |
| `--input-scaling` | `unit` | `standardize` = `(x - mean) / std`; `unit` = plain `[0, 1]` pixels. |
| `--validation-split` | `0.1` | Fraction of the training set held out (seeded shuffle), strictly inside `(0, 1)`. Never the test set. |
| `--architecture` | `default` | Hidden-width preset: `small`, `default`, `large`, `deep` (table below). |
| `--k` | `2` | Power of the `ReLU-k` activation. |
| `--dropout-rate` | `0.1` | Dropout after each hidden layer, in `[0, 1)`. |
| `--batch-normalization` / `--no-batch-normalization` | per dataset: on for `mnist`, off for `cifar10` | Batch normalization after each hidden layer. An explicit flag always wins. |
| `--kernel-initializer` | `lecun_normal` | `glorot_normal`, `he_normal`, `lecun_normal` or `glorot_uniform`; sets the initial logit scale. |
| `--epochs` | `100` | Maximum number of training epochs. |
| `--batch-size` | `128` | Training batch size. |
| `--learning-rate` | `0.0003` | Initial learning rate. |
| `--optimizer` | `adam` | `adam`, `adamw`, `sgd` or `rmsprop` (built through `optimizer_builder`). |
| `--weight-decay` | `0.0` | Decoupled optimizer weight decay. Never also applied as a kernel regularizer. |
| `--patience` | `15` | `EarlyStopping` patience in epochs on `val_loss`. |
| `--seed` | `42` | Seed for the weights, the shuffling and the validation split. |
| `--epoch-analysis` | off | Run the per-epoch `ModelAnalyzer` callback. Off by default: it measured about 8 s per epoch and its per-epoch files carry only weight and spectral fields. The final analysis always runs. |
| `--output-dir` | `results` | Output root; a relative path is anchored at the repo root. |
| `--experiment-name` | `powermlp_<dataset>_<architecture>_<timestamp>` | Run directory name. |
| `--gpu` | unset | GPU device index. Unset: memory growth on the visible GPUs; training is single-device either way. |

`--no-epoch-analysis` (the old opt-out spelling) was removed and is rejected (exit 2), not
aliased.

Presets (hidden widths only; the input width and the 10 classes are prepended and appended
by `effective_hidden_units`):

| `--architecture` | MNIST hidden widths | CIFAR-10 hidden widths |
|---|---|---|
| `small` | 128, 64 | 256, 128 |
| `default` | 256, 128, 64 | 512, 256, 128 |
| `large` | 512, 256, 128, 64 | 1024, 512, 256, 128 |
| `deep` | 256, 128, 128, 64, 64, 32 | 512, 256, 256, 128, 128, 64 |

Parameter counts pinned by tests: MNIST `default` 486,218 with batch normalization
(484,426 without: batch normalization adds `4 * (256 + 128 + 64) = 1,792`); CIFAR-10
`default` 3,475,594 without batch normalization (3,479,178 with).

Fixed by design, not flags: batch norm is the only normalization, the output is a softmax
(`from_logits=False`), gradient clipping is a per-variable norm of 1.0, `ReduceLROnPlateau`
(factor 0.5, patience 5, floor `1e-7`) and `EarlyStopping(restore_best_weights=True)` come
from `train.common.create_callbacks`, and `kernel_regularizer` is always `None`.

### The batch-normalization default is per dataset

`TrainingConfig.batch_normalization` is `None` by default and is resolved in `__post_init__`
from `BATCH_NORMALIZATION_BY_DATASET = {"mnist": True, "cifar10": False}`, so
`config.json` and the summary always record the value actually used. Each entry comes from
a pre-registered 3-seed x 10-epoch grid on that dataset (numbers below): MNIST gains
+0.0024 test accuracy from batch normalization, CIFAR-10 loses 0.031. A dataset absent from
the table is a `KeyError`, not a fallback: a third dataset must bring its own grid.

---

## Run-directory layout

```
results/<experiment_name>/
    config.json                 resolved TrainingConfig (batch_normalization is a bool here)
    training_log.csv            one row per epoch: epoch, accuracy, loss, lr, val_accuracy, val_loss
    training_history.json       per-epoch lists: accuracy, loss, val_accuracy, val_loss, lr, learning_rate
    best_model.keras            checkpoint of the lowest-val_loss epoch
    final_model.keras           the in-memory model at the end (best weights restored, see below)
    results_summary.json        the run's record (strict JSON, keys below)
    run.log                     the dl logger for this run (not Keras' own stdout lines)
    visualizations/
        training_dashboard.png  six panels, redrawn on a cadence during training
        confusion_matrix.png    counts and row-normalized panels
        per_class_metrics.png   per-class precision / recall / F1
        confidence_calibration.png   reliability diagram with ECE
        misclassifications.png  the most confident errors
        classification_report.json   per-class report, macro F1, accuracy
    model_analysis/             ModelAnalyzer output of the final analysis
    epoch_analysis/             ONLY with --epoch-analysis
```

Notes on individual files:

- **`training_log.csv`**: the `epoch` column is **0-based**. `lr` is the learning rate of
  that epoch (from `train.common.LearningRateLogger`).
- **`training_history.json`**: carries both `lr` and `learning_rate` with the same values
  (one from `LearningRateLogger`, one that Keras' `ReduceLROnPlateau` puts into the logs).
  Removing either means editing `train.common`, so the duplicate stays.
- **`best_model.keras` / `final_model.keras`**: `EarlyStopping` restores the best weights at
  the end of every run, whether or not it stopped early, so the final weights equal the best
  ones by construction. The trainer compares the two and validates that the best
  checkpoint loads (`model_loading_validated`).
- **`run.log`**: the initial-loss sanity line, the data shapes, the optimizer, the test
  results, the analyzer status, plus two derived lines that Keras only prints to stdout
  (`ReduceLROnPlateau: learning rate a -> b from epoch N`, `EarlyStopping: stopped after
  epoch N of M ...`), which the trainer reconstructs from the history. Keras' own stdout
  progress lines are not in it. The file handler is attached for the run and removed when
  `train_model` returns or raises.
- **`training_dashboard.png`**: six panels: Loss (log axis, with the epoch-0 baseline star,
  clipped and annotated only if it is more than 10x the curves; under batch normalization
  the star is the untrained model measured in TRAINING mode on the validation set, the same
  measurement as the initial-loss guard, and its legend says so), Accuracy, Learning rate,
  Generalization gap, Per-epoch time (bars above 3x the median of epochs 2 and later are
  drawn at the ceiling, hatched, and annotated with their true value: epoch 1 is the XLA
  warmup), and Smoothed loss (moving average, present from 6 epochs). The train curve is
  the running mean over the epoch while the validation curve is measured at the epoch end,
  which makes the epoch-1 generalization gap negative; the caption says so.
- **Dashboard cadence**: redrawn after epoch 1, then every `max(1, planned_epochs // 20)`
  epochs, plus once more when training ends if the last epoch was not on the cadence. A
  100-epoch run draws 21 times, not 100. Mid-run the PNG on disk is therefore up to
  `max(1, planned_epochs // 20) - 1` epochs stale (4 epochs behind on a 100-epoch run); a
  process killed between two draws leaves the last cadence draw, not the last epoch. A render costs about 1.0 to 1.2 s (measured, CPU)
  against a training epoch of about 1.8 s; drawing every epoch made a 100-epoch run take
  348 s of wall-clock instead of 259 s with identical metrics.
- **`confidence_calibration.png`**: bins with fewer than 20 samples are drawn hatched, pale
  and annotated with their `n`, because a handful of samples is not evidence; the ECE is not
  affected by the drawing.

### `results_summary.json` keys

Written by one function with `allow_nan=False`: strict JSON, non-finite values become
`null`. Read it with `json.load(..., parse_constant=raise)` to check.

| Key | Meaning |
|---|---|
| `status` | `"ok"` or `"diverged"`. A run whose `loss` or `val_loss` turned non-finite is `diverged`: no evaluation, no figures, no analysis, no `final_model.keras`; a reduced summary (`best_epoch: null`, `non_finite_metrics`, history with `null`s) is written FIRST and only then a `RuntimeError` is raised. |
| `run_dir`, `experiment_name`, `dataset`, `architecture`, `effective_hidden_units`, `params`, `k`, `dropout_rate`, `batch_normalization`, `kernel_initializer`, `input_scaling`, `optimizer`, `learning_rate`, `weight_decay`, `batch_size`, `seed`, `input_normalization`, `epochs_requested`, `monitor` | The configuration actually used. |
| `initial_loss_sanity_eval`, `initial_loss_ratio`, `init_scale_warning`, `initial_loss_mode` | The pre-fit guard: loss on 1024 train samples, its ratio to `ln(C)`, whether the ratio exceeded 10 (strict; a WARNING is logged, the run still starts), and `"inference"` or `"training"` (see Known limits). |
| `epochs_run`, `stopped_early` | Epochs actually trained; whether `EarlyStopping` ended the run (`null` on a `diverged` run: a non-finite loss ended it, not `EarlyStopping`, and no best weights were restored). |
| `best_epoch` | **1-based** epoch of the lowest `val_loss`. |
| `best_epoch_csv_index` | The 0-based twin: `best_epoch - 1`, the row in `training_log.csv`. |
| `lr_reduction_epochs` | 1-based epochs trained at a lower rate than the epoch before, derived from the `lr` history. |
| `best_val_metrics`, `final_val_metrics` | Validation metrics of the best epoch and of the LAST epoch. |
| `test_metrics_final`, `test_metrics_best` | Test metrics of the in-memory weights after best-weight restore, and of `best_model.keras`. Equal by construction. |
| `epoch_times` | Seconds per epoch (epoch 1 includes the XLA warmup). |
| `ece` | Expected calibration error on the FULL test set. |
| `visualizations` | The files written, the ECE, and `failed` (a list, empty on a healthy run; a figure error is swallowed into it rather than killing the run). |
| `model_loading_validated`, `best_checkpoint_load_error` | Result of loading `best_model.keras` back and comparing predictions. |
| `analyzer` | Status, loss, accuracy (first 1000 test samples) and path of the `ModelAnalyzer` run. |
| `notes` | Plain-language caveats for the reader of the file (initial-loss mode, the guard is a floor, epoch indexing, ECE slice, final == best). |

---

## Measured results

All numbers below were copied from `results_summary.json` files. Test accuracy is
`test_metrics_final.accuracy`, which equals the best-`val_loss` weights' accuracy.

### MNIST, 10 epochs, `default` preset, `k=2`, `unit`, 3 seeds (`results/powermlp_seed_grid/*`)

The pre-registered rule for the MNIST default (fixed before any of these runs; decisions
D-021 and D-025 of the planning record): adopt batch
normalization iff the best batch-normalized arm's mean beats the best non-batch-normalized
arm's mean by more than 0.002 AND its worst seed is above that best non-BN mean; then pick
the initializer inside the chosen family by the same 0.002 tie rule. 12 runs, all valid
(status ok, every epoch finite, no initial-loss warning, no failed figure).

| initializer | batch norm | seed 0 | seed 1 | seed 2 | mean | min | sd |
|---|---|---|---|---|---|---|---|
| `lecun_normal` | off | 0.9739 | 0.9802 | 0.9788 | 0.97763 | 0.9739 | 0.00331 |
| `glorot_normal` | off | 0.9720 | 0.9788 | 0.9771 | 0.97597 | 0.9720 | 0.00354 |
| `lecun_normal` | on | 0.9787 | 0.9810 | 0.9805 | **0.98007** | 0.9787 | 0.00121 |
| `glorot_normal` | on | 0.9797 | 0.9814 | 0.9788 | 0.97997 | 0.9788 | 0.00132 |

Outcome: batch normalization gap `0.98007 - 0.97763 = +0.00243` (bar 0.002), worst BN seed
0.9787 above 0.97763, so batch normalization is on; inside it `glorot - lecun = -0.00010`,
a tie, so `lecun_normal`. The margin over the bar is thin (0.00043) and rests on 3 seeds;
the direction is not carried by one seed (batch normalization beat no-BN in 6 of 6
same-seed same-initializer pairs, gaps +0.0008 to +0.0077). The non-BN arms' seed sd
(0.0033 to 0.0035) is larger than the 0.003 the planners had assumed would make the bar
unresolvable; the rule had no sd clause and was applied as written.

### CIFAR-10, 10 epochs, `default` preset, `k=2`, `unit`, `lecun_normal`, 3 seeds (`results/powermlp_cifar10_grid/*`)

Same style of rule, fixed before the runs (decisions D-027 and D-028): adopt batch
normalization for CIFAR-10 iff the mean gain is above 0.002 and the worst BN seed is above the
no-BN mean; otherwise it stays off. 6 runs, all valid.

| batch norm | seed 0 | seed 1 | seed 2 | mean | min | sd | mean final val loss | params |
|---|---|---|---|---|---|---|---|---|
| off | 0.5098 | 0.5122 | 0.5087 | **0.51023** | 0.5087 | 0.00179 | 1.3819 | 3,475,594 |
| on | 0.4785 | 0.4588 | 0.4999 | 0.47907 | 0.4588 | 0.02056 | 1.5544 | 3,479,178 |

Outcome: gap `-0.0312`, batch normalization worse in 3 of 3 seeds (paired differences
-0.0313, -0.0534, -0.0088), so the CIFAR-10 default is off. The mechanism (under-trained
3.5M-parameter model, batch-statistics noise raising validation loss) is a guess and was not
tested.

### Long runs (MNIST, `default`, `lecun_normal`, `unit`, batch norm on, seed 0)

| run | flags | epochs run | best epoch | test accuracy | test loss | wall-clock |
|---|---|---|---|---|---|---|
| A: `results/powermlp_mnist_100ep_seed0` | `--epochs 100` (patience 15) | 37 (early stop) | 22 | 0.9836 | 0.0681 | 153 s |
| B: `results/powermlp_mnist_100ep_seed0_patience100` | `--epochs 100 --patience 100`, dashboard redrawn every epoch | 100 | 22 | 0.9836 | 0.0681 | 348 s |
| final: `results/powermlp_mnist_100ep_final_seed0` | `--epochs 100 --patience 100`, dashboard on the cadence | 100 | 22 | 0.9836 | 0.0681 | 259 s |

The three runs share the same best epoch and bit-identical test metrics because
`EarlyStopping` restores the epoch-22 weights in each of them. **Runs A, B and "final" were
produced before the epoch-0 dashboard baseline was moved to training mode** (see the
control below): at HEAD the same command gives a different, equally valid seed-0
trajectory for a batch-normalized run. Details of the final run:

- Test accuracy 0.9836, test loss 0.0681; validation accuracy 0.9842 at the best epoch,
  best `val_loss` 0.06259 (epoch 22), last-epoch `val_loss` 0.06479.
- Parameters 486,218; initial loss 2.7601, ratio 1.20 to `ln(10)` (measured in training
  mode); `init_scale_warning` false.
- `lr_reduction_epochs` = 14, 20, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73 (12 halvings from
  `3e-4`; the rate is at its `1e-7` floor from epoch 73).
- Sum of `epoch_times` 196.5 s (epoch 1: 15.6 s; median of the rest 1.80 s), against a wall
  of 259 s: 62.5 s went to startup, data loading, evaluation, the final analysis, the
  figures and the dashboard renders. Run B: sum 195.9 s against 348 s (152 s of overhead),
  so the redraw cadence saved 89 s (25.6%) and changed no metric.
- Top-level `ece` 0.0085 (full test set); the analyzer reports accuracy 0.9820 and loss
  0.0644 on the first 1000 test samples.
- GPU memory was sampled every 20 s beside runs A and B (`nvidia-smi`, whole device): 733 to
  737 MiB after the first sample, flat, peak 737 MiB (about 15 MiB before the process
  started). The final run was not sampled and the summary does not record memory.

### Long-horizon control: batch normalization on vs off, 100 epochs (MNIST, `default`, `lecun_normal`, `unit`, seed 0, `--patience 100`)

The MNIST batch-normalization default rests on a 10-epoch margin of +0.00243. This control
trains both arms for 100 epochs at HEAD (`cc050e3a0`), GPU 1, one job at a time, from the
repository root. The batch-normalized command was run twice: "final" is the earlier run
(before the epoch-0 baseline change), "final2" is the same command at HEAD.

| run | batch norm | best epoch | test accuracy | test loss | best `val_loss` | last-epoch `val_loss` | wall-clock |
|---|---|---|---|---|---|---|---|
| `powermlp_mnist_100ep_final_seed0` (earlier code) | on | 22 | 0.9836 | 0.0681 | 0.0626 | 0.0648 | 259 s |
| `powermlp_mnist_100ep_final2_seed0` (HEAD) | on | 8 | 0.9797 | 0.0664 | 0.0615 | 0.0657 | 292 s |
| `powermlp_mnist_100ep_nobn_seed0` (HEAD) | off | 13 | 0.9806 | 0.0784 | 0.0923 | 0.1255 | 262 s |

Honest reading:

- The two batch-normalized runs have the same configuration and seed and differ by 0.0039
  test accuracy (0.9836 vs 0.9797). The cause that was measured: the trainer's epoch-0
  baseline pass runs the batch-normalized model in training mode, which draws dropout masks
  and advances the model's three dropout seed generators (CPU probe: 3 of 3 change), so a
  batch-normalized run starts `fit` from a different dropout stream than before; a
  non-batch-normalized run uses `model.evaluate` (dropout off) and is unaffected. That the
  accuracy difference comes from this stream change is an inference: the earlier code is
  bit-identical run to run (runs B and "final"), and no run isolates the cause.
- So the seed-0 test accuracy of one configuration moves by about 0.004 with the dropout
  stream, which is larger than the batch-normalization effect measured at 10 epochs (+0.0024)
  and larger than the 100-epoch gaps here: +0.0030 (earlier run) and -0.0009 (HEAD run)
  against the non-batch-normalized run. **At 100 epochs the accuracy benefit of batch
  normalization is neither confirmed nor refuted by one seed.**
- What is consistent across both batch-normalized runs is the loss: best `val_loss` 0.0615
  and 0.0626 against 0.0923, test loss 0.0664 and 0.0681 against 0.0784, and a last-epoch
  `val_loss` of 0.066 against 0.126 (the non-batch-normalized run overfits more: training
  loss 5.1e-3 against 1.3e-3 to 1.4e-3 at epoch 100). Test loss is at the
  `val_loss`-selected weights, so this is the selection metric's own advantage.
- The default is unchanged: the control does not reverse the pre-registered 10-epoch,
  3-seed decision, it only shows that decision's accuracy margin is small relative to
  run-to-run noise at the horizon the quick start recommends.

### CIFAR-10, 3 epochs, seed 0, `default` preset (single runs)

| batch norm | test accuracy | test loss | best val loss | ECE | run |
|---|---|---|---|---|---|
| off | 0.4486 | 1.5365 | 1.5229 | 0.0037 | `results/powermlp_cifar10_default_20260919_022617` |
| on (run C, **superseded**) | 0.4345 | 1.6056 | 1.6100 | 0.0311 | `results/powermlp_cifar10_default_20260919_041145` |

Run C ran under the earlier global batch-normalization default, before CIFAR-10 was given
its own (off) default; it is kept only as provenance. These single 3-epoch runs pointed the
same way as the 10-epoch, 3-seed grid above and are not the evidence for the default.

---

## Known limits

Each item is backed by a measurement above or a test.

1. **`--architecture deep` and `--k 3` can diverge without batch normalization.** With
   batch normalization off, a real tiny `deep` + `--k 3` run on MNIST warns at the initial
   loss (ratio about 9e5), then trains to NaN (the pinned test run is 2 epochs long); the trainer reports
   `status: "diverged"` with a strict-JSON summary and raises. This is pinned by
   `tests/test_train/test_power_mlp/test_run_health.py` (the tiny run uses
   `--no-batch-normalization`). Initial-loss ratio to `ln(C)` at the **shipped defaults**
   (`lecun_normal`, `unit`, dropout 0.1, per-dataset batch normalization), measured on CPU
   on the first 1024 real training samples, untrained models. The three values in a cell
   differ ONLY in the weight-initialization seed (0, 1, 2); the 1024-sample slice is the
   same in all three (the first 1024 rows of the seed-0 train/validation split), so a
   ratio near 10 is itself seed-dependent (with a per-seed data split, CIFAR-10 `deep`
   `k=2` seed 1 would read 17.8 instead of 3.09, across the warning threshold). **Measured
   init only: no training was run for `large` or `deep`, and none of the numbers below says
   the run trains well.**

   | preset | k | MNIST (BN on, training mode) | MNIST with `--no-batch-normalization` | CIFAR-10 (BN off, the default) | CIFAR-10 with `--batch-normalization` |
   |---|---|---|---|---|---|
   | `small` | 2 | 1.23 / 1.16 / 1.21 | 1.01 / 1.00 / 1.01 | 1.02 / 1.05 / 1.14 | 1.17 / 1.18 / 1.22 |
   | `small` | 3 | 1.19 / 1.16 / 1.19 | 1.01 / 1.01 / 1.01 | 1.16 / 1.28 / 2.06 | 1.15 / 1.16 / 1.19 |
   | `default` | 2 | 1.20 / 1.22 / 1.17 | 1.00 / 1.00 / 1.00 | 1.07 / 1.03 / 1.09 | 1.18 / 1.16 / 1.22 |
   | `default` | 3 | 1.14 / 1.14 / 1.16 | 1.00 / 1.00 / 1.00 | **39.5 / 16.0 / 231** | 1.13 / 1.12 / 1.12 |
   | `large` | 2 | 1.18 / 1.17 / 1.18 | 1.00 / 1.00 / 1.00 | 1.20 / 1.28 / 1.09 | 1.13 / 1.16 / 1.15 |
   | `large` | 3 | 1.11 / 1.11 / 1.10 | 1.00 / 1.00 / 1.00 | **1.1e11 / 1.4e12 / 2.2e7** | 1.07 / 1.11 / 1.09 |
   | `deep` | 2 | 1.08 / 1.09 / 1.13 | 1.00 / 1.00 / 1.00 | **35.0** / 3.09 / **429** | 1.07 / 1.08 / 1.08 |
   | `deep` | 3 | 1.05 / 1.08 / 1.07 | **9.3e5 / 56.8 / 2.6e17** | **NaN / NaN / NaN** | 1.05 / 1.07 / 1.09 |

   Bold cells are above the guard's warn threshold of 10 (NaN raises). Reading: on MNIST
   with the shipped default (batch normalization on) every preset and `k` starts at about
   `ln(10)`; the divergence is a no-batch-normalization phenomenon on the deepest
   configuration, and on CIFAR-10 (3072-wide input, batch normalization off by default)
   `k=3` and `deep` start far above `ln(10)`. Batch normalization removes the initial blow-up
   in every row, but whether those runs then train stably or accurately was not measured.
   The guard warns above 10x and never raises on a large finite loss; a ratio between 3x
   and 10x raises no warning and is not healthy (a floor for hazards, not a health
   certificate).
2. **The batch-normalization guard is measured in training mode.** At init the moving
   statistics are `(0, 1)`, so an inference-mode pass sees un-normalized activations
   (measured 4.2e11 for BN + `glorot_normal` + `standardize` + `k=3`, against 2.68 in
   training mode) and would warn on a healthy run. The guard therefore evaluates a
   batch-normalized model in training mode on the 1024-sample slice, then restores every
   weight (the moving statistics are bit-identical afterwards). Two consequences: dropout
   is active in that pass, and the batch statistics come from 1024 samples instead of the
   128-sample training batches (the dashboard's epoch-0 star reuses the same measurement
   on the validation set (6,000 samples on MNIST), in chunks of 1024, so on a batch-normalized run it can
   differ from `initial_loss_sanity_eval.loss` only by the data, never by the mode), so BN ratios (1.05 to 1.37 measured) sit a little above the
   non-BN ones (0.998 to 1.09). `initial_loss_mode` in the summary says which regime
   produced the number.
3. **First-epoch XLA warmup.** Epoch 1 costs about 15 to 26 s against about 1.8 s for
   the later epochs (15.3 to 16.0 s in the 100-epoch runs, 22.8 s and 25.5 s in the earlier
   3-epoch runs). The dashboard's time panel clips and annotates that bar.
4. **Two ECE numbers.** Top-level `ece` (and `visualizations.ece`) is computed on the full
   test set; the analyzer's calibration numbers and accuracy in `model_analysis/` use the
   first 1000 test samples, so they differ (final run: full-test accuracy 0.9836, analyzer
   0.9820). Both are stated in `notes`.
5. **A 100-epoch MNIST run overfits and gains nothing after epoch 22.** In the final
   run the training loss reaches 1.3e-3 (training accuracy 0.99976) while the validation
   loss is 6.5e-2 at epoch 100 (validation accuracy 0.9842, the same as at the best epoch).
   `ReduceLROnPlateau` halves the rate 12 times and reaches its `1e-7` floor at epoch 73,
   which with `--patience 100` (early stopping cannot fire) means epochs 73 to 100 train at
   that floor, a rate that changed nothing measurable (`val_loss` 0.06486 at epoch 73,
   0.06479 at epoch 100). The default patience of 15 stops the same run at epoch 37 with
   the identical test accuracy in 153 s instead of 259 s. `--patience 100` exists to make
   the run train the requested 100 epochs, not because it helps.
6. **`best_epoch` semantics.** `best_epoch` is 1-based, the CSV `epoch` column is 0-based
   (`best_epoch_csv_index` is the CSV row). `test_metrics_final` describes the weights after
   the best epoch was restored, so it equals `test_metrics_best` on every run (difference
   0.0 in all 18 grid runs); the LAST epoch's validation metrics are `final_val_metrics`,
   and `final_val_metrics.val_loss` is above `best_val_metrics.val_loss` whenever the best
   epoch is not the last (9 of the 12 MNIST grid runs).
7. **Per-dataset batch-normalization default.** MNIST on, CIFAR-10 off, each from its own
   grid (above). The MNIST evidence is thin (+0.00243 against a 0.002 bar, 3 seeds, a
   10-epoch horizon); the 100-epoch control does not settle it in accuracy (one seed, the
   same batch-normalized configuration ranges over 0.9797 to 0.9836 against 0.9806 without)
   but favours batch normalization consistently in loss. Override with `--batch-normalization` / `--no-batch-normalization`; the resolved value is in
   `config.json` and the summary.
8. **Shared writers.** `training_log.csv` and `training_history.json` come from
   `train.common` and can carry `NaN` tokens on a diverged run (only `results_summary.json`
   is guaranteed strict JSON); `training_history.json` repeats the learning rate under two
   keys (`lr`, `learning_rate`).
9. **What the numbers do not cover.** One dataset pair (MNIST, CIFAR-10), the `default`
   preset for every comparison, `k=2` for every trained run, a single seed (0) for the
   100-epoch runs (the trainer's own default seed is 42), and 10-epoch grids for the
   defaults. Nothing here compares PowerMLP with other models.
10. **The CIFAR-10 batch-normalization default is measured for one configuration.** The grid
    that chose "off" ran `default` / `k=2` for 10 epochs only. Off is also the
    divergence-prone arm for `deep` and `k=3` on CIFAR-10 (limit 1: `default` `k=3` starts
    at 16x to 231x `ln(10)`, `deep` `k=2` at 35x and 429x in 2 of 3 seeds, `large` `k=3`
    at 2.2e7 to 1.4e12, `deep` `k=3` raises on a non-finite initial loss in 3 of 3 seeds,
    while `--batch-normalization` starts every
    one of them at 1.05 to 1.22). Use `--batch-normalization` for those configurations.
11. **Flags that were unit-tested but never executed end to end** (only construction,
    parsing or a stubbed call is pinned, no real training run used them): `--optimizer sgd`,
    `rmsprop` and `adamw`; `--weight-decay` above 0; `--epoch-analysis` (the real per-epoch
    analyzer last ran in the first iteration, before batch normalization was adopted);
    `--patience 1`; `--validation-split` near its edges (only the rejected values are
    tested); and training with the `large` or `deep` presets or `k` other than 2 (initial
    loss only, limit 1). There is no resume path.
