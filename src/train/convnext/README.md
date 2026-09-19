# convnext - ConvNeXt V1 / V2 classification training

Training scripts for the two supervised ConvNeXt families
(`dl_techniques.models.vision.convnext.convnext_v1` and `.convnext_v2`) on **MNIST**,
**CIFAR-10** and **CIFAR-100**, a driver that compares the two stochastic-regularization
modes (`depth` vs `gradient`), and a MAE pretraining script that is **not** part of the
normalized trainers (see "The MAE trainer").

Both supervised trainers are thin wrappers over one shared orchestrator,
`train/convnext/common.py`: the CLI, data pipeline, optimizer, schedule, callbacks, evaluation
and every artifact live there once, and the wrappers only fix the model family. The trainer is
a measurement tool as much as a training script: every run writes a self-contained run
directory with a strict-JSON `results_summary.json`, so a number quoted anywhere can be traced
to a file the trainer itself wrote.

---

## Directory layout

| File | Role |
|------|------|
| `common.py` | The orchestrator: `TrainingConfig`, `_build_parser`, `parse_arguments(argv, model_family)`, `prepare_data`, `make_train_dataset`, `build_lr_schedule`, the initial-loss guard, `train(config)` (the whole run) and `main(model_family, argv)`. Not run directly. |
| `train_convnext_v1.py` | ConvNeXt V1 wrapper: `main(argv)` calls `common.main("v1", argv)`. `--variant` offers exactly the keys of `ConvNeXtV1.MODEL_VARIANTS`. |
| `train_convnext_v2.py` | ConvNeXt V2 wrapper (GRN blocks): `common.main("v2", argv)`. `--variant` offers exactly the keys of `ConvNeXtV2.MODEL_VARIANTS`. |
| `run_stochastic_comparison.py` | Driver: trains one trainer twice (`depth`, then `gradient`, same seed) and emits a comparison report. |
| `train_convnext_v2_mae.py` | ConvNeXt V2 with MAE pretraining. **Not normalized**, out of scope of everything below except its own section. |
| `__init__.py` | Package marker. |

Shared code used by the orchestrator lives in `src/train/common/`
(`run_artifacts.py`: `refuse_existing_run`, `attach_run_log`, `write_summary_json`;
`classification_viz.py`: the dashboard and the four end-of-run figures, shared with the
PowerMLP trainer).

---

## Quick start

Run from the repository root with the project virtualenv and a non-interactive matplotlib
backend. Pick the GPU with `CUDA_VISIBLE_DEVICES` in the shell (see "GPU selection").

```bash
cd <repo root>

# V1, CIFAR-10, the 2-stage cifar10 variant, 100 epochs (the defaults)
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.convnext.train_convnext_v1 \
    --dataset cifar10 --variant cifar10

# V2, CIFAR-100 (drop path and dropout default to 0.2 for this dataset), named run
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.convnext.train_convnext_v2 \
    --dataset cifar100 --variant cifar10 --epochs 100 --experiment-name my_c100_v2

# A 4-stage variant with the standard 32 -> 16 -> 8 -> 4 -> 2 schedule (see "Geometry")
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.convnext.train_convnext_v1 \
    --dataset cifar10 --variant tiny --strides 2

# Smoke run: 512 samples, 2 epochs
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.convnext.train_convnext_v1 \
    --dataset cifar10 --variant cifar10 --epochs 2 --max-samples 512 --experiment-name smoke
```

`--help` parses first and exits, so it allocates no GPU and no run directory.

- Everything is written under `<repo root>/results/<experiment_name>/` whatever the current
  directory is (`--output-dir` is anchored at the repo root when relative).
- **A reused `--experiment-name` is refused.** If the resolved run directory already holds any
  of `results_summary.json`, `config.json`, `best_model.keras`, `run.log` or `training_log.csv`,
  `train()` raises `FileExistsError` naming the path before anything is written; nothing is
  overwritten, merged or deleted. The default name is
  `convnext_<v1|v2>_<dataset>_<variant>_<timestamp>`, so it never collides.
- One GPU job at a time.
- The datasets come from the local Keras cache (`~/.keras/datasets`, through
  `train.common.load_dataset`). A missing file makes Keras try to download it and fail loudly;
  nothing here retries.

---

## Command-line flags

Defaults are read off `TrainingConfig` (the parser cannot drift from the config; a test pins
it). Both wrappers have the same flags; only `--variant` differs.

| Flag | Default | Meaning |
|---|---|---|
| `--dataset {mnist,cifar10,cifar100}` | `cifar10` | Dataset. `imagenet` is refused with an explicit error (see "Data"). |
| `--validation-split` | `0.1` | Fraction of the TRAIN set held out (seeded) for early stopping and checkpoint selection. Strictly inside (0, 1); 0 is refused because it would silently validate on the test set. |
| `--max-samples` | none | Cap the train pool and the test set at this many samples (smoke runs). Must be at least 2. |
| `--variant` | `cifar10` | V1: `cifar10, tiny, small, base, large, xlarge`. V2: `cifar10, atto, femto, pico, nano, tiny, base, large, huge`. Sets depths and dims. |
| `--kernel-size` | `7` | Depthwise convolution kernel size. |
| `--strides` | `4` | Stem patch size AND every inter-stage downsample stride (one knob). See "Geometry". |
| `--drop-path-rate` | per dataset | Maximum stochastic-depth rate. `0.1` for mnist and cifar10, `0.2` for cifar100. |
| `--stochastic-mode {depth,gradient}` | `depth` | `depth` = `StochasticDepth`, `gradient` = `StochasticGradient` (see below). |
| `--dropout-rate` | per dataset | Dropout inside each block. `0.1` for mnist and cifar10, `0.2` for cifar100. |
| `--use-gamma` / `--no-use-gamma` | on | Learnable per-channel layer scale in each block. |
| `--epochs` | `100` | Maximum epochs. The cosine spans exactly this many. |
| `--batch-size` | `64` | Training batch size. |
| `--learning-rate` | `0.001` | Peak learning rate. |
| `--weight-decay` | `0.0001` | Decoupled AdamW weight decay (never combined with an L2 regularizer). |
| `--lr-schedule {cosine,exponential,constant}` | `cosine` | Schedule over the whole run. `constant` passes a plain float and adds `ReduceLROnPlateau` (factor 0.5, patience 5, on `val_loss`). |
| `--warmup-epochs` | `0` | Linear warmup epochs before the cosine. Only valid with `cosine` and strictly below `--epochs`; otherwise refused at config time. |
| `--patience` | `50` | Early-stopping patience in epochs, on `val_loss`. |
| `--seed` | `42` | Seed for weights, shuffling, augmentation and the splits. |
| `--epoch-analysis` | off | Also run the per-epoch `ModelAnalyzer` callback into `epoch_analysis/`. The end-of-run analysis always runs. |
| `--output-dir` | `results` | Output root; a relative path is anchored at the repo root. |
| `--experiment-name` | `convnext_<family>_<dataset>_<variant>_<timestamp>` | Run directory name. A name that already holds a run is refused. |
| `--gpu` | none | GPU index; see "GPU selection". Not a config field. |

Every default above is the value the pre-normalization scripts shipped with. None of them was
measured for this model, and some are arguable (a weight decay of `1e-4` is small for AdamW on a
ConvNeXt, a patience of 50 cannot fire in a short run, there is no warmup by default). They are
kept unchanged so the first audited runs measure what a user actually gets; a change to any
default belongs to a measured comparison, not to this file.

### `stochastic_mode`: `depth` vs `gradient`

Both are threaded to the model constructor and are two different regularizers, not two
implementations of the same one:

- **`depth` -> `StochasticDepth`**: per-sample Bernoulli drop of the entire residual branch
  with the `/keep_prob` rescale (classic drop path).
- **`gradient` -> `StochasticGradient`**: forward-identity; only the gradient through the branch
  is stochastically `stop_gradient`-ed.

No claim is made here about which one is better. The earlier A/B numbers that used to sit in
this file were measured under the collapsed learning-rate schedule described in "Optimization"
and were removed; the driver below is how to measure it again.

---

## Data

- **Source**: `train.common.load_dataset` (`keras.datasets.{mnist,cifar10,cifar100}`), pixels
  scaled to `[0, 1]`. MNIST comes as 28x28 with its single channel repeated into 3 (input shape
  `28x28x3`).
- **Validation split**: a seeded permutation of the train set; the first
  `int(n * validation_split)` samples are the validation set, the rest is the fit set. With
  `--max-samples`, the train pool and the test set are first cut to a seeded subset of that
  size.
- **The test set is used only for reporting**, after training. It never drives early stopping,
  the best checkpoint or `best_epoch`. The old scripts passed the test set as
  `validation_data`, so the reported test number was selected on itself; that leak is gone.
- **Normalization**: per-channel mean and std computed on the fit split only, applied to fit,
  validation and test. Both statistics are recorded in the summary (`input_normalization`).
- **Augmentation** (train pipeline only, tf.data, reshuffled every epoch): a pad-4 random crop
  back to the image size for every dataset, plus a random horizontal flip for CIFAR-10 and
  CIFAR-100. **MNIST gets no flip** (a flipped digit is a different digit). The padding is
  added after standardization, so the border is the per-channel mean colour, not black.
  Validation and test are not augmented.
- **`imagenet` is not supported.** The `tfds` `imagenet2012` data is not available on this
  machine, and the full-test-set figures (confusion matrix, calibration, confident errors)
  cannot be built for a streaming dataset. `--dataset imagenet` fails with an explicit message
  at the parser, and `TrainingConfig` refuses it too, before any GPU or run directory is
  touched.

---

## Optimization

- One optimizer, identical for V1 and V2: `AdamW(learning_rate=<schedule>, weight_decay=wd,
  clipnorm=1.0)`. The model is built without a kernel regularizer, so decay is applied exactly
  once.
- Loss `SparseCategoricalCrossentropy(from_logits=True)`: the V1/V2 head is a bare
  `Dense(num_classes)` and emits logits. Metrics are `accuracy`, plus `top_5_accuracy` (a metric
  object, not the unresolvable string alias) when there are more than 10 classes.
- **The cosine spans the whole run.** `steps_per_epoch = n_train // batch_size` (the train pipeline drops the incomplete last
  batch, see "First epoch" under Measured results) is always passed to the schedule. Without it the library counts optimizer steps against
  `decay_steps = epochs` and the cosine reaches its 1% floor after `epochs` batches: the old
  trainers trained at about `1e-5` from the second epoch on. A guard pins this.
- **Warmup** is engaged through `warmup_steps = warmup_epochs * steps_per_epoch` (the library's
  `warmup_epochs` argument is a reserved no-op).
- **Model selection** uses `val_loss` (early stopping, `best_model.keras`, `best_epoch`). This
  is inherited from the PowerMLP trainer, not a measured preference; whether `val_accuracy`
  would pick a different epoch is an open audit question.
- **Best versus final.** Keras 3.8 `EarlyStopping(restore_best_weights=True)` restores the best
  weights at every train end, so after `fit` the in-memory model is always the best one. The
  trainer copies the weights at the end of every epoch and, after the figures and the analyzer
  (which describe the best weights), swaps the last epoch's weights back in: that model is
  `final_model.keras` and is what `test_metrics_final` measures. `test_metrics_best` is the
  reloaded `best_model.keras`. The two differ whenever the last epoch is not the best one
  (`final_is_best`).
- **Health guard.** Before `fit` the untrained model is evaluated once on the validation split;
  the ratio to `ln(num_classes)` is stored as `initial_loss_ratio` and a WARNING is logged when
  it is strictly above 10. A non-finite initial loss raises before `fit`. The dashboard's
  epoch-0 marker reuses this same measurement.
- **Divergence.** `TerminateOnNaN` is on. A run whose `loss` or `val_loss` turns non-finite gets
  a reduced strict-JSON `results_summary.json` with `status: "diverged"`, and then a
  `RuntimeError` is raised; no evaluation, figures, analysis or `final_model.keras` are produced.

---

## Geometry: `--strides` is one knob

The stem is a `strides x strides` convolution at stride `strides` with `"valid"` padding
(`"same"` when `strides == 1`), giving `n // strides`; every downsample between stages is
`"same"`, giving `ceil(n / strides)`. There is no separate stem stride.

| Input | Stages | `--strides` | Feature map per stage |
|---|---|---|---|
| 32x32 | 4 | 4 (default) | 8, 2, 1, 1 |
| 32x32 | 4 | 2 | 16, 8, 4, 2 |
| 32x32 | 2 (`cifar10` variant) | 4 (default) | 8, 2 |
| 28x28 (MNIST) | 4 | 4 (default) | 7, 2, 1, 1 |

So the default `strides=4` builds and trains on 32x32 four-stage models (there is no crash),
but the last two stages run on a single pixel, so the spatial mixing of the depthwise
convolution there is degenerate. `--strides 2` gives the usual `32 -> 16 -> 8 -> 4 -> 2` schedule for
`tiny/small/base/...`. Every summary records `stage_feature_map_sizes` (computed by
`stage_feature_map_sizes`, which a test checks against the real `include_top=False` model) and
the run log prints it, so a degenerate geometry is visible in the record of the run.

---

## GPU selection

**`CUDA_VISIBLE_DEVICES` exported in the shell is the supported control.** `--gpu N` also works
in the current code: `main()` parses first, then calls `setup_gpu(gpu_id=N)`, which sets
`CUDA_VISIBLE_DEVICES=N` before TensorFlow first enumerates devices, and it overrides an
exported value. That was measured with device queries only (no training): importing `train.convnext.common`
enumerates no device, so the flag takes effect; but after ANY earlier device query in the same
process, changing the variable is inert (TensorFlow keeps the first device). That effectiveness
rests on nothing enumerating devices at import time, which a later import could break, so do not
rely on the flag without checking.

The check is in the record: the summary stores `gpu_name` and `tf_visible_devices` (what
TensorFlow actually used) next to `cuda_visible_devices` (the environment value at call time),
and the run log names all three before data loading. Without either control, memory growth is
enabled on every visible GPU; training is single-device.

---

## Run-directory layout

```
results/<experiment_name>/
    config.json                 resolved TrainingConfig (drop_path_rate / dropout_rate are floats here)
    training_log.csv            one row per epoch: epoch, accuracy, loss, lr, val_accuracy, val_loss (+ top-5 for 100 classes)
    training_history.json       per-epoch lists of every history metric
    best_model.keras            checkpoint of the lowest-val_loss epoch
    final_model.keras           the LAST epoch's weights (not the best; see "Best versus final")
    results_summary.json        the run's record (strict JSON, keys below)
    run.log                     the dl logger for this run
    visualizations/
        training_dashboard.png  per-epoch curves, redrawn on a cadence during training
        confusion_matrix.png
        per_class_metrics.png
        confidence_calibration.png   reliability diagram with ECE
        misclassifications.png       the most confident errors
        classification_report.json
    model_analysis/             ModelAnalyzer output of the final analysis (analysis_results.json, ...)
    epoch_analysis/             ONLY with --epoch-analysis
```

Notes:

- **`training_log.csv`**: the `epoch` column is 0-based, `best_epoch` in the summary is 1-based
  (`best_epoch_csv_index` = `best_epoch - 1`). `lr` is the rate at the START of the epoch (its
  first step): epoch 1 shows the configured base rate (or the warmup start value under
  warmup), and the last epoch shows the rate it began at, not the schedule's value past its
  end.
- **Per-epoch line and the progress bar**: `run.log` and the console carry one line per epoch,
  `Epoch N/E - loss X - accuracy X - [top_5_accuracy X -] val_loss X - val_accuracy X -
  [val_top_5_accuracy X -] lr X - time Ns`, built from the true epoch logs (the same values as
  the CSV row, 4 decimals). The Keras progress bar is kept, but its TRAIN numbers read low: it
  averages the already-running-mean metrics a second time (`keras/src/utils/progbar.py`, no
  `stateful_metrics`), most in a fast-learning first epoch (run 1: bar 0.3152 vs true 0.3709
  accuracy). Its validation numbers are exact. The CSV and the log line are the reference.
- **`best_model.keras` vs `final_model.keras`** differ whenever the last epoch is not the best.
  `model_loading_validated` reports whether `final_model.keras` reloads and reproduces its
  predictions; `best_checkpoint_max_abs_diff` is the gap between the reloaded best checkpoint
  and the in-memory best weights on the test set (a value above `1e-6` is logged as a bug
  signal).
- **Figures and `model_analysis/` describe the best weights.** The figures are drawn from
  probabilities (softmax of the logits). Each figure is isolated: a failure is logged and listed
  under `visualizations.failed`, and the others still render.
- **`run.log`** holds the `dl` logger output of the run (devices, geometry, LR line, the
  untrained-loss line, test results, analyzer status). The handler is attached for the run and
  removed when `train()` returns or raises. Keras' own progress bars are not in it.
- **The analyzer status is read back from disk** (`model_analysis/analysis_results.json`),
  because `run_model_analysis` logs "completed successfully" even when its evaluation failed.
  The analyzer's accuracy and calibration numbers use the first 1000 test samples; the top-level
  `ece` uses the full test set.

### `results_summary.json` keys

Written with `allow_nan=False`: strict JSON, non-finite values become `null`. Keys present on
EVERY summary (also `diverged`), from `_summary_head`:

| Key | Meaning |
|---|---|
| `run_dir`, `experiment_name`, `model_family`, `dataset`, `variant` | Identity of the run. |
| `params`, `depths`, `dims` | Parameter count and the variant's structure. |
| `strides`, `kernel_size`, `drop_path_rate`, `stochastic_mode`, `dropout_rate`, `use_gamma` | The model configuration actually used (the rates are the resolved per-dataset values). |
| `input_shape`, `num_classes` | Data geometry. |
| `stage_feature_map_sizes` | Spatial size of the map each stage runs on, stage 0 first. |
| `optimizer`, `gradient_clip_norm`, `learning_rate`, `lr_schedule`, `warmup_epochs`, `steps_per_epoch`, `weight_decay`, `batch_size`, `seed` | Optimization. |
| `validation_split`, `max_samples`, `n_train`, `n_val`, `n_test`, `input_normalization` | Data split sizes and the mean / std applied. |
| `epochs_requested`, `monitor` | Requested epochs; the monitored metric (`val_loss`). |
| `initial_loss_sanity_eval`, `initial_loss_ratio`, `init_scale_warning` | The pre-fit guard: loss on the validation split (with `n_samples`, `split`, `before_fit`), its ratio to `ln(C)`, and whether the ratio exceeded 10. |
| `gpu_name`, `tf_visible_devices`, `cuda_visible_devices` | The device TensorFlow used and the environment at call time. |

Keys added by a finished run (`status: "ok"`):

| Key | Meaning |
|---|---|
| `status` | `"ok"` or `"diverged"`. |
| `epochs_run`, `stopped_early` | Epochs actually trained; whether `EarlyStopping` ended the run. |
| `best_epoch`, `best_epoch_csv_index`, `final_is_best` | 1-based best epoch by `val_loss`; the 0-based twin; whether the last epoch is the best. |
| `lr_first_epoch`, `lr_last_epoch` | The CSV `lr` of the first and last epoch: the rate at the START of each (`lr_first_epoch` is the configured base rate, or the warmup start value under warmup). |
| `best_val_metrics`, `final_val_metrics` | Validation metrics of the best epoch and of the last epoch. |
| `test_metrics_best`, `test_metrics_final` | Test metrics of the reloaded `best_model.keras` and of the last epoch's weights (`final_model.keras`). |
| `best_checkpoint_load_error`, `best_checkpoint_max_abs_diff` | Whether the best checkpoint loaded and its gap to the in-memory best weights. |
| `epoch_times`, `fit_wall_seconds` | Seconds per epoch and the wall time of `fit`; the difference is time outside the epoch clock (dashboard redraws, checkpoint saves, train-end restore). |
| `ece` | Expected calibration error on the FULL test set. |
| `visualizations` | The files written, the ECE, and `failed` (empty on a healthy run). |
| `model_loading_validated` | Result of reloading `final_model.keras` and comparing predictions. |
| `analyzer` | Status, loss, accuracy, error and path of the `ModelAnalyzer` run, read back from disk. |
| `notes` | Plain-language caveats for the reader of the file. |

A `diverged` summary carries the shared keys plus `status`, `epochs_run`, `stopped_early`
(`null`: `TerminateOnNaN` ended the run, not `EarlyStopping`), `best_epoch` (`null`),
`non_finite_metrics`, `history` (with `null`s), `epoch_times`, `fit_wall_seconds` and `notes`.

---

## Stochastic-mode comparison driver

`run_stochastic_comparison.py` trains the production trainer once per mode, serially, under an
identical seed, and reports the pair with `train.common.compare_runs`.

```bash
# 2-stage cifar10 variant, quick
MPLBACKEND=Agg .venv/bin/python -m train.convnext.run_stochastic_comparison \
    --model v1 --dataset cifar10 --variant cifar10 --epochs 30 --gpu 0

# 4-stage variant at strides 2
MPLBACKEND=Agg .venv/bin/python -m train.convnext.run_stochastic_comparison \
    --model v1 --dataset cifar10 --variant tiny --strides 2 --epochs 100 --gpu 0
```

| Flag | Default | Meaning |
|---|---|---|
| `--model {v1,v2}` | `v1` | Which trainer to launch (`train.convnext.train_convnext_<model>`). |
| `--variant` | `cifar10` | Forwarded. |
| `--dataset` | `cifar10` | Forwarded. |
| `--epochs` | `5` | Forwarded. |
| `--batch-size` | `64` | Forwarded. |
| `--seed` | `42` | Forwarded; the same for both modes. |
| `--gpu` | `0` | GPU index handed to each child as `CUDA_VISIBLE_DEVICES`; a negative value hides every GPU (CPU training). |
| `--strides` | `4` | Forwarded. |
| `--kernel-size` | `7` | Forwarded. |
| `--output-dir` | `results` | Output root of the two runs and the comparison (anchored at the repo root by the trainer when relative). |
| `--max-samples` | none | Forwarded only when given. |
| `--modes A B` | `depth gradient` | The two `stochastic_mode` values, in order. |

Behaviour:

- Each mode is a subprocess with an explicit `--experiment-name` `<stem>_<mode>`, where
  `<stem>` is `convnext_<model>_stochastic_<dataset>_<variant>_<timestamp>` (one timestamp per
  invocation). The driver reads `<output-dir>/<name>/results_summary.json` directly and requires
  `status: "ok"`; there is no snapshot-diffing of `results/`. A name that already holds a run is
  refused by the trainer, so a rerun never overwrites an earlier comparison.
- The comparison lands in `<output-dir>/<stem>_compare/` (`comparison.md`, `loss_curves.png`,
  `metric_curves.png`).
- **The driver itself is CPU-only by design.** It sets `CUDA_VISIBLE_DEVICES=''` before it
  imports TensorFlow (via `compare_runs`): a second GPU context on the trainer's GPU can
  fragment the XLA allocator and abort the trainer (`Check failed: h != kInvalidChunkHandle`).
  Each child gets its GPU through its own environment. A few harmless `cuInit: NO_DEVICE` log
  lines from the driver are expected.
- It does not forward learning rate, weight decay, schedule or patience, and it has no flag for
  `--epoch-analysis` (the trainer's default, off, applies).

---

## The MAE trainer (`train_convnext_v2_mae.py`): NOT normalized

`train_convnext_v2_mae.py` (MAE self-supervised pretraining, then two fine-tuning stages) was
left untouched by the audit that normalized V1/V2. It is out of scope of every contract above:
none of the run-directory, summary, validation-split or schedule statements in this file applies
to it. Known issues, read off the code:

- **Test-set validation.** `validation_data=(x_test, ...)` drives early stopping and the
  checkpoints in all three `fit` calls (pretraining and both fine-tuning stages), so the
  reported test accuracy was selected on itself.
- **No run-directory contract.** No `--output-dir` / `--experiment-name`, no `config.json`, no
  `results_summary.json`, no reused-name refusal; the directory is
  `results/mae_convnext_<dataset>_<variant>_<timestamp>` (CWD-relative), and the callbacks nest
  extra timestamped directories inside it. The summary is free text (`training_summary.txt`).
- **Analyzer default on.** `create_callbacks` is called without `include_analyzer=False`, so the
  per-epoch `EpochAnalyzerCallback` runs on the MAE model and on both classifier stages.
- `--run-analysis` is `store_true` with `default=True`, so it is a no-op; only `--no-analysis`
  does anything. `--gpu` comes from the shared base parser.
- The fine-tune classifier calls the encoder with `training=False`, so the "unfrozen" stage
  trains with drop path and dropout disabled.

Use it as research code; do not read its numbers as comparable to the V1/V2 trainers.

---

## Tests

Scoped to what this package touches (never run the full suite; see the repository `CLAUDE.md`):

```bash
# the ConvNeXt trainer: CLI contract, one real tiny end-to-end run, schedule / split / geometry guards
.venv/bin/python -m pytest tests/test_train/test_convnext/ -q

# every TrainingConfig field is read by its trainer (covers this trainer's config)
.venv/bin/python -m pytest tests/test_train/test_config_fields_are_live.py -q

# shared helpers the orchestrator uses (run_artifacts.py, classification_viz.py)
.venv/bin/python -m pytest tests/test_train/test_power_mlp/ -q
```

`test_cli_contract.py` pins every flag to its config field (once per family), the completeness
of that table against the real parser, `--gpu` into `setup_gpu`, the variant choices against
`MODEL_VARIANTS` and the `imagenet` refusal. `test_train_convnext.py` runs one tiny real
training and checks the artifact set, the strict-JSON summary, that the CSV `lr` column is a live
cosine, that the figures are fed softmax probabilities, that `final_model.keras` is the last
epoch and differs from the best one, the reused-name refusal, the 100-class top-5 metric, the
seeded disjoint validation split and the geometry helper. The tests write only under
`tmp_path` and use synthetic data injected through `common.load_dataset`.

---

## Measured results

Every number below is copied from a `results_summary.json` (or the run's `training_log.csv`) written
by the trainer. `results/` is untracked, so the run directory is named for each row. The code state
is named too: numbers from earlier code states are kept only when labelled as such.

| Run | Command (GPU 0, RTX 4090, seed 42) | Test acc | Test loss | ECE | Epoch times (s) | Notes |
|---|---|---|---|---|---|---|
| `convnext_v1_cifar10_cifar10_iter1_run1` | `train_convnext_v1 --dataset cifar10 --variant cifar10 --epochs 5` (2.23M params, strides 4, feature maps 8x8 then 2x2, batch 64, cosine 1e-3) | 0.6148 | 1.0911 | 0.0124 | 84.0, 13.1, 11.9, 12.2, 12.4 | Code state of the iteration-1 audit. Val acc 0.6078. Best epoch = last epoch (5), loss still falling. Wall 198 s: fit 138 s, post-fit 41 s. |

Reading the epoch times: epoch 1 is much slower than the rest because the XLA-compiled train step
is built on the first batch, and it is not step time (steady state is 17 to 18 ms per step at batch
64), so per-epoch cost comparisons use epochs 2 onward. The run-1 row above predates the fix
described next.

**First epoch (measured, GPU 0, 2-epoch probes on the full CIFAR-10 data, cifar10 variant, batch
64, seed 42, `findings/iter2-f1-epoch1.md`).** Keras compiles the train step with XLA by default
here. The first epoch cost 80.5 s against 12.1 s for the second, in two parts: 47.5 s before the
first train step returned (the XLA compile of the whole step) and 20.7 s at the very last step of
the epoch, when the incomplete 8-sample batch (45000 % 64) forced a second compile. The train
pipeline therefore drops the incomplete last batch (`steps_per_epoch = n_train // batch_size`, 703
here; the pool is reshuffled every epoch so no sample is permanently left out), which brought epoch
1 to 58.0 s. What remains is the one-time compile of the first step, about 45 s, and it is a fixed
cost of the run, not a per-epoch one. Ruled out by measurement: the input pipeline (0.4 s to the
first batch on CPU, 0.6 s per epoch), the end-of-epoch validation (0.2 s), cuDNN autotuning
(`TF_CUDNN_USE_AUTOTUNE=0` and `XLA_FLAGS=--xla_gpu_autotune_level=0` left the first step at 46.5 s
and 44.8 s), and turning XLA off (`jit_compile=False`: 30 s to the first step but 89 ms per step
for the whole run, five times slower). GPU utilization is about 1 percent during the compile and
15 to 20 percent in steady state (a small model at batch 64 is launch bound, not compute bound).

The console progress bar's train metrics are a second average of the running mean and read
lower than `training_log.csv` (epoch 1: bar 0.3152 vs CSV 0.3709 accuracy); the CSV, the summary and
the dashboard are the reference.
