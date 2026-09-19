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

# V1, CIFAR-10, the 2-stage cifar10 variant, 100 epochs (the defaults, strides 2)
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.convnext.train_convnext_v1 \
    --dataset cifar10 --variant cifar10

# V2, CIFAR-100 (drop path and dropout default to 0.2 for this dataset), named run
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.convnext.train_convnext_v2 \
    --dataset cifar100 --variant cifar10 --epochs 100 --experiment-name my_c100_v2

# A 4-stage variant: the default strides 2 gives the standard 32 -> 16 -> 8 -> 4 -> 2 schedule
# (see "Geometry" and "Choosing strides")
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.convnext.train_convnext_v1 \
    --dataset cifar10 --variant tiny

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
| `--max-samples` | none | Cap the train pool and the test set at this many samples (smoke runs). Refused at config time, before any run directory exists, unless the resulting train split (after the `--validation-split` cut, which must hold out at least one sample) holds at least one full `--batch-size` batch: `--max-samples 64` with the default batch 64 and split 0.1 leaves 58 train samples and is refused; lower `--batch-size` or raise `--max-samples`. |
| `--variant` | `cifar10` | V1: `cifar10, tiny, small, base, large, xlarge`. V2: `cifar10, atto, femto, pico, nano, tiny, base, large, huge`. Sets depths and dims. |
| `--kernel-size` | `7` | Depthwise convolution kernel size. |
| `--strides` | `2` | Stem patch size AND every inter-stage downsample stride (one knob). The default is 2 on measured evidence ("Choosing strides"); the model classes keep their own default 4. See "Geometry". |
| `--drop-path-rate` | per dataset | Maximum stochastic-depth rate. `0.1` for mnist and cifar10, `0.2` for cifar100. |
| `--stochastic-mode {depth,gradient}` | `depth` | `depth` = `StochasticDepth`, `gradient` = `StochasticGradient` (see below). |
| `--dropout-rate` | per dataset | Dropout inside each block. `0.1` for mnist and cifar10, `0.2` for cifar100. |
| `--use-gamma` / `--no-use-gamma` | on | Learnable per-channel layer scale in each block. |
| `--epochs` | `100` | Maximum epochs. The cosine spans exactly this many. |
| `--batch-size` | `64` | Training batch size. |
| `--learning-rate` | `0.001` | Peak learning rate. |
| `--weight-decay` | `0.0001` | Decoupled AdamW weight decay (never combined with an L2 regularizer). |
| `--label-smoothing` | `0.0` | Opt-in label smoothing `a` in `[0, 1)`; anything else is refused at config time. `0.0` keeps the stock loss; above 0 see "Label smoothing" under Optimization. |
| `--lr-schedule {cosine,exponential,constant}` | `cosine` | Schedule over the whole run. `constant` passes a plain float and adds `ReduceLROnPlateau` (factor 0.5, patience 5, on `val_loss`). |
| `--warmup-epochs` | `0` | Linear warmup epochs before the cosine. Only valid with `cosine` and strictly below `--epochs`; otherwise refused at config time. |
| `--patience` | `50` | Early-stopping patience in epochs, on `val_loss`. |
| `--seed` | `42` | Seed for weights, shuffling, augmentation and the splits. |
| `--epoch-analysis` | off | Also run the per-epoch `ModelAnalyzer` callback into `epoch_analysis/`. Independent of `--model-analysis`. |
| `--model-analysis` / `--no-model-analysis` | on | The end-of-run `ModelAnalyzer` into `model_analysis/`. `--no-model-analysis` skips it and the summary records `analyzer.status = "skipped"`. See "The end-of-run analysis". |
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
- Loss `SparseCategoricalCrossentropy(from_logits=True)` (unless `--label-smoothing` is above
  0, next bullet): the V1/V2 head is a bare `Dense(num_classes)` and emits logits. Metrics are `accuracy`, plus `top_5_accuracy` (a metric
  object, not the unresolvable string alias) when there are more than 10 classes.
- **Label smoothing (opt-in, `--label-smoothing a`, default 0.0).** Trains against a softened
  target instead of a hard one, which discourages over-confident logits. The convention is the
  one of `keras.losses.CategoricalCrossentropy(label_smoothing=a)`: the target of a sample of
  class `y` over `C` classes is `onehot(y) * (1 - a) + a / C`, so every class, the true one
  included, gets `a / C`, and `loss = (1 - a) * CE(y) + a * mean over classes of (-log_softmax)`.
  With `a = 0` the loss is exactly the stock `SparseCategoricalCrossentropy(from_logits=True)`
  object and nothing else changes. Above 0 it is `SmoothedSparseCategoricalCrossentropy`
  (in `common.py`, registered and serializable, sparse labels and logits, XLA-compatible), so the
  pipeline, the sparse `accuracy` / `top_5_accuracy` metrics and every figure are untouched, and
  `best_model.keras` / `final_model.keras` reload with the compile state. The summary records
  `label_smoothing` and, above 0, a note: `val_loss`, `test_loss` and the training loss include
  the smoothing (a smoothed target cannot be fit to zero loss, so the loss floor is above 0) and
  are **not comparable with unsmoothed rows**, while accuracy, top-5 and ECE keep their
  definitions. The initial-loss guard stays valid: uniform logits give `ln(C)` for any `a`.
  No effect of smoothing is claimed here; measured rows appear in "Measured results" only after
  runs 4a and 4b.
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
| 32x32 | 4 | 2 (default) | 16, 8, 4, 2 |
| 32x32 | 4 | 4 | 8, 2, 1, 1 |
| 32x32 | 2 (`cifar10` variant) | 2 (default) | 16, 8 |
| 32x32 | 2 (`cifar10` variant) | 4 | 8, 2 |
| 28x28 (MNIST) | 4 | 2 (default) | 14, 7, 4, 2 |
| 28x28 (MNIST) | 2 (`cifar10` variant) | 2 (default) | 14, 7 |
| 28x28 (MNIST) | 4 | 4 | 7, 2, 1, 1 |

`--strides 4` builds and trains on 32x32 four-stage models (there is no crash), but the last two
stages run on a single pixel, so the spatial mixing of the depthwise convolution there is
degenerate. The default `--strides 2` gives the usual `32 -> 16 -> 8 -> 4 -> 2` schedule for
`tiny/small/base/...` and no stage below 2x2 on any of the three datasets. Every summary records `stage_feature_map_sizes` (computed by
`stage_feature_map_sizes`, which a test checks against the real `include_top=False` model) and
the run log prints it, so a degenerate geometry is visible in the record of the run.

### Choosing strides

The default changed from 4 to 2 on one paired measurement (CIFAR-10, `cifar10` variant, V1,
5 epochs, seed 42, batch 64, cosine 1e-3, GPU 0):

| Run directory | `--strides` | Feature maps | Params | Test acc | Test loss | Steady epoch |
|---|---|---|---|---|---|---|
| `results/convnext_v1_cifar10_cifar10_iter1_run1` | 4 | 8x8, 2x2 | 2,229,226 | 0.6148 | 1.0911 | about 12.4 s |
| `results/convnext_v1_cifar10_cifar10_strides2_iter2_run2a` | 2 | 16x16, 8x8 | 2,004,586 | 0.7206 | 0.8134 | about 13.9 s |

That is +10.6 accuracy points and -0.28 test loss for about 12 percent more time per steady
epoch (epochs 2 to 5, mean) and 10 percent fewer parameters. It is a single seed per arm; seed
noise is measured separately (the run-3a repeat of the strides-2 arm), and a gap this size is far
above any plausible seed spread but is not a statement about differences of a point or two. The two
runs were measured at slightly different code states (run 1 before the LR log point moved, the
`drop_remainder` pipeline change and the step count 704 to 703; run 2a after them); none of those
changes what is trained beyond 8 dropped samples per epoch, so the pairing holds to that extent.
What is not measured: 4-stage variants (only the `cifar10` variant was paired), CIFAR-100, MNIST,
and longer runs. Pass `--strides 4` to reproduce the old default.

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
        confusion_matrix.png         up to 20 classes: counts + row-normalized; more: heatmap + the 15 most confused pairs (bounded size)
        per_class_metrics.png        up to 20 classes: grouped bars; more: the 15 worst and 15 best classes by F1 (the full table is in classification_report.json)
        confidence_calibration.png   reliability diagram with ECE
        misclassifications.png       the most confident errors
        classification_report.json
    model_analysis/             ModelAnalyzer output of the final analysis (analysis_results.json, ...); absent with --no-model-analysis
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
  With `--no-model-analysis` there is nothing to read and the status is `"skipped"`.
- **The end-of-run analysis** (`model_analysis/`, on by default, `--no-model-analysis` to skip)
  costs about 24 s of a 198 s CIFAR-10 5-epoch run (seven PNGs), so a short run or a sweep
  usually wants it off. Read it with three caveats. Its spectral (WeightWatcher) verdicts such
  as "overfit / over-trained" are heuristics that read a 5-epoch model as over-trained, and they
  are unreliable for short runs and for depthwise kernels. Its `Final Acc` and ECE come from
  the first 1000 test samples and differ from the summary's full-test-set `test_metrics_*` and
  `ece`; the `Final Acc` is not the last epoch's accuracy. `summary_dashboard.png` can carry an
  empty "No weight PCA data available" panel. Three more label facts, all library output that
  this trainer does not change: `training_dynamics.png` counts "Best Epoch" from 0 (the summary's
  `best_epoch` is 1-based); "Final Acc" is the first-1000-test-sample accuracy in
  `summary_dashboard.png` but the validation accuracy in `training_dynamics.png` (so it can read
  below "Best Acc" although the last epoch is the best); and the panels of one analyzer figure can
  show different layer counts (`information_flow_analysis.png` in runs 2a and 2b: 10 layers on top, 8 below). The
  summary's `notes` repeat the caveats when the analysis ran, and say it was skipped when it did
  not.

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
| `optimizer`, `gradient_clip_norm`, `learning_rate`, `lr_schedule`, `warmup_epochs`, `steps_per_epoch`, `weight_decay`, `label_smoothing`, `batch_size`, `seed` | Optimization (`label_smoothing` is 0.0 for the stock loss). |
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
| `visualizations` | `files`: the post-fit figure and report files that exist on disk (the dashboard is written by the training callback and is not listed; a perfect classifier writes no `misclassifications.png`, and the name is then absent, not listed), the ECE, and `failed` (figures that raised; empty on a healthy run). |
| `lr_reduction_epochs` | 1-based epochs trained at a lower rate than the epoch before, from the `lr` history: only meaningful under `--lr-schedule constant` (which adds `ReduceLROnPlateau`), where an empty list means no plateau reduction; `null` under `cosine` and `exponential`, where the rate falls every epoch by design and `ReduceLROnPlateau` is not installed. |
| `model_loading_validated` | Result of reloading `final_model.keras` and comparing predictions. |
| `analyzer` | Status, loss, accuracy, error and path of the `ModelAnalyzer` run, read back from disk; `{"status": "skipped", ...}` with `null` for the rest under `--no-model-analysis`. |
| `notes` | Plain-language caveats for the reader of the file. |

A `diverged` summary carries the shared keys plus `status`, `epochs_run`, `stopped_early`
(`null`: `TerminateOnNaN` ended the run, not `EarlyStopping`), `best_epoch` (`null`),
`non_finite_metrics`, `lr_reduction_epochs`, `history` (with `null`s), `epoch_times`, `fit_wall_seconds` and `notes`.

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
| `--strides` | none | Forwarded only when given, so the trainer default (2) is the one source. |
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
- `--strides` still defaults to 4 here, unlike the normalized trainers (default 2, measured
  10.6 points better on CIFAR-10; see "Choosing strides"), so its geometry and its numbers are
  not comparable to theirs at default flags.

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

| Run | Command (GPU 0, RTX 4090, seed 42 unless the command says otherwise) | Test acc | Test loss | ECE | Epoch times (s) | Notes |
|---|---|---|---|---|---|---|
| `convnext_v1_cifar10_cifar10_iter1_run1` | `train_convnext_v1 --dataset cifar10 --variant cifar10 --epochs 5` (2.23M params, strides 4, feature maps 8x8 then 2x2, batch 64, cosine 1e-3) | 0.6148 | 1.0911 | 0.0124 | 84.0, 13.1, 11.9, 12.2, 12.4 | Code state of the iteration-1 audit. Val acc 0.6078. Best epoch = last epoch (5), loss still falling. Wall 198 s: fit 138 s, post-fit 41 s. |
| `convnext_v1_cifar10_cifar10_strides2_iter2_run2a` | `train_convnext_v1 --dataset cifar10 --variant cifar10 --epochs 5 --strides 2` (2.00M params, 2,004,586, feature maps 16x16 then 8x8, depths 5, 5, dims 96, 192, batch 64, cosine 1e-3, 703 steps per epoch) | 0.7206 | 0.8134 | 0.0136 | 63.7, 13.6, 13.7, 14.0, 14.3 | Code state HEAD after `25e4afc84`. Val acc 0.7154, fit 124.0 s. Best epoch = last epoch (5). Paired with run 1 in "Choosing strides". |
| `convnext_v2_cifar10_cifar10_strides2_iter2_run2b` | `train_convnext_v2 --dataset cifar10 --variant cifar10 --epochs 5 --strides 2` (2.02M params, 2,016,106, feature maps 16x16 then 8x8, depths 5, 5, dims 96, 192, batch 64, cosine 1e-3, 703 steps per epoch) | 0.7098 | 0.8386 | 0.0084 | 73.8, 14.7, 15.1, 15.3, 15.3 | Code state HEAD after `25e4afc84`. Val acc 0.7136, fit 139.0 s. Best epoch = last epoch (5). The 1.1-point gap to 2a is one seed each and not attributable to V1 versus V2. |
| `convnext_v1_cifar10_cifar10_seed43_iter3_run3a` | `train_convnext_v1 --variant cifar10 --dataset cifar10 --epochs 5 --seed 43 (default strides 2)` | 0.7255 | 0.8009 | 0.0117 | 63.9, 13.7, 14.4, 14.3, 14.5 | Seed 43. Against 2a (seed 42) only the seed differs: 0.5 points test accuracy, 1.2 points val accuracy. Code state HEAD after 21fe4af47. |
| `convnext_v1_cifar10_cifar10_gradient_iter3_run3c` | `train_convnext_v1 --variant cifar10 --dataset cifar10 --epochs 5 --stochastic-mode gradient --no-model-analysis` | 0.7052 | 0.8452 | 0.0169 | 64.8, 15.7, 15.7, 15.7, 15.7 | Gradient mode: 1.8 points below the mean of the two depth-mode seeds (0.7231), about 3.6 times the seed spread of 0.49, and about 12% slower steady epochs. Code state HEAD after 21fe4af47. |
| `convnext_v1_cifar10_cifar10_bs256_iter3_run3d` | `train_convnext_v1 --variant cifar10 --dataset cifar10 --epochs 5 --batch-size 256 --learning-rate 2e-3 --no-model-analysis` | 0.6677 | 0.9505 | 0.0136 | 56.8, 7.6, 7.5, 7.7, 7.7 | 175 steps per epoch: steady epochs 1.85x faster, 5.5 points worse at equal epochs; fit wall time only 27% shorter because about 50 s is the fixed compile. Code state HEAD after 21fe4af47. |
| `convnext_v1_cifar10_cifar10_warmup1_iter3_run3e` | `train_convnext_v1 --variant cifar10 --dataset cifar10 --epochs 5 --warmup-epochs 1 --no-model-analysis` | 0.7150 | 0.8197 | 0.0165 | 73.1, 14.0, 14.0, 14.5, 14.2 | One warmup epoch (lr at epoch start 1e-8, 1e-3, 8.55e-4, 5.05e-4, 1.55e-4). Within noise of the no-warmup runs; epoch 1 about 9 s slower. Code state HEAD after 21fe4af47. |
| `convnext_v1_cifar10_cifar10_warmup1_iter3_run3e2` | `same command as run 3e, run again after the figure fixes` | 0.7208 | 0.8108 | 0.0120 | 73.6, 14.1, 14.2, 14.4, 14.3 | Same seed and command as 3e: 0.7208 vs 0.7150, so identical-seed GPU runs differ by about 0.6 points. |
| `convnext_v1_cifar100_cifar10_iter3_run3b` | `train_convnext_v1 --variant cifar10 --dataset cifar100 --epochs 5` | 0.3905 (top-5 0.6921) | 2.4209 | 0.0552 | 65.1, 14.3, 14.7, 14.9, 14.3 | CIFAR-100, 5 epochs, before the 100-class figure fixes (its per-class and confusion figures were unreadable). |
| `convnext_v1_cifar100_cifar10_iter3_run3b2` | `same command as run 3b, run again after the figure fixes` | 0.3851 (top-5 0.6916) | 2.4093 | 0.0436 | 65.2, 14.5, 14.8, 14.6, 14.8 | Same seed and command as 3b: 0.3851 vs 0.3905. Figures readable (worst/best 15 classes, confusion pairs, CIFAR-100 names). |

Seed and run-to-run noise: two runs that differ only in the seed (2a vs 3a) differ by 0.5 points of test accuracy, and two runs with the identical seed and command (3b vs 3b2, 3e vs 3e2) by 0.5 to 0.6 points (GPU non-determinism). The identical-seed re-runs also moved the ECE (3b 0.0552 vs 3b2 0.0436), so ECE differences of about 1 point are not attributable either. Differences below about 1 point in this table are not attributable. The strides effect (run 1 vs 2a, 10.6 points) and the batch-256 loss (5.5 points) are far outside that band; the V1 vs V2 gap (1.1 points) and warmup are not.

Reading the epoch times: epoch 1 is much slower than the rest because the XLA-compiled train step
is built on the first batch, and it is not step time (steady state is 17 to 18 ms per step at batch
64), so per-epoch cost comparisons use epochs 2 onward. Only the run-1 row predates the fix described
next (it used 704 steps per epoch); every later row uses 703.

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
