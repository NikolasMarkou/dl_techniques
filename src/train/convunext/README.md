# convunext - ConvUNext semantic-segmentation training (Oxford-IIIT Pet, 3 classes)

Trains `create_convunext` (`use_bias=True`, `output_channels=3`, a linear head that emits logits)
as a per-pixel classifier on Oxford-IIIT Pet: pet / background / border. One entry point,
`train_convunext_segmentation.py`; the orchestrator is `common.py`, the figures and the
per-epoch grid callback are `segmentation_viz.py`.

## How it relates to its neighbours

- **`src/train/convnext/` (sibling).** Same run-directory contract and the same shared pieces
  from `src/train/common/` (run directory and log, `LearningRateLogger`, `EpochLogLine`,
  `TrainingDashboardCallback`, summary helpers, `run_model_analysis`), but a separate package:
  the ConvNeXt trainer threads classification through about thirty places (dataset enum,
  per-image labels, `ln(C)` initial-loss check, calibration and misclassification figures), so a
  task switch would be an `if` ladder through a trainer that has its own audit history. The
  two share the split arithmetic and constants by import (`split_sizes`, `steps_per_epoch_for`,
  `MONITOR`, `GRADIENT_CLIP_NORM`), not by copy. It does not reuse the classification figures,
  which read probabilities per image.
- **`src/train/bfunet/` (the denoiser).** The bias-free ConvUNeXt DENOISER (`use_bias=False`,
  noisy image in, clean image out) is trained by `src/train/bfunet/train_convunext_denoiser.py`
  and stays there; it shares a `common.py` with the unet and bfcnn denoisers, which is why it
  was not moved. This package is the biased segmentation arm of the same model.
- **The model** is `src/dl_techniques/models/vision/convunext/model.py`; its README is next to it.

## Quick start

Repo root, `.venv` active, GPU chosen with `CUDA_VISIBLE_DEVICES` (see "GPU selection"). The
dataset is read from the local TFDS cache with `download=False`; set `TFDS_DATA_DIR` to the
directory that holds `oxford_iiit_pet/4.0.0` if it is not your default.

```bash
# The defaults: variant tiny, 128 px, 30 epochs, batch 16, whole train split
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.convunext.train_convunext_segmentation

# A named short run: 2 epochs on 1600 train images
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.convunext.train_convunext_segmentation \
    --variant tiny --image-size 128 --epochs 2 --batch-size 16 --max-samples 1600 \
    --experiment-name my_segmentation_run

# Smoke run: 64 train images at 64 px, 2 epochs, a grid after every epoch
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.convunext.train_convunext_segmentation \
    --variant tiny --image-size 64 --epochs 2 --max-samples 64 --batch-size 8 --viz-freq 1 \
    --experiment-name convunext_seg_smoke

# The flag list is the source of truth for names and defaults
.venv/bin/python -m train.convunext.train_convunext_segmentation --help
```

A reused `--experiment-name` is refused before anything is written. `--help` allocates no GPU
and no directory.

## Command-line flags

Defaults are read off `SegTrainingConfig` (the parser cannot drift from the config; a test
pins each one) and were checked against `--help`.

| Flag | Default | Meaning |
|---|---|---|
| `--variant` | `tiny` | ConvUNext variant: `tiny`, `small`, `base`, `large`, `xlarge` (the keys of `CONVUNEXT_CONFIGS`). |
| `--image-size` | `128` | Square side images and masks are resized to. Refused below `2 ** depth` of the variant (tiny and small 8, base and large 16, xlarge 32); odd sizes are fine. |
| `--validation-split` | `0.1` | Fraction of the TRAIN split held out (seeded) for early stopping and checkpoint selection, strictly inside (0, 1). |
| `--max-samples` | none (whole train split) | Caps the train pool (fit and validation together). The test split is never capped. |
| `--epochs` | `30` | Maximum epochs; the cosine schedule spans this many. |
| `--batch-size` | `16` | Training batch size; the fit split must hold at least one full batch (refused at config time otherwise). |
| `--learning-rate` | `0.001` | Peak learning rate. |
| `--weight-decay` | `0.0001` | Decoupled AdamW weight decay. |
| `--warmup-epochs` | `0` | Linear warmup epochs before the cosine; smaller than `--epochs`. |
| `--patience` | `10` | Early-stopping patience in epochs on `val_loss`. |
| `--seed` | `42` | Seeds weights, shuffling, augmentation, the validation split and the grid's sample choice. |
| `--viz-freq` | `1` | Write the segmentation grid every this many epochs (the last epoch is always written). |
| `--viz-samples` | `4` | Validation images in each grid. |
| `--model-analysis` / `--no-model-analysis` | on | End-of-run ModelAnalyzer, weights and spectral analyses only, into `model_analysis/`. |
| `--output-dir` | `results` | Output root; a relative path is anchored at the repo root. |
| `--experiment-name` | `convunext_seg_<variant>_<timestamp>` | Run directory name. |
| `--gpu` | none | Sets `CUDA_VISIBLE_DEVICES` and overrides an exported value. Not a config field. |

Fixed, not flags: optimizer AdamW with `clipnorm=1.0`, cosine schedule, loss
`SparseCategoricalCrossentropy(from_logits=True)`, monitor `val_loss`, random horizontal flip
(image and mask together) as the only augmentation, deep supervision off. The training defaults are
the ConvNeXt trainer's and have NOT been tuned for this task.

## Data

- **Oxford-IIIT Pet 4.x** from the local TFDS cache (`oxford_iiit_pet:4.*.*`,
  `download=False`); the trainer never downloads. 3680 train and 3669 test images. If the cache
  is missing or the wrong version, `load_oxford_pet` raises a `RuntimeError` naming
  `TFDS_DATA_DIR` before any run directory exists.
- **3 classes**: TFDS mask values 1, 2, 3 minus 1 give 0 `pet`, 1 `background`, 2 `border`.
- **Resize** to `--image-size` squares: images bilinear with antialiasing, masks NEAREST (no
  invented class value), stored as uint8 arrays in memory and scaled to [0, 1] in the pipeline.
- **Split**: a seeded slice of the TRAIN split is the validation set (drives early stopping,
  checkpoint selection and the grids); the rest is the fit split. The TEST split is read only
  after `fit` and selects nothing. `--max-samples` caps the train pool only, so test numbers
  are comparable across runs.
- The train pipeline drops the incomplete last batch and reshuffles every epoch. The horizontal
  flip is a STATELESS draw keyed by `(seed, position in the epoch's shuffled order)`, so the
  batches (images and masks) of every epoch are reproducible from `--seed` whatever order the
  parallel map workers run in; a stateful `tf.random.uniform` in that map assigned the flips in
  worker-scheduling order and changed the batches between two same-seed runs (measured on random
  1440 x 128 px arrays with the global and shuffle seed both fixed: at least 39 of 90 batches
  differed between two builds of the old pipeline, compared only on each batch's first sample).

## What mIoU means here

`mIoU` is the mean over classes of `TP / (true + predicted - TP)` computed from a pixel
confusion matrix; a class with no true and no predicted pixel has no IoU and is left out of the
mean (the `keras.metrics.MeanIoU` convention). The summary carries two independent measurements
of it, `miou` and `miou_from_confusion`. A TEST block is ONE pass over the split (the loss is the
compiled loss of the predicted logits, `accuracy` and `miou` are read off the confusion matrix, so
they equal `pixel_accuracy` and `miou_from_confusion`); `test_the_one_pass_evaluation_equals_keras_evaluate`
compares them with Keras' `evaluate` to 1e-5. The validation metrics are Keras' own.

Read a mIoU next to `trivial_baseline`: the scores of predicting the majority class of the
test masks for every pixel (its pixel accuracy is that class's share of pixels, its mIoU that
share divided by 3). A segmenter that does not beat it, or that predicts one class, has learned
nothing about the image. `accuracy` is pixel accuracy and is dominated by the large classes;
per-class IoU (and the `border` class in particular, a thin band) is the honest view.

## GPU selection

`CUDA_VISIBLE_DEVICES` exported in the shell is the supported control. `--gpu N` also works
(`main()` parses first, then calls `setup_gpu(gpu_id=N)`, which sets the variable before
TensorFlow first enumerates devices and overrides an exported value); do not pass it together
with an exported variable and do not rely on it after any earlier device query in the process.
The summary stores `gpu_name`, `tf_visible_devices` and `cuda_visible_devices`, and `run.log`
names all three before data loading.

## Run-directory layout

```
results/<experiment_name>/
    config.json                     resolved SegTrainingConfig
    run.log                         the dl logger for this run
    training_log.csv                one row per epoch: epoch (0-based), accuracy, loss, lr, miou, val_* ...
    training_history.json           per-epoch lists of every history metric
    best_model.keras                checkpoint of the lowest-val_loss epoch
    final_model.keras               the LAST epoch's weights (not the best)
    results_summary.json            the run's record (strict JSON, keys below)
    visualizations/
        training_dashboard.png      shared per-epoch curves (loss, accuracy, mIoU, lr, ...), redrawn on a cadence; the best epoch (lowest val_loss so far) is a green dashed line and star on the val curves
        epoch_000_seg_grid.png      the UNTRAINED model on the fixed validation batch
        epoch_NNN_seg_grid.png      after epoch NNN (every --viz-freq epochs, and the last epoch)
        confusion_matrix.png        TEST split, best weights: pixel counts and row-normalized recall
        per_class_metrics.png       grouped bars of IoU, Dice, precision and recall per class
        segmentation_report.json    the same per-class numbers plus the confusion counts (strict JSON, null where undefined)
        best_vs_final_predictions.png   image | ground truth | best weights | final weights, on the grid's samples (NOT written when the best epoch is the last: `visualizations.skipped` says so)
        miou_curve.png              train and validation mIoU per epoch, the best epoch marked
    model_analysis/                 weights + spectral ModelAnalyzer output (analysis_results.json, PNGs); absent with --no-model-analysis
```

Grid figures show `viz_samples` validation images chosen ONCE with `numpy.random.RandomState(seed)`,
so every epoch and every run of one seed shows the same images. Rows are image | ground truth |
prediction; ground truth and prediction share one palette (pet orange, background dark blue,
border yellow) with a legend, and each prediction title carries that row's pixel accuracy. The
figure title carries the validation mIoU of the WHOLE validation set, written
`val mIoU 0.390 (all 160)`; the sheet itself shows only `--viz-samples` of those images.

Notes:

- `epoch` in the CSV is 0-based, `best_epoch` in the summary is 1-based. CSV `lr` is the rate at
  the START of the epoch.
- `final_model.keras` is the real last epoch: `EarlyStopping` is built with
  `restore_best_weights=False` (Keras 3.8 would otherwise restore the best weights at train end).
  When the best epoch is the last one the two files hold the same weights.
- `model_analysis/` describes the LAST epoch's weights (the in-memory model). Its spectral
  verdicts are heuristics and unreliable for a short run. Calibration, information flow and
  training dynamics are off: they read per-image labels and probabilities.
- `model_analysis/` holds `analysis_results.json` plus three PNGs: `spectral_summary.png`,
  `spectral_funnel_diagram.png` and `weight_learning_journey.png`. The library also writes
  `summary_dashboard.png`, which is EMPTY for a data-free analysis (every panel reads "No ...
  data available"); the run deletes that one file after an `ok` analysis and records it as
  `analyzer.removed: ["summary_dashboard.png"]`. Two smoke runs (tiny, 64 px, `convunext_seg_smoke_step4` and `convunext_seg_smoke_step6`) read `analyzer.seconds` 18.1 and 16.4.
- Against the denoiser, this trainer writes no `model_summary.txt` and no `tensorboard/`: the ConvNeXt
  reference has neither, so the segmentation trainer follows the reference (the parameter total
  is in `run.log`; the denoiser's `initial_loss_sanity_eval` carries `steps`, this
  one `n_samples`, because the denoiser's validation stream is repeated).
- `run.log` names what was written: one `Segmentation grid written: <file>` line per grid and one
  `Visualizations: N written (...), M failed[: names], K skipped[: names], grids plus end-of-run figures took S s` line after the
  end-of-run figures.
- Every figure is isolated: one that raises is recorded by name under `visualizations.failed` (a
  list, as in the ConvNeXt reference; the error text is the `Visualization <name> failed: ...`
  WARNING in `run.log`) and the run stays `status: "ok"`. The analyzer cannot fail a finished run either: an
  exception is recorded as `analyzer.status: "error"`.

### `results_summary.json` keys

Identity and setup (present for `status` `ok` and `diverged`): `status`, `run_dir`,
`experiment_name`, `model_family`, `dataset`, `variant`, `params`, the architecture keys
(below), `input_shape`, `image_range` (`[0.0, 1.0]`, the image after `/255`; named `data_range` in iteration 3), `num_classes`,
`class_names`, `optimizer`, `gradient_clip_norm`, `learning_rate`, `lr_schedule`,
`warmup_epochs`, `steps_per_epoch`, `weight_decay`, `batch_size`, `seed`, `validation_split`,
`max_samples`, `n_train`, `n_val`, `n_test`, `data_load_seconds`, `epochs_requested`, `monitor`,
`initial_loss_sanity_eval`, `initial_loss_ratio`, `init_scale_warning`, `gpu_name`,
`tf_visible_devices`, `cuda_visible_devices`, `epochs_run`, `stopped_early`, `best_epoch`,
`epoch_times`, `fit_wall_seconds`, `notes`.

| Key (status `ok` only) | Meaning |
|---|---|
| `best_epoch_csv_index` | `best_epoch - 1`, the CSV row of the best epoch. |
| `final_is_best` | Whether the last epoch is the best one. |
| `final_reused_best` | True when `final_is_best` and `best_model.keras` loaded: the test split was scored ONCE and `test_metrics_final` is the same result as `test_metrics_best` (both keys stay populated). |
| `lr_first_epoch`, `lr_last_epoch`, `lr_last_step` | CSV `lr` of the first and last epoch; the schedule's value at the final optimizer step. |
| `best_val_metrics`, `final_val_metrics` | Validation `loss`, `accuracy`, `miou` of the best and the last epoch. |
| `test_metrics_best`, `test_metrics_final` | Test-split `loss`, `accuracy`, `miou`, `miou_from_confusion`, `pixel_accuracy`, `per_class_iou`, `confusion` for the reloaded `best_model.keras` and for the last-epoch weights. |
| `trivial_baseline` | Scores of the majority-class predictor on the test masks (`predicted_class`, `miou`, `per_class_iou`, `pixel_accuracy`, `confusion`). |
| `visualizations` | `{files, failed, skipped, seconds}`: the names that exist on disk (dashboard, per-epoch grids, end-of-run figures), the list of figure names that raised, `{figure: reason}` for those deliberately not drawn (`best_vs_final_predictions.png` when the best epoch is the last), wall seconds of the per-epoch grids plus the end-of-run figures (the dashboard redraw is not timed). |
| `analyzer` | `{status, analyzers, error, path, seconds}` read back from `model_analysis/analysis_results.json`, plus `removed: ["summary_dashboard.png"]` when the empty library figure was deleted after an `ok` read-back (the key is absent otherwise): status `ok`, `partial`, `missing`, `unreadable` or `error`; `analyzers` lists which of `weights` / `spectral` wrote results. With `--no-model-analysis`: the skipped block (`status: "skipped"`). |
| `test_eval_seconds` | Wall seconds of the test evaluation (one prediction pass per scored weight set; the split is scored once when `final_reused_best`, twice otherwise) plus the baseline. |
| `best_checkpoint_load_error`, `best_checkpoint_max_abs_diff` | Whether the best checkpoint reloaded, and the largest gap between its validation metrics and what `fit` recorded for the best epoch. |
| `model_loading_validated` | Whether `final_model.keras` reloads and reproduces its predictions. |

Architecture keys, resolved by `architecture_of(variant)` the way `create_convunext_variant`
builds (the variant's row of `CONVUNEXT_CONFIGS` over `create_convunext`'s signature defaults,
then `use_bias=True`, 3 output channels, linear head), and cross-checked by a test against the
layers of the built model: `convnext_version`, `depth`, `blocks_per_level`, `dims` (channels of
the encoder levels and the bottleneck, the model's own `Filter progression`), `kernel_size` (the
block kernel), `stem_kernel_size`, `block_normalization`, `drop_path_rate` (the configured
maximum; the model spreads it linearly over the blocks, so the largest per-block rate is below
it), `dropout_rate`, `use_bias`, `final_activation`. The ConvNeXt summary's `strides`,
`use_gamma` and `stochastic_mode` are absent on purpose: this model has no such setting
(its ConvNeXt V2 blocks carry a GRN instead of a layer scale).

A diverged run (`status: "diverged"`) writes the identity block plus `non_finite_metrics` and
`history`, no evaluation, no figures beyond what training drew, no `final_model.keras`, and then
raises.

## Tests

```bash
# config, CLI contract, refusals, --help
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_train/test_convunext/test_config_and_cli.py -q
# one real tiny end-to-end run on synthetic band images (loader stubbed), artifact, split and last-epoch guards
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_train/test_convunext/test_run.py -q
# the figures and the grid callback
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_train/test_convunext/test_segmentation_viz.py -q
# the shared analysis (run and read back) and the shared dashboard with its optional best-epoch marker
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_train/test_common_run_summary.py tests/test_train/test_common_classification_viz.py -q
```

`test_run.py` is slow (several real fits). No test reads the TFDS cache or writes under the
repo-root `results/`.

## Measured runs

### Loop 1 (before the loop-2 fixes)

Two runs, audited in plan step 5 (`plans/plan-2026-09-19T224205-49c8bf80/findings/audit-loop-1-seg.md`),
run names `convunext_audit_seg_l1` and `convunext_audit_seg_l1_rerun` (both under `results/`, which
is untracked). Command (only the name differs), on GPU 1 (RTX 4070), commit `2c9050f57`:

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.convunext.train_convunext_segmentation \
    --variant tiny --image-size 128 --epochs 2 --batch-size 16 --max-samples 1600 --viz-freq 1 \
    --experiment-name convunext_audit_seg_l1        # and ..._l1_rerun
```

1440 fit images, 160 validation images, the full 3669-image test split, 90 steps per epoch, seed
42, every other flag at its default. Both runs exited 0 with `status: "ok"`. These are two epochs:
the schedule spanned 180 steps and the validation loss was still falling, so they show the wiring
works, not what the recipe reaches.

| Test split, best weights (= final, epoch 2) | `..._l1` | `..._l1_rerun` | Majority-class baseline |
|---|---|---|---|
| mIoU | 0.4296 | 0.4272 | 0.1924 |
| pixel accuracy | 0.7280 | 0.7262 | 0.5772 |
| IoU pet / background / border | 0.4898 / 0.7021 / 0.0970 | 0.4910 / 0.7007 / 0.0899 | 0 / 0.5772 / 0 |
| val mIoU, epoch 1 then 2 | 0.3901, 0.4284 | 0.3896, 0.4269 | |
| val loss, epoch 1 then 2 | 0.7113, 0.6530 | 0.7179, 0.6565 | |
| `epoch_times` (s) | 83.30, 9.59 | 84.60, 9.64 | |
| `fit_wall_seconds` | 95.84 | 97.28 | |

Test mIoU is 2.2 times the baseline in both runs and all three classes are predicted, but `border`
(a thin band, 11 to 12 percent of the pixels) is barely learned: IoU about 0.09 to 0.10, recall
0.11. A numpy recompute of the confusion matrix from the saved `best_model.keras` equalled the
summary's exactly. Wall clock: the first epoch is 74 s of graph build (steady epoch 9.6 s);
`test_eval_seconds` was 59.3 and 58.5 s and the analyzer 16.6 and 17.9 s. The test evaluation
figure predates the `final_reused_best` change, which scores the split once when the best epoch is
the last one (it was scored twice in these runs although both blocks were identical).

Difference between these two same-seed runs (n = 2, measured BEFORE the stateless-flip fix, so
it is not a noise floor of the current code): 0.0024 in test mIoU, 0.0018 in pixel accuracy,
0.0028 in test loss, 0.0071 in `border` IoU. The loop-2 measurement below replaces it as the
reference: the spread between SEEDS is 0.0174, 7.3 times larger.

A separate measurement on `jit_compile` (tiny, 128 px, batch 16, 20 steps on synthetic batches, GPU 1,
one process per row; a probe script, not a run directory):

| | first step | steady step |
|---|---|---|
| default (XLA) | 73.4 s, 73.5 s, 73.3 s | 0.086 s |
| `jit_compile=False` | 51.3 s, 51.5 s | 0.287 s |

Turning XLA off saves about 22 s of build and costs about 0.2 s per step, so it breaks even near
110 steps (a little over one epoch at 90 steps) and loses beyond; the default is unchanged.

### Loop 2 (the fixed code of iteration 1, then the loop-2 fixes)

Every number below was read from the named run's `results_summary.json` or `training_log.csv`
when this section was written; the run directories are under `results/` (untracked). Audit note:
`plans/plan-2026-09-19T224205-49c8bf80/findings/audit-loop-2.md`. Same command as loop 1 (tiny,
128 px, 2 epochs, batch 16, `--max-samples 1600`, `--viz-freq 1`), GPU 1, one job at a time,
1440 fit / 160 validation images, the full 3669-image test split. All exited 0 with
`status: "ok"`, `best_epoch == epochs_run == 2`, `final_reused_best: true`.

| Run (test split, best weights) | Seed | mIoU | Pixel acc. | IoU pet / background / border | `epoch_times` (s) | `fit_wall_seconds` | `test_eval_seconds` |
|---|---|---|---|---|---|---|---|
| `convunext_audit_seg_l2` | 42 | 0.4281 | 0.7277 | 0.4947 / 0.7027 / 0.0870 | 81.82, 9.77 | 94.90 | 34.09 |
| `convunext_audit_seg_l2_s1` | 1 | 0.4392 | 0.7304 | 0.4959 / 0.7044 / 0.1174 | 84.27, 9.56 | 97.10 | 33.74 |
| `convunext_audit_seg_l2_s2` | 2 | 0.4455 | 0.7339 | 0.5072 / 0.7071 / 0.1222 | 82.34, 9.82 | 95.38 | 34.06 |

**Seed spread, not a same-seed floor.** Across the three seeds test mIoU spans 0.0174 (0.4281 to
0.4455; sample standard deviation 0.0088), `border` IoU 0.035 (0.0870 to 0.1222). A seed changes the
initial weights AND the validation split, so this is the spread of the recipe over seeds; it is not
the run-to-run variation of one seed. At this 2-epoch recipe a difference between two
configurations below about 0.02 mIoU on one seed is not attributable. The trivial baseline is 0.1924
in all three (a majority-class predictor).

**Same seed, fixed code (`convunext_audit_seg_l2_fix`, seed 42, commit `1fb374e81`).** Test mIoU
0.42811 against 0.42812 in `..._l2` (1.7e-5 apart), per-class IoU identical to four decimals,
test loss 0.66023 against 0.66021. `test_eval_seconds` 34.09 to 26.18: the test split is scored in
ONE prediction pass now. The two runs differ in the evaluation code as well as in the run, so this
pair says the same-seed difference is far below the seed spread, not that two runs are bit-identical
(the GPU kernel non-determinism in "Open items" is unchanged). `visualizations.seconds` 1.29 to 3.40
because the per-epoch grid seconds are now counted, and `best_vs_final_predictions.png` is absent
(`visualizations.skipped` names it) since the best epoch is the last; `analyzer.removed` is
`["summary_dashboard.png"]` and that file is gone from `model_analysis/`.

**The 10-epoch probe (`convunext_probe_seg_l2_10ep`).** Seed 42, whole train split (3312 fit / 368
validation images, 207 steps per epoch), every other flag at its default:

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.convunext.train_convunext_segmentation \
    --variant tiny --image-size 128 --epochs 10 --batch-size 16 --experiment-name convunext_probe_seg_l2_10ep
```

Test mIoU 0.6579 (pixel accuracy 0.8611, loss 0.3584), per-class IoU pet / background / border
0.7446 / 0.8519 / 0.3771, 3.4 times the trivial baseline. `fit_wall_seconds` 291.3, the whole
process 371.6 s (measured around the process, not stored in the run directory); epoch 1 93.39 s,
epochs 2 to 10 about 20.5 to 21.6 s. Verdicts, on rules written before the run:

| Question | Rule | Measured | Verdict |
|---|---|---|---|
| Is `border` stuck? | test border IoU below 0.20 | 0.3771 (recall 0.462, precision 0.672) | No: above even the 0.35 upper edge of "lagging"; no class weight or Dice term is indicated |
| Does the LR anneal? | `lr_last_step` at most 2e-5 and CSV `lr` falls every epoch | `lr_last_step` 1.0001e-05, ten strictly falling CSV values | Yes |
| Is mIoU still rising? | `val_miou` at epoch 10 exceeds the value at epoch 5 by 0.01 | 0.6480 against 0.5930 (+0.0550) | Yes by the rule, but flat at the end: `val_miou` 0.6487 at epoch 9, 0.6480 at epoch 10 (0.6056 at epoch 6) |

`val_miou` by epoch: 0.4113, 0.4881, 0.5367, 0.5810, 0.5930, 0.6056, 0.6251, 0.6432, 0.6487,
0.6480. The 10-epoch cosine has run its course; whether the 30-epoch default ends above 0.648 is not
measured (only 10 epochs, one seed). The predicted `border` share is still below the true 12.3
percent (8.5 percent), so `border` is under-predicted by a third, not collapsed as in the 2-epoch
runs (2.6 to 3.4 percent).

Wall clock: `run.log` timestamps put the `final_model.keras` save plus `validate_model_loading` at
11.7 s in `..._l2` (12.8 s in loop 1); `analyzer.seconds` 9.5, 13.3, 9.5 in the three seeds and
18.7 in `..._fix`.

### Loop 3 (same-seed floor, the default 30-epoch recipe)

Every number below was read from the named run's `results_summary.json` or `training_log.csv`
when this section was written; the run directories are under `results/` (untracked). Audit note:
`plans/plan-2026-09-19T224205-49c8bf80/findings/audit-loop-3.md`. GPU 1, one job at a time.

**The same-seed floor.** Three runs of the loop-2 command at seed 42 on the fixed code
(`convunext_audit_seg_l3`, `_rep1`, `_rep2`; tiny, 128 px, 2 epochs, batch 16, `--max-samples 1600`,
`--viz-freq 1`):

| Run | Test mIoU (best weights) | IoU pet / background / border | Test loss | `epoch_times` (s) | `fit_wall_seconds` | `test_eval_seconds` |
|---|---|---|---|---|---|---|
| `convunext_audit_seg_l3` | 0.428110 | 0.494694 / 0.702653 / 0.086982 | 0.660235 | 83.11, 9.71 | 96.29 | 25.28 |
| `convunext_audit_seg_l3_rep1` | 0.428109 | 0.494705 / 0.702655 / 0.086967 | 0.660236 | 84.44, 9.55 | 97.44 | 25.48 |
| `convunext_audit_seg_l3_rep2` | 0.428123 | 0.494720 / 0.702670 / 0.086979 | 0.660208 | 84.22, 9.48 | 97.21 | 26.12 |

The test mIoU range is 0.000014, and 0.000017 over the five seed-42 runs of the fixed code
(`convunext_audit_seg_l2` 0.428124, `convunext_audit_seg_l2_fix` 0.428107 and the three above; three
code states). That is about a thousandth of the loop-2 seed spread (0.0174, n = 3): the spread in "Loop 2" is a
seed effect (a seed changes the initial weights and the fit/validation split), not run-to-run noise.
The same-seed floor is the GPU kernel non-determinism of "Open items" (weights differ at the 1e-6 level
after 20 steps, measured in iteration 1; test mIoU differs by about 1e-5 after 180 steps (2 epochs of 90), measured here); before the stateless flip of iteration 1 the same pair differed by
0.0024. So a change of about 1e-4 mIoU on a fixed seed and split is attributable in a 2-epoch run,
unless the change alters how the random number generators are consumed (an added layer, a different
initialiser draw order), which makes it a different-seed comparison.

**The default recipe, once (`convunext_probe_seg_l3_default`).** Seed 42, `--variant tiny --image-size 128
--batch-size 16` and every other flag at its default: 30 epochs planned, patience 10, cosine LR 1e-3,
whole train split (3312 fit / 368 validation images, 207 steps per epoch), the full 3669-image test
split. Exit 0, `status: "ok"`, 28 epochs run, `fit_wall_seconds` 675.15, `test_eval_seconds` 48.73
(two scored passes, best and final), `analyzer.seconds` 22.10. Epoch 1 took 95.76 s, epochs 2 to 28 19.80
to 20.86 s (mean 20.30). Verdicts, on rules written before the run:

| Question | Rule | Measured | Verdict |
|---|---|---|---|
| Did it run to the end? | `epochs_run` 30, else early stop | `stopped_early` true after epoch 28 of 30 (patience 10 on `val_loss`) | Early stop: it fired exactly 10 epochs after the best epoch (18 + 10), saved 2 epochs and did not change the scored checkpoint |
| Is the best epoch the last? | `best_epoch == epochs_run` | `best_epoch` 18 of 28 | No: the best-versus-final path ran on a real segmentation run |
| Overfit? | `val_loss` last minus best at least 0.02 with train loss lower | 0.3316 - 0.2990 = +0.0327; train loss 0.1736 against 0.2383 | Yes |
| Plateau? | `val_miou` range over the last 10 epochs below 0.01 | 0.0150 (epochs 19 to 28); 0.0033 over epochs 24 to 28 | No by the rule; the last four epochs are flat (0.7109, 0.7114, 0.7109, 0.7102) |
| Above the 10-epoch probe? | test mIoU above 0.6579 by more than the 0.0174 seed spread | 0.7100 (best), 0.7197 (final) | Above (0.0521 and 0.0618, 3.0 and 3.6 times the seed spread; the same-seed floor is 1.4e-5). Same seed and split, only the epoch count and so the schedule differ |
| Is `border` stuck? | test border IoU below 0.20 | 0.4593 (best), 0.4726 (final); recall 0.568 / 0.586, precision 0.706 / 0.710; predicted share 9.87 / 10.14 percent against 12.29 percent | No (the 10-epoch probe: 0.3771) |
| Does the LR anneal? | `lr_last_step` at most 2e-5 and CSV `lr` falls every epoch | `lr_last_step` 2.0869e-05 (4.3 percent above the line), CSV `lr` strictly falling from 1.0000e-03 to 3.4227e-05 | Borderline: the run stopped at step 5796 of 6210, and the 1e-5 floor is reached only at step 6210 |

Best weights (`best_model.keras`, epoch 18) against the final weights on the test split (60,112,896
pixels, both confusion matrices sum to it, `miou == miou_from_confusion`):

| | IoU pet / background / border | mIoU | Pixel accuracy | Test loss |
|---|---|---|---|---|
| best (epoch 18) | 0.7883 / 0.8824 / 0.4593 | 0.7100 | 0.8866 | 0.3061 |
| final (epoch 28) | 0.7985 / 0.8880 / 0.4726 | 0.7197 | 0.8911 | 0.3302 |

**The `val_loss` checkpoint scores below the final weights here.** `val_loss` selects the epoch whose
weights every headline number is scored on, and it turns at epoch 18 and then rises while `val_miou`
keeps improving to epoch 26 (0.7114). On the test split the final weights beat the best ones by 0.0097 mIoU,
0.0133 border IoU and 0.0045 pixel accuracy and lose only on cross-entropy (0.3302 against 0.3061), which is
over-confidence and not worse masks. One seed: the 0.0097 gap is exact for this run (both checkpoints come from it; the same-seed floor of
the test mIoU is 1.4e-5), and whether its sign generalizes across seeds is measured in loop 4 (seeds 43 and 44 of the
default recipe). It is a finding and nothing was changed: the monitor stays `val_loss` (a new
monitor would change a measured default). Only the best weights get the confusion and per-class figures (their
titles say so); the final weights' confusion is in `results_summary.json` (`test_metrics_final`).

**Best differs from final: what appeared.** `final_reused_best` false; `test_metrics_final` differs from
`test_metrics_best`; `best_vs_final_predictions.png` is present (four images, two visibly different prediction
columns); `visualizations.skipped` is `{}` and `failed` `[]`; 35 files listed and 35 on disk (34 PNG and
`segmentation_report.json`); `analyzer.removed` is `["summary_dashboard.png"]`. The TensorFlow retracing
warning that loop 2 saw once in a denoiser run with best differing from final was not reproduced here: no
`retracing` line in the 1.5 MB stdout. The two scored passes cost 48.73 s against about 26 s for one.

## Open items

- The training defaults are the ConvNeXt trainer's, untuned for segmentation. `border` is not stuck:
  the 10-epoch probe reaches test IoU 0.3771 (2-epoch runs 0.087 to 0.122), so no class weight or
  Dice term is justified by the evidence. The 30-epoch default was measured once ("Loop 3"): it
  stops at epoch 28, ends at test mIoU 0.7100 (best weights) and 0.7197 (final), above the probe's
  0.6579, with `border` IoU 0.4593 and 0.4726. Whether the `val_loss` checkpoint costs mIoU beyond this
  one seed is unmeasured (one seed, 0.0097 apart).
- `--deep-supervision`, a Dice or focal loss and a softmax head are not offered: the stock
  sparse cross-entropy on logits is the whole loss (the library's `SegmentationLosses` need
  one-hot targets and probabilities).
- Two same-seed runs can still differ. The input batches are reproducible (tested), but a
  measurement on this model (20 steps from one seed on the SAME numpy batches, two builds, GPU 1)
  left 172 of 172 weight tensors not bit-identical, largest absolute difference 3.6e-6 to 4.4e-6:
  GPU kernel non-determinism (convolution backward / XLA), not investigated further. The
  same-seed spread of the current code is 0.000014 test mIoU (n = 3, "Loop 3"); the seed spread is
  0.0174 (n = 3).
- `visualizations.seconds` counts the per-epoch grids and the end-of-run figures, not the
  dashboard redraw (the shared callback has no clock); `fit_wall_seconds` minus the sum of
  `epoch_times` is the only account of it.
- `run.log` records the analyzer, figures and grids; `validate_model_loading` plus the
  `final_model.keras` save took 11.7 s in `convunext_audit_seg_l2`.
