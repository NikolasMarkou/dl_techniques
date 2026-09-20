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
- The train pipeline drops the incomplete last batch and reshuffles every epoch.

## What mIoU means here

`mIoU` is the mean over classes of `TP / (true + predicted - TP)` computed from a pixel
confusion matrix; a class with no true and no predicted pixel has no IoU and is left out of the
mean (the `keras.metrics.MeanIoU` convention). The summary carries two independent measurements
of it, Keras' `miou` and `miou_from_confusion` (a numpy count), and a test cross-checks them.

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
        training_dashboard.png      shared per-epoch curves (loss, accuracy, lr, ...), redrawn on a cadence
        epoch_000_seg_grid.png      the UNTRAINED model on the fixed validation batch
        epoch_NNN_seg_grid.png      after epoch NNN (every --viz-freq epochs, and the last epoch)
        confusion_matrix.png        TEST split, best weights: pixel counts and row-normalized recall
        per_class_metrics.png       grouped bars of IoU, Dice, precision and recall per class
        segmentation_report.json    the same per-class numbers plus the confusion counts (strict JSON, null where undefined)
        best_vs_final_predictions.png   image | ground truth | best weights | final weights, on the grid's samples
        miou_curve.png              train and validation mIoU per epoch, the best epoch marked
    model_analysis/                 weights + spectral ModelAnalyzer output (analysis_results.json, PNGs); absent with --no-model-analysis
```

Grid figures show `viz_samples` validation images chosen ONCE with `numpy.random.RandomState(seed)`,
so every epoch and every run of one seed shows the same images. Rows are image | ground truth |
prediction; ground truth and prediction share one palette (pet orange, background dark blue,
border yellow) with a legend, and each prediction title carries that row's pixel accuracy.

Notes:

- `epoch` in the CSV is 0-based, `best_epoch` in the summary is 1-based. CSV `lr` is the rate at
  the START of the epoch.
- `final_model.keras` is the real last epoch: `EarlyStopping` is built with
  `restore_best_weights=False` (Keras 3.8 would otherwise restore the best weights at train end).
  When the best epoch is the last one the two files hold the same weights.
- `model_analysis/` describes the LAST epoch's weights (the in-memory model). Its spectral
  verdicts are heuristics and unreliable for a short run. Calibration, information flow and
  training dynamics are off: they read per-image labels and probabilities.
- `model_analysis/` holds `analysis_results.json` plus four PNGs. `summary_dashboard.png` is
  library output that is EMPTY for a data-free analysis (every panel reads "No ... data
  available"); the useful ones are `spectral_summary.png`, `spectral_funnel_diagram.png` and
  `weight_learning_journey.png`. A smoke run (tiny, 64 px) spent about 18 s on it (`analyzer.seconds`).
- Every figure is isolated: one that raises is recorded under `visualizations.failed` with its
  error and the run stays `status: "ok"`. The analyzer cannot fail a finished run either: an
  exception is recorded as `analyzer.status: "error"`.

### `results_summary.json` keys

Identity and setup (present for `status` `ok` and `diverged`): `status`, `run_dir`,
`experiment_name`, `model_family`, `dataset`, `variant`, `params`, `input_shape`, `num_classes`,
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
| `lr_first_epoch`, `lr_last_epoch`, `lr_last_step` | CSV `lr` of the first and last epoch; the schedule's value at the final optimizer step. |
| `best_val_metrics`, `final_val_metrics` | Validation `loss`, `accuracy`, `miou` of the best and the last epoch. |
| `test_metrics_best`, `test_metrics_final` | Test-split `loss`, `accuracy`, `miou`, `miou_from_confusion`, `pixel_accuracy`, `per_class_iou`, `confusion` for the reloaded `best_model.keras` and for the last-epoch weights. |
| `trivial_baseline` | Scores of the majority-class predictor on the test masks (`predicted_class`, `miou`, `per_class_iou`, `pixel_accuracy`, `confusion`). |
| `visualizations` | `{files, failed, seconds}`: the names that exist on disk (dashboard, per-epoch grids, end-of-run figures), `{figure: error}` for those that raised, wall seconds of the end-of-run figures. |
| `analyzer` | `{status, analyzers, error, path, seconds}` read back from `model_analysis/analysis_results.json`: status `ok`, `partial`, `missing`, `unreadable` or `error`; `analyzers` lists which of `weights` / `spectral` wrote results. With `--no-model-analysis`: the skipped block (`status: "skipped"`). |
| `test_eval_seconds` | Wall seconds of the best and final test evaluations plus the baseline. |
| `best_checkpoint_load_error`, `best_checkpoint_max_abs_diff` | Whether the best checkpoint reloaded, and the largest gap between its validation metrics and what `fit` recorded for the best epoch. |
| `model_loading_validated` | Whether `final_model.keras` reloads and reproduces its predictions. |

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
# the shared readback of a weights + spectral analysis
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_train/test_common_run_summary.py -q
```

`test_run.py` is slow (several real fits). No test reads the TFDS cache or writes under the
repo-root `results/`.

## Measured runs

None audited yet. Every number in this section must come from a run directory and its
summary; the first audit run (`--variant tiny --image-size 128 --epochs 2`) is the next step of the
plan that added this package.

## Open items

- The training defaults are the ConvNeXt trainer's, untuned for segmentation.
- `--deep-supervision`, a Dice or focal loss and a softmax head are not offered: the stock
  sparse cross-entropy on logits is the whole loss (the library's `SegmentationLosses` need
  one-hot targets and probabilities).
- Best-versus-final test evaluations each cost a full pass over the 3669 test images; when the
  best epoch is the last one they are the same weights evaluated twice.
- The noise floor between two same-seed runs is not measured yet.
