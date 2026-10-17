# Depth Anything Training

Pattern-5 (depth-estimation) training scaffold for
`dl_techniques.models.vision.depth_anything.DepthAnything` on MegaDepth RGB+depth pairs.

> The default encoder is the in-tree `dl_techniques.models.vision.vit.ViT`
> (`--encoder-kind real`). The decoder ends in a ReLU, as the reference head does. `train_step`
> dispatches to a clean labeled path or a semi-sup path that adds FAL +
> stop-gradient pseudo-label consistency (affine-invariant per CutMix region, eq. 6-8). Semi-supervised data feed
> is wired end-to-end via `--unlabeled-image-glob`, which builds an
> `UnlabeledImageDataset` paired with `MegaDepthDataset` into
> `((x_lab, x_unlab), y_lab)` batches. On-step EMA decay is driven by
> `TeacherEMACallback` (`--ema-schedule {cosine,linear,none}`). External
> encoder weights can be loaded via `--pretrained-encoder-weights`
> (`from_pretrained_encoder` re-syncs the teacher).

## Quick start

```bash
# Smoke run (2 epochs on 100 train pairs / 20 val pairs)
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.depth_anything.train_depth_anything \
    --encoder-type vit_s \
    --epochs 2 --batch-size 4 --patch-size 256 \
    --max-train-files 100 --max-val-files 20

# Real run
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.depth_anything.train_depth_anything \
    --encoder-type vit_l --epochs 100 --batch-size 16 --patch-size 384 \
    --learning-rate 5e-6 --weight-decay 1e-5 \
    --warmup-epochs 5 --early-stopping-patience 20 \
    --seed 42

# With pretrained transfer (from a prior compatible .keras checkpoint)
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.depth_anything.train_depth_anything \
    --encoder-type vit_l --epochs 100 --batch-size 16 \
    --init-from results/depth_anything_pretrain_*/model_inference.keras

# Semi-supervised run with EMA teacher + paired unlabeled images
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.depth_anything.train_depth_anything \
    --encoder-type vit_l --epochs 100 --batch-size 8 --patch-size 384 \
    --use-feature-alignment --enable-semi-supervised \
    --unlabeled-image-glob '/data/unlabeled/**/*.jpg' \
    --ema-schedule cosine --ema-decay-start 0.5 --ema-decay-end 0.999 \
    --ema-warmup-steps 0 \
    --pretrained-encoder-weights /path/to/vit_encoder.keras
```

`MPLBACKEND=Agg` is required to avoid X11 crashes on headless/remote systems.
Select the GPU with the `CUDA_VISIBLE_DEVICES` export and pass no `--gpu`: the
flag counts within the visible devices (D-110), so without an export an index
picks a GPU of the whole machine. Never run two GPU jobs in parallel on this
machine.

## What this script does

- Loads MegaDepth RGB+depth pairs via `train.common.megadepth.MegaDepthDataset`.
  Three rules there, each fixed on 2026-10-17 (plan-2026-10-14T062616-81938e81,
  iteration 2):
  - The training augmentation is a horizontal flip only, the reference's only
    labeled-image augmentation (arXiv 2401.10891, section 4.1). Up-down flips and
    90-degree rotations kept a random-init model at a nearly constant map: on the
    RTX 4070, `vit_s`, 1500 steps, training loss 1.26 and held-out Pearson 0.05,
    against 0.82 and 0.23 with the flip only (D-124). Every run before that date
    trained with them.
  - A crop whose valid depth is one constant is skipped like a crop with too few
    valid pixels. Part of MegaDepth stores ordinal depth (2.0 or 5.0 on every
    valid pixel); its target is all zeros, scores exactly 1.0 for any prediction
    and has no gradient (D-125). The first validation files of a small run are
    mostly such maps (60 of 94 in scene `0001`).
  - Batch `i` reads files `i*B .. i*B+B-1` once each. Before the fix, a batch read
    every other file and overlapped the next batch by half (D-126).
- Splits MegaDepth by scene with `split_megadepth_pairs`: `round(n_scenes * (1 -
  --train-split))` scenes, spread evenly over the sorted scene list, are validation
  scenes, the rest training scenes, and no scene is in both. `--max-train-files` and
  `--max-val-files` are then taken evenly across each side's scenes. Until 2026-10-17
  the trainer took the first `max_train_files + max_val_files` pairs in scene order
  and split that list at 90%: the defaults covered scenes `0000`-`0004` of 195, and
  the validation pairs came from scene `0004`, whose other images were in training
  (D-129).
- The labeled batch trains as loaded: the horizontal flip above is its only
  augmentation, as in the paper. CutMix (`--cutmix-prob`) and colour jitter
  (`--color-jitter-strength`) perturb the unlabeled batches of a semi-supervised run
  only. Until 2026-10-17 `DepthAnything.train_step` also CutMixed and jittered every
  labeled batch (D-131).
- `UnlabeledImageDataset` raises `OSError`, naming the file, for an image it cannot
  read. It used to substitute an all-zero image (D-130).
- Builds a `DepthAnything` model via `create_depth_anything(...)`.
- Supervises the Depth Anything V1 relative-depth target (arXiv 2401.10891,
  section 3.1): `MegaDepthDataset` turns each depth map into disparity `1 / t`
  normalized to `[0, 1]` per map over its valid pixels
  (`train.common.megadepth.normalized_disparity`; nearest valid pixel 1,
  farthest 0). The model's final projection ends in a ReLU, as in the reference
  head, so its output range `[0, inf)` contains the target's.
- Compiles the paper's **affine-invariant mean absolute error** (eq. 1-3,
  `build_depth_loss` = `AffineInvariantLoss(masked=True)`): per map, prediction
  and target are each shifted by their median and divided by their mean absolute
  deviation over the valid pixels, then compared with L1. It is invariant to the
  prediction's scale and shift. Until 2026-10-16 the target was min-max depth in
  `[-1, 1]` (half of it below what the ReLU can output) and the loss a masked L1
  plus gradient matching on the raw output; runs before that date trained a
  different objective.
- Compiles the five depth metrics from `dl_techniques.metrics.depth_metrics`
  (`build_depth_metrics`). They take `MegaDepthDataset`'s two-channel
  `y_true = concat([disparity, mask], -1)` whole, mask invalid pixels
  themselves, and score the prediction after per-image least-squares
  scale-and-shift alignment (`align_scale_and_shift`), so their values are on
  the normalized-disparity scale, not metric depth.
  Until 2026-10-16 a `_DepthOnlyMetricWrapper` sliced the mask channel off
  first, which left no valid pixel: every metric column of every run since the
  trainer was added (b9dccdcea, 2026-05-10) reads 0.0.
- Adds `DepthMetricsCurveCallback` and `DepthPredictionGridCallback` from
  `dl_techniques.callbacks.depth_visualization` for periodic visualization.
- Uses `train.common.create_callbacks(...)` for EarlyStopping +
  ModelCheckpoint + CSVLogger + TensorBoard + TerminateOnNaN. `monitor=val_loss`.
- Optimizer: `AdamW` via `dl_techniques.optimization.optimizer_builder` with
  cosine-decay LR schedule + warmup. **Does not** combine `kernel_regularizer=L2`
  with `AdamW.weight_decay` (avoids double weight decay).
- Optional pretrained backbone init via `--init-from <ckpt.keras>`
  (`init_from_checkpoint`): every layer of the checkpoint except the decoder, which is
  skipped by the name of `model.decoder` (`depth_head`) and so starts
  freshly initialized.

## CLI flags

| Flag                          | Default                                   | Notes |
|-------------------------------|-------------------------------------------|-------|
| `--megadepth-root`            | `/media/arxwn/data0_4tb/datasets/Megadepth` | Path to MegaDepth root. |
| `--train-split`               | `0.9`                                     | Labeled train/val split fraction. |
| `--min-valid-ratio`           | `0.1`                                     | Minimum fraction of valid depth pixels for a pair to be kept. |
| `--max-train-files`           | `10000`                                   | Cap on training pairs. |
| `--max-val-files`             | `1000`                                    | Cap on validation pairs. |
| `--encoder-type`              | `vit_l`                                   | One of `{vit_s,vit_b,vit_l}`. |
| `--encoder-kind`              | `real`                                    | `real` (in-tree `ViT`) or `placeholder` (Conv-BN-ReLU compat). |
| `--enable-semi-supervised`    | off                                       | Activates the pseudo-label consistency train_step path on `((x_lab,x_unlab),y_lab)`, plus the FAL term when `--use-feature-alignment` is also on. Combine with `--unlabeled-image-glob`. |
| `--unlabeled-image-glob`      | `None`                                    | Glob to unlabeled RGB images (e.g. `'/data/unlab/**/*.jpg'`); paired with `MegaDepthDataset` via `pair_labeled_unlabeled`. |
| `--pretrained-encoder-weights`| `None`                                    | Path to a saved encoder `.keras` file. Loaded via `DepthAnything.from_pretrained_encoder`; re-syncs the EMA teacher. |
| `--ema-schedule`              | `none`                                    | `{cosine,linear,none}`. Drives `TeacherEMACallback` when `--use-feature-alignment` **or** `--enable-semi-supervised` is on (both read the teacher). |
| `--ema-decay-start`           | `0.5`                                     | EMA decay at step 0. |
| `--ema-decay-end`             | `0.999`                                   | EMA decay asymptote at the end of the run. |
| `--ema-warmup-steps`          | `0`                                       | Number of training steps before EMA updates begin. |
| `--use-feature-alignment`     | off                                       | Enables `frozen_encoder` (weight-shared teacher; EMA via `update_teacher_ema`). Default off. |
| `--cutmix-prob`               | `0.5`                                     | Forwarded to `StrongAugmentation`, for the unlabeled batches of `--enable-semi-supervised` only: the teacher's pseudo-depth is CutMixed by the same box as the image, inside `DepthAnything.train_step`. The labeled batch is never mixed (paper: horizontal flip only). |
| `--color-jitter-strength`     | `0.2`                                     | Forwarded to `StrongAugmentation`; unlabeled batches only. |
| `--epochs`                    | `100`                                     | |
| `--batch-size`                | `16`                                      | |
| `--patch-size`                | `384`                                     | RGB+depth crop size; matches default `input_shape`. |
| `--learning-rate`             | `5e-6` with `--init-from` or `--pretrained-encoder-weights`, else `2e-4` | `5e-6` is the paper's rate for a pretrained encoder; `2e-4` was measured for a random encoder (see "Learning rate" below). An explicit value always wins. |
| `--weight-decay`              | `1e-5`                                    | AdamW decoupled weight decay. |
| `--gradient-clipping`         | `1.0`                                     | Forwarded to `optimizer_builder`. |
| `--warmup-epochs`             | `5`                                       | Cosine warmup epoch count. |
| `--lr-schedule`               | `cosine_decay`                            | Forwarded to `learning_rate_schedule_builder`. |
| `--early-stopping-patience`   | `20`                                      | Epochs without `val_loss` improvement. |
| `--monitor-every`             | `5`                                       | Visualization callback frequency. |
| `--steps-per-epoch`           | `None`                                    | If set, overrides `len(train_ds)`. |
| `--validation-steps`          | `200`                                     | |
| `--init-from`                 | `None`                                    | Path to a `.keras` checkpoint for backbone transfer. |
| `--seed`                      | `42`                                      | Seeds Python/NumPy/TF/Keras. |
| `--gpu`                       | `None`                                    | Index within the visible devices (D-110). Select the GPU with `CUDA_VISIBLE_DEVICES` instead and leave this unset. |
| `--output-dir`                | `results`                                 | Results parent directory. |
| `--experiment-name`           | auto                                      | Defaults to `depth_anything_<encoder>_<timestamp>`. |

### Learning rate

The paper (arXiv 2401.10891, section 4.1) trains a *pretrained* encoder at 5e-6 and
the randomly initialised decoder at 10 times that. This trainer has one rate for
the whole model, and no pretrained encoder ships, so the default depends on the
start: 5e-6 when `--init-from` or `--pretrained-encoder-weights` is given, else 2e-4.
The 2e-4 was measured (plan-2026-10-14T062616-81938e81 D-115) on the RTX 4070:
`vit_s`, batch 4, 256 px, 10 epochs of 40 steps, 1 warm-up epoch, 512 training and
58 validation pairs, one run per rate.

| LR | train loss, epoch 1 -> 10 | mean of last 3 | val loss, epoch 1 -> 10 | mean of last 3 | best val |
|---|---|---|---|---|---|
| 5e-6 | 1.688 -> 1.493 | 1.481 | 1.682 -> 1.495 | 1.533 | 1.495 |
| 5e-5 | 1.659 -> 1.340 | 1.324 | 1.972 -> 1.452 | 1.404 | 1.351 |
| 2e-4 | 1.578 -> 1.304 | 1.292 | 1.686 -> 1.378 | 1.459 | 1.314 |
| 1e-3 | 1.504 -> 1.307 | 1.288 | 1.845 -> 1.359 | 1.413 | 1.359 |

5e-6 trains measurably slower; 5e-5 to 1e-3 are not separable at this budget (the
validation loss moves by up to 0.37 between consecutive epochs of one run). 2e-4 is
the middle of that range. A constant prediction scores exactly 1.0 under this loss,
and every run here is still above that after 400 steps: these numbers rank
learning rates, they do not show a trained model.

## Output layout

```
results/
└── depth_anything_<encoder>_<timestamp>/
    ├── config.json
    ├── training_history.json
    ├── training_summary.txt
    ├── model_inference.keras
    ├── visualization_plots/
    │   ├── metrics_curve_*.png
    │   └── prediction_grid_epoch_*.png
    └── (TensorBoard logs, CSV logs, model checkpoints)
```

## Semi-supervised usage

Combine `--enable-semi-supervised` with `--unlabeled-image-glob` and (for
EMA) `--use-feature-alignment` + `--ema-schedule`. The script then:

1. Builds a labeled `MegaDepthDataset` and an `UnlabeledImageDataset` from
   the glob.
2. Pairs them via `pair_labeled_unlabeled` into a `tf.data.Dataset`
   yielding `((x_lab, x_unlab), y_lab)`.
3. Routes batches into `DepthAnything._train_step_semi_supervised`, which
   adds two loss terms on top of the labeled `compute_loss`:
   * **FAL** between pooled student/teacher features on `x_unlab`.
   * **Pseudo-label consistency** between the student's depth on
     `x_unlab` and the EMA teacher's stop-gradient pseudo-depth: the
     paper's affine-invariant loss per CutMix region (eq. 6-8).
4. Drives the EMA teacher per training batch via `TeacherEMACallback`
   using the chosen schedule.

The labeled component is unchanged: depth supervision still uses MegaDepth
RGB+depth pairs (sparse SfM ground truth caveats below).

## Known Issues — depth quality

Even with this script working end-to-end, **do not expect publishable Depth
Anything numbers**:

1. **Encoder is in-tree `ViT`, not DINOv2.** No external pretrained weights
   are loaded by default. Use `--pretrained-encoder-weights <ckpt.keras>`
   (encoder-only) or `--init-from <ckpt.keras>` (a full DepthAnything checkpoint,
   decoder skipped) if you have a compatible checkpoint; otherwise initialization
   is random.
2. **MegaDepth ground truth is sparse** (Structure-from-Motion). Even a
   correctly trained Depth Anything model will produce dense depth maps
   that look *qualitatively* good but disagree with sparse SfM ground
   truth on non-Lambertian / textureless regions.

The script is the right shape for when these issues are addressed — at that
point `git pull` + re-run should be all that's needed. Until then, this is
a Pattern-5 scaffold: it runs, it produces visualizations, it logs metrics,
but the model behind it isn't trained as advertised.

## Reference implementation

This is the only Pattern-5 (depth-estimation) trainer in the repo. Its shape —
config dataclass, callback wiring, optimizer setup, MegaDepth dataset usage —
comes from the former CliffordNet depth trainer, from which it differs in:

- Single-output model (no deep supervision), so no `_MultiScaleDataset` or
  `DeepSupervisionWeightScheduler`.
- Metrics take MegaDepth's two-channel `y_true = concat([disparity, mask], -1)`
  directly; the depth metrics read the mask channel themselves.
- `init_from` skips the decoder by `model.decoder.name` (`depth_head`), not by the
  `head_` prefix CliffordNet used. It skipped the literal `dpt_decoder` until 2026-10-15, a name
  no layer carried, so the decoder was transferred as well.
