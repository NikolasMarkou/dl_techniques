# Depth Anything Training

Pattern-5 (depth-estimation) training scaffold for
`dl_techniques.models.depth_anything.DepthAnything` on MegaDepth RGB+depth pairs.

> **Status (post-`plan_2026-05-10_54e6e303`)**: in-tree
> `dl_techniques.models.vit.ViT` as the default encoder
> (`--encoder-kind real`). DPT decoder default is `linear`. `train_step`
> dispatches to a clean labeled path or a semi-sup path that adds FAL +
> stop-gradient pseudo-label L1 consistency. Semi-supervised data feed
> is now wired end-to-end via `--unlabeled-image-glob`, which builds an
> `UnlabeledImageDataset` paired with `MegaDepthDataset` into
> `((x_lab, x_unlab), y_lab)` batches. On-step EMA decay is driven by
> `TeacherEMACallback` (`--ema-schedule {cosine,linear,none}`). External
> encoder weights can be loaded via `--pretrained-encoder-weights`
> (`from_pretrained_encoder` re-syncs the teacher).

## Quick start

```bash
# Smoke run (2 epochs on 100 train pairs / 20 val pairs)
MPLBACKEND=Agg .venv/bin/python -m train.depth_anything.train_depth_anything \
    --encoder-type vit_s \
    --epochs 2 --batch-size 4 --patch-size 256 \
    --max-train-files 100 --max-val-files 20 \
    --gpu 0

# Real run
MPLBACKEND=Agg .venv/bin/python -m train.depth_anything.train_depth_anything \
    --encoder-type vit_l --epochs 100 --batch-size 16 --patch-size 384 \
    --learning-rate 5e-6 --weight-decay 1e-5 \
    --warmup-epochs 5 --early-stopping-patience 20 \
    --seed 42 --gpu 0

# With pretrained transfer (from a prior compatible .keras checkpoint)
MPLBACKEND=Agg .venv/bin/python -m train.depth_anything.train_depth_anything \
    --encoder-type vit_l --epochs 100 --batch-size 16 \
    --init-from results/depth_anything_pretrain_*/model_inference.keras \
    --gpu 0

# Semi-supervised run with EMA teacher + paired unlabeled images
MPLBACKEND=Agg .venv/bin/python -m train.depth_anything.train_depth_anything \
    --encoder-type vit_l --epochs 100 --batch-size 8 --patch-size 384 \
    --use-feature-alignment --enable-semi-supervised \
    --unlabeled-image-glob '/data/unlabeled/**/*.jpg' \
    --ema-schedule cosine --ema-decay-start 0.5 --ema-decay-end 0.999 \
    --ema-warmup-steps 0 \
    --pretrained-encoder-weights /path/to/vit_encoder.keras \
    --gpu 0
```

`MPLBACKEND=Agg` is required to avoid X11 crashes on headless/remote systems.
Use a single GPU (`--gpu 0` or `--gpu 1`); never run two GPU jobs in parallel
on this machine.

## What this script does

- Loads MegaDepth RGB+depth pairs via `train.common.megadepth.MegaDepthDataset`.
- Builds a `DepthAnything` model via `create_depth_anything(...)`.
- Compiles with a **masked L1 + multi-scale gradient-matching loss**
  (locally-defined `DepthEstimationLoss`, copy of the CliffordNet trainer's
  loss). This is compatible with the model's default sigmoid output range.
- Wraps the five depth metrics from `dl_techniques.metrics.depth_metrics`
  with a small `_DepthOnlyMetricWrapper` so they ignore the mask channel
  produced by `MegaDepthDataset`.
- Adds `DepthMetricsCurveCallback` and `DepthPredictionGridCallback` from
  `dl_techniques.callbacks.depth_visualization` for periodic visualization.
- Uses `train.common.create_callbacks(...)` for EarlyStopping +
  ModelCheckpoint + CSVLogger + TensorBoard + TerminateOnNaN. `monitor=val_loss`.
- Optimizer: `AdamW` via `dl_techniques.optimization.optimizer_builder` with
  cosine-decay LR schedule + warmup. **Does not** combine `kernel_regularizer=L2`
  with `AdamW.weight_decay` (avoids double weight decay).
- Optional pretrained backbone init via `--init-from <ckpt.keras>` (skips
  layers prefixed with `dpt_decoder` so the head is freshly initialized).

## CLI flags

| Flag                          | Default                                   | Notes |
|-------------------------------|-------------------------------------------|-------|
| `--megadepth-root`            | `/media/arxwn/data0_4tb/datasets/Megadepth` | Path to MegaDepth root. |
| `--max-train-files`           | `10000`                                   | Cap on training pairs. |
| `--max-val-files`             | `1000`                                    | Cap on validation pairs. |
| `--encoder-type`              | `vit_l`                                   | One of `{vit_s,vit_b,vit_l}`. |
| `--encoder-kind`              | `real`                                    | `real` (in-tree `ViT`) or `placeholder` (Conv-BN-ReLU compat). |
| `--enable-semi-supervised`    | off                                       | Activates FAL + pseudo-label consistency train_step path on `((x_lab,x_unlab),y_lab)`. Combine with `--unlabeled-image-glob`. |
| `--unlabeled-image-glob`      | `None`                                    | Glob to unlabeled RGB images (e.g. `'/data/unlab/**/*.jpg'`); paired with `MegaDepthDataset` via `pair_labeled_unlabeled`. |
| `--pretrained-encoder-weights`| `None`                                    | Path to a saved encoder `.keras` file. Loaded via `DepthAnything.from_pretrained_encoder`; re-syncs the EMA teacher. |
| `--ema-schedule`              | `none`                                    | `{cosine,linear,none}`. Drives `TeacherEMACallback` when `--use-feature-alignment` is on. |
| `--ema-decay-start`           | `0.5`                                     | EMA decay at step 0. |
| `--ema-decay-end`             | `0.999`                                   | EMA decay asymptote at the end of the run. |
| `--ema-warmup-steps`          | `0`                                       | Number of training steps before EMA updates begin. |
| `--use-feature-alignment`     | off                                       | Enables `frozen_encoder` (weight-shared teacher; EMA via `update_teacher_ema`). Default off. |
| `--cutmix-prob`               | `0.5`                                     | Forwarded to `StrongAugmentation`. |
| `--color-jitter-strength`     | `0.2`                                     | Forwarded to `StrongAugmentation`. |
| `--epochs`                    | `100`                                     | |
| `--batch-size`                | `16`                                      | |
| `--patch-size`                | `384`                                     | RGB+depth crop size; matches default `input_shape`. |
| `--learning-rate`             | `5e-6`                                    | |
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
| `--gpu`                       | `None`                                    | GPU index (or `None` for all visible). |
| `--output-dir`                | `results`                                 | Results parent directory. |
| `--experiment-name`           | auto                                      | Defaults to `depth_anything_<encoder>_<timestamp>`. |

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
   * **L1 pseudo-label consistency** between the student's depth on
     `x_unlab` and the EMA teacher's stop-gradient pseudo-depth.
4. Drives the EMA teacher per training batch via `TeacherEMACallback`
   using the chosen schedule.

The labeled component is unchanged: depth supervision still uses MegaDepth
RGB+depth pairs (sparse SfM ground truth caveats below).

## Known Issues — depth quality

Even with this script working end-to-end, **do not expect publishable Depth
Anything numbers**:

1. **Encoder is in-tree `ViT`, not DINOv2.** No external pretrained weights
   are loaded by default. Use `--pretrained-encoder-weights <ckpt.keras>`
   (encoder-only) or `--init-from <ckpt.keras>` (full model) if you have a
   compatible checkpoint; otherwise initialization is random.
2. **MegaDepth ground truth is sparse** (Structure-from-Motion). Even a
   correctly trained Depth Anything model will produce dense depth maps
   that look *qualitatively* good but disagree with sparse SfM ground
   truth on non-Lambertian / textureless regions.

The script is the right shape for when these issues are addressed — at that
point `git pull` + re-run should be all that's needed. Until then, this is
a Pattern-5 scaffold: it runs, it produces visualizations, it logs metrics,
but the model behind it isn't trained as advertised.

## Reference implementation

This trainer mirrors `src/train/cliffordnet/train_depth_estimation.py` 1:1 in
structure (config dataclass, callback wiring, optimizer setup, MegaDepth
dataset usage). Differences:

- Single-output model (no deep supervision), so no `_MultiScaleDataset` or
  `DeepSupervisionWeightScheduler`.
- Metrics are wrapped with `_DepthOnlyMetricWrapper` because MegaDepth's
  `y_true = concat([depth, mask], -1)` shape is two-channel.
- `init_from` skips layers prefixed `dpt_decoder` (vs `head_` for CliffordNet).
