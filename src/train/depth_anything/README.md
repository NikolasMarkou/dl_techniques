# Depth Anything Training

Pattern-5 (depth-estimation) training scaffold for
`dl_techniques.models.depth_anything.DepthAnything` on MegaDepth RGB+depth pairs.

> **Status (post-`plan_2026-05-10_bd098beb`)**: the in-tree
> `dl_techniques.models.vit.ViT` is now wired as the default encoder
> (`--encoder-kind real`); the legacy Conv-BN-ReLU stack is preserved
> behind `--encoder-kind placeholder`. The decoder default is `linear`
> (no longer sigmoid-clamped to `[0,1]`). The `train_step` Keras-3 API
> bug remains fixed. Save/load round-trip is verified on CPU
> (max-abs-diff = 0.0).
> **Open**: `--enable-semi-supervised` switches the model to the FAL
> train_step path on `((x_lab, x_unlab), y_lab)` data, but the in-tree
> `MegaDepthDataset` only yields labeled batches — feeding paired
> unlabeled imagery requires extending the data pipeline. See
> "Semi-supervised usage" below.

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
| `--enable-semi-supervised`    | off                                       | Activates FAL train_step path on `((x_lab,x_unlab),y_lab)`. Requires extending `MegaDepthDataset` — see Semi-supervised usage below. |
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

`--enable-semi-supervised` flips `DepthAnything.train_step` into a path that
expects `((x_labeled, x_unlabeled), y_labeled)` per-batch. The model then
runs:

1. Supervised forward pass + `compute_loss` on the labeled batch.
2. (When `--use-feature-alignment` is also set) `FeatureAlignmentLoss`
   between `self.encoder(x_unlabeled)` and the weight-shared
   `self.frozen_encoder(x_unlabeled)`. The teacher is initialized from a
   `keras.models.clone_model` of the student at `build()` time and can be
   advanced by calling `model.update_teacher_ema(decay=...)` from a custom
   on-step callback.

Caveat — the in-tree `train.common.megadepth.MegaDepthDataset` yields plain
`(rgb, depth_with_mask)` tensors only. Wiring an unlabeled stream is
**out of scope** for this plan; the model-side infrastructure is in place
so a future plan can add a `_PairedUnlabeledDataset` wrapper that yields
`((x_lab, x_unlab), y_lab)` and the script will pick it up unchanged.

## Known Issues — depth quality

Even with this script working end-to-end, **do not expect publishable Depth
Anything numbers**:

1. **Encoder is in-tree `ViT`, not DINOv2.** No external pretrained weights
   are loaded by default. Use `--init-from <pretrained.keras>` if you have
   a compatible backbone (otherwise initialization is random).
2. **Semi-supervised data feed not wired** — only labeled MegaDepth depth
   supervision is used unless you extend the dataset to yield
   `((x_lab, x_unlab), y_lab)`. The "62M unlabeled images" claim from the
   original paper is not implemented at the data level.
3. **MegaDepth ground truth is sparse** (Structure-from-Motion). Even a
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
