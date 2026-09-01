# train/video_jepa — Video-JEPA-Clifford

Self-supervised pretraining of a patch-level latent video model. The encoder is
`PatchEmbedding2D` + a `CliffordNetBlock` stack that keeps the 2D patch grid
intact; a causal predictor forecasts future patch latents at several horizons,
with a V-JEPA-style tube-masked latent objective and SIGReg for collapse
prevention. Model code:
`src/dl_techniques/models/vision/video_jepa/{config,model}.py`. Single script:
`train_video_jepa.py`.

There is no pixel decoder. Outputs live in latent patch space, e.g.
`(B, T, 14, 14, 128)` at `img_size=112, patch_size=8, embed_dim=128`. There is
no conditioning input — `call({"pixels": ...})` and `stream_step(frame)` take
pixels only.

## Run

```bash
# Smoke test, CPU or any GPU, ~seconds
MPLBACKEND=Agg .venv/bin/python -m train.video_jepa.train_video_jepa --smoke

# Synthetic sanity on a GPU
MPLBACKEND=Agg .venv/bin/python -m train.video_jepa.train_video_jepa \
    --dataset synthetic --gpu 1 \
    --epochs 2 --steps-per-epoch 4 --batch-size 2 \
    --T 4 --img-size 64 --patch-size 8

# Real BDD100K run with validation and EarlyStopping (many hours)
MPLBACKEND=Agg .venv/bin/python -m train.video_jepa.train_video_jepa \
    --dataset bdd100k \
    --videos-root /media/arxwn/data0_4tb/datasets/bdd_data/train/videos \
    --gpu 0 \
    --epochs 100 --steps-per-epoch 1000 \
    --val-steps 100 --val-fraction 0.1 --early-stopping-patience 15 \
    --batch-size 4 --T 24 --predict-horizons 1 4 15 \
    --img-size 112 --patch-size 8 --embed-dim 128 \
    --sigreg-num-proj 1024 --visualization-frequency 1 --seed 0 \
    --output-dir results/video_jepa_bdd100k_run_XX
```

`--smoke` is a preset (synthetic, T=4, img=64, patch=8, embed=64, depth=2,
batch=2, epochs=2, steps=4, sigreg-num-proj=64, horizons `1 2`). Flags you pass
explicitly still win over it.

## Loss

```
total = lambda_next * sum_h L_pred_h  +  lambda_mask * L_mask  +  sigreg_weight * L_sigreg
```

`L_pred_h` is MSE between a per-horizon `Dense(D, use_bias=False)` head on the
shared causal predictor and the encoder latents `h` frames ahead. `lambda_next`
is applied **per horizon**, not split across them, so adding a horizon adds
loss mass. `L_mask` is MSE on tube-masked patch positions. `L_sigreg` is an
Epps-Pulley Gaussianity statistic averaged over random projections.

Trackers: `next_frame_loss_h{h}` per horizon, plus an aggregated
`next_frame_loss`, `mask_loss`, `sigreg_loss`. `CSVLogger` writes one column
each, with `val_` twins when validation is on.

## CLI

| Flag | Default | Notes |
|---|---|---|
| `--dataset` | `synthetic` | `synthetic` or `bdd100k`. `bdd100k` requires `--videos-root`. |
| `--videos-root` | none | Directory of BDD100K `.mov` files, flat layout. |
| `--epochs` | `100` | |
| `--steps-per-epoch` | `1000` | |
| `--batch-size` | `4` | Must be >= 2: `CliffordNetBlock` uses BatchNormalization in its context stream. |
| `--learning-rate` | `3e-4` | |
| `--weight-decay` | `1e-4` | |
| `--seed` | `0` | |
| `--gpu` | none | GPU index; sets `CUDA_VISIBLE_DEVICES`. Nothing stops two concurrent jobs — keep GPU jobs serial yourself. |
| `--output-dir` | `results/video_jepa_<timestamp>` | |
| `--smoke` | off | Tiny preset, see above. |
| `--T` | `24` | Frames per clip = `num_frames` = `history_size_k`. Must exceed `max(--predict-horizons)`. |
| `--img-size` | `112` | |
| `--img-channels` | `3` | |
| `--patch-size` | `8` | The patch grid must be >= 2 on each axis for the spatial Clifford block's depthwise convs. |
| `--embed-dim` | `128` | |
| `--encoder-clifford-depth` | `2` | |
| `--encoder-shifts` | `1 2` | |
| `--predictor-depth` | `2` | |
| `--predictor-num-heads` | `8` | |
| `--predictor-dim-head` | `16` | |
| `--predictor-mlp-dim` | `256` | |
| `--predictor-shifts` | `1 2` | |
| `--dropout-rate` | `0.0` | |
| `--sigreg-knots` | `17` | |
| `--sigreg-num-proj` | `1024` | `--smoke` drops it to 64. |
| `--sigreg-weight` | `0.09` | |
| `--mask-prediction-enabled` / `--no-mask-prediction-enabled` | enabled | Turn off to fall back to two-loss training. |
| `--mask-ratio` | `0.6` | Fraction of spatial patch positions masked. |
| `--lambda-next-frame` | `1.0` | Applied per horizon. |
| `--lambda-mask` | `1.0` | |
| `--ema-momentum` | `0.996` | EMA target encoder momentum. Strict bound `[0.0, 1.0)`. |
| `--ema-schedule` | `none` | `none` holds it; `cosine` ramps to 1.0 over the run. Logged as metric `ema_m`. |
| `--predict-horizons` | `1 4 15` | Strictly positive, sorted ascending, unique, `max(h) < T`. |
| `--val-steps` | `0` | Validation batches per epoch. 0 disables validation. BDD100K only. |
| `--val-fraction` | `0.1` | Fraction of BDD100K files held out, disjoint from train by seeded permutation. |
| `--early-stopping-patience` | `0` | 0 disables. |
| `--visualization-frequency` | `1` | Write visualization PNGs every N epochs. |

`--image-size`, `--patience`, `--lr-schedule` and `--show-plots` are inherited
from `create_base_argument_parser()` and **this script reads none of them**.
Use `--img-size` and `--early-stopping-patience` instead.

## What lands on disk

```
<output-dir>/
  training_log.csv       epoch + per-horizon, mask, sigreg losses (val_ twins if validating)
  last.keras             every epoch
  best.keras             save_best_only on val_loss, or loss when no validation
  final_model.keras      written after fit()
  training_curves/loss.png
  jepa_viz/epoch_NNN_mask_overlay.png, epoch_NNN_patch_error.png
```

After training the script reloads `final_model.keras` and asserts the forward
pass reproduces to `max|delta| < 1e-4`.

## Gotchas

- **I/O bounds real runs.** opencv random-frame seek on BDD100K MOV files
  saturates several CPU cores and leaves the GPU idle. Measured ~0.84 s/step at
  `B=4, T=8, 112^2`.
- **Checkpoint size is mostly optimizer state.** ~865k trainable parameters
  (~3.3 MB fp32), but `last.keras` / `final_model.keras` land around 10.7 MB
  because AdamW state rides along.
- **Masking only happens under `training=True`.** `TubeMaskGenerator` uses
  unseeded `keras.random.uniform`; if masks were applied at inference the model
  would be non-deterministic on its own forward pass. This is why
  `model(x, training=False)` is reproducible.
- **EMA is required, not an ablation.** With a live target encoder the
  per-horizon heads all collapse to the same value; multi-horizon JEPA at 30 fps
  needs the EMA target.
- **Pixel-space masking is impossible here.** The Clifford encoder needs an
  intact 2D patch grid, so encode-then-mask is the only path.
- No mixed precision and no multi-GPU support in this trainer.

## Tests

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_train/test_video_jepa/ \
    tests/test_models/test_video_jepa/ tests/test_callbacks/test_jepa_visualization.py -q
```
