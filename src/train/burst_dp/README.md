# BurstDP training

Train [`dl_techniques.models.burst_dp.BurstDP`](../../dl_techniques/models/burst_dp/)
on COCO 2017 with on-the-fly synthetic auxiliary-view generation.

## Quick start

```bash
MPLBACKEND=Agg .venv/bin/python -m train.burst_dp.train_burst_dp \
    --preset burst_dp_small \
    --image-size 256 \
    --patch-size 16 \
    --n-max 5 --n-min 1 \
    --batch-size 4 \
    --epochs 40 \
    --coco-root /media/arxwn/data0_4tb/datasets/coco_2017 \
    --out-dir src/results/burst_dp/run01 \
    --gpu 0 \
    --mixed-precision
```

## Smoke run (a few images, fast)

```bash
MPLBACKEND=Agg .venv/bin/python -m train.burst_dp.train_burst_dp \
    --preset burst_dp_pico \
    --image-size 128 \
    --patch-size 16 \
    --n-max 3 --n-min 1 \
    --batch-size 2 \
    --epochs 1 \
    --max-train-images 64 --max-val-images 16 \
    --coco-root /media/arxwn/data0_4tb/datasets/coco_2017 \
    --out-dir src/results/burst_dp/smoke \
    --gpu 0
```

## Visualizations

The trainer saves a `(num_samples x 6)` recon + segmentation comparison
grid (ref / aux[0] / recon pred / recon target / seg pred / seg target)
under `out_dir/viz/` on the configured cadence:

- `--viz-every-steps N` (default `500`) — save every N optimizer steps,
  files `viz/step_NNNNNNN.png`. Pass `0` to disable.
- `--viz-every-epochs M` (default `1`) — save every M epochs, files
  `viz/epoch_NNNN.png`. Pass `0` to disable.
- `--viz-num-samples K` (default `4`) — rows per grid, capped to the
  val batch size.

Both triggers can fire independently; setting both to `0` skips the
callback entirely.

## Outputs

- `burst_dp_best.keras` — best validation checkpoint
- `burst_dp_final.keras` — final epoch
- `tb/` — TensorBoard logs (per-head losses)
- `viz/` — periodic recon + segmentation comparison PNGs
- `history.csv` — epoch metrics
- `run_config.json` — args + model config + distortion spec for reproduction
