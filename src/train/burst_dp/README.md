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

## Datasets

`--dataset {coco,div2k,vggface2}` selects the training source. Default is
`coco` and is bit-for-bit back-compatible with the previous behavior.

| Dataset    | Mode         | Source layout                                                 | Heads trained         |
|------------|--------------|---------------------------------------------------------------|-----------------------|
| `coco`     | multi-task   | `<coco-root>/{train2017,val2017,annotations}/...`             | `recon` + `segmentation` |
| `div2k`    | fidelity-only| `<div2k-root>/{train,validation}/*.png` (800 / 100 images)    | `recon` (seg head zero-weighted) |
| `vggface2` | fidelity-only| `<vggface2-root>/{train_list.txt,test_list.txt,train/,test/}` | `recon` (seg head zero-weighted) |

**Fidelity-only mode**: DIV2K and VGG-Face2 lack segmentation labels.
The `BurstDP` model is hardcoded dual-head, so we keep the segmentation
head in the graph but zero its loss weight and drop its metrics. The
seg head still runs forward + backward but contributes nothing to the
total loss — a ~5-15% compute overhead in exchange for zero blast
radius on the model / config / saved-checkpoint surface. If you set
`--loss-seg` to a non-zero value with a fidelity-only dataset, the
trainer logs a warning and forces it to `0.0`.

Example invocations:

```bash
# DIV2K, single GPU, full 800 train images
MPLBACKEND=Agg .venv/bin/python -m train.burst_dp.train_burst_dp \
    --dataset div2k \
    --div2k-root /media/arxwn/data0_4tb/datasets/div2k \
    --preset burst_dp_small --image-size 256 --patch-size 16 \
    --n-max 5 --n-min 1 --batch-size 4 --epochs 40 \
    --out-dir src/results/burst_dp/div2k_run01 --gpu 0 --mixed-precision
```

```bash
# VGG-Face2 — always cap train/val with --max-*-images
# (3.1M training JPGs otherwise)
MPLBACKEND=Agg .venv/bin/python -m train.burst_dp.train_burst_dp \
    --dataset vggface2 \
    --vggface2-root /media/arxwn/data0_4tb/datasets/VGG-Face2/data \
    --preset burst_dp_small --image-size 256 --patch-size 16 \
    --n-max 5 --n-min 1 --batch-size 4 --epochs 10 \
    --max-train-images 100000 --max-val-images 5000 \
    --out-dir src/results/burst_dp/vggface2_run01 --gpu 0 --mixed-precision
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
