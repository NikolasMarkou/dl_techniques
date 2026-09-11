# `train/omnipoint` — OmniPoint training pipeline

Trains `dl_techniques.models.vision.omnipoint.OmniPoint` (architecture details: that package's
own `README.md` — this file does not duplicate them) on locally available pinhole imagery: KITTI
depth-benchmark + MegaDepth. No custom `train_step` (repo hard invariant) — supervision goes
through a training-only wrapper's `self.add_loss()` (`decisions.md` D-020), because Keras 3.8's
tuple-output loss routing cannot express `OmniPointCombinedLoss`'s single-shared-scale
requirement across a whole tuple at once (see D-020 for the measured mechanism).

## THIS TRAINER DOES NOT TRAIN ON FISHEYE OR PANORAMA DATA

Every training and validation sample this pipeline consumes is pinhole imagery
(KITTI + MegaDepth). `generate_synthetic_camera_eval_set.py` in this same directory reprojects a
small subset of that pinhole data through the fisheye/equirectangular functions in
`dl_techniques.utils.camera_models` and stages the result to
`/media/arxwn/data0_4tb/datasets/omnipoint_synth/` — but this is separate, standalone
**validation-only** tooling, grep-confirmed to have zero references from `train_omnipoint.py`
(`decisions.md` step-10 report). Running it does not add fisheye/panorama samples to this
trainer's training loop. If you need to prove the non-pinhole code paths execute, run that script
or `tests/test_models/test_omnipoint/test_cross_camera_synthetic.py`, not this trainer.

## Running it

```bash
# what the CLI offers -- allocates nothing, touches no GPU, no dataset discovery
.venv/bin/python -m train.omnipoint.train_omnipoint --help

# a fast end-to-end sanity check: tiny image size, tiny batch, 1 epoch, a handful of
# files per source -- overrides --image-size/--batch-size/--epochs/--max-*-files/--workers
MPLBACKEND=Agg .venv/bin/python -m train.omnipoint.train_omnipoint --smoke --gpu 0

# a real run
MPLBACKEND=Agg .venv/bin/python -m train.omnipoint.train_omnipoint \
    --omnipoint-variant omnipoint_base --epochs 20 --gpu 0
```

Key flags (`--help` is authoritative; this is a summary):

| Flag | Purpose |
|---|---|
| `--kitti-root` / `--megadepth-root` | Dataset roots, or `none` to skip a source. Defaults point at this machine's local copies under `/media/arxwn/data0_4tb/datasets/`. |
| `--max-kitti-files` / `--max-megadepth-files` | Cap discovered pairs (smoke/dev runs). |
| `--omnipoint-variant` | `OmniPoint.MODEL_VARIANTS` key (`omnipoint_base` / `omnipoint_large`). |
| `--enable-conditioning` | Enables the optional intrinsics-ray-map / sparse-depth conditioning path (`OmniPoint`'s `enable_conditioning=True`). |
| `--lambda-ray` / `--lambda-metric` / `--lambda-normal` / `--lambda-local` / `--lambda-mask` | `OmniPointCombinedLoss` term weights. |
| `--smoke` | Tiny integration-smoke run — see above. |

Results land under repo-root `results/` (never `src/`), monitoring `val_loss`, per the repo's
standing training-script conventions.

## Data sources and the derived-intrinsics caveat (restated here, not just in the model README)

- **KITTI**: `src/train/omnipoint/kitti_depth.py` discovers real RGB+depth pairs under
  `/media/arxwn/data0_4tb/datasets/KITTI/data/depth/` — 92,554 matched pairs found across the
  `depth_values/`(train) and `val/` splits, both `image_02`/`image_03` cameras, measured at
  Step 7 execution time.
- **MegaDepth**: `src/train/omnipoint/data.py` adapts `train.common.megadepth`'s pair discovery
  and `.h5` depth decoding into the same `(rgb, depth, valid_mask, K)` contract KITTI uses, since
  MegaDepth's own `[-1,1]`-normalized loaders are unusable for ray/distance GT derivation.
  MegaDepth's raw depth values are its own MVS/SfM reconstruction scale, not calibrated
  real-world meters, and are used as-is with no rescaling attempt (`decisions.md` D-019) — a
  second, independent approximation on top of the one below.
- **Neither dataset stores real camera intrinsics colocated with its depth.** `K` is instead
  DERIVED per image from a fixed assumed horizontal field-of-view and the image's own
  width/height (`fx = W / (2*tan(FOV_h/2))`, square pixels, principal point at the image center)
  — `decisions.md` D-006. This is a documented approximation, applied identically to every KITTI
  and MegaDepth sample: every `gt_ray`/`gt_distance` value derived from this `K` inherits its
  systematic bias against each dataset's true, locally unrecoverable camera. **A reader of only
  this README, without the model package's own `README.md`, should take away the same warning:
  nothing this trainer produces should be read as trained against calibrated ground truth.**

## The mask-as-validity-proxy simplification

`MaskLoss`'s ground-truth target is each loader's own `valid_mask` (1.0 = a real, finite, nonzero
depth measurement was present; 0.0 = invalid) — not a semantic sky/foreground mask, because
neither KITTI nor MegaDepth carries any such annotation locally, and no zero-new-data alternative
exists (`decisions.md` D-017). The mask head therefore learns "was this pixel measurable" (a mix
of sensor dropout, occlusion and out-of-range geometry), not the paper's intended semantic sky
mask. This is a real, non-fabricated training signal, but a materially different one — do not
read a trained `MaskHead` output as a sky/foreground segmentation.

## See also

- `dl_techniques.models.vision.omnipoint`'s own `README.md` — architecture, construction,
  camera-agnosticism caveat, output-resolution and loss-simplification limitations.
- `decisions.md` in `plans/plan-2026-09-11T050223-1b47bcf6/` — D-006 (derived intrinsics),
  D-016 (planar-to-radial GT conversion), D-017 (mask proxy), D-018 (KITTI/MegaDepth mixing),
  D-019 (MegaDepth scale), D-020/D-021 (training-wrapper loss routing and GT downsampling).
