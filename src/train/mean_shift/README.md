# Mean shift experiments (Phase B)

A research harness that puts the Phase A mean shift components on real images and measures
them. It is not a trainer for a model package: there is no `models/` package behind it.
`research/2026_mean_shift_deep_learning.md` is the theory note; this directory is the
experiments.

- Experiment 1 compares Mode Pooling (`ModePooling`, one start and four starts) against mean
  pooling and CLS pooling on `vit_pico`, on clean, occluded and noisy test images.
- Experiment 2 compares MeanShift Attention against `multi_head` attention in a small ViT
  (`VisionEncoder`), with a 16-config ablation grid.

Results are not in this file yet. See the Results section at the end.

## Shape of the harness

Each run is a sweep of cells. A cell is one trained model, identified by the tuple
`(experiment, arm, config, seed)`. Every reader of the rows uses that whole tuple, never the
seed alone: two arms share their seeds, so a reader keyed on the seed alone silently merges
them.

| File | Role |
|---|---|
| `common.py` | Cell runner (`run_sweep`), idle-GPU gate, manifest, `rows.jsonl`, resume, the shared recipe, the worker helpers. Imports without TensorFlow. |
| `corruption.py` | `occlude` and `add_noise`, pure numpy, keyed by `(seed, condition)` |
| `exp1_mode_pooling.py` | Experiment 1: driver and worker in one module |
| `exp2_mean_shift_attention.py` | Experiment 2: the 17-config table, driver and worker |
| `report.py` | Aggregation of `rows.jsonl` into `summary.json` and `summary.md`. Imports without TensorFlow. |

Each experiment module is both the sweep driver (default) and the cell worker (`--worker`).
The driver runs one subprocess per cell, so a failure, an XLA state or a seed state never
leaks into the next cell, and so the idle gate can run before every cell. A cell costs about
90 s of start-up on top of its epochs.

### Layout of a sweep root

```
results/mean_shift/<timestamp>_<name>/
    manifest.json    written first: the full cell list, resolved epochs, recipe, recipe hash, git sha, env
    rows.jsonl       one fsynced line per finished cell (status ok or failed)
    cells/
        <experiment>__<arm>__<config>__s<seed>__try<k>/
            cell.log     the command, CUDA_VISIBLE_DEVICES, the gate samples, then the worker output
            result.json  written by the worker: params, history, seconds per epoch, metrics
```

A row carries the key, `status`, `attempt`, `cell_dir`, `seconds`, the `device` string, the
`gate` samples, `returncode`, `metrics` (the worker's `result.json`) and `error_tail`. A
failed cell appends a `failed` row with the tail of its log. A killed sweep resumes at cell
level with `--resume <root>`: cells with an `ok` row are skipped, the rest run in a new
`__try<k>` directory. A resume with a different cell list or recipe hash is refused. No
directory is ever deleted, and nothing under `results/` is ever overwritten.

## GPU and gate rules

- GPU 0 only. The driver accepts `--gpu 0` or no `--gpu`, and refuses any other value. It
  also refuses an exported `CUDA_VISIBLE_DEVICES` other than `0`, because `train.common.gpu`
  counts `--gpu` within an export and `1` would select the broken GPU 1.
- One GPU job at a time. Before every cell the driver waits until GPU 0 is idle: 3
  consecutive `nvidia-smi -i 0` samples, each at most 2048 MiB used and at most 20 %
  utilization. A refusal sleeps 60 s. A failed `nvidia-smi`, a hung one or an unparsable line
  counts as busy. The cumulative wait is capped (`--gate-max-wait-hours`, default 6) and then
  the sweep raises. A foreign job that starts mid-cell is not detected until the next cell
  boundary.
- The driver process itself holds no GPU (it hides them from itself); only the children get
  `CUDA_VISIBLE_DEVICES=0`.
- `--cpu` runs the cells on CPU without the gate. It exists for tests.

## Usage

Run from the repository root so a relative `--out-dir` lands in the repo-root `results/`.
Always set `MPLBACKEND=Agg`. The launch recipe, with `--gpu 0` explicit:

```bash
MPLBACKEND=Agg .venv/bin/python -m train.mean_shift.exp1_mode_pooling \
    --out-dir results/mean_shift/<timestamp>_exp1 --dataset cifar10 --epochs <E> --gpu 0
```

Use a new `--out-dir` for every run. To continue an interrupted sweep:

```bash
MPLBACKEND=Agg .venv/bin/python -m train.mean_shift.exp1_mode_pooling \
    --resume results/mean_shift/<timestamp>_exp1 --dataset cifar10 --epochs <E> --gpu 0
```

The resume must repeat the original `--dataset`, `--epochs`, `--arms` and `--seeds`, or the
manifest check refuses it.

Experiment 1 flags: `--arms` (default `mean cls mode mode_s4`), `--seeds` (default `0 1 2`),
`--dataset` (`cifar10` default, `cifar100`), `--epochs` (required), `--max-samples` (smoke
cap), `--cell-timeout-s`, `--gate-max-wait-hours`.

Experiment 2 has three modes, all with `--dataset cifar100` by default:

```bash
# stability smoke: mha and the default MeanShift config, 1 seed
MPLBACKEND=Agg .venv/bin/python -m train.mean_shift.exp2_mean_shift_attention \
    --mode smoke --out-dir results/mean_shift/<timestamp>_exp2_smoke --epochs 5 --gpu 0

# grid: mha and the 16 MeanShift configs, 1 seed, cheapest first
MPLBACKEND=Agg .venv/bin/python -m train.mean_shift.exp2_mean_shift_attention \
    --mode grid --out-dir results/mean_shift/<timestamp>_exp2_grid --epochs <E_abl> --gpu 0

# main: mha and the chosen configs, 3 seeds
MPLBACKEND=Agg .venv/bin/python -m train.mean_shift.exp2_mean_shift_attention \
    --mode main --configs <config id> <config id> \
    --out-dir results/mean_shift/<timestamp>_exp2_main --epochs <E2> --gpu 0
```

Aggregate finished roots (a report never overwrites an existing one):

```bash
.venv/bin/python -m train.mean_shift.report \
    --root results/mean_shift/<timestamp>_exp1 --out results/mean_shift/<timestamp>_exp1_report
```

`--root` repeats. Two roots that hold the same ok cell are refused: report the exp2 grid and
the exp2 main run separately, because both contain `mha` at seed 0.

## The recipe

Shared by every arm of every experiment (`common.Recipe`), so the only difference between
arms is the thing under test:

- AdamW, learning rate 3e-4, weight decay 0.05, batch 128, gradient clip 1.0
- linear warmup then cosine decay; the warmup is a tenth of the run, between 1 and 5 epochs
- stock `fit`, float32 policy (refused otherwise), no custom `train_step`
- no early stopping, no analyzer callbacks, no best-checkpoint restore: the weights scored
  are the FINAL-epoch weights
- data through `train.common.classification_data` only, `normalize="unit"` (model input equals
  the `[0, 1]` pixel domain), the `cifar_light` augmentation (flip, 4-pixel reflect pad crop,
  brightness 0.1, contrast 0.9 to 1.1)
- the epoch count is set per run from a measured timing smoke, not copied from the 50-epoch
  precedent; it is recorded in the manifest (`resolved.epochs`) and in the report

Experiment 1 model: `create_vit("vit_pico", ..., input_shape=(32, 32, 3), patch_size=4,
include_top=False, pooling=None, dropout_rate=0.1)`, giving a `[B, 65, D]` sequence with the
CLS token at position 0, then one pooling head and a `Dense` logits layer. Patch size 4 is
passed explicitly: the `vit_pico` default of 16 silently trains on 4 tokens at 32x32.

| Arm | Pooling |
|---|---|
| `mean` | `SequencePooling("mean", exclude_positions=[0])`, the reference |
| `cls` | `SequencePooling("cls")` |
| `mode` | `ModePooling(num_steps=5, bandwidth="auto", exclude_positions=[0], num_starts=1)` |
| `mode_s4` | the same with `num_starts=4` |

All four arms have the same parameter count (2,710,948 at 100 classes).

Experiment 2 model: `VisionEncoder(img_size=32, patch_size=4, embed_dim=192, depth=6,
num_heads=3, mlp_ratio=4, use_cls_token=False, output_mode="mean",
normalization_position="pre")` plus a `Dense` logits layer. The configs are the table
`CONFIGS` in `exp2_mean_shift_attention.py`: `mha` (`multi_head`, the reference) and 16
MeanShift configs, the product of blurring {off, on}, steps {1, 3}, bandwidth mode
{`per_head`, `per_token`} and kernel {`gaussian`, `vmf`}. Config ids read
`<blur|noblur>_s<steps>_<mode>_<kernel>`. The default config is
`blur_s3_per_head_gaussian`. Every vMF config sets `bandwidth_init=0.3`; Gaussian configs use
the layer default.

Test-time conditions of experiment 1: `clean`, `occ25`, `occ50`, `noise0.05`, `noise0.1`,
`noise0.2`.

## Conventions and confounds

Read these before comparing any two numbers.

- Parameters are reported, not matched, in experiment 2. Tied Q, K and V make the MeanShift
  stack about 16 % smaller than `multi_head` at the same width and depth (2,265,910 against
  2,710,564 parameters at 100 classes). Matching them would change width or depth, which is a
  second variable. Any gap between `mha` and a MeanShift config mixes the attention
  mechanism with that capacity difference.
- Experiment 1 and experiment 2 use different backbones: `vit_pico` (truncated-normal
  initializer, post-norm, CLS token) against `VisionEncoder` (the `glorot_uniform` default,
  pre-norm, no CLS token). Nothing is claimed across the two experiments. Within each
  experiment the initializer is identical in every arm.
- In experiment 1 the mean and mode arms drop the CLS token from the pooled set, but it still
  runs through attention, so those arms are not "CLS-free" models.
- vMF bandwidth: the layer default makes the vMF kernel nearly uniform on unit vectors, so
  every vMF config starts at `bandwidth_init=0.3`. A vMF result is a result for that
  initialization, not for the layer default.
- Selection against test: the selection split is a seeded 10 % slice of TRAIN that the loader
  holds out (5,000 CIFAR images). Every ranking and every choice between configs reads the
  selection split only. The test split (10,000 images) only scores, once, after the final
  epoch. The `exp2` grid ranking and the choice of the two configs for the main run use the
  selection accuracy, ties broken by fewer steps and then by config id.
- Occlusion zeroes `round(fraction * 64)` distinct 4x4-pixel blocks of every image, 16 blocks
  at 25 % and 32 at 50 %, in the model input domain. Because the input is in `[0, 1]`, zero is
  black. Each image draws its own blocks. This is one convention among several (a mean-colour
  fill would give other absolute accuracies).
- Noise is Gaussian with the stated sigma, added in the `[0, 1]` pixel domain and then
  clipped to `[0, 1]`, so inputs stay in the training range. The clipping makes the effective
  noise smaller than the nominal sigma near the range ends.
- The corruption random generator is keyed by `(seed, condition)` and nothing else, so every
  arm of a seed scores the same occluded and noisy images.
- Throughput in experiment 2 is images per second of `predict` over the test split after one
  untimed pass, on whatever else GPU 0 was doing; it is a rough ratio between configs, not a
  benchmark.
- The grid uses a short epoch budget and 1 seed, so its ranking is weaker than the main
  run's; the best two configs may not carry over.
- Learned bandwidths: `per_head` reports `softplus(raw) + min_bandwidth` per layer and head;
  `per_token` reports mean, min and max per layer over 512 selection images, read through the
  layer's own `_bandwidth` helper. A bandwidth within the clip range `[0.01, 100]` is
  "inside"; one at a bound is flagged in the report.

## Reading the report

`report.py` reads `rows.jsonl` keyed by the full tuple and writes `summary.json` (strict
JSON, no NaN tokens) and `summary.md`.

- Mean and std over seeds per condition, with the explicit `n_seeds` beside each cell. The std
  is the sample std (`ddof=1`, Bessel), as `train.common.stats.mean_std` computes it; it is
  `null` (printed `n/a`) for a cell with one seed.
- Paired differences against the reference (`mean` in experiment 1, `mha` in experiment 2),
  formed seed by seed over the seeds both variants finished, with the per-seed values listed.
- No significance test. With 3 seeds an exact sign-flip test has a minimum two-sided p of
  0.25, so the report states mean, std and per-seed paired differences and nothing more.
- Failed cells (a `failed` row and no `ok` row) and missing cells (in the manifest, no row)
  are listed, never dropped.
- Parameters, seconds per epoch (the median over epochs 2 and later, since epoch 1 carries the
  one-off trace) and, for experiment 2, throughput.
- Experiment 2: the learned bandwidths per config and the grid ranking by selection accuracy.

## Tests

CPU only: `CUDA_VISIBLE_DEVICES= MPLBACKEND=Agg .venv/bin/python -m pytest -q
tests/test_train/test_mean_shift/`. The directory takes about ten minutes because the worker
tests train tiny models end to end.

| File | What it pins |
|---|---|
| `tests/test_train/test_mean_shift/test_corruption.py` | exact block counts (16 and 32 of 64), determinism per `(seed, condition)`, conditions differ, arm independence, noise std, clipping |
| `tests/test_train/test_mean_shift/test_cells.py` | the idle gate (3 consecutive samples, a failed `nvidia-smi` is busy), manifest first, fsynced rows, resume, retry in a new directory, manifest mismatch refused, no deleting call, import without TensorFlow |
| `tests/test_train/test_mean_shift/test_exp_workers.py` | the four exp1 arms and their equal parameter counts, paired corruption across arms, the selection split, the 17 exp2 configs, vMF `bandwidth_init`, the measured parameter ratio, bandwidth extraction on a toy layer, end-to-end CPU workers |
| `tests/test_train/test_mean_shift/test_report.py` | two arms by two seeds (a seed-only reader fails), paired arithmetic against hand values, pairing by seed not position, sample std, failed and missing cells, ranking on the selection split, strict JSON, a golden markdown table, no TensorFlow import |
| `tests/test_train/test_every_trainer_seeds_through_the_shared_helper.py` | classifies both experiment modules as multi-seed trainers and `report` as not training |

## Results

Empty. The runs are launched after the harness is verified, and this section, a findings
file and the research note are filled in from the finished `summary.json` files only. No
number is quoted here before then.
