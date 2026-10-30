# Mean shift experiments (Phase B)

A research harness that puts the Phase A mean shift components on real images and measures
them. It is not a trainer for a model package: there is no `models/` package behind it.
`research/2026_mean_shift_deep_learning.md` is the theory note; this directory is the
experiments.

- Experiment 1 compares Mode Pooling (`ModePooling`, one start and four starts) against mean
  pooling and CLS pooling on `vit_pico`, on clean, occluded and noisy test images.
- Experiment 2 compares MeanShift Attention against `multi_head` attention in a small ViT
  (`VisionEncoder`), with a 16-config ablation grid and an initializer control for the
  baseline.

Results are in the Results section at the end and, numbered, in `FINDINGS.md`.

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
| `exp2_mean_shift_attention.py` | Experiment 2: the 17-config table, the `mha_oinit` control (`CONTROL_CONFIGS`, outside the grid), driver and worker |
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
cap), `--cell-timeout-s` (default 7200; every run reported below used 2400),
`--gate-max-wait-hours`.

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

`--configs` also accepts the control `mha_oinit` (see the confounds below), which is not part of
the grid; `--mode grid` never runs it.

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

All four arms have the same parameter count (2,693,578 at 10 classes, as run on CIFAR-10;
2,710,948 at 100 classes).

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
  pre-norm, no CLS token). Nothing is claimed across the two experiments. In experiment 1 the
  arms built from one seed start from identical weights (checked for seed 1); in experiment 2
  they do not, see the next bullet.
- Initializer replay in `mha` (experiment 2, measured). `MultiHeadCrossAttention` reuses one
  seeded initializer instance for `qkv` and `proj`, so after `set_seeds(0)` the 192x192
  `proj/kernel` of every `mha` block is a rescaled copy of the first 36,864 values of its
  `qkv/kernel` (correlation 1.000000 in all six blocks). The MeanShift `projection` and
  `output_projection` are independent (largest absolute correlation 0.012 over the six blocks of
  the default and the two chosen configs). The control `mha_oinit` is `multi_head` with an
  independent output projection initializer (largest absolute correlation 0.0099). It scored the
  same as `mha` on every seed (section "Experiment 2: initializer control"), so this asymmetry
  does not explain the gap to MeanShift. The `mha` rows were left exactly as run.
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
  benchmark. Another process was resident on GPU 0 during 5 of the 9 main-run cells (idle-gate
  samples of 1378 to 1951 MiB in those cells against at most 1002 MiB in the other four and at
  most 866 MiB in every other sweep), within the gate rule. That affects the s/epoch and
  images/s of the main run and nothing else.
- The grid has 1 seed, so its ranking is weaker than the main run's; the best two configs may
  not carry over. The grid and the main run used the same 50 epochs.
- Learned bandwidths: `per_head` reports `softplus(raw) + min_bandwidth` per layer and head;
  `per_token` reports mean, min and max per layer over 512 selection images, read through the
  layer's own `_bandwidth` helper. A bandwidth within the clip range `[0.01, 100]` is
  "inside"; one at a bound is flagged in the report. "Inside" is a binary flag and says nothing
  about the distance to the floor: the lowest per-token bandwidth of a config sits as little as
  4.0e-6 above the 0.01 floor (grid, `blur_s3_per_token_gaussian`) and 4.1e-4 to 8.1e-4 above it
  in the main run (layer 0 on every seed). In the grid 5 of the 8 per-token configs have a
  minimum within 2.5e-3 of the floor; the other three are 1.8e-2 to 0.23 above it. The gradient reaching the raw
  parameter is scaled by `sigmoid(raw)`, which equals `1 - exp(-(h - 0.01))`, about the
  distance to the floor, so those tokens are saturated at the floor: for the vMF kernel
  `kappa = 1 / h^2` is about 9e3 there, hard nearest-key attention.
- Weight decay acts on the bandwidth parameters. The shared recipe passes AdamW weight decay
  0.05 with no `exclude_from_weight_decay` (`common.build_optimizer` calls
  `train.common.optimizer.build_optimizer`), so every trainable variable is decayed, including
  `raw_bandwidth` and, for `per_token`, the `bandwidth_projection` kernel and bias. The decay
  pulls the raw parameter toward 0, that is `h` toward `0.01 + softplus(0)` = 0.703: upward for
  the vMF configs that start at 0.3, downward for the Gaussian configs that start at 2.83. Its
  size against the gradient was not measured, so the learned bandwidths mix learning with that
  prior.
- Query and key normalisation (untested confound). The vMF kernel L2-normalises the queries and
  the keys, and the layer re-normalises the moved tokens after every step, which acts like QK
  normalisation; `mha` has none (`qk_norm_type` unset). Both chosen configs of the main run are
  vMF. The Gaussian configs do not normalise and still beat `mha` by 9.98 to 12.36 points in the
  grid, so normalisation is not the whole gap, but its share was not measured: the matching
  control is a `multi_head` arm with `qk_norm_type`, which was not run.
- Provenance. The manifests of the smoke, exp1, stability and grid runs record the git shas
  `59955ebec`, `65804ba43` and `f5fd449e2`, which are unreachable from the current history
  after another session rebased the branch; the rebased equivalents are `67b9564d9`,
  `458ca3bf7` and `c15384002`. A manifest records the sha but not whether the working tree was
  dirty. No file on the experiment code paths changed between the run shas and the rebased ones
  except docstring-only edits and the class token refactor of the `vit` family (`5e8f8124f`,
  made after exp1 finished, zeros initializer before and after; its message reports a maximum
  absolute output difference of 0.0). The code paths were checked equivalent; nothing was
  re-run.

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
  A `multi_head` variant other than `mha` (the `mha_oinit` control) is paired against `mha`
  like any variant but is not ranked.

## Tests

CPU only: `CUDA_VISIBLE_DEVICES= MPLBACKEND=Agg .venv/bin/python -m pytest -q
tests/test_train/test_mean_shift/`. The directory takes about ten minutes because the worker
tests train tiny models end to end.

| File | What it pins |
|---|---|
| `tests/test_train/test_mean_shift/test_corruption.py` | exact block counts (16 and 32 of 64), determinism per `(seed, condition)`, conditions differ, arm independence, noise std, clipping |
| `tests/test_train/test_mean_shift/test_cells.py` | the idle gate (3 consecutive samples, a failed `nvidia-smi` is busy), manifest first, fsynced rows, resume, retry in a new directory, manifest mismatch refused, no deleting call, import without TensorFlow |
| `tests/test_train/test_mean_shift/test_exp_workers.py` | the four exp1 arms and their equal parameter counts, paired corruption across arms, the selection split, the 17 exp2 configs, vMF `bandwidth_init`, the measured parameter ratio, the `mha_oinit` control (outside the grid; `mha` replays one draw, correlation above 0.999, the control and the MeanShift configs do not), `fit` validating on the selection split, bandwidth extraction on a toy layer, end-to-end CPU workers |
| `tests/test_train/test_mean_shift/test_report.py` | two arms by two seeds (a seed-only reader fails), paired arithmetic against hand values, pairing by seed not position, sample std, failed and missing cells, ranking on the selection split, strict JSON, a golden markdown table, no TensorFlow import |
| `tests/test_train/test_every_trainer_seeds_through_the_shared_helper.py` | classifies both experiment modules as multi-seed trainers and `report` as not training |

## Results

Every number in this section is copied from a `summary.json` written by `report.py` from the
finished sweep roots; the numbered record with the same numbers and the caveats is
`src/train/mean_shift/FINDINGS.md`. "Points" means percentage points of accuracy. Standard
deviations are sample stds over seeds (`ddof=1`). No significance test is applied: with 3
seeds the smallest possible two-sided sign-flip p is 0.25, so the tables give mean +- std and
the per-seed paired differences, and nothing more. Accuracies are final-epoch weights scored
once on the CIFAR test split (the `selection` column is the 5,000-image selection split).

### What was run

All runs used GPU 0, `MPLBACKEND=Agg`, the shared recipe above, and 50 epochs. The epoch count
came from the budget rule applied to a 2-epoch timing smoke (slowest exp1 epoch 5.39 s, mha
4.44 s, default MeanShift config 5.18 s); every one of the three budgets allowed the 50-epoch
cap, so no run was shortened.

| Run | Directory under `results/mean_shift/` | Cells | Epochs | Seeds |
|---|---|---|---|---|
| timing smoke, exp1 | `20261030_053319_smoke_exp1` | 2 | 2 | 0 |
| timing smoke, exp2 | `20261030_053319_smoke_exp2` | 2 | 2 | 0 |
| exp1, CIFAR-10 | `20261030_054304_exp1_cifar10` | 12 | 50 | 0, 1, 2 |
| exp2 stability smoke, CIFAR-100 | `20261030_065822_exp2_stability` | 2 | 5 | 0 |
| exp2 grid, CIFAR-100 | `20261030_070258_exp2_grid` | 17 | 50 | 0 |
| exp2 main, CIFAR-100 | `20261030_090740_exp2_main` | 9 | 50 | 0, 1, 2 |
| exp2 control, CIFAR-100 (`mha` and `mha_oinit`) | `20261030_135834_exp2_control` | 6 | 50 | 0, 1, 2 |

All cells of all seven sweeps finished with an `ok` row; there are no failed or missing cells.
The sum of the cell seconds (start-up included, gate waits excluded) is 1.21 h for
exp1, 1.80 h for the grid, 1.01 h for the main run and 0.63 h for the control.

```bash
# exp1: 4 arms x 3 seeds
MPLBACKEND=Agg .venv/bin/python -m train.mean_shift.exp1_mode_pooling \
    --out-dir results/mean_shift/20261030_054304_exp1_cifar10 --dataset cifar10 --epochs 50 \
    --cell-timeout-s 2400 --gpu 0
# exp2 stability smoke, then grid, then main (the two chosen configs from the grid)
MPLBACKEND=Agg .venv/bin/python -m train.mean_shift.exp2_mean_shift_attention \
    --mode smoke --out-dir results/mean_shift/20261030_065822_exp2_stability --epochs 5 \
    --cell-timeout-s 2400 --gpu 0
MPLBACKEND=Agg .venv/bin/python -m train.mean_shift.exp2_mean_shift_attention \
    --mode grid --out-dir results/mean_shift/20261030_070258_exp2_grid --epochs 50 \
    --cell-timeout-s 2400 --gpu 0
MPLBACKEND=Agg .venv/bin/python -m train.mean_shift.exp2_mean_shift_attention \
    --mode main --configs noblur_s3_per_token_vmf blur_s1_per_head_vmf \
    --out-dir results/mean_shift/20261030_090740_exp2_main --epochs 50 \
    --cell-timeout-s 2400 --gpu 0
# exp2 initializer control: mha is re-run too, a reproducibility check
MPLBACKEND=Agg .venv/bin/python -m train.mean_shift.exp2_mean_shift_attention \
    --mode main --configs mha_oinit --epochs 50 --dataset cifar100 --cell-timeout-s 2400 \
    --out-dir results/mean_shift/20261030_135834_exp2_control --gpu 0
```

The control command runs `mha` and `mha_oinit` for seeds 0, 1, 2 (`--mode main` always adds
the reference). The reports are in `results/mean_shift/20261030_reports/` (`exp1`, `exp2_grid`,
`exp2_main`, `exp2_control`, `exp2_stability`, `smoke`), each with `summary.json` and
`summary.md`. The grid, the main run and the control were reported separately because each
holds `mha` at seed 0 (reporting two of them together is refused).

Provenance and sharing, in short (details under "Conventions and confounds"): the manifests of
the first four sweeps name git shas that no longer exist in the history, and another process
was resident on GPU 0 during 5 of the 9 main-run cells, which only touches that run's
throughput figures.

### Experiment 1: Mode Pooling on vit_pico, CIFAR-10

Accuracy in percent, mean +- std over seeds 0, 1, 2 (`s/epoch` is the median over epochs 2 and
later). The parameter count is for 10 classes. The four arms have the same count.

| arm | seeds | params | s/epoch | clean | occ25 | occ50 | noise 0.05 | noise 0.1 | noise 0.2 | selection |
|---|---|---|---|---|---|---|---|---|---|---|
| mean | 3 | 2,693,578 | 5.18 +- 0.02 | 82.48 +- 0.16 | 63.33 +- 1.21 | 41.80 +- 0.82 | 74.71 +- 0.77 | 52.28 +- 3.04 | 22.98 +- 3.38 | 82.30 +- 0.32 |
| cls | 3 | 2,693,578 | 5.15 +- 0.05 | 81.35 +- 0.55 | 62.92 +- 1.11 | 45.76 +- 0.61 | 73.76 +- 1.26 | 51.67 +- 3.07 | 22.32 +- 1.76 | 81.14 +- 0.47 |
| mode | 3 | 2,693,578 | 5.33 +- 0.06 | 82.24 +- 0.27 | 63.29 +- 1.69 | 42.82 +- 2.08 | 74.68 +- 1.26 | 52.51 +- 2.15 | 22.19 +- 0.82 | 82.27 +- 0.44 |
| mode_s4 | 3 | 2,693,578 | 5.55 +- 0.05 | 82.24 +- 0.28 | 63.26 +- 1.65 | 42.80 +- 2.13 | 74.68 +- 1.25 | 52.51 +- 2.18 | 22.19 +- 0.81 | 82.27 +- 0.40 |

Paired difference against `mean`, seed by seed (arm minus `mean`, points):

| arm | condition | mean diff vs mean (points) | per-seed diffs, seeds 0, 1, 2 (points) |
|---|---|---|---|
| cls | clean | -1.13 | -0.46, -1.12, -1.81 |
| cls | occ25 | -0.41 | +0.00, -2.89, +1.66 |
| cls | occ50 | +3.96 | +4.60, +3.26, +4.03 |
| cls | noise0.05 | -0.95 | -0.30, -1.40, -1.14 |
| cls | noise0.1 | -0.61 | -0.99, -2.20, +1.36 |
| cls | noise0.2 | -0.66 | -3.39, -0.77, +2.19 |
| cls | select | -1.16 | -0.44, -1.06, -1.98 |
| mode | clean | -0.24 | +0.15, -0.27, -0.59 |
| mode | occ25 | -0.04 | -1.91, -0.67, +2.46 |
| mode | occ50 | +1.02 | +0.09, +0.32, +2.64 |
| mode | noise0.05 | -0.02 | +0.41, +0.09, -0.57 |
| mode | noise0.1 | +0.23 | -1.01, +1.45, +0.25 |
| mode | noise0.2 | -0.79 | -3.97, +0.24, +1.36 |
| mode | select | -0.03 | -0.28, +0.24, -0.04 |
| mode_s4 | clean | -0.23 | +0.17, -0.28, -0.59 |
| mode_s4 | occ25 | -0.07 | -1.89, -0.71, +2.40 |
| mode_s4 | occ50 | +1.00 | +0.02, +0.32, +2.66 |
| mode_s4 | noise0.05 | -0.03 | +0.39, +0.10, -0.58 |
| mode_s4 | noise0.1 | +0.23 | -0.97, +1.44, +0.22 |
| mode_s4 | noise0.2 | -0.79 | -3.97, +0.25, +1.35 |
| mode_s4 | select | -0.03 | -0.22, +0.16, -0.02 |

Reading, limited to what the numbers support:

- Mode Pooling matches mean pooling within the seed noise on every condition. The condition
  means differ by -0.79 to +1.02 points and the sign of the
  per-seed difference changes across seeds in 6 of the 7 metrics. The one
  exception to the sign changes is `occ50`, positive on all three seeds (+0.09, +0.32, +2.64
  points), but that mean of +1.02 rests on one seed.
- `mode_s4` (four starts) is the same as `mode` (one start): the largest difference in a
  condition mean is 0.03 points, and the largest in any single seed and metric is
  0.08 points.
- The only effect that holds on every seed is on the CLS arm: 3.26 to
  4.60 points better than `mean` at 50 % occlusion on all three seeds, at 0.46
  to 1.81 points lower clean accuracy (also lower on all three seeds, as is `noise0.05`). At 25 %
  occlusion and at the two larger noise levels the CLS differences change sign across seeds.
- A null result: no robustness advantage for Mode Pooling over mean pooling was found in this
  setting (clean-trained `vit_pico`, 50 epochs, black-block occlusion, clipped Gaussian noise).
  The following explanations of the null are hypotheses and were not tested: the final-layer
  tokens may already be near-uniform after the attention blocks have mixed them, so the token
  set has no minority outlier cluster for a mode to ignore; the median-heuristic bandwidth
  (`bandwidth="auto"`) may be wide enough that five steps barely move away from the mean; the
  occluded tokens may form a coherent second cluster rather than scattered outliers; and a model
  trained on clean images with the pooling in the loop may adapt its tokens to either pooling.
  The distance between the mode and the mean of the trained model's tokens was not measured.
  Why CLS holds up better under heavy occlusion is likewise untested.

### Experiment 2: MeanShift Attention in a small ViT, CIFAR-100

Model: `VisionEncoder`, patch 4, width 192, depth 6, 3 heads, plus a dense head. MeanShift
configs have 16.4 % (per head) and 16.3 % (per token)
fewer parameters than `mha` because Q, K and V are tied.

Stability smoke (`20261030_065822_exp2_stability`, 5 epochs, seed 0, the default config
`blur_s3_per_head_gaussian` against `mha`). Pre-registered pass conditions, all met:

| Condition | Result |
|---|---|
| no non-finite loss in any epoch | met in both runs (`finite` true, 5 epochs each) |
| epoch-5 training loss below epoch-1 | MeanShift 4.457 to 3.725; `mha` 4.333 to 3.065 |
| selection accuracy at epoch 5 at least 3 % | MeanShift 13.92 %; `mha` 25.60 % |
| learned bandwidths strictly inside the clip range [0.01, 100], none at a bound | per-head values 2.722 to 2.850, all six layers inside |

MeanShift learned more slowly than `mha` over these 5 epochs (selection accuracy 13.92 %
against 25.60 %). The per-head bandwidths barely moved from the Gaussian
initialization `head_dim ** 0.25` = 2.83.

Grid (`20261030_070258_exp2_grid`): 16 MeanShift configs plus `mha`, 1 seed, 50 epochs, ranked by
selection accuracy (the test column is shown, never used for the ranking). Images per second is
`predict` over the test split (a rough ratio, see Conventions).

| rank | config | params | s/epoch | images/s | selection | test | selection minus mha (points) | bandwidth min to max |
|---|---|---|---|---|---|---|---|---|
| ref | mha | 2,710,564 | 4.39 | 16945 | 43.26 | 44.13 | - | n/a |
| 1 | noblur_s3_per_token_vmf | 2,269,366 | 6.79 | 15346 | 57.08 | 57.38 | +13.82 | 0.011 to 10.344 |
| 2 | blur_s1_per_head_vmf | 2,265,910 | 5.14 | 14983 | 56.82 | 57.88 | +13.56 | 0.192 to 1.314 |
| 3 | noblur_s1_per_head_vmf | 2,265,910 | 5.11 | 15053 | 56.66 | 58.15 | +13.40 | 0.193 to 1.308 |
| 4 | noblur_s3_per_head_vmf | 2,265,910 | 7.01 | 14188 | 56.10 | 57.89 | +12.84 | 0.186 to 0.795 |
| 5 | blur_s3_per_token_vmf | 2,269,366 | 6.62 | 12161 | 55.76 | 56.25 | +12.50 | 0.012 to 14.286 |
| 6 | blur_s1_per_head_gaussian | 2,265,910 | 4.57 | 16548 | 55.62 | 56.64 | +12.36 | 1.800 to 4.527 |
| 7 | blur_s3_per_head_vmf | 2,265,910 | 6.51 | 12678 | 55.58 | 56.89 | +12.32 | 0.188 to 0.712 |
| 8 | noblur_s1_per_head_gaussian | 2,265,910 | 4.42 | 15974 | 55.28 | 56.05 | +12.02 | 1.799 to 4.515 |
| 9 | noblur_s1_per_token_gaussian | 2,269,366 | 4.90 | 16609 | 55.00 | 56.44 | +11.74 | 0.037 to 34.028 |
| 10 | blur_s1_per_token_vmf | 2,269,366 | 5.46 | 15035 | 54.92 | 56.96 | +11.66 | 0.012 to 8.725 |
| 11 | blur_s1_per_token_gaussian | 2,269,366 | 4.99 | 16341 | 54.70 | 56.17 | +11.44 | 0.028 to 33.326 |
| 12 | noblur_s1_per_token_vmf | 2,269,366 | 5.41 | 15521 | 54.70 | 57.02 | +11.44 | 0.012 to 8.215 |
| 13 | noblur_s3_per_head_gaussian | 2,265,910 | 5.93 | 14707 | 53.94 | 55.46 | +10.68 | 1.987 to 3.306 |
| 14 | noblur_s3_per_token_gaussian | 2,269,366 | 6.44 | 14941 | 53.76 | 55.81 | +10.50 | 0.241 to 16.298 |
| 15 | blur_s3_per_head_gaussian | 2,265,910 | 5.66 | 14531 | 53.68 | 55.55 | +10.42 | 1.924 to 3.238 |
| 16 | blur_s3_per_token_gaussian | 2,269,366 | 5.99 | 15817 | 53.24 | 54.98 | +9.98 | 0.010 to 34.899 |

Ranking caveat: one seed. The top four configs are within 0.98 points of selection
accuracy, about three times the spread of the one-step blurring and non-blurring twins, which
differ only after the first step (0.16 to 0.34 points on this seed), and about the size of the
seed std in the main run (0.40 to 0.91), so the order of the best few is weak. The two
configs for the main run were `noblur_s3_per_token_vmf` and `blur_s1_per_head_vmf`; there were
no ties: rank 1 to rank 2 is 0.26 points, and rank 2 and rank 3 are the one-step
blurring and non-blurring versions of the same config, which compute the same function at one
step, so their gap of 0.16 points is run-to-run noise.

Main (`20261030_090740_exp2_main`): `mha` and the two chosen configs, seeds 0, 1, 2, 50 epochs.
Percent, mean +- std over 3 seeds. "Final-epoch train accuracy" is Keras' running training
accuracy over the last epoch, on augmented batches while the weights were still changing, so it
is a rough measure and is used here only to compare the three configs.

| config | seeds | params | s/epoch | images/s | selection | test | final-epoch train accuracy |
|---|---|---|---|---|---|---|---|
| mha | 3 | 2,710,564 | 5.08 +- 0.21 | 13986 +- 620 | 42.81 +- 0.83 | 43.63 +- 0.48 | 76.33 +- 0.83 |
| blur_s1_per_head_vmf | 3 | 2,265,910 | 5.00 +- 0.09 | 15831 +- 1456 | 56.85 +- 0.40 | 57.97 +- 0.16 | 76.08 +- 0.27 |
| noblur_s3_per_token_vmf | 3 | 2,269,366 | 6.76 +- 0.13 | 14527 +- 212 | 56.57 +- 0.91 | 57.07 +- 0.55 | 80.49 +- 0.29 |

Paired difference against `mha`, seed by seed (config minus `mha`, points):

| config | metric | mean diff vs mha (points) | per-seed diffs, seeds 0, 1, 2 (points) |
|---|---|---|---|
| blur_s1_per_head_vmf | select | +14.05 | +13.02, +14.26, +14.86 |
| blur_s1_per_head_vmf | test | +14.33 | +14.00, +14.67, +14.33 |
| noblur_s3_per_token_vmf | select | +13.76 | +13.66, +13.98, +13.64 |
| noblur_s3_per_token_vmf | test | +13.44 | +13.61, +13.53, +13.18 |

Learned bandwidths in the main run (mean over seeds and heads per layer; for `per_token`, over
512 selection images as well; initial value `bandwidth_init` = 0.3 for every vMF config; all
values inside the clip range [0.01, 100]):

| config | mode | per-layer mean, layers 0 to 5 | min | max |
|---|---|---|---|---|
| blur_s1_per_head_vmf | per_head | 0.303, 0.523, 0.599, 0.813, 1.161, 1.284 | 0.192 | 1.309 |
| noblur_s3_per_token_vmf | per_token | 2.654, 1.174, 0.876, 2.464, 3.159, 2.787 | 0.010 | 18.594 |

The `per_head` config moved from 0.3 in layer 0 to 1.284 in layer 5; the three
seeds agree to within 0.027 on each layer mean. The `per_token` config has a wide
spread of per-token bandwidths within the run (smallest 0.010, largest
18.594) and a layer 0 that differs by seed: the layer-0 mean is 0.152,
3.851 and 3.960 on seeds 0, 1, 2. In seeds 1 and 2 one layer-0 head
has a mean per-token bandwidth of 11.29 and 11.61; in seed 0 no
layer-0 head is above 0.33.

#### Experiment 2: initializer control

Control (`20261030_135834_exp2_control`, 6 cells, seeds 0, 1, 2, 50 epochs, CIFAR-100, GPU 0
under the idle gate, the same recipe): `mha_oinit` is `multi_head` with
`attention_args={"output_kernel_initializer": "glorot_uniform"}`, which builds an independent
initializer for `proj` so that it no longer replays the `qkv` draw (see Conventions). `mha` was
re-run in the same root, which doubles as a reproducibility check of the main run. Percent, mean
+- std over 3 seeds; per-seed values in the order seeds 0, 1, 2.

| config | seeds | params | s/epoch | selection | test | final-epoch train accuracy |
|---|---|---|---|---|---|---|
| mha (rerun) | 3 | 2,710,564 | 5.02 +- 0.31 | 42.71 +- 0.91 | 43.74 +- 0.57 | 76.40 +- 0.85 |
| mha_oinit | 3 | 2,710,564 | 4.96 +- 0.26 | 42.74 +- 0.70 | 43.77 +- 0.28 | 76.46 +- 0.58 |

| config | selection, seeds 0, 1, 2 | test, seeds 0, 1, 2 | final-epoch train accuracy, seeds 0, 1, 2 |
|---|---|---|---|
| mha (rerun) | 43.46, 42.98, 41.70 | 44.35, 43.23, 43.63 | 75.69, 77.35, 76.16 |
| mha_oinit | 42.70, 43.46, 42.06 | 43.79, 44.03, 43.48 | 76.52, 77.00, 75.85 |

| comparison (first minus second) | metric | mean diff (points) | per-seed diffs, seeds 0, 1, 2 (points) |
|---|---|---|---|
| mha_oinit minus mha rerun | select | +0.03 | -0.76, +0.48, +0.36 |
| mha_oinit minus mha rerun | test | +0.03 | -0.56, +0.80, -0.15 |
| mha rerun minus mha main run | select | -0.09 | -0.04, -0.06, -0.18 |
| mha rerun minus mha main run | test | +0.10 | +0.27, +0.11, -0.07 |

- The independent output initializer does not change the baseline: `mha_oinit` is within noise
  of `mha` on every seed (selection differences -0.76 to +0.48 points, test -0.56 to +0.80, both
  with changing signs, means +0.03), at the same train accuracy. The shared-initializer artefact
  therefore does NOT explain the 13 to 14 point gap to the MeanShift configs. Against
  `mha_oinit` the two chosen configs are still 13.46 to 14.68 points ahead on selection and
  12.62 to 14.55 on test, on every seed.
- The rerun of `mha` reproduces the main run: the per-seed selection accuracy differs by 0.04,
  0.06 and 0.18 points and the test accuracy by at most 0.27. The baseline rows of the main
  run and the grid were left as run; the rerun is reported beside them, not instead.
- Three seeds and a noise floor of a few tenths of a point: the control excludes an
  initializer effect of the size of the gap, not one of a few tenths.

Reading, limited to what the numbers support:

- In this recipe all 16 MeanShift configs of the grid beat `mha` on selection accuracy by
  9.98 to 13.82 points, at 16 % fewer parameters. The main run repeats it
  for the two chosen configs on three seeds: 13.02 to 14.86
  points on selection and 13.18 to 14.67 on the test split, with the
  same sign on every seed. This headline now includes the initializer control and the baseline
  rerun above: with an independent output initializer the baseline is unchanged and the gap
  stays at 13 to 14 points.
- Of the two chosen configs neither is ahead: selection 56.85 against 56.57 %
  (std 0.40 and 0.91), the opposite order from the 1-seed grid.
- `mha` reaches 76.33 % running training accuracy and scores 42.81 % on
  selection, a gap of 33.5 points. `blur_s1_per_head_vmf` has about the same training
  accuracy (76.08 %) with 56.85 % on selection (gap 19.2);
  `noblur_s3_per_token_vmf` trains higher (80.49 %) and reaches 56.57 %
  (gap 23.9). On this rough measure the advantage of `blur_s1_per_head_vmf` is in the gap
  between training and selection accuracy, not in a better fit of the training data;
  `noblur_s3_per_token_vmf` fits the training data better as well.
- Cost: `noblur_s3_per_token_vmf` takes 6.76 s per epoch against 5.08 for
  `mha`; `blur_s1_per_head_vmf` 5.00.
- Within the grid, the vMF kernel beat its Gaussian twin in 7 of 8 matched pairs, by 0.22 to
  3.32 points (one pair -0.30). That mixes the kernel with the initialization: every vMF config
  starts at `bandwidth_init` 0.3 and the Gaussian configs at the layer default 2.83.

What this does not show:

- The baseline recipe is untuned: no dropout, one learning rate and one weight decay for every
  arm, no baseline-specific tuning. A `multi_head` ViT with a tuned recipe may close some or all
  of the gap. The large training-to-selection gap of `mha` above is consistent with an
  under-regularized baseline; that was not tested.
- MeanShift has fewer parameters and tied Q, K, V; a regularization effect is not excluded. The
  initializer replay of `mha` is excluded as the explanation by the control above. The vMF
  configs also use `bandwidth_init` 0.3 against the Gaussian default and normalise queries and
  keys, which `mha` does not (untested, see Conventions); weight decay also acts on the
  learned bandwidths.
- One dataset (CIFAR-100), a small ViT, 50 epochs, 3 seeds, no significance test.
- Nothing is claimed across experiment 1 and experiment 2: the backbones differ.

### What was not done

No hyperparameter tuning of the baseline or of any arm, no dataset other than CIFAR-10 (exp1)
and CIFAR-100 (exp2), no budget longer than 50 epochs, no more than 3 seeds (1 for the grid), no
training-time occlusion or noise augmentation for exp1, no measurement of the mode-to-mean
distance of the trained models, no ablation separating the kernel from its bandwidth
initialization, no `multi_head` arm with QK normalisation, no run with the bandwidth parameters
exempt from weight decay, and no significance test. The empirical Tweedie denoiser oracle was run on CPU
and is documented in `src/train/bfunet/FINDINGS.md` section 10; it compares no trained denoiser.
