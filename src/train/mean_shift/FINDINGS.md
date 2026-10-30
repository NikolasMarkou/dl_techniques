# Mean shift experiments (Phase B) - findings

**Question this answers:** when the Phase A mean shift components are put on real images, does
Mode Pooling (`ModePooling`) make a ViT classifier more robust to occlusion and noise than mean
pooling, and does MeanShift Attention (`MeanShiftAttention`) do as well as `multi_head`
attention in a small ViT?

**Short answer:**
- Mode Pooling: no. It matches mean pooling within the seed noise on clean, occluded and noisy
  CIFAR-10 test images (`vit_pico`, 3 seeds). One start and four starts are the same. This is a
  null result in this setting.
- MeanShift Attention: in this recipe every one of the 16 configs of the grid scored
  9.98 to 13.82 points above `multi_head` on CIFAR-100 selection accuracy
  at 16 % fewer parameters, and the two configs of the 3-seed main run scored
  13.02 to 14.86 points above it. The baseline recipe is
  untuned, so this is a statement about this recipe, not about the attention mechanisms. The
  numbers now include two checks of the baseline: a rerun of `mha` (it reproduces the main run
  to within 0.18 points of selection accuracy) and a control with an independent output
  projection initializer (`mha_oinit`, same accuracy as `mha` on every seed within noise), so
  the initializer replay found in `mha` does not explain the gap (section 6).

Every number below is copied from the `summary.json` of a `report.py` run (the reports are in
`results/mean_shift/20261030_reports/`). "Points" means percentage points of accuracy; standard
deviations are sample stds over seeds; no significance test is applied (3 seeds cannot go below
a two-sided sign-flip p of 0.25). The harness, the recipe and the confounds are in
`src/train/mean_shift/README.md`.

---

## 1. What was run

All runs: GPU 0, 50 epochs (the cap; the budget rule applied to a 2-epoch timing smoke allowed
it for every run), the shared recipe in the README, final-epoch weights scored once on the test
split. All cells finished `ok`; none failed or is missing.

| Run | Directory under `results/mean_shift/` | Cells | Epochs | Seeds |
|---|---|---|---|---|
| timing smoke, exp1 and exp2 | `20261030_053319_smoke_exp1`, `20261030_053319_smoke_exp2` | 2 + 2 | 2 | 0 |
| exp1, CIFAR-10 | `20261030_054304_exp1_cifar10` | 12 | 50 | 0, 1, 2 |
| exp2 stability smoke | `20261030_065822_exp2_stability` | 2 | 5 | 0 |
| exp2 grid, CIFAR-100 | `20261030_070258_exp2_grid` | 17 | 50 | 0 |
| exp2 main, CIFAR-100 | `20261030_090740_exp2_main` | 9 | 50 | 0, 1, 2 |
| exp2 control, CIFAR-100 | `20261030_135834_exp2_control` | 6 | 50 | 0, 1, 2 |

Commands (repo root, `MPLBACKEND=Agg .venv/bin/python -m ...`, each with `--gpu 0` and
`--cell-timeout-s 2400`, which every cell used; the default is 7200):
`train.mean_shift.exp1_mode_pooling --dataset cifar10 --epochs 50`;
`train.mean_shift.exp2_mean_shift_attention --mode smoke --epochs 5`, `--mode grid --epochs 50`
and `--mode main --configs noblur_s3_per_token_vmf blur_s1_per_head_vmf --epochs 50`, each with
its own `--out-dir`; the control is
`train.mean_shift.exp2_mean_shift_attention --mode main --configs mha_oinit --epochs 50 --dataset cifar100 --cell-timeout-s 2400`
(it runs `mha` as well). Reports:
`.venv/bin/python -m train.mean_shift.report --root <run dir> --out results/mean_shift/20261030_reports/<name>`
with one root per report (`exp1`, `exp2_grid`, `exp2_main`, `exp2_control`, `exp2_stability`,
`smoke`); the grid, the main run and the control each hold `mha` at seed 0, so they cannot be
reported together.

Provenance and sharing:
- The manifests of the smoke, exp1, stability and grid runs record the git shas `59955ebec`,
  `65804ba43` and `f5fd449e2`, which are unreachable from the current history after another
  session rebased the branch (the rebased equivalents are `67b9564d9`, `458ca3bf7` and
  `c15384002`). A manifest records the sha but not whether the working tree was dirty. No file
  on the experiment code paths changed between the run shas and the rebased ones except
  docstring-only edits and the class token refactor of the `vit` family (`5e8f8124f`, after
  exp1 finished; zeros initializer before and after, maximum absolute output difference 0.0 in
  its message). The code paths were checked equivalent; nothing was re-run.
- Another process was resident on GPU 0 during 5 of the 9 main-run cells (idle-gate samples of
  1378 to 1951 MiB in those cells against at most 1002 MiB in the other four and at most 866 MiB
  in every other sweep), within the gate rule. It affects only the s/epoch and images/s figures
  of the main run, not the accuracies.

## 2. Experiment 1: Mode Pooling matches mean pooling

Accuracy in percent, mean +- std over seeds 0, 1, 2 (`s/epoch`: median over epochs 2 and later;
parameters for 10 classes). Directory `results/mean_shift/20261030_054304_exp1_cifar10`.

| arm | seeds | params | s/epoch | clean | occ25 | occ50 | noise 0.05 | noise 0.1 | noise 0.2 | selection |
|---|---|---|---|---|---|---|---|---|---|---|
| mean | 3 | 2,693,578 | 5.18 +- 0.02 | 82.48 +- 0.16 | 63.33 +- 1.21 | 41.80 +- 0.82 | 74.71 +- 0.77 | 52.28 +- 3.04 | 22.98 +- 3.38 | 82.30 +- 0.32 |
| cls | 3 | 2,693,578 | 5.15 +- 0.05 | 81.35 +- 0.55 | 62.92 +- 1.11 | 45.76 +- 0.61 | 73.76 +- 1.26 | 51.67 +- 3.07 | 22.32 +- 1.76 | 81.14 +- 0.47 |
| mode | 3 | 2,693,578 | 5.33 +- 0.06 | 82.24 +- 0.27 | 63.29 +- 1.69 | 42.82 +- 2.08 | 74.68 +- 1.26 | 52.51 +- 2.15 | 22.19 +- 0.82 | 82.27 +- 0.44 |
| mode_s4 | 3 | 2,693,578 | 5.55 +- 0.05 | 82.24 +- 0.28 | 63.26 +- 1.65 | 42.80 +- 2.13 | 74.68 +- 1.25 | 52.51 +- 2.18 | 22.19 +- 0.81 | 82.27 +- 0.40 |

Paired differences against `mean` (arm minus `mean`, per seed 0, 1, 2, points), the rows that
carry the findings:

| arm | condition | mean diff | per-seed diffs |
|---|---|---|---|
| cls | clean | -1.13 | -0.46, -1.12, -1.81 |
| cls | occ25 | -0.41 | +0.00, -2.89, +1.66 |
| cls | occ50 | +3.96 | +4.60, +3.26, +4.03 |
| cls | noise0.05 | -0.95 | -0.30, -1.40, -1.14 |
| mode | clean | -0.24 | +0.15, -0.27, -0.59 |
| mode | occ25 | -0.04 | -1.91, -0.67, +2.46 |
| mode | occ50 | +1.02 | +0.09, +0.32, +2.64 |
| mode | noise0.2 | -0.79 | -3.97, +0.24, +1.36 |
| mode_s4 | occ50 | +1.00 | +0.02, +0.32, +2.66 |

- `mode` minus `mean`: the condition means differ by -0.79 to
  +1.02 points; the per-seed sign changes in 6 of 7 metrics
  (6 conditions plus the selection split). `occ50` is positive on all three seeds
  (+0.09, +0.32, +2.64 points) but the mean of +1.02 comes from one seed.
- `mode_s4` minus `mode`: at most 0.03 points in any condition mean and at most
  0.08 points in any single seed and metric. Four starts add cost (5.55 against 5.33
  s per epoch) and nothing else here.
- `cls` minus `mean`: 3.26 to 4.60 points better at 50 %
  occlusion on all three seeds, at 0.46 to 1.81 points lower clean accuracy (negative on all
  three seeds, as is `noise0.05`). At `occ25`, `noise0.1` and `noise0.2` its per-seed differences
  change sign.
- Hypotheses for the null, none tested: the final tokens are already near-uniform so a mode has
  no outlier cluster to ignore; the median-heuristic bandwidth is wide enough that five steps
  stay close to the mean; occluded tokens form a coherent second cluster; a model trained on
  clean images adapts its tokens to whichever pooling is in the loop. The mode-to-mean distance
  of the trained tokens was not measured.

## 3. Experiment 2: stability smoke

`results/mean_shift/20261030_065822_exp2_stability`, 5 epochs, seed 0, CIFAR-100, the default
config `blur_s3_per_head_gaussian` against `mha`. The pass conditions were fixed before the run
and all were met:

- no non-finite loss (both runs finite for 5 epochs);
- epoch-5 training loss below epoch-1: MeanShift 4.457 to 3.725, `mha`
  4.333 to 3.065;
- selection accuracy at epoch 5 at least 3 %: MeanShift 13.92 %, `mha` 25.60 %;
- learned bandwidths strictly inside the clip range [0.01, 100]: 2.722 to
  2.850 over all layers and heads (initial value 2.83).

MeanShift learned more slowly than `mha` over the first 5 epochs.

## 4. Experiment 2: the grid

`results/mean_shift/20261030_070258_exp2_grid`: 16 MeanShift configs plus `mha`, 1 seed, 50
epochs, ranked by selection accuracy only. Percent; the last column is selection accuracy minus
that of `mha`, in points.

| rank | config | selection | test | selection minus mha (points) |
|---|---|---|---|---|
| ref | mha | 43.26 | 44.13 | - |
| 1 | noblur_s3_per_token_vmf | 57.08 | 57.38 | +13.82 |
| 2 | blur_s1_per_head_vmf | 56.82 | 57.88 | +13.56 |
| 3 | noblur_s1_per_head_vmf | 56.66 | 58.15 | +13.40 |
| 4 | noblur_s3_per_head_vmf | 56.10 | 57.89 | +12.84 |
| 5 | blur_s3_per_token_vmf | 55.76 | 56.25 | +12.50 |
| 6 | blur_s1_per_head_gaussian | 55.62 | 56.64 | +12.36 |
| 7 | blur_s3_per_head_vmf | 55.58 | 56.89 | +12.32 |
| 8 | noblur_s1_per_head_gaussian | 55.28 | 56.05 | +12.02 |
| 9 | noblur_s1_per_token_gaussian | 55.00 | 56.44 | +11.74 |
| 10 | blur_s1_per_token_vmf | 54.92 | 56.96 | +11.66 |
| 11 | blur_s1_per_token_gaussian | 54.70 | 56.17 | +11.44 |
| 12 | noblur_s1_per_token_vmf | 54.70 | 57.02 | +11.44 |
| 13 | noblur_s3_per_head_gaussian | 53.94 | 55.46 | +10.68 |
| 14 | noblur_s3_per_token_gaussian | 53.76 | 55.81 | +10.50 |
| 15 | blur_s3_per_head_gaussian | 53.68 | 55.55 | +10.42 |
| 16 | blur_s3_per_token_gaussian | 53.24 | 54.98 | +9.98 |

- Parameters: 2,265,910 (per_head) and 2,269,366 (per_token) against
  2,710,564 for `mha` (16.4 % and 16.3 %
  fewer).
- Caveat: one seed. The top four are within 0.98 points, about three times the spread of the
  one-step blurring and non-blurring twins (0.16 to 0.34 points) and about the size of the seed
  std in the main run (0.40 to 0.91); rank 1 to 2 is 0.26 points, and ranks 2
  and 3 are the one-step blurring and non-blurring versions of one config, so their
  0.16 points are run-to-run noise. The two configs chosen for the main run,
  `noblur_s3_per_token_vmf` and `blur_s1_per_head_vmf`, were a weak choice.
- The vMF kernel beat its Gaussian twin in 7 of 8 matched pairs, by 0.22 to 3.32 points (one pair
  -0.30). This mixes the kernel with the initialization (vMF configs start at `bandwidth_init`
  0.3, Gaussian at 2.83).

## 5. Experiment 2: the main run

`results/mean_shift/20261030_090740_exp2_main`: seeds 0, 1, 2, 50 epochs. Percent, mean +- std.
"Final-epoch train accuracy" is Keras' running accuracy over the last epoch on augmented
batches, a rough figure used only to compare the three rows.

| config | seeds | params | s/epoch | images/s | selection | test | final-epoch train accuracy |
|---|---|---|---|---|---|---|---|
| mha | 3 | 2,710,564 | 5.08 +- 0.21 | 13986 +- 620 | 42.81 +- 0.83 | 43.63 +- 0.48 | 76.33 +- 0.83 |
| blur_s1_per_head_vmf | 3 | 2,265,910 | 5.00 +- 0.09 | 15831 +- 1456 | 56.85 +- 0.40 | 57.97 +- 0.16 | 76.08 +- 0.27 |
| noblur_s3_per_token_vmf | 3 | 2,269,366 | 6.76 +- 0.13 | 14527 +- 212 | 56.57 +- 0.91 | 57.07 +- 0.55 | 80.49 +- 0.29 |

Paired differences against `mha` (config minus `mha`, per seed 0, 1, 2, points):

| config | metric | mean diff vs mha (points) | per-seed diffs, seeds 0, 1, 2 (points) |
|---|---|---|---|
| blur_s1_per_head_vmf | select | +14.05 | +13.02, +14.26, +14.86 |
| blur_s1_per_head_vmf | test | +14.33 | +14.00, +14.67, +14.33 |
| noblur_s3_per_token_vmf | select | +13.76 | +13.66, +13.98, +13.64 |
| noblur_s3_per_token_vmf | test | +13.44 | +13.61, +13.53, +13.18 |

- Both configs are above `mha` on every seed. The two configs are not separated from each other
  (56.85 against 56.57 % selection, stds 0.40 and
  0.91); the order is the reverse of the 1-seed grid.
- `mha` has a train-to-selection gap of 33.5 points; `blur_s1_per_head_vmf`
  19.2 (about the same training accuracy); `noblur_s3_per_token_vmf`
  23.9 (higher training accuracy).
- Cost: 5.00 s per epoch for `blur_s1_per_head_vmf`, 6.76 for
  `noblur_s3_per_token_vmf`, 5.08 for `mha`.

## 6. Experiment 2: the initializer control

`results/mean_shift/20261030_135834_exp2_control`: 6 cells, seeds 0, 1, 2, 50 epochs, CIFAR-100,
GPU 0 under the idle gate, the same recipe. The finding behind it: `MultiHeadCrossAttention`
reuses one seeded initializer instance for `qkv` and `proj`, so after `set_seeds(0)` the 192x192
`proj/kernel` of every `mha` block is a rescaled copy of the first 36,864 values of its
`qkv/kernel` (correlation 1.000000 in all six blocks), while the MeanShift `projection` and
`output_projection` are independent (largest absolute correlation 0.012). `mha_oinit` is
`multi_head` with `attention_args={"output_kernel_initializer": "glorot_uniform"}`, an
independent initializer for `proj` (largest absolute correlation 0.0099). `mha` was re-run in the
same root. The `mha` rows of the grid and the main run were left as run. Percent, mean +- std
over 3 seeds:

| config | seeds | params | s/epoch | selection | test | final-epoch train accuracy |
|---|---|---|---|---|---|---|
| mha (rerun) | 3 | 2,710,564 | 5.02 +- 0.31 | 42.71 +- 0.91 | 43.74 +- 0.57 | 76.40 +- 0.85 |
| mha_oinit | 3 | 2,710,564 | 4.96 +- 0.26 | 42.74 +- 0.70 | 43.77 +- 0.28 | 76.46 +- 0.58 |

| config | selection, seeds 0, 1, 2 | test, seeds 0, 1, 2 | final-epoch train accuracy, seeds 0, 1, 2 |
|---|---|---|---|
| mha (rerun) | 43.46, 42.98, 41.70 | 44.35, 43.23, 43.63 | 75.69, 77.35, 76.16 |
| mha_oinit | 42.70, 43.46, 42.06 | 43.79, 44.03, 43.48 | 76.52, 77.00, 75.85 |

| comparison (first minus second) | metric | mean diff | per-seed diffs, seeds 0, 1, 2 |
|---|---|---|---|
| mha_oinit minus mha rerun | select | +0.03 | -0.76, +0.48, +0.36 |
| mha_oinit minus mha rerun | test | +0.03 | -0.56, +0.80, -0.15 |
| mha rerun minus mha main run | select | -0.09 | -0.04, -0.06, -0.18 |
| mha rerun minus mha main run | test | +0.10 | +0.27, +0.11, -0.07 |

- **The independent output initializer does not change the baseline.** `mha_oinit` is within
  noise of `mha` on every seed (selection differences -0.76 to +0.48 points, test -0.56 to
  +0.80, signs change across seeds, means +0.03) at the same training accuracy. The
  shared-initializer artefact therefore does not explain the 13 to 14 point gap: against
  `mha_oinit` the two chosen configs are still 13.46 to 14.68 points ahead on selection and
  12.62 to 14.55 on test, on every seed.
- **The baseline is reproducible.** The rerun of `mha` differs from the main run by 0.04, 0.06
  and 0.18 points of selection accuracy and by at most 0.27 of test accuracy, per seed.
- Limits: 3 seeds and a noise floor of a few tenths of a point; the control excludes an
  initializer effect of the size of the gap, not one of a few tenths. The correlations are a
  CPU probe of the freshly built models after `set_seeds(0)` (pinned by a test in
  `tests/test_train/test_mean_shift/test_exp_workers.py`), not a measurement of the trained
  weights.

## 7. Learned bandwidths

Main run, mean over seeds and heads per layer (layers 0 to 5); `per_token` additionally over 512
selection images. All values are inside the clip range [0.01, 100], but "inside" is a binary
flag: see the distance to the floor below.

| config | mode | per-layer mean, layers 0 to 5 | min | max |
|---|---|---|---|---|
| blur_s1_per_head_vmf | per_head | 0.303, 0.523, 0.599, 0.813, 1.161, 1.284 | 0.192 | 1.309 |
| noblur_s3_per_token_vmf | per_token | 2.654, 1.174, 0.876, 2.464, 3.159, 2.787 | 0.010 | 18.594 |

- `per_head`: layer 0 stays near its 0.3 initialization, deeper layers grow to 1.284;
  the seeds agree to within 0.027 per layer.
- `per_token`: smallest 0.010, largest 18.594; the layer-0 mean is
  0.152, 3.851 and 3.960 on seeds 0, 1, 2 (one layer-0
  head at 11.29 and 11.61 on seeds 1 and 2, none above
  0.33 on seed 0).
- The lower tail of the per-token bandwidths is saturated at the 0.01 floor in the main run
  and in 5 of the 8 grid per-token configs. The lowest
  per-token bandwidth of `noblur_s3_per_token_vmf` is 4.1e-4 to 8.1e-4 above the floor on the
  three seeds (in layer 0 each time); in the grid the lowest per-token bandwidth of the 8
  per-token configs is between 4.0e-6 (`blur_s3_per_token_gaussian`) and 0.23 above it, and
  5 of the 8 are within 2.5e-3. With `h = 0.01 + softplus(raw)` the gradient reaching `raw` is
  `sigmoid(raw)` = `1 - exp(-(h - 0.01))`, about the distance to the floor, so such a token
  barely moves back; for the vMF kernel `kappa = 1 / h^2` is about 9e3 there, hard
  nearest-key attention.
- Weight decay acts on the bandwidth parameters: the shared AdamW has weight decay 0.05 and no
  `exclude_from_weight_decay`, so `raw_bandwidth` and the `bandwidth_projection` kernel and bias
  are decayed toward raw 0, that is `h` toward `0.01 + softplus(0)` = 0.703 (upward for the vMF
  configs that start at 0.3, downward for the Gaussian ones that start at 2.83). Its size
  against the gradient was not measured, so the learned bandwidths mix learning with that prior.
- Gaussian configs of the grid move away from the 2.83 initialization in both directions (the
  per-layer means of `blur_s1_per_head_gaussian` run from 1.978 to 4.489); the
  full per-config values are in `results/mean_shift/20261030_reports/exp2_grid/summary.md`.

## 8. Empirical Tweedie denoiser oracle (WP5)

Documented in `src/train/bfunet/FINDINGS.md` section 10 (run directory
`results/mean_shift_denoiser/20261030_060609`), not repeated here. Oracle only; no trained
denoiser was compared. On held-out DIV2K crops the oracle median PSNR is flat across sigma
(23.47 to 23.61 dB) and its median is below the noisy-input median at sigma 5, 10 and 15
(23.56, 23.56, 23.57 against 34.15, 28.15, 24.61 dB); that holds for the median only, the
oracle mean (27.13, 26.88, 26.73 dB) is above the noisy input at sigma 15 (24.60 dB). On in-bank
queries it copies (median at the 120 dB cap).

## 9. Caveats and what was not done

- The baseline recipe is untuned: no dropout, one learning rate and weight decay for all arms.
- MeanShift has 16 % fewer parameters and tied Q, K, V; a regularization effect is not
  excluded. The initializer replay of `mha` is excluded as the explanation by the control of
  section 6. vMF configs use `bandwidth_init` 0.3, Gaussian the layer default.
- The vMF kernel L2-normalises queries and keys (and the moved tokens after every step), which
  acts like QK normalisation; `mha` has none. Untested confound of every vMF result; the
  Gaussian configs (no normalisation) are still 9.98 to 12.36 points above `mha` in the grid.
  The matching control, `multi_head` with `qk_norm_type`, was not run.
- Weight decay acts on the bandwidth parameters (section 7).
- One dataset per experiment (CIFAR-10 for exp1, CIFAR-100 for exp2), small models, 50 epochs,
  3 seeds (1 for the grid), no significance test.
- exp1 and exp2 use different backbones; nothing is claimed across them.
- Not done: baseline tuning, other datasets, longer budgets, more seeds, training-time
  corruption augmentation, a mode-to-mean distance measurement, an ablation that separates the
  vMF kernel from its bandwidth initialization, a `multi_head` arm with QK normalisation, a run
  with the bandwidth parameters exempt from weight decay.
