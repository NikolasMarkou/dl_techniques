# RMSNorm Variants Comprehensive Study

A multi-experiment harness comparing four normalization layers across diverse
models, tasks, and regimes — designed to deliver a defensible
**PASS / FAIL / INDISTINGUISHABLE** verdict for each variant's specific
theoretical claim, not merely a test-accuracy column.

Plan: `plans/plan_2026-05-14_3764496e`.

## The four variants

| Norm | Trainable params per layer | Distinguishing mechanism |
|------|----------------------------|--------------------------|
| `rms_norm` (baseline) | `d` (per-feature γ) | RMS rescaling only |
| `band_rms` | 1 scalar | Constrains output RMS to `[1-α, 1]` band |
| `zero_centered_rms_norm` | `d` (per-feature γ) | Centers inputs before RMS (DC removal) |
| `zero_centered_band_rms_norm` | 1 scalar | Centers + band constraint |

The parameter-count asymmetry (1 vs `d`) is a confound. Every experiment is
run in **two modes**:

- **`oob`** — out-of-the-box defaults. Reflects the "drop-in usability" answer.
- **`param_matched`** — RMSNorm-family variants run with `use_scale=False`
  (0 trainable params per norm), matching BandRMS variants' 1 scalar.

Both modes are reported side-by-side; the verdict requires **consistency
across modes**.

## Experiment matrix

| ID | Model | Task | Regime | Hypothesis under test |
|----|-------|------|--------|------------------------|
| E1 | ViT-pico | CIFAR-10 | fp32, AdamW, cosine, 50ep | OOB accuracy + γ-growth in residual stream |
| E2 | ResNet-18 | CIFAR-100 | fp32, AdamW, cosine, 80ep | Norm choice in conv stack vs transformer |
| E3 | TinyTransformer | IMDb seq=128 | fp32, AdamW, 30ep | NLP-domain transferability; activation RMS stats |
| E4 | DeepResidual (24 blocks) | Synthetic polynomial reg | **fp16, batch=16**, 60ep | γ-growth + DC drift under adversarial regime |
| E5 | NormLayerMicrobench | Synthetic Gaussian reg | fp32, K=16 stack, 30ep | Layer-level baseline / callback sanity |

## Probes

Four mechanistic callbacks log per-epoch CSV rows, directly targeting each
variant's theoretical claim:

1. **`GradientNormCallback`** — global gradient L2-norm trajectory.
2. **`WeightNormTrajectoryCallback`** — per-norm-layer `scale` / `band_param`
   L2 trajectory. Direct test of the "γ-growth suppression" claim
   (ZeroCenteredRMSNorm).
3. **`NormLayerActivationCallback`** — mean and per-sample-RMS std of each
   norm layer's output, evaluated on a fixed calibration batch. Direct test
   of the "thick spherical shell" (BandRMS) and "zero-mean output"
   (ZeroCentered\*) claims.
4. **`NormInternalStatsCallback`** — scalar internal state per norm layer
   (`scale` L2 for RMSNorm-family, `band_param` post-sigmoid value for
   BandRMS-family).

Additionally, `EpochAnalyzerCallback` (data-free) logs weight and spectral
statistics every 5 epochs.

## Reproducing

Single cell:

```bash
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.rms_variants_train.experiments.e5_norm_layer_microbench \
    --norm-type rms_norm --seed 0 --epochs 5 --out-dir /tmp/rms_e5
```

Full sweep:

```bash
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.rms_variants_train.sweep \
    --experiments e1,e2,e3,e4,e5 \
    --norms rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm \
    --seeds 0,1,2,3,4 --mode oob --out-dir results/rms_variants_train/full_oob
```

Then aggregate:

```bash
.venv/bin/python -m train.rms_variants_train.report --in-dir results/rms_variants_train/full_oob
```

## Verdict rules

Per variant, the report writer applies these decision rules:

- **PASS**: the variant's hypothesis-test column rejects the null vs. RMSNorm
  baseline at `p < 0.05` (paired permutation, B=10000) AND the direction
  matches the claim, in **at least 2 of the 5 experiments** AND **in both
  modes**.
- **FAIL**: the same column rejects the null in the **opposite** direction
  in any experiment.
- **INDISTINGUISHABLE**: neither PASS nor FAIL — within seed noise at n=5
  (LESSONS L77).

The headline accuracy column is reported but is NOT the verdict driver —
saturation routinely makes it uninformative (LESSONS L61).

## Statistical inference

All aggregation routes through `train.logic.multiseed_stats` (re-exported
via `train.rms_variants_train.stats`):

- `mean_std` — NaN-tolerant, Bessel-corrected sample std.
- `bootstrap_ci(B=2000)` — vectorized percentile CI.
- `paired_permutation_test(B=10000)` — sign-flip with Phipson-Smyth
  correction; degenerate all-zero-diffs returns `p=1.0`.

## Layout

```
src/train/rms_variants_train/
├── __init__.py
├── config.py                 ExperimentConfig + build_norm_kwargs + NORM_VARIANTS (8-tuple as of Phase 3)
├── seed_utils.py             set_seeds(seed)
├── stats.py                  re-export from train.logic.multiseed_stats
├── callbacks.py              6 callbacks: 4 probes + CalibrationCallback + RobustnessProbe (Phase 3)
├── sweep.py                  subprocess sweep driver (Phase 3: --gpu CLI hard-sets CUDA_VISIBLE_DEVICES)
├── report.py                 summary.md writer + 3 Phase 3 post-hoc derivations
├── README.md                 this file
├── RESULTS.md                Phase 1 verdict (published) + Phase 3 design appendix
├── PHASE3_PLAN.md            Phase 3 operational plan (per-chunk commands + falsification signals)
└── experiments/
    ├── __init__.py           (each trainer accepts a --regime sub-experiment flag as of Phase 3)
    ├── e1_vit_cifar10.py
    ├── e2_resnet_cifar100.py
    ├── e3_tinytransformer_imdb.py
    ├── e4_deep_residual_reg.py
    └── e5_norm_layer_microbench.py
```

## Phase 3 additions

Phase 3 (plan `plan_2026-05-18_63121227`) extends the 7-norm harness to 8
by integrating `zero_centered_adaptive_band_rms_norm` (the 8th library
variant). It also broadens the metrics suite (calibration ECE-15 + Brier;
input-perturbation robustness over 4 Gaussian sigmas; epochs-to-threshold
convergence; late-training stability) and parameterises four new
sub-experiment axes (LR / batch / mixed-precision / depth) through a
`--regime` argparse flag on each trainer. The Phase 2 GPU env propagation
bug is fixed: `sweep.py` now exposes a `--gpu` CLI that hard-sets
`CUDA_VISIBLE_DEVICES` for each cell subprocess (the parent shell's value
can no longer leak through). See `PHASE3_PLAN.md` for the operational
sweep recipe.

## Phase 3 refinements (plan `plan_2026-05-18_e1f12eab`)

The harness is refined with four falsifiability-oriented features. None
modify the Phase 1 RESULTS.md verdict block (I2 invariant), the
`VARIANT_CRITERIA` PASS/FAIL rule (I4), or the GPU `--gpu` hard-set contract.
Everything below is additive.

### Hypothesis Registry

`train.rms_variants_train.hypotheses.VARIANT_HYPOTHESES` is a dict mapping each
of the 8 norm variants to a single falsifiable claim with a numerical STOP-IF
threshold on a metric column the harness already collects. Each entry is a
`HypothesisSpec(claim, metric_column, comparator, threshold, applicable_experiments, applicable_modes, min_samples, reduction, notes)`.

`evaluate_hypothesis(variant, df, experiment=..., mode=...) -> Verdict` returns
one of `CONFIRMED` / `REJECTED` / `INCONCLUSIVE` / `N/A`. `evaluate_all(df)`
groups by `(experiment, norm_type, mode)` and emits a per-cell verdict frame.

`report.write_report` consumes `evaluate_all`, attaches a `hypothesis_verdict`
column to `headline_summary.csv`, writes `hypothesis_verdicts.csv` (per-cell
observed vs threshold), and renders a "Hypothesis verdicts" block in
`summary.md` alongside the existing PASS/FAIL block. Thresholds are stated
from each layer's documented design claim — not tuned post-hoc to match
observed Phase 1 data.

### Distribution-Shift Probe

`callbacks.DistributionShiftProbe` evaluates val accuracy on
**CIFAR-10-C** (Hendrycks & Dietterich 2019) corruptions via
`tensorflow_datasets` `cifar10_corrupted/{corruption}_{severity}`. Wired into
E1 by default (severity 3, 5 corruption-type subset: gaussian_noise,
defocus_blur, brightness, contrast, jpeg_compression; max 500 samples per
corruption). E2 wiring (CIFAR-100-C) is identical and is a one-line addition
when needed.

**Hard contract — never raises.** On missing TFDS dataset, missing
`tensorflow_datasets` package, or any load error, the probe writes a
`dist_shift.csv` with a `reason` column populated and logs a WARNING. No
cell is poisoned by probe failure. This is anchored at
`DECISION plan_2026-05-18_e1f12eab/D-002`.

### Sweep `--regimes`

`sweep.py --regimes <csv>` makes the regime axis a first-class sweep dimension
instead of a shell loop. Cells multiply by the regime count. Unsupported
(experiment, regime) pairs are filtered with a one-line log via the
`EXPERIMENT_REGIMES` table (mirroring each trainer's `_REGIME_MAP`).

A hard `--max-cells` build-time guard (default 1000) refuses oversized
builds **before any subprocess is launched** — prevents the
partial-sweep / inconsistent-results-dir failure mode. Anchored at
`DECISION plan_2026-05-18_e1f12eab/D-003`. Bump `--max-cells` explicitly
for full-Cartesian-product sweeps; the default flags any > 1000-cell build
as a smell to chunk instead.

Non-default regimes get a `regime_<name>` segment in the out-dir path so
the default-regime layout (under which Phase 1 RESULTS.md was generated)
remains byte-identical (I2 invariant).

### Generalization Gap Metric

Every trainer's `results.csv` now includes a `generalization_gap` column:

- **Classification (E1/E2/E3)**: `final_acc - final_val_acc` — positive value
  indicates overfitting.
- **Regression (E4/E5)**: `final_val_loss - final_loss` — positive value
  indicates worse generalization.

NaN-tolerant (any computation error → NaN). Consumed by the
`band_logit_norm` hypothesis registry entry (claim: classification
generalization gap ≤ 0.20) and reported in `headline_summary.csv`.
