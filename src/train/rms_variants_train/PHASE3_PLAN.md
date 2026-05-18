# Phase 3 — RMSNorm Variants Full 8-Variant Campaign Plan

*Plan: `plans/plan_2026-05-18_63121227`. Status: design complete; sweep pending user execution.*

This document captures the per-chunk command sequences, cell tally, wall-clock budget, and falsification signals for the Phase 3 execution. It is the operational counterpart to the design section appended to `RESULTS.md`.

## Pre-flight check

Before launching any chunk, verify the harness is green:

```bash
cd src
CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m pytest \
    tests/test_train/test_rms_variants_train/ \
    tests/test_layers/test_norms/test_zero_centered_adaptive_band_rms_norm.py -vvv
```

All must be green. The harness import smoke is the canonical SC-12 check:

```bash
cd src
CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -c \
  "from train.rms_variants_train import config, callbacks, sweep, report; \
   from train.rms_variants_train.experiments import \
     e1_vit_cifar10, e2_resnet_cifar100, e3_tinytransformer_imdb, \
     e4_deep_residual_reg, e5_norm_layer_microbench; print('OK')"
```

## Smoke gate (required before Chunk 1)

8 variants × 5 experiments × seed 0 × 5-min global cap. Must complete in ≤ 5 minutes total and emit non-empty `all_runs.csv`.

```bash
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.rms_variants_train.sweep \
    --gpu 0 \
    --experiments e1,e2,e3,e4,e5 \
    --norms rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm,adaptive_band_rms,band_logit_norm,dynamic_tanh,zero_centered_adaptive_band_rms_norm \
    --modes oob \
    --seeds 0 \
    --epochs 1 \
    --cell-timeout-s 90 \
    --global-cap-s 300 \
    --out-dir results/rms_variants_train/p3_smoke \
    --no-report
```

**Falsification A (smoke)** — STOP IF: `cat results/rms_variants_train/p3_smoke/e2/zero_centered_adaptive_band_rms_norm/oob/seed_0/cell.log | tail -5` shows NaN gradients or `best_val_acc < 0.20`.

**GPU binding check** — first 3 lines of any `cell.log` must include `CUDA_VISIBLE_DEVICES=0`. If they show `=1` or absent: review `sweep.run_one` and re-run with explicit `--gpu 0`.

## Chunk 1 — Core 8 × 5-exp × 5-seed (OOB)

200 cells. Wall-clock estimate: 14-17h on RTX 4090.

```bash
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.rms_variants_train.sweep \
    --gpu 0 \
    --experiments e1,e2,e3,e4,e5 \
    --norms rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm,adaptive_band_rms,band_logit_norm,dynamic_tanh,zero_centered_adaptive_band_rms_norm \
    --modes oob \
    --seeds 0,1,2,3,4 \
    --out-dir results/rms_variants_train/p3_chunk1 \
    --global-cap-s 64800
```

**Falsification B (chunk 1)** — STOP IF: estimated remaining wall-clock at the 6h mark exceeds 24h. Drop E2 from this chunk (the longest per-cell trainer) and rerun the missing E2 cells as Chunk 2's prefix.

## Chunk 2 — param_matched on E3/E4/E5 (4 norms only)

40 cells (4 variants × 3 experiments × 5 seeds with the variants that have a real `use_scale` toggle: rms_norm, zero_centered_rms_norm, band_rms, zero_centered_band_rms_norm). Wall-clock ~6h.

```bash
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.rms_variants_train.sweep \
    --gpu 0 \
    --experiments e3,e4,e5 \
    --norms rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm \
    --modes param_matched \
    --seeds 0,1,2,3,4 \
    --out-dir results/rms_variants_train/p3_chunk2 \
    --global-cap-s 25200
```

## Chunk 3 — Regime sub-experiments

Per-trainer regime cells. ~400 cells aggregate; the budget can be trimmed by reducing `--seeds` or `--norms`.

```bash
# E1 regimes (lr_low, lr_high, mp_fp16) × 8 norms × 5 seeds
for regime in lr_low lr_high mp_fp16; do
  CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
      train.rms_variants_train.experiments.e1_vit_cifar10 \
      --regime "$regime" \
      --norm-type rms_norm --seed 0 \
      --out-dir "results/rms_variants_train/p3_chunk3/e1/${regime}/rms_norm/seed_0"
  # ... loop over 8 norms × 5 seeds. Cleaner: emit a per-regime sweep.py invocation
  # when sweep.py grows --regime support (deferred to follow-up).
done
```

**Falsification C (regimes triple wall-clock)** — STOP IF: regime-sweep estimated wall-clock for any chunk exceeds 24h. Action: drop regimes from this chunk; keep core 8-variant × 5-exp × 5-seed only. Regime axis becomes a follow-up plan.

## Analysis day

```bash
# Concatenate all_runs.csv across chunks into one master frame.
.venv/bin/python -c "
import pandas as pd, glob
frames = [pd.read_csv(p) for p in glob.glob('results/rms_variants_train/p3_chunk*/all_runs.csv')]
pd.concat(frames, ignore_index=True).to_csv(
    'results/rms_variants_train/p3_full/all_runs.csv', index=False)
"

# Regenerate the full Phase 3 report.
.venv/bin/python -m train.rms_variants_train.report \
    --in-dir results/rms_variants_train/p3_full
```

Outputs: `summary.md`, `headline_summary.csv` (with `late_stability_var`), `probes_summary.csv`, `convergence_summary.csv`, `regime_delta_summary.csv`.

Update `RESULTS.md` Phase 3 "Results" subsection with the verdict table from `summary.md`.

## Falsification signals (summary)

| ID | Signal | STOP IF | Action |
|----|--------|---------|--------|
| A  | New variant broken | smoke shows NaN or `best_val_acc < 0.20` on E2/zc-adaptive/seed_0 | inspect ctor wiring; compare to ZeroCenteredBandRMSNorm |
| B  | Chunk 1 over budget | est. remaining at 6h mark > 24h | drop E2; reschedule to Chunk 2 prefix |
| C  | Regimes triple wall-clock | any regime chunk est. > 24h | defer regime axis to follow-up |
| D  | GPU binding leak | `cell.log` first lines show CVD ≠ 0 | inspect `sweep.run_one` env construction; rerun with explicit `--gpu 0` |

## Wall-clock budget

| Chunk | Cells | Wall-clock | Notes |
|-------|-------|------------|-------|
| Smoke | 40   | ≤ 5 min   | E2/zc-adaptive falsification gate |
| 1     | 200  | 14-17h    | Core 8-variant OOB sweep |
| 2     | 40   | ~6h       | param_matched on E3/E4/E5 (4 norms) |
| 3     | ~400 | 12-18h    | regime sub-experiments (trimmable) |
| Analysis | -- | ~1h     | Concatenate + regenerate report |

Total optimistic: ~33h. Total budget allocation: 3 overnight chunks ≤ 18h + 1 analysis day, matching F-004's hard constraint.

## References

- Decisions: `plans/plan_2026-05-18_63121227/decisions.md` D-001 (8-tuple), D-002 (GPU env hard-set), D-003 (calibration + robustness probes).
- Findings: same plan dir, `findings/{01..04}*.md`.
- Failure-mode log from prior attempt: `plans/LESSONS.md` L93 (setdefault footgun).
