# Phase 3 — RMSNorm Variants Full 8-Variant Campaign Plan (v2)

*Plan: `plans/plan_2026-05-18_e1f12eab` (refinement) — supersedes the v1 doc from `plans/plan_2026-05-18_63121227`.*

*Status: harness refined per plan_e1f12eab Steps 1-5 (VARIANT_HYPOTHESES registry, DistributionShiftProbe, generalization_gap column, sweep --regimes + --max-cells guard). Smoke gate at Step 9 of the same plan; chunks 1-3 remain user-initiated.*

This document captures the per-chunk command sequences, cell tally, wall-clock budget, and falsification signals for the Phase 3 execution. It is the operational counterpart to the design section appended to `RESULTS.md` and the falsifiable hypothesis registry in `train.rms_variants_train.hypotheses` (`VARIANT_HYPOTHESES`).

## What changed since v1

| Feature | Where | Verdict source |
|---|---|---|
| Falsifiable hypothesis registry — one claim + STOP-IF threshold per variant | `train.rms_variants_train.hypotheses:VARIANT_HYPOTHESES` | `hypothesis_verdict` column on `headline_summary.csv` + `hypothesis_verdicts.csv` |
| Generalization-gap metric on every trainer | `results.csv:generalization_gap` (E1/E2/E3: `train_acc - val_acc`; E4/E5: `val_loss - train_loss`) | Aggregated into headline tables; consumed by the BandLogitNorm hypothesis |
| Real distribution-shift probe (CIFAR-10-C) | `callbacks.DistributionShiftProbe` wired into E1 with `cifar10_corrupted/{corruption}_3` TFDS dataset | per-cell `dist_shift.csv` |
| Sweep `--regimes` first-class dimension | `sweep.py --regimes` enumerates `(exp, norm, mode, regime, seed)` | per-cell `regime` column on `results.csv` |
| `--max-cells` build-time guard | `sweep.py --max-cells` (default 1000) — raises BEFORE any subprocess | Build error, not run-time |

## Pre-flight check

Before launching any chunk, verify the harness is green:

```bash
cd src
CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m pytest \
    tests/test_train/test_rms_variants_train/ \
    tests/test_layers/test_norms/ -vvv
```

All must be green (≥ 488 PASS post-plan_e1f12eab: 347 norm-suite + 141 harness-suite).

Hypothesis-registry import check:

```bash
cd src
CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -c \
  "from train.rms_variants_train.hypotheses import VARIANT_HYPOTHESES, evaluate_all; \
   assert len(VARIANT_HYPOTHESES) == 8; print('OK — 8 hypotheses registered')"
```

## Smoke gate (required before Chunk 1)

8 variants × 5 experiments × seed 0 × 5-min global cap. Must complete in ≤ 5 minutes total and emit non-empty `all_runs.csv` with `hypothesis_verdict` + `generalization_gap` columns populated.

```bash
cd src
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.rms_variants_train.sweep \
    --gpu 0 \
    --experiments e1,e2,e3,e4,e5 \
    --norms rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm,adaptive_band_rms,band_logit_norm,dynamic_tanh,zero_centered_adaptive_band_rms_norm \
    --modes oob \
    --regimes default \
    --max-cells 200 \
    --seeds 0 \
    --epochs 1 \
    --cell-timeout-s 90 \
    --global-cap-s 300 \
    --out-dir results/rms_variants_train/p3_smoke
```

After completion (per plan_e1f12eab Step 9 / SC13):

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('results/rms_variants_train/p3_smoke/all_runs.csv')
assert len(df) == 40, f'Expected 40 rows; got {len(df)}'
assert df['hypothesis_verdict'].notna().all(), 'hypothesis_verdict column has NaN'
assert df['generalization_gap'].notna().all(), 'generalization_gap column has NaN'
print('OK — smoke gate PASS')
"
```

Then inspect E1 dist-shift:

```bash
.venv/bin/python -c "
import pandas as pd, glob
csvs = glob.glob('results/rms_variants_train/p3_smoke/e1/*/oob/seed_0/dist_shift.csv')
for c in csvs:
    df = pd.read_csv(c)
    bad = df[(df['reason'].astype(str) == '') & df['val_acc'].isna()]
    assert bad.empty, f'{c}: rows missing both val_acc and reason: {bad}'
print('OK — DistributionShiftProbe ran or soft-failed cleanly on all E1 cells')
"
```

**Falsification A (smoke)** — STOP IF: `cat results/rms_variants_train/p3_smoke/e2/zero_centered_adaptive_band_rms_norm/oob/seed_0/cell.log | tail -5` shows NaN gradients or `best_val_acc < 0.20`.

**GPU binding check** — first 3 lines of any `cell.log` must include `CUDA_VISIBLE_DEVICES=0`. If they show `=1` or absent: review `sweep.run_one` and re-run with explicit `--gpu 0`.

## Chunk 1 — Core 8 × 5-exp × 5-seed (OOB)

200 cells. Wall-clock estimate: 14-17h on RTX 4090 (unchanged from v1 — refinements add < 30s per cell from CIFAR-10-C download + dist-shift eval, amortized).

```bash
cd src
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.rms_variants_train.sweep \
    --gpu 0 \
    --experiments e1,e2,e3,e4,e5 \
    --norms rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm,adaptive_band_rms,band_logit_norm,dynamic_tanh,zero_centered_adaptive_band_rms_norm \
    --modes oob \
    --regimes default \
    --seeds 0,1,2,3,4 \
    --max-cells 250 \
    --out-dir results/rms_variants_train/p3_chunk1 \
    --global-cap-s 64800
```

**Verdict source**: post-sweep `headline_summary.csv` has both `verdict` (existing PASS/FAIL/INDISTINGUISHABLE per VARIANT_CRITERIA) and `hypothesis_verdict` (CONFIRMED/REJECTED/INCONCLUSIVE/N/A per VARIANT_HYPOTHESES). Cross-reference both when writing RESULTS.md.

**Falsification B (chunk 1)** — STOP IF: estimated remaining wall-clock at the 6h mark exceeds 24h. Drop E2 from this chunk (the longest per-cell trainer) and rerun the missing E2 cells as Chunk 2's prefix.

## Chunk 2 — param_matched on E3/E4/E5 (4 norms only)

40 cells (4 variants × 3 experiments × 5 seeds with the variants that have a real `use_scale` toggle: rms_norm, zero_centered_rms_norm, band_rms, zero_centered_band_rms_norm). Wall-clock ~6h.

```bash
cd src
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.rms_variants_train.sweep \
    --gpu 0 \
    --experiments e3,e4,e5 \
    --norms rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm \
    --modes param_matched \
    --regimes default \
    --seeds 0,1,2,3,4 \
    --max-cells 100 \
    --out-dir results/rms_variants_train/p3_chunk2 \
    --global-cap-s 25200
```

## Chunk 3 — Regime sub-experiments (first-class via `--regimes`)

Now driven by `sweep.py --regimes` directly — no more shell loop. Cells multiply by the regime axis, so `--max-cells` is critical.

```bash
# Example: E5 microbench across all 5 supported regimes × 8 norms × 5 seeds = 200 cells.
cd src
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.rms_variants_train.sweep \
    --gpu 0 \
    --experiments e5 \
    --norms rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm,adaptive_band_rms,band_logit_norm,dynamic_tanh,zero_centered_adaptive_band_rms_norm \
    --modes oob \
    --regimes default,bs_32,bs_256,lr_low,lr_high \
    --seeds 0,1,2,3,4 \
    --max-cells 250 \
    --out-dir results/rms_variants_train/p3_chunk3_e5 \
    --global-cap-s 21600
```

```bash
# E1 regimes (lr_low, lr_high, mp_fp16) × 8 norms × 5 seeds = 120 cells.
cd src
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.rms_variants_train.sweep \
    --gpu 0 \
    --experiments e1 \
    --norms rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm,adaptive_band_rms,band_logit_norm,dynamic_tanh,zero_centered_adaptive_band_rms_norm \
    --modes oob \
    --regimes default,lr_low,lr_high,mp_fp16 \
    --seeds 0,1,2,3,4 \
    --max-cells 200 \
    --out-dir results/rms_variants_train/p3_chunk3_e1 \
    --global-cap-s 43200
```

```bash
# E4 regimes (depth_12, depth_48) × 8 norms × 5 seeds = 120 cells.
cd src
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.rms_variants_train.sweep \
    --gpu 0 \
    --experiments e4 \
    --norms rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm,adaptive_band_rms,band_logit_norm,dynamic_tanh,zero_centered_adaptive_band_rms_norm \
    --modes oob \
    --regimes default,depth_12,depth_48 \
    --seeds 0,1,2,3,4 \
    --max-cells 150 \
    --out-dir results/rms_variants_train/p3_chunk3_e4 \
    --global-cap-s 32400
```

E2 has only `default`. E3 has `default,mp_fp16` → a separate sweep if needed (`--regimes default,mp_fp16`).

**Falsification C (regimes triple wall-clock)** — STOP IF: regime-sweep estimated wall-clock for any chunk exceeds 24h. Action: drop regimes from this chunk; keep core 8-variant × 5-exp × 5-seed only. Regime axis becomes a follow-up plan. The `--max-cells` guard prevents accidentally launching the full Cartesian product (8 × 5 × 5 × 5 × 5 = 5000 cells would refuse to build).

## Analysis day

```bash
# Concatenate all_runs.csv across chunks into one master frame.
mkdir -p results/rms_variants_train/p3_full
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

Outputs:
- `summary.md` — narrative report with PASS/FAIL verdict block AND hypothesis-verdict block.
- `headline_summary.csv` — adds `hypothesis_verdict` + `late_stability_var` columns.
- `hypothesis_verdicts.csv` — per-cell observed-vs-threshold details.
- `probes_summary.csv`, `convergence_summary.csv`, `regime_delta_summary.csv` — unchanged from v1.

Update `RESULTS.md` Phase 3 "Results" subsection with both verdict tables from `summary.md`.

## Falsification signals (summary)

| ID | Signal | STOP IF | Action |
|----|--------|---------|--------|
| A  | New variant broken | smoke shows NaN or `best_val_acc < 0.20` on E2/zc-adaptive/seed_0 | inspect ctor wiring; compare to ZeroCenteredBandRMSNorm |
| B  | Chunk 1 over budget | est. remaining at 6h mark > 24h | drop E2; reschedule to Chunk 2 prefix |
| C  | Regimes triple wall-clock | any regime chunk est. > 24h | defer regime axis to follow-up |
| D  | GPU binding leak | `cell.log` first lines show CVD ≠ 0 | inspect `sweep.run_one` env construction; rerun with explicit `--gpu 0` |
| E  | Hypothesis registry unfalsifiable | ≥ 2 variants always INCONCLUSIVE post-sweep, OR ≥ 4 evaluate-to-N/A on standard chunks | revisit VARIANT_HYPOTHESES thresholds — they may not bind to collected metrics. See plan_e1f12eab pre-mortem A. |
| F  | DistributionShiftProbe missing | E1 `dist_shift.csv` rows all have `reason=dataset_missing` on a system where `cifar10` itself loads | pivot per plan_e1f12eab pre-mortem B — hand-rolled corruption suite |

## Wall-clock budget (v2)

| Chunk | Cells | Wall-clock | Notes |
|-------|-------|------------|-------|
| Smoke | 40   | ≤ 5 min   | + ~30s amortized for CIFAR-10-C download on first run; SC13 gate of plan_e1f12eab |
| 1     | 200  | 14-17h    | Core 8-variant OOB sweep; emits `hypothesis_verdict` + `generalization_gap` |
| 2     | 40   | ~6h       | param_matched on E3/E4/E5 (4 norms) |
| 3a    | 200  | ~6h       | E5 × 5 regimes |
| 3b    | 120  | ~12h      | E1 × 4 regimes |
| 3c    | 120  | ~9h       | E4 × 3 regimes |
| Analysis | -- | ~1h     | Concatenate + regenerate report |

Total optimistic: ~50h. Total realistic: 4-5 overnight chunks ≤ 18h + 1 analysis day. Regimes (Chunks 3a-3c) can be trimmed by reducing `--seeds` to 0,1,2 (3 seeds) → halves wall-clock at the cost of statistical power on the regime axis.

## References

- Decisions (this plan): `plans/plan_2026-05-18_e1f12eab/decisions.md` D-001 (`VARIANT_HYPOTHESES` registry), D-002 (`DistributionShiftProbe` soft-fail), D-003 (`--max-cells` guard), D-005 (scope b smoke).
- Decisions (prior plans): `plans/plan_2026-05-18_63121227/decisions.md` D-001 (8-tuple), D-002 (GPU env hard-set), D-003 (calibration + robustness probes).
- Hypothesis registry: `src/train/rms_variants_train/hypotheses.py` — VARIANT_HYPOTHESES dict + evaluate_hypothesis(variant, df) -> Verdict.
- Findings: `plans/plan_2026-05-18_e1f12eab/findings/*` (3 indexed findings from the prior-work synthesis).
- Failure-mode log from prior attempt: `plans/LESSONS.md` L93 (setdefault footgun).
