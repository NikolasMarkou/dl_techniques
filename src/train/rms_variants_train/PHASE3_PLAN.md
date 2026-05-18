# Phase 3 — RMSNorm Variants Full 8-Variant Campaign Plan (v3)

*Plan: `plans/plan_2026-05-18_6776f8ba` (MAXIMUM-EFFORT design refinement) — supersedes the v2 doc from `plan_2026-05-18_e1f12eab`.*

*Status: harness refined per plan_6776f8ba Steps 1-9. Smoke gate at Step 12 is USER-launched (HC12, LESSONS L110/L111). Sweep execution is USER-operated; this document ships the runnable commands.*

This document captures the per-chunk command sequences, cell tally, wall-clock budget, pre-registered analysis rules, and falsification signals for the Phase 3 execution. It is the operational counterpart to the design section appended to `RESULTS.md` and the falsifiable hypothesis registry in `train.rms_variants_train.hypotheses` (`VARIANT_HYPOTHESES`).

## What changed since v2 (9-refinement summary)

| ID | Refinement | Where | Verdict source |
|----|------------|-------|----------------|
| A | Per-norm compute-overhead benchmark | `norm_overhead_bench.py` (standalone) | `overhead.csv` columns: `(norm, params, mean_step_ms_fp32, mean_step_ms_fp16, peak_mem_mb_fp32, peak_mem_mb_fp16)` |
| B | Pre-registered `OVERALL_RULES` + `compute_overall_recommendation` | `report.py` (frozen rules + 4-slot taxonomy) | `overall_recommendation.csv` post-sweep |
| C | E6 — 4-layer causal-LM × Wikipedia 10k | `experiments/e6_clm_wiki.py` (4L, d=192, 4 heads, tiktoken cl100k_base, packed CLM) | `results.csv:final_val_perplexity` |
| D | Stress regimes (lr_extreme, wd_zero, bs_4, mp_fp16_lowloss) | `_REGIME_MAP` on E1/E3/E4/E5/E6; `sweep.EXPERIMENT_REGIMES` advertises each | per-cell `regime` column |
| E | Pure-function `stats.py` (mean_std / bootstrap_ci / paired_permutation) | `stats.py` (replaces 22-line shim) | All headline aggregation now NaN-tolerant + deterministic-RNG |
| G | ViT + ResNet `normalization_kwargs` plumbing | `models/vit/model.py`, `models/resnet/model.py`, `standard_blocks.py` (additive, default-off bit-exact) | E1/E2 PM mode now legitimately param-matched |
| F | (DEFERRED) Standalone power-analysis script | follow-up plan | n/a |
| H | (DEFERRED) E1b off-label-boundary experiment (BandLogitNorm in head slot, DyT in feature-map slot) | follow-up plan | n/a |
| I | Decision anchors uplifted to plan-id-prefixed `# DECISION plan_2026-05-18_6776f8ba/D-NNN` | norm_overhead_bench.py:D-001, report.py:D-002, vit/resnet/standard_blocks:D-003, e6_clm_wiki.py:D-004 | grep audit at CLOSE |

## Pre-flight check

Before launching any chunk, verify the harness is green:

```bash
cd src
CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m pytest \
    tests/test_train/test_rms_variants_train/ \
    tests/test_layers/test_norms/ \
    tests/test_models/test_vit/ \
    tests/test_models/test_resnet/ -vvv
```

All must be green (≥ 488 PASS pre-Phase-3 baseline + ~80 new from refinements A/B/C/E/G). Hypothesis-registry import check:

```bash
cd src
CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -c \
  "from train.rms_variants_train.hypotheses import VARIANT_HYPOTHESES; \
   assert len(VARIANT_HYPOTHESES) == 8; print('OK — 8 hypotheses registered')"
```

## Pre-warm TFDS + HF caches (LESSONS L110/L114 mitigation)

Run ONCE before the first overnight chunk. The chunk launch itself MUST NOT be blocked on first-time downloads.

```bash
# CIFAR-10-C (~2.78 GB, 95 corruption/severity files).
.venv/bin/python -c "
import tensorflow_datasets as tfds
print('cifar10_corrupted size:', tfds.builder('cifar10_corrupted/gaussian_noise_3').info.download_size)
_ = tfds.load('cifar10_corrupted/gaussian_noise_3', split='test')  # forces download
"

# CIFAR-100-C (E2 dist-shift) — only if the namespace is registered on this
# system. plan_6776f8ba step 7 confirmed it is NOT in the default TFDS build;
# the E2 probe soft-fails with reason='dataset_missing' rows in dist_shift.csv.
.venv/bin/python -c "
import tensorflow_datasets as tfds
try:
    print('cifar100_corrupted size:', tfds.builder('cifar100_corrupted/gaussian_noise_3').info.download_size)
except Exception as e:
    print('cifar100_corrupted NOT in TFDS namespace:', repr(e))
    print('  → E2 dist_shift.csv will emit per-corruption reason rows; SC7 acceptable')
"

# E6 — tiktoken cl100k_base (~50MB) + Wikipedia 20231101.en (~20GB; cached at
# /media/arxwn/data0_4tb/datasets/wikipedia on the dev box).
.venv/bin/python -c "
import tiktoken
_ = tiktoken.get_encoding('cl100k_base')
print('OK — cl100k_base ready')
from dl_techniques.datasets.nlp import load_wikipedia_train_val
tr, va, n, _ = load_wikipedia_train_val(max_train_samples=10, max_val_samples=2, return_counts=True)
print('OK — Wikipedia HF cache resolves:', n, 'sample train articles')
"
```

## Smoke gate (required before Chunk 1 — plan_6776f8ba SC13)

8 variants × 5 image experiments × seed 0 × 5-min global cap. Emits non-empty `aggregated.csv` with `hypothesis_verdict` + `generalization_gap` columns populated. E6 is excluded from smoke — its 1-step CPU smoke runs via `test_e6_clm.py` (synthetic-smoke path), so SC8 already verifies trainer non-NaN.

```bash
mkdir -p results/rms_variants_train/p3_smoke logs
cd src
# Durable tee logging — LESSONS L110 mitigation.
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
    --out-dir results/rms_variants_train/p3_smoke \
  2>&1 | tee logs/p3_smoke.$(date +%F_%H%M%S).log
```

After completion (SC13 acceptance assertions):

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('results/rms_variants_train/p3_smoke/aggregated.csv')
assert len(df) == 40, f'Expected 40 rows; got {len(df)}'
assert df['hypothesis_verdict'].notna().all(), 'hypothesis_verdict has NaN'
assert df['generalization_gap'].notna().all(), 'generalization_gap has NaN'
print('OK — smoke gate PASS')
"

# Dist-shift soft-fail OR populated check:
.venv/bin/python -c "
import pandas as pd, glob
csvs = glob.glob('results/rms_variants_train/p3_smoke/e1/*/oob/seed_0/dist_shift.csv')
for c in csvs:
    df = pd.read_csv(c)
    bad = df[(df['reason'].astype(str) == '') & df['val_acc'].isna()]
    assert bad.empty, f'{c}: rows missing both val_acc and reason: {bad}'
print('OK — DistributionShiftProbe ran or soft-failed cleanly')
"
```

**Falsification A (smoke)** — STOP IF: `cat results/rms_variants_train/p3_smoke/e2/zero_centered_adaptive_band_rms_norm/oob/seed_0/cell.log | tail -5` shows NaN gradients or `best_val_acc < 0.20`, OR baseline `rms_norm` shows NaN — the latter means the harness itself is broken (plan_6776f8ba Pre-Mortem 5).

**GPU binding check** — first 3 lines of any `cell.log` must include `CUDA_VISIBLE_DEVICES=0`. If they show `=1` or absent: review `sweep.run_one` and re-run with explicit `--gpu 0`.

## Chunk 1 — Core 8 × 5-exp × 5-seed (OOB)

200 cells. Wall-clock estimate: 14-17h on RTX 4090.

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
    --global-cap-s 64800 \
  2>&1 | tee logs/p3_chunk1.$(date +%F_%H%M%S).log
```

**Falsification B (chunk 1)** — STOP IF: estimated remaining wall-clock at the 6h mark exceeds 24h. Drop E2 from this chunk and rerun E2 cells as Chunk 2's prefix.

## Chunk 2 — param_matched on E1/E2/E3/E4/E5 (4 norms only)

E1 and E2 now plumb `normalization_kwargs={'use_scale': False}` via Refinement G — they are legitimately param-matched-able. 4-norm subset: `rms_norm, band_rms, zero_centered_rms_norm, zero_centered_band_rms_norm`.

```bash
cd src
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.rms_variants_train.sweep \
    --gpu 0 \
    --experiments e1,e2,e3,e4,e5 \
    --norms rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm \
    --modes param_matched \
    --regimes default \
    --seeds 0,1,2,3,4 \
    --max-cells 150 \
    --out-dir results/rms_variants_train/p3_chunk2 \
    --global-cap-s 32400 \
  2>&1 | tee logs/p3_chunk2.$(date +%F_%H%M%S).log
```

## Chunk 3 — Regime sub-experiments (stress regimes EXPECTED to break some norms)

```bash
# E5 microbench across regimes × 8 norms × 5 seeds.
cd src
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.rms_variants_train.sweep \
    --gpu 0 --experiments e5 \
    --norms rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm,adaptive_band_rms,band_logit_norm,dynamic_tanh,zero_centered_adaptive_band_rms_norm \
    --modes oob \
    --regimes default,bs_32,bs_256,lr_low,lr_high,lr_extreme,wd_zero \
    --seeds 0,1,2,3,4 \
    --max-cells 300 \
    --out-dir results/rms_variants_train/p3_chunk3_e5 \
    --global-cap-s 21600 \
  2>&1 | tee logs/p3_chunk3_e5.$(date +%F_%H%M%S).log

# E1 regimes (lr_low, lr_high, mp_fp16 + stress: lr_extreme, wd_zero, bs_4, mp_fp16_lowloss).
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.rms_variants_train.sweep \
    --gpu 0 --experiments e1 \
    --norms rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm,adaptive_band_rms,band_logit_norm,dynamic_tanh,zero_centered_adaptive_band_rms_norm \
    --modes oob \
    --regimes default,lr_low,lr_high,mp_fp16,lr_extreme,wd_zero,bs_4,mp_fp16_lowloss \
    --seeds 0,1,2,3,4 \
    --max-cells 350 \
    --out-dir results/rms_variants_train/p3_chunk3_e1 \
    --global-cap-s 43200 \
  2>&1 | tee logs/p3_chunk3_e1.$(date +%F_%H%M%S).log
```

E2 has only `default`. E3 has `default,mp_fp16,lr_extreme,wd_zero`. E4 has `default,depth_12,depth_48,wd_zero,mp_fp16_lowloss`. Cells will fail / record NaN for some (norm, regime) pairs — that is the SIGNAL of stress regimes, not a failure (E3 invariant + LESSONS L92).

**Falsification C** — STOP IF: regime chunk est. > 24h. Drop stress regimes; keep `default,lr_low,lr_high,mp_fp16` and reschedule stress to a follow-up.

## Chunk 4 — E6 causal-LM × Wikipedia 10k

E6 adds the causal-LM data point (the largest single architecture gap in v2). 4-epoch run on 10k Wikipedia articles per seed. Per-cell wall-clock ~30min on RTX 4090.

```bash
cd src
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.rms_variants_train.sweep \
    --gpu 0 --experiments e6 \
    --norms rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm,adaptive_band_rms,band_logit_norm,dynamic_tanh,zero_centered_adaptive_band_rms_norm \
    --modes oob \
    --regimes default \
    --seeds 0,1,2 \
    --max-cells 50 \
    --out-dir results/rms_variants_train/p3_chunk4_e6 \
    --global-cap-s 14400 \
  2>&1 | tee logs/p3_chunk4_e6.$(date +%F_%H%M%S).log
```

**Falsification scenario 1 (plan_6776f8ba)** — STOP IF: E6 GPU full-sweep estimate exceeds 12h. Action: drop E6 from the main sweep, ship as opt-in `--experiments e6` only.

## Compute overhead bench (Refinement A — runs once, machine-portable)

```bash
cd src
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.rms_variants_train.norm_overhead_bench \
    --out results/rms_variants_train/p3_full/overhead.csv \
  2>&1 | tee logs/p3_overhead.$(date +%F_%H%M%S).log
```

Emits 8 rows with per-norm fp32/fp16 step-time and peak memory. Consumed by `compute_overall_recommendation` to enforce the `overhead_ceiling_step_time_ratio = 1.5×` gate.

## Pre-registered analysis rules (`OVERALL_RULES` frozen at PLAN approval — D-002)

Verbatim from `report.py` (DECISION plan_2026-05-18_6776f8ba/D-002):

| Rule | Value | Meaning |
|------|-------|---------|
| `headline_pass_required_for_recommend` | `True` | A variant MUST headline-PASS over baseline at p < 0.05 (correct direction) to receive any RECOMMENDED slot |
| `hypothesis_confirm_required_for_default` | `True` | RECOMMENDED_DEFAULT additionally requires hypothesis registry CONFIRMED on at least one cell |
| `overhead_ceiling_step_time_ratio` | `1.5` | Compute overhead vs `rms_norm` must be ≤ 1.5× to receive ANY RECOMMENDED slot |
| `calibration_ece_delta_max` | `0.02` | Calibration ECE delta vs baseline must NOT regress by more than +0.02 |
| `robustness_shift_acc_delta_min` | `-0.05` | Dist-shift accuracy delta vs baseline must NOT lose more than 5 percentage points |
| `avoid_on_headline_fail` | `True` | A FAIL on any non-off-label cell triggers AVOID |
| `avoid_on_hypothesis_rejected` | `True` | A REJECTED hypothesis verdict triggers AVOID |

The 4-slot taxonomy:
- `RECOMMENDED_DEFAULT` — passes every gate including hypothesis CONFIRMED.
- `RECOMMENDED_NICHE` — passes headline + overhead + robustness, but hypothesis INCONCLUSIVE (mechanism unconfirmed, still a viable substitute).
- `NULL` — indistinguishable from baseline; no harm, no gain.
- `AVOID` — at least one AVOID trigger fires.

**Changing any of these rules after plan approval requires a new PIVOT entry in `decisions.md`** — NOT a silent edit. This is the curation-bias firewall (plan.md Pre-Mortem 3).

## Analysis day

```bash
mkdir -p results/rms_variants_train/p3_full
.venv/bin/python -c "
import pandas as pd, glob
frames = [pd.read_csv(p) for p in glob.glob('results/rms_variants_train/p3_chunk*/aggregated.csv')]
pd.concat(frames, ignore_index=True).to_csv(
    'results/rms_variants_train/p3_full/aggregated.csv', index=False)
"
# overhead.csv is the same on every run — copy in place once.
cp results/rms_variants_train/p3_full/overhead.csv \
   results/rms_variants_train/p3_full/overhead.csv 2>/dev/null || true

.venv/bin/python -m train.rms_variants_train.report \
    --in-dir results/rms_variants_train/p3_full
```

Outputs (additive vs v2):
- `summary.md` — narrative report with PASS/FAIL verdict block AND hypothesis-verdict block.
- `headline_summary.csv` — adds `hypothesis_verdict` + `late_stability_var` columns.
- `hypothesis_verdicts.csv` — per-cell observed-vs-threshold details.
- `overall_recommendation.csv` — **NEW** (Refinement B): one row per `norm_type` with `(recommendation, reason)`.
- `probes_summary.csv`, `convergence_summary.csv`, `regime_delta_summary.csv` — unchanged.

Update `RESULTS.md` Phase 3 "Results" subsection with all three verdict tables from `summary.md`. RESULTS.md lines 1-209 are BYTE-FROZEN (plan_6776f8ba I2).

## Falsification signals (v3 — adds G + H)

| ID | Signal | STOP IF | Action |
|----|--------|---------|--------|
| A  | New variant broken | smoke shows NaN or `best_val_acc < 0.20` on E2/zc-adaptive/seed_0 | inspect ctor wiring; compare to ZeroCenteredBandRMSNorm |
| B  | Chunk 1 over budget | est. remaining at 6h mark > 24h | drop E2; reschedule to Chunk 2 prefix |
| C  | Regime chunk over budget | any regime chunk est. > 24h | defer stress regimes; keep `default,lr_low,lr_high,mp_fp16` only |
| D  | GPU binding leak | `cell.log` first lines show CVD ≠ 0 | inspect `sweep.run_one` env construction; rerun with explicit `--gpu 0` |
| E  | Hypothesis registry unfalsifiable | ≥ 2 variants always INCONCLUSIVE post-sweep, OR ≥ 4 evaluate-to-N/A on standard chunks | revisit `VARIANT_HYPOTHESES` thresholds |
| F  | DistributionShiftProbe missing | E1 `dist_shift.csv` rows all have `reason=dataset_missing` on a system where `cifar10` itself loads | pivot per plan_e1f12eab pre-mortem B — hand-rolled corruption suite |
| **G** | **Overhead-bench shows > 1.5× step-time at no accuracy gain** | `overhead.csv` mean_step_ms_fp32 / baseline > 1.5 AND headline `verdict != PASS` for that norm | Set `overall_recommendation` for that norm to AVOID; document in `summary.md` |
| **H** | **E6 perplexity diverges under stress regimes for ALL 8 norms** | Chunk 3 E6 + stress (`lr_extreme`, `wd_zero`) records `final_val_perplexity = NaN` for every norm | sweep-design failure — stress is too harsh; halve the lr_extreme multiplier OR drop E6 stress regimes; document in PHASE3_PLAN v4 follow-up |
| **I** | **CIFAR-100-C unavailable** | `tfds.builder('cifar100_corrupted')` fails (confirmed at plan_6776f8ba step 7) | E2 dist_shift.csv soft-fails with `reason='dataset_missing:...'` per-corruption rows; deferred until user installs the TFDS build |

## Wall-clock budget (v3)

| Chunk | Cells | Wall-clock | Notes |
|-------|-------|------------|-------|
| Pre-warm | 0 | ~30min one-time | TFDS + HF Wikipedia cache materialisation |
| Smoke | 40 | ≤ 5 min | SC13 gate; image experiments only (E6 verified at unit-test layer SC8) |
| 1 | 200 | 14-17h | Core 8-variant × E1-E5 OOB sweep |
| 2 | 100 | ~5h | param_matched on E1-E5 (Refinement G unlocks E1/E2) |
| 3a | 280 | ~7h | E5 × 7 regimes |
| 3b | 320 | ~14h | E1 × 8 regimes |
| 4 (E6) | 24 | ~12h | 8 norms × 3 seeds × 4 epochs |
| Overhead bench | 0 (1-time) | ~3 min | Refinement A standalone |
| Analysis | -- | ~1h | Concat + regenerate report + overall_recommendation.csv |

Total optimistic: ~55h. Total realistic: 5-6 overnight chunks ≤ 18h + 1 analysis day. Trim levers: drop stress regimes (`-Chunk 3 stress entries`) → ~-10h; reduce E6 to 1 seed → ~-8h; reduce OOB seeds to 0,1,2 → halves Chunk 1 wall-clock at the cost of statistical power.

## Deferred items (NOT in this sweep)

- **Refinement F (standalone power-analysis script)** — offline, no harness coupling.
- **Refinement H/I (E1b off-label-boundary experiment, 9th norm variant)** — append-only invariant (I1) lets these land as a follow-up plan.
- **E6 stress regimes (lr_extreme, wd_zero, mp_fp16 on CLM)** — wired in the trainer but not in Chunk 4's defaults; user-opt-in via `--regimes default,lr_extreme,wd_zero --experiments e6` if budget allows.

## References

- Decisions (this plan): `plans/plan_2026-05-18_6776f8ba/decisions.md`
  - D-001 — overhead bench (standalone, not per-cell callback).
  - D-002 — `OVERALL_RULES` frozen pre-sweep (curation-bias firewall).
  - D-003 — ViT/ResNet `normalization_kwargs` plumbing (default-off bit-exact).
  - D-004 — E6 norm at all 3 positions (block-input + block-output + final pre-logits).
  - D-005 — stress regimes additive on existing maps (no abstractions).
  - D-006 — sweep + smoke remain USER-launched per HC12 / LESSONS L110/L111.
- Decisions (prior plans): plan_2026-05-18_e1f12eab (D-001 hypotheses, D-002 dist-shift, D-003 max-cells), plan_2026-05-18_63121227 (D-001 8-tuple, D-002 GPU env hard-set).
- Hypothesis registry: `src/train/rms_variants_train/hypotheses.py:VARIANT_HYPOTHESES` + `evaluate_hypothesis(variant, df) -> Verdict`.
- Frozen analysis rules: `report.py:OVERALL_RULES` + `compute_overall_recommendation`.
- Pattern-3 NLP conventions: `src/train/CLAUDE.md` — train.common.nlp helpers (`preprocess_clm_packed_dataset`, `estimate_clm_steps_per_epoch`, `build_clm_metrics`, `prepare_dict_keyed_compile`).
- Failure-mode log: `plans/LESSONS.md` (L110 sweep-launch, L111 sweep is user-operated, L113 design-plan LOC undershoot, L114 TFDS pre-warm).
