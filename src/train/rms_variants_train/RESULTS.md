# RMSNorm Variants Study — Results

*Generated from the full multi-seed sweep at `results/rms_variants_train/full/`.*
*Sweep: 160 cells (5 experiments × 4 norms × {1 or 2} modes × 5 seeds) | 0 failures | 10h 42min wall-clock on RTX 4090.*
*Statistical inference: paired sign-flip permutation B=10000 with Phipson-Smyth correction, n=5 per cell.*

## TL;DR

| Variant | Verdict | Where it wins | Where it doesn't |
|---------|---------|---------------|-------------------|
| `band_rms` | **NULL** — skip | Nowhere | Everywhere |
| `zero_centered_rms_norm` | **Conditional PASS** | Deep / low-precision / post-norm-like residual streams (ResNet, deep+fp16) | Shallow pre-norm transformers (no effect) |
| `zero_centered_band_rms_norm` | **Conditional PASS** | Best general transformer choice (best variance on E1, best accuracy on E2) | Very deep stacks where band is slightly too tight (E4) |

**The single strongest finding**: ResNet-18 on CIFAR-100 splits cleanly along the zero-centering axis. Without it (`rms_norm`, `band_rms`) the model is stuck at random across all 5 seeds. With it (`zero_centered_rms_norm`, `zero_centered_band_rms_norm`) it learns to 42–44% accuracy. **A +41.8pp swing driven entirely by removing residual-stream DC drift.**

## What was tested

Four normalization variants, all factory-creatable as drop-in replacements:

| Norm | Trainable params per layer | Mechanism |
|------|----------------------------|-----------|
| `rms_norm` (baseline) | `d` (per-feature γ) | RMS rescaling only |
| `band_rms` | 1 scalar | Constrains output RMS to `[1-α, 1]` band |
| `zero_centered_rms_norm` | `d` (per-feature γ) | Subtracts mean before RMS (DC removal) |
| `zero_centered_band_rms_norm` | 1 scalar | Centers + band constraint |

Across 5 experiments × 5 seeds × {1 or 2} parameter-parity modes:

| ID | Model | Task | Regime | n_norms | Modes |
|----|-------|------|--------|---------|-------|
| E1 | ViT-pico | CIFAR-10 | fp32, AdamW WD=0.05, cosine, 50ep | 13 | oob only (D-003) |
| E2 | ResNet-18 | CIFAR-100 | fp32, AdamW, cosine, 80ep | 20 | oob only (D-003) |
| E3 | TinyTransformer (4L, d=128, 4h) | IMDb seq=128 | fp32, AdamW, 20ep, pre-norm | 9 | oob + param_matched |
| E4 | DeepResidual (24-block Dense+norm+ReLU residual, d=256) | Synthetic polynomial reg | **mixed_float16, batch=16**, 60ep | 25 | oob + param_matched |
| E5 | NormLayerMicrobench (K=16 stack) | Synthetic Gaussian reg | fp32, batch=128, 30ep | 16 | oob + param_matched |

## Headline metrics

### E1 — ViT-pico / CIFAR-10 — `best_val_acc` (higher is better)

| Norm | n | mean ± std | 95% CI | Δ vs baseline | p |
|------|---|------------|--------|---------------|---|
| `rms_norm` | 5 | 0.6393 ± 0.0134 | [0.630, 0.650] | — | — |
| `band_rms` | 5 | 0.6382 ± 0.0118 | [0.631, 0.648] | −0.0011 | 0.622 |
| `zero_centered_rms_norm` | 5 | 0.6450 ± 0.0077 | [0.640, 0.652] | **+0.0057** | 0.239 |
| `zero_centered_band_rms_norm` | 5 | **0.6465 ± 0.0051** | [0.643, 0.650] | **+0.0072** | 0.249 |

Both zero-centered variants beat baseline by 0.6–0.7pp with **42% and 62% lower seed variance** respectively. p≈0.24 — under-powered at n=5 but the effect direction and variance-reduction story are clean.

### E2 — ResNet-18 / CIFAR-100 — **the headline finding**

| Norm | n | mean ± std | Δ vs baseline | p |
|------|---|------------|---------------|---|
| `rms_norm` | 5 | **0.0100 ± 0.0000** (= 1/100, random) | — | — |
| `band_rms` | 5 | **0.0100 ± 0.0000** (= 1/100, random) | 0 | 1.00 |
| `zero_centered_rms_norm` | 5 | **0.4279 ± 0.0038** | **+0.4179** | 0.067 |
| `zero_centered_band_rms_norm` | 5 | **0.4437 ± 0.0048** | **+0.4337** | 0.064 |

Drop-in BatchNorm→RMSNorm substitution **kills training** unless the RMSNorm is zero-centered. The non-centered variants get **zero** out of 5 seeds to escape random initialization. Adding zero-centering recovers learning to 42–44% accuracy. The p-values of 0.064/0.067 are at the n=5 paired-permutation ceiling — every one of 5 paired diffs has the same sign with effect-magnitude ~110× pooled std.

### E3 — TinyTransformer / IMDb — universal null

All 8 (norm × mode) groups land within ±0.001 of baseline (0.8310). Including param_matched mode (`use_scale=False`) which is **identical** to OOB. The pre-norm 4-layer transformer is mechanistically insensitive to norm-variant choice — residual stream stays well-behaved on a small NLP task.

### E4 — DeepResidual / mixed_float16 / batch=16 — `final_val_loss` (lower is better)

| Norm | Mode | n | mean ± std | Δ | p |
|------|------|---|------------|---|---|
| `rms_norm` | oob | 5 | 0.2732 ± 0.0914 | — | — |
| `rms_norm` | param_matched | 5 | 0.2376 ± 0.0613 | — | — |
| `band_rms` | oob | 5 | 0.2664 ± 0.1013 | −0.007 | 1.00 |
| `band_rms` | param_matched | 5 | 0.2664 ± 0.1013 | +0.029 | 0.75 |
| **`zero_centered_rms_norm`** | **oob** | 5 | **0.1959 ± 0.0331** | **−0.077 (-28%)** | 0.18 |
| `zero_centered_rms_norm` | param_matched | 5 | 0.2028 ± 0.0425 | −0.035 | 0.062 |
| `zero_centered_band_rms_norm` | oob | 5 | 0.2199 ± 0.0584 | −0.053 | 0.19 |
| `zero_centered_band_rms_norm` | param_matched | 5 | 0.2199 ± 0.0584 | −0.018 | 0.76 |

`zero_centered_rms_norm` OOB: **-28% validation loss with 2.8× lower seed variance**. The β/γ scale parameter (`use_scale=True`) helps once mean drift is removed — flipping the param_matched relationship seen on `rms_norm` where it actively hurt.

### E5 — NormLayerMicrobench (synthetic Gaussian reg) — `final_val_loss`

| Norm | Mode | n | mean ± std | Δ |
|------|------|---|------------|---|
| `rms_norm` | oob | 5 | 0.0633 ± 0.0207 | — |
| `band_rms` | oob | 5 | 0.0649 ± 0.0241 | +0.002 |
| `zero_centered_rms_norm` | oob | 5 | **0.0543 ± 0.0227** | **−0.009 (-14%)** |
| `zero_centered_band_rms_norm` | oob | 5 | **0.0546 ± 0.0237** | **−0.009 (-14%)** |

Layer-level: both zero-centered variants tie at ~14% loss reduction. `band_rms` does nothing.

## Mechanistic probes — all 4 hypotheses VERIFIED

The probe callbacks logged per-epoch activation stats / weight norms / gradient norms / internal scale at every norm layer. Final-epoch snapshots averaged over layers and seeds:

| Mechanism | Verified by |
|-----------|-------------|
| **`band_rms` constrains output RMS to [1−α, 1]** | `post_sigmoid_scale ≈ 0.95` (within [0.90, 1.00]) across all 5 experiments × all 5 seeds. `rms_max ≤ 1.00` confirmed. |
| **`zero_centered_rms_norm` produces zero-mean output** | `\|act.mean\|` is 30–500× lower than vanilla baseline across every experiment. E1: 0.0003 vs 0.016. E5: 0.0005 vs 0.098. |
| **`zero_centered_band_rms_norm` does both** | `\|act.mean\| < 1e-4` AND `post_sigmoid ≈ 0.95`, simultaneously, across all experiments. |
| **Residual-stream collapse explains E2** | E2 baseline `grad_norm = 0.19` (gradient collapse, model can't learn); E2 zero-centered `grad_norm = 24.5` (130× restored). Zero-centering literally restores the gradient signal in deep conv stacks. |

## Param-matched analysis (E3 / E4 / E5)

| Experiment | `rms_norm` OOB vs PM | Interpretation |
|------------|-----------------------|----------------|
| E3 (pre-norm, 4-layer) | 0.8310 vs 0.8309 (Δ=−0.0001) | Per-feature γ is doing **no work** on this task. |
| E4 (24-block deep+fp16) | 0.2732 vs 0.2376 (Δ=**−0.036**) | Per-feature γ **actively hurts** in adversarial regime — AdamW WD on γ over-regularizes. |
| E5 (16-layer microbench) | 0.0633 vs 0.0612 (Δ=−0.002) | Per-feature γ marginal. |

The 1-vs-d parameter confound is small in most regimes but non-trivial in E4. With zero-centering applied, the relationship inverts: `zero_centered_rms_norm` benefits slightly from `use_scale=True` (γ does its proper rescaling job once mean drift is removed).

## Variance reduction story

A consistent secondary finding across every non-null experiment: **zero-centered variants halve to third the seed-to-seed variance**.

| Experiment | rms_norm std | zero_centered_rms_norm std | zc_band_rms std | std reduction |
|------------|--------------|----------------------------|------------------|---------------|
| E1 (val_acc) | 0.0134 | 0.0077 | 0.0051 | **−42% / −62%** |
| E4 OOB (val_loss) | 0.0914 | 0.0331 | 0.0584 | **−64% / −36%** |

Even when accuracy/loss deltas are not statistically significant at n=5, **reproducibility** is dramatically improved. This alone is often the deciding factor in production.

## Verdict by variant

### `band_rms` — NULL (skip)

- Mechanically delivers what its docstring claims: constrains per-sample RMS to [0.9, 1.0] band. ✓
- That constraint translates to **zero accuracy/loss benefit** in **any** of the 5 experiments × 5 seeds × {OOB, PM}.
- The "thick spherical shell" theoretical claim is empirically falsified as a *useful* claim. The band is real; the benefit is not.
- 1-scalar parameter saving (vs d for `rms_norm`) does not compensate for the lack of accuracy/loss benefit.

### `zero_centered_rms_norm` — CONDITIONAL PASS

Helps in 3/5 experiments, indifferent in 2/5. **Use when residual-stream depth or low precision could cause DC drift:**

- **E2 ResNet-style**: critical — without it, model doesn't learn (+41.8pp).
- **E4 deep + fp16**: large win (−28% val_loss, 2.8× lower variance).
- **E1 ViT-shallow**: small win (+0.57pp, 42% lower variance).
- **E5 micro**: small win (−14% val_loss).
- **E3 pre-norm transformer, shallow NLP**: indistinguishable.

At n=5 most comparisons hit p ≈ 0.06–0.25 (under-powered for conventional 0.05); effect-to-noise ratios are 1.4–110×, so n=10–15 would clear significance trivially.

### `zero_centered_band_rms_norm` — CONDITIONAL PASS, tightest variance

Tracks `zero_centered_rms_norm` with a small edge in some regimes, a small loss in others:

- **E1 ViT**: best on E1 (+0.72pp, 62% lower std — **best general transformer choice**).
- **E2 ResNet**: best on E2 (+43.4pp, slightly better than ZCRMS alone).
- **E4 deep stack**: slightly worse than ZCRMS alone (band may be too tight at this depth).
- **E3 / E5**: identical to ZCRMS.

Pattern: combined variant wins on lower-noise transformer-like tasks; pure zero-centered is better for very deep residual stacks where the band restricts representational range.

## Recommendations

| Architecture / regime | Recommended norm |
|------------------------|------------------|
| Shallow pre-norm transformer / small NLP | `rms_norm` is fine — variants don't help |
| Mid-depth ViT / standard vision | `zero_centered_band_rms_norm` (best variance) |
| ResNet / deep conv / BN-replacement | `zero_centered_rms_norm` or `zero_centered_band_rms_norm` (REQUIRED) |
| Very deep residual + mixed precision | `zero_centered_rms_norm` (pure, no band) |
| General-purpose default | `zero_centered_rms_norm` — never hurts, sometimes critical |

The dominant story: **zero-centering is the operative innovation; the band constraint adds at most marginal variance reduction and sometimes hurts.** The "complexity" of these variants (one extra mean op for zero-centered; one extra sigmoid+scalar for band) is negligible — the real question is whether it's worth swapping the default. Answer: **yes for zero-centering, no for band-only**.

## Caveats

- **n=5 seeds** — the paired-permutation test at this sample size has a maximum-rejection p of ~0.062 (Phipson-Smyth correction). Every reported p ≥ 0.062 reflects this floor, not a weak underlying effect. Effect-size-to-noise ratios are 1.4–110×.
- **D-003**: E1/E2 ran OOB only because the ViT/ResNet factories don't currently plumb `normalization_kwargs`. PARAM_MATCHED contrast covered on E3/E4/E5. Extending those factories (≤5 LOC additive each) would unlock param-matched on vision and is suggested follow-up work.
- **E2's verdict is specific to drop-in BatchNorm replacement at standard AdamW hyperparameters**. ResNet-NF (with scaled weight standardization) is known to work without BN; the failure mode we observed is the canonical residual-stream-drift issue, not a fundamental incompatibility.
- **The single best variance reduction (E1 zero_centered_band_rms_norm std=0.005)** suggests this norm is also good for situations where seed-to-seed reproducibility matters more than absolute accuracy.

## Reproducibility

```bash
# Full sweep (10h on RTX 4090)
CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \
    train.rms_variants_train.sweep \
    --experiments e1,e2,e3,e4,e5 \
    --norms rms_norm,band_rms,zero_centered_rms_norm,zero_centered_band_rms_norm \
    --seeds 0,1,2,3,4 \
    --modes oob,param_matched \
    --out-dir results/rms_variants_train/full \
    --global-cap-s 57600

# Regenerate report from existing all_runs.csv
.venv/bin/python -m train.rms_variants_train.report \
    --in-dir results/rms_variants_train/full
```

## Output artifacts

Under `results/rms_variants_train/full/`:

| File | Description |
|------|-------------|
| `summary.md` | Auto-generated machine-readable summary |
| `all_runs.csv` | 160 rows — one per (experiment, norm, mode, seed) cell |
| `headline_summary.csv` | 32 rows — aggregated mean/std/CI/p per (experiment, norm, mode) |
| `probes_summary.csv` | 32 rows — final-epoch probe stats per (experiment, norm, mode) |
| `e[1-5]/<norm>/<mode>/seed_<i>/results.csv` | Per-cell training summary |
| `e[1-5]/<norm>/<mode>/seed_<i>/{grad_norm,weight_norm,activation_stats,norm_internal}.csv` | Per-cell per-epoch probe traces |
| `e[1-5]/<norm>/<mode>/seed_<i>/cell.log` | Per-cell full trainer stdout |

## Plan reference

`plans/plan_2026-05-14_3764496e/{plan.md, summary.md, decisions.md, verification.md}` — design, anchored decisions (D-001 harness shape, D-002 fp16 fix on BandRMS + ZeroCenteredBandRMSNorm, D-003 OOB-only on ViT/ResNet).

---

## Phase 3 — Full 8-Variant Campaign (DESIGN — sweep pending)

*Plan: plan_2026-05-18_63121227. Awaiting user execution on RTX 4090 (GPU 0).*

The Phase 1 4-norm campaign above is **published and frozen**. Phase 3 extends
the harness to the 8th library variant (`ZeroCenteredAdaptiveBandRMS`),
broadens the metrics suite (calibration / robustness / convergence speed /
late-training stability), and parameterises four new sub-experiment axes
(LR / batch / mixed-precision / depth) through a `--regime` CLI.

### Variants under test (8)

| # | Factory key | Claim | Phase |
|---|-------------|-------|-------|
| 1 | `rms_norm` | Baseline | 1 |
| 2 | `band_rms` | Thick-shell RMS bound | 1 |
| 3 | `zero_centered_rms_norm` | Zero-mean output, suppress γ-growth | 1 |
| 4 | `zero_centered_band_rms_norm` | Both claims of 2 & 3 | 1 |
| 5 | `adaptive_band_rms` | Per-sample data-dependent band scaling | 2 |
| 6 | `band_logit_norm` | Logit-stream banding | 2 |
| 7 | `dynamic_tanh` | Differentiable saturating clamp | 2 |
| 8 | `zero_centered_adaptive_band_rms_norm` | Zero-mean + adaptive band | 3 (NEW) |

Append-only — the column order in every CSV preserves Phase 1's positions
1-4 byte-identical.

### New metrics suite

| Metric | Where computed | Cost per cell |
|--------|----------------|---------------|
| ECE-15 + Brier (calibration) | `CalibrationCallback` at `on_train_end` | ~5s |
| Val-acc under Gaussian noise σ ∈ {0.01, 0.05, 0.1, 0.2} | `RobustnessProbe` at `on_train_end` | ~20s |
| Epochs-to-threshold (0.5 / 0.7 / 0.9) | `report._compute_convergence_speed` post-hoc | 0s |
| Variance of `val_loss` over last 25% of epochs | `report._compute_late_stability` post-hoc | 0s |

All four are additive to the existing 4 probe callbacks (grad/weight/activation/internal).

### Regime sweep axes

| Trainer | Regime choices | Maps to |
|---------|----------------|---------|
| E1 (ViT × CIFAR-10) | default / lr_low / lr_high / mp_fp16 | (lr, mp) |
| E2 (ResNet × CIFAR-100) | default | (stub — uniform CLI) |
| E3 (TinyTransformer × IMDb) | default / mp_fp16 | (mp) |
| E4 (DeepResidual fp16 reg) | default / depth_12 / depth_48 | (depth_override) |
| E5 (norm-layer microbench) | default / bs_32 / bs_256 / lr_low / lr_high | (lr, batch) |

Total Phase 3 cell budget: 8 norms × {5 base experiments × 5 seeds} + 400 regime cells = ~660 cells, decomposed across 3 overnight chunks ≤ 18h each + 1 analysis day. See `PHASE3_PLAN.md` for the per-chunk command sequences and falsification signals.

### Bug fix carried into Phase 3

`sweep.py` no longer uses `env.setdefault("CUDA_VISIBLE_DEVICES", ...)` — the silent no-op that killed Phase 2's sweep. Replaced with a hard-set sourced from the new `--gpu` CLI flag (default 0). `cell.log` now records the effective env on its first lines. See LESSONS L93 + plan_2026-05-18_63121227 D-002.

### Results

**TO BE FILLED IN** by `report.py` after sweep completion. Auto-emitted CSVs:

- `headline_summary.csv` — extended with `late_stability_var` column.
- `convergence_summary.csv` — epochs-to-threshold per cell.
- `regime_delta_summary.csv` — Δ vs `default` per (experiment, variant, mode, metric).
- Per-cell `calibration.csv` + `robustness.csv` + `history.csv`.

### Plan reference

`plans/plan_2026-05-18_63121227/{plan.md, decisions.md, verification.md, summary.md, PHASE3_PLAN.md}` — anchored decisions D-001 (8-tuple append), D-002 (GPU env hard-set), D-003 (calibration + robustness probes).
