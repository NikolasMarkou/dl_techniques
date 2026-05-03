# Clifford-Algebra-Compliant Downsampling — Experimental Findings

**Status:** Complete. Single seed. CIFAR-100, 100 epochs.
**Hardware:** RTX 4090 (GPU 0), serial. ~21h total wall time.
**Result artifact:** `results/cliffordnet_downsampling_experiments_20260430_164707/comparison.csv`
**Source:** `src/train/cliffordnet/train_downsampling_experiments.py`
**Block:** `dl_techniques.layers.geometric.clifford_block.CliffordNetBlockDSv2`

---

## TL;DR

Eleven downsampling-block configurations tested at strides>1 in a 4-stage
backbone (96, 192, 384, 768 channels; 12 blocks; ~18M params). The
analysis prior (`analyses/analysis_2026-04-30_41b5e415/summary.md`)
predicted that decoupling stream and skip pool families and switching to
pixel-unshuffle skip would deliver +0.5..+1.0pp at multi-stage scale.
Empirically, the simplest single change won:

| Rank | Variant | best val_acc | Δ vs V0 | Predicted | Verdict |
|---:|---|---:|---:|---:|---|
| **1** | **V1_blur_blur** (BlurPool stream + BlurPool skip) | **0.7573** | **+0.59pp** | +0.2..+0.5 | ✓ slightly above |
| 2 | V7_int_abs (V4 + ctx_mode=abs at strides>1) | 0.7562 | +0.48pp | +0.1..+0.3 over V4 | ✓ exceeded |
| 3 | V4_blur_pxsh_int (BlurPool + pxsh + internal exp.) | 0.7516 | +0.02pp | +0.7..+1.4 | ✗ flat |
| 4 | V0_baseline_avg_avg (anchor) | 0.7514 | — | anchor | — |
| 5 | V10_resnetd (AvgPool + 1x1, both paths) | 0.7498 | −0.16pp | +0.1..+0.3 | ✗ flat |
| 6 | V3_gauss_pxsh | 0.7496 | −0.18pp | +0.6..+1.1 | ✗ wrong sign |
| 7 | V6_pyramid_diff | 0.7473 | −0.41pp | +0.8..+1.5 | ✗ wrong sign |
| 8 | V11_kitchen_sink (V5+V6 stack) | 0.7463 | −0.51pp | +0.8..+1.5 | ✗ wrong sign |
| 9 | V12_k=3,s=2 (negative control) | 0.7446 | −0.68pp | −0.3..−0.8 | ✓ within band |
| 10 | V2_blur_pxsh (BlurPool stream + pxsh skip) | 0.7372 | −1.42pp | +0.5..+1.0 | ✗ wrong sign |
| 11 | V5_GN-everywhere | 0.7292 | −2.22pp | +0.8..+1.6 | ✗ confounded (see caveat) |

**Adopted default:** `CliffordNetBlockDSv2` ships with `stream_pool="blur"`,
`skip_pool="blur"` — the V1 configuration. All other defaults match the
v1 `CliffordNetBlockDS` block.

---

## Method

### Backbone
4-stage hierarchy with stride-2 transitions inside the block (not between
blocks). Stem: patch1 (no spatial reduction). Stages
``(96, 2), (192, 2), (384, 4), (768, 4)``. The first block of stages 1–3
is the strided ``CliffordNetBlockDSv2`` whose pool, ctx_mode, channel-
expansion, and norm choices are the variant's signature; the remaining
n−1 blocks per stage are isotropic at the same channel count. Head:
GAP → LN → Dense.

### Training
AdamW (β1=0.9, β2=0.999, weight_decay=0.1), cosine LR schedule with peak
1e-3, 5-epoch warm-up, 100 epochs total. Batch 128. CIFAR-10
AutoAugment + RandomErasing(p=0.25). 1 seed per variant. Total wall:
~115 min/variant × 11 = ~21 h serial on RTX 4090.

### Variant axes (from the analysis design space)

| Axis | Knob (CliffordNetBlockDSv2) | Values explored |
|---|---|---|
| A | `stream_pool` | avg, max, blur, gaussian_dw, pixel_unshuffle, resnetd |
| B | `skip_pool` | same set |
| C | `kernel_size` | 3, 5, 7 (must satisfy `k ≥ 2*strides`) |
| D | `ctx_mode` | diff, abs, pyramid_diff |
| E | `out_channels` | None (external 1×1) vs C_next (internal) |
| G | `ctx_norm_type` | bn, gn, ln, none |
| H | `layer_scale_init` (CLI) | 1e-5 default, swept via `--layer-scale-init` |

V8 (full-resolution geometric product, axis F) and V9 (grade-aware
grouped pool) were deferred — see `plans/DECISIONS.md` D-001.

---

## Findings

### 1. The simplest principled change won
V1 (replace avg-pool with the Zhang [1,3,3,1]/8 BlurPool on **both**
stream and skip paths, no other change) is the empirical winner. The
+0.59pp gain over V0 sits at the upper end of the analysis's predicted
+0.2..+0.5pp band. Anti-aliasing matters; nothing else helped consistently.

### 2. The ctx_mode lever is much larger than predicted
V7 (V4 + `ctx_mode=abs` at strides>1) came in second at +0.48pp over V0
— the analysis predicted only +0.1..+0.3pp over V4 for axis D. Dropping
the broken `diff` Laplacian semantics at stage transitions gives a real
gain. **`ctx_mode="diff"` is genuinely a bug at strides>1**; either `abs`
or a future principled replacement should be the default for strided
blocks.

### 3. Pixel-unshuffle skip alone is harmful
V2 (BlurPool stream + pixel-unshuffle skip + 1×1 projection, no internal
channel expansion) lost 1.42pp vs V0. The "info-preserving gradient
highway" prior (analysis H11) does not pay off at this scale; the extra
1×1 projection on the skip path may be fighting the SRGP path early in
training. V2 was the analysis's HIGH-priority pick — it underperformed
both the baseline and V1.

### 4. Internal channel expansion (axis E) is a clean win conditional on
   pixel-unshuffle skip
V4 vs V2 isolates axis E exactly (+1.44pp). V4 (internal expansion)
recovers V2's deficit and ties V0. Internal expansion is therefore a
**fix for** the pixel-unshuffle-skip configuration, not an independent
gain — V4 ≈ V0, not V0 + axis E.

### 5. Stacking does not compose
V11 (kitchen sink: V5 + V6 stacked on V4) lost 0.51pp vs V0 and 1.10pp
vs V1. Predicted +0.8..+1.5pp. The "principled" levers don't add — they
cancel. Either each lever is fragile to the others' presence, or the
analysis's per-axis effect sizes are over-attributed.

### 6. Negative control held
V12 (`kernel_size=3, strides=2`, violating k≥2s) landed at −0.68pp,
inside the predicted −0.3..−0.8pp band. The k≥2s anti-alias-support
necessary condition is empirically validated.

### 7. ResNet-D is a wash
V10 (`AvgPool + 1×1` on both paths) ties V0 within noise (−0.16pp). The
literature-standard pattern offers no gain over plain `AvgPool` at this
scale; the 1×1 projection it adds doesn't help.

### 8. GroupNorm everywhere is harmful — but the test was confounded
V5 (V4 + GroupNorm context-stream norm at every stage) lost 2.22pp.
**Caveat:** the analysis specified "GroupNorm at H/4 and below" — only
the deeper stages where BN's sample-thinness shows up. Our script applied
GN at *all* stages, including the high-resolution ones where BN is fine.
This is a script-level deviation; V5's number does not falsify axis G,
only "GN-everywhere." A clean re-run of V5 with stage-conditional norm
is a useful follow-up.

---

## Caveats

- **Single seed.** All deltas should be read against an unmeasured run-
  to-run variance ≈ ±0.2..0.4pp at this scale (cf. E04→E05 spreads in
  `VARIATIONS_COMPARISON.md`). The V0/V4/V10/V3/V6 cluster
  (0.7473..0.7516) is plausibly inside seed noise; V1 vs V7 (+0.11pp)
  is plausibly inside noise. V2 and V5's deficits, V1/V7's leads, and
  V12's regression are large enough to credit.
- **18M params, not 10M.** The 4-stage 96-192-384-768 backbone landed
  at ~18M params, not the ~10M the analysis was implicitly calibrated
  against. Magnitudes (not signs) of all predictions should be discounted
  to account for this scale shift.
- **V5 confounded** as noted above.
- **V8 and V9 not measured** (deferred).
- **Macro-arch coupling untested.** All variants share the same stage
  layout. The H_SCOPE_MACRO hypothesis (analysis §6) is open.

---

## Adopted default

`CliffordNetBlockDSv2` ships with the **V1 configuration** as the
default:

```python
CliffordNetBlockDSv2(
    channels=...,
    shifts=[1, 2],
    stream_pool="blur",   # was "avg" pre-experiment
    skip_pool="blur",     # was "avg" pre-experiment
    # All other defaults unchanged: kernel_size=7, ctx_mode="diff",
    # ctx_norm_type="bn", layer_scale_init=1e-5, out_channels=None.
)
```

For users who want the in-block downsampling pattern with no further
choice, instantiating the block with no pool overrides yields the
empirically best configuration measured by this experiment.

For users wanting to push higher than V1, the recommended next step is
`stream_pool="blur"`, `skip_pool="blur"`, plus `ctx_mode="abs"` for
strided transitions (V7-style). That requires per-block plumbing
because non-strided blocks should keep `ctx_mode="diff"`.

---

## How to reproduce

```bash
# Smoke test (~75 min total, all 11 variants × 3 epochs × batch 32)
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_experiments \
    --variant all --smoke-test --gpu 0

# Full sweep (~21 h serial on RTX 4090)
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_experiments \
    --variant all --epochs 100 --batch-size 128 --gpu 0

# Single variant
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_experiments \
    --variant V1_blur_blur --epochs 100 --batch-size 128 --gpu 0
```

---

## Files

- `src/train/cliffordnet/train_downsampling_experiments.py` — campaign script
- `src/dl_techniques/layers/geometric/clifford_block.py` — `CliffordNetBlockDSv2`
- `src/dl_techniques/layers/blur_pool.py` — `BlurPool2D` (Zhang binomial)
- `src/dl_techniques/layers/pixel_unshuffle.py` — `PixelUnshuffle2D`
- `analyses/analysis_2026-04-30_41b5e415/summary.md` — design-space report (priors)
- `results/cliffordnet_downsampling_experiments_20260430_164707/comparison.csv` — raw results

---

## Follow-Up Campaign (iter-2)

After iter-1 produced V1 (blur/blur, 0.7573, +0.59pp) and V7 (blur/pxsh
+abs+int, 0.7562, +0.48pp) as the two strongest single variants, four
open questions remained:

1. **What is σ on this benchmark?** The +0.59pp / +0.48pp leads sit close
   to estimated ±0.2..0.4pp seed noise. Without σ we cannot credit
   sub-half-point deltas.
2. **Does V1 + V7 stack?** The prior campaign never measured V1's
   stream/skip with V7's ctx_mode on the V0 substrate (V11 stacked
   different axes on V4 and lost).
3. **Is GN-at-H/4-and-below clean?** V5 applied GN at *every* stage,
   confounding the result.
4. **Is LN viable at low resolution?** LN is the modern transformer
   default for sample-thin regimes; never measured here.

### Variant table (iter-2)

All four follow-up variants use the **V1 substrate** (blur/blur
stream+skip, external channel expansion) and stack one axis on top.
A0 anchor seed-panel uses the existing V0/V1/V7 entries with the new
`--seed` arg.

| Block | Variant | What it changes vs V1 | Predicted Δ |
|-------|---------|-----------------------|------------:|
| A0 | `V0/V1/V7 × seed=42,137,2025` | seed only | establishes σ |
| B1 | `B1_blur_blur_abs` | `ctx_mode=abs` at strided transitions | +0.1..+0.5pp vs V1 |
| B2 | `B2_blur_blur_pyrdiff` | `ctx_mode=pyramid_diff` at strided | -0.2..+0.3pp vs V1 |
| C1 | `C1_blur_blur_gn_late` | GN ctx-norm at stages 2,3 (BN at stage 1) | +0.0..+0.3pp vs V1 |
| C2 | `C2_blur_blur_ln_late` | LN ctx-norm at stages 2,3 | -0.2..+0.2pp vs V1 |

13 runs total. All variants are ~18M params (matching V0..V12).

### Engineering changes (committed iter-1/step-1-3)

* `--seed <int>` CLI arg seeds Python/NumPy/TF/Keras at startup. Default
  None preserves the original non-deterministic behaviour. Seed is
  persisted into each run's `config.json`.
* Per-stage `ctx_norm_type` override: pass either a scalar str
  (uniform across stages 1/2/3 — back-compat) OR a length-3 list
  `[stage1, stage2, stage3]`. Used by C1 (`["bn","gn","gn"]`) and
  C2 (`["bn","ln","ln"]`).
* Determinism verified: two seeded builds produce bit-identical initial
  weights and forward-pass outputs.

### Launch command (full 13-run STRETCH campaign)

Serial on GPU 0. ~115 min/run × 13 runs ≈ 25h.

```bash
# A0: anchor seed panel — 9 runs (V0/V1/V7 × seeds 42/137/2025)
for V in V0_baseline_avg_avg V1_blur_blur V7_blur_pxsh_int_abs; do
  for S in 42 137 2025; do
    MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_experiments \
        --variant "$V" --seed "$S" --epochs 100 --batch-size 128 --gpu 0
  done
done

# B + C: 4 follow-up cells, single seed (42)
for V in B1_blur_blur_abs B2_blur_blur_pyrdiff C1_blur_blur_gn_late C2_blur_blur_ln_late; do
  MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_experiments \
      --variant "$V" --seed 42 --epochs 100 --batch-size 128 --gpu 0
done
```

Each invocation creates its own
`results/cliffordnet_downsampling_experiments_<ts>/` run-root with a
per-run `comparison.csv`. To merge results across runs, concatenate the
13 individual `comparison.csv` files post-hoc (the script does NOT
chain runs — each launch is its own root).

### Expected outcomes

* If A0 σ ≥ 0.4pp → V1's +0.59pp lead over V0 is borderline.
  Re-evaluate the entire iter-1 ranking under the new error bar.
* B1 wins (>V1 + 1σ) → ctx_mode=abs at strides matters; recommended
  default for downsampling-heavy backbones.
* C1 wins (>V1 + 1σ) → GN-at-H/4-and-below is a free improvement.
  Roll into the default block factory.
* B2/C2 not significantly different from V1 → confirms axes D and G
  saturated; close them.

### Smoke-test status (committed iter-1/step-1-3)

All 4 new variants pass `--smoke-test` (3 epochs / batch 32) on RTX 4090:

| Variant | Smoke val_acc (3 ep) | Wall | Notes |
|---------|---------------------:|-----:|-------|
| B1_blur_blur_abs | 0.4993 | 7.6 min | round-trip-load OK |
| B2_blur_blur_pyrdiff | 0.4945 | 7.7 min | — |
| C1_blur_blur_gn_late | 0.5140 | 7.6 min | round-trip-load OK |
| C2_blur_blur_ln_late | 0.5128 | 7.7 min | — |

Smoke numbers are not predictive of 100-epoch outcomes — they only
prove forward+backward+save are wired correctly.

### Iter-2 results (executed 2026-05-02, 24h53m wall, RTX 4090)

13/13 runs completed cleanly, no NaN, no crashes. Aggregated results
in ``results/iter2_sweep_20260502_031640/aggregated.csv`` and
``aggregated.md``.

#### Anchor seed panel (axis-A0): σ-calibration

| Variant | seed=42 | seed=137 | seed=2025 | mean | σ (pp) |
|---------|--------:|---------:|----------:|-----:|-------:|
| V0_baseline_avg_avg | 0.7537 | 0.7477 | 0.7517 | **0.7510** | 0.31 |
| V1_blur_blur | 0.7564 | 0.7541 | 0.7551 | **0.7552** | 0.12 |
| V7_blur_pxsh_int_abs | 0.7499 | 0.7472 | 0.7477 | **0.7483** | 0.14 |

**Pooled within-variant σ ≈ 0.20pp** across the 9 anchor seeds. The
V0 σ of 0.31pp is the conservative single-variant estimate; the V1
and V7 panels happened to land tighter.

#### B/C single-seed cells (seed=42)

| Variant | best val_acc | Δ vs V1 mean | Verdict |
|---------|-------------:|-------------:|---------|
| **B1_blur_blur_abs** | 0.7562 | +0.10pp | within noise — axis D null on V1 substrate |
| B2_blur_blur_pyrdiff | 0.7534 | -0.18pp | within noise (slight slack) — pyramid_diff null |
| C1_blur_blur_gn_late | 0.7472 | -0.80pp | clearly worse — GN-late harmful (clean test) |
| C2_blur_blur_ln_late | 0.7510 | -0.42pp | worse — LN-late also harmful |

#### Verdicts (all four open questions answered)

1. **σ on this benchmark = ~0.20pp pooled** (0.31pp single-variant for V0).
   The iter-1 V1 lead of +0.59pp is real (≈2 pooled-σ); the iter-1 V7 lead of
   +0.48pp was **a single-seed high-tail outlier**. V7's true mean is
   −0.27pp **below** V0 (~1.3σ deficit).
2. **V1 + V7 does not stack** — but B1 (V1+abs@strided) ≈ V1 within noise,
   so axis D (`ctx_mode=abs` at strided transitions) is *neutral* on V1
   substrate. The V7 deficit comes from V7's pxsh-skip + internal-expansion
   combination, NOT from the abs mode itself.
3. **GN-at-H/4-and-below is harmful** (−0.80pp). The "axis G is good but
   was confounded" hypothesis is falsified by the clean test. V5's
   GN-everywhere number was harmful for the same reason, just amplified.
4. **LN-at-H/4-and-below is also harmful** (−0.42pp). Alternative norms
   at low resolution do not help on this benchmark with this recipe.

#### Final ranking (across iter-1 + iter-2)

| Rank | Variant | best | mean (n) | seed σ | vs V0 mean (Δ pp / σ-units) |
|---:|---------|-----:|---------:|-------:|----------------------------:|
| 1 | **V1_blur_blur** | 0.7564 | 0.7552 (3) | 0.12 | **+0.42pp / +2.1σ** ✓ |
| 2 | B1_blur_blur_abs | 0.7562 | 0.7562 (1) | — | +0.52pp / +2.6σ (single seed) |
| 3 | B2_blur_blur_pyrdiff | 0.7534 | 0.7534 (1) | — | +0.24pp / +1.2σ (single seed) |
| 4 | V0_baseline_avg_avg | 0.7537 | 0.7510 (3) | 0.31 | (anchor) |
| 5 | C2_blur_blur_ln_late | 0.7510 | 0.7510 (1) | — | +0.00pp |
| 6 | V7_blur_pxsh_int_abs | 0.7499 | 0.7483 (3) | 0.14 | −0.27pp / −1.3σ |
| 7 | C1_blur_blur_gn_late | 0.7472 | 0.7472 (1) | — | −0.38pp |

The currently-shipped V1 default in ``CliffordNetBlockDSv2`` is the
empirical winner across every test run so far on this benchmark. No
default change recommended.

#### What further runs would be useful (not executed)

* **B1 multi-seed**: B1 single-seed lands +0.10pp above V1 mean. Within
  V1's 0.12pp σ, so the result is "≈ V1 within noise". A 2-seed B1
  follow-up would tell us if B1 is actually better than V1 (rather than
  just inside its tight σ-cluster). Cost: ~4h on GPU 0.
* **V8 / V9** still deferred (need block-level refactor / open-research
  grade-grouping design — see ``plans/DECISIONS.md`` D-001).

#### Wall-time budget (actual)

* 13 × 100-epoch runs × ~1h54m each = 24h53m end-to-end serial on RTX 4090.
* No retries or restarts. Sweep ran clean.
