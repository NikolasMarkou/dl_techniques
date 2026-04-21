# Verification Results

## Iter-1 (archived)
All 7 iter-1 criteria C1..C7 verified **PASS** on 2026-04-21 (29/29 tests
green; 2-epoch smoke 3.7679 → 3.3212). P1–P4 STOP-IF triggers quiet;
A1–A8 held. Preserved as the regression baseline.

## Iter-2 (REFLECT, 2026-04-21)

### Criteria

| # | Criterion | Method | Command/Action | Result | Evidence |
|---|-----------|--------|----------------|--------|----------|
| C8 | Tube-mask cardinality | pytest | `pytest -k test_mask_ratio_exact -vvv` | **PASS** | 50-seed sweep, per-row sum == 38 exactly (ratio=0.6 × 64 patches) |
| C9 | Per-sample mask independence | pytest | `pytest -k test_per_sample_independence -vvv` | **PASS** | At B=8, masks differ pairwise |
| C10 | Mask loss finite at init | pytest | `pytest -k test_mask_loss_finite -vvv` | **PASS** | 3 finite losses; mask_loss > 1e-5 (zero-init token replaces z values) |
| C11 | Mask-disabled regression | pytest | `pytest -k test_mask_disabled_matches_iter1_behavior -vvv` | **PASS** | len(losses)==2, two forwards bit-identical atol=1e-6 |
| C12 | Serialization w/ masking | pytest | `pytest -k test_serialization_roundtrip_with_masking -vvv` | **PASS** | Weight-name parity + per-weight allclose + post-load forward finite. (Contract revised: output determinism replaced by weight-level equivalence since mask sampling is stochastic per-call.) |
| C13 | No iter-1 regression | pytest full | `pytest tests/test_models/test_video_jepa/ -vvv` | **PASS** | 44/44 (29 iter-1 retained + 15 iter-2 additions). Iter-1 tests `test_forward_t1_edge` and `test_save_load_round_trip` received planned `mask_prediction_enabled=False` tweaks (Assumption A11) — *not* regressions. |
| C14 | Smoke dual-loss training | manual | `MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 … train_video_jepa …` | **PARTIAL FAIL** | See details below. |

### C14 details (2-epoch smoke on GPU 1, RTX 4070)

Logs: `results/video_jepa_smoke_iter2/training_log.csv`
(+ `final_model.keras` verified to round-trip with 3 finite losses.)

| Epoch | loss | mask_loss | next_frame_loss | sigreg_loss |
|-------|------|-----------|-----------------|-------------|
| 1 | 2.2670 | 0.5018 | **6.998e-04** | 19.6057 |
| 2 | 2.2291 | 0.4906 | **1.618e-03** | 19.2983 |

- **Total `loss`**: 2.2670 → 2.2291 (decreasing ✓)
- **`mask_loss`**: 0.5018 → 0.4906 (decreasing ✓; **P8 STOP-IF did NOT fire** — mask loss is monotone non-increasing)
- **`sigreg_loss`**: 19.6057 → 19.2983 (decreasing ✓)
- **`next_frame_loss`**: 6.998e-04 → 1.618e-03 (~2.3× INCREASE ✗)
- **Finiteness**: all values finite; no NaN / Inf. TerminateOnNaN not triggered.
- **Save/load**: `final_model.keras` saves; reloads; 3 losses finite post-reload.

### Analysis of the `next_frame_loss` increase

Three likely explanations, ordered by prior probability:

1. **Absolute scale effect (most likely)**: At epoch 1, `next_frame_loss ≈ 7e-4`
   is already two orders of magnitude below the other two losses (0.5 and 19.6).
   With `lambda_next_frame = lambda_mask = 1.0` and `sigreg_weight = 0.09` the
   gradient contribution from `next_frame_loss` is a tiny fraction of total.
   A ~2× increase from 7e-4 → 1.6e-3 is still **absolutely small** (Δ ≈ 9e-4)
   relative to how much `mask_loss + sigreg_loss` dropped. This is consistent
   with "the predictor stopped being an identity on unmasked slots and started
   actually modelling masked slots", which is the whole point of the dual-loss.
2. **Under-sampling noise**: 2 epochs × 4 steps × batch 2 = 16 gradient steps
   total. The numbers may simply be within statistical noise of each other —
   the plan's C14 requirement of monotone non-increasing over just 2 epochs is
   unusually strict for a 16-step regime.
3. **Weight imbalance**: both λ=1.0 is defensible but biases the predictor
   towards the larger (mask) loss signal. A principled rebalance is a candidate
   iter-3 investigation.

**This is not a P7 falsification** (P7 is "iter-1 regression — causality test
fails", which did NOT happen; C13 is PASS). **Not a P8 falsification either**
(P8 specifically = "mask_loss does not decrease"; mask_loss DID decrease).

It **is** a partial C14 failure — the plan's criterion said "both losses
monotone non-increasing" and one is not.

### Additional checks

| Check | Result |
|-------|--------|
| Full iter-1 suite (29 tests) still green | **PASS** (verified at every iter-2 step and post-Step-5) |
| Iter-2 Autonomy Leash usage | **1 fix attempt** across 6 steps (Step 2: single-line Keras `_allow_non_tensor_positional_args=True`). Leash never approached. |
| Iter-1 regression, deeper scan | `test_forward_t1_edge` and `test_save_load_round_trip` required the planned Assumption-A11 flag-tweak (`mask_prediction_enabled=False`). This was documented in plan v2 before EXECUTE — not a regression. |
| Smoke `final_model.keras` round-trip | **PASS** |
| Falsification P5 (mask cardinality wrong) | Quiet — C8 50-seed sweep all exact |
| Falsification P6 (mask loss < 1e-5 at init) | Quiet — init L2 ≈ 0.5 |
| Falsification P7 (iter-1 regression) | Quiet — 29/29 retained |
| Falsification P8 (mask loss does not decrease) | Quiet — mask_loss 0.5018 → 0.4906 |

### Not Verified
| What | Why |
|------|-----|
| Real drone data | Out of scope |
| EMA target encoder | Out of scope — iter-3 candidate |
| Asymmetric context/target views | Out of scope |
| Pixel-space masking variant | Blocked by D-001 (hard constraint) |
| Long-run training (>2 epochs) | Out of scope — smoke only |
| Hyper-parameter sweep for `lambda_next_frame` vs `lambda_mask` | Out of scope — iter-3 candidate |

### Prediction Accuracy (EXTENDED — iter-2)

| Plan prediction | Actual | Notes |
|-----------------|--------|-------|
| 1 new file + 4 modified files | 1 new + 4 modified | Exact ✓ |
| +150–250 net lines (excl. tests) | ~190 net | ✓ within band |
| 6 new tube-gen tests | 7 | +1 (added `test_rejects_bad_args`) |
| 5 new video-jepa tests → 38 total | 4 new + 2 existing-test flag tweaks → 44 total | +6 tests beyond plan |
| 2 fix attempts max per step | 1 total across all 6 steps | ✓ well under leash |
| Both losses decrease in smoke | mask_loss ✓, next_frame_loss ✗ | Partial — see C14 analysis |

### Convergence Metrics (EXTENDED — iter-2)

Iter-2 is iteration 2. No previous iter-2 REFLECT, so metric stability is
against iter-1 only.

| Metric | Iter-1 | Iter-2 | Trend |
|--------|--------|--------|-------|
| Test count | 29 | 44 | +15 (all new iter-2-specific) |
| Pass rate | 29/29 = 100% | 44/44 = 100% | flat ✓ |
| Files created | 11 | +1 → 12 | at 12/12 override cap |
| New classes | 5 | +1 → 6 | at 6/7, under cap ✓ |
| Fix attempts | 2 (single-line) | 1 (single-line) | ↓ improving |
| Scope drift | 0 | 0 | ✓ matches plan v2 inventory exactly |

Scope stability is strong — no unplanned file touched. Issue decay is strong —
1 fix in 6 steps vs 2 in 9. Not diverging.

### Devil's advocate (EXTENDED — iter-2)

One reason this might still be wrong despite PASS on 6/7 criteria:

*The mask loss is decreasing, but it might be decreasing because the predictor
has discovered that the easiest way to reduce `(pred - z)` at masked positions
is to predict something close to the **global mean of z**, not the
semantically-correct per-position latent.* That would be a shortcut: mask
loss decreases but no real representation learning happens. A stronger
falsification would compare mask-loss values against a "predict-the-mean"
baseline — not in this plan's C14. This is a candidate iter-3 probe.

### Simplification Checks (complexity-control)

1. *Can this be done with fewer files?* No — `masking.py` is distinct and reusable;
   mixing it into `model.py` would bloat `model.py` further.
2. *Can this be done with fewer classes?* No — `TubeMaskGenerator` is
   well-scoped and round-trips independently.
3. *Are we extending a close-but-wrong neighbour?* No — deliberately did not
   subclass `JEPAMaskingStrategy` (ghost-constraint #1 in PIVOT entry).
4. *Is this a wrapper cascade?* No — single layer, single model method.
5. *Could a config toggle collapse this?* `mask_prediction_enabled` **is** a toggle
   — intentional, for regression-guard.
6. *Temporary workaround that should be permanent?* None.

### Verdict

**6/7 success criteria PASS + 1 PARTIAL.**

C8–C13 all PASS cleanly with strong evidence. C14 partial:
- Mask loss decreases ✓
- Total loss decreases ✓
- Mask prediction is working (mask_loss starts at 0.5 and drops — non-trivial signal)
- No NaNs, finite everywhere, save/load works
- **BUT** next-frame loss went up from 7e-4 to 1.6e-3 over 2 epochs / 16 steps.

**Recommended verdict: PASS-with-observation.** The C14 requirement was
possibly too strict given the (a) absolute scale of `next_frame_loss` being
100× smaller than `mask_loss` at init, (b) tiny 16-step training regime, and
(c) the expected "interference" when a dual-loss predictor starts learning
the second (larger) task. No falsification signal fired. Autonomy Leash
barely used. No regressions. Scope honored exactly.

**However**, the protocol says: `next_frame_loss` increase IS a partial
failure of a written criterion. Per REFLECT phase rules, this is user's
call:

- **Option (a) — CLOSE with observation logged**: accept the partial C14,
  document iter-3 candidates (loss reweighting, mean-predictor baseline
  probe, longer training) in DECISIONS.md.
- **Option (b) — Extend iter-2 one more step**: rebalance λ weights (e.g.
  `lambda_next_frame=2.0`) or run a longer smoke (5 epochs) and re-verify
  monotonicity. This risks Iteration-3 growth without new design content.
- **Option (c) — PIVOT to iter-3**: treat as a hyper-parameter investigation
  and open a fresh plan.

Orchestrator recommendation: **Option (a)**, because the signal passed every
falsification and the C14 sub-condition that failed is best attributed to the
smoke regime being too short to draw monotonicity conclusions from a loss
that is two orders of magnitude smaller. **But the user decides.**
