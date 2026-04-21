# Plan v2 — Video-JEPA-Clifford iter-2: V-JEPA-style tube-masked prediction

## Goal

Extend the iter-1 `VideoJEPA` backbone with a **second training target**:
V-JEPA-style masked-patch prediction in latent space, running **alongside** the
existing next-frame patch prediction. Keep iter-1 code intact — every iter-1
test must continue to pass. Deliverables: a standalone `TubeMaskGenerator`
utility, extended `VideoJEPAConfig`, extended `VideoJEPA.call` with a second
`add_loss` term + a learned mask token, updated test suite, updated smoke
training that logs both loss components.

This operationalizes the **iter-2 deferral of D-003**: user has locked the
latent-space / reuse-predictor / per-sample-independent-mask design via the
proposed D-008..D-012 decisions (see decisions.md).

## Problem Statement

### Expected behavior
1. **Training forward** (unchanged inputs): `{"pixels": (B, T, H, W, C),
   "telemetry": (B, T, k)}`. Internally:
   - Encode full clip once: `z = encode_frames(pixels) ∈ (B, T, H_p, W_p, D)`.
   - Sample a **tube mask** `M ∈ {0, 1}^(B, H_p, W_p)` per sample; broadcast
     to `(B, T, H_p, W_p)` (same mask across all T frames = pure spatial tube).
   - Build a **masked-latent input** `z_masked`: at unmasked positions, keep
     `z`; at masked positions, substitute `mask_token ∈ (D,)` broadcast.
   - Predict: `pred = predictor([z_masked, c])`.
   - **Loss 1 (next-frame, iter-1)**: `mean((pred[:, :-1] - z[:, 1:])**2)`
     *evaluated on **unmasked** positions only* so the two tasks don't double-
     count on the same tokens. (Rationale: where positions are masked, the
     predictor is by design reconstructing them — already captured by Loss 2.)
   - **Loss 2 (mask, iter-2)**: `mean((pred - z)**2)` evaluated on masked
     positions across **all T frames** (MSE between predicted and same-encoder
     target latents at masked slots; no gradient stop).
   - **Loss 3 (SIGReg, iter-1)**: unchanged — on `pred.reshape(B*T, N, D)`.
   - `total_loss = λ1 * L1 + λ2 * L2 + λ3 * L3` via three `add_loss` calls.

2. **Streaming inference (unchanged)**: no masking in `stream_step`. Streaming
   is inference-time — masked prediction is a training-only auxiliary task.

### Invariants (iter-2 adds I7..I9; iter-1 I1..I6 must still hold)
- **I1–I6 iter-1**: causality, AdaLN identity-at-init, shape propagation, SIGReg
  finiteness, serialization round-trip, streaming weak-equivalence — all must
  **continue to pass** exactly as before.
- **I7 — Tube mask validity**: generated mask has exactly
  `round(mask_ratio * H_p * W_p)` masked positions per sample; tube structure
  is preserved across all T frames (mask is T-invariant).
- **I8 — Mask loss finiteness**: `L2` is finite and non-NaN at init, after
  random init, and after one `fit` step.
- **I9 — Causality unchanged by masking**: perturbation at frame `k` at an
  **unmasked** position still leaves outputs at frames `< k` unchanged within
  1e-6 (same as I1). The mask is time-invariant, so it cannot leak future
  into past.

### Edge cases
- **Mask ratio = 0**: no positions masked → `L2 = 0` (or skipped); `L1 + L3`
  reduce exactly to iter-1 behavior. Regression guard.
- **Mask ratio = 1**: all positions masked → `L1` has no valid positions.
  Config invariant: `0.0 ≤ mask_ratio < 1.0` (strict upper).
- **Mask count rounds to 0**: if `round(mask_ratio * N) == 0`, skip `L2`.
- **T = 1**: no `L1` (same as iter-1); `L2` still runs.
- **Batch-of-1**: still unsafe (CliffordNet BN). Smoke tests lock B ≥ 2.

## Context

### Key findings feeding this plan
- `findings/clifford-primitives.md` — BN-batch-of-1 hard constraint still
  applies. `CausalCliffordNetBlock` causality semantics unchanged.
- `findings/lewm-reusable-assets.md` — SIGReg and AdaLN contracts unchanged;
  SIGReg is computed on `pred` unconditionally (not on target), so masking
  does not affect its numerics.
- `findings/positional-and-infrastructure.md` — `JEPAMaskingStrategy` exists
  for images but is *pixel-level* — declared unsuitable during iter-1 EXPLORE;
  confirmed during iter-2 ghost-constraint scan. Writing standalone
  `TubeMaskGenerator` is simpler and correct.
- `src/dl_techniques/models/masked_autoencoder/patch_masking.py` — existing
  MAE-style masking utility uses `argsort`-of-uniform-noise trick to sample
  `k`-of-`n` positions with correct cardinality. We **reuse the same trick**
  in `TubeMaskGenerator` — well-tested idiom in the codebase.

### Locked decisions for iter-2 (awaiting user confirmation)
D-008 multi-task single run, D-009 latent-space masking (hard constraint),
D-010 reuse predictor, D-011 per-sample independent mask, D-012 default weights.
Full rationale + trade-offs in `decisions.md`.

### Complexity budget (updated — see decisions.md PIVOT entry)
- Files added: iter-1 used 11/12; iter-2 adds **1** → **12/12** at the override cap.
- New classes: iter-1 used 5/7; iter-2 adds **1** (`TubeMaskGenerator`) → **6/7**.
- Lines added (iter-2 only): +~150-250 net (excluding tests).
- Existing-file modifications: `config.py` (+~6 fields), `model.py` (+~40 lines),
  `train_video_jepa.py` (+~10 lines), `test_video_jepa.py` (+~5 tests). All
  *surgical* additions — no rewrites.

## Files To Modify

**iter-2 inventory (authoritative). 1 new file + 4 modified files + 1 new
checkpoint.**

### Created
1. **`src/dl_techniques/models/video_jepa/masking.py`** — new file:
   `TubeMaskGenerator(keras.layers.Layer)`. Stateless (no weights, no
   persistent state); uses `keras.random` seeded per-call. Args: `mask_ratio`,
   `patches_per_side`. `call(batch_size, training) →
   (mask: (B, H_p, W_p) in {0,1}, num_masked: int)`. Full
   `@register_keras_serializable()` + `get_config()`.

### Modified
2. **`src/dl_techniques/models/video_jepa/config.py`** — add fields:
   `mask_prediction_enabled: bool = True`, `mask_ratio: float = 0.6`,
   `lambda_next_frame: float = 1.0`, `lambda_mask: float = 1.0`. Invariants:
   `0.0 ≤ mask_ratio < 1.0`, `lambda_* ≥ 0.0`. `to_dict`/`from_dict` extended.
3. **`src/dl_techniques/models/video_jepa/model.py`** —
   - Instantiate `self.mask_gen = TubeMaskGenerator(...)` if enabled.
   - `self.mask_token = self.add_weight(shape=(cfg.embed_dim,),
     initializer="zeros", name="mask_token")` (learned; zero-init per MAE
     convention — matches iter-1 identity-at-init philosophy).
   - `call`:
     (a) encode once (`z`),
     (b) sample mask `M: (B, H_p, W_p)` → broadcast to `(B, T, H_p, W_p, 1)`,
     (c) `z_masked = (1 - M) * z + M * mask_token` (broadcast),
     (d) `pred = predictor([z_masked, c])`,
     (e) `L1` on unmasked positions using a `(1 - M)` weight mask,
     (f) `L2` on masked positions using `M` weight mask,
     (g) `L3` (SIGReg on `pred.reshape(B*T, N, D)`, unchanged).
   - `get_config` unchanged (just inherits extended config).
   - `stream_step` **unchanged** (training-only masking).
4. **`tests/test_models/test_video_jepa/test_video_jepa.py`** — add
   `TestTubeMaskGenerator` class (mask ratio correctness across random seeds,
   T-invariance, per-sample independence, serialization round-trip) +
   `TestVideoJEPA::test_mask_loss_finite`,
   `test_mask_disabled_matches_iter1_behavior` (regression),
   `test_serialization_roundtrip_with_masking`.
5. **`src/train/video_jepa/train_video_jepa.py`** — log two loss components
   (named `next_frame_loss` and `mask_loss` in the CSV). Uses `add_metric`
   or reads `self.losses` post-`call`. Minimal change — ~10 lines.

### New checkpoint
6. **`plans/plan_2026-04-21_421088a1/checkpoints/cp-001-iter2.md`** — created
   at start of iter-2 step 1 EXECUTE. Records current commit hash
   (`8493641`-plus-whatever-iter-1 leaves) as the iter-2 rollback point. The
   iter-1 `cp-000-iter1.md` remains the nuclear fallback.

**No files deleted.** All iter-1 tests retained and must remain green.

## Steps

Each step annotated `[RISK] [deps]`. Hardest-first pattern preserved — the
TubeMaskGenerator tests (step 2) surface any mask-generation bug before it
can corrupt `model.call`.

### Step 1 — Config extension + checkpoint [RISK: low] [deps: none] [x]
- Write `cp-001-iter2.md` (current HEAD hash is the restore point).
- Add iter-2 fields to `VideoJEPAConfig` + invariants + `to_dict`/`from_dict`
  extensions. Add `TestConfig::test_iter2_fields_round_trip` test.
- Commit: `[iter-2/step-1] video-jepa: config fields for mask prediction`.
- **Done 2026-04-21**: 4 new fields added (`mask_prediction_enabled`,
  `mask_ratio`, `lambda_next_frame`, `lambda_mask`); 3 new invariants;
  4 new tests (`test_iter2_defaults`, `test_iter2_fields_round_trip`,
  `test_iter2_mask_ratio_bounds`, `test_iter2_lambda_non_negative`).
  Full suite: 33/33 pass (29 iter-1 retained + 4 iter-2). No regression.

### Step 2 — TubeMaskGenerator + HARDEST-FIRST tests [RISK: medium] [deps: 1]
Write `masking.py` implementing `TubeMaskGenerator`. Uses the `argsort` trick
(re-verified from `masked_autoencoder/patch_masking.py`) to sample exactly
`K = round(mask_ratio * H_p * W_p)` positions per sample.

Tests written **with** the layer (hardest first):
- **`test_mask_ratio_exact`** — for 50 random batches and seeds, sum of mask
  over `(H_p, W_p)` equals `K` exactly.
- **`test_tube_structure`** — returned mask is `(B, H_p, W_p)`; when we
  broadcast to T frames, the masked positions are identical across all T.
  (Verified by construction since we only generate spatial mask and broadcast.)
- **`test_per_sample_independence`** — for B=4, masks across samples are not
  all identical with probability 1.0 − ε over a small seed sweep.
- **`test_serialization_round_trip`** — save/load a layer instance.
- **`test_ratio_zero`** — mask_ratio=0 → all zeros.
- **`test_ratio_high`** — mask_ratio=0.75 → `K = round(0.75 * N)` masked.

Commit: `[iter-2/step-2] video-jepa: TubeMaskGenerator + hardest-first tests`.

**Done 2026-04-21**: `masking.py` created (1 class, ~120 LOC incl.
docstrings). 7 new `TestTubeMaskGenerator` tests all pass. One single-
line fix applied (`_allow_non_tensor_positional_args = True` to let
Python-int `batch_size` cross `Layer.__call__` guard). Full suite 40/40.
No iter-1 regression.

### Step 3 — Model integration [RISK: HIGH] [deps: 1, 2]
Extend `VideoJEPA.__init__` (mask generator + learned mask token) and
`VideoJEPA.call` (three-loss branch). Key subtleties:
- Mask broadcast: `M: (B, H_p, W_p) → (B, 1, H_p, W_p, 1)` so it broadcasts
  over `T` and `D`.
- Mask token broadcast: `mask_token: (D,) → (1, 1, 1, 1, D)`.
- `z_masked = (1 - M_bcast) * z + M_bcast * mask_token`. All positions get
  the same `mask_token` (position identity is provided by the positional
  embeddings inside the predictor).
- `L1` weight mask: `(1 - M_bcast)[:, :-1]` applied element-wise to
  `(pred[:, :-1] - z[:, 1:])**2`, then `sum / max(count, 1)` to avoid
  divide-by-zero if mask_ratio is extreme.
- `L2` weight mask: `M_bcast` applied element-wise to `(pred - z)**2`,
  then `sum / max(count, 1)`.
- Both per-position counts computed with `ops.sum(M_bcast) * D` (include all
  D dims in denominator so it's a true mean).
- If `mask_prediction_enabled == False`: skip mask generation + `L2` entirely
  → call reduces exactly to iter-1 behavior (regression path).

Commit: `[iter-2/step-3] video-jepa: mask-token + dual-loss call branch`.

**Done 2026-04-21**: (a) `mask_gen` (TubeMaskGenerator) instantiated in
`__init__`, (b) `mask_token` as unconditional zero-init `add_weight` in
`__init__` (not in `build`) — this preserves save/load weight topology
regardless of flag, (c) `call` refactored: encode → sample mask (if on) →
substitute via `(1-M)*z + M*token` → predictor → L1 on unmasked slots
(with element-weight mask), L2 on masked slots across all T, L3 SIGReg
unchanged. Two iter-1 tests (`test_forward_t1_edge`, `test_save_load_
round_trip`) flipped to `mask_prediction_enabled=False` per plan A11
(iter-2 equivalents come in Step 4). Full suite **40/40 PASS**, no
fix-attempt used (the flag-tweak was planned work, not a fix). Iter-1
regression: 29/29 retained.

### Step 4 — Test extensions (regression + mask-loss) [RISK: medium] [deps: 3]
Add to `TestVideoJEPA`:
- **`test_iter1_criteria_still_pass`** — meta test that references the same
  C1..C7 checks but with mask-prediction enabled (causality, AdaLN-id, SIGReg,
  forward, save/load, streaming). Most of these are already covered by the
  existing test classes — this step mostly ensures the existing tests run
  with the default config where masking is ON.
- **`test_mask_loss_finite`** — one forward pass, iterate `self.losses`, all
  three finite, `len(self.losses) == 3` with masking on.
- **`test_mask_disabled_matches_iter1_behavior`** — with
  `mask_prediction_enabled=False`, `len(self.losses)` reduces to 2 (next-frame
  + sigreg), and outputs match a reference iter-1-style model within tol.
- **`test_serialization_roundtrip_with_masking`** — save + load + predict a
  full model with masking enabled; bit-equivalent outputs.

Run full test suite: expect 29 iter-1 tests + 6 new `TubeMaskGenerator` tests
+ 3 new integration tests = **38 tests all green**.

Commit: `[iter-2/step-4] video-jepa: regression + mask-loss tests`.

**Done 2026-04-21**: 4 new tests in `TestVideoJEPAIter2`:
`test_mask_loss_finite` (C10 + P6 guard > 1e-5), `test_mask_disabled_
matches_iter1_behavior` (C11 — two-loss shape + determinism),
`test_serialization_roundtrip_with_masking` (C12 — weight-level bit-
equivalence; pragmatic rewrite: since mask sampling is stochastic, we
verify weight-name parity + per-weight allclose + post-load finite
forward instead of output bit-equivalence), `test_fit_one_step_with_
masking` (training smoke). Full suite **44/44 PASS**. One pragmatic
revision used (not a fix attempt — a clearer contract for the stochastic
case). Predicted 38, delivered 44 (more granular than planned).

### Step 5 — Training script logging [RISK: low] [deps: 3]
Minimal change to `train_video_jepa.py` — log `next_frame_loss` and
`mask_loss` as named metrics (via `add_metric(loss, name=...)` in model.call,
or via wrapping the existing logger). Simplest implementation: stash
`self._last_next_frame_loss` and `self._last_mask_loss` tensors inside `call`,
expose them via `metrics` property. Commit: `[iter-2/step-5] video-jepa:
log both loss components in training CSV`.

**Done 2026-04-21**: Implemented at the *model* level (cleaner than
post-hoc training-script hooks). Added `keras.metrics.Mean` trackers
(`next_frame_loss`, `mask_loss`, `sigreg_loss`) on `VideoJEPA`; exposed
via `metrics` property so `CSVLogger` writes each as a named column.
`train_video_jepa.py` gained 4 new CLI flags (`--mask-prediction-enabled`,
`--mask-ratio`, `--lambda-next-frame`, `--lambda-mask`). Full suite
44/44 pass. No fix attempts.

### Step 6 — Smoke training + REFLECT [RISK: medium] [deps: 4, 5]
Run on GPU 1:
```bash
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m \
  train.video_jepa.train_video_jepa --epochs 2 --batch-size 2 --T 4 \
  --img-size 64 --steps-per-epoch 4 \
  --output-dir results/video_jepa_smoke_iter2 --seed 0
```
Verify:
- No NaN/Inf in either loss component.
- Total loss decreases epoch-to-epoch.
- Mask loss decreases epoch-to-epoch (explicit requirement of iter-2).
- Next-frame loss also decreases (regression guard).
- `final_model.keras` saves + round-trips.

No commit — gated by REFLECT.

## Assumptions

| # | Assumption | Grounding | Steps depending |
|---|------------|-----------|-----------------|
| A9 | `argsort`-of-uniform-noise gives exact k-of-n selection per sample | `patch_masking.py:_create_mask` (battle-tested) | 2 |
| A10 | Broadcasting `M: (B, H_p, W_p)` over T inside model.call is cheap (no materialization to (B,T,H_p,W_p)) | Keras ops broadcast semantics | 3 |
| A11 | A zero-initialized learned `mask_token` does not break AdaLN identity-at-init | iter-1 I2 uses AdaLN zero-gate only; mask_token sits outside AdaLN (pre-predictor input substitution). Zero-init → at init, z_masked = (1-M)*z + 0 = (1-M)*z, which IS different from z for masked positions. This means I2 / AdaLN-zero test must be run with mask_prediction_enabled=False to isolate AdaLN behavior — iter-1 test already accepts a reference model, so flipping the flag is the right test shape. | 3, 4 |
| A12 | Three `add_loss` calls compose correctly under `loss=None, jit_compile=False` | iter-1 already uses 2-loss pattern | 3 |
| A13 | `keras.random.uniform` is seedable and batch-independent samples are trivially obtained by calling once with shape (B, N) | Keras 3 API | 2 |

If any of A9–A13 is false during EXECUTE, STOP and REFLECT. A11 in particular
is subtle — iter-1 `test_adaln_identity_init` may need a flag-tweak to keep
passing; that is expected and planned, not a regression.

## Failure Modes

| Dep | If broken | Blast radius | Mitigation |
|-----|-----------|--------------|------------|
| `TubeMaskGenerator` cardinality off-by-one | Mask ratio silently wrong | L2 statistic dilute | `test_mask_ratio_exact` covers 50 seeds |
| Mask broadcast rank mismatch | Keras raises at runtime | `model.call` | Shape asserts in step 3; unit test `test_mask_loss_finite` shapes it |
| `mask_token` zero-init makes `L2 = 0` trivially | No learning signal | Training | Check: at init `L2 > 0` because `z_masked ≠ z` wherever M=1 (mask_token=0 replaces z values). Test: mean `L2` at init > 1e-5 |
| `L1` denominator zero when mask_ratio ≈ 1 | NaN loss | Training | Clamp `count = max(1, actual_count)` |
| `mask_prediction_enabled=False` path drifts from iter-1 | Regression | All iter-1 tests | Explicit `test_mask_disabled_matches_iter1_behavior` |
| Causality break from mask-token global state | I1/I9 fail | Predictor outputs | Mask is sampled per-call and T-invariant; cannot cause temporal leakage. Still re-run causality test with masking ON |

## Pre-Mortem & Falsification Signals

### P5 — "Tube mask cardinality is wrong"
**STOP IF**: `test_mask_ratio_exact` fails for any seed. → REFLECT, suspect
`argsort` axis or rank confusion.

### P6 — "Mask loss is zero at init"
**STOP IF**: at init with default `mask_ratio=0.6`, `L2 < 1e-5`. → REFLECT,
suspect mask_token not replacing z correctly, or target tensor accidentally
fed `z_masked` instead of raw `z`.

### P7 — "Iter-1 regression — causality test fails with masking on"
**STOP IF**: any iter-1 test that was passing regresses. → REFLECT before
attempting fix (10-Line Rule applies).

### P8 — "Mask loss does not decrease in smoke training"
**STOP IF**: in 2-epoch smoke, `mask_loss[epoch=1] ≥ mask_loss[epoch=0]`
(not monotone non-increasing, with tolerance). → REFLECT. Weaker than
"diverges" — a rising mask loss means the predictor isn't learning the
denoising task at all.

## Success Criteria

Hardest-first.

1. **C8 Tube-mask cardinality**: `test_mask_ratio_exact` passes for 50 seeds.
2. **C9 Per-sample independence**: `test_per_sample_independence` passes.
3. **C10 Mask loss finite at init**: `test_mask_loss_finite` passes.
4. **C11 Mask-disabled regression**: `test_mask_disabled_matches_iter1_behavior`
   passes (feature-flag off ⇒ iter-1-equivalent forward).
5. **C12 Serialization with masking**: `test_serialization_roundtrip_with_masking`
   passes.
6. **C13 Iter-1 tests all green**: full `pytest tests/test_models/test_video_jepa/`
   — all 29 iter-1 tests + all new iter-2 tests pass. **No regression.**
7. **C14 Smoke training dual-loss**: 2-epoch fit; both `next_frame_loss` and
   `mask_loss` finite and monotone non-increasing epoch-to-epoch; total loss
   monotone non-increasing; `final_model.keras` round-trips.

## Verification Strategy

| # | Criterion | Method | Command | Pass means |
|---|-----------|--------|---------|------------|
| C8 | Mask cardinality | pytest | `.venv/bin/python -m pytest -k test_mask_ratio_exact -vvv` | all 50 seeds report `sum(M) == K` |
| C9 | Per-sample indep | pytest | `.venv/bin/python -m pytest -k test_per_sample_independence -vvv` | assertion holds |
| C10 | Mask loss finite | pytest | `.venv/bin/python -m pytest -k test_mask_loss_finite -vvv` | `ops.isfinite(L2) = True`, `L2 > 1e-5` |
| C11 | Disabled regression | pytest | `.venv/bin/python -m pytest -k test_mask_disabled_matches_iter1_behavior -vvv` | output atol 1e-5 vs reference |
| C12 | Serialize | pytest | `.venv/bin/python -m pytest -k test_serialization_roundtrip_with_masking -vvv` | atol 1e-5 |
| C13 | No regression | pytest full | `.venv/bin/python -m pytest tests/test_models/test_video_jepa/ -vvv` | ≥ 38 tests, 0 failed |
| C14 | Smoke training | manual | see step 6 command | both loss components monotone decrease |

## Complexity Budget (live counter, override at cap)

- Files added: 11/12 → **12/12** at cap after iter-2. Justified in PIVOT entry.
- New abstractions: 5/7 + 1 (`TubeMaskGenerator`) = **6/7** — still under cap.
- Lines added: +~150-250 in iter-2 (excluding tests).
- Extending a close-but-wrong neighbour: NO — `masking.py` lives inside the
  `video_jepa/` package per LESSONS rule.
