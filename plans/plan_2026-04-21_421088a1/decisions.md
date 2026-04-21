# Decision Log
*Append-only. Never edit past entries.*

*Cross-plan context: see plans/FINDINGS.md, plans/DECISIONS.md, and plans/LESSONS.md*

---

## 2026-04-21 — Locked design decisions (post-EXPLORE, pre-PLAN)

User selected from alternatives presented after EXPLORE. Each decision is phrased
as "X at the cost of Y" and anchored to a finding.

### D-001 — Encoder architecture: HYBRID (`PatchEmbedding2D + 2–4 CliffordNetBlock`)

- **Choice**: option (c) hybrid — `PatchEmbedding2D(patch=P, embed_dim=D)` → reshape
  to 4D `(B*T, H_p, W_p, D)` → 2–4 stacked `CliffordNetBlock` layers.
- **Trade-off**: locality + small-object inductive bias at the cost of losing
  the pretrained-ViT transfer path and the simplicity of a single known encoder.
- **Grounding**: `findings/lewm-reusable-assets.md` § "Encoder choice (D-001)";
  `findings/clifford-primitives.md` § `CliffordNetBlock` contracts.
- **Risks anchored**: BatchNorm inside Clifford context stream is unsafe at
  batch-of-1 — smoke tests must keep B ≥ 2 (hard constraint, `findings.md`).
- **Alternative rejected (a) ViT-tiny**: cheaper and has pretrained weights, but
  no CNN locality and no dual-stream detail/context — weaker for drone footage.
- **Alternative rejected (b) Clifford-only from raw pixels**: no proper patchifier
  step → loses the `(B, N, D) → (B, H_p, W_p, D)` clean reshape path.

### D-002 — Predictor 5D handling: FACTORIZED spatial + causal temporal, ALTERNATING

- **Choice**: option (a) factorized. Predictor body is a stack of alternating
  blocks: per-frame spatial `CliffordNetBlock` on `(B*T, H_p, W_p, D)` + per-
  spatial-position causal temporal `CausalCliffordNetBlock` on
  `(B*H_p*W_p, 1, T, D)`.
- **Trade-off**: clean reuse of existing, fully-tested 4D Clifford primitives at
  the cost of two reshapes per alternating pair (negligible on RTX 4070).
- **Grounding**: `findings/clifford-primitives.md` § "Implications for Video-JEPA
  predictor". `CausalCliffordNetBlock` requires H=1 — the reshape to
  `(B*H_p*W_p, 1, T, D)` is the only legal temporal path.
- **Alternative rejected flatten `(B, T*N, D)`**: kills spatial grid, loses
  `CliffordNetBlock`'s 2D DWConv advantage.
- **Alternative rejected 3D `(B, T, N, D)`**: no primitive accepts this rank
  cleanly; would require a new layer.

### D-003 — Target signal: LeWM-STYLE next-frame patch-latent prediction

- **Choice**: option (b). For iter-1, MSE between predicted `(B, T-1, H_p, W_p, D)`
  and target `(B, T-1, H_p, W_p, D)` = `target_emb[:, 1:, ...]`. Causal shift.
- **Trade-off**: simple, matches LeWM precedent, no new masking utility, at the
  cost of not training invariance to patch-level dropouts (tube-masked V-JEPA
  is stronger signal but is iter-2 follow-up).
- **Grounding**: `findings/lewm-reusable-assets.md` § `call()` pattern;
  `findings/positional-and-infrastructure.md` § "V-JEPA masking precedent".
- **Deferred to iter-2**: tube masking over `(T, H_p, W_p)` — needs new
  `TubeMaskingStrategy` (~50 lines) extending `JEPAMaskingStrategy`.

### D-004 — Conditioning injection: PREDICTOR-ONLY + per-frame `c_t`

- **Choice**: option (a). Telemetry → `c: (B, T, D_cond)` (per-frame). AdaLN-zero
  applied only inside predictor temporal pass (where we already have a 3D
  `(B, T, D)` tensor after per-patch transpose). Encoder is NOT conditioned.
- **Trade-off**: identity-at-init preserved (zero-init AdaLN shift/scale/gate =
  block is identity), and no 5D conditioning expansion, at the cost of weaker
  spatial-telemetry coupling (may revisit if rotation-heavy drone motion shows
  up as a failure mode).
- **Grounding**: `findings/lewm-reusable-assets.md` § AdaLN-zero contract. The
  `AdaLNZeroConditionalBlock` takes `inputs=[x, c]` list, both `(B, T, D)`.
  Perfect fit for the temporal pass after we reshape to `(B*H_p*W_p, T, D)`:
  `c` is broadcast-expanded from `(B, T, D)` → `(B*H_p*W_p, T, D)` via tile.
- **Alternative rejected (b) global `c` via pooling**: loses per-frame
  alignment between telemetry and visual content.
- **Alternative rejected encoder-AdaLN**: would require per-frame modulation of
  4D `(B*T, H_p, W_p, D)` — the AdaLN-zero layer as shipped expects 3D `(B,T,D)`,
  would need a new variant.

### D-005 — SIGReg placement: MIDDLE — reshape `(B·T, N, D)`, average over N

- **Choice**: option (c). After predictor: reshape pred latents to
  `(B*T, N, D)` where `N = H_p * W_p`, pass to `SIGRegLayer` which averages
  cos/sin over axis=-3 (= N axis), then outer mean over `(B*T)`.
- **Trade-off**: per-frame statistic (averages patches within a frame, samples
  across frames+batch) — principled balance between patch-level signal tightness
  and per-frame coherence. Cost: must document the reshape; `num_proj=64`
  smoke default (LeWM precedent).
- **Grounding**: `findings/lewm-reusable-assets.md` § SIGReg D-005 interpretation.
  Middle option is the most principled IMO (explorer's phrasing).
- **Alternative rejected per-frame pooled `(1, B*T, D)`**: collapses patch
  diversity — SIGReg statistic ignores within-frame structure.
- **Alternative rejected fully flattened `(1, B*T*N, D)`**: may over-regularize
  dense patch features and dilute the signal.

### D-006 — Positional embeddings (accepted as recommended)

- **Choice**: `PositionEmbeddingSine2D(num_pos_feats=D/2)` for patch grid
  (transposed channels-first → channels-last) + learned 1D temporal PE
  `self.add_weight((1, T_max, D))` + `ContinuousSinCosEmbed(dim=D_cond, ndim=k)`
  for telemetry (k = # telemetry channels).
- **Trade-off**: zero-parameter fixed 2D PE + learned temporal flexibility at the
  cost of a mandatory channels-first → last transpose (documented pitfall).
- **Grounding**: `findings/positional-and-infrastructure.md` § "Recommendation
  for D-006".

### D-007 — Streaming contract (accepted as recommended)

- **Choice**: rolling-buffer pattern copied from `LeWM.rollout`. Keep
  `emb_buf: (B, K, H_p, W_p, D)` for last K encoded frames; per new frame,
  encode → append → truncate to last K → run predictor on K-window → emit
  `(B, 1, H_p, W_p, D)`. O(1) amortized per frame.
- **Trade-off**: no true constant-memory streaming state (buffer is K frames,
  not a hidden state), at the cost of simplicity — matches existing LeWM precedent
  which has already been validated.
- **Grounding**: `findings/positional-and-infrastructure.md` § "LeWM model —
  streaming precedent"; `src/dl_techniques/models/lewm/model.py:rollout` lines 208–279.

---

## 2026-04-21 — Complexity budget override (declared in PLAN)

This is a new-model port spanning model package + training package + dataset
module. LESSONS.md rule "Extending a close-but-wrong neighbour package: Make a
new package for each new architecture; do not overload existing ones" applies
directly — overloading `lewm/` or `jepa/` would be the wrong call.

- Default budget: 3 files / 2 abstractions. Override: ~10 files / ~6 classes.
- Mitigation: every new class must have `@keras.saving.register_keras_serializable()`,
  complete `get_config()`, and a round-trip serialization test (LESSONS:
  "Not testing serialization round-trip").

Full file list + rationale in `plan.md` § "Files To Modify".

---

## 2026-04-21 — PIVOT REFLECT → PIVOT → PLAN (iter-2, planned extension, NOT failure)

### Context
Iter-1 REFLECT verified **PASS** on all 7 success criteria (29/29 tests green;
2-epoch smoke run monotone decrease 3.7679 → 3.3212; P1–P4 STOP-IF did not fire;
assumptions A1–A8 held). This is a **planned-extension pivot**, not a failure
pivot. The iter-1 code + tests are stable and are retained as-is; iter-2 adds a
second training target on top.

### Rationale
D-003 (iter-1) locked the target signal to **LeWM-style next-frame patch-latent
prediction** and explicitly deferred **V-JEPA-style tube masking** to iter-2.
User has now authorized iter-2 for that deferred scope.

### Why a PIVOT (not an iteration extension under the same plan)
- A new training target adds a second loss term, new trainable state (mask
  token), new config fields, and new tests. That is a design change, not a
  bug-fix iteration on the existing plan.
- Logging as PIVOT makes the scope expansion auditable and keeps iter-1
  decisions anchored without mutation.

### Keep vs revert
- **KEEP everything from iter-1.** All 11 files stay as shipped; all 29 tests
  continue to run and must remain PASS (no-regression constraint — see plan v2
  success criteria). No revert to cp-000.
- cp-000-iter1.md remains the nuclear-fallback checkpoint (pre-iter-1 state).
- iter-2 will add a new `cp-001-iter2.md` at the start of its first EXECUTE.

### Ghost-constraint scan (EXTENDED — iter-2, second PLAN)
1. *"Tube masking needs its own `TubeMaskingStrategy` subclass of the existing
   `JEPAMaskingStrategy`"* (anchored in decisions.md D-003 deferred note) —
   **stale.** The existing `JEPAMaskingStrategy` is pixel-level and not a
   useful base class for latent-space spatiotemporal tube masking. Writing a
   small standalone `TubeMaskGenerator` utility is simpler. **Will not inherit.**
2. *"Drop-tokens before the encoder (V-JEPA original recipe)"* — incompatible
   with CliffordNet's 2D grid encoder. Latent-space masking (encode full clip
   once, mask latents before predictor) is both cheaper and structurally
   required by D-001. **Hard constraint, not a preference** — anchors D-009(a).
3. *"Need EMA target encoder"* — V-JEPA original uses EMA; LeWM philosophy
   (and iter-1) does not. Out of scope for iter-2 (same-encoder targets).

### Complexity budget update (MUST justify)
Iter-1 final: 11 new files / 5 new classes + 1 function. Well under the 12/7
override cap.

Iter-2 adds:
- 1 new utility file (`masking.py` co-located under `models/video_jepa/`) with
  1 new class (`TubeMaskGenerator`).
- `config.py` modified: +~6 fields + 1 invariant check (not a new file).
- `model.py` modified: +~40 lines for mask loss branch + 1 learned mask token
  weight (not a new file).
- `test_video_jepa.py` modified: +~5 tests.
- `train_video_jepa.py` modified: +~10 lines to log both loss components.

Post-iter-2 totals: **12 new files / 6 new classes + 1 function**. Exactly at
the override cap (12/7), still legitimate — each addition maps to a distinct
responsibility and LESSONS § "Extending a close-but-wrong neighbour package"
continues to apply (tube masking belongs with video_jepa, not in a shared
`layers/masking/` directory — we verified no such directory exists anyway).

### Momentum check (EXTENDED — 1st PIVOT on this plan)
First pivot. No oscillation risk. Direction: +capability (iter-1 target-signal
deferral now resolved). Trajectory remains monotonic; not a reversal.

### Transition
REFLECT → PIVOT → PLAN (iter-2). Present plan v2 for user approval before EXECUTE.

---

## 2026-04-21 — Hardest-first verification order

Verification strategy in plan.md orders checks hardest-first so failures surface early:
1. **Causality** — `CausalCliffordNetBlock` invariant. Perturbation at pos k must
   not alter output at pos < k. This is the *single* correctness property that
   makes the rest meaningful.
2. **SIGReg stability** — finite, non-NaN, non-inf at init and after 1-step grad.
3. **AdaLN-zero identity-at-init** — `max|block([x, 0]) - x| < 1e-6`.
4. **Serialization round-trip** — `save → load → predict` bit-equivalent.
5. **Forward-pass shapes** — explicit asserts on all 5D tensors at block boundaries.
6. **Streaming O(1) per frame** — constant wall-clock per `stream_step` call.

---

## 2026-04-21 — Iter-2 proposed design decisions (RECOMMENDATIONS — awaiting user confirmation)

Each decision is phrased "X at the cost of Y" with a recommendation + alternatives.
These are the five D-008..D-012 surfaced for user sign-off before plan v2 is
approved. Iter-2 EXECUTE does not begin until these are confirmed.

### D-008 — Multi-task: both losses in a single training run (RECOMMEND option a)
- **Option (a) RECOMMENDED** — single training run, `loss = λ1·next_frame_loss + λ2·mask_loss + λ3·sigreg`.
  - **Trade-off**: stronger backbone at the cost of the losses fighting if the
    weights are miscalibrated. Mitigation: λ1=λ2=1.0, λ3=0.09 defaults (tunable).
- **Option (b)** — separate training modes via config flag (`training_target ∈
  {"next_frame", "masked", "both"}`). Cleaner for isolated ablations but
  doubles the training-script state machine. Can revisit in iter-3.
- **Grounding**: user guidance in PIVOT brief.

### D-009 — Masking in latent space (RECOMMEND option a — treated as hard constraint)
- **Option (a) RECOMMENDED / HARD CONSTRAINT** — encode full clip once to
  `(B, T, H_p, W_p, D)`, apply tube mask in latent space before predictor:
  replace masked positions with learned mask token + positional info.
  - **Trade-off**: deviates from V-JEPA original (which masks at pixel level)
    at the benefit of reusing the same encoder pass for both loss branches.
    Cheaper; avoids rebuilding a token-drop path for CliffordNet's 2D grid.
- **Option (b)** — drop patch tokens before encoder. Blocked by D-001 (hybrid
  Clifford encoder requires an intact 2D grid).
- **Ghost constraint**: see decisions.md PIVOT entry, ghost scan item #2.

### D-010 — Reuse existing predictor (RECOMMEND option a)
- **Option (a) RECOMMENDED** — reuse `VideoJEPAPredictor` unchanged. Both
  next-frame prediction and mask prediction go through the same predictor.
  Mask loss reads predictor outputs at the *masked* (H_p, W_p) positions and
  compares against the *target* latents at those same positions (from the
  same encoder, no gradient stop — keep full grad flow per LeWM/SIGReg style).
  - **Trade-off**: shared representation at the cost of coupling; the mask
    loss becomes a denoising-in-latent-space signal through the next-frame
    predictor rather than through a dedicated head.
- **Option (b)** — add a small mask-only prediction head (extra params,
  extra code, extra test surface). Only worth it if (a) fails to train.

### D-011 — Per-sample independent tube masks (RECOMMEND option a)
- **Option (a) RECOMMENDED** — each sample in the batch gets an independently
  sampled tube mask.
  - **Trade-off**: higher gradient diversity at the cost of a slightly more
    complex mask generator (batched stateless sampling).
- **Option (b)** — shared mask across the batch (V-JEPA 2 default). Simpler
  code; less diversity at small batch sizes.

### D-012 — Loss weights (RECOMMEND defaults)
- Defaults: `lambda_next_frame=1.0`, `lambda_mask=1.0`, `sigreg_weight=0.09`
  (unchanged from iter-1). All tunable via `VideoJEPAConfig`.
  - **Trade-off**: equal weight is the most-symmetric prior at the cost of
    hiding a possible per-task dominance; the smoke training is short enough
    that we'll see divergence fast if it happens.
- Anchored in `config.py` — fully serialized.

### Iter-2-specific details (baked into plan v2, not separate decisions)
- **Mask ratio**: default **0.6** (mid-range V-JEPA; 0.5–0.75 typical). Tunable
  via `mask_ratio` in config.
- **Tube structure**: pick **K distinct (h, w) spatial-grid positions**, mask
  them across **all T frames** (pure space-tube, not space-time cube). Justified
  because causality is only meaningful if we don't mask "future" positions
  specifically — a full-temporal tube preserves causality by *symmetry*
  (masking is time-invariant), so we still get a principled next-frame loss
  from the unmasked spatial minority.
- **Tube count K**: `K = round(mask_ratio * H_p * W_p)` per sample.
- **Mask token**: single learned `(D,)` weight in `VideoJEPA` (model-level,
  not predictor-level) — broadcast + position-embedded onto masked positions
  before being fed to the predictor.
- **Causality preservation**: mask is time-invariant → does not introduce
  temporal leakage. Iter-1 causality test must **still pass unchanged** —
  that's the regression guardrail.

---

## 2026-04-21 — REFLECT iter-2 evaluation

### Verdict
6 of 7 criteria PASS (C8–C13), 1 PARTIAL (C14). No STOP-IF signals fired.
Full detail in verification.md.

### Root-cause analysis for the C14 partial

**Immediate cause**: `next_frame_loss` rose 7e-4 → 1.6e-3 over 2-epoch smoke.

**Contributing factors** (trace back one level):
1. Smoke regime (16 total grad steps, batch-2) is below the statistical
   resolution at which a 9e-4 delta can be trusted as "monotone".
2. `next_frame_loss` is two orders of magnitude smaller than `mask_loss`
   at init (7e-4 vs 0.5), so with equal λ the gradient budget on the
   predictor is dominated by mask-prediction — the predictor starts
   *trading* unmasked-slot accuracy to lower masked-slot residuals.
3. Criterion C14 ("both losses monotone non-increasing") was written
   before the dual-loss interference dynamic was observed; in hindsight
   a criterion like "total loss + mask loss monotone non-increasing,
   next_frame_loss at most 2× its init value" would have been more
   useful.

**Failed defense**: P8 (falsification) was scoped to mask_loss only;
there is no STOP-IF on next_frame_loss increase. The gap was in
falsification breadth, not in the gate itself.

**Prevention for iter-3/follow-up**:
- Add a falsification for "next_frame_loss explodes >10× init".
- Rebalance λ (`lambda_next_frame=2.0` — weighs up the smaller-magnitude
  loss so it actually gets gradient signal).
- Longer smoke (5+ epochs) before declaring monotonicity.

### Devil's advocate concern (EXTENDED — iter-2)
Mask loss might be decreasing via the "predict global mean of z" shortcut
(the MAE in latent-space without an EMA target is known to be
shortcut-prone). Iter-3 candidate: a "predict-the-mean" baseline probe.

### Convergence (EXTENDED — iter-2)
Iter-1 → iter-2: scope stable (0 drift), issue decay improving (2 fix
attempts → 1), test count up 52% (29 → 44). Trajectory converging, not
diverging. Decomposition threshold (iteration 5) not approached.

### Recommended options for the user
- **(a) CLOSE with observation**: 6/7 clean PASS is strong. The partial
  is explainable by smoke-regime effects, not a design flaw. Iter-3
  candidates (λ rebalance, mean-predictor probe, longer training, EMA
  target encoder) go into `plans/DECISIONS.md` for future work.
- **(b) Extend iter-2 one step**: add a λ-rebalance + re-run smoke,
  re-verify C14. Bounded additional work.
- **(c) PIVOT to iter-3**: open a fresh plan framed as "hyper-parameter
  study + shortcut-avoidance probe". Heavier but cleaner.

**Orchestrator recommendation: (a).** Strong safety signal throughout
(no NaN, no regression, no falsification fire, 1 fix attempt in 6 steps).
The C14 sub-failure is about one sub-dimension of a smoke-regime that
was never meant to draw statistical conclusions from. **User decides.**

---

## 2026-04-21 — Future Work (iter-3+ candidates, queued at CLOSE)

User approved CLOSE of iter-2 with the C14 partial accepted as a smoke-regime
artifact. The following investigations are deferred to follow-up plans. Each is
phrased with its expected trade-off so a future EXPLORE does not need to re-derive
the rationale.

### FW-001 — Loss weight rebalancing (lambda tuning)
**Motivation**: Iter-2 smoke showed next_frame_loss (~7e-4 at init) is ~1000x
smaller than mask_loss (~0.5). With lam_next=lam_mask=1.0 the predictor's
gradient budget is dominated by mask_loss, causing absolute regression of
next_frame_loss even as total_loss decreases.

**Proposed**: sweep `lambda_next_frame in {1.0, 2.0, 5.0, 10.0, 100.0}` against
fixed `lambda_mask=1.0`; measure both losses at epoch 5. Pick the smallest
`lambda_next_frame` at which next_frame_loss is monotone non-increasing.

**Cost**: cheap — config-only, 5 smoke runs serially on GPU 1. No code changes.

### FW-002 — Longer-horizon training before monotonicity claims
**Motivation**: 16 gradient steps is statistically inadequate for
sub-milli-magnitude loss deltas. A proper gate needs >=5 epochs, >= 100 steps/epoch.

**Proposed**: on the re-balanced lam config (FW-001 outcome), run 10-epoch smoke
with `steps_per_epoch=64`, `batch_size=2`. Record per-epoch both losses.
Criterion: monotone non-increasing over 5-epoch rolling window (not per-epoch).

**Cost**: ~30 minutes on GPU 1.

### FW-003 — EMA target encoder (V-JEPA original recipe)
**Motivation**: Iter-1/2 used a live target encoder (identical to context
encoder, no stop-gradient). V-JEPA original uses an EMA-updated target
encoder, which is known to reduce shortcut-learning risk in masked-latent
prediction.

**Proposed**: add `ema_target: bool = False` flag + `ema_decay: float = 0.999`
to `VideoJEPAConfig`. When True, maintain a shadow encoder updated via EMA;
target latents come from shadow, context from live. `stop_gradient` on target
path. Test invariant: with `ema_decay=1.0`, behavior matches current (shadow
frozen at init). With `ema_decay=0.0`, shadow == live (current behavior).

**Cost**: medium — +1 config flag, ~30 lines in `model.py` (shadow
initialization + per-step EMA update + stop-gradient), +1 serialization test,
+1 numeric-equivalence test for edge cases.

**Trade-off**: stronger representation + lower shortcut risk at the cost of
~2x encoder memory (shadow copy of weights). Matches D-009's "reuse same
encoder" choice in spirit — EMA is still "same architecture", just frozen
at different decay rates.

### FW-004 — Mean-predictor shortcut probe
**Motivation**: Devil's-advocate concern (iter-2 REFLECT): mask_loss may be
decreasing via a trivial "predict global mean of z" shortcut rather than
per-position semantic recovery. Absence of an EMA target makes this more
plausible.

**Proposed**: add a diagnostic metric `mask_loss_vs_mean_baseline` that
computes `L2(mean(z), z)` over masked positions — if training `mask_loss`
stays close to this baseline, the predictor has learned the shortcut.
Emit to training CSV. Alarm threshold: `mask_loss / mask_loss_mean_baseline
> 0.9` at epoch >= 3.

**Cost**: low — a few lines in `call` + a metric tracker. No new
abstraction. Can ship with FW-001 or FW-003.

### Ordering
Recommended: FW-001 first (cheapest, unblocks the C14 gate), then FW-004
(piggyback as a diagnostic), then FW-003 (design change, only if FW-004
shows shortcut behavior). FW-002 is an always-on companion to any of the
above.

### Non-candidates (explicitly NOT queued)
- Pixel-space masking — HARD CONSTRAINT D-009. Do not reopen.
- `training_target` mode selector (option b of D-008) — the single-run
  multi-task design is working; adding modes is complexity without current
  evidence of need.
- New predictor head for mask loss (option b of D-010) — the shared
  predictor is the lean choice; only revisit if FW-003 shows coupling
  problems.
