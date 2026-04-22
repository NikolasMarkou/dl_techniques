# Consolidated Decisions
*Cross-plan decision archive. Entries merged from per-plan decisions.md on close. Newest first.*

<!-- COMPRESSED-SUMMARY -->
## Summary (compressed)
*Auto-compressed. Read full content below if needed.*

### Key Decisions (outcomes + active constraints)
- **video_jepa training viz reuse** (`plan_2026-04-22_016e549b`): reused
  `TrainingCurvesCallback` as-is; added `LatentMaskOverlayCallback` +
  `PatchPredictionErrorCallback` in `src/dl_techniques/callbacks/jepa_visualization.py`
  (skeleton cloned from `DepthPredictionGridCallback`). Zero model edits.
  `callbacks/__init__.py` is empty by convention — full-module-path imports.
  Tests live under `tests/test_callbacks/`, not `tests/dl_techniques/callbacks/`.
- **video_jepa telemetry strip + BDD100K** (`plan_2026-04-22_4f29c76f`):
  dropped drone-telemetry conditioning path (scope A) + kept iter-2 per-loss
  Mean trackers; replaced AdaLN-zero predictor block with `CausalSelfAttnMLPBlock`
  (LayerScale γ=1e-5 restores identity-at-init); new BDD100K loader via
  opencv-python; sanity-run-first cadence. Architecture otherwise frozen.
- **video_jepa iter-1/2** (`plan_2026-04-21_421088a1`): hybrid Clifford encoder
  (PatchEmbedding2D + 2-4 CliffordNetBlock); factorized spatial+causal-temporal
  predictor; next-frame + tube-masked latent dual-loss; SIGReg middle placement
  `(B*T, N, D)`; rolling-buffer streaming. Latent-space masking = HARD CONSTRAINT
  (pixel-space incompatible with 2D-grid encoder). Future work: FW-001 lambda
  rebalance, FW-002 longer training, FW-003 EMA target encoder, FW-004
  mean-predictor shortcut probe.
- **cliffordnet detection** (`plan_2026-04-21_5dadc8ce`): YOLOv12 head reused
  via subclass-and-override (CliffordDetectionLoss rebuilds anchors/strides
  after super().__init__); 3-tap multi-level; declarative head specs via
  JSON-serialized dict; MultiTaskHead in vision_heads/factory.py is BROKEN
  (dict sublayers untracked by Keras).
- **lewm port** (`plan_2026-04-21_8416bc0b`): PyTorch->Keras 3; D-001 live
  target encoder (no EMA, no stop_gradient); D-002 LayerNorm not BatchNorm
  in MLPProjector.
- **weight transfer** (`plan_2026-04-21_4c4451e5`): D-001 layer-by-layer
  set_weights is the only robust name-based path in Keras 3.8 `.keras` files;
  `load_weights(.keras, by_name=True)` RAISES — latent bug in 3 places.
- **cliffordnet depth refactor** (`plan_2026-04-21_c49eca98`): minimal inline
  heads beat shared factories; declarative head specs; deep_supervision as
  per-head flag.

### Things NOT to do (failed approaches / hard rejections)
- Do not subclass `JEPAMaskingStrategy` for latent tube masking (pixel-level;
  wrong abstraction).
- Do not use `load_weights(path, by_name=True)` on `.keras` files (Keras 3.8).
- Do not drop patch tokens before a 2D-grid encoder (hard constraint).
- Do not use `MultiTaskHead` factory until dict-tracking bug is fixed.
- Do not run GPU jobs in parallel (GPU 0 = 4090 24GB, GPU 1 = 4070 12GB).
- Do not fork YOLOv12 loss/head; subclass and rebuild baked constants.
- Do not run full `make test` as routine regression (~1.5h; pre-push hook).
- Do not add decord when opencv-python random-seek suffices for sanity-scale.
<!-- /COMPRESSED-SUMMARY -->

## plan_2026-04-22_016e549b
### D-001 (2026-04-22, PLAN iter-1) — Library placement for new viz callbacks

**Decision.** Add two new callbacks to
`src/dl_techniques/callbacks/jepa_visualization.py` as:
- `LatentMaskOverlayCallback`
- `PatchPredictionErrorCallback`

**Why.** User directive: prefer library placement with an accurate name.
Names reflect the actual contract: the callbacks require a model exposing
`encode_frames + predictor + mask_gen + mask_token` — i.e. the JEPA
latent-masking trainer. Not generic "video" viz, not strictly "video_jepa"
(could apply to still-image JEPA or other latent-masking variants).

**Trade-off.** Library surface grows by one module and two classes at the
cost of leanness; in return, future JEPA-style models (image-JEPA, etc.)
can reuse these callbacks without copy-paste from the train script.

**Alternatives rejected.**
- Keep callbacks local to `train_video_jepa.py`: leaner, no public surface,
  but non-reusable and inconsistent with `depth_visualization` precedent.
- Name them `video_jepa_*`: too specific — the API they target is JEPA
  latent masking, not video-specific.
- Name them generically (`MaskOverlay`, `PatchL2Heatmap`): aspirational —
  the implementation has no generic abstraction over arbitrary models.

---

### D-002 (2026-04-22, PLAN iter-1) — Skip post-fit summary PNG (Pattern C)

**Decision.** Do not call `generate_training_curves(history, results_dir)`
after `model.fit`.

**Why.** User directive. Per-epoch `TrainingCurvesCallback` output is
sufficient; post-fit call is redundant for this workflow.

**Trade-off.** No publication-style final PNG at the cost of a nominal
one-line call; acceptable since loss curves already exist per-epoch.

---

### D-003 (2026-04-22, PLAN iter-1) — Fixed eval batch = first batch of training dataset

**Decision.** At callback construction time, pull one batch from the
training dataset via `next(iter(train_dataset))`, convert pixels to numpy,
and cache on the callback. Both JEPA viz callbacks share this fixed batch.

**Why.** User directive. Simplest and matches `DepthPredictionGridCallback`'s
"fixed eval batch" spirit from `findings/existing-callbacks.md`. Provides
cross-epoch visual stability without needing a separate validation pipeline.

**Trade-off.** Visualization shows a sample that the model has been trained
on, at the cost of "true" held-out fidelity; acceptable for this smoke /
sanity-check use case. If held-out viz is ever needed, swap the batch source
in the train script without touching the callbacks.

**Note on mask stochasticity.** `model.mask_gen` re-samples on each call;
the fixed *pixels* batch is cached, but the mask will differ epoch-to-epoch.
Documented in plan.md Failure Modes. If deterministic masks are needed
later, cache `M_spatial` once at construction — not in scope for this plan.

---

### D-004 (2026-04-22, PLAN iter-1) — Chosen approach: three callbacks, no model edits

**Decision.** Use `TrainingCurvesCallback` as-is for loss curves, and add
the two JEPA-specific callbacks (D-001) for mask overlay + patch error
heatmap. No edits to `src/dl_techniques/models/video_jepa/**`.

**Why.** All required tensors are reachable through the public submodule
API (`findings/video-jepa-tensors.md`). The only gap is a viz layer; that
belongs in `callbacks/`, not in the model.

**Trade-off.** A new module + two classes in the library at the cost of
isolation (would be zero new surface if we inlined into the train script);
in return, reusability and consistency with the `depth_visualization`
pattern.

**Alternatives rejected.**
- Inline callbacks in the train script: rejected per D-001.
- Extend the `visualization` plugin framework: heavier than needed — a
  full manager/plugin pair for one domain is over-abstraction.
- Add a forward-hook in `model.py` that stores intermediate tensors for
  a passive callback to read: rejected — mutates model for viz, violates
  the "no model edits" invariant.

---

### D-005 (2026-04-22, EXECUTE iter-1, Step 0) — Skeleton source & callbacks package surface

**Findings.**
- `src/dl_techniques/callbacks/__init__.py` is effectively empty (1 line,
  per the subpackage CLAUDE: "import directly from modules"). So A4 is
  **FALSE** — we do NOT export the new callbacks from `__init__.py`.
  Step 2 collapses to a no-op; the train script imports by full module path.
- Skeleton source confirmed: `src/dl_techniques/callbacks/depth_visualization.py`
  (`DepthPredictionGridCallback`). Pattern: lazy `_import_matplotlib()`,
  `try/except` with `logger.warning`, `gc.collect()`, `plt.close`,
  `epoch_{NNN:03d}_<name>.png` filenames, `(epoch + 1) % frequency != 0`
  guard, handle `n == 1` axes shape corner.
- Depth callbacks do **NOT** use `@keras.saving.register_keras_serializable()`.
  Callbacks aren't saved with models; plan.md's note to add it is overruled
  by following the actual repo precedent (depth_visualization).
- Dataset yield shape (synthetic_drone_video): `({"pixels": (B, T, H, W, C)}, 0.0)`.
  Extraction for eval batch: `batch[0]["pixels"]`. A3 is grounded.

**Decision.** Follow `depth_visualization.py` skeleton verbatim (no serialization
decorator, lazy matplotlib, try/except+logger.warning). Skip `__init__.py`
export (Step 2 no-op). Use `batch[0]["pixels"]` to pull pixels from the
dataset in the train script.

**Trade-off.** Matching depth precedent (no serialization decorator) over
plan.md's aspirational note, at the cost of minor inconsistency with the
broader "register everything serializable" convention — acceptable because
callbacks aren't serialized as part of models anywhere in this repo.

---

### D-006 (2026-04-22, EXECUTE iter-1, Step 1) — LOC overrun on jepa_visualization.py

**Finding.** `jepa_visualization.py` lands at 395 LOC (332 non-blank/non-comment)
vs. the plan-v1 target of ≤220.

**Cause.** Plan's ≤220 target scoped "the two callback classes"; it did not
budget the module-level helpers (`_import_matplotlib`, `_normalize_pixels`,
`_to_display_image`, `_upsample_mask`) nor the Google-style docstrings
required by repo convention (CLAUDE.md). The two class bodies themselves
are ~220 LOC combined; docstrings + helpers + import block add the rest.

**Decision.** Accept overrun; surface at REFLECT. Criterion #1 ("file ≤220
LOC") is in scope-drift status. Rationale: code is straightforward, split
into small functions, heavily commented — tightening would remove docstrings
(violates project convention) or inline helpers (reduces clarity).

**Trade-off.** Accept ~+175 LOC over budget for docstring completeness and
helper clarity, at the cost of the literal ≤220 target. If REFLECT flags
this as a blocker, we can strip docstrings to one-liners and inline
helpers to land ≤250 quickly.

---

### D-007 (2026-04-22, EXECUTE iter-1, Step 3) — Test directory correction

**Finding.** Plan.md specifies `tests/dl_techniques/callbacks/test_jepa_visualization.py`
but the repo's actual test tree is `tests/test_callbacks/` (mirrored from
`src/dl_techniques/callbacks/` via the `test_<pkg>/` convention, see
`tests/test_losses/`, `tests/test_layers/`, etc.). User's reminder echoed the
plan text verbatim — intent is to place tests under the standard tree.

**Decision.** Put the test at `tests/test_callbacks/test_jepa_visualization.py`
following the existing convention. Verification command updated in practice
(pytest target path): `tests/test_callbacks/test_jepa_visualization.py`.

**Trade-off.** Deviate from plan-v1 literal path at the cost of plan/file drift,
in return for matching the actual repo convention (single source of truth
for test discovery).

## plan_2026-04-22_4f29c76f
### 2026-04-22 iter-1 PLAN — scope & architecture decisions (chosen)

### D-P1: Telemetry scope = A only (drop drone conditioning; keep per-loss Mean trackers B)
**Trade-off**: Lose drone-telemetry feature path **at the cost of** discarding a model capability that was never validated on real drone data. Synthetic-only usage means the cost is near-zero. Keeping B (trackers) preserves CSV logging — the feature the user is actively using.

### D-P2: Replace AdaLN-zero conditional block with plain causal self-attn + MLP + LayerScale
**Trade-off**: Predictor becomes strictly unconditional (simpler) **at the cost of** giving up the documented "identity-at-init via zero-gated modulation" guarantee. Mitigation: LayerScale γ=1e-5 on both attn and MLP residual paths provides equivalent near-identity-at-init (same mechanism `CausalCliffordNetBlock` already uses). The renamed test (`test_predictor_identity_init`) validates this. Causality is preserved structurally (causal mask unchanged).

### D-P3: Architecture otherwise frozen (no EMA, no lambda rebalance, no predictor simplification, no loss drop)
**Trade-off**: Avoid compounding refactor risk **at the cost of** deferring iter-2 future-work items (FW-001..FW-004). If sanity run surfaces issues (e.g. lambda imbalance from iter-2 REFLECT), we address them in a follow-up plan — not in this surgical pass.

### D-P4: BDD100K with opencv-python (no decord)
**Trade-off**: opencv-python has broader compatibility and is already a likely transitive dep **at the cost of** slower random-frame access than decord. We accept the cost because sanity config (200 steps × B=4 × T=8) is I/O-bound at only ~6400 frames total — well within opencv's capability.

### D-P5: Sanity-run-first training cadence
**Trade-off**: 1 epoch × 200 steps at B=4/T=8/112² gives a fast signal (target ≤ 15 min) **at the cost of** not detecting long-horizon instabilities that only appear after 1k+ steps. Mitigation: if sanity passes, user will decide whether to schedule a full run as a follow-up plan.

### D-P6: GPU 0 (RTX 4090 24 GB) for sanity
**Trade-off**: Use the larger GPU even though B=4/T=8/112² fits on the 4070 **at the cost of** tying up the 4090. Benefit: maximum headroom if we bump config for the full run later; consistent with "sanity first, full later" plan cadence.

### D-P7: Synthetic dataset simplified (drop telemetry emission) rather than kept as-is
**Trade-off**: Cleaner synthetic loader (30 LOC less) **at the cost of** removing the feature that exercised the now-deleted telemetry path. The feature is dead code once D-P1 lands; keeping it would be a ghost constraint.

### Alternatives considered (rejected)
- **Keep AdaLN block with zero-tensor c** (A2 from findings) — rejected: leaves vestigial conditioning code + config fields for no benefit.
- **Switch predictor to pure-temporal V-JEPA block** — rejected by user (D-P3 freeze).
- **decord decoder** — rejected by user (D-P4).
- **Stanford Drone Dataset** — rejected: user chose BDD100K.

## plan_2026-04-21_421088a1
### 2026-04-21 — Locked design decisions (post-EXPLORE, pre-PLAN)

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

### 2026-04-21 — Complexity budget override (declared in PLAN)

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

### 2026-04-21 — PIVOT REFLECT → PIVOT → PLAN (iter-2, planned extension, NOT failure)

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

### 2026-04-21 — Hardest-first verification order

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

### 2026-04-21 — Iter-2 proposed design decisions (RECOMMENDATIONS — awaiting user confirmation)

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

### 2026-04-21 — REFLECT iter-2 evaluation

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

### 2026-04-21 — Future Work (iter-3+ candidates, queued at CLOSE)

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

## plan_2026-04-21_5dadc8ce
### D-001 — Natural decoder strides `[2, 4, 8]`, not YOLO-standard `[8, 16, 32]`
**Decision**: Tap decoder levels `[1, 2, 3]` on 4-level variants → strides `[2, 4, 8]`.  Do not add a detection neck.  Un-hardcode strides in the detection loss.
**Why**: User choice (Q1 answered "natural, follow the U-Net"). Preserves backbone alignment between detection and cls/seg/depth (no detection-only weights).  Anchor count at 256px = 21,504 vs YOLO's 8400; workable.
**Trade-off**: Denser anchors → slower TAL assignment + more memory **at the cost of** structural simplicity and shared feature reuse.  If anchor blowup becomes a real problem, pivot to a detection neck is the fallback.

### D-002 — Full COCO mAP in this plan (not deferred)
**Decision**: Ship `COCOMAPCallback` with pycocotools `COCOeval` computing mAP@50 and mAP@50-95 at every validation epoch.
**Why**: User choice (Q2 = "now").  Detection training without real mAP is nearly useless for model selection — train loss alone doesn't correlate reliably with detection quality.
**Trade-off**: ~400 LOC callback with several integration subtleties (box decode, NMS, cat-id remap, pixel coord conversion) **at the cost of** honest evaluation signal.  Large step in the plan (Step 6).

### D-003 — Multi-task integration, not a standalone detection script
**Decision**: Extend `train_coco_multitask.py` with `--enable-detection` to add a 3rd head (cls + seg + det simultaneously).
**Why**: User choice (Q3).  Keeps the single "one pretrained backbone" training story. Avoids a second training script.
**Trade-off**: 3-task loss competition + memory pressure **at the cost of** implementation simplicity.  Mitigated by per-task loss weights and keeping detection optional via the flag (invariant SC6).

### D-004 — Extend `COCO2017MultiTaskLoader` with `emit_boxes`, not a sibling class
**Decision**: Add `emit_boxes: bool = False` to `COCOMultiTaskConfig`. When true, labels dict gains a `"detection"` key with `(B, max_boxes, 5)` in xyxy-normalized layout, padded with `-1.0`.
**Why**: User choice (Q4).  Single data pipeline; avoid duplicating COCO JSON parsing.
**Trade-off**: Larger `COCO2017MultiTaskLoader` surface **at the cost of** code duplication elimination.

### D-005 — Subclass `YOLOv12ObjectDetectionLoss` rather than fork
**Decision**: `CliffordDetectionLoss(YOLOv12ObjectDetectionLoss)` with configurable `strides` — inherits CIoU + focal BCE + DFL + TAL assigner untouched.  Override only `_make_anchors` (or rebuild `self.anchors` / `self.strides` after `super().__init__()`).
**Why**: Reuse YOLOv12's battle-tested logic; keep the original loss available for yolo12 users.  Minimizes code duplication.
**Trade-off**: Coupling to YOLOv12's internal representation of anchors/strides **at the cost of** LOC efficiency.  If the parent refactors, we update once.

### D-006 — Padding sentinel `-1.0` for empty box slots
**Decision**: Pad `(max_boxes, 5)` to `max_boxes` with rows of `-1.0`. The loss mask `sum(boxes, axis=-1) > 0` correctly excludes these (sum = -4 < 0).
**Why**: Unambiguous sentinel; impossible for any real box to sum to -4.  Matches Ultralytics YOLOv5/v8 convention.
**Trade-off**: Disagrees with a reading of the existing `COCODatasetBuilder` (which may use 0-padding — unclear) **at the cost of** robustness.  We're writing a new loader path, not modifying the existing builder, so no compatibility break.

### D-007 — YOLO FPN ordering preserved (shallow-to-deep), not auto-sorted
**Decision**: `tap: [1, 2, 3]` is used as-is by the detection head.  Order convention: smallest stride first (shallowest / highest resolution) = P3-equivalent.
**Why**: `YOLOv12DetectionHead` and `YOLOv12ObjectDetectionLoss` both expect `[P3, P4, P5]` ordering.  Auto-sorting would be invisible but catastrophic if a user ever passes `[3, 2, 1]` or mixes a bottleneck in.
**Trade-off**: User must get the order right **at the cost of** silent breakage if we sorted.

### D-008 — DFL decoder lifted into a standalone utility
**Decision**: `src/dl_techniques/utils/yolo_decode.py` exports `decode_dfl_logits`, `make_anchors`, `decode_predictions`, `nms_per_class` — reusable by inference + the mAP callback + future users.
**Why**: DFL decode is currently inlined in `yolo12_multitask_loss.py:387-401`.  Inference script at `inference.py:262` references a non-existent `decode_predictions` method.  Lifting fixes dead code and de-duplicates future work.
**Trade-off**: One extra file **at the cost of** zero duplication and a clean API surface.

### D-009 — Eager numpy NMS in the mAP callback (not graph NMS)
**Decision**: `nms_per_class` in `yolo_decode.py` uses eager numpy greedy NMS.
**Why**: The mAP callback runs at epoch end, outside the compiled training step.  No need for graph-compatibility.  Existing `bbox_nms` (`bounding_box.py:413`) is already Python-loop-based; we write a cleaner version inline to avoid coupling to that legacy utility.
**Trade-off**: Slower than a TF-native NMS **at the cost of** simplicity. Val mAP computation time is dominated by the model forward pass, not NMS.

## plan_2026-04-21_8416bc0b
### 2026-04-21T10:15:00Z — PLAN iter-1: chosen approach

**Decision**: Port LeWM as a new model package under `src/dl_techniques/models/lewm/`, reusing the existing `ViT` model for the encoder backbone, writing new custom layers for the parts that have no analogue (`AdaLNZeroConditionalBlock`, `SIGRegLayer`), and a thin dataset module supporting both synthetic smoke-test data and a PushT HDF5 skeleton.

**At the cost of**: 13 new files (above the default 3-file complexity budget), ~1200 net LOC added, two non-trivial new custom layers that must be tested carefully for serialization. We accept this cost because the port is inherently a new-model task — there is no shortcut. We pay the complexity up front, bounded by tests + serialization round-trip + AdaLN-zero identity check.

**Alternatives rejected**:
- *Extend existing `models/jepa/`* — that package is masked I/V-JEPA, not action-conditioned; name collision would confuse users. Use distinct `models/lewm/`.
- *Extend `layers/film.py` to 6-way modulation* — AdaLN-zero is a different pattern (gated residual + zero-init). Writing a dedicated layer is cleaner than overloading FiLM.
- *EMA target encoder* — upstream LeWM does not use EMA (target encoder = live encoder, gradient flows). We match upstream. Noted as future ablation.
- *Rewrite ViT-tiny inline* — wasteful; existing ViT with `include_top=False, pooling='cls', scale='tiny', patch_size=14` is a drop-in fit.

**Assumptions that could invalidate**: A1 (ViT CLS pooling returns (B, 192) cleanly), A4 (no EMA target), A6 (BatchNorm safe under fit). Tracked in plan.md Assumptions table.

### 2026-04-21T10:15:01Z — D-001 anchor: target encoder is live, not EMA

**Decision**: In `LeWM.call`, the target embedding for the JEPA loss comes from the **same live encoder** as the context embedding. Gradient flows through the target path (no `stop_gradient`). This matches upstream `/tmp/lewm_source/jepa.py:29-45` where `self.encoder` is called once per forward and no `.detach()` appears before the loss.

**Why this matters for future changes**: Many JEPA variants (I-JEPA, V-JEPA, BYOL-style) use an EMA target encoder with stop-gradient. A future reader of this code might "helpfully" add stop-gradient thinking it's a bug. It is not. Upstream LeWM is the ablation where the target encoder is live.

**Anchor location**: Inline `# DECISION D-001` comment in `models/lewm/model.py` at the target-emb computation site.

### 2026-04-21 — D-002 anchor: MLPProjector uses LayerNorm, not BatchNorm

**Decision**: `MLPProjector` (models/lewm/projector.py) uses `LayerNormalization` on the hidden activation, not `BatchNormalization`.

**Why**: Plan.md Step 4 described "Linear → BatchNorm → GELU → Linear" (matching findings #1 summary). Re-reading the actual upstream class `/tmp/lewm_source/module.py:159-172`, `MLP.__init__` defaults `norm_fn=nn.LayerNorm`, not BatchNorm. The JEPA wiring uses the default. We match upstream truth (code), not the plan's paraphrase. This also avoids the BN-batch-of-1 failure mode flagged in plan.md Pre-Mortem Scenario 2, so assumption A6 is moot here.

**Impact**: No downstream change — the projector's external shape contract is identical. Loss curves may differ slightly vs the hypothetical-BN version, but the smoke test just needs finite loss.

**Anchor location**: inline docstring in `models/lewm/projector.py` explaining the LayerNorm default and the reason.

### 2026-04-21T13:11:00Z — REFLECT iter-1: all criteria PASS

**Outcome**: All 6 success criteria (C1–C6) PASS on first attempt across 10 plan steps with zero fix attempts. No surprises, no failed falsification signals, no scope drift.

**Evidence**:
- `pytest tests/test_layers/test_adaln_zero.py tests/test_regularizers/test_sigreg.py tests/test_models/test_lewm.py -vvv` → 11/11 passed in 20.53s (CPU).
- `MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.lewm.train_lewm --synthetic --batch-size=2 --epochs=1 --steps-per-epoch=2` → exit 0, final loss 0.8782 (finite), `training_log.csv` + `final_model.keras` + `last.keras` written under `results/lewm_20260421_130931/`.

**Simplification Checks**: all 6 clean. 13 files added was within the plan's declared budget (justified: full-model port), and every abstraction maps 1:1 to an upstream class.

**Devil's advocate**: the smoke defaults downscale (56×56, depth=2, num_proj=64). Full-spec LeWM at 224×224 / depth=6 / num_proj=1024 is not exercised here. Any scale-only issue would not invalidate the port but would require tuning on a full run — explicitly out of scope for this plan.

**Recommendation**: CLOSE. Follow-up work (real HDF5, full-scale training, eval.py / MPC) should become new plans.

## plan_2026-04-21_4c4451e5
### D-001 — Layer-by-layer `set_weights` rather than `load_weights(by_name=True)`
**Decision**: In `load_weights_from_checkpoint`, load the full source model with `keras.models.load_model`, then iterate source layers and copy each compatible layer's weights into the target via `target.get_layer(name).set_weights(source_layer.get_weights())`.
**Why**: Keras 3.8 raises `ValueError("Invalid keyword arguments: {'by_name': True}")` when `load_weights` is called on a `.keras` file with `by_name=True`. Our checkpoints are `.keras`. The only robust name-based transfer path is manual layer-by-layer. Three existing helpers in the repo (`cliffordnet/model.py:413`, `bfunet.py:515`, `convnext_v2.py:400`) pass `by_name=by_name` and will crash on `.keras` checkpoints — latent bug, acknowledged, out of scope to fix here.
**Trade-off**: Requires loading the full source model (memory spike) **at the cost of** correctness. Worth it — our checkpoints are small (≈200 MB max) and transfers are rare.
**Anchor**: `# DECISION D-001` comment near the helper definition.

### D-002 — Skip prefixes are a constructor argument, not hardcoded
**Decision**: `load_weights_from_checkpoint(target, ckpt_path, skip_prefixes=("head_primary_", "head_aux_"), strict=False)` — the default matches `CliffordNetUNet`'s naming, but callers can override for other models.
**Why**: We want to ship a generic utility in `utils/`, not a CliffordNet-specific helper. Generic keeps it reusable when the same bug bites other architectures.
**Trade-off**: Slightly more ceremony in the default case **at the cost of** discoverability and reuse.

### D-003 — `TransferReport` dataclass over raw dicts
**Decision**: Return a small `TransferReport` dataclass with named fields (`loaded`, `skipped_by_prefix`, `shape_mismatch`, `missing_in_source`, `unused_in_source`) and a `summary_string()` method. Do not return a plain `Dict[str, List[str]]`.
**Why**: The report is the *primary* audit artifact when debugging "why did training not improve?" — structured access matters more than brevity here.
**Trade-off**: ~30 extra lines **at the cost of** IDE autocomplete, type safety, and a clean `logger.info(report.summary_string())` at the call site.

### D-004 — Fix seeds in `train_depth_estimation.py`, flag-controlled
**Decision**: Add `--seed <int>` (default 42) and set: `numpy.random.seed`, `tensorflow.random.set_seed`, `keras.utils.set_random_seed`, `random.seed`. Serialize the seed into `config.json`.
**Why**: From-scratch vs from-COCO comparison (user-triggered next step) needs bitwise-reproducible initialization to attribute any metric difference to the pretraining rather than to RNG luck.  Findings explicitly note zero seed fixing currently.
**Trade-off**: ~5 lines **at the cost of** minor reproducibility debt (we don't seed `MegaDepthDataset` internals because it doesn't expose a seed; document non-reproducibility of dataset shuffle ordering in `config.json`).

### D-005 — Add all five depth metrics, not the current two
**Decision**: Change `train_depth_estimation.py`'s compile call to include `AbsRelMetric`, `SqRelMetric`, `RMSEMetric`, `RMSELogMetric`, `DeltaThresholdMetric(1.25)`. Current script only compiles the first and last.
**Why**: The user wants to compare from-scratch vs from-COCO quantitatively. More metrics = richer comparison. Cost is negligible (5 extra numbers per epoch in the CSV).
**Trade-off**: Slightly slower per-epoch eval (~0.5 extra seconds per epoch) **at the cost of** a much richer comparison surface.

### D-006 — Compare-runs tool reads CSVs, not TensorBoard events or in-memory history
**Decision**: `compare_runs.py` reads `training_log.csv` (standard output of `CSVLogger` in `create_callbacks`). Does not parse TensorBoard events; does not depend on the Keras `History` object.
**Why**: CSV is stable, parseable without any Keras/TF runtime, and persists after the training process exits. TensorBoard events require parsing protobufs; `History` requires the trainer be in-process.
**Trade-off**: Can only compare what was logged via CSV **at the cost of** simplicity. If the user later wants per-step granularity they'll have to extend.

### D-007 — Three-file latent-bug: document but do not fix
**Decision**: Note the `load_pretrained_weights(by_name=True)` bug in `cliffordnet/model.py:413`, `bfunet.py:515`, `convnext_v2.py:400` in `plans/LESSONS.md` at CLOSE. Do not modify these files in this plan.
**Why**: Fixing three unrelated model helpers is a separate concern. Each fix requires testing with their own model families (`CliffordNet`, `BFUNet`, `ConvNeXtV2`) and we have no signal any user is hitting these today. Scope discipline.
**Trade-off**: The bug remains latent **at the cost of** keeping this plan focused. Tracked for a future cleanup plan.

### D-008 — Defer detection to Plan B
**Decision**: Detection head engineering is out of scope for this plan. Tracked as "Plan B" — will need: multi-tap head system extension to `CliffordNetUNet`, a `CliffordNetDetectionHead` (custom because YOLOv12DetectionHead cannot dispatch through our single-tap head_configs), adapting `YOLOv12ObjectDetectionLoss` to un-hardcode its `[8, 16, 32]` strides, a detection dataset adapter, and a training script.
**Why**: Detection is substantially more engineering than transfer plumbing; folding them into one plan would 15+ the step count, span multiple sessions, and block the critical path (depth bootstrap). User confirmed the split in Q1 of the pre-plan dialogue.
**Trade-off**: Detection availability pushed out **at the cost of** a bounded, shippable plan today.

## plan_2026-04-21_c49eca98
### D-001 — Build minimal heads inline in `unet.py` instead of reusing `vision_heads/factory.py`
**Decision**: Write `_ClassificationHeadBlock` and `_SpatialHeadBlock` as small internal layers inside `unet.py` (mirroring existing depth head style in `depth.py:357-370`), rather than using `ClassificationHead` / `SegmentationHead` / `DepthEstimationHead` from `src/dl_techniques/layers/vision_heads/factory.py`.
**Why**: The factory heads are over-parameterized (optional attention + FFN + dropout stack), targeted at transformer pipelines, and `MultiTaskHead` stores child heads in a plain Python `dict` that Keras does not track for weight serialization (`findings/multihead-patterns.md`). Minimal inline heads (LayerNorm → Conv/Dense) stay faithful to the current depth head and keep serialization simple.
**Trade-off**: Loses `create_vision_head(...)` ergonomics **at the cost of** ~60 extra lines in `unet.py`. Acceptable; refactor toward the factory later if other CliffordNet models need it.
**Anchor**: `# DECISION D-001` comment in `unet.py` near `_ClassificationHeadBlock` definition.

### D-002 — Decoder output is a numeric level; `"bottleneck"` is a separate tap name
**Decision**: `tap: int` = decoder output at that level (0 = full-res, N-1 = coarsest decoder output). `tap: "bottleneck"` = post-bottleneck features (same spatial size as level N-1 but before any decoder block).
**Why**: Classification heads semantically tap the bottleneck (max receptive field, no decoder involvement). Dense heads tap decoder outputs. Separating these prevents confusion between "deepest decoder output" and "pure encoder/bottleneck feature".
**Trade-off**: Adds a special-case string literal **at the cost of** the simpler "everything is an int level" scheme.

### D-003 — Delete `depth.py` rather than deprecate (user waived backward compat)
**Decision**: Remove `src/dl_techniques/models/cliffordnet/depth.py`. Replace all `CliffordNetDepthEstimator` call sites with the `create_cliffordnet_depth(...)` factory in `unet.py`.
**Why**: User's answer to Q4 was "don't care about backwards compatibility". Cleaner codebase, no dead code.
**Trade-off**: Any undiscovered external consumer breaks **at the cost of** a cleaner codebase. Mitigated by Step 1 grep audit.

### D-004 — COCO classification target is multi-hot from instance annotations
**Decision**: Per-image label = 80-dim float32 vector, 1.0 at every class index with ≥1 instance. Loss: `BinaryCrossentropy(from_logits=True)`.
**Why**: User's Q2 answer confirmed multi-label. Standard COCO-ML formulation.

### D-005 — Local COCO loader, not tfds-backed
**Decision**: Read directly from `/media/arxwn/data0_4tb/datasets/coco_2017/{train2017,val2017,annotations}/` via `pycocotools` (fallback `skimage.draw.polygon` for masks). Do not use `COCODatasetBuilder` (tfds).
**Why**: User has COCO locally (Q5). tfds would redownload. `COCODatasetBuilder` is detection-focused.
**Trade-off**: Extra dataset module **at the cost of** vendor-lock-in to local path — mitigated by parameterization.

### D-006 — Deep supervision is a per-head flag, default-on via factory/training-script config
**Decision**: `deep_supervision` flag per head spec. When True on a spatial-tap head, aux heads auto-attach at every deeper decoder level. Defaulted to True in `create_cliffordnet_depth` and in the COCO seg head.
**Why**: User's "always enabled" phrasing interpreted as default-on. Hard-coding it (no flag) would break unit testing; default-True flag is cleanest.

### D-007 — Serialize `head_configs` as JSON string in `get_config()`
**Decision**: `json.dumps(head_configs, default=str)` in `get_config()`, `json.loads(...)` in `from_config()`.
**Why**: `head_configs` is `Dict[str, Dict[str, Any]]` with mixed types (int tap, string `"bottleneck"` tap). Explicit JSON marshaling avoids Keras default-config nested-dict pitfalls.
**Trade-off**: Config file less Keras-idiomatic **at the cost of** guaranteed round-trip correctness.
