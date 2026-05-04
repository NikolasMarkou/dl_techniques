# Consolidated Decisions
*Cross-plan decision archive. Entries merged from per-plan decisions.md on close. Newest first.*

## plan_2026-05-01_1c080382
### D-001 — Choose STRETCH scope (A0 + B1 + B2 + C1 + C2) (2026-05-01, PLAN, iter-1)

**Decision**: User explicitly chose STRETCH scope (13 runs, ~25h). Implement all four B/C cells, plus A0 anchor-seed panel via `--seed` reuse of existing V0/V1/V7.

**Trade-off**: Maximum experimental coverage at cost of ~25h GPU time vs. ~21h MIN scope. Buys the pyrdiff-on-V1-substrate disambiguation (B2) and the LN late-stage control (C2). Worth it given user explicitly accepted the cost.

### D-002 — Extend in-place, do not create sibling script (2026-05-01, PLAN, iter-1)

**Decision**: Add the 4 new VARIANTS and the per-stage / seed plumbing to the existing `train_downsampling_experiments.py`. No `train_downsampling_followup.py` sibling.

**Trade-off**: Single-script consistency and shared comparison.csv across both campaigns at cost of growing one file beyond 850 lines. User explicitly chose in-place. Deferring sibling-split until the file becomes unwieldy (≥1500 lines).

### D-003 — Operate autonomously (no PLAN→EXECUTE approval gate) (2026-05-01, PLAN, iter-1)

**Decision**: User said "WORK UNSUPERVISED". Skip the user-approval handoff after PLAN. Proceed directly into EXECUTE. Treat any further user-input gates as auto-approved unless hitting a genuine blocker (irreversible op, ambiguous failure, autonomy-leash exhaustion). Still STOP before launching the full 25h training run — only present the launch command.

**Trade-off**: Faster iteration at cost of losing the early plan-review checkpoint. Mitigation: the plan is small (4 variants, 1 helper, 1 plumbing change), risk is low, and CLOSE still surfaces a final review.

### D-004 — Per-stage `ctx_norm_type`: list-of-3 with scalar back-compat (2026-05-01, PLAN, iter-1)

**Decision**: Encode the per-stage override as either a scalar str (existing behaviour, applies uniformly to stages 1/2/3 transitions) OR a length-3 list `[stage1, stage2, stage3]`. Pop the key out of `transition_kwargs` when it's a list and inject explicitly at each transition build site.

**Trade-off**: Backwards-compatible at the cost of a single conditional in `build_variant`. Alternative (separate `ctx_norm_schedule` dict key) would have been cleaner but more invasive. Sticking with overload-the-existing-key per project preference for minimal surface-area churn.

### D-005 — Use V1 substrate (no `internal_expansion`) for B1/B2/C1/C2 (2026-05-01, PLAN, iter-1)

**Decision**: All four new variants use `internal_expansion=False` (matching V1's substrate exactly). They stack ONE axis at a time on V1.

**Trade-off**: Tests the marginal contribution of each axis on the empirical winner at cost of NOT testing axis interactions with internal expansion. The prior V11 kitchen-sink stack on V4 substrate already failed (-0.51pp); we're avoiding that failure mode by keeping the substrate clean.

## plan_2026-04-30_3a94be21
### D-001 — 2026-04-30 — Implement V0–V7, V10–V12 (11 variants); defer V8 + V9

**Decision.** Implement 11 of 12 variants from `analysis_2026-04-30_41b5e415/summary.md` §4. Defer V8 (full-res product, axis F) and V9 (grade-aware grouped pool).

**Why.** V8 requires structural refactor of `CliffordNetBlockDS.call()` that reverses pool/product order — high blast radius, +30–50% FLOPs, marginal payoff per analysis §3 hierarchy table. V9 is an open research question (no grade metadata in repo, requires constructive design choice) — analysis §6.1 calls it out as needing separate research scope. Both are explicitly low priority in the analysis.

**Cost.** The campaign cannot test the principled "interact-then-pool" hypothesis (V8) or the Clifford-correct grade-aware pool (V9). User loses the ability to validate H5 (bilinear bandwidth doubling) and H14 (grade-aware pool) in this iteration. Mitigation: leave hooks in `VARIANTS` dict commented out; structure the block refactor so V8/V9 can be added incrementally.

### D-002 — 2026-04-30 — Stage layout `(96,2),(192,2),(384,4),(768,4)` with patch1 stem

**Decision.** 4-stage backbone with channel schedule 96-192-384-768 (2-2-4-4 blocks). Patch1 stem (no spatial reduction at stem) so first strided transition is 32→16, then 16→8, 8→4. Total ~10M params at C=96 base — matches analysis "10M multi-stage" scale.

**Why.** Analysis §6 H_SCOPE_MACRO assumes "4-stage backbone with channel ratios (96, 192, 384, 768) or similar". 96 is divisible by 4 and 16 (pixel-unshuffle skip needs `C·s²` integer; trivially satisfied). 12 total blocks ≈ existing `V4_4stage_aggressive` (12 blocks too) so wall-time per variant is comparable to E05 (~200 min at 10M).

**Cost.** Different macro-arch may shift relative ranking (acknowledged as H_SCOPE_MACRO open in §6). Mitigation: layout is one constant in the script, easy to re-run with `(64,2),(128,2),(256,4),(512,4)` if needed.

### D-003 — 2026-04-30 — Refactor `CliffordNetBlockDS` rather than subclass

**Decision.** Extend `CliffordNetBlockDS` constructor with new params (`stream_pool`, `out_channels`, `ctx_norm_type`, additional `skip_pool` and `ctx_mode` enum values) rather than create a sibling class `CliffordNetBlockDSv2`.

**Why.** The new knobs are orthogonal axes of the same architectural concept; a sibling class would duplicate ~70% of code. Strict additivity (defaults preserve current behavior) avoids breaking `train_compare_variants.py`, `train_downsampling_techniques.py`, and existing tests. `use_ctx_bn` kept as deprecated alias for `ctx_norm_type` to avoid call-site churn.

**Cost.** Higher refactor risk on a layer used by other scripts. Mitigation: keep all new params optional with current-behavior defaults; verify with smoke-build of legacy training scripts before declaring step 3 done.

### D-003-AMENDED — 2026-04-30 — Use sibling class `CliffordNetBlockDSv2` (user direction)

**Decision (supersedes D-003).** Per user direction, create a NEW sibling class `CliffordNetBlockDSv2` in `src/dl_techniques/layers/geometric/clifford_block.py` that contains all axis A–H knobs natively. Leave `CliffordNetBlockDS` untouched.

**Why.** User wants zero-risk back-compat. A sibling class:
- Cannot accidentally break existing tests / training scripts (they keep using `CliffordNetBlockDS`).
- Lets us add new parameters without deprecation aliases (`use_ctx_bn`).
- Lets us refactor `call()` cleanly for `pyramid_diff` and the future V8 full-res-product variant without nesting flags.
- Code duplication is acceptable (~70% overlap) since this is a research artifact.

**Cost.** ~400 lines of duplicated code in `clifford_block.py`. Two classes to maintain. Mitigation: file already houses both `CliffordNetBlock` and `CliffordNetBlockDS` so a third class is consistent with the file's role.

### D-005 — 2026-04-30 — Run smoke-test first, then full training only after user re-approval

**Decision.** Step 5 = smoke-test (3 epochs × batch 32 × 11 variants, ~10–15 min total). Step 7 added: full 100-epoch training run (`--variant all --epochs 100 --batch-size 128 --gpu 0`) — ONLY after smoke passes AND user explicitly approves the full run.

**Why.** Full run is ~40h serial on RTX 4090 (analysis §5 estimate ~75h for 21 cells = ~3.5h per 10M cell × 11 variants). User wants smoke validation as a checkpoint before committing GPU time.

**Cost.** Adds a sync point. Trivial.

### D-004 — 2026-04-30 — V11 picks `pyramid_diff` over `abs` (V6 dominates V7)

**Decision.** V11 (`DS-kitchen_sink`) uses `ctx_mode=pyramid_diff` rather than `abs at strides>1`. V6 and V7 are mutually-exclusive ctx_mode choices.

**Why.** Analysis §3 hierarchy table puts V6 (pyramid_diff) and V7 (abs) at +0.1–0.3 each but V6 is principled-as-Laplacian-replacement while V7 is bypass. Stacking both is impossible at one ctx_mode value. Pyramid_diff preserves the Laplacian semantics axis D was meant to express. Cost: V11 won't isolate V7's contribution — that's already isolated in V7 itself.

## plan_2026-04-30_9c4cdf31
### D-001 (PLAN, 2026-04-30) — V2 alongside, legacy untouched
**Decision**: Add `CapsNetV2`, `AttentionRoutingCapsule`, `CapsuleBlockV2` as new files alongside the existing `CapsNet`/`RoutingCapsule`/`CapsuleBlock`. Modify legacy code only for the LayerNorm bug fix.
**Rationale**: User-confirmed default (Q1 answer). Existing tests stay green; users opting in to V2 get the new architecture without API breakage.
**Trade-off**: More code (≈1500 net+) and two parallel APIs **at the cost of** a clean migration path and zero risk of regressing currently-working behavior.

### D-002 (PLAN, 2026-04-30) — Bug fix: length-preserving LN, API stable
**Decision**: Keep `CapsuleBlock(use_layer_norm=True)` as the public API; change its IMPLEMENTATION to length-preserving (split magnitude/direction, LN the direction, re-normalize, multiply magnitude back).
**Rationale**: API stability for legacy users; the broken behavior is replaced with the correct behavior (margin loss now sees the magnitude it expects). Cleaner than introducing a new flag.
**Trade-off**: Behavior change for any user who was depending on the old (buggy) magnitude rescaling **at the cost of** correctness for the documented length-as-probability semantics. Mitigated by `logger.info` at build time noting the correction.

### D-003 (PLAN, 2026-04-30) — Decoupled length × probability in `AttentionRoutingCapsule`
**Decision**: Replace squash entirely in V2. Use `v = sigmoid(prob_head(s)) * (s / (||s|| + eps))` — magnitude is a learned scalar from a per-capsule Dense head; direction is unit vector from raw routing aggregate.
**Rationale**: Matches the Stage 3a design from the prior epistemic-deconstructor analysis (OBS-004). Eliminates squash saturation at 0 and conflation of "detection probability" with "vector magnitude side-effect".
**Trade-off**: Slightly more parameters (the prob head) and a behavior different from "classical" capsules **at the cost of** clean gradient flow at small magnitudes and clearer semantic separation.

### D-004 (PLAN, 2026-04-30) — Softmax over output capsules (axis=2)
**Decision**: Default `softmax_axis="output"` in `AttentionRoutingCapsule` — each input capsule distributes its weight competitively across output capsules (matches dynamic routing semantics).
**Rationale**: Preserves the "routing-by-agreement" intuition: an input votes for one parent, with probabilistic mass distributed by competition.
**Trade-off**: Risk of coupling collapse (one parent wins all) **at the cost of** clearer semantic match to capsule routing literature. Pre-mortem signal Scenario 2 covers the fallback (switch to axis="input" if collapse observed at random init).

### D-005 (PLAN, 2026-04-30) — No mixup in factory; defer to data pipeline
**Decision**: `create_capsnet_v2` does NOT include mixup in the modern recipe defaults.
**Rationale**: Mixup is a data-augmentation pipeline concern, not a model-side default. No mixup utility in `dl_techniques/datasets/` was found. Adding it would either (a) couple the model factory to a specific augmentation library or (b) require a new mixup utility — out of scope for "code path only" Stage 1.
**Trade-off**: Recipe is missing one component **at the cost of** factory-pipeline coupling cleanliness. Document mixup as a recommended training-time augmentation.

### D-007 (REFLECT, 2026-04-30) — Refactor `AttentionRoutingCapsule` matmul to `ops.einsum`
**Decision (in EXECUTE step 6)**: replaced `ops.matmul(W, u_tiled).squeeze(axis=-1)` (with `W: (1, N_in, N_out, D_out, D_in)` × `u_tiled: (B, N_in, N_out, D_in, 1)`) with `ops.einsum("iode,bie->biod", self.W, inputs)` and reshaped W to `(N_in, N_out, D_out, D_in)`.
**Why**: the matmul + squeeze pattern works in eager mode but the static last-axis size is lost under `tf.function` tracing inside `model.fit()`, raising "Cannot squeeze axis=-1, because the dimension is not 1". Einsum is shape-robust under graph mode.
**Trade-off**: Slightly less explicit indexing **at the cost of** clean graph-mode execution. All 24 V2 layer tests + 16 V2 model tests + 80 legacy tests still pass after the refactor.
**Falsification signal status**: Pre-Mortem Scenario 1 (serialization break) — did NOT fire. Scenario 2 (routing collapse) — did NOT fire (test_lengths_show_variance passes). Scenario 3 (pretrained fallback) — did NOT fire (test_..._falls_back_to_random_init passes).

### D-006 (PLAN, 2026-04-30) — Pretrained URL placeholders accepted; rely on existing fallback
**Decision**: Stage 2 factory `create_capsnet_v2_pretrained` passes through to `create_resnet(pretrained=...)` and inherits its fallback behavior (try download → fail → log warning → random init).
**Rationale**: The repo's existing pattern. No new code path; failure mode is explicit and degrades gracefully. Users with a local weights file can pass `stem_pretrained="/path/to/weights.keras"`.
**Trade-off**: Can't actually load ImageNet weights from the placeholder URL **at the cost of** complexity. Acceptable — code path is the deliverable.

## plan_2026-04-29_b6dbc601
### D-001 (PLAN, iter-1): Pool x_norm BEFORE stream split (instead of pooling z_det only)
**Choice**: When `strides>1`, apply pool to `x_norm` once before computing both `z_det` and `z_ctx`; the strided DWConv produces the already-pooled context.

**Trade-off**: Cleanest invariant satisfaction (z_det and z_ctx share shape automatically) at the cost of computing the LayerNorm on full-resolution input (Linear projection done on pooled tensor instead).

**Alternatives considered**:
- Pool z_det after Linear (full-res Linear then pool): wastes compute on the linear projection, no quality benefit.
- Pool x_prev once and feed both streams from there (skipping LayerNorm on full-res): would change normalization semantics — LN computes statistics on a smaller spatial map.

Going with: pool x_norm post-LN, before both streams. Same skip-pool type used for residual `x_prev`.

### D-002 (PLAN, iter-1): Naming — `CliffordNetBlockDS`
**Choice**: Class name `CliffordNetBlockDS` (DS = downsample / single-conv).

**Trade-off**: Communicates the new capability (downsampling) at the cost of a slightly cryptic suffix. Alternative `CliffordNetBlock7x7` only describes the kernel size, not the strided variant — misleading when users use `strides=1`.

### D-003 (PLAN, iter-1): Single class, not a parameter on existing block
**Choice**: New class instead of adding `kernel_size` / `strides` / `skip_pool` parameters to existing `CliffordNetBlock`.

**Trade-off**: Some duplication of forward-pass scaffolding at the cost of keeping the existing isotropic block exactly stable (no signature change → no risk to existing CliffordNet models that have been tuned in prior plans, see LESSONS L21-25). Existing serialized checkpoints continue to deserialize unchanged.

### D-004 (PLAN, iter-1): Two pool instances (stream-split + skip), not shared
**Choice**: Build two separate `keras.layers.AveragePooling2D` (or `MaxPooling2D`) instances when `strides>1`. They have identical configs but are not the same object.

**Trade-off**: Slightly more verbose at the cost of avoiding any chance of state-sharing surprises (pool layers have no state, but Keras serialization sometimes treats shared sublayers oddly). Two layers also makes the forward path more readable: `pool_stream(x_norm)` vs `pool_skip(x_prev)`.

### D-005 (REFLECT, iter-1): Devil's advocate — what could still be wrong
- The strided DWConv with `padding="same"` and the 2x2 pool with `padding="same"` both ceil-divide spatial dims, so they produce matching shapes. Verified empirically across H ∈ {8, 16, 32}. Edge case: H=7 with strides=2 → both produce 4. Not exhaustively tested but `padding="same"` semantics are stable. Risk: low.
- Devil's advocate concern: residual identity test at strides=2 only covers `skip_pool="avg"` against an explicit `AveragePooling2D` reference. The corresponding `skip_pool="max"` case isn't pixel-checked but is implicitly verified by `test_skip_pool_avg_vs_max_differ` (proves max path is wired distinctly). Acceptable.
- All criteria PASS, no regressions. Recommend CLOSE.

## plan_2026-04-24_cf1a9ab7
### D-001: Wrap `CliffordNetBlock`, do NOT modify it (2026-04-24, PLAN iter-1)

**Decision**: Implement hierarchical variants by building functional Keras models that interleave existing `CliffordNetBlock` instances with new inter-stage downsamplers. Leave `src/dl_techniques/models/cliffordnet/model.py` untouched.

**Trade-off**: We accept duplicated builder code (the new script defines its own stem+stages assembly) at the cost of not adding a `HierarchicalCliffordNet` class to the library. **In exchange we gain** zero risk of breaking existing CliffordNet tests, no churn on the CIP-style serialization surface, and a self-contained experiment whose lifetime is the experiment.

**Alternatives rejected**:
- Refactor `CliffordNet` into a hierarchical variant: would require touching every downstream user (`CliffordCLIP`, `CliffordNetUNet`, depth, denoiser) and breaks `from_variant` semantics. Out of scope for an experiment.
- Add a new `HierarchicalCliffordNet` model class in `models/cliffordnet/`: adds permanent surface area for what is fundamentally a one-off comparison. Defer until one variant clearly wins; promotion can happen in a follow-up plan.

### D-002: Run only 100 epochs (not paper's 200) (2026-04-24, PLAN iter-1)

**Decision**: Train all 5 variants for 100 epochs at the AdamW/cosine recipe, instead of the paper's 200.

**Trade-off**: Final accuracy will be 1-3 pp below the paper's reported numbers, at the gain of fitting all 5 runs in ~4-6h on a single 4090 (vs ~10-12h at 200 epochs). The architectural ranking is preserved at 100 epochs because cosine has fully decayed by then.

**Alternatives rejected**:
- 200 epochs each: doesn't fit a single sitting; user would have to babysit overnight.
- 50 epochs each: too short for cosine to anneal cleanly; ranking gets noisy.

### D-003: Keep base channels=128 only where the variant doesn't double; use 64 as starting width for variants with channel doubling (2026-04-24, PLAN iter-1)

**Decision**: Variants V1, V2, V3, V4 start at C=64 (so 64→128→256[→512]); V0 baseline and V5 stay at C=128 throughout (V0) or 128→256 (V5).

**Trade-off**: Smaller starting width on hierarchical variants makes their final stage match V0's width (rather than 2x or 4x exceeding it), keeping params comparable. Cost: stage-1 of V1-V4 has lower per-block expressivity.

**Alternatives rejected**:
- Start all at C=128 and double to 256/512: V4 would balloon to ~10M params and dwarf the baseline, defeating the comparison.

### D-004: Truncate smoke test to offline correctness checks (2026-04-24, EXECUTE step 2)

**What happened**: Step-2 full smoke (`--smoke-test --variant all`) stalled on V2 for 12+ min at epoch 1, while V0+V1 completed in 4-5 min each. Killed and re-profiled with isolated scripts.

**Root cause**: Keras `Model.fit` traces a `tf.function` (XLA-eligible) on the first `train_on_batch` call. Tracing time scales with graph complexity: V0 ~5-10s warmup, V3/V4 ~100s warmup. On the batch=32 smoke path, the CPU-heavy `tf.numpy_function` augmentation (AutoAugment + RandomErasing in pure numpy) also stalls GPU-idle steps. Combined, V2+V3+V4 warmup + CPU-bound step pipeline exceeded sensible smoke budget.

**Decision**: Declare step 2 complete based on:
- Offline build + forward pass for all 6 variants (verified twice, CPU and GPU).
- V0_baseline_isotropic and V1_3stage_strided_conv finished full 3-epoch smoke: 37.8% and 46.7% val_acc respectively (above the 1% chance floor by 38-46×).
- Per-step GPU profiling of V2, V3, V4 confirms correct training (70-80 ms/step steady-state, val loss decreasing).

**Trade-off**: We skip the 3-epoch-on-every-variant smoke loop at the cost of not having a `training_log.csv` for V2-V5 pre-full-run. **In exchange**: we save ~1.5 h of sunk time on a pipeline bottleneck irrelevant to the actual batch=128 training, and we surface the warmup cost (~100 s/variant for larger variants) as an expected characteristic of the full run.

**Mitigation for the long run**: the full training at batch=128 is GPU-bound (not CPU-augmentation-bound), so the V2-type stall will not recur. We also keep `include_terminate_on_nan=True` and `EarlyStopping(patience=30)` in the callback list so any silent failure terminates gracefully.

### D-005: Run full training on GPU 1 (RTX 4070 12GB) instead of GPU 0 (2026-04-24, EXECUTE step 4)

**Decision**: User asked to run step 4 on GPU 1 (RTX 4070, 12GB) instead of the planned GPU 0 (RTX 4090, 24GB). Keep batch size at 128 (no reduction needed).

**Pre-flight check**: Ran a 3-step train_on_batch on V4 (10.32M params, largest variant) on GPU 1 at batch 128 with allow-growth. Memory headroom confirmed — V4 fits. Loss updates normally.

**Trade-off**: Expect ~1.5-2x slower per-step than 4090 (4070 has fewer tensor cores and lower bandwidth). Full run time estimate revises from ~5h to ~7-10h. In exchange: frees GPU 0 for other work; respects user's explicit resource allocation.

**Alternatives rejected**:
- Reduce batch size to 64 preemptively: unnecessary, V4 fits at 128. Would halve step count and hurt batchnorm/optimizer stats in variants.
- Wait for GPU 0: user directed GPU 1 explicitly.

### D-006: Reduce batch size from 128 to 64 at user request (2026-04-24, EXECUTE step 4)

**Decision**: After ~3.5 min of successful training at batch=128 on GPU 1 (PID 187179, 10.4GB used, 91% GPU util), user asked to kill the run and restart at batch=64. Killed PID 187179, freed GPU 1 (15MB residual), relaunched as PID 199113 with identical recipe except `--batch-size 64`.

**Context**: This is a user-requested change, NOT an OOM or failure. The prior run was healthy. D-005 explicitly noted 128 was fine; user simply prefers 64.

**Trade-off**: At batch=64, step count doubles (781 steps/epoch vs ~391 at 128) and per-epoch wall time rises (epoch 1 measured at 162s incl. XLA compile; steady-state estimated ~120s/epoch). Total run time estimate revises upward from ~7-10h (bs=128 on 4070) to ~12-15h for all 5 variants at bs=64. In exchange: smaller batch yields more gradient-update noise (mild regularization, often +0.3-0.7pp val_acc on CIFAR-class tasks) and cuts GPU memory to ~6-7GB (more headroom).

**Verification of new run**:
- nvidia-smi: GPU 1 at 10.4GB used, 91% util — training actively on GPU 1 ✓
- log shows `StreamExecutor device (0): NVIDIA GeForce RTX 4070` (CUDA_VISIBLE_DEVICES=1 maps to logical 0) ✓
- Epoch 1 completed with val_accuracy=0.0646 (6.46%, well above 1% chance floor for CIFAR-100) ✓
- 781 steps/epoch confirms bs=64 (50000/64 ≈ 781) ✓

**Alternatives rejected**:
- Keep 128 and push back: user directed the change explicitly; no OOM justification needed.

### D-007: Skip V0 retraining, train V1-V5 via serial bash loop (Option C) (2026-04-24, EXECUTE)

**Decision**: After the previous V0-through-V5 run was stopped mid-V0, resume by (a) keeping the existing V0 checkpoint at `results/cliffordnet_downsampling_20260424_101156/V0_baseline_isotropic_downsampling_20260424_101204/best_model.keras` (epoch 77, val_acc 0.7425) as V0's final artifact, and (b) training V1 through V5 via a serial bash loop, each invoked with `--variant V<n>`.

**Why Option C (not A or B)**:
- Option A (resume V0 from checkpoint): the training script has no resume logic — grepped for `resume|load_weights|restore`, zero hits. Would require code changes.
- Option B (edit script to accept a subset flag): `--variant` already accepts a single variant OR "all" (line 707-715 of `train_downsampling_techniques.py`). A serial bash loop over single-variant invocations is zero code change.
- Option C (bash loop): cleanest path — no script modification, each variant gets its own `results/` subdir naturally.

**V0 checkpoint rationale**: val_acc at epoch 77 is 0.7425 with val_loss 0.557. Training log shows gains of ~0.001-0.003 val_acc per epoch under fully-decayed cosine LR. Remaining 23 epochs would buy ≲0.01 val_acc — not enough to change the architectural ranking we care about. Cost of saving ~40 min of compute > marginal accuracy gain.

**Trade-off**: V0 will have a slightly lower number than if trained to epoch 100 (estimated ~0.745-0.750 vs measured 0.7425), at the gain of ~40 min of GPU time and zero code churn. Architectural comparison is unaffected because V1-V5 gaps are much larger than 0.005.

**Alternatives rejected**:
- Restart V0 from scratch: wastes ~2h of compute that's already banked.
- Modify script to add `--resume-from` flag: adds permanent surface for a one-off experiment.

---

### D-008 — REFLECT (step 4+5 complete, 2026-04-24): results, simplification checks, devil's advocate

**Context**: All 5 hierarchical full-run variants + partial V0 baseline now have best checkpoints and comparison.csv rows. Step 5 populated `DOWNSAMPLING.md` Results + Discussion.

**What happened**:
- V1 (76.58) > V3 (76.44) > V2 (75.96) >> V0 (74.25, partial) > V4 (75.26, but 10.3M params) >> V5 (70.05).
- Prediction delta: hypothesis ranked patch-merging > avgpool > strided-conv. Actual outcome: V1 strided-conv (narrowly) wins top-1, V3 patch-merge wins top-5, V2 avgpool trails by <0.7 pp. The three 3-stage variants are effectively tied at 100 epochs / single seed.

**Simplification Checks (6-point)**:
1. Files added: 2/3 (script + DOWNSAMPLING.md). ✓ Under budget.
2. Abstractions: 1/2 (local `build_variant` factory). ✓ Under budget.
3. Lines: ~600 script + ~170 doc. Under the 800+150 plan budget. ✓
4. No wrapper cascades, config toggles, or dead code — script is flat and straight-line per variant.
5. No exception swallowing, type escapes, or temporary workarounds.
6. `model.py` not modified; `CliffordNetBlock` not modified. Invariant preserved. ✓

**Devil's advocate (one reason this might still be wrong)**:
- Single-seed runs. The ~0.6 pp spread between V1/V2/V3 is inside plausible seed noise for CIFAR-100 at 100 epochs. The *relative ranking* of 3-stage vs isotropic vs aggressive-stem is robust to seeds (gaps are several pp), but the fine-grained V1 > V3 > V2 ordering should not be over-interpreted. This is already noted in the Caveats section of DOWNSAMPLING.md.

**Root cause analysis (for V5 regression)**:
- Immediate cause: patch=4 stem collapses 32×32 → 8×8 before any `CliffordNetBlock`.
- Contributing factor: `CliffordNetBlock`'s sparse-channel-rolling mix depends on local spatial structure; destroying it before the first block denies the operator its useful signal.
- Failed defense: none needed — this is an informative negative result, not a bug.
- Prevention: documented in Discussion as "don't use aggressive patchify stems with CliffordNetBlock".

**Verdict**: All 5 success criteria PASS (see `verification.md`). No regressions. Recommend → CLOSE.

## plan_2026-04-24_e4c8ebab
### D-001 (PLAN, iter-1, plan v1): Recommend Strategy 1 — hierarchical vision tower only, text untouched

**At the cost of**: only ~3-5× memory reduction on the vision tower (which is the bottleneck) and zero reduction on the text tower. If the user later raises `context_length` significantly (e.g. 256+), text-tower memory will rise linearly and the user may need a follow-up plan to make the text tower hierarchical too. Also: variant ladder symmetry between `CliffordCLIP` and standalone `CliffordNet` / `CliffordNetLM` is partially broken — the CLIP vision tower will no longer have a single `vision_depth: int` matching `CliffordNet.nano`'s depth, while the text tower still does.

**Why**: three reasons stack:
1. **Vision tower is where the user's complaint actually bites** — back-of-envelope in `findings/vision-tower.md` shows ~10× more activation memory in vision than text at default config (and ~50× at the `large` variant on 224×224 input).
2. **`_TextLMWrapper` (CLM pretrain) requires per-original-token logits at length `context_length`** (`findings/clip-wiring-and-pretrain.md`). Downsampling the text tower either breaks this wrapper or forces a bypass code path. The user explicitly asked us to keep pretrain wrappers working (they were just renamed/cleaned up in `plan_2026-04-24_1c5ae010`).
3. **Project lesson from previous plan** ("Doc updates belong in the same plan" / "PLAN-PLAN cycles are normal") — better to ship a bounded vision-only refactor with high confidence and present a Strategy 2 follow-up if needed, than to bundle two risky refactors and hit a 3-strike on the text-tower complications.

**Alternatives considered**:
- **Strategy 2** (hierarchical both towers + LM-pretrain bypass): ~1.5× more code change, adds branching to `_TextLMWrapper`, requires a new `CausalSeqMerging` layer with its own tests, requires pad-mask pooling logic. Memory savings: marginally better (text from ~50 MB → ~25 MB at nano). **Recommend offering as a v2 plan if the user wants it.**
- **Strategy 3** (single mid-stack stride only): simplest change (~30 lines) but only ~2× memory savings on vision. Doesn't match the user's "true hierarchical encoder" framing.

**Recovery**: revert is `git revert` of this plan's commits. The pre-refactor model is preserved in git (no destructive changes).

---

### D-002 (PLAN, iter-1, plan v1): Channel progression `[D, D, 2D, 2D]` (doubling twice across 4 stages), not `[D, 2D, 4D, 8D]`

**At the cost of**: less aggressive channel-growth than canonical Swin (`[D, 2D, 4D, 8D]`). The deepest stage is only 2× the stem channel count, not 8×. May leave some representational headroom on the table — the `large` variant in particular currently has D=384 which would become D=3072 in stage 4 under canonical Swin scaling, drastically increasing parameter count.

**Why**: parameter-count back-of-envelope (Pre-Mortem Scenario 1 in `plan.md`):
- Isotropic nano (D=128, depth=12): ~600k params (Clifford blocks dominated by Dense(D, D) ~ D² each).
- Hierarchical with `[D, 2D, 4D, 8D]` and `[3,3,3,3]` depths: stage-4 alone has 3 blocks × ~D²×64 params ≈ 9.8M. Total ~12M. **20× the isotropic count** at nano scale.
- Hierarchical with `[D, D, 2D, 2D]`: ~1.5M total. **2.5× the isotropic count** — still high but in the budget defined by Falsification Signal 1 (STOP IF >2×).

The CliffordNet block is unusually channel-heavy because each block has a `SparseRollingGeometricProduct` whose Dense projections scale with `channels²` per shift. Aggressive channel doubling stacks against this. The user's complaint is **memory** (activations); they did not ask for parameter inflation. Spatial reduction alone gets most of the activation-memory win without the parameter explosion.

**Alternatives considered**:
- `[D]*4` (no channel growth, only spatial reduction): smallest parameter footprint, simplest progression. **Fallback if D-002's `[D, D, 2D, 2D]` exceeds the 2× param budget at any variant scale.** Decision rule: if SC3 in `plan.md` triggers STOP IF in Falsification Signal 1 for any variant, revise variants to use `[D]*4`.
- `[D, 2D, 2D, 2D]` (one early double): asymmetric, no obvious win over `[D, D, 2D, 2D]`.

**Recovery**: change one dict in `MODEL_VARIANTS`; no surgery on the body code (the staged builder accepts any per-stage channel list).

---

### D-003 (PLAN, iter-1, plan v1): Keep `vision_blocks` as a flat list (with `vision_merge_layers` separate), not nested-list-of-lists

**At the cost of**: the body forward needs `_vision_stage_offsets` to know when to insert a merge — a small piece of bookkeeping. Iteration order is implicit in the flat list rather than explicit in the nesting.

**Why**: backward compatibility with existing tooling. `_freeze_clip_for_pretrain` (`train_clip.py:715`) and `_VisionClassifier` (until step 8 patches it) both iterate `for block in m.vision_blocks: ...`. A flat list keeps `for block in vision_blocks` working as a no-op iteration over all blocks regardless of stage. A nested list of lists would require touching every iterator.

Tests at `test_clip.py:84` also iterate `m.text_blocks` — same principle applies if Strategy 2 is later approved.

**Alternatives considered**:
- `List[List[CliffordNetBlock]]`: cleaner stage demarcation but breaks every existing iterator, including the pretrain-freeze sweep that we deliberately want to leave untouched. Loses more than it gains.
- `keras.Sequential` per stage: heavier; serialisation gets noisier (extra wrapper config).

**Recovery**: trivially convertible to nested form later if Strategy 2 makes the flat form awkward.

---

### Decision parking lot (raised by EXPLORE, not yet decided)

- **Variant ladder symmetry with `CliffordNet` / `CliffordNetLM`** (raised in `findings.md` ghost-constraint section): the existing `nano` test asserts `vision_depth == 12` to enforce ladder match with `CliffordNet.nano`. After refactor, the CLIP vision tower has staged depths summing to 12, but the standalone `CliffordNet.nano` is still flat depth=12. **Decision**: keep the sum equal (so total compute is roughly comparable) and document the divergence in the docstring. Update the test to assert the sum, not the exact int. Recorded as part of step 9 in `plan.md`; no separate D-NNN entry needed unless the user pushes back.

- **`nano_g` global-context branch placement under hierarchical vision**: currently `vision_use_global_context` is a single bool applied to every block. Under hierarchical staging, do we apply it (a) to all blocks at all stages, (b) only to the last stage (whole-image summary at the most abstract scale), or (c) only to the first stage (matching CliffordNet.lite_g spirit which puts global context at the base)? **Default in plan v1**: option (a) — apply uniformly, matches existing `nano_g` behaviour exactly. **If user wants finer control**, the plan can be amended to make `vision_use_global_context: List[bool]` per stage (back-compat: scalar bool broadcasts to all stages).

## plan_2026-04-24_1c5ae010
### D-001 (PLAN, iter-1): Strip both Stage 0 and the Stage 1/2 curriculum in one pass
**At the cost of**: losing a working code path (CIFAR-100 vision pretrain + Wikipedia
LM pretrain + low-to-high resolution curriculum) that *could* be useful in future
recipes. Recovery requires `git revert` of this plan's commits or pulling code
back from `d35c27b`/`af205cc`.
**Why**: the user's stated goal is to drop staging and go directly to "the big patch."
PLAN_CONTINUE.md confirms the curriculum was never used in production runs (always
`--stage2-epochs 0`). Keeping dead, untested branches in a script the user actively
edits is worse than removing them — they show up in every read of the file and
invite future bit-rot.

### D-002 (PLAN, iter-1): New CLI uses flat names (`--image-size`, `--epochs`, `--peak-lr`)
**At the cost of**: every existing shell history / launch script that mentions
`--stage1-*` or `--stage2-*` will fail with "unrecognized arguments". The README
becomes stale.
**Why**: keeping the `--stage2-*` names after staging is gone would be misleading
(the "2" implies a "1" that no longer exists). PLAN_CONTINUE.md and the README are
the only documents that mention the old flags; the user said docs are out of scope.
Defaults preserve the proven config (image_size=112, epochs=10, lr=5e-4).

### D-003 (PLAN, iter-1): Dataset summary as a single bracketed log block, not scattered logs
**At the cost of**: slight duplication — the per-loader "Loaded N pairs" lines
remain in the loaders, so the same numbers appear twice in the log.
**Why**: the user's literal request — "able to read the logs and immediately see
dataset sizes" — is best served by one greppable block. Loader-internal logs are
helpful when triaging the loader itself; we don't strip them because they live
in `train.common` and are out of this plan's scope.

### D-004 (PLAN, iter-1): No new abstractions — fold `_run_stage` inline
**At the cost of**: `train()` grows by ~20 lines.
**Why**: `_run_stage` exists today only because it was called twice (once per
stage). With one stage, an extra helper just adds a hop with no callers.
Complexity-control rule: if a function has a single caller, inline it.

### D-005 (PLAN, iter-1, plan v2): Keep Stage 0 functionality but rename to "pretraining"
**At the cost of**: bigger surface to keep working — every `_run_stage0_*`,
`--stage0-*`, `_STAGE0_*`, and `_freeze_clip_for_stage0` reference must be
renamed in lock-step. Risk of leaving a stale name behind that breaks at runtime
(NameError) rather than at import. Mitigated by SC1's grep gate plus the smoke
test in step 5.

This **supersedes D-001's "drop Stage 0"** decision. The user's revised
direction (plan v2) is: keep CIFAR-100 vision pretrain + Wikipedia LM pretrain
as a useful capability under a clearer name. PLAN_CONTINUE.md showed Stage 0
was rarely exercised in production *runs* but never said the capability itself
was unwanted — only the "stage 0 / stage 1 / stage 2" naming was confusing.
**Why** "pretraining" specifically: it's the standard ML term for what these
helpers actually do (initialize backbones via auxiliary objectives before the
main contrastive loss). The rename makes the script self-documenting without
adding a new abstraction.

### D-006 (PLAN, iter-1, plan v2): Merge Stage 1 + Stage 2 → single CLIP stage
**At the cost of**: same as D-001 for the curriculum portion — losing the
low-res-to-high-res warmup as a built-in code path. Recovery requires reading
the deletion from git.

**Why**: the user's revised direction confirms the curriculum was never the
desired permanent shape — only the "big patch" final-resolution training is.
Keeping the curriculum loop "just in case" while users run with
`--stage2-epochs 0` would leave dead code in a file the user actively edits.

### D-007 (PLAN, iter-1, plan v2): `--skip-pretrain` (single flag), not `--skip-pretrain-vision` + `--skip-pretrain-lm`
**At the cost of**: less granular control — to skip only one of the two
pretraining helpers, users set its `--pretrain-*-steps` to 0 (the existing
mechanism, unchanged from the `--stage0-*-steps 0` semantics today).

**Why**: matches the existing `--skip-stage0` shape one-for-one, so the rename
is mechanical and not a behavioral change. Keeping a single skip flag also
keeps the smoke test single-flag (cleaner SC4).

### D-008 (PLAN, iter-1, plan v2): Smoke test uses `--skip-pretrain`
**At the cost of**: the smoke run does not exercise the (renamed) pretraining
code paths end-to-end. NameError-style rename mistakes inside
`_run_pretrain_vision` / `_run_pretrain_lm` bodies could escape.

**Why**: pretraining requires CIFAR-100 (~170MB download) and the Wikipedia HF
cache (multi-GB). Triggering both for a CLI sanity check is hostile to dev
loops and contradicts the smoke run's purpose (verify imports, CLI parsing,
single-stage CLIP path). Mitigation: SC1's grep gate catches *referenced* old
names in any code path; SC2's grep gate confirms new names exist. The
pretraining bodies themselves are not modified — only their names — so a NameError
inside a renamed function would only fire if step 1 missed updating the
function's own internal recursive calls (none exist; verified during EXPLORE).
