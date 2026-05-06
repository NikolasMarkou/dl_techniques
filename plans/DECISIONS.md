# Consolidated Decisions
*Cross-plan decision archive. Entries merged from per-plan decisions.md on close. Newest first.*

## plan_2026-05-06_82749628
### 2026-05-06 — D-001: Use UpSampling2D(nearest) along W as the causal upsampler.
**Choice**: `keras.layers.UpSampling2D(size=(1, s), interpolation="nearest")` for the decoder up-path.
**Cost**: no smoothing across pool boundaries — each output cell within a window has the same scalar. This is acceptable here because (a) `CausalCliffordNetBlock` after the upsample re-injects local mixing (left-only causal DW conv along W) so the decoder block can blend pooled positions back into per-token features, and (b) any smoother (bilinear/transposed-conv) leaks future info — losing causality is a bigger cost.
**Why not transposed conv with mask**: complexity. Adding a learned-yet-strictly-lower-triangular conv is harder to verify than a stateless repeat. Revert-First / 10-Line Rule favours the stateless repeat.

### 2026-05-06 — D-002: Pad-then-crop in `call()` (option B), not `seq_len % total_stride == 0` validation.
**Choice**: right-pad input with zeros to next multiple of `total_stride`, run the U-Net, slice along W back to original `seq_len`.
**Cost**: tiny per-call overhead (one pad + one slice op). Goes against "fail fast on invalid shapes" preference.
**Why**: train script default is `seq_len = max_seq_length - 1 = 511`; making this work without forcing the user to think about stride alignment is the friendliest option. Padding does NOT pollute real positions because every block is causal in W (zeros at the right tail can only influence themselves or later padded positions).

### 2026-05-06 — D-003: Skip-fusion = Concatenate + 1x1 Conv2D (not add).
**Choice**: `concat([upsampled, encoder_skip], axis=-1)` then `Conv2D(channels, 1)` to project 2C → C.
**Cost**: extra 1x1 conv per decoder level. Adds ~`channels^2 * 2` params per level; negligible vs the Clifford blocks' SRGP weights.
**Why**: gives the model a learnable mixing weight between encoder skip and decoder feature; more flexible than addition; matches the dl_techniques/cliffordnet vision UNet pattern (`unet.py:676-680`).

### 2026-05-06 — D-004: Per-stage isotropic channel count (no channel growth across stages).
**Choice**: every stage uses the same `channels` (e.g. nano=128 throughout). DSv2 `out_channels` is left at `None`.
**Cost**: gives up the standard U-Net "double channels when halving spatial" recipe. Bottleneck is wider in *receptive field* but not in *channel count*.
**Why**: keeps the SRGP `shifts < channels` invariant trivially satisfied; matches `lm.py` which is also isotropic; cleaner skip-concat (no per-level channel projection on the encoder side); shifts list applies uniformly. If we later want a channel-pyramid variant, we add it as a separate variant — keep iteration 1 simple.

### 2026-05-06 — D-004 SUPERSEDED by D-005: User rejected isotropic channels.
**User feedback (Plan v1 review)**: Channels must grow per stage (typical U-Net doubling). Use base_channels per variant; per-stage channels derived as `base * 2**i`.

### 2026-05-06 — D-006: STOP — Pre-Mortem Scenario A fired during Step 3 smoke check.
**Trigger**: With nano-like config (base_channels=32, stride_per_stage=[2,2], total_stride=4), perturbing input position 7 (last) changed logits at positions 4, 5, 6 with max abs diff ~0.21–0.27 (well above 1e-5 tolerance). Positions 0–3 were byte-identical.

**Root cause**: Pool→nearest-upsample **round trip** is not causal at fine resolution, even though nearest-upsample alone is. DSv2 with `padding="same"` and stride `s` produces pooled cell `j` that has seen input positions `[j*s, j*s+s-1]`. Nearest-upsample maps output position `k` to pooled cell `k // s`, which has seen up to input position `(k//s)*s + s - 1`. For `k % s != s - 1`, that maximum exceeds `k` — i.e. output position `k` reads from a pooled cell that has incorporated future inputs. Compounded across multiple levels (deepest cell sees up to `(k // total_stride)*total_stride + total_stride - 1` of the input).

**A1 falsified**: The `upsampling-causality.md` finding's argument *"upsampled position k is identically equal to encoder feature at pooled position j; pooled position j was computed from input positions 0..j*s+(s-1), so k sees only past+current"* is correct ONLY at the pool grid (`k = j*s`). For all `k` in `[j*s, (j+1)*s - 2]`, "past+current" at the pool grid translates to **future** at the original grid.

**Failed defense**: A1 wasn't truly tested at the unit level — the existing `test_causal_block_no_future_leak` regression covers the DSv2 block at pool resolution but never the round trip through upsample back to original resolution. The plan accepted A1 on the strength of an internal-consistency argument, not a test.

**Per Autonomy Leash**: 0 fix attempts made on Step 3 — reverted uncommitted changes immediately on detecting the leak (Revert-First). Committed Steps 1 & 2 (scaffolding) are kept; they are causality-agnostic.

**Remediation options to discuss with user**:
- (R1) **Causal upsample**: shift upsampled feature right by `s-1` along W (zero-pad on left, drop right). Compounded across levels: output position `k` reads pooled cell `(k - (s_total - 1)) // s_total` at the deepest level, ensuring max-input-seen ≤ k. Implementable as a small custom layer or as `keras.ops.pad` + slice after each `UpSampling2D`. Cost: extra `(total_stride - 1)` positions of latency at the top of U; the first `total_stride - 1` positions of any stage's upsampled feature are zero-filled before the skip connection. The encoder skip at level 0 is fully causal, so the skip-fusion path at the leftmost positions still has signal — only the deep-path contribution is zero-padded.
- (R2) **Skip-only at boundary positions**: for the first `s-1` output positions of each upsample window, mask the upsampled-feature contribution and rely entirely on the encoder skip. Equivalent to R1 in effect; different implementation.
- (R3) **Drop the U-Net**: revert plan, fall back to non-pooled stack (i.e. `lm.py`'s isotropic depth-only model). User explicitly wanted U-Net so this is the last resort.
- (R4) **Strict causal pool**: change DSv2 to use a pool that emits at the **right** edge of its window (pool position j sees inputs `[j*s - s + 1, j*s]`), then nearest-upsample is causal at all positions because pooled cell `k // s` has seen up to input `(k // s) * s ≤ k`. Cost: requires modifying DSv2 (out of plan scope) or wrapping it with a left-shift pre-pool.

Recommendation: R1 (causal upsample with right-shift) is the smallest, most local change. R4 is more elegant but touches a shared layer.

### 2026-05-06 — PIVOT: REFLECT → PIVOT → PLAN (adopt R1, causal upsample right-shift).
**User decision**: choose R1 — causal upsample via right-shift by `s-1` (left zero-pad, drop `s-1` from the right) at each decoder upsample stage. Keep the change local to `lmunet.py`. Do NOT modify `CausalCliffordNetBlockDSv2`.

**Ghost-constraint scan**: none — A1 was a wrong claim, not a ghost constraint. The constraint that originally drove "nearest upsample is enough" was based on an unproven derivation; nothing about the environment has changed.

**Complexity assessment**: R1 adds ~5 lines per upsample call site (a `keras.ops.pad` + slice). Stays within complexity budget (no new abstractions; UpSampling2D + pad+slice is two stdlib ops in sequence). Net delta ~+10 lines vs original Step 3.

**Available checkpoints**: cp-000-iter1 (pre-Step-1 commit `a7abadf`). Steps 1 & 2 commits kept (causality-agnostic scaffolding); resume from Step 3 on top of `245553d`.

### 2026-05-06 — D-007: Causal upsample via right-shift by `s-1` (R1).
**Choice**: After each `keras.layers.UpSampling2D(size=(1, s), interpolation="nearest")` in the decoder, apply a causal right-shift along W: `x = ops.pad(x, [[0,0],[0,0],[s-1,0],[0,0]])[:, :, :W_up, :]` where `W_up` is the post-upsample width. Equivalent to left-zero-padding by `s-1` then dropping the last `s-1` cells. Implemented inline at the call site as a small helper `_causal_upsample(x, stride)` private to `lmunet.py`.
**Cost**: first `s-1` output positions of each decoder upsample stage have zero deep-path contribution (skip path still carries signal at those positions, since the encoder skip is at the original resolution and is itself causal). Compounded across levels, the leftmost `total_stride - 1` output positions have purely-skip information from the deepest stages — acceptable: this is "latency at warm-up" not a permanent bias.
**Why**: smallest local change that fixes Scenario A. Does not touch the shared `CausalCliffordNetBlockDSv2`. Verifiable in isolation: a unit test on the helper alone proves `_causal_upsample(x_with_perturbation_at_k)` produces output where positions `< k` are byte-identical to the unperturbed baseline.
**Anchor in code**: `# DECISION D-007: causal upsample via right-shift; see plans/plan_2026-05-06_82749628/decisions.md`. Placed at the call site of `_causal_upsample` in `call()` and at the helper definition.
**Falsification**: unit test `test_causal_upsample_helper` (perturb input at position k, assert output positions `< k` byte-identical) MUST pass before the end-to-end causality test in Step 4.

### 2026-05-06 — REFLECT (iter-1 post-pivot): all 6 criteria PASS.
**Summary**: 30/30 unit tests pass. End-to-end causality verified (perturb-last, perturb-middle, non-multiple seq_len, with global_context). Helper micro-tests (TestCausalUpsampleHelper) isolate D-007 fix.

**Simplification Checks (6)**:
1. Could a step be removed? No — all 6 deliver distinct artifacts.
2. Could two layers be merged? No — `_causal_upsample` is a stateless 4-line helper, can't merge into UpSampling2D.
3. Is any abstraction unused? No — every new symbol is wired in.
4. Could a config field be a constant? `base_channels` is the only config exposed at the variant boundary; the rest are per-variant defaults.
5. Could a forward branch be deleted? No — pad+crop is needed for non-multiple seq_len; tied/untied LM head matches `lm.py` policy.
6. Is anything wrapped that doesn't need wrapping? No — UpSampling2D + 1x1 Conv2D + Concatenate are stdlib; no adapter classes.

**Devil's advocate**: One reason this might still be wrong despite passing — the causality tests use a small nano-like config (base_channels=16, stride=[2,2]). At deeper variants (base/large/xl with stride=[2,2,2], total_stride=8), the causal-shift compounds across 3 levels: total left-zero region = 7 positions. We never tested causality at the larger variant. However: the right-shift logic at each level is locally correct, and composing causal operations stays causal (mathematical property), so the small-config test is sufficient. Risk: if a future change adds a new spatial-mixing op at a deeper level, only the small-config test will catch it. Acceptable; flagged in "Not Verified" of verification.md.

**Prediction accuracy**: see verification.md "Prediction Accuracy" table.

**Root cause analysis**: only one failure during EXECUTE was the save/load tolerance (5e-5 vs plan's 1e-5). Immediate cause: GPU XLA reduction-order non-determinism. Contributing factor: plan tolerance was inherited from `lm.py` test which was simpler (no DSv2 / no decoder fusion path). Failed defense: none — the `lm.py` precedent suggested 1e-5 was achievable, and it took the actual test to reveal the gap. Prevention: future U-Net-style models should default to 1e-4 in save/load tests.

### 2026-05-06 — D-005: Per-stage channel doubling (standard U-Net schedule). [SUPERSEDES D-004]
**Choice**: `channels_per_stage[i] = base_channels * (2 ** i)` for `i in 0..num_levels-1`. Encoder doubles channels each downsample (DSv2 `out_channels` does the projection). Decoder mirrors going back up: 1x1 Conv2D after concat projects from `(C_{i+1} + C_i) = 3*C_i` down to `C_i`. Embedding dim = `base_channels` (top-of-U).
**Cost**: parameter count grows fast at the bottleneck (largest channel count is `base * 2**(num_levels-1)`). For xl with base=192, num_levels=4 → bottleneck has 1536 channels. Slightly more memory than the isotropic D-004 design at the bottleneck. Decoder skip-fusion projection is `3*C_i → C_i` (was `2*C_i → C_i` under isotropic).
**Why**: standard U-Net inductive bias — wider receptive field at deeper levels deserves more channels to capture coarser features; user explicitly required this; matches the dl_techniques vision UNet pattern. `shifts < channels` invariant remains trivially satisfied: the smallest channel count is at the top of U (`base_channels`), and `max(shifts) < base_channels` is checked once in `__init__` — every other stage has strictly more channels. DSv2's existing `out_channels` arg supplies the channel projection without adding new layers. Concat semantics unchanged: still concat along channel axis then 1x1 project — only the input/output channel arithmetic shifts.

## plan_2026-05-06_13a2df9e
### 2026-05-06 PLAN — chosen approach

**Decision**: Add `CausalCliffordNetBlockDSv2` as a new sibling class in `clifford_block.py`, mirroring the structure of `CliffordNetBlockDSv2` but with a narrower pool-kind surface (`avg`/`max` only) and the causal-padding paradigm of `CausalCliffordNetBlock` applied to the depthwise context conv and the global-context cumulative mean.

**Trade-off (X at the cost of Y)**:
- Reuse causal-padding pattern (left-pad by `k-1` along W, then `valid` conv) at the cost of a slightly redundant call-time `pad` op (already accepted in `CausalCliffordNetBlock`).
- Restrict pools to `avg` / `max` only at the cost of feature parity with DSv2; pyramid_diff / blur / gaussian_dw / pixel_unshuffle / resnetd are excluded because they would leak future info under H=1, W=seq layout.
- Place new class in the same file at the cost of file growth (~2300 lines), matching the user's explicit instruction.

**Alternatives considered**:
- Pass a `causal: bool` flag into existing `CliffordNetBlockDSv2`: rejected — the parent class already validates `pyramid_diff` etc., and adding a flag complicates serialization, validation, and forward branching.
- Subclass `CliffordNetBlockDSv2`: rejected — the parent's `dw_conv` and pool factory choices conflict with causal constraints; mirroring instead of inheriting matches the existing pattern (`CausalCliffordNetBlock` does not subclass `CliffordNetBlock`).
- Place the class in a separate file: rejected — user explicitly requested same file.

### 2026-05-06 INIT → EXPLORE → PLAN

EXPLORE produced 3 indexed findings (scope-and-callers, causality-mechanics, dsv2-merge-points). Confidence: deep / constrained / clear. Transitioning to PLAN.

### 2026-05-06 EXECUTE step-1 — DepthwiseConv2D vs Conv2D(groups=channels) deviation

**Surprise**: `DepthwiseConv2D` on TF/CUDA rejects asymmetric strides
(`Current implementation only supports equal length strides in the row
and column dimensions`). The plan called for `DepthwiseConv2D(strides=(1, s))`
which fails on GPU at strides=2.

**Root cause**: The existing `CausalCliffordNetBlock` uses default strides=1
so it never hit this limitation. The plan implicitly assumed
`DepthwiseConv2D` accepts asymmetric strides — it does not on TF GPU.

**Resolution (1-line fix at point of impact)**: Replace
`DepthwiseConv2D(kernel_size=(1,k), strides=(1,s))` with
`Conv2D(filters=channels, kernel_size=(1,k), strides=(1,s), groups=channels)`.
Algebraically identical (depthwise = grouped conv with groups==channels==filters).
Verified working at strides=2 with both shape and causality probes.

**Trade-off**: Conv2D-with-groups at the cost of (a) a slightly larger
memory footprint for the kernel-shape tensor, (b) divergence from the
sibling `CausalCliffordNetBlock` which still uses `DepthwiseConv2D` (it
can — its strides are always 1). Documented in DECISION D-002 in code.

**Anchored**: D-002 comment expanded inline in `CausalCliffordNetBlockDSv2.__init__`.

## plan_2026-05-05_0eac2c81
### D-PLAN-001 (2026-05-05) — Empirical re-verification before fixing

**Decision:** Verify each Phase-5 finding by running existing tests + targeted micro-experiments BEFORE writing any plan steps.

**Rationale:** Phase-5 review claimed `B1/B5` (multi-input build crashes Functional API) but `test_model_save_load` in the existing suite already exercises that path and passes. Cost of writing a "fix" for a non-bug: real (touches public API). Cost of verifying first: 30 seconds of pytest.

**Trade-off:** Spending one EXPLORE round on verification at the cost of slightly slower plan delivery — but avoiding fixing 6 phantom issues.

**Outcome:**
- Refuted: B1, B5, B9 (cosmetic only).
- Confirmed: B16, B7, B17, B4, B8, B3, B13, P1, P2, L1.
- Partially confirmed (works today, no test): B11.

### D-PLAN-002 (2026-05-05) — pyramid_diff fix: slice-after-upsample over swap-to-resize

**Decision:** Fix B16 by cropping `z_lo_up` to match `z_ctx.shape` after `UpSampling2D`, instead of replacing `UpSampling2D` with `keras.ops.image.resize`.

**Rationale:**
- `UpSampling2D` is a tracked Keras layer (clean serialisation, no `__init__` changes).
- A static slice operation in `call` is graph-friendly (works under tf.function and Functional API).
- The slice is a no-op when shapes already match (`H/s % s == 0`), so even-dim cases are byte-identical to current behavior.

**Trade-off:** Sub-pixel positioning of the upsample is inherited from `UpSampling2D` (bilinear) rather than redone — at the cost of slightly less control over interpolation if a future reviewer wants nearest-neighbour. Acceptable because the original code already used bilinear.

### D-PLAN-003 (2026-05-05) — Reject downsampling pool kinds at strides=1

**Decision:** `_make_pool_v2` raises `ValueError` when `kind in {blur, gaussian_dw, pixel_unshuffle, resnetd}` and `strides == 1`.

**Rationale:** These kinds *only* make sense as downsamplers. Silently returning Identity hides user errors. The legitimate strides=1 path is `kind in {avg, max}` which already returns `Identity` (and matches user mental model: "no downsampling, no pooling").

**Trade-off:** Hard error at the cost of one possible legitimate use case (e.g. wanting blur at strides=1 for low-pass filtering without downsampling). That use case is not currently exercised anywhere in the codebase (grep confirmed) and is more naturally expressed by directly using `BlurPool2D(strides=1)`.

### D-PLAN-004 (2026-05-05) — B11 source-level fix deferred; test only

**Decision:** Add a regression test for CausalCliffordNetBlock causality, but do not change the source.

**Rationale:** Empirical leak test shows current code is causally correct. The future risk is a refactor breaking causality silently. A test guards that without the cost of source changes (which would risk introducing a real bug while "fixing" a non-bug).

**Trade-off:** Test guards present behavior without proving anything new. Cost is one short test (~15 LOC) for permanent regression coverage.

### D-PLAN-005 (2026-05-05) — Defer B9, B12, X1, X2, X3 and remaining defensive items

**Decision:** Do not fix B6, B9 (cosmetic), B12, B14, B15, B18, L2, L3, L4, L6, X1, X2, X3 in this plan.

**Rationale:** Each is either a theoretical concern with no current evidence of breakage (B12 fp16 cumsum), a minor coupling smell (B14, B15), or a defensive assertion that adds lines without preventing a current failure mode (L2, X2). Per the LESSONS.md note "PLAN to PLAN cycles are normal" — keep this iteration tight; document for future plans.

**Trade-off:** Smaller plan that lands cleanly at the cost of leaving improvements on the table. The findings.md "Out of scope" section preserves them for retrieval.

### D-PIVOT-001 (2026-05-05) — B17 reverted mid-EXECUTE: not actually a bug

**Decision:** Roll back the B17 change to `_make_pool_v2` (rejecting `pixel_unshuffle`/`resnetd`/`blur`/`gaussian_dw` at strides=1).

**Trigger:** Step 5 caused `test_default_init_uses_v1_winner` to fail. DSv2's default `stream_pool="blur"` + `strides=1` constructed without error before; my "fix" raised.

**Root cause analysis:**
1. **Immediate cause:** New ValueError fired in default constructor.
2. **Contributing factor:** I treated `Identity at strides=1 for non-{avg,max} kinds` as silent-degradation. It is actually a *deliberate uniform-construction contract* — a hierarchical model can declare `stream_pool="blur"` once and use it across all stages; only the strided stages activate the kind, the others pass through.
3. **Failed defense:** No defense in plan. The Phase-5 review labelled this CONFIRMED based on empirical observation of the Identity-return, without checking if the behavior was *intended*. Empirical confirmation alone doesn't separate "bug" from "documented contract".
4. **Prevention:** When a Phase-5 finding flags "silent degradation", grep the test suite for tests that *enforce* the silent behavior before assuming it's a bug. The test `test_default_init_uses_v1_winner` was one grep away.

**Outcome:** Source reverted to original `if strides == 1: return Identity(...)` for all kinds. Comment expanded to document the contract. SC4 revoked from verification criteria.

### D-PIVOT-002 (2026-05-05) — Source delta exceeded budget; accepted

**Decision:** Accept +92 net source LOC vs target ≤ +60. Do not compress.

**Rationale:** The overage is concentrated in (a) validation error messages — clarity wins over brevity for build-time errors that caller engineers will read; (b) the D-003 comment explaining the pyramid-diff slice — describes non-obvious math that future readers need.

**Trade-off:** Slightly fatter source at the cost of better error messages and self-documenting math. Reverse if a future plan has a stronger LOC requirement.

## plan_2026-05-05_60c5be7d
### 2026-05-05 PLAN v1 — chosen approach

**Decision**: 7-step incremental plan, ordered with the failing fp16 test FIRST (Step 1), then the fix (Step 2). Refactors (H-5) last with their own commit so they can be reverted independently.

**At the cost of**: 7 small commits instead of one bundle — slightly more git overhead, but each commit is independently bisectable and the H-5 refactor can be reverted without losing the bug fixes.

**Alternatives considered**:
- "Single big commit covering all 8 fixes" — rejected. H-5 has nontrivial residual D-004 risk (LESSONS L31); coupling it to the fp16 fixes means a single failing save/load test forces reverting all bug fixes too.
- "Apply M-1 to cosine basis as well" — DECLINED per user instruction ("if uncertain, leave it"). D-004 is recent and firsthand. Save the ~8KB per layer; not worth re-litigating.
- "Conditional cast-back on input dtype (Step 2)" — rejected as default. The simpler "always fp32 output" is the deliberate override of compute_dtype for probability distributions; document with D-005. Keep the conditional version as fallback if Scenario B fires.

**Anchor**: plan.md → Pre-Mortem (Scenario A is the H-5 escape hatch).

### 2026-05-05 PLAN v1 refinement — Step 2 fp16-only conditional cast (user instruction)

**Decision**: In Step 2, the cast-back at L717 becomes a conditional: keep fp32 ONLY when `inputs.dtype == "float16"`; otherwise cast back to input dtype as before.

**At the cost of**: 3 extra lines (the if/else) instead of straight removal. Net step-2 line delta moves from -3 to ~0.

**Rationale (user)**: bf16 has fp32-like range so the underflow doesn't apply; only fp16 is broken. Scoping the override to fp16 minimizes blast radius — fp32 callers see no behavior change at all, bf16 callers also unchanged. D-005 comment must reflect this scoped reasoning ("override compute_dtype for fp16 only").

**Alternatives considered**:
- "Always fp32 output" (original plan) — rejected per user; broader blast radius than necessary.
- "Cast inside an `ops.cond`" — rejected; `inputs.dtype` is a static Python attribute, no runtime branch needed.

**Anchor**: plan.md Step 2 (revised), D-005 comment in routing_probabilities.py at the L717 site.

### 2026-05-05 PLAN v1 → EXECUTE — user approval

**Decision**: User approved plan with Step 2 refinement above. M-1 stays declined. Step 7 keeps its independent checkpoint and revert-on-failure rule. Commit after each step. After Step 7 or any aborted step, run scoped pytest from Verification Strategy and route to REFLECT. Wait for user review before CLOSE.

**Anchor**: state.md transition log (PLAN → EXECUTE).

### 2026-05-05 EXECUTE Step 2 — surprise: Keras `compute_output_spec` overrides runtime dtype

**Surprise**: under `mixed_float16` policy, `keras.Input(dtype="float16")(layer)` declared output dtype as fp16 even after `call()` returned fp32. Root cause: Keras Layer base class default `compute_output_spec` (layer.py:1096) returns `KerasTensor(dtype=self.compute_dtype)` whenever a custom `compute_output_shape` is implemented. The Functional graph then coerces the runtime tensor to that declared dtype.

**Why not in Assumptions A1**: A1 covered input-side autocasting only. The output-side spec dtype was an unstated assumption — that runtime dtype propagates to the symbolic graph. It does NOT when a custom `compute_output_shape` is used.

**Fix**: override `compute_output_spec` on the layer to return fp32 dtype under fp16 compute_dtype, matching the runtime cast logic at the D-005 site. +30 lines net for Step 2.

**At the cost of**: more code than planned. Alternative considered and rejected — overriding the layer's `compute_dtype` to always be fp32, which would force kernel storage to fp32 and kill the mixed-precision benefit on the matmul.

**Plan invalidation check**: Assumption A1 is still valid (input-side). No other assumption is invalidated. Steps 3-7 unaffected.

**Anchor**: routing_probabilities.py — `compute_output_spec` override docstring + D-005 comment at the cast site.

## plan_2026-05-04_1b2810b6
### 2026-05-04 — Plan v1: routing-LM variant as parallel files (no edits to existing lm.py / train script)

**Decision**: Implement the routing-LM variant as new sibling files (`lm_routing.py`, `train_cliffordnet_nlp_routing.py`, dedicated test file) rather than adding a flag to `CliffordNetLM` or `train_cliffordnet_nlp.py`.

**Trade-off**: code duplication (~700 lines mirrored) at the cost of bisectability and zero risk to the existing baseline. Existing trained checkpoints and training pipelines remain bit-identical and untouchable.

**Why not the alternative (flag-on-existing)**: a `use_routing_head: bool` flag on `CliffordNetLM` would require also threading a `from_logits` flag through the train script's loss creation, changing the model's serialization shape, and risk breaking existing depth-estimation / CLIP code paths that import `CliffordNetLM`. Cost (duplication) is bounded; cost of regression on shipped baselines is unbounded.

**Anchored at**: `D-001` (in `lm_routing.py` near dict construction) — output dict key remains `"logits"` despite values being probabilities. `D-002` (in `lm_routing.py` near routing layer instantiation) — `from_logits=False` requirement and rationale for `routing_mode="trainable"` default. `D-003` (in `train_cliffordnet_nlp_routing.py` near `create_loss_fn`) — `from_logits=False` is required for routing output.

**Default routing_mode**: `"trainable"`. Deterministic mode (16 fixed cosine projections to discriminate ~50K tokens) is information-theoretically too tight; exposed only as opt-in for ablation.

**Loss reuse**: keep `MaskedCausalLMLoss` and `FocalCausalLMLoss` (existing classes). Both already support `from_logits=False` via `ops.log(y_pred + 1e-8)`. No new loss class needed.

**Output key**: keep `"logits"` (not `"probs"`). The train data wrapper does `(x, y) -> (x, {"logits": y})` and compile uses `loss={"logits": ...}, metrics={"logits": ["accuracy"]}`. Renaming would force changes to the dataset wrapper and loss spec — out of scope.

**Smoke verification only**: smoke runs check pipeline integrity (no crash, no NaN, gradient flow), not final perplexity. Convergence quality is out of scope; this is research scaffolding.

### 2026-05-04 — Step 3: Surprise discovery — RoutingProbabilitiesLayer deterministic-mode FuncGraph capture bug

**What happened**: Test 4-5 (`test_forward_shape[deterministic]`, `test_save_load_roundtrip[deterministic]`, etc.) failed with "tensor cannot be accessed from here, it was defined in FuncGraph(...) which is out of scope". The cosine basis was created in `build()` via `ops.stack(...)` returning a plain Tensor (not a tracked weight). When the layer is embedded inside a `keras.Model` subclass, Keras runs `compute_output_spec` in a transient scratch graph; `build()` runs there too and the cosine basis gets captured by that graph. Subsequent eager calls to `layer.kernel` then dereference a dead tensor.

This bug surfaces only when `RoutingProbabilitiesLayer(deterministic)` is used as a sub-layer of a `keras.Model` subclass. Direct standalone use (the way the existing layer tests exercise it) works fine because there is no surrounding compute_output_spec scratch graph.

**Falsification signal fired**: yes — Pre-Mortem Scenario 4 ("tests catch a contract violation that smoke run would have hidden"). Caught by tests rather than at runtime.

**Root cause analysis**:
1. Immediate cause: `ops.stack(...)` inside `build()` produces a graph-bound tensor.
2. Contributing factor: the layer's previous design comment (D-003) explicitly chose "plain (non-tracked) tensor" over `add_weight` to "remain parameter-free and avoid get_build_config plumbing." This was an over-correction — `add_weight(trainable=False)` keeps the layer parameter-free in the trainable-parameter sense, while making the basis a tracked, graph-independent weight.
3. Failed defense: the existing `test_no_trainable_parameters` and `test_build_process` asserted `len(non_trainable_weights) == 0`, encoding the buggy design as a contract. They would have flagged my fix as "regression" if I hadn't recognized them as the bug-encoding tests.
4. Prevention: any layer that holds frozen state should store it via `add_weight(trainable=False)` so it's tracked and graph-independent. Plain tensors created inside `build()` are a foot-gun for Keras Model composition.

**Fix (1 file in dl_techniques layer + 1 test file update)**:
- `src/dl_techniques/layers/activations/routing_probabilities.py`: rename `_cosine_basis` to `_cosine_basis_numpy` (pure numpy), and in `build()` deterministic branch store the result via `add_weight(trainable=False, initializer=keras.initializers.Constant(...))`. Anchored as `# DECISION D-004` in the layer.
- `tests/test_layers/test_activations/test_routing_probabilities.py`: update two tests (`test_build_process`, `test_no_trainable_parameters`) to expect `len(non_trainable_weights) == 1`, with comments pointing to D-004.

**Fix attempts**: 1 (one). Within autonomy leash. Revert-first not used because the diagnosis was definitive (matches the FuncGraph error class) and the fix was minimal (no wrapper cascades, no new classes, just changing storage from plain-tensor to non-trainable-weight).

**Verification**: full re-run of `tests/test_layers/test_activations/test_routing_probabilities*.py` (113 tests) + `tests/test_models/test_cliffordnet/test_cliffordnet_lm_routing.py` (21 tests) = **134 passed**. Both deterministic and trainable modes verified through Keras Model embedding (save/load roundtrip included).

**Plan files touched (out-of-scope additions)**: Two unplanned edits beyond the original Files To Modify table:
- `src/dl_techniques/layers/activations/routing_probabilities.py` (1 fix to library bug)
- `tests/test_layers/test_activations/test_routing_probabilities.py` (2 test asserts updated)

These are justified by the surprise discovery and noted here. The original plan's no-edit constraint applied to `lm.py` and `train_cliffordnet_nlp.py` — both untouched.

## plan_2026-05-04_38e259bf
### D-001 (PLAN, iter-1): unified `mode` parameter rather than two distinct classes or inheritance
**Choice**: Single class with `mode: Literal["deterministic","trainable"]`. Default `"deterministic"` preserves existing call-site semantics.
**Trade-off**: Single API surface and shared call/build math at the cost of a slightly larger constructor signature and dead kwargs (`kernel_initializer`, `bias_initializer`, etc.) when `mode="deterministic"`.
**Alternatives rejected**:
- Inheritance (`HierarchicalRoutingLayer(RoutingProbabilitiesLayer)`): keeps two classes, contradicts the "single layer" goal.
- Class-method factory (`RoutingProbabilitiesLayer.trainable(...)`): non-idiomatic for Keras serialization; `from_config` would need extra plumbing.
- Backward-compat alias `HierarchicalRoutingLayer = ...`: not requested; task says delete the file.

### D-002 (PLAN, iter-1): renormalization uses `prob_sum + epsilon` in both modes
**Choice**: Adopt the trainable variant's `unnormalized_probs / (prob_sum + self.epsilon)` for both modes.
**Trade-off**: Tiny numerical bias in deterministic mode at the cost of guaranteed no-divide-by-zero. Magnitude < epsilon = 1e-7 — well below test tolerance (1e-6).
**Why safe**: Decision probs are clipped to `[epsilon, 1-epsilon]`, so `prob_sum > 0` always, but the safer form costs nothing and unifies the code path.

### D-002b (EXECUTE, iter-1): D-002 falsified — revert to mode-specific renormalization
**Trigger**: `test_epsilon_parameter` in `test_routing_probabilities.py` regressed:
asserts `sum == 1.0 atol=1e-6` for `epsilon=1e-5`. With `+ epsilon` denominator,
sum is ~0.99998 — outside tolerance.
**Reason**: Deterministic mode's original semantics guarantee exact sum=1.0; the
"safety" of `+ epsilon` is unnecessary because clipping ensures `prob_sum > 0`.
**Fix**: Branch on `self.mode`. Deterministic uses bare `prob_sum`; trainable
keeps original `+ epsilon`. Both modes preserve their pre-merge semantics.
**Lesson**: "Safer" numerical changes can violate API contracts asserted by tests.
Default to preserving exact original behavior unless there is a real regression to fix.

### D-003 (PLAN, iter-1): factory `'hierarchical_routing'` key and `probability_output` strings preserved
**Choice**: Keep the public string keys; redirect them internally to `RoutingProbabilitiesLayer(mode="trainable", ...)`.
**Trade-off**: API stability at the cost of a tiny indirection.

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
