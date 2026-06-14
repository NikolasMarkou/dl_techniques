# Consolidated Decisions
*Cross-plan decision archive. Entries merged from per-plan decisions.md on close. Newest first.*

<!-- COMPRESSED-SUMMARY -->
## Summary (compressed)
*Auto-compressed from 628 lines on 2026-05-13 (refreshed after plan_2026-05-13_a1c9a52d close — merged layers/ntm/ into layers/memory/, deleted ntm package; no new active constraints introduced, but note new constraint below). Read full content below for details on each plan's decisions.*

### Active Constraints (anchored, do-not-break)
- **3-name encoder public surface** (`<Model>`, `create_<model>`, `create_<model>_with_head`) — locked in tree_transformer/bert/cliffordnet; gpt2 is 2-name (LM head intrinsic); cliffordnet now hosts 4+3 names (multiple model classes).
- **`_download_weights` raises `NotImplementedError`** + **`from_variant` narrow `except (IOError, OSError, ValueError)`** — no silent random-init fallback. Anchored in tree_transformer, bert, gpt2, vit, cliffordnet, cliffordnet/embedding_unet.
- **`pad_token_id=<tokenizer_pad>` must be wired from trainer config to encoder ctor** (silent semantic bug otherwise). tiktoken cl100k_base pad = 100266; gpt2 enc pad differs.
- **Output dict key `"logits"`** + **`prepare_dict_keyed_compile(model, output_key="logits")`** required for every Pattern-3 CLM trainer before `model.compile`.
- **`build_clm_metrics(encoding_name, ignore_index)`** — required metric floor for every CLM trainer (replaces bare `["accuracy"]`).
- **`SegmentationWrapperLoss`** is the canonical save/load-friendly seg loss; no more `compile=False` workarounds in trainers.
- **`save_own_variables`/`load_own_variables`** on outer Model classes wrapping inner Models (DepthAnything pattern) — required for `.keras` round-trip when sub-Model weights would otherwise re-initialize.
- **memory_bank dual-optimizer**: register one optimizer with `super().compile`, apply second manually; prefix split via `name.split('/')[0].startswith(p)` (leading-component, NOT substring).
- **U-Net `.keras` round-trip tolerance is atol=1e-4** (not 1e-5) on fp32 GPU due to reduction-order noise. Applies to lmunet + embedding_unet + AccUNet.
- **`dl_techniques.layers.ntm` no longer exists** — all NTM / MANN / SOM imports go through `dl_techniques.layers.memory` (plan_2026-05-13_a1c9a52d D-002). Top-level (`NTMCell`, `NTMConfig`, `create_ntm`, `MannLayer`, `SOM2dLayer`, `SOMLayer`, `SoftSOMLayer`) and deep-submodule paths both supported.

### Failed Approaches (do NOT retry)
- "Modify `lmunet.py` in place with a `causal` flag" — REJECTED (plan_632605aa D-001); also "modify Clifford block classes with `causal` flag" — REJECTED. Sibling-stack additive file is correct.
- `keras.ops.cond` for runtime branch skipping inside `call()` — both branches trace under TF; use multiply-by-zero (plan_0f39a086 D-003).
- Mocking the database in tests / using `compile=False` to dodge a custom-loss round-trip bug — both are workarounds, not fixes (LESSONS).
- SimCSE / contrastive sentence-pair training as iter-1 for an encoder package — explicit deferral pattern (plan_632605aa D-003; plan_146ae899 — staged plans only).
- LR sweep on "smooth-train + cliff-val + sub-random val" signature — that fingerprint = data-pipeline divergence, NOT hparams (plan_f2d29729 D-006/D-007).

### Decision-Anchor Conventions
- Format: `# DECISION plan_<id>/D-NNN: <one-line>` at point of impact. Block, hash, double-dash variants supported. Unqualified `D-NNN` anchors from old plans are tolerated but WARN; new code MUST use qualified form.
- 5 triggers: failure-driven, non-obvious, rejected-alternative, constraint-workaround, 3-strike.
- Anchor at impact site (not at decision definition). One anchor per impact site, even if shared with sibling decision.
<!-- /COMPRESSED-SUMMARY -->

## plan_2026-06-14_ab855e7e
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-14_ab855e7e/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-14
**Context**: `attention/` was fully resolved earlier today by 5 plans (final b9456f74, 940 tests). A fresh 4-explorer adversarial pass + per-claim source verification found residue those plans' scoped findings missed: F2 (ops.sqrt-on-static-scalar in 3 more files), F3 (graph-safety in ring/anchor), F4 (factory optional_params drop real class args), F5 (mobile_mqa ignored mask), F6 (training= omission to attn_prob).
**Decision**: Execute a behavior-preserving residue-cleanup of F2-F6 using established precedent patterns (D-002 for scale, capsule/PFA static-shape+fail-loud for graph-safety, additive registry completion); defer F7/F8 as logged low-value/judgment items.
**Trade-off**: Closing the residue with precedent-matched minimal edits **at the cost of** not re-litigating the whole sub-package (the GHOST "resolved" zone) and deferring the two judgment-call items (F7 latent no-caller bug, F8 factory-registration of 2 standard-sig layers).
**Reasoning**: Every fix has an established in-repo pattern and a test oracle (940 tests + new factory test). Alternatives rejected: (a) full re-review and re-fix — wasteful, the prior 5 plans were thorough; (b) registering wave_field/single_window (F8) — non-trivial signature/contract questions, belongs in its own plan; (c) fixing F7 — no caller exercises it, already inline-commented.
**Anchor-Refs**: (anchors added at EXECUTE for F2 scale sites + F3 fail-loud guards)

### D-002 | PLAN | 2026-06-14
**Context**: F3 makes ring/anchor raise ValueError on a dynamic (None) sequence dim.
**Decision**: Fail-loud on None seq, do NOT refactor to a fully-dynamic loop.
**Trade-off**: Graph-safety + a clear error **at the cost of** ring/anchor no longer accepting symbolic variable-length sequences (they require statically-known N).
**Reasoning**: The block loop (ring) and anchor/query split fundamentally need a Python-int count; a dynamic-loop refactor would be a large behavior-touching rewrite. capsule/PFA set this exact precedent and were accepted.
**Anchor-Refs**: `src/dl_techniques/layers/attention/ring_attention.py` (D-002 comment at seq_len read in `_blockwise_attention`), `src/dl_techniques/layers/attention/anchor_attention.py` (D-002 comment at seq_len read in `_hierarchical_attention`)

D-001 in-code anchors (F2 scale): `src/dl_techniques/layers/attention/gated_attention.py` (self.scale in `__init__`), `src/dl_techniques/layers/attention/group_query_attention.py` (self.scale in `__init__`).

## plan_2026-06-14_b9456f74
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-14_b9456f74/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-14
**Context**: EXPLORE (source-verified) confirmed three deferred xfail-gated attention bugs (S2 PFA SW-MSA mask dead-on-forward; S3 Performer causal 5-D einsum crash; S4 Ring `slice_update` has no eager-TF gradient), 22+5 build() methods missing the Keras-3 idempotency guard (L1/F20), and two doc/export gaps (D1 README caveat table, D2 unexported Ideogram4/MMDiT). All three bugs have bounded fixes (<=40 lines each). Three further findings (C2 missing `package=`, call() param renames, extra graph tests) are regression-risk or no-op and were ruled out of scope.
**Decision**: Execute a fixed-scope 5-step plan in riskiest-isolated-first order — S3 (1), S4 (2), S2 (3), guard sweep (4), docs/exports/test (5) — fixing only the source-verified bounded subset; explicitly defer C2/`package=`, call() renames, and additional `tf.function` tests.
**Trade-off**: Resolve the three operate-as-advertised bugs + lifecycle conformance with a low blast radius (each fix scoped + individually verified before the mechanical sweep) **at the cost of** leaving the latent bare-name serialization-collision hazard (C2) and the call()-signature inconsistencies unaddressed (documented, not fixed).
**Reasoning**: C2's `package=` fix changes the registration KEY and breaks deserialization of already-saved `.keras` models (a breaking change, not a fix); call() renames break serialized configs (D-007, plan 0c5d4a21). Both are net-negative under the goal "operate as advertised + Keras compliance". The riskiest fix (S2, live model callers in pft_sr/thera/swin) is isolated behind a CHECKPOINT with a minimal-fix fallback (PM1) so a general-geometry failure cannot block the bounded S3/S4 wins. The guard sweep runs after the bug fixes so a guard regression is isolated from the numeric edits.
**Anchor-Refs**: `src/dl_techniques/layers/attention/progressive_focused_attention.py:423` (S2 mask site, SW-MSA static-H,W requirement)

### D-002 | REFLECT (iter-1) | 2026-06-14
**Context**: All 6 EXECUTE steps (5 planned + 1 review-driven test) succeeded. Verifier: 5/5 criteria PASS; full attention suite **939 passed / 0 failed / 0 xfailed** (baseline 935/0/3); live PFA caller `thera/tails` 20/20 PASS after the S2 mask fix; zero scope drift; no debug artifacts. Adversarial reviewer (ip-reviewer, run despite iter-1 due to MAXIMUM-EFFORT + 25-file blast radius): **0 CRITICAL, 0 WARNING**, verdict READY_TO_CLOSE, with all three high-risk fixes confirmed mathematically correct by construction (S3 canonical FAVOR+ causal prefix-sum no off-by-one; S4 concat==slice_update byte-identical; S2 numpy mask partition byte-identical to `_window_partition` ordering, tile B-major/window-minor matches attn_scores). 4 advisory NOTEs raised.
**Decision**: Route to CLOSE. Address review NOTE 4 (fixed); accept/defer NOTE 1/2/3.
**Trade-off**: Ship the verified contract/robustness/operate-as-advertised fixes now **at the cost of** leaving two pre-existing micro-smells (performer dead non-causal compute on the causal path; mobile_mqa add_weight-after-super().build) as documented optional follow-ups rather than touching just-fixed/by-design code.
**Reasoning / NOTE disposition**:
- **NOTE 1** (performer computes+discards the non-causal block when `causal=True`): pre-existing wasted FLOPs, correctness-neutral. Touching the just-fixed causal path for a marginal efficiency gain adds risk for no correctness value. DEFER (optional `if not self.causal:` guard).
- **NOTE 2** (mobile_mqa `add_weight` after `super().build()` sets built=True): pre-existing latent smell; baseline passed and the new top-of-build guard correctly fixes the second-build idempotency case. No regression introduced. ACCEPT as-is.
- **NOTE 3** (in-code anchor reuses strategy D-001 rather than a dedicated D-NNN for the fail-loud-on-dynamic-H/W fork): D-001 Anchor-Refs point at `:423` and its Reasoning covers the PM1 fallback, so it is traceable. ACCEPT (granularity preference, not a defect).
- **NOTE 4** (README `differential_attention` rationale imprecise — pitfall is a positional `training` binding to `layer_idx`, not `attention_mask`): FIXED in commit `d5bafaa4` (step-7).
- **Blind-spot closure**: reviewer flagged general-H×W PFA SW-MSA correctness was verified only by an ephemeral manual forward → added committed regression `test_shifted_window_forward_general_geometry` (4 geometries incl. non-square nW>1) in commit `721a6e83` (step-6).
**Anchor-Refs**: none (REFLECT routing decision; no new in-code anchor).

## plan_2026-06-14_7734bacd
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-14_7734bacd/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-14
**Context**: 4 parallel explorers audited all 30 attention files (33 classes), factory, docs, tests. Package largely healthy (prior F18/F19/hopfield/cross-attn fixes verified present). Found 2 real robustness bugs (capsule `range(graph_tensor)` graph crash A1, capsule Dense-in-build A2), 4 doc/wiring defects (W1-W4), 3 contract-limit findings (C1-C3), 6 untested/partial layers (T1/T2). Explorers over-flagged `ops.sqrt`-in-`call()` as HARD; downgraded to non-bug (N1, dropped).
**Decision**: Scoped plan = Bugs + docs + tests. Fix A1/A2/W1-W4, document C1/C2/C3, add forward+config+`.keras` tests for the 6 layers. Defer L1 mass-guard pass + L2/F20 + C1 renames.
**Trade-off**: Fix verified-real bugs and close the regression-gate gap **at the cost of** leaving ~25 harmless missing-guard build()s and 7 non-standard call sigs documented-but-unchanged (renames break serialized configs).
**Reasoning**: User-selected scope. L1 guards are pure-additive cosmetic conformance with near-zero real payoff and high review cost; C1 renames unsafe. The 6 untested layers are the real robustness risk — tests close coverage AND surface latent dead-on-forward bugs (LESSONS: never-executed code hides multi-bug chains). User directed fix-in-plan, per-step leash still gates (2 attempts → STOP).
**Anchor-Refs**: none yet (code anchors added during EXECUTE if a fix meets the 5 trigger conditions)

### D-002 | EXECUTE step 2 | 2026-06-14
**Context**: Capsule A2 fix. The four Dense projections (query/key/value/output) are None-init in `__init__` and created in `build()` because their units depend on `embed_dim = input_shape[-1]` (`actual_key_dim` defaults to `embed_dim // num_heads`; `output_dense` uses `embed_dim`). Build had no idempotency guard, so a second `build()` (functional reuse / from_config) re-creates and discards already-built Dense weights.
**Decision**: Keep the Denses created in `build()` (None-sentinel pattern, same as HopfieldAttention precedent in this package) and add `if self.built: return` as the first line of `build()` to make it idempotent. Do NOT move the Denses to `__init__`.
**Trade-off**: Idempotent build with weight-correct reload **at the cost of** retaining the None-sentinel-in-build pattern (units genuinely need input shape, so moving to `__init__` is impossible without an extra ctor arg).
**Reasoning**: Trigger conditions: non-obvious ("why not move Denses to __init__?") + framework constraint (units depend on input_shape[-1]) + a simpler-looking alternative (move to __init__) deliberately rejected. The guard alone is the actual A2 fix; the `.keras` round-trip test is the detection oracle.
**Anchor-Refs**: `src/dl_techniques/layers/attention/capsule_routing_attention.py:328-334`

### D-003 | EXECUTE step 3 | 2026-06-14
**Context**: First-ever `.keras` round-trip test for `PerceiverAttention` surfaced a latent serialization bug. `build()` branched on `isinstance(input_shape, list)` to decide "two inputs (cross-attention) vs single input". Keras serializes a shape tuple to a plain list, so on `build_from_config` during `load_model` a single 3D query shape arrives as `[None, 8, 32]` (a 3-element list of scalars). The old check treated that as 3 inputs and raised `ValueError("Expected 2 inputs for cross-attention, got 3")`. Forward/eager and symbolic-build worked (shape passed as a real tuple); the bug bit ONLY via the functional `.keras` round-trip (build_config replay) — the classic "never-executed code" failure mode.
**Decision**: Disambiguate a genuine list-of-shapes (elements are themselves list/tuple) from a single serialized shape (elements are int/None) via a local `_is_list_of_shapes` helper, instead of `isinstance(list)` alone. 7-line contained fix; forward semantics unchanged on every previously-working path (cross-attention suite 65 green, perceiver_transformer consumer smoke green).
**Trade-off**: Robust round-trip + multi-input support **at the cost of** a small structural-typing helper inside build() (vs the simpler-but-wrong isinstance check).
**Reasoning**: Trigger conditions: framework constraint (Keras tuple→list shape serialization) + non-obvious ("why not just isinstance(list)?") + a simpler alternative (the original check) deliberately rejected because it is the bug. A separately-surfaced PFA SW-MSA mask bug (dead-on-forward, 5D index + mismatched tile + collapsing reshape) was NOT chased — it needs a >10-line restructure altering masked-attention semantics, so its test is `xfail(strict=False)` and surfaced for a dedicated plan (per leash).
**Anchor-Refs**: `src/dl_techniques/layers/attention/perceiver_attention.py:174-198`

### D-004 | EXECUTE step 4 | 2026-06-14
**Context**: First-ever tests for RingAttention + PerformerAttention surfaced two latent bugs in never-executed code paths (per LESSONS). (1) Performer `causal=True` forward is dead-on-forward: `_linear_attention` causal branch (`performer_attention.py:333-349`) builds a 5-D tensor via `ops.einsum('bhnf,bhnd->bhnfd', ...)`, `cumsum`es it, then feeds it (with a mis-ranked `expand_dims(q, -2)`) into a second einsum `'bhnf,bhnfd->bhnd'` whose subscripts are inconsistent with the actual tensor ranks → `InvalidArgumentError` (rank 5 vs expected 4). (2) RingAttention forward/config/.keras all pass, but backprop fails: blockwise output placement uses `ops.slice_update` → `XlaDynamicUpdateSlice`, which has no registered gradient in eager TF.
**Decision**: Per HARD leash (multi-bug chain / >10-line restructure → STOP, xfail, do NOT chase). Both bugs marked `@pytest.mark.xfail(strict=False)` with root-cause reasons and kept as gates. NO source edits — revert-clean (working paths untouched). Performer non-causal forward + Ring forward/config/.keras are fully functional and covered by passing tests.
**Trade-off**: Closes coverage on the working paths + documents the two latent bugs as standing gates **at the cost of** leaving causal-Performer and Ring-gradient-flow unfixed (each needs a dedicated plan: streaming-causal math rewrite; gradient-friendly scatter/concat block placement).
**Reasoning**: 3-strike / multi-bug-chain leash. The causal fix requires re-deriving the streaming-causal cumsum (>10 lines, semantics-altering); the Ring gradient fix requires replacing `slice_update` with a differentiable block-placement (backend-level, semantics-sensitive). Neither is a contained ≤10-line fix. Surfaced for dedicated follow-up plans (the F18/F19 / PFA-SW-MSA deferral pattern).
**Anchor-Refs**: none (no source edited; xfail gates at `tests/test_layers/test_attention/test_performer_attention.py` TestForward::test_output_shape_causal and `tests/test_layers/test_attention/test_ring_attention.py` TestEdgeCases::test_gradient_flow)

### D-005 | REFLECT iter-1 | 2026-06-14
**Context**: All 5 steps complete. Final regression gate: 935 passed / 3 xfailed / 0 failed / 0 errors (+109 vs 826 baseline). 6/6 success criteria PASS. Scope drift none, diff clean, validate-plan introduces zero new ERRORs (31 pre-existing orphan anchors from other plans). EXECUTE surfaced 4 latent bugs: S1 (Perceiver .keras) FIXED in-plan (D-003); S2/S3/S4 (PFA shifted-window mask, Performer causal, Ring gradient-flow) DEFERRED as xfail-gated multi-bug chains per the autonomy leash.
**Decision**: Recommend CLOSE. The plan's scoped goal (fix A1/A2/W1-W4, document C1-C3, add T1/T2 coverage) is fully met. The 3 deferred bugs are correctly out-of-scope (each a >10-line multi-bug restructure) and now permanently gated by xfail regression tests + findings S2/S3/S4 for dedicated follow-up plans.
**Trade-off**: Ship a verified, fully-tested attention package with 3 documented-and-gated known-broken modes **at the cost of** not resolving S2/S3/S4 here (genuine multi-bug chains; chasing them would violate the leash and risk derailing 1 clean fix + 6 clean test-additions — the F18/F19 deferral pattern).
**Reasoning**: User directed "fix all in-plan" but acknowledged the per-step leash still gates; S2/S3/S4 each blew the 10-line / multi-bug threshold on diagnosis, so xfail+defer is the correct leash application. Devil's advocate: 3 layer modes remain broken in source — but they were ALREADY broken (never-executed dead code), this plan did not regress them, and it converted silent breakage into loud xfail-gated + documented breakage, strictly better.
**Anchor-Refs**: none (REFLECT routing decision, no new source anchor)

## plan_2026-06-14_adaddf34
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-14_adaddf34/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-14
**Context**: Two pre-existing `NonLocalAttention` bugs surfaced (not caused) by plan_2026-06-14_0c5d4a21, deferred to this plan. Both source-verified AND reproduced: F18/G1 — DEFAULT `attention_mode='gaussian'` reduces K/V to `attention_channels//8` but leaves `query_conv` at full `attention_channels`, so `Q@Kᵀ` is shape-incompatible (`Matrix size-incompatible [2,256,32] x [2,4,256]`); F19/G2 — `kernel_size` normalization uses `isinstance(kernel_size, tuple)`, False for the LIST Keras produces on `.keras` reload → wraps to `([7,7],[7,7])`, breaking round-trip.
**Decision**: Two surgical fixes in `non_local_attention.py` (G2 normalize via `tuple(...)`; G1 align Q to embedded `key_value_channels` + `max(1,//8)` guard + docstring) plus a NEW `test_non_local_attention.py` regression gate. Byte-identical on working paths (dot_product; int/tuple kernel_size).
**Trade-off**: Correct Non-local embedded-Gaussian + robust serialization **at the cost of** changing the gaussian-mode query channel count (a path that currently crashes → no working caller affected) and 1 new test file.
**Reasoning**: `Q@Kᵀ` REQUIRES a matched contraction dim, so reducing only K/V is a bug, not a design choice; aligning Q to `key_value_channels` is minimal + byte-identical in dot_product. `tuple(kernel_size)` uniformly handles int/tuple/list/TrackedList. Rejected: separate Q-projection (extra params, not the paper); custom from_config (band-aid — root cause is __init__ normalization); changing `//8` (out of scope).
**Anchor-Refs**: `src/dl_techniques/layers/attention/non_local_attention.py` (D-002 anchor added at the gaussian channel-alignment site during EXECUTE step 2)

### D-002 | EXECUTE step 2 | 2026-06-14
**Context**: F18/G1 — gaussian mode left `query_conv` at full `attention_channels` while K/V were reduced to `attention_channels//8`, so `Q@Kᵀ` contracted mismatched dims and crashed. The embedded-Gaussian formulation (Wang et al.) projects theta/phi/g all to the same reduced dim.
**Decision**: In gaussian mode set `key_value_channels = max(1, attention_channels // 8)` and route `query_conv` (and the `call()` q-reshape, `build()` q_seq_shape, and dot_product `d_k` scaling) through `key_value_channels` so Q,K,V share one embedded dim. `dot_product` keeps `key_value_channels == attention_channels` → byte-identical.
**Trade-off**: Correct, well-formed gaussian attention **at the cost of** reducing the gaussian-mode query channel count (a path that only ever crashed → no working caller affected).
**Reasoning**: `Q@Kᵀ` needs a matched contraction dim; reducing only K/V is a bug. `max(1, ...)` clamps `attention_channels<8` to embedded dim 1 (avoids a 0-filter Conv2D). Rejected: a separate Q-projection (extra params, not the paper); keeping Q full via padding (incorrect). Verified SC2 dot_product `array_equal` (max_abs_diff 0.0), SC1 gaussian forward finite, SC5 (channels=4) guard forward finite.
**Anchor-Refs**: `src/dl_techniques/layers/attention/non_local_attention.py:279-284`
