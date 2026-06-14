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

## plan_2026-06-14_33b77a7a
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-14_33b77a7a/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | sweep scope + byte-identity gate | 2026-06-14
**Context**: D-003 deferred the ops.sqrt-in-call() sweep. Census found 6 static call()-time sqrt-scale sites (performer:245/289, lighthouse:651/763, capsule:519, non_local:514) and 3 genuinely-dynamic sites (window:489, wave_field:556, capsule:599 — tensor norms / runtime shape).
**Decision**: Precompute the 6 static scales as Python floats (reuse performer.self.scale + lighthouse._scale where they already exist; add 3 new attrs). Leave the 3 dynamic sites. Gate every site with `np.float32(new)==np.float32(old)` + the existing test suite; revert any site that shifts a test.
**Trade-off**: Eliminating per-forward ops.sqrt tensor-node creation + D-002 consistency **at the cost of** touching 4 live numeric paths where a >1-ULP float drift would be a regression.
**Reasoning**: The precompute pattern is the blessed D-002 idiom already in ~15 layers. The risk is bounded to float32 rounding and fully covered by the regression oracle. Alternatives (leave deferred) rejected — the user explicitly requested the sweep.
**Anchor-Refs**: `src/dl_techniques/layers/attention/performer_attention.py:247` (D-001), `src/dl_techniques/layers/attention/lighthouse_attention.py:650` (D-002), `src/dl_techniques/layers/attention/non_local_attention.py:515` (D-003), `src/dl_techniques/layers/attention/capsule_routing_attention.py:344` (D-004) — per-step in-code anchors.

### D-002 | PLAN | capsule precompute in build(), not __init__ | 2026-06-14
**Context**: capsule `actual_key_dim` is None in __init__ (resolved in build() from input_shape embed_dim).
**Decision**: Precompute `_inv_sqrt_key_dim` in build() after actual_key_dim resolves; non_local/performer precompute in __init__ (init-static dims).
**Trade-off**: Correct value at every call **at the cost of** one non-uniform precompute location (build vs init).
**Reasoning**: Precomputing in __init__ would read None and crash. Mirrors the build-time-resolution pattern already in the layer.
**Anchor-Refs**: `src/dl_techniques/layers/attention/lighthouse_attention.py:650` (in-code `D-002` anchor — D-002 precompute-reuse pattern), `src/dl_techniques/layers/attention/capsule_routing_attention.py:344` (D-004, build())

### D-003 | EXECUTE | non_local dot_product-only scale precompute | 2026-06-14
**Context**: `non_local_attention.py:514` computed `scores / ops.sqrt(cast(key_value_channels))` per forward, inside the `dot_product` branch only; `gaussian` mode is intentionally unscaled (matches keras Attention use_scale=False).
**Decision**: Precompute `self._inv_sqrt_kv = 1.0/math.sqrt(float(key_value_channels))` in `__init__` (key_value_channels is final at init); replace 514 with `scores * self._inv_sqrt_kv`; remove the dead `d_k` line. Gaussian branch untouched.
**Trade-off**: Drop per-call ops.sqrt + D-002 consistency **at the cost of** a scale constant captured at init (correct: key_value_channels never mutates post-init).
**Reasoning**: np.float32 probe equal (0 ULP: 0.125 both); 16 non_local tests green incl. gaussian.
**Anchor-Refs**: `src/dl_techniques/layers/attention/non_local_attention.py:515`

### D-004 | EXECUTE | capsule scale precompute in build() | 2026-06-14
**Context**: `capsule_routing_attention.py:519` computed `attention_logits / ops.sqrt(cast(actual_key_dim))` per forward. `actual_key_dim` is None in __init__, resolved in build() from input_shape.
**Decision**: Precompute `self._inv_sqrt_key_dim = 1.0/math.sqrt(float(actual_key_dim))` in build() after actual_key_dim resolves; replace 519 with `attention_logits * self._inv_sqrt_key_dim`.
**Trade-off**: Drop per-call ops.sqrt **at the cost of** a non-uniform precompute location (build, not init) — mandated by actual_key_dim's resolution timing.
**Reasoning**: np.float32 probe equal (0 ULP: 0.25 both); 34 capsule tests green incl. graph-mode + .keras round-trip.
**Anchor-Refs**: `src/dl_techniques/layers/attention/capsule_routing_attention.py:344`

## plan_2026-06-14_077a2a35
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-14_077a2a35/D-NNN` anchor exists in source)
-->

### D-001 | Hopfield F7 cross-attn KV-dim derivation | 2026-06-14
**Context**: `hopfield_attention.py:373-375` builds `key_dense`/`value_dense` with `query_shape` ("Assuming key has same shape as query"). Cross-attention with a different K/V feature dim locks in a wrong-shaped kernel. Zero callers trigger it today; self-attention is correct.
**Decision**: Derive `key_shape`/`val_shape` from the list-of-shapes branch (element-type check `isinstance(input_shape[0], (list,tuple))`), falling back to `query_shape` for the flat single-shape (self-attention) case; build K/V Dense from those.
**Trade-off**: Correct cross-attention build **at the cost of** four extra lines in `build()` and a wider input contract that the self-attn path must provably pass through unchanged.
**Reasoning**: The element-type disambiguation is the established pattern (perceiver S1, LESSONS tuple->list rule). Self-attn input is a flat tuple so the fallback keeps it byte-identical (the live-path invariant). Alternative (rename/restructure inputs) rejected: out of scope, no caller demand. STOP-IF self-attn diverges -> revert this step, ship AF1/AF2 only.
**Anchor-Refs**: `src/dl_techniques/layers/attention/hopfield_attention.py:382-396` (build() K/V shape fix)

### D-002 | Performer NOTE 1 causal/non-causal branch | 2026-06-14
**Context**: `performer_attention.py:_linear_attention()` always computes the non-causal block (lines 320-333, incl. a rank-4 `kv` materialization) then overwrites `out` when `self.causal`. The non-causal compute is fully dead on the causal path.
**Decision**: Wrap the body in `if self.causal: <causal block> else: <non-causal block>`, each block byte-identical to its current form (causal block is the S3-fixed version).
**Trade-off**: Eliminated dead FLOPs on the causal path **at the cost of** touching a hot live numeric path where any byte-level drift is a regression.
**Reasoning**: Pure compute-elision; both branches are exactly today's code. Gated by atol-0 forwards for both `causal` values + the full performer test file. Alternative (leave as-is) rejected because the deferred NOTE 1 is a named open item and the fix is <=8 lines and behavior-identical.
**Anchor-Refs**: `src/dl_techniques/layers/attention/performer_attention.py:316` (causal/non-causal guard)

### D-003 | Scope discipline — defer SQRT sweep + not-a-bug items | 2026-06-14
**Context**: Adversarial pass flagged `ops.sqrt`-in-`call()` at performer:245/289, lighthouse:651/763, capsule:519, non_local:514 (consistency, not correctness), plus AF3 mobile_mqa get_config and NOTE 2 add_weight ordering.
**Decision**: Fix only hopfield AF2 (folds into Step 1); defer the rest of the SQRT sweep; classify AF3 + NOTE 2 as NOT-A-BUG and leave untouched.
**Trade-off**: A smaller, fully behavior-verifiable plan **at the cost of** leaving a known cosmetic SQRT-consistency gap across four files.
**Reasoning**: Those call-time sqrt sites produce correct output and are behavior-neutral (YAGNI); a broad sweep would expand the live-path blast radius for zero functional gain. AF3/NOTE 2 are provably safe under standard Keras `from_config`/idempotency guard (findings). Surface-and-defer keeps both this plan and any future SQRT-consistency plan clean.
**Anchor-Refs**: (none — scope decision, no source anchor)

### D-004 | FLAKY capsule test — diagnostic-first, not source-contort | 2026-06-14
**Context**: `test_capsule_routing_attention.py::...test_graph_mode_positional_routing_concrete_seqlen` passes isolated (6.67s), fails in full-suite run. Classified as test-isolation/global-state pollution, not a source defect.
**Decision**: Diagnose root cause; apply a fix ONLY if it is a quick, clearly-correct test-isolation change (missing reset / stale registration / deterministic seed). Otherwise document and leave source AND test untouched.
**Trade-off**: Honest diagnosis with a hard autonomy leash **at the cost of** possibly leaving a known CI-flake unfixed this iteration.
**Reasoning**: Contorting source to satisfy polluted global state would introduce a real defect to mask a test-infra problem (LESSONS: diagnostic-vs-fix framing logged before code). If diagnosis reveals a real source defect, that falsifies A7 -> NEEDS_EXPLORE, not a quick fix.
**Anchor-Refs**: (none until a concrete test-isolation fix is chosen)

### D-005 | FLAKY root cause = eager-vs-graph tolerance, fix applied | 2026-06-14
**Context**: Diagnosed D-004. Isolated 20-run stress (GPU1) on the eager-vs-graph comparison: 0 failures but worst `maxabs = 5.96e-7`, sitting right under the test's `atol=1e-6`. No random-order plugin; conftest has no seeding/policy-reset fixtures. The input was UNSEEDED (`keras.random.normal` w/o seed), so per-run magnitude varied and under full-suite GPU memory pressure (different XLA kernel selection) the delta occasionally exceeded 1e-6. NOT a source defect, NOT global-state pollution — genuine eager-vs-XLA float nondeterminism with an over-tight tolerance.
**Decision**: Seed the test input (`seed=1337`) for determinism and loosen the eager-vs-graph assertion from `1e-6` to `1e-5` (a realistic GPU eager-vs-graph delta). Source untouched. This satisfied A7 (test-infra, not source) and the D-004 "quick clearly-correct test-isolation fix" gate.
**Trade-off**: A deterministic, non-flaky A1 lock **at the cost of** a 10x looser numeric tolerance on that one assertion.
**Reasoning**: The A1 invariant (static-shape loop unrolls; graph matches eager) is still fully locked at 1e-5 — a broken unroll diverges by orders of magnitude, not 1e-6. Seeding removes run-to-run variance; the tolerance reflects the true eager-vs-XLA delta. The compiled-model sibling (test...in_compiled_model) only checks shape/NaN, and the roundtrip test compares same-path predict() calls — neither is at risk, so no further edits. Verified: full capsule file 34 passed.
**Anchor-Refs**: (none — test-only change)

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
