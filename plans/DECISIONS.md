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

## plan_2026-05-13_3a2f1d23
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-13_3a2f1d23/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-05-13
**Context**: Reviewer surfaced 5 critical + 10 high + 10 medium claims against logic/. Empirical verification of all 5 critical: C1 (Gumbel form) confirmed differential statistical effect; C2 (smooth-divide dipole) is the LESSONS L44 / D-001-anchored design choice — reviewer's math is right but classification as bug is wrong; C3 (1-D selection) confirmed by code inspection; C4 (sigmoid-stacking unsoundness) confirmed by tracing call graph; C5 (to_symbolic gumbel leak) confirmed.
**Decision**: Phase changes by risk into A (LOW-risk correctness fixes + default flips), B (MED-risk feature additions: alias, force-clip), C (HIGH-risk arch: per-channel selection mode), D (NET-NEW features: t-norms, walker, callback, diversity). Present two scope options to user (Quick Wins = Phase A only; Full Scope = all). Reject reviewer H9 (remove explicit child.build) — LESSONS L42 empirically reversed this earlier.
**Trade-off**: comprehensive implementation **at the cost of** ~1100 LOC, 16 commits, and a non-trivial architectural addition (per-channel mode) that doubles the weight count for selection in the new mode.
**Reasoning**: User asked MAXIMUM EFFORT. Phasing by risk lets the user revert mid-plan if Phase C/D goes sideways without losing the correctness wins. LESSONS L42 takes precedence over reviewer H9 because we have a documented empirical reversal anchor.
**Anchor-Refs**: `findings/critical-claims-verification.md`, `findings/scope-and-risk-assessment.md`, `src/dl_techniques/layers/logic/arithmetic_operators.py:442`, `src/dl_techniques/layers/logic/logic_operators.py:467`

### D-002 | EXECUTE step-7 | 2026-05-13
**Context**: H6 review claim — `load_balance_coefficient` is a misnomer; the loss it gates is the Shazeer (2017) gate-entropy regularizer (mean over batch of -p·log(p)), not a load-balance loss. External callers (only `src/train/latent_reasoning_vision/circuit.py`) currently pass `load_balance_coefficient`.
**Decision**: Add `gate_entropy_coefficient` as the canonical kwarg on `CircuitDepthLayer` + `LearnableNeuralCircuit`. Keep `load_balance_coefficient` as a deprecated alias that emits a one-time `DeprecationWarning` and resolves to the same internal `_gate_entropy_coef`. `get_config()` emits the new name.
**Trade-off**: API clarity **at the cost of** carrying a deprecated alias indefinitely (cannot remove without breaking external consumer).
**Reasoning**: Renaming silently would break the external consumer. Adding the canonical name with an alias preserves bit-exact behavior for existing callers while letting new code use the correct term. Aligns with library convention that misnamed APIs get aliased, not replaced.
**Anchor-Refs**: `src/dl_techniques/layers/logic/neural_circuit.py:45`

## plan_2026-05-13_a2b0f17b
### D-001 | EXPLORE → PLAN | 2026-05-13
**Context**: Prior plan `plan_2026-05-13_e52a5ac8` already empirically verified this exact review and deferred most fixes for documented reasons (consumer break, `.keras` round-trip break, "documented footgun not bug" per LESSONS L38). User has explicitly overridden these conclusions and selected "Everything including B, E, F, G (full rewrite)" via AskUserQuestion.
**Decision**: Implement all overrides A-G plus the 5 truly-new safe items in a single multi-phase plan, with **opt-in flags preserving back-compat** for every behavior-changing change to maximize chance of passing existing 118 tests.
**Trade-off**: Comprehensive coverage of every review item **at the cost of** API surface explosion (≈8 new optional params on `LearnableLogicOperator` alone) and a guaranteed need to update the consumer `circuit.py`. Justified because user explicitly selected the maximum-scope option after being shown the alternative.
**Reasoning**: Previously rejected items (softplus reparam, smooth divide, routing rewrite, MoE aux losses) are real engineering improvements; the prior plan rejected them on prudence grounds (don't fix what consumers depend on). User has now overridden prudence with eyes open. The opt-in default scheme (`circuit_routing='output_only'` new default but `'classic'` preserves old; `softplus_temperature=False` default; `safe_divide_mode='hard_clamp'` default) is the right balance: new code gets fixed behavior, archive load still works.
**Alternatives rejected**:
- Full rewrite with no back-compat shims: would invalidate 118 tests + every saved `.keras` → too brittle.
- Implement only safe items: ignores user's explicit override.
- Decompose into 3-4 separate plans: matches LESSONS guidance for >iter-5 work, but user wanted single-pass; we'll decompose if iter-3 hits.
**Anchor-Refs**: pending — will anchor at the actual code sites in EXECUTE.

### D-002 | PLAN | 2026-05-13
**Context**: Plan v1 has 18 steps across 4 phases touching 10 files. Step 12 (routing rewrite) is the highest-blast-radius change because it alters the math the consumer was trained against.
**Decision**: Make `circuit_routing` a `Literal['classic','output_only']` with new default `'output_only'`. Pin consumer `circuit.py` to `'classic'` only if its smoke test fails on the new default.
**Trade-off**: Honest default that fixes the math **at the cost of** breaking any external trained `.keras` files that depend on the old attenuated forward pass.
**Reasoning**: The user's override extends to consumer changes. If a deployed model was trained against the broken routing, retraining is the honest fix; backward-compat as opt-in is sufficient.
**Anchor-Refs**: `src/dl_techniques/layers/logic/neural_circuit.py` (CircuitDepthLayer ctor + call), `src/dl_techniques/layers/logic/arithmetic_operators.py` (sign-preserving power, smooth divide), `src/dl_techniques/layers/logic/logic_operators.py` (allow_unary_degenerate raise).
**Outcome**: Consumer `train/latent_reasoning_vision/circuit.py` smoke-tested with new default (1033690 params, finite forward) — no consumer patch needed.

### D-003 | EXECUTE | 2026-05-13
**Context**: After moving sublayer creation to __init__ (intending lazy build via __call__), `test_factory_layer_round_trip[circuit_depth-kwargs2]` failed with "Layer 'arithmetic_op_0' was never built and thus it doesn't have any variables. However the weights file lists 3 variables for this layer." The Keras 3 saving contract requires that the parent's `build()` method explicitly create state of all children, not rely on later `__call__` to do it.
**Decision**: Restore explicit `child.build(input_shape)` calls in `CircuitDepthLayer.build()` and `LearnableNeuralCircuit.build()` for every sublayer. Children are still constructed in `__init__` (for serialization-config matching), but their state is created in the parent's `build()`.
**Trade-off**: Two-stage child management (construct in __init__, build in parent.build) **at the cost of** non-idiomatic Keras 3 pattern (the docs encourage lazy build). Reversed-the-prior-judgment: prior plan's "cargo-cult manual build" call was actually correct.
**Reasoning**: Keras 3 round-trip requires the saved weights file to map cleanly to a built layer hierarchy on load. Without manual build, the loader sees variables for children that haven't been built and raises. Documented for future readers.

## plan_2026-05-13_e52a5ac8
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-13_e52a5ac8/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-05-13
**Context**: Prior review (assistant turn) flagged ~30 issues in `layers/logic/`. Per LESSONS L20, every claim was empirically verified via `findings/verify_claims.py`. 6 claims confirmed actionable + low-risk; others either overstated (H4), false positive (L4), inherent-to-math (C5), or already documented per LESSONS L38 (C6). Cross-plan context: predecessor plan_2aaad563 already aligned this package's surface and deliberately deferred unary degeneracy to README.
**Decision**: Apply only the 6 empirically-confirmed, additive, low-risk fixes (C1, M2, H10, H9, C3, L2); defer high-risk semantic changes (C2, C4, C5) and design opportunities (H1-H8, O1-O10).
**Trade-off**: Safety and zero-regression risk **at the cost of** leaving real but invasive issues unaddressed (input-side routing pathology C2, temperature reparam C4, division gradient hazard C5 — documented as README warnings instead).
**Reasoning**:
- C2 (output-side routing) is a semantic break with high blast radius; user has working consumer (`train/latent_reasoning_vision/circuit.py`) depending on current behavior — defer until concrete demand.
- C4 (softplus temperature) changes `.keras` archive semantics — temperature_init=1.0 would no longer mean temperature=1.0; breaks existing saved configs (LESSONS L94/L118 risk).
- C5 (safe_divide) — huge gradients near zero are inherent to division; current `_safe_divide` is arguably correct; smoother alternatives (`x1*x2/(x2²+ε²)`) change function semantics significantly. Document hazard instead.
- C6 (NOT-in-binary-pool) — explicitly classified as documented footgun by LESSONS L38.
- H4 (zeros init) — empirically 1.4% softmax spread; cosmetic, not functional. Skip.
- L4 (initializer round-trip) — false positive; works correctly.
**Anchor-Refs**: TBD at EXECUTE step boundaries (only C1 + C3 fixes warrant `# DECISION` anchors per the 5-trigger rule in `references/decision-anchoring.md` — they encode "do NOT revert to false default" and "apply_sigmoid=False is intentional for stacked use").

## plan_2026-05-13_2aaad563
### iter-1 PLAN — recommended approach
**Chosen**: minimal-additive refactor: add `factory.py` + populate `__init__.py` + rank-relax + small fixes + write `README.md` + scope-add tests. **At the cost of**: not addressing the unary `subtract`/`divide` footgun in code (documented in README only) — fixing it in code would either change semantics (potential test break) or require new ctor flag (over-scope).

Why this over alternatives:
- **Alt A (full rewrite, unified `LearnableOperator` base class)**: high blast radius, no consumer demand, would require relocating logic between modules → breaks `.keras` archive serialization keys per LESSONS L94/L118. Rejected.
- **Alt B (deprecate `LearnableArithmeticOperator`+`LearnableLogicOperator`, keep only `LearnableNeuralCircuit`)**: removes user surface that exists in tests + sole consumer's analyze path. Rejected.
- **Alt C (relocate classes into one file)**: same `.keras` archive break risk. Rejected.

Constraint inputs:
- HARD: bare `@register_keras_serializable()` ties registered key to `__module__` — keep current module homes (LESSONS L94/L118).
- HARD: 78 PASS baseline must remain green.
- HARD: external consumer `train/latent_reasoning_vision/circuit.py` import path locked.
- SOFT: repo convention "package with factory.py → populated __init__.py" (FFN/memory/norms precedent in layers/CLAUDE.md).

### Available checkpoints
None yet — iter-1 will create `cp-000-iter1.md` at first EXECUTE step (per protocol).

## plan_2026-05-13_8c1dc6fd
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-13_8c1dc6fd/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-05-13
**Context**: review.md identified 4 bug-grade defects (B1, B2, I13, I8 LSP) plus medium/low issues and 13 refactor recommendations. Source state verified at every flagged line. Only in-repo callers of `MannLayer` and `SOMLayer*` are research/experimental code.
**Decision**: Do all 4 must-fix bugs + R1/R3/R5/R6/R13/I18 (small, local, low-risk). Defer R4/R8/R9/R10/R11/R12 (structural / public API change).
**Trade-off**: A clean correctness pass and stable public API **at the cost of** carrying duplicate NTM addressing implementations (DifferentiableAddressingHead vs NTMReadHead vs MannLayer._calculate_head_addressing) until the user approves a structural pass.
**Reasoning**: Structural merges (R8/R10) break public surface (970+ LOC removed, deprecation cycles) and require a separate explicit approval. Doing them silently here would violate the "minimal surgical edits" preference. The duplication is inert (not a correctness issue) and can be fixed in a follow-up plan.
**Anchor-Refs**: pending (no code anchors required for this plan — all fixes are surgical and self-explanatory at the call site).

### D-002 | PLAN (v2) | 2026-05-13
**Context**: User authorized maximum-effort expansion ("YES LETS GO, MAXIMUM EFFORT"): pull in deferred R4, R7, R8, R9, R10, R11, R12. For R10 (MannLayer rewrite), instructions said "pick lower-risk path between LSTMCell rewrite (R4) and collapse-to-factory-over-NTM, justify".
**Decision**: Lower-risk path is **additive** factory: introduce `create_mann()` in new `factory.py` that returns a `NeuralTuringMachine` configured with MANN-equivalent knobs; **keep `MannLayer` class as-is** (with I13 fix from step 3). Do NOT delete or rewrite MannLayer's internals — that would either (a) silently change numerical behavior of `qwen3_mega.py` (semantics differ between MannLayer.call's concat-output and NTM's `Dense(output_dim)` projection of cell state), or (b) trigger user-visible test failures and the autonomy leash. Same additive philosophy for R9: `SOM2dLayer` becomes a registered subclass alias (`_SOM2dLayerImpl`) returned by the `create_som_2d` factory — preserves `isinstance(x, SOM2dLayer)`, ctor signature, `get_weights_as_grid` method, and the 6 tests in `tests/test_models/test_som/test_model.py`. The user's caveat "do not fight irrecoverable breaks" governs.
**Trade-off**: A clean public-surface refactor (R7-R12 done) and clean dead-code removal (R8) **at the cost of** keeping `MannLayer` class (and `SOM2dLayer` alias) in the public surface as legacy entry points alongside the new factory. The duplication is small and inert.
**Reasoning**: The user explicitly invited the autonomy leash to fire on irrecoverable structural breaks. The factory-as-additive-layer pattern lets every consumer pick: legacy class (MannLayer / SOM2dLayer) or factory (create_mann / create_som_2d). For R4: by using `create_mann()` callers automatically get the LSTM**Cell**-based RNN that `NeuralTuringMachine` already uses internally — R4's perf benefit is captured for new callers without rewriting `MannLayer.call`. R8 deletes 970 LOC of `base_layers.py` outright (no in-src callers, only `test_base_layers.py` consumes it — both go). R12 prunes enum values that no caller uses (`LOCATION/SPARSE/TEMPORAL/LEARNED`) — only `HYBRID` and `CONTENT` kept, matching usage.
**Anchor-Refs**: pending (will add `# DECISION plan_2026-05-13_8c1dc6fd/D-002` anchor in `factory.py` create_mann body documenting the BC contract).

## plan_2026-05-13_a40908e7
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-13_a40908e7/D-NNN` anchor exists in source)
-->

## plan_2026-05-13_a1c9a52d
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-13_a1c9a52d/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-05-13
**Context**: `layers/memory/` (4 files: MannLayer + 3 SOMs, empty `__init__.py`) and `layers/ntm/` (4 files: interfaces + base layers + baseline NTM, populated `__init__.py` with 29-name public API) are sibling layer subpackages. 21 known call sites split across 12 deep-submodule `layers.memory.*` imports + 9 `layers.ntm.*` imports (2 top-level, 7 submodule).
**Decision**: Adopt Option A — flat-siblings layout. `git mv` the 3 NTM files into `memory/`, then create a 4-file shim package at `layers/ntm/` (rewritten `__init__.py` + 3 thin submodule shims) that re-exports from `dl_techniques.layers.memory`. Populate `memory/__init__.py` as the canonical public surface (NTM names + MANN + 3 SOM classes).
**Trade-off**: zero-blast-radius backward compatibility (3 shim files + 1 rewritten `__init__.py`) **at the cost of** keeping a grab-bag flat namespace under `memory/` (NTM files sit alongside SOM files; no internal subpackage grouping).
**Reasoning**: Option A is 3 `git mv` + 4 shim files + 0 source modifications across the 21 call sites. Option B (subpackages `memory/ntm/`, `memory/som/`) would require rewriting intra-package relative imports across 4 files and forces an uglier 2-level path `dl_techniques.layers.memory.ntm.baseline_ntm`. Option C (no merge) does not satisfy the goal. Serialization key risk (`__module__`-keyed `@register_keras_serializable` registration changes when files move — LESSONS L118) is mitigated by verifying that all in-repo `.keras` files live under `results/` (gitignored runtime output, not fixtures).
**Anchor-Refs**: (no in-source anchors required — pure relocation, no behavioral decision baked into call())

### D-002 | PLAN → PLAN (user revision) | 2026-05-13
**Context**: User rejected the shim approach in D-001. Backward-compat shim package at `layers/ntm/` is unwanted technical debt; user prefers a clean break and explicit caller rewrites.
**Decision**: Drop all shim creation. `git mv` the 3 NTM files into `memory/`, rewrite all 9 `layers.ntm.*` caller import statements to use `layers.memory.*`, then `git rm` the `layers/ntm/` directory entirely. Populate `memory/__init__.py` as the canonical public surface and add README.md.
**Trade-off**: clean single-package layout with zero legacy shims **at the cost of** touching 9 caller files (broader edit blast radius) and a hard break for any out-of-tree caller that imports `dl_techniques.layers.ntm` (none known, but the grep guarantee is repo-scoped).
**Reasoning**: User explicitly prefers atomic cutover over compatibility cruft. The 9 caller edits are mechanical, all enumerated in F-002, and reviewable as a single diff. Top-level callers (`models/ntm/model.py:27`, `train/ntm/train_ntm.py:8`) now import from `dl_techniques.layers.memory` directly (`NTMCell`, `NTMConfig`, `create_ntm` re-exported from `memory/__init__.py`). Serialization risk unchanged from D-001 (no in-repo `.keras` fixtures).
**Anchor-Refs**: (none — pure relocation + import rewrites)

### D-003 | REFLECT → CLOSE recommendation | 2026-05-13
**Context**: All 7 plan steps executed cleanly. Zero fix attempts. 287/287 scoped tests pass after `make clean`. All 7 SCs PASS. Diff review found no debug artifacts, no print(), no TODOs. Scope drift: none. Simplification checks: all 6 clear. Falsification signals A, B, C: none fired.
**Decision**: Recommend CLOSE.
**Trade-off**: closing now **at the cost of** not running the full repo test suite (out of scope per plan; user-instructed to avoid `make test` routinely; scoped suite covers all touched code).
**Reasoning**: Verification is complete against the documented criteria. The merge was structural (relocation + import rewrites + new README) with mechanical execution. No behavioral changes. Top-level public API re-exports all NTM names and verified by SC5 smoke. Devil's advocate: the only failure mode untested is downstream/notebook consumers importing `dl_techniques.layers.ntm` — A5 explicitly accepted by user as hard break. No further verification value at this iteration.
**Anchor-Refs**: (none)

## plan_2026-05-13_8e866056
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-13_8e866056/D-NNN` anchor exists in source)
-->
