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

## plan_2026-05-13_16ac1621
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-13_16ac1621/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE -> PLAN | 2026-05-13
**Context**: Five previous documentation tasks in this session built sibling benchmark files via a single research agent. The same pattern fits here: METRICS.md is a single greenfield doc with a well-defined inventory of 16 task families. Iterative-planner protocol followed for rigor since user explicitly invoked the skill.
**Decision**: Single research agent writes METRICS.md in one pass, with the task-metric inventory and style template as the brief. Update CLAUDE.md in a second step. No parallel split unless agent output is shallow.
**Trade-off**: simpler dispatch and a single coherent voice across the doc, **at the cost of** longer single-agent turn and risk of context exhaustion on the agent side.
**Reasoning**: The four prior benchmark files each fit in one agent without truncation. Pre-mortem covers the truncation scenario with a clear STOP trigger. Splitting into 2-3 agents would produce stylistic seams and double the verification surface for no clear gain.

### D-002 | PLAN | 2026-05-13
**Context**: Math notation choice for the doc.
**Decision**: Use plain-text math (`AbsRel = (1/N) sum |d - d_hat| / d`, multi-line in fenced code blocks) instead of LaTeX `$...$`.
**Trade-off**: universal readability in any markdown viewer, **at the cost of** less typographic polish.
**Reasoning**: The repo's IDE setups vary (PyCharm without KaTeX plugin, GitHub web, plain code editors). The sibling benchmark files avoid LaTeX. Plain-text math is the lowest-common-denominator that still encodes the formula unambiguously.

## plan_2026-05-13_03176394
### D-001 | EXPLORE → PLAN | 2026-05-13
**Context**: `tversky_projection.py` and `kan_linear.py` are top-level layer files but conceptually are alternative projection / FFN-style linear layers. The user wants them integrated into the `ffn/` factory.
**Decision**: Move both files into `layers/ffn/` and register both in the factory; do NOT modify either layer's `call()` body (per user constraint).
**Trade-off**: Factory inclusion of Tversky at the cost of factory consumers receiving a layer whose `call()` is rank-2-only — must be advertised in description + README.
**Reasoning**: Modifying `call()` to be rank-generic is out of scope. The 2D-only limitation is real but acceptable as a documented quirk; consumers building rank-3 transformers will simply not pick `'tversky'`.

### D-002 | PLAN | 2026-05-13
**Context**: Existing KAN consumers import via deep path `dl_techniques.layers.kan_linear`. After the move that path no longer exists.
**Decision**: Update all 7 consumer sites to the new deep path `dl_techniques.layers.ffn.kan_linear`. Do NOT add a shim re-export at the old path.
**Trade-off**: Explicit breakage of stale imports at the cost of one-time edit churn across 7 files.
**Reasoning**: A shim accumulates as technical debt; with only 7 sites the explicit update is cleaner and matches LESSONS file-split refactor pattern (`plan_2026-05-11_46ecfa0b`).

### D-003 | REFLECT iter-1 | 2026-05-13
**Context**: All 9 plan steps completed single-pass. Scoped pytest (492 tests across `test_ffn/`, `test_kan/`, `test_window_attention.py`) green in 130s. All 8 success criteria PASS.
**Decision**: Recommend CLOSE — no regressions, no scope drift, no simplification blockers, no new orphan decision anchors introduced.
**Simplification check (6 checks)**:
1. Necessity: every change is required by the goal (relocation + factory wiring). PASS.
2. Smallest change: file moves via `git mv`, additive registry entries, surgical edits. No new abstractions. PASS.
3. No premature generalization: did not add a shim/alias module. PASS.
4. No dead code: every added symbol referenced by tests or factory. PASS.
5. Forbidden patterns: none (no wrappers, no config toggles, no exception swallowing). PASS.
6. Complexity budget: files +1 (Tversky test) / 3 budget, abstractions 0/2, lines ~+225 (test code dominates — non-runtime). PASS.
**Devil's advocate**: One thing that could still be wrong — README's "fourteen different FFN layer types" count is now correct (12 → 14), but the registry actually had 12 before this plan and now has 14; verified by `len(get_ffn_info())` returning 14 via the updated `test_layer_count_in_info` test. Risk: nil.
**Validator pre-existing errors**: 26 `anchor-orphan` / `anchor-unknown-plan` ERRORs are all in unrelated source files (routing_probabilities.py, clifford_block.py, train_cliffordnet_nlp.py, gpt2/*, nam/*, depth_anything/*, common/nlp.py). None introduced by this plan. Pre-existing repo hygiene debt — not a CLOSE blocker for this plan.
**Trade-off**: Accept "1.5h full suite not run" at the cost of scoped-only verification; mitigated by scoped suite covering every touched module + cascade (single_window_attention via ffn.kan_linear).
**Anchor-Refs**: none — this plan introduces no `# DECISION` anchors (no trigger conditions applied to the relocation).

## plan_2026-05-12_13c70aed
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-12_13c70aed/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-05-12
**Context**: User asked for MRL + auxiliary L2-normalized embedding head on `CliffordNetLMUNet`. The model is a causal U-Net language model with `"logits"` as its sole output key — a SYSTEM.md invariant for CLM trainers. The existing head ingests `h_top: (B, T, base_channels)` after `head_norm`. EXPLORE confirmed (a) no existing matryoshka utilities in `src/dl_techniques`, (b) MRL plugs in purely post-decoder (zero blast radius on the encoder/bottleneck/decoder stack), (c) generation probe reads `"logits"` only — unaffected if we keep the primary largest-width head named `"logits"`, (d) `prepare_dict_keyed_compile` is the established mechanism for routing dict-keyed loss/metrics through subclassed Keras 3 models.

**Decision**: Add MRL by walking a `mrl_widths` list (default per-variant: halving from `base_channels` to floor 16) at the head; emit flat-keyed outputs `{"logits", "logits_w{w}", "embedding_w{w}"}`. Trainer wires per-key `MaskedCausalLMLoss` instances via Keras's `loss=` + `loss_weights=` dicts; `prepare_dict_keyed_compile` is extended to accept a list of output keys. Embedding head is identity-by-default (slice + L2-norm); learnable `Dense(C0)` opt-in via `--emb-head`. Embedding side outputs never participate in loss.

**Trade-off**: A multi-head LM (N untied paths or N tied biases plus per-width LayerNorms) at the cost of slightly higher parameter + activation count and a wider model output dict.

**Reasoning**: Pure additive change. Causality is structural (slicing + per-position projection). `"logits"` primary-head name is preserved → zero impact on the generation probe and on other CLM trainers. Reusing `MaskedCausalLMLoss` avoids new custom_objects. Halving widths down to 16 mirrors common matryoshka recipes and gives consistent semantics across non-power-of-2 base variants. The `loss_weights` dict is the cleanest plan A; a self-contained matryoshka loss class (summing CE internally) is the documented fallback if Keras's dict routing misbehaves on subclassed models (Pre-Mortem Scenario A). Alternatives rejected: nested dict outputs (complicates compile path), contrastive loss for the embedding head (out of scope), per-width Dense embedding projection (dead weight without supervision signal).

**Anchor-Refs**: `src/dl_techniques/models/cliffordnet/lmunet.py` (head section to be edited at ~lines 626-641 in the existing file).

### D-002 | PLAN (revision) → EXECUTE | 2026-05-12
**Context**: PC-PLAN presentation surfaced Assumption A7 (default width sequence at non-power-of-2 `base_channels`). User explicitly chose width rule: **"Power-of-2 anchored, base preserved"** — largest width is `base_channels` preserved as-is (even when not a power of 2); remaining widths are strict powers of 2 strictly less than `base_channels`, descending, floor 16. This replaces the original "halving from base_channels" rule in plan v1.

**Decision**: Apply the new rule to:
1. Step 1 validation: first element MUST equal `base_channels` (may be non-power-of-2); every subsequent element MUST be a strict power of 2 AND strictly less than `base_channels`.
2. Step 1b `MODEL_VARIANTS` defaults: nano `[128,64,32,16]`; mini `[192,128,64,32,16]`; base `[384,256,128,64,32,16]`; large `[512,256,128,64,32,16]`; xl `[768,512,256,128,64,32,16]`.
3. Helper `_default_mrl_widths(base, floor=16)` computes `[base] + [2**k for k in descending powers-of-2 < base, terminating at >= floor]`.
4. Test `test_mrl_default_widths_per_variant` updated to the new table.
5. Loss weighting default stays `uniform` (weights `[1.0] * N`, sum=N — unchanged from plan v1).

**Trade-off**: Preserves full `base_channels` capacity at the largest head at the cost of a slightly irregular descent on non-power-of-2 variants (e.g. `192 → 128` is not a halving). Compared to strict-power-of-2-only (which would drop the `base_channels` head when `base ≠ 2^k`), this keeps the primary head matching the model's actual width and keeps `"logits"` semantically identical to the existing single-head path. Compared to plan v1's halving, this gives cleaner power-of-2 representations at all sub-widths and aligns with common matryoshka recipes that prefer power-of-2 truncations for hardware efficiency.

**Reasoning**: User chose to preserve base_channels capacity at largest width. Strict-power-of-2-only would silently drop capacity at the primary head for mini/base/xl (and break the SYSTEM.md `"logits"` invariant if the primary head no longer matched `base_channels`). Halving was a reasonable default but produced non-power-of-2 sub-widths (e.g. 192→96→48→24) that are less hardware-friendly and less aligned with matryoshka conventions. The "anchored" rule resolves both concerns cleanly.

**Anchor-Refs**: `src/dl_techniques/models/cliffordnet/lmunet.py:415` (in-code anchor `# DECISION plan_2026-05-12_13c70aed/D-002` at the MRL widths resolution + validation block).

### D-003 | EXECUTE → REFLECT | 2026-05-12
**Context**: All 5 plan steps executed in a single pass with zero fix attempts. Full scoped pytest run: 43/43 PASS in 124.22s. SC1-SC8 all green. Diff review clean (no debug artifacts). Scope drift: zero (exactly 4 planned files modified). `validate-plan.mjs` errors are all pre-existing orphan anchors from prior plans, unrelated to this plan's edits.

**Decision**: Route REFLECT → CLOSE.

**Trade-off**: Closing now at the cost of not running the full ~1.5h `make test` suite. SC1-SC8 are scoped to the modules edited; SC6 demonstrates the cross-trainer mechanism is preserved; the per-edit blast-radius scoring (LOW/MED) and the manifest-vs-plan equality argue against broader breakage. Net trade-off accepted given user pre-push policy.

**Reasoning**: 6 Simplification Checks pass. Pre-Mortem Scenarios A/B/C all averted via dedicated tests. Devil's advocate: only "unknown" is MRL convergence dynamics, which is explicitly out of scope and recorded in Not Verified. No regressions, no scope drift, no simplification blockers. Recommendation justified.

**Anchor-Refs**: none (no new in-code anchors introduced for this plan; trigger conditions in references/decision-anchoring.md not met for slicing/L2-norm additive code).
