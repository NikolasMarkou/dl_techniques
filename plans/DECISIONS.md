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

## plan_2026-06-15_2485b951
### D-001 | EXPLORE → PLAN | 2026-06-15
**Context**: Source-verified review of all 14 norm classes found 4 correctness bugs (B1 squeeze crash, B3 regularizer-not-deserialized x2, B4 axis mutation, B2 degenerate BandLogitNorm), a package-wide missing built-guard (C1), a broken `__init__.py` (C2), and low-severity factory/doc gaps (F1, D1). User constraint: fix-only, no functionality expansion, no eager, full Keras-3 compliance.
**Decision**: Apply a tight 10-step remediation grouped by file (correctness bugs first, then mechanical built-guard sweep, then factory/init/docs, then pin with tests). Author plan directly (orchestrator holds line-accurate verified context) rather than spawn plan-writer blind to source.
**Trade-off**: Maximum coverage of confirmed defects **at the cost of** leaving design-level redesigns (B2 math, FW-2/3/4) out of scope.
**Reasoning**: "Fix what's there, don't expand" rules out redesigns; mechanical fixes are zero-regression (LESSONS-confirmed across 4 prior sub-package sweeps).

### D-002 | PLAN | 2026-06-15
**Context**: BandLogitNorm's `LayerNormalization(axis=-1)` applied to the `[...,1]` L2-norm tensor is mathematically degenerate (output ≡ 0 → constant scale). Used by `train/rms_variants_train/`. A real fix is a redesign.
**Decision**: Preserve the degenerate math; apply only the Keras-3 contract fixes (create sublayer in `__init__`, add built-guard, `super().build()` last) and DOCUMENT the limitation honestly in the docstring.
**Trade-off**: Honest contract + docs **at the cost of** the layer remaining non-adaptive in practice.
**Reasoning**: Changing the math = functionality redesign = out of scope + would alter a production experiment harness's variant. Anchor: textual (see summary Decision Anchors Registry at CLOSE).

### D-003 | PLAN | 2026-06-15
**Context**: Factory injects `epsilon=1e-6` via `setdefault` while the 8 custom classes default to `1e-7`; Keras `layer_norm`/`batch_norm` default to `1e-3` (already shifted to 1e-6 by the factory).
**Decision**: Leave the factory's 1e-6 default behavior unchanged; clarify in the factory docstring that it is factory-imposed and may differ from a class's own default.
**Trade-off**: Numeric stability of already-saved factory-built models **at the cost of** factory/class default divergence (documented, not silent).
**Reasoning**: Silently changing epsilon would alter numerics of existing trained/saved models — higher risk than the cosmetic divergence. Revisit only if user requests alignment.

### D-004 | PLAN | 2026-06-15
**Context**: Deferred item FW-1 (register PolarWeightNorm in the norms factory) from plan_2026-05-29_4538aa62.
**Decision**: WON'T-FIX BY DESIGN — PolarWeightNorm is a *weight* reparameterization (Dense replacement: `units`, radius+angles), not an activation-normalization layer; it does not fit `create_normalization_layer`'s contract. Already documented as not-registered in `layers/CLAUDE.md` + module docstring.
**Trade-off**: Contract purity of the norm factory **at the cost of** one-line factory convenience for PolarWeightNorm.
**Reasoning**: Registering it would force a semantically wrong key (a transform layer among normalizers). FW-2/3/4 + kernel-cache are new functionality → out of scope.

<!-- Schema example — DO NOT REMOVE. Real entries follow this shape.
     See references/file-formats.md "Entry Schema by Type" for required fields per entry type.
     In-code anchors carry the plan-id prefix: `# DECISION plan_2026-06-15_2485b951/D-NNN` (see references/decision-anchoring.md).

### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-15_2485b951/D-NNN` anchor exists in source)
-->

## plan_2026-06-15_0205772c
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-15_0205772c/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-15
**Context**: Audit of `layers/activations` (4 parallel explorers, then orchestrator source-verification). Graph-safety found already sound (no `tf.*`/`.numpy()`/`int(tensor)`); several explorer "graph-unsafe" claims (`len(inputs.shape)`, `keras.activations.elu`) refuted as static-rank/traceable. Real defects: missing `if self.built: return` guards (4 weight-bearing builds = real weight-doubling), factory/class default divergence (thresh_max trainable_slope, differentiable_step shift_constraint), one mutable-default anti-pattern, and `MonotonicityLayer._sigmoid` not actually monotonic.
**Decision**: Fix the 7 build guards + DifferentiableStep mutable defaults + factory default alignment + `_sigmoid` flexibility math; exclude all refuted/expansion items; verify against 367/0 baseline + new monotonicity test.
**Trade-off**: Targeted contract+correctness fixes **at the cost of** leaving documented-but-out-of-scope items (ProbabilityOutput factory registration, type_config serialization hardening, relu_k TypeError style) for a follow-up — honoring the user's fix-only/no-expansion directive.
**Reasoning**: Class defaults are source-of-truth (test_threshmax asserts `trainable_slope=False`); factory keys + MonotonicityLayer have zero src callers so blast radius is minimal. Aligning factory to class fixes silent mis-config without changing any real caller. `_sigmoid` fix `flexibility=1/(n-1)` is the minimal in-spirit change that provably restores monotonicity.
**Anchor-Refs**: none (mechanical/contract fixes; no in-source DECISION anchors warranted — all changes are self-explanatory contract conformance)

## plan_2026-06-15_9dbb87c1
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-15_9dbb87c1/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-15
**Context**: continuous_sin_cos:237 and continuous_rope:224 call `ops.convert_to_numpy(min_coords)` inside `call()` under default `assert_positive=True`, breaking graph trace. The block only emits `logger.warning` on negative coords — no functional effect.
**Decision**: Delete the eager warning block; keep `assert_positive` accepted/stored/serialized as a documented no-op for config back-compat.
**Trade-off**: graph compatibility + zero behavior change for valid inputs **at the cost of** losing a debug-only negative-coordinate warning.
**Reasoning**: A graph-safe assertion would add complexity for a log line nobody consumes; removal is the KISS fix. assert_positive kept so existing configs/`.keras` still load.

### D-002 | EXPLORE → PLAN | 2026-06-15
**Context**: PositionEmbeddingSine2D, ModernBertEmbeddings, AlbertFactorizedEmbedding exist + are `@register_keras_serializable` but are NOT in `factory.py`; `layers/CLAUDE.md` already implies ModernBERT factory support. User explicitly asked for factory + doc wiring.
**Decision**: Register all 3 (standard ctors) in the factory with exact param lists read from each `__init__`.
**Trade-off**: closing the factory/doc gap + honoring the user instruction **at the cost of** 3 more registry keys to maintain.
**Reasoning**: Factory only constructs (never calls) so a non-standard `call()` sig is irrelevant; this is wiring of existing classes, not functional expansion. Pre-mortem Scenario B drops any key whose ctor proves non-standard.

### D-003 | EXPLORE → PLAN | 2026-06-15
**Context**: continuous_rope `compute_output_shape` returns `dim` but actual `call()` output is `dim/2` (phase angles). 0 callers.
**Decision**: Fix compute_output_shape + docstring to the true phase width; do NOT change `call()` output.
**Trade-off**: correct shape contract + conventional RoPE phase semantics **at the cost of** the docstring's original `(..., dim)` promise being corrected downward.
**Reasoning**: A wrong `compute_output_shape` is worse than none (LESSONS); dim/2 phase width is the standard RoPE design; changing call() output would be riskier with zero benefit (no callers).

### D-004 | EXPLORE → PLAN | 2026-06-15
**Context**: multi_axis_rope uses `register_keras_serializable(package="dl_techniques.layers")` while siblings use the default package.
**Decision**: Leave the package= string unchanged.
**Trade-off**: preserve loadability of already-saved `.keras` models **at the cost of** cross-layer package-string inconsistency.
**Reasoning**: Changing a package= string changes the registration KEY and breaks deserialization of existing saved models (attention-plan C2 lesson).

### D-005 | EXPLORE → PLAN | 2026-06-15
**Context**: Several cosmetic items found (zeros+assign vs Constant init; patch stride=None config asymmetry; tests using custom_objects / tf.GradientTape).
**Decision**: Skip them; fix only functional/contract defects.
**Trade-off**: stay in scope + minimize regression surface **at the cost of** leaving minor style inconsistencies.
**Reasoning**: User said fix what's there / no expansion; these have no functional/contract impact.

## plan_2026-06-14_5e80bd3e
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-14_5e80bd3e/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-14
**Context**: Prior plan plan_2026-06-14_7384c2e3/D-001 set the mixtures training guards to `if training is True:` — graph-safe but it SKIPS training-only side-effects when `training` is a symbolic `tf.Tensor` (custom @tf.function train loop). The user wants this foot-gun fixed so side-effects fire under symbolic-True while staying graph-safe.
**Decision**: Replace the `training is True` gate with a shared `resolve_training_factor(training, dtype)` helper + tensor-MASKING: full unmasked side-effect on the python-True fast path (exact), and a `cast(training)`-scaled side-effect on the symbolic path so symbolic-False is a true no-op.
**Trade-off**: Support symbolic training via masking **at the cost of** running a zeroed op (assign of zero delta / add_loss of 0) on the symbolic-False runtime path, plus one new shared helper + a slightly larger `_update_centroids`.
**Reasoning**: Masking is graph-safe (no tensor→bool coercion; all python branches are on identity/type, static at trace time) and cleaner than `keras.ops.cond` for side effects (cond branches must return matching structures; add_loss/.assign inside cond is awkward). The `isinstance(factor, float)` fast path guarantees ZERO numeric regression on the python-True training path. REJECTED `ops.cond` (awkward for side effects), REJECTED always-mask (would perturb python-True numerics). This SUPERSEDES plan_2026-06-14_7384c2e3/D-001's tensor-skip trade-off. Empirically validated (findings F-PROTO).
**Anchor-Refs**: `src/dl_techniques/layers/mixtures/kmeans.py` (Step 2), `src/dl_techniques/layers/mixtures/gmm.py` (Step 3), `src/dl_techniques/layers/mixtures/radial_basis_function.py` (Step 4)

### D-002 | PLAN | 2026-06-14
**Context**: The factor logic (None/False→skip, True→1.0, tensor→cast) is subtle and needed at 3 sites.
**Decision**: Put it in ONE shared helper `resolve_training_factor` in `utils/tensors.py`, unit-tested once.
**Trade-off**: One new util function (new public surface) **at the cost of** vs inlining 3×; chosen for DRY + a single tested source of truth.
**Reasoning**: 3 call sites = earned abstraction; subtle correctness logic should not be duplicated. Repo convention is per-site `ops.cond`, but that idiom doesn't fit side-effects; a tiny resolver is the better fit.
