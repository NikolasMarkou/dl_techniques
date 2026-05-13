# Consolidated Findings
*Cross-plan findings archive. Entries merged from per-plan findings.md on close. Newest first.*

<!-- COMPRESSED-SUMMARY -->
## Summary (compressed)
*Auto-compressed. Read full content below for plan-specific detail.*

### Key Findings
- **TreeTransformer (`models/tree_transformer/`)** is structurally sound — save/load + gradient flow + MLM-wrapper integration all correct. Four real bugs fixed in plan_3c3ed037: B-1 fp16 NaN (dtype-aware mask sentinel `-1e4` under float16, plus fp32 cast on GroupAttention DP log/matmul/exp block); B-3 explicit `attention_mask` honored in dict input; B-4 `load_pretrained_weights` via `weight_transfer.load_weights_from_checkpoint` (Keras 3.8 `by_name=True` broken); B-5 `PRETRAINED_WEIGHTS={}` + `NotImplementedError` (no public checkpoints). Trainer `src/train/tree_transformer/{pretrain,finetune}.py` mirrors `bert/`. Anchor: `model.py:318` D-001. **Trainer config MUST pass `pad_token_id=config.pad_token_id` (tiktoken cl100k_base = 100266) to encoder — model default 0 is silent semantic bug.** Aligned to `bert/`/`resnet/` conventions in plan_0a5779e8: bare `create_tree_transformer(variant, ...)` factory added, `__init__.py` trimmed to 3 names (`TreeTransformer`, `create_tree_transformer`, `create_tree_transformer_with_head`; internal layer classes remain importable from `.model` for `nam/` consumers), and `from_variant(pretrained=True)` now raises `NotImplementedError` loudly instead of silently random-initializing (D-001 anchor at `model.py:1133`, narrowed try/except to `(IOError, OSError, ValueError)`).
- **TinyRecursiveModel (`models/tiny_recursive_model/`)** — save/load clean. B-3 Q-learn lookahead `training=False` + `keras.ops.stop_gradient` on `target_q`; B-5 inference halts on learned signal. `hrm_loss`/`HRMMetrics` API-compatible with TRM output schema. Anchor: `model.py:370` D-001 (plan_e6309bd5).
- **`keras.ops.expand_dims(axis=tuple)` works** on Keras 3.8 / TF 2.18 eager + `@tf.function` (B-1 false-positive in plan_e6309bd5).
- **DepthAnything** is now full-feature — real ViT encoder, DPTDecoder linear default + `upsample_factor`, weight-shared frozen teacher via `clone_model`, semi-sup `train_step` (FAL + L1 pseudo-label stop-gradient), on-step EMA via `TeacherEMACallback`, `from_pretrained_encoder(path)`, `StrongAugmentation` + dynamic cutmix.
- **Keras 3 / TF 2.18 idioms**: `keras.random.*` (NOT `keras.ops.random.*`); `keras.ops.*` for backend-agnostic ops; `@keras.saving.register_keras_serializable()` + `get_config()` round-trip; `dl_techniques.utils.logger` only.
- **Save/load on subclassed Models wrapping inner Models**: weights drop unless outer class overrides `save_own_variables` / `load_own_variables` (D-004).
- **Keras 3.8 `model.load_weights(path.keras, by_name=True)` is broken** — use `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`.
- **AdamW double weight decay footgun**: never combine `AdamW(weight_decay=...)` with `kernel_regularizer=L2(weight_decay)`.
- **CLM training**: use `train.common.nlp.estimate_clm_steps_per_epoch`; `min_article_length=0` correct for packed pipelines.
- **Two-optimizer differential-LR**: register one with `super().compile`; apply second manually inside `train_step` via name-prefix variable routing (leading-component match).
- **`keras.ops.cond` traces BOTH branches under `tf.function`** — multiply-by-zero for compute-amount differences.
- **Frozen state in layers**: `add_weight(trainable=False)` or numpy-on-self — never plain tensors in `build()` (FuncGraph dead-tensor).
- **BERT (`models/bert/`)** aligned to resnet/tree_transformer template in plan_9357982a: `create_bert` bare-encoder factory added; `__init__.py` trimmed to 3-name surface `{BERT, create_bert, create_bert_with_head}` (drop `create_nlp_head` re-export); `_download_weights` raises `NotImplementedError`, `from_variant` try/except narrowed to `(IOError, OSError, ValueError)` (D-001 anchor at `bert.py:687`); docstring/README path fixed to `dl_techniques.layers.nlp_heads`. 28/28 pytest PASS, 0 fix attempts.
- **AccUNet** requires H,W divisible by 16; validation in `call()` raising `ValueError` (plan_bdb2c84d D-001/D-002).
- **`SegmentationWrapperLoss`** is canonical save/load-friendly segmentation loss; `compile=False` workaround removed (plan_17633038 D-002).

### Key Decisions
- **D-001 plan_3c3ed037 (TreeTransformer bundle)**: 4 model bugs + Pattern-3 trainer in one iteration — 5 new files / +950 LOC at the cost of 2 over file-budget; trainer depends on Step 5 re-exports and Step 2 attention_mask honoring, so splitting would force pinning to broken imports.
- **D-001 plan_e6309bd5 (TRM bundle)**: bug fixes + factory + tests + trainer in one plan — at cost of larger review surface; B-5 testable only with same harness as trainer eval path.
- **Pseudo-label loss**: plain L1 + `stop_gradient`, not `compute_loss` against synthetic mask (plan_54e6e303 D-002).
- **Encoder weight-loading**: keep `--pretrained-encoder-weights` + `--init-from` distinct (plan_54e6e303 D-003).
- **D-004 (save_own_variables override)**: canonical Keras-3 fix when `.keras` round-trip drops sub-Model weights.
- **D-003 (Keras-3 canonical train_step)**: `compute_loss(x,y,y_pred)` adds `self.losses` internally — no manual regularization addition.
- **D-005 (StrongAugmentation graph-mode safety)**: symbolic gate; `keras.random.*` not `keras.ops.random.*`.
- **CLM metrics architecture**: math in `dl_techniques/metrics/`; list in `train/common/nlp/build_clm_metrics()`; fresh instances each call.
- **`current_phase` / `_global_step` counters**: `add_weight(trainable=False, dtype="float32")` — int32 fails CPU/GPU device placement.
<!-- /COMPRESSED-SUMMARY -->

## plan_2026-05-13_a2b0f17b
### Index
- [prior-work-audit](plan_2026-05-13_a2b0f17b/findings/prior-work-audit.md) — `plan_2026-05-13_e52a5ac8` already double-checked the same review; 6 fixes shipped (commit `b562bd0`); most other items deferred for empirical reasons. Genuinely new items enumerated.

### Key Constraints
- **HARD**: Do not break consumer `src/train/latent_reasoning_vision/circuit.py:122` (depends on current `LearnableNeuralCircuit` shape contract + post-training `operation_weights` access).
- **HARD**: Do not break `.keras` deserialization for already-saved models — rules out `softplus(temperature)` reparam (LESSONS L94/L118).
- **SOFT**: Match prior plan's discipline — empirical verification BEFORE implementation; defaults preserve back-compat.
- **GHOST**: User's emphatic "implement everything MAXIMUM EFFORT" reads as overriding prior empirical conclusions, but those conclusions are still load-bearing — no new evidence has invalidated them. Need explicit override before re-litigating.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-13_e52a5ac8
### Index

- **F-001 Empirical verification of prior-review claims** — `findings/verify_claims.py` script + output below.
- **F-002 Test sensitivity scan for proposed fixes** — which tests would break for each candidate fix.
- **F-003 Cross-plan precedent** — prior plan_2aaad563 already addressed same package; LESSONS L38 explicitly rules unary degeneracy out-of-scope.

### F-001: Empirical results (one per prior-review claim)

| Claim | Verdict | Evidence |
|---|---|---|
| **C1** `LearnableNeuralCircuit` default `use_residual=False` while `CircuitDepthLayer` default is `True` | **CONFIRMED** | `inspect.signature` shows mismatch; default-config ‖y‖/‖x‖ = 0.38, with `use_residual=True` = 1.69. |
| **C2** Input-side routing shrinks signal | **CONFIRMED, MILDER than claimed** | Ratio 0.38 (not 1/N²). Real but recoverable with residual. |
| **C3** Stacked `LearnableLogicOperator` collapses to constant | **CONFIRMED, SEVERE** | std: 1.759 → 0.056 → 0.003 → 0.000 across 3 layers. Unusable as building block. |
| **C4** Temperature gradient is exactly 0 when raw value < 1e-7 | **CONFIRMED** | At temp=-0.5, ∂L/∂T = 0.0 exactly. At temp=+0.5, ∂L/∂T = 0.31. |
| **C5** `_safe_divide` gradient blow-up near zero | **CONFIRMED but partially expected** | At x2=0, d/dx2 = 0 (discontinuity); at x2=1e-3, d/dx2 = -1e6. Huge gradients near zero are inherent to division — current behavior is *arguably correct*. Documentation/awareness fix more appropriate than code change. |
| **C6** `NOT` competes in same softmax as binary ops, silently drops x2 | **CONFIRMED — already documented** | Per LESSONS L38: "documented footgun, not a bug to fix in code". Defer. |
| **H4** `random_uniform` arch-init biases selection | **OVERSTATED** | 7-way softmax probs from `random_uniform(-0.05, 0.05)` init: max-min = 0.0137 (≈1.4% spread). Essentially uniform. Not worth changing. |
| **H10** `validate_logic_config` accepts `bool` as positive int | **CONFIRMED** | `validate_logic_config("circuit_depth", num_logic_ops=True)` passes silently. |
| **M2** `compute_output_shape` returns `None` for list-deserialized shape | **CONFIRMED** | `compute_output_shape([None, 32])` returns `None`; `(None, 32)` returns `(None, 32)`. Same in `LearnableArithmeticOperator` and `LearnableLogicOperator`. |
| **L4** Initializer round-trip broken | **FALSE POSITIVE** | `keras.initializers.get(serialized_dict)` handles dict form; `GlorotNormal` round-trips correctly. |

### F-002: Test sensitivity for proposed fixes

| Fix | Touches | Test impact |
|---|---|---|
| **C1** flip `LearnableNeuralCircuit.use_residual=True` default | `neural_circuit.py:389` | None — no test asserts default value; `test_with_residual_connections` passes both explicit values. |
| **M2** disambiguate list-shape in `compute_output_shape` | `arithmetic_operators.py:422`, `logic_operators.py:411` | None — no test currently calls `compute_output_shape([None, D])` directly. |
| **H10** reject `bool` in validator | `factory.py:215` | None — no test passes bool. |
| **H9** deepcopy `get_logic_info()` | `factory.py:175` | None — no test mutates the returned dict. |
| **C3** add `apply_sigmoid=True` flag | `logic_operators.py` (new ctor param) | None — default `True` preserves all existing behavior incl. `test_sigmoid_input_normalization`. |
| **L2** `logger.info` → `logger.debug` for noisy init logs | All four files | None — no test asserts on log content. |

### F-003: Cross-plan precedent

- **plan_2026-05-13_2aaad563** (the predecessor) already aligned this package: added `factory.py`, populated `__init__.py`, relaxed 4-D ghost constraint, documented unary footgun in README. Chose to NOT fix unary degeneracy in code.
- **LESSONS L38** (codified outcome): "Unary-input degeneracy in DARTS-style 'softmax over primitives' layers ... is a documented footgun, not a bug to fix in code."
- **LESSONS L20** (governing principle): "Verify reviewer claims empirically before applying fixes from a 'deep review'. Reviews contain false positives."
- **LESSONS L21**: "A correctly-flagged bug can mask a worse adjacent bug. Sweep the category."

### Key Constraints

- **HARD**: 107 PASS baseline must remain green (`tests/test_layers/test_logic/`).
- **HARD**: `src/train/latent_reasoning_vision/circuit.py` import path locked.
- **HARD**: bare `@register_keras_serializable()` — no module relocation (LESSONS L94/L118).
- **HARD**: `.keras` archive compatibility for existing checkpoints — any change to parameter semantics (e.g. softplus reparam of `temperature`) breaks reload. Defer such changes.
- **SOFT**: prefer additive over destructive — new ctor flags with backward-compatible defaults.

### Corrections

None.

## plan_2026-05-13_2aaad563
### Index
- F-001 Package structure & public surface
- F-002 Design intent & sibling-pattern alignment
- F-003 Code-quality / correctness issues
- F-004 Integration consumers
- F-005 Test coverage baseline

### F-001 Package structure
- `src/dl_techniques/layers/logic/` 4 files, ~1409 LOC src code:
  - `arithmetic_operators.py` (449) — `LearnableArithmeticOperator` (DARTS-style softmax over {add, multiply, subtract, divide, power, max, min}).
  - `logic_operators.py` (435) — `LearnableLogicOperator` (sigmoid-normalize then softmax over {and, or, xor, not, nand, nor} fuzzy-logic gates).
  - `neural_circuit.py` (525) — `CircuitDepthLayer` (MoE over logic+arithmetic experts with soft routing + soft combination), `LearnableNeuralCircuit` (D-deep stack of CircuitDepthLayer + optional LayerNorm).
  - `__init__.py` (0 bytes / empty).
- All 3 classes use bare `@keras.saving.register_keras_serializable()` (no package= arg).

### F-002 Design intent & sibling alignment
- The 3 layers share the same mathematical core: a softmax over learnable weights selects (or convex-combines) a primitive op from a discrete set. This is DARTS continuous relaxation.
- Sibling factory-bearing packages (`ffn/`, `norms/`, `embedding/`, `activations/`, `attention/`, `memory/`, `nlp_heads/`, `vision_heads/`, `vlm_heads/`) all expose `factory.py` + populated `__init__.py`. `logic/` is the only package with ≥3 homogeneous related classes lacking these conventions.
- `dl_techniques.layers.ffn.logic_ffn.LogicFFN` is *different*: a Dense-projection FFN that internally uses soft AND/OR/XOR. Not a duplicate. Distinguish in README.

### F-003 Code-quality / correctness issues
- **Unary footgun**: `LearnableArithmeticOperator.call(inputs=x)` sets `x2 = inputs = x`. For `subtract` → identically 0; for `divide` → identically 1. The softmax mixes these with other ops; the result is silently corrupted. Same shape in `LearnableLogicOperator` (`x2=x` ⇒ AND/OR/XOR all degenerate). Fix: raise on unary inputs when non-self-canceling ops are present, or document in README + `__init__` validation as a warning.
- **Strict 4-D constraint** in `CircuitDepthLayer.build()` and `LearnableNeuralCircuit.build()`: `if len(input_shape) != 4`. The actual `call()` body is rank-agnostic (uses `expand_dims` based on `ops.shape(inputs)`). The strict check is artificial and blocks NLP/seq use. Fix: relax to `len(input_shape) >= 2`.
- **Stray underscore** in `arithmetic_operators.py` line 7 docstring (`_   Architecture Search`).
- **Initializer round-trip asymmetry**: `get_config()` calls `keras.initializers.serialize(...)` but `__init__` uses `keras.initializers.get(...)`. `get` handles serialized dicts so round-trip works, but explicit `deserialize` in `from_config` would match memory/norms sibling conventions. Cheap.
- **`logger.info` in `__init__`** is fine per repo convention but noisy when the circuit is stacked deep — `logger.debug` would be more appropriate. Not blocking.
- **Manual sub-layer build inside parent build()** (`logic_op.build(input_shape)` inside `CircuitDepthLayer.build`) is verbose but safe in Keras 3. Keep.

### F-004 Integration consumers
- `src/train/latent_reasoning_vision/circuit.py` — only non-test consumer; imports `LearnableNeuralCircuit` via fully-qualified path. Will keep working under any non-relocation refactor.
- No model in `src/dl_techniques/models/` imports `layers.logic.*`.
- Library has no `create_logic_layer` factory; consumers must direct-import.

### F-005 Test coverage baseline
- `.venv/bin/python -m pytest tests/test_layers/test_logic/ -q` → **78 PASS / 0 FAIL** in 17.3s.
- Files: `test_arithmetic_operators.py` (377 LOC), `test_logic_operators.py` (435 LOC), `test_neural_circuit.py` (588 LOC). Each covers init, forward pass, training mode, serialization round-trip, and edge cases.
- No factory tests (no factory exists). No rank-relaxed shape tests (currently impossible — code rejects).

### Key Constraints

### HARD
- 78 PASS baseline must remain green.
- Bare `@register_keras_serializable` ties registered key to `__module__`. **Do not relocate classes between modules** — would invalidate any pre-existing `.keras` archives.
- External consumer path `dl_techniques.layers.logic.neural_circuit.LearnableNeuralCircuit` must remain valid.
- Repo convention (layers/CLAUDE.md): packages WITH `factory.py` populate `__init__.py`; packages WITHOUT factory keep `__init__.py` empty.

### SOFT
- Unary semantics of `subtract`/`divide` (degenerate to 0/1) — fix or document.
- Strict 4-D check artificial; relax to rank ≥ 2.
- Initializer round-trip explicit `deserialize`.
- Stray underscore in docstring.

### GHOST
- "Manual sub-layer build in parent build()" was needed pre-Keras-3.0. Now belt-and-suspenders. Keep as-is to avoid test churn.
- "4-D only" inherited from author's vision use case, not a math constraint.

### Exploration Confidence
- **scope**: deep — all 4 src files end-to-end, related FFN logic_ffn, sole external consumer, all 3 test modules, layers/CLAUDE.md, train/CLAUDE.md.
- **solutions**: constrained — repo conventions tightly lock surface; minimal vs. opportunistic options enumerated.
- **risks**: clear — relocation forbidden, public API unchanged, behavior changes scoped to rank relaxation + new factory + unary-input docstring/validation.

### Synthesis
`logic/` is functionally complete and well-tested (78 PASS) but underintegrated relative to its sibling layer packages. All three classes share an identical DARTS-style mathematical core, making them a textbook factory candidate. The integration improvement is: (1) add `factory.py` with `create_logic_layer(layer_type, **kwargs)` over `{'arithmetic', 'logic', 'circuit_depth', 'neural_circuit'}`; (2) populate `__init__.py` to re-export the four classes + factory (consistent with FFN/memory/norms); (3) relax the artificial strict 4-D check in `CircuitDepthLayer`/`LearnableNeuralCircuit` to `rank >= 2` (unblocks NLP/seq use without touching math); (4) fix small correctness/cosmetic issues (stray underscore, unary-divide/subtract footgun — at minimum documented in README, ideally a warning); (5) write `src/dl_techniques/layers/logic/README.md` describing math, classes, factory, integration patterns, and the documented limitation. No class relocation. No public API removal. Tests added incrementally for factory + rank-relaxation. Single-iteration scope per LESSONS line 28 (audit-driven full-coverage with file:line references + design notes per fix).

## plan_2026-05-13_8c1dc6fd
### Index

| # | Topic | File | Coverage |
|---|-------|------|----------|
| F1 | Review summary + triage | findings/review-summary.md | Bug list (B1/B2/I8/I13), medium fixes, refactor triage |
| F2 | Source state at flagged lines | findings/source-state.md | Verified line-refs in README, mann, baseline_ntm, som_nd, som_nd_soft |
| F3 | Tests + stale docs | findings/test-and-doc-state.md | Test paths, scoped pytest sets, doc regen plan |

### Key Constraints

**HARD**
- Keras 3 / TF 2.18 conventions: `@keras.saving.register_keras_serializable`, `keras.ops`, full `get_config()`.
- Use `.venv`. Never run `make test` (1.5h). Scope pytest to touched modules.
- User pushes commits themselves.
- B1, B2, I13 (dead memory_matrix), I8 (LSP) MUST be fixed (user requirement).
- Centralized logger; no prints.

**SOFT**
- Match repo convention for serialization in get_config (`serialize` always).
- Prefer minimal, surgical edits. Avoid breaking class signatures unless user approves.
- Default-do list: R1 (README typos), R3 (doc regen + CLAUDE.md), R5 (LSP loud-fail), R6 (SoftSOM docstring), R13 (topological_sigma), I13 fix (dead weight), I18 fix ('sample' loud-fail).
- Defer: R4, R8, R9, R10, R11, R12 (structural / API-breaking).

**GHOST**
- "MannLayer must keep memory_matrix for checkpoint BC" — weight is unused; only caller is qwen3_mega.py.
- "Can't fix map_size in README without API change" — API is already `grid_shape=`; doc is wrong, not API.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-13_a40908e7
### Index
*To be populated during EXPLORE.*

### Key Constraints
*To be populated during EXPLORE.*

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-13_a1c9a52d
### Index

| ID | Topic | File |
|----|-------|------|
| F-001 | Source files inventory (memory/ + ntm/) | findings/F-001-source-files.md |
| F-002 | Consumers / import sites (12 memory + 9 ntm) | findings/F-002-consumers.md |
| F-003 | Merge strategy options + recommendation | findings/F-003-merge-strategy.md |

### Key Constraints

### HARD
- **No breaking imports.** All 21 current call sites must keep working unchanged:
  - 12 `from dl_techniques.layers.memory.<submodule> import ...` callers (deep-submodule).
  - 7 `from dl_techniques.layers.ntm.<submodule> import ...` callers (deep-submodule).
  - 2 `from dl_techniques.layers.ntm import ...` callers (top-level public API).
- **Keras 3 / TF 2.18 idioms** — `@register_keras_serializable()` preserved on every moved class; full `get_config()` round-trip; `dl_techniques.utils.logger` only.
- **`@register_keras_serializable()` keys are `__module__`-based** when `package=` is omitted (LESSONS L118). Moving files mutates `__module__`. Mitigated: zero in-repo `.keras` fixtures reference NTM classes (verified — all `.keras` files live under `results/`, which is gitignored).
- **Tests must pass** — scope to `tests/test_layers/test_ntm/` + `tests/test_layers/test_som_*.py` + `tests/test_models/test_ntm/` + `tests/test_models/test_som/`. Never `make test` (1.5h hook).
- **Per `layers/CLAUDE.md`** — sibling packages may keep populated `__init__.py` when they export a public API; deep-submodule imports are the canonical pattern.

### SOFT
- **Recommendation: Option A** (flat siblings inside `memory/`). 3 file moves + a 3-file shim package at `layers/ntm/`.
- **Populate `memory/__init__.py`** as the canonical public surface going forward (the empty `__init__.py` is legacy state, not a hard constraint).
- **README** at `src/dl_techniques/layers/memory/README.md` covering family overview, class taxonomy (MANN / NTM / SOM), public surface, usage examples, references.
- **SOM tests stay flat** at `tests/test_layers/test_som_*.py` (out of scope to relocate).

### GHOST (considered & rejected)
- *Unify `MannLayer` and `NeuralTuringMachine` under one base class* — REJECTED. Independent implementations; `mann.py` does NOT use `BaseNTM`. Would change runtime semantics.
- *Move tests into `tests/test_layers/test_memory/`* — REJECTED. Renames 7 test files for zero API benefit.
- *Use `package=` on `@register_keras_serializable()` to stabilize registration keys across the move* — REJECTED. Would itself change semantics; no in-repo `.keras` consumers, so the move is safe as-is.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-13_8e866056
### Index
*To be populated during EXPLORE.*

### Key Constraints
*To be populated during EXPLORE.*

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-13_16ac1621
### Index
1. `findings/style-template.md` - prior benchmark files set the conventions (snapshot date, no emojis, no em-dashes, tables, sources). METRICS.md diverges by requiring math + per-metric prose, no leaderboard tables.
2. `findings/task-metric-inventory.md` - target coverage: 16 task families spanning ~80 distinct metrics. Defines depth required (formula + 2-6 sentence prose + edge cases) and resolves open questions (plain-text math, length ~700-900 lines, include subjective metrics + loss-as-metric briefly).
3. `findings/repo-implementations.md` - existing `dl_techniques/metrics/` modules. METRICS.md should cross-reference them so training scripts can find existing Keras implementations.

### Key Constraints
- **HARD**: File must live at `src/train/benchmarks/METRICS.md` (user-specified path).
- **HARD**: For every metric: computation formula + 2-6 sentence description + edge cases (user-specified).
- **HARD**: Web search required (user-specified, ensures up-to-date conventions and authoritative formulas).
- **HARD**: Style must match the other four benchmark files (no emojis, no em-dashes, snapshot date, sources section).
- **SOFT**: Tone and structure consistent with sibling files - prior pattern is "What X measures -> tables -> themes -> sources"; METRICS.md adapts to "What X is -> per-metric definitions grouped by task -> cross-cutting pitfalls -> sources".
- **SOFT**: Cross-reference existing `dl_techniques/metrics/` modules when relevant.
- **GHOST**: None identified - this is greenfield documentation.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*
