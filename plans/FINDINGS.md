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

## plan_2026-05-13_03176394
### Index

| ID | Topic | File |
|----|-------|------|
| F-001 | Source files structure and ctor signatures | findings/F-001-source-files.md |
| F-002 | All consumers / deep imports / test locations | findings/F-002-consumers.md |
| F-003 | FFN factory integration shape (registry, validation, exports) | findings/F-003-factory-integration.md |

### Key Constraints

- **HARD**: All consumer imports of `dl_techniques.layers.kan_linear` must be updated to the new path; otherwise `models/kan/`, `train/kan`, `train/coshkan`, `layers/attention/single_window_attention.py`, and existing tests break. (F-002)
- **HARD**: Both layers carry bare `@keras.saving.register_keras_serializable()` — moving changes `__module__`. No public .keras checkpoints reference these classes in-repo; safe to move. (F-001)
- **HARD**: `TverskyProjectionLayer.call()` is rank-2-only despite rank-generic `compute_output_shape`. Goal forbids modifying `call()` — document the 2D-only limitation in factory description + README; test only rank-2. (F-001, F-003)
- **SOFT**: Add `features` and `num_features` to factory `positive_dims` whitelist for validation consistency. (F-003)
- **SOFT**: docs/ generated files reference old paths; regenerated via `make docs`. Out of scope.
- **GHOST**: None identified — pure relocation + factory wiring.

### Corrections
*(none yet)*

## plan_2026-05-12_13c70aed
### Index

| ID | Topic | File | Summary |
|----|-------|------|---------|
| F-001 | CliffordNetLMUNet MRL integration surface | `findings/lmunet-mrl-integration.md` | MRL plugs in at the post-`head_norm`, post-squeeze `h_top: (B, T, C0)` tensor in `call()`. The existing `"logits"` head is just the slice at `w_0 = base_channels`. Tied mode reuses `token_embedding.embeddings[:, :w]`; untied mode adds per-width `Dense(V)`. Output dict shape: flat keys `{"logits": ..., "logits_w{w}": ..., "embedding_w{w}": ...}` rather than nested — keeps `prepare_dict_keyed_compile` simple and `compute_output_shape` clean. Width sequence: halve from base_channels down to a floor of 16. Causality preserved (head is per-position). Memory acceptable. |
| F-002 | Auxiliary L2-normalized embedding head — design | `findings/embedding-head-design.md` | Pool at last array position by default (causal model — last position has seen all real tokens). Expose `pool ∈ {last, cls, auto}`; default `last`. Default to identity projection (slice+norm); `--emb-head` flag enables a single learnable `Dense(C0, use_bias=False)` shared across widths. L2-norm per width independently. Numerical safety: epsilon `1e-12` under sqrt; cast to fp32 inside the norm op (LESSONS L34/L100). Embedding output: flat keys `{f"embedding_w{w}": (B, w)}` — side output, never participates in loss. Trainer-side `output_names` excludes embedding keys. |
| F-003 | Trainer + loss wiring | `findings/trainer-and-loss-wiring.md` | `prepare_dict_keyed_compile` gets a new `output_keys=None` parameter (backwards compatible); trainer passes `output_keys=["logits", "logits_w128", ...]`. Loss dict uses N `MaskedCausalLMLoss` instances (one per width). `loss_weights` dict: `uniform` default; `inv-log2` optional. Labels are duplicated across keys via `(x, y) -> (x, {k: y for k in lm_keys})`. CLI flags: `--mrl-widths`, `--mrl-weights`, `--emb-head`, `--mrl-head-norm`. No new `custom_objects` entries. Generation probe is unaffected — reads `"logits"`. |

### Key Constraints

### HARD
- Keras 3 / TF 2.18 idioms: `@keras.saving.register_keras_serializable()`, `keras.ops`, full `get_config()` round-trip, `dl_techniques.utils.logger` only.
- Causality (LESSONS L33, D-007 of plan_82749628) must be preserved at every slice width. Verified by test.
- `tie_word_embeddings=True` default honored at every width: slice `token_embedding.embeddings[:, :w]` transposed; per-width learnable bias.
- `"logits"` key is the SYSTEM.md output-key invariant. Must remain the primary (largest-width) head. Smaller widths get `f"logits_w{w}"` suffix.
- `prepare_dict_keyed_compile` extension must be backwards compatible (existing 6 CLM trainers unaffected).
- Numerical safety on L2-norm: epsilon `1e-12` under sqrt; fp32 cast for the norm op.
- No new external dependencies.
- Don't run `make test` — scope pytest to `tests/test_models/test_cliffordnet/`.
- `MPLBACKEND=Agg`; single GPU; user pushes commits.
- `.keras` round-trip atol = 1e-4 (LESSONS — fp32 reduction-order noise on U-Net).
- Width floor 16; widths halve from `base_channels`. nano `[128,64,32,16]`; mini `[192,96,48,24]`; base `[384,192,96,48,24]`; large `[512,256,128,64,32,16]`; xl `[768,384,192,96,48,24]`.
- Slice widths are static Python ints (resolved in `__init__`).
- MRL must support both tied and untied LM head modes.

### SOFT
- Output dict flat keys (`"logits_w64"`, `"embedding_w64"`).
- Default `--mrl-weights uniform`.
- Default `--emb-head False`.
- Default `--mrl-head-norm True`.
- Default `pool="last"`.
- Existing `"logits"` semantics unchanged.
- Extend existing test file rather than add a parallel file.

### GHOST (considered & rejected)
- Contrastive loss for embeddings — out of scope; `emb_head=False` default avoids dead weight.
- Nested dict output — rejected for `prepare_dict_keyed_compile` simplicity.
- CLS-at-0 default — rejected; causal model, position 0 sees only itself.
- Per-width Dense embedding projection — dead weight without contrastive signal.
- MRL inside attention/blocks — out of scope.
- Per-width perplexity metrics — adds memory for negligible signal.

### Exploration Confidence
- Scope: deep. All 4 target/relevant files read end-to-end. SYSTEM.md, LESSONS.md, prior plans plan_82749628 + plan_632605aa reviewed. No existing matryoshka utilities in the repo (grep returned only docstring references, none reusable).
- Solutions: constrained. Single architecturally-correct approach: post-`head_norm` slicing + per-width vocab projection (tied/untied), L2-normed side-output, flat-keyed output dict.
- Risks: clear. `loss_weights` dict on subclassed Keras models works via the same `output_names` fix; memory fits nano/mini at batch=8; `.keras` round-trip with per-width bias weights persists naturally; causality at small widths is structural.

### Synthesis
Three coupled additive changes, zero blast radius on non-MRL paths.
1. `src/dl_techniques/models/cliffordnet/lmunet.py` (~+200 LOC) — MRL+embedding head wiring.
2. `src/train/common/nlp.py` (~+10 LOC) — `prepare_dict_keyed_compile` accepts `output_keys`.
3. `src/train/cliffordnet/train_cliffordnet_nlp_unet.py` (~+80 LOC) — CLI + loss/label dicts.
4. `tests/test_models/test_cliffordnet/test_cliffordnet_lmunet.py` (~+150 LOC) — MRL/embedding/serialization tests.

### Corrections
*None yet.*
