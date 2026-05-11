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

## plan_2026-05-11_9357982a
### Index

| ID | Topic | File | Summary |
|----|-------|------|---------|
| F-001 | bert/ current state | `findings/bert-current-state.md` | 3 files, exports `BERT, create_bert_with_head, create_nlp_head`; no `create_bert`; `_download_weights` uses placeholder URLs; broad `except Exception` in `from_variant` (bert.py:716). |
| F-002 | resnet + tree_transformer template | `findings/resnet-tree-transformer-template.md` | Target shape: 3-name `__init__`, module-level `create_<model>` factory, `_download_weights` raises `NotImplementedError`, narrow `(IOError, OSError, ValueError)` except in `from_variant`, 3 lock-in tests, per-step commits `[iter-N/step-M] bert: ...`. |
| F-003 | bert/ issues (post double-check) | `findings/bert-issues.md` | I-01 HIGH silent random-init footgun; I-02 HIGH missing `create_bert` factory; I-03 MED stale `dl_techniques.nlp.heads` paths in docstring+README; I-04 MED `__init__` re-exports unused `create_nlp_head`; I-05/06 LOW (kept). |

### Key Constraints

**HARD**
- Keras 3 / TF 2.18; `@keras.saving.register_keras_serializable()`; full `get_config()` round-trip — must not break `test_config_serialization` or `test_model_save_load`.
- All external callers of the bert package (4 train scripts + tests) import only `BERT` and `create_bert_with_head`. Public API must keep these names.
- No `print` in library code (use `dl_techniques.utils.logger`).
- Scope pytest to `tests/test_models/test_bert/` — never run full suite.

**SOFT**
- Match resnet template (the reference) but tree_transformer post-refactor (plan_2026-05-11_0a5779e8) is the more recent worked example for NLP encoders — prefer its idioms (3-name `__init__`, `vocab_size` convenience kwarg in factory, lock-in test class layout).
- Commit-message style: `[iter-N/step-M] bert: <description>` (mirrors recent tree_transformer commits).

**GHOST**
- "We might publish pretrained BERT weights soon" — no evidence in repo. URLs are `example.com` placeholders; treat as permanently unavailable for now (raise `NotImplementedError`, leave the dict in place for API parity). This unlocks the cleanest fix.
- "`create_nlp_head` must be re-exported because some downstream consumer imports it from `bert`" — falsified by grep across `src/` and `tests/`.

### Corrections
*None yet.*

## plan_2026-05-11_0a5779e8
### Index

| ID | Topic | File | Summary |
|----|-------|------|---------|
| F-001 | Structure comparison: tree_transformer vs resnet | `findings/structure-comparison.md` | Both flat modules. Differences: tree_transformer over-exports 4 internal layer classes in `__init__.py`; lacks bare `create_tree_transformer(...)` factory analogous to `create_resnet(...)`; uses Sphinx-style docstrings vs ResNet/BERT Google-style; README is 405 LOC vs ResNet's 2486 (acceptable — domain difference). Both have `MODEL_VARIANTS`, `from_variant`, `get_config`/`from_config`. tree_transformer correctly uses `weight_transfer.load_weights_from_checkpoint` (Keras 3.8 `by_name=True` lesson); ResNet still uses the broken pattern (out of scope, 4th confirmed instance). |
| F-002 | model.py post-iter-1 audit (bugs/issues/gaps) | `findings/model-py-audit.md` | After plan_3c3ed037's 4 fixes (B-1/B-3/B-4/B-5), module is functionally correct. Issues found: I-03 (MEDIUM) `from_variant(pretrained=True)` silently swallows `NotImplementedError` from `_download_weights` via `except Exception` — contract violation. I-09 stale docstring import path. I-10 stale `weights_dataset` docstring text. I-06 `__init__.py` over-exports. I-07 missing bare encoder factory. Several false positives ruled out. |
| F-003 | Refactor targets aligned to resnet pattern | `findings/refactor-targets.md` | 5 in-scope changes: R-01 add `create_tree_transformer(...)` mirroring `create_resnet`; R-02 trim `__init__.py` to public API; R-03 fix I-03 silent-error contract violation; R-04/R-05 fix stale docstrings. Out of scope: Sphinx→Google docstring sweep, deep supervision, README expansion. Caller audit: only `dl_techniques.models.nam.*` and `src/train/nam/train_dfsa.py` import the internal layer classes — they import from `.model` directly, so trimming `__init__` exports is safe. |

### Key Constraints

**HARD**
- Existing tests (`tests/test_models/test_tree_transformer/test_model.py`, 744 LOC, 31 tests) must remain passing.
- B-1/B-3/B-4/B-5 fixes from plan_3c3ed037 must remain intact (locked by `TestTreeTransformerIter1Fixes`).
- `TreeTransformer`, `create_tree_transformer_with_head` must remain importable from `dl_techniques.models.tree_transformer` (package-level).
- `GroupAttention, TreeMHA, PositionalEncoding, TreeTransformerBlock` must remain importable from `dl_techniques.models.tree_transformer.model` (used by nam package).
- No commits to remote; user pushes themselves.
- Full pytest suite is ~1.5h — only run the tree_transformer subdir test.

**SOFT**
- Mirror `resnet/` package structure as closely as the domain (NLP encoder vs vision classifier) allows.
- Match `bert/__init__.py` style (model class + factories) for `__init__.py` exports.
- Google-style docstrings preferred — full sweep deferred (cosmetic).

**GHOST**
- "tree_transformer should expose internal layers in `__init__`" — no caller in src/ relies on this; was an early default, not a deliberate API decision.
- "Refactor implies a full rewrite" — bugs are already fixed; this plan is alignment + small bug-fixes.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*

## plan_2026-05-11_3c3ed037
### Index

| ID    | Topic                                              | File                              | Headline                                                                                                                  |
|-------|----------------------------------------------------|-----------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| F-001 | tree_transformer `model.py` code review            | `findings/model-py-review.md`     | 14 issues. 1 HIGH (B-1 mixed_float16 NaN), 4 MEDIUM (B-2 fp16 sentinel, B-3 attention_mask ignored, B-4 by_name=True, B-5 fake pretrained URLs), 9 LOW/INFO. Save/load round-trip, gradient flow, basic forward all CORRECT. |
| F-002 | README ↔ implementation cross-check                 | `findings/readme-vs-impl.md`      | 6 drifts. HIGH: §9 mixed_float16 promo is broken. MEDIUM: §8 MLM example uses `-100` ignored-index incorrectly. §5/§6 import path + missing PositionalEncoding. LOW: §9 jit_compile unverified. |
| F-003 | `src/train/<model>/` conventions survey             | `findings/train-conventions.md`   | Pattern 3 (NLP Pretrain). Mirror `src/train/bert/{pretrain.py,finetune.py}`. `MaskedLanguageModel` wrapper accepts TreeTransformer as-is (`hidden_size` attr + `last_hidden_state` key both present, empirically verified). Trainer config MUST set `pad_token_id=100266` (tiktoken), not the model default 0. |
| F-004 | Existing test coverage analysis                     | `findings/test-coverage.md`       | 9 test classes / 15+ tests exist at `tests/test_models/test_tree_transformer/test_model.py`. No coverage for: mixed precision, attention_mask honoring, `load_pretrained_weights`, `from_variant(pretrained=True)`, `model.fit` smoke, MLM-wrapper compatibility, pad_token_id != 0. |

### Key Constraints

### HARD
- Keras 3 / TF 2.18, `keras.ops` / `keras.random.*` only.
- `@keras.saving.register_keras_serializable()` + full `get_config()` round-trip on every Layer/Model — currently SATISFIED.
- `dl_techniques.utils.logger` only — currently SATISFIED.
- Save/load via `keras.models.load_model(path)` without `custom_objects` — currently SATISFIED (verified, max-abs-diff 0.0 on logits).
- Tests must remain scoped to `tests/test_models/test_tree_transformer/` — full suite is ~1.5h, do NOT run.
- Single GPU only; `MPLBACKEND=Agg`; never parallel jobs.
- User pushes commits themselves; we commit locally with `[iter-N/step-M] desc` prefix.
- `MaskedLanguageModel` requires encoder with `.hidden_size` attribute and `{"last_hidden_state": ...}` in call output — both present in TreeTransformer.
- `pad_token_id` in TreeTransformer constructor MUST match the trainer's tokenizer pad token (default 0 vs tiktoken cl100k_base pad=100266 — mismatch silently mis-masks every input).
- Pattern 3 mandate: use `train.common.nlp` helpers (`create_tokenizer`, `load_text_dataset`, `preprocess_mlm_dataset`, `create_warmup_lr_schedule`, `create_nlp_callbacks`). Do NOT roll local versions.
- Trainer file naming: per `src/train/CLAUDE.md`, `train_<model>.py` is the rule, but the BERT sibling uses `pretrain.py`/`finetune.py` and is the closest precedent. Recommendation: follow BERT (`pretrain.py` + `finetune.py`).
- 10-Line Rule + Autonomy Leash (2 fix attempts max per step).

### SOFT
- README §11 ("comprehensive test suite") is overstated — gap visible in F-004. Closing the gap is recommended but not blocking.
- `__init__.py` is currently 0 LOC. Sibling models split between empty and re-exporting public API. Re-exporting (`TreeTransformer`, `create_tree_transformer_with_head`, optionally layer classes) is cleaner but optional.
- `lm_head` in the foundation model is dead weight for pure-encoder use (B-9). Removing it would be a breaking change to `get_config`/round-trip — out of scope unless user requests.
- `keras.mixed_precision` fix: change `-1e9` magic constants to dtype-aware sentinel (`-1e4` for fp16, `-1e9` otherwise). ≤6 LOC delta total.

### GHOST (considered and rejected)
- "TreeTransformer needs its own MLM loss" — NO; `MaskedLanguageModel` wrapper is encoder-agnostic and works.
- "TreeTransformer needs a custom train_step for the tree induction aux-loss" — NO; the Shen 2019 paper does NOT add an auxiliary loss; group_prob is supervised purely through the MLM gradient signal flowing back through `TreeMHA`'s multiplicative modulation. Confirmed in §4 of paper and absence of any aux-loss surface in the model code.
- "PRETRAINED_WEIGHTS should point at real HuggingFace URLs" — NO; no public official checkpoint exists for the dl_techniques implementation. Either remove the surface or document as TBD.
- "Refactor `_build_architecture` inline into `__init__`" — NO; cosmetic only, breaks nothing, not worth the diff.
- "Add `LayerCKYTreeAttention` as a public layer in `dl_techniques/layers/`" — NO; the layer is tree-transformer-specific and not requested.

### Exploration Confidence

- **scope: deep** — entire `model.py` (1246 LOC) read, README (371 LOC) read, sibling BERT pretrain (258 LOC) read, MLM wrapper internals read, existing 549-LOC test suite read, empirical smoke confirmed forward/serialization/gradient/edge cases. Mixed-precision NaN reproducer is a 5-line smoke.
- **solutions: adequate** — bug-fix shapes are all bounded (≤10 LOC each). Trainer is a Pattern-3 mirror of `bert/pretrain.py` with one constructor override.
- **risks: clear** — primary risk is `attention_mask`-honoring change (B-3) inadvertently breaking the existing `test_padding_mask_functionality` test. Mitigation: preserve fallback derivation; only honor explicit mask when present.

### Synthesis

The tree_transformer package is structurally sound. Save/load, get_config, gradient flow, MLM-wrapper integration are all empirically correct. The defects fall into three buckets:

(a) **Two real correctness bugs** affecting documented usage paths — B-1 (`mixed_float16` NaN, contradicts README §9) and B-4 (`load_pretrained_weights` uses by_name=True, broken in Keras 3.8 per LESSONS).

(b) **Two API/UX bugs** — B-3 (explicit `attention_mask` ignored, BERT-API contract gap) and B-5 (placeholder `example.com` URLs in `PRETRAINED_WEIGHTS` guarantee runtime error for anyone following the module docstring).

(c) **Doc and surface gaps** — empty `__init__.py`, README §8 MLM example uses the wrong ignored-index recipe, README §11 "comprehensive tests" claim is overstated.

The training pipeline is a near-trivial mirror of `src/train/bert/{pretrain.py, finetune.py}` — `MaskedLanguageModel` wraps TreeTransformer cleanly. Single biggest pitfall: trainer MUST pass `pad_token_id=<tiktoken_pad>` (100266 for cl100k_base) to `TreeTransformer(...)` or the encoder's pad mask will be all-ones (silent semantic bug).

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-10_e6309bd5
### Index

| ID | Topic | File | Headline |
|----|-------|------|----------|
| F-001 | `model.py` (TRM) bug & gap review | `findings/model-py-review.md` | 14 issues; 2 HIGH (B-1 expand_dims tuple axis, B-5 ACT-at-inference regression vs paper), 1 HIGH (B-3 Q-learn lookahead uses training=True) |
| F-002 | `components.py` (TRMReasoningModule, TRMInner) review | `findings/components-py-review.md` | Components clean. Composite-layer pattern correctly implemented. No critical bugs. |
| F-003 | `__init__.py`, README, serialization, tests | `findings/api-serialization-review.md` | Save/load OK. Missing factory, missing test module, 3 README doc-vs-code drifts. |
| F-004 | Training conventions for `train/tiny_recursive_model/` | `findings/training-conventions-survey.md` | HRM trainer is the canonical template; `hrm_loss` and `HRMMetrics` already match TRM output schema; reuse them. |

### Key Constraints

### HARD constraints
- Keras 3 / TF 2.18, `keras.ops` only; `keras.random.*` only.
- `model.save(...keras)` + `keras.models.load_model(...)` must round-trip without `compile=False` or `custom_objects` (currently passes — smoke-tested).
- Output schema compatible with `dl_techniques.losses.hrm_loss.HRMLoss`: keys `logits`, `q_halt_logits`, `q_continue_logits`, optional `target_q_continue` (currently compatible).
- Single GPU only; `MPLBACKEND=Agg`; no `print`; no AdamW WD + L2 kernel reg double-up.
- Trainer file MUST be named `train_<model>.py` not `train.py` (package-shadow rule).
- `stop_gradient` on inner carry is structural — gradient does not flow across ACT steps. Trainer cannot backprop through the whole unroll.

### SOFT constraints
- HRM-style class-based trainer, no `model.fit()`. Custom `GradientTape` per-step or per-unroll.
- `dl_techniques.optimization.optimizer_builder` / `learning_rate_schedule_builder`.
- CLI via `train.common.create_base_argument_parser`.
- Synthetic dataset primary; optional ARC behind a flag.
- Test module at `tests/test_models/test_tiny_recursive_model/`.
- Add `create_trm(...)` factory to match `create_hierarchical_reasoning_model` precedent.
- README + CLAUDE.md updates land in same plan as the code change (LESSONS).

### GHOST constraints (considered and rejected)
- "TRM needs its own loss" — NO; `HRMLoss` is API-compatible.
- "TRM needs its own metrics" — NO; `HRMMetrics` works on the same schema.
- "`tf.stop_gradient` is canonical (per README)" — NO; `keras.ops.stop_gradient` is.
- "Puzzle embedding must be wired (paper fidelity) before training" — NO for this plan. Zero-pad is inert (B-11); wiring `HRMSparsePuzzleEmbedding` is a larger paper-fidelity plan. Document as a residual.

### Exploration Confidence
- **scope**: deep (entire package read; empirical build, forward, Q-branch, save/load verified).
- **solutions**: adequate (clear bug-fix set + HRM-template trainer blueprint).
- **risks**: clear (4 doc/code drifts; B-1 needs a graph-mode reproducer to lock as bug-or-non-bug).

### Synthesis
The TRM package is structurally sound and serializes correctly. There is no foundational rewrite needed. The defects fall into three buckets: (a) two correctness bugs in the ACT loop (B-1 expand_dims tuple-axis is graph-unsafe, B-5 inference always max-steps), (b) one training-stability bug in the Q-learning lookahead (B-3 lookahead in training mode), and (c) doc/test gaps (no factory, no test module, README drift). The training script is straightforward — `hrm_loss` and `HRMMetrics` already accept TRM's exact output schema, so the trainer is essentially a thin HRM-trainer mirror with `_forward_step` replaced by the standard `model(carry, batch, training=...)` and a `keras.ops.all(carry["halted"])` finish check. Recommend bundling the bug fixes, factory function, README reconciliation, test module, and training script into a single iteration-1 plan.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

- **[CORRECTED iter-0] F-001 B-1 (`ops.expand_dims(axis=tuple)`) — FALSE POSITIVE.** Empirical reproducer (eager + `@tf.function` graph) on Keras 3.8 / TF 2.18 confirms `keras.ops.expand_dims` accepts tuple-of-int `axis` and broadcasts correctly. The textual prior from older keras-numpy compat warnings does not apply. B-1 is dropped from the bug list — no fix needed. F-001 summary table downgrades B-1 from CRITICAL to N/A.
- **[CONFIRMED iter-0] F-001 B-5 (ACT-at-inference regression).** Empirical reproducer: with `training=False`, even sequences whose `q_halt_logits > 0` at every step never halt before `step == halt_max_steps`. This is a genuine paper regression and the primary correctness bug to fix.

## plan_2026-05-10_17633038
### Index

| ID | Topic | File | Severity | Status |
|----|-------|------|----------|--------|
| F-000 | Seed bug report from accunet plan (user-supplied) | `findings/seed-bug-report.md` | CRITICAL | confirmed |
| F-001 | WrappedLoss anatomy — `reduction` round-trip + closure-capture bug | `findings/wrapped-loss-anatomy.md` | CRITICAL | confirmed (static + Keras-3 source-trace) |
| F-002 | Blast radius — callers, trainers, public API | `findings/blast-radius.md` | mixed | confirmed (grep) |
| F-003 | Recommended fix — module-level WrappedLoss with reconstructable loss_fn | `findings/fix-design.md` | n/a | design |
| F-004 | Sibling-loss conventions (file/class naming, decorator, exports, tests) | `findings/loss-module-conventions.md` | HARD-CONSTRAINTS | confirmed (4+ sibling reads, init/CLAUDE/README) |

### Key Constraints

### HARD
- Keras 3 idioms — `@keras.saving.register_keras_serializable()`, full `get_config`/`from_config` round-trip.
- `WrappedLoss` is currently defined INSIDE `create_loss_function` (closure) — must hoist to module scope for true serializability.
- `loss_fn` is a closure-captured bound method → NOT in get_config → unrecoverable on load. Fix MUST reconstruct from a name+config, not pickle the function.
- `LossConfig` (dataclass) must be Keras-serializable for round-trip — verify and patch if needed.
- Public API surface (`create_loss_function`, `create_segmentation_loss_function` re-export) must remain backward-compatible.
- Active production trainer (`src/train/accunet/train_accunet.py`) currently uses `compile=False` workaround — fix must allow removing it.
- Existing test `test_loss_serialization_and_deserialization` swallows the exception in a `try/except` — must be updated to assert success (currently encodes the bug as a contract; see LESSONS).
- No `print` calls, use `dl_techniques.utils.logger` only.
- Scope pytest to `tests/test_losses/test_segmentation_loss.py` + `tests/test_models/test_accunet/` for regression.

### SOFT
- Three "candidate" fixes from user (a/b/c) are all incomplete because they only address `reduction`, not the deeper closure problem. EXPLORE has falsified all three as standalone fixes.

### GHOST
- "WrappedLoss must stay inside the factory for parametric specialization" — no such constraint; module-level class with `loss_name` param works identically and is serializable.
- "WrappedLoss should live inside `segmentation_loss.py`" — no such constraint; per F-004 sibling conventions, every loss is its own module. Hoisting AND extracting to a sibling module (`segmentation_wrapper_loss.py`) is the conformant choice.

### HARD (added iter-0, post-revision)
- F-004: new wrapper class lives in its own module `src/dl_techniques/losses/segmentation_wrapper_loss.py`.
- F-004: class is `SegmentationWrapperLoss` with bare `@keras.saving.register_keras_serializable()` (no package= argument).
- F-004: backward compat — `from dl_techniques.losses.segmentation_loss import create_loss_function, LossConfig` MUST keep working; `create_segmentation_loss_function` re-export MUST keep working.
- F-004: new test file `tests/test_losses/test_segmentation_wrapper_loss.py` mirrors module name.
- F-004: must update `losses/__init__.py`, `losses/CLAUDE.md`, `losses/README.md`.

### Corrections
*None yet.*

### Exploration Confidence
- scope: deep (full WrappedLoss source read, blast radius grep'd, parent class behavior known, test contract reviewed)
- solutions: constrained (one design path — module-level class with reconstructable loss_fn; minor open question on LossConfig serializability)
- risks: clear (only risk is LossConfig being non-trivially nested; verifiable in 5 lines during PLAN)

## plan_2026-05-10_bdb2c84d
### Index

| ID | Topic | File | Severity | Status |
|----|-------|------|----------|--------|
| F-001 | accunet code review (model.py + README + __init__.py) | `findings/accunet-code-review.md` | mixed | confirmed (empirical + static) |
| F-002 | src/train/<model>/ conventions for AccUNet trainer | `findings/train-conventions.md` | n/a | reference |
| F-003 | Dependent layers (HANCBlock/HANCLayer/ResPath/MLFCLayer/SE) | `findings/dependent-layers.md` | n/a | reference |

### Key Constraints

### HARD
- Keras 3 idioms — `@keras.saving.register_keras_serializable()`, `keras.ops`, full `get_config()`.
- Single GPU jobs only; `MPLBACKEND=Agg` for any training-script invocation.
- AdamW vs L2 double weight decay — pick one.
- No `print` calls — use `dl_techniques.utils.logger`.
- Trainer file naming `train_<model>.py`, never `train.py`.
- Trainer must use `train.common` (`setup_gpu`, `create_callbacks`, `optimizer_builder`, `learning_rate_schedule_builder`).
- Do not run full test suite. Scope pytest to `tests/test_models/test_accunet/`.
- Keras 3.8 `model.load_weights(by_name=True)` broken on `.keras` — use `load_weights_from_checkpoint`.
- AccUNet input dimensions MUST be divisible by 16 in current implementation (verified empirically). Trainer must enforce on data loading until the fix lands.
- MLFCLayer requires exactly 4 input tensors.
- HANCBlock requires explicit `input_channels` matching the actual input.

### SOFT
- Output activation hard-coded sigmoid/softmax (could be made optional; out of scope unless user asks).
- 5-level fixed depth (paper convention).
- 16-pixel divisibility intrinsic to depth=5 with `padding='valid'` pooling — switching to `padding='same'` removes it.

### GHOST
- None identified.

### Corrections
*None yet.*

### Exploration Confidence
- scope: deep (target files read; sibling trainers and dependent layers read; empirical input-size sweep performed)
- solutions: constrained (fixes localised; trainer = Pattern-4)
- risks: clear (B1 only real user-visible bug; everything else is hygiene + new code)

## plan_2026-05-10_54e6e303
### Index

| ID | Topic | Detail |
|----|-------|--------|
| F-001 | Inherited OPEN inventory after plan_bd098beb | inline below |
| F-002 | EMA decay scheduling shape (warmup → asymptote) + callback wiring | inline below |
| F-003 | `from_pretrained_encoder` weight-loading hook design | inline below |
| F-004 | Pseudo-label + dataset pairing design | inline below |
| F-005 | train_step refactor design (clean labeled / semi-sup branches) | inline below |
| F-006 | StrongAugmentation #9 fix shape (per-sample factors + dynamic channels) | inline below |
| F-007 | Multi-epoch FAL stability test recipe | inline below |
| F-008 | Train script + README impact | inline below |

### Key Constraints

### Hard
- Keras 3 / TF 2.18 idioms; full `get_config()` round-trip; `@keras.saving.register_keras_serializable()`; `keras.ops` / `keras.random.*` only; `dl_techniques.utils.logger`.
- Preserve `# DECISION plan_2026-05-10_44694bc9/D-003` (Keras-3 train_step) and `# DECISION plan_2026-05-10_bd098beb/D-004` (save_own_variables override) anchors.
- AdamW WD only — no `kernel_regularizer=L2`.
- CPU verification with `CUDA_VISIBLE_DEVICES=""`. Single GPU jobs only — never parallel.
- Per-step commit `[iter-N/step-M] desc`; user pushes.
- 10-Line Rule + Autonomy Leash (2 fix attempts max per step).
- Existing 8 tests in `tests/test_models/test_depth_anything/test_depth_anything.py` must continue to pass.
- StrongAugmentation graph-mode safety: no Python-bool branches on symbolic tensors; `keras.random.*` not `keras.ops.random.*`.

### Soft
- New flag defaults backward-compatible (off / no-op when not requested).
- TeacherEMACallback as a small standalone class in the model package, not a sub-attribute of DepthAnything.
- Pseudo-label generation as a helper method on the model — keeps train_step locally readable.
- Dataset pairing utility lives under `train.common.megadepth`.
- `from_pretrained_encoder` stays a small wrapper around existing `load_weights_from_checkpoint`.

### Ghost / out-of-scope
- True DINOv2 weight loading from HuggingFace.
- Distributed / multi-GPU.
- `make test` (full 1.5h suite).
- Item #5 (`tf.GradientTape`) — semi-sup path needs the explicit tape.

### Exploration Confidence
- **Problem scope: deep** — read entire `model.py` (752 LOC), `components.py` (309 LOC), `strong_augmentation.py` (258 LOC), train script (711 LOC), train README (166 LOC), model README (337 LOC), test module (178 LOC), `weight_transfer.py` API, FAL header. Reviewed all prior decisions.
- **Solution space: constrained** — every OPEN item has a known fix shape; the EMA callback / pseudo-label / dataset-pairing pieces are small, additive surfaces.
- **Risk visibility: clear** — main risks: (a) `update_teacher_ema` is numpy-based per-step (CPU host roundtrip); (b) pseudo-label loss must be plain L1 over 1-channel depth (no mask); (c) per-sample brightness/contrast must broadcast `(B,1,1,1)`; (d) multi-epoch test must complete <60s on CPU.

### F-001 — Inherited OPEN inventory

From model README "STILL OPEN" after plan_bd098beb:
- **#5** — `tf.GradientTape` in custom train_step. Keep & document.
- **#2-deeper** — On-step EMA decay schedule + integration with a real pretrained student.
- **#3-deeper** — Pseudo-label depth on unlabeled stream + `((x_lab, x_unlab), y_lab)` pairing.
- **#9** — `_apply_cutmix` channels=3 hard-coded; brightness/contrast batch-scalar.

Plus user-explicit:
- train_step refactor (clean labeled / semi-sup paths)
- Multi-epoch FAL stability test (≥3 epochs, finite + non-increasing-on-average + teacher moves).

### F-002 — EMA decay scheduling + callback wiring

Existing surface: `DepthAnything.update_teacher_ema(decay)` at model.py:286-305 (numpy-based).

**Schedule shape**: standard Mean-Teacher / DINO recipe:
- cosine: `decay(t) = end - (end-start) * 0.5 * (1 + cos(pi * min(t,T)/T))`
- linear: `decay(t) = start + (end-start) * min(t/T, 1.0)`
- typical: start≈0.5, end≈0.999, total_steps = epochs * steps_per_epoch.

**New module** `dl_techniques/models/depth_anything/teacher_ema.py`:
- `cosine_ema_schedule(decay_start, decay_end, total_steps) -> Callable[[int], float]`
- `linear_ema_schedule(decay_start, decay_end, total_steps) -> Callable[[int], float]`
- `class TeacherEMACallback(keras.callbacks.Callback)` — constructor `(schedule: Callable[[int], float], warmup_steps: int = 0)`. On `on_train_batch_end`: if `step >= warmup_steps`, calls `model.update_teacher_ema(decay=schedule(step - warmup_steps))`. Increments step counter.

Re-exported via package `__init__`.

### F-003 — `from_pretrained_encoder` design

Wrap `load_weights_from_checkpoint` against `model.encoder`. Re-sync teacher.

```python
def from_pretrained_encoder(self, weights_path, skip_prefixes=()):
    if not self.built:
        _ = self(keras.ops.zeros((1,)+tuple(self.image_shape)), training=False)
    report = load_weights_from_checkpoint(
        target=self.encoder, ckpt_path=weights_path, skip_prefixes=skip_prefixes,
    )
    logger.info(f"from_pretrained_encoder: loaded={report.num_loaded}")
    if self.frozen_encoder is not None and self.use_feature_alignment:
        self.frozen_encoder.set_weights(self.encoder.get_weights())
```

### F-004 — Pseudo-label + dataset pairing

### Pseudo-label (model side)
Helper method:
```python
def _pseudo_label_depth(self, x_unlab):
    feat = self.frozen_encoder(x_unlab, training=False)
    feat = self._features_to_spatial(feat)
    pseudo = self.decoder(feat, training=False)
    return ops.stop_gradient(pseudo)
```

In semi-sup train_step add a consistency term:
```python
y_pred_unlab = self(x_unlab, training=True)
pseudo = self._pseudo_label_depth(x_unlab)
consistency = ops.mean(ops.abs(y_pred_unlab - pseudo))
loss += self.loss_weights.get('unlabeled', 0.5) * consistency
```

### Dataset pairing
New helpers in `train.common.megadepth`:
- `class UnlabeledImageDataset(keras.utils.PyDataset)` — small, returns batched RGB.
- `pair_labeled_unlabeled(labeled_ds, unlabeled_ds) -> tf.data.Dataset` yielding `((x_lab, x_unlab), y_lab)` via `tf.data.Dataset.zip` + map.

Train script: behind `--enable-semi-supervised` AND `--unlabeled-image-glob`, build paired dataset.

### F-005 — train_step refactor

Replace single mega-block with split helpers:
```python
def train_step(self, data):
    x, y = data
    if self.enable_semi_supervised and isinstance(x, (tuple, list)) and len(x) == 2:
        return self._train_step_semi_supervised(x[0], x[1], y)
    return self._train_step_labeled(x, y)
```

`_train_step_labeled`: short — single forward + compute_loss + apply grads + update metrics.
`_train_step_semi_supervised`: clearly delimited — labeled forward + (optional) FAL term + (optional) consistency term.

Preserve `# DECISION plan_2026-05-10_44694bc9/D-003` anchor at the `compute_loss` call.

### F-006 — StrongAugmentation #9 fix

In `_apply_cutmix`:
- `ops.tile(mask, [1, 1, 3])` → `ops.tile(mask, [1, 1, ops.shape(x)[-1]])`.

In `_apply_color_jitter`:
- brightness/contrast `shape=()` → `shape=(ops.shape(x)[0], 1, 1, 1)` for per-sample factors.

Per-sample factors broadcast cleanly. Graph-traceability preserved.

### F-007 — Multi-epoch FAL stability test

CPU pytest: build small ViT-S DepthAnything at 64x64, snapshot teacher weights, run 3 epochs * 2 steps semi-sup synthetic, assert: (a) all losses finite; (b) `losses[-1] <= 1.5 * losses[0]` (loose tolerance); (c) sum-abs-diff(teacher_after - teacher_before) > 0.

### F-008 — Train script + README impact

Train script CLI: add `--ema-decay-end`, `--ema-warmup-steps`, `--ema-schedule {cosine,linear,none}`, `--unlabeled-image-glob`, `--pretrained-encoder-weights`.

Wire `from_pretrained_encoder`, `pair_labeled_unlabeled`, `TeacherEMACallback`.

Train README: drop "labeled-only" caveat; new semi-sup recipe with EMA.

Model README: move #2-deeper, #3-deeper, #5 (refactored), #9 to FIXED. Only #5 base note (custom GradientTape) remains as documented LOW. Add EMA-callback usage example.

### Corrections
*None yet.*

## plan_2026-05-10_bd098beb
### Index

| ID  | Topic | Detail |
|-----|-------|--------|
| F-001 | Inherited issue inventory (README #1-#14 + D-005) | inline below |
| F-002 | Real encoder availability (ViT vs DINOv2) | inline below |
| F-003 | Decoder upsampling design | inline below |
| F-004 | Semi-supervised pipeline + feature-alignment wiring | inline below |
| F-005 | Train-script + README sync impact | inline below |

### Key Constraints

### Hard
- Keras 3 / TF 2.18 idioms, full `get_config()` round-trip, `@keras.saving.register_keras_serializable()`, `keras.ops`, `dl_techniques.utils.logger` (no `print`).
- Single GPU, never parallel jobs. CPU smoke (`CUDA_VISIBLE_DEVICES=""`) preferred for verification.
- Per-step commit `[iter-N/step-M] desc`; user pushes themselves.
- Update both READMEs (model + train) when API changes.
- Pattern `mlm.py:309-343` is canonical Keras-3 train_step. Already applied — preserve `# DECISION plan_2026-05-10_44694bc9/D-003` anchor unless we materially change semantics.
- AdamW WD only — never combine with `kernel_regularizer=L2`.
- 10-Line Rule + Autonomy Leash + 3-Strike Rule.

### Soft
- Real encoder: prefer `dl_techniques.models.vit.ViT(include_top=False, pooling=None)` — `DINOv2VisionTransformer` requires three Functional inputs (image+masks+is_training), much harder to compose.
- DPTDecoder default activation: change `'sigmoid'` → `'linear'` so AffineInvariantLoss is usable.
- Semi-supervised: gate behind `enable_semi_supervised` flag (default OFF) for backward compatibility.
- Feature alignment: rewire `frozen_encoder` to be a `keras.models.clone_model(encoder)` weight-share teacher; expose `update_teacher_ema(decay)`. Default OFF.

### Ghost / out-of-scope
- True DINOv2 weight loading from HuggingFace — separate plan.
- Pseudo-label depth-on-unlabeled training (needs a real pretrained teacher) — separate plan.
- Multi-GPU / distributed.
- `make test` (full 1.5h suite).

### Exploration Confidence
- **Problem scope: deep** — read entire `model.py`, `components.py`, `__init__.py`, `strong_augmentation.py`, `affine_invariant_loss.py`, `feature_alignment_loss.py`, current train script, ViT and DINOv2 model surface, prior plan summary + findings + decisions.
- **Solution space: constrained** — fixes are mechanical for #5/#6/#7/#8/#10/#11/#13/#14 + D-005; substantive for #1 (real ViT encoder), #2/#3 (semi-sup + FAL wiring). Pattern: gate behind flags; default backward-compatible.
- **Risk visibility: clear** — main risks: (a) real ViT encoder produces sequence `(B,N,D)` while decoder expects 4D — must reshape + upsample; (b) frozen_encoder clone ordering vs build; (c) save/load round-trip with new sub-model topology. Mitigated by step ordering + per-step CPU smoke.

### F-001 — Inherited issue inventory

From model README Known Issues + plan_2026-05-10_44694bc9 D-005:

| # | Severity | Description | Fix shape |
|---|----------|-------------|-----------|
| 1 | HIGH | Placeholder Conv-BN-ReLU encoder, not real ViT/DINOv2 | Wire `dl_techniques.models.vit.ViT(include_top=False, pooling=None)`; reshape (B,N,D)→(B,h,w,D); placeholder kept behind `encoder_kind='conv'` |
| 2 | HIGH | Frozen teacher has independent random weights | `keras.models.clone_model(encoder)` + copy weights post-build; expose `update_teacher_ema(decay)` |
| 3 | HIGH | Semi-supervised pipeline unimplemented | `enable_semi_supervised` flag (default False). When True: `train_step` accepts `((x_lab, x_unlab), y_lab)`, computes labeled loss + FAL on unlabeled feats. Pseudo-label-depth deferred. |
| 4 | FIXED | Keras-3 train_step (prior plan) | preserve `# DECISION plan_2026-05-10_44694bc9/D-003` |
| 5 | LOW | tf.GradientTape | Acceptable; keep — Keras 3 supports TF backend tape. Note in README. |
| 6 | HIGH | DPTDecoder default sigmoid incompatible with AIL | Change default to `'linear'`. Update train script + README. |
| 7 | MEDIUM | Functional encoder built in `build()` is fragile under save/load | When encoder is ViT (declared in `__init__`), the issue disappears. |
| 8 | LOW | Dead `if X is not None` checks + `_create_fallback_decoder` | Remove. |
| 9 | LOW | StrongAugmentation cutmix per-batch + 3-channel hardcoded | Make channels dynamic; per-sample brightness/contrast factors. + D-005 fix. |
| 10 | MEDIUM | `encoder_type` validated but unused | Map `vit_s/vit_b/vit_l` → ViT scales `small/base/large`; `conv` = placeholder. |
| 11 | LOW | `input_shape` shadows Layer.input_shape | Rename to `image_shape` (back-compat alias retained). |
| 12 | MEDIUM | No tests | Add `tests/test_models/test_depth_anything/test_depth_anything.py`. |
| 13 | LOW | `frozen_encoder.trainable = trainable` after Functional construction | Replaced by clone_model approach. |
| 14 | LOW | `compile()` mutates dead state | Remove `self.depth_loss`/`self.feature_loss`; just `super().compile()`. |
| D-005 | HIGH | `keras.ops.random.uniform` doesn't exist in Keras 3.8 | Replace with `keras.random.uniform` in StrongAugmentation `_apply_color_jitter` and `_apply_cutmix`. |

### F-002 — Real encoder availability

`src/dl_techniques/models/vit/model.py` — `ViT(keras.Model)`:
- `ViT(input_shape=(384,384,3), scale='small'/'base'/'large', patch_size=16, include_top=False, pooling=None)` returns `(B, num_patches+1, embed_dim)`.
- Strip CLS: `x[:, 1:, :]` → `(B, num_patches, embed_dim)`.
- Reshape to spatial: `(B, h, w, embed_dim)` where `h=H//patch_size`, `w=W//patch_size` (statically known from constructor).

`DINOv2VisionTransformer` requires three Functional inputs (`[inputs, masks_input, is_training_input]`) — too invasive. Defer.

### F-003 — Decoder upsampling design

ViT encoder at patch_size=16 → encoder output is H/16 × W/16 (24×24 for 384×384). Need to upsample to full resolution.

Add `upsample_factor: int = 1` to `DPTDecoder.__init__`. With 4-stage `dims=[256,128,64,32]`, distribute upsampling across stages: each non-final conv block followed by 2× bilinear upsample → cumulative 16× (one per stage; last stage no upsample, output_conv at full res).

`compute_output_shape` updated to multiply h,w by upsample_factor.

DepthAnything passes `upsample_factor = image_size // encoder_stride`. For placeholder Conv encoder (stride 32 actually — initial stride-2 conv + 4 maxpools = 32×), placeholder mode passes 32; for ViT (patch_size=16), passes 16.

### F-004 — Semi-supervised + feature-alignment wiring

`train_step(data)`: detect `x = data[0]; y = data[1]`. If `x` is a 2-tuple `(x_lab, x_unlab)`, run semi-sup path; else single-batch labeled-only path (current behavior).

When `enable_semi_supervised` AND `use_feature_alignment`:
1. Forward labeled: `y_pred_lab = self(x_lab, training=True)`. Capture student feat via `feat_student = self.encoder(x_unlab, training=True)`.
2. Teacher: `feat_teacher = self.frozen_encoder(x_unlab, training=False)`. Stop-gradient on teacher path (frozen weights, no tape track).
3. Loss: `loss = w_lab * compute_loss(x=x_lab, y=y, y_pred=y_pred_lab) + w_feat * FeatureAlignmentLoss()(feat_teacher, feat_student)`.

`feature_alignment_loss` expects `(B, feature_dim)` shape. We pool `(B, h, w, D)` to `(B, D)` via global average pool BEFORE passing to FAL. (Per-token FAL would need broadcasting; pooled is the standard distillation form.)

`frozen_encoder` build: in `DepthAnything.build()`, after `self.encoder = ...`, when `use_feature_alignment`: `self.frozen_encoder = keras.models.clone_model(self.encoder); self.frozen_encoder.set_weights(self.encoder.get_weights()); self.frozen_encoder.trainable = False`.

`update_teacher_ema(decay=0.999)` method copies EMA weights from student → teacher. Future plan can wire a callback to invoke this each step.

Pseudo-label-depth-on-unlabeled-data path: deferred. Documented in README as residual.

### F-005 — Train script + README impact

`src/train/depth_anything/train_depth_anything.py`:
- `create_model()` adds `encoder_kind`, `output_activation='linear'` (now valid via DepthEstimationLoss + AIL both), `image_shape` (renamed), `enable_semi_supervised`.
- Add `--encoder-kind {real,placeholder}` (default `real`) and `--enable-semi-supervised` flag (default off).
- Loss: `DepthEstimationLoss` continues to work (linear output is fine; sigmoid was a constraint, removing it is a simplification).

Model README rewrite Known Issues:
- FIXED in this plan: #1, #2(weight-shared teacher), #3(infrastructure for semi-sup; FAL wired), #6, #7, #8, #10, #11, #13, #14, D-005, partial #9 (D-005 only — per-batch/3-channel cutmix is a separate refactor).
- STILL OPEN: #5 (LOW; documented), #2-deeper (EMA decay default + callback to invoke it), #3-deeper (pseudo-label depth on unlabeled), #9-deeper (per-sample cutmix/color factors).
- ADDED: #12 — tests added in this plan, so REMOVED from open list.

### Corrections
*None yet.*
