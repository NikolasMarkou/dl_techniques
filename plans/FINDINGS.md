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

## plan_2026-06-03_943569ad
### Index
| # | Finding | File | Covers |
|---|---------|------|--------|
| 1 | ConvNeXt drop_path is `keras.layers.Dropout(noise_shape=(None,1,1,1))` applied per-block on the residual branch, with a linear rate schedule `rate*i/(N-1)`; blocks themselves have no drop_path. | findings/convnext-droppath-usage.md | problem scope, existing pattern |
| 2 | `StochasticDepth(drop_path_rate)` is a per-sample drop+rescale layer — a TRUE drop-in for the current `Dropout(noise_shape=(None,1,1,1))`. `StochasticGradient(drop_path_rate)` is forward-identity, only stochastically `stop_gradient`s — a DIFFERENT regularizer, not drop_path. | findings/stochastic-layers-api.md | solution space, constraints |
| 3 | Blast radius: 4 exported symbols, 3 training scripts pass `drop_path_rate`, tests assert `model.drop_path_rate` attr + config round-trip + a `test_stochastic_depth` forward-shape test. Block layers (used by 5 other models) carry NO drop_path, so block-level blast radius is zero. | findings/convnext-blast-radius.md | affected files, test gates |

### Key Constraints
- [HARD] Public kwarg name `drop_path_rate` is serialized in `get_config()` (convnext_v1.py:626, convnext_v2.py:682) and asserted by tests + passed by 3 training scripts. **Keep the kwarg name `drop_path_rate`** — only swap the internal mechanism, do not rename the public API.
- [HARD] `StochasticDepth` rescales by `1/keep_prob` and is per-sample — semantically equivalent to the current `Dropout(noise_shape=(None,1,1,1))`. So swapping to `StochasticDepth` preserves training behavior.
- [HARD] Current per-block `Dropout` objects are stored in a plain `list` of `dict`s (`stage_blocks`), NOT as tracked sublayer attributes. Keras may not auto-track layers in nested Python containers — the replacement should keep the same storage pattern (Dropout/StochasticDepth have no weights, so this is a tracking/naming concern, not a correctness one).
- [SOFT] `StochasticDepth` is the established repo idiom (10+ call sites incl. swin, clifford, dino); `StochasticGradient` has exactly one usage (an experiment script), zero production-model usage.
- [SEMANTIC] `StochasticGradient` is NOT a drop-in for drop_path: it leaves the forward pass unchanged, so it removes the forward-activation regularization that drop_path provides. Using it changes model behavior materially.
- [GHOST] None identified.

### Resolved Decision
User chose **configurable: both available**. Add a `stochastic_mode: str = 'depth'` kwarg to both models. `'depth'` -> `StochasticDepth` (default, preserves current behavior); `'gradient'` -> `StochasticGradient`. Keep `drop_path_rate` public kwarg unchanged. Plumb `stochastic_mode` through `get_config()` and validate against `{'depth','gradient'}`.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-06-02_da7698bc
### Index

| # | Finding file | Covers | Key takeaway |
|---|--------------|--------|--------------|
| 1 | `findings/target-files.md` | Current `polar_weight_norm.py` (348 ln) + `.md` (130 ln) | `.py` is ALREADY Keras-3 compliant (typed, `get_config` complete, `compute_output_shape`, `logger`, `keras.ops`-only `call`, Google-style class docstring w/ `**Intent**`/`**Architecture**`). `.md` carries extra content: usage example, guarantees, per-forward caveat, provenance/arXiv, AND full `PolarInitializer` docs (a *different* file). |
| 2 | `findings/template-and-instructions.md` | RBF template + 2026 instruction doc (M1-M15) | Instruction doc = authority. Code satisfies all 15 MUST rules already. Only real divergence is docstring dialect (Google authoritative; module docstring currently has Sphinx `:func:`/`:class:` roles). RBF sets the bar for module-docstring richness (summary + math + references). |
| 3 | `findings/butterfly-precedent.md` | `orthogonal_butterfly.py` did the IDENTICAL task (commits `836a7798`+`30b4a1dd`) | Pattern = (A) merge `.md` into module docstring + add `# ---` divider before decorator and at EOF (`.py` only, commit 1); (B) delete `.md` + fix doc pointers (commit 2). Section order: one-liner -> description -> Mathematical Foundation -> Properties -> Constraints -> When to Use -> References. |

### Verified facts (this plan)
- Code passes M1-M15 of `2026_keras_custom_models_instructions.md` already (verified line-by-line in finding 2). "Refine code to pass instructions" is largely VERIFY-ONLY + cosmetic (dialect cleanup, `# ---` dividers).
- Exports OK: `src/dl_techniques/layers/norms/__init__.py:7` exports `PolarWeightNorm, polar_encode, polar_decode`.
- Test file exists: `tests/test_layers/test_norms/test_polar_weight_norm.py` (8.3 KB) -> scoped-pytest verification target.
- **4 pointer sites** reference `polar_weight_norm.md` (must fix on deletion):
  1. `src/dl_techniques/layers/CLAUDE.md:16` — "see `norms/polar_weight_norm.md`."
  2. `src/dl_techniques/layers/norms/README.md:22` — "See `polar_weight_norm.md` for the full design..."
  3. `src/dl_techniques/initializers/README.md:171` — "(see `dl_techniques/layers/norms/polar_weight_norm.md`)."
  4. `src/dl_techniques/layers/orthogonal_butterfly.py:71` — docstring cross-ref "see norms/polar_weight_norm.md."

### Key Constraints

- **[HARD]** `@keras.saving.register_keras_serializable()` bare (no `package=`) — LESSONS confirms the bare form is correct here; do NOT add `package=`.
- **[HARD]** `keras.ops`-only in `call`; `logger` not `print`; full `get_config` round-trip — all already satisfied, must be PRESERVED through edits.
- **[HARD]** Existing module docstring (lines 1-23) and class docstring (146-195) content must be PRESERVED/extended, not dropped.
- **[HARD]** Google docstring dialect is authoritative over Sphinx (instruction doc > RBF template dialect).
- **[SOFT]** `# ---` dividers: one before class decorator, one at EOF (butterfly precedent).
- **[SOFT]** Merge-then-delete in TWO commits (precedent).
- **[GHOST]** "match RBF quality" does NOT mean copy RBF's Sphinx `:param:` dialect — it means structural completeness/polish. The `.py` already exceeds RBF on Keras-3 compliance (RBF lacks `logger`).
- **[DECISION-PENDING]** `PolarInitializer` lives in a *separate* file (`initializers/polar_initializer.py`) with its OWN rich module docstring. Default decision: CROSS-REFERENCE it from the merged docstring, do NOT duplicate its full args table/example (DRY / single-source). Surface for user approval at PC-PLAN.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-06-02_2a0b8192
### Index

| # | Topic | File | Key takeaway |
|---|-------|------|--------------|
| 1 | Current state of orthogonal_butterfly (src + doc) | findings/orthogonal-butterfly-current.md | Layer is already high-quality and instruction-compliant; only the bare decorator was flagged. `.md` (86 lines) is a richer reference than the current top docstring. Comprehensive test file exists. Not exported from `__init__` (intentional). No production importers. |
| 2 | RBF quality template | findings/rbf-quality-template.md | RBF = quality bar: rich module docstring (bold-prose sections + unicode math + References), `# ---` 69-dash rules before/after class, complete get_config, compute_output_shape always present. NOTE: RBF uses Sphinx `:param:` docstrings, NO logger, NO `package=`. |
| 3 | Keras custom-layer instructions checklist | findings/keras-instructions-checklist.md | HARD: register decorator, all args stored, add_weight only in build(), super().build() last, keras.ops only, compute_output_shape required, get_config complete, round-trip test. Google-style docstrings + logger. `package=` is SOFT/optional. |

### Key Constraints

### HARD
- Custom layer must keep `@keras.saving.register_keras_serializable()`, `keras.ops`-only `call()`, `add_weight` only in `build()`, `super().build()` last, complete `get_config()`, `compute_output_shape()` present, round-trip serialization test passing. **The current code already satisfies all of these.**
- Google-style docstrings are the instruction-mandated standard (`Args:` / `Input shape:` / `Output shape:`). The current butterfly code already uses Google style.

### SOFT
- `# ---` horizontal rule (69 dashes) before and after the class block (RBF convention) — not currently present in butterfly.
- Rich module-level docstring with bold-prose section headings + References bullets (RBF convention).
- `package=` arg in the register decorator: OPTIONAL per instructions; RBF (the named quality template) OMITS it. → keep bare.

### GHOST (constraints that look binding but are not)
- "Bare decorator is a violation" (flagged by explorer 1) — FALSE. Instructions mark `package=` optional; RBF, the explicit quality template, omits it. Keeping it bare satisfies both authorities and avoids changing the registered serialization key (zero risk on an unused layer).
- "Match RBF means convert to Sphinx `:param:` docstrings" — FALSE. Instructions mandate Google style; converting would VIOLATE the instructions. "Match quality" = structural completeness/polish, not docstring dialect.
- "RBF has no logger, so strip logger from butterfly" — not required. Instructions permit logger; the 3 existing `logger.debug` calls are compliant and harmless. Keep to avoid churn.

### Decision points (surfaced at PC-PLAN)
- The `.md` will be DELETED after merging into the top-of-file docstring ("merge" = consume the source). `layers/CLAUDE.md` references `orthogonal_butterfly.md` and must be updated. The `.md` itself links to `polar_weight_norm.md` (sibling kept separate) — that sibling is unaffected.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

- [CORRECTED iter-0] findings/orthogonal-butterfly-current.md called the bare `@register_keras_serializable()` "the only Keras 3 convention violation." Cross-checked against findings/keras-instructions-checklist.md ([SOFT] §8.1) and findings/rbf-quality-template.md ([GHOST], RBF omits it): it is NOT a violation. Reclassified as a GHOST constraint. Decorator stays bare.

## plan_2026-06-02_e3da3ff9
### Index

| # | Finding | Location |
|---|---------|----------|
| F1 | `neuro_grid.py` (41KB) currently at `src/dl_techniques/layers/neuro_grid.py`; target dir `src/dl_techniques/layers/memory/` exists with siblings som/mann/ntm. | `src/dl_techniques/layers/neuro_grid.py` |
| F2 | `neuro_grid.py` has two upward relative imports: `from ..regularizers.soft_orthogonal` and `from ..initializers.hypersphere_orthogonal_initializer` (lines 87-88). `..` = `dl_techniques`. After moving one level deeper these MUST become `...`. | `neuro_grid.py:87-88` |
| F3 | Only two code references import it: `tests/test_layers/test_neuro_grid.py:9` and `src/experiments/neuro_grid/mnist_reconstruction.py:26`, both `from dl_techniques.layers.neuro_grid import NeuroGrid`. | grep |
| F4 | `neuro_grid` is NOT exported from `layers/__init__.py` (no reference). Memory siblings SOM/MANN/NTM ARE exported from `memory/__init__.py`. USER DECISION: add NeuroGrid export to `memory/__init__.py`. | `memory/__init__.py` |
| F5 | Doc mention "neuro grid" in prose layer list at `layers/CLAUDE.md:86` (not an import). Low priority — informational list. | `layers/CLAUDE.md:86` |
| F6 | Test convention: memory-family tests (som/mann) currently flat in `tests/test_layers/`; subdir `test_ntm/` exists with empty `__init__.py`. USER DECISION: create `tests/test_layers/test_memory/` (with empty `__init__.py`) and move test there. | `tests/test_layers/` |
| F7 | [EXPANDED SCOPE] User: move ALL memory-package tests into `test_memory/`. Memory-package tests = test_neuro_grid.py, test_som_2d_layer.py, test_som_nd_layer.py, test_som_nd_soft_layer.py, and the `test_ntm/` dir. All use ABSOLUTE imports (no relative breakage on move). No external/CI refs to these paths. | `tests/test_layers/` |
| F8 | [EXCLUSION] `test_hierarchical_memory_system.py` imports `dl_techniques.layers.experimental.hierarchical_memory_system` — NOT the memory package. Despite the name, EXCLUDE from the move. No test_mann.py / factory test exists. | `tests/test_layers/test_hierarchical_memory_system.py:8` |

### Key Constraints

- **HARD**: The two relative imports in `neuro_grid.py` (`..regularizers`, `..initializers`) must change to `...` after the move, or the module fails to import. Verified by package-depth math.
- **HARD**: Both importers (`test_neuro_grid.py`, `mnist_reconstruction.py`) must update path to `dl_techniques.layers.memory.neuro_grid` or they break.
- **HARD (convention)**: New test subdir `test_memory/` needs an empty `__init__.py` (matches `test_ntm/`).
- **SOFT**: `memory/__init__.py` export placement should follow the existing sectioned style (NTM / MANN / SOM blocks + `__all__`).
- **GHOST**: None — no stale constraints inherited.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*
