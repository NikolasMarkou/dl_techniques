# Consolidated Findings
*Cross-plan findings archive. Entries merged from per-plan findings.md on close. Newest first.*

<!-- COMPRESSED-SUMMARY -->
## Summary (compressed)
*Auto-compressed. Read full content below for plan-specific detail.*

### Key Findings
- **models-vs-layers audit REPLACE verdicts are hypotheses (plan_2026-06-13_ae26345d)**: 3 of 4 Tier-1 "drop-in" REPLACE verdicts in `research/2026_models_layer_reuse_audit.md` were refuted on implementation-time source read: DINOv2Block (6 structural mismatches with TransformerLayer), ByteTokenizer (different special-token attr names + 4 shared BLT callers), TRMReasoningModule (incompatible positional `call()` signature). Only `_LayerScale1D → LearnableMultiplier` was confirmed and executed. Correction addendum appended at `research/2026_models_layer_reuse_audit.md:464`. **Any future implementation of that audit MUST re-verify each finding before acting.**
- **`LearnableMultiplier` default divergence (plan_2026-06-13_ae26345d)**: `constraint='non_neg'`, `initializer='ones'` are the defaults; `_LayerScale1D` used no constraint and `Constant(1e-5)`. Both MUST be overridden explicitly (`constraint=None, initializer=Constant(1e-5)`) or the swap silently changes gradient dynamics.
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

## plan_2026-06-15_0205772c
### Index

| # | Finding | Severity | File:line | Status |
|---|---------|----------|-----------|--------|
| F1 | `ExpandedActivation.build()` (parent of xATLU/xGELU/xSiLU) lacks `if self.built: return` guard; `add_weight('alpha')` doubles on re-build | HIGH | expanded_activations.py:286 | VERIFIED |
| F2 | `DifferentiableStep.build()` lacks guard; doubles slope+shift weights | HIGH | differentiable_step.py:160 | VERIFIED |
| F3 | `ThreshMax.build()` lacks guard; doubles slope weight | HIGH | thresh_max.py:174 | VERIFIED |
| F4 | `RoutingProbabilitiesLayer.build()` lacks `if self.built: return` (7 add_weight); super().build() at 505 is fine but no top guard | HIGH | routing_probabilities.py:499 | VERIFIED |
| F5 | `DifferentiableStep.__init__` mutable default instances: `keras.regularizers.L2(1e-3)`, `ValueRangeConstraint(-1,+1)` (Pitfall 8; stateless so benign at runtime, contract violation) | MEDIUM | differentiable_step.py:139-140 | VERIFIED |
| F6 | Factory `differentiable_step` registry sets `shift_constraint: None`, OVERRIDING class default `ValueRangeConstraint(-1,+1)` → factory-built layer silently loses constraint. Also shared L2 instance. | HIGH | factory.py:102-103 | VERIFIED |
| F7 | Factory `thresh_max` registry: `trainable_slope: True` contradicts class default `False` (test asserts False); `slope_regularizer L2_custom(-1e-3)` vs class `-1e-4`; shared mutable instances | HIGH | factory.py:269-272 | VERIFIED |
| F8 | `MonotonicityLayer._sigmoid` does NOT guarantee monotonicity (advertised "non-decreasing"). Per-element deviation `±0.25*(max-min)` can exceed adjacent spacing `(max-min)/(n-1)` → inversions; clip does not restore order. | MEDIUM (correctness) | monotonicity_layer.py:505-509 | VERIFIED |
| F9 | Build guards absent on weightless build() overrides (adaptive_softmax:192, monotonicity build, probability_output:211) — re-validate/re-build-sublayer on double call; cheap consistency fix | LOW | (3 files) | VERIFIED |
| F10 | `elu_plus_one_plus_epsilon` uses `keras.activations.elu` + `keras.backend.epsilon()` — graph-SAFE (traceable + python float) but not `keras.ops`; purity nit | LOW | expanded_activations.py:523 | VERIFIED |

### Explorer claims DOWNGRADED / REFUTED on source-verify (do NOT fix)
- `len(inputs.shape)` in monotonicity_layer (319/324/495), routing (663/508), monotonicity build: returns **static rank** as Python int → graph-SAFE. NOT a bug. (Explorer A/C/D overstated "graph-unsafe".)
- `keras.activations.elu/softmax`, `keras.backend.epsilon()`: graph-traceable / python float → NOT eager. Downgraded to F10 LOW.
- `ProbabilityOutput` absent from factory: INTENTIONAL — it is a meta-dispatcher that itself instantiates activation layers (softmax/sparsemax/routing strategies). Direct-import-only is correct; do NOT register. Document only.
- `probability_output.get_config` stores `type_config` verbatim: edge case (nested keras objects); current serialization tests pass. OUT of scope (no expansion).
- `SaturatedMish.mish_at_alpha` not in config: deterministically rederived from `alpha`. Not a bug.
- `relu_k.py:136` raises TypeError vs ValueError: LOW style, OUT of scope.

### Key Constraints

- **[HARD]** Fix-only, NO functionality expansion. Target existing defects + contract gaps; do not add features. (user directive)
- **[HARD]** All `call()` paths must be graph-compatible — no eager ops. (user: "no EAGER SHIT"). VERIFIED: grep shows NO `tf.*`, NO `.numpy()`, NO `int(tensor)` anywhere in the package. Graph-safety is already largely sound; the genuine issue space is idempotency guards, factory/class default consistency, and one math-correctness bug.
- **[HARD]** Keras-3 contract per `research/2026_keras_custom_models_instructions.md`: `if self.built: return` first line of every build() (repo-wide convention, established for embeddings iter-1/step-2 commit b8b88fe3, and FFN/attention packages per SYSTEM.md).
- **[HARD]** Full save/load round-trip must keep passing. Baseline: **367 passed / 0 failed** (`tests/test_layers/test_activations/`, CUDA_VISIBLE_DEVICES=1, 69s).
- **[SOFT]** Factory registry `optional_params` should mirror class defaults (advertised via `get_activation_info()`); divergence is the bug.
- **Blast radius**: `MonotonicityLayer` has ZERO src callers outside the package. Factory keys `thresh_max`/`differentiable_step` have ZERO src callers. `ThreshMax`/`DifferentiableStep` used directly only by capsule_routing_attention.py (direct class, unaffected by factory changes). Low risk across the board.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*

## plan_2026-06-15_9dbb87c1
### Index

| ID | Topic | File | Status |
|----|-------|------|--------|
| F1 | Factory/spine wiring + `__init__` defects | findings/spine-factory.md | OPEN |
| F2 | RoPE family review (rotary/dual/continuous/multi_axis) | findings/rope-family.md | pending explorer |
| F3 | Continuous/scalar/positional family review | findings/continuous-positional.md | pending explorer |
| F4 | BERT-family review (bert/modern_bert/albert) | findings/bert-family.md | pending explorer |
| F5 | Patch + hierarchical-codebook review | findings/patch-codebook.md | pending explorer |

### Key Constraints

### HARD
- Keras 3.8 / TF 2.18; `keras.ops` only (no raw TF in library code, narrow documented exceptions allowed).
- **Graph-compatible**: NO eager ops reachable from `call()` (no `convert_to_numpy`, `.numpy()`, `float(tensor)` on traced tensors). User explicit: "no EAGER SHIT, all must be graph compatible".
- Canonical Keras-3 lifecycle: `@register_keras_serializable`, `None`-sentinel ctor for build-time dims, `if self.built: return` FIRST line of `build()`, explicit sublayer `.build()` in parent `build()`, full `get_config`/`from_config` round-trip, `compute_output_shape`.
- `training is True` Python identity idiom (never `if training:` / `if training is not False:`).
- Scope: FIX existing functionality only — NO new functionality/expansion. Target DEFERRED items.
- Work autonomously, no questions.

### SOFT
- Keras guide: `research/2026_keras_custom_models_instructions.md` (authoritative).
- Docstrings Google-style; logging via `dl_techniques.utils.logger` only.

### GHOST (to verify)
- README.md + layers/CLAUDE.md describe an older roster (mention ModernBERT but NOT albert/hierarchical_codebook/sine_2d/scalar_sinusoidal/mrope) — doc drift, not a code constraint.

### Preliminary direct-read facts (orchestrator, pre-explorer)

- **14 layer files** in `src/dl_techniques/layers/embedding/` + `factory.py` + `__init__.py` + `README.md`.
- **Factory registry = 10 keys**: patch_1d, patch_2d, positional_learned, rope, dual_rope, continuous_rope, continuous_sincos, bert_embeddings, scalar_sinusoidal, mrope_ideogram4.
- **NOT factory-registered (4 classes)**: `AlbertFactorizedEmbedding`, `ModernBertEmbeddings`, `HierarchicalCodebookEmbedding`, `PositionalEmbeddingSine2D`. (Verify whether each SHOULD be — standard call sig? — or is intentionally direct-import.)
- **`__init__.py` `__all__` BUG**: `__all__ = [create_embedding_from_config, create_embedding_layer, validate_embedding_config]` lists the function OBJECTS, not their string names. `__all__` must be a list of `str`. Real defect (breaks `from ... import *` / tooling).
- **`__init__.py` does NOT export layer classes** (only 3 factory funcs). Consistent with "import from submodules" convention, but factory funcs are the only re-exports.
- **EAGER SUSPECTS (graph-breaking, prime targets)**:
  - `continuous_sin_cos_embedding.py:237: if ops.convert_to_numpy(min_coords) < 0:` — if reachable from `call()` under `assert_positive`, breaks graph trace.
  - `continuous_rope_embedding.py:224: if ops.convert_to_numpy(min_coords) < 0:` — same pattern.
  - (np.arange/np.eye in `build()`/`__init__` are trace-time constant folding — graph-SAFE, not targets.)
- **Test coverage GAPS** (no test file): `continuous_rope_embedding`, `continuous_sin_cos_embedding`, `positional_embedding_sine_2d`. Factory has only `test_factory_ideogram4.py` (no general factory test covering all 10 keys / param passthrough).
- Existing tests present for: dual_rotary, patch, hierarchical_codebook, modern_bert, multi_axis_rope, rotary_position, scalar_sinusoidal, positional_embedding, albert_factorized, bert.

### Verified-by-orchestrator (direct source read, supersedes explorer claims)

- **[REFUTED] bert_embeddings.py:222 was NOT a CRITICAL bug.** `_create_normalization_layer(self, name)` takes `name` as the *sublayer name string*; its body (lines 241-265) branches on `self.normalization_type` and returns LayerNorm/RMSNorm/BandRMS/BatchNorm accordingly. Passing `"layer_norm"` is just the sublayer's `.name`. `normalization_type` IS honored. The only cosmetic quirk: the sublayer is always named "layer_norm" even when it is an RMSNorm. NOT a defect. (bert build() also correctly has `if self.built: return` at :275 + explicit sublayer builds.)
- **[VERIFIED] continuous_sincos `compute_output_shape` returns `dim` and that is CORRECT.** Math: merged trailing = ndim·effective_dim_per_wave = (dim−padding); plus `padding` zero cols = dim. No fix needed.
- **[VERIFIED] continuous_rope `compute_output_shape` returns `dim` but ACTUAL output is `ndim·(effective_dim_per_wave//2) + padding//2 = dim/2`** (phase angles, no sin/cos doubling). REAL bug — `compute_output_shape` over-reports by 2x. Docstring also says `(..., dim)`. ZERO callers in repo (safe to fix). Fix = correct compute_output_shape + docstring to actual phase width; do NOT change call() (dim/2 phases is conventional RoPE design).
- **[VERIFIED] `__init__.py` `__all__` lists function OBJECTS not strings** — real defect (`__all__ = [create_embedding_from_config, ...]`).
- **[VERIFIED] eager graph-breaks** continuous_sin_cos:237 + continuous_rope:224 (`ops.convert_to_numpy` in call() under default `assert_positive=True`); both are pure advisory `logger.warning` blocks with no functional effect → graph-safe fix = delete the block, keep `assert_positive` param accepted+serialized for config compat.
- **[DECISION] multi_axis_rope.py:56 `package="dl_techniques.layers"` — do NOT "align".** Changing a `register_keras_serializable` package= string changes the registration KEY and breaks deserialization of already-saved `.keras` models (attention-plan C2 lesson). Leave as-is.

### Real callers (grep-verified)
- `ContinuousRoPE`: **ZERO callers** anywhere in src/.
- `ContinuousSinCosEmbed`: text_decoder.py, text_encoder.py (via factory 'continuous_sincos'), supernode_pooling.py (direct). Eager bug affects real callers.
- `PositionEmbeddingSine2D`: detr/model.py, video_jepa/encoder.py (direct import). Class name is `PositionEmbeddingSine2D`.
- `ModernBertEmbeddings`: modern_bert/modern_bert.py (direct import).
- `AlbertFactorizedEmbedding`: no src callers (standalone).

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*
</content>
</invoke>

## plan_2026-06-14_5e80bd3e
### Index

| ID | Topic | Key takeaway |
|----|-------|--------------|
| F-SITES | The 3 side-effect sites | KMeans `call:619` gates `self._update_centroids()` (which does TWO state writes: `centroid_momentum.assign` kmeans.py:571 + `centroids.assign_add` kmeans.py:574). GMM `call:573` gates `add_loss(self._isometric_regularization_loss())`. RBF `call:300` gates `add_loss(self._compute_repulsion_loss())`. All three currently gated by `if training is True:` (from prior plan plan_2026-06-14_7384c2e3/D-001) → skip under symbolic tensor. |
| F-PRECEDENT | Graph-safe conditional idiom in repo | Repo uses `keras.ops.cond` for graph-safe VALUE selection (stochastic_gradient.py:141, prism_blocks.py:499). No training-resolution helper exists anywhere. `utils/tensors.py` is the natural home for one. For SIDE EFFECTS (add_loss / .assign), `ops.cond` branches are awkward (must return matching structures); tensor MASKING (multiply the delta/loss by `cast(training)`) is cleaner and equally graph-safe. |
| F-PROTO | Empirically validated fix idiom | Standalone @tf.function prototype confirms: gate `if training is not None and training is not False:` + factor `1.0 if training is True else keras.ops.cast(training, dtype)`, then MASK state-deltas/loss by factor → symbolic True fires both side-effects, symbolic False is a TRUE no-op (centroid unchanged, loss=0), python True preserved exactly, None graph-safe under tf.function. No graph break (all branching is on python identity/type, never on a tensor value). |
| F-NUMERIC | Zero-regression guard | To preserve EXACT numerics on the existing python-True training tests, the python-True path must keep the unmasked `assign`/`add_loss` (factor==1.0 float → exact); only the symbolic-tensor path uses the masked formula. Branch on `isinstance(factor, float)` (a python-type check, static at trace time, graph-safe). |

### Key Constraints

- **[HARD]** Side-effects MUST fire when `training` is a symbolic `tf.Tensor` that is True at runtime (the user's foot-gun). This REVISES prior plan plan_2026-06-14_7384c2e3/D-001 which intentionally skipped the tensor case.
- **[HARD]** Everything stays graph-compatible — no eager, no bool-coercion of a tensor. (carried constraint)
- **[HARD]** Zero numeric regression on the python-bool / None call paths (existing 112 tests must stay green). Preserve exact python-True update via `isinstance(factor, float)` fast path (F-NUMERIC).
- **[HARD]** Symbolic-False at runtime must be a true no-op for STATE (no centroid/momentum drift at inference) and zero loss contribution.
- **[SOFT]** Single source of truth for the factor logic (shared `resolve_training_factor` helper in utils/tensors.py) → uniform across all 3 layers, testable once. (DRY; 3 call sites = earned abstraction)
- **[SOFT]** Update the now-stale in-code comments (added last plan) that claim "symbolic/None/False skip the side-effect" — symbolic no longer skips.
- **[GHOST]** Prior LESSONS/SYSTEM say `training is True` is THE canonical idiom. That holds for layers whose side-effects are inference-irrelevant, but is the wrong contract for layers that must support symbolic-training custom loops. Will reconcile at CLOSE (idiom is context-dependent, not absolute).

### Corrections
*None yet.*

## plan_2026-06-14_7384c2e3
### Index

| ID | Topic | File | Key takeaway |
|----|-------|------|--------------|
| F-GRAPH | Graph/eager audit | findings/graph-eager-audit.md | The ONLY graph-breaker class is bare `if training:`/`if training and ...` on a value that may be a symbolic tensor — at kmeans.py:619, gmm.py:571, rbf:298. All 3 raise `OperatorNotAllowedInGraphError` when `training` is a `tf.Tensor`. No `.numpy()`/`convert_to_numpy`/`py_function`/dynamic-loop escapes anywhere. `.assign`/`add_loss` are graph-safe. Empirically: all 3 PASS @tf.function with python-bool, FAIL with tensor-bool. |
| F-CONTRACT | Contract & keras-rules compliance | findings/contract-keras-compliance.md | HIGH: `compute_output_shape` in GMM (gmm.py:391) + KMeans (kmeans.py:402) reads build-mutated `self.cluster_axis` → wrong/inconsistent for multi-axis pre-build. MED: KMeans+RBF lack `from_config` (GMM has it); GMM+KMeans `build()`→`_setup_cluster_axes()` mutates `self.cluster_axis` in place with no reset from `_cluster_axis_arg` → double-build corruption. LOW: kmeans.py:219 stores tuple raw (GMM uses `list()`); kmeans regularizer serialize guard inconsistency. |
| F-FACTORY | Factory/docs/tests/as-advertised | findings/factory-docs-tests.md | Factory wiring: CLEAN (all 3 wired, exports complete, matches ffn/norms sibling contract). Advertised math: all correct. Docs: 1 soft notation mismatch (README.md:9 RBF formula uses textbook `2*sigma^2` vs impl `exp(-gamma*dist_sq)`). Tests: 105 pass / 0 fail; gaps = KMeans has NO `.keras` round-trip test, and ZERO graph-mode (@tf.function tensor-training) tests across all 3. |
| F-IDIOM | Canonical graph-safe training-gate idiom | (this file) | Repo canonical idiom is `if training is True:` — documented verbatim in residual_acf.py:288-291, and used in mdn_layer.py:291, deep_kernel_pca.py:560, vector_quantizer_rotation_trick.py:369/390/392, logic operators. `is` identity check never coerces a tensor to bool (graph-safe); `None`/`False`/symbolic all skip the training-only side-effect, which is the intended contract for EMA updates + add_loss. This SUPERSEDES the explorer's `is not False` suggestion (see Corrections). |

### Key Constraints

- **[HARD]** All layers must be graph-compatible: no construct in `call()` (or anything reachable from it) may coerce a possibly-symbolic tensor to a python bool. Fix the 3 `if training` guards. (user: "no EAGER SHIT, all graph compatible")
- **[HARD]** Fix-only, no functionality expansion. Target the existing defects + deferred items; do not add features. (user directive)
- **[HARD]** `compute_output_shape` must return correct shapes without the layer being built (Keras functional-API tracing calls it pre-build).
- **[HARD]** The training-gate fix MUST use the repo's canonical `training is True` idiom (F-IDIOM), not a bare truthiness or an ad-hoc variant, for keras-rules compliance + cross-layer uniformity.
- **[SOFT]** The 3 layers should comply to the SAME contract (uniform `from_config`, uniform `cluster_axis` normalization, uniform serialize style). (user: "comply to the same contract")
- **[SOFT]** README RBF formula notation should match the implementation.
- **[GHOST]** Explorer-suggested `if training is not False:` — REJECTED; it would fire training-only side effects under `training=None`/symbolic, contradicting the repo's documented `training is True` contract. Not a real constraint.
- Existing `# DECISION plan_2026-06-14_8c7365d0/D-005` (cluster_axis stash) and `plan_2026-06-08_57a975d1/D-002` (orthonormal lazy-resolve) anchors are CORRECT-BY-DESIGN — preserve, do not touch.

### Corrections

- **[CORRECTED iter-0]** findings/graph-eager-audit.md recommends the fix `if training is not False:` (claimed KAN/SwinMLP precedent). VERIFICATION FAILED: grep shows kan_linear.py / swin_mlp.py do NOT use that idiom, and no layer in the repo uses `is not False`. The actual canonical idiom is `if training is True:` (residual_acf.py:288-291 documents it explicitly; also mdn_layer, deep_kernel_pca, vector_quantizer_rotation_trick, logic operators). `is not False` is semantically WRONG (fires under `training=None`). Use `training is True`. See F-IDIOM.
