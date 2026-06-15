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

## plan_2026-06-15_2485b951
### Scope
13 norm layer files in `src/dl_techniques/layers/norms/` (+ `factory.py`, `__init__.py`, `README.md`).
Classes: RMSNorm, ZeroCenteredRMSNorm, BandRMS, AdaptiveBandRMS, ZeroCenteredBandRMSNorm,
ZeroCenteredAdaptiveBandRMS, BandLogitNorm, LogitNorm, MaxLogitNorm, DecoupledMaxLogit, DMLPlus,
GlobalResponseNormalization, DynamicTanh, PolarWeightNorm.

### Index
| # | File (detailed) | Topic | Key conclusion |
|---|-----------------|-------|----------------|
| F-contract | findings/contract-conformance.md | Keras-3 lifecycle conformance | explorer hypotheses; SOURCE-VERIFIED below |
| F-graph | findings/graph-safety-correctness.md | graph-safety + "as advertised" | explorer hypotheses; SOURCE-VERIFIED below |
| F-factory | findings/factory-docs-deferred.md | factory/docs/deferred items | explorer hypotheses; SOURCE-VERIFIED below |

### SOURCE-VERIFIED ISSUE LIST (orchestrator confirmed against current source)

### Correctness bugs (confirmed)
- **B1 [HIGH] MaxLogit squeeze crash.** `max_logit_norm.py:318` (DecoupledMaxLogit.call) + `:492` (DMLPlus center): `ops.squeeze(ops.max(norm, axis=self.axis), axis=-1)`. `ops.max` over a keepdims=True norm already drops that axis, so `squeeze(...,axis=-1)` targets the batch dim → `Cannot squeeze axis=-1` for batch>1. Fix: drop the squeeze, use `ops.max(norm, axis=self.axis)` (matches `max_cosine` pattern on line 315). **ZERO src consumers** (grep) → unexercised but real; pin with a test.
- **B3 [MEDIUM] band_regularizer not deserialized.** `band_rms.py:172` + `zero_centered_band_rms_norm.py:187`: `self.band_regularizer = band_regularizer or keras.regularizers.L2(1e-5)` — skips `keras.regularizers.get()` (unlike `band_initializer` which IS `.get()`-wrapped). A serialized dict from `from_config` is stored verbatim (truthy dict), so the next `get_config`→`regularizers.serialize(dict)` breaks for any non-None custom regularizer. Fix: `keras.regularizers.get(band_regularizer) or keras.regularizers.L2(1e-5)`. NOTE: the default-None→L2(1e-5) behavior is ADVERTISED in the docstring — keep it (round-trips fine for the None case because add_weight re-gets internally). AdaptiveBandRMS/ZeroCenteredAdaptiveBandRMS correctly use `.get()` and default to None (no injected L2) — inter-layer inconsistency is intentional/documented, NOT a bug.
- **B4 [MEDIUM] DynamicTanh mutates ctor state in build.** `dynamic_tanh.py:167` `self.axis = normalized_axis` overwrites the ctor `axis` with the build-normalized (positive) list; `get_config` then serializes post-build state (build-state-dependent). Production-used (vit/vit_siglip/vit_hmlp). Fix: keep `self.axis` = ctor value; store normalized axes in a separate attr (e.g. `self._norm_axis`) used by build/call. Zero behavior change.

### "Operate as advertised" — design-level (decision required)
- **B2 [MEDIUM] BandLogitNorm degenerate adaptive mechanism.** `band_logit_norm.py:162-172`: creates `LayerNormalization(axis=-1)` and applies it to `x_length` of shape `[...,1]`. LayerNorm over a size-1 axis is identically 0 → `tanh(0)=0` → scale collapses to the constant `1 - 0.5*max_band_width`. The "adaptive"/learnable claim is non-functional; behaves as L2-normalize × constant. Also `build()` calls `super().build()` FIRST (line 158) then creates the sublayer (anti-pattern). **Used by `train/rms_variants_train/`** (harness even references the inner LayerNorm). Real math fix = redesign = EXPANSION → out of scope. DECISION: apply Keras-3 contract fixes (create LayerNormalization in `__init__`, add built-guard) WITHOUT changing the degenerate math, and DOCUMENT the limitation honestly in the docstring.

### Keras-3 contract (mechanical, zero-regression)
- **C1 [MEDIUM] Missing `if self.built: return` guard.** ZERO grep hits for `if self.built` across the whole package. All weight/sublayer-creating `build()` methods must have it as the FIRST line (canonical sweep, confirmed zero-regression across attention/ffn/embedding/activations). Affected (10): rms_norm, zero_centered_rms_norm, band_rms, adaptive_band_rms (Dense in build), zero_centered_band_rms_norm, zero_centered_adaptive_band_rms_norm (Dense in build), global_response_norm, polar_weight_norm, dynamic_tanh, band_logit_norm. Stateless (no build/weights, N/A): logit_norm, max_logit_norm (all 3 classes — `constant` is a plain float, not a weight).
- **C2 [HIGH] `__init__.py` broken.** (a) `__all__` contains class OBJECTS, not strings — must be `List[str]`. (b) Missing imports/exports for 5 classes: AdaptiveBandRMS, GlobalResponseNormalization, DynamicTanh, ZeroCenteredRMSNorm, ZeroCenteredBandRMSNorm (README's direct-import example for ZeroCenteredBandRMSNorm is a broken ImportError). (c) factory helpers `get_normalization_info`/`validate_normalization_config` not exported. Fix: import all 14 classes + factory symbols; `__all__` as strings.

### Factory / docs (low)
- **F1 [LOW] `validate_normalization_config` whitelist incomplete.** `get_normalization_info()` 'parameters' lists omit valid ctor params, causing false `ValueError` via `validate`: band_rms/adaptive_band_rms missing `band_initializer`/`band_regularizer`; global_response_norm missing `gamma_regularizer`/`beta_regularizer`/`activity_regularizer`. Fix: complete the lists (source-of-truth = class `__init__`).
- **D1 [LOW] doc nits.** LogitNorm docstring says `norm = sqrt(sum(x²)+ε)` but code is `sqrt(max(sum(x²),ε))` (logit_norm.py:175-176); also calls temperature "learnable" though it's a fixed float. DecoupledMaxLogit/DMLPlus comment "learned weight" but `constant` is a fixed float. Fix docstrings to match code.
- **README** — add the 5 layers missing from the export example; verify factory-key table matches the 16 Literal types.

### Confirmed NOT bugs (explorer over-flags — per LESSONS "claims are hypotheses")
- PolarWeightNorm `convert_to_numpy`/`np.*` are in `build()` ONLY (init-time seed-kernel→polar encode, lines 311/321). `build()` runs eagerly in Keras 3; `call()`/`_reconstruct_kernel()` are pure `keras.ops` = graph-safe. Host materialization is necessary (LESSONS L46). NOT a fix target.
- BandRMS `band_regularizer or L2` default-injection is ADVERTISED and round-trips for None case (downgraded from "bug" to the narrower B3 `.get()` asymmetry).
- `len(input_shape)`/`len(tensor.shape)` static-rank usages are graph-safe (LESSONS).

### Deferred items (from prior plans) — verified disposition
- **FW-1** PolarWeightNorm factory registration → WON'T-FIX BY DESIGN. PolarWeightNorm is a *weight* reparameterization (Dense replacement: radius+angles, units param), NOT an activation-normalization layer; it does not fit `create_normalization_layer`'s contract. Documented as not-registered in `layers/CLAUDE.md` + module docstring. Keep documented.
- **FW-2** angle_regularizer toward π/4 prior; **FW-3** benchmark vs Dense; **FW-4** PolarOrthogonalLayer; **kernel-cache** caveat → all NEW functionality → OUT OF SCOPE ("don't expand functionality").
- **Coverage gap** → NO tests for logit_norm, max_logit_norm (3 classes), and NO factory test. Adding regression tests for existing code is in-scope (pins B1).

### Key Constraints
- **HARD**: No eager ops in any traced (`call`) path; must be `@tf.function`/graph-compatible. (Build-time eager init is allowed per Keras 3 + LESSONS L46.)
- **HARD**: Full Keras-3 serialization round-trip (`get_config`/`from_config`, `.keras` save/load) must hold for every layer.
- **HARD**: Fix existing behavior only — DO NOT expand functionality / add features (no new layers, no new params, no redesigns).
- **HARD**: `@keras.saving.register_keras_serializable()`; do NOT change any existing `package=` (registration key) — would break already-saved models.
- **SOFT**: Follow `research/2026_keras_custom_models_instructions.md` canonical patterns (built-guard, sublayers in `__init__`).
- **SOFT**: Net-neutral/negative line count; mechanical guard sweep preferred over rewrites.
- **GHOST (to verify, not assume)**: factory `epsilon=1e-6` default vs class `1e-7` defaults (F2). Changing it alters numerics of existing factory-built/saved models (layer_norm/batch_norm would shift 1e-6→Keras 1e-3). Lean: leave documented default, do NOT silently change. Decide at PLAN.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*

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
