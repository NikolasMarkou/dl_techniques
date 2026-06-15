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

## plan_2026-06-15_3028e33c
### Index

| # | Finding | Severity | Source+verify | Decision |
|---|---------|----------|---------------|----------|
| A1 | `DMLPlus.compute_output_shape` center model returns `input_shape` `(B,C)` for the norm_factor output, but `call` returns keepdims `norm` `(B,1)`. REPRO: declared `(8,5)` vs actual `(8,1)`. max_logit_norm.py:520. | [BUG] | logit-family#1, repro CONFIRMED | FIX (A) |
| A2 | Factory `get_normalization_info['dynamic_tanh']` lists only 3 params; DynamicTanh ctor accepts 5 more (bias_initializer, kernel_regularizer, bias_regularizer, kernel_constraint, bias_constraint). `validate` REJECTS what `create` ACCEPTS. factory.py:280 vs dynamic_tanh.py:106-110. | [BUG] | factory-wiring#F-1, repro CONFIRMED | FIX (A) |
| A3 | Factory validate `max_band_width` checks only `<=0`; all 5 band classes enforce `0<x<1`. REPRO: `validate(band_rms, max_band_width=1.5)` accepts, `create` rejects "must be between 0 and 1". factory.py:322. Fixing + aligning message also makes README:348 (F-3) correct. | [BUG] | factory-wiring#F-2/F-3, repro CONFIRMED | FIX (A) |
| B1 | `AdaptiveBandRMS` + `ZeroCenteredAdaptiveBandRMS`: a `None` dim on a normalized axis is silently skipped in num_params (`if dim is not None`), yielding wrong Dense units + a `None` in reshape target. adaptive_band_rms.py:280-283 + 351; zero_centered_adaptive_band_rms_norm.py:241-249. Should fail-loud at build. | [BUG/RISK] | rms-family#2, source CONFIRMED | FIX (B) |
| B2 | `AdaptiveBandRMS._reshape_scaling_factors(self, scaling_factors, input_shape)` — `input_shape` is dead (only `self._param_shape` used); sibling omits it. adaptive_band_rms.py:335-339 + call 403. | [cleanup] | rms-family#1, source CONFIRMED | FIX (B) |
| B3 | Axis-type validation absent in `BandRMS` + `AdaptiveBandRMS` (`_validate_inputs`); the other 4 RMS classes have it. band_rms.py:181-196, adaptive_band_rms.py:173-188. Contract asymmetry. | [RISK] | rms-family#5, source CONFIRMED | FIX (B) |
| C1 | logit_norm.py:6 module docstring still says "**learned** temperature" (pass-1 fixed class docstring/comment, missed module line). | [DOC] | logit-family#2 | FIX (C) |
| C2 | `DecoupledMaxLogit` `constant` called "learnable weighting"/"Weight" (docstrings 182-183, 228, 249, 275) but is a plain float (266); inline comment 321 says "fixed hyperparameter". | [DOC] | logit-family#3 | FIX (C) |
| C3 | band_logit_norm.py:11 module docstring step 3 `tanh(norm_scaled)` omits the `4×` (code:217 `tanh(4*x)`; class docstring+diagram already correct). | [DOC] | logit-family#4 | FIX (C) |
| C4 | global_response_norm.py docstring step1 says `N_c=||X_c||_2` (exact) but code (247) computes `sqrt(sum(x²)+eps)`. | [DOC] | other-layers#3 | FIX (C) |
| C5 | README param lists: dynamic_tanh (F-4) omits 5 params; global_response_norm (F-5) omits gamma/beta/activity_regularizer. README.md:267, 222. | [DOC] | factory-wiring#F-4/F-5 | FIX (C) |
| C6 | Factory keys `dml_plus_center` + `decoupled_max_logit` return `(score, norm)` TUPLES, not a single tensor — undocumented in factory/README. | [DOC] | factory-wiring#bonus | FIX (C) |

### Deferred / WON'T-FIX (judgment — behavior-change or by-design, matching first-pass D-002/D-003/D-004 precedent)

| # | Finding | Why NOT changing |
|---|---------|------------------|
| D1 | GRN `gamma_initializer='ones'` diverges from ConvNeXt V2 paper (gamma=0 identity-at-init). global_response_norm.py:135. | 3 production model families (convnext_v2_block, bfconvunext, convunext) use the DEFAULT. Changing it alters their training dynamics. Per first-pass D-002 precedent: preserve production behavior, DOCUMENT the divergence + how to get paper init. → DOC only (no default change). |
| D2 | `PolarWeightNorm.build()` calls `ops.convert_to_numpy(...)` (314,324); bare `@tf.function` on an UNBUILT layer fails. | `build()` is eager BY DESIGN in Keras (creates weights); the numpy seeding is legitimate one-time work; the `call`/forward path is fully graph-safe (verified). All standard workflows (Functional/Sequential/`model.fit`) pre-build before trace. Per LESSONS "don't add eager guards when host materialization is necessary". → DOC note only (must build before tracing). |
| D3 | Epsilon strategy floor(`maximum`) vs additive(`+eps`) across logit/maxlogit families; regularizer-default L2(1e-5) present in band classes but not adaptive classes. | Unifying either = a numerical/training behavior change to production-used layers. → acknowledge; no code change (out of "fix-don't-break" scope). |
| D4 | `_scaling_axes` computed+stored but unused in both adaptive classes (dead scaffolding). | Build-time attr, not serialized; removal is optional cleanup folded into B1/B2 only if low-risk; else leave. |

### Key Constraints

- **[HARD] No eager / graph-compatible.** VERIFIED: all `call()` paths across 16 types are graph-safe (no `.numpy()`/`int(tensor)`/dynamic-dim loops). The only eager surfaced (D2 PolarWeightNorm `build()`) is build-time-by-design, not a call-path break.
- **[HARD] Don't break production behavior.** GRN gamma default (D1), epsilon strategy + regularizer defaults (D3) are behavior-changing → DOC/WON'T-FIX, NOT code changes. Mirrors first pass's D-002.
- **[HARD] Never re-flag first-pass WON'T-FIX** (BandLogitNorm degenerate math, factory epsilon=1e-6 divergence, PolarWeightNorm-not-factory-registered) — none re-flagged.
- **[SOFT] Same contract across siblings**: axis validation (B3), None-dim fail-loud (B1), `_reshape` signature (B2) bring divergent RMS classes into line.
- **Baseline: 424 norms tests pass.** Fixes must keep all green + add targeted regressions.

### Corrections

- **[CORRECTED iter-0]** other-layers#1 flagged GRN `gamma_initializer='ones'` as a **[BUG]**. Downgraded to **[DOC]/WON'T-CHANGE-DEFAULT**: it IS a divergence from the cited ConvNeXt V2 paper, but 3 production model families rely on the current default; changing it is a training-behavior change out of "fix-don't-break" scope. Documenting the divergence instead (first-pass D-002 precedent).
- **[CORRECTED iter-0]** other-layers#2 flagged PolarWeightNorm `build()` `convert_to_numpy` as **[EAGER]** hard-constraint break. Downgraded to **DOC/by-design**: `build()` runs eagerly in all standard Keras workflows; the seeding is necessary one-time host work; the forward path is graph-safe. Only an unsupported usage (bare `@tf.function` on an unbuilt layer) fails. Per LESSONS, not an over-flag to fix.

## plan_2026-06-15_c8f516c3
### Index

| # | Finding | Severity | Source | Decision |
|---|---------|----------|--------|----------|
| F1 | All 5 MoE layer `build()` methods lack `if self.built: return` first-line guard. Explicit double-build raises ValueError (CONFIRMED by repro). Canonical Keras-3 pattern (LESSONS). LinearGating:212, CosineGating:433, SoftMoEGating:643, FFNExpert:190, MixtureOfExperts:139. | [BUG-hygiene] | gating.md#2, experts-layer#2/3, repro | FIX (Tier A) |
| F2 | `SoftMoEGating` `raw_gate_probs` has wrong shape `[b,s,e,slots]` (softmax axis=-1 over slots) vs contract `[b,s,e]` that the other 2 gatings emit. gating.py:721. Latent (layer.py:252 suppresses aux-loss for softmoe) but violates the shared aux_info contract. | [BUG-latent] | gating.md#5 | FIX (Tier A) |
| F3 | README stale/wrong: line 632 `GatingConfig(train_capacity_factor=1.0)` raises TypeError if copied; lines 178-179 show removed `train_/eval_capacity_factor` as live `MoEConfig` fields; `capacity_factor` documented as functional (143/344/534/654/718) but is a never-consumed no-op stub; norm fields undocumented. | [DOC-BUG] | config-and-wiring#11/12/13, gating.md#6 | FIX (Tier A) |
| F4 | `layer.py:286` `num_tokens = ops.shape(inputs_flat)[0]` is a dead assignment (only referenced in comments). | [cleanup] | (orchestrator verify) | FIX (Tier A) |
| F5 | `GatingConfig` has NO `__post_init__` validation; `ExpertConfig` HAS one (config.py:79). Contract asymmetry. Invalid `gating_type`/`top_k<1`/`num_slots<1` pass silently until layer construction. | [RISK/contract] | config-and-wiring#2 | FIX (Tier B) |
| F6 | `CosineGating` learnable temperature unconstrained (gating.py:443); division `cosine_sim / temperature_value` (489). temperature→0 ⇒ NaN; negative ⇒ inverted routing. | [RISK-robustness] | gating.md#4 | FIX (Tier B) |
| F7 | `ExpertConfig.use_bias/kernel_initializer/bias_initializer/kernel_regularizer/bias_regularizer` (config.py:65-69) are serialized+round-tripped but never forwarded to FFNExpert. Docstring scopes them to "additional layers (not part of the FFN itself)" — which do not exist ⇒ inert BY DESIGN. | [DEFERRED] | experts-layer#10, config:50-69 | DOC-note only; wiring=expand (REJECT) |
| F8 | `register_keras_serializable()` has no `package=` on any MoE class (gating ×3, FFNExpert, MixtureOfExperts). | [REJECT] | gating.md#3 | DO NOT FIX — adding `package=` CHANGES the registration key and breaks already-saved `.keras` models (LESSONS). Repo convention is bare per experts.py:90. |
| F9 | `capacity_factor`/`drop_tokens`/`use_residual_connection` are reserved no-ops, already commented in code (layer.py:329-333). Only the README over-advertises capacity_factor (folded into F3). | [DEFERRED] | all 3 findings | Code OK; doc via F3 |
| F10 | `integration.py:168 _apply_moe_learning_rate_multipliers` warning branch fires for every standard optimizer; sets duck-typed `optimizer.learning_rate_multipliers` attr. Pre-existing training-glue, not layer contract. | [RISK-out-of-scope] | experts-layer#9 | DEFER — training integration, not the MoE layer contract. Note only. |

### Key Constraints

- **[HARD] No EAGER, all graph-compatible.** VERIFIED already satisfied: all 3 gating types + full MoE layer trace cleanly under `@tf.function` with `TensorSpec([None,5,16])`, including `top_k==num_experts`. The dense "run-all-experts" combine (layer.py:315-327) and static `for expert_id in range(self.num_experts)` are graph-safe. No `.numpy()`/`int(tensor)`/dynamic-dim python loops exist.
- **[HARD] Keras-3 serialization round-trips.** VERIFIED: save/load roundtrip PASS (max|Δ|=0) for linear/cosine/softmoe. The missing `if self.built: return` guard does NOT break the Keras `__call__`-driven build path (framework guards it) — only explicit re-`build()` raises. Fix is hygiene/contract, not a roundtrip bug.
- **[HARD] Never change/add a `package=` on existing registered classes** — breaks deserialization of saved models (LESSONS). ⇒ F8 REJECTED.
- **[HARD] Don't expand functionality / fix what's there.** ⇒ wiring ExpertConfig dead fields (F7) and implementing capacity_factor (F9) are OUT. Removing reserved fields breaks serialization ⇒ also OUT.
- **[SOFT] Same contract across pieces.** ExpertConfig validates in `__post_init__`; GatingConfig should too (F5). All gatings emit aux_info `raw_gate_probs` shaped `[...,num_experts]` (F2 makes softmoe comply).
- **[GHOST] `train_capacity_factor`/`eval_capacity_factor`** — removed from code, silently dropped in `MoEConfig.from_dict`, still shown live in README (F3).
- **Baseline: 57 MoE tests pass** (`tests/test_layers/test_moe/`). Fixes must keep all green + add targeted regressions.

### Corrections

- **[CORRECTED iter-0]** `gating.md#1` (and Risks) claimed an **EAGER BREAK / functional-trace crash** in the `top_k==num_experts` "use all experts" branch via `ops.broadcast_to(..., (ops.shape(gate_logits)[0], N))` at gating.py:280-283 / 512-515. **REFUTED by direct repro**: `LinearGating(num_experts=4,top_k=4)` and `CosineGating(...,top_k=4)` build cleanly via `keras.Input`, AND the full MoE traces under `@tf.function`/`TensorSpec([None,...])`. On the TF backend `keras.ops.shape(x)[0]` is a dynamic scalar tensor (not Python `None`), so `broadcast_to` is fine. Matches the recurring "explorer over-flags graph-safety" pattern in LESSONS. NO FIX NEEDED.
- **[CORRECTED iter-0]** `experts-layer#5` flagged `if training and ...` (layer.py:218,252) as `[EAGER]`/`[RISK]`. This is the standard Keras `training`-bool guard; Keras passes a Python bool in `model.fit` (LESSONS). Graph-trace repro passes. NOT a defect. NO FIX.

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
