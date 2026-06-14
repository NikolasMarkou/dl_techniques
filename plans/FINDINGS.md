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

## plan_2026-06-14_33b77a7a
### Goal
Complete the deferred `ops.sqrt`-in-`call()` consistency sweep (D-003 of plan_2026-06-14_077a2a35).
Precompute STATIC attention scales as Python floats (D-002 pattern: `1.0/math.sqrt(float(static_int))`),
reuse existing precomputed attrs where present. Behavior-preserving. Leave genuinely-dynamic sqrt.

### Full sqrt-site census (grep of attention/*.py)

### STATIC — precompute / reuse (IN SCOPE)
| Site | Current | Static source | Fix | Where to precompute |
|------|---------|---------------|-----|---------------------|
| `performer_attention.py:245` | `projection / ops.sqrt(ops.cast(self.head_dim, dtype))` | head_dim (init:162) | REUSE `self.scale` (init:163 = 1/np.sqrt(head_dim)) -> `projection * self.scale` | already exists |
| `performer_attention.py:289` | `features * ops.sqrt(2.0/ops.cast(self.nb_features, dtype))` | nb_features (init:149) | new `self._feature_scale = math.sqrt(2.0/float(nb_features))` -> `features * self._feature_scale` | `__init__` |
| `lighthouse_attention.py:651` | `ops.cast(1.0/ops.sqrt(ops.cast(self.head_dim,dt)),dt)` | head_dim | REUSE `self._scale` (init:340 = 1/float(head_dim)**0.5) -> `ops.cast(self._scale, q_t.dtype)` | already exists |
| `lighthouse_attention.py:763` | `ops.cast(1.0/ops.sqrt(ops.cast(D,dt)),dt)` D=head_dim (742) | head_dim | REUSE `self._scale` | already exists |
| `capsule_routing_attention.py:519` | `attention_logits / ops.sqrt(ops.cast(self.actual_key_dim,dt))` | actual_key_dim resolved in build (341) | new `self._inv_sqrt_key_dim = 1.0/math.sqrt(float(actual_key_dim))` -> `attention_logits * self._inv_sqrt_key_dim` | **`build()`** (after line 342) |
| `non_local_attention.py:514` | `scores / ops.sqrt(d_k)` d_k=cast(key_value_channels) | key_value_channels (init:284) | new `self._inv_sqrt_kv = 1.0/math.sqrt(float(key_value_channels))` -> `scores * self._inv_sqrt_kv` (dot_product mode only; gaussian unaffected) | `__init__` (after 284) |

### DYNAMIC — genuine runtime tensors (OUT OF SCOPE, leave as-is)
| Site | Why dynamic |
|------|-------------|
| `window_attention.py:489` | `sqrt(cast(N_actual))` N_actual = `ops.shape(inputs)[1]` runtime seq len |
| `wave_field_attention.py:556` | `sqrt(sum(square(k)))` — per-vector L2 norm over a tensor |
| `capsule_routing_attention.py:599` | `sqrt(squared_norm + eps)` — squash norm over a tensor |

### ALREADY-FIXED precedent (do not touch)
gated:249, gqa:202, mla:242, ring:162, shared_weights_cross:169, diff:246, mhxa:237, hopfield:242 (AF2),
wave_field:249, ideogram4:126, mmdit:158, performer:163(self.scale), lighthouse:340(_scale), anchor:222, rpc:191.

### Index
| ID | Topic | File | Summary |
|----|-------|------|---------|
| CENSUS | sqrt-site classification | findings.md (this file) | 6 static call()-time sites to fix across 5 files; 3 dynamic left; reuse self.scale/_scale at 4 sites, 2 new attrs. |
| TIMING | precompute location | findings.md | non_local+performer in __init__; capsule in build() (actual_key_dim needs input_shape). |
| MP | mixed-precision test coverage | findings.md | NONE of lighthouse/performer/capsule/non_local have MP tests. float32 is the live/tested path. |

### Key Constraints
- **HARD**: behavior-preserving. Precomputed Python-float scale must equal the old `ops.sqrt` result in
  float32 (the tested path). Verify per-site with `np.float32(new) == np.float32(old)`; tests are the oracle.
  Any forward test shift -> revert that site (it is consistency-only, not worth a regression).
- **HARD**: precompute timing — capsule MUST be in build() (actual_key_dim is None until build resolves
  embed_dim from input_shape). Precomputing in __init__ would crash/use None.
- **HARD**: non_local fix applies ONLY to `dot_product` mode (line 514); `gaussian` mode has no scaling — do not add any.
- **SOFT**: reuse existing precomputed attrs (self.scale / self._scale) rather than add duplicates (DRY).

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*

## plan_2026-06-14_077a2a35
### Prior-work context (CRITICAL — read before exploring)

`src/dl_techniques/layers/attention/` has been reviewed by ~6 closed plans TODAY (2026-06-14).
Current ground-truth baseline: **989 passed / 27 skipped / 0 failed** (per ab855e7e CLOSE).
HEAD = 97996298. This is a 7th pass — value is in (a) deferred items, (b) fresh adversarial residue.

Prior plans + what they fixed:
- `0c5d4a21`: F1-F16. Factory registration 23->27, build-guard sentinels (hopfield/nonlocal/PFA/tripse),
  fnet fft guard, performer/rpc regularizer round-trip, README/GUIDE param-name fixes, __init__ exports.
- `adaddf34`: NonLocal F18 (gaussian Q/K/V dim alignment) + F19 (kernel_size normalize) + new test.
- `7734bacd`: "Deep review" — capsule A1/A2, W1-W4 docs/wiring, C1-C3 contract docs, Perceiver .keras
  fix (D-003). Deferred S1(fixed)/S2/S3/S4 as xfail.
- `b9456f74`: Resolved S2 (PFA SW-MSA mask), S3 (Performer causal einsum), S4 (Ring gradient list+concat),
  L1/L2/F20 (29 build() idempotency guards across 25 files), D1-D3 docs/exports. 939 passed.
- `ab855e7e`: Fresh adversarial pass F2-F6. Precompute static scale (gated/gqa/mobile_mqa),
  ring/anchor static-seq fail-loud guards, factory optional_params completion, mobile_mqa mask doc,
  MLA training= forward. 989 passed.

### KNOWN DEFERRED / OPEN ITEMS (the user explicitly asked to check these)
- **F7 (ab855e7e)**: `hopfield_attention.py` builds K/V Dense with query_shape only; cross-attention with
  a different K/V feature dim gets a wrong-shaped kernel. No caller today; self-attention path correct.
  Inline-commented. -> VERIFY + assess fix.
- **F8 (ab855e7e)**: `wave_field_attention` and `single_window_attention` have STANDARD call sigs but are
  NOT factory-registered. -> VERIFY + assess registration.
- **NOTE 1 (b9456f74)**: Performer runs dead non-causal compute even when causal=True (wasted FLOPs,
  correctness-neutral). Optional `if not self.causal:` guard.
- **NOTE 2 (b9456f74)**: mobile_mqa add_weight after super().build() — latent smell, idempotent-safe.
- **C2 (b9456f74)**: package= on bare-registered classes changes registration KEY -> breaks already-saved
  .keras deserialization. Document-only, NOT a fix.
- **call() param renames**: D-007 (0c5d4a21) — renaming breaks serialized configs. Document-only.

### ANCHOR DEBT
~38 pre-existing foreign-plan orphan ERRORs from validate-plan.mjs at REFLECT. NOT blockers; triage by
plan-id (project_validate_plan_anchor_debt memory).

### Index
| ID | Topic | File | Summary |
|----|-------|------|---------|
| FCT | Factory optional_params gaps | findings/factory-docs-audit.md | CONFIRMED: 7 registry entries silently drop real ctor params — capsule_routing/multi_head/group_query/perceiver/shared_weights_cross miss probability_type/probability_config/qk_norm_type/qk_norm_kwargs; non_local misses output_activation_args; window/window_zigzag miss kan_* . Same class ab855e7e fixed for anchor/channel/spatial/tripse — these were MISSED. |
| F8 | wave_field+single_window unregistered | findings/deferred-items.md | CLEAN-FIX: both standard call sigs, additive registry+Literal+import. |
| F7 | hopfield cross-attn KV-dim build | findings/deferred-items.md | CLEAN-FIX: build() uses query_shape for K/V Dense; wrong for cross-attn; byte-identical for self-attn (zero callers). |
| DOC | README/__all__ doc gaps | findings/factory-docs-audit.md | README:82 wrongly says capsule_routing has no probability hooks; window class column shows class not factory fn; caveats table missing spatial/non_local; __all__ omits list_attention_types/get_attention_requirements. |
| N1 | performer dead non-causal compute | findings/deferred-items.md | CLEAN-FIX (≤8 lines): non-causal block fully recomputed+discarded when causal=True. if/else guard, behavior-identical. |
| AF1 | hopfield value_dim=None get_config | findings/adversarial-correctness.md | LOW: get_config serializes resolved int not original None. Round-trip functionally equiv. 1-line raw-arg store. |
| SQRT | ops.sqrt-in-call() consistency | findings/adversarial-correctness.md | MED/consistency, NOT bugs: hopfield:431, performer:245/289, lighthouse:651/763, capsule:519, non_local:514 use ops.sqrt in call(). Call-time = correct output; only a precompute-consistency gap. Candidate DEFER (YAGNI). |
| AF3 | mobile_mqa get_config key-delete | findings/adversarial-correctness.md | NOT-A-BUG: standard from_config re-injects; works. Skip. |
| FLAKY | capsule graph test suite-order flake | findings/test-ground-truth.md | MED: fails in full suite, passes isolated. Test-isolation/global-state, not source. Diagnose. |

### Key Constraints
- **HARD**: behavior-preserving on all live call paths (zero regressions vs 989 baseline).
- **HARD**: no registration-KEY changes (breaks saved .keras) — C2/param-renames stay document-only.
- **SOFT**: follow established precedent patterns (None-sentinel build, fail-loud dynamic-N, math.sqrt scale).

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*

## plan_2026-06-14_ab855e7e
### Index
| # | Finding | File | Tag/Sev |
|---|---------|------|---------|
| F1 | Baseline `tests/test_layers/test_attention/` = **940 passed, 0 fail** (CUDA1). Regression oracle. | findings/review-* | CONFIRMED-OK |
| F2 | `ops.sqrt(ops.cast(static_head_dim,...))` (D-002 pattern) survives in 3 files: `gated_attention.py:512-513`, `group_query_attention.py:457`, `mobile_mqa.py:252`. Prior D-002 sweep fixed only cross/diff. Correctness-neutral (leak version-conditional) but violates "same contract". | gated/gqa/mobile_mqa | NEW-BUG LOW |
| F3 | Graph-safety residue (static-shape defect class fixed in capsule/PFA, missed here): `ring_attention.py:384` `seq_len=ops.shape(q)[2]` → `range(num_blocks)`; `anchor_attention.py:434` `if num_anchor_tokens >= seq_len` (dynamic). Break under `@tf.function`/jit with None seq dim. | ring/anchor | NEW-BUG MED |
| F4 | Factory `optional_params` silently drop real class args (factory filters kwargs to required∪optional). CONFIRMED: `anchor` missing head_dim/probability_type/probability_config; `channel` missing intermediate_activation_{type,args}+gate_activation_{type,args}; `spatial` missing gate_activation_{type,args}; `tripse1-4` missing gate_activation_{type,args} (+tripse4 se_reduction_activation_{type,args}). Fails "wired up in factory". Additive zero-risk fix. | factory.py | NEW-BUG MED |
| F5 | `mobile_mqa.call` accepts+documents `attention_mask` but silently ignores it (mobile_mqa.py:255-257). Advertised-vs-actual mismatch. Mask under downsampling is non-trivial. Fix: document (spatial precedent) + README caveat. | mobile_mqa.py | NEW-BUG MED |
| F6 | `attn_prob(scores)` called without `training=` (ProbabilityOutput.call DOES accept training): MLA:534, gated:533, others. Systematic; default softmax unaffected. | multiple | NEW-SMELL LOW |
| F7 | `hopfield_attention.py:373-375` builds K/V Dense with `query_shape` only; cross-attn with different K/V feature dim → wrong kernel. No current caller (self-attn path OK). | hopfield | NEW-BUG LOW (latent) |
| F8 | Unregistered files: `ideogram4`, `mmdit_joint`, `progressive_focused` JUSTIFIED (non-standard call sig). `wave_field`, `single_window` have standard `call(inputs, attention_mask, training)` sig → registration is possible but not done (explorer: not structurally forced). | factory.py | WIRING (review) |
| F9 | All prior-plan fixes CONFIRMED holding: `if self.built: return` guards (all files); Performer causal einsum; Ring list+concatenate assembly; capsule/PFA/non_local static-shape+failloud; __all__ exports; README caveat table accurate (8 entries). | all | CONFIRMED-OK |

Detailed reports: `findings/review-core-mha.md`, `findings/review-efficient.md`, `findings/review-vision.md`, `findings/review-factory-docs.md`.

### Key Constraints
- **[HARD] Behavior-preserving for passing layers.** 940 tests pass now; any numeric delta on existing tests is a regression → revert. (LESSONS: guide-conformance is behavior-preserving.)
- **[HARD] D-002 fix pattern is `self.scale = 1.0/math.sqrt(float(head_dim))` in `__init__`** (Python float, math.* not ops.*). F2 fixes must match it.
- **[HARD] Graph-safe shape pattern = static `.shape[N]` + fail-loud `ValueError` on None** (capsule/PFA precedent). F3 fixes must match it.
- **[HARD] Adding `package=` to @register_keras_serializable is FORBIDDEN** (breaks already-saved .keras models) — documented known-open, NOT to "fix".
- **[HARD] Do NOT rename performer.call (no attention_mask) / rpc.call (`mask`)** — documented intentional quirks (D-007), behavior-touching, out of scope.
- **[SOFT] Factory registry additions (F4) are additive** — add keys with class-default values; do not change required_params or existing defaults.
- **[GHOST] "attention/ already fully resolved" — TRUE for the items prior plans targeted, but a fresh adversarial pass found F2-F7 residue. The "resolved" label was scoped to those plans' findings, not exhaustive.**

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-06-14_b9456f74
### Baseline
- `pytest tests/test_layers/test_attention/` = **935 passed / 3 xfailed / 0 failed** (284s, GPU1). Repo at clean post-`plan_2026-06-14_7734bacd` state. The 3 xfails are S2/S3/S4 below.
- This directory had 3 prior plans: `0c5d4a21` (contract/factory/docs/lifecycle), `adaddf34` (NonLocal F18/F19), `7734bacd` (deep review + tests, deferred S2/S3/S4 + L1/L2).

### Index
| ID | File | Finding | Verdict |
|----|------|---------|---------|
| S2 | findings/deferred-bugs.md | PFA SW-MSA mask dead-on-forward: `_attn_mask[None,None,:,:]` → 5-D, mismatched tile, reshape collapse → InvalidArgumentError. ALSO `_compute_attention_mask` hardcodes 2×2 grid (`mask_size=ws*2`) regardless of actual H×W → silently wrong mask for production callers (pft_sr, thera, swin) with larger maps. | FIX (operate-as-advertised); has prod callers |
| S3 | findings/deferred-bugs.md | Performer `causal=True` crash: `ops.expand_dims(v,-2)` makes einsum 2nd arg rank-5 (line 340); `expand_dims(q,-2)`+`squeeze` rank bug (347-349) → InvalidArgumentError. | FIX BOUNDED (3 lines) |
| S4 | findings/deferred-bugs.md | Ring gradient crash: `ops.slice_update` (line 485) → `XlaDynamicUpdateSlice` has no eager-TF gradient. Forward OK. | FIX BOUNDED (list+concat, ~6 lines) |
| L1 | findings/lifecycle-robustness.md | 22 build() methods + 5 tripse classes lack `if self.built: return` first-line guard while calling `child.build()`/`add_weight()`. Weight-creating ones (fnet:184, differential:335/362, mobile_mqa, single_window, wave_field) silently DUPLICATE weights on 2nd build. Repo-standard fix used 5x already. | FIX (keras-compliant / same-contract) |
| L2/F20 | findings/lifecycle-robustness.md | tripse `TripletAttentionBranch`(139), `TripSE1`(350), `TripSE2`(487), `TripSE3`(676), `TripSE4`(1012) all lack outer guard; only `_SEWeights`(820) has it. Subset of L1. | FIX with L1 |
| D1 | findings/contract-factory-docs.md | README caveat table lists 7 non-standard call() sigs but MISSES: `mobile_mqa`(175, swapped order + `return_attention_weights`), `differential`(504, extra `layer_idx=0`). | FIX docs |
| D2 | findings/contract-factory-docs.md | `Ideogram4Attention` + `MMDiTJointAttention` are registered-serializable but NOT in `__init__.py __all__` → not importable via public API. | FIX exports (additive) |
| D3 | findings/lifecycle-robustness.md | `test_shared_weights_cross_attention.py` lacks a real `.keras` file save/load round-trip test (only from_config). | ADD test (also regression oracle for guard fix) |
| C1 | findings/contract-factory-docs.md | Factory: 27 registered types CONFIRMED. 5 unregistered (ideogram4, mmdit, progressive_focused, single_window, wave_field) for valid architectural reasons. | CONFIRMED-GOOD |
| C2 | findings/contract-factory-docs.md | 32/34 serializable classes lack `package=`. Latent bare-name collision risk BUT adding package= changes the registration KEY → breaks already-saved `.keras` models. | DEFER (document; regression risk) |

### Key Constraints
- **[HARD]** RingAttention forward numerics must stay identical (online-softmax single==multi verified atol 1e-5). `concatenate` of blocks in order == slice_update assembly. (S4)
- **[HARD]** Keras 3 / `keras.ops` only — no raw TF ops. Static scalars via `math.*`.
- **[HARD]** `if self.built: return` MUST be FIRST statement of any build() that calls `child.build()` or `add_weight()` — Keras does NOT self-guard explicit child `.build()` (2nd build raises lock violation / duplicates weights). Established repo pattern (used 5x). (L1)
- **[HARD]** Do NOT rename call() params (rpc `mask`, anchor `x`, performer no-mask, etc.) — D-007 (plan 0c5d4a21): renaming breaks serialized configs. Document only. (D1)
- **[HARD]** Do NOT add `package=` to the 32 bare-registered classes — changes registration key, breaks deserialization of already-saved models. Document as known latent hazard. (C2)
- **[SOFT]** S2 mask should be correct for general H×W (standard Swin), not just the H=W=2·ws test case — "operate as advertised". Reference: `swin_transformer_block.py:446-483`.
- **[GHOST]** PFA comment line 799 "Broadcasting handles batch and head dimensions" is false — broadcasting never worked. (S2)
- **[GHOST]** Prior "gated/performer/ring/rpc not registered" concern — all 4 now registered (D-007 resolved). Don't re-investigate.

### Corrections
*None yet.*
