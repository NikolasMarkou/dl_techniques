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

## plan_2026-06-14_7734bacd
### Index

| # | Finding | Severity | Evidence | Detail file |
|---|---------|----------|----------|-------------|
| A1 | **Capsule `range(graph_tensor)` crash.** `_horizontal_routing` does `seq_len = ops.shape(attention_weights)[2]` then `for l in range(seq_len)`. `use_positional_routing=True` is the DEFAULT. Works eagerly (passing test is eager-only); crashes under `tf.function`/jit (`range()` on a symbolic tensor). | HIGH | capsule_routing_attention.py:721-727; default :201 | lifecycle-group-b.md |
| A2 | **Capsule Dense projections created in `build()` not `__init__`** (`query_dense`/`key_dense`/`value_dense`/`output_dense` = None in init, instantiated in build) + no `if self.built: return` guard → double-build replaces sublayers, `.keras` round-trip weight-corruption risk. | MEDIUM | capsule_routing_attention.py:303-381 | lifecycle-group-b.md |
| W1 | **`__init__.py:46` stale comment** says gated/performer/ring/rpc are "not yet in factory registry" — all four ARE registered (factory.py:768-901). Doc-accuracy defect. | MEDIUM | __init__.py:46; factory.py:768-901 | wiring-docs.md |
| W2 | **GUIDE.md:213-214 wrong dict keys** in the "add a new attention type" example: uses `'required'`/`'optional'` instead of `'required_params'`/`'optional_params'`. A developer following it silently bypasses all required-param validation. | MEDIUM | GUIDE.md:213-214 | wiring-docs.md |
| W3 | `TripletAttentionBranch` exported from `__init__.__all__` though it is a tripse-internal helper (should be `_`-private or dropped from `__all__`). | LOW | __init__.py; tripse_attention.py:45 | wiring-docs.md |
| W4 | `validate_attention_config`: no `dim == num_heads*head_dim` consistency check for `differential`; `key_dim` (hopfield-required) still unchecked; `window`/`window_zigzag` registry entries store factory FUNCTIONS not classes (works at runtime, breaks any `isinstance(info['class'], type)` guard). | LOW | factory.py:790-901 | wiring-docs.md |
| T1 | **4 zero-coverage layers** (no forward/config/save-load test anywhere): `RingAttention`, `PerceiverAttention`, `PerformerAttention`, `ProgressiveFocusedAttention`. Untested layers in this repo have historically masked dead-on-forward bug chains (LESSONS). | HIGH | tests/test_layers/test_attention/ (absent) | test-coverage.md |
| T2 | **2 partial-coverage layers**: `SingleWindowAttention` (only `isinstance`-checked in test_window_attention.py:71), `SpatialAttention` (only as CBAM sublayer in test_convolutional_block_attention.py:93). No standalone forward/serialize/round-trip. | MEDIUM | test-coverage.md | test-coverage.md |
| T3 | **Weak serialization gates**: test_wave_field_attention.py (63 tests, 3 `.keras` hits), test_shared_weights_cross_attention.py (22 tests, 2 `.keras` hits) — config-heavy, thin save/load gate. Collection clean (826 tests, 0 errors). | LOW | test-coverage.md | test-coverage.md |
| C1 | **7 non-standard `call()` signatures** vs the contract `call(inputs, attention_mask=None, training=None)`: `rpc`(`mask=`), `shared_weights_cross`(required positional `split_sizes`), `anchor`/`performer`/`lighthouse`(no mask), `group_query`/`ring`(`training` before `attention_mask`). Factory is construction-only so none block registration. Mostly architectural or risky-to-rename. | MEDIUM (document) | lifecycle-group-a.md | lifecycle-group-a.md |
| C2 | **Lighthouse requires static seq-len**: `call()` raises `RuntimeError` when `_N_static is None` (dynamic/functional-API None seq-len). Fail-loud, intentional, but NOT documented in the docstring; combined with no-mask sig it is a limited-contract layer. | MEDIUM (document) | lighthouse_attention.py:717-726 | lifecycle-group-a.md |
| C3 | **SpatialAttention silently ignores `attention_mask`** — param accepted, never used (vision layer). Misleading API: remove param or document. | LOW (document) | spatial_attention.py:197-229 | lifecycle-group-b.md |
| L1 | **`build()` idempotency guard (`if self.built: return`) absent from ~25 of the layer/helper `build()` methods.** Only `NonLocalAttention`, `ProgressiveFocusedAttention`, `_SEWeights` have it. Pure-additive, behavior-preserving conformance gap; only bites layers that explicitly child-`.build()` AND get rebuilt (functional reuse / `from_config` on built parent). `.keras` round-trip is the regression gate. | LOW (mechanical) | lifecycle-group-a/b.md | both |
| L2 | **F20 (known open): TripSE1-4 `build()` lack idempotency guard.** Single-build flow green (58 tests). Subset of L1. | LOW | tripse_attention.py:350,487,676,1012 | lifecycle-group-b.md |
| N1 | **`ops.sqrt` used inside `call()`** in gated/hopfield/lighthouse/group_query to compute `1/sqrt(head_dim)` fresh per forward. Explorers flagged HARD; **DOWNGRADED**: D-002 is an `__init__`-time graph-leak rule — inside `call()` this is a harmless scalar recompute, NOT a correctness bug. Optional micro-efficiency cleanup (precompute `math.sqrt` float in `__init__`). | INFO (not a bug) | group_query:454, gated:510, hopfield:428, lighthouse:638,749 | lifecycle-group-a.md |
| V1 | **Prior fixes VERIFIED present** (not regressed): non_local F18 (`query_conv` uses `key_value_channels`, :291) + F19 (kernel_size norm, :204); hopfield bounded loop (:556) + output_dense None-sentinel (:354); cross-attn `math.sqrt` (:237); performer/rpc regularizers via `regularizers.get()`. Atlas claims hold. | INFO | multiple | all |

### Key Constraints

- **[HARD] Behavior-preserving**: any conformance edit (idempotency guards, sublayer-in-init moves, scale precompute) MUST be byte-identical on forward. The `.keras` round-trip + existing 826-test suite is the regression gate. Any numeric delta = bug, revert.
- **[HARD] Factory is construction-only**: ideogram4/mmdit correctly excluded (non-standard sigs). Do NOT factory-register them. Non-standard `call()` sigs (C1) do not block registration.
- **[HARD] Untested-layer fixes need the test FIRST**: for A1 (capsule graph-safety) and any bug surfaced in T1, the new/expanded test is the regression gate — write it before/with the fix.
- **[SOFT] GUIDE naming standard** (`dim`/`num_heads`/`head_dim`): legacy param names (`channels`, `attention_channels`, `key_dim`) are PRE-EXISTING; renaming breaks callers + serialized models → document, do not rename.
- **[SOFT] Call-signature standardization (C1)**: renaming `mask`→`attention_mask` or reordering args risks breaking callers/serialized configs. Prefer document over rename unless a safe additive alias exists.
- **[GHOST] "lifecycle conformance done"** (atlas claim): TRUE only for the specific layers prior plans targeted. ~25 build()s still lack the guard (L1) — but most are harmless. Do NOT treat the atlas as asserting full-package guard coverage.
- **[GHOST] `ops.sqrt`-in-`call()` as a D-002 violation** (N1): the D-002 lesson is `__init__`-scoped; does not apply to `call()`. Not a ghost to chase.

### Surfaced during EXECUTE (test-tier discoveries)

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| S1 | **Perceiver `.keras` round-trip bug** — `build()` used `isinstance(input_shape, list)` to pick single-vs-two-input; Keras serializes shape tuples to lists, so `load_model` mis-read a single query shape as "3 inputs" → `ValueError("Expected 2 inputs, got 3")`. Bit ONLY via `.keras` reload (forward/symbolic worked). | HIGH | FIXED step-3, D-003, `perceiver_attention.py:175-198`; cross-attn suite (65) green |
| S2 | **PFA shifted-window (`shift_size>0`) mask path dead-on-forward** — `self._attn_mask` is 3D but `[None,None,:,:]` makes 5D, `tile` uses a mismatched 4-tuple, final `reshape` collapses `(B*heads,wa,wa)`→`(1,1,wa,wa)` → `InvalidArgumentError` (4096 vs 256). Multi-bug chain needing >10-line Swin-mask restructure. W-MSA (`shift_size=0`) path is fine. | HIGH | DEFERRED — xfail'd gate `test_shifted_window_forward` in test_progressive_focused_attention.py; **needs dedicated plan** |
| S3 | **Performer `causal=True` dead-on-forward** — `_linear_attention` causal branch (`performer_attention.py:333-349`) builds 5-D einsum `'bhnf,bhnd->bhnfd'`, cumsums, then feeds into `'bhnf,bhnfd->bhnd'` with a mis-ranked `expand_dims(q,-2)` → `InvalidArgumentError` (rank 5 vs 4). Non-causal path is fine. | HIGH | DEFERRED — xfail'd gate in test_performer_attention.py; **needs dedicated plan** |
| S4 | **Ring gradient-flow unsupported** — forward/config/`.keras` PASS, but backprop fails: blockwise output placement uses `ops.slice_update` → `XlaDynamicUpdateSlice`, no registered eager-TF gradient. Online-softmax forward exactness confirmed (single==multi block, atol 1e-5). | MEDIUM | DEFERRED — xfail'd gradient gate in test_ring_attention.py; **needs gradient-friendly scatter rewrite (dedicated plan)** |

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-06-14_adaddf34
### Index

| # | Finding (source-verified + reproduced) | Severity | Evidence |
|---|----------------------------------------|----------|----------|
| G1 | **F18 — gaussian mode shape crash (DEFAULT mode).** `non_local_attention.py:285-289`: in `'gaussian'` mode `key_value_channels = attention_channels // 8`, but `query_conv` (`:277-281`) keeps full `attention_channels` filters. In `call()` (`:494-504`) `q` is `(B,N,attention_channels)`, `k` is `(B,N,kv)`, and `scores = matmul(q, transpose(k))` contracts q's last dim (attention_channels) with k's (kv=attention_channels//8) → **size-incompatible**. REPRODUCED: `NonLocalAttention(attention_channels=32)` forward `(2,16,16,64)` → `InvalidArgumentError: Matrix size-incompatible: In[0]: [2,256,32] ...`. `attention_mode='gaussian'` is the constructor DEFAULT, so `create_attention_layer('non_local', attention_channels=N)` + forward crashes. | CRITICAL | repro: matmul `[2,256,32]` x `[2,4,256]` |
| G2 | **F19 — `.keras` round-trip kernel_size list-wrap.** `non_local_attention.py:203`: `self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)`. On `.keras` reload the serialized tuple `(7,7)` returns as a LIST `[7,7]` (TrackedList); `isinstance([7,7], tuple)` is False → else-branch wraps it to `([7,7],[7,7])`, then `DepthwiseConv2D(kernel_size=([7,7],[7,7]))` raises. REPRODUCED: Functional `.keras` save/load → `TypeError ... could not be deserialized`; isolate `NonLocalAttention(kernel_size=[7,7])` → `ValueError: kernel_size argument must be a tuple of 2 integers. Received kernel_size=([7, 7], [7, 7]) ... type TrackedList`. | HIGH | repro: ValueError on list kernel_size |

### Root causes (both confirmed by reproduction)

- **G1 fix direction**: the original Non-local embedded-Gaussian reduces Q, K, AND V to the same embedded channel dim (theta/phi/g all project to C_hat). The layer wrongly reduces only K/V. Make `query_conv` use `self.key_value_channels` too (and the `call()` q-reshape + dot_product `d_k` scaling reference `key_value_channels`). In `dot_product` mode `key_value_channels == attention_channels`, so the change is **byte-identical** there; in `gaussian` mode it makes Q@Kᵀ well-formed (all C//8). Docstring currently says "key/value channels are reduced" — update to "query, key, and value channels are reduced".
- **G2 fix direction**: normalize `kernel_size` for int/tuple/list/TrackedList uniformly: `self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)`. Byte-identical for int/tuple inputs; fixes the list/reload path.

### Key Constraints

- **HARD — behavior-preserving for dot_product + int/tuple inputs**: the G1 query-channel change must be byte-identical in `dot_product` mode (key_value_channels==attention_channels); the G2 normalization must be byte-identical for int and tuple `kernel_size` args. Any numeric delta in the unaffected paths = bug.
- **HARD — no test currently exists** for non_local (parent plan finding). The regression gate is a NEW test file + inline smoke. This plan SHOULD add `tests/test_layers/test_attention/test_non_local_attention.py` covering: gaussian forward, dot_product forward, `.keras` Functional round-trip, int + tuple + list kernel_size, get_config/from_config.
- **SOFT — gaussian channel reduction factor (//8)**: keep the existing `//8` (do not change to //2); only fix the query side to match. Caveat: `attention_channels < 8` makes `key_value_channels == 0` (degenerate) — consider a guard (`max(1, ...)`) or document; decide in PLAN.
- **GHOST — "reduce only K/V" intent**: the docstring's claim that only K/V are reduced is mathematically impossible for Q@Kᵀ; it is a bug, not a constraint. Reducing Q to match is the correct Non-local behavior.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*
