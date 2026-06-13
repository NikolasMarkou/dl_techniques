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

## plan_2026-06-13_5b933e7f
### Index

| # | Topic | File | Summary |
|---|-------|------|---------|
| F1 | Optimization package structure, conventions, __init__.py | `findings/optimization-package.md` | Package exports only 3 builder functions; Muon/SGLD not exported. OptimizerType enum + dispatch in optimizer.py. Constants in constants.py. 5 test classes per optimizer. Muon = stateful template (add_variable_from_reference). SGLD = stateless template. |
| F2 | Keras 3 optimizer API + test patterns | `findings/test-and-keras-optimizer-patterns.md` | Must implement build(var_list), update_step(gradient, variable, learning_rate), get_config(), from_config(cls, config). @register_keras_serializable required. Per-variable state via add_variable_from_reference. self.iterations is global step counter (already incremented before update_step). |
| F3 | VSGD algorithm mapping (from provided PyTorch source) | (this file) | VSGD has per-variable state (mug, bg, bhg) + scalar constants (pa2, pbg2, pbhg2) derivable from hyperparams. step counter = self.iterations. Step-1 branch needed (ops.where). Weight decay is AdamW-style. |

### Key Constraints

- **[HARD] @keras.saving.register_keras_serializable()** required on the class for save/load round-trip
- **[HARD] build(var_list)** must guard with `if self.built: return`, call `super().build(var_list)`, allocate per-variable mug/bg/bhg via `add_variable_from_reference`
- **[HARD] update_step(gradient, variable, learning_rate)** — `learning_rate` is already evaluated scalar; use `keras.ops` only (no raw tf.*)
- **[HARD] Step-1 branching** must use `ops.where` (not Python if) for graph-safety
- **[HARD] No print statements** — use `dl_techniques.utils.logger`
- **[HARD] get_config() must call super().get_config()** first; base class handles learning_rate schedule serialization
- **[SOFT] weight_decay**: pass `weight_decay=0.0` to super() and manage manually (Muon pattern) OR pass through (SGLD pattern) — VSGD manages it manually in update_step (AdamW-style)
- **[SOFT] Add VSGD to OptimizerType enum** and optimizer_builder dispatch in optimizer.py
- **[SOFT] Export VSGD from __init__.py** (fix the existing Muon/SGLD omission inconsistency)
- **[SOFT] Constants in constants.py** following SGLD defaults block pattern

### F3: VSGD Algorithm State Mapping

### Per-variable state (same shape as parameter — use add_variable_from_reference)
- `mug`: running mean estimate of gradient g_t, initialized to zeros
- `bg`: running variance estimate of g_t, initialized to zeros
- `bhg`: running variance estimate of ghat_t, initialized to zeros

### Scalar constants (derived from hyperparams — not stored as state)
- `pa2 = 2*ps + 1.0 + 1e-4` (prior shape param)
- `pbg2 = 2*ps` (prior scale for g variance)
- `pbhg2 = 2*ghattg*ps` (prior scale for ghat variance)

### Global step counter
- `step = self.iterations` (Keras base class, incremented BEFORE update_step is called)
- On first call: self.iterations == 1

### Update rule (Keras 3 translation)
```
step = cast(self.iterations, dtype)
is_first = (step <= 1.0)
sg = where(is_first, pbg2/(pa2-1), bg/pa2)
shg = where(is_first, pbhg2/(pa2-1), bhg/pa2)
mug_prev = copy(mug)
mug_new = (ghat*sg + mug_prev*shg) / (sg + shg)
sigg = sg * shg / (sg + shg)
mug_sq = sigg + mug_new**2
bg2 = pbg2 + mug_sq - 2*mug_new*mug_prev + mug_prev**2
bhg2 = pbhg2 + mug_sq - 2*ghat*mug_new + ghat**2
rho1 = step**(-tau1); rho2 = step**(-tau2)
bg_new = bg*(1-rho1) + bg2*rho1
bhg_new = bhg*(1-rho2) + bhg2*rho2
variable *= (1 - lr*weight_decay)   # AdamW weight decay
mug.assign(mug_new); bg.assign(bg_new); bhg.assign(bhg_new)
variable -= lr / (sqrt(mug_sq) + eps) * mug_new
```

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-06-13_250487cb
### Index

| # | Topic | File | Summary |
|---|-------|------|---------|
| F1 | Transformer layer audit (all 15 files) | `findings/transformer-layer-files.md` | All concrete classes have @register, build, compute_output_shape, get_config. Primary violations: `BinaryMapper` uses raw `tf.*`; `FreeTransformerLayer` + `PFTBlock` create sublayers in `build()` (not `__init__`); `EomtTransformer` mutates Python int in `call()`; `TextDecoder.get_config()` omits 6 params. |
| F2 | Keras guide requirements | `findings/keras-guide-requirements.md` | HARD: sublayers in `__init__`, `add_weight` only in `build()`, no raw `tf.*` in call, all `__init__` params in `get_config()`. SOFT: consistent package= in register. |
| F3 | Factory wiring + `__init__.py` exports | `findings/factory-wiring.md` | 10 classes exported; `sd3_adaln`, `ideogram4_block` intentionally direct-import. `free_transformer.py` and `progressive_focused_transformer.py` not exported (unclear if intentional). No `create_text_decoder` factory. |

### Key Constraints

- **[HARD] `BinaryMapper` uses raw `tf.*`** -- `tf.constant` in `__init__`, `tf.nn.sigmoid_cross_entropy_with_logits` in `call()` -- must be replaced with `keras.ops` equivalents or removed
- **[HARD] `FreeTransformerLayer` creates sublayers in `build()`** -- 9 sublayers (`encoder_attention`, `encoder_ffn`, etc.) deferred to `build()` -- must move to `__init__`
- **[HARD] `PFTBlock` creates sublayers in `build()`** -- `_norm1`, `_norm2`, `_attn`, `_ffn`, `_drop_path` deferred to `build()` -- must move to `__init__`
- **[HARD] `EomtTransformer.current_step`** -- plain Python int mutated in `call()`, resets on reload, not serialized -- must become a `keras.Variable` or be removed if unused for correctness
- **[HARD] `TextDecoder.get_config()` omits 6 params** -- `embedding_type`, `positional_type`, `attention_type`, `normalization_type`, `normalization_position`, `ffn_type` missing -- deserialization reverts to defaults silently
- **[SOFT] `sd3_adaln.py` and `ideogram4_block.py` are intentionally direct-import** -- SYSTEM.md confirms this; do NOT add to `__init__.py`
- **[SOFT] `free_transformer.py` and `progressive_focused_transformer.py` are absent from `__init__.py`** -- status unclear; export only after fixing violations
- **[SOFT] `TextEncoder.cls_token`** -- `add_weight` in `build()` is acceptable but inconsistent with `VisionEncoder` (which does it in `__init__`); low priority

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

- **[NOTE]** `sd3_adaln.py` and `ideogram4_block.py` missing from `__init__.py` is INTENTIONAL per SYSTEM.md ("Direct-import only") -- factory-wiring finding F3 initially flagged as a gap but is correct behavior.

## plan_2026-06-13_28f0b453
### Index

| # | Topic | File | Summary |
|---|-------|------|---------|
| F1 | DETR model code -- all bugs | `findings/detr-model-code.md` | 10 confirmed bugs: wrong attention_type key (crash), bad build signatures (crash), NCHW pos_embed not transposed (silent shape corruption), DetrDecoderLayer reimplements TransformerDecoderLayer from scratch, query_embed weight access fragile, mask shape mismatch, compute_output_shape fails on flat input. |
| F2 | Factory/transformer compatibility | `findings/detr-factory-compatibility.md` | `'multi_head'` is the valid key (not `'multi_head_attention'`). `TransformerDecoderLayer(use_causal_mask=False)` is a drop-in for `DetrDecoderLayer`. Full-refactor path eliminates ~200 lines. |
| F3 | Test patterns + DETR test plan | `findings/detr-test-patterns.md` | No DETR tests exist. Must use a minimal stub backbone (avoid ResNet50 download). 5 test classes needed; canonical pattern from `test_thera_model.py` / `test_vit/`. |

### Key Constraints

- **[HARD] `attention_type='multi_head'`** -- the correct registry key; `'multi_head_attention'` raises `ValueError` at construction. `DetrTransformer.__init__` currently uses the wrong string at `model.py:158`.
- **[HARD] `PositionEmbeddingSine2D` returns NCHW** `(B, C, H, W)` -- must transpose to `(B, H, W, C)` before reshape to `(B, H*W, C)`. Current code silently corrupts positional encodings (`model.py:594`).
- **[HARD] `TransformerDecoderLayer.build(input_shape)`** expects a single 3-tuple `(B, T, H)` for the decoder input, NOT a 2-tuple `(tgt_shape, memory_shape)` as `DetrDecoderLayer.build` currently does. `DetrTransformer.build` at `model.py:191` must be updated accordingly.
- **[HARD] `TransformerDecoderLayer.call(inputs, encoder_output, ...)`** -- `encoder_output` is a required positional arg (not a kwarg `memory=`).
- **[HARD] `use_causal_mask=False`** on `TransformerDecoderLayer` -- DETR queries are not autoregressive.
- **[HARD] No test suite exists** -- tests must be written; tests must use a minimal stub backbone, NOT `create_detr` which downloads ResNet50 weights.
- **[SOFT] `DetrDecoderLayer` should be deleted** and replaced with `TransformerDecoderLayer` -- repo convention is reuse over bespoke reimplementation.
- **[SOFT] `self.query_embed.embeddings`** is safer than `self.query_embed.weights[0]` for accessing the Embedding weight.
- **[GHOST] `'multi_head_attention'` was never a valid factory key** -- this string was never registered.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*

## plan_2026-06-13_e7b5704d
### Index
| # | Topic | File | Summary |
|---|-------|------|---------|
| F1 | Repo-wide AST scan | (this file) | 572 keras Layer/Model subclasses. Raw hits: 54 layers w/o compute_output_shape, 11 w/o register, 10 add_weight-in-init, 1 w/o get_config. |
| F2 | compute_output_shape cluster A (layers core/ts/attn) | `findings/cos-cluster-a.md` | 13 ADD, rest SKIP (abstract bases, RNN cells, inherited). |
| F3 | compute_output_shape cluster B (heads/experimental) | `findings/cos-cluster-b.md` | 6 ADD (dict-output), SKIP abstract + dynamic-dict multitask. |
| F4 | compute_output_shape cluster C (models) | `findings/cos-cluster-c.md` | 18 ADD, 5 SKIP (multi-input/non-tensor-arg/3-output cell). |
| F5 | register / get_config / add_weight | `findings/register-and-addweight.md` | 6 ADD register (5 tabm pkg=TabM incl MLPBlock collision, 1 experimental); Base* abstract SKIP; ALL 10 add_weight-in-init are ACCEPTABLE-LEAVE (config-shape, not input-shape). |

### Key Constraints
- **[HARD] `MLPBlock` name collision**: `tabm_blocks.py:MLPBlock` vs already-registered `ffn/mlp.py:MLPBlock`. Must register tabm classes with `package="TabM"` or import raises.
- **[HARD] Abstract base classes are NOT violations**: `BaseMemory/Head/Controller/NTM` (ntm_interface), `BaseExpert` (moe), `BaseGating` (moe), `BaseVisionHead/BaseVLMHead`, `ComplexLayer` are `ABC`/@abstractmethod — do NOT register or add compute_output_shape.
- **[HARD] RNN cells exempt from compute_output_shape**: `DeepARCell`, `mLSTMCell`, `sLSTMCell`, `NAMCell`, `TRMInner` use the `state_size`/`output_size` cell contract. SKIP.
- **[HARD] All 10 `add_weight`-in-`__init__` are config-shaped (learnable tokens/temperatures/pos-embeds), NOT input-shaped → working pattern, NOT the guide's anti-pattern. LEAVE (documented).
- **[SOFT] Dynamic-dict-output heads** (`MultiTaskHead`, `MultiTaskVLMHead`) build their return dict from runtime task names → no static compute_output_shape. SKIP.
- **[HARD] A wrong compute_output_shape is worse than none**: each added method must be verified against a real forward pass before commit.

### Corrections / Discoveries
- **[DISCOVERY iter-1]** `mst_correlation_filter.py` was UNIMPORTABLE pre-change (`keras.KerasTensorShape` nonexistent attr in annotations). Fixed (D-002). SystemicGraphFilter was effectively dead code.
- **[DISCOVERY iter-1] DETR is pre-existing broken/untested**: `DetrTransformer` encoder constructs `TransformerLayer(attention_type='multi_head_attention')` — not a valid factory key ('multi_head' is) → cannot construct. `DetrDecoderLayer.build` unpacks a 2-shape tuple, failing on isolated Keras auto-build. No `tests/test_models/test_detr/` exists. Out of scope (conformance only); compute_output_shape added mirrors detr's own build() unpacking. Flag for a future DETR-fix plan.
- **[DISCOVERY iter-1] Pre-existing flaky test**: `test_cliffordnet_lmunet.py::TestMRLSerialization::test_save_load_keras_with_mrl` fails at ~2e-5 save/load mismatch — FAILS IDENTICALLY AT BASELINE (my edits stashed). Not a regression; pre-existing numeric/XLA non-determinism.
