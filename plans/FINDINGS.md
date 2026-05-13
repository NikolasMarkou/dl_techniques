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

## plan_2026-05-12_6a2cd5b3
### Index

| ID | Topic | File | Summary |
|----|-------|------|---------|
| F-001 | bert/wikipedia/pretrain.py structure | `findings/bert-wikipedia-pretrain.md` | Class-level `PretrainConfig`; HF streaming `datasets.load_dataset("wikipedia",...,streaming=True)` interleaved with BookCorpus; `tf.py_function` tokenize -> batch -> prefetch; `MirroredStrategy` + `mixed_float16`; `WarmupSchedule(...,CosineDecay(...))` + `AdamW(weight_decay=0.01, clipnorm=1.0, jit_compile=True)`; `ModelCheckpoint(save_freq=5000) + TensorBoard + BackupAndRestore`; single mega-epoch `steps_per_epoch=total_steps=100000`. Encoder-only saved at end. NO argparse. NO val split. BookCorpus often 401s on HF. `pretrain_english.py` adds ASCII-density english filter. |
| F-002 | Data path trade-off | `findings/data-path-tradeoff.md` | Recommend `load_wikipedia_train_val` (Pattern B) over HF streaming (Pattern A): local cache `/media/arxwn/data0_4tb/datasets/wikipedia/wikimedia___wikipedia/` already populated, gives val split, reproducible, parallel tokenization via `num_shards>1`, matches established CliffordNet sibling trainers. Cost: no BookCorpus interleave (acceptable - restricted on HF). For MLM (one-article-per-example), use `min_article_length>=500` (LESSONS note: 0 is correct for *packed* CLM, NOT for MLM). |
| F-003 | GPU budget + deps + horizon | `findings/gpu-budget-and-deps.md` | Empirically measured: nano=29M MLM params, mini=51M, base=275M. RTX 4090 24GB: nano @ batch=64 seq=512 comfortable; mini @ batch=32; base @ batch=16. `datasets 4.4.1` and `tiktoken 0.12.0` present in .venv. Wikipedia local cache verified. Smoke run = nano/batch=16/seq=128/max_samples=2000/total_steps=400 (~5 min). Real run = nano/batch=32/seq=512/total_steps<=50000 (multi-hour). Must run in background, single GPU, MPLBACKEND=Agg. Use `mixed_float16` for real run. |

### Key Constraints

### HARD
- **Single GPU only** - GPU 0 = RTX 4090 24GB. Never run training jobs in parallel.
- **`CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg`** prefix every training invocation.
- **`run_in_background=True`** for real training run (multi-hour).
- **Keras 3 / TF 2.18 idioms** - `@keras.saving.register_keras_serializable()`, `keras.ops`, `dl_techniques.utils.logger`, `get_config()` round-trip.
- **`pad_token_id=100266`** (tiktoken cl100k_base) wired into encoder ctor - LESSONS canonical silent semantic bug.
- **AdamW WD only** - no `kernel_regularizer=L2(...)` combined (LESSONS L72).
- **No `make test`** as a regression check (~1.5h pre-push hook). Scope pytest to changed module.
- **User pushes commits** - we commit locally only.
- **Naming**: training script is `pretrain.py` under `src/train/cliffordnet/wikipedia/`, matches `bert/wikipedia/pretrain.py`. Add `__init__.py` (empty).
- **For MLM with `load_wikipedia_train_val`, `min_article_length>=500`** - opposite of packed CLM convention.
- **`prepare_dict_keyed_compile` NOT required** - MLM returns scalar loss via internal `compute_loss`. Verified in `train_embeddings.py`.

### SOFT
- Mirror `src/train/bert/wikipedia/pretrain.py` (data + callbacks + mega-epoch shape) + `src/train/cliffordnet/train_embeddings.py` (model + MLM + AdamW + argparse).
- Wikipedia-only (skip BookCorpus).
- Use `WarmupSchedule(... CosineDecay(...))` from `dl_techniques.optimization.warmup_schedule` (NOT `create_warmup_lr_schedule` which assumes epoch-shaped schedule).
- Drop `tf.distribute.MirroredStrategy` wrapping - single GPU.
- Save encoder-only at end via `mlm_model.encoder.save(...)`; periodic `ModelCheckpoint` during training saves full MLM model.
- Use `BackupAndRestore` for preemption tolerance.
- Smoke recipe: `--smoke` -> nano/batch=16/seq=128/max_train_samples=2000/total_steps=400/warmup=40/mixed_precision=False/save_freq=200/log_freq=10. Acceptance: train loss decreases.
- Real recipe defaults: variant=nano/batch=32/seq=512/max_train_samples=None/total_steps=50000/warmup=5000/lr=5e-4/wd=0.01/mask_ratio=0.15/mixed_precision=True/save_freq=5000/log_freq=100. Acceptance: train+val loss decline; encoder saved.
- argparse: `--gpu --variant --max-seq-length --batch-size --total-steps --warmup-steps --learning-rate --weight-decay --mask-ratio --pooling-strategy --save-dir --no-mixed-precision --max-train-samples --min-article-length --num-shards --seed --smoke`.
- `__init__.py` empty (matches `train/bert/wikipedia/__init__.py`).
- Verification: 1) py_compile new files; 2) import smoke; 3) `--smoke` GPU 0 run (~5 min) - assert final loss < initial; 4) real GPU 0 run (multi-hour, background) - observe train/val curves, save checkpoints.

### GHOST (considered & rejected)
- *HF streaming a la `bert/wikipedia/pretrain.py`* - REJECTED. Local cache gives val split + reproducibility + parallel tokenization. See F-002.
- *`MirroredStrategy` for multi-GPU* - REJECTED. Single GPU constraint; heterogeneous GPUs (24GB+12GB) bottleneck.
- *Mirror `train_embeddings.py` and just swap IMDB->Wikipedia* - PARTIALLY rejected. Yes for model/optim; NO for callbacks - need `BackupAndRestore + save_freq ModelCheckpoint` for multi-hour, not epoch-end `create_nlp_callbacks`.
- *Include BookCorpus* - REJECTED. HF restricts; landmine. Wiki-only is honest.
- *Add MLM smoke test inside this plan* - REJECTED. MLM<->CliffordNetEmbedding already covered by `plan_2026-05-12_632605aa` SC7 + existing test suite.
- *Validate via existing `train_embeddings.py --smoke`* - REJECTED. New file is structurally different (single mega-epoch, save_freq, BackupAndRestore, mixed precision); smoke must run the NEW code path.

### Exploration Confidence
- **Scope: deep** - read `bert/wikipedia/{pretrain.py,pretrain_english.py}` end-to-end, `train_embeddings.py`, `datasets/nlp.py:load_wikipedia_train_val`. SYSTEM.md/LESSONS.md/just-closed FINDINGS reviewed.
- **Solutions: constrained** - single correct shape: Pattern B data + bert/wikipedia callback set + train_embeddings.py model wiring. Variant/seq/batch empirically grounded.
- **Risks: clear** - (a) BookCorpus 401 - avoided; (b) `tf.py_function` graph shapes - proven by `preprocess_mlm_dataset`; (c) GPU OOM - mitigated by nano-first; (d) BackupAndRestore + mixed precision - well-tested upstream; (e) multi-hour run interruption - mitigated by BackupAndRestore + frequent ModelCheckpoint.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-12_632605aa
### Index

| ID | Topic | File | Summary |
|----|-------|------|---------|
| F-001 | BERT encoder API + MLM contract | `findings/bert-encoder-api.md` | BERT is a pure encoder, no head. Dict in/out: `{input_ids, attention_mask, ...}` -> `{last_hidden_state (B,T,C), attention_mask}`. Required attr: `encoder.hidden_size`. Public surface = 3 names: `<Model>, create_<model>, create_<model>_with_head`. `MaskedLanguageModel` owns the MLM head and calls `encoder(inputs)["last_hidden_state"]`. Template: `MODEL_VARIANTS`, `PRETRAINED_WEIGHTS`, `_download_weights` raises `NotImplementedError`, `from_variant` with narrow `try/except (IOError, OSError, ValueError)` (plan_9357982a D-001). Pooling for embeddings: expose `pooling_strategy` ("mean"/"cls"/"max", default "mean"), compute pooled_output inside `call()`, emit dict `{last_hidden_state, pooled_output, attention_mask}`. |
| F-002 | lmunet structure + non-causal Clifford blocks | `findings/lmunet-and-clifford-block.md` | Non-causal `CliffordNetBlock` (clifford_block.py:481) and `CliffordNetBlockDSv2` (line 1695) ALREADY EXIST with matching API. Zero changes to `clifford_block.py` needed. Surgical diff vs `lmunet.py`: swap causal classes for non-causal siblings, DELETE `_causal_upsample(x, s)` helper + its call site, REPLACE LM head with pooling head, change input to BERT-style dict, change output to embedding dict. Keep: variants ladder (nano/mini/base/large/xl), `channel_multiplier=1.5`, right-pad-to-stride + crop-back, linear DropPath schedule, channel ladder math, all invariants. `hidden_size == base_channels` (U-Net restores to top-level C0 at head). |
| F-003 | Training pattern + objective + naming | `findings/trainer-pattern-and-objective.md` | Mirror `src/train/bert/pretrain.py` (~258 LOC) — Pattern-3 NLP MLM. Objective: MLM (BERT-style) — only self-supervised single-trainer option with existing infra (`MaskedLanguageModel`, `preprocess_mlm_dataset`). Encoder + `MaskedLanguageModel` wrapper -> `model.compile(optimizer=AdamW(...))` -> `model.fit(train_ds, validation_data=val_ds)`. Helpers in `train.common.nlp`: `create_tokenizer`, `load_text_dataset`, `preprocess_mlm_dataset`, `create_warmup_lr_schedule`, `create_nlp_callbacks`. Use tiktoken cl100k_base (vocab=100277, pad=100266, mask=100267). New file path: `src/dl_techniques/models/cliffordnet/embedding_unet.py` with class `CliffordNetEmbedding`. Trainer: `src/train/cliffordnet/train_embeddings.py`. Update `cliffordnet/__init__.py` to re-export 3 new names. |

### Key Constraints

### HARD
- **Keras 3 / TF 2.18 idioms**: `@keras.saving.register_keras_serializable()` on every new class, `keras.ops` only, full `get_config()` round-trip, `dl_techniques.utils.logger` only (no `print`).
- **MaskedLanguageModel encoder contract**: encoder must accept dict `{input_ids, attention_mask?}` and return dict with `"last_hidden_state": (B, T, hidden_size)`; must expose `self.hidden_size: int`.
- **Public surface = 3 names** per __init__.py recipe (matches resnet/bert/tree_transformer): `<Model>, create_<model>, create_<model>_with_head`.
- **`_download_weights` raises `NotImplementedError`** + `from_variant` narrow `try/except (IOError, OSError, ValueError)` (plan_9357982a D-001 ghost — no silent random-init fallback).
- **`pad_token_id=100266`** wired into model config; matches tiktoken cl100k_base used by `create_tokenizer`. Mismatched pad is a silent semantic bug (LESSONS — tree_transformer pad_token_id incident).
- **AdamW WD only — no `kernel_regularizer=L2`** combined with AdamW (LESSONS L72).
- **Single GPU jobs only**; `MPLBACKEND=Agg` mandatory for any training-script invocation.
- **No `make test`** as a regression check (~1.5h pre-push hook). Scope pytest to changed module only.
- **Naming**: training script `train_embeddings.py`, NOT `train.py` (shadowing per train/CLAUDE.md). Model file `embedding_unet.py` to disambiguate from `lmunet.py` (causal LM variant).
- **User pushes commits**; we commit locally only.
- **Non-causal U-Net = standard symmetric upsample**: DELETE `_causal_upsample(x, stride)` from the new file's `call()` path. KEEP `pad-to-multiple-of-total_stride` + final crop (still needed for arbitrary seq_len divisibility — AccUNet lesson L102).

### SOFT
- Follow BERT pretrain.py structure: `TrainingConfig` class with class-level defaults, `create_<model>_mlm_model(config)`, `compile_model(...)`, `train(...)`, `evaluate_model(...)`, `main()` with argparse.
- Mirror lmunet.py MODEL_VARIANTS shape (nano/mini/base/large/xl); use the SAME hyperparameters except for any block-class-specific quirks (e.g. non-causal DSv2 default `stream_pool="blur"` but we can pin to `"avg"` to match the causal baseline behavior).
- Default `pooling_strategy="mean"` (mask-aware mean over tokens) — robust without special-token assumption. Also expose "cls" and "max".
- Default `dataset_name="imdb_reviews"` for smoke; match BERT pretrain default.
- Re-use the same defaults BERT uses for MLM hyperparams: `mask_ratio=0.15`, `random_token_ratio=0.1`, `unchanged_ratio=0.1`, `mlm_head_activation="gelu"`, `mlm_head_dropout=0.1`, `layer_norm_eps=1e-12`.
- Skip `token_type_ids` entirely (single-sequence; lmunet doesn't model it). Encoder accepts dict with `{input_ids, attention_mask}`; ignore `token_type_ids` if passed.
- For test file, mirror sibling test structure: `tests/test_models/test_cliffordnet/test_embedding_unet.py` with the standard 5-6 tests (init, forward, gradient, serialization round-trip, variants, pooling-strategy parity).

### GHOST (considered & rejected)
- *"Add a `causal=False` flag to `CausalCliffordNetBlock`"* — REJECTED. Non-causal sibling already exists. Flag would bloat 2 classes for zero benefit.
- *"Modify `lmunet.py` in place to support a `causal` flag"* — REJECTED. lmunet is the causal CLM variant locked by the existing `train_cliffordnet_nlp_unet.py` trainer + tests. Adding a flag risks regression. Pure additive new file is zero blast radius.
- *"Use `CausalLanguageModel` wrapper instead of `MaskedLanguageModel`"* — REJECTED. The model is bidirectional; CLM is an autoregressive objective that requires causal masking.
- *"Use the existing `lm_routing.py` head"* — REJECTED. RoutingProbabilitiesLayer is a vocab-probability head, not an embedding head.
- *"Train via SimCSE / contrastive from scratch"* — REJECTED for iter-1. No precedent in dl_techniques; needs sentence-pair generation. MLM gives a usable embedding encoder; SimCSE/sentence-transformer fine-tune is a separate follow-up plan.
- *"Place the embedding model under a new `cliffordnet_embedding/` subdir"* — REJECTED. User said "or a new subdir if appropriate — let exploration decide". The model is conceptually a cliffordnet variant (shares lmunet's U-Net body 95%), and the existing `cliffordnet/` package already hosts multiple model classes (CliffordNet, CliffordCLIP, CliffordNetLMRouting, CliffordNetLMUNet). Adding `CliffordNetEmbedding` to the same package matches established convention.

### Exploration Confidence
- **Scope: deep** — all 4 in-scope files read end-to-end (`bert.py` 1010 LOC, `lmunet.py` 710 LOC, `clifford_block.py` skimmed at class-header level — both causal/non-causal siblings confirmed). Sibling references: `train/bert/pretrain.py` + `finetune.py`, `train/cliffordnet/train_cliffordnet_nlp_unet.py`, `MaskedLanguageModel` source, `train.common.nlp` helper inventory. SYSTEM.md + LESSONS.md fully reviewed for ghost constraints.
- **Solutions: constrained** — exactly one architecturally-correct approach: new file mirroring lmunet structure with non-causal swaps + pooling head + BERT-style dict I/O. MLM is the only single-trainer self-supervised objective with existing infra.
- **Risks: clear** — main risks: (a) `MaskedLanguageModel` strictly requires `hidden_size` attr — covered, set explicitly; (b) attention_mask passthrough — covered, mirror BERT's pattern; (c) right-pad-to-stride zeroing real token embeddings — same as lmunet, already proven safe; (d) `_causal_upsample` removal — mathematically trivial since bidirectional has no causality constraint; (e) `pad_token_id` semantic bug — covered by HARD constraint above.

### Synthesis

Three deliverables, all additive, zero blast radius on existing code:

1. **`src/dl_techniques/models/cliffordnet/embedding_unet.py`** (~600-700 LOC) — class `CliffordNetEmbedding(keras.Model)` mirroring `CliffordNetLMUNet` structure. Surgical changes: non-causal block classes, remove `_causal_upsample`, BERT-style dict I/O, pooling layer + pooled_output in output dict, expose `self.hidden_size`. Plus `create_cliffordnet_embedding(...)` factory and `create_cliffordnet_embedding_with_head(...)` (NLPTaskConfig-based). 5 variants (nano/mini/base/large/xl) using lmunet's hyperparam ladder.

2. **`src/dl_techniques/models/cliffordnet/__init__.py`** — re-export 3 new names (verbatim 9-line edit).

3. **`src/train/cliffordnet/train_embeddings.py`** (~260 LOC) — Pattern-3 NLP MLM mirror of `train/bert/pretrain.py`. Uses `MaskedLanguageModel` wrapper, `create_tokenizer`, `preprocess_mlm_dataset`, `create_warmup_lr_schedule`, `create_nlp_callbacks`. Smoke recipe `--variant nano --epochs 1 --max-samples 256 --batch-size 8`.

4. **`tests/test_models/test_cliffordnet/test_embedding_unet.py`** (~200 LOC) — class-based test mirroring sibling tests in `test_cliffordnet/`. 6 tests: init+forward, gradient flow, save/load round-trip (`.keras`), each pooling_strategy variant, MaskedLanguageModel-integration smoke, from_variant for nano.

Gate: pytest on the new test file PASS; py_compile + import smoke for the trainer; 1-epoch smoke training run on GPU 0 with synthetic config completes without error.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-12_e9584ff4
### Index

| ID | Topic | File | Summary |
|----|-------|------|---------|
| F-001 | BLT architecture inventory | `findings/blt-architecture.md` | `ByteLatentTransformer` is a complete end-to-end byte-LM (737 LOC `model.py`). Input: int32 byte-token IDs `(B, T_bytes)` in `[0, 260)` (offset +4). Output: plain `(B, T, 260)` logits (NOT dict-keyed). Has dynamic entropy patcher (`EntropyModel + DynamicPatcher`), `LocalEncoder` cross-attn pooling, `GlobalTransformer`, `LocalDecoder` cross-attn back to bytes. `__init__.py` is empty. Existing `src/train/blt/train_blt.py` uses synthetic data, multi-stage trainer, does NOT touch `train.common.nlp`. No public tests for BLT. `ByteTokenizer` at `dl_techniques/layers/blt_blocks.py:232` is a Python-side text↔byte helper, not a tf.data pipeline. |
| F-002 | Current CliffordNetLMRouting integration surface | `findings/lm-routing-current.md` | Token-ID-keyed LM: `vocab_size=50261` (tiktoken gpt2 + 4 specials), `max_seq_length=512`, `channels∈{128..768}` is the feature dim D (NOT Clifford algebraic dim). 3 embedding strategies (`hce`/`albert`/`dense`), CausalCliffordNetBlock×depth stack on 4-D tensors, head = `RoutingProbabilitiesLayer(output_dim=vocab_size, mode={trainable,deterministic})` producing **probabilities** in `[eps, 1-eps]` summing to 1. Output dict key `"logits"` (values are probs — D-001 anchored). Loss must use `from_logits=False` (D-002 anchored). 5 variants nano/mini/base/large/xl. Three plug-in seams identified (A: full byte-vocab swap = 260 classes, d=9 decisions; B: BLT front-end + Clifford stack; C: drop CliffordNet → use BLT). |
| F-003 | Existing training pipeline + byte scaffolding | `findings/training-pipeline.md` | 1299-LOC Pattern-3 trainer: tiktoken gpt2, `MaskedCausalLMLoss(from_logits=False)`, `AdamW + warmup-cosine + clipnorm=1.0` (no kernel reg), `prepare_dict_keyed_compile + build_clm_metrics("gpt2")`, `StepCheckpointCallback + GenerationProbeCallback + augment_probe_results hook`, data wrapper `(x, y) → (x, {"logits": y})`, `.repeat()` on train, HF Wikipedia OR TFDS sources, `data_seed = seed + initial_step` resume. **NO byte-level tf.data pipeline exists anywhere** — `train/blt/train_blt.py` uses synthetic data. `train.common.nlp` knows only tiktoken. Three deltas required for raw-byte path: (1) write `preprocess_clm_byte_dataset` (graph-mode `tf.io.decode_raw` recommended over `tf.py_function`), (2) byte-aware probe callback using `ByteTokenizer.tokens_to_text`, (3) `build_clm_metrics` adjustment (vocab=260; BitsPerToken loses meaning, BitsPerCharacter mostly preserved for ASCII). |

### Key Constraints

### HARD

- **Keras 3 / TF 2.18 idioms**: `@keras.saving.register_keras_serializable()` on every new class, `keras.ops` only, full `get_config()` round-trip, `dl_techniques.utils.logger` only (no `print`).
- **Output dict key MUST be `"logits"`** even when values are probabilities — required by data wrapper + `prepare_dict_keyed_compile` + `build_clm_metrics`. Anchored as `# DECISION D-001` in current `lm_routing.py`.
- **`from_logits=False` is required** when the head is `RoutingProbabilitiesLayer` (probabilities, not logits). Anchored as `# DECISION D-002`. If replaced with plain Dense, flips to `from_logits=True`.
- **Single GPU jobs only.** GPU 0 = RTX 4090 24GB, GPU 1 = RTX 4070 12GB. Never parallel. `MPLBACKEND=Agg` mandatory.
- **`__init__.py` for `byte_latent_transformer/` is empty** (matches `models/CLAUDE.md`). Import from `.model` directly.
- **`make test` is forbidden** as a regression check (~1.5h pre-push hook). Scope pytest to changed module only.
- **AdamW double-WD footgun** — never combine `AdamW(weight_decay=W)` with `kernel_regularizer=L2(W)`.
- **`prepare_dict_keyed_compile(model, output_key="logits")` is mandatory** for dict-output trainers (SYSTEM.md invariant).
- **`train_<model>.py` naming** — must NOT be `train.py`. The new script must be `train_cliffordnet_nlp_routing_blt.py`.
- **User pushes commits** — commit locally only.

### SOFT

- Follow Pattern-3 NLP CLM conventions: `StepCounter` first, `StepCheckpointCallback`, `GenerationProbeCallback._post_generate_hook = augment_probe_results`, AdamW + warmup-cosine, `prepare_dict_keyed_compile`, dict label wrapper, `.repeat()` on train, `data_seed = seed + initial_step` resume.
- CLI uniformity: `--steps-per-epoch`, `--seed`, `--min-article-length`, `--shuffle-shards`, `--resume`.
- Mirror existing `train_cliffordnet_nlp_routing.py` shape where byte-domain doesn't force divergence.
- Prefer graph-mode `tf.strings.bytes_split → tf.io.decode_raw(uint8) → +byte_offset → concat-and-chunk` for the byte pipeline.
- Do NOT copy the class-based multi-stage `train/blt/train_blt.py` pattern; keep Pattern-3 functional flow.

### GHOST (considered & rejected unless re-validated)

- *"d=8 for 256 bytes is the Clifford block algebraic dim"* — FALSE. `channels` (feature dim D = 128..768) is NOT the Clifford algebraic dim. The "d" in routing-tree language is `ceil(log2(padded_vocab))`. For 260-vocab → padded 512 → d=9; for 256-vocab → padded 256 → d=8.
- *"Routing-tree pathologies apply identically to bytes"* — FALSE. The 16-decisions-for-50K-classes ceiling, BPE leaf-arrangement penalty, and gradient asymmetry documented in `train_cliffordnet_nlp_routing.py:42-99` **vanish** at byte vocab: d=9 ≫ info floor; byte values are a meaningful natural ordering (adjacent bytes share lower bits = mostly meaningful for ASCII).
- *"Byte path needs new layers"* — partially false. `ByteTokenizer + EntropyModel + DynamicPatcher + LocalEncoder + GlobalTransformer + LocalDecoder` already exist in `dl_techniques/layers/blt_blocks.py`. Only the tf.data pipeline + trainer wiring are missing.
- *"BLT vocab is 256"* — FALSE. BLT uses `vocab_size=260` (256 bytes + 4 specials at id 0..3). Byte values offset +4 before becoming token IDs.
- *"Probe must continue to use tiktoken"* — FALSE. Bytes ARE the tokens; decode via `ByteTokenizer.tokens_to_text`, not tiktoken.

### Exploration Confidence

- **Scope**: deep. All three target files read end-to-end (BLT model 737 LOC; lm_routing 503 LOC; routing trainer 1299 LOC). `train.common.nlp` + `routing_probabilities.py` + `blt_blocks.py` + `train/blt/train_blt.py` sampled. Grep across `src/` for byte-level scaffolding exhausted.
- **Solutions**: open — at least 3 architecturally distinct options (A: byte-vocab swap on CliffordNetLMRouting, B: BLT front-end + Clifford global stack, C: drop CliffordNet entirely and wrap BLT). User must choose before PLAN.
- **Risks**: clear. Main risks: (a) tf.data byte-pipeline correctness (graph-mode vs py_function), (b) probe callback rewrite (tiktoken is hard-wired across `_generate`), (c) loss/metrics: `from_logits` flag flips depending on head; `BitsPerToken` loses meaning at byte level, (d) sequence-length budget — bytes are 4× longer than BPE tokens.

### Synthesis

Three coupled but independent choices PLAN cannot resolve without user input:

1. **Architecture**: Option A keeps `CliffordNetLMRouting` and swaps the token vocab → bytes (smallest delta, ~1 day, ~50-100 LOC model changes). Option B uses BLT's entropy patcher + local encoder to feed the CausalCliffordNetBlock stack with patch reps (novel hybrid, largest delta). Option C abandons CliffordNet — the new file becomes a Pattern-3 trainer over `ByteLatentTransformer` (smallest LOC but maximal semantic change).
2. **Routing-head fate**: Keep `RoutingProbabilitiesLayer(output_dim=260, mode={trainable, deterministic})` — d=9 decisions for 260 classes is well above the info floor; the original pathologies vanish — OR replace with plain `Dense(260)` softmax (simpler, but loses the routing-tree research angle the original module was built around).
3. **Byte-data path**: Graph-mode `tf.io.decode_raw(uint8)` over `tf.strings.bytes_split` (faster, no GIL) vs `tf.py_function` wrapping `ByteTokenizer.text_to_bytes` (simpler but slower). Must be packed (concat-and-chunk) to match the existing CLM pipeline.

The byte-level layer primitives already exist (`blt_blocks.py`). Missing pieces are the **tf.data byte pipeline** and the **trainer wiring**. The "Rewrite `lm_routing.py`" portion is small under Option A (~50-100 LOC); the "add `train_cliffordnet_nlp_routing_blt.py`" portion is ~700-900 LOC regardless of option (mirror of existing trainer with byte deltas).

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-12_995a621a
### Index

| # | File | Topic | Iter |
|---|------|-------|------|
| F1 | findings/topic-a-readme-and-intent.md | NAM history + 5-iteration arc + iter-5 end-state (zero-param arithmetic) | 0 |
| F2 | findings/topic-b-training-scripts.md | train_dfsa.py vs train_dfsa_ste.py mechanics | 0 |
| F3 | findings/topic-c-data-and-eval.md | Curriculum data generator + eval harness | 0 |
| F4 | findings/gradient-breaks.md | Census of every non-diff op in DFSA forward pass; what STE bridges and doesn't | 0 |
| F5 | findings/tree-encoder-intent.md | What "reviving tree-transformer" can concretely mean (interpretations A/B/C) | 0 |

### Key Constraints

### HARD
- **HARD**: float32 precision caps operands at ~10 digits. Architectural ceiling.
- **HARD**: `act_steps` bounds max operators per expression (default 3-4).
- **HARD**: Arithmetic tokenizer vocab (digits 0-9, `+-*/`, parens, space, `[RES]`, BOS/EOS/PAD). Out-of-scope = out-of-scope.
- **HARD**: Arithmetic correctness is currently **100% with random weights** (iter-5). Any revival that regresses accuracy is a non-starter unless the user accepts the trade.
- **HARD**: `argmax` on `op_pos_hard` and integer re-tokenization sever cross-step gradients structurally — no STE recovers this without architectural change.

### SOFT
- **SOFT**: Tree encoder (~1.5M params) currently dead in arithmetic path. `reduction_scorer` (Dense 1) declared but never called. `op_classifier` (Dense 4) only called when `use_ste=True`.
- **SOFT**: STE path in `train_dfsa_ste.py` opens *one* gradient channel (host loss → `op_one_hot` → `op_classifier` → `x`), but learning signal is weak: `L_result = 0` on correct samples; `L_operator` supervises a trivial `token_id - 14` look-up.

### GHOST
- **GHOST**: "Tree encoder must learn PEMDAS / arithmetic mechanism." Iter-5 falsified this. The honest reframe: it should learn a **representation** grounded by some signal (correctness, span supervision, constituency).
- **GHOST**: "Phase 3 of `train_dfsa_ste.py` is unimplemented because gradient flow is broken." Actually, gradient flow is fine (the probe at lines 277-289 confirms all relevant vars receive grads). Phase 3 is unimplemented because (i) there is no host model to integrate with, and (ii) the existing learning signal is too weak. Blocker is loss surface, not plumbing.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-12_ebb5fac5
### Index

| ID | Topic | File | Summary |
|----|-------|------|---------|
| F-001 | prism model + README review | findings/prism-model-review.md | 12 issues (3 HIGH doc bugs, 3 MEDIUM doc gaps, 6 LOW). NO model.py bugs. Tests already present. Recommend prose-only README rewrite + trainer scaffold rewrite + new export.py. |
| F-002 | tirex -> prism trainer mapping | findings/tirex-pattern-mapping.md | Existing train_prism.py is a Pattern-2 mirror of tirex but drifts: wrong backronym in docstring, missing export_onnx hook + _export_to_onnx method, no standalone export.py, os._exit cleanup, missing preset validation. export.py recipe = near-verbatim tirex copy. PRISM emits single tensor so no --output_key needed. |

### Key Constraints

### HARD
- NO model.py code edits in scope (informational review; matches LESSONS L117).
- Sibling consistency: README mirrors adaptive_ema/tirex template; trainer is Pattern-2 (mirrors train/tirex).
- Keras 3 idioms preserved; dl_techniques.utils.logger only; MPLBACKEND=Agg documented.
- Existing tests at tests/test_models/test_prism/ must remain green. Scope pytest to that dir.
- __init__.py stays empty per models/CLAUDE.md.
- Acronym lock-in: "Partitioned Representations for Iterative Sequence Modeling". Update model.py + train_prism.py module docstrings (docstring-only, no code).
- User pushes commits themselves; serial GPU only (memory).

### SOFT
- Do not touch predict_quantiles() self-mutation (I-8); document only.
- ONNX export OFF by default in trainer (--no-onnx pattern).
- Drop placeholder image entirely.
- Surface I-1..I-12 in README Limitations section.

### GHOST
- Rewriting model.py for I-7/I-8/I-9/I-12 polish: rejected (would require regression tests; out of scope).
- --output_key in export.py: rejected (PRISM returns single tensor).
- Live PRISM ONNX smoke in plan: rejected (out of scope; opt-in only).
- Replacing ops.cond in PRISMNode: rejected (touches prism_blocks.py, no test coverage for perf change).

### Exploration Confidence
- Scope: deep. Read all 6 source files end-to-end plus sibling precedents.
- Solutions: constrained. Two deliverables with explicit line-by-line precedents.
- Risks: clear. (a) PRISM ONNX path never tested -> opt-in only; (b) acronym lock-in must update model.py docstring too; (c) test_model.py presumed-green -> pytest as CORE gate.

### Synthesis
Three-step additive plan, zero model.py code edits (only docstring acronym fix):
1. Rewrite README.md per adaptive_ema/tirex template; fix I-1/I-2/I-3, expand tables (I-4), add quantile mode section (I-5), drop placeholder + add Limitations folding I-1..I-12 (I-6). Sync model.py module docstring acronym.
2. Rewrite src/train/prism/train_prism.py to byte-align tirex Pattern-2 (acronym fix, export_onnx hook, try/finally cleanup, preset validation, sync flag spellings).
3. Create src/train/prism/export.py as near-verbatim tirex copy (CPU-only env-pin, auto-detect context_len from get_config, single-tensor verification at rtol=atol=1e-4).

Gate: pytest tests/test_models/test_prism/ -x green after step 1; py_compile + import smoke for steps 2+3. No make test. No live training in plan scope.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*

## plan_2026-05-12_5f0e087c
### Index

| ID | Topic | File | Summary |
|----|-------|------|---------|
| F-001 | EMA vectorization (I-3) | `findings/ema-vectorization.md` | `ExponentialMovingAverage.call()` Python loop in `ema_layer.py:119`. Only consumer = `AdaptiveEMASlopeFilterModel` (sibling `EMASlopeFilter` in same file is unused). `keras.ops.scan` and `keras.ops.associative_scan` both available in Keras 3.8. Recommend `ops.scan` (bit-equivalent, both modes); `associative_scan` reserved as optional perf optimization with ~1e-6 drift. Note: existing `adjust=True` branch is a CUSTOM variant (not pandas semantics) — vectorization target is the existing recurrence. No existing `tests/test_layers/test_time_series/test_ema_layer.py` — new file needed. |
| F-002 | Polish items I-14..I-19 | `findings/polish-items.md` | I-14 add `compute_output_shape`. I-15 verify-only. I-16 docstring sentence (hard vs soft exclusivity). I-17 add References section to module docstring (LeBeau 1992, Bollinger, Koenker & Bassett 1978 — mirror README §13). I-18 raise `ValueError` when `F>1` AND head enabled (user-chosen option (a)); update README §11 with L-7, add a test. I-19 `logger.warning` when `learnable_thresholds=True` AND `quantile_head_config=None`; add caplog test. Total ~70 LOC model.py + ~25 LOC tests + ~5 LOC README. |
| F-003 | Trainer smoke + ONNX export | `findings/trainer-and-onnx-smoke.md` | Trainer at `src/train/adaptive_ema/train_adaptive_ema.py`, never executed. Pattern-2; synthetic data via `TimeSeriesGenerator`; wrapper selects single tensor (signal_between for classification, slope_quantiles for quantile) sliced to trailing prediction_horizon. Smoke recipe: 3 epochs, batch 32, 20 steps/epoch, input_length=64, prediction_horizon=16, GPU 0, no-warmup. Run BOTH modes sequentially. ONNX via Keras 3.8 `model.export(format="onnx")`; `onnxruntime 1.23.2` confirmed installed. Vectorize EMA (F-001) BEFORE smoke + ONNX so we exercise final code path (Python-loop unrolling in tf2onnx is the biggest known ONNX risk; `keras.ops.scan` collapses to a single Scan op). |

### Key Constraints

### HARD
- **Numerical equivalence**: vectorized `ExponentialMovingAverage` must match existing Python-loop output within `atol=1e-6, rtol=1e-6` for BOTH `adjust=True` AND `adjust=False`, across `period ∈ {1, 5, 25, 100}` and `T ∈ {1, 16, 128, 512}`.
- **Existing test suite must remain green**: `tests/test_models/test_adaptive_ema/test_model.py` (14 tests, currently passing) — `test_serialization_round_trip` is an implicit equivalence gate.
- **Keras 3 / TF 2.18 idioms**: `@keras.saving.register_keras_serializable()`, `keras.ops`, full `get_config()` round-trip, `dl_techniques.utils.logger` only, no `print`.
- **Test scope**: pytest scoped to `tests/test_models/test_adaptive_ema/` AND `tests/test_layers/test_time_series/test_ema_layer.py` (NEW). Never run `make test`.
- **GPU policy**: serial training runs only; pin GPU 0 (RTX 4090) via `CUDA_VISIBLE_DEVICES=0`; `MPLBACKEND=Agg` mandatory.
- **Commit prefix**: `[iter-N/step-M] adaptive_ema: <description>` (or `ema_layer:` for the layer step).
- **User pushes commits themselves** — commit locally only.
- **I-18 design choice locked by user**: option (a) — raise `ValueError` when `F>1` AND `quantile_head_config is not None`. Update README L-7. Add test.

### SOFT
- Prefer `keras.ops.scan` over `associative_scan` for bit-equivalent semantics. Document trade-off in layer docstring if relevant.
- Reference section (I-17) — keep academic-only (LeBeau 1992, Bollinger, Koenker & Bassett 1978) mirroring README §13; do NOT fabricate URLs. A "Background reading:" line with a TradingView search URL is acceptable but optional.
- For I-18 raise site, prefer `call()` over `build()` (Keras passes a `KerasTensor` with known static shape at first call).
- For I-19 warning, prefer `dl_techniques.utils.logger.logger.warning` over `warnings.warn`. Test via `caplog` after confirming propagation; fall back to monkeypatching `logger.warning` if caplog doesn't catch it.
- Smoke run uses `--no-warmup` to skip the WarmupSchedule for a faster, simpler smoke (3 epochs cannot meaningfully exercise warmup anyway).
- ONNX export verification tolerance: `rtol=1e-4, atol=1e-4` (export.py default).

### GHOST (considered & rejected)
- *"Switch `adjust=True` semantics to pandas-canonical while we're vectorizing."* — No. The current variant has been the contract for all 14 existing tests. Changing math is out of scope. Use `ops.scan` to preserve the exact recurrence + division.
- *"Use `associative_scan` for maximum speed."* — Defer. Bit-equivalence trumps perf for a refactor of a shared layer.
- *"Apply I-18 head per-feature (option (b))."* — Rejected by user; option (a) is the clean contract.
- *"Run trainer smoke before vectorizing the EMA loop."* — Rejected; we'd be smoke-testing a deprecated code path. Vectorize first, then smoke + ONNX.

### Exploration Confidence
- **Scope: deep** — all 5 in-scope files read end-to-end (`ema_layer.py`, `adaptive_ema/model.py`, `adaptive_ema/__init__.py`, `adaptive_ema/README.md`, `train_adaptive_ema.py`, `export.py`); all consumers grep-confirmed; `keras.ops.scan`/`associative_scan` empirically verified; `onnxruntime` install verified.
- **Solutions: constrained** — I-3 has 3 candidates; recommendation is `ops.scan` for bit-equivalence. All polish items (I-14..I-19) are mechanical. Trainer/ONNX smoke recipes are direct mirrors of `train/tirex/` precedents.
- **Risks: clear** — main risks: (a) `ops.scan` carry-shape mismatch for `(B,T)` vs `(B,T,F)` (mitigated by always lifting to 3D inside scan, squeezing on return); (b) trainer crash on first end-to-end run (mitigated by tiny smoke + revert-first); (c) ONNX `model.export` complaining about the new scan op (low risk — tf2onnx supports `tf.scan` natively); (d) caplog vs `dl_techniques.utils.logger` propagation (fallback: monkeypatch).

### Synthesis
4-deliverable plan: (1) **I-3** vectorize `ExponentialMovingAverage` (shared layer; one direct consumer + sibling in same file). Numerical equivalence at 1e-6 is the hard gate, enforced by a new equivalence test that pastes the old loop as a reference. (2) **I-14..I-19** polish — mechanical edits to `model.py` + README + 2 new tests; ~100 LOC total. (3) **Trainer smoke** — execute BOTH modes serially on GPU 0 with a tiny 3-epoch config; fix the trainer if it crashes. (4) **ONNX numerical-match** — export the quantile-smoke checkpoint and verify via `export.py --verify --output-key slope_quantiles` within `atol=1e-4`. Ordering matters: I-3 before smoke (so we ONNX-export the vectorized path) and before polish (so polish lands on the vectorized layer). All four fit a single iteration.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-12_86f14c6e
### Index

| ID | Topic | File | Summary |
|----|-------|------|---------|
| F-001 | Scope from prior code review | `findings/scope-from-prior-review.md` | Enumerates I-1, I-5..I-13 fixes + I-2 head redesign (recommend option b — Conv1D featurization). |
| F-002 | System context | `findings/system-context.md` | Files involved, current vs target ctor signature, weight inventory, round-trip risk. |
| F-003 | Test conventions | `findings/test-conventions.md` | Mirrors `tests/test_models/test_tirex/test_model.py`; class-based, fixtures, atol=1e-5. |

### Key Constraints

### HARD
- Serialization round-trip (`.keras` save/load) must pass.
- Gradient flow on the 2 new learnable threshold weights (`midpoint_var`, `log_half_range_var`) must be non-zero when `learnable_thresholds=True`.
- `train_adaptive_ema.py --mode quantile` must keep working — `slope_quantiles` output shape stays `(B, T, K)`.
- Soft signals must be in `[0, 1]` (sigmoid outputs — guaranteed by new formula).
- Do NOT run `make test` (1.5h pre-push hook). Scope pytest to `tests/test_models/test_adaptive_ema/` only.
- Use `dl_techniques.utils.logger` only; no `print`.

### SOFT
- Mirror `models/vit/__init__.py` `__init__.py` shape (docstring + `__all__`).
- Mirror `tests/test_models/test_tirex/test_model.py` test class structure.
- Keep threshold weights in `float32` even under mixed precision (I-10).
- README + trainer doc updates land in the same plan as code changes (LESSONS).

### GHOST
- None — prior plan_94b9fab5 was informational; no inherited stale constraints. The previous `softplus(abs(...))` parameterization was an isolated mistake, not an artifact of an obsolete constraint.

### Exploration Confidence
- Scope: **deep** — all target `model.py` lines identified; all consumer files audited (trainer, README, export); no other callers.
- Solution space: **constrained** — I-2 has a clear (a)/(b) trade-off resolved; all other items mechanical.
- Risk visibility: **clear** — weight-rename breaks old `.keras` checkpoints, none exist in CI/user workflows; accepted in F-002.

### Synthesis
Contained, corrective fix to a single 354-LOC `model.py` plus peer-pattern packaging (factory + `__init__.py` re-exports) plus a new test file. New tests are the verification floor. Only design call needing PLAN attention is I-2: recommend option (b) — small causal `Conv1D(slope_feature_dim=16, kernel_size=5)` over the slope window before `QuantileSequenceHead`. Trainer keeps working: head output shape unchanged. README + trainer doc updates bundled into same iteration.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*
