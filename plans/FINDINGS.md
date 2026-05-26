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

## plan_2026-05-26_d7a342f2
### Index

| # | Topic | File | Key Takeaway |
|---|-------|------|--------------|
| F-001 | Training script — dataset dispatch | findings/training-script.md | `build_dataset()` is a hard if/elif chain on `config.dataset: str`. Both ADE20K and COCO share `_build_filesystem_dataset()` with identical `/255.0` normalization. No multi-dataset path exists. `steps_per_epoch_override` already present as escape hatch. |
| F-002 | Config, callbacks, mixing primitives | findings/callbacks-and-config.md | `TrainingConfig.dataset: str` (single). No mixer utility for images in `src/train/`. `tf.data.Dataset.sample_from_datasets` is available in TF 2.18 and used in `dl_techniques/datasets/nlp.py:184`. Callbacks are dataset-agnostic (accept any numpy batch). |
| F-003 | Tests and TFDS availability | findings/tests-and-datasets.md | Tests use synthetic numpy only — no test changes needed. COCO/2017 available locally. ADE20K at `/media/arxwn/data0_4tb/datasets/ade20k` (filesystem glob, not TFDS). Mixing CIFAR+filesystem is incompatible (different norm spaces). |

### Key Constraints

### HARD
- `TrainingConfig.dataset: str` — single string, no list field exists. Multi-dataset requires adding `datasets: List[str]`.
- Both ADE20K and COCO use `/255.0` → [0,1] normalization — identical, safe to mix.
- CIFAR+filesystem mixing is INVALID: CIFAR uses mean/std normalization in MSE mode, producing incompatible pixel ranges.
- `tf.data.Dataset.sample_from_datasets` works at batch OR file level; using it at file-path level (before batch) gives finer-grained interleaving.
- `_build_filesystem_dataset` returns batched `.repeat()` train + non-repeating val. Must keep backward compat.
- All tests use synthetic numpy — no test changes needed for multi-dataset.
- `val_steps=None` for all filesystem datasets (Keras exhausts naturally).
- `MPLBACKEND=Agg`, single GPU, `results/` at repo root.

### SOFT
- `experiment_name` auto-generated from `config.dataset` — needs to produce `"ade20k+coco_base"` for multi-dataset runs.
- Default `steps_per_epoch` for mixed runs = sum of individual dataset steps (proportional to combined corpus).
- Mix weights should be size-proportional by default (COCO ~118k vs ADE20K ~20k).
- Warning guard at line 980 should trigger when any dataset in the list is a large-image dataset and `image_size==32`.

### GHOST
- "Must restructure `_build_filesystem_dataset`" — it stays unchanged; new `_build_mixed_filesystem_dataset` adds the path on top.
- "Need architectural changes for mixed datasets" — model is fully resolution-agnostic; only the trainer's data pipeline changes.
- "Callbacks need per-dataset awareness" — callbacks accept any numpy array, they are already dataset-agnostic.

### Exploration Confidence
- **scope**: deep — full 1029-line trainer read, all dataset builders analyzed, callbacks read, tests confirmed
- **solutions**: constrained — targeted changes to trainer only; model untouched; 4 sites to update
- **risks**: clear — no architectural risk; regression covered by existing synthetic tests; CIFAR+filesystem mixing validated to fail at config level

### Corrections
*None — first iteration.*

## plan_2026-05-26_d8c33dca
### Index

| # | Topic | File | Key Takeaway |
|---|-------|------|--------------|
| F-001 | Training script and existing viz | findings/training-script-and-viz.md | Only one viz callback (ReconVisualizationCallback): 3-row grid (orig/recon/fixed-z). No latent space viz, no interpolation. |
| F-002 | Model architecture and latent space | findings/model-architecture.md | Latent is 4D spatial `(B, Hp, Wp, latent_dim)`. encode→(mu, log_var), decode(z)→pixels. API: encode(), decode(), sample(). |
| F-003 | Visualization patterns and callbacks | findings/viz-patterns-and-callbacks.md | Pure matplotlib + savefig. No TSNE/PCA in this module. New callbacks go inline in training script or in a new callbacks file alongside it. |

### Key Constraints

**HARD constraints:**
- Latent is 4D spatial `(B, Hp, Wp, latent_dim)` — NOT flat. Must reshape/pool before PCA/t-SNE.
- `MPLBACKEND=Agg` required — headless, no display. All plt calls must work offscreen.
- `jit_compile=False` on compile — no XLA; encode/decode from Python callbacks is safe.
- `model.decode(z)` applies sigmoid for BCE, identity for MSE — always call decode() not decoder directly.
- Val samples are fixed 8 images grabbed at callback construction time.

**SOFT constraints:**
- New callbacks should follow try/except + logger.warning pattern to avoid crashing training.
- Visualization callbacks should use `plt.close(fig)` + `gc.collect()` after every save.
- Fixed random seeds for reproducibility (existing code uses seed=42).
- Callbacks should produce output under `results/.../` subdirectories matching the run's results_dir.

**GHOST constraints:**
- `EpochAnalyzerCallback` was explicitly disabled (`include_analyzer=False`) — do not re-enable.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*

## plan_2026-05-26_b11b0e90
### Index

| ID | Topic | File | One-line summary |
|----|-------|------|------------------|
| F1 | Dataset availability | `findings/dataset-availability.md` | ADE20K (scene_parse150) NOT locally available. Imagenette/320px-v2 and COCO/2017 ARE available locally. TFDS pattern: `tfds.load(..., data_dir=TFDS_DATA_DIR)` → map → batch(drop_remainder=True) → prefetch. |
| F2 | Trainer current state | `findings/trainer-current.md` | `build_dataset()` raises ValueError for anything other than cifar10/cifar100. No --data-dir flag. `val_steps` must be None for non-repeating TFDS val. `_CIFAR_STATS` fallback in viz callback must be None-safe. `set_defaults(image_size=32)` is overrideable via CLI. |
| F3 | Model constraints | `findings/model-constraints.md` | Only hard constraint: `img_size % patch_size == 0`. Model is fully resolution-agnostic (no hardcoded pixel values, no abs positional embeddings). At 256x256 patch_size=8: safe on RTX 4090 at batch 16-32. At 256x256 patch_size=4: marginal (~20GB). SIGReg warning if num_patches < sigreg_knots (not error). |

### Key Constraints

### HARD
- `img_size % patch_size == 0` enforced at `config.py:142` and `encoder.py:160-168`
- ADE20K (`scene_parse150`) is not locally available — needs ~3.5GB download to run
- Imagenette (`imagenette/320px-v2`) and COCO (`coco/2017`) ARE locally available
- `model.compile(loss=None, jit_compile=False)` — all losses via `add_loss`
- All existing tests must pass unchanged
- `CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg` for training; output to repo-root `results/`
- No `git push` — user handles
- Single GPU only

### SOFT
- For large natural image datasets: normalize to `[0, 1]` (divide by 255); use MSE not BCE
- Warning if user selects large dataset but img_size <= 64 (accidental CIFAR default)
- `val_steps=None` for non-repeating TFDS val pipelines
- `--data-dir` flag defaulting to `os.environ.get("TFDS_DATA_DIR")`
- Recommended configs for ADE20K-scale: `base` preset, `img_size=256`, `patch_size=8`

### GHOST
- "ADE20K must be the primary dataset" — Imagenette and COCO are locally available and large enough for development; ADE20K support can be added as a code path even without local data
- "Need architectural changes to support larger images" — model is already resolution-agnostic; only the trainer's data pipeline needs extension

### Exploration Confidence
- **scope**: deep — full trainer read, all locally available TFDS datasets inventoried, model config/encoder/decoder all read, VRAM estimates computed
- **solutions**: constrained — targeted trainer changes only; model is untouched; 3 datasets to wire (imagenette, coco, ade20k)
- **risks**: clear — no architectural risk; main risk is VRAM at aggressive img_size/patch_size combos; CIFAR regression testable with existing pytest

### Corrections
*None — first iteration.*

## plan_2026-05-25_a8325e3f
### Index

| ID | Topic | File | One-line summary |
|----|-------|------|------------------|
| F1 | Model files deep review | `findings/model-files.md` | config.py has "mae" default bug; encoder bottleneck is single Conv2D(2*latent_dim) with Glorot init (no zero-init); decoder has single Conv2DTranspose(patch_size) head; model._beta_kl is plain float (mutable from callback); no skip connections or multi-stage upsampling. |
| F2 | Trainer deep review | `findings/trainer-review.md` | Warmup is flat (initial_lr == warmup_target == config.learning_rate); no beta annealing infrastructure; beta_kl is baked into model config at build time but model._beta_kl is mutable at runtime. |
| F3 | Fix scope decision | (inline) | Fixes 1-4 from analysis (config default, warmup, beta annealing, log_var zero-init) are non-breaking. Fixes 5-7 (skip connections, multi-stage decoder, perceptual loss) require architectural API changes — deferred to next plan. |

### Key Constraints

### HARD
- `model.compile(loss=None, jit_compile=False)` — all losses via `add_loss`.
- All 11 existing tests must PASS unchanged (or with minimal updates for bottleneck rename).
- `CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg` for any run. Output to repo-root `results/`.
- No `git push` — user handles.
- Single GPU only.
- `self._beta_kl` is referenced as `self._beta_kl * kl_loss` in `call()` — mutating it from a Keras callback works (Python attribute lookup per call).

### SOFT
- `beta_anneal_epochs=15` as default ramp window (analysis recommendation: 10-15 epochs).
- `beta_kl_start=0.0` default (start collapsed-free).
- zero-init for log_var_head kernel AND bias.

### GHOST (do NOT inherit)
- "Need to change model API to support beta annealing" — `self._beta_kl` is a Python float, mutating it from a callback is sufficient; no tf.Variable needed.
- "Perceptual loss required now" — needed only after collapse is confirmed resolved; no frozen VGG in repo.
- "Skip connections must be done in same iteration" — architectural change; do after training validates collapse fix.

### Corrections
*None — first iteration.*

### Exploration Confidence
- **scope**: deep — all 5 model source files reviewed via sub-agent; full trainer reviewed; test suite structure confirmed.
- **solutions**: constrained — three targeted fixes: (a) 1-line config default, (b) encoder bottleneck split, (c) trainer warmup + beta annealing callback.
- **risks**: clear — scoped pytest confirms no regression; bottleneck rename doesn't affect test correctness (tests explicitly pass recon_loss_type="mse" and don't inspect layer names).
