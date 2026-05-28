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

## plan_2026-05-27_75849a91
### Index
- `findings/convnext-patch-vae-v2-trainer.md` — CLI flags, dataset pipeline, losses (7 components), optimizer/clip, callbacks, custom train_step. Substitution points identified.
- `findings/convnext-patch-vae-v2-model.md` — Flat single-scale VAE: stem → N ConvNextV2Block → mu/log_var → Sampling → M ConvNextV2Block → ConvTranspose. Constant `(B, Hp, Wp, embed_dim)`. Block I/O contract documented.
- `findings/cliffordnet-blocks.md` — `CliffordNetBlock` (`layers/geometric/clifford_block.py:482`) is isotropic, no channel divisibility constraints, internal residual via GatedGeometricResidual + LayerScale + DropPath. `CliffordNetBlockDSv2` available for downsampling stages.
- `findings/train-conventions-and-lpips.md` — `src/train/` package conventions, dataclass-config pattern with `to_model_config()`, `create_base_argument_parser`, `create_callbacks`, output to repo-root `results/`. LPIPSLoss API documented.

### Key Constraints

### HARD
- Keras 3 / TF 2.18, Python 3.11+. `keras.ops` for backend-agnostic ops; `@keras.saving.register_keras_serializable()` on every custom class; full `get_config()` round-trip.
- Block I/O contract on the swap target: `(B, Hp, Wp, embed_dim) → (B, Hp, Wp, embed_dim)`, stride=1, full spatial resolution preserved (MAE mask token application + SIGReg reshape both require the complete `(Hp, Wp)` grid).
- **Residual semantics mismatch**: existing ConvNeXtV2 encoder loop adds the residual **externally** (`encoder.py:239-243`). `CliffordNetBlock` adds residual **internally** via `GatedGeometricResidual`. New encoder/decoder loop must NOT add an outer residual when using CliffordNetBlock — else double residual.
- Training outputs always go to repo-root `results/`, never `src/results/` (per `feedback_results_dir` memory).
- GPU jobs strictly serial (per `feedback_no_parallel_gpu` memory).
- Push with `--no-verify` (user pushes themselves).
- Centralized logger via `dl_techniques.utils.logger` — no print statements.

### SOFT
- New training package should mirror v2's directory layout (`__init__.py`, `callbacks.py`, `train_<name>.py`, optional README.md). Script name must NOT be `train.py` (package shadow).
- Dataclass config + `to_model_config()` bridge is the repo idiom.
- Re-use existing v2 callbacks where possible (`BetaAnnealingCallback`, `MaskedReconViz`, `TrainingCurvesCallback`).

### GHOST (worth checking)
- None identified — CliffordNetBlock has no channel-divisibility constraint (was a concern, ruled out by exploration).

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*

## plan_2026-05-27_4a444b14
### Iteration 0 (EXPLORE)

- F1 [HARD constraint] **V1 architecture, APIs, and conventions** — `findings/v1-architecture.md`
- F2 [SCAFFOLD] **Existing losses inventory + reuse map** — `findings/losses-inventory.md`
- F3 [DESIGN] **Multi-task head + MAE masking design notes** — `findings/v2-design.md`
- F4 [HARD constraint] **Repo conventions (training, serialization, no-EMA)** — `findings/repo-conventions.md`
- F5 [SCOPE] **iter-1 scope boundaries and deferred work** — `findings/scope-boundaries.md`

### Exploration Confidence
- Problem scope: **deep** (V1 fully characterized in the prior epistemic deconstruction; user's "recommended path" explicit from prior conversation)
- Solution space: **constrained** (V1 patterns + repo conventions narrow design; multi-task pretraining recipe is canonical)
- Risk visibility: **clear** (LESSONS.md captures every footgun for this codebase + V1 specifically)

### Key Constraints (HARD/SOFT/GHOST)
- **HARD**: `compile(loss=None)` + `add_loss` pattern (V1 contract — D-001 of V1)
- **HARD**: `jit_compile=False` (XLA breaks SIGReg)
- **HARD**: `Sampling` is stochastic — reload checks must use deterministic `mu`
- **HARD**: SIGReg call site MUST `× ops.cast(Hp*Wp, "float32")` for resolution-invariance
- **HARD**: AdamW WD only — no L2 kernel_regularizer (double-WD footgun)
- **HARD**: Resolution-agnostic invariant — no GAP, no learned PE, no Dense over flattened spatial
- **HARD**: Losses live in `src/dl_techniques/losses/` (user instruction)
- **HARD**: Custom train_step must explicitly create `loss_tracker` (Keras 3.8 contract)
- **GHOST**: EMA target encoder (V1 README rejects; VAE forbids identity, no symmetry to break)
- **GHOST**: Learned absolute positional embedding (kills resolution invariance)
- **SOFT**: training scripts use `train.common` utilities and follow patterns 1-5
- **SOFT**: Greenfield model packages land in +1000-1300 code-only LOC (predict accordingly)

## plan_2026-05-27_84f6180d
### Index

### F1 — CLAUDE.md inventory (17 files, 1 over 400 lines)
- Only `src/train/CLAUDE.md` (451 lines) exceeds the 400-line cap. All others are ≤185 lines.
- Largest others: train/ccnets (185), models/ccnets (157), models/ (105), dl_techniques/ (99), layers/ (95), CLAUDE.md root (84).

### F2 — Inflated count claims (root + dl_techniques + models + layers)
- Claim: "150+ models, 290+ layers". Actual: 75 model directories (excluding `__pycache__`), 231 layer `.py` files.
- Affected files: `CLAUDE.md`, `src/dl_techniques/CLAUDE.md`, `src/dl_techniques/models/CLAUDE.md`, `src/dl_techniques/layers/CLAUDE.md`.

### F3 — Root CLAUDE.md `src/results/` is wrong
- Tree comment in `CLAUDE.md` lists `src/results/`; actual path is repo-root `results/` (confirmed by `ls src/` and memory entry `feedback_results_dir`).

### F4 — models/CLAUDE.md missing entries
Disk has but doc omits: `burst_dp`, `gpt2`, `lewm`, `memory_bank`, `nam`, `video_jepa`, `vq_vae_rotation`, `wave_field_llm`.

### F5 — losses/CLAUDE.md missing entries and outdated count
Doc says "28+"; actual 33 `.py` modules. Missing in doc: `clifford_detection_loss.py`, `focal_causal_lm_loss.py`, `masked_causal_lm_loss.py`, `multi_task_loss.py`, `scaled_mse_loss.py`, `utilization_loss.py`.

### F6 — optimization/CLAUDE.md missing SGLD + duplicate sled line
Recent commits added `sgld_optimizer.py` (commits b23f769e, 9342eaec, 70deb5e9). Not in doc. Also lists `sled_supervision.py` twice. Public API block doesn't expose `Muon`/`SGLD`. `train_vision/` subpackage referenced is real.

### F7 — utils/CLAUDE.md missing modules
Disk modules missing in doc: `deep_supervision.py`, `drop_path.py`, `weight_transfer.py`, `yolo_decode.py`. (`weight_transfer.py` IS used heavily in train/CLAUDE.md depth section, so its omission from utils doc is a real gap.)

### F8 — datasets/CLAUDE.md drift
Lists `universal_dataset_loader.py` twice. Missing: `bdd100k_video.py`, `nlp.py`, `pusht_hdf5.py`, `synthetic_drone_video.py`.

### F9 — train/CLAUDE.md is 451 lines (over cap)
Content largely accurate vs codebase. Trimming candidates: (a) long benchmark file descriptions in §"Reference Documents" can be condensed (each file gets 4–6 lines now); (b) Pattern 5 depth section has a long Keras-3.8 `by_name` gotcha that could move into utils/CLAUDE.md or be condensed; (c) Pattern code blocks could lose some inline comments.

### F10 — Smaller CLAUDE.md files are accurate
analyzer, callbacks, constraints, initializers, metrics, regularizers, visualization, models/ccnets, train/ccnets all consistent with current code. No edits needed.

### Key Constraints

**HARD**
- 400-line cap (only train/CLAUDE.md violates).
- Must remain factually consistent with current codebase (file/module listings, count claims).

**SOFT**
- Existing prose style and section structure should be preserved where possible (these are reference docs).
- Don't aggressively expand small files past what's needed for accuracy.

**GHOST (suspected)**
- "150+ / 290+" numbers may reflect an older aspirational repo state and have been blindly inherited across docs. Either bring them in line with reality or drop the count entirely.

### Corrections
*None yet.*

## plan_2026-05-27_68c7fcd6
### Index
1. **F-001** Keras 3 optimizer template — `src/dl_techniques/optimization/muon_optimizer.py` is the canonical example: subclass `keras.optimizers.Optimizer`, `@keras.saving.register_keras_serializable()`, implement `build` / `update_step(gradient, variable, learning_rate)` / `get_config` / `from_config`. Base class handles `weight_decay`, `clipnorm`, `clipvalue`, `global_clipnorm`, LR schedules.
2. **F-002** Optimization package conventions (`src/dl_techniques/optimization/CLAUDE.md`): config-driven builders, centralized `dl_techniques.utils.logger`, mirror test layout in `tests/test_optimization/`. SGLD does not need to be wired into `optimizer_builder` (Muon isn't either).
3. **F-003** Test template (`tests/test_optimization/test_muon_optimizer.py`): class-based pytest, covers Instantiation / Build / Update / Serialization / Integration (model.fit + save/load) / EdgeCases.
4. **F-004** SGLD update formula (canonical, Welling & Teh 2011): `w_{t+1} = w_t − lr·∇L(w_t) + sqrt(2·lr)·ε`, `ε ~ N(0, I)`. Reference PyTorch snippet in prompt uses `sqrt(lr)` — incorrect vs canonical formula. We follow canonical and document.
5. **F-005** Random number generation in Keras 3: use `keras.random.SeedGenerator` + `keras.random.normal(shape, seed=self._seed_generator)` for graph-safe, reproducible, backend-agnostic noise.

### Key Constraints
- **HARD**: Keras 3 / TF 2.18, `@keras.saving.register_keras_serializable()`, full round-trip `get_config`/`from_config`, `keras.ops` only, `dl_techniques.utils.logger` not `print`.
- **HARD**: Tests under `tests/test_optimization/test_sgld_optimizer.py`, mirror Muon test classes; tolerance `1e-6`–`1e-7`.
- **SOFT**: Default `noise_scale=1.0` (canonical SGLD); allow scaling factor (Bayesian temperature).
- **SOFT**: SGLD is stateless — minimal `build()`.

### Corrections
*(none)*
