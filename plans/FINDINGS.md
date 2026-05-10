# Consolidated Findings
*Cross-plan findings archive. Entries merged from per-plan findings.md on close. Newest first.*

## plan_2026-05-10_bd098beb
### Index

| ID  | Topic | Detail |
|-----|-------|--------|
| F-001 | Inherited issue inventory (README #1-#14 + D-005) | inline below |
| F-002 | Real encoder availability (ViT vs DINOv2) | inline below |
| F-003 | Decoder upsampling design | inline below |
| F-004 | Semi-supervised pipeline + feature-alignment wiring | inline below |
| F-005 | Train-script + README sync impact | inline below |

### Key Constraints

### Hard
- Keras 3 / TF 2.18 idioms, full `get_config()` round-trip, `@keras.saving.register_keras_serializable()`, `keras.ops`, `dl_techniques.utils.logger` (no `print`).
- Single GPU, never parallel jobs. CPU smoke (`CUDA_VISIBLE_DEVICES=""`) preferred for verification.
- Per-step commit `[iter-N/step-M] desc`; user pushes themselves.
- Update both READMEs (model + train) when API changes.
- Pattern `mlm.py:309-343` is canonical Keras-3 train_step. Already applied — preserve `# DECISION plan_2026-05-10_44694bc9/D-003` anchor unless we materially change semantics.
- AdamW WD only — never combine with `kernel_regularizer=L2`.
- 10-Line Rule + Autonomy Leash + 3-Strike Rule.

### Soft
- Real encoder: prefer `dl_techniques.models.vit.ViT(include_top=False, pooling=None)` — `DINOv2VisionTransformer` requires three Functional inputs (image+masks+is_training), much harder to compose.
- DPTDecoder default activation: change `'sigmoid'` → `'linear'` so AffineInvariantLoss is usable.
- Semi-supervised: gate behind `enable_semi_supervised` flag (default OFF) for backward compatibility.
- Feature alignment: rewire `frozen_encoder` to be a `keras.models.clone_model(encoder)` weight-share teacher; expose `update_teacher_ema(decay)`. Default OFF.

### Ghost / out-of-scope
- True DINOv2 weight loading from HuggingFace — separate plan.
- Pseudo-label depth-on-unlabeled training (needs a real pretrained teacher) — separate plan.
- Multi-GPU / distributed.
- `make test` (full 1.5h suite).

### Exploration Confidence
- **Problem scope: deep** — read entire `model.py`, `components.py`, `__init__.py`, `strong_augmentation.py`, `affine_invariant_loss.py`, `feature_alignment_loss.py`, current train script, ViT and DINOv2 model surface, prior plan summary + findings + decisions.
- **Solution space: constrained** — fixes are mechanical for #5/#6/#7/#8/#10/#11/#13/#14 + D-005; substantive for #1 (real ViT encoder), #2/#3 (semi-sup + FAL wiring). Pattern: gate behind flags; default backward-compatible.
- **Risk visibility: clear** — main risks: (a) real ViT encoder produces sequence `(B,N,D)` while decoder expects 4D — must reshape + upsample; (b) frozen_encoder clone ordering vs build; (c) save/load round-trip with new sub-model topology. Mitigated by step ordering + per-step CPU smoke.

### F-001 — Inherited issue inventory

From model README Known Issues + plan_2026-05-10_44694bc9 D-005:

| # | Severity | Description | Fix shape |
|---|----------|-------------|-----------|
| 1 | HIGH | Placeholder Conv-BN-ReLU encoder, not real ViT/DINOv2 | Wire `dl_techniques.models.vit.ViT(include_top=False, pooling=None)`; reshape (B,N,D)→(B,h,w,D); placeholder kept behind `encoder_kind='conv'` |
| 2 | HIGH | Frozen teacher has independent random weights | `keras.models.clone_model(encoder)` + copy weights post-build; expose `update_teacher_ema(decay)` |
| 3 | HIGH | Semi-supervised pipeline unimplemented | `enable_semi_supervised` flag (default False). When True: `train_step` accepts `((x_lab, x_unlab), y_lab)`, computes labeled loss + FAL on unlabeled feats. Pseudo-label-depth deferred. |
| 4 | FIXED | Keras-3 train_step (prior plan) | preserve `# DECISION plan_2026-05-10_44694bc9/D-003` |
| 5 | LOW | tf.GradientTape | Acceptable; keep — Keras 3 supports TF backend tape. Note in README. |
| 6 | HIGH | DPTDecoder default sigmoid incompatible with AIL | Change default to `'linear'`. Update train script + README. |
| 7 | MEDIUM | Functional encoder built in `build()` is fragile under save/load | When encoder is ViT (declared in `__init__`), the issue disappears. |
| 8 | LOW | Dead `if X is not None` checks + `_create_fallback_decoder` | Remove. |
| 9 | LOW | StrongAugmentation cutmix per-batch + 3-channel hardcoded | Make channels dynamic; per-sample brightness/contrast factors. + D-005 fix. |
| 10 | MEDIUM | `encoder_type` validated but unused | Map `vit_s/vit_b/vit_l` → ViT scales `small/base/large`; `conv` = placeholder. |
| 11 | LOW | `input_shape` shadows Layer.input_shape | Rename to `image_shape` (back-compat alias retained). |
| 12 | MEDIUM | No tests | Add `tests/test_models/test_depth_anything/test_depth_anything.py`. |
| 13 | LOW | `frozen_encoder.trainable = trainable` after Functional construction | Replaced by clone_model approach. |
| 14 | LOW | `compile()` mutates dead state | Remove `self.depth_loss`/`self.feature_loss`; just `super().compile()`. |
| D-005 | HIGH | `keras.ops.random.uniform` doesn't exist in Keras 3.8 | Replace with `keras.random.uniform` in StrongAugmentation `_apply_color_jitter` and `_apply_cutmix`. |

### F-002 — Real encoder availability

`src/dl_techniques/models/vit/model.py` — `ViT(keras.Model)`:
- `ViT(input_shape=(384,384,3), scale='small'/'base'/'large', patch_size=16, include_top=False, pooling=None)` returns `(B, num_patches+1, embed_dim)`.
- Strip CLS: `x[:, 1:, :]` → `(B, num_patches, embed_dim)`.
- Reshape to spatial: `(B, h, w, embed_dim)` where `h=H//patch_size`, `w=W//patch_size` (statically known from constructor).

`DINOv2VisionTransformer` requires three Functional inputs (`[inputs, masks_input, is_training_input]`) — too invasive. Defer.

### F-003 — Decoder upsampling design

ViT encoder at patch_size=16 → encoder output is H/16 × W/16 (24×24 for 384×384). Need to upsample to full resolution.

Add `upsample_factor: int = 1` to `DPTDecoder.__init__`. With 4-stage `dims=[256,128,64,32]`, distribute upsampling across stages: each non-final conv block followed by 2× bilinear upsample → cumulative 16× (one per stage; last stage no upsample, output_conv at full res).

`compute_output_shape` updated to multiply h,w by upsample_factor.

DepthAnything passes `upsample_factor = image_size // encoder_stride`. For placeholder Conv encoder (stride 32 actually — initial stride-2 conv + 4 maxpools = 32×), placeholder mode passes 32; for ViT (patch_size=16), passes 16.

### F-004 — Semi-supervised + feature-alignment wiring

`train_step(data)`: detect `x = data[0]; y = data[1]`. If `x` is a 2-tuple `(x_lab, x_unlab)`, run semi-sup path; else single-batch labeled-only path (current behavior).

When `enable_semi_supervised` AND `use_feature_alignment`:
1. Forward labeled: `y_pred_lab = self(x_lab, training=True)`. Capture student feat via `feat_student = self.encoder(x_unlab, training=True)`.
2. Teacher: `feat_teacher = self.frozen_encoder(x_unlab, training=False)`. Stop-gradient on teacher path (frozen weights, no tape track).
3. Loss: `loss = w_lab * compute_loss(x=x_lab, y=y, y_pred=y_pred_lab) + w_feat * FeatureAlignmentLoss()(feat_teacher, feat_student)`.

`feature_alignment_loss` expects `(B, feature_dim)` shape. We pool `(B, h, w, D)` to `(B, D)` via global average pool BEFORE passing to FAL. (Per-token FAL would need broadcasting; pooled is the standard distillation form.)

`frozen_encoder` build: in `DepthAnything.build()`, after `self.encoder = ...`, when `use_feature_alignment`: `self.frozen_encoder = keras.models.clone_model(self.encoder); self.frozen_encoder.set_weights(self.encoder.get_weights()); self.frozen_encoder.trainable = False`.

`update_teacher_ema(decay=0.999)` method copies EMA weights from student → teacher. Future plan can wire a callback to invoke this each step.

Pseudo-label-depth-on-unlabeled-data path: deferred. Documented in README as residual.

### F-005 — Train script + README impact

`src/train/depth_anything/train_depth_anything.py`:
- `create_model()` adds `encoder_kind`, `output_activation='linear'` (now valid via DepthEstimationLoss + AIL both), `image_shape` (renamed), `enable_semi_supervised`.
- Add `--encoder-kind {real,placeholder}` (default `real`) and `--enable-semi-supervised` flag (default off).
- Loss: `DepthEstimationLoss` continues to work (linear output is fine; sigmoid was a constraint, removing it is a simplification).

Model README rewrite Known Issues:
- FIXED in this plan: #1, #2(weight-shared teacher), #3(infrastructure for semi-sup; FAL wired), #6, #7, #8, #10, #11, #13, #14, D-005, partial #9 (D-005 only — per-batch/3-channel cutmix is a separate refactor).
- STILL OPEN: #5 (LOW; documented), #2-deeper (EMA decay default + callback to invoke it), #3-deeper (pseudo-label depth on unlabeled), #9-deeper (per-sample cutmix/color factors).
- ADDED: #12 — tests added in this plan, so REMOVED from open list.

### Corrections
*None yet.*

## plan_2026-05-10_44694bc9
### Index

| ID | Topic | Detail |
|----|-------|--------|
| F-001 | depth_anything code review (model.py / components.py / __init__.py) | inline below |
| F-002 | Pattern-5 train reference (train_depth_estimation.py + helpers) | inline below |
| F-003 | Deliverables scope and constraints | inline below |

### Key Constraints

### Hard
- Goal = review + README + train PLAN scaffolding. Do NOT execute training; do NOT modify model.py/components.py source in this plan.
- README at `src/dl_techniques/models/depth_anything/README.md`.
- Train scaffolding at `src/train/depth_anything/` follows Pattern 5 (Depth Estimation) per `src/train/CLAUDE.md`.
- Keras 3 / TF 2.18 idioms: `@keras.saving.register_keras_serializable()`, `keras.ops`, `dl_techniques.utils.logger`, no print, full `get_config()` round-trip.
- `MPLBACKEND=Agg`, single GPU, `.venv/bin/python -m train.depth_anything.<script>`.
- Use `train.common.megadepth` (`discover_megadepth_pairs`, `MegaDepthDataset`) for data — verified present.
- Use `dl_techniques.metrics.depth_metrics` (`AbsRelMetric`, `DeltaThresholdMetric`, `SqRelMetric`, `RMSEMetric`, `RMSELogMetric`) — verified present.
- Use `dl_techniques.callbacks.depth_visualization` (`DepthPredictionGridCallback`, `DepthMetricsCurveCallback`) — verified present.
- `train.common.create_callbacks(monitor="val_loss", ...)` + append depth-specific.
- AdamW WD only — never combine with `kernel_regularizer=L2`.
- Commit prefix `[iter-N/step-M] description`. User pushes; commit locally only.

### Soft
- README must flag as known issues: placeholder encoder (not real DINOv2); semi-supervised pipeline described in module docstring is not implemented in `call()`/`train_step()`; sigmoid output activation is incompatible with affine-invariant loss; `compiled_loss`/`compiled_metrics` Keras-2 attributes will likely fail under Keras 3.8.
- Train script overrides decoder `output_activation='linear'` (or `softplus`) instead of default `sigmoid`.
- Expose `--init-from <pretrained.keras>` per Pattern 5 — pretrained encoder is necessary for usable depth quality (LESSONS / user memory). Without it, README must warn the run will not converge to publishable numbers.

### Ghost / out-of-scope
- Fixing the DepthAnything code (placeholder encoder, semi-supervised pipeline, sigmoid output, Keras-3 train_step API) — separate plan after this review.
- Implementing real DINOv2 — separate effort.
- Multi-GPU / distributed.
- `make test` / full training execution.

### Exploration Confidence
- **Problem scope: deep** — all 3 depth_anything files read in full; every imported dep (`StrongAugmentation`, `AffineInvariantLoss`, `FeatureAlignmentLoss`) read; Pattern-5 reference (`train_depth_estimation.py` head) inspected; `train/CLAUDE.md`, `dl_techniques/CLAUDE.md`, `models/CLAUDE.md`, `losses/CLAUDE.md`, `layers/CLAUDE.md` re-read.
- **Solution space: constrained** — README format conventional; train scaffolding mirrors Pattern 5 1:1.
- **Risk visibility: clear** — main risk is overstating fixes (user wants review-only + plan); README must flag gaps without rewriting code. Second risk is sigmoid×affine-invariant mismatch; mitigated by overriding `output_activation='linear'` in trainer.

### F-001 — Code review of depth_anything

### Files
- `__init__.py` — empty (1 line). Per `models/__init__.py` convention this is OK; but per-package `__init__.py` for a model usually re-exports the public API. Currently consumers must import from `.model` and `.components` directly. Minor gap.
- `components.py` — `DPTDecoder` (keras.Layer). Self-contained, serializable. Implements only a final conv head — no multi-scale fusion or upsampling. Docstring is honest about this. NOT a true DPT decoder.
- `model.py` — `DepthAnything` (keras.Model) + `create_depth_anything` factory.

### Bugs / gaps (verified)
1. **Placeholder encoder, not DINOv2.** `_create_encoder()` builds a small Conv-BN-ReLU stack. Module docstring "Inherits semantic priors from pre-trained encoders" is unfulfilled. (HIGH)
2. **Frozen "teacher" encoder shares no weights with student.** `self.frozen_encoder = self._create_encoder(trainable=False)` builds an *independent* Functional model with fresh random weights. FeatureAlignmentLoss against random features is meaningless. (HIGH)
3. **Semi-supervised pipeline implied by docstring is NOT implemented.** `call()` accepts `(x_labeled, x_unlabeled)` but uses only `x_labeled`. `train_step()` doesn't use the unlabeled half, the `feature` loss term, the `frozen_encoder`, or any consistency loss. The `loss_weights={'labeled','unlabeled','feature'}` dict is dead state. (HIGH)
4. **`self.compiled_loss` / `self.compiled_metrics` are removed/deprecated in Keras 3.** Canonical Keras-3 pattern: `compute_loss(y, y_pred)` + iterate `self.metrics`. Calling `self.compiled_loss(y, y_pred)` and `self.compiled_metrics.update_state(...)` will raise `AttributeError` on Keras 3.8+. **`model.fit` will crash.** (HIGH — blocks any training)
5. **`tf.GradientTape` instead of relying on Keras default `train_step` or `keras.ops`.** Couples to TF backend; minor convention violation. (LOW)
6. **`DPTDecoder.output_activation='sigmoid'` is incompatible with `AffineInvariantLoss`.** Sigmoid clamps prediction to [0,1] making global scale ill-defined; AIL specifically expects unbounded scale to median-shift / MAD-normalize meaningfully. Standard depth output: `linear` or `softplus`/`exp`. (HIGH for training quality)
7. **Functional encoder built inside `build()` is fragile under serialization.** Convention in dl_techniques: declare sub-layers in `__init__`. Save/load round-trip not tested. (MEDIUM)
8. **Dead defensive checks**: `if DPTDecoder is not None`, `if StrongAugmentation is not None`, `if AffineInvariantLoss is not None`/`FeatureAlignmentLoss is not None` — all are class objects imported at top of file, never None. `_create_fallback_decoder` is dead code. (LOW)
9. **`StrongAugmentation._apply_cutmix` hard-codes 3 channels** (`ops.tile(mask, [1, 1, 3])`); brightness/contrast factors are batch-scalar (single value applied to whole batch), not per-sample — much weaker than typical CutMix. (LOW; pre-existing layer)
10. **`encoder_type` in {'vit_s','vit_b','vit_l'} validated but never used.** Placeholder encoder is identical for all three values. Misleading API. (MEDIUM)
11. **`input_shape` parameter shadows Keras `Layer.input_shape`.** Mitigated as `self.input_shape_param`. `get_config` exports as `"input_shape"` so `from_config(**config)` round-trips. Error-prone naming. (LOW)
12. **No tests** under `tests/test_models/test_depth_anything/` — verified absent.
13. **`encoder.trainable = trainable`** set after Functional construction. Works, but combined with bug 2 above, the frozen "teacher" never gets a chance to be a real teacher.
14. **`compile()` override silently mutates `self.loss_weights`** and stores `self.depth_loss` / `self.feature_loss` that are then never read by `train_step` — dead state. (LOW)

### Things that ARE OK (no false positives)
- `get_config` correctly serializes `kernel_initializer` / `kernel_regularizer` via `keras.initializers.serialize` / `keras.regularizers.serialize`; `__init__` accepts both string and dict (`keras.initializers.get(...)` handles both). Round-trip OK for these.
- `@keras.saving.register_keras_serializable()` on `DepthAnything` and `DPTDecoder` — both decorated correctly.
- `from_config(cls, config) -> cls(**config)` — fine because `keras.Model.__init__` accepts `**kwargs` (so `name`, `trainable`, `dtype` from base `get_config()` are absorbed).
- `DPTDecoder` provides `get_build_config` / `build_from_config` — correct Keras 3 idiom for layers built from input shape.
- `DPTDecoder.compute_output_shape` is correct.

### F-002 — Pattern-5 train reference

`src/train/cliffordnet/train_depth_estimation.py` is the canonical reference:
- `DepthTrainingConfig` dataclass: `megadepth_root`, `train_split`, `patch_size`, `min_valid_ratio`, `max_train_files`, `max_val_files`, `dataset_shuffle_buffer`, `model_variant`, `batch_size`, `epochs`, `patches_per_image`, `augment_data`, `steps_per_epoch`, `learning_rate`, `optimizer_type`, `lr_schedule_type`, `warmup_epochs`, `weight_decay`, `gradient_clipping`, `enable_deep_supervision`, `monitor_every_n_epochs`, `early_stopping_patience`, `validation_steps`, `output_dir`, `experiment_name`, `init_from`, `seed`.
- `DepthEstimationLoss(keras.losses.Loss)` — masked L1 + multi-scale gradient matching, **locally defined**.
- Data: `discover_megadepth_pairs(megadepth_root)` → `(rgb_paths, depth_paths)` → `MegaDepthDataset(rgb, depth, batch_size, patch_size, ...)`. Yields `(rgb, y_true)` with `y_true = concat([depth, mask], axis=-1)`.
- Optimizer/LR: `optimizer_builder` + `learning_rate_schedule_builder` from `dl_techniques.optimization`.
- Callbacks: `create_common_callbacks(monitor="val_loss", patience, use_lr_schedule=True, include_terminate_on_nan=True, include_analyzer=False)` + append `DepthMetricsCurveCallback` + `DepthPredictionGridCallback`.
- Pretrained init: `--init-from <ckpt.keras>` → `load_weights_from_checkpoint(model, ckpt, skip_prefixes=("head_",))`.
- Reproducibility: `--seed` seeds Python/NumPy/TF/Keras at startup.

### Verified present
- `src/train/common/megadepth.py`
- `src/dl_techniques/metrics/depth_metrics.py`
- `src/dl_techniques/callbacks/depth_visualization.py`
- `src/dl_techniques/utils/weight_transfer.py` (`load_weights_from_checkpoint`)

### F-003 — Deliverables

### Deliverable 1: README
Path: `src/dl_techniques/models/depth_anything/README.md`

Sections: Overview, Architecture (encoder placeholder, decoder, augmentation, frozen teacher), Components (`DepthAnything`, `DPTDecoder`, `StrongAugmentation`), Usage (forward pass, factory, compile), Configuration parameters, Example, **Known Issues / Caveats** (bullet list mapping to F-001 #1..#14), References.

### Deliverable 2: Train scaffold
Directory: `src/train/depth_anything/`

Files:
- `__init__.py` — empty (per train-package convention).
- `train_depth_anything.py` — Pattern-5 trainer mirroring `cliffordnet/train_depth_estimation.py`, adapted to call `create_depth_anything(...)`. Override decoder `output_activation='linear'` at construction (constructor exposure or post-hoc `model.decoder.output_activation = 'linear'` reset would not propagate; recommended path: construct `DepthAnything` then replace `model.decoder.output_conv.activation = keras.activations.linear` BEFORE first forward pass — but this is fragile). Cleaner path: file local helper `build_depth_anything_for_training(args)` that constructs the model and a *new* `DPTDecoder(output_activation='linear', ...)` directly, swapping the decoder. Will document in plan.
- README in `src/train/depth_anything/` — optional; add a short usage block + smoke recipe.

### Compile path
Two reasonable choices, both honest:
1. Use `dl_techniques.losses.AffineInvariantLoss` — matches DepthAnything's docstring claim; requires linear output. NOTE: AIL ignores the validity mask channel — it processes the full 4D tensor as flat. We must either (a) slice off the mask before passing to AIL, or (b) use a wrapper that applies AIL only on `y_true[..., :1]` with `y_true[..., 1:]` as mask. (a) is simpler in trainer code.
2. Use the proven `DepthEstimationLoss` from `cliffordnet/train_depth_estimation.py` (copy-paste local; mask-aware L1 + gradient matching). More predictable; less faithful to the paper recipe.

Plan PROPOSAL: option (1) with a tiny `MaskedAffineInvariantLoss` wrapper inside the train script that splits `y_true` into `(depth, mask)`, masks invalid pixels in both `y_true` and `y_pred`, then calls AIL on the masked tensors. ~25 LOC.

### CLI flags (mirror Pattern 5 + DepthAnything-specific)
`--encoder-type {vit_s,vit_b,vit_l}`, `--input-size`, `--decoder-dims`, `--megadepth-root`, `--patch-size`, `--batch-size`, `--epochs`, `--learning-rate`, `--lr-schedule`, `--weight-decay`, `--gradient-clipping`, `--patience`, `--steps-per-epoch`, `--validation-steps`, `--max-train-files`, `--max-val-files`, `--init-from`, `--seed`, `--gpu`, `--output-dir`, `--experiment-name`, `--use-feature-alignment` (off by default until placeholder-vs-pretrained encoder is resolved), `--cutmix-prob`, `--color-jitter-strength`, `--show-plots`.

### Corrections
*None yet.*

## plan_2026-05-09_0f39a086
### Index

| ID | Topic | File |
|----|-------|------|
| F-001 | Audit summary — 6 bugs / 6 robustness / 5 design / 9 opportunities; per-finding fix shape | findings/01-audit-summary.md |
| F-002 | Importers of `memory_bank` — test scope | findings/02-importers.md |
| F-003 | Design notes for non-trivial fixes (B1 W_Q-projected KMeans, B2 InfoNCE redesign, R1 cond, O4 plumbing, O7 schedule, D2 constants) | findings/03-design-notes.md |

### Key Constraints

### Hard
- All 32 existing tests under `tests/test_models/test_memory_bank/` must pass.
- Variable-name prefixes (`memory_*`, `gate_*`) are load-bearing for the dual-optimizer split — don't rename existing weights.
- `keras.ops` / `keras.random` only (no `import tensorflow as _tf` for randomness — B4/O5).
- `@keras.saving.register_keras_serializable()` + full `get_config()` round-trip on every Layer/Model.
- `dl_techniques.utils.logger` (no `print`).
- User pushes commits themselves; we commit locally only.
- Single GPU; never run training in parallel (project memory).
- Tests scoped to `tests/test_models/test_memory_bank/` — full suite is ~1.5h, do not run.

### Soft
- O1 (KV-cache for incremental decode) is a stretch goal — skip if it requires touching `WaveFieldDecoderBlock` (audit says so, mark scope-creep).
- O4 (multi_head_keys) defaults to False everywhere to preserve current behavior.
- B2 (InfoNCE) and O4 (multi_head_keys) are the largest fixes — own steps.

### Ghost / out-of-scope
- Modifying `WaveFieldDecoderBlock` or `WaveFieldAttention` (62 tests lock the latter).
- Pushing to remote.
- Running `make test` (full suite).

### Exploration Confidence
- **Problem scope: deep** — audit doc covers every finding with file:line references and prescribed fix shape; all source files read in full; existing tests catalogued.
- **Solution space: constrained** — most fixes are one-shot; B2/R1/O4/O7 design choices resolved in F-003.
- **Risk visibility: clear** — main risks identified (B2 InfoNCE redesign, R1 cond aux-loss interaction, R3 prefix-match reroute) and mitigations in F-003.

Ready for PLAN.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-08_146ae899
### Index

| ID | Topic | File |
|----|-------|------|
| F-001 | `src/train/wave_field_llm/pretrain.py` Pattern-3 NLP CLM reference (skeleton, invariants, helpers) | findings/01-pretrain-py-reference.md |
| F-002 | `WaveFieldLLM` model surface — extension shape for dual-tap memory | findings/02-wave-field-llm-model.md |
| F-003 | Existing memory/routing layers in `dl_techniques` — none reusable as-is; reference patterns extracted | findings/03-existing-memory-and-routing-layers.md |
| F-004 | Blueprint mapping to concrete code (top-K STE, gate init b=-3, 4 aux losses, 4-phase curriculum, custom train_step) | findings/04-blueprint-decomposition.md |
| F-005 | Scope clarification + recommended file layout (training script + new wave_field_llm_memory model package) | findings/05-scope-and-file-layout.md |

### Key Constraints

### Hard
- Output dict key MUST be `"logits"` (SYSTEM atlas invariant).
- `prepare_dict_keyed_compile(model, output_key="logits")` must run BEFORE `model.compile`.
- `metrics={"logits": build_clm_metrics(config.encoding_name)}` — Pattern-3 metric floor.
- `@keras.saving.register_keras_serializable()` + full `get_config()` round-trip on every new Layer/Model.
- Static `k` int for `ops.top_k`; static `num_classes` for `ops.one_hot` (set as `M_static = S_lt + max_seq_len`).
- `d_v < embed_dim` (bottleneck enforced); `d_k != d_v` (blueprint).
- No positional encodings on memory queries/keys.
- `b_g` init = -3.0 (so sigmoid≈0.04, memory ~96% bypassed at init).
- `K_lt` initialized via offline K-Means (`sklearn.cluster.MiniBatchKMeans`) on warmup hidden states, applied via `K_lt.assign(centroids)` at start of Phase 2.
- Frozen state in layers via `add_weight(trainable=False, initializer=...)` (LESSONS — Keras FuncGraph dead-tensor bug).
- STE: `routing = soft + ops.stop_gradient(hard - soft)` (canonical idiom from vector_quantizer.py).
- Strict causal mask for M_WM retrieval (position t may only attend to WM tokens at positions <= t).
- Sibling-stack pattern: NO modification of existing `WaveFieldLLM` / `WaveFieldAttention` / `WaveFieldDecoderBlock` (mirrors plan_2026-05-07_1519e34f D-001).
- Custom `train_step` for differential LRs (backbone 1e-5, memory 3e-4) using two AdamW optimizers, gradient split by var name prefix.
- `compiled_metrics.update_state(y, y_pred)` must be called inside the custom `train_step`.
- `load_model` `custom_objects` must list both losses + every new custom Layer/Model class.
- Single GPU only; `MPLBACKEND=Agg`; `dl_techniques.utils.logger` (no `print`).
- Required CLM CLI flags: `--steps-per-epoch`, `--seed`, `--min-article-length`, `--shuffle-shards`, `--resume`.
- AdamW WD only — no `kernel_regularizer=L2`.
- Test scope: pytest only on touched modules.

### Soft
- Memory hyperparameters scale with variant: tiny=(d_k=64, d_v=128, S_lt=4096); small=(128, 256, 16384); medium=(128, 512, 32768); large/xl=(128, 512, 65536).
- Default top-k retrieval = 32.
- Default aux loss weights: λ_gate_entropy=1e-3, λ_load_balance=1e-2, λ_z_loss=1e-3, λ_diversity=1e-3, λ_infonce=5e-3.
- Key diversity penalty subsamples 1024 keys per step (full S_lt×S_lt is too large at S_lt=65536).
- InfoNCE uses 256 random negatives per step.
- Default phase boundaries (steps): phase1=50000, phase2=25000, phase3=100000.
- Variant ladder mirrors `WaveFieldLLM.MODEL_VARIANTS`: tiny/small/medium/large/xl.

### Ghost / out-of-scope
- "Reuse `KMeansLayer` for K_lt online updates" — false; offline `sklearn.MiniBatchKMeans` matches blueprint better.
- "Reuse `KeyValueMemoryStore` from `experimental/contextual_memory.py`" — false; same K/V dim, no top-K, no aux losses.
- "Modify `WaveFieldAttention` to accept memory queries" — false; locked by 62 tests.
- "Single-file deliverable strict reading" — contradicts project convention; recommended scope is small new model package + training script (S2 in F-005).
- Phase-4 long-context dataset construction — separate plan.
- Distributed / multi-GPU — out of scope per project memory.

### Exploration Confidence

- **Problem scope: deep** — `pretrain.py` (887 LOC), `wave_field_llm.py` (582 LOC), `train.common.nlp` API, sibling memory layers, and 5 prior related plans all read in detail.
- **Solution space: constrained** — Pattern-3 NLP CLM trainer template is locked by 6 existing trainers; the memory architecture is fully specified by the user-supplied blueprint. Open knobs: (a) scope S1 single-file vs S2 idiomatic split, (b) curriculum control via custom `train_step` vs callback-only.
- **Risk visibility: clear** — primary risks identified: custom `train_step` interaction with `compiled_metrics` + `prepare_dict_keyed_compile`; save/load round-trip with phase counter; K-Means warmup cost at large S_lt (mitigated by MiniBatchKMeans + once-only at phase boundary); `ops.one_hot` static `num_classes` (mitigated by right-padding WM to `max_seq_len`).

Ready for PLAN.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-07_824e5687
### Index
*To be populated during EXPLORE.*

### Key Constraints
*To be populated during EXPLORE.*

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-07_3f461682
### Index
- [F-001 LLM trainer inventory & current metric setup](plan_2026-05-07_3f461682/findings/01-llm-trainer-inventory.md) — 5 in-scope Pattern-3 CLM trainers (gpt2 pretrain/finetune, cliffordnet_nlp, _unet, _routing, wave_field_llm) all share `metrics={"logits": ["accuracy"]}`. qwen/nano_vlm/blt/bert/fnet are out of scope (custom trainers / different objectives).
- [F-002 Existing metrics infrastructure](plan_2026-05-07_3f461682/findings/02-existing-metrics-infra.md) — `dl_techniques.metrics.perplexity_metric.Perplexity` already exists, drop-in compatible. No BPC/BPW/BLEU exists. `_post_generate_hook` is an empty extension point on every probe.
- [F-003 Integration shape](plan_2026-05-07_3f461682/findings/03-integration-shape.md) — DRY recipe: one `dl_techniques/metrics/llm_metrics.py` module + one `train/common/nlp.py::build_clm_metrics` helper. Per-trainer delta = ~3-5 LOC. Optional Self-BLEU/distinct-N/aggregate-tok/s via `_post_generate_hook` override.

### Key Constraints

### Hard
- **DRY single-source-of-truth**: one shared metric module + one builder helper; no metric math copy-pasted into the 5 trainers (user explicit: "make good use of the common").
- **Reuse `Perplexity` from `perplexity_metric.py`** — do not re-implement.
- **Output dict key MUST stay `"logits"`** (SYSTEM atlas invariant; loss + train wrappers + probe pivot on it).
- **Keras 3 idioms**: `@keras.saving.register_keras_serializable()`, `keras.ops`, `dl_techniques.utils.logger`, full `get_config()` round-trip.
- **`ignore_class`** for PPL must mirror each trainer's loss `ignore_index`. Default for `MaskedCausalLMLoss` is `-1`; packed-CLM pipeline does not actually emit ignore tokens, so this is a no-op in practice but should be wired for symmetry / correctness if labels ever change.
- **Mixed precision**: keep PPL accumulator fp32 (default `add_weight` dtype). AMP isn't currently enabled in CLM training, but write metric to be AMP-safe.
- **No full-training smoke**. Verify by unit tests + (only on user request) a few-step `model.fit` smoke.
- **No emojis, no print, single GPU, MPLBACKEND=Agg, .venv/bin/python** (project conventions).

### Soft
- BPC `chars_per_token` defaults to **4.0** for `gpt2` encoding on EN-Wikipedia (paper-ish constant). Configurable via `build_clm_metrics`.
- BPW `tokens_per_word` ~1.3. Less commonly reported. Recommend including but flagged optional via flag (or simply use BPT + BPC and skip BPW).
- Self-BLEU @ n=4 across probe outputs; distinct-2; aggregate tok/s. NLTK-free pure-Python implementations.
- Defer probe-class extraction (5x duplication) to a separate refactor plan — outside scope here.

### Ghost / out-of-scope
- True BLEU/ROUGE (need references) → offline eval harness only.
- Hallucination, Toxicity, Coherence → LLM-as-judge / classifiers, separate harness.
- Inference Latency / VRAM as live training-loop metrics → recommend deferring or once-per-epoch logger only.
- qwen, nano_vlm, blt, bert/fnet (MLM), CLIP — different shapes, defer.

### Exploration Confidence
- **Problem scope: deep** — every CLM trainer's `compile_model` and probe class read; existing metrics package mapped; tokenizer access confirmed; `ignore_index` semantics traced through `MaskedCausalLMLoss` and the packed-CLM pipeline.
- **Solution space: constrained** — DRY single-module + single-helper is the only design that satisfies the user's "make good use of the common" instruction without per-model duplication.
- **Risk visibility: clear** — main risk is `ignore_class` mismatch (mitigated by reading each trainer's loss config) + AMP-safety of accumulator (existing `Perplexity` is correct).

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

- [CORRECTED iter-1] F-002 (`findings/02-existing-metrics-infra.md` line 38): "All five GenerationProbeCallback classes implement an empty `_post_generate_hook`" — actually only `gpt2/pretrain.py` and `wave_field_llm/pretrain.py` do. The 3 cliffordnet probes lack the extension point. See D-002 in decisions.md for resolution (add the extension to the 3 cliffordnet probes during EXECUTE).

## plan_2026-05-07_08aaf818
### Index
- `findings/reference-fix-shape.md` — exact code shape of the 1fe2088 fix; explains both halves.
- `findings/call-sites.md` — inventory of decode sites; confirms scope incl. routing-variant outlier and out-of-scope verification.
- `findings/per-file-patch-shape.md` — per-file insertion line numbers and routing-specific tuple-return shape.

### Key Constraints
- **Hard**: behaviour parity with 1fe2088 (mask + try/except). Each patch mirrors the reference exactly modulo the routing tuple return.
- **Hard**: `dl_techniques.utils.logger` only (no print).
- **Hard**: 4 reserved specials (50257..50260) baked in (`power_sampling.py:86` confirms). `tiktoken.get_encoding("gpt2").n_vocab == 50257`. Reference's `range(n_vocab, max(n_vocab+1, 50261))` covers exactly these four.
- **Hard**: routing returns `Tuple[str, int]` — both branches of try/except must keep this shape.
- **Soft**: one commit per file. Verify each with `python -m py_compile`; no smoke training (too expensive).

### Out of scope (verified)
- `src/dl_techniques/models/nam/tokenizer.py` — custom 21-symbol tokenizer, not tiktoken.
- `src/dl_techniques/models/cliffordnet/power_sampling.py:625` — pre-sample mask at line 207-209 makes bad ids impossible.
- `src/train/wave_field_llm/pretrain.py` — already fixed (1fe2088).

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-07_1519e34f
### Index

- [F-001 gpt2.py architecture & attention wiring](plan_2026-05-07_1519e34f/findings/01-gpt2-architecture.md) — `GPT2(keras.Model)` wraps `TextDecoder` which wraps `TransformerLayer` stack with `create_attention_layer` factory. WaveFieldAttention is NOT in the factory; mask shape is (B,N) not (B,N,N); QKV+gate+output projections are internal. Conclusion: build a parallel decoder stack rather than retrofit `TextDecoder`.
- [F-002 src/train/gpt2/pretrain.py conventions](plan_2026-05-07_1519e34f/findings/02-train-gpt2-conventions.md) — Pattern 3 NLP CLM training script, ~95% generic. `TrainingConfig` dataclass + `StepCheckpointCallback` + `GenerationProbeCallback` + AdamW/warmup-cosine + tiktoken/Wikipedia + `MaskedCausalLMLoss`. Mirror file-by-file, swap model class and variant list.
- [F-003 WaveFieldAttention call signature & integration](plan_2026-05-07_1519e34f/findings/03-wave-field-attention-integration.md) — `(B,N,D)` in/out, optional `(B,N)` padding mask, causal-by-construction, internal QKV+gate+output, FFT in fp32 under AMP. New hyperparameter `field_size` (default `2*max_seq_len`). Trainable-var count == 10 is locked by tests.

### Key Constraints

### Hard
- `WaveFieldAttention` API is locked (62 tests). Do NOT modify the layer.
- Mask shape mismatch: WaveField uses `(B,N)`; standard `TransformerLayer` passes `(B,N,N)`. Cannot reuse `TextDecoder`/`TransformerLayer` without invasive changes that would touch unrelated attention types.
- Output dict key MUST be `"logits"` — train wrappers, `MaskedCausalLMLoss`, and `model.compile(loss={"logits": ...})` all key on it.
- Keras 3.8 / TF 2.18 idioms: `@keras.saving.register_keras_serializable()`, `keras.ops`, `dl_techniques.utils.logger` (no `print`), full `get_config()` round-trip.
- Causality: WaveFieldAttention is causal by construction. No additional causal mask needed. Padding mask only.
- `dim % num_heads == 0`. `field_size > 1`. `max_seq_len > 0`.
- Tokens beyond `max_seq_len` alias to the last field cell — warn-only. So `max_seq_len` of the layer MUST equal the model's `max_seq_len`.
- AdamW WD only — no `kernel_regularizer=L2`.
- Single GPU jobs. `MPLBACKEND=Agg`. `.venv`.
- Direct import path: `from dl_techniques.models.wave_field_llm.wave_field_llm import WaveFieldLLM`.
- Required CLM CLI flags: `--steps-per-epoch`, `--seed`, `--min-article-length`, `--shuffle-shards`, `--resume` (D-001..D-006 from plan_2026-05-07_c6dd7cc1).
- Resume seed shift: `data_seed = config.seed + initial_step`.
- `custom_objects` on `keras.models.load_model` must include both losses; `WaveFieldAttention`+`_IdentityPlusNoise` are auto-registered via decorator.
- Mirror gpt2.py variant ladder: `tiny / small / medium / large / xl`.
- Test scope: pytest only on the new module. Do NOT run `make test`.

### Soft
- `field_size = 2 * max_seq_len` default. Document trade-off (memory vs accuracy).
- GELU FFN with 4x expansion (matches GPT-2 reference).
- Pre-norm + residual structure.
- Variants: copy gpt2.py MODEL_VARIANTS 1:1, add `field_size` per-variant.
- Tests: mirror `tests/test_models/test_gpt2/test_gpt2.py`. Add `keras.Model`-wrapped save/load round-trip (LESSONS L48-49).

### Ghost (not present)
- "Register WaveFieldAttention in attention factory" — invasive. Defer to a separate plan; build self-contained stack instead.
- "Custom causal mask layer" — false; kernel is causal.
- "3-D attend mask" — false; WaveFieldAttention takes (B,N).

### Exploration Confidence

- **Problem scope**: deep — every reference file read in full (gpt2.py 417 LOC, wave_field_attention.py 591 LOC, pretrain.py 945 LOC, text_decoder.py 475 LOC). Mask shape mismatch confirmed.
- **Solution space**: constrained — single viable composition (sibling stack with WaveFieldAttention block). FFN type, variant ladder, tie/untie are 3 minor knobs.
- **Risk visibility**: clear — main risks: padding-mask under variable seq_len, max_seq_len mismatch, save/load round-trip with wave-kernel weights. All mitigated by prior-plan patterns.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*
