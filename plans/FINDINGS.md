# Consolidated Findings
*Cross-plan findings archive. Entries merged from per-plan findings.md on close. Newest first.*

## plan_2026-04-21_421088a1
### User-Provided Context (seed)

### Target
Drone video streaming (mixed altitude), ~30 FPS RTX 4070 12GB. Patch-level anomaly/surprise
detection + SSL backbone. Telemetry (IMU Δr/p/y, GPS velocity, altitude) via AdaLN-zero.

### LeWM assets (plan_2026-04-21_8416bc0b)
- `src/dl_techniques/models/lewm/` — global-CLS JEPA.
- `src/dl_techniques/layers/adaln_zero.py` — AdaLN-zero (identity-at-init).
- `src/dl_techniques/regularizers/sigreg.py` — SIGReg (no EMA needed).

### Clifford primitives (`src/dl_techniques/layers/geometric/clifford_block.py`)
- `SparseRollingGeometricProduct`, `GatedGeometricResidual` (LayerScale).
- `CliffordNetBlock` — 4D `(B,H,W,D)` dual-stream.
- `CausalCliffordNetBlock` — 4D `(B,1,T,D)` left-padded causal.

### Open design decisions (must present in PLAN)
D-001 encoder (ViT vs Clifford vs hybrid); D-002 predictor 5D handling (factorized/flatten/3D);
D-003 target (tube-mask / next-frame / both); D-004 conditioning (where, per-frame vs global);
D-005 SIGReg placement (per-patch vs pooled); D-006 positional; D-007 streaming contract.

### Scope
IN: `models/video_jepa/`, synthetic drone dataset, training smoke test, tests
(shape/causality/serialization/AdaLN-id/SIGReg-finite), streaming API.
OUT: real drone data, surprise viz, downstream heads, full-scale training.

### Ops constraints
`.venv/bin/python`, `CUDA_VISIBLE_DEVICES=1`, `MPLBACKEND=Agg`, never parallel GPU.
Commit per step. Verify hardest case (causality + SIGReg stability) first.

### Index

1. **[clifford-primitives.md](plan_2026-04-21_421088a1/findings/clifford-primitives.md)** — Full contracts of
   `SparseRollingGeometricProduct`, `GatedGeometricResidual`, `CliffordNetBlock`
   (4D `(B,H,W,D)`), `CausalCliffordNetBlock` (4D `(B,1,T,D)` with left-pad & causal
   cumsum mean). Factorized 5D handling is the clean path. BatchNorm-batch-of-1
   risk noted. Identity-at-init possible via LayerScale γ=1e-5.
2. **[lewm-reusable-assets.md](plan_2026-04-21_421088a1/findings/lewm-reusable-assets.md)** —
   `AdaLNZeroConditionalBlock`: `inputs=[x,c]` (B,T,D)(B,T,D), zero-init ⇒
   identity-at-init. `SIGRegLayer`: `(..., N, D)` averages over axis=-3, 3 placement
   options for 5D latents. `LeWM.rollout` is the exact streaming precedent.
   `encode_pixels` pattern (reshape B*T) lifts to patch-level with `pooling=None`.
3. **[positional-and-infrastructure.md](plan_2026-04-21_421088a1/findings/positional-and-infrastructure.md)** —
   `PatchEmbedding2D` (Conv2D) + `PositionEmbeddingSine2D` (channels-first!) +
   `ContinuousSinCosEmbed` for telemetry. Training template copied from
   `src/train/lewm/train_lewm.py`. `JEPAMaskingStrategy` exists for images but not
   temporal tubes — small extension if D-003 = tube masking.

### Key Constraints

### Hard
- **Clifford blocks BatchNormalization** inside context stream — unsafe at batch-of-1.
  Smoke tests must keep B ≥ 2; fallback is swap BN→LayerNorm (structural change,
  noted as risk).
- **`CausalCliffordNetBlock` requires H=1** input. Any 5D handling must reshape
  `(B, H_p, W_p, T, D)` → `(B*H_p*W_p, 1, T, D)` for the temporal pass.
- **`AdaLNZeroConditionalBlock` expects `inputs=[x,c]`** (list, length 2). Keras
  serialization of dict inputs has trapped prior plans — use list.
- **`SIGRegLayer` input convention `(..., N, D)`** — averaging over axis=-3.
  Upstream LeWM passes `(T, B, D)`. For 5D latents we must choose a reshape and
  document it (D-005).
- **Keras 3.8 serialization rules** (LESSONS): every custom class
  `@keras.saving.register_keras_serializable()` + complete `get_config()`,
  sublayers as explicit attrs (not dict).
- **GPU policy**: `CUDA_VISIBLE_DEVICES=1` (RTX 4070 12 GB), never parallel, `MPLBACKEND=Agg`.
- **Causality test = hardest case**: perturbation at temporal position k must not
  alter outputs at positions < k. This invariant MUST be tested first.

### Soft
- ViT-tiny (192d) is the default carryover from LeWM — but a hybrid encoder
  (PatchEmbed + 2-4 Clifford blocks) is parameter-cheap and better for drone
  small-object sensitivity.
- Smoke-scale defaults mirror LeWM: `num_proj=64`, small depth, B=2, short T.
- Commit cadence `[iter-N/step-M] desc`; user pushes.

### Ghost / deferred
- Real drone dataset — not in scope, synthetic only.
- Downstream fine-tuning heads (tracking, detection) — follow-up plan.
- Patch-level surprise visualization — follow-up.
- Scaling beyond smoke — follow-up.

### Key reusable components
- `src/dl_techniques/layers/geometric/clifford_block.py` — `CliffordNetBlock`,
  `CausalCliffordNetBlock`, `SparseRollingGeometricProduct`, `GatedGeometricResidual`.
- `src/dl_techniques/layers/adaln_zero.py` — `AdaLNZeroConditionalBlock`.
- `src/dl_techniques/regularizers/sigreg.py` — `SIGRegLayer`.
- `src/dl_techniques/layers/embedding/patch_embedding.py` — `PatchEmbedding2D`.
- `src/dl_techniques/layers/embedding/positional_embedding_sine_2d.py` — fixed 2D PE.
- `src/dl_techniques/layers/embedding/continuous_sin_cos_embedding.py` — telemetry PE.
- `src/dl_techniques/models/lewm/model.py` — `encode_pixels`, `rollout` patterns.
- `src/dl_techniques/models/vit/model.py` — fallback encoder (include_top=False, pooling=None).
- `src/train/lewm/train_lewm.py` + `train.common` — training script template.
- `src/dl_techniques/datasets/pusht_hdf5.py:synthetic_lewm_dataset` — synthetic
  data generator precedent.

### Exploration Confidence
- **Problem scope**: deep — drone streaming target + patch-level prediction +
  telemetry + Clifford primitives are all concrete with code references.
- **Solution space**: open — seven genuine design decisions (D-001..D-007) each
  with defensible alternatives.
- **Risk visibility**: clear — causality (test first), BN-batch-of-1, serialization
  round-trip, and AdaLN identity-at-init are all testable invariants.

## plan_2026-04-21_5dadc8ce
### Index

1. **[yolo12-interfaces.md](plan_2026-04-21_5dadc8ce/findings/yolo12-interfaces.md)** — Exact signatures of `YOLOv12DetectionHead` (takes list-of-3, stride-agnostic, serializable) and `YOLOv12ObjectDetectionLoss` (strides hardcoded `[8,16,32]` at line 123 of the init, returns scalar, boxes xyxy-normalized with mask-by-sum>0, TAL assigner built-in). DFL decoder inlined in loss (lines 387–401), not standalone. No NMS utility graph-compatible. No COCOeval anywhere.
2. **[box-format-and-eval.md](plan_2026-04-21_5dadc8ce/findings/box-format-and-eval.md)** — Confirms `y_true = (B, max_gt, 5) = [class_id, x1_norm, y1_norm, x2_norm, y2_norm]`. `COCODatasetBuilder` agrees. COCO cat_ids non-contiguous → remap via `idx_to_cat_id`. Risk hotspots: image_id mapping, pixel vs normalized, NMS implementation.
3. **[multi-tap-design.md](plan_2026-04-21_5dadc8ce/findings/multi-tap-design.md)** — Minimal diff to `unet.py` (~30 lines): extend `Tap` union, add `_DETECTION_HEAD_TYPES`, list-tap validation, reject multi-tap + DS, new `_DetectionHeadBlock` inline class wrapping `YOLOv12DetectionHead`, rank-3 `compute_output_shape` case. `tap: [1,2,3]` is accepted user-ordered (shallow-to-deep). Serialization via JSON already handles lists.

### Key Constraints

### Hard
- **YOLOv12ObjectDetectionLoss hardcodes strides in __init__ (line 123).** Any subclass must rebuild `self.anchors` / `self.strides` after `super().__init__()` (or override `_make_anchors`).
- **YOLOv12DetectionHead.call() requires exactly 3 input tensors** (`yolo12_heads.py:292`). Detection heads in our system must declare exactly 3 taps.
- **Box format is xyxy-normalized `[0, 1]`** end-to-end: loader → loss → decoder. No silent unit conversion.
- **COCO category IDs are non-contiguous** (gaps at 12/26/29/30/45/66/68/69/71/83). Model trains on 0..79; detection JSON for pycocotools needs remapping back to COCO IDs via `idx_to_cat_id`.
- **pycocotools `COCOeval` expects pixel coords in each image's original (H, W)**, not the training-resized size. Callback must look up `coco.loadImgs(image_id)[0]["height"/"width"]`.
- **Multi-tap heads cannot use deep_supervision.** Validation must reject `tap=list + deep_supervision=True`.

### Soft
- **YOLO defaults (reg_max=16, TAL top-k=10, focal γ=1.5, box_weight=7.5, cls_weight=0.5, dfl_weight=1.5)** per user Q5. Don't retune at start.
- **Padding sentinel `-1.0`** for empty box slots — D-006.

### Ghost / deferred
- **Detection neck (YOLO-standard `[8, 16, 32]` strides)** — user explicitly chose natural decoder strides in Q1. Revisit only if anchor count causes OOM / slowness (S1 pre-mortem).
- **Detection-only training script** — user chose multi-task integration in Q3.

### Key Reusable Components

- `YOLOv12DetectionHead` at `src/dl_techniques/layers/yolo12_heads.py:68` — our detection head block will wrap it.
- `YOLOv12ObjectDetectionLoss` at `src/dl_techniques/losses/yolo12_multitask_loss.py:51` — our loss will subclass it.
- `COCO2017MultiTaskLoader` — we extend it with `emit_boxes`. Already has `self.image_ids`, `self.idx_to_cat_id` — needed by mAP callback.
- `pycocotools` (already installed) — COCO + COCOeval classes.
- `TrainingCurvesCallback` (previous plan) — auto-picks up new `val_map50` / `val_map5095` keys via the "other" group or a new "detection" group.

### Known Pitfalls (from prior plan's LESSONS)

- Keras 3.8 `.keras` + `by_name=True` — not relevant here (we're not transferring weights in this plan).
- `MultiTaskHead.task_heads` dict bug — we don't use MultiTaskHead (D-001 of the prior plan).
- Multi-label accuracy illusion — not relevant; detection has its own metric (mAP).
- Always probe actual layer names at runtime — we'll do this during step 2 tests.

## plan_2026-04-21_8416bc0b
### Index
1. **lewm-source-schema.md** — (seed, user-provided) PyTorch LeWM architecture at `/tmp/lewm_source/`. JEPA top-level, ARPredictor (6-deep ConditionalBlock stack), Embedder (Conv1d + 2-layer MLP), MLP projector/pred_proj, SIGReg regularizer, ViT-tiny encoder (patch=14, img=224). Config: embed_dim=192, history_size=3, num_preds=1, sigreg.weight=0.09, knots=17, num_proj=1024. Loss: `pred_loss + 0.09 * sigreg_loss`.
2. **framework-conventions.md** (inline, below) — dl_techniques conventions: `@keras.saving.register_keras_serializable()`, `keras.ops`, `utils.logger`, factory-driven construction, explicit-create-vs-build rule, `compute_output_shape()` required, Dict config serialization. Models live under `src/dl_techniques/models/<name>/` with one model subdirectory per architecture. Training scripts under `src/train/<name>/train_<name>.py` using `train.common` utilities. Key guide: `research/2026_keras_custom_models_instructions.md`.
3. **reusable-layers.md** (inline, below) — Primitives available; what's missing. `TransformerLayer` (pre/post LN, factory-driven attention+FFN) at `layers/transformers/transformer.py`. Attention factory + `multi_head_attention`. `MLP` FFN at `layers/ffn/mlp.py`. Norms factory (`layer_norm`, `rms_norm`, …) at `layers/norms/factory.py`. Embeddings factory incl. `patch_2d` + `positional_learned` at `layers/embedding/factory.py`. **FiLM** at `layers/film.py` — closest to AdaLN but scale/shift only (6-way modulation would need extension). **No AdaLN-zero layer exists** → write new `AdaLNZeroConditionalBlock` layer. **No Conv1d-based action embedder primitive** → write as small custom layer or inline in model. **SIGReg** has no analogue → write new `layers/regularizers/...` or inline.
4. **existing-vit-and-training.md** (inline, below) — `src/dl_techniques/models/vit/model.py` — fully featured ViT (`include_top`, `pooling`, patch+pos embedding, CLS token, factory). `SCALE_CONFIGS["tiny"] = (192, 3, 12, 4.0)` — but LeWM uses `patch_size=14`, `img_size=224` which gives `num_patches = 256`, `max_seq_len = 257`. ViT encoder with `include_top=False, pooling='cls'` returns `(B, 192)` — perfect for our `encode()` body. **No `interpolate_pos_encoding` needed** because we control the input size. Training pattern: Pattern 2 (synthetic data) with `create_callbacks(monitor='val_loss', include_terminate_on_nan=True, use_lr_schedule=False)`. Existing `jepa/` model directory implements I-JEPA/V-JEPA (masked prediction), not our LeWM (action-conditioned). Must use distinct name: **`lewm/`** (matches upstream name, distinguishes from masked-JEPA).

### Key Constraints

### Hard (from user message + LESSONS)
- **Smoke test only.** Synthetic dataset generator + a few batches. No long training runs on real data.
- **GPU policy.** `CUDA_VISIBLE_DEVICES=1` (RTX 4070, 12 GB). Never run GPU jobs in parallel. Use `MPLBACKEND=Agg`.
- **Keras 3.8 serialization.** `@keras.saving.register_keras_serializable()` + complete `get_config()` on every custom layer/model. Sublayers tracked via `setattr`, never inside a dict.
- **Keras 3.8 weight-load trap.** `load_weights(path.keras, by_name=True)` fails — noted for completeness (no transfer needed here).
- **No `stable_worldmodel` / `stable_pretraining` dependency** — must be reimplemented with tf.data + keras + numpy.
- **No HDF5 dataset available.** Design loader against the PyTorch schema; provide synthetic generator for smoke test.

### Soft
- Python 3.11+, TF 2.18, Keras >= 3.8.
- Single-file models preferred per dl_techniques convention (config-driven factory functions).
- `dl_techniques.utils.logger` — no print statements.
- Commit cadence: `[iter-N/step-M] desc`. User pushes.

### Out of scope (follow-up)
- `eval.py` / MPC / planning (depends on stable_worldmodel).
- Real PushT HDF5 data loading.
- Multi-GPU / bf16 training infra.

### User-provided architecture summary (from /tmp/lewm_source/)

- **JEPA** top-level: `encode` / `predict` / `rollout` / `criterion` / `get_cost`. Encode: pixels (B,T,C,H,W) → flatten time, run ViT encoder, take CLS token, project to embed_dim. Training loss: `(pred_emb - tgt_emb).pow(2).mean() + λ·sigreg(emb.transpose(0,1))`.
- **ARPredictor**: learned pos embedding (1, num_frames, input_dim) + Transformer with `ConditionalBlock` (AdaLN-zero). Forward (x, c): x=context emb (B, T=3, D=192), c=action emb (B, T=3, D=192). Causal self-attn.
- **ConditionalBlock**: LayerNorm (no affine) + modulate(shift,scale) + self-attn + gated residual. AdaLN module = `SiLU → Linear(dim → 6*dim)`, zero-init.
- **Embedder**: `Conv1d(action_dim, smoothed_dim, k=1) → permute → 2-layer MLP (SiLU)`. Outputs (B, T, emb_dim).
- **MLP projector / pred_proj**: `Linear → BatchNorm1d → GELU → Linear`. BN1d over embeddings.
- **SIGReg**: sketch-based isotropic-Gaussian regularizer. Random projections (D, num_proj) each forward; computes characteristic-function residual against Gaussian target.
- **Encoder**: ViT-tiny patch=14, img=224, CLS token, `interpolate_pos_encoding=True` behavior.
- **Config**: embed_dim=192, history_size=3, num_preds=1, depth=6, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1, sigreg.weight=0.09.

## plan_2026-04-21_4c4451e5
### Index

1. **[weight-transfer.md](plan_2026-04-21_4c4451e5/findings/weight-transfer.md)** — Keras 3.8 `.keras` file + `by_name=True` is a hard ValueError. Three latent-bug helpers in the repo. Correct path: layer-by-layer `set_weights`. Three implementation options ranked.
2. **[depth-training-status.md](plan_2026-04-21_4c4451e5/findings/depth-training-status.md)** — `train_depth_estimation.py` insertion point (between build and probe). Baseline config identified. All five depth metrics present but only two compiled. No seed fixing in current code. No comparison tooling in repo.
3. **[detection-patterns.md](plan_2026-04-21_4c4451e5/findings/detection-patterns.md)** — (Retained for Plan B.) YOLOv12 head/loss infrastructure. `CliffordNetUNet` needs multi-tap head support for detection. `YOLOv12ObjectDetectionLoss` strides hardcoded `[8, 16, 32]`; our strides are `[1, 2, 4]` or `[1, 2, 4, 8]`.

### Key Constraints

### Hard
- **Keras 3.8 `.keras` + `by_name`**: `model.load_weights(path.keras, by_name=True, skip_mismatch=True)` raises `ValueError`. Must load full model and iterate layers. Source: `keras/src/saving/saving_api.py` (verified).
- **Layer-name stability** (from prior plan): encoder/decoder/bottleneck layer names are identical across head configurations — the foundation for name-based transfer. Tested by `TestLayerNameStability`.
- **`@keras.saving.register_keras_serializable()` + imports required** to `load_model` a custom-class checkpoint. All our training scripts import `dl_techniques.models.cliffordnet.unet`, so this is satisfied.

### Soft
- Baseline depth config (`cliffordnet_depth_base_20260420_130715`) is the target of any future from-scratch vs from-COCO comparison. Configuration: base variant, batch 16, patch 256, no two_phase, DS=true, lr=1e-3.
- Five depth metrics exist (`AbsRel`, `SqRel`, `RMSE`, `RMSELog`, `Delta@1.25`); only two are compiled into the current training script.

### Ghost / deferred
- The three latent-bug `load_pretrained_weights` helpers — document, do not fix (D-007).
- YOLOv12 detection infrastructure — gathered for Plan B, not consumed here.

### Key Reusable Components

- `keras.models.load_model(path)` + `model.get_layer(name).get_weights()` → works with our registered custom types.
- `src/train/common/create_callbacks` — unchanged, includes CSV logger producing `training_log.csv`.
- `CliffordNetUNet` naming scheme (`stem_*`, `enc_*`, `bottleneck_*`, `dec_*`, `head_primary_*`, `head_aux_*`) — backbone prefixes are stable; head prefixes are skipped during transfer.
- `pandas` (via `pyproject.toml` deps) — for the compare tool.

## plan_2026-04-21_c49eca98
### Index

1. **[cliffordnet-architecture.md](plan_2026-04-21_c49eca98/findings/cliffordnet-architecture.md)** — Current depth.py structure, `CliffordNetBlock` interface, CliffordNet family files, design implications for generalization.
2. **[training-scripts.md](plan_2026-04-21_c49eca98/findings/training-scripts.md)** — Training patterns in `src/train/cliffordnet/`, reusable seams (`train.common.create_callbacks`, `optimizer_builder`, `DeepSupervisionWeightScheduler`), COCO loader situation.
3. **[multihead-patterns.md](plan_2026-04-21_c49eca98/findings/multihead-patterns.md)** — Existing `vision_heads/factory.py` with full head hierarchy (Base/Classification/Seg/Det/Depth/MultiTask), `create_vision_head()` factory, ConvUNeXt backbone/head separation pattern, `include_top` convention.

### Key Constraints

### Hard
- **Keras 3 serialization**: all custom layers/models need `@keras.saving.register_keras_serializable()` + complete `get_config()`. Head layers must be stored as explicit attributes (not plain `dict`) for weight tracking.
- **No COCO classification loader exists** in the repo. `COCODatasetBuilder` (tfds) covers detection/segmentation only; `load_coco2017_local_split` is for caption pairs. A multi-label classification wrapper over `instances_*.json` must be built.
- **Backbone weight transfer across tasks** requires identical layer names across configurations (ConvUNeXt pattern). Layer-name stability is a real constraint, not a preference.
- `MultiTaskHead.task_heads` in `vision_heads/factory.py:905` uses a plain Python `dict` — Keras can't track weights through it. When reusing `MultiTaskHead` we must patch or replace with setattr-based registration.

### Soft
- `CliffordNetDepthEstimator` name exists publicly; breaking API would force downstream script updates. Prefer keeping a thin `CliffordNetDepthEstimator` factory wrapper over the new generic class.
- ConvUNeXt's `include_top: bool` + `enable_deep_supervision: bool` dual-flag API is the in-repo convention.

### Ghost (candidates to question)
- "Aux head channels = `level_channels[0]`" hard-wire (`depth.py:381`) — was a fix for depth, not a general constraint.
- `setattr(self, f"aux_norm_{level}", ...)` string-based aux-head registration (`depth.py:402-404`) — works but fragile; replace with explicit layer lists.

### Key Reusable Components

- `src/dl_techniques/layers/vision_heads/factory.py` — `BaseVisionHead`, `ClassificationHead`, `SegmentationHead`, `DetectionHead`, `DepthEstimationHead`, `InstanceSegmentationHead`, `create_vision_head(task_type, **kwargs)`.
- `src/dl_techniques/layers/geometric/clifford_block.py:545` — `CliffordNetBlock` (unchanged; the U-Net building block).
- `src/train/common/` — `create_callbacks`, `DeepSupervisionWeightScheduler`, optimizer/LR builders.
- `src/dl_techniques/datasets/vision/coco.py:218` — `COCODatasetBuilder` (detection/segmentation; would need a classification wrapper).
- `src/dl_techniques/models/convunext/model.py:442` — closest precedent for backbone+head+deep-supervision UNet with `include_top`.
