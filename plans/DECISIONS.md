# Consolidated Decisions
*Cross-plan decision archive. Entries merged from per-plan decisions.md on close. Newest first.*

## plan_2026-04-21_8416bc0b
### 2026-04-21T10:15:00Z — PLAN iter-1: chosen approach

**Decision**: Port LeWM as a new model package under `src/dl_techniques/models/lewm/`, reusing the existing `ViT` model for the encoder backbone, writing new custom layers for the parts that have no analogue (`AdaLNZeroConditionalBlock`, `SIGRegLayer`), and a thin dataset module supporting both synthetic smoke-test data and a PushT HDF5 skeleton.

**At the cost of**: 13 new files (above the default 3-file complexity budget), ~1200 net LOC added, two non-trivial new custom layers that must be tested carefully for serialization. We accept this cost because the port is inherently a new-model task — there is no shortcut. We pay the complexity up front, bounded by tests + serialization round-trip + AdaLN-zero identity check.

**Alternatives rejected**:
- *Extend existing `models/jepa/`* — that package is masked I/V-JEPA, not action-conditioned; name collision would confuse users. Use distinct `models/lewm/`.
- *Extend `layers/film.py` to 6-way modulation* — AdaLN-zero is a different pattern (gated residual + zero-init). Writing a dedicated layer is cleaner than overloading FiLM.
- *EMA target encoder* — upstream LeWM does not use EMA (target encoder = live encoder, gradient flows). We match upstream. Noted as future ablation.
- *Rewrite ViT-tiny inline* — wasteful; existing ViT with `include_top=False, pooling='cls', scale='tiny', patch_size=14` is a drop-in fit.

**Assumptions that could invalidate**: A1 (ViT CLS pooling returns (B, 192) cleanly), A4 (no EMA target), A6 (BatchNorm safe under fit). Tracked in plan.md Assumptions table.

### 2026-04-21T10:15:01Z — D-001 anchor: target encoder is live, not EMA

**Decision**: In `LeWM.call`, the target embedding for the JEPA loss comes from the **same live encoder** as the context embedding. Gradient flows through the target path (no `stop_gradient`). This matches upstream `/tmp/lewm_source/jepa.py:29-45` where `self.encoder` is called once per forward and no `.detach()` appears before the loss.

**Why this matters for future changes**: Many JEPA variants (I-JEPA, V-JEPA, BYOL-style) use an EMA target encoder with stop-gradient. A future reader of this code might "helpfully" add stop-gradient thinking it's a bug. It is not. Upstream LeWM is the ablation where the target encoder is live.

**Anchor location**: Inline `# DECISION D-001` comment in `models/lewm/model.py` at the target-emb computation site.

### 2026-04-21 — D-002 anchor: MLPProjector uses LayerNorm, not BatchNorm

**Decision**: `MLPProjector` (models/lewm/projector.py) uses `LayerNormalization` on the hidden activation, not `BatchNormalization`.

**Why**: Plan.md Step 4 described "Linear → BatchNorm → GELU → Linear" (matching findings #1 summary). Re-reading the actual upstream class `/tmp/lewm_source/module.py:159-172`, `MLP.__init__` defaults `norm_fn=nn.LayerNorm`, not BatchNorm. The JEPA wiring uses the default. We match upstream truth (code), not the plan's paraphrase. This also avoids the BN-batch-of-1 failure mode flagged in plan.md Pre-Mortem Scenario 2, so assumption A6 is moot here.

**Impact**: No downstream change — the projector's external shape contract is identical. Loss curves may differ slightly vs the hypothetical-BN version, but the smoke test just needs finite loss.

**Anchor location**: inline docstring in `models/lewm/projector.py` explaining the LayerNorm default and the reason.

### 2026-04-21T13:11:00Z — REFLECT iter-1: all criteria PASS

**Outcome**: All 6 success criteria (C1–C6) PASS on first attempt across 10 plan steps with zero fix attempts. No surprises, no failed falsification signals, no scope drift.

**Evidence**:
- `pytest tests/test_layers/test_adaln_zero.py tests/test_regularizers/test_sigreg.py tests/test_models/test_lewm.py -vvv` → 11/11 passed in 20.53s (CPU).
- `MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.lewm.train_lewm --synthetic --batch-size=2 --epochs=1 --steps-per-epoch=2` → exit 0, final loss 0.8782 (finite), `training_log.csv` + `final_model.keras` + `last.keras` written under `results/lewm_20260421_130931/`.

**Simplification Checks**: all 6 clean. 13 files added was within the plan's declared budget (justified: full-model port), and every abstraction maps 1:1 to an upstream class.

**Devil's advocate**: the smoke defaults downscale (56×56, depth=2, num_proj=64). Full-spec LeWM at 224×224 / depth=6 / num_proj=1024 is not exercised here. Any scale-only issue would not invalidate the port but would require tuning on a full run — explicitly out of scope for this plan.

**Recommendation**: CLOSE. Follow-up work (real HDF5, full-scale training, eval.py / MPC) should become new plans.

## plan_2026-04-21_4c4451e5
### D-001 — Layer-by-layer `set_weights` rather than `load_weights(by_name=True)`
**Decision**: In `load_weights_from_checkpoint`, load the full source model with `keras.models.load_model`, then iterate source layers and copy each compatible layer's weights into the target via `target.get_layer(name).set_weights(source_layer.get_weights())`.
**Why**: Keras 3.8 raises `ValueError("Invalid keyword arguments: {'by_name': True}")` when `load_weights` is called on a `.keras` file with `by_name=True`. Our checkpoints are `.keras`. The only robust name-based transfer path is manual layer-by-layer. Three existing helpers in the repo (`cliffordnet/model.py:413`, `bfunet.py:515`, `convnext_v2.py:400`) pass `by_name=by_name` and will crash on `.keras` checkpoints — latent bug, acknowledged, out of scope to fix here.
**Trade-off**: Requires loading the full source model (memory spike) **at the cost of** correctness. Worth it — our checkpoints are small (≈200 MB max) and transfers are rare.
**Anchor**: `# DECISION D-001` comment near the helper definition.

### D-002 — Skip prefixes are a constructor argument, not hardcoded
**Decision**: `load_weights_from_checkpoint(target, ckpt_path, skip_prefixes=("head_primary_", "head_aux_"), strict=False)` — the default matches `CliffordNetUNet`'s naming, but callers can override for other models.
**Why**: We want to ship a generic utility in `utils/`, not a CliffordNet-specific helper. Generic keeps it reusable when the same bug bites other architectures.
**Trade-off**: Slightly more ceremony in the default case **at the cost of** discoverability and reuse.

### D-003 — `TransferReport` dataclass over raw dicts
**Decision**: Return a small `TransferReport` dataclass with named fields (`loaded`, `skipped_by_prefix`, `shape_mismatch`, `missing_in_source`, `unused_in_source`) and a `summary_string()` method. Do not return a plain `Dict[str, List[str]]`.
**Why**: The report is the *primary* audit artifact when debugging "why did training not improve?" — structured access matters more than brevity here.
**Trade-off**: ~30 extra lines **at the cost of** IDE autocomplete, type safety, and a clean `logger.info(report.summary_string())` at the call site.

### D-004 — Fix seeds in `train_depth_estimation.py`, flag-controlled
**Decision**: Add `--seed <int>` (default 42) and set: `numpy.random.seed`, `tensorflow.random.set_seed`, `keras.utils.set_random_seed`, `random.seed`. Serialize the seed into `config.json`.
**Why**: From-scratch vs from-COCO comparison (user-triggered next step) needs bitwise-reproducible initialization to attribute any metric difference to the pretraining rather than to RNG luck.  Findings explicitly note zero seed fixing currently.
**Trade-off**: ~5 lines **at the cost of** minor reproducibility debt (we don't seed `MegaDepthDataset` internals because it doesn't expose a seed; document non-reproducibility of dataset shuffle ordering in `config.json`).

### D-005 — Add all five depth metrics, not the current two
**Decision**: Change `train_depth_estimation.py`'s compile call to include `AbsRelMetric`, `SqRelMetric`, `RMSEMetric`, `RMSELogMetric`, `DeltaThresholdMetric(1.25)`. Current script only compiles the first and last.
**Why**: The user wants to compare from-scratch vs from-COCO quantitatively. More metrics = richer comparison. Cost is negligible (5 extra numbers per epoch in the CSV).
**Trade-off**: Slightly slower per-epoch eval (~0.5 extra seconds per epoch) **at the cost of** a much richer comparison surface.

### D-006 — Compare-runs tool reads CSVs, not TensorBoard events or in-memory history
**Decision**: `compare_runs.py` reads `training_log.csv` (standard output of `CSVLogger` in `create_callbacks`). Does not parse TensorBoard events; does not depend on the Keras `History` object.
**Why**: CSV is stable, parseable without any Keras/TF runtime, and persists after the training process exits. TensorBoard events require parsing protobufs; `History` requires the trainer be in-process.
**Trade-off**: Can only compare what was logged via CSV **at the cost of** simplicity. If the user later wants per-step granularity they'll have to extend.

### D-007 — Three-file latent-bug: document but do not fix
**Decision**: Note the `load_pretrained_weights(by_name=True)` bug in `cliffordnet/model.py:413`, `bfunet.py:515`, `convnext_v2.py:400` in `plans/LESSONS.md` at CLOSE. Do not modify these files in this plan.
**Why**: Fixing three unrelated model helpers is a separate concern. Each fix requires testing with their own model families (`CliffordNet`, `BFUNet`, `ConvNeXtV2`) and we have no signal any user is hitting these today. Scope discipline.
**Trade-off**: The bug remains latent **at the cost of** keeping this plan focused. Tracked for a future cleanup plan.

### D-008 — Defer detection to Plan B
**Decision**: Detection head engineering is out of scope for this plan. Tracked as "Plan B" — will need: multi-tap head system extension to `CliffordNetUNet`, a `CliffordNetDetectionHead` (custom because YOLOv12DetectionHead cannot dispatch through our single-tap head_configs), adapting `YOLOv12ObjectDetectionLoss` to un-hardcode its `[8, 16, 32]` strides, a detection dataset adapter, and a training script.
**Why**: Detection is substantially more engineering than transfer plumbing; folding them into one plan would 15+ the step count, span multiple sessions, and block the critical path (depth bootstrap). User confirmed the split in Q1 of the pre-plan dialogue.
**Trade-off**: Detection availability pushed out **at the cost of** a bounded, shippable plan today.

## plan_2026-04-21_c49eca98
### D-001 — Build minimal heads inline in `unet.py` instead of reusing `vision_heads/factory.py`
**Decision**: Write `_ClassificationHeadBlock` and `_SpatialHeadBlock` as small internal layers inside `unet.py` (mirroring existing depth head style in `depth.py:357-370`), rather than using `ClassificationHead` / `SegmentationHead` / `DepthEstimationHead` from `src/dl_techniques/layers/vision_heads/factory.py`.
**Why**: The factory heads are over-parameterized (optional attention + FFN + dropout stack), targeted at transformer pipelines, and `MultiTaskHead` stores child heads in a plain Python `dict` that Keras does not track for weight serialization (`findings/multihead-patterns.md`). Minimal inline heads (LayerNorm → Conv/Dense) stay faithful to the current depth head and keep serialization simple.
**Trade-off**: Loses `create_vision_head(...)` ergonomics **at the cost of** ~60 extra lines in `unet.py`. Acceptable; refactor toward the factory later if other CliffordNet models need it.
**Anchor**: `# DECISION D-001` comment in `unet.py` near `_ClassificationHeadBlock` definition.

### D-002 — Decoder output is a numeric level; `"bottleneck"` is a separate tap name
**Decision**: `tap: int` = decoder output at that level (0 = full-res, N-1 = coarsest decoder output). `tap: "bottleneck"` = post-bottleneck features (same spatial size as level N-1 but before any decoder block).
**Why**: Classification heads semantically tap the bottleneck (max receptive field, no decoder involvement). Dense heads tap decoder outputs. Separating these prevents confusion between "deepest decoder output" and "pure encoder/bottleneck feature".
**Trade-off**: Adds a special-case string literal **at the cost of** the simpler "everything is an int level" scheme.

### D-003 — Delete `depth.py` rather than deprecate (user waived backward compat)
**Decision**: Remove `src/dl_techniques/models/cliffordnet/depth.py`. Replace all `CliffordNetDepthEstimator` call sites with the `create_cliffordnet_depth(...)` factory in `unet.py`.
**Why**: User's answer to Q4 was "don't care about backwards compatibility". Cleaner codebase, no dead code.
**Trade-off**: Any undiscovered external consumer breaks **at the cost of** a cleaner codebase. Mitigated by Step 1 grep audit.

### D-004 — COCO classification target is multi-hot from instance annotations
**Decision**: Per-image label = 80-dim float32 vector, 1.0 at every class index with ≥1 instance. Loss: `BinaryCrossentropy(from_logits=True)`.
**Why**: User's Q2 answer confirmed multi-label. Standard COCO-ML formulation.

### D-005 — Local COCO loader, not tfds-backed
**Decision**: Read directly from `/media/arxwn/data0_4tb/datasets/coco_2017/{train2017,val2017,annotations}/` via `pycocotools` (fallback `skimage.draw.polygon` for masks). Do not use `COCODatasetBuilder` (tfds).
**Why**: User has COCO locally (Q5). tfds would redownload. `COCODatasetBuilder` is detection-focused.
**Trade-off**: Extra dataset module **at the cost of** vendor-lock-in to local path — mitigated by parameterization.

### D-006 — Deep supervision is a per-head flag, default-on via factory/training-script config
**Decision**: `deep_supervision` flag per head spec. When True on a spatial-tap head, aux heads auto-attach at every deeper decoder level. Defaulted to True in `create_cliffordnet_depth` and in the COCO seg head.
**Why**: User's "always enabled" phrasing interpreted as default-on. Hard-coding it (no flag) would break unit testing; default-True flag is cleanest.

### D-007 — Serialize `head_configs` as JSON string in `get_config()`
**Decision**: `json.dumps(head_configs, default=str)` in `get_config()`, `json.loads(...)` in `from_config()`.
**Why**: `head_configs` is `Dict[str, Dict[str, Any]]` with mixed types (int tap, string `"bottleneck"` tap). Explicit JSON marshaling avoids Keras default-config nested-dict pitfalls.
**Trade-off**: Config file less Keras-idiomatic **at the cost of** guaranteed round-trip correctness.
