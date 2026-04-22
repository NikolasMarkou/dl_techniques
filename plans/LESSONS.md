# Lessons Learned
*Cross-plan lessons. Consolidated on close. Max 200 lines. Read before PLAN.*

## Patterns That Work

- **Layer-name stability + per-layer `set_weights` for cross-task weight transfer (Keras 3)**:
  assign explicit, config-independent layer names at construction; for transfer, load source
  via `keras.models.load_model`, iterate `source.layers`, call
  `target.get_layer(name).set_weights(src_layer.get_weights())`. Only robust name-based path
  in Keras 3.8 for `.keras`. See `src/dl_techniques/utils/weight_transfer.py` (D-001).

- **Subclass-and-override pattern for third-party loss classes**: When a loss bakes constants at
  `__init__` time (e.g. strides, anchors), subclass it, call `super().__init__()`, then immediately
  rebuild the baked attributes. Works cleanly: `CliffordDetectionLoss` rebuilds `self.anchors` /
  `self.strides` post-`super().__init__()`. Keeps parent logic fully intact.

- **Declarative head specs via dict + JSON serialization**: Store head configs as a plain Python
  dict at construction time; serialize via `json.dumps(head_configs, default=str)` in
  `get_config()` and `json.loads(...)` in `from_config()`.

- **setattr-based Keras layer registration**: Track sublayers via `setattr(self, name, layer)`.
  Keras walks object attributes for weight tracking; it does not recurse into dicts.

- **Minimal inline heads beat shared factories** when the factory is over-engineered or has
  tracking bugs. LayerNorm + Conv1x1 (or GAP + Dense) is sufficient for task heads in a U-Net.

- **Deep supervision as a per-head flag**: `deep_supervision: bool` in each head spec, defaulting
  to True. Hard-coding always-on breaks unit testing.

- **Smoke test before declaring done**: 1-epoch run on small data with save-reload-predict
  determinism check is a fast, reliable gate.

- **Checkpoint at start of every plan**: `cp-000-iter1.md` with the pre-change commit hash costs
  nothing and gives a clean rollback point.

- **`add_loss` inside `call` composes with `model.fit`**: No custom training loop needed for
  multi-term losses. Call `self.add_loss(scalar_tensor)` inside `call()`; keep `jit_compile=False`.

- **Identity-at-init invariants are cheap gates**: For layers that should be identity at init,
  write a test that asserts `max|block(x,c) - x| < 1e-6` before any training.

- **Re-read actual source at implementation time**: Plan.md often paraphrases findings. Before
  writing the layer, open the upstream file and verify. Plans drift; code is ground truth.

- **CSV-based run comparison**: `compare_runs.py` pattern — read `training_log.csv` + `config.json`
  from each of two result dirs, use pandas `merge` on epoch, emit markdown + PNGs.

- **YOLO/COCO detection glue**: (a) DFL decode = softmax expectation over `reg_max` bins ×
  per-anchor stride; `xyxy = anchor_xy ± [l,t,r,b]` in pixel coords — lift to `yolo_decode.py`.
  (b) NMS must preserve global indices — keep `order` as list of globals, not slice-and-pop.
  (c) `COCOeval(gt,dt,'bbox').evaluate().accumulate().summarize()`; `stats[0]`=mAP@50-95,
  `stats[1]`=mAP@50; guard empty results (loadRes raises) → return 0.0.

- **Keras callback ordering for CSVLogger**: Any callback that injects new keys into the `logs`
  dict must be placed BEFORE `CSVLogger` in the callback list. Use `callbacks.insert(0, cb)` or
  build the list with the injecting callbacks first.

- **Multi-tap detection head validation**: `tap: List[int]` must have len == 3 (YOLOv12 constraint),
  unique values, all in `[0, num_levels)`, shallow-to-deep order (user-enforced), and cannot
  coexist with `deep_supervision=True`. Validate at construct time with a clear ValueError.

- **Latent-space masking for 2D-grid encoders** (CliffordNet, any dual-stream geometric block):
  pixel-space token dropout is structurally incompatible. Encode full clip, mask latents before
  predictor. Tube masks broadcast across T preserve causality. Zero-init mask token
  (`z_masked = (1-M)*z`) differs from `z` only at masked positions → non-trivial mask_loss
  (~0.5) at init gives gradient from step 1.

- **Feature-flag regression path for multi-iteration plans**: gate iter-N additions on a
  config flag where `flag=False` reduces exactly to iter-(N-1) behavior. Same suite asserts
  both paths; predicted flag-tweaks to existing tests go in plan Assumptions, not surprises.

- **LayerScale γ=1e-5 as drop-in for AdaLN-zero identity-at-init**: wrap MHA + MLP residual
  adds with a trainable scalar init=1e-5. Same guarantee without conditioning wiring. Used by
  `CausalCliffordNetBlock` and `video_jepa`'s `CausalSelfAttnMLPBlock`.

- **Ghost-constraint audit during PLAN**: for each deleted feature, grep every caller and ask
  whether the surrounding code still makes sense. Clean up in the same plan — don't leave dead
  code. Example: synthetic_drone_video.py's telemetry emission became dead the moment
  TelemetryEmbedder was removed.

- **`DepthPredictionGridCallback` is the skeleton for any fixed-eval-batch viz callback**:
  cache eval batch at `__init__`, `on_epoch_end` gated on `frequency`, lazy `import matplotlib`
  inside save method, try/except wrap, `gc.collect()` after each figure. Copy the skeleton,
  not the content. Reference: `src/dl_techniques/callbacks/depth_visualization.py`, and
  `jepa_visualization.py` which follows it exactly.

- **Reuse existing callbacks before writing new ones**: `TrainingCurvesCallback` in
  `dl_techniques.callbacks.training_curves` auto-groups log keys by `_loss` suffix into a
  single PNG; drops into any training script in 2 lines. Verify via
  `src/train/cliffordnet/train_coco_multitask.py` (live user).

## What To Avoid

- **`load_weights(path.keras, by_name=True)` in Keras 3.8** raises. `by_name` is `.h5`/`.hdf5`
  only. Latent bug in cliffordnet/model.py:413, bfunet.py:515, convnext_v2.py:400 — do NOT
  use with `.keras`. (`.keras`/`.weights.h5` use object-graph positional matching.)

- **`vision_heads/factory.py` MultiTaskHead**: `task_heads` is a plain Python dict — Keras cannot
  track weights through it. Do not use until fixed.

- **Assuming pycocotools is installed**: Install first: `.venv/bin/pip install pycocotools`.

- **tfds for local COCO**: Read `instances_{split}.json` directly via `pycocotools.COCO` when
  data is already local.

- **Depth training on MegaDepth without a pretrained encoder**: Bias-free architecture + sparse
  depth maps do not converge from scratch. Pretrained encoder required.

- **Multi-loss λ=1.0 when losses differ 100×+**: gradient budget dominated by larger loss;
  smaller losses can regress absolutely while aggregate metrics look healthy. Rebalance λ or
  use absolute-bound criteria ("loss at most 2× init"), not strict monotonicity. Smoke regimes
  <100 steps can't distinguish monotone-decrease from noise on sub-milli losses.

- **"Just swap the import"** for list→dict output refactors. Budget a dedicated migration step
  covering dataset yield, compile loss dict, loss_weights, callbacks, all training scripts, docs.

- **Extending a close-but-wrong neighbour package**: Make a new package for each new architecture;
  do not overload existing ones.

- **Forking upstream loss/head classes**: Subclass and override the specific method that bakes
  constants. Forks accumulate divergence silently.

## Codebase Gotchas

- **COCO dataset path**: `/media/arxwn/data0_4tb/datasets/coco_2017/` — subdirs `train2017/`,
  `val2017/`, `annotations/instances_train2017.json`, `annotations/instances_val2017.json`.

- **Decoder level numbering**: In `CliffordNetUNet`, level 0 = full-resolution decoder output,
  level N-1 = coarsest. `"bottleneck"` is a distinct tap (pre-decoder, post-encoder).

- **Head layer naming in `CliffordNetUNet`**: attr `_head_primary_{name}` → Keras name
  `head_{name}`; attr `_head_aux_{name}_{L}` → `head_{name}_aux_{L}`. Cross-task skip
  prefix: `("head_",)`.

- **YOLOv12 hardcodes**: `ObjectDetectionLoss` bakes strides `[8,16,32]` + anchors as constants
  at `__init__` (line 123) — subclass + rebuild post-`super().__init__()` for custom strides.
  `DetectionHead.call()` (line 292) enforces exactly 3 input tensors — validate at construct time.

- **COCO eval gotchas**: (a) `COCOeval` expects pixel coords in each image's original `(H, W)`,
  not training-resized; use `coco.loadImgs(image_id)[0]`. (b) category IDs are non-contiguous
  (gaps at 12, 26, 29, 30, 45, 66, 68, 69, 71, 83); model trains on 0..79, detection JSON needs
  remapping via `idx_to_cat_id` (on `COCO2017MultiTaskLoader`).

- **Two-phase training in `train_depth_estimation.py`**: Phase 1 = full-image resize, Phase 2 =
  random patches. `EarlyStopping` state resets between phases. Do not break this when modifying.

- **MegaDepth dataset seeding**: `MegaDepthDataset` does not accept a seed argument; shuffle
  ordering is not reproducible even with `--seed`.

- **CliffordNetBlock unchanged across architectures**: `clifford_block.py:545` works at all
  encoder/decoder/bottleneck levels. Do not modify when adding new head types.

- **Keras 3 serialization requirements**: every custom layer/model needs
  `@keras.saving.register_keras_serializable()` + complete `get_config()`; missing → silent
  failures on load. Sublayers via `setattr`, never inside dict.

- **GPU layout**: GPU 0 = RTX 4090 (24 GB), GPU 1 = RTX 4070 (12 GB). Never run GPU jobs in
  parallel. Use `MPLBACKEND=Agg` for all training scripts on headless/remote systems.

- **Full `make test` takes ~1.5h** (also the pre-push hook). Do not run as a routine regression.
  Scope pytest to touched modules + their importers. Documented in root CLAUDE.md.

- **Test dir convention**: `tests/test_<subpkg>/test_<name>.py` — does NOT mirror
  `src/dl_techniques/<subpkg>/` paths. Place callback tests under `tests/test_callbacks/`,
  not `tests/dl_techniques/callbacks/`.

- **`callbacks/__init__.py` is empty by convention**: do not add re-exports. Train scripts
  import by full module path: `from dl_techniques.callbacks.<module> import <Class>`.

- **BDD100K video dataset path**: `/media/arxwn/data0_4tb/datasets/bdd_data/train/videos/` — 28641
  .mov files, flat layout, ~1200 frames @ 30fps @ 1280×720. opencv-python random-frame seek is
  adequate for sanity-scale video-JEPA (~0.84 s/step at B=4/T=8/112²); decord not needed below
  T>16 or full-epoch runs.

- **ViT-tiny @ patch=14, img=224**: `src/dl_techniques/models/vit/model.py` with
  `include_top=False, pooling='cls', scale='tiny', patch_size=14` returns `(B, 192)` per frame.

## Recurring Traps

- **Coding skip filters from attribute names, not Keras layer names**: Always
  `print([l.name for l in model.layers])` on a built model before hardcoding any name-based filter.

- **Skipping the call-site audit before deleting a class**: Always grep for ClassName across
  `src/` and `tests/` before deletion. `__init__.py` export lists are the most frequent miss.

- **Underestimating API migration scope**: output-format changes (list→dict, shape, new keys)
  require a checklist: dataset yield, compile loss dict, loss_weights, callbacks, training
  scripts, docs.

- **Not testing serialization round-trip**: `model.save → keras.models.load_model → predict`
  must be a required test. Silent failures (weights not restored) are the most common bug.

- **Forgetting to install a dependency and coding around it**: Check, install, then use the
  idiomatic API. Do not write fallback implementations preemptively.

- **Trusting plan.md paraphrase of upstream architecture**: Always re-read the actual upstream
  source file at implementation time. Plans drift; code is ground truth.

- **CSVLogger before log-injecting callbacks**: If `COCOMAPCallback` (or any callback that adds
  new keys to `logs`) is appended after `CSVLogger`, those keys appear in the next epoch's CSV
  row but not the current one — or are silently dropped. Always insert injecting callbacks first.

- **LOC-delta budgets miss new-loader additions**: a "~150 LOC small loader" plan estimate
  routinely lands at 190-210 LOC. Budget loader LOC as additive, not offsetable by core
  deletions.
