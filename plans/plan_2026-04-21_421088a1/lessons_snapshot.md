# Lessons Learned
*Cross-plan lessons. Updated and consolidated on close. Max 200 lines — rewrite, don't append forever.*
*Read before any PLAN state. This is institutional memory.*

## Patterns That Work

- **Layer-name stability for weight transfer**: Assign explicit, config-independent names to all
  backbone layers at construction time. Test it: assert two models with different head configs have
  identical encoder/decoder layer names. Enables name-based weight transfer across tasks.

- **Layer-by-layer `set_weights` for cross-task weight transfer (Keras 3)**: Load the full source
  model via `keras.models.load_model`, then iterate `source.layers`, call
  `target.get_layer(name).set_weights(source_layer.get_weights())` for each non-skipped layer.
  This is the only robust name-based path in Keras 3.8 for `.keras` files. See
  `src/dl_techniques/utils/weight_transfer.py` (DECISION D-001).

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

- **DFL decode formula**: softmax expectation across `reg_max` bins per edge, multiplied by per-
  anchor stride; `xyxy = anchor_xy ± [l, t, r, b]` in pixel coords. Lift from loss internals into
  a standalone utility (`yolo_decode.py`) — the loss does not expose this as a callable.

- **Eager numpy NMS index tracking**: NMS must preserve global indices throughout — a naive
  slice-and-pop on a shrinking list loses index tracking. Keep `order` as a list of global indices.

- **pycocotools COCOeval usage**: `COCOeval(gt, dt, 'bbox').evaluate().accumulate().summarize()`;
  `stats[0]` = mAP@50-95, `stats[1]` = mAP@50. Guard before `loadRes`: if results list is empty,
  `loadRes` raises — return early with `mAP = 0.0`.

- **Keras callback ordering for CSVLogger**: Any callback that injects new keys into the `logs`
  dict must be placed BEFORE `CSVLogger` in the callback list. Use `callbacks.insert(0, cb)` or
  build the list with the injecting callbacks first.

- **Multi-tap detection head validation**: `tap: List[int]` must have len == 3 (YOLOv12 constraint),
  unique values, all in `[0, num_levels)`, shallow-to-deep order (user-enforced), and cannot
  coexist with `deep_supervision=True`. Validate at construct time with a clear ValueError.

- **Latent-space masking is a hard constraint for 2D-grid-preserving encoders** (CliffordNet,
  any dual-stream geometric block): pixel-space token dropout is structurally incompatible.
  Encode full clip once, then mask latents before the predictor. Tube masks broadcast across
  all T frames preserve causality by construction (no "future position" selectively masked).

- **Zero-init mask token = clean denoising at init**: `z_masked = (1-M)*z + M*0` differs
  from `z` only at masked positions. Non-trivial mask_loss at init (~0.5) gives gradient
  from step 1. Matches identity-at-init philosophy.

- **Feature-flag regression path for multi-iteration plans**: gate iter-N additions on a
  config flag where `flag=False` reduces exactly to iter-(N-1) behavior. Same suite asserts
  both paths; predicted flag-tweaks to existing tests go in plan Assumptions, not surprises.

## What To Avoid

- **`load_weights(path.keras, by_name=True)` in Keras 3.8** — raises
  `ValueError: Invalid keyword arguments: {'by_name': True}`. The `by_name` kwarg only works for
  legacy `.h5`/`.hdf5` files. Latent bug in:
  - `src/dl_techniques/models/cliffordnet/model.py:413`
  - `src/dl_techniques/models/bias_free_denoisers/bfunet.py:515`
  - `src/dl_techniques/models/convnext/convnext_v2.py:400`
  Do NOT call these with `.keras` checkpoints. Separate cleanup plan needed.

- **`vision_heads/factory.py` MultiTaskHead**: `task_heads` is a plain Python dict — Keras cannot
  track weights through it. Do not use until fixed.

- **Assuming pycocotools is installed**: Install first: `.venv/bin/pip install pycocotools`.

- **tfds for local COCO**: Read `instances_{split}.json` directly via `pycocotools.COCO` when
  data is already local.

- **Depth training on MegaDepth without a pretrained encoder**: Bias-free architecture + sparse
  depth maps do not converge from scratch. Pretrained encoder required.

- **Multi-loss training with lambda=1.0 when losses differ by 100x+**: gradient budget is
  dominated by the larger-magnitude loss; smaller losses can regress absolutely while
  aggregate metrics look healthy. Rebalance lambdas or write absolute-bound criteria
  ("loss at most 2x init") rather than strict monotonicity. Smoke regimes under ~100
  gradient steps cannot distinguish monotone-decrease from noise on sub-milli-magnitude
  losses.

- **"Just swap the import"** when refactoring list-output to dict-output models. Budget a dedicated
  migration step: (1) dataset yield, (2) compile loss dict, (3) loss_weights, (4) callbacks,
  (5) all training scripts, (6) docs.

- **Extending a close-but-wrong neighbour package**: Make a new package for each new architecture;
  do not overload existing ones.

- **Forking upstream loss/head classes**: Subclass and override the specific method that bakes
  constants. Forks accumulate divergence silently.

## Codebase Gotchas

- **Keras 3 `load_weights` format/flag matrix**:
  - `.keras` / `.weights.h5`: `by_name` raises; uses object-graph positional matching.
  - `.h5` / `.hdf5`: `by_name=True` works (legacy format).

- **COCO dataset path**: `/media/arxwn/data0_4tb/datasets/coco_2017/` — subdirs `train2017/`,
  `val2017/`, `annotations/instances_train2017.json`, `annotations/instances_val2017.json`.

- **Decoder level numbering**: In `CliffordNetUNet`, level 0 = full-resolution decoder output,
  level N-1 = coarsest. `"bottleneck"` is a distinct tap (pre-decoder, post-encoder).

- **Head layer naming in `CliffordNetUNet`**: Python attribute `_head_primary_{name}` → Keras
  layer name `head_{name}`. Python attribute `_head_aux_{name}_{L}` → Keras layer name
  `head_{name}_aux_{L}`. Skip prefix for cross-task transfer: `("head_",)`.

- **YOLOv12ObjectDetectionLoss hardcodes strides `[8, 16, 32]` at line 123** of its `__init__`.
  `self.anchors` and `self.strides` are built as constant tensors at construction time. To use
  custom strides, subclass and rebuild them immediately after `super().__init__()`.

- **YOLOv12DetectionHead enforces exactly 3 input tensors** at `call()` line 292. Any wrapper
  must provide exactly 3 tap tensors; validate at construct time.

- **COCOeval pixel coords**: `pycocotools COCOeval` expects pixel coords in each image's original
  `(H, W)`, not the training-resized size. Use `coco.loadImgs(image_id)[0]["height"/"width"]`.

- **COCO category IDs are non-contiguous** (gaps at 12, 26, 29, 30, 45, 66, 68, 69, 71, 83).
  Model trains on contiguous 0..79; detection JSON for pycocotools needs remapping back via
  `idx_to_cat_id` (already on `COCO2017MultiTaskLoader`).

- **Two-phase training in `train_depth_estimation.py`**: Phase 1 = full-image resize, Phase 2 =
  random patches. `EarlyStopping` state resets between phases. Do not break this when modifying.

- **MegaDepth dataset seeding**: `MegaDepthDataset` does not accept a seed argument; shuffle
  ordering is not reproducible even with `--seed`.

- **CliffordNetBlock unchanged across architectures**: `clifford_block.py:545` works at all
  encoder/decoder/bottleneck levels. Do not modify when adding new head types.

- **Keras 3 serialization requirements**: Every custom layer/model needs
  `@keras.saving.register_keras_serializable()` + complete `get_config()`. Missing these causes
  silent failures on `keras.models.load_model`. Sublayers via `setattr`, never inside a dict.

- **GPU layout**: GPU 0 = RTX 4090 (24 GB), GPU 1 = RTX 4070 (12 GB). Never run GPU jobs in
  parallel. Use `MPLBACKEND=Agg` for all training scripts on headless/remote systems.

- **ViT-tiny @ patch=14, img=224**: `src/dl_techniques/models/vit/model.py` with
  `include_top=False, pooling='cls', scale='tiny', patch_size=14` returns `(B, 192)` per frame.

## Recurring Traps

- **Coding skip filters from attribute names, not Keras layer names**: Always
  `print([l.name for l in model.layers])` on a built model before hardcoding any name-based filter.

- **Skipping the call-site audit before deleting a class**: Always grep for ClassName across
  `src/` and `tests/` before deletion. `__init__.py` export lists are the most frequent miss.

- **Underestimating API migration scope**: Any change to model output format (list → dict, shape
  change, new keys) requires a checklist: (1) dataset yield, (2) compile loss dict, (3)
  loss_weights, (4) callbacks, (5) all training scripts, (6) docs.

- **Not testing serialization round-trip**: `model.save → keras.models.load_model → predict`
  must be a required test. Silent failures (weights not restored) are the most common bug.

- **Forgetting to install a dependency and coding around it**: Check, install, then use the
  idiomatic API. Do not write fallback implementations preemptively.

- **Trusting plan.md paraphrase of upstream architecture**: Always re-read the actual upstream
  source file at implementation time. Plans drift; code is ground truth.

- **CSVLogger before log-injecting callbacks**: If `COCOMAPCallback` (or any callback that adds
  new keys to `logs`) is appended after `CSVLogger`, those keys appear in the next epoch's CSV
  row but not the current one — or are silently dropped. Always insert injecting callbacks first.
