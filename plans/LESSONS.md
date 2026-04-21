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

- **`TransferReport` dataclass over raw dict**: When a utility returns an audit result, a
  dataclass with named fields and a `summary_string()` method beats a plain dict. IDE autocomplete
  matters when debugging "why didn't transfer work?".

- **Probe actual Keras layer names before coding skip filters**: Python attribute names
  (`_head_primary_foo`) and Keras layer names (`head_foo`) differ. Always
  `model.build(...); [l.name for l in model.layers]` before hardcoding a name filter.

- **TDD for utility functions**: Write tests first, including edge cases. In this codebase, the
  test suite caught two bugs (wrong default prefix, bad no-overlap guard) before commit.

- **"No overlap" guard must count shape-mismatched layers**: If source and target share a layer
  name but shapes differ, that still counts as overlap. `ValueError("No overlapping layers found")`
  should only fire when zero layers share a name — not when shape-mismatches exist.

- **CSV-based run comparison**: `compare_runs.py` pattern — read `training_log.csv` +
  `config.json` from each of two result dirs, use pandas `merge` on epoch, emit markdown + PNGs.
  Graceful degradation when pandas unavailable. Output to timestamped dir under `results/`.

- **Minimal inline heads beat shared factories** when the factory is over-engineered or has
  tracking bugs. LayerNorm + Conv1x1 (or GAP + Dense) is sufficient for task heads in a U-Net.

- **Declarative head specs via dict + JSON serialization**: Store head configs as a plain Python
  dict at construction time; serialize via `json.dumps(head_configs, default=str)` in
  `get_config()` and `json.loads(...)` in `from_config()`.

- **setattr-based Keras layer registration**: Track sublayers via `setattr(self, name, layer)`.
  Keras walks object attributes for weight tracking; it does not recurse into dicts.

- **Deep supervision as a per-head flag**: `deep_supervision: bool` in each head spec, defaulting
  to True. Hard-coding always-on breaks unit testing.

- **Smoke test before declaring done**: 1-epoch run on small data with save-reload-predict
  determinism check is a fast, reliable gate.

- **Checkpoint at start of every plan**: `cp-000-iter1.md` with the pre-change commit hash costs
  nothing and gives a clean rollback point.

- **`add_loss` inside `call` composes with `model.fit`**: No custom training loop needed for
  multi-term losses (e.g. JEPA MSE + SIGReg). Call `self.add_loss(scalar_tensor)` inside
  `call()`; reported CSV-log loss includes it automatically. Keep `jit_compile=False` to avoid
  tracing issues with dynamic add_loss.

- **Identity-at-init invariants are cheap gates**: For layers that should be identity at init
  (e.g. AdaLN-zero with zero-init modulation), write a test that builds and calls `block(x, c)`
  then asserts `max|block(x,c) - x| < 1e-6` before any training. Catches zero-init bugs instantly.

- **Re-read actual source at implementation time**: Plan.md often paraphrases findings ("Linear
  → BatchNorm → GELU → Linear"). Before writing the layer, open the upstream file and verify.
  The plan's paraphrase can drift from truth (example: LeWM MLP defaults to LayerNorm, not
  BatchNorm). Trust code over paraphrase.

- **Stateless random projections inside a layer `call()`**: `keras.random.normal(...)` inside
  `call` is allowed; per-forward non-determinism matches many regularizers (SIGReg) and is not
  a bug. Don't try to make it deterministic by stashing a seed — upstream doesn't either.

## What To Avoid

- **`load_weights(path.keras, by_name=True)` in Keras 3.8** — raises
  `ValueError: Invalid keyword arguments: {'by_name': True}`. The `by_name` kwarg only works for
  legacy `.h5`/`.hdf5` files. For `.keras` and `.weights.h5`, Keras uses object-graph positional
  matching. The three existing helpers carry this latent bug:
  - `src/dl_techniques/models/cliffordnet/model.py:413`
  - `src/dl_techniques/models/bias_free_denoisers/bfunet.py:515`
  - `src/dl_techniques/models/convnext/convnext_v2.py:400`
  Do NOT call these with `.keras` checkpoints until fixed. Separate cleanup plan needed.

- **`vision_heads/factory.py` MultiTaskHead**: `task_heads` is a plain Python dict — Keras cannot
  track weights through it. Do not use until fixed (replace dict with setattr-based registration).

- **Assuming pycocotools is installed**: Not in `.venv` by default. Install first:
  `.venv/bin/pip install pycocotools`.

- **tfds for local COCO**: Read `instances_{split}.json` directly via `pycocotools.COCO` when
  data is already local.

- **Depth training on MegaDepth without a pretrained encoder**: A bias-free architecture and
  sparse depth maps do not converge from scratch. Pretrained encoder (ImageNet or COCO) is
  required. See `feedback_depth_estimation.md`.

- **"Just swap the import"** when refactoring list-output to dict-output models. The change
  cascades into: dataset yield format, `model.compile(loss=...)` dict, loss_weights dict, all
  callbacks, and all training scripts. Budget a dedicated migration step; scope is 3–5× what
  it appears.

- **Extending a close-but-wrong neighbour package**: `models/jepa/` (masked I/V-JEPA) and
  `models/lewm/` (action-conditioned JEPA) share lineage but not architecture. Make a new
  package for each new architecture; do not overload existing ones.

- **"Helpfully" adding stop-gradient to JEPA target paths**: Some JEPA variants use EMA target
  encoders with stop-gradient; LeWM does not (target = live encoder, gradient flows both ways).
  A future reader could break this by "fixing" a perceived bug. Anchor with `# DECISION D-NNN`
  at the target-emb computation site. See `models/lewm/model.py:177` (D-001).

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

- **Two-phase training in `train_depth_estimation.py`**: Phase 1 = full-image resize, Phase 2 =
  random patches. `EarlyStopping` state resets between phases. Do not break this when modifying.

- **MegaDepth dataset seeding**: `MegaDepthDataset` does not accept a seed argument; shuffle
  ordering is not reproducible even with `--seed`. Document this caveat in config.json.

- **CliffordNetBlock unchanged across architectures**: `clifford_block.py:545` — works at all
  encoder/decoder levels and at the bottleneck. Do not modify when adding new head types.

- **Keras 3 serialization requirements**: Every custom layer/model needs
  `@keras.saving.register_keras_serializable()` + complete `get_config()`. Missing these causes
  silent failures on `keras.models.load_model`. Sublayers must be tracked via `setattr`, never
  inside a dict.

- **`train.common.DeepSupervisionWeightScheduler`**: Existing callback for annealing deep
  supervision loss weights. Reuse; do not reimplement per script.

- **GPU layout**: GPU 0 = RTX 4090 (24 GB), GPU 1 = RTX 4070 (12 GB). Never run GPU jobs in
  parallel. Use `MPLBACKEND=Agg` for all training scripts on headless/remote systems.

- **ViT-tiny @ patch=14, img=224**: `src/dl_techniques/models/vit/model.py` with
  `include_top=False, pooling='cls', scale='tiny', patch_size=14` returns `(B, 192)` per frame
  and round-trips through `.keras`. Use as a drop-in encoder backbone (e.g. for LeWM).

## Recurring Traps

- **Coding skip filters from attribute names, not Keras layer names**: Always `print([l.name for
  l in model.layers])` on a built model before hardcoding any name-based filter.

- **Skipping the call-site audit before deleting a class**: Always grep for ClassName across
  `src/` and `tests/` before deletion. `__init__.py` export lists are the most frequent miss.

- **Underestimating API migration scope**: Any change to model output format (list → dict, shape
  change, new keys) requires a checklist: (1) dataset yield, (2) compile loss dict, (3)
  loss_weights, (4) callbacks, (5) all training scripts, (6) docs.

- **Not testing serialization round-trip**: `model.save → keras.models.load_model → predict`
  must be a required test. Silent failures (weights not restored) are the most common bug.

- **Forgetting to install a dependency and coding around it**: Check (`python -c "import X"`),
  install if missing, then use the idiomatic API. Do not write fallback implementations
  preemptively.

- **Trusting plan.md paraphrase of upstream architecture**: Always re-read the actual upstream
  source file at implementation time. Plans drift; code is ground truth.
