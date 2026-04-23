# Video-JEPA-Clifford — Training

Self-supervised video model built on CliffordNet primitives and the JEPA
(Joint Embedding Predictive Architecture) objective. Currently trained on
dashcam video (BDD100K); a synthetic dataset is kept for smoke/CI runs.

- **Model package**: `src/dl_techniques/models/video_jepa/`
- **Datasets**:
  - `src/dl_techniques/datasets/bdd100k_video.py` — opencv-python loader
    for BDD100K `.mov` files, with deterministic seeded train/val split.
  - `src/dl_techniques/datasets/synthetic_drone_video.py` — colored
    rectangle / noise clip generator, pixels-only (telemetry emission
    was removed; see "What changed since iter-2" below).
- **Trainer (this directory)**: `train_video_jepa.py`
- **Tests**: `tests/test_models/test_video_jepa/` — **41 tests** (4 legacy
  telemetry-embedder tests + 1 `cond_dim` invariant test were removed
  when the drone-telemetry conditioning path was stripped; remaining
  tests cover causality, identity-at-init, SIGReg finiteness, shape
  flow, serialization, streaming, and smoke training).
  Visualization callbacks have their own unit tests at
  `tests/test_callbacks/test_jepa_visualization.py` (6 tests).

---

## What changed since iter-2

Two closed plans in this session modified the training stack. In order:

1. `plans/plan_2026-04-22_4f29c76f/` — **telemetry-A strip + BDD100K
   loader**:
   - Deleted `telemetry_embedder.py`. Dropped `cond_dim`, `telemetry_dim`
     config fields and the `cond_dim == embed_dim` invariant.
   - Replaced the AdaLN-zero conditional predictor block with a plain
     `CausalSelfAttnMLPBlock` (LayerNorm → causal MHA → **LayerScale
     γ=1e-5** → LayerNorm → MLP → **LayerScale γ=1e-5**). LayerScale
     restores near-identity-at-init without any conditioning wiring.
   - Synthetic dataset simplified to pixels-only.
   - New `bdd100k_video_dataset` loader (opencv-python, `cv2.VideoCapture`
     random-start seeks, BGR→RGB, INTER_AREA resize, float32 in [0, 1]).
   - Added `--dataset {synthetic,bdd100k}` CLI switch + `--videos-root`.
2. `plans/plan_2026-04-22_016e549b/` — **training visualization (reuse)**:
   - Wired `dl_techniques.callbacks.training_curves.TrainingCurvesCallback`
     as-is (auto-detects `val_` twins and plots them alongside train).
   - Added two new JEPA-specific callbacks in
     `src/dl_techniques/callbacks/jepa_visualization.py`:
     `LatentMaskOverlayCallback` (red tube-mask overlay on a cached
     eval batch, every N epochs) and `PatchPredictionErrorCallback`
     (per-patch L2 heatmap, `viridis`, shared scale, every N epochs).
   - Both callbacks run the model in `training=False`, cache a fixed
     eval batch at construction time, use lazy `matplotlib` imports,
     wrap figure save in try/except, and `gc.collect()` after each epoch.

Two subsequent commits added the current training loop features:

3. Loader fix: the train script no longer passes `num_steps` to the
   BDD100K loader (it was being interpreted as `.take(num_steps)` and
   exhausted the dataset after epoch 1). Keras's `steps_per_epoch`
   alone bounds the per-epoch iteration; the generator is `while True`.
4. **Train/val split, EarlyStopping, best checkpoint** — see below.

---

## What this trains

A patch-level latent video model that:

- **Encodes** frames with a hybrid `PatchEmbedding2D` + 2 `CliffordNetBlock`
  stack — keeps the 2D patch grid intact for geometric-product
  interactions. At `img_size=112, patch_size=8` the grid is `14×14`,
  embed_dim=128.
- **Predicts** future patch latents via a factorized spatial /
  causal-temporal stack. A `CliffordNetBlock` handles per-frame spatial
  structure, a `CausalCliffordNetBlock` handles per-spatial-position
  temporal causality, and a `CausalSelfAttnMLPBlock` (new, non-conditional)
  sits in between to give the predictor dense cross-token interactions
  along the spatial grid without any conditioning input.
- **Regularizes** latents with SIGReg (Sketch Isotropic Gaussian
  Regularizer) — collapse prevention without EMA or stop-gradient.
- **Masked-latent objective** (iter-2): V-JEPA-style tube-masked
  latent prediction as a second training target. Single training run,
  three additive losses.

**Not present (intentionally)**:

- No conditioning input. `TelemetryEmbedder`, AdaLN-zero, and all
  IMU/GPS/altitude plumbing were stripped. `stream_step(frame)` and
  the model's `call({"pixels": ...})` accept pixels only.
- No EMA target encoder — targets come from the same live encoder
  (LeWM design choice, FW-003 still queued as an ablation).
- No pixel decoder — outputs live in `(B, T, 14, 14, 128)` latent
  patch space.

Streaming inference is supported via a rolling buffer of the last K
patch grids (O(1) amortized per frame).

## Loss

```
total = λ_next · L_next_frame + λ_mask · L_mask + λ_sigreg · L_sigreg
```

Defaults: `λ_next=1.0`, `λ_mask=1.0`, `λ_sigreg=0.09`.

- `L_next_frame` — MSE between predicted patch latents for frame `t+1`
  and the encoder's output on the actual `t+1` frame.
- `L_mask` — MSE on masked patch positions between predictor output
  and the same encoder's output at the masked slots. Latent-space
  masking only — pixel-space is structurally incompatible with the
  Clifford encoder's intact-grid requirement (hard constraint).
- `L_sigreg` — Epps-Pulley Gaussianity statistic on `(B·T, N, D)`
  reshape, averaged over projections. Default `sigreg_num_proj=64`
  for smoke; use `1024` for real training.

The model exposes three `keras.metrics.Mean` trackers
(`next_frame_loss`, `mask_loss`, `sigreg_loss`) via its `metrics`
property, so `CSVLogger` automatically writes a per-component column
next to the aggregated `loss`. Validation produces the same columns
prefixed with `val_`.

## Callbacks (what the trainer wires up)

Registered in `_build_callbacks()` in order:

1. `TerminateOnNaN` — first, so NaN losses short-circuit before any
   visualization callback.
2. `CSVLogger` → `training_log.csv` (epoch + 3 train losses + 3 val
   losses + aggregate + `val_loss`).
3. `ModelCheckpoint` → `last.keras` every epoch.
4. `ModelCheckpoint` → `best.keras`, `save_best_only=True`, monitored
   on `val_loss` when validation is enabled, else `loss`.
5. `TrainingCurvesCallback` → `training_curves/loss.png`. Reused
   from `dl_techniques.callbacks.training_curves`; it detects `val_`
   twins automatically and overlays them on the same axes.
6. `LatentMaskOverlayCallback` → `jepa_viz/epoch_NNN_mask_overlay.png`.
   T columns × 2 rows per sample, row 0 = RGB frames, row 1 = RGB
   with red tube-mask overlay (mask broadcast across T).
7. `PatchPredictionErrorCallback` → `jepa_viz/epoch_NNN_patch_error.png`.
   T columns × B rows, `viridis` heatmap of per-patch L2 error, shared
   color scale per epoch.
8. `EarlyStopping` (if `--early-stopping-patience > 0`) — monitors the
   same key as `best.keras`, `restore_best_weights=True`, verbose.

## Quick start

### Sanity / smoke on GPU 1 (synthetic)

```bash
MPLBACKEND=Agg .venv/bin/python -m train.video_jepa.train_video_jepa \
    --dataset synthetic --gpu 1 \
    --epochs 2 --steps-per-epoch 4 --batch-size 2 \
    --T 4 --img-size 64 --patch-size 8
```

### Sanity on BDD100K (GPU 0, real data, ~15 min)

```bash
MPLBACKEND=Agg .venv/bin/python -m train.video_jepa.train_video_jepa \
    --dataset bdd100k \
    --videos-root /media/arxwn/data0_4tb/datasets/bdd_data/train/videos \
    --gpu 0 \
    --epochs 1 --steps-per-epoch 200 --batch-size 4 \
    --T 8 --img-size 112 --patch-size 8 --embed-dim 128 \
    --sigreg-num-proj 1024 --seed 0 \
    --output-dir results/video_jepa_sanity_bdd100k
```

### Full run with val + EarlyStopping (GPU 0, many hours)

```bash
MPLBACKEND=Agg .venv/bin/python -m train.video_jepa.train_video_jepa \
    --dataset bdd100k \
    --videos-root /media/arxwn/data0_4tb/datasets/bdd_data/train/videos \
    --gpu 0 \
    --epochs 100 --steps-per-epoch 1000 \
    --val-steps 100 --val-fraction 0.1 \
    --early-stopping-patience 15 \
    --batch-size 4 --T 8 --img-size 112 --patch-size 8 --embed-dim 128 \
    --sigreg-num-proj 1024 --visualization-frequency 1 --seed 0 \
    --output-dir results/video_jepa_bdd100k_run_XX
```

Batch size must be ≥ 2 — `CliffordNetBlock` uses BatchNormalization in
its context stream and is unsafe at batch-of-1.

### CLI reference (selected flags)

| Flag | Default | Notes |
|---|---|---|
| `--dataset` | `synthetic` | `synthetic` or `bdd100k`. |
| `--videos-root` | — | required for `--dataset bdd100k`. |
| `--val-steps` | `0` | 0 disables validation entirely. |
| `--val-fraction` | `0.1` | fraction of BDD paths held out for val; disjoint from train by seeded permutation. |
| `--early-stopping-patience` | `0` | 0 disables. |
| `--visualization-frequency` | `1` | write viz PNGs every N epochs. |
| `--sigreg-num-proj` | `64` | bump to `1024` for real training. |
| `--mask-prediction-enabled / --no-mask-prediction-enabled` | True | iter-2 tube-masked loss. |
| `--mask-ratio` | `0.6` | fraction of spatial patch positions masked. |
| `--lambda-next-frame`, `--lambda-mask` | `1.0, 1.0` | loss weights; see iter-2 caveat. |
| `--gpu` | None | sets `CUDA_VISIBLE_DEVICES`; serial GPU use only. |

## Artifacts produced by a run

```
results/<output-dir>/
├── training_log.csv          # epoch + 3 train losses + 3 val losses + aggregates
├── last.keras                # checkpoint every epoch
├── best.keras                # save_best_only on val_loss (or loss if no val)
├── final_model.keras         # saved at end of fit()
├── training_curves/
│   └── loss.png              # auto-regenerated every visualization-frequency epochs; overlays val_*
└── jepa_viz/
    ├── epoch_000_mask_overlay.png
    ├── epoch_000_patch_error.png
    └── epoch_NNN_... (N = number of epochs completed)
```

## Verified behavior

41 / 41 module tests + 6 / 6 visualization-callback tests passing.
Verification order is hardest-first:

1. **Causality** — perturb frame `k ∈ [1, T-1]`; frames `< k` are
   identical to baseline within `1e-5`. Future-frame diff `> 1e-6`
   confirms the predictor isn't collapsing.
2. **SIGReg finiteness** at init and after one fit step.
3. **Predictor identity-at-init** (renamed from `test_adaln_identity_init`)
   — `max|predictor(z) - (z + pos_t)| < 5e-3` with the LayerScale
   γ=1e-5 scheme.
4. **Serialization round-trip** — all components + full model within
   `atol=1e-5`, including post-training reload. Also covers
   `TelemetryEmbedder`-era legacy config keys (they are silently
   scrubbed by `from_dict`).
5. **Shape flow** across the predictor pipeline.
6. **Streaming O(1)** — buffer capped at K, shape-tested over K+2
   frames.
7. **Visualization callbacks** — on `on_epoch_end`, PNGs land at the
   expected paths; weights are unchanged after the callback runs; the
   frequency gate is respected; the masked-prediction callback skips
   gracefully when `mask_prediction_enabled=False`.
8. **Smoke training** — monotonically-decreasing total loss on 2 × 4
   steps of synthetic data.

### Real-data convergence signal (BDD100K)

On the full-run config above (100 epochs, 1000 steps, B=4, T=8, 112²,
GPU 0, EarlyStopping patience=15), we observed clean monotone descent
in both train and val losses across dozens of epochs:

- train_loss  ≈ 1.25 (ep 0) → ≈ 0.041 (ep 52 of the in-flight run)
- val_loss    ≈ 0.426 (ep 0) → best ≈ 0.041 mid-run, ES counter under
  control
- train and val stay within a few percent of each other — no
  overfitting signal

Wall time ~12–13 min/epoch on the 4090 (I/O-bound by opencv MOV seeks).

The iter-2 caveat about `L_next_frame` being starved by `L_mask` did
**not** materialize at real scale — `next_frame_loss` descended
proportionally with `mask_loss` (e.g. 0.067 → 0.00018 by ep 50,
alongside 0.224 → 0.0016 for `mask_loss`). The smoke-regime observation
was a statistical-resolution artifact, as suspected; it remains a valid
thing to re-check on any config change but is not blocking.

Smoke runs are architectural correctness checks, not convergence
proofs. BDD runs under the above settings give a convergence signal,
but no downstream-task evaluation has been performed yet — `val_loss`
is still the SSL objective, not a linear-probe accuracy.

## Known issues / caveats

- **I/O is the bottleneck on real runs.** opencv-python random-frame
  seek on BDD100K MOV files saturates 3–4 CPU cores and keeps GPU
  utilization low. At sanity scale (B=4/T=8/112²) we measured ~0.84
  s/step; the 4090 is underutilized. Decord would likely help for
  T≥16 or full-epoch throughput, but is not worth the dep at current
  scale.
- **Checkpoint size is dominated by optimizer state.** The model is
  ~865k trainable parameters (~3.3 MB fp32) but `final_model.keras`
  and `last.keras` land at ~10.7 MB each because they include the
  AdamW state.
- **No EMA target encoder.** SIGReg is the sole collapse-prevention
  mechanism. Works at the scale we've trained; FW-003 is queued to
  test an EMA variant.
- **Pixel-space masking is blocked.** Hard constraint, not a
  preference — a CliffordNet encoder requires an intact 2D patch
  grid. Encode-then-mask is the only viable path under the current
  encoder choice.
- **Patch grid must be ≥ 2 on each axis** for the spatial Clifford
  block's depthwise convolutions. At `img_size=112, patch_size=8`
  the grid is `14×14` ✓.
- **Full `pytest tests/` takes ~1.5 h** (also the pre-push hook).
  Scope pytest to `tests/test_models/test_video_jepa/` and
  `tests/test_callbacks/test_jepa_visualization.py` during
  development. See root `CLAUDE.md`.
- **No serial-GPU protection in the trainer.** User convention is
  to keep GPU jobs serial (GPU 0 = 4090 24 GB, GPU 1 = 4070 12 GB);
  nothing in the script enforces this.

## What the trained checkpoint is usable for

Given the latent-patch encoder + causal predictor and no decoder, the
natural downstream uses are:

- **Frozen-feature downstream tasks** — linear probe or small head on
  `encode_frames(...)` for driving-event classification, detection,
  segmentation, etc. BDD100K provides labels for these.
- **Per-patch surprise / anomaly maps** — exactly what the
  `patch_error` viz shows on a fixed batch, applied to live streams
  via `stream_step`. Natural fit for dashcam anomaly detection.
- **Latent-space rollouts** — the predictor is causal; feed `z_{1..t}`
  through repeatedly for short-horizon latent forecasting without
  decoding back to pixels.
- **Clip retrieval / clustering** — pooled latent grid as a per-clip
  embedding.
- **Pretraining checkpoint for a larger stage** — weight-transfer
  the encoder into a deeper / wider model and continue training.

The checkpoint cannot generate pixels, is not text-conditioned, and
has no action input (telemetry was removed).

## Future work

Queued in `plans/DECISIONS.md` as FW-001 … FW-004 from the original
iter-2 close (`plan_2026-04-21_421088a1`):

- **FW-001** — **λ rebalance**. Original motivation was the
  smoke-regime asymmetry; the BDD-scale runs do not show the pathology
  but learned loss balancing (e.g. uncertainty weighting) is still a
  reasonable ablation.
- **FW-002** — **Longer training**. Now partially resolved — full
  BDD runs are in place with EarlyStopping. Remaining work: ablate
  against other recipes.
- **FW-003** — **EMA target encoder** variant as an ablation against
  live targets + SIGReg.
- **FW-004** — **Predict-the-mean shortcut probe.** Verify `L_mask`
  is decreasing via per-position semantic recovery and not via a
  trivial latent-mean predictor.

Newly surfaced directions from this session:

- **Downstream linear probe on BDD100K labels** — most informative
  next step. Turns "training converged" into "features are useful".
- **Decord or hardware MOV decode** — unblock GPU utilization at
  T≥16 or multi-hour epoch throughput.
- **Larger model sweep** — current config is 865k params. Grid over
  `embed_dim ∈ {128, 256, 384}` and `encoder_clifford_depth ∈ {2, 4}`
  to see how val_loss scales.
- **Per-patch surprise callback for streaming eval** — like
  `PatchPredictionErrorCallback` but driven by `stream_step` on a
  held-out clip.
- **Reintroduce conditioning as an input, not as AdaLN** — if a
  future dataset has useful control signals, consider concatenating
  conditioning tokens rather than rebuilding AdaLN-zero.

Out of scope for the current trainer:

- Mixed-precision (bf16) and multi-GPU training.
- Ablations: SIGReg on/off, Clifford-depth sweep, spatial-only vs
  spatiotemporal tube masks.

## References

- **Closed plans (this session)**:
  - `plans/plan_2026-04-22_4f29c76f/` — telemetry-A strip + BDD100K
    loader.
  - `plans/plan_2026-04-22_016e549b/` — training visualization via
    callback reuse.
- **Earlier plans**: `plan_2026-04-21_421088a1/` — iter-2 tube masking
  + dual-loss training (original architecture decisions D-001 … D-012,
  FW-001 … FW-004). Kept as historical context; the AdaLN-zero
  conditioning described there has been removed.
- **Consolidated cross-plan summaries**: `plans/DECISIONS.md`,
  `plans/FINDINGS.md`, `plans/LESSONS.md`.
- **Model package CLAUDE.md**: `src/dl_techniques/models/video_jepa/`
  and module docstrings.
- **Callbacks reused**:
  `src/dl_techniques/callbacks/training_curves.py`,
  `src/dl_techniques/callbacks/jepa_visualization.py`.
- **Upstream LeWM port**: `src/train/lewm/` — single-CLS JEPA, same
  SIGReg primitive, simpler predictor. The video-JEPA model generalizes
  it to patch level.
- **Clifford primitives**:
  `src/dl_techniques/layers/geometric/clifford_block.py` —
  `CliffordNetBlock`, `CausalCliffordNetBlock`,
  `SparseRollingGeometricProduct`, `GatedGeometricResidual`.
