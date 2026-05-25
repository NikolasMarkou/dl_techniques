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
- No pixel decoder — outputs live in `(B, T, 14, 14, 128)` latent
  patch space.

Streaming inference is supported via a rolling buffer of the last K
patch grids (O(1) amortized per frame).

## Loss

```
total = λ_next · Σ_{h∈H} L_pred_h  +  λ_mask · L_mask  +  λ_sigreg · L_sigreg
```

Defaults: `H = predict_horizons = (1, 4, 15)`, `λ_next=1.0` per horizon,
`λ_mask=1.0`, `λ_sigreg=0.09`, `T = num_frames = 24`.

- `L_pred_h` — **multi-horizon prediction**: for each horizon `h ∈ H`,
  a linear pointwise `Dense(D, use_bias=False)` head named
  `pred_head_h{h}` projects the shared causal predictor output, and
  the loss is `MSE(pred_head_h(pred[:, :-h]), z[:, h:])` over unmasked
  positions. The shared causal predictor is unchanged across horizons;
  per-horizon heads provide an independent linear sub-objective per `h`.
  The same `λ_next` weight is applied to every horizon (NOT split-equally
  across `|H|`) — this keeps per-horizon contribution stable when the
  horizon set changes.
- `L_mask` — MSE on masked patch positions between predictor output
  and the same encoder's output at the masked slots. Latent-space
  masking only — pixel-space is structurally incompatible with the
  Clifford encoder's intact-grid requirement (hard constraint).
- `L_sigreg` — Epps-Pulley Gaussianity statistic on `(B·T, N, D)`
  reshape, averaged over projections. Default `sigreg_num_proj=64`
  for smoke; use `1024` for real training.

The model exposes per-horizon trackers `next_frame_loss_h{h}` (one per
`h ∈ H`) **plus** a combined `next_frame_loss` tracker (mean across
horizons — kept under that name for CSV back-compat with iter-2 logs),
`mask_loss`, and `sigreg_loss` via its `metrics` property, so
`CSVLogger` automatically writes a column per horizon next to the
aggregated `loss`. Validation produces the same columns prefixed
with `val_`.

### Why multi-horizon

Single-horizon `t+1` next-frame prediction is degenerate on dashcam
video at 30 fps: at that frame rate, `z[t+1] ≈ z[t]`, so the trivial
identity predictor (`pred(z) = z`) is hard to beat. In the iter-1
single-horizon run, the identity baseline beat the trained model 46×
in latent MSE and training was halted at epoch 23. Adding longer
horizons (`h=4` ≈ 130 ms, `h=15` ≈ 500 ms at 30 fps) gives the
predictor sub-objectives where identity *cannot* be the optimum, while
keeping the easy `h=1` head as a smoothness regularizer. The shared
causal predictor is unchanged; only `|H|` cheap linear heads
(`~3·D²` extra params at default `|H|=3, D=128`) are added.

See `plan_2026-05-23_0b664700` / `D-001` for the full design rationale
and the rejected alternatives (horizon-conditioned predictor with
horizon embedding; equal-split `λ_next/|H|`).

### Why EMA target

Multi-horizon eval on the post-multi-horizon checkpoint surfaced a
deeper pathology: the trained model was **84-300x WORSE than the
trivial identity predictor at every horizon**. Root cause: the
**live** target encoder (`z_target == z_online`, single encoder
shared between predictor input and prediction target) plus SIGReg
co-adapted the encoder and predictor into a near-time-invariant
feature map (`z_t ≈ z_{t+15}`). At that point the trivial identity
predictor is the global optimum of the next-frame objective in
latent space, and the trained model cannot beat it. SIGReg
prevents rank-collapse (the failure mode it was designed for); it
does not prevent time-invariance — a distinct failure mode.

Fix: switched to a **V-JEPA / BYOL / DINO-style EMA target encoder**
with `stop_gradient` on the target branch
(`plan_2026-05-23_15151c75/D-001`). Architecture:

- A second `VideoJEPACliffordEncoder` (`self.target_encoder`) with
  identical architecture to `self.encoder`. `trainable=False` so its
  weights are excluded from `model.trainable_variables` and the
  optimizer never sees them.
- Online encoder feeds predictor + SIGReg.
- Target encoder produces `z_target` (with `ops.stop_gradient`)
  consumed as the regression target for both `L_pred_h` and
  `L_mask`.
- After each `optimizer.apply_gradients`, a custom `train_step`
  EMA-copies `target_w <- m * target_w + (1 - m) * encoder_w`.
  Default `m = 0.996` (V-JEPA / BYOL).
- Optional `--ema-schedule cosine` ramps `m` from `--ema-momentum`
  to `1.0` across the run; current value is exposed as the `ema_m`
  metric. The step counter is a non-trainable weight so cosine
  progress survives `.keras` checkpoint reload.
- Note: `ema_m` logs the **EMA momentum scalar** applied at this
  step (per `ema_schedule`), NOT the divergence between online
  and target encoders.
- `ema_divergence` — the companion metric to `ema_m`. It is the
  weight-space L2 ratio
  `||W_target - W_online||_2 / ||W_online||_2` over the paired
  target/online encoder weights (BYOL/MoCo Option A). Cold-start
  ≈ 0 (the constructor bitwise-syncs target ← encoder); asymptotic
  0.01–0.3 in healthy SSL runs; sustained > 1.0 is the published
  collapse signal. Measured on a 100-step BDD100K diagnostic at
  momentum 0.996: 0.001 → 0.120, accumulating monotonically inside
  the healthy band — the online encoder is doing real gradient
  work, the target lags by design, no collapse. Logged every
  `train_step`; lands in `training_log.csv` and
  `training_curves/other.png` next to `ema_m`.
- SIGReg input was moved from `pred` (predictor output) to
  `z_online` (encoder output), per
  `plan_2026-05-23_15151c75/D-002` — regularization targets the
  representation directly under the JEPA framing, and the target
  encoder is not regularized because it carries no gradient.

The EMA target reverses the explicit iter-1 D-001 (live target,
inherited from LeWM); LeWM's single-CLS static-pool architecture did
not face the time-invariance failure mode that the 30 fps
frame-grid video setting surfaces. The reversal is anchored in
source as `# DECISION plan_2026-05-23_15151c75/D-001`.

Concrete numbers from this session:

- iter-1 (single-horizon, live target): identity baseline beat the trained model **46x** at h=1.
- iter-2 (multi-horizon, live target): identity baseline beat the trained model **84-300x** across h in {1, 4, 15}, AND all per-horizon heads collapsed to the same value (around 0.0046).
- iter-3 (multi-horizon + EMA, momentum=0.996): identity-baseline gap at h=15 dropped to **2.6x** by epoch 16; the trained model beats cross-encoder identity at every horizon. EMA is what made multi-horizon viable.

## Graph-mode tracing

`VideoJEPA` is fully usable from standard `keras.Model.fit` and from
eager `model(...)` calls. There is one residual sharp edge when
wrapping the model in your own `@tf.function`:

- `model.fit(...)` — passes Python `True`/`False` for `training`.
  Fully graph-safe. No action needed.
- `@tf.function`-wrapped inference with `training=None` or Python
  bool — OK at every configuration.
- `@tf.function`-wrapped call with `training=tf.constant(True/False)`:
  - At production default `dropout=0.0` — OK (the predictor's
    `mlp_drop` is `None`; mask gate uses Python identity
    `training is True`). Anchored as `# DECISION
    plan_2026-05-24_ca745a6c/D-005`.
  - At `dropout > 0` — raises `OperatorNotAllowedInGraphError`
    inside `keras.layers.Dropout.call`. This is a generic Keras 3
    limitation, not specific to this model. Workaround: pass a
    Python `True`/`False` instead of a symbolic tensor — which is
    what `keras.Model.fit` does.

Sibling anchor: `# DECISION plan_2026-05-24_ca745a6c/D-003` covers
the analogous mask-gate constraint in `model.py`.

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
    --T 24 --predict-horizons 1 4 15 \
    --img-size 112 --patch-size 8 --embed-dim 128 \
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
    --batch-size 4 --T 24 --predict-horizons 1 4 15 \
    --img-size 112 --patch-size 8 --embed-dim 128 \
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
| `--lambda-next-frame`, `--lambda-mask` | `1.0, 1.0` | loss weights; applied **per horizon** for `--lambda-next-frame`. |
| `--predict-horizons` | `1 4 15` | strictly-positive, sorted ascending, unique; `max(h) < T`. `--smoke` overrides to `1 2`. |
| `--ema-momentum` | `0.996` | V-JEPA / BYOL default for the EMA target encoder. Strict bound `[0.0, 1.0)`. |
| `--ema-schedule` | `none` | `none` holds m constant; `cosine` ramps from `--ema-momentum` to `1.0` over the run. Logged as metric `ema_m`. |
| `--T` | `24` | frames per clip; raised from `8` so multi-horizon at `h=15` (~0.5s @ 30fps) fits. |
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

### Per-step loss dynamics (100-step BDD100K diagnostic)

A 100-optimizer-step BDD100K run at smoke geometry (img=64, T=4, B=2,
RTX 4090, 7m15s wall-clock) gives a per-step CSV that resolves what
the epoch-grain log above smears together. All four learned signals
descend, in different regimes:

| Component | step 0 | step 99 | ratio | role in the descent |
|---|---|---|---|---|
| `next_frame_loss` (mean over h) | 0.876 | 0.084 | **10.4×** | fastest — per-horizon `Dense` heads find a useful linear projection of the predictor output almost immediately |
| `mask_loss` | 0.534 | 0.092 | **5.8×** | mid — requires the encoder to produce features predictable from neighbouring spatial context; a genuine representation-learning signal, not just temporal extrapolation |
| `sigreg_loss` | 38.5 | 18.9 | 2.0× | slowest in ratio — already two orders above the predictive losses in absolute terms; the `λ_sigreg=0.09` weight balances it back into the same order of magnitude in gradient contribution |
| `ema_divergence` | 0.001 | 0.120 | grows | not a loss — the encoder ↔ target gap, accumulating monotonically inside the published 0.01–0.3 healthy band |

**Gradient flow.** The three predictive components live in different
parts of the loss landscape and the ordering of their descent rates is
the right one: cheap-projection heads first, representation-learning
mask term second, regularizer last. No component is starving any other.
The fact that `mask_loss` is descending 5.8× while `sigreg_loss` only
manages 2.0× is direct evidence that the regularizer is not crowding
out the mask-prediction gradient — it would be the opposite ordering
if it were. `ema_divergence` rising in lockstep with the predictive
losses says the online encoder is doing real gradient work and the
target encoder is lagging by design (at momentum 0.996, a single step
moves the target by 0.4% of the online update, so the gap accumulates
naturally). Nothing here looks like collapse — collapse manifests as
`ema_divergence` exploding past 1.0 once the encoder finds a
degenerate fixed point the EMA can't track.

**What this falsifies.** An earlier 8-step BDD100K smoke had recorded
`mask_loss` apparently flat near 0.52, which seeded five candidate
explanations for stagnation. At 100 steps, all five are falsified by
the same evidence:

- *Sigreg dominance crowding out mask*: ruled out — `mask_loss` makes
  5.8× progress while sigreg manages 2.0×.
- *Zero mask token + slow EMA target*: ruled out — both the mask loss
  and the EMA gap evolve at healthy rates.
- *Predictor depth=2 capacity bottleneck*: ruled out — both predictive
  losses descend; capacity is not the constraint at this scale.
- *`λ_next` dilution under multi-horizon*: ruled out — `next_frame_loss`
  leads the descent, not lags it.
- *Genuine slow convergence*: reframed — the 8-step "flatness" was
  below the noise floor of the optimizer + I/O pipeline. Most of the
  relative drop happens inside the first ~30 steps; the 8-step window
  ended before the signal cleared the noise.

Net: the loss surface at this geometry is well-behaved, mask
prediction works without rebalancing, and the smoke-regime alarm was
a statistical-resolution artifact.

**Caveat.** This is smoke geometry (`img=64, T=4, sigreg_num_proj=64`).
Loss balance at full resolution (`img=112, T=24, sigreg_num_proj=1024`)
is not guaranteed identical, but the full-run signal documented above
(mask and next-frame descending proportionally over 50+ epochs) is
consistent with this finding at a different geometry.

## Training results

Four BDD100K runs on RTX 4090 (GPU 0), batch=4, monomic-precision.

| Run | Output dir | Config | Result | Verdict |
|---|---|---|---|---|
| iter-1 single-horizon, no-EMA | `results/video_jepa_20260523_091820/` | T=8, embed=128, depth=2, h=1, sigreg_proj=1024, 23 epochs (stopped) | h1=0.0014 train loss; identity-baseline eval: model 46x WORSE than identity at h=1 | FAILED: predictor degenerates under live target encoder; identity unbeatable |
| iter-2 multi-horizon, no-EMA | `results/video_jepa_20260523_144517/` | T=24, horizons=[1,4,15], 3 epochs (stopped) | All heads collapsed to same value (~0.0046); identity-baseline: 84-300x worse | FAILED: per-horizon heads have no symmetry-breaking signal when targets are too smooth |
| iter-3 multi-horizon + EMA | `results/video_jepa_20260523_161131/` | T=24, h=[1,4,15], ema_momentum=0.996, 24 epochs (stopped) | h15=0.0025 plateau; identity-baseline at epoch 16: model 2.6x worse at h=15 (vs 84x in iter-2); model BEATS cross-encoder identity at all horizons | SUCCESS: EMA decoupling fixed the identity-baseline pathology. Real predictive signal. |
| iter-4 hires + EMA + bigger | `results/video_jepa_20260523_234811/` | img=224, patch=16, embed=192, depths=4/4, T=24, h=[8,15], 30 epochs completed | h8=h15 approx 8.6e-4 final; mask=1.1e-3; sigreg=0.44 | TRAINED OK but reload check FAILED with max\|delta\|=8.30. See Known issues. |

Headline arc: live target encoder + SIGReg is unable to produce a model that beats the trivial identity predictor on dashcam video at 30 fps. The EMA target encoder (V-JEPA / BYOL / DINO recipe) is the fix that turned the 84-300x identity-baseline gap into 2.6x at h=15. Multi-horizon heads alone do not break the symmetry; without EMA decoupling, per-horizon heads collapse to the same value because the live target shifts with the predictor.

## Known issues / caveats

- **Reload-check failure ("hires bug") — FIXED in `plan_2026-05-24_ca745a6c`.** Iter-4 reported `max|delta|=8.30` between the in-memory model and the reloaded model at the scaled config and was originally attributed to the EMA `target_encoder` eager dummy-sync interacting with `from_config`. That diagnosis was wrong. The actual root cause is **scale-independent**: `TubeMaskGenerator` calls unseeded `keras.random.uniform`, and tube-mask substitution was running under `training=False`, which made `model(x, training=False)` self-non-deterministic for any `mask_prediction_enabled=True` config (reproduced on a fresh untrained model at small AND scaled configs with `max|delta|≈1.10`; scaled-config trained weights amplify it to ≈8.30). Fix: mask substitution is now gated on `training=True` (the V-JEPA/MAE convention — masks are a pretraining augmentation, not an inference contract). Anchored at `# DECISION plan_2026-05-24_ca745a6c/D-001` in `src/dl_techniques/models/video_jepa/model.py`. Locked in by `test_inference_determinism_with_masking`, `test_serialization_forward_parity_with_masking`, and `test_end_to_end_fit_and_reload_with_masking`. Both encoders are now reload-deterministic — no downstream workaround required.
- **Multi-horizon head collapse without EMA.** Per-horizon `Dense(D, no bias)` heads on top of a shared causal predictor do NOT break time-invariance symmetry on their own. When the target encoder is live (no EMA), all heads converge to the same numerical value (iter-2: all heads at approximately 0.0046). EMA target encoder is a necessary ingredient, not an optional ablation, when training multi-horizon JEPA at 30 fps. See `plan_2026-05-23_15151c75` and "Why EMA target" above.
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
- **FW-003** — **EMA target encoder** — **DONE in `plan_2026-05-23_15151c75`**.
  No longer an ablation: live target + SIGReg failed the multi-horizon
  identity-baseline test (84-300x worse than identity), and the EMA
  variant is now the default. See "Why EMA target" above and the Deep
  Review "Fixed" entry below.
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

## Deep Review

Audit mirroring the LeWM deep review (`plan_2026-05-23_692fd5e5`) applied
to `src/train/video_jepa/`. Items below were the punch list from reading
the trainer + dataset loaders + model package end-to-end.

### Pattern Conformance

- Trainer is **not** Pattern 1/2/3/4/5 in `src/train/CLAUDE.md`. Like
  LeWM, it rolls its own callback list because the shared
  `EpochAnalyzerCallback` does not understand dict inputs or
  `add_loss`-only training. Acceptable trade-off — same reasoning.
- Uses `setup_gpu(args.gpu)` from `train.common`.
- Writes to `results/` at repo root (matches `feedback_results_dir`).
- No `kernel_regularizer=L2(...)` alongside AdamW — avoids the
  double-weight-decay foot-gun.

### Issues found (before fixes)

1. **No CLI arg validation.** `_build_config` performed zero CLI-side
   validation. `embed_dim` must be EVEN (encoder asserts `D % 2 == 0`
   for sine2D PE) and must equal `predictor_num_heads * predictor_dim_head`
   (temporal MHA wiring) — both unenforced, fail-late deep inside build.
2. **No reload-after-save round-trip check.** Trainer saved
   `final_model.keras` and stopped. Any serialization regression would
   sail past CI silently.
3. **`rollout` silent broadcast — N/A.** Video-JEPA has no
   `rollout(pixels_history, action_sequence)` method (no actions).
   `stream_step(frame)` is shape-explicit `(B, H, W, C)` per call.
   Marked N/A for this category.
4. **Dataset code smells (BDD100K loader):**
   - `bdd100k_video_dataset.num_steps` parameter is a footgun:
     `.take(num_steps)` exhausted the dataset after epoch 1; the trainer
     stopped passing it (README "What changed" section), but the
     parameter remained on the public API.
   - `np.random.seed(seed)` mutated the global numpy RNG process-wide,
     just so `_read_clip` could use `np.random.randint`.
5. **No `--smoke` preset.** Trainer defaults were smoke-sized; the
   README's full-spec config required the user to override 8+ flags
   manually.
6. **`create_base_argument_parser` not adopted.** Local argparse
   repeated `--epochs`, `--batch-size`, `--learning-rate`,
   `--weight-decay`, `--gpu`, `--output-dir`. Drift from sibling
   trainers and LeWM.
7. **Zero coverage at `tests/test_train/test_video_jepa/`.** Model
   tests at `tests/test_models/test_video_jepa/` are thorough (33+
   tests), but the trainer script itself had no regression coverage.
8. **README lacked Deep Review section.** Verified-behavior section
   existed; the audit / issues / fixed trail did not.

### Intentionally left / N/A

- **Issue 3 (rollout S>1)** — N/A. video_jepa has no rollout method.
- **Per-patch (no CLS pool) output** — intentional. `(B,T,H_p,W_p,D)`
  is the contract.
- **`--T` controls both `num_frames` and `history_size_k`** — by
  design. If a user wants `K > T`, they edit code; low value to add a
  flag.
- **`MPLBACKEND=Agg` belt-and-braces** at import time — harmless,
  matches LeWM.
- **`metrics` property already exposes per-loss trackers**
  (`next_frame_loss`, `mask_loss`, `sigreg_loss`) — same shape as
  LeWM post-iter1, nothing to change.

### Fixed in plan_2026-05-23_c573e591

- **Issue 1 (arg validation)**: `_validate_args(args)` called from
  `main()` rejects `img_size % patch_size != 0`, odd `embed_dim`,
  `embed_dim != predictor_num_heads * predictor_dim_head`, and
  `batch_size < 2`.
- **Issue 2 (reload check)**: trainer now reloads `final_model.keras`,
  forward-pass-diffs against the in-memory model, raises `RuntimeError`
  and `sys.exit(1)` on `max|delta| >= 1e-4` or any reload exception.
- **Issue 4 (BDD100K loader)**:
  - Removed `num_steps` parameter and the `.take(num_steps)` call.
    Anchored as `# DECISION plan_2026-05-23_c573e591/D-001` in source.
    Grep confirmed no external callers passed `num_steps=...`.
  - Removed global `np.random.seed(seed)` side effect; loader now
    threads a local `np.random.default_rng(seed)` into
    `_read_clip(rng=...)`.
- **Issue 5 (`--smoke` preset)**: defaults promoted to BDD full-spec
  (`T=8, img=112, embed=128, batch=4, sigreg_num_proj=1024, epochs=100,
  steps_per_epoch=1000`). New `--smoke` flag applies the prior
  smoke-sized config (`synthetic, T=4, img=64, patch=8, embed=64,
  depth=2, batch=2, epochs=2, steps_per_epoch=4, sigreg_num_proj=64`).
  User-provided flags still win.
- **Issue 6 (`create_base_argument_parser`)**: trainer now extends
  the base parser; `--dataset` repurposed to
  `{synthetic, bdd100k}`; common flags (`--epochs`, `--batch-size`,
  `--learning-rate`, `--weight-decay`, `--gpu`) come from the base
  parser. Video-jepa-specific flags layered on top.
- **Issue 7 (regression tests)**: new
  `tests/test_train/test_video_jepa/test_train_video_jepa.py` covers
  arg validation (4 cases), `--smoke` preset application, and an
  integration-marked end-to-end 1-step fit + reload round-trip.
  Runtime <60s on CPU.
- **Issue 8 (Deep Review section)**: this section.

### Fixed in plan_2026-05-23_15151c75 (EMA target encoder)

- **Identity-baseline pathology at every horizon.** Multi-horizon eval
  on the post-multi-horizon checkpoint showed the trained model
  84-300x WORSE than the trivial identity predictor at h ∈ {1, 4, 15}.
  Root cause: the live target encoder + SIGReg co-adapted into a
  near-time-invariant latent space (z_t ≈ z_{t+15}), making identity
  the global optimum of the next-frame objective. SIGReg prevents
  rank-collapse, not time-invariance. Fix: V-JEPA-style EMA target
  encoder with stop-gradient, custom `train_step` that EMA-updates
  the target after each `optimizer.apply_gradients`, default
  momentum 0.996, optional cosine schedule. Online encoder feeds
  predictor + SIGReg; target encoder produces the regression target
  for both per-horizon next-frame loss and mask loss. Anchored as
  `# DECISION plan_2026-05-23_15151c75/D-001` in
  `src/dl_techniques/models/video_jepa/model.py`. New trainer flags
  `--ema-momentum`, `--ema-schedule`; the cosine step counter is a
  non-trainable weight so schedule progress survives `.keras` reload.
  Also (sub-decision D-002): SIGReg input switched from `pred` to
  `z_online` so it regularizes the representation directly under the
  JEPA framing. Six new regression tests under
  `tests/test_models/test_video_jepa/` cover initial sync,
  not-trainable, EMA math after one step, no gradient through target,
  serialization round-trip of both encoders, and cosine schedule
  monotonicity. All 47 model tests + 20 trainer tests pass.

### Fixed in plan_2026-05-23_0b664700 (multi-horizon)

- **Single-horizon `t+1` degeneracy at 30 fps.** Identity baseline
  beat the iter-1 trained model 46× in latent MSE (training stopped
  at epoch 23). The objective was a degenerate target on dashcam
  video at frame rate, not a model-capacity problem. Fix: replaced
  with multi-horizon prediction `H = (1, 4, 15)` via per-horizon
  linear `Dense(D, no bias)` heads on top of the **unchanged** shared
  causal predictor. Anchored as
  `# DECISION plan_2026-05-23_0b664700/D-001`. New trainer flag
  `--predict-horizons`; `--T` default raised `8 → 24` so `h=15` fits.
  Per-horizon trackers `next_frame_loss_h{h}` plus a combined
  `next_frame_loss` (mean across horizons; kept under that name for
  CSV back-compat). New regression tests cover arg validation,
  multi-horizon forward (3 trackers + combined), and an
  integration-marked fit + reload round-trip that confirms all
  per-horizon trackers survive serialization. Encoder, SIGReg, mask
  token, and mask loss are untouched.

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
