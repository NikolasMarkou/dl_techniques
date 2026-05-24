# LeWM Training Script

Smoke-test trainer for **LeWM** (Learning the World with Minimal Supervision),
a JEPA-style action-conditioned world model. Trains on a synthetic random
dataset by default; an HDF5 PushT loader skeleton is provided for real data.

- **Script**: `src/train/lewm/train_lewm.py`
- **Model**: `src/dl_techniques/models/lewm/`
- **Dataset producers**: `src/dl_techniques/datasets/pusht_hdf5.py`
- **Loss regularizer**: `src/dl_techniques/regularizers/sigreg.py`
- **Upstream reference**: Sobal et al., 2024 (LeWM) — PyTorch source at `/tmp/lewm_source/`

This script is explicitly framed by its author as a **smoke test** — it
validates the end-to-end pipeline (encoder, action embedder, AR predictor,
SIGReg, optimizer step, serialization round-trip), not a production training
recipe. Defaults target a 56x56 / 2-step / 64-projection configuration that
runs in seconds on CPU.

---

## Model Overview

`LeWM` (`src/dl_techniques/models/lewm/model.py`) is a self-supervised
world model that predicts future visual embeddings conditioned on actions.

### Components

| Component | File | Role |
|-----------|------|------|
| `ViT` encoder | `dl_techniques.models.vit.model` | Per-frame patch encoder, CLS-pooled. Default scale `tiny` (192d). |
| `MLPProjector` (`projector`) | `lewm/projector.py` | Refines encoder features: `Linear -> LayerNorm -> GELU -> Linear`. Identity-shaped (192->192) by default. |
| `ActionEmbedder` (`action_encoder`) | `lewm/embedder.py` | `Conv1D(k=1) -> Dense -> SiLU -> Dense`, maps `(B,T,A) -> (B,T,D)`. |
| `ARPredictor` (`predictor`) | `lewm/predictor.py` | Stack of `AdaLNZeroConditionalBlock` transformer blocks with learned positional embedding (init stddev 0.02). Conditions on action embeddings. |
| `MLPProjector` (`pred_proj`) | `lewm/projector.py` | Post-prediction projection mirror of `projector`. |
| `SIGRegLayer` | `regularizers/sigreg.py` | Sketch Isotropic Gaussian Regularizer, applied to projected embeddings shaped `(T, B, D)`. |

### Forward Contract (`LeWM.call`)

Inputs: `dict` with
- `pixels`: `(B, T, H, W, C)` float — history + future frames, ImageNet-normalized.
- `action`: `(B, T-1, A)` float — actions between successive frames.

Returns: predicted embeddings `(B, T, D)`.

Losses are added via `self.add_loss()` inside `call`:
- **MSE prediction loss**: `mean((pred_emb[:, :-1] - emb[:, 1:])**2)` — predict next-frame embedding from current.
- **SIGReg loss**: applied on `transpose(emb, (1,0,2))`, weighted by `config.sigreg_weight` (default 0.09).

Per-component trackers `pred_loss` and `sigreg_loss` are exposed as Keras
metrics so they appear next to `loss` in the CSV log.

### Anchored Design Decisions

- **D-001**: target encoder is **live** (no EMA, no `stop_gradient`).
  Gradients flow through both context and target paths, matching upstream
  LeWM (distinct from BYOL/DINO/JEPA conventions). See model.py call().
  Note: the sibling `src/train/video_jepa/` model now diverges further
  from LeWM by adopting an EMA target encoder with `stop_gradient`
  (`plan_2026-05-23_15151c75/D-001`), because the patch-grid 30 fps video
  setting hit a time-invariance failure mode that single-CLS LeWM does not.
- **D-002**: `MLPProjector` uses **LayerNorm**, not BatchNorm. Follows
  upstream `MLP(norm_fn=nn.LayerNorm)`. Sidesteps BN-batch-of-1 failures.
- **D-002 (config)**: `num_frames` is a serialized field with a sentinel
  `0` that gets derived to `history_size + num_preds` in `__post_init__`.
  Explicit values must cover the training sequence length or `__post_init__`
  raises.

### Inference: `rollout(pixels_history, action_sequence)`

Autoregressive rollout from a history of pixel observations.

- Inputs: `pixels_history` `(B, S, HS, H, W, C)`, `action_sequence` `(B, S, T, A)`.
- Output: `predicted_emb` shaped `(B, S, T+1, D)`.
- **S must equal 1.** Only `pixels_history[:, 0]` is encoded; passing
  `S > 1` raises `ValueError` (DECISION plan_2026-05-23_692fd5e5/D-001).
  Tile externally or call `rollout` once per history.
- The first `HS` time entries are encoder-derived; the remainder are
  predictor-derived. Score only the predictor-derived tail against ground
  truth.
- Eager-only (Python `for` over `n_steps`). Raises if `T < HS` or `S != 1`.

---

## Training Pipeline

### Loss Wiring

The model uses `self.add_loss()` inside `call`, so it is compiled with
`loss=None`. `jit_compile=False` avoids XLA tracing issues with the dynamic
rollout / add_loss path.

### Dataset Schema

Both producers emit:

```python
({"pixels": (B, T, H, W, C), "action": (B, T-1, A)}, dummy_y)
```

where `T = history_size + num_preds`. `dummy_y` is a zero scalar placeholder
required for `model.fit` with `loss=None`. Datasets are `.repeat()`ed so that
`steps_per_epoch * epochs` budgets exceeding the underlying sample count do
not crash mid-fit.

### Callbacks

Intentionally minimal (does NOT use `train.common.create_callbacks` — the
shared `EpochAnalyzerCallback` does not understand dict inputs / `add_loss`):

- `TerminateOnNaN`
- `CSVLogger(training_log.csv)`
- `ModelCheckpoint(last.keras, save_best_only=False)`

### Outputs

Written under `results/lewm_<YYYYMMDD_HHMMSS>/`:

- `training_log.csv` — per-epoch `loss`, `pred_loss`, `sigreg_loss`.
- `last.keras` — checkpoint at end of every epoch.
- `final_model.keras` — explicit final save.

After training, the script reloads `final_model.keras`, runs a forward pass
on one batch, and compares against the original model's output. Logs PASSED
if `max|delta| < 1e-4`. On failure (delta too large, or any reload
exception) the script logs the error and exits with status 1 so CI catches
serialization regressions.

---

## Usage

### Synthetic Smoke Test

```bash
MPLBACKEND=Agg .venv/bin/python -m train.lewm.train_lewm --smoke --synthetic
```

### Full-Spec Build (upstream defaults — now the script defaults)

```bash
MPLBACKEND=Agg .venv/bin/python -m train.lewm.train_lewm --synthetic --gpu 0
```

### Real Data (HDF5 PushT — UNTESTED)

```bash
MPLBACKEND=Agg .venv/bin/python -m train.lewm.train_lewm \
    --hdf5-path /path/to/pusht.h5 \
    --img-size 224 --batch-size 8 --epochs 10 --steps-per-epoch 100 \
    --gpu 0
```

The HDF5 schema expected (per upstream) is:
- `/pixels`: `(N, H0, W0, 3)` uint8
- `/action`: `(N, A)` float (NaN sentinels at episode breaks become 0)
- `/episode_ends`: int boundary indices

---

## CLI Arguments

Defaults track **upstream LeWM** (full-spec build). Use `--smoke` for the
fast CPU iteration preset. Common training flags come from
`train.common.create_base_argument_parser` (inherited `--dataset`,
`--image-size`, `--lr-schedule`, `--patience`, `--show-plots` are unused
by this script).

| Group | Flag | Default | Notes |
|-------|------|---------|-------|
| Preset | `--smoke` | False | Tiny CPU preset (see below). User flags still win. |
| Data | `--synthetic` | True (fallback) | Use random data. |
| Data | `--hdf5-path` | None | Mutually exclusive with `--synthetic`. |
| Train | `--batch-size` | 16 | |
| Train | `--epochs` | 50 | |
| Train | `--steps-per-epoch` | 200 | |
| Train | `--learning-rate` | 5e-5 | AdamW LR. |
| Train | `--weight-decay` | 1e-3 | AdamW WD. |
| Train | `--seed` | 42 | Seeds Python, NumPy, TF, Keras. |
| Train | `--gpu` | None | `setup_gpu(gpu)` from `train.common`. |
| Model | `--img-size` | 224 | |
| Model | `--patch-size` | 14 | |
| Model | `--encoder-scale` | `tiny` | ViT scale string. |
| Model | `--embed-dim` | 192 | Validated == `ViT.SCALE_CONFIGS[scale][0]`. |
| Model | `--history-size` | 3 | |
| Model | `--num-preds` | 1 | |
| Model | `--depth` | 6 | |
| Model | `--heads` | 16 | |
| Model | `--dim-head` | 64 | |
| Model | `--mlp-dim` | 2048 | |
| Model | `--dropout` | 0.0 | |
| Action | `--action-dim` | 2 | PushT = 2. |
| Action | `--smoothed-dim` | 10 | ActionEmbedder intermediate. |
| Action | `--mlp-scale` | 4 | |
| Data | `--frameskip` | 1 | HDF5 only. |
| SIGReg | `--sigreg-weight` | 0.09 | |
| SIGReg | `--sigreg-knots` | 17 | |
| SIGReg | `--sigreg-num-proj` | 1024 | |

`--smoke` overrides (only for defaults the user did not set):
`img_size=56, patch_size=14, encoder_scale=tiny, embed_dim=192,
history_size=2, num_preds=1, depth=2, heads=4, dim_head=48, mlp_dim=256,
sigreg_num_proj=64, batch_size=2, epochs=1, steps_per_epoch=2`.

If neither `--synthetic` nor `--hdf5-path` is supplied, `--synthetic` is
auto-enabled.

`_build_model` fail-fasts at the CLI level when:
- `img_size % patch_size != 0`,
- `encoder_scale` is not in `ViT.SCALE_CONFIGS`,
- `embed_dim != ViT.SCALE_CONFIGS[encoder_scale][0]` (the projector is
  identity-shaped; a mismatch would otherwise crash deep in the encoder).

---

## Dependencies

- `keras >= 3.8`, `tensorflow 2.18`, `numpy`
- `h5py` (only when `--hdf5-path` is used; imported lazily)
- Internal: `dl_techniques.models.vit`, `dl_techniques.layers.transformers.adaln_zero`,
  `dl_techniques.regularizers.sigreg`, `dl_techniques.utils.logger`,
  `train.common.setup_gpu`

---

## Deep Review

Original punch list from reading the script and every model component.
The "Fixed" subsection below records what landed in
`plan_2026-05-23_692fd5e5`.

### Pattern Conformance

- The script intentionally **does not** follow Pattern 1/2/3/4/5 in
  `src/train/CLAUDE.md`. It rolls its own minimal callback set because the
  shared `EpochAnalyzerCallback` does not understand dict inputs or
  `add_loss`-only training. The reason is documented at the call site
  (`_build_callbacks` docstring) — acceptable, but means the script is
  outside the analyzer ecosystem and will not produce the standard
  visualizations.
- It correctly uses `setup_gpu(args.gpu)` from `train.common`.
- It writes to `results/` at repo root (matches the user feedback memory
  `feedback_results_dir`).
- No `kernel_regularizer=L2(...)` is added alongside AdamW — avoids the
  double-weight-decay foot-gun called out in CLAUDE.md.

### Smells / Issues

1. **`--img-size`/`--patch-size`/`--encoder-scale` decoupling.** The
   defaults are 56/14/`tiny`, but `ViT(scale='tiny')` may have its own
   internal expectations on input shape and patch count. There is no
   assertion that `img_size % patch_size == 0` or that the chosen ViT
   variant tolerates a 56x56 input. A wrong combo will fail deep inside the
   encoder build, far from the CLI.

2. **`--embed-dim` is silently load-bearing.** It must equal the ViT
   encoder output dim for the projector identity assumption (model.py:75-80
   comment notes this). There is no check; setting `--embed-dim 256`
   silently produces a 256-dim projector input vs 192-dim encoder output
   and crashes at first matmul.

3. **`--num-frames` is not a CLI arg.** It is always derived as
   `history_size + num_preds`. Fine for training; but if a user wants the
   predictor's positional table to be larger than the training window (so
   `rollout` can extend beyond `T`), there is no way to do it from CLI.
   Currently `rollout` reuses positions `0..HS-1` via slicing — fine — but
   any future use case that wants longer rollouts than `num_frames` would
   silently OOB the positional embedding.

4. **Reload check uses `next(iter(dataset))` after `model.fit`.** The
   dataset is `.repeat()`ed but `next(iter(...))` creates a fresh iterator
   — that is fine, but means the first batch is materialized twice (once
   for fit warm-up, once here). For a dataset backed by `from_generator`
   with a seeded RNG, both iterations produce the same data, so the check
   is deterministic; for a stateful generator it could surprise.

5. **Reload check failure is logged, not raised.** Line 226-228 logs an
   error if `max|delta| >= 1e-4` but the script exits 0. CI / wrappers
   would not catch a regression in serialization. Consider exiting non-zero
   on failure, or at minimum setting a flag the caller can check.

6. **`rollout` silently broadcasts `pixels_history[:, 0]` over `S`.** The
   docstring warns about it, but no `ValueError` is raised when the caller
   passes distinct per-`S` histories — they are dropped without complaint.
   A `tf.debugging.assert_equal` (or a Python-side `if S > 1: log warning`)
   would surface this.

7. **`PushTHDF5Dataset` is **untested on real data**.** The author flags
   this in the docstring. Specific risks:
   - `tf.image.resize` is called from inside a Python generator via
     `.numpy()` — works on CPU eagerly but adds per-window TF kernel
     launch overhead. A `tf.data.Dataset.map` would be more idiomatic.
   - `_load_raw` reads the entire `/pixels` array into memory. PushT is
     small, but a larger HDF5 file would OOM.
   - No support for proprioceptive state despite the `proprio_key`
     parameter — it is stored but never consumed.

8. **`history_size = 2` default differs from upstream `3`.** OK for smoke
   testing, but easy to forget when comparing against published numbers.

9. **`sigreg_num_proj = 64` default vs upstream `1024`.** Same issue. The
   `--help` text calls this out, but it is the kind of default that
   silently makes training look noisier than it should at full scale.

10. **No `train.common.create_base_argument_parser` use.** This script
    repeats `--batch-size`, `--epochs`, `--learning-rate`, `--weight-decay`,
    `--seed`, `--gpu` locally. The base parser would have given consistent
    flag names. Not a bug, but drift from the rest of the codebase.

11. **`train.common.setup_gpu` is imported but the script also has
    `os.environ.setdefault("MPLBACKEND", "Agg")` at import time.** The
    CLAUDE.md convention is `MPLBACKEND=Agg` on the command line. Either is
    fine; both is belt-and-braces and harmless.

12. **No deterministic dataset shuffle seed.** `_set_seed` seeds the
    Python/NumPy/TF RNGs, but `synthetic_lewm_dataset` uses
    `np.random.default_rng(seed)` once at construction, then iterates
    deterministically. Fine. `PushTHDF5Dataset` has no shuffle whatsoever —
    windows are produced in file order — and there is no `.shuffle()` call
    anywhere. For real training this is a problem.

13. **`encode_pixels` rebuilds `H, W, C` from `self.config`, not from the
    actual input tensor shape.** If a caller passes mismatched pixel
    dimensions at inference, `ops.reshape` will fail with an opaque shape
    error rather than a `ValueError` naming the inputs.

14. **Missing CLI mirror for `emb_dropout`, `projector_hidden_dim`,
    `img_channels`.** All hardcoded in `_build_model` to either CLI or
    config defaults. Low priority.

### Suggestions

- Add an `assert img_size % patch_size == 0` and an
  `assert embed_dim == ViT(scale).embed_dim` check at the top of
  `_build_model`. Fail fast with a clear message.
- Promote the reload check to a `sys.exit(1)` on failure (or behind a
  `--strict-reload` flag) so CI catches regressions.
- Add `--shuffle-buffer` for the HDF5 path.
- Consider adopting `create_base_argument_parser(..., default_dataset=None)`
  and adding LeWM-specific flags on top, even if `--dataset` is ignored.
- Add a regression test under `tests/test_models/test_lewm/` that drives
  one step of `train_lewm.py` via `subprocess` against `--synthetic`
  defaults (the test suite already has the pieces; the script is small
  enough to invoke directly).
- If `rollout` distinct-per-S histories matter, change the input contract
  to `(B, HS, H, W, C)` and remove the `S` dimension from
  `pixels_history` entirely (callers can replicate themselves).

### What I did NOT verify

- Did not run the script (per instructions).
- Did not run the test suite.
- Did not inspect `AdaLNZeroConditionalBlock` internals.
- Did not check whether `ViT(scale='tiny', patch_size=14)` builds cleanly
  for `img_size=56`.
- Did not validate `PushTHDF5Dataset` against any real HDF5 file.

### Fixed in plan_2026-05-23_692fd5e5

- **Issue 1, 2 (--img-size/--patch-size/--encoder-scale/--embed-dim
  decoupling)**: `_build_model` now fail-fasts at the CLI on
  `img_size % patch_size != 0`, unknown `encoder_scale`, or `embed_dim`
  mismatch with `ViT.SCALE_CONFIGS[scale][0]`.
- **Issue 5 (reload check non-fatal)**: reload check now raises
  `RuntimeError` on `max|delta| >= 1e-4` and `sys.exit(1)` on any reload
  exception. Existing log output preserved.
- **Issue 6 (rollout silent broadcast)**: `rollout` now raises
  `ValueError` when `S != 1`. Anchored as
  `DECISION plan_2026-05-23_692fd5e5/D-001` at the check site.
- **Issue 7 (PushTHDF5Dataset)**:
  - Removed dead `proprio_key` parameter (grep confirmed zero callers).
  - Replaced eager full `/pixels` load with on-demand `h5py`-indexed
    reads inside the generator.
  - Added per-epoch index-level shuffle (seeded via new
    `shuffle_seed` parameter).
  - Class docstring marks **UNTESTED SKELETON**.
- **Issue 8, 9 (smoke defaults divergence)**: script defaults now match
  upstream (`history_size=3, depth=6, heads=16, dim_head=64, mlp_dim=2048,
  sigreg_num_proj=1024, img_size=224, batch_size=16, epochs=50,
  steps_per_epoch=200`). New `--smoke` flag applies the fast-CPU preset;
  explicit user flags still win.
- **Issue 10 (`create_base_argument_parser`)**: script now extends
  `train.common.create_base_argument_parser`; LeWM-specific flags layered
  on top.
- **New regression test** under `tests/test_train/test_lewm/`: arg
  validation (3 cases), rollout `S>1` guard, 1-epoch synthetic fit +
  reload round-trip. Runtime ~53s on CPU.

### Intentionally left

- **Issue 3 (`--num-frames` not on CLI)**: low-value; derived in
  `LeWMConfig.__post_init__`.
- **Issue 4 (reload check uses `next(iter(dataset))`)**: working as
  intended; synthetic dataset is deterministic per `--seed`.
- **Issue 11 (`MPLBACKEND=Agg` belt-and-braces)**: harmless.
- **Issue 12 (synthetic dataset shuffle determinism)**: addressed for
  HDF5 path; synthetic stays deterministic by design.
- **Issue 13 (`encode_pixels` rebuilds H,W,C from config)**: low priority.
- **Issue 14 (CLI mirrors for `emb_dropout`, `projector_hidden_dim`,
  `img_channels`)**: low priority.
