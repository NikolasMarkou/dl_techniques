# LeWM Training Script

Smoke-test trainer for **LeWM** (Learning the World with Minimal Supervision),
a JEPA-style action-conditioned world model. Trains on a synthetic random
dataset by default; an HDF5 PushT loader skeleton is provided for real data.

- **Script**: `src/train/lewm/train_lewm.py`
- **Model**: `src/dl_techniques/models/vision/lewm/`
- **Dataset producers**: `src/dl_techniques/datasets/pusht_hdf5.py`
- **Loss regularizer**: `src/dl_techniques/regularizers/sigreg.py`
- **Upstream reference**: Sobal et al., 2024 (LeWM), PyTorch

The script is a **smoke test**: it validates the end-to-end pipeline (encoder,
action embedder, AR predictor, SIGReg, optimizer step, serialization
round-trip), not a production training recipe. Its defaults track upstream LeWM
(224x224, `history_size=3`, `sigreg_num_proj=1024`); `--smoke` swaps in a
56x56 / 2-step / 64-projection preset that runs in seconds on CPU.

---

## Model Overview

`LeWM` (`src/dl_techniques/models/vision/lewm/model.py`) is a self-supervised
world model that predicts future visual embeddings conditioned on actions.

### Components

| Component | File | Role |
|-----------|------|------|
| `ViT` encoder | `dl_techniques.models.vision.vit.model` | Per-frame patch encoder, CLS-pooled. Default scale `tiny` (192d). |
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

### Design points worth knowing

- The target encoder is **live** (no EMA, no `stop_gradient`). Gradients flow
  through both context and target paths, matching upstream LeWM (distinct from
  BYOL/DINO/JEPA conventions). See `model.py`'s `call()`. The sibling
  `src/train/video_jepa/` model diverges here: it uses an EMA target encoder
  with `stop_gradient`, because the patch-grid 30 fps video setting hits a
  time-invariance failure mode that single-CLS LeWM does not.
- `MLPProjector` uses **LayerNorm**, not BatchNorm, following upstream
  `MLP(norm_fn=nn.LayerNorm)`. Sidesteps BN-batch-of-1 failures.
- `num_frames` is a serialized config field with a sentinel `0` that is derived
  to `history_size + num_preds` in `__post_init__`. Explicit values must cover
  the training sequence length or `__post_init__` raises.

### Inference: `rollout(pixels_history, action_sequence)`

Autoregressive rollout from a history of pixel observations.

- Inputs: `pixels_history` `(B, S, HS, H, W, C)`, `action_sequence` `(B, S, T, A)`.
- Output: `predicted_emb` shaped `(B, S, T+1, D)`.
- **S must equal 1.** Only `pixels_history[:, 0]` is encoded; passing
  `S > 1` raises `ValueError`. Tile externally or call `rollout` once per history.
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
| Model | `--dropout-rate` | 0.0 | |
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
- Internal: `dl_techniques.models.vision.vit`, `dl_techniques.layers.transformers.adaln_zero`,
  `dl_techniques.regularizers.sigreg`, `dl_techniques.utils.logger`,
  `train.common.setup_gpu`

---

## Known limitations

- **`PushTHDF5Dataset` is an UNTESTED SKELETON.** It has never been run against
  a real PushT HDF5 file. Windows are read on demand via `h5py` indexing with a
  per-epoch index-level shuffle (`shuffle_seed`); `tf.image.resize` is called
  inside the Python generator, which works but is not the idiomatic
  `tf.data.map` path.
- **Outside the analyzer ecosystem.** The script rolls its own minimal callback
  set (see Callbacks above), so it produces none of the standard analyzer
  visualizations.
- **`emb_dropout_rate`, `projector_hidden_dim` and `img_channels` have no CLI
  mirror** — they are set in `_build_model` from config defaults.
- **`num_frames` is not a CLI flag.** It is always `history_size + num_preds`,
  so a rollout longer than the training window would run past the predictor's
  positional table.
- **`encode_pixels` rebuilds `H, W, C` from the config, not from the input
  tensor**, so a mismatched pixel shape fails with an opaque reshape error
  rather than a named `ValueError`.
