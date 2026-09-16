# LeWM - Action-Conditioned World Model

A Keras 3 model that predicts future visual embeddings conditioned on actions,
in the JEPA family: a ViT encoder produces per-frame embeddings, and an
autoregressive conditional transformer predicts the next frame's embedding
from past embeddings plus the action taken. There is no pixel decoder and no
EMA target encoder — the same live encoder produces both the prediction
target and the context, and a SIGReg term keeps the embedding space from
collapsing instead.

**References**:
- Sobal et al., 2024. Learning the World with Minimal Supervision (LeWM).
- Assran et al., 2023. Self-Supervised Learning from Images with a Joint-
  Embedding Predictive Architecture (I-JEPA). CVPR 2023.
  (https://arxiv.org/abs/2301.08243)
- LeCun, 2022. A Path Towards Autonomous Machine Intelligence.
  (OpenReview: BZ5a1r-kVsf)
- Skean et al., 2025. SIGReg / hyperspherical-energy anti-collapse
  regularization, as implemented in `dl_techniques.layers.sigreg`.

## What's here

- `model.py::LeWM` — the top-level `keras.Model`, plus the `create_lewm(...)`
  factory. Registered as `dl_techniques.models.lewm.model`.
- `config.py::LeWMConfig` — the single dataclass holding every hyperparameter;
  there is no named-scale (`MODEL_VARIANTS`) table (see Config below).
- `embedder.py::ActionEmbedder` — maps raw per-timestep action vectors to the
  model's embedding space.
- `predictor.py::ARPredictor` — the autoregressive predictor: AdaLN-zero
  conditional transformer blocks conditioned on action embeddings.
- `projector.py::MLPProjector` — a small 2-layer MLP head, instantiated twice
  inside `LeWM` (`projector` and `pred_proj`, see Components below).

## Components

| Component | Attribute name | File | Role |
|-----------|-----------------|------|------|
| `ViT` encoder | `self.encoder` | `dl_techniques.models.vision.vit.model` | Per-frame patch encoder, CLS-pooled. `include_top=False`; default scale `tiny` (192d). |
| `MLPProjector` | `self.projector` | `projector.py` | Refines the encoder's CLS feature: `Dense -> LayerNorm -> GELU -> Dense`. Identity-shaped (`embed_dim -> embed_dim`) by default. |
| `ActionEmbedder` | `self.action_encoder` | `embedder.py` | `Conv1D(k=1) -> Dense -> SiLU -> Dense`, maps `(B, T, action_dim) -> (B, T, embed_dim)`. |
| `ARPredictor` | `self.predictor` | `predictor.py` | Stack of `AdaLNZeroConditionalBlock` transformer blocks (`dl_techniques.layers.transformers.adaln_zero`), with a learned positional embedding (init stddev 0.02) sized to `config.num_frames`. Conditions on the action embeddings. |
| `MLPProjector` | `self.pred_proj` | `projector.py` | A **second, independently-weighted instance of the same `MLPProjector` class** as `self.projector` — post-prediction projection, same architecture, not a shared/tied layer. |
| `SIGRegLayer` | `self.sigreg` | `dl_techniques.regularizers.sigreg` | Sketch Isotropic Gaussian Regularizer, applied to the encoder's own embeddings (not the prediction) to keep the embedding space from collapsing. |

`MLPProjector` is written once and instantiated twice under different names
(`projector`, `pred_proj`) with independent weights — reuse of the *class*,
not weight sharing between the two roles.

## Forward Contract (`LeWM.call`)

Inputs: a `dict` with
- `"pixels"`: `(B, T, H, W, C)` float — a history of `T` frames (the last one
  is the future frame the model is trained to predict).
- `"action"`: `(B, T-1, A)` float — the `T-1` actions taken between
  successive frames.

Returns: predicted embeddings `pred_emb` of shape `(B, T, D)`.

Two losses are added internally via `self.add_loss(...)` inside `call()`, so
`model.fit(...)` trains correctly with `loss=None` and no custom `train_step`:

- **MSE prediction loss** — `mean((pred_emb[:, :-1] - emb[:, 1:]) ** 2)`:
  predict the next frame's embedding from the current one.
- **Weighted SIGReg loss** — `SIGRegLayer` applied to
  `transpose(emb, (1, 0, 2))` (the SIGReg layer expects `(T, B, D)`),
  multiplied by `config.sigreg_weight` (default `0.09`).

`self.pred_loss_tracker` / `self.sigreg_loss_tracker` (`keras.metrics.Mean`)
expose both terms separately via the `metrics` property, so a `CSVLogger`
sees `pred_loss` and `sigreg_loss` alongside the summed `loss`.

## Design point: the encoder is live, not an EMA target

Unlike the sibling `models/vision/video_jepa/` package, `LeWM` uses **one
live encoder for both the prediction target and the context** — no EMA
shadow copy, no `stop_gradient` anywhere in `call()`. Gradients flow through
both paths, matching upstream LeWM. `video_jepa` instead uses an EMA target
encoder with `stop_gradient`, because its patch-grid, 30fps video setting
hits a time-invariance failure mode that single-CLS-token LeWM does not.
Anti-collapse is handled by the SIGReg term instead of the asymmetry an EMA
target would otherwise provide.

## `build()` re-derives shapes; it does not trace `call()`

`build()` repeats `call()`'s forward shapes through the same
`encode_pixels()` / `encode_actions()` / `predict_next()` helpers (skipping
only the loss tail — `add_loss` plus the trackers, which own no weights), but
it builds its own `keras.KerasTensor` placeholders from `self.config` rather
than tracing the real `call()` graph. This is deliberate: `add_loss` inside a
traced `call()` raises when the tensors involved are `KerasTensor`
placeholders, so `build()` cannot simply call `self(inputs)`. Consequently
`self.config` — not the shape of whatever `inputs` a caller happens to pass
to `build()` — determines the built weight shapes. `encode_pixels()`'s own
runtime shape guard (see below) validates the pixel tensor `call()` actually
receives against that same config, so a config/data mismatch is caught with a
clear error rather than surfacing as a silent wrong reshape.

## `rollout(pixels_history, action_sequence)` — autoregressive inference

Use `rollout()`, not `call()`, for autoregressive inference from an observed
history:

- Inputs: `pixels_history` `(B, S, HS, H, W, C)` (`HS = history_size`; only
  the `s = 0` plane is ever encoded) and `action_sequence`
  `(B, S, T, action_dim)` (full horizon `T >= HS`).
- Output: `{"predicted_emb": ...}` of shape `(B, S, T + 1, D)`. Every step the
  rollout produces is kept — the first `HS` entries along the time axis are
  encoder-derived embeddings of the observed history, and the remaining
  `T + 1 - HS` entries are predictor-derived. Score only the predictor-
  derived tail against ground truth.
- **`S` must equal 1.** Passing distinct per-sample histories (`S > 1`) would
  otherwise be silently dropped, since only `pixels_history[:, 0]` is
  encoded — `rollout` raises `ValueError` instead. Tile externally, or call
  `rollout` once per history.
- **Eager-only.** The autoregressive step loop is a plain Python `for` over
  `n_steps = T - HS`, so `rollout` cannot be traced/compiled as a graph op.
  It also raises `ValueError` if `T < HS`.

## `LeWMConfig` (`config.py`)

One dataclass holds every hyperparameter — there is no `MODEL_VARIANTS`
table. LeWM ships a single configuration mirroring the upstream defaults, and
is retuned field-by-field rather than by picking a named scale; the one scale
knob it does have is `encoder_scale`, forwarded straight to `ViT`.

| Field | Default | Notes |
|-------|---------|-------|
| `img_size` | `224` | |
| `patch_size` | `14` | |
| `img_channels` | `3` | |
| `encoder_scale` | `"tiny"` | `dl_techniques` ViT scale name; `"tiny"` = 192 dims / 3 heads / 12 layers. |
| `embed_dim` | `192` | Must equal `ViT.SCALE_CONFIGS[encoder_scale][0]` (enforced by callers such as `src/train/lewm/train_lewm.py::_build_model`, not by this dataclass itself). |
| `history_size` | `3` | |
| `num_preds` | `1` | |
| `num_frames` | `0` (sentinel) | Sizes the predictor's learned positional embedding; see below. |
| `depth` | `6` | `ARPredictor` block count. |
| `heads` | `16` | |
| `dim_head` | `64` | |
| `mlp_dim` | `2048` | |
| `dropout_rate` | `0.1` | |
| `emb_dropout_rate` | `0.0` | |
| `projector_hidden_dim` | `192` | Shared by both `projector` and `pred_proj`. |
| `action_dim` | `2` | e.g. 2 for PushT. |
| `smoothed_dim` | `10` | `ActionEmbedder`'s intermediate width. |
| `mlp_scale` | `4` | `ActionEmbedder` MLP hidden-width multiplier. |
| `sigreg_weight` | `0.09` | |
| `sigreg_knots` | `17` | |
| `sigreg_num_proj` | `1024` | |

### `num_frames` sentinel-0 derivation

`num_frames` is a **stored** dataclass field (not a computed property), so it
round-trips through `to_dict()` / `from_dict()` like every other field. Its
default is the sentinel `0`, and `__post_init__` derives it:

- if `num_frames <= 0` (including the default), it is set to
  `history_size + num_preds`;
- if an explicit `num_frames` is given and it is `< history_size + num_preds`,
  `__post_init__` raises `ValueError` — an explicit value is accepted only
  when it is large enough to cover the training sequence length;
- `create_lewm(...)`'s override path re-triggers this: if the caller
  overrides `history_size`/`num_preds` without also restating `num_frames`,
  the merged config's `num_frames` is reset to `0` before reconstruction, so
  it re-derives from the new horizon instead of carrying the old value
  forward into a spurious `ValueError`.

## Serialization

`get_config()` nests the entire model configuration as one field:
`config.update({"config": self.config.to_dict()})`. `from_config()` reverses
this — `LeWMConfig.from_dict(cfg_dict)` if present, otherwise a fresh
`LeWMConfig()` — and reconstructs `LeWM(config=cfg, **config)`. A full
`model.save("model.keras")` / `keras.models.load_model(...)` round trip is
exercised by the trainer's post-training reload check (see
`src/train/lewm/README.md`) and by the model test suite
(`tests/test_models/test_lewm.py`, `tests/test_models/test_lewm_causality.py`).

## `create_lewm(...)` factory

```python
from dl_techniques.models.vision.lewm.model import create_lewm

model = create_lewm(img_size=64, patch_size=16, depth=1, history_size=2)
out = model({"pixels": pixels, "action": actions})
```

Any keyword that names a `LeWMConfig` field overrides that field (on top of
an optional `config=` base); any other keyword is forwarded to `keras.Model`
instead (e.g. `name`).

## See also

`src/train/lewm/README.md` documents the training pipeline in full: the
`--smoke` CLI preset, dataset schema (`{"pixels", "action"} -> dummy_y`),
callback wiring, output artifacts (`config.json`, `training_log.csv`,
`last.keras`, `final_model.keras`), the full CLI argument table, and known
limitations of the HDF5 PushT loader. This README covers the model itself;
that one covers running it.
