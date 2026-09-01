# LeWM: A JEPA-Style Action-Conditioned World Model

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 port of **LeWorldModel (LeWM)**, the end-to-end Joint-Embedding Predictive Architecture of Maes et al., *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels* (arXiv:2603.19312). LeWM learns latent dynamics by predicting the **embedding** of the next frame instead of its pixels, so no decoder and no pixel reconstruction loss are needed.

The pieces: a **ViT encoder shared between the context and target paths** (live encoder, no EMA), an **AdaLN-zero conditional Transformer** as the autoregressive predictor, action conditioning through a per-timestep **action embedder**, and **SIGReg**, the sketched-isotropic-Gaussian regularizer that keeps the embedding space from collapsing without a target network.

---

## 1. Overview: Predicting in Embedding Space

### What is LeWM?

**LeWM** is a JEPA for **action-conditioned video**. Given a short history of frames and the actions taken between them, it predicts the **embedding** of the next frame. Training is self-supervised: one encoder produces both context and target embeddings, and the predictor is asked to match that encoder's own future output.

### Characteristics of this port

1.  **Live target encoder (no EMA)**: one encoder serves both the context path and the target path. Gradients flow through both, and SIGReg supplies the anti-collapse pressure an EMA target network would otherwise provide.
2.  **AdaLN-zero conditioning**: the predictor stacks `AdaLNZeroConditionalBlock` Transformer blocks (DiT-style adaptive layer norm with zero-initialized residual gates), conditioned on the action embedding sequence.
3.  **SIGReg anti-collapse**: a sliced-Gaussian regularizer that fits random 1-D projections of the embedding batch to a standard normal. Without it, the live-target setup collapses to a constant embedding.
4.  **Action-aware rollout**: `rollout()` generates multi-step predicted embeddings from a history window and an action sequence, with the shape contracts (`S == 1`, `T >= history_size`) enforced rather than silently worked around.
5.  **Keras 3 native**: full `get_config()` round-trip, per-component loss trackers (`pred_loss`, `sigreg_loss`) on `model.metrics`, and `add_loss()`-based training so `model.compile(loss=None)` works.

### LeWM vs pixel-space world models

| | Pixel-space (Dreamer, classic video prediction) | LeWM (embedding space) |
| :--- | :--- | :--- |
| **Target** | future RGB frames | future ViT-CLS embeddings |
| **Loss** | pixel MSE / ELBO / GAN | embedding MSE + SIGReg |
| **Failure mode** | capacity spent on texture, lighting, sensor noise | trivial collapse to a constant embedding, prevented by SIGReg |

---

## 2. The Problem: Pixel Prediction vs Latent Prediction

A frame carries far more information than a controller needs. A pixel-space predictor optimizes

```
loss = || decoder(predict(enc(x_t), a_t)) - x_{t+1} ||^2
```

so the decoder must reconstruct every pixel, and capacity goes to reconstruction rather than dynamics.

LeWM drops the decoder. The target is the encoder's own embedding of the next frame:

```
z_t        = encoder(x_t)
z_hat_t+1  = predictor(z_t, a_t)
loss       = || z_hat_t+1 - encoder(x_{t+1}) ||^2 + lambda * SIGReg(Z)
```

Because the same live encoder sits on both sides, a constant-output encoder would drive the MSE to zero. SIGReg blocks that solution by penalizing any embedding distribution whose random 1-D projections drift from a standard normal.

---

## 3. How LeWM Works

```
pixels (B, T, H, W, C)                 action (B, T-1, A)
      │                                       │
      ▼                                       ▼
 ViT encoder (CLS pool)                pad one zero action  -> (B, T, A)
      │                                       │
      ▼                                       ▼
 MLPProjector "projector"              ActionEmbedder (Conv1D -> MLP/SiLU)
      │                                       │
      │ emb (B, T, D)                         │ act_emb (B, T, D)
      └───────────────┬───────────────────────┘
                      ▼
        ARPredictor:  x + pos_embedding -> emb_dropout
                      -> depth x AdaLNZeroConditionalBlock(x, c=act_emb)
                      -> LayerNorm
                      ▼
                 pred_proj (MLPProjector)  ->  pred_emb (B, T, D)
                      │
        ┌─────────────┴──────────────┐
        ▼                            ▼
 MSE(pred[:, :-1], emb[:, 1:])   SIGReg(transpose(emb) -> (T, B, D))
        └────────────► add_loss(pred + sigreg_weight * sigreg) ◄─────┘
```

Step by step:

1.  **Input**: `{"pixels": (B, T, H, W, C), "action": (B, T-1, A)}` with `T = history_size + num_preds`.
2.  **Pixel encoding**: pixels are flattened to `(B*T, H, W, C)`, run through the ViT (CLS pooled), reshaped to `(B, T, D)`, then refined by `MLPProjector`.
3.  **Action encoding**: actions are right-padded with one zero step so their time axis matches the pixel one, then embedded to `(B, T, D)`.
4.  **Prediction**: `ARPredictor` adds a learned positional embedding, runs `depth` AdaLN-zero blocks with `act_emb` as the conditioning signal, then `pred_proj` refines per timestep.
5.  **Losses** (added inside `call`): prediction MSE `mean((pred_emb[:, :-1] - emb[:, 1:]) ** 2)`, plus SIGReg on `emb` transposed to `(T, B, D)` and scaled by `sigreg_weight`.
6.  **Inference**: `model.rollout(pixels_history, action_sequence)` runs an eager loop, appending one predicted embedding per future action and truncating the predictor input to the last `history_size` steps.

---

## 4. Architecture Deep Dive

### 4.1 Vision encoder (`ViT`)

A `dl_techniques.models.vision.vit.ViT` backbone, by default `scale="tiny"` (192-dim, patch 14, image 224), `pooling="cls"`, `include_top=False`. It is **shared** between context (`emb[:, :-1]`) and target (`emb[:, 1:]`); both paths receive gradient — no EMA, no stop-gradient.

### 4.2 `MLPProjector` (used twice)

`Dense(hidden) -> LayerNorm -> GELU -> Dense(out)`:

*   **`projector`** after the ViT, refining `(B, T, D)` embeddings before they serve as both target and predictor input.
*   **`pred_proj`** after the predictor, refining the predicted `(B, T, D)` embeddings.

The norm is `LayerNormalization`, matching the upstream `MLP` default; BatchNorm would also fail at batch size 1.

### 4.3 `ActionEmbedder`

```
Conv1D(action_dim -> smoothed_dim, kernel=1)
-> Dense(mlp_scale * emb_dim) -> SiLU
-> Dense(emb_dim)
```

In Keras (channels-last) the upstream `permute(0, 2, 1)` pair is a no-op; input and output are `(B, T, A)` and `(B, T, D)`.

### 4.4 `ARPredictor` (AdaLN-zero Transformer)

A stack of `AdaLNZeroConditionalBlock` layers — DiT-style blocks where the conditioning vector (the per-step action embedding `c`) produces shift, scale and **zero-initialized residual gates** for the attention and MLP sublayers:

1.  **Learned positional embedding** of shape `(1, num_frames, input_dim)`, `RandomNormal(stddev=0.02)`, sliced to the current `T`.
2.  **`emb_dropout`** on the positionally-encoded sequence.
3.  **Input / cond projections** `Dense(input_dim -> hidden_dim)`, created only when the dimensions differ.
4.  **`depth` AdaLN-zero blocks**, each consuming `[x, c]`. Zero-init gates make the predictor start as an identity, which stabilizes early training against a live target encoder.
5.  **Final LayerNorm and output projection** back to `output_dim`.

### 4.5 `SIGReg` (sketched isotropic Gaussian regularizer)

A `keras.layers.Layer` in `dl_techniques.regularizers.sigreg` returning a scalar loss. For `Z in R^{N x D}` and a freshly sampled, column-normalized Gaussian projection `A in R^{D x P}`:

```
SIGReg(Z) = mean_j sum_k w_k * [
                (mean_n cos(t_k * (Z A)_{n,j}) - phi(t_k))^2
              + (mean_n sin(t_k * (Z A)_{n,j}))^2
            ] * N
```

with knots `t_k` on `[0, 3]`, `phi(t) = exp(-t^2/2)` (the standard-normal characteristic function) and trapezoidal weights `w_k`. It pushes random 1-D projections of the embedding batch towards a standard Gaussian. **This is what replaces the EMA target network.**

LeWM passes embeddings as `(T, B, D)`, so the reduction axis is the batch axis: one statistic per timestep, averaged.

### 4.6 Top-level `LeWM`

Wires the components together, owns the `LeWMConfig`, registers two `keras.metrics.Mean` trackers (`pred_loss`, `sigreg_loss`) so a CSV log records both weighted contributions next to the summed `loss`, and exposes `encode_pixels` / `encode_actions` / `predict_next` / `rollout`.

---

## 5. Quick Start Guide

```bash
pip install keras>=3.8.0 tensorflow>=2.18.0
```

```python
import keras
import numpy as np

from dl_techniques.models.vision.lewm.config import LeWMConfig
from dl_techniques.models.vision.lewm.model import LeWM, create_lewm

# 1. Default config: ViT-tiny, history=3, predict=1, embed_dim=192.
cfg = LeWMConfig()
model = LeWM(config=cfg)          # or: create_lewm()

# 2. Compile. The loss is added internally via add_loss(), so loss=None.
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-4), loss=None)

# 3. Forward pass on dummy data.
B, T = 2, cfg.history_size + cfg.num_preds          # T = 4
pixels = np.random.normal(size=(B, T, 224, 224, 3)).astype("float32")
action = np.random.normal(size=(B, T - 1, cfg.action_dim)).astype("float32")

pred_emb = model({"pixels": pixels, "action": action}, training=False)
print(pred_emb.shape)   # (2, 4, 192)
```

### Inference rollout

```python
# Roll a 3-frame history forward by 5 actions.
HS = cfg.history_size                     # 3
H_future = 5
pixels_history = np.random.normal(
    size=(B, 1, HS, 224, 224, 3)
).astype("float32")
action_sequence = np.random.normal(
    size=(B, 1, HS + H_future, cfg.action_dim)
).astype("float32")

out = model.rollout(pixels_history, action_sequence)
print(out["predicted_emb"].shape)   # (2, 1, 9, 192)  == (B, 1, HS + H_future + 1, D)
```

---

## 6. Component Reference

### 6.1 `LeWM` and `create_lewm`

`LeWM` is the top-level `keras.Model`; `create_lewm(config=None, **overrides)` is the factory. Any
override that names a `LeWMConfig` field is applied to the config (and `num_frames` is re-derived
unless you restate it); anything else goes to `keras.Model`, e.g. `name`.

```python
from dl_techniques.models.vision.lewm.model import create_lewm

model = create_lewm(img_size=64, patch_size=16, depth=1, history_size=2)
```

**Public methods**:

*   `call(inputs, training)`: training forward; adds `pred_loss` and `sigreg_weight * sigreg_loss` via `add_loss`. Returns `pred_emb (B, T, D)`.
*   `encode_pixels(pixels, training)`: `(B, T, H, W, C) -> (B, T, D)`.
*   `encode_actions(action)`: `(B, T_a, A) -> (B, T_a, D)`.
*   `predict_next(emb, act_emb, training)`: `ARPredictor + pred_proj` over `(B, T, D)`.
*   `rollout(pixels_history, action_sequence)`: eager autoregressive rollout returning `{"predicted_emb": (B, S, T + 1, D)}`; requires `S == 1` and `T >= history_size`.
*   `metrics`: exposes `pred_loss_tracker` and `sigreg_loss_tracker` alongside Keras' own.

### 6.2 `LeWMConfig`

A dataclass with `to_dict()` / `from_dict()` round-trip.

| Field | Default | Meaning |
|:------|:-------:|:--------|
| `img_size`, `patch_size`, `img_channels` | 224, 14, 3 | ViT input |
| `encoder_scale` | `"tiny"` | ViT scale (`dl_techniques.models.vision.vit`) |
| `embed_dim` | 192 | model width `D` |
| `history_size` | 3 | observed frames per sample |
| `num_preds` | 1 | predicted frames per sample |
| `num_frames` | 0 (auto) | predictor positional-embedding length; `0` derives `history_size + num_preds`, and an explicit value must be at least that |
| `depth`, `heads`, `dim_head`, `mlp_dim` | 6, 16, 64, 2048 | predictor Transformer |
| `dropout_rate`, `emb_dropout_rate` | 0.1, 0.0 | predictor dropouts |
| `projector_hidden_dim` | 192 | hidden dim of `projector` and `pred_proj` |
| `action_dim`, `smoothed_dim`, `mlp_scale` | 2, 10, 4 | action embedder |
| `sigreg_weight`, `sigreg_knots`, `sigreg_num_proj` | 0.09, 17, 1024 | SIGReg |

### 6.3 Layers

| Component | Shape contract |
|:---|:---|
| **`ActionEmbedder`** | `(B, T, action_dim) -> (B, T, emb_dim)`; Conv1D(k=1) -> Dense/SiLU -> Dense |
| **`MLPProjector`** | `Dense(hidden) -> LayerNorm -> GELU -> Dense(out)` |
| **`ARPredictor`** | inputs `[x, c]`, both `(B, T, D)`; AdaLN-zero stack with learned positional embedding |

---

## 7. Configuration & Model Variants

There is **no `MODEL_VARIANTS` table and no named scale family** — LeWM ships one configuration and
is retuned field by field. `encoder_scale` is the one scale knob; everything else is a
`LeWMConfig` override.

| Use case | Overrides |
|:---------|:----------|
| **Smoke test / CI** | `img_size=64, patch_size=16, history_size=2, num_preds=1, depth=2, heads=4, dim_head=32, mlp_dim=256, sigreg_num_proj=64` |
| **Upstream default** | all defaults |
| **Larger action space** | `action_dim=<A>, smoothed_dim=max(10, 5*A)` |
| **Longer horizon** | `num_preds>=k`; `num_frames` is auto-derived |
| **Larger encoder** | `encoder_scale="small"` or larger; set `embed_dim` to that scale's CLS dim and `projector_hidden_dim` to match |

---

## 8. Usage Examples

### Example 1: A short training run

```python
import keras
import numpy as np

from dl_techniques.models.vision.lewm.model import create_lewm

# A small config keeps the example runnable on CPU.
model = create_lewm(
    img_size=64, patch_size=16, history_size=2, num_preds=1,
    depth=2, heads=4, dim_head=32, mlp_dim=256, sigreg_num_proj=64,
)
model.compile(optimizer=keras.optimizers.AdamW(1e-4), loss=None)

cfg = model.config
T = cfg.history_size + cfg.num_preds

def gen():
    while True:
        pixels = np.random.normal(size=(4, T, 64, 64, 3)).astype("float32")
        action = np.random.normal(size=(4, T - 1, cfg.action_dim)).astype("float32")
        # The target is ignored: the loss is added internally.
        yield {"pixels": pixels, "action": action}, np.zeros((4,), "float32")

model.fit(gen(), steps_per_epoch=2, epochs=1)
# Logged columns: loss, pred_loss, sigreg_loss
```

### Example 2: Inspecting the loss components

```python
pixels = np.random.normal(size=(4, T, 64, 64, 3)).astype("float32")
action = np.random.normal(size=(4, T - 1, cfg.action_dim)).astype("float32")

model({"pixels": pixels, "action": action}, training=True)
print("pred_loss   :", float(model.pred_loss_tracker.result()))
print("sigreg_loss :", float(model.sigreg_loss_tracker.result()))
# Both trackers hold WEIGHTED contributions, so their sum is the total loss.
```

### Example 3: Scoring only the predicted tail of a rollout

```python
HS = cfg.history_size
pixels_history = np.random.normal(size=(2, 1, HS, 64, 64, 3)).astype("float32")
action_sequence = np.random.normal(size=(2, 1, HS + 4, cfg.action_dim)).astype("float32")

out = model.rollout(pixels_history, action_sequence)
pred = out["predicted_emb"][0, 0]          # (T + 1, D)
predictor_tail = pred[HS:]                 # only these are predictor-derived
print(pred.shape, predictor_tail.shape)
```

The first `HS` entries are encoder-derived embeddings of the observed history. Comparing them with
ground truth measures the encoder, not the predictor.

---

## 9. Advanced Usage Patterns

### Disentangling prediction loss from SIGReg

```python
# Zero out SIGReg to check that pred_loss can fall on its own.
model._sigreg_weight = 0.0
# With SIGReg off and a shared live encoder the embeddings collapse to a
# constant within a few hundred steps. Diagnostic use only.
```

### Custom encoder

Subclass `LeWM` and replace `self.encoder` with any `keras.Model` whose pooled output is
`(B*T, D_enc)`; set `projector` so its input dimension is `D_enc`.

### Multi-horizon training

Raise `num_preds` to train the predictor on multi-step targets. `num_frames` is re-derived (or set
it explicitly, at least `history_size + num_preds`). The MSE extends naturally because it always
compares `pred[:, :-1]` against `emb[:, 1:]`.

### Rolling out several distinct histories

`rollout` requires `S == 1`, because only `pixels_history[:, 0]` is encoded and a larger `S` would
silently drop the other histories. Call `rollout` once per history, or tile externally.

---

## 10. Training and Best Practices

*   **Optimizer**: AdamW at `1e-4` with weight decay. Remember the live encoder receives gradient from both the context and target paths.
*   **Batch size**: SIGReg's empirical characteristic function sharpens with `N` (the batch axis after the `(T, B, D)` transpose). Aim for `B >= 32`; it degrades at very small batches.
*   **`sigreg_weight`**: 0.09 by default. If embeddings collapse (`pred_loss` crashes to zero while `sigreg_loss` spikes), raise it; if SIGReg dominates for many epochs, lower it.
*   **Observability**: always read `pred_loss` and `sigreg_loss` separately — a diverging or dominating term is invisible in the summed `loss`.
*   **Warmup**: the AdaLN-zero gates start at zero, so the predictor starts as an identity. That is already an implicit warmup; an LR warmup on top is usually unnecessary.

---

## 11. Serialization & Deployment

`LeWM` and every sublayer register through `@register_dl_technique(...)` from
`dl_techniques.utils.keras_registration`. The registration key is the defining module's dotted path
with the `vision/` family directory stripped, so `LeWM` resolves to
`dl_techniques.models.lewm.model>LeWM`, and likewise `...lewm.embedder>ActionEmbedder`,
`...lewm.predictor>ARPredictor`, `...lewm.projector>MLPProjector`. The helper also binds the legacy
`Custom>ClassName` alias, so older archives keep loading.

```python
model.save("lewm.keras")
restored = keras.models.load_model("lewm.keras")
```

`LeWM.get_config()` serializes the full `LeWMConfig.to_dict()`, and `from_config()` rebuilds the
dataclass. `num_frames` is a stored field rather than a property precisely so that round-trip works
for old and new configs alike.

No separate inference conversion is needed: `rollout()` is the inference API and runs with
`training=False` internally.

---

## 12. Troubleshooting

-   **`pred_loss` crashes to zero in the first epoch** — embedding collapse. Check `sigreg_loss`, verify `sigreg_weight > 0`, raise it, and use a batch large enough for the characteristic-function estimate (`B >= 32`).
-   **`rollout: S must equal 1`** — pass `pixels_history` shaped `(B, 1, history_size, H, W, C)`; roll multiple histories one at a time.
-   **`rollout: action_sequence horizon T must be >= history_size`** — the action sequence must cover the history window plus zero or more future steps.
-   **`ValueError` about `num_frames`** — an explicit `num_frames` must be at least `history_size + num_preds`; leave it at `0` to auto-derive.
-   **Dropping SIGReg to save time** — not possible with a live target encoder; SIGReg is the only thing preventing collapse. Removing it requires an EMA target and a stop-gradient, which is a different architecture.
-   **Rollout output has length `T + 1`** — by construction: `HS` history-encoded steps plus `(T - HS) + 1` predicted steps. Score the tail only.

---

## 13. Technical Details

### Why no EMA target encoder?

The upstream model uses one shared, live encoder for context and target and relies on SIGReg to
prevent collapse. This port keeps that design. An EMA-target variant is possible but needs a
separate codepath and different semantics.

### AdaLN-zero conditioning

Each block computes `shift`, `scale` and a `gate` from the conditioning vector `c`. The gate is
zero-initialized, so at step 0 the block is exactly the identity; training moves it away from zero
only as the predictor learns to use the action signal.

### SIGReg input convention

LeWM passes `(T, B, D)` to `SIGRegLayer`, which reduces over the batch axis, producing one
characteristic-function statistic per timestep and averaging them. The projection matrix
(`P = sigreg_num_proj` columns, normalized) is resampled on every call, so the estimator is
stochastic.

### Action padding

Actions live between frames, so `T` frames carry `T - 1` actions. `LeWM.call` right-pads a zero
action:

```python
zero_pad = ops.zeros((B, 1, action_dim))
action_padded = ops.concatenate([action, zero_pad], axis=1)   # (B, T, A)
```

The padded final action conditions the prediction at step `T`, which the loss then ignores — it
compares `pred[:, :-1]` against `emb[:, 1:]`.

### Loss bookkeeping

`call` issues `add_loss(pred_loss)` and `add_loss(sigreg_weight * sigreg_loss)`, and the two `Mean`
trackers record the **weighted** contributions, so `pred_loss + sigreg_loss == loss` in the log.

---

## 14. Citation

-   **LeWorldModel**, arXiv:2603.19312:
    ```bibtex
    @article{maes2026leworldmodel,
      title   = {LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
      author  = {Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
      journal = {arXiv preprint arXiv:2603.19312},
      year    = {2026}
    }
    ```
