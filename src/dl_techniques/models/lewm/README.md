# LeWM: LeCun's World Model (JEPA-Style Action-Conditioned Predictor)

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 port of **LeWM** (Learning the World with Minimal supervision), the JEPA-style action-conditioned world model from Sobal et al. (2024). LeWM learns latent dynamics by predicting **future embeddings** of observed frames rather than future pixels, eliminating the need to model high-bandwidth pixel detail.

Key architectural features include a **ViT encoder shared between context and target paths** (live encoder, no EMA), an **AdaLN-zero conditional Transformer** as the autoregressive predictor, action conditioning via a per-timestep **action embedder**, and the **SIGReg** sliced-Gaussian regularizer that prevents the embedding space from collapsing without requiring a target network.

---

## Table of Contents

1. [Overview: Predicting in Embedding Space](#1-overview-predicting-in-embedding-space)
2. [The Problem: Pixel Prediction vs. Latent Prediction](#2-the-problem-pixel-prediction-vs-latent-prediction)
3. [How LeWM Works](#3-how-lewm-works)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Component Reference](#6-component-reference)
7. [Configuration & Model Variants](#7-configuration--model-variants)
8. [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9. [Advanced Usage Patterns](#9-advanced-usage-patterns)
10. [Training and Best Practices](#10-training-and-best-practices)
11. [Serialization & Deployment](#11-serialization--deployment)
12. [Troubleshooting & FAQs](#12-troubleshooting--faqs)
13. [Technical Details](#13-technical-details)
14. [Citation](#14-citation)

---

## 1. Overview: Predicting in Embedding Space

### What is LeWM?

**LeWM** is a Joint Embedding Predictive Architecture (JEPA) for **action-conditioned video** dynamics. Given a short history of frames and the actions taken between them, the model predicts the **embedding** of the next frame rather than its pixels. Training is fully self-supervised: the same encoder produces both context and target embeddings, and the predictor is asked to match the encoder's own future output.

The implementation faithfully mirrors the upstream PyTorch reference (`/tmp/lewm_source/`) while adopting Keras 3 conventions: every component is a `@keras.saving.register_keras_serializable()` layer/model with a `LeWMConfig` dataclass driving construction.

### Key Innovations of this Implementation

1.  **Live target encoder (no EMA)**: Upstream LeWM uses a single encoder for both the context path and the target path. Gradients flow through both, and SIGReg supplies the anti-collapse pressure that an EMA target network would otherwise provide. This is preserved here (`DECISION D-001`).
2.  **AdaLN-zero conditioning**: The predictor is a stack of `AdaLNZeroConditionalBlock` Transformer blocks (DiT-style adaptive layer-norm with zero-initialized residual gates), conditioned on the action embedding sequence.
3.  **SIGReg anti-collapse**: A sliced Gaussian regularizer fits random 1-D projections of the embedding batch to a standard normal characteristic function; without it the live-target setup collapses to a constant embedding.
4.  **Action-aware autoregressive rollout**: A first-class `rollout()` method generates multi-step predicted embeddings from a history window and a full action sequence, with shape contracts enforced (`S == 1`, `T >= history_size`).
5.  **Keras 3 native**: Full `get_config()` round-trip, per-component loss trackers (`pred_loss`, `sigreg_loss`) exposed via `model.metrics`, and `add_loss()`-based training so `model.compile(loss=None)` works out of the box.

### LeWM vs. Pixel-Space World Models

**Pixel-space world model (e.g. Dreamer, classic video prediction)**:

```
- Target: future RGB frames (high-dimensional, lots of irrelevant detail).
- Loss: pixel MSE / VAE ELBO / GAN.
- Failure mode: spends capacity modeling textures, lighting, noise.
```

**LeWM (JEPA, embedding-space)**:

```
- Target: future ViT-CLS embeddings (low-dimensional, semantically dense).
- Loss: embedding MSE + SIGReg sliced-Gaussian penalty.
- Failure mode: trivial collapse to constant embedding (prevented by SIGReg).
```

---

## 2. The Problem: Pixel Prediction vs. Latent Prediction

### The Cost of Pixel Targets

Video frames carry far more information than a controller needs. A pixel-space predictor must allocate capacity to texture, lighting, and sensor noise, even though downstream control only depends on a low-dimensional geometric/semantic state.

```
┌─────────────────────────────────────────────────────────────┐
│  Pixel-Space World Model                                    │
│                                                             │
│  loss = || decoder(predict(enc(x_t), a_t)) - x_{t+1} ||^2   │
│                                                             │
│  The decoder must reconstruct every pixel. Capacity is      │
│  spent on reconstruction, not dynamics.                     │
└─────────────────────────────────────────────────────────────┘
```

### The LeWM Solution

LeWM removes the decoder entirely. The target is the encoder's own embedding of the next frame, so the predictor only needs to model **change in latent state**, not pixels.

```
┌─────────────────────────────────────────────────────────────┐
│  Embedding-Space World Model (LeWM)                         │
│                                                             │
│  z_t       = encoder(x_t)                                   │
│  z_hat_t+1 = predictor(z_t, a_t)                            │
│  loss      = || z_hat_t+1 - encoder(x_{t+1}) ||^2           │
│              + lambda * SIGReg(Z)                           │
│                                                             │
│  Same encoder on both sides. SIGReg keeps Z non-degenerate. │
└─────────────────────────────────────────────────────────────┘
```

The shared (live) encoder means a constant-output encoder would minimize the MSE trivially. SIGReg blocks that solution by penalizing any embedding distribution whose random 1-D projections diverge from a standard normal.

---

## 3. How LeWM Works

### The High-Level Architecture

A history of frames is encoded; actions are embedded; the AdaLN-zero predictor consumes both and produces predicted embeddings of the same shape. The MSE is computed between the predictor's output (shifted by one) and the encoder's own output on the next frame.

```
┌────────────────────────────────────────────────────────────────────┐
│                           LeWM Forward Pass                        │
│                                                                    │
│  pixels (B, T, H, W, C)              action (B, T-1, A)            │
│       │                                   │                        │
│       ▼                                   ▼                        │
│  ┌───────────┐                       ┌──────────────────────┐      │
│  │ ViT-tiny  │                       │ pad zero action at T │      │
│  │ encoder   │                       └──────────┬───────────┘      │
│  │ (CLS pool)│                                  ▼                  │
│  └─────┬─────┘                       ┌─────────────────────┐       │
│        ▼                             │ ActionEmbedder      │       │
│  ┌───────────┐                       │ Conv1D->MLP(SiLU)   │       │
│  │ Projector │                       └──────────┬──────────┘       │
│  │  (MLP+LN) │                                  │                  │
│  └─────┬─────┘                                  │                  │
│        │ emb (B, T, D)              act_emb (B, T, D)              │
│        ├──────────────┐                         │                  │
│        ▼              │                         │                  │
│   ┌─────────────────────────────────────────────┘                  │
│   │ ARPredictor:                                                   │
│   │   x + pos_embedding -> emb_dropout                             │
│   │   -> N x AdaLNZeroConditionalBlock(x, c=act_emb)               │
│   │   -> LayerNorm                                                 │
│   └────────┬───────────────────────────────────────────────────────┘
│            ▼                                                       │
│       ┌──────────┐                                                 │
│       │ pred_proj│  (MLPProjector)                                 │
│       └────┬─────┘                                                 │
│            ▼                                                       │
│   pred_emb (B, T, D)                                               │
│            │                                                       │
│            ▼                                                       │
│   ┌────────────────────────────────┐    ┌─────────────────────┐    │
│   │ MSE(pred[:, :-1], emb[:, 1:])  │    │ SIGReg(emb^T)       │    │
│   └─────────────┬──────────────────┘    └──────────┬──────────┘    │
│                 │                                  │               │
│                 └──────────┐         ┌─────────────┘               │
│                            ▼         ▼                             │
│                       add_loss(pred + lambda * sigreg)             │
└────────────────────────────────────────────────────────────────────┘
```

### Data Flow Step by Step

1.  **Input**: a dict `{"pixels": (B, T, H, W, C), "action": (B, T-1, A)}` where `T = history_size + num_preds`.
2.  **Pixel encoding**: pixels are flattened to `(B*T, H, W, C)`, run through ViT (CLS-pooled), reshaped to `(B, T, D)`, then refined by `MLPProjector` ("projector").
3.  **Action encoding**: actions are right-padded with a zero step so the action time axis equals the pixel time axis (`T-1 -> T`), then embedded by `ActionEmbedder` to `(B, T, D)`.
4.  **Prediction**: `ARPredictor` adds a learned positional embedding, projects in/out as needed, runs `depth` AdaLN-zero blocks with `act_emb` as the conditioning signal, then `pred_proj` refines per-timestep.
5.  **Losses (added inside `call`)**:
    *   **Prediction MSE**: `mean((pred_emb[:, :-1] - emb[:, 1:]) ** 2)`.
    *   **SIGReg**: applied to `emb` reshaped to `(T, B, D)` (matching upstream's reduction axis), scaled by `sigreg_weight`.
6.  **Inference rollout**: `model.rollout(pixels_history, action_sequence)` runs an eager Python loop, autoregressively appending one predicted embedding per future action while truncating the predictor input to the last `history_size` steps.

---

## 4. Architecture Deep Dive

### 4.1 Vision encoder (`ViT`)

A `dl_techniques.models.vit.ViT` backbone, by default `scale="tiny"` (192-dim, 3 heads, 12 layers, patch=14, img=224), `pooling="cls"`, `include_top=False`. The encoder is **shared** between context (`emb[:, :-1]`) and target (`emb[:, 1:]`); both paths receive gradient (`DECISION D-001` — no EMA, no stop-gradient, matching upstream).

### 4.2 `MLPProjector` (projector and pred_proj)

A 2-layer MLP: `Dense(hidden) -> LayerNorm -> GELU -> Dense(out)`. Used twice:

*   **`projector`** after the ViT, refining `(B, T, D)` embeddings before they are consumed as both target and predictor input.
*   **`pred_proj`** after the predictor, refining the predicted `(B, T, D)` embeddings.

`DECISION D-002` documents that upstream's `MLP` uses `nn.LayerNorm` (not `BatchNorm1d`); this Keras port follows the upstream-code truth. BatchNorm would also fail at batch-size 1.

### 4.3 `ActionEmbedder`

Per-timestep action lifting:

```
Conv1D(action_dim -> smoothed_dim, kernel=1)
-> Dense(smoothed_dim -> mlp_scale * emb_dim) -> SiLU
-> Dense(mlp_scale * emb_dim -> emb_dim)
```

In Keras (channels-last) the upstream `permute(0,2,1)` pair is a no-op; input/output are `(B, T, A)` / `(B, T, D)`.

### 4.4 `ARPredictor` (AdaLN-zero Transformer)

A stack of `AdaLNZeroConditionalBlock` layers — DiT-style adaptive layer-norm Transformer blocks where the conditioning vector (here, the per-step action embedding `c`) produces shift, scale, and **zero-initialized residual gates** for the attention and MLP sublayers. Components:

1.  **Learned positional embedding**: shape `(1, num_frames, input_dim)`, `RandomNormal(stddev=0.02)`; sliced to current `T` at call time.
2.  **`emb_dropout`** on the positionally-encoded sequence.
3.  **Input / cond projections**: `Dense(input_dim -> hidden_dim)` for `x` and `c`, instantiated only when dims differ (Identity otherwise).
4.  **`depth` AdaLN-zero blocks**: each consumes `[x, c]` and returns updated `x`. Zero-init gates mean the predictor starts as an identity, which stabilizes early training on top of the live target encoder.
5.  **Final LayerNorm + output projection** back to `output_dim`.

### 4.5 `SIGReg` (Sketch Isotropic Gaussian Regularizer)

Implemented as a `keras.layers.Layer` (in `dl_techniques.regularizers.sigreg`) returning a scalar loss. For input `Z in R^{N x D}` and a freshly-sampled column-normalized Gaussian projection `A in R^{D x P}`:

```
SIGReg(Z) = mean_j sum_k w_k * [
                (mean_n cos(t_k * (Z A)_{n,j}) - phi(t_k))^2
              + (mean_n sin(t_k * (Z A)_{n,j}))^2
            ] * N
```

with `t_k` integration knots on `[0, 3]`, `phi(t) = exp(-t^2/2)` (standard-normal characteristic function), and trapezoidal weights `w_k`. It pushes 1-D random projections of the embedding batch toward a standard Gaussian, in the spirit of sliced-Wasserstein regularization. **This is what replaces the EMA target network** as the anti-collapse mechanism.

LeWM passes embeddings as `(T, B, D)` so the reduction axis is the batch axis `B`, producing one statistic per timestep and averaging.

### 4.6 Top-level `LeWM` model

Wires the six components together, owns the `LeWMConfig`, registers two `keras.metrics.Mean` trackers (`pred_loss`, `sigreg_loss`) so the CSV log records both weighted contributions next to the summed `loss`, and exposes `encode_pixels` / `encode_actions` / `predict_next` / `rollout` helpers.

---

## 5. Quick Start Guide

### Installation

```bash
pip install keras>=3.8.0 tensorflow>=2.18.0
```

### Your First LeWM Model

```python
import keras
import numpy as np
from dl_techniques.models.lewm.config import LeWMConfig
from dl_techniques.models.lewm.model import LeWM

# 1. Default config: ViT-tiny, history=3, predict=1, embed_dim=192.
cfg = LeWMConfig()
model = LeWM(config=cfg)

# 2. Compile. Loss is added internally via add_loss(), so loss=None is correct.
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-4), loss=None)

# 3. Forward pass on dummy data.
B, T = 2, cfg.history_size + cfg.num_preds          # T = 4
pixels = np.random.normal(size=(B, T, 224, 224, 3)).astype("float32")
action = np.random.normal(size=(B, T - 1, cfg.action_dim)).astype("float32")

pred_emb = model({"pixels": pixels, "action": action}, training=False)
print(pred_emb.shape)   # (2, 4, 192)
```

### Inference Rollout

```python
# Roll a 3-frame history forward by 5 actions.
HS = cfg.history_size                     # 3
H_future = 5
pixels_history = np.random.normal(size=(B, 1, HS, 224, 224, 3)).astype("float32")
action_sequence = np.random.normal(size=(B, 1, HS + H_future, cfg.action_dim)).astype("float32")

out = model.rollout(pixels_history, action_sequence)
print(out["predicted_emb"].shape)   # (B, 1, HS + H_future + 1, D)
```

---

## 6. Component Reference

### 6.1 `LeWM`

The top-level `keras.Model`. Construction is config-driven via `LeWMConfig`.

**Public methods**:

*   `call(inputs, training)`: training forward; adds `pred_loss` and `sigreg_weight * sigreg_loss` via `add_loss`. Returns `pred_emb (B, T, D)`.
*   `encode_pixels(pixels, training)`: `(B, T, H, W, C) -> (B, T, D)`.
*   `encode_actions(action)`: `(B, T_a, A) -> (B, T_a, D)`.
*   `predict_next(emb, act_emb, training)`: `ARPredictor + pred_proj` over `(B, T, D)`.
*   `rollout(pixels_history, action_sequence)`: eager autoregressive rollout. Returns `{"predicted_emb": (B, S, T+1, D)}`. Enforces `S == 1` and `T >= history_size`.
*   `metrics`: exposes `pred_loss_tracker` and `sigreg_loss_tracker` so per-epoch CSV logs include both components.

### 6.2 `LeWMConfig`

Dataclass with `to_dict()` / `from_dict()` round-trip. Fields:

| Field | Default | Meaning |
|:------|:-------:|:--------|
| `img_size`, `patch_size`, `img_channels` | 224, 14, 3 | ViT input |
| `encoder_scale` | `"tiny"` | ViT scale (`dl_techniques.models.vit`) |
| `embed_dim` | 192 | model width `D` |
| `history_size` | 3 | observed frames per sample |
| `num_preds` | 1 | predicted frames per sample |
| `num_frames` | 0 (auto) | predictor positional-embedding length; `0` means `history_size + num_preds`; explicit values must satisfy `num_frames >= history_size + num_preds` |
| `depth`, `heads`, `dim_head`, `mlp_dim` | 6, 16, 64, 2048 | predictor Transformer |
| `dropout`, `emb_dropout` | 0.1, 0.0 | predictor dropouts |
| `projector_hidden_dim` | 192 | hidden dim of `projector` and `pred_proj` |
| `action_dim`, `smoothed_dim`, `mlp_scale` | 2, 10, 4 | action embedder |
| `sigreg_weight`, `sigreg_knots`, `sigreg_num_proj` | 0.09, 17, 1024 | SIGReg |

### 6.3 `ActionEmbedder`

`(B, T, action_dim) -> (B, T, emb_dim)`. Conv1D(k=1) -> Dense(SiLU) -> Dense.

### 6.4 `MLPProjector`

`Dense(hidden) -> LayerNorm -> GELU -> Dense(out)`. Used for both `projector` and `pred_proj`.

### 6.5 `ARPredictor`

AdaLN-zero conditional Transformer stack with learned positional embedding, optional in/cond/out projections, and a final LayerNorm. Inputs: `[x, c]` both `(B, T, D)`.

---

## 7. Configuration & Model Variants

LeWM is configured by `LeWMConfig`. The upstream defaults target PushT-scale tasks (ViT-tiny, action_dim=2). Variants are produced by overriding fields rather than via named presets.

| Use case | Recommended overrides |
|:---------|:----------------------|
| **Smoke test / CI** | `history_size=2, num_preds=1, depth=2, heads=4, dim_head=32, mlp_dim=256, sigreg_num_proj=64` |
| **PushT (upstream default)** | All defaults |
| **Larger action space** | `action_dim=<A>, smoothed_dim=max(10, 5*A)` |
| **Longer horizon** | `num_preds>=k`; `num_frames` auto-derived to `history_size + num_preds` |
| **Larger encoder** | `encoder_scale="small"` (or larger); set `embed_dim` to that scale's CLS dim and `projector_hidden_dim` accordingly |

There are no "tiny/small/base/large" named factory variants — the `tiny` ViT IS the default; scale up via `encoder_scale` and the transformer hyperparameters.

---

## 8. Comprehensive Usage Examples

### Example 1: Training Loop

```python
import keras, numpy as np
from dl_techniques.models.lewm.config import LeWMConfig
from dl_techniques.models.lewm.model import LeWM

cfg = LeWMConfig()
model = LeWM(config=cfg)
model.compile(optimizer=keras.optimizers.AdamW(1e-4), loss=None)

def gen():
    while True:
        T = cfg.history_size + cfg.num_preds
        pixels = np.random.normal(size=(8, T, 224, 224, 3)).astype("float32")
        action = np.random.normal(size=(8, T - 1, cfg.action_dim)).astype("float32")
        # Target is ignored: loss is added internally.
        yield {"pixels": pixels, "action": action}, np.zeros((8,), "float32")

model.fit(gen(), steps_per_epoch=10, epochs=2)
# CSV log columns: loss, pred_loss, sigreg_loss
```

### Example 2: Inspecting Loss Components

```python
out = model({"pixels": pixels, "action": action}, training=True)
print("pred_loss   :", float(model.pred_loss_tracker.result()))
print("sigreg_loss :", float(model.sigreg_loss_tracker.result()))
# pred_loss + sigreg_loss == total loss (both weighted contributions).
```

### Example 3: Per-Step Embedding Distance During Rollout

```python
out = model.rollout(pixels_history, action_sequence)
pred = out["predicted_emb"][0, 0]                 # (T+1, D)
# First HS entries are encoder-derived; only the tail is predictor-derived.
HS = cfg.history_size
predictor_tail = pred[HS:]                        # score against GT here
```

---

## 9. Advanced Usage Patterns

### Disentangling Pred-Loss from SIGReg

To debug a stalled training run, freeze one term:

```python
# Zero out SIGReg to verify pred_loss can decrease on its own.
model._sigreg_weight = 0.0
# WARNING: with SIGReg off and a shared live encoder, embeddings will collapse
# to a constant within a few hundred steps. Diagnostic use only.
```

### Custom Encoder

Subclass `LeWM` and swap `self.encoder` for a different `keras.Model` whose output is `(B*T, D_enc)` after pooling. Adjust `projector` so its `input_dim == D_enc`.

### Multi-Horizon Training

Bump `num_preds` to train the predictor on multi-step targets. `num_frames` is auto-derived (or set explicitly with `num_frames >= history_size + num_preds`). The MSE loss naturally extends because it always compares `pred[:, :-1]` against `emb[:, 1:]`.

### Rollout with Multiple Distinct Histories

`rollout` enforces `S == 1` to avoid the upstream footgun where `pixels_history[:, 0]` is silently broadcast. To roll out multiple distinct histories, call `rollout` once per history or tile externally.

---

## 10. Training and Best Practices

*   **Optimizer**: AdamW with `learning_rate=1e-4` and weight decay is the upstream default. Mind that the live target encoder receives gradient from both context and target paths.
*   **Batch size**: SIGReg's empirical characteristic function is more accurate with larger `N` (the batch axis after the `(T, B, D)` transpose). Aim for `B >= 32` for stable SIGReg; SIGReg degrades at very small batches.
*   **`sigreg_weight`**: 0.09 is the upstream default. If you see embedding collapse (pred_loss crashes to zero, sigreg_loss spikes), increase weight; if SIGReg dominates (sigreg_loss >> pred_loss for many epochs), decrease it.
*   **Loss observability**: always inspect `pred_loss` and `sigreg_loss` separately in the CSV log. A diverging or dominating term is invisible in the summed `loss`.
*   **Warmup**: the AdaLN-zero blocks initialize their gates to zero, so the predictor starts as identity. This already provides implicit warmup; you usually do not need an LR warmup schedule on top.

---

## 11. Serialization & Deployment

`LeWM` and every sublayer are `@keras.saving.register_keras_serializable()`. Round-trip:

```python
model.save("lewm.keras")
restored = keras.models.load_model("lewm.keras")
```

`LeWM.get_config()` serializes the full `LeWMConfig.to_dict()`; `from_config()` reconstructs the dataclass and rebuilds. `num_frames` is a stored field (not a `@property`) precisely so to_dict/from_dict round-trips both old and new configs cleanly (`DECISION D-002`).

For inference-only deployment, no separate "inference model" conversion is required: `model.rollout()` is the inference API and uses `training=False` internally.

---

## 12. Troubleshooting & FAQs

**Q: My `pred_loss` crashes to zero in the first epoch.**

A: Embedding collapse. Check `sigreg_loss` — if it spiked at the same time, the encoder is producing a near-constant embedding. Verify `sigreg_weight > 0`, increase it if needed, and ensure batch size is large enough for SIGReg's characteristic-function estimate to be informative (`B >= 32` recommended).

**Q: `rollout` raises `S must equal 1`.**

A: Intentional. Pass `pixels_history` with shape `(B, 1, history_size, H, W, C)`. To roll multiple distinct histories, call `rollout` once per history. See `DECISION plan_2026-05-23_692fd5e5/D-001`.

**Q: `rollout` raises `T must be >= history_size`.**

A: `action_sequence` must cover both the history window and at least zero future steps. Pad to at least `history_size`.

**Q: I get a `num_frames` `ValueError` from `LeWMConfig`.**

A: An explicit `num_frames` must be `>= history_size + num_preds` (it sizes the predictor's positional embedding). Leave it at `0` to auto-derive.

**Q: The model returns a list of outputs / multiple tensors.**

A: It does not — `LeWM.call` returns a single tensor `pred_emb (B, T, D)`. If you mean training metrics, those come from `model.metrics` (`loss`, `pred_loss`, `sigreg_loss`).

**Q: Can I drop SIGReg to speed up training?**

A: No, not with the live target encoder. SIGReg is the only thing preventing trivial collapse. If you want to remove SIGReg you must add an EMA target encoder and stop-gradient (which would diverge from upstream and `DECISION D-001`).

**Q: Why does the rollout output have length `T + 1` rather than `T`?**

A: By construction the rollout retains every step it produces: `HS` history-encoded steps plus `(T - HS) + 1` predicted steps. The first `HS` entries are **encoder-derived**; only the tail is **predictor-derived**. Score predictions on the tail only.

---

## 13. Technical Details

### Why no EMA target encoder?

Upstream LeWM (Sobal et al.) uses a shared, live encoder for both context and target, and relies on **SIGReg** to prevent collapse. This Keras port preserves that design (`DECISION D-001` in `plans/plan_2026-04-21_8416bc0b/decisions.md`). Variants with an EMA target are possible but require a separate codepath and divergent semantics.

### AdaLN-zero conditioning

Each Transformer block computes `shift`, `scale`, and a `gate` from the conditioning `c`. The gate is **zero-initialized**, so at step 0 the block is exactly the identity. Training pushes the gate away from zero only as the predictor learns to use the action signal — a strong stabilizer when the target encoder is live.

### SIGReg input convention

LeWM passes `(T, B, D)` to `SIGRegLayer`. The layer reduces over the **last-but-one axis** (the batch axis here), producing one characteristic-function statistic per timestep, averaged into a scalar. With a fresh column-normalized random projection `A in R^{D x P}` each call (`P = num_proj`), the estimator is stochastic but unbiased.

### Action padding trick

Actions live between frames, so there are `T - 1` actions for `T` frames. Upstream right-pads a zero action to match the time axis. This implementation does the same inside `LeWM.call`:

```python
zero_pad = ops.zeros((B, 1, action_dim))
action_padded = ops.concatenate([action, zero_pad], axis=1)   # (B, T, A)
```

The padded final action conditions the prediction at step `T`, which is then unused by the loss (the loss compares `pred[:, :-1]` to `emb[:, 1:]`).

### Loss bookkeeping

`call` calls `add_loss(pred_loss)` and `add_loss(sigreg_weight * sigreg_loss)`. The two `Mean` trackers (`pred_loss_tracker`, `sigreg_loss_tracker`) record the **weighted** contributions, so `pred_loss + sigreg_loss == loss` in the CSV log, which makes a diverging or dominating term immediately visible.

### Decisions log

Key non-obvious design choices are pinned in `plans/`:

*   **D-001** (live target encoder, no EMA, no stop-gradient): matches upstream `/tmp/lewm_source/jepa.py`.
*   **D-002** (`MLPProjector` uses `LayerNorm`, not `BatchNorm1d`): upstream `MLP` defaults to `nn.LayerNorm`; BN would also break at batch-size 1.
*   **D-002** (`num_frames` is a stored field, not a property): preserves `to_dict`/`from_dict` round-trip across config evolution.
*   **plan_2026-05-23_692fd5e5/D-001** (`rollout` enforces `S == 1`): surfaces the upstream broadcast footgun rather than silently dropping per-S histories.

---

## 14. Citation

If you use this implementation, please cite the upstream LeWM paper and this library:

```bibtex
@article{sobal2024lewm,
  title   = {Learning the World with Minimal Supervision (LeWM)},
  author  = {Sobal, Vlad and others},
  year    = {2024},
  note    = {Upstream PyTorch reference}
}

@software{dl_techniques,
  title  = {dl_techniques: A Deep Learning Research Library},
  author = {Markou, Nikolas},
  year   = {2024},
  url    = {https://github.com/NikolasMarkou/dl_techniques}
}
```

Upstream PyTorch reference: `/tmp/lewm_source/` (`jepa.py`, `module.py`). See `DECISIONS` in `plans/` for design choices that diverge from a literal port.
