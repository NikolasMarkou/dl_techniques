# PRISM: Partitioned Representations for Iterative Sequence Modeling

![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg) ![Python 3.11](https://img.shields.io/badge/Python-3.11+-blue.svg) ![TF 2.18](https://img.shields.io/badge/TF-2.18-orange.svg)

A Keras 3 implementation of **PRISM (Partitioned Representations for Iterative Sequence Modeling)**, a hierarchical time-series forecaster that replaces standard attention with a learnable **binary time tree** combined with **Haar Wavelet** frequency decomposition. PRISM supports both **point forecasting** (single tensor) and **probabilistic forecasting** via an optional quantile head with monotonicity enforcement.

> **Identity, up front.** PRISM is a forecasting model. The architecture is a "Split / Transform / Weight / Merge" pipeline over time: time gets recursively bisected into overlapping segments, each segment is wavelet-decomposed into frequency bands, a small MLP router assigns soft importance weights to those bands, and the bands are recombined and stitched back together with linear cross-fading. The output is `(B, F_out, num_features)` in point mode or `(B, F_out, num_features, num_quantiles)` in quantile mode.

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Problem PRISM Solves](#2-the-problem-prism-solves)
3. [How PRISM Works: Core Concepts](#3-how-prism-works-core-concepts)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Forecasting Modes (Point vs Quantile)](#5-forecasting-modes-point-vs-quantile)
6. [Quick Start](#6-quick-start)
7. [Component Reference](#7-component-reference)
8. [Configuration & Presets](#8-configuration--presets)
9. [Comprehensive Usage Examples](#9-comprehensive-usage-examples)
10. [Training & Best Practices](#10-training--best-practices)
11. [Serialization & Deployment](#11-serialization--deployment)
12. [Interpretability](#12-interpretability)
13. [Limitations, Troubleshooting & FAQs](#13-limitations-troubleshooting--faqs)
14. [References](#14-references)

---

## 1. Overview

`PRISMModel` is a `keras.Model` that, given a context window of length `context_len` with `num_features` channels, produces:

- A **point forecast** of shape `(B, forecast_len, num_features)`, OR
- A **quantile forecast** of shape `(B, forecast_len, num_features, num_quantiles)` (when `use_quantile_head=True`).

The model has three structural ideas that decide what it does:

| Knob | Off (default) | On |
|------|---------------|-----|
| `use_quantile_head` | Point forecast via MLP head (FFN expansion) | Probabilistic forecast via `QuantileHead` |
| `enforce_monotonicity` | (point mode: ignored) | Strictly non-crossing quantiles via cumulative softplus |
| `tree_depth` | 0 = no time split (single node) | N = `2^N` overlapping segments per layer |

---

## 2. The Problem PRISM Solves

Real-world time series mix global trends, local fine-grained structure, and features on multiple scales in between. Two common failure modes in modern forecasting:

1. **Transformers (e.g. PatchTST)**: strong on global dependencies but quadratic in sequence length and prone to noise leakage from high-frequency channels.
2. **Linear / decomposition models (e.g. DLinear)**: very efficient but lack the nonlinear capacity to fuse interactions across scales.

PRISM is positioned in between. It is both **hierarchical** (a binary tree over time) and **adaptive** (a learnable router decides which frequency bands matter where), at near-linear cost in sequence length.

```
┌─────────────────────────────────────────────────────────────┐
│  The PRISM Solution                                         │
│                                                             │
│  1. Unified hierarchy: instead of flattening time, PRISM    │
│     builds a binary tree of overlapping time segments.       │
│     Coarse scales (root) set context for fine scales (leaf). │
│                                                             │
│  2. Adaptive filtering: an MLP router computes soft weights │
│     over Haar wavelet bands per node, so high-freq jitter   │
│     gets suppressed in trend regions and emphasized in       │
│     spike / seasonality regions.                            │
│                                                             │
│  3. Efficiency: Haar wavelets are O(N); the router and FFN   │
│     are small MLPs; the temporal projector is a single       │
│     channel-shared Dense (DLinear-style).                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How PRISM Works: Core Concepts

**Time tree.** At each PRISM layer, the time axis is recursively bisected into `2^tree_depth` overlapping segments (overlap controlled by `overlap_ratio`). Each segment is processed independently by a `PRISMNode` and the outputs are stitched back together with linear cross-fading at the segment boundaries.

**Node mechanism.** Each `PRISMNode` runs four steps on its segment:

1. **Haar DWT** decomposes the segment into `num_wavelet_levels` frequency bands (low-pass + high-pass at each level).
2. **Statistics extraction** computes per-band summary stats (mean, std, max amplitude, first/second derivatives).
3. **Importance router** is a small MLP from stats to per-band weights, normalized via `softmax(stats / router_temperature)`.
4. **Weighted reconstruction** sums the bands with the router weights.

**Stacking.** `num_layers` PRISM layers are stacked, each with residual connection + LayerNorm + dropout. Each layer keeps the hidden dimension fixed at `hidden_dim`.

**Decoder.** A channel-independent **DLinear-style** decoder maps `context_len -> forecast_len` using a single shared `Dense`, then either:
- a small **MLP forecast head** (point mode), or
- a **`QuantileHead`** with optional monotonicity (quantile mode).

The forecast head is applied to a collapsed shape `(B*forecast_len, hidden_dim)` and reshaped back. This keeps head parameters at `O(hidden_dim * num_features * num_quantiles)` regardless of `forecast_len`.

---

## 4. Architecture Deep Dive

```
Input:  context   shape (B, context_len, num_features)
                  │
                  ▼
         ┌──────────────────────────┐
         │  Dense(hidden_dim)       │   input projection
         └────────────┬─────────────┘
                      │
                      ▼            (B, context_len, hidden_dim)
         ┌──────────────────────────┐
         │   N x PRISMLayer         │
         │  - PRISMTimeTree:        │
         │    bisect time into       │
         │    2^tree_depth segments  │
         │    -> per-node:          │
         │       Haar DWT +         │
         │       stats +            │
         │       router MLP +       │
         │       weighted recon     │
         │    -> stitch (crossfade) │
         │  - residual + LN +       │
         │    dropout               │
         └────────────┬─────────────┘
                      │
                      ▼            (B, context_len, hidden_dim)
         ┌──────────────────────────┐
         │ DLinear-style projector  │
         │  transpose -> Dense(F_out)
         │  -> transpose            │
         └────────────┬─────────────┘
                      │
                      ▼            (B, F_out, hidden_dim)
         reshape to (B*F_out, hidden_dim)
                      │
                      ▼
         head_dropout
                      │
                ┌─────┴─────┐
                ▼           ▼
       point head:    QuantileHead:
       MLP +          Dense ->
       Dense(F)       cumulative
                      softplus
                      (if monotonic)
                      │           │
                      ▼           ▼
       (B, F_out,    (B, F_out,
        F)            F, Q)
```

The `(B*forecast_len, hidden_dim)` collapse before the head is intentional: it makes the head fully time-shared (no per-step parameters) and keeps the head's parameter count independent of `forecast_len`.

---

## 5. Forecasting Modes (Point vs Quantile)

`use_quantile_head` switches between two output regimes. The contract is rigid: a single tensor in both cases (no dict outputs), with rank 3 (point) or rank 4 (quantile).

### Mode A — Point forecast (default)

```python
model = PRISMModel(
    context_len=168, forecast_len=24, num_features=1,
    use_quantile_head=False,
)
# Output shape: (B, forecast_len, num_features)
# Output rank:  3
# Typical loss: MSE, MAE, Huber
```

### Mode B — Quantile Mode

```python
model = PRISMModel(
    context_len=168, forecast_len=24, num_features=1,
    use_quantile_head=True,
    num_quantiles=9,
    quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    enforce_monotonicity=True,
)
# Output shape: (B, F_out, num_features, num_quantiles)
# Output rank:  4
# Typical loss: dl_techniques.losses.QuantileLoss(quantiles=quantile_levels)
```

Output shape summary:

| Mode | `use_quantile_head` | Output shape | Output rank |
|------|---------------------|--------------|-------------|
| Point | `False` | `(B, F_out, num_features)` | 3 |
| Quantile | `True` | `(B, F_out, num_features, num_quantiles)` | 4 |

`PRISMModel.predict_quantiles(context, quantile_levels=None, batch_size=64)` is available only in quantile mode and returns `(quantile_preds, point_preds)` where `point_preds` is the median slice.

---

## 6. Quick Start

```python
import keras
import numpy as np
from dl_techniques.models.prism.model import PRISMModel

# Synthetic data: 1000 windows of length 96 -> forecast next 24 steps, 1 channel.
x = np.linspace(0, 100, 1000 + 96 + 24)
data = np.sin(x).astype("float32")[:, None]
X = np.stack([data[i : i + 96] for i in range(1000)])              # (1000, 96, 1)
y = np.stack([data[i + 96 : i + 96 + 24] for i in range(1000)])    # (1000, 24, 1)

model = PRISMModel.from_preset(
    "small",
    context_len=96,
    forecast_len=24,
    num_features=1,
)
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, batch_size=32, epochs=5, verbose=0)

forecast = model.predict(X[:1], verbose=0)
print(forecast.shape)  # (1, 24, 1)
```

---

## 7. Component Reference

### `PRISMModel`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context_len` | `int` | required | Length of the input history window. Must be `> 0`. |
| `forecast_len` | `int` | required | Length of the prediction horizon. Must be `> 0`. |
| `num_features` | `int` | required | Number of input/output channels. Must be `> 0`. |
| `hidden_dim` | `Optional[int]` | `None` | Hidden dim for projection and PRISM layers. If `None`, falls back to `num_features`. |
| `num_layers` | `int` | `2` | Number of stacked PRISM layers (each = TimeTree + residual + LN + dropout). |
| `tree_depth` | `int` | `2` | Depth of the time binary tree inside each layer. `2^tree_depth` segments per layer. `0` disables splitting. |
| `overlap_ratio` | `float` | `0.25` | Overlap fraction between adjacent time segments. Range `[0.0, 0.5]`. Larger values smooth segment boundaries. |
| `num_wavelet_levels` | `int` | `3` | Number of Haar DWT levels per node. Produces `num_wavelet_levels + 1` frequency bands. |
| `router_hidden_dim` | `int` | `64` | Hidden dim of the per-node importance-router MLP. |
| `router_temperature` | `float` | `1.0` | Temperature for the router softmax. Lower (`< 1.0`) sharpens band selection; higher (`> 1.0`) smooths it. |
| `dropout_rate` | `float` | `0.1` | Dropout applied inside each PRISM layer and before the forecast head. |
| `ffn_expansion` | `int` | `4` | Expansion factor for the point-mode MLP forecast head (`hidden_dim * ffn_expansion`). Ignored when `use_quantile_head=True`. |
| `use_quantile_head` | `bool` | `False` | If `True`, swap the point MLP head for a `QuantileHead`. Output rank becomes 4. |
| `num_quantiles` | `int` | `3` | Number of quantiles emitted when `use_quantile_head=True`. Ignored in point mode (still stored / serialized — see L-7). |
| `quantile_levels` | `Optional[List[float]]` | `None` | Optional explicit quantile levels (e.g. `[0.1, 0.5, 0.9]`). Length must equal `num_quantiles`. If `None` and `use_quantile_head=True`, auto-generates a linear space. |
| `enforce_monotonicity` | `bool` | `True` | When `use_quantile_head=True`, forces `Q_i <= Q_{i+1}` via a cumulative softplus reparameterization. Eliminates quantile crossing. |
| `kernel_initializer` | `Union[str, Initializer]` | `"glorot_uniform"` | Kernel initializer for all `Dense` layers. Round-tripped via `get_config`. |
| `kernel_regularizer` | `Optional[Regularizer]` | `None` | Kernel regularizer for all `Dense` layers. Round-tripped via `get_config`. |

### Output tensor

| Mode | Shape | Notes |
|------|-------|-------|
| Point (`use_quantile_head=False`) | `(B, forecast_len, num_features)` | Single dense tensor, rank 3. |
| Quantile (`use_quantile_head=True`) | `(B, forecast_len, num_features, num_quantiles)` | Single dense tensor, rank 4. Monotonic along the last axis when `enforce_monotonicity=True`. |

### `PRISMLayer` (internal, exposed for custom composition)

| Parameter | Description |
|-----------|-------------|
| `tree_depth` | Depth of the per-layer time tree. `2^tree_depth` overlapping segments. |
| `overlap_ratio` | Overlap fraction between segments. `[0.0, 0.5]`. |
| `num_wavelet_levels` | Number of Haar DWT levels per node. |
| `router_hidden_dim` | Router MLP hidden dim. |
| `router_temperature` | Softmax temperature for the router. |
| `dropout_rate` | Dropout applied after the time tree, before residual + LN. |
| `use_residual` | Whether to add `x + Tree(x)` (default `True`). |
| `use_output_norm` | Whether to LayerNorm after residual (default `True`). |

### `PRESETS`

`PRISMModel.PRESETS` is a class-level `Dict[str, Dict[str, Any]]` with the keys `"tiny"`, `"small"`, `"base"`, `"large"`. Used by `PRISMModel.from_preset(name, ...)`.

---

## 8. Configuration & Presets

```python
PRISMModel.PRESETS = {
    "tiny":  {"hidden_dim":  32, "num_layers": 1, "tree_depth": 1, "num_wavelet_levels": 2, "router_hidden_dim":  32, "ffn_expansion": 2},
    "small": {"hidden_dim":  64, "num_layers": 2, "tree_depth": 2, "num_wavelet_levels": 3, "router_hidden_dim":  64, "ffn_expansion": 4},
    "base":  {"hidden_dim": 128, "num_layers": 3, "tree_depth": 2, "num_wavelet_levels": 3, "router_hidden_dim": 128, "ffn_expansion": 4},
    "large": {"hidden_dim": 256, "num_layers": 4, "tree_depth": 2, "num_wavelet_levels": 4, "router_hidden_dim": 256, "ffn_expansion": 4},
}
```

- **`tiny`** — debug / short sequences. 1 layer, depth 1.
- **`small`** — standard baseline. 2 layers, depth 2.
- **`base`** — wider for multivariate (e.g. ETT, Weather). 3 layers, depth 2.
- **`large`** — long context / large-scale pre-training. 4 layers, depth 2, deeper wavelets.

`from_preset` accepts any preset key plus the three required positional fields (`context_len`, `forecast_len`, `num_features`) plus any override kwargs (e.g. `use_quantile_head=True`, `num_quantiles=9`). Preset fields are merged with user kwargs (user kwargs win).

---

## 9. Comprehensive Usage Examples

### Example 1 — Point forecast, preset

```python
from dl_techniques.models.prism.model import PRISMModel

model = PRISMModel.from_preset(
    "small",
    context_len=168, forecast_len=24, num_features=1,
)
model.compile(optimizer="adamw", loss="mse", metrics=["mae"])
# model.fit(X, y, ...)
```

### Example 2 — Multivariate point forecast, custom config

```python
model = PRISMModel(
    context_len=336, forecast_len=96, num_features=7,
    hidden_dim=128, num_layers=3, tree_depth=2,
    num_wavelet_levels=4, dropout_rate=0.2,
)
model.compile(optimizer="adamw", loss="mae")
```

### Example 3 — Quantile mode

```python
from dl_techniques.losses.quantile_loss import QuantileLoss

quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
model = PRISMModel.from_preset(
    "small",
    context_len=168, forecast_len=24, num_features=1,
    use_quantile_head=True,
    num_quantiles=len(quantile_levels),
    quantile_levels=quantile_levels,
    enforce_monotonicity=True,
)
model.compile(optimizer="adamw", loss=QuantileLoss(quantiles=quantile_levels))
# model.fit(X, y, ...)  # y shape: (B, forecast_len, num_features)
# Output shape: (B, forecast_len, num_features, num_quantiles)
```

### Example 4 — `predict_quantiles` helper (quantile mode only)

```python
quantile_preds, point_preds = model.predict_quantiles(
    context=X_test,
    quantile_levels=[0.1, 0.5, 0.9],   # subset, must be a subset of training quantile_levels
    batch_size=64,
)
# quantile_preds shape: (B, forecast_len, num_features, len(quantile_levels))
# point_preds   shape: (B, forecast_len, num_features)   <- median slice
```

> See L-2 in Limitations: `predict_quantiles` self-mutates `self.quantile_levels` if it was deserialized empty. Set `quantile_levels` explicitly at construction time to avoid that path.

### Example 5 — Serialization round-trip

```python
model.save("prism_small.keras")
restored = keras.models.load_model("prism_small.keras")
pred = restored.predict(X[:1], verbose=0)
```

---

## 10. Training & Best Practices

A ready-to-run trainer lives at [`src/train/prism/train_prism.py`](../../../../train/prism/train_prism.py). It mirrors `src/train/tirex/` (Pattern 2 — time-series / probabilistic) and supports both point and quantile modes via `--use_quantile_head`.

**Headless smoke run** (always required on remote / no-X11 machines):

```bash
MPLBACKEND=Agg .venv/bin/python -m train.prism.train_prism \
    --epochs 1 --steps_per_epoch 50 --batch_size 32 --gpu 0
```

A standalone CPU-only ONNX exporter lives at [`src/train/prism/export.py`](../../../../train/prism/export.py). ONNX export is **off by default** in the trainer; pass nothing to skip it (use `--no_onnx` is unnecessary — the default is already off).

**Best-practice notes**

- **Normalize inputs.** PRISM has internal LayerNorm but does not include instance-level (ReVIN) normalization. Per-instance Z-score normalization is implemented in the bundled trainer (`--no-normalize` to disable) and recommended for benchmark datasets.
- **Single-feature default.** `num_features=1` is the bundled trainer default and the only path exercised in CI. Multivariate (`num_features>1`) is supported by the architecture but lightly tested — start by validating point mode with a small `forecast_len`.
- **Context length.** PRISM benefits from longer context windows due to hierarchical splitting. `168` -> `336` -> `512` are good progression points.
- **Tree depth.** Depth `2` (4 segments per layer) is the standard sweet spot. Depths `> 3` rarely help and inflate node count exponentially.
- **Overlap ratio.** Default `0.25` is robust. Increase to `0.3`-`0.4` if you see "jumpy" predictions at segment boundaries.
- **Quantile losses.** Use `dl_techniques.losses.QuantileLoss(quantiles=quantile_levels)`. Set `enforce_monotonicity=True` (default) to eliminate quantile crossing at the head level.
- **Single GPU only.** Pass `--gpu 0` or `--gpu 1` to the trainer. Do not run two trainers in parallel.
- **`MPLBACKEND=Agg`** is mandatory on headless boxes — the visualization callback writes PNGs.

---

## 11. Serialization & Deployment

### Keras native (`.keras`)

```python
model.save("prism.keras")
restored = keras.models.load_model("prism.keras")
```

Round-trip works because:

- `PRISMModel`, `PRISMLayer`, `PRISMTimeTree`, `PRISMNode`, `FrequencyBandRouter`, `QuantileHead` all use `@keras.saving.register_keras_serializable()`.
- `kernel_initializer` and `kernel_regularizer` are normalized via `keras.initializers.get` / `keras.regularizers.get` in `get_config()`.
- `quantile_levels` is round-tripped as a list (or `None` in point mode).

### ONNX

A standalone exporter is provided at [`src/train/prism/export.py`](../../../../train/prism/export.py). It mirrors `train/tirex/export.py`:

- CPU-only env-pin (`CUDA_VISIBLE_DEVICES=""` set before `import keras`) to avoid CudnnRNN ops in the export trace.
- Auto-detection of `context_len` from `model.get_config()` (falls back to `--input_length`).
- Single-tensor verification at `rtol=atol=1e-4` between Keras and the ONNX runtime output.

```bash
.venv/bin/python -m train.prism.export \
    --model_path results/prism_small_point_xxx/best_model.keras \
    --opset_version 17 --verify
```

PRISM emits a single dense tensor (not a dict), so the exporter does **not** have an `--output_key` flag. The same exporter handles both rank-3 (point) and rank-4 (quantile) outputs transparently.

---

## 12. Interpretability

PRISM's router weights are inspectable per node. Each `PRISMNode` exposes its router as a sublayer, and the time tree stores its nodes in a flat list (`time_tree.all_nodes`).

```python
# Inspect the router of the first node of the first PRISM layer.
node = model.prism_layers[0].time_tree.all_nodes[0]
router = node.router
# Run a forward pass and pull the router's intermediate output via a keras.Model
# extractor. See PRISMNode.call() for the exact tensor name to extract.
```

Higher weight on low-pass (approximation) bands indicates the node is focused on slow trend; higher weight on high-pass (detail) bands indicates focus on rapid fluctuations / spikes. This is per-node and per-input, so the same model can interpret different inputs at different scales.

---

## 13. Limitations, Troubleshooting & FAQs

### Limitations (read these before filing issues)

- **L-1. Single-tensor output, two ranks.** Point mode returns rank 3 (`(B, F_out, F)`); quantile mode returns rank 4 (`(B, F_out, F, Q)`). Downstream code that handles both must branch on rank, not on key names.
- **L-2. `predict_quantiles` self-mutates `self.quantile_levels` on first call** if `quantile_levels` was empty or `None` at construction. Pass `quantile_levels` explicitly at construction time (or via `from_preset`) to avoid this footgun. Surfaced from issue I-8.
- **L-3. `num_quantiles` and `enforce_monotonicity` are stored even in point mode.** When `use_quantile_head=False` these values are kept in `get_config()` for round-trip fidelity but never used at inference. Not a bug, just noisy config. Surfaced from issue I-9.
- **L-4. `num_features=1` is the well-tested default.** The architecture supports multivariate `num_features>1`, but the bundled trainer and the existing test suite focus on `num_features=1`. Validate point mode first when going multivariate, then enable quantile.
- **L-5. Tree depth is exponential in node count.** Each layer has `2^tree_depth - 1` internal segments plus leaves. Setting `tree_depth > 3` rarely helps and inflates parameter count and graph build time significantly.
- **L-6. `PRISMNode.call()` uses `keras.ops.cond` for the interpolation branch.** Under the TF backend, `ops.cond` traces both branches, so the conditional is purely a control-flow nicety, not a perf optimization. Acceptable for forward pass, but a latent inefficiency if you are benchmarking on very large trees. Surfaced from issue I-12.
- **L-7. ONNX export is not exercised in CI.** The exporter at `train/prism/export.py` is a near-verbatim copy of `train/tirex/export.py` (which is exercised), but the PRISM-specific path has not been smoke-tested end-to-end. ONNX export is opt-in only (off by default in the trainer).
- **L-8. No instance normalization.** PRISM does not include ReVIN-style per-instance normalization. The bundled trainer does per-instance Z-score normalization in the data pipeline; if you build a custom pipeline, normalize inputs yourself.

### Troubleshooting / FAQs

**Q. My model output rank is 4 but I expected 3.** You set `use_quantile_head=True`. Either pass `use_quantile_head=False` for point mode, or take the median slice via `model.predict_quantiles(...)` (returns `(quantile_preds, point_preds)`).

**Q. `predict_quantiles` raises about `self.quantile_levels`.** You loaded a model where `quantile_levels` was `None`. Set it explicitly at construction. See L-2.

**Q. Why Haar wavelets, not FFT?** FFT assumes periodicity and global stationarity. Wavelets are localized in time and frequency, which aligns naturally with the time tree's block-based processing, and Haar is `O(N)`.

**Q. Is this faster than Transformers?** Yes for forward pass. Cost is roughly linear in sequence length; attention is `O(L^2)`. The router and FFN are small MLPs.

**Q. Can I use this for classification?** Not directly. `PRISMModel` is forecast-only. You can lift the `PRISMLayer` stack into a custom model and attach a classification head (e.g. GlobalAveragePooling + Dense) yourself.

**Q. ONNX export fails.** The exporter pins `CUDA_VISIBLE_DEVICES=""` before `import keras` to dodge `CudnnRNN`. If you set the env var after import, the pin is silently ignored. Run `python -m train.prism.export --verify` from a fresh process.

**Q. Where are the router weights during training?** Inside each `PRISMNode.router`. Access them via `model.prism_layers[i].time_tree.all_nodes[k].router`. See section 12.

---

## 14. References

- Chen, Z. et al. (2025) — *PRISM: A Hierarchical Multiscale Approach for Time Series Forecasting*. arXiv:2512.24898.
- Mallat, S. (1989) — *A theory for multiresolution signal decomposition: the wavelet representation*. IEEE TPAMI.
- Zeng et al. (2023) — *Are Transformers Effective for Time Series Forecasting?* (DLinear baseline).
- Nie et al. (2023) — *A Time Series is Worth 64 Words* (PatchTST).
- Koenker & Bassett (1978) — *Regression Quantiles*. Econometrica. Underpins `QuantileLoss`.

**Related code:**

- Model: `dl_techniques/models/prism/model.py`
- Blocks: `dl_techniques/layers/time_series/prism_blocks.py` (`PRISMLayer`, `PRISMTimeTree`, `PRISMNode`, `FrequencyBandRouter`, `FrequencyBandStatistics`)
- Quantile head: `dl_techniques/layers/time_series/quantile_head_fixed_io.py`
- Loss: `dl_techniques/losses/quantile_loss.py`
- Trainer: `train/prism/train_prism.py`
- ONNX export: `train/prism/export.py`
- Peer time-series models: `dl_techniques/models/{tirex, adaptive_ema, nbeats, mdn, deepar}`
- Tests: `tests/test_models/test_prism/test_model.py`

```bibtex
@article{chen2025prism,
  title={PRISM: A Hierarchical Multiscale Approach for Time Series Forecasting},
  author={Chen, Zihao and Andre, Alexandre and Ma, Wenrui and Knight, Ian and Shuvaev, Sergey and Dyer, Eva},
  journal={arXiv preprint arXiv:2512.24898},
  year={2025}
}
```
