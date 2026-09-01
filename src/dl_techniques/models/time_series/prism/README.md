# PRISM: Partitioned Representations for Iterative Sequence Modeling

![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg) ![Python 3.11](https://img.shields.io/badge/Python-3.11+-blue.svg) ![TF 2.18](https://img.shields.io/badge/TF-2.18-orange.svg)

A Keras 3 implementation of **PRISM**, a hierarchical forecaster that replaces attention with a learnable **binary time tree** over **Haar wavelet** frequency bands. It supports point forecasting (a single tensor) and probabilistic forecasting through an optional quantile head with monotonicity enforcement.

> **Identity, up front.** PRISM is a forecasting model built as a Split / Transform / Weight / Merge pipeline over time. Time is split into `2^i` overlapping segments at each level `i` (a loop over levels, not a recursion over children); each segment is wavelet-decomposed into frequency bands; a small MLP router assigns soft importance weights to those bands; the bands are recombined and stitched back with linear cross-fading. Output is `(B, F_out, num_features)` in point mode, `(B, F_out, num_features, num_quantiles)` in quantile mode.

---

## 1. Overview

`PRISMModel` is a `keras.Model` that maps a context window of length `context_len` with `num_features` channels to one of two output regimes. Three knobs decide what it does:

| Knob | Off (default) | On |
|------|---------------|-----|
| `use_quantile_head` | Point forecast via an MLP head | Probabilistic forecast via `QuantileHead` |
| `enforce_monotonicity` | (ignored in point mode) | Strictly non-crossing quantiles via cumulative softplus |
| `tree_depth` | `0` = no time split (single node) | `N` = `2^N` overlapping segments per layer |

---

## 2. The Problem PRISM Solves

Real series mix a global trend, local fine structure, and features at every scale between. The two dominant families each miss one side:

1. **Transformers (PatchTST and kin)** are strong on global dependency but quadratic in sequence length, and they leak noise in from high-frequency channels.
2. **Linear / decomposition models (DLinear and kin)** are very efficient but lack the non-linear capacity to fuse interactions across scales.

PRISM sits between them: **hierarchical** (a binary tree over time) and **adaptive** (a learnable router decides which frequency bands matter where), at near-linear cost in sequence length.

```
┌─────────────────────────────────────────────────────────────┐
│  1. Unified hierarchy: build a binary tree of overlapping    │
│     time segments instead of flattening time. Coarse scales  │
│     set the context for fine scales.                         │
│                                                              │
│  2. Adaptive filtering: an MLP router weights Haar wavelet   │
│     bands per node, suppressing high-frequency jitter in     │
│     trend regions and emphasizing it around spikes.          │
│                                                              │
│  3. Efficiency: Haar is O(N); router and FFN are small MLPs; │
│     the temporal projector is one shared Dense (DLinear).    │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How PRISM Works: Core Concepts

**Time tree.** Each PRISM layer splits the time axis into `2^tree_depth` overlapping segments (overlap set by `overlap_ratio`), processes each with a `PRISMNode`, and stitches the outputs back with linear cross-fading at the boundaries. This is a **loop, not a recursion**: level `i` re-splits the full, re-stitched sequence into `2^i` segments rather than bisecting level `i-1`'s children, so the deepest leaf's length comes from one application of the split formula at `num_segments = 2^tree_depth`, not from `tree_depth` successive halvings.

**Node mechanism.** Each `PRISMNode` runs four steps on its segment:

1. **Haar DWT** into `num_wavelet_levels + 1` bands: one detail (high-pass) band per level plus the final approximation (low-pass) band. Each level floor-halves the length, so the deepest band is `segment_len // 2 ** num_wavelet_levels` long (see L-4).
2. **Statistics extraction** produces exactly six per-band summaries: `mean`, `std`, `min`, `max`, and the `mean` and `std` of the band's FIRST difference. There is no second derivative. On a band of length 1 the first difference does not exist and both diff stats are defined as `0.0`; a band of length 0 is refused at construction.
3. **Importance router**: a small MLP from the concatenated per-band stats to one score per band, normalized by a SINGLE joint `softmax(scores / router_temperature)` across all bands of the node. Because the softmax is joint, a non-finite statistic in any one band propagates to every band's weight.
4. **Weighted reconstruction** sums the bands with those weights.

**Stacking.** `num_layers` PRISM layers, each with residual + LayerNorm + dropout, all at the same `hidden_dim`.

**Decoder.** A channel-independent DLinear-style decoder maps `context_len -> forecast_len` with one shared `Dense`, then either a small MLP forecast head (point mode) or a `QuantileHead` (quantile mode).

---

## 4. Architecture Deep Dive

```
Input:  context   (B, context_len, num_features)
                  │
         ┌────────▼─────────────────┐
         │  Dense(hidden_dim)       │   input projection
         └────────────┬─────────────┘
                      ▼            (B, context_len, hidden_dim)
         ┌──────────────────────────┐
         │   N x PRISMLayer         │
         │  - PRISMTimeTree:        │
         │      split into          │
         │      2^tree_depth        │
         │      segments -> per     │
         │      node: Haar DWT +    │
         │      stats + router MLP  │
         │      + weighted recon    │
         │      -> stitch (crossfade)│
         │  - residual + LN + dropout│
         └────────────┬─────────────┘
                      ▼            (B, context_len, hidden_dim)
         ┌──────────────────────────┐
         │ DLinear-style projector  │
         │  transpose -> Dense(F_out)
         │  -> transpose            │
         └────────────┬─────────────┘
                      ▼            (B, F_out, hidden_dim)
         reshape to (B * F_out, hidden_dim) -> head_dropout
                      │
                ┌─────┴─────┐
                ▼           ▼
       point head:    QuantileHead:
       MLP +          Dense -> cumulative
       Dense(F)       softplus (if monotonic)
                │           │
                ▼           ▼
       (B, F_out, F)  (B, F_out, F, Q)
```

The `(B * forecast_len, hidden_dim)` collapse before the head is deliberate: it makes the head fully time-shared and keeps its parameter count at `O(hidden_dim * num_features * num_quantiles)` regardless of `forecast_len`.

---

## 5. Forecasting Modes (Point vs Quantile)

`use_quantile_head` switches between two output regimes. The contract is rigid: a single tensor in both cases (never a dict), rank 3 or rank 4.

| Mode | `use_quantile_head` | Output shape | Rank | Typical loss |
|------|---------------------|--------------|:----:|--------------|
| Point | `False` | `(B, F_out, num_features)` | 3 | MSE, MAE, Huber |
| Quantile | `True` | `(B, F_out, num_features, num_quantiles)` | 4 | `QuantileLoss(quantiles=...)` |

`PRISMModel.predict_quantiles(context, quantile_levels=None, batch_size=64)` works only in quantile mode and returns `(quantile_preds, point_preds)`, where `point_preds` is the median slice.

---

## 6. Quick Start

```python
import keras
import numpy as np
from dl_techniques.models.time_series.prism.model import PRISMModel

# 1000 windows of length 96 -> forecast the next 24 steps, 1 channel.
x = np.linspace(0, 100, 1000 + 96 + 24)
data = np.sin(x).astype("float32")[:, None]
X = np.stack([data[i : i + 96] for i in range(1000)])              # (1000, 96, 1)
y = np.stack([data[i + 96 : i + 96 + 24] for i in range(1000)])    # (1000, 24, 1)

model = PRISMModel.from_variant(
    "small", context_len=96, forecast_len=24, num_features=1,
)
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, batch_size=32, epochs=5, verbose=0)

forecast = model.predict(X[:1], verbose=0)
print(forecast.shape)          # (1, 24, 1)
print(model.count_params())    # 26959
```

---

## 7. Component Reference

### `PRISMModel`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context_len` | `int` | required | Length of the input history window. `> 0`. |
| `forecast_len` | `int` | required | Prediction horizon. `> 0`. |
| `num_features` | `int` | required | Input/output channels. `> 0`. |
| `hidden_dim` | `Optional[int]` | `None` | Hidden dim for projection and layers. `None` falls back to `num_features`. |
| `num_layers` | `int` | `2` | Stacked PRISM layers (TimeTree + residual + LN + dropout). |
| `tree_depth` | `int` | `2` | `2^tree_depth` segments per layer; `0` disables splitting. **No standalone valid range** — constrained jointly with `context_len`, `overlap_ratio` and `num_wavelet_levels` (L-4). |
| `overlap_ratio` | `float` | `0.25` | Overlap between adjacent segments. Half-open range `[0, 0.5)`, validated in `__init__`. Larger values smooth boundaries and lengthen segments. |
| `num_wavelet_levels` | `int` | `3` | Haar DWT levels per node, giving `levels + 1` bands. Each level floor-halves the band length. |
| `router_hidden_dim` | `int` | `64` | Hidden dim of the per-node router MLP. |
| `router_temperature` | `float` | `1.0` | Router softmax temperature. `< 1.0` sharpens band selection, `> 1.0` smooths it. |
| `dropout_rate` | `float` | `0.1` | Dropout in each layer and before the head. |
| `ffn_expansion` | `int` | `4` | Expansion for the point-mode MLP head. Ignored in quantile mode. |
| `use_quantile_head` | `bool` | `False` | Swap the point head for a `QuantileHead`; output rank becomes 4. |
| `num_quantiles` | `int` | `3` | Quantiles emitted in quantile mode. Stored and serialized even in point mode. |
| `quantile_levels` | `Optional[List[float]]` | `None` | Explicit levels; length must equal `num_quantiles`. When `None` in quantile mode: at `num_quantiles == 3` it is `PRISMModel.DEFAULT_QUANTILES = [0.1, 0.5, 0.9]`; at any other length it falls back to `np.linspace(0, 1, num_quantiles + 2)[1:-1]`. |
| `enforce_monotonicity` | `bool` | `True` | In quantile mode, forces `Q_i <= Q_{i+1}` via cumulative softplus. Eliminates crossing. |
| `kernel_initializer` | `str` or `Initializer` | `"glorot_uniform"` | For all `Dense` layers. Round-tripped in `get_config`. |
| `kernel_regularizer` | `Optional[Regularizer]` | `None` | For all `Dense` layers. Round-tripped in `get_config`. |

### `PRISMLayer` (internal, exposed for custom composition)

`tree_depth`, `overlap_ratio`, `num_wavelet_levels`, `router_hidden_dim`, `router_temperature`, `dropout_rate` mean the same as above, plus `use_residual` (add `x + Tree(x)`, default `True`) and `use_output_norm` (LayerNorm after the residual, default `True`).

The `prism/` package `__init__.py` is intentionally empty, so import from `.model`. The `time_series` family package re-exports the public names, so `from dl_techniques.models.time_series import PRISMModel, create_prism_model` also works.

---

## 8. Configuration & Presets

```python
PRISMModel.MODEL_VARIANTS = {
    "tiny":  {"hidden_dim":  32, "num_layers": 1, "tree_depth": 1, "num_wavelet_levels": 2, "router_hidden_dim":  32, "ffn_expansion": 2},
    "small": {"hidden_dim":  64, "num_layers": 2, "tree_depth": 2, "num_wavelet_levels": 3, "router_hidden_dim":  64, "ffn_expansion": 4},
    "base":  {"hidden_dim": 128, "num_layers": 3, "tree_depth": 2, "num_wavelet_levels": 3, "router_hidden_dim": 128, "ffn_expansion": 4},
    "large": {"hidden_dim": 256, "num_layers": 4, "tree_depth": 2, "num_wavelet_levels": 4, "router_hidden_dim": 256, "ffn_expansion": 4},
}
```

- **`tiny`** — debug and short sequences.
- **`small`** — standard baseline.
- **`base`** — wider, for multivariate sets (ETT, Weather).
- **`large`** — long context, deeper wavelets.

### Band budget per preset (measured, not hand-computed)

Every preset multiplies `tree_depth` and `num_wavelet_levels` into one number, `min_band_len` (L-4). It also depends on `context_len`, a caller argument, so a preset alone does not determine it. At `context_len=96, overlap_ratio=0.25`:

| variant | `tree_depth` | `num_wavelet_levels` | `deepest_leaf_seg` | `min_band_len` @ 96 | forward finite | smallest supported `context_len` |
|---|---|---|---|---|---|---|
| `tiny`  | 1 | 2 | 54 | 13 | yes | 8 |
| `small` | 2 | 3 | 25 | 3  | yes | 31 |
| `base`  | 2 | 3 | 25 | 3  | yes | 31 |
| `large` | 2 | 4 | 25 | **1** | yes | 61 |

`large` sits **exactly on the degenerate boundary** at `context_len=96`: its deepest bands carry a single timestep, so `mean == min == max` and both first-difference features are exactly `0.0` by definition. `__init__` logs a warning saying so. It is supported, not an error — but it is why `large` is documented as the *long context* preset. Below `context_len=61`, `large` is refused with a `ValueError` (`min_band_len == 0`).

`from_variant` takes any variant key plus the three required fields (`context_len`, `forecast_len`, `num_features`) plus overrides (e.g. `use_quantile_head=True`, `num_quantiles=9`). Overrides win over preset fields.

---

## 9. Comprehensive Usage Examples

### Example 1 — Multivariate point forecast, custom config

```python
from dl_techniques.models.time_series.prism.model import create_prism_model

model = create_prism_model(
    context_len=336, forecast_len=96, num_features=7,
    hidden_dim=128, num_layers=3, tree_depth=2,
    num_wavelet_levels=4, dropout_rate=0.2,
)
model.compile(optimizer="adamw", loss="mae")
print(model.count_params())   # 114808 -- the factory returns a BUILT model
```

### Example 2 — Quantile mode

```python
from dl_techniques.losses.quantile_loss import QuantileLoss

quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
model = PRISMModel.from_variant(
    "small", context_len=96, forecast_len=24, num_features=1,
    use_quantile_head=True,
    num_quantiles=len(quantile_levels),
    quantile_levels=quantile_levels,
    enforce_monotonicity=True,
)
model.compile(optimizer="adamw", loss=QuantileLoss(quantiles=quantile_levels))
model.fit(X, y, ...)                     # y shape: (B, forecast_len, num_features)
print(model.predict(X[:2]).shape)        # (2, 24, 1, 9)
```

### Example 3 — `predict_quantiles` (quantile mode only)

```python
quantile_preds, point_preds = model.predict_quantiles(
    context=X_test,
    quantile_levels=[0.1, 0.5, 0.9],     # must be a subset of the trained levels
    batch_size=64,
)
# quantile_preds: (B, forecast_len, num_features, 3)
# point_preds:    (B, forecast_len, num_features)   <- median slice
```

Set `quantile_levels` explicitly at construction: see L-2.

### Example 4 — Serialization round trip

```python
model.save("prism_small.keras")
restored = keras.models.load_model("prism_small.keras")
pred = restored.predict(X[:1], verbose=0)
```

---

## 10. Training & Best Practices

A ready-to-run trainer lives at [`src/train/time_series/prism/train_prism.py`](../../../../train/time_series/prism/train_prism.py). It supports both modes via `--use_quantile_head`.

```bash
MPLBACKEND=Agg .venv/bin/python -m train.time_series.prism.train_prism \
    --epochs 1 --steps_per_epoch 50 --batch_size 32 --gpu 0
```

- **Normalize inputs.** PRISM has internal LayerNorm but no instance-level (RevIN) normalization. The bundled trainer does per-instance Z-scoring (`--no-normalize` to disable); a custom pipeline must do its own.
- **Single-feature default.** `num_features=1` is the trainer default and the exercised path. Multivariate is supported by the architecture but lightly tested — validate point mode with a small `forecast_len` first.
- **Context length.** PRISM benefits from long context because of the hierarchical splitting. `168` -> `336` -> `512` is a good progression.
- **Tree depth.** Depth `2` (4 segments per layer) is the sweet spot. No depth is safe on its own — it is bounded by `min_band_len` jointly with `context_len` and `num_wavelet_levels` (L-4). Deepen the tree and the context window together.
- **Overlap ratio.** `0.25` is robust. Raise to `0.3`-`0.4` if predictions look jumpy at segment boundaries.
- **Quantiles.** Use `dl_techniques.losses.QuantileLoss` and leave `enforce_monotonicity=True`.
- **Single GPU.** Pass `--gpu 0` or `--gpu 1`; do not run two trainers in parallel. `MPLBACKEND=Agg` is mandatory on headless boxes — the visualization callback writes PNGs.

---

## 11. Serialization & Deployment

### Keras native (`.keras`)

```python
model.save("prism.keras")
restored = keras.models.load_model("prism.keras")     # no custom_objects
```

`PRISMModel`, `PRISMLayer`, `PRISMTimeTree`, `PRISMNode`, `FrequencyBandRouter` and `QuantileHead` all register through `register_dl_technique` (`dl_techniques.utils.keras_registration`). The package string is the *defining* module's dotted path, and only one of the six is defined here, so the keys are not uniform: `dl_techniques.models.prism.model>PRISMModel`, `dl_techniques.layers.time_series.prism_blocks><ClassName>` for the four block classes, and `dl_techniques.layers.time_series.quantile_head_fixed_io>QuantileHead`. Pre-2026-08-29 archives still load through the legacy `Custom>ClassName` alias the helper also binds.

`kernel_initializer` and `kernel_regularizer` are normalized via `keras.initializers.get` / `keras.regularizers.get` in `get_config()`; `quantile_levels` round-trips as a list (or `None` in point mode).

### ONNX

A standalone exporter is at [`src/train/time_series/prism/export.py`](../../../../train/time_series/prism/export.py). It pins CPU (`CUDA_VISIBLE_DEVICES=""` before `import keras`, to keep CudnnRNN ops out of the trace), auto-detects `context_len` from `model.get_config()`, and verifies Keras against the ONNX runtime at `rtol=atol=1e-4`.

```bash
.venv/bin/python -m train.time_series.prism.export \
    --model_path results/prism_small_point_xxx/best_model.keras \
    --opset_version 17 --verify
```

PRISM emits a single dense tensor, so the exporter has no `--output_key` flag; it handles rank-3 and rank-4 outputs alike. Export is off by default in the trainer.

---

## 12. Interpretability

Router weights are inspectable per node. Each `PRISMNode` exposes its router as a sublayer, and the time tree keeps its nodes in a flat list:

```python
node = model.prism_layers[0].time_tree.all_nodes[0]
router = node.router          # FrequencyBandRouter
```

Run a forward pass and pull the router's intermediate output through a `keras.Model` extractor; `PRISMNode.call()` names the tensor. High weight on the low-pass (approximation) band means the node is tracking slow trend; high weight on the detail bands means it is tracking rapid fluctuation. The weights are per-node *and* per-input, so one model reads different inputs at different scales.

---

## 13. Limitations, Troubleshooting & FAQs

### Limitations

- **L-1. Single-tensor output, two ranks.** Point mode is rank 3, quantile mode rank 4. Code that handles both must branch on rank, not on key names.
- **L-2. `predict_quantiles` self-mutates `self.quantile_levels`** on first call if the levels were empty or `None` at construction. Pass `quantile_levels` explicitly (directly or through `from_variant`) to avoid that path.
- **L-3. `num_quantiles` and `enforce_monotonicity` are stored even in point mode.** Kept in `get_config()` for round-trip fidelity, never used at inference. Noisy config, not a bug.
- **L-4. `tree_depth`, `context_len`, `overlap_ratio` and `num_wavelet_levels` are jointly constrained — no `tree_depth` range is valid on its own.** The binding quantity is the length of the deepest band:

  ```
  overlap_size     = int(context_len * overlap_ratio / 2 ** tree_depth)
  non_overlap_len  = (context_len - overlap_size * (2 ** tree_depth - 1)) // 2 ** tree_depth
  deepest_leaf_seg = non_overlap_len + overlap_size      # == context_len when tree_depth == 0
  min_band_len     = deepest_leaf_seg // 2 ** num_wavelet_levels
  ```

  `min_band_len` must be `>= 1`; `__init__` raises `ValueError` at 0, naming all four knobs, the computed lengths, and a `context_len` that does work. Measured over a 36-cell grid, **depth alone does not separate working configurations from broken ones**: `context_len=96, tree_depth=2, num_wavelet_levels=4` is refused, while `context_len=256, tree_depth=4, num_wavelet_levels=3` is fine. Do not read a "`tree_depth > 3` is bad" rule out of this — compute `min_band_len`.

  At `min_band_len == 1` the configuration is supported but statistically degenerate: that band is one timestep, so the router sees `mean == min == max` and both first-difference statistics are a fabricated exact `0.0`. Prefer `>= 2` if the deepest band is meant to carry information; `__init__` warns at 1, because README prose is not reachable from a `from_variant(...)` call site.

  **Legacy checkpoints.** `from_config` routes through `__init__`, so a `.keras` file whose config has `min_band_len == 0` now raises at `load_model` instead of loading. This is deliberate: such a model produced `inf`/NaN band statistics on every forward pass. Re-train at a supported configuration; no migration makes those weights meaningful.

  **A dynamic time axis defeats the guard.** The degenerate-band check branches on the STATIC band length and deliberately falls through when it is unknown. Measured: a `tree_depth=3` model traced as `tf.function(input_signature=[tf.TensorSpec([None, None, 7])])` returns `nan_frac == 1.0` where the same model returns `0.0` eager. `PRISMModel.input_spec` pins `axes={1: context_len}` and refuses a WRONG static length, but Keras' `assert_input_compatibility` explicitly accepts an unknown dimension against an `axes` constraint. Pin the time axis statically in any `tf.function`, `saved_model` signature, or `padded_batch` pipeline — the shipped ONNX exporter already does.

  Cost is separately exponential in depth: each layer instantiates `2^tree_depth` leaf segments plus the shallower levels.
- **L-5. `PRISMNode.call()` uses `keras.ops.cond`** for the interpolation branch. Under the TF backend `ops.cond` traces both branches, so it is control-flow tidiness, not a speed-up — a latent inefficiency when benchmarking very large trees.
- **L-6. ONNX export is not exercised in CI.** The exporter is a near-verbatim copy of the TiRex one (which is exercised), but the PRISM path has not been smoke-tested end to end.
- **L-7. No instance normalization.** See §10.

### Troubleshooting / FAQs

**Q. My output rank is 4 but I expected 3.** `use_quantile_head=True`. Pass `False`, or take the median slice via `predict_quantiles`.

**Q. `predict_quantiles` raises about `self.quantile_levels`.** The loaded model had `quantile_levels=None`. Set it at construction (L-2).

**Q. Why Haar wavelets, not FFT?** FFT assumes periodicity and global stationarity. Wavelets are localized in time *and* frequency, which matches the tree's block-based processing, and Haar is `O(N)`.

**Q. Is this faster than a Transformer?** For the forward pass, yes — cost is roughly linear in sequence length against attention's `O(L^2)`, and the router and FFN are small MLPs.

**Q. Can I use this for classification?** Not directly; `PRISMModel` is forecast-only. Lift the `PRISMLayer` stack into a custom model and attach your own head.

**Q. ONNX export fails.** The exporter pins `CUDA_VISIBLE_DEVICES=""` before `import keras`. Setting it after the import silently does nothing — run the exporter from a fresh process.

---

## 14. References

- Chen, Z. et al. (2025) — *PRISM: A Hierarchical Multiscale Approach for Time Series Forecasting*. arXiv:2512.24898.
- Mallat, S. (1989) — *A theory for multiresolution signal decomposition: the wavelet representation*. IEEE TPAMI.
- Zeng et al. (2023) — *Are Transformers Effective for Time Series Forecasting?* (DLinear baseline).
- Nie et al. (2023) — *A Time Series is Worth 64 Words* (PatchTST).
- Koenker & Bassett (1978) — *Regression Quantiles*. Econometrica. Underpins `QuantileLoss`.

**Related code**

- Model: `dl_techniques/models/time_series/prism/model.py`
- Blocks: `dl_techniques/layers/time_series/prism_blocks.py` (`PRISMLayer`, `PRISMTimeTree`, `PRISMNode`, `FrequencyBandRouter`, `FrequencyBandStatistics`)
- Quantile head: `dl_techniques/layers/time_series/quantile_head_fixed_io.py`
- Loss: `dl_techniques/losses/quantile_loss.py`
- Trainer: `train/time_series/prism/train_prism.py`; ONNX export: `train/time_series/prism/export.py`
- Tests: `tests/test_models/test_prism/test_model.py`

```bibtex
@article{chen2025prism,
  title={PRISM: A Hierarchical Multiscale Approach for Time Series Forecasting},
  author={Chen, Zihao and Andre, Alexandre and Ma, Wenrui and Knight, Ian and
          Shuvaev, Sergey and Dyer, Eva},
  journal={arXiv preprint arXiv:2512.24898},
  year={2025}
}
```
