# Adaptive EMA Slope Filter

![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg) ![Python 3.11](https://img.shields.io/badge/Python-3.11+-blue.svg) ![TF 2.18](https://img.shields.io/badge/TF-2.18-orange.svg)

A small, transparent **EMA-slope trading-signal generator** with two optional learnable extensions: trainable slope thresholds and a probabilistic slope-quantile head. The package is intentionally simple: a single `keras.Model` that wraps `ExponentialMovingAverage`, computes the slope of the EMA over a configurable look-back, and emits three trading signals — `signal_above`, `signal_below`, `signal_between` — gated by an upper / lower slope threshold.

> **Identity, up front.** This is **not** a learned forecaster. With fixed thresholds (`learnable_thresholds=False`, default) the model has **zero trainable parameters** in the threshold path — outputs are hard binary masks. With `learnable_thresholds=True` you train **two scalars** (a midpoint and a width). For real probabilistic forecasting use [`models/tirex`](../tirex/README.md), [`models/prism`](../prism/README.md), or [`models/nbeats`](../nbeats/README.md).

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Problem This Solves](#2-the-problem-this-solves)
3. [How It Works: Core Concepts](#3-how-it-works-core-concepts)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Quick Start](#5-quick-start)
6. [Component Reference](#6-component-reference)
7. [Configuration & Usage Modes](#7-configuration--usage-modes)
8. [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9. [Training & Best Practices](#9-training--best-practices)
10. [Serialization & Deployment](#10-serialization--deployment)
11. [Limitations, Troubleshooting & FAQs](#11-limitations-troubleshooting--faqs)
12. [When to Use This vs. a Real Forecaster](#12-when-to-use-this-vs-a-real-forecaster)
13. [References](#13-references)

---

## 1. Overview

`AdaptiveEMASlopeFilterModel` is a `keras.Model` that, given a price-like time series, produces:

- The **EMA** of the input,
- The **slope** of the EMA over `lookback_period` bars,
- Three binary (or soft) **trading signals** — `slope > upper`, `slope < lower`, and `lower ≤ slope ≤ upper`,
- The two **threshold scalars** themselves, optionally trainable,
- Optionally, **slope quantiles** from a `QuantileSequenceHead`.

The model has three knobs that decide what it does:

| Knob | Off (default) | On |
|------|---------------|-----|
| `learnable_thresholds` | Hard binary signals from fixed thresholds | Soft sigmoid signals + 2 trainable threshold scalars |
| `quantile_head_config` | No quantile head | Adds `slope_quantiles` to outputs |
| `adjust_ema` | Plain recursive EMA | Bias-corrected EMA (Pandas-style `adjust=True`) |

---

## 2. The Problem This Solves

A common observation in systematic trading is that filtering trades by the **slope** of a moving average can produce better risk-adjusted returns than always-long / always-short rules. Specifically:

- A **steep positive slope** (`slope > upper`) often correlates with strong trend regimes — entry signal.
- A **steep negative slope** (`slope < lower`) marks downtrends — short / exit signal.
- A **flat slope** (`lower ≤ slope ≤ upper`) marks ranging / chop regimes — sit out.

`AdaptiveEMASlopeFilterModel` packages this rule as a Keras model so that:

1. The slope thresholds can optionally be **learned** rather than guessed.
2. The whole pipeline (EMA, slope, signal) is **differentiable in soft mode**, so it can sit inside a larger training graph.
3. The model produces a **dict output** containing all intermediate tensors — useful for diagnostics, visualizations, and downstream rule composition.

---

## 3. How It Works: Core Concepts

**EMA.** Standard exponential moving average with smoothing factor `α = 2 / (period + 1)`. The `ExponentialMovingAverage` layer iterates over the time axis in pure Keras ops (one Python step per time index — see Limitations).

**Slope.** Defined as `slope_t = EMA_t − EMA_{t−L}` where `L = lookback_period`. For the first `L` steps the lagged EMA is zero-padded, so `slope[:, :L]` should be ignored in any downstream loss / metric.

**Threshold parameterization (learnable).** When `learnable_thresholds=True` the raw scalar weights are passed through a midpoint + softplus(half-range) reparameterization to enforce `lower ≤ upper`:

```
midpoint    = (U_var + L_var) / 2
half_range  = softplus(|U_var − L_var| / 2)
upper       = midpoint + half_range
lower       = midpoint − half_range
```

(See Limitations §11 for a known geometric quirk of this parameterization.)

**Signal generation.** Inference always uses **hard** comparisons (`slope > upper`, etc.) and produces a 0/1 mask. During training **with** `learnable_thresholds=True` the signals are computed with sigmoids for gradient flow:

```
signal_above   = σ((slope − upper) / T)
signal_below   = σ((lower − slope) / T)
signal_between = max(1 − signal_above − signal_below, 0)
```

The temperature `T` is currently fixed at `1.0`.

---

## 4. Architecture Deep Dive

```
Input: price = [p_0, p_1, ..., p_T]
Shape: (batch, time_steps) or (batch, time_steps, features)
                    │
                    ▼
       ┌────────────────────────────┐
       │  ExponentialMovingAverage  │
       │      period = ema_period   │
       └─────────────┬──────────────┘
                     │
                     ▼
        EMA = [EMA_0, EMA_1, ..., EMA_T]
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
   EMA current              EMA lagged (L steps)
        │                         │
        └──────────┬──────────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │    Slope Calculation  │
       │  slope_t = EMA_t -    │
       │           EMA_{t-L}   │
       └───────────┬───────────┘
                   │
    ┌──────────────┴──────────────┐
    │                             │
    ▼                             ▼
(if quantile_head_config)    (threshold path)
    │                             │
    ▼                             ▼
┌───────────────────┐    ┌──────────────────────────────┐
│ QuantileSequence  │    │    Learnable Thresholds      │
│      Head         │    │  upper_threshold (trainable) │
│                   │    │  lower_threshold (trainable) │
│ Outputs quantiles │    │                              │
│ of slope values   │    │  Constraint: lower ≤ upper   │
└─────────┬─────────┘    │  via midpoint + softplus     │
          │              └──────────────┬───────────────┘
          │                             │
          │              ┌──────────────┼────────────────┐
          │              │              │                │
          │              ▼              ▼                ▼
          │         soft / hard    soft / hard      soft / hard
          │        signal_above   signal_below     signal_between
          │
          ▼
     slope_quantiles
     (optional)

Output Dict:
{ "ema", "slope", "signal_above", "signal_below", "signal_between",
  "upper_threshold", "lower_threshold", "slope_quantiles"? }
```

**Why a `keras.Model` and not a `keras.Layer`?** The package exposes a model so it can be `compile`-d, `fit`-ted, saved as `.keras`, and exported to ONNX as a standalone artifact. The core machinery is genuinely layer-like (no internal state across batches); composing it as a sublayer of a bigger model is also supported — pass `inputs` of the right shape and consume any subset of the dict outputs.

---

## 5. Quick Start

```python
import keras
import numpy as np
from dl_techniques.models.adaptive_ema.model import AdaptiveEMASlopeFilterModel

# Synthetic price-like series: 32 samples, 256 bars
prices = np.cumsum(np.random.randn(32, 256), axis=1).astype("float32")

model = AdaptiveEMASlopeFilterModel(
    ema_period=25,
    lookback_period=25,
    initial_upper_threshold=1.5,
    initial_lower_threshold=-1.5,
    learnable_thresholds=False,
)

outputs = model(prices, training=False)
print(outputs["ema"].shape)              # (32, 256)
print(outputs["signal_between"].shape)   # (32, 256)
print(outputs["upper_threshold"])        # 1.5
```

---

## 6. Component Reference

### `AdaptiveEMASlopeFilterModel`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ema_period` | `int` | `25` | Period of the underlying `ExponentialMovingAverage` (`α = 2/(period+1)`). Must be `≥ 1`. |
| `lookback_period` | `int` | `25` | Number of bars back used in the slope: `slope_t = EMA_t − EMA_{t−L}`. Must be `≥ 1`. |
| `initial_upper_threshold` | `float` | `15.0` | Initial upper slope threshold. With `learnable_thresholds=False` this is the fixed cutoff. |
| `initial_lower_threshold` | `float` | `-15.0` | Initial lower slope threshold. Must be `≤ initial_upper_threshold`. |
| `learnable_thresholds` | `bool` | `False` | If `True`, the two thresholds become trainable scalars and training uses soft (sigmoid) signals. |
| `adjust_ema` | `bool` | `True` | Pass-through to the EMA layer's `adjust` flag (Pandas-style bias-corrected EMA). |
| `quantile_head_config` | `Optional[Dict[str, Any]]` | `None` | If provided, attaches a `QuantileSequenceHead`. Required key: `num_quantiles`. Optional: `dropout_rate`, `enforce_monotonicity`, `use_bias`. |

### Output dictionary

| Key | Shape | Notes |
|-----|-------|-------|
| `ema` | `(B, T)` or `(B, T, F)` | EMA of the input. |
| `slope` | `(B, T)` or `(B, T, F)` | EMA slope. First `L` steps are zero-padded — ignore in loss. |
| `signal_above` | same as `slope` | Hard at inference; soft at training when `learnable_thresholds=True`. |
| `signal_below` | same as `slope` | Same. |
| `signal_between` | same as `slope` | Same. |
| `upper_threshold` | `()` | Scalar (after constraint application). |
| `lower_threshold` | `()` | Scalar. |
| `slope_quantiles` | `(B, T, num_quantiles)` | Present only if `quantile_head_config` is set. See Limitations. |

---

## 7. Configuration & Usage Modes

The model has three useful operating modes. Pick by configuration:

### Mode A — Pure rule-based filter (default)
```python
model = AdaptiveEMASlopeFilterModel(
    ema_period=25, lookback_period=25,
    initial_upper_threshold=1.5, initial_lower_threshold=-1.5,
    learnable_thresholds=False,
)
```
- **Trainable params:** 0.
- **Use case:** drop into a backtest, generate signals, never call `fit`. The model is purely inference.

### Mode B — Learnable thresholds
```python
model = AdaptiveEMASlopeFilterModel(
    ema_period=25, lookback_period=25,
    initial_upper_threshold=1.5, initial_lower_threshold=-1.5,
    learnable_thresholds=True,
)
```
- **Trainable params:** 2 (`upper_threshold_var`, `lower_threshold_var`).
- **Use case:** fit the cutoffs to maximize a soft-signal proxy of some downstream PnL or hit-rate target. See "Training & Best Practices."

### Mode C — Probabilistic slope head
```python
model = AdaptiveEMASlopeFilterModel(
    ema_period=25, lookback_period=25,
    quantile_head_config={
        "num_quantiles": 5,
        "dropout_rate": 0.1,
        "enforce_monotonicity": True,
    },
)
```
- **Trainable params:** small Dense projection (size `1 → num_quantiles`).
- **Use case:** experimentation. As of today the head is fed a scalar slope per timestep — see Limitations.

You can combine B and C (learnable thresholds + quantile head) freely.

---

## 8. Comprehensive Usage Examples

### Example 1 — Fixed thresholds, batch inference
```python
from dl_techniques.models.adaptive_ema.model import AdaptiveEMASlopeFilterModel
import numpy as np

prices = np.cumsum(np.random.randn(16, 512), axis=1).astype("float32")

model = AdaptiveEMASlopeFilterModel(
    ema_period=20, lookback_period=10,
    initial_upper_threshold=2.0, initial_lower_threshold=-2.0,
)
out = model(prices, training=False)

# Boolean trade-mask: in regime, slope between thresholds
in_regime = out["signal_between"].numpy().astype(bool)
```

### Example 2 — Learnable thresholds, fit on a labeled mask
```python
import keras, tensorflow as tf

# Wrap the model to return just signal_between so standard losses work
class _BetweenHead(keras.Model):
    def __init__(self, base, **kw):
        super().__init__(**kw)
        self.base = base
    def call(self, x, training=None):
        return self.base(x, training=training)["signal_between"]

base = AdaptiveEMASlopeFilterModel(
    ema_period=25, lookback_period=25,
    initial_upper_threshold=2.0, initial_lower_threshold=-2.0,
    learnable_thresholds=True,
)
wrapper = _BetweenHead(base)
wrapper.compile(optimizer="adam", loss=keras.losses.BinaryCrossentropy())

# x: (B, T)  prices;  y: (B, T)  binary in-regime mask
# wrapper.fit(x, y, epochs=5, batch_size=32)
```

### Example 3 — Quantile head for slope distribution
```python
model = AdaptiveEMASlopeFilterModel(
    ema_period=25, lookback_period=25,
    quantile_head_config={"num_quantiles": 5},
)
out = model(np.random.randn(8, 256).astype("float32"), training=False)
print(out["slope_quantiles"].shape)  # (8, 256, 5)
```

### Example 4 — Serialization round-trip
```python
model.save("adaptive_ema.keras")
restored = keras.saving.load_model("adaptive_ema.keras")
```

---

## 9. Training & Best Practices

A ready-to-run trainer lives at [`src/train/adaptive_ema/train_adaptive_ema.py`](../../../../train/adaptive_ema/train_adaptive_ema.py). It mirrors `src/train/tirex/` (Pattern 2 — time-series / probabilistic) and supports two modes:

- `--mode classification`: trains `signal_between` against a binary "in regime" target derived from the realized future slope, using `BinaryCrossentropy`. Requires `--learnable-thresholds` to be meaningful (the wrapped head only has gradient when thresholds are trainable).
- `--mode quantile`: trains `slope_quantiles` against the realized future slope using `QuantileLoss`.

Headless run (always required on remote / no-X11 machines):
```bash
MPLBACKEND=Agg .venv/bin/python -m train.adaptive_ema.train_adaptive_ema \
    --mode quantile --epochs 10 --batch_size 64 \
    --enable_quantile_head --gpu 0
```

**Best-practice notes**
- **Input scale matters.** The default thresholds (`±15`) were tuned for raw price slopes. For normalized or returns-based inputs, set `initial_upper/lower_threshold` to match — otherwise `signal_between` collapses to 1 (or 0) at initialization. The trainer's default config uses `±1.5` for synthetic price series.
- **Ignore the first `L` bars.** `slope[:, :L]` is zero by construction. Mask them in any loss.
- **Don't expect more than 2 scalars to fit in classification mode.** That is the entire trainable surface. Use the quantile head if you want a real loss landscape.
- **Single GPU only.** Set `--gpu 0` or `--gpu 1`. Never run two trainers in parallel.

---

## 10. Serialization & Deployment

### Keras native (`.keras`)
```python
model.save("adaptive_ema.keras")
restored = keras.saving.load_model("adaptive_ema.keras")
```
Round-trip works because:
- The model is decorated with `@keras.saving.register_keras_serializable()`.
- All constructor args are scalars or a small JSON-able dict.

### ONNX
A standalone exporter is provided at [`src/train/adaptive_ema/export.py`](../../../../train/adaptive_ema/export.py). It mirrors `train/tirex/export.py` (CPU-only env preset, auto input-length detection, optional output-key verification round-trip):

```bash
python -m train.adaptive_ema.export \
    --model_path results/adaptive_ema_xxx/best_model.keras \
    --opset_version 17 --verify --output_key slope_quantiles
```

`--output_key` selects which dict head to verify against the Keras model (default: the first output). ONNX will export all dict outputs; `verify_onnx` only compares one to keep the script simple.

---

## 11. Limitations, Troubleshooting & FAQs

### Limitations (read these before filing issues)

- **L-1. Hard-signal mode is zero-gradient.** With `learnable_thresholds=False`, the three signals are `cast(bool→float)` — they have no gradient. Do **not** apply a loss to `signal_*` in this mode; train on `slope` or `slope_quantiles` instead, or flip to `learnable_thresholds=True`.
- **L-2. The quantile head is fed a scalar slope per timestep.** `QuantileSequenceHead` is a per-timestep Dense projection. Mapping `(B, T, 1) → (B, T, K)` from a 1-D slope leaves the head with very little information to extract. Calibrated uncertainty estimates require either (a) feeding a learned representation of price history into the head, or (b) stacking lagged slopes as features. As wired today the head is best treated as a smoke test, not a production probabilistic forecaster.
- **L-3. Only 2 trainable scalars in classification mode.** Don't expect convergence behavior comparable to even a small MLP. The optimizer is searching a 2-D landscape.
- **L-4. EMA layer is O(T) Python-loop.** The underlying `ExponentialMovingAverage` iterates over the time axis in Python. Acceptable for `T ≤ 256`; expect a slowdown / JIT failure for very long sequences. Keep `input_length` modest.
- **L-5. Learnable-threshold parameterization is mathematically awkward.** The midpoint + `softplus(|ΔU,L|/2)` reparameterization always produces a non-zero floor band-width (`softplus(0) = ln 2 ≈ 0.693`), is non-injective in the raw weight signs, and `abs()` discards gradient signs. Acceptable for the intended toy use case; don't read too much into the learned scalars.
- **L-6. First `lookback_period` slope values are zero-padded.** Ignore them in any loss / metric / signal aggregation.
- **L-7. No factory function / no `__init__.py` re-exports.** Import the class directly from `dl_techniques.models.adaptive_ema.model`. (Future work — see code review.)
- **L-8. No tests.** The package has zero coverage. A follow-up plan should add `tests/test_models/test_adaptive_ema/test_model.py` covering forward pass, serialization, gradient flow, soft-signal training.

### Troubleshooting / FAQs

**Q. My `signal_between` is all ones (or all zeros).** Your thresholds don't match the scale of your input. Either rescale the input (normalize prices to unit-variance returns) or set `initial_upper/lower_threshold` to match the empirical slope magnitude.

**Q. Training loss is identical across epochs.** You are probably in `learnable_thresholds=False` mode and applying a loss to one of the signals — there is no gradient. Flip the flag, or train on `slope_quantiles`.

**Q. Quantile head outputs the same quantile values regardless of input.** Expected gotcha — see L-2. The head doesn't see anything but a scalar slope per timestep.

**Q. Mixed precision warnings on the threshold weights.** The weights are created with `dtype=self.dtype`, which becomes `compute_dtype` under mixed precision. A read-out cast back to `slope.dtype` happens inside `call`, so behavior is correct, but cleaner practice is to keep variables in float32. To be addressed in a future patch.

**Q. ONNX export fails / produces a strange output set.** The model emits a dict. `model.export(format="onnx")` exports all of them; the verifier in `train/adaptive_ema/export.py` only checks one head — pass `--output_key` to select.

---

## 12. When to Use This vs. a Real Forecaster

| You want… | Use… |
|-----------|-----|
| A transparent EMA-slope regime filter, no training needed | **`AdaptiveEMASlopeFilterModel`** with `learnable_thresholds=False` |
| To learn the slope cutoffs for a labeled regime mask | **`AdaptiveEMASlopeFilterModel`** with `learnable_thresholds=True` |
| Calibrated probabilistic forecasts of future prices | [`models/tirex`](../tirex/README.md) or [`models/prism`](../prism/README.md) |
| Interpretable basis-expansion forecasting | [`models/nbeats`](../nbeats/README.md) |
| Mixture-density (heteroscedastic) targets | [`models/mdn`](../mdn/README.md) |

Adaptive EMA is the right tool when you already know your strategy is "trade only when slope is flat" and you just want a small, fast, serializable, ONNX-exportable wrapper for that rule. It is the wrong tool for anything that smells like prediction.

---

## 13. References

- Charles LeBeau & David Lucas — *Technical Traders' Guide to Computer Analysis of the Futures Markets* (1992). Early discussion of EMA-slope regime filtering.
- John Bollinger — Various writings on volatility-band trading rules.
- Koenker & Bassett (1978) — *Regression Quantiles*, Econometrica. Underpins `QuantileLoss` used by the optional head.

**Related code:**
- Layer: `dl_techniques/layers/time_series/ema_layer.py`
- Layer: `dl_techniques/layers/time_series/quantile_head_variable_io.py`
- Trainer: `train/adaptive_ema/train_adaptive_ema.py`
- ONNX export: `train/adaptive_ema/export.py`
- Peer time-series models: `dl_techniques/models/{tirex, prism, nbeats, mdn, deepar}`
