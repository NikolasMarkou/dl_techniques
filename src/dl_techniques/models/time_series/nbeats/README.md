# N-BEATS: Neural Basis Expansion Analysis for Time Series

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of **N-BEATS**, a forecasting architecture built entirely from fully-connected layers and basis expansions — no recurrence, no attention. The package ships two models: `NBeatsNet` (the original) and `NBeatsXNet` (the NBEATSx variant, which adds exogenous covariates through a TCN encoder).

---

## 1. Overview: What is N-BEATS and Why It Matters

### What is N-BEATS?

N-BEATS predicts by **basis expansion**. A deep stack of dense layers learns coefficients (`theta`) for a set of basis functions, and those functions generate the output. Choosing the basis — polynomial, Fourier, or learned — decides whether the model is interpretable or a flexible black box.

### Key Innovations

1. **Doubly residual stacking.** Each block produces a *forecast* and a *backcast* (a reconstruction of its own input). The backcast is subtracted, and the residual goes to the next block. The signal is progressively decomposed.
2. **Interpretable by design.** `TrendBlock` uses polynomials; `SeasonalityBlock` uses a Fourier series. A trend stack followed by a seasonality stack yields a forecast you can read component by component.
3. **No recurrence, no convolution.** Dense layers only, so training is fast and there are no sequential dependencies to unroll.
4. **Reversible instance normalization.** Each input window is normalized by its own mean and std and the forecast is denormalized back, which makes the model robust to the distribution shift that dominates real series.

### Why N-BEATS Matters

```
ARIMA / ETS:        strong statistics, weak on non-linearity and
                    multiple seasonalities.
LSTM / Transformer: powerful, slower, and hard to interrogate when
                    a forecast goes wrong.

N-BEATS:            decompose with residual stacks, constrain each stack
                    to an interpretable basis, normalize per window.
```

---

## 2. The Problem N-BEATS Solves

### Non-stationarity

Real series change their mean, variance, and seasonal shape over time. A model trained before a marketing campaign will not hold after it. And when a black-box model produces a bad forecast, there is no way to ask whether the trend, a seasonal effect, or an event caused it.

```
┌─────────────────────────────────────────────────────────────┐
│  The past is not always like the future                     │
│    - the training distribution drifts away from serving      │
│  The black-box problem                                       │
│    - a bad forecast gives you no component to inspect        │
└─────────────────────────────────────────────────────────────┘
```

### How N-BEATS answers both

- **Instance normalization** makes the modelling task stationary: the network sees zero-mean, unit-variance windows and learns temporal shape, not scale. The forecast is scaled back afterwards.
- **Residual stacks** split the signal: the trend stack models and removes the trend, so the seasonality stack sees a de-trended series and can only explain periodicity. The decomposition is the explanation.

---

## 3. How N-BEATS Works: Core Concepts

```
┌──────────────────────────────────────────────────────────────────┐
│  Input ───► Instance norm ──►┌──────────────────┐ (Trend)        │
│  (B, backcast, D_in)         │     Stack 1      ├─► Forecast₁    │
│                              └────────┬─────────┘                │
│                                       │ residual₁ = x - backcast₁│
│                              ┌────────▼─────────┐ (Seasonality)  │
│                              │     Stack 2      ├─► Forecast₂    │
│                              └────────┬─────────┘                │
│                                       │ residual₂                │
│  Final = Forecast₁ + Forecast₂ + ... ◄── denormalize ◄───────────┘
│  (B, forecast, D_out)  +  final residual (B, backcast * D_in)    │
└──────────────────────────────────────────────────────────────────┘
```

Data flow, step by step:

1. **Normalize.** Compute per-instance mean and std over the time axis; store them.
2. **Stack loop.** Within stack *i*, each block *j* receives residual *j-1*, emits `backcast_j` and `forecast_j`, and updates `residual_j = residual_{j-1} - backcast_j`. The stack forecast is the sum over its blocks.
3. **Aggregate and denormalize.** Sum all stack forecasts, then apply `x * std + mean`. The returned *residual* is scaled by `std` alone — it is a difference of normalized quantities, so re-adding the mean would corrupt it.
4. **Project.** If `input_dim != output_dim`, a final dense layer maps the forecast to `output_dim`.

**`call` returns a tuple**, `(forecast, final_residual)`, not a single tensor. That shapes how you compile and how you read `predict` — see §5 and §11.

---

## 4. Architecture Deep Dive

All blocks live in `dl_techniques/layers/time_series/nbeats_blocks.py`.

### 4.1 Instance normalization (`use_normalization`)

Implemented inline in `NBeatsNet.call`, not as a separate layer. Forward: subtract the per-window mean, divide by the per-window std (floored away from zero). Reverse: multiply by std, add mean. On by default.

### 4.2 `NBeatsBlock` (base class)

Four fully-connected `Dense` layers (default activation `relu`) over the flattened input window, then two linear "theta" heads that project the features into backcast and forecast coefficients. The base class does not generate a signal; its subclasses supply the basis.

### 4.3 `TrendBlock`

Polynomial basis `t^0, t^1, t^2, ...`. `thetas_dim` is the number of polynomial terms, so `thetas_dim=4` is a cubic. The basis is defined on a continuous time index spanning backcast **and** forecast, which is what makes the extrapolation smooth.

### 4.4 `SeasonalityBlock`

Fourier basis. The harmonic count is `thetas_dim // 2` — each harmonic contributes one cosine row and one sine row, at frequency `2πk / (backcast_length + forecast_length)`. With `normalize_basis`, both rows of a harmonic are divided by that harmonic's norm over the full sequence.

### 4.5 `GenericBlock`

A fully learnable linear basis, implemented as two `Dense` layers. Maximum flexibility, zero interpretability.

---

## 5. Quick Start Guide

```python
import keras
import numpy as np
from dl_techniques.models.time_series.nbeats.nbeats import create_nbeats_model

BACKCAST, FORECAST = 96, 24     # 4 days of hourly history -> next day

# The factory returns an UN-compiled model.
model = create_nbeats_model(
    backcast_length=BACKCAST,
    forecast_length=FORECAST,
    stack_types=["trend", "seasonality"],   # interpretable configuration
)

# The model has TWO outputs: (forecast, residual). Compile a loss for the
# forecast head and None for the residual, or give both a loss (see §11).
model.compile(
    optimizer=keras.optimizers.Adam(1e-4, clipnorm=1.0),
    loss=["mae", None],
)

x = np.random.randn(16, BACKCAST, 1).astype("float32")
y = np.random.randn(16, FORECAST, 1).astype("float32")
r = np.zeros((16, BACKCAST), dtype="float32")     # residual target placeholder

print(model.train_on_batch(x, [y, r]))

forecast, residual = model.predict(x)
print(forecast.shape, residual.shape)   # (16, 24, 1) (16, 96)
print(model.count_params())             # 1386336

# Or take the unified contract, which returns the forecast head only.
print(model.predict_forecast(x).point.shape)   # (16, 24, 1)
```

---

## 6. Component Reference

| Component | Location | Purpose |
|:---|:---|:---|
| `NBeatsNet` | `models.time_series.nbeats.nbeats` | The main model. Also a `ForecastMixin`. |
| `create_nbeats_model` | `models.time_series.nbeats.nbeats` | Factory with auto theta dims. Returns un-compiled. |
| `NBeatsXNet` | `models.time_series.nbeats.nbeatsx` | NBEATSx: exogenous covariates via a TCN stack. |
| `create_nbeatsx_model` | `models.time_series.nbeats.nbeatsx` | NBEATSx factory. |
| `NBeatsBlock` | `layers.time_series.nbeats_blocks` | Abstract base for all blocks. |
| `TrendBlock` | `layers.time_series.nbeats_blocks` | Polynomial basis. |
| `SeasonalityBlock` | `layers.time_series.nbeats_blocks` | Fourier basis. |
| `GenericBlock` | `layers.time_series.nbeats_blocks` | Learnable basis. |

All four public names are re-exported by `nbeats/__init__.py` and by the family package, so these all work:

```python
from dl_techniques.models.time_series.nbeats.nbeats import NBeatsNet, create_nbeats_model
from dl_techniques.models.time_series.nbeats import NBeatsNet, create_nbeats_model
from dl_techniques.models.time_series import NBeatsNet, create_nbeats_model
```

---

## 7. Configuration & Model Variants

`stack_types` decides the character of the model.

| Configuration | `stack_types` | Description |
|:---|:---|:---|
| **Interpretable** | `['trend', 'seasonality']` | Standard interpretable setup: trend plus seasonal component. |
| **Generic** | `['generic', 'generic']` | Black box. Usually the most accurate, gives you nothing to read. |
| **Hybrid** | `['trend', 'seasonality', 'generic']` | Strip trend and seasonality, then let a generic stack absorb the remainder. Package default. |

When `thetas_dim` is not given, the factory picks per stack type:

| Stack | Auto `thetas_dim` |
|:---|:---|
| `trend` | `4` (cubic polynomial) |
| `seasonality` | `2 * min(forecast_length // 2, 16)` |
| `generic` | `max(16, forecast_length * 2)` |

---

## 8. Comprehensive Usage Examples

### Example 1: Interpretable univariate

```python
model = create_nbeats_model(
    backcast_length=168,                     # 7 days of hourly data
    forecast_length=24,
    stack_types=["trend", "seasonality"],
    hidden_layer_units=512,
)
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=["mae", None])
```

### Example 2: High-capacity generic

```python
model = create_nbeats_model(
    backcast_length=168,
    forecast_length=24,
    stack_types=["generic", "generic", "generic"],
    nb_blocks_per_stack=4,
    hidden_layer_units=1024,
)
# 41,152,512 parameters -- generic stacks are expensive.
```

### Example 3: Multivariate

```python
model = create_nbeats_model(
    backcast_length=96,
    forecast_length=24,
    stack_types=["trend", "seasonality", "generic"],
    input_dim=7,
    output_dim=3,
    hidden_layer_units=512,
)
forecast, residual = model(np.random.randn(4, 96, 7).astype("float32"))
# forecast (4, 24, 3), residual (4, 672)   # 672 = 96 * 7
```

The window is flattened to `(batch, backcast_length * input_dim)` before the blocks, so cross-variable relationships are learnable. The forecast is produced in `input_dim` space first (so the denormalization statistics line up), then projected to `output_dim`.

### Example 4: NBEATSx with exogenous covariates

`NBeatsXNet` takes a **dict** with three keys and returns a single tensor.

```python
from dl_techniques.models.time_series.nbeats.nbeatsx import create_nbeatsx_model

model = create_nbeatsx_model(
    backcast_length=168, forecast_length=24, exogenous_dim=3,
    stack_types=("trend", "seasonality", "exogenous"),
)
out = model({
    "target_history": np.random.randn(2, 168, 1).astype("float32"),
    "exog_history":   np.random.randn(2, 168, 3).astype("float32"),
    "exog_forecast":  np.random.randn(2,  24, 3).astype("float32"),
})
print(out.shape)   # (2, 24, 1)
```

Note the flag asymmetry: in `NBeatsXNet`, `use_normalization` gates *both* the target normalization and the RMSNorm layers inside every block trunk, while in `NBeatsNet` it gates only the target normalization. `use_block_normalization=None` (the default) reproduces the historical coupling.

---

## 9. Advanced Usage Patterns

### Pattern 1: Hand-tuned theta dimensions

```python
model = create_nbeats_model(
    backcast_length=96,
    forecast_length=24,
    stack_types=["trend", "seasonality"],
    thetas_dim=[2, 20],   # trend: linear (t^0, t^1); seasonality: 10 harmonics
)
```

### Pattern 2: Weight sharing within a stack

`share_weights_in_stack=True` reuses one block object at every position in a stack, as in the original paper's interpretable configuration. It cuts parameters sharply and regularizes small datasets.

---

## 10. Performance Optimization

- **Mixed precision.** N-BEATS is dense-layer bound and gains a lot from `keras.mixed_precision.set_global_policy('mixed_float16')` set before construction.
- **Backcast/forecast ratio.** Use `backcast_length` between 3x and 7x `forecast_length`. Below 3x the blocks have too little context; well above 7x the flattened input inflates the first dense layer for little gain. The factory does not check this for you.
- **Generic stacks dominate the parameter count.** `max(16, 2 * forecast_length)` thetas plus a learnable basis is far heavier than a polynomial or Fourier basis; prefer interpretable stacks when memory matters.

---

## 11. Training and Best Practices

### Compiling a two-output model

`NBeatsNet.call` returns `(forecast, residual)`. Keras therefore expects a loss *structure*, not a single loss:

```python
# Forecast head only -- the residual is ignored.
model.compile(optimizer="adam", loss=["mae", None])

# Or supervise the reconstruction too (what the shipped trainer does when
# reconstruction_loss_weight > 0):
model.compile(
    optimizer="adam",
    loss=[keras.losses.MeanAbsoluteError(),
          keras.losses.MeanAbsoluteError(name="residual_loss")],
    loss_weights=[1.0, 0.1],
)
```

Passing a bare `loss='mae'` raises `KeyError: "The path: (0,) in the loss argument ..."`.

### Other notes

- **Optimizer.** Adam or AdamW. Gradient clipping (`clipnorm=1.0`) is strongly recommended — N-BEATS training can be unstable — but nothing sets it for you.
- **Loss.** MAE is usually preferred over MSE: time series outliers are common and MSE chases them.
- **Learning rate.** `1e-4` with a cosine decay schedule is a reliable starting point.

---

## 12. Serialization & Deployment

Both models register through `register_dl_technique` (`dl_techniques.utils.keras_registration`) — `dl_techniques.models.nbeats.nbeats>NBeatsNet` and `dl_techniques.models.nbeats.nbeatsx>NBeatsXNet` — so `.keras` archives round-trip with no `custom_objects`.

```python
model.save("my_nbeats_model.keras")
loaded = keras.models.load_model("my_nbeats_model.keras")
```

`build()` explicitly builds every block before weight restore, which is what makes the round trip lossless. Under `share_weights_in_stack` a block is built once even though it appears at several positions.

---

## 13. Testing & Validation

`tests/test_models/test_nbeats/` covers configuration validation, serialization, and the forward contract. A minimal multivariate check:

```python
import numpy as np
from dl_techniques.models.time_series.nbeats.nbeats import NBeatsNet

model = NBeatsNet(backcast_length=50, forecast_length=20,
                  input_dim=5, output_dim=3)
forecast, residual = model(np.random.randn(4, 50, 5).astype("float32"))
assert forecast.shape == (4, 20, 3)
assert residual.shape == (4, 250)      # 50 * 5
```

---

## 14. Failure Modes

- **`KeyError: "The path: (0,) in the loss argument ..."`** — you compiled with a single loss. The model has two outputs; see §11.
- **`predict` output does not have the shape you expected** — it is a tuple `(forecast, residual)`. Unpack it, or use `predict_forecast`.
- **Loss goes NaN** — lower the learning rate to `1e-5`-`5e-5`, add `clipnorm=1.0`, and keep `use_normalization=True`.
- **`count_params()` raises "the layer isn't built"** — the factory does not build the model. Call it once on a dummy batch first.
- **Poor accuracy with a short backcast** — a backcast shorter than one full cycle of the dominant seasonality cannot see the pattern at all.

---

## 15. Technical Details

### Doubly residual stacking

With input `x`:

1. Stack 1 (trend) produces `backcast_1`, `forecast_1`; `residual_1 = x - backcast_1`.
2. Stack 2 (seasonality) processes `residual_1`, produces `backcast_2`, `forecast_2`.
3. `forecast = forecast_1 + forecast_2`.

Removing the trend before the seasonality stack sees the signal is what makes the components separable rather than merely additive.

### Basis functions

- **Trend (polynomial):** `y = Σ_{i=0}^{p} θ_i t^i`, with `t` a normalized time vector over backcast + forecast and `p = thetas_dim - 1`.
- **Seasonality (Fourier):** `y = Σ_{k=1}^{n} [ θ_{k,1} cos(2πkt) + θ_{k,2} sin(2πkt) ]`, with `n = thetas_dim // 2` harmonics.

---

## 16. Citation

- **N-BEATS**:
    ```bibtex
    @inproceedings{oreshkin2020n,
      title={N-BEATS: Neural basis expansion analysis for interpretable time series forecasting},
      author={Oreshkin, Boris N and Carpov, Dmitri and Chapados, Nicolas and Bengio, Yoshua},
      booktitle={International Conference on Learning Representations},
      year={2020}
    }
    ```
- **NBEATSx**:
    ```bibtex
    @article{olivares2023neural,
      title={Neural basis expansion analysis with exogenous variables:
             Forecasting electricity prices with NBEATSx},
      author={Olivares, Kin G and Challu, Cristian and Marcjasz, Grzegorz and
              Weron, Rafa{\l} and Dubrawski, Artur},
      journal={International Journal of Forecasting},
      volume={39}, number={2}, pages={884--900}, year={2023}
    }
    ```
- **RevIN** (the reversible instance normalization idea):
    ```bibtex
    @inproceedings{kim2022reversible,
      title={Reversible Instance Normalization for Accurate Time-Series
             Forecasting against Distribution Shift},
      author={Kim, Taesung and Kim, Jinhee and Tae, Yungi and Park, Cheonbok
              and Choi, Jang-Ho and Choo, Jaegul},
      booktitle={International Conference on Learning Representations},
      year={2022}
    }
    ```
