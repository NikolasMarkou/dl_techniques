# `models/time_series/` — the forecasting family

There is no README here until 2026-08-18, but **every subpackage already had one**;
this file is the index and the shared-contract page, not a summary of the seven.

## The subpackages

| package | what it is | README |
|:---|:---|:---|
| `adaptive_ema/` | Adaptive EMA slope filter | [README](adaptive_ema/README.md) |
| `deepar/` | Autoregressive probabilistic forecasting (Monte-Carlo sampled) | [README](deepar/README.md) |
| `mdn/` | Mixture-density heads | [README](mdn/README.md) |
| `nbeats/` | N-BEATS basis expansion, plus the exogenous `nbeatsx` variant | [README](nbeats/README.md) |
| `prism/` | Partitioned / wavelet-band forecasting with a soft router | [README](prism/README.md) |
| `tirex/` | Mixed sequential (SSM + attention) forecaster | [README](tirex/README.md) |
| `xlstm/` | Extended LSTM, with a dedicated `xLSTMForecaster` wrapper | [README](xlstm/README.md) |

They are **independent architectures sharing no backbone**, so there is deliberately
no cross-family variant table and no common `MODEL_VARIANTS`. Each package's own
README is the authority for its variants and shapes.

## What they DO share: `forecast.py`

`forecast.py` is the one thing that spans the family — a uniform inference contract
so a caller does not branch on the concrete model type.

**`Forecast`** is a plain `@dataclass` of already-materialized numpy arrays. It is
inert data: not a Keras layer, never registered with Keras, never serialized into a
`.keras` file.

| field | shape | notes |
|:---|:---|:---|
| `point` | `[B, H, F]` | required — point / median forecast |
| `quantiles` | `[B, H, F, Q]` or `None` | `None` for point models |
| `quantile_levels` | `Q` floats | must match `quantiles` |
| `samples` | `[S, B, H, F]` or `None` | Monte-Carlo draws, where a model produces them |

**`ForecastMixin`** gives a model `predict_forecast(x) -> Forecast`, delegating to a
model-specific `_forecast` hook.

**The point-vs-probabilistic rule is load-bearing.** A point model exposes
`quantiles=None` and **must not fabricate intervals**; a probabilistic model
populates `quantiles` together with the matching `quantile_levels`. Callers test
`forecast.has_quantiles()` (or catch the `ValueError` from `forecast.interval(low,
high)`) rather than checking which class they were handed.

## Import surface

Everything is re-exported at the package root:

```python
from dl_techniques.models.time_series import (
    AdaptiveEMASlopeFilterModel, create_adaptive_ema_slope_filter,
    DeepAR, create_deepar,
    MDNModel, create_mdn_model,
    NBeatsNet, create_nbeats_model,
    NBeatsXNet, create_nbeatsx_model,
    PRISMModel, create_prism_model,
    TiRexCore, create_tirex_model, create_tirex_by_variant,
    xLSTM, create_xlstm,
    xLSTMForecaster, create_xlstm_forecaster,
)
```

`tirex/`, `prism/` and `deepar/` have empty `__init__.py` files, so import them
through this root or through their `.model` submodule, not from the subpackage name.
