# Time Series Layers Module

`dl_techniques.layers.time_series` holds the layers used to build forecasting and
sequence models: N-BEATS blocks, xLSTM cells and blocks, the PRISM wavelet tree,
DeepAR-style probabilistic heads, quantile heads, and a few signal-processing
utilities.

## Overview

The package exports **33 names**, listed in `__all__` in `__init__.py`. Every
one of them appears in the tables below.

Two things to know before you start:

- **There is no factory.** Unlike `layers/ffn/` and `layers/attention/`, this
  package has no `create_*_layer` dispatcher. Import the class you want by name.
- **One export is not a layer.** `create_manokhin_compliant_model` is a builder
  that returns a full `keras.Model`. Everything else is a `keras.Layer`.

## Available Components

### 1. Sequence encoders and architecture blocks

| Name | Description | Use case |
|---|---|---|
| `MixedSequentialBlock` | Pre-LN block running LSTM, self-attention, or both | Capturing local and global dependencies in one block |
| `TemporalConvNet` | Stack of dilated causal residual blocks | Encoding a long history with a fixed receptive field |
| `TemporalBlock` | One residual block of a TCN | Building a custom TCN stack level by level |
| `sLSTMBlock` | sLSTM recurrence + norm + FFN, post-norm, residual | Sequence modeling needing long-term memory (xLSTM Fig. 10) |
| `sLSTMLayer` | Runs `sLSTMCell` over a sequence | Using the sLSTM recurrence without the block wrapper |
| `sLSTMCell` | Scalar LSTM cell, exponential gating, normalizer state | Feeding a custom `keras.layers.RNN` |
| `mLSTMBlock` | Up-projection, causal Conv1D, mLSTM, down-projection | High-throughput sequence processing (xLSTM Fig. 11) |
| `mLSTMLayer` | Runs `mLSTMCell` over a sequence | Using the mLSTM recurrence without the block wrapper |
| `mLSTMCell` | Matrix LSTM cell with a matrix memory and covariance update | Feeding a custom `keras.layers.RNN` |

### 2. N-BEATS family

| Name | Description | Use case |
|---|---|---|
| `NBeatsBlock` | Base class: 4-layer dense stack + dual theta projection | Subclass it and supply the two basis methods |
| `GenericBlock` | N-BEATS block with a learnable Dense basis | Flexible, least interpretable forecasting |
| `TrendBlock` | N-BEATS block with a polynomial basis | Modeling trend explicitly |
| `SeasonalityBlock` | N-BEATS block with a Fourier basis | Modeling periodic and seasonal patterns |
| `ExogenousBlock` | N-BEATSx block whose basis is built from covariates | Bringing exogenous inputs into N-BEATS |

`NBeatsBlock` is the shared base. Its `_generate_backcast` and
`_generate_forecast` are marked `@abstractmethod`, but a Keras layer is not an
ABC, so Python will still let you instantiate it. Use one of the four
subclasses instead.

### 3. PRISM (multi-resolution decomposition)

| Name | Description | Use case |
|---|---|---|
| `PRISMLayer` | Time tree + dropout + optional residual and norm | The PRISM entry point; channel count is unchanged |
| `PRISMTimeTree` | Runs a node bank over the sequence once per tree level | Building PRISM into a larger model directly |
| `PRISMNode` | Splits one segment into bands and recombines them by weight | One node of the tree |
| `FrequencyBandRouter` | Scores frequency bands, turns scores into weights | Deciding which band matters for a segment |
| `FrequencyBandStatistics` | Six summary statistics for one band | Feature input to the router |

### 4. Probabilistic and output heads

| Name | Description | Use case |
|---|---|---|
| `QuantileHead` | Fixed-horizon quantile projection | Horizon-based probabilistic forecasting |
| `QuantileSequenceHead` | Quantiles at every sequence position | Pointwise uncertainty estimation |
| `GaussianLikelihoodHead` | DeepAR Gaussian parameters (`mu`, `sigma`) | Probabilistic forecasting for real values |
| `NegativeBinomialLikelihoodHead` | DeepAR negative-binomial parameters | Probabilistic forecasting for count data |
| `DeepARCell` | One autoregressive DeepAR step, wrapping `LSTMCell` | Feeding a `keras.layers.RNN` |
| `ScaleLayer` | Divides inputs by a per-item scale, or multiplies back up | Handling series whose magnitudes differ by orders |
| `TemporalFusionLayer` | Gated blend of a context forecast and a lag forecast | Combining deep features with autoregression |
| `AdaptiveLagAttentionLayer` | Per-lag sigmoid weights plus a master gate | Dynamic feature selection from history |

`AdaptiveLagAttentionLayer` uses sigmoid rather than softmax, so the lag weights
do not compete: several lags can be important at once.

### 5. Scientific forecasting and signal processing

| Name | Description | Use case |
|---|---|---|
| `NaiveResidual` | Adds the network output on top of a random-walk baseline | Learning only what beats the naive forecast |
| `ForecastabilityGate` | Learned per-sample blend of deep and naive forecasts | Preventing overfitting on noisy series |
| `ConformalQuantileHead` | Emits lower/median/upper; `predict_intervals` widens them by `q_hat` | Conformalized quantile regression |
| `create_manokhin_compliant_model` | Builder wiring the three layers above into a model | Getting a compliant baseline in one call |
| `ExponentialMovingAverage` | EMA over the time axis | Smoothing a series |
| `EMASlopeFilter` | EMA slope plus three 0/1 threshold signals | Generating trend-based trading signals |

`EMASlopeFilter` has four `output_mode` values (`all`, `signals_only`,
`ema_only`, `slope_only`) and the return changes with each. `all` returns five
tensors: the EMA, the slope, and the three signals.

## Usage Examples

Every example below has been executed as written.

### N-BEATS architecture

Interpretable stack built from a trend block and a seasonality block.

```python
import keras
from dl_techniques.layers.time_series import TrendBlock, SeasonalityBlock

# 4 polynomial rows: degrees 0, 1, 2, 3.
trend = TrendBlock(
    units=256,
    thetas_dim=4,
    backcast_length=168,
    forecast_length=24,
)

# 8 basis rows: 4 harmonics, each a cosine row and a sine row.
seasonality = SeasonalityBlock(
    units=256,
    thetas_dim=8,
    backcast_length=168,
    forecast_length=24,
)

# Input is flat: (batch, backcast_length * input_dim).
inputs = keras.Input(shape=(168 * 1,))
b1, f1 = trend(inputs)

# Doubly residual stacking: each block sees what the last one could not explain.
residual = inputs - b1
b2, f2 = seasonality(residual)

final_forecast = f1 + f2
model = keras.Model(inputs=inputs, outputs=final_forecast)
print(model.output_shape)  # (None, 24)
```

`thetas_dim` is a row count, not a degree and not a harmonic count. For
`TrendBlock` it is the number of powers of time. For `SeasonalityBlock` it is
the number of basis rows, filled by `thetas_dim // 2` harmonics.

### xLSTM stack

```python
import keras
from dl_techniques.layers.time_series import sLSTMBlock, mLSTMBlock

inputs = keras.Input(shape=(64, 128))
x = sLSTMBlock(units=128)(inputs)
x = mLSTMBlock(units=128, num_heads=4)(x)
model = keras.Model(inputs=inputs, outputs=x)
print(model.output_shape)  # (None, 64, 128)
```

### PRISM multi-resolution block

```python
import keras
from dl_techniques.layers.time_series import PRISMLayer

inputs = keras.Input(shape=(128, 32))
outputs = PRISMLayer(tree_depth=2, num_wavelet_levels=3)(inputs)
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.output_shape)  # (None, 128, 32)
```

### DeepAR probabilistic forecasting

```python
import keras
from dl_techniques.layers.time_series import (
    DeepARCell,
    ScaleLayer,
    GaussianLikelihoodHead,
    NegativeBinomialLikelihoodHead,
)

inputs = keras.Input(shape=(48, 8))
scaled = ScaleLayer()(inputs)
hidden = keras.layers.RNN(DeepARCell(units=64), return_sequences=True)(scaled)

mu, sigma = GaussianLikelihoodHead(units=1)(hidden)
total_count, probs = NegativeBinomialLikelihoodHead(units=1)(hidden)

model = keras.Model(inputs=inputs, outputs=[mu, sigma, total_count, probs])
print([o.shape for o in model.outputs])
# [(None, 48, 1), (None, 48, 1), (None, 48, 1), (None, 48, 1)]
```

### Encoder, lag attention and quantile heads

```python
import keras
from dl_techniques.layers.time_series import (
    TemporalConvNet,
    AdaptiveLagAttentionLayer,
    TemporalFusionLayer,
    QuantileHead,
    QuantileSequenceHead,
)

context = keras.Input(shape=(168, 16))
lags = keras.Input(shape=(12,))

encoded = TemporalConvNet(filters=32, kernel_size=2, num_levels=4)(context)
pooled = keras.layers.GlobalAveragePooling1D()(encoded)

weighted_lags = AdaptiveLagAttentionLayer(num_lags=12)([pooled, lags])
fused = TemporalFusionLayer(output_dim=24, num_lags=12)([pooled, lags])

fixed_q = QuantileHead(num_quantiles=3, output_length=24)(pooled)
seq_q = QuantileSequenceHead(num_quantiles=3)(encoded)

model = keras.Model(inputs=[context, lags],
                    outputs=[weighted_lags, fused, fixed_q, seq_q])
print([o.shape for o in model.outputs])
# [(None, 1), (None, 24), (None, 24, 3), (None, 168, 3)]
```

`TemporalConvNet` is built from `TemporalBlock` levels, so use `TemporalBlock`
only when you want to control the dilation schedule yourself.

### Naive-benchmark model builder

```python
import numpy as np
from dl_techniques.layers.time_series import create_manokhin_compliant_model

model = create_manokhin_compliant_model(
    input_shape=(168, 1),
    forecast_length=24,
    hidden_units=128,
)

outputs = model.predict(np.random.randn(4, 168, 1).astype("float32"), verbose=0)
print([o.shape for o in outputs])   # [(4, 24, 1), (4, 24, 1, 3)]
print(model.output_shape)           # [(None, 24, 1), (None, 24, 1, 3)]
```

The builder returns a two-output model: the gated point forecast and the raw
quantiles. The declared shapes match the runtime ones.

`ForecastabilityGate` takes `deep_forecast` and `pure_naive` as extra call
arguments, not as part of `inputs`, so under the functional API Keras hands its
`compute_output_shape` only the backcast shape — the layer cannot infer its own
forecast length. The builder therefore passes `forecast_length=` down to the
gate, and `compute_output_shape` substitutes it for the time axis. If you
construct a `ForecastabilityGate` yourself and omit `forecast_length`, the
older behaviour is kept: the declared shape is the backcast shape, a warning is
logged, and you must read the runtime shape instead.

### EMA smoothing and slope signals

```python
import keras
from dl_techniques.layers.time_series import ExponentialMovingAverage, EMASlopeFilter

inputs = keras.Input(shape=(256, 1))
smoothed = ExponentialMovingAverage(period=25)(inputs)
signals = EMASlopeFilter(ema_period=25, lookback_period=25, output_mode="all")(inputs)

model = keras.Model(inputs=inputs, outputs=[smoothed, signals])
print([o.shape for o in model.outputs])
# [(None, 256, 1), (None, 256, 1), (None, 256, 1),
#  (None, 256, 1), (None, 256, 1), (None, 256, 1)]
```
