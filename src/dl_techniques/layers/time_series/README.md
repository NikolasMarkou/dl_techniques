# Time Series Layers Module

The `dl_techniques.layers.time_series` module provides a comprehensive collection of advanced neural network layers designed specifically for time series forecasting, signal processing, and sequence modeling.

## Overview

This module consolidates state-of-the-art architectures (like N-BEATS and xLSTM), probabilistic output heads, and scientifically grounded forecasting utilities into production-ready Keras 3 components.

## Available Components

### 1. Advanced Architectures (xLSTM, N-BEATS, PRISM)

| Class | Description | Use Case |
|---|---|---|
| `sLSTMBlock` | Scalar xLSTM with exponential gating | Sequence modeling requiring long-term memory |
| `mLSTMBlock` | Matrix xLSTM with parallelizable memory | High-throughput sequence processing |
| `GenericBlock` | N-BEATS block with learnable basis | General purpose interpretable forecasting |
| `TrendBlock` | N-BEATS block with polynomial basis | Modeling trend components explicitly |
| `SeasonalityBlock` | N-BEATS block with Fourier basis | Modeling periodic/seasonal patterns |
| `ExogenousBlock` | N-BEATSx block with TCN encoder | Incorporating covariates into N-BEATS |
| `PRISMLayer` | Hierarchical wavelet decomposition | Multi-resolution time-frequency analysis |
| `MixedSequentialBlock` | Hybrid LSTM/Transformer block | capturing both local and global dependencies |

### 2. Probabilistic & Output Heads

| Class | Description | Use Case |
|---|---|---|
| `QuantileHead` | Fixed-horizon quantile projection | Horizon-based probabilistic forecasting |
| `QuantileSequenceHead` | Sequence-to-sequence quantiles | Pointwise uncertainty estimation |
| `GaussianLikelihoodHead` | DeepAR Gaussian parameters | Probabilistic forecasting for real-values |
| `NegBinLikelihoodHead` | DeepAR Negative Binomial params | Probabilistic forecasting for count data |
| `TemporalFusionLayer` | Gated Context/AR fusion | Combining deep features with autoregression |
| `AdaptiveLagAttention` | Context-aware lag weighting | Dynamic feature selection from history |

### 3. Scientific Forecasting & Signal Processing

| Class | Description | Use Case |
|---|---|---|
| `NaiveResidual` | Enforces Naive Benchmark Principle | Learning only value-added over random walk |
| `ForecastabilityGate` | Complexity-based switching | Preventing overfitting on noisy data |
| `ScaleLayer` | Instance-wise scaling | Handling power-law scale distributions |
| `EMASlopeFilter` | Adaptive EMA slope detection | Generating trend-based trading signals |

## Usage Examples

### N-BEATS Architecture
Constructing an interpretable stack using Trend and Seasonality blocks.

```python
from dl_techniques.layers.time_series import TrendBlock, SeasonalityBlock

# Create a trend block
trend = TrendBlock(
    units=256,
    thetas_dim=4,  # Polynomial degree
    backcast_length=168,
    forecast_length=24
)

# Create a seasonality block
seasonality = SeasonalityBlock(
    units=256,
    thetas_dim=8,  # Number of harmonics
    backcast_length=168,
    forecast_length=24
)

# Input shape: (batch, backcast_length * input_dim)
inputs = keras.Input(shape=(168 * 1,))
b1, f1 = trend(inputs)
# Subtract backcast from input (doubly residual stacking)
residual = inputs - b1 
b2, f2 = seasonality(residual)

final_forecast = f1 + f2