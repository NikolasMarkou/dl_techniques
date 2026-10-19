# Timestral

Keras 3 implementation of the [Timestral](https://mistral.ai/news/timestral) family of time series foundation models for multivariate probabilistic forecasting.

## Overview

Timestral is a patch-based transformer trained with Decoupled Horizon Forecasting (DHF), where forecasts are generated from separate prediction tokens that read the causally encoded context. Each variate is a target, a past-only covariate, or a known-future covariate: covariates pass information to targets but never read them, and only known-future covariates see the forecast horizon. A quantile head outputs 21 non-crossing quantiles per time step, and each forecast patch is denormalized with the statistics of the context and the block's earlier predicted medians.

## Features

- **Multivariate**: Variates of a series are forecast jointly
- **Covariates**: Past-only, known-future, and categorical covariates supported
- **Probabilistic**: 21 quantiles per time step (0.01 to 0.99)
- **Long context**: Up to 32,768 steps
- **Block-autoregressive decoding**: Horizons longer than one decode block (32 patches = 1024 steps) are forecast block by block
- **Test-time augmentation**: Symmetric averaging (clampsym) and non-negative clipping

## Models

Four model sizes are available, matching the Hugging Face Hub releases:

| Variant | d_model | num_heads | layers | mlp_dim | params |
|---------|---------|-----------|--------|---------|--------|
| 5M | 256 | 8 | 8 | 1024 | ~5M |
| 28M | 512 | 8 | 12 | 2048 | ~28M |
| 123M | 768 | 12 | 24 | 3072 | ~123M |
| 209M | 1024 | 16 | 24 | 4096 | ~209M |

## Installation

```bash
pip install -e .  # from repo root with dl_techniques
```

Requires: `tensorflow>=2.18`, `keras>=3.8`, `safetensors` (checkpoint files only), `huggingface_hub` (optional, for Hub loading)

## Quick Start

### Inference with Pretrained Weights

```python
import numpy as np
from dl_techniques.models.time_series import TimestralForecaster

forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-209M")

# Context: (batch, variates, time) - NaN marks missing values
context = np.random.randn(2, 1, 2000).cumsum(-1)
forecast = forecaster.forecast(context, prediction_length=96)

print(forecast.quantiles.shape)   # (2, 1, 96, 21)
print(forecast.quantile_levels)   # (0.01, 0.05, ..., 0.95, 0.99)
print(forecast.median.shape)      # (2, 1, 96)
print(forecast.quantile(0.9).shape)  # (2, 1, 96)
```

### With Covariates

```python
forecast = forecaster.forecast(
    context,
    prediction_length=96,
    past_covariates=past,           # (B, C_past, T)
    known_covariates=known,         # (B, C_known, T + 96)
)
```

### Categorical Covariates

```python
forecast = forecaster.forecast(
    context,
    prediction_length=96,
    categorical_past_covariates=codes_past,     # (B, C, T) numeric codes, NaN = missing
    categorical_known_covariates=codes_known,   # (B, C, T + 96)
)
```

### Training

```bash
MPLBACKEND=Agg .venv/bin/python -m train.time_series.timestral.train_timestral \
    --variant 28M \
    --prediction_length 96 \
    --epochs 100 \
    --batch_size 32
```

## Architecture

- **Patch embedding**: Each patch of 32 time steps is embedded from `[values, missing]` channels
- **Time/Variate attention**: Alternating attention over time axis (with RoPE) and variate axis
- **DHF decoding**: Decoder tokens attend to full context; block-autoregressive for long horizons
- **Quantile head**: Monotone quantiles via softplus cumulative sum from median
- **Normalization**: Causal per-patch mean/std with asinh squash; advancing refine frame for decoding
- **Covariate encoding**: Categorical covariates target-encoded per series

## API Reference

### `TimestralConfig`

Configuration dataclass for model architecture.

```python
config = TimestralConfig(
    patch_size=32,
    d_model=512,
    num_heads=8,
    mlp_hidden_dim=2048,
    block_schedule=("time", "time_variate", ...),
    quantile_knots=(0.01, 0.05, ..., 0.99),
    block_size=32,
    context_length=32768,
    # ... other params
)
```

### `TimestralModel(config)`

Core Keras Model implementing the transformer.

```python
model = TimestralModel(config)
# For training, use TimestralTrainer which calls model.predict_block directly
```

`TimestralModel.from_variant(variant, prediction_length=96, patch_size=32, quantile_levels=None, **kwargs)`
builds one of the four sizes in `TimestralModel.MODEL_VARIANTS`; both factories below delegate to it.

`model.build(...)` creates every weight. `predict_block` does not mark the model built, so call
`model.build(...)` on a fresh model before `model.save(...)`; the `.keras` round trip then restores
every weight exactly.

### `TimestralForecaster.from_pretrained(repo_id)`

High-level forecaster with block-autoregressive decoding and clampsym.

```python
forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-209M")
forecast = forecaster.forecast(context, prediction_length=96)
```

### `create_timestral_model(variant, prediction_length, ...)`

Factory for creating models from predefined variants.

```python
model = create_timestral_model("28M", prediction_length=96)
```

### `create_timestral_by_variant(variant, prediction_length, ...)`

Alias for `create_timestral_model`.

## Weight Loading

Weights are loaded from Hugging Face Hub (`mistralai/Timestral-{5M,28M,123M,209M}`) or local directories containing `config.json` and `model.safetensors`. The loading uses `safetensors` with `strict=True`.

## Training

The training pipeline follows Pattern 2 (time-series / probabilistic) from the codebase conventions:

- Synthetic pattern generators from `dl_techniques.datasets.time_series`
- `QuantileLoss` with configurable quantile levels
- Warmup + cosine LR schedule
- Per-epoch quantile band visualization
- ONNX export support

## Differences from Original PyTorch Implementation

- Pure Keras 3 implementation using `keras.ops` for backend-agnostic ops
- Full `get_config()` / `from_config()` round-trip serialization
- Registered with `@register_dl_technique("dl_techniques.models.timestral.model")`
- Integrates with `dl_techniques` training infrastructure (`BaseTimeSeriesTrainer`)
- Forecaster is pure Keras 3 + NumPy; checkpoints are read with a NumPy safetensors
  parser (no PyTorch anywhere in this package)

## License

Apache 2.0 (code). Model weights on Hugging Face Hub are under a Modified MIT license.