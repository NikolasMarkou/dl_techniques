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

Four model sizes are available. `TimestralModel.MODEL_VARIANTS` is read from the `config.json` of each Hugging Face Hub release; every other field is a `TimestralConfig` default there, except `quantile_width_factor`, which is 3.0 in all four. Each schedule repeats `(time, time, time, time_variate)`.

| Variant | d_model | num_heads | layers | mlp_dim | trainable params |
|---------|---------|-----------|--------|---------|------------------|
| 5M | 192 | 3 | 8 | 512 | 5,208,960 |
| 28M | 384 | 6 | 12 | 1024 | 27,821,184 |
| 123M | 640 | 10 | 20 | 1712 | 122,606,720 |
| 209M | 768 | 12 | 24 | 2048 | 209,312,256 |

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
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.time_series.timestral.train_timestral \
    --variant 5M \
    --input_length 1024 \
    --prediction_length 96 \
    --epochs 20 --steps_per_epoch 500 --batch_size 64 --learning_rate 3e-4 \
    --no-deep-analysis \
    --result_dir <run root>
```

This trains from scratch on synthetic series. Fine-tuning released weights is not wired into the trainer.

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
quantiles = model(context)  # (B, T) -> (B, H, Q); (B, T, V) -> (B, H, V, Q)
```

`model(context)` is the training forward: one decode block after each context, the same
path as `predict_block`, every horizon patch in the frame of the last context patch. The batch
size must be static (batch a `tf.data` pipeline with `drop_remainder=True`). `compute_loss` is
the training objective, so `model.compile(optimizer=...)` and `model.fit(context, horizon)` train
it with no `loss=` argument; see Training below.

`TimestralModel.from_variant(variant, prediction_length=96, patch_size=32, quantile_levels=None, **kwargs)`
builds one of the four sizes in `TimestralModel.MODEL_VARIANTS`; both factories below delegate to it.

`model.build(...)` creates every weight. `predict_block` does not mark the model built, so call
`model.build(...)` on a fresh model before `model.save(...)` (it also completes a model that
`predict_block` already built); the `.keras` round trip then restores every weight exactly. A model
from `load_model` or `TimestralForecaster.from_pretrained` is already built.

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

Weights are loaded from Hugging Face Hub (`mistralai/Timestral-{5M,28M,123M,209M}`) or local directories containing `config.json` and `model.safetensors`, read with a NumPy safetensors parser.

- The weight file uses the reference PyTorch state-dict names and layouts, the format of the released checkpoints. `loading.reference_weight_map(model)` derives each name from the layer structure, never from `Variable.path`, so a lazily built, an explicitly built and a loaded model all map the same way.
- Linear weights are transposed, and the fused QKV projection is reordered from the reference's `(qkv, head_dim, heads)` split to this port's `(qkv, heads, head_dim)`.
- Loading is strict: a missing, unused or misshapen tensor raises.
- `loading.save_checkpoint(model, directory)` writes the same format; `load_model` of its output restores every weight exactly.
- `load_model` builds the model explicitly, so a loaded model is marked built and `model.save(...)` round-trips every weight.
- On the released Timestral-5M weights, run in true float32, the port matches the reference PyTorch package: `predict_block` to within 1e-4 (values up to 75), a 1200-step forecast with clampsym to within 2.2e-3 (values up to 370). `tests/test_models/test_timestral/test_torch_reference.py` checks `predict_block` and the multi-block forecaster against a frozen reference fixture, without torch.
- `TimestralModel(config, dtype=...)` builds and computes in that dtype policy in every sublayer; without `dtype` it takes the global policy. float32, bfloat16 and mixed_bfloat16 are accepted. A float16 compute dtype raises `ValueError`: on the released 5M weights, with every layer in float16, the forecasts stayed finite but their error against the true continuation was 62 times the float32 error, while bfloat16 matched float32 (plan-2026-10-14T062616-81938e81 D-178; `tests/test_models/test_timestral/test_dtype.py`). The same check on the 28M, 123M and 209M releases (16 synthetic series, 96-step forecasts, GPU) keeps bfloat16 within 2.5% of the float32 error on each: the mean absolute error of the median forecast per unit scale, at value scale 1, is 0.1311 against 0.1279, 0.1280 against 0.1278, and 0.1267 against 0.1252 (D-182).
- On an NVIDIA GPU, TensorFlow runs float32 matmuls in TF32 by default. On the same 5M weights that moved a 512-step-context `predict_block` by 0.048 and the 1200-step forecast by 49, because each block's median is fed back as context. For reference-faithful long horizons on a GPU, turn TF32 off first: `tf.config.experimental.enable_tensor_float_32_execution(False)`. The forecaster does not change this process-wide flag for you.

## Training

The reference repository publishes inference code only. The training objective comes from the Mistral announcement: "The objective is a pinball loss over 21 quantiles, a discrete approximation of CRPS." The port implements it as follows (plan-2026-10-14T062616-81938e81 D-169):

- `TimestralModel.call(context)` is the training forward. It runs one DHF decode block after each context, the context-only path of `predict_block`, with byte-identical output.
- `TimestralModel.compute_loss` is the objective, and stock `fit` calls it. It scores the pinball loss over the model's quantile levels in normalized (asinh) space.
- Each horizon patch is scored in its own frame: the causal statistics of the true series up to the end of the previous patch. This is the advancing refine frame the forecaster decodes in, with perfect medians.
- `NaN` targets are left out, and `sample_weight` weights whole series.
- Two choices are this port's own, because the reference does not publish them: scoring one block at the end of each context, and the asinh space.
- `compile(optimizer=...)` needs no `loss=`. The batch size must be static.

`train.time_series.timestral.train_timestral` trains it on the synthetic pattern generators from `dl_techniques.datasets.time_series`:

- Warmup and cosine LR.
- Held-out evaluation with `TimestralForecaster` on the test split. It records MAE, rMAE, MASE and coverage, and also the MAE of the naive (last value) and seasonal-naive forecasts of the same windows. The seasonal-naive period is chosen per series by a backtest on its own context.
- Per-epoch forecast plots, drawn by the forecaster.

Measured with `5M` from scratch, the command above, one run on GPU 1 (20 epochs x 500 steps). Training loss fell from 0.209 to 0.115 and validation loss from 0.179 to 0.117. All MAE values below are held out.

| forecast | trainer's test split, 4416 windows | 256 fresh test-split windows, 69 patterns |
|---|---|---|
| Timestral, clampsym on | 6.63 | 2.41 |
| Timestral, clampsym off | not measured | 2.17 |
| naive (last value) | 17.71 | 3.31 |
| seasonal naive | 7.94 | 3.16 |

- On the fresh windows the model beats naive on 83% of windows and seasonal naive on 87.5%.
- It loses to naive on the `industrial` and `regime` categories.
- Coverage of the 0.01-0.99 band on the trainer's split is 0.971, against a nominal 0.98.
- These numbers are for synthetic data only. They say nothing about the benchmark accuracy of the released checkpoints.

## Differences from Original PyTorch Implementation

- Pure Keras 3 implementation using `keras.ops` for backend-agnostic ops
- Full `get_config()` / `from_config()` round-trip serialization
- Registered with `@register_dl_technique("dl_techniques.models.timestral.model")`
- Integrates with `dl_techniques` training infrastructure (`BaseTimeSeriesTrainer`)
- Adds a training forward and objective (`call`, `compute_loss`); the reference ships inference only
- The fused QKV projection splits as `(qkv, heads, head_dim)` here and `(qkv, head_dim, heads)` in the reference; the loader and `save_checkpoint` reorder it
- Forecaster is pure Keras 3 + NumPy; checkpoints are read with a NumPy safetensors
  parser (no PyTorch anywhere in this package)

## License

Apache 2.0 (code). Model weights on Hugging Face Hub are under a Modified MIT license.