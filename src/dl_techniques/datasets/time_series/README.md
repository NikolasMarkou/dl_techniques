# Time Series Data Module

A production-ready module for the `dl_techniques` framework that provides end-to-end support for time series deep learning. This module includes state-of-the-art benchmark loaders, a comprehensive synthetic data generator, robust normalization tools, and efficient TensorFlow data pipelines.

## Directory Structure

```text
dl_techniques/datasets/time_series/
├── __init__.py       # Exports
├── base.py           # Abstract base classes
├── config.py         # Configuration dataclasses (Benchmarks & Pipelines)
├── generator.py      # Synthetic data generator & GeneratorConfig
├── normalizer.py     # Advanced reversible normalization
├── pipeline.py       # tf.data pipeline utilities
├── utils.py          # Download/extraction helpers
├── long_horizon.py   # ETT, ECL, Weather, Traffic
├── m4.py             # M4 Competition loader
└── favorita.py       # Favorita Grocery Sales loader
```

## 1. Benchmark Datasets

Load popular academic and competition datasets with automatic downloading, caching, and parsing.

### Long Horizon Benchmarks (ETT, Weather, Traffic)

Used for evaluating Transformer-based models (Informer, Autoformer, etc.).

```python
from dl_techniques.datasets.time_series import (
    LongHorizonDataset, 
    load_ett, 
    LONG_HORIZON_CONFIGS
)

# Option 1: Direct Loader
dataset = LongHorizonDataset(root_dir='./data')
# Returns: Y_df (targets), X_df (exogenous), S_df (static - None for ETT)
y_df, x_df, _ = dataset.load('ETTh1', multivariate=True)

# Option 2: Convenience Function
# Loads ETTh1 (Hourly)
y_df, x_df, _ = load_ett('h1', multivariate=True)

print(f"Dataset shape: {y_df.shape}")
print(f"Horizon: {LONG_HORIZON_CONFIGS['ETTh1'].horizon}")
```

### M4 Competition

Standard univariate forecasting benchmark.

```python
from dl_techniques.datasets.time_series import load_m4, get_m4_horizon

# Load Monthly data
# Returns: Y_df, None (no exogenous), S_df (static metadata)
y_df, _, s_df = load_m4('Monthly')
horizon = get_m4_horizon('Monthly')

print(f"Series count: {y_df['unique_id'].nunique()}")
print(f"Forecast Horizon: {horizon}")
```

## 2. Synthetic Data Generator

Generate diverse, realistic time series patterns for pre-training, stress testing, or research. Supports over 80 patterns across financial, weather, industrial, and chaotic domains.

> **Note:** The generator uses its own configuration class defined in `generator.py`.

```python
from dl_techniques.datasets.time_series import TimeSeriesGenerator, TimeSeriesGeneratorConfig

# Configure
config = TimeSeriesGeneratorConfig(
    n_samples=5000,
    random_seed=42,
    default_noise_level=0.1,
    seasonal_periods=[24, 168]
)
gen = TimeSeriesGenerator(config)

# Generate specific patterns
trend = gen.generate_task_data("linear_trend_strong")
crypto = gen.generate_task_data("crypto_prices")
chaos = gen.generate_task_data("mackey_glass")

# Generate custom patterns
custom_ar = gen.generate_custom_pattern(
    "stochastic",
    process_type="arma",
    ar_coeffs=[0.7, -0.2],
    ma_coeffs=[0.4]
)

# Augmentation (Time Warping)
aug_series = gen.augment_series(trend, [{"name": "time_warp", "strength": 0.1, "n_knots": 4}])
```

## 3. Advanced Normalization

The `TimeSeriesNormalizer` provides reversible transformations, crucial for evaluating model performance in the original data scale.

```python
from dl_techniques.datasets.time_series import TimeSeriesNormalizer, NormalizationMethod

# Initialize (Robust scaling handles outliers better than Standard)
scaler = TimeSeriesNormalizer(method=NormalizationMethod.ROBUST)

# Fit and Transform
train_norm = scaler.fit_transform(train_data)
test_norm = scaler.transform(test_data)

# ... Train Model ...
# predictions_norm = model.predict(test_norm)

# Inverse Transform for metrics
# predictions_original = scaler.inverse_transform(predictions_norm)
```

**Available Methods:** `minmax`, `standard`, `robust` (IQR), `max_abs`, `quantile_uniform`, `quantile_normal`, `tanh`, `power`, `decimal`.

## 4. TensorFlow Data Pipelines

Convert DataFrames or Arrays into production-ready `tf.data.Dataset` objects with windowing, batching, and prefetching.

### From DataFrame

```python
from dl_techniques.datasets.time_series import make_tf_dataset, PipelineConfig

# Configuration for batching and shuffling
pipeline_config = PipelineConfig(
    batch_size=32,
    shuffle=True,
    prefetch_buffer_size=-1  # AUTOTUNE
)

# Create Dataset
train_ds = make_tf_dataset(
    df=y_df,
    window_size=96,
    horizon=24,
    target_col='y',
    id_col='unique_id',
    pipeline_config=pipeline_config
)

# Use in Keras
# model.fit(train_ds, epochs=10)
```

### Complete Train/Val/Test Split

```python
from dl_techniques.datasets.time_series import create_train_val_test_datasets, WindowConfig

window_config = WindowConfig(
    window_size=96, 
    horizon=24, 
    target_col='y'
)

train_ds, val_ds, test_ds = create_train_val_test_datasets(
    train_df, val_df, test_df,
    window_config=window_config
)
```

## Configuration Reference

### 1. Pipeline Config
Used for `make_tf_dataset` and pipeline creation.
```python
@dataclass
class PipelineConfig:
    batch_size: int = 32
    shuffle: bool = True
    shuffle_buffer_size: int = 10000
    prefetch_buffer_size: int = -1  # tf.data.AUTOTUNE
    drop_remainder: bool = False
    cache: bool = False
```

### 2. Generator Config
Used specifically for `TimeSeriesGenerator` (import from `generator.py`).
```python
@dataclass
class TimeSeriesConfig:
    n_samples: int = 5000
    random_seed: int = 42
    default_noise_level: float = 0.1
    trend_strengths: Tuple[float, float] = (0.00005, 0.01)
    seasonal_periods: List[int] = [12, 24, 48, 60, 96, 168, 336, 720, 8760]
    # ... plus domain specific params (ar_coeffs, volatility_range, etc.)
```

### 3. Benchmark Config
Used for defining properties of loaded datasets (ETT, M4, etc.).
```python
@dataclass
class TimeSeriesConfig:
    name: str
    freq: str
    seasonality: int
    horizon: int
    n_ts: int
    test_size: int
    val_size: Optional[int]
    # ...
```
