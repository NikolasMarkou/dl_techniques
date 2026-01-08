# Time Series Dataset Module

## Overview

The `dl_techniques.datasets.time_series` module is a production-grade toolkit designed to streamline the data engineering lifecycle for time series deep learning. It bridges the gap between raw data sources and TensorFlow model training.

This module provides end-to-end support for:
1.  **Benchmarking:** Automatic downloading and standardizing of academic datasets (ETT, M4, Favorita).
2.  **Synthetic Generation:** Creating complex, domain-specific synthetic data for pre-training and stress testing.
3.  **Preprocessing:** Robust, reversible normalization strategies designed to prevent data leakage.
4.  **Pipelines:** Efficient, memory-safe TensorFlow `tf.data` pipelines supporting windowing and batching.

## Directory Structure

```text
dl_techniques/datasets/time_series/
├── base.py           # Abstract base classes and splitting logic
├── config.py         # Configuration dataclasses
├── generator.py      # Synthetic data generation engine
├── normalizer.py     # Reversible normalization logic
├── pipeline.py       # TensorFlow data pipeline utilities
├── utils.py          # Download and caching helpers
├── long_horizon.py   # Loaders for ETT, ECL, Weather, Traffic
├── m4.py             # Loaders for M4 Competition data
└── favorita.py       # Loaders for Favorita Grocery Sales
```

## Data Format Standard

To ensure compatibility across the module, all DataFrames follow a strict "long format" convention similar to the Nixtla ecosystem:

*   **`unique_id`**: Identifier for the specific time series (string or int).
*   **`ds`**: The timestamp column (datetime or int).
*   **`y`**: The target variable to forecast (numeric).

## 1. Benchmark Datasets

The module includes loaders for standard datasets used in forecasting literature (Informer, Autoformer, N-BEATS).

### Long Horizon Datasets (ETT, Weather, Traffic)

These datasets are widely used for evaluating long-sequence time-series forecasting (LSTF).

```python
from dl_techniques.datasets.time_series import LongHorizonDataset, load_ett

# Option 1: Convenience function for ETT (Electricity Transformer Temperature)
# Returns:
#   y_df: Target data (unique_id, ds, y)
#   x_df: Exogenous temporal features
#   s_df: Static features (None for ETT)
y_df, x_df, _ = load_ett(variant='h1', multivariate=True)

# Option 2: Generic loader for other datasets (e.g., Weather, Traffic, ECL)
dataset = LongHorizonDataset(root_dir='./data')
y_df, x_df, _ = dataset.load('Weather', multivariate=True)
```

### M4 Competition

Load data from the M4 forecasting competition, categorized by frequency.

```python
from dl_techniques.datasets.time_series import load_m4, get_m4_horizon

# Load Monthly data
# Returns: y_df (targets), None (no temporal features), s_df (static categories)
y_df, _, s_df = load_m4('Monthly')

# Get standard horizon for this frequency
horizon = get_m4_horizon('Monthly')
print(f"Standard Horizon: {horizon}")
```

### Favorita Grocery Sales

Includes optimization for handling large datasets via chunked loading.

```python
from dl_techniques.datasets.time_series import load_favorita

# Load a subset (top 200 items) to save memory
y_df, x_df, s_df = load_favorita(group='Favorita200')
```

## 2. Synthetic Data Generation

The `TimeSeriesGenerator` creates realistic patterns across multiple domains (Finance, IoT, Weather, Chaos Theory). It supports augmentation and parameter randomization.

```python
from dl_techniques.datasets.time_series import (
    TimeSeriesGenerator, 
    TimeSeriesGeneratorConfig
)

# 1. Configure the generator
config = TimeSeriesGeneratorConfig(
    n_samples=5000,
    random_seed=42,
    default_noise_level=0.1
)
gen = TimeSeriesGenerator(config)

# 2. Generate specific domain patterns
trend_data = gen.generate_task_data("linear_trend_strong")
crypto_data = gen.generate_task_data("crypto_prices")
chaos_data = gen.generate_task_data("mackey_glass")

# 3. Create custom patterns
custom_arma = gen.generate_custom_pattern(
    "stochastic",
    process_type="arma",
    ar_coeffs=[0.7, -0.2],
    ma_coeffs=[0.4]
)
```

## 3. Normalization

The `TimeSeriesNormalizer` handles scaling while strictly preventing data leakage. It supports reversible transformations to evaluate model performance on the original scale.

```python
from dl_techniques.datasets.time_series import TimeSeriesNormalizer, NormalizationMethod

# Initialize normalizer (Robust method handles outliers better than Standard)
scaler = TimeSeriesNormalizer(method=NormalizationMethod.ROBUST)

# Fit ONLY on training data
scaler.fit(train_array)

# Transform all splits
train_norm = scaler.transform(train_array)
test_norm = scaler.transform(test_array)

# Inverse transform predictions for evaluation
predictions_original = scaler.inverse_transform(predictions_norm)
```

**Available Methods:** `minmax`, `standard`, `robust`, `max_abs`, `quantile_uniform`, `quantile_normal`, `tanh`, `power`.

## 4. TensorFlow Data Pipelines

Convert Pandas DataFrames or Numpy arrays into production-ready `tf.data.Dataset` objects. These pipelines handle windowing, batching, shuffling, and prefetching.

### Standard Pipeline

```python
from dl_techniques.datasets.time_series import make_tf_dataset, PipelineConfig

# Define pipeline behavior
pipeline_config = PipelineConfig(
    batch_size=32,
    shuffle=True,
    prefetch_buffer_size=-1  # AUTOTUNE
)

# Create tf.data.Dataset
train_ds = make_tf_dataset(
    df=y_df,
    window_size=96,
    horizon=24,
    target_col='y',
    id_col='unique_id',
    pipeline_config=pipeline_config
)

# Iterate
for x, y in train_ds.take(1):
    print(x.shape, y.shape)
```

### Full Train/Val/Test Split

This utility splits the DataFrame temporally (respecting time order) and creates datasets for all three splits.

```python
from dl_techniques.datasets.time_series import create_train_val_test_datasets, WindowConfig

# Define windowing parameters
window_config = WindowConfig(
    window_size=96,
    horizon=24,
    target_col='y'
)

# Generate datasets
train_ds, val_ds, test_ds = create_train_val_test_datasets(
    train_df, val_df, test_df,
    window_config=window_config
)
```

## Configuration Reference

The module relies on typed dataclasses for configuration management.

### PipelineConfig
Controls the `tf.data` behavior.
*   `batch_size` (int): Batch size.
*   `shuffle` (bool): Whether to shuffle training data.
*   `prefetch_buffer_size` (int): Number of batches to prefetch.
*   `cache` (bool): Whether to cache dataset in memory.

### WindowConfig
Controls how sliding windows are generated.
*   `window_size` (int): Lookback period.
*   `horizon` (int): Forecast period.
*   `stride` (int): Step size between windows.
*   `target_col` (str): Name of the target column.
*   `feature_cols` (List[str]): Specific features to include (default: all numeric).

### TimeSeriesConfig
Metadata for benchmark datasets.
*   `name` (str): Dataset name.
*   `freq` (str): Pandas frequency string (e.g., 'H', 'D').
*   `seasonality` (int): Primary seasonal period.
*   `horizon` (int): Default forecast horizon.