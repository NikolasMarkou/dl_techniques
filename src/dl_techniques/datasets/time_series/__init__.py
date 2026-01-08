"""
Time Series Dataset Module.

This module provides utilities for loading, preprocessing, generating, and
creating data pipelines for time series forecasting.

It includes:
1. Benchmark Datasets (ETT, M4, Favorita, etc.)
2. Synthetic Data Generators (Trend, Seasonal, Chaotic, Financial)
3. Advanced Normalization (Reversible, Robust, Quantile)
4. TensorFlow Data Pipelines

Usage:
    >>> from dl_techniques.datasets.time_series import (
    ...     LongHorizonDataset, make_tf_dataset, TimeSeriesGenerator
    ... )
"""

# Configuration
from .config import (
    DatasetSplits,
    NormalizationConfig,
    PipelineConfig,
    TimeSeriesConfig,
    WindowConfig,
)

# Base & Utilities
from .base import BaseTimeSeriesDataset
from .utils import (
    clean_cache,
    download_file,
    ensure_directory,
    extract_file,
    get_cache_path,
    validate_dataframe_schema,
)

# Data Processing & Pipelines
from .normalizer import (
    NormalizationMethod,
    TimeSeriesNormalizer,
)

from .pipeline import (
    add_time_features,
    create_sliding_windows,
    create_train_val_test_datasets,
    get_dataset_info,
    make_tf_dataset,
    make_tf_dataset_from_arrays,
)

# Synthetic Data Generation
from .generator import (
    TimeSeriesGeneratorConfig,
    TimeSeriesGenerator,
)

# Dataset Loaders
from .long_horizon import (
    LONG_HORIZON_CONFIGS,
    LongHorizonDataset,
    load_ecl,
    load_ett,
    load_weather,
)

from .m4 import (
    M4_CONFIGS,
    M4Dataset,
    get_m4_horizon,
    get_m4_seasonality,
    load_m4,
)

from .favorita import (
    FAVORITA_CONFIG,
    FAVORITA_CONFIGS,
    FavoritaDataset,
    load_favorita,
)

__all__ = [
    # Configuration
    'TimeSeriesConfig',
    'WindowConfig',
    'DatasetSplits',
    'PipelineConfig',
    'NormalizationConfig',

    # Core Components
    'BaseTimeSeriesDataset',
    'TimeSeriesNormalizer',
    'NormalizationMethod',
    'TimeSeriesGenerator',
    'TimeSeriesGeneratorConfig',

    # Utilities
    'download_file',
    'extract_file',
    'ensure_directory',
    'get_cache_path',
    'clean_cache',
    'validate_dataframe_schema',

    # Pipeline
    'make_tf_dataset',
    'make_tf_dataset_from_arrays',
    'create_sliding_windows',
    'create_train_val_test_datasets',
    'add_time_features',
    'get_dataset_info',

    # Long Horizon Benchmarks
    'LongHorizonDataset',
    'LONG_HORIZON_CONFIGS',
    'load_ett',
    'load_weather',
    'load_ecl',

    # M4 Benchmarks
    'M4Dataset',
    'M4_CONFIGS',
    'load_m4',
    'get_m4_horizon',
    'get_m4_seasonality',

    # Favorita
    'FavoritaDataset',
    'FAVORITA_CONFIG',
    'FAVORITA_CONFIGS',
    'load_favorita',
]
