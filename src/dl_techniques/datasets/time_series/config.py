"""
Time Series Dataset Configuration Module.

This module provides configuration dataclasses for time series datasets,
including dataset metadata, split information, and deep learning parameters.

Example:
    >>> from dl_techniques.datasets.time_series.config import (
    ...     TimeSeriesConfig, DatasetSplits, WindowConfig
    ... )
    >>> config = TimeSeriesConfig(
    ...     name='ETTh1',
    ...     freq='H',
    ...     seasonality=24,
    ...     horizon=96
    ... )
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Any, Literal

# ---------------------------------------------------------------------

@dataclass
class TimeSeriesConfig:
    """
    Configuration for Time Series Datasets.

    This dataclass encapsulates metadata and parameters for time series
    forecasting datasets, including frequency, seasonality, and horizon.

    :param name: Name of the dataset (e.g., 'ETTh1', 'Yearly').
    :type name: str
    :param freq: Pandas frequency string (e.g., 'H', 'D', 'M').
    :type freq: str
    :param seasonality: Primary seasonal period in time steps.
    :type seasonality: int
    :param horizon: Default forecasting horizon.
    :type horizon: int
    :param n_ts: Number of time series in the dataset.
    :type n_ts: int
    :param test_size: Number of time steps reserved for testing.
    :type test_size: int
    :param val_size: Number of time steps reserved for validation.
        Defaults to None (no validation set).
    :type val_size: Optional[int]
    :param url: Source URL for downloading the dataset.
    :type url: Optional[str]
    :param window_size: Default lookback window size for DL models.
    :type window_size: int
    :param stride: Default stride for sliding window generation.
    :type stride: int
    :param n_features: Number of features per time step.
    :type n_features: int

    Example:
        >>> config = TimeSeriesConfig(
        ...     name='ETTh1',
        ...     freq='H',
        ...     seasonality=24,
        ...     horizon=96,
        ...     n_ts=1,
        ...     test_size=11520
        ... )
    """

    name: str
    freq: str
    seasonality: int
    horizon: int
    n_ts: int
    test_size: int
    val_size: Optional[int] = None
    url: Optional[str] = None
    window_size: int = 96
    stride: int = 1
    n_features: int = 1


@dataclass
class WindowConfig:
    """
    Configuration for sliding window generation.

    This dataclass specifies parameters for creating input-output windows
    from time series data for sequence-to-sequence forecasting.

    :param window_size: Length of input sequence (lookback period).
    :type window_size: int
    :param horizon: Length of target sequence (forecast period).
    :type horizon: int
    :param stride: Step size between consecutive windows.
    :type stride: int
    :param target_col: Name of the target column in the DataFrame.
    :type target_col: str
    :param id_col: Name of the column identifying individual time series.
    :type id_col: str
    :param time_col: Name of the timestamp column.
    :type time_col: str
    :param feature_cols: List of feature column names to include.
        If None, all columns except id_col and time_col are used.
    :type feature_cols: Optional[List[str]]
    :param include_target_in_features: Whether to include target column in input features.
    :type include_target_in_features: bool

    Example:
        >>> window_config = WindowConfig(
        ...     window_size=96,
        ...     horizon=24,
        ...     stride=1,
        ...     target_col='OT'
        ... )
    """

    window_size: int = 96
    horizon: int = 24
    stride: int = 1
    target_col: str = 'y'
    id_col: str = 'unique_id'
    time_col: str = 'ds'
    feature_cols: Optional[List[str]] = None
    include_target_in_features: bool = True


@dataclass
class DatasetSplits:
    """
    Container for train/validation/test dataset splits.

    This dataclass holds the split data arrays or DataFrames along with
    optional normalization statistics computed from the training set.

    :param train: Training data (array or DataFrame).
    :type train: Any
    :param val: Validation data (array or DataFrame). Defaults to None.
    :type val: Optional[Any]
    :param test: Test data (array or DataFrame). Defaults to None.
    :type test: Optional[Any]
    :param train_mean: Mean of training data for denormalization.
    :type train_mean: Optional[np.ndarray]
    :param train_std: Standard deviation of training data for denormalization.
    :type train_std: Optional[np.ndarray]

    Example:
        >>> splits = DatasetSplits(
        ...     train=train_df,
        ...     val=val_df,
        ...     test=test_df,
        ...     train_mean=np.array([0.0]),
        ...     train_std=np.array([1.0])
        ... )
    """

    train: Any
    val: Optional[Any] = None
    test: Optional[Any] = None
    train_mean: Optional[np.ndarray] = None
    train_std: Optional[np.ndarray] = None


@dataclass
class PipelineConfig:
    """
    Configuration for tf.data pipeline construction.

    This dataclass specifies batch sizes, shuffling, and prefetching
    parameters for creating efficient TensorFlow data pipelines.

    :param batch_size: Number of samples per batch.
    :type batch_size: int
    :param shuffle: Whether to shuffle the dataset.
    :type shuffle: bool
    :param shuffle_buffer_size: Size of the shuffle buffer.
    :type shuffle_buffer_size: int
    :param prefetch_buffer_size: Number of batches to prefetch.
        Use -1 for tf.data.AUTOTUNE.
    :type prefetch_buffer_size: int
    :param drop_remainder: Whether to drop the last incomplete batch.
    :type drop_remainder: bool
    :param cache: Whether to cache the dataset in memory.
    :type cache: bool
    :param seed: Random seed for reproducible shuffling.
    :type seed: Optional[int]

    Example:
        >>> pipeline_config = PipelineConfig(
        ...     batch_size=32,
        ...     shuffle=True,
        ...     shuffle_buffer_size=10000
        ... )
    """

    batch_size: int = 32
    shuffle: bool = True
    shuffle_buffer_size: int = 10000
    prefetch_buffer_size: int = -1  # -1 for AUTOTUNE
    drop_remainder: bool = False
    cache: bool = False
    seed: Optional[int] = None


@dataclass
class NormalizationConfig:
    """
    Configuration for time series normalization.

    This dataclass specifies the normalization strategy and parameters
    for preprocessing time series data before model training.

    :param method: Normalization method to apply.
        Options: 'standard', 'minmax', 'robust', 'revin', 'none'.
    :type method: Literal['standard', 'minmax', 'robust', 'revin', 'none']
    :param per_series: Whether to normalize each series independently.
    :type per_series: bool
    :param per_feature: Whether to normalize each feature independently.
    :type per_feature: bool
    :param epsilon: Small constant for numerical stability.
    :type epsilon: float
    :param clip_value: Maximum absolute value after normalization.
        None for no clipping.
    :type clip_value: Optional[float]

    Example:
        >>> norm_config = NormalizationConfig(
        ...     method='standard',
        ...     per_series=True,
        ...     per_feature=True
        ... )
    """

    method: Literal['standard', 'minmax', 'robust', 'revin', 'none'] = 'standard'
    per_series: bool = False
    per_feature: bool = True
    epsilon: float = 1e-8
    clip_value: Optional[float] = None
