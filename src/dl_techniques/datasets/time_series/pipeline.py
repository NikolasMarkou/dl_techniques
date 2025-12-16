"""
Time Series Data Pipeline Module.

This module provides utilities for converting time series data into
TensorFlow data pipelines suitable for Keras model training, with
support for windowing, batching, and preprocessing.

Example:
    >>> from dl_techniques.datasets.time_series.pipeline import (
    ...     make_tf_dataset, create_sliding_windows
    ... )
    >>> dataset = make_tf_dataset(df, window_size=96, horizon=24)
    >>> for batch_x, batch_y in dataset.take(1):
    ...     print(batch_x.shape, batch_y.shape)
"""

import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .config import PipelineConfig, WindowConfig

# ---------------------------------------------------------------------

def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    horizon: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from a numpy array.

    Generates input-output window pairs for sequence-to-sequence
    forecasting tasks using an efficient NumPy implementation.

    :param data: Input array of shape (timesteps, features).
    :type data: np.ndarray
    :param window_size: Length of input sequence (lookback period).
    :type window_size: int
    :param horizon: Length of target sequence (forecast horizon).
    :type horizon: int
    :param stride: Step size between consecutive windows.
    :type stride: int
    :return: Tuple of (inputs, targets) arrays with shapes:
        - inputs: (n_windows, window_size, features)
        - targets: (n_windows, horizon, features)
    :rtype: Tuple[np.ndarray, np.ndarray]
    :raises ValueError: If data is too short for the given parameters.

    Example:
        >>> data = np.random.randn(1000, 7)
        >>> x, y = create_sliding_windows(data, window_size=96, horizon=24)
        >>> print(x.shape, y.shape)
        (881, 96, 7) (881, 24, 7)
    """
    n_samples = (len(data) - window_size - horizon) // stride + 1

    if n_samples <= 0:
        raise ValueError(
            f"Data length ({len(data)}) is too short for "
            f"window_size={window_size}, horizon={horizon}, stride={stride}. "
            f"Need at least {window_size + horizon} timesteps."
        )

    # Pre-allocate arrays for efficiency
    n_features = data.shape[1] if data.ndim > 1 else 1
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    x = np.zeros((n_samples, window_size, n_features), dtype=data.dtype)
    y = np.zeros((n_samples, horizon, n_features), dtype=data.dtype)

    for idx, i in enumerate(range(0, n_samples * stride, stride)):
        x[idx] = data[i:i + window_size]
        y[idx] = data[i + window_size:i + window_size + horizon]

    return x, y


def make_tf_dataset(
    df: pd.DataFrame,
    window_size: int,
    horizon: int,
    batch_size: int = 32,
    shuffle: bool = True,
    target_col: str = 'y',
    id_col: str = 'unique_id',
    time_col: str = 'ds',
    feature_cols: Optional[List[str]] = None,
    pipeline_config: Optional[PipelineConfig] = None
) -> tf.data.Dataset:
    """
    Convert a pandas DataFrame to a tf.data.Dataset for time series forecasting.

    Creates sliding windows from multiple time series, handling each series
    independently to prevent cross-series contamination. Supports various
    batching and shuffling configurations.

    :param df: DataFrame with columns [id_col, time_col, target_col, ...].
    :type df: pd.DataFrame
    :param window_size: Length of input sequence (lookback period).
    :type window_size: int
    :param horizon: Length of target sequence (forecast horizon).
    :type horizon: int
    :param batch_size: Number of samples per batch.
    :type batch_size: int
    :param shuffle: Whether to shuffle the dataset.
    :type shuffle: bool
    :param target_col: Name of the target column.
    :type target_col: str
    :param id_col: Name of the series identifier column.
    :type id_col: str
    :param time_col: Name of the timestamp column.
    :type time_col: str
    :param feature_cols: List of feature columns to include.
        If None, uses all numeric columns except id_col and time_col.
    :type feature_cols: Optional[List[str]]
    :param pipeline_config: Optional pipeline configuration for advanced settings.
    :type pipeline_config: Optional[PipelineConfig]
    :return: A tf.data.Dataset yielding (batch_x, batch_y) tuples.
    :rtype: tf.data.Dataset
    :raises ValueError: If the DataFrame is empty or missing required columns.

    Example:
        >>> dataset = make_tf_dataset(
        ...     df,
        ...     window_size=96,
        ...     horizon=24,
        ...     batch_size=32,
        ...     target_col='OT'
        ... )
        >>> for x, y in dataset.take(1):
        ...     print(f"Input: {x.shape}, Target: {y.shape}")
    """
    if pipeline_config is None:
        pipeline_config = PipelineConfig(
            batch_size=batch_size,
            shuffle=shuffle
        )

    # Validate required columns
    required_cols = [id_col, target_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    # Determine feature columns
    exclude_cols = {id_col, time_col}
    if feature_cols is not None:
        cols_to_use = feature_cols
    else:
        # Use all numeric columns except id and time
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_use = [c for c in numeric_cols if c not in exclude_cols]

    # Ensure target is included in features
    if target_col not in cols_to_use:
        cols_to_use = [target_col] + cols_to_use

    target_idx = cols_to_use.index(target_col)

    all_x = []
    all_y = []

    # Process each series independently
    for series_id, group in df.groupby(id_col):
        # Sort by time
        if time_col in group.columns:
            group = group.sort_values(time_col)

        # Extract feature values
        series_data = group[cols_to_use].values.astype(np.float32)

        # Check if series is long enough
        min_length = window_size + horizon
        if len(series_data) < min_length:
            logger.warning(
                f"Series '{series_id}' has only {len(series_data)} timesteps, "
                f"need at least {min_length}. Skipping."
            )
            continue

        # Create sliding windows
        try:
            x_windows, y_windows = create_sliding_windows(
                series_data, window_size, horizon
            )
        except ValueError as e:
            logger.warning(f"Could not create windows for series '{series_id}': {e}")
            continue

        # Extract only target column for y
        y_windows = y_windows[:, :, target_idx:target_idx + 1]

        all_x.append(x_windows)
        all_y.append(y_windows)

    if not all_x:
        raise ValueError(
            "No valid windows could be generated from the DataFrame. "
            "Check that series are long enough for the given window_size and horizon."
        )

    # Concatenate all windows
    x_data = np.concatenate(all_x, axis=0)
    y_data = np.concatenate(all_y, axis=0)

    logger.info(
        f"Created dataset with {len(x_data)} windows, "
        f"input shape: {x_data.shape[1:]}, target shape: {y_data.shape[1:]}"
    )

    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

    # Apply pipeline configuration
    if pipeline_config.cache:
        dataset = dataset.cache()

    if pipeline_config.shuffle:
        seed = pipeline_config.seed
        dataset = dataset.shuffle(
            buffer_size=pipeline_config.shuffle_buffer_size,
            seed=seed,
            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(
        pipeline_config.batch_size,
        drop_remainder=pipeline_config.drop_remainder
    )

    # Configure prefetching
    prefetch = pipeline_config.prefetch_buffer_size
    if prefetch == -1:
        prefetch = tf.data.AUTOTUNE
    dataset = dataset.prefetch(prefetch)

    return dataset


def make_tf_dataset_from_arrays(
    x_data: np.ndarray,
    y_data: np.ndarray,
    pipeline_config: Optional[PipelineConfig] = None
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from pre-computed arrays.

    Convenient wrapper for creating datasets from arrays already
    processed by other utilities.

    :param x_data: Input array of shape (n_samples, window_size, n_features).
    :type x_data: np.ndarray
    :param y_data: Target array of shape (n_samples, horizon, n_targets).
    :type y_data: np.ndarray
    :param pipeline_config: Pipeline configuration for batching and shuffling.
    :type pipeline_config: Optional[PipelineConfig]
    :return: A tf.data.Dataset yielding (batch_x, batch_y) tuples.
    :rtype: tf.data.Dataset

    Example:
        >>> x = np.random.randn(1000, 96, 7).astype(np.float32)
        >>> y = np.random.randn(1000, 24, 1).astype(np.float32)
        >>> dataset = make_tf_dataset_from_arrays(x, y)
    """
    if pipeline_config is None:
        pipeline_config = PipelineConfig()

    dataset = tf.data.Dataset.from_tensor_slices((
        x_data.astype(np.float32),
        y_data.astype(np.float32)
    ))

    if pipeline_config.cache:
        dataset = dataset.cache()

    if pipeline_config.shuffle:
        dataset = dataset.shuffle(
            buffer_size=pipeline_config.shuffle_buffer_size,
            seed=pipeline_config.seed,
            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(
        pipeline_config.batch_size,
        drop_remainder=pipeline_config.drop_remainder
    )

    prefetch = pipeline_config.prefetch_buffer_size
    if prefetch == -1:
        prefetch = tf.data.AUTOTUNE
    dataset = dataset.prefetch(prefetch)

    return dataset


def create_train_val_test_datasets(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
    window_config: WindowConfig,
    pipeline_config: Optional[PipelineConfig] = None
) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset], tf.data.Dataset]:
    """
    Create train, validation, and test tf.data.Datasets from DataFrames.

    Convenience function for creating all three datasets with consistent
    configuration. Training dataset is shuffled by default, while
    validation and test datasets are not.

    :param train_df: Training DataFrame.
    :type train_df: pd.DataFrame
    :param val_df: Validation DataFrame (or None).
    :type val_df: Optional[pd.DataFrame]
    :param test_df: Test DataFrame.
    :type test_df: pd.DataFrame
    :param window_config: Window generation configuration.
    :type window_config: WindowConfig
    :param pipeline_config: Pipeline configuration (optional).
    :type pipeline_config: Optional[PipelineConfig]
    :return: Tuple of (train_dataset, val_dataset, test_dataset).
    :rtype: Tuple[tf.data.Dataset, Optional[tf.data.Dataset], tf.data.Dataset]

    Example:
        >>> train_ds, val_ds, test_ds = create_train_val_test_datasets(
        ...     train_df, val_df, test_df,
        ...     window_config=WindowConfig(window_size=96, horizon=24)
        ... )
    """
    if pipeline_config is None:
        pipeline_config = PipelineConfig()

    # Create training dataset (shuffled)
    train_ds = make_tf_dataset(
        train_df,
        window_size=window_config.window_size,
        horizon=window_config.horizon,
        target_col=window_config.target_col,
        id_col=window_config.id_col,
        time_col=window_config.time_col,
        feature_cols=window_config.feature_cols,
        pipeline_config=pipeline_config
    )

    # Create validation dataset (not shuffled)
    val_ds = None
    if val_df is not None and len(val_df) > 0:
        val_config = PipelineConfig(
            batch_size=pipeline_config.batch_size,
            shuffle=False,
            prefetch_buffer_size=pipeline_config.prefetch_buffer_size
        )
        val_ds = make_tf_dataset(
            val_df,
            window_size=window_config.window_size,
            horizon=window_config.horizon,
            target_col=window_config.target_col,
            id_col=window_config.id_col,
            time_col=window_config.time_col,
            feature_cols=window_config.feature_cols,
            pipeline_config=val_config
        )

    # Create test dataset (not shuffled)
    test_config = PipelineConfig(
        batch_size=pipeline_config.batch_size,
        shuffle=False,
        prefetch_buffer_size=pipeline_config.prefetch_buffer_size
    )
    test_ds = make_tf_dataset(
        test_df,
        window_size=window_config.window_size,
        horizon=window_config.horizon,
        target_col=window_config.target_col,
        id_col=window_config.id_col,
        time_col=window_config.time_col,
        feature_cols=window_config.feature_cols,
        pipeline_config=test_config
    )

    return train_ds, val_ds, test_ds


def add_time_features(
    dataset: tf.data.Dataset,
    time_indices: np.ndarray,
    feature_extractors: Optional[List[Callable[[int], float]]] = None
) -> tf.data.Dataset:
    """
    Add time-based features to an existing dataset.

    Extracts temporal features (e.g., hour of day, day of week) and
    concatenates them to the input features.

    :param dataset: Existing tf.data.Dataset.
    :type dataset: tf.data.Dataset
    :param time_indices: Array of time indices corresponding to each window.
    :type time_indices: np.ndarray
    :param feature_extractors: List of functions that extract features
        from time indices. If None, uses default extractors.
    :type feature_extractors: Optional[List[Callable[[int], float]]]
    :return: Dataset with time features added to inputs.
    :rtype: tf.data.Dataset

    Example:
        >>> dataset = add_time_features(dataset, time_indices)
    """
    if feature_extractors is None:
        # Default: hour of day (assuming hourly data starting at index 0)
        feature_extractors = [
            lambda t: np.sin(2 * np.pi * (t % 24) / 24),
            lambda t: np.cos(2 * np.pi * (t % 24) / 24),
            lambda t: np.sin(2 * np.pi * (t % 168) / 168),  # Week
            lambda t: np.cos(2 * np.pi * (t % 168) / 168),
        ]

    # Extract time features
    time_features = np.column_stack([
        np.array([f(t) for t in time_indices])
        for f in feature_extractors
    ]).astype(np.float32)

    # Create a dataset of time features
    time_ds = tf.data.Dataset.from_tensor_slices(time_features)

    # Zip and map to concatenate
    def add_features(inputs_targets, time_feat):
        x, y = inputs_targets
        # Broadcast time features across the sequence dimension
        time_feat_expanded = tf.tile(
            tf.expand_dims(time_feat, 0),
            [tf.shape(x)[0], 1]
        )
        x_with_time = tf.concat([x, time_feat_expanded], axis=-1)
        return x_with_time, y

    return tf.data.Dataset.zip((dataset, time_ds)).map(
        add_features,
        num_parallel_calls=tf.data.AUTOTUNE
    )


def get_dataset_info(dataset: tf.data.Dataset) -> Dict[str, Any]:
    """
    Get information about a tf.data.Dataset.

    :param dataset: A tf.data.Dataset to inspect.
    :type dataset: tf.data.Dataset
    :return: Dictionary with dataset information.
    :rtype: Dict[str, Any]

    Example:
        >>> info = get_dataset_info(dataset)
        >>> print(info['input_shape'], info['output_shape'])
    """
    # Get element spec
    element_spec = dataset.element_spec

    if isinstance(element_spec, tuple) and len(element_spec) == 2:
        input_spec, output_spec = element_spec
        input_shape = input_spec.shape.as_list()
        output_shape = output_spec.shape.as_list()
        input_dtype = input_spec.dtype.name
        output_dtype = output_spec.dtype.name
    else:
        input_shape = element_spec.shape.as_list()
        output_shape = None
        input_dtype = element_spec.dtype.name
        output_dtype = None

    # Estimate cardinality
    cardinality = dataset.cardinality().numpy()
    if cardinality == tf.data.INFINITE_CARDINALITY:
        cardinality_str = 'infinite'
    elif cardinality == tf.data.UNKNOWN_CARDINALITY:
        cardinality_str = 'unknown'
    else:
        cardinality_str = int(cardinality)

    return {
        'input_shape': input_shape,
        'output_shape': output_shape,
        'input_dtype': input_dtype,
        'output_dtype': output_dtype,
        'cardinality': cardinality_str,
    }
