"""
Time Series Data Pipeline Module.

This module provides utilities for converting time series data into
TensorFlow data pipelines suitable for Keras model training, with
support for windowing, batching, and preprocessing.

Refactored to improve memory efficiency using generators and stride views,
avoiding the creation of massive intermediate arrays.

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
from typing import Any, Callable, Dict, List, Optional, Tuple, Generator

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
    Create sliding windows from a numpy array using efficient memory views.

    Generates input-output window pairs for sequence-to-sequence
    forecasting tasks. Uses numpy sliding_window_view to create virtual
    views rather than copying data, significantly reducing memory usage
    before the data is consumed.

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
    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_timesteps, n_features = data.shape

    # Calculate total sequence length needed for one sample (input + target)
    total_len = window_size + horizon

    if n_timesteps < total_len:
        raise ValueError(
            f"Data length ({len(data)}) is too short for "
            f"window_size={window_size}, horizon={horizon}. "
            f"Need at least {total_len} timesteps."
        )

    # Use sliding_window_view for memory efficiency.
    # This creates a view (N, window_len, features) without copying.
    # We create a window of size (window_size + horizon) to capture both X and y.
    try:
        from numpy.lib.stride_tricks import sliding_window_view

        # window_shape specifies the size of the window in each dimension.
        # We want to window over the time dimension (axis 0) with size total_len.
        # We don't want to window over features, so we don't specify axis 1 or set it to full size.
        # However, sliding_window_view appends the window dimensions at the end.

        # Shape: (n_windows, features, total_len) if we window over axis 0
        windows = sliding_window_view(data, window_shape=total_len, axis=0)[::stride]

        # sliding_window_view output shape for axis=0 is (n_windows, n_features, total_len)
        # We need (n_windows, total_len, n_features).
        # Note: The behavior of axis parameter puts window dim at the end.
        # Let's verify shape: (N - total_len + 1, features, total_len)

        # Swap axes to get (n_windows, total_len, n_features)
        windows = windows.transpose(0, 2, 1)

        # Split into inputs and targets
        # Note: These are still views into the original array (mostly)
        x = windows[:, :window_size, :]
        y = windows[:, window_size:, :]

        return x, y

    except (ImportError, AttributeError):
        # Fallback for older numpy versions or if stride_tricks fails
        logger.debug("numpy.lib.stride_tricks.sliding_window_view not available, using manual striding")
        n_samples = (n_timesteps - total_len) // stride + 1

        # Calculate strides
        # data.strides is (bytes_per_step, bytes_per_feature)
        step_bytes = data.strides[0]

        new_shape = (n_samples, total_len, n_features)
        new_strides = (step_bytes * stride, step_bytes, data.strides[1])

        # Create view
        combined = np.lib.stride_tricks.as_strided(
            data, shape=new_shape, strides=new_strides, writeable=False
        )

        x = combined[:, :window_size, :]
        y = combined[:, window_size:, :]

        return x, y


def _data_generator(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    cols_to_use: List[str],
    target_idx: int,
    window_size: int,
    horizon: int,
    stride: int
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Internal generator that yields single window tuples (x, y).

    Used by make_tf_dataset to stream data into tf.data.Dataset
    without loading everything into memory at once.
    """
    # Group by series ID
    grouped = df.groupby(id_col)

    for series_id, group in grouped:
        # Sort by time to ensure temporal order
        if time_col in group.columns:
            # We use values directly for speed, assuming caller handled basic sort if needed,
            # but sorting here is safer for the generator contract.
            series_data = group.sort_values(time_col)[cols_to_use].values.astype(np.float32)
        else:
            series_data = group[cols_to_use].values.astype(np.float32)

        n_timesteps = len(series_data)
        total_len = window_size + horizon

        # Skip if too short
        if n_timesteps < total_len:
            continue

        # Use stride tricks locally to generate windows efficiently
        # We manually iterate indices to yield one by one to keep memory low
        # and compatible with from_generator signature constraints
        for i in range(0, n_timesteps - total_len + 1, stride):
            window_data = series_data[i : i + total_len]
            x_window = window_data[:window_size]
            # Target is just the target column(s) in the horizon
            y_window = window_data[window_size:, target_idx:target_idx + 1]

            yield x_window, y_window


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
    Convert a pandas DataFrame to a tf.data.Dataset using a memory-efficient generator.

    This method handles multiple time series by processing them sequentially
    and streaming windows, avoiding the creation of massive intermediate arrays
    that can cause OOM errors on large datasets.

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

    # Ensure target is included in features (standard for AR models)
    if target_col not in cols_to_use:
        cols_to_use = [target_col] + cols_to_use

    target_idx = cols_to_use.index(target_col)
    n_features = len(cols_to_use)

    # Define output signature for the generator
    output_signature = (
        tf.TensorSpec(shape=(window_size, n_features), dtype=tf.float32),
        tf.TensorSpec(shape=(horizon, 1), dtype=tf.float32)
    )

    # We use a generator to stream data into the dataset.
    # This avoids the "2GB graph limit" of from_tensor_slices and RAM spikes.
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: _data_generator(
            df=df,
            id_col=id_col,
            time_col=time_col,
            cols_to_use=cols_to_use,
            target_idx=target_idx,
            window_size=window_size,
            horizon=horizon,
            stride=1  # Assuming stride 1 for basic usage
        ),
        output_signature=output_signature
    )

    # Apply pipeline configuration
    if pipeline_config.cache:
        # Warning: Caching a massive dataset might still OOM.
        # Use with caution or provide a filename to cache() for disk caching.
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

    logger.info(
        f"Created streamed tf.data.Dataset (window={window_size}, horizon={horizon})"
    )

    return dataset


def make_tf_dataset_from_arrays(
    x_data: np.ndarray,
    y_data: np.ndarray,
    pipeline_config: Optional[PipelineConfig] = None
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from pre-computed arrays.

    Uses `from_tensor_slices` for smaller datasets and `from_generator`
    for large datasets to avoid the TensorFlow Graph 2GB protobuf limit.

    :param x_data: Input array of shape (n_samples, window_size, n_features).
    :type x_data: np.ndarray
    :param y_data: Target array of shape (n_samples, horizon, n_targets).
    :type y_data: np.ndarray
    :param pipeline_config: Pipeline configuration for batching and shuffling.
    :type pipeline_config: Optional[PipelineConfig]
    :return: A tf.data.Dataset yielding (batch_x, batch_y) tuples.
    :rtype: tf.data.Dataset
    """
    if pipeline_config is None:
        pipeline_config = PipelineConfig()

    # Heuristic for using generator vs tensor_slices
    # 2GB limit is approx 500M floats.
    # Safe threshold: 250M elements to be sure.
    is_large_dataset = x_data.size > 250_000_000

    if is_large_dataset:
        logger.info("Large dataset detected, using generator to avoid Graph limits.")

        def gen():
            for i in range(len(x_data)):
                yield x_data[i], y_data[i]

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=x_data.shape[1:], dtype=tf.float32),
                tf.TensorSpec(shape=y_data.shape[1:], dtype=tf.float32)
            )
        )
    else:
        # Efficient for small/medium datasets
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

    Note: This assumes the dataset yields (x, y) tuples where x
    aligns sequentially with time_indices.

    :param dataset: Existing tf.data.Dataset.
    :type dataset: tf.data.Dataset
    :param time_indices: Array of time indices corresponding to each window.
    :type time_indices: np.ndarray
    :param feature_extractors: List of functions that extract features
        from time indices. If None, uses default extractors.
    :type feature_extractors: Optional[List[Callable[[int], float]]]
    :return: Dataset with time features added to inputs.
    :rtype: tf.data.Dataset
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
        # Shape of time_feat is (n_features,)
        # Shape of x is (batch, time, features) if batched, or (time, features) if not

        # NOTE: This function's implementation depends on whether the dataset
        # was already batched or not. Assuming element-wise operation (unbatched)
        # based on from_tensor_slices usage.

        # If unbatched x: (time, feat)
        # We want to append time_feat to every time step
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
        # Handle cases with dictionary inputs or other structures
        if hasattr(element_spec, 'shape'):
             input_shape = element_spec.shape.as_list()
             input_dtype = element_spec.dtype.name
        else:
             input_shape = "complex_structure"
             input_dtype = "mixed"
        output_shape = None
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
