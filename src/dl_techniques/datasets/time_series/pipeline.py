"""
Time Series Data Pipeline Module.

This module provides utilities for converting time series data into
TensorFlow data pipelines suitable for Keras model training, with
support for windowing, batching, and preprocessing.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Any, Callable, Dict, List, Optional, Tuple, Generator

# ---------------------------------------------------------------------
# Local Imports
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

    Generates input-output window pairs for sequence-to-sequence tasks.

    :param data: Input array of shape (timesteps, features).
    :type data: np.ndarray
    :param window_size: Length of input sequence.
    :type window_size: int
    :param horizon: Length of forecast horizon.
    :type horizon: int
    :param stride: Step size between windows.
    :type stride: int
    :return: Tuple of (x, y) arrays.
    :rtype: Tuple[np.ndarray, np.ndarray]
    :raises ValueError: If data is too short.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_timesteps, n_features = data.shape
    total_len = window_size + horizon

    if n_timesteps < total_len:
        raise ValueError(
            f"Data length ({len(data)}) is too short for "
            f"window_size={window_size}, horizon={horizon}. "
            f"Need at least {total_len} timesteps."
        )

    try:
        from numpy.lib.stride_tricks import sliding_window_view
        # Shape: (n_windows, features, total_len) if windowing over axis 0
        # sliding_window_view behavior depends on numpy version, usually puts window axis last
        windows = sliding_window_view(data, window_shape=total_len, axis=0)[::stride]

        # Standardize shape to (n_windows, total_len, n_features)
        # Current: (n_windows, n_features, total_len)
        windows = windows.transpose(0, 2, 1)

        x = windows[:, :window_size, :]
        y = windows[:, window_size:, :]
        return x, y

    except (ImportError, AttributeError):
        logger.debug("numpy.lib.stride_tricks.sliding_window_view not available, using manual striding")
        n_samples = (n_timesteps - total_len) // stride + 1
        step_bytes = data.strides[0]
        new_shape = (n_samples, total_len, n_features)
        new_strides = (step_bytes * stride, step_bytes, data.strides[1])

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
    """Internal generator that yields single window tuples (x, y)."""
    grouped = df.groupby(id_col)

    for series_id, group in grouped:
        if time_col in group.columns:
            series_data = group.sort_values(time_col)[cols_to_use].values.astype(np.float32)
        else:
            series_data = group[cols_to_use].values.astype(np.float32)

        n_timesteps = len(series_data)
        total_len = window_size + horizon

        if n_timesteps < total_len:
            continue

        for i in range(0, n_timesteps - total_len + 1, stride):
            window_data = series_data[i : i + total_len]
            x_window = window_data[:window_size]
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

    :param df: Input DataFrame.
    :type df: pd.DataFrame
    :param window_size: Input window size.
    :type window_size: int
    :param horizon: Forecast horizon.
    :type horizon: int
    :param batch_size: Batch size.
    :type batch_size: int
    :return: TensorFlow dataset yielding (x, y).
    :rtype: tf.data.Dataset
    """
    if pipeline_config is None:
        pipeline_config = PipelineConfig(batch_size=batch_size, shuffle=shuffle)

    required_cols = [id_col, target_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    exclude_cols = {id_col, time_col}
    if feature_cols is not None:
        cols_to_use = feature_cols
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_use = [c for c in numeric_cols if c not in exclude_cols]

    if target_col not in cols_to_use:
        cols_to_use = [target_col] + cols_to_use

    target_idx = cols_to_use.index(target_col)
    n_features = len(cols_to_use)

    output_signature = (
        tf.TensorSpec(shape=(window_size, n_features), dtype=tf.float32),
        tf.TensorSpec(shape=(horizon, 1), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        generator=lambda: _data_generator(
            df=df,
            id_col=id_col,
            time_col=time_col,
            cols_to_use=cols_to_use,
            target_idx=target_idx,
            window_size=window_size,
            horizon=horizon,
            stride=1
        ),
        output_signature=output_signature
    )

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

    logger.info(f"Created streamed tf.data.Dataset (window={window_size}, horizon={horizon})")
    return dataset


def make_tf_dataset_from_arrays(
    x_data: np.ndarray,
    y_data: np.ndarray,
    pipeline_config: Optional[PipelineConfig] = None
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from pre-computed arrays.

    Uses generator if data size > 250M elements to avoid protobuf limits.
    """
    if pipeline_config is None:
        pipeline_config = PipelineConfig()

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
    """Create train, validation, and test tf.data.Datasets from DataFrames."""
    if pipeline_config is None:
        pipeline_config = PipelineConfig()

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

    Robustly handles both unbatched (time, feat) and batched (batch, time, feat) inputs.

    :param dataset: Input tf.data.Dataset.
    :type dataset: tf.data.Dataset
    :param time_indices: Array of time indices aligning with dataset.
    :type time_indices: np.ndarray
    :return: Dataset with concatenated time features.
    :rtype: tf.data.Dataset
    """
    if feature_extractors is None:
        feature_extractors = [
            lambda t: np.sin(2 * np.pi * (t % 24) / 24),
            lambda t: np.cos(2 * np.pi * (t % 24) / 24),
            lambda t: np.sin(2 * np.pi * (t % 168) / 168),
            lambda t: np.cos(2 * np.pi * (t % 168) / 168),
        ]

    time_features = np.column_stack([
        np.array([f(t) for t in time_indices])
        for f in feature_extractors
    ]).astype(np.float32)

    time_ds = tf.data.Dataset.from_tensor_slices(time_features)

    def add_features(inputs_targets, time_feat):
        x, y = inputs_targets

        # Determine rank to handle batching correctly
        # rank 2: (time, features) -> unbatched
        # rank 3: (batch, time, features) -> batched
        rank = tf.rank(x)

        if rank == 2:
            # Unbatched case: append time_feat to every timestep
            # Expand time_feat to (1, n_time_feats) then tile to (window, n_time_feats)
            time_feat_expanded = tf.tile(
                tf.expand_dims(time_feat, 0),
                [tf.shape(x)[0], 1]
            )
            x_with_time = tf.concat([x, time_feat_expanded], axis=-1)

        else:
            # Batched case (rank 3 or higher)
            # Expand to (1, 1, n_time_feats) then tile to (batch, window, n_time_feats)
            time_feat_expanded = tf.tile(
                tf.reshape(time_feat, [1, 1, -1]),
                [tf.shape(x)[0], tf.shape(x)[1], 1]
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

    :param dataset: Input dataset.
    :return: Dictionary containing shape, dtype, and cardinality.
    """
    element_spec = dataset.element_spec

    if isinstance(element_spec, tuple) and len(element_spec) == 2:
        input_spec, output_spec = element_spec
        input_shape = input_spec.shape.as_list()
        output_shape = output_spec.shape.as_list()
        input_dtype = input_spec.dtype.name
        output_dtype = output_spec.dtype.name
    else:
        if hasattr(element_spec, 'shape'):
             input_shape = element_spec.shape.as_list()
             input_dtype = element_spec.dtype.name
        else:
             input_shape = "complex_structure"
             input_dtype = "mixed"
        output_shape = None
        output_dtype = None

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