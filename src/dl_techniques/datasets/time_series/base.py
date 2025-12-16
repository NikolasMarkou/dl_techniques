"""
Time Series Dataset Base Classes.

This module provides abstract base classes for time series dataset loaders,
defining a consistent interface for downloading, loading, and splitting data.

Example:
    >>> from dl_techniques.datasets.time_series.base import BaseTimeSeriesDataset
    >>> class MyDataset(BaseTimeSeriesDataset):
    ...     def download(self) -> None:
    ...         # Download implementation
    ...         pass
    ...     def load(self, group: str):
    ...         # Load implementation
    ...         pass
"""

import numpy as np
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .config import (
    DatasetSplits,
    NormalizationConfig,
    TimeSeriesConfig,
    WindowConfig,
)

# ---------------------------------------------------------------------

class BaseTimeSeriesDataset(ABC):
    """
    Abstract base class for time series dataset loaders.

    This class defines the interface for loading and preprocessing
    time series forecasting datasets. Subclasses must implement
    the download() and load() methods.

    :param root_dir: Root directory for storing downloaded data.
    :type root_dir: str
    :param verbose: Whether to enable verbose logging.
    :type verbose: bool

    Example:
        >>> class ETTDataset(BaseTimeSeriesDataset):
        ...     def download(self):
        ...         # Implementation
        ...         pass
        ...     def load(self, group):
        ...         # Implementation
        ...         pass
    """

    # Default configuration for the dataset (override in subclasses)
    CONFIGS: Dict[str, TimeSeriesConfig] = {}
    SOURCE_URL: str = ''

    def __init__(
        self,
        root_dir: str = './data',
        verbose: bool = True
    ) -> None:
        """
        Initialize the dataset loader.

        :param root_dir: Root directory for data storage.
        :type root_dir: str
        :param verbose: Enable verbose output.
        :type verbose: bool
        """
        self.root_dir = Path(root_dir)
        self.verbose = verbose

    @abstractmethod
    def download(self, **kwargs) -> None:
        """
        Download the raw dataset files.

        Subclasses must implement this method to download dataset files
        from their source URLs.

        :raises NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement download()")

    @abstractmethod
    def load(
        self,
        group: str,
        cache: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load the dataset into pandas DataFrames.

        :param group: Name of the dataset group/variant to load.
        :type group: str
        :param cache: Whether to use/create cached pickle files.
        :type cache: bool
        :return: Tuple of (Y_df, X_df, S_df) where:
            - Y_df: Target time series with columns [unique_id, ds, y]
            - X_df: Exogenous temporal variables (or None)
            - S_df: Static variables per series (or None)
        :rtype: Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]
        :raises NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement load()")

    def get_config(self, group: str) -> TimeSeriesConfig:
        """
        Get the configuration for a specific dataset group.

        :param group: Name of the dataset group.
        :type group: str
        :return: Configuration dataclass for the group.
        :rtype: TimeSeriesConfig
        :raises ValueError: If the group is not found.

        Example:
            >>> config = dataset.get_config('ETTh1')
            >>> print(config.horizon)
            96
        """
        if group not in self.CONFIGS:
            raise ValueError(
                f"Unknown group '{group}'. "
                f"Available groups: {list(self.CONFIGS.keys())}"
            )
        return self.CONFIGS[group]

    def list_groups(self) -> List[str]:
        """
        List all available dataset groups.

        :return: List of group names.
        :rtype: List[str]

        Example:
            >>> dataset.list_groups()
            ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
        """
        return list(self.CONFIGS.keys())

    def split_data(
        self,
        df: pd.DataFrame,
        config: TimeSeriesConfig,
        normalize: bool = True,
        norm_config: Optional[NormalizationConfig] = None
    ) -> DatasetSplits:
        """
        Split a DataFrame into train/validation/test sets.

        Uses the test_size and val_size from the configuration to perform
        a temporal split, ensuring no data leakage from future to past.

        :param df: DataFrame with time series data.
        :type df: pd.DataFrame
        :param config: Dataset configuration with split sizes.
        :type config: TimeSeriesConfig
        :param normalize: Whether to apply normalization.
        :type normalize: bool
        :param norm_config: Normalization configuration. If None, uses
            standard normalization.
        :type norm_config: Optional[NormalizationConfig]
        :return: DatasetSplits containing train, val, test data.
        :rtype: DatasetSplits

        Example:
            >>> splits = dataset.split_data(df, config, normalize=True)
            >>> print(len(splits.train), len(splits.test))
        """
        if norm_config is None:
            norm_config = NormalizationConfig()

        # Sort by time within each series
        df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

        # Calculate split boundaries
        n_total = len(df)
        test_size = config.test_size
        val_size = config.val_size or 0

        train_end = n_total - test_size - val_size
        val_end = n_total - test_size if val_size > 0 else train_end

        # Perform splits
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy() if val_size > 0 else None
        test_df = df.iloc[val_end:].copy()

        # Compute normalization statistics from training data
        train_mean = None
        train_std = None

        if normalize and norm_config.method != 'none':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if c not in ['unique_id', 'ds']]

            train_mean = train_df[numeric_cols].mean().values
            train_std = train_df[numeric_cols].std().values + norm_config.epsilon

            # Apply normalization
            train_df[numeric_cols] = (
                train_df[numeric_cols] - train_mean
            ) / train_std

            if val_df is not None:
                val_df[numeric_cols] = (
                    val_df[numeric_cols] - train_mean
                ) / train_std

            test_df[numeric_cols] = (
                test_df[numeric_cols] - train_mean
            ) / train_std

        return DatasetSplits(
            train=train_df,
            val=val_df,
            test=test_df,
            train_mean=train_mean,
            train_std=train_std
        )

    def prepare_arrays(
        self,
        df: pd.DataFrame,
        window_config: WindowConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a DataFrame to windowed numpy arrays.

        Creates sliding windows from time series data, handling multiple
        series independently to prevent cross-series contamination.

        :param df: DataFrame with time series data.
        :type df: pd.DataFrame
        :param window_config: Window generation configuration.
        :type window_config: WindowConfig
        :return: Tuple of (X, y) arrays where:
            - X: Input windows of shape (n_windows, window_size, n_features)
            - y: Target windows of shape (n_windows, horizon, n_targets)
        :rtype: Tuple[np.ndarray, np.ndarray]

        Example:
            >>> X, y = dataset.prepare_arrays(df, window_config)
            >>> print(X.shape, y.shape)
            (1000, 96, 7) (1000, 24, 1)
        """
        # Determine feature columns
        exclude_cols = {window_config.id_col, window_config.time_col}
        if window_config.feature_cols is not None:
            feature_cols = window_config.feature_cols
        else:
            feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Ensure target column is in feature columns if needed
        if (window_config.include_target_in_features and
                window_config.target_col not in feature_cols):
            feature_cols.append(window_config.target_col)

        all_x = []
        all_y = []

        # Process each series independently
        for _, group in df.groupby(window_config.id_col):
            series_data = group[feature_cols].values.astype(np.float32)

            # Find target column index
            target_idx = feature_cols.index(window_config.target_col)

            # Create sliding windows
            x_windows, y_windows = self._create_sliding_windows(
                data=series_data,
                window_size=window_config.window_size,
                horizon=window_config.horizon,
                stride=window_config.stride,
                target_idx=target_idx
            )

            if len(x_windows) > 0:
                all_x.append(x_windows)
                all_y.append(y_windows)

        if not all_x:
            raise ValueError(
                "No data windows generated. Check that DataFrame has enough "
                f"rows (>= window_size + horizon = "
                f"{window_config.window_size + window_config.horizon})"
            )

        X = np.concatenate(all_x, axis=0)
        y = np.concatenate(all_y, axis=0)

        return X, y

    @staticmethod
    def _create_sliding_windows(
        data: np.ndarray,
        window_size: int,
        horizon: int,
        stride: int = 1,
        target_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from a numpy array.

        Internal method for generating input-output window pairs.

        :param data: Input array of shape (timesteps, features).
        :type data: np.ndarray
        :param window_size: Length of input sequence.
        :type window_size: int
        :param horizon: Length of target sequence.
        :type horizon: int
        :param stride: Step size between windows.
        :type stride: int
        :param target_idx: Index of the target column in features.
        :type target_idx: int
        :return: Tuple of (inputs, targets) arrays.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        n_samples = (len(data) - window_size - horizon) // stride + 1

        if n_samples <= 0:
            return np.array([]), np.array([])

        x_list = []
        y_list = []

        for i in range(0, n_samples * stride, stride):
            x_list.append(data[i:i + window_size])
            # Extract only target column for y
            y_list.append(data[i + window_size:i + window_size + horizon, target_idx:target_idx + 1])

        return np.array(x_list), np.array(y_list)

    def get_info(self, group: str) -> Dict[str, Any]:
        """
        Get detailed information about a dataset group.

        :param group: Name of the dataset group.
        :type group: str
        :return: Dictionary with dataset metadata and statistics.
        :rtype: Dict[str, Any]

        Example:
            >>> info = dataset.get_info('ETTh1')
            >>> print(info['horizon'], info['seasonality'])
            96 24
        """
        config = self.get_config(group)
        return {
            'name': config.name,
            'frequency': config.freq,
            'seasonality': config.seasonality,
            'horizon': config.horizon,
            'num_series': config.n_ts,
            'test_size': config.test_size,
            'val_size': config.val_size,
            'default_window_size': config.window_size,
            'num_features': config.n_features,
        }

    def __repr__(self) -> str:
        """Return string representation of the dataset loader."""
        return (
            f"{self.__class__.__name__}("
            f"root_dir='{self.root_dir}', "
            f"groups={self.list_groups()})"
        )
