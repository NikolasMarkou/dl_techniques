"""
Time Series Dataset Base Classes.

This module provides abstract base classes for time series dataset loaders,
defining a consistent interface for downloading, loading, and splitting data.
It enforces strict separation of concerns between data fetching, loading,
and preprocessing.

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
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .config import (
    DatasetSplits,
    NormalizationConfig,
    TimeSeriesConfig,
    WindowConfig,
)
from .normalizer import TimeSeriesNormalizer

# ---------------------------------------------------------------------

class BaseTimeSeriesDataset(ABC):
    """
    Abstract base class for time series dataset loaders.

    This class defines the interface for loading and preprocessing
    time series forecasting datasets. Subclasses must implement
    the `download()` and `load()` methods.

    Attributes:
        CONFIGS (Dict[str, TimeSeriesConfig]): Registry of dataset configurations.
        SOURCE_URL (str): Default URL for data download.
        root_dir (Path): Base directory for data storage.
        verbose (bool): Whether to log detailed operations.
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
    def download(self, **kwargs: Any) -> None:
        """
        Download the raw dataset files.

        Subclasses must implement this method to download dataset files
        from their source URLs.

        :param kwargs: Additional download parameters (e.g., specific group).
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

        This method performs a temporal split per 'unique_id' group to ensure
        data integrity across multiple time series. It prevents future leakage
        between train/test splits even when series are stacked.

        :param df: DataFrame with time series data.
        :type df: pd.DataFrame
        :param config: Dataset configuration with split sizes.
        :type config: TimeSeriesConfig
        :param normalize: Whether to apply normalization.
        :type normalize: bool
        :param norm_config: Normalization configuration.
        :type norm_config: Optional[NormalizationConfig]
        :return: DatasetSplits containing train, val, test data.
        :rtype: DatasetSplits
        """
        if norm_config is None:
            norm_config = NormalizationConfig()

        # Sort by time within each series to ensure correct temporal splitting
        df = df.sort_values(['unique_id', 'ds'])

        train_dfs: List[pd.DataFrame] = []
        val_dfs: List[pd.DataFrame] = []
        test_dfs: List[pd.DataFrame] = []

        # Process each series independently to prevent leakage
        for _, group in df.groupby('unique_id'):
            n_total = len(group)
            test_size = config.test_size
            val_size = config.val_size or 0

            train_end = n_total - test_size - val_size
            val_end = n_total - test_size if val_size > 0 else train_end

            # Check for insufficient data
            if train_end <= 0:
                logger.warning(
                    f"Series {group['unique_id'].iloc[0]} too short for requested split. "
                    f"Length: {n_total}, Req: {test_size + val_size}"
                )
                continue

            train_dfs.append(group.iloc[:train_end])
            if val_size > 0:
                val_dfs.append(group.iloc[train_end:val_end])
            test_dfs.append(group.iloc[val_end:])

        # Concatenate results
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else None
        test_df = pd.concat(test_dfs, ignore_index=True)

        # Handle Normalization
        train_mean: Optional[np.ndarray] = None
        train_std: Optional[np.ndarray] = None

        if normalize and norm_config.method != 'none':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            # Exclude ID and likely Time/DS columns if they are numeric
            cols_to_norm = [c for c in numeric_cols if c not in ['unique_id', 'ds', 'date']]

            if not cols_to_norm:
                logger.warning("No numeric columns found to normalize.")
            else:
                # Initialize Normalizer
                normalizer = TimeSeriesNormalizer(
                    method=norm_config.method,
                    epsilon=norm_config.epsilon,
                    feature_range=(-1, 1) if norm_config.method == 'minmax' else (0, 1)
                )

                # Fit on Training Data ONLY to prevent leakage
                train_data = train_df[cols_to_norm].values
                normalizer.fit(train_data)

                # Transform all sets
                train_df[cols_to_norm] = normalizer.transform(train_data)

                if val_df is not None:
                    val_df[cols_to_norm] = normalizer.transform(val_df[cols_to_norm].values)

                test_df[cols_to_norm] = normalizer.transform(test_df[cols_to_norm].values)

                # Extract statistics for backward compatibility with DatasetSplits
                stats = normalizer.get_statistics()
                if norm_config.method == 'standard':
                    train_mean = np.array(stats.get('mean_val'))
                    train_std = np.array(stats.get('std_val'))

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
        :raises ValueError: If no windows can be generated.
        """
        # Lazy import to avoid circular dependencies
        from .pipeline import create_sliding_windows as pipeline_create_windows

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

        target_idx = feature_cols.index(window_config.target_col)

        all_x: List[np.ndarray] = []
        all_y: List[np.ndarray] = []

        # Process each series independently
        for _, group in df.groupby(window_config.id_col):
            series_data = group[feature_cols].values.astype(np.float32)

            # Use pipeline logic for efficient windowing
            try:
                x_win, y_win_full = pipeline_create_windows(
                    data=series_data,
                    window_size=window_config.window_size,
                    horizon=window_config.horizon,
                    stride=window_config.stride
                )

                # Slice Y to get only the target index (pipeline returns full features)
                # Shape: (n_windows, horizon, 1)
                y_win = y_win_full[:, :, target_idx:target_idx + 1]

                if len(x_win) > 0:
                    all_x.append(x_win)
                    all_y.append(y_win)

            except ValueError:
                # Skip series that are too short
                continue

        if not all_x:
            raise ValueError(
                "No data windows generated. Check that DataFrame has enough "
                f"rows (>= {window_config.window_size + window_config.horizon})"
            )

        X = np.concatenate(all_x, axis=0)
        y = np.concatenate(all_y, axis=0)

        return X, y

    def get_info(self, group: str) -> Dict[str, Any]:
        """
        Get detailed information about a dataset group.

        :param group: Name of the dataset group.
        :type group: str
        :return: Dictionary with dataset metadata and statistics.
        :rtype: Dict[str, Any]
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