"""
M4 Competition Dataset Loader.

This module provides loaders for the M4 Forecasting Competition dataset,
which contains 100,000 time series across various frequencies (Yearly,
Quarterly, Monthly, Weekly, Daily, Hourly).

The M4 competition is a standard benchmark for evaluating forecasting
methods, with well-defined horizons and evaluation metrics.

Example:
    >>> from dl_techniques.datasets.time_series.m4 import (
    ...     M4Dataset, M4_CONFIGS
    ... )
    >>> dataset = M4Dataset(root_dir='./data')
    >>> y_df, x_df, s_df = dataset.load('Monthly')
    >>> print(f"Number of series: {y_df['unique_id'].nunique()}")
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import BaseTimeSeriesDataset
from .config import TimeSeriesConfig
from .utils import download_file, ensure_directory, get_cache_path

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

# M4 Competition dataset configurations
M4_CONFIGS: Dict[str, TimeSeriesConfig] = {
    'Yearly': TimeSeriesConfig(
        name='Yearly',
        freq='Y',
        seasonality=1,
        horizon=6,
        n_ts=23000,
        test_size=6,
        val_size=None,
        window_size=6,
        n_features=1
    ),
    'Quarterly': TimeSeriesConfig(
        name='Quarterly',
        freq='Q',
        seasonality=4,
        horizon=8,
        n_ts=24000,
        test_size=8,
        val_size=None,
        window_size=8,
        n_features=1
    ),
    'Monthly': TimeSeriesConfig(
        name='Monthly',
        freq='M',
        seasonality=12,
        horizon=18,
        n_ts=48000,
        test_size=18,
        val_size=None,
        window_size=18,
        n_features=1
    ),
    'Weekly': TimeSeriesConfig(
        name='Weekly',
        freq='W',
        seasonality=52,
        horizon=13,
        n_ts=359,
        test_size=13,
        val_size=None,
        window_size=13,
        n_features=1
    ),
    'Daily': TimeSeriesConfig(
        name='Daily',
        freq='D',
        seasonality=7,
        horizon=14,
        n_ts=4227,
        test_size=14,
        val_size=None,
        window_size=14,
        n_features=1
    ),
    'Hourly': TimeSeriesConfig(
        name='Hourly',
        freq='H',
        seasonality=24,
        horizon=48,
        n_ts=414,
        test_size=48,
        val_size=None,
        window_size=48,
        n_features=1
    ),
}


class M4Dataset(BaseTimeSeriesDataset):
    """
    Loader for M4 Competition Dataset.

    The M4 dataset consists of 100,000 time series categorized by
    frequency: Yearly, Quarterly, Monthly, Weekly, Daily, and Hourly.
    Each frequency has its own forecasting horizon.

    Dataset includes:
        - Train/Test splits for all series
        - Static metadata (category) for each series
        - Predefined forecasting horizons

    :param root_dir: Root directory for data storage.
    :type root_dir: str
    :param verbose: Enable verbose logging.
    :type verbose: bool

    Example:
        >>> dataset = M4Dataset(root_dir='./data')
        >>> y_df, _, s_df = dataset.load('Monthly')
        >>> print(f"Series: {y_df['unique_id'].nunique()}")
        >>> print(f"Categories: {s_df['category'].unique()}")
    """

    CONFIGS = M4_CONFIGS
    SOURCE_URL = 'https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset'

    def __init__(
        self,
        root_dir: str = './data',
        verbose: bool = True
    ) -> None:
        """
        Initialize the M4 dataset loader.

        :param root_dir: Root directory for data storage.
        :type root_dir: str
        :param verbose: Enable verbose logging.
        :type verbose: bool
        """
        super().__init__(root_dir=root_dir, verbose=verbose)
        self._data_dir = self.root_dir / 'm4'

    def _get_download_urls(self, group: str) -> List[str]:
        """
        Get download URLs for a specific M4 group.

        :param group: Dataset frequency group name.
        :type group: str
        :return: List of URLs to download.
        :rtype: List[str]
        """
        urls = [
            f'{self.SOURCE_URL}/Train/{group}-train.csv',
            f'{self.SOURCE_URL}/Test/{group}-test.csv',
        ]
        # Info file is shared across groups
        if not (self._data_dir / 'datasets' / 'M4-info.csv').exists():
            urls.append(f'{self.SOURCE_URL}/M4-info.csv')
        return urls

    def download(self, group: Optional[str] = None) -> None:
        """
        Download M4 dataset files.

        :param group: Specific group to download. If None, downloads all.
        :type group: Optional[str]

        Example:
            >>> dataset = M4Dataset()
            >>> dataset.download('Monthly')  # Download Monthly only
            >>> dataset.download()  # Download all groups
        """
        target_dir = self._data_dir / 'datasets'
        ensure_directory(target_dir)

        groups = [group] if group else list(self.CONFIGS.keys())

        for grp in groups:
            if grp not in self.CONFIGS:
                logger.warning(f"Unknown group '{grp}', skipping")
                continue

            urls = self._get_download_urls(grp)
            for url in urls:
                try:
                    download_file(target_dir, url)
                except Exception as e:
                    logger.error(f"Failed to download {url}: {e}")

    def load(
        self,
        group: str,
        cache: bool = True,
        include_test: bool = True
    ) -> Tuple[pd.DataFrame, None, pd.DataFrame]:
        """
        Load a specific M4 frequency group.

        :param group: Frequency group name (e.g., 'Monthly', 'Hourly').
        :type group: str
        :param cache: Whether to use/create cached pickle files.
        :type cache: bool
        :param include_test: Whether to append test data to training series.
        :type include_test: bool
        :return: Tuple of (Y_df, None, S_df) where:
            - Y_df: Time series data with [unique_id, ds, y]
            - None: No temporal exogenous features
            - S_df: Static features with [unique_id, category]
        :rtype: Tuple[pd.DataFrame, None, pd.DataFrame]
        :raises ValueError: If the group is not recognized.

        Example:
            >>> y_df, _, s_df = dataset.load('Monthly')
            >>> print(y_df.groupby('unique_id').size().describe())
        """
        if group not in self.CONFIGS:
            raise ValueError(
                f"Unknown group '{group}'. "
                f"Available: {list(self.CONFIGS.keys())}"
            )

        # Ensure data is downloaded
        self.download(group)

        cache_file = get_cache_path(
            self._data_dir, 'cache', f'{group}.pkl'
        )

        # Return cached data if available
        if cache and cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_pickle(cache_file)

        data_dir = self._data_dir / 'datasets'

        # Load static info
        s_df = self._load_info(data_dir, group)

        # Load time series data
        y_df = self._load_series(data_dir, group, include_test)

        # Filter static info to match loaded series
        s_df = s_df[s_df['unique_id'].isin(y_df['unique_id'].unique())]
        s_df = s_df.reset_index(drop=True)

        # Cache results
        if cache:
            ensure_directory(cache_file.parent)
            pd.to_pickle((y_df, None, s_df), cache_file)
            logger.info(f"Cached data to {cache_file}")

        return y_df, None, s_df

    def _load_info(self, data_dir: Path, group: str) -> pd.DataFrame:
        """
        Load M4 metadata (info) file.

        :param data_dir: Directory containing data files.
        :type data_dir: Path
        :param group: Frequency group name.
        :type group: str
        :return: DataFrame with static metadata.
        :rtype: pd.DataFrame
        """
        info_path = data_dir / 'M4-info.csv'
        if not info_path.exists():
            raise FileNotFoundError(f"M4-info.csv not found at {info_path}")

        s_df = pd.read_csv(info_path, usecols=['M4id', 'category', 'SP'])

        # Filter to specified group
        # SP column indicates the frequency (Yearly, Quarterly, etc.)
        s_df = s_df[s_df['SP'] == group].copy()

        # Encode category as integer
        s_df['category_code'] = s_df['category'].astype('category').cat.codes

        # Rename columns
        s_df = s_df.rename(columns={'M4id': 'unique_id'})
        s_df = s_df[['unique_id', 'category', 'category_code']]

        return s_df

    def _load_series(
        self,
        data_dir: Path,
        group: str,
        include_test: bool
    ) -> pd.DataFrame:
        """
        Load and combine train/test series data.

        :param data_dir: Directory containing data files.
        :type data_dir: Path
        :param group: Frequency group name.
        :type group: str
        :param include_test: Whether to append test data.
        :type include_test: bool
        :return: Combined time series DataFrame.
        :rtype: pd.DataFrame
        """
        def read_and_melt(filepath: Path) -> pd.DataFrame:
            """Read M4 CSV and convert to long format."""
            df = pd.read_csv(filepath)
            # First column is series ID (V1), rest are values
            id_col = df.columns[0]
            # Create numeric column names for melting
            value_cols = list(range(1, len(df.columns)))
            df.columns = [id_col] + value_cols

            melted = pd.melt(
                df,
                id_vars=[id_col],
                var_name='ds',
                value_name='y'
            )
            melted = melted.dropna(subset=['y'])
            melted = melted.rename(columns={id_col: 'unique_id'})
            return melted

        # Load training data
        train_path = data_dir / f'{group}-train.csv'
        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_path}")

        df_train = read_and_melt(train_path)
        logger.info(f"Loaded {len(df_train)} training observations")

        if not include_test:
            df_train = df_train.sort_values(['unique_id', 'ds'])
            df_train = df_train.reset_index(drop=True)
            return df_train

        # Load test data
        test_path = data_dir / f'{group}-test.csv'
        if not test_path.exists():
            logger.warning(f"Test file not found: {test_path}")
            df_train = df_train.sort_values(['unique_id', 'ds'])
            df_train = df_train.reset_index(drop=True)
            return df_train

        df_test = read_and_melt(test_path)
        logger.info(f"Loaded {len(df_test)} test observations")

        # Adjust test timestamps to continue from training
        # Get max timestamp per series from training
        len_train = df_train.groupby('unique_id')['ds'].max().reset_index()
        len_train.columns = ['unique_id', 'train_max_ds']

        # Merge and adjust
        df_test = df_test.merge(len_train, on='unique_id', how='left')
        df_test['ds'] = df_test['ds'].astype(int) + df_test['train_max_ds'].astype(int)
        df_test = df_test.drop(columns=['train_max_ds'])

        # Combine train and test
        df = pd.concat([df_train, df_test], ignore_index=True)
        df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

        logger.info(f"Combined dataset has {len(df)} observations")
        return df

    def load_train_test_split(
        self,
        group: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load M4 data with explicit train/test split.

        Returns separate DataFrames for training and test data,
        useful for evaluation against M4 benchmark results.

        :param group: Frequency group name.
        :type group: str
        :return: Tuple of (train_df, test_df, info_df).
        :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

        Example:
            >>> train, test, info = dataset.load_train_test_split('Monthly')
            >>> # Evaluate forecasts
            >>> horizon = M4_CONFIGS['Monthly'].horizon
        """
        self.download(group)
        data_dir = self._data_dir / 'datasets'

        # Load info
        s_df = self._load_info(data_dir, group)

        # Load train
        train_path = data_dir / f'{group}-train.csv'
        df_train = pd.read_csv(train_path)
        id_col = df_train.columns[0]
        value_cols = list(range(1, len(df_train.columns)))
        df_train.columns = [id_col] + value_cols

        train_melted = pd.melt(
            df_train,
            id_vars=[id_col],
            var_name='ds',
            value_name='y'
        ).dropna().rename(columns={id_col: 'unique_id'})

        # Load test
        test_path = data_dir / f'{group}-test.csv'
        df_test = pd.read_csv(test_path)
        df_test.columns = [id_col] + value_cols[:len(df_test.columns) - 1]

        test_melted = pd.melt(
            df_test,
            id_vars=[id_col],
            var_name='ds',
            value_name='y'
        ).dropna().rename(columns={id_col: 'unique_id'})

        return train_melted, test_melted, s_df

    def get_naive2_forecasts(self, group: str) -> Optional[pd.DataFrame]:
        """
        Get Naive2 baseline forecasts for comparison.

        The Naive2 method is a common baseline in the M4 competition.
        Returns None if baseline forecasts are not available.

        :param group: Frequency group name.
        :type group: str
        :return: DataFrame with Naive2 forecasts or None.
        :rtype: Optional[pd.DataFrame]
        """
        # Naive2 forecasts would need to be downloaded separately
        # This is a placeholder for the interface
        logger.warning(
            "Naive2 forecasts not included. "
            "Download from M4 repository if needed."
        )
        return None


# Convenience functions

def load_m4(
    group: str,
    root_dir: str = './data',
    include_test: bool = True
) -> Tuple[pd.DataFrame, None, pd.DataFrame]:
    """
    Load M4 dataset with default settings.

    :param group: Frequency group (Yearly, Quarterly, Monthly, etc.).
    :type group: str
    :param root_dir: Data directory.
    :type root_dir: str
    :param include_test: Whether to include test data.
    :type include_test: bool
    :return: Tuple of (Y_df, None, S_df).
    :rtype: Tuple[pd.DataFrame, None, pd.DataFrame]

    Example:
        >>> y_df, _, s_df = load_m4('Monthly')
        >>> print(y_df['unique_id'].nunique())
        48000
    """
    dataset = M4Dataset(root_dir=root_dir)
    return dataset.load(group, include_test=include_test)


def get_m4_horizon(group: str) -> int:
    """
    Get the standard forecast horizon for an M4 group.

    :param group: Frequency group name.
    :type group: str
    :return: Forecast horizon in time steps.
    :rtype: int

    Example:
        >>> get_m4_horizon('Monthly')
        18
    """
    if group not in M4_CONFIGS:
        raise ValueError(f"Unknown group '{group}'")
    return M4_CONFIGS[group].horizon


def get_m4_seasonality(group: str) -> int:
    """
    Get the seasonality period for an M4 group.

    :param group: Frequency group name.
    :type group: str
    :return: Seasonality period in time steps.
    :rtype: int

    Example:
        >>> get_m4_seasonality('Monthly')
        12
    """
    if group not in M4_CONFIGS:
        raise ValueError(f"Unknown group '{group}'")
    return M4_CONFIGS[group].seasonality
