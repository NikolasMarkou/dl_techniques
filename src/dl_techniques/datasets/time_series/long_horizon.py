"""
Long Horizon Forecasting Benchmark Datasets.

This module provides loaders for popular long-horizon time series
forecasting benchmarks including ETT (Electricity Transformer Temperature),
ECL (Electricity), Traffic, Weather, ILI (Influenza-Like Illness),
and Exchange Rate datasets.

These datasets are commonly used for evaluating Transformer-based
forecasting models like Informer, Autoformer, and FEDformer.

Example:
    >>> from dl_techniques.datasets.time_series.long_horizon import (
    ...     LongHorizonDataset, LONG_HORIZON_CONFIGS
    ... )
    >>> dataset = LongHorizonDataset(root_dir='./data')
    >>> y_df, x_df, s_df = dataset.load('ETTh1')
    >>> print(y_df.shape)
"""

import pandas as pd
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import BaseTimeSeriesDataset
from .config import TimeSeriesConfig
from .utils import download_file, ensure_directory, get_cache_path
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

# Dataset configurations for long-horizon benchmarks
LONG_HORIZON_CONFIGS: Dict[str, TimeSeriesConfig] = {
    'ETTh1': TimeSeriesConfig(
        name='ETTh1',
        freq='H',
        seasonality=24,
        horizon=96,
        n_ts=1,
        test_size=2880,
        val_size=2880,
        window_size=96,
        n_features=7
    ),
    'ETTh2': TimeSeriesConfig(
        name='ETTh2',
        freq='H',
        seasonality=24,
        horizon=96,
        n_ts=1,
        test_size=2880,
        val_size=2880,
        window_size=96,
        n_features=7
    ),
    'ETTm1': TimeSeriesConfig(
        name='ETTm1',
        freq='15T',
        seasonality=96,
        horizon=96,
        n_ts=1,
        test_size=11520,
        val_size=11520,
        window_size=96,
        n_features=7
    ),
    'ETTm2': TimeSeriesConfig(
        name='ETTm2',
        freq='15T',
        seasonality=96,
        horizon=96,
        n_ts=1,
        test_size=11520,
        val_size=11520,
        window_size=96,
        n_features=7
    ),
    'ECL': TimeSeriesConfig(
        name='ECL',
        freq='H',
        seasonality=24,
        horizon=96,
        n_ts=321,
        test_size=5260,
        val_size=2632,
        window_size=96,
        n_features=321
    ),
    'Traffic': TimeSeriesConfig(
        name='traffic',
        freq='H',
        seasonality=24,
        horizon=96,
        n_ts=862,
        test_size=3508,
        val_size=1756,
        window_size=96,
        n_features=862
    ),
    'Weather': TimeSeriesConfig(
        name='weather',
        freq='10T',
        seasonality=144,
        horizon=96,
        n_ts=21,
        test_size=10539,
        val_size=5270,
        window_size=96,
        n_features=21
    ),
    'ILI': TimeSeriesConfig(
        name='ili',
        freq='W',
        seasonality=52,
        horizon=24,
        n_ts=7,
        test_size=193,
        val_size=97,
        window_size=36,
        n_features=7
    ),
    'Exchange': TimeSeriesConfig(
        name='exchange_rate',
        freq='D',
        seasonality=1,
        horizon=96,
        n_ts=8,
        test_size=1517,
        val_size=760,
        window_size=96,
        n_features=8
    )
}


class LongHorizonDataset(BaseTimeSeriesDataset):
    """
    Loader for Long-Horizon Forecasting Benchmark Datasets.

    Provides access to popular time series forecasting benchmarks
    used in academic research. Datasets are downloaded from the
    NHiTS experiments repository.

    Supported datasets:
        - ETTh1, ETTh2: Electricity Transformer Temperature (hourly)
        - ETTm1, ETTm2: Electricity Transformer Temperature (15-min)
        - ECL: Electricity Load Diagrams
        - Traffic: California road occupancy rates
        - Weather: Max Planck Institute weather data
        - ILI: CDC Influenza-Like Illness data
        - Exchange: Currency exchange rates

    :param root_dir: Root directory for data storage.
    :type root_dir: str
    :param verbose: Enable verbose logging.
    :type verbose: bool

    Example:
        >>> dataset = LongHorizonDataset(root_dir='./data')
        >>> y_df, x_df, s_df = dataset.load('ETTh1')
        >>> print(y_df.columns.tolist())
        ['unique_id', 'ds', 'y']
    """

    CONFIGS = LONG_HORIZON_CONFIGS
    SOURCE_URL = 'https://nhits-experiments.s3.amazonaws.com/datasets.zip'

    # Alternative source URLs for individual datasets
    ETT_SOURCE_URL = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small'
    ALTERNATE_URLS = {
        'ETTh1': f'{ETT_SOURCE_URL}/ETTh1.csv',
        'ETTh2': f'{ETT_SOURCE_URL}/ETTh2.csv',
        'ETTm1': f'{ETT_SOURCE_URL}/ETTm1.csv',
        'ETTm2': f'{ETT_SOURCE_URL}/ETTm2.csv',
    }

    def __init__(
        self,
        root_dir: str = './data',
        verbose: bool = True
    ) -> None:
        """
        Initialize the Long Horizon dataset loader.

        :param root_dir: Root directory for data storage.
        :type root_dir: str
        :param verbose: Enable verbose logging.
        :type verbose: bool
        """
        super().__init__(root_dir=root_dir, verbose=verbose)
        self._data_dir = self.root_dir / 'longhorizon'

    def download(self, group: Optional[str] = None) -> None:
        """
        Download the long horizon datasets.

        Downloads the complete dataset archive from the source URL.
        For ETT datasets, can also download individual files.

        :param group: Optional specific group to download.
            If None, downloads the complete archive.
        :type group: Optional[str]

        Example:
            >>> dataset = LongHorizonDataset()
            >>> dataset.download()  # Downloads all datasets
            >>> dataset.download('ETTh1')  # Downloads only ETTh1
        """
        target_dir = self._data_dir / 'datasets'

        # Check if already downloaded
        if target_dir.exists() and any(target_dir.iterdir()):
            logger.info(f"Datasets already exist in {target_dir}")
            return

        ensure_directory(target_dir)

        # Download specific ETT dataset if requested
        if group in self.ALTERNATE_URLS:
            url = self.ALTERNATE_URLS[group]
            logger.info(f"Downloading {group} from {url}")
            download_file(target_dir, url)
            return

        # Download complete archive
        logger.info(f"Downloading datasets from {self.SOURCE_URL}")
        download_file(self._data_dir, self.SOURCE_URL, decompress=True)

    def load(
        self,
        group: str,
        cache: bool = True,
        multivariate: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load a specific long-horizon dataset.

        :param group: Name of the dataset to load (e.g., 'ETTh1', 'Weather').
        :type group: str
        :param cache: Whether to use/create cached pickle files.
        :type cache: bool
        :param multivariate: If True, loads multivariate version.
            If False, loads univariate version (only target column).
        :type multivariate: bool
        :return: Tuple of (Y_df, X_df, S_df) where:
            - Y_df: DataFrame with [unique_id, ds, y] columns
            - X_df: DataFrame with exogenous features (or None)
            - S_df: Static features (always None for these datasets)
        :rtype: Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]
        :raises ValueError: If the group is not recognized.

        Example:
            >>> y_df, x_df, s_df = dataset.load('ETTh1', multivariate=True)
            >>> print(y_df.head())
        """
        if group not in self.CONFIGS:
            raise ValueError(
                f"Unknown group '{group}'. "
                f"Available: {list(self.CONFIGS.keys())}"
            )

        # Ensure data is downloaded
        self.download(group)

        config = self.CONFIGS[group]
        cache_suffix = '_mv' if multivariate else '_uv'
        cache_file = get_cache_path(
            self._data_dir, 'cache', f'{group}{cache_suffix}.pkl'
        )

        # Return cached data if available
        if cache and cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_pickle(cache_file)

        # Determine file path based on dataset type
        if group.startswith('ETT'):
            y_df, x_df = self._load_ett_dataset(group, multivariate)
        elif group in ['ECL', 'Traffic', 'Weather', 'ILI', 'Exchange']:
            y_df, x_df = self._load_standard_dataset(group, config, multivariate)
        else:
            raise ValueError(f"Loading logic not implemented for {group}")

        s_df = None  # No static features for these datasets

        # Cache results
        if cache:
            ensure_directory(cache_file.parent)
            pd.to_pickle((y_df, x_df, s_df), cache_file)
            logger.info(f"Cached data to {cache_file}")

        return y_df, x_df, s_df

    def _load_ett_dataset(
        self,
        group: str,
        multivariate: bool
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load ETT (Electricity Transformer Temperature) dataset.

        :param group: Dataset name (ETTh1, ETTh2, ETTm1, ETTm2).
        :type group: str
        :param multivariate: Whether to load all features.
        :type multivariate: bool
        :return: Tuple of (Y_df, X_df).
        :rtype: Tuple[pd.DataFrame, Optional[pd.DataFrame]]
        """
        # Try multiple possible paths
        possible_paths = [
            self._data_dir / 'datasets' / f'{group}.csv',
            self._data_dir / 'datasets' / group / 'M' / 'df_y.csv',
            self._data_dir / f'{group}.csv',
        ]

        df = None
        for path in possible_paths:
            if path.exists():
                df = pd.read_csv(path)
                break

        if df is None:
            # Try downloading directly
            self.download(group)
            csv_path = self._data_dir / 'datasets' / f'{group}.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                raise FileNotFoundError(
                    f"Could not find {group} data. "
                    f"Tried paths: {possible_paths}"
                )

        # Standard ETT format: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
        date_col = df.columns[0]  # Usually 'date'
        target_col = 'OT'

        # Create standardized DataFrame
        y_df = pd.DataFrame({
            'unique_id': group,
            'ds': pd.to_datetime(df[date_col]),
            'y': df[target_col].astype(float)
        })

        x_df = None
        if multivariate:
            feature_cols = [c for c in df.columns if c not in [date_col]]
            x_df = df[[date_col] + feature_cols].copy()
            x_df.rename(columns={date_col: 'ds'}, inplace=True)
            x_df['ds'] = pd.to_datetime(x_df['ds'])
            x_df['unique_id'] = group

        return y_df, x_df

    def _load_standard_dataset(
        self,
        group: str,
        config: TimeSeriesConfig,
        multivariate: bool
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load standard format datasets (ECL, Traffic, Weather, etc.).

        :param group: Dataset name.
        :type group: str
        :param config: Dataset configuration.
        :type config: TimeSeriesConfig
        :param multivariate: Whether to load all features.
        :type multivariate: bool
        :return: Tuple of (Y_df, X_df).
        :rtype: Tuple[pd.DataFrame, Optional[pd.DataFrame]]
        """
        kind = 'M' if multivariate else 'S'
        base_path = self._data_dir / 'datasets' / config.name / kind

        # Load Y (targets)
        y_path = base_path / 'df_y.csv'
        if not y_path.exists():
            raise FileNotFoundError(f"Target file not found: {y_path}")

        y_df = pd.read_csv(y_path)
        y_df = y_df.sort_values(['unique_id', 'ds'], ignore_index=True)
        y_df = y_df[['unique_id', 'ds', 'y']]

        # Load X (exogenous features)
        x_df = None
        x_path = base_path / 'df_x.csv'
        if x_path.exists() and multivariate:
            x_df = pd.read_csv(x_path)
            # Merge to align timestamps
            x_df = y_df[['unique_id', 'ds']].merge(x_df, how='left', on=['ds'])

        return y_df, x_df

    def load_raw(self, group: str) -> pd.DataFrame:
        """
        Load raw CSV data without preprocessing.

        Useful for custom preprocessing pipelines.

        :param group: Dataset name.
        :type group: str
        :return: Raw DataFrame from CSV.
        :rtype: pd.DataFrame

        Example:
            >>> raw_df = dataset.load_raw('ETTh1')
            >>> print(raw_df.columns)
        """
        self.download(group)

        if group.startswith('ETT'):
            csv_path = self._data_dir / 'datasets' / f'{group}.csv'
            if not csv_path.exists():
                csv_path = self._data_dir / f'{group}.csv'
        else:
            config = self.CONFIGS[group]
            csv_path = self._data_dir / 'datasets' / config.name / 'M' / 'df_y.csv'

        if not csv_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {csv_path}")

        return pd.read_csv(csv_path)

    def get_splits(
        self,
        group: str,
        normalize: bool = True,
        multivariate: bool = True
    ):
        """
        Load and split a dataset into train/val/test sets.

        Convenience method that loads the data and applies the
        standard train/validation/test split for the dataset.

        :param group: Dataset name.
        :type group: str
        :param normalize: Whether to normalize the data.
        :type normalize: bool
        :param multivariate: Whether to load multivariate version.
        :type multivariate: bool
        :return: DatasetSplits object with train, val, test DataFrames.
        :rtype: DatasetSplits

        Example:
            >>> splits = dataset.get_splits('ETTh1')
            >>> print(len(splits.train), len(splits.val), len(splits.test))
        """
        y_df, x_df, _ = self.load(group, multivariate=multivariate)
        config = self.CONFIGS[group]

        # Merge Y and X for splitting
        if x_df is not None:
            df = y_df.merge(
                x_df.drop(columns=['unique_id'], errors='ignore'),
                on='ds',
                how='left'
            )
        else:
            df = y_df.copy()

        return self.split_data(df, config, normalize=normalize)


# Convenience functions for common use cases

def load_ett(
    variant: str = 'h1',
    root_dir: str = './data',
    multivariate: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load ETT (Electricity Transformer Temperature) dataset.

    Convenience function for loading ETT variants.

    :param variant: ETT variant ('h1', 'h2', 'm1', 'm2').
    :type variant: str
    :param root_dir: Data directory.
    :type root_dir: str
    :param multivariate: Load multivariate version.
    :type multivariate: bool
    :return: Tuple of (Y_df, X_df, S_df).
    :rtype: Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]

    Example:
        >>> y_df, x_df, _ = load_ett('h1', multivariate=True)
    """
    variant_map = {
        'h1': 'ETTh1', 'h2': 'ETTh2',
        'm1': 'ETTm1', 'm2': 'ETTm2',
        'ETTh1': 'ETTh1', 'ETTh2': 'ETTh2',
        'ETTm1': 'ETTm1', 'ETTm2': 'ETTm2',
    }

    group = variant_map.get(variant.lower(), variant)
    dataset = LongHorizonDataset(root_dir=root_dir)
    return dataset.load(group, multivariate=multivariate)


def load_weather(
    root_dir: str = './data',
    multivariate: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load Weather dataset.

    :param root_dir: Data directory.
    :type root_dir: str
    :param multivariate: Load multivariate version.
    :type multivariate: bool
    :return: Tuple of (Y_df, X_df, S_df).
    :rtype: Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]
    """
    dataset = LongHorizonDataset(root_dir=root_dir)
    return dataset.load('Weather', multivariate=multivariate)


def load_ecl(
    root_dir: str = './data',
    multivariate: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load ECL (Electricity) dataset.

    :param root_dir: Data directory.
    :type root_dir: str
    :param multivariate: Load multivariate version.
    :type multivariate: bool
    :return: Tuple of (Y_df, X_df, S_df).
    :rtype: Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]
    """
    dataset = LongHorizonDataset(root_dir=root_dir)
    return dataset.load('ECL', multivariate=multivariate)
