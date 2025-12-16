"""
Favorita Grocery Sales Forecasting Dataset.

This module provides a loader for the Corporación Favorita grocery sales
forecasting dataset, originally from a Kaggle competition. The dataset
contains sales data from grocery stores in Ecuador.

Note: This dataset requires significant preprocessing. The full pipeline
is available but may take considerable time and memory to execute.

Example:
    >>> from dl_techniques.datasets.time_series.favorita import (
    ...     FavoritaDataset, FAVORITA_CONFIG
    ... )
    >>> dataset = FavoritaDataset(root_dir='./data')
    >>> # Load preprocessed subset
    >>> y_df, x_df, s_df = dataset.load('Favorita200')
"""

import pandas as pd
from typing import Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import BaseTimeSeriesDataset
from .config import TimeSeriesConfig
from .utils import download_file, extract_file, ensure_directory, get_cache_path
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

# Favorita dataset configuration
FAVORITA_CONFIG = TimeSeriesConfig(
    name='Favorita',
    freq='D',
    seasonality=7,
    horizon=16,
    n_ts=3995,
    test_size=16,
    val_size=30,
    window_size=90,
    n_features=1
)

# Subset configurations for easier experimentation
FAVORITA_CONFIGS: Dict[str, TimeSeriesConfig] = {
    'Favorita200': TimeSeriesConfig(
        name='Favorita200',
        freq='D',
        seasonality=7,
        horizon=16,
        n_ts=200,
        test_size=16,
        val_size=30,
        window_size=90,
        n_features=1
    ),
    'Favorita500': TimeSeriesConfig(
        name='Favorita500',
        freq='D',
        seasonality=7,
        horizon=16,
        n_ts=500,
        test_size=16,
        val_size=30,
        window_size=90,
        n_features=1
    ),
    'FavoritaComplete': FAVORITA_CONFIG,
}


class FavoritaDataset(BaseTimeSeriesDataset):
    """
    Loader for Favorita Grocery Sales Forecasting Dataset.

    The Favorita dataset contains daily sales data from Ecuador grocery
    stores, with rich metadata including store information, item details,
    promotions, and external factors (oil prices, holidays).

    Available variants:
        - Favorita200: 200 top-selling store-item combinations
        - Favorita500: 500 top-selling combinations
        - FavoritaComplete: All series (~4000 combinations)

    :param root_dir: Root directory for data storage.
    :type root_dir: str
    :param verbose: Enable verbose logging.
    :type verbose: bool

    Note:
        Processing the complete dataset requires substantial memory
        and computation time. Use Favorita200 or Favorita500 for
        initial experiments.

    Example:
        >>> dataset = FavoritaDataset(root_dir='./data')
        >>> y_df, x_df, s_df = dataset.load('Favorita200')
    """

    CONFIGS = FAVORITA_CONFIGS
    SOURCE_URL = 'https://www.dropbox.com/s/xi019gtvdtmsj9j/favorita-grocery-sales-forecasting2.zip?dl=1'

    # Files in the Favorita archive
    DATA_FILES = [
        'train.csv.zip',
        'test.csv.zip',
        'items.csv.zip',
        'stores.csv.zip',
        'transactions.csv.zip',
        'holidays_events.csv.zip',
        'oil.csv.zip'
    ]

    def __init__(
        self,
        root_dir: str = './data',
        verbose: bool = True
    ) -> None:
        """
        Initialize the Favorita dataset loader.

        :param root_dir: Root directory for data storage.
        :type root_dir: str
        :param verbose: Enable verbose logging.
        :type verbose: bool
        """
        super().__init__(root_dir=root_dir, verbose=verbose)
        self._data_dir = self.root_dir / 'favorita'

    def download(self) -> None:
        """
        Download the Favorita dataset.

        Downloads and extracts the main archive and internal zip files.

        Example:
            >>> dataset = FavoritaDataset()
            >>> dataset.download()
        """
        target_dir = self._data_dir

        # Check if already extracted
        if (target_dir / 'train.csv').exists():
            logger.info("Favorita data already downloaded and extracted")
            return

        ensure_directory(target_dir)

        # Download main archive
        logger.info(f"Downloading Favorita dataset from {self.SOURCE_URL}")
        archive_path = download_file(
            target_dir,
            self.SOURCE_URL,
            filename='favorita.zip'
        )

        # Extract main archive
        extract_file(archive_path, target_dir)

        # Extract internal zip files
        for data_file in self.DATA_FILES:
            zip_path = target_dir / data_file
            if zip_path.exists():
                csv_name = data_file.replace('.zip', '')
                if not (target_dir / csv_name).exists():
                    logger.info(f"Extracting {data_file}")
                    extract_file(zip_path, target_dir)

        logger.info("Favorita dataset ready")

    def load(
        self,
        group: str = 'Favorita200',
        cache: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load the Favorita dataset.

        :param group: Dataset variant ('Favorita200', 'Favorita500', 'FavoritaComplete').
        :type group: str
        :param cache: Whether to use/create cached pickle files.
        :type cache: bool
        :return: Tuple of (Y_df, X_df, S_df) where:
            - Y_df: Daily sales with [unique_id, ds, y]
            - X_df: Temporal features (promotions, oil price, holidays)
            - S_df: Static features (store info, item category)
        :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        :raises ValueError: If the group is not recognized.

        Example:
            >>> y_df, x_df, s_df = dataset.load('Favorita200')
            >>> print(y_df.head())
        """
        if group not in self.CONFIGS:
            raise ValueError(
                f"Unknown group '{group}'. "
                f"Available: {list(self.CONFIGS.keys())}"
            )

        # Check cache
        cache_file = get_cache_path(
            self._data_dir, 'cache', f'{group}.pkl'
        )

        if cache and cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_pickle(cache_file)

        # Ensure data is downloaded
        self.download()

        config = self.CONFIGS[group]

        # Process data
        y_df, x_df, s_df = self._process_data(config.n_ts)

        # Cache results
        if cache:
            ensure_directory(cache_file.parent)
            pd.to_pickle((y_df, x_df, s_df), cache_file)
            logger.info(f"Cached data to {cache_file}")

        return y_df, x_df, s_df

    def _process_data(
        self,
        n_series: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Process raw Favorita data files.

        :param n_series: Number of series to include (top sellers).
        :type n_series: int
        :return: Tuple of processed DataFrames.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        """
        logger.info("Processing Favorita data (this may take a while)...")

        # Load main sales data
        train_path = self._data_dir / 'train.csv'
        if not train_path.exists():
            raise FileNotFoundError(
                f"train.csv not found at {train_path}. "
                "Run download() first."
            )

        # Read with chunking for large files
        logger.info("Loading train.csv...")
        df = pd.read_csv(
            train_path,
            parse_dates=['date'],
            dtype={
                'id': 'int64',
                'store_nbr': 'int16',
                'item_nbr': 'int32',
                'unit_sales': 'float32',
                'onpromotion': 'object'
            }
        )

        # Create unique series identifier
        df['unique_id'] = (
            df['store_nbr'].astype(str) + '_' +
            df['item_nbr'].astype(str)
        )

        # Select top n_series by total sales
        logger.info(f"Selecting top {n_series} series by sales volume...")
        series_totals = df.groupby('unique_id')['unit_sales'].sum()
        top_series = series_totals.nlargest(n_series).index.tolist()
        df = df[df['unique_id'].isin(top_series)]

        # Create Y_df
        y_df = df[['unique_id', 'date', 'unit_sales']].copy()
        y_df.columns = ['unique_id', 'ds', 'y']
        y_df['y'] = y_df['y'].clip(lower=0)  # No negative sales
        y_df = y_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

        # Create X_df (temporal features)
        x_df = self._create_temporal_features(df)

        # Create S_df (static features)
        s_df = self._create_static_features(df)

        logger.info(
            f"Processed {y_df['unique_id'].nunique()} series, "
            f"{len(y_df)} observations"
        )

        return y_df, x_df, s_df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal exogenous features.

        :param df: Main sales DataFrame.
        :type df: pd.DataFrame
        :return: Temporal features DataFrame.
        :rtype: pd.DataFrame
        """
        # Promotion indicator
        x_df = df[['unique_id', 'date', 'onpromotion']].copy()
        x_df.columns = ['unique_id', 'ds', 'onpromotion']

        # Convert promotion to numeric
        x_df['onpromotion'] = x_df['onpromotion'].map(
            {'True': 1.0, 'False': 0.0, True: 1.0, False: 0.0}
        ).fillna(0.0).astype('float32')

        # Load oil prices
        oil_path = self._data_dir / 'oil.csv'
        if oil_path.exists():
            oil_df = pd.read_csv(oil_path, parse_dates=['date'])
            oil_df = oil_df.rename(columns={'date': 'ds', 'dcoilwtico': 'oil_price'})
            oil_df['oil_price'] = oil_df['oil_price'].ffill().bfill()
            x_df = x_df.merge(oil_df[['ds', 'oil_price']], on='ds', how='left')
            x_df['oil_price'] = x_df['oil_price'].ffill().bfill().astype('float32')
        else:
            logger.warning("oil.csv not found, skipping oil price feature")

        # Load holidays
        holidays_path = self._data_dir / 'holidays_events.csv'
        if holidays_path.exists():
            holidays_df = pd.read_csv(holidays_path, parse_dates=['date'])
            holidays_df = holidays_df[holidays_df['transferred'] == False]
            holidays_df['is_holiday'] = 1.0
            holidays_df = holidays_df[['date', 'is_holiday']].drop_duplicates()
            holidays_df.columns = ['ds', 'is_holiday']
            x_df = x_df.merge(holidays_df, on='ds', how='left')
            x_df['is_holiday'] = x_df['is_holiday'].fillna(0.0).astype('float32')
        else:
            logger.warning("holidays_events.csv not found, skipping holiday feature")

        x_df = x_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        return x_df

    def _create_static_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create static features for each series.

        :param df: Main sales DataFrame.
        :type df: pd.DataFrame
        :return: Static features DataFrame.
        :rtype: pd.DataFrame
        """
        # Extract store and item IDs from unique_id
        series_ids = df['unique_id'].unique()
        s_df = pd.DataFrame({'unique_id': series_ids})
        s_df[['store_nbr', 'item_nbr']] = s_df['unique_id'].str.split('_', expand=True)
        s_df['store_nbr'] = s_df['store_nbr'].astype('int16')
        s_df['item_nbr'] = s_df['item_nbr'].astype('int32')

        # Load store info
        stores_path = self._data_dir / 'stores.csv'
        if stores_path.exists():
            stores_df = pd.read_csv(stores_path)
            # Encode categorical features
            for col in ['city', 'state', 'type', 'cluster']:
                if col in stores_df.columns:
                    stores_df[f'{col}_code'] = stores_df[col].astype('category').cat.codes
            s_df = s_df.merge(stores_df, on='store_nbr', how='left')
        else:
            logger.warning("stores.csv not found, skipping store features")

        # Load item info
        items_path = self._data_dir / 'items.csv'
        if items_path.exists():
            items_df = pd.read_csv(items_path)
            # Encode categorical features
            for col in ['family', 'class']:
                if col in items_df.columns:
                    items_df[f'{col}_code'] = items_df[col].astype('category').cat.codes
            s_df = s_df.merge(items_df, on='item_nbr', how='left')
        else:
            logger.warning("items.csv not found, skipping item features")

        s_df = s_df.reset_index(drop=True)
        return s_df

    def load_raw(self, filename: str) -> pd.DataFrame:
        """
        Load a raw CSV file from the Favorita dataset.

        :param filename: Name of the file (e.g., 'train.csv', 'stores.csv').
        :type filename: str
        :return: Raw DataFrame.
        :rtype: pd.DataFrame

        Example:
            >>> stores_df = dataset.load_raw('stores.csv')
            >>> print(stores_df.columns)
        """
        self.download()

        filepath = self._data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        return pd.read_csv(filepath)


# Convenience functions

def load_favorita(
    group: str = 'Favorita200',
    root_dir: str = './data'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load Favorita dataset with default settings.

    :param group: Dataset variant.
    :type group: str
    :param root_dir: Data directory.
    :type root_dir: str
    :return: Tuple of (Y_df, X_df, S_df).
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

    Example:
        >>> y_df, x_df, s_df = load_favorita('Favorita200')
    """
    dataset = FavoritaDataset(root_dir=root_dir)
    return dataset.load(group)
