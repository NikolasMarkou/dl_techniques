"""
Favorita Grocery Sales Forecasting Dataset.

This module provides a loader for the Corporación Favorita grocery sales
forecasting dataset, originally from a Kaggle competition. It includes
optimizations for handling the massive dataset size via chunked loading.
"""

import pandas as pd
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .base import BaseTimeSeriesDataset
from .config import TimeSeriesConfig
from .utils import download_file, extract_file, ensure_directory, get_cache_path

# ---------------------------------------------------------------------

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

    Includes optimizations for memory-safe loading of the large train.csv file.

    :param root_dir: Root directory for data storage.
    :type root_dir: str
    :param verbose: Enable verbose logging.
    :type verbose: bool
    """

    CONFIGS = FAVORITA_CONFIGS
    SOURCE_URL = 'https://www.dropbox.com/s/xi019gtvdtmsj9j/favorita-grocery-sales-forecasting2.zip?dl=1'

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
        super().__init__(root_dir=root_dir, verbose=verbose)
        self._data_dir = self.root_dir / 'favorita'

    def download(self, **kwargs) -> None:
        """Download and extract dataset files."""
        target_dir = self._data_dir
        if (target_dir / 'train.csv').exists():
            logger.info("Favorita data already downloaded and extracted")
            return

        ensure_directory(target_dir)
        logger.info(f"Downloading Favorita dataset from {self.SOURCE_URL}")
        archive_path = download_file(
            target_dir,
            self.SOURCE_URL,
            filename='favorita.zip'
        )
        extract_file(archive_path, target_dir)

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
        Load the Favorita dataset using chunked processing.

        :param group: Dataset variant (Favorita200, Favorita500, etc).
        :type group: str
        :param cache: Whether to use cached pickles.
        :type cache: bool
        :return: Tuple of (Y_df, X_df, S_df).
        :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        """
        if group not in self.CONFIGS:
            raise ValueError(f"Unknown group '{group}'. Available: {list(self.CONFIGS.keys())}")

        cache_file = get_cache_path(self._data_dir, 'cache', f'{group}.pkl')

        if cache and cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_pickle(cache_file)

        self.download()
        config = self.CONFIGS[group]

        # Determine if we need to load everything or just a subset
        full_dataset = (group == 'FavoritaComplete')
        y_df, x_df, s_df = self._process_data(config.n_ts, full_dataset=full_dataset)

        if cache:
            ensure_directory(cache_file.parent)
            pd.to_pickle((y_df, x_df, s_df), cache_file)
            logger.info(f"Cached data to {cache_file}")

        return y_df, x_df, s_df

    def _process_data(
        self,
        n_series: int,
        full_dataset: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Process raw Favorita data files using chunked reading.

        Strategy:
        1. Aggregation Pass: Read chunks, sum sales per series to find top N.
        2. Filtering Pass: Read chunks again, keeping only rows for top N series.
        """
        logger.info("Processing Favorita data (chunked mode)...")

        train_path = self._data_dir / 'train.csv'
        if not train_path.exists():
            raise FileNotFoundError("train.csv not found.")

        top_series = None

        # Pass 1: Identify top series by volume if not loading everything
        if not full_dataset:
            logger.info("Pass 1: Identifying top series by volume...")
            series_totals = {}
            chunk_size = 500_000

            for chunk in pd.read_csv(
                train_path,
                usecols=['store_nbr', 'item_nbr', 'unit_sales'],
                chunksize=chunk_size
            ):
                chunk['unique_id'] = chunk['store_nbr'].astype(str) + '_' + chunk['item_nbr'].astype(str)
                # Aggregate locally within chunk
                agg = chunk.groupby('unique_id')['unit_sales'].sum()
                for uid, val in agg.items():
                    series_totals[uid] = series_totals.get(uid, 0.0) + val

            # Sort and select top N
            sorted_series = sorted(series_totals.items(), key=lambda item: item[1], reverse=True)
            top_series = set([item[0] for item in sorted_series[:n_series]])
            logger.info(f"Identified top {n_series} series.")
        else:
            logger.info("Loading complete dataset (WARNING: High RAM usage).")

        # Pass 2: Load and Filter
        logger.info("Pass 2: Loading and filtering data...")
        df_list = []

        dtypes = {
            'id': 'int32',
            'store_nbr': 'int16',
            'item_nbr': 'int32',
            'unit_sales': 'float32',
            'onpromotion': 'object'
        }

        chunk_size = 500_000
        for chunk in pd.read_csv(
            train_path,
            parse_dates=['date'],
            dtype=dtypes,
            chunksize=chunk_size
        ):
            chunk['unique_id'] = chunk['store_nbr'].astype(str) + '_' + chunk['item_nbr'].astype(str)

            if top_series is not None:
                chunk = chunk[chunk['unique_id'].isin(top_series)]

            if not chunk.empty:
                df_list.append(chunk)

        df = pd.concat(df_list, ignore_index=True)

        # Create Y_df
        y_df = df[['unique_id', 'date', 'unit_sales']].copy()
        y_df.columns = ['unique_id', 'ds', 'y']
        y_df['y'] = y_df['y'].clip(lower=0)
        y_df = y_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

        x_df = self._create_temporal_features(df)
        s_df = self._create_static_features(df)

        logger.info(f"Processed {y_df['unique_id'].nunique()} series, {len(y_df)} observations")
        return y_df, x_df, s_df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features (holidays, oil price)."""
        x_df = df[['unique_id', 'date', 'onpromotion']].copy()
        x_df.columns = ['unique_id', 'ds', 'onpromotion']

        x_df['onpromotion'] = x_df['onpromotion'].map(
            {'True': 1.0, 'False': 0.0, True: 1.0, False: 0.0}
        ).fillna(0.0).astype('float32')

        oil_path = self._data_dir / 'oil.csv'
        if oil_path.exists():
            oil_df = pd.read_csv(oil_path, parse_dates=['date'])
            oil_df = oil_df.rename(columns={'date': 'ds', 'dcoilwtico': 'oil_price'})
            oil_df['oil_price'] = oil_df['oil_price'].ffill().bfill()
            x_df = x_df.merge(oil_df[['ds', 'oil_price']], on='ds', how='left')
            x_df['oil_price'] = x_df['oil_price'].ffill().bfill().astype('float32')

        holidays_path = self._data_dir / 'holidays_events.csv'
        if holidays_path.exists():
            holidays_df = pd.read_csv(holidays_path, parse_dates=['date'])
            holidays_df = holidays_df[holidays_df['transferred'] == False]
            holidays_df['is_holiday'] = 1.0
            holidays_df = holidays_df[['date', 'is_holiday']].drop_duplicates()
            holidays_df.columns = ['ds', 'is_holiday']
            x_df = x_df.merge(holidays_df, on='ds', how='left')
            x_df['is_holiday'] = x_df['is_holiday'].fillna(0.0).astype('float32')

        x_df = x_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        return x_df

    def _create_static_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract static features (store, item info)."""
        series_ids = df['unique_id'].unique()
        s_df = pd.DataFrame({'unique_id': series_ids})
        s_df[['store_nbr', 'item_nbr']] = s_df['unique_id'].str.split('_', expand=True)
        s_df['store_nbr'] = s_df['store_nbr'].astype('int16')
        s_df['item_nbr'] = s_df['item_nbr'].astype('int32')

        stores_path = self._data_dir / 'stores.csv'
        if stores_path.exists():
            stores_df = pd.read_csv(stores_path)
            for col in ['city', 'state', 'type', 'cluster']:
                if col in stores_df.columns:
                    stores_df[f'{col}_code'] = stores_df[col].astype('category').cat.codes
            s_df = s_df.merge(stores_df, on='store_nbr', how='left')

        items_path = self._data_dir / 'items.csv'
        if items_path.exists():
            items_df = pd.read_csv(items_path)
            for col in ['family', 'class']:
                if col in items_df.columns:
                    items_df[f'{col}_code'] = items_df[col].astype('category').cat.codes
            s_df = s_df.merge(items_df, on='item_nbr', how='left')

        s_df = s_df.reset_index(drop=True)
        return s_df

    def load_raw(self, filename: str) -> pd.DataFrame:
        """Load a raw CSV file directly."""
        self.download()
        filepath = self._data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return pd.read_csv(filepath)


def load_favorita(
    group: str = 'Favorita200',
    root_dir: str = './data'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to load Favorita data."""
    dataset = FavoritaDataset(root_dir=root_dir)
    return dataset.load(group)