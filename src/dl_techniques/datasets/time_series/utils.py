"""
Time Series Dataset Utilities.

This module provides utility functions for downloading, extracting,
and managing time series dataset files with progress tracking.

Example:
    >>> from dl_techniques.datasets.time_series.utils import (
    ...     download_file, extract_file
    ... )
    >>> filepath = download_file('./data', 'https://example.com/data.zip')
    >>> extract_file(filepath, './data/extracted')
"""

import logging
import shutil
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

def extract_file(
    filepath: Union[str, Path],
    directory: Union[str, Path],
    remove_archive: bool = False
) -> Path:
    """
    Extract a compressed archive to a specified directory.

    Supports zip files natively and delegates other formats to
    shutil.unpack_archive for broader compatibility.

    :param filepath: Path to the compressed archive file.
    :type filepath: Union[str, Path]
    :param directory: Target directory for extraction.
    :type directory: Union[str, Path]
    :param remove_archive: Whether to delete the archive after extraction.
    :type remove_archive: bool
    :return: Path to the extraction directory.
    :rtype: Path
    :raises FileNotFoundError: If the archive file does not exist.
    :raises zipfile.BadZipFile: If the zip file is corrupted.

    Example:
        >>> extract_file('./data/dataset.zip', './data/extracted')
        PosixPath('./data/extracted')
    """
    filepath = Path(filepath)
    directory = Path(directory)

    if not filepath.exists():
        raise FileNotFoundError(f"Archive not found: {filepath}")

    directory.mkdir(parents=True, exist_ok=True)

    if filepath.suffix == '.zip':
        logger.info(f"Decompressing {filepath.name} to {directory}")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(directory)
    else:
        logger.info(f"Extracting {filepath.name} using shutil")
        shutil.unpack_archive(str(filepath), str(directory))

    logger.info(f"Successfully extracted to {directory}")

    if remove_archive:
        filepath.unlink()
        logger.info(f"Removed archive: {filepath}")

    return directory


def download_file(
    directory: Union[str, Path],
    source_url: str,
    decompress: bool = False,
    filename: Optional[str] = None,
    chunk_size: int = 8192,
    timeout: int = 30,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Path:
    """
    Download a file from a URL with progress tracking.

    Downloads a file to the specified directory with optional automatic
    decompression. Skips download if the file already exists.

    :param directory: Target directory for the downloaded file.
    :type directory: Union[str, Path]
    :param source_url: URL to download from.
    :type source_url: str
    :param decompress: Whether to extract the file after downloading.
    :type decompress: bool
    :param filename: Optional filename override. If None, extracted from URL.
    :type filename: Optional[str]
    :param chunk_size: Size of download chunks in bytes.
    :type chunk_size: int
    :param timeout: Request timeout in seconds.
    :type timeout: int
    :param progress_callback: Optional callback function receiving
        (bytes_downloaded, total_bytes).
    :type progress_callback: Optional[Callable[[int, int], None]]
    :return: Path to the downloaded file.
    :rtype: Path
    :raises requests.RequestException: If the download fails.
    :raises ValueError: If the URL is invalid or inaccessible.

    Example:
        >>> download_file('./data', 'https://example.com/dataset.zip', decompress=True)
        PosixPath('./data/dataset.zip')
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if filename:
        fname = filename
    else:
        # Extract filename from URL, handling query parameters
        fname = source_url.split('/')[-1].split('?')[0]

    filepath = directory / fname

    if filepath.exists():
        logger.info(f"File {fname} already exists in {directory}")
        if decompress:
            extract_file(filepath, directory)
        return filepath

    logger.info(f"Downloading {fname} from {source_url}")

    try:
        response = requests.get(source_url, stream=True, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to download from {source_url}: {e}") from e

    total_size = int(response.headers.get('content-length', 0))
    downloaded_bytes = 0

    with tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        desc=fname,
        disable=total_size == 0
    ) as progress_bar:
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    progress_bar.update(len(chunk))
                    if progress_callback:
                        progress_callback(downloaded_bytes, total_size)

    logger.info(f"Successfully downloaded {fname}")

    if decompress:
        extract_file(filepath, directory)

    return filepath


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    :param path: Path to the directory.
    :type path: Union[str, Path]
    :return: Path object for the directory.
    :rtype: Path

    Example:
        >>> ensure_directory('./data/cache')
        PosixPath('./data/cache')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_path(
    root_dir: Union[str, Path],
    dataset_name: str,
    filename: str
) -> Path:
    """
    Get the standardized cache file path for a dataset.

    :param root_dir: Root data directory.
    :type root_dir: Union[str, Path]
    :param dataset_name: Name of the dataset.
    :type dataset_name: str
    :param filename: Cache filename (e.g., 'ETTh1.pkl').
    :type filename: str
    :return: Full path to the cache file.
    :rtype: Path

    Example:
        >>> get_cache_path('./data', 'longhorizon', 'ETTh1.pkl')
        PosixPath('./data/longhorizon/cache/ETTh1.pkl')
    """
    cache_dir = Path(root_dir) / dataset_name / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / filename


def clean_cache(
    root_dir: Union[str, Path],
    dataset_name: Optional[str] = None
) -> int:
    """
    Remove cached files for a dataset or all datasets.

    :param root_dir: Root data directory.
    :type root_dir: Union[str, Path]
    :param dataset_name: Specific dataset to clean. If None, cleans all.
    :type dataset_name: Optional[str]
    :return: Number of files removed.
    :rtype: int

    Example:
        >>> clean_cache('./data', 'longhorizon')
        5
    """
    root_dir = Path(root_dir)
    files_removed = 0

    if dataset_name:
        cache_dirs = [root_dir / dataset_name / 'cache']
    else:
        cache_dirs = list(root_dir.glob('*/cache'))

    for cache_dir in cache_dirs:
        if cache_dir.exists():
            for cache_file in cache_dir.iterdir():
                if cache_file.is_file():
                    cache_file.unlink()
                    files_removed += 1
                    logger.debug(f"Removed cache file: {cache_file}")

    logger.info(f"Cleaned {files_removed} cached files")
    return files_removed


def validate_dataframe_schema(
    df,
    required_columns: list,
    dataset_name: str = 'dataset'
) -> bool:
    """
    Validate that a DataFrame contains required columns.

    :param df: DataFrame to validate.
    :type df: pandas.DataFrame
    :param required_columns: List of required column names.
    :type required_columns: list
    :param dataset_name: Name for error messages.
    :type dataset_name: str
    :return: True if validation passes.
    :rtype: bool
    :raises ValueError: If required columns are missing.

    Example:
        >>> validate_dataframe_schema(df, ['unique_id', 'ds', 'y'], 'ETTh1')
        True
    """
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )
    return True
