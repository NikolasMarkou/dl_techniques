"""
Time Series Dataset Utilities.

This module provides utility functions for downloading, extraction, and cache management.
"""

import shutil
import zipfile
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Callable, Optional, Union

# ---------------------------------------------------------------------
# Local Imports
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

    :param filepath: Archive path.
    :param directory: Target directory.
    :param remove_archive: Delete archive after extraction.
    :return: Target directory path.
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

    :param directory: Target directory.
    :param source_url: Download URL.
    :param decompress: Auto-extract after download.
    :return: Path to downloaded file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if filename:
        fname = filename
    else:
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
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_path(
    root_dir: Union[str, Path],
    dataset_name: str,
    filename: str
) -> Path:
    """Get standardized cache path."""
    cache_dir = Path(root_dir) / dataset_name / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / filename


def clean_cache(
    root_dir: Union[str, Path],
    dataset_name: Optional[str] = None
) -> int:
    """
    Remove cached files.

    Includes safety checks to avoid deleting system files.

    :param root_dir: Base data directory.
    :param dataset_name: Specific dataset name (optional).
    :return: Count of removed files.
    """
    root_dir = Path(root_dir).resolve()

    # Safety Check: Prevent deletion of root/system dirs
    if len(root_dir.parts) < 2:
        raise ValueError(f"Root directory {root_dir} seems unsafe to clean. Please specify a deeper path.")

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
    df: pd.DataFrame,
    required_columns: list,
    dataset_name: str = 'dataset'
) -> bool:
    """Check if DataFrame contains required columns."""
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )
    return True