from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

def count_available_files(directories: List[str],
                         extensions: List[str],
                         max_files: Optional[int] = None) -> int:
    """
    Count available files without loading them into memory.
    Used for logging and steps_per_epoch calculation.

    Args:
        directories: List of directories to search
        extensions: List of valid file extensions
        max_files: Maximum number of files to count

    Returns:
        int: Number of available files
    """
    if not directories:
        return 0

    extensions_set = set(ext.lower() for ext in extensions)
    extensions_set.update(ext.upper() for ext in extensions)

    count = 0

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue

        try:
            for file_path in dir_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in extensions_set:
                    count += 1
                    if max_files and count >= max_files:
                        return count
        except Exception as e:
            logger.error(f"Error counting files in {directory}: {e}")
            continue

    return count

# ---------------------------------------------------------------------

def image_file_generator(directories: List[str],
                        extensions: List[str],
                        max_files: Optional[int] = None,
                        patches_per_image: int = 1):
    """
    Generator that yields image file paths on-the-fly without storing them in memory.

    Args:
        directories: List of directories to search
        extensions: List of valid file extensions
        max_files: Maximum number of files to discover (None = no limit)
        patches_per_image: Number of times to yield each file path

    Yields:
        str: Image file path
    """
    if not directories:
        logger.warning("No directories provided for file discovery")
        return

    extensions_set = set(ext.lower() for ext in extensions)
    extensions_set.update(ext.upper() for ext in extensions)

    file_count = 0
    total_yielded = 0

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            continue

        logger.info(f"Processing files from: {directory}")

        try:
            for file_path in dir_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in extensions_set:
                    file_count += 1

                    # Yield this file path multiple times for patch extraction
                    for _ in range(patches_per_image):
                        yield str(file_path)
                        total_yielded += 1

                    # Check file limit
                    if max_files and file_count >= max_files:
                        logger.info(f"Reached max files limit: {max_files}")
                        logger.info(f"Total paths yielded: {total_yielded}")
                        return

                    # Progress logging
                    if file_count % 1000 == 0:
                        logger.info(f"Processed {file_count} files, yielded {total_yielded} paths...")

        except Exception as e:
            logger.error(f"Error processing files in {directory}: {e}")
            continue

    logger.info(f"Generator completed: {file_count} files processed, {total_yielded} paths yielded")

# ---------------------------------------------------------------------
