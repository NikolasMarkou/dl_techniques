import numpy as np
from keras import datasets, utils
from typing import Tuple, Optional, NamedTuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

class Dataset(NamedTuple):
    """Container for Dataset split.

    Attributes:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
    """
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

# ---------------------------------------------------------------------

def normalize_images(
        images: np.ndarray,
        dtype: str = "float32",
        scale: float = 255.0
) -> np.ndarray:
    """Normalize image pixel values to [0, 1] range.

    Args:
        images: Input images array
        dtype: Data type for the normalized images
        scale: Scaling factor for normalization

    Returns:
        Normalized images array

    Raises:
        ValueError: If images array is empty or has invalid values
    """
    if images.size == 0:
        raise ValueError("Empty image array provided")

    if np.any(np.isnan(images)) or np.any(np.isinf(images)):
        raise ValueError("Image array contains NaN or Inf values")

    return images.astype(dtype) / scale

# ---------------------------------------------------------------------

def get_data_shape(data: Dataset) -> Tuple[Tuple[int, ...], ...]:
    """Get shapes of all arrays in the dataset.

    Args:
        data: MNISTData instance

    Returns:
        Tuple of shapes for all arrays
    """
    return tuple(arr.shape for arr in data)

# ---------------------------------------------------------------------
