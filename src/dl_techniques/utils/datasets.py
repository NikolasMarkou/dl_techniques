import numpy as np
from keras import datasets, utils
from typing import Tuple, Optional, NamedTuple

from .logger import logger


class MNISTData(NamedTuple):
    """Container for MNIST dataset splits.

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


def load_and_preprocess_mnist(
        num_classes: int = 10,
        add_channel_dim: bool = True,
        validation_split: Optional[float] = None
) -> MNISTData:
    """Load and preprocess MNIST dataset.

    Args:
        num_classes: Number of output classes for one-hot encoding
        add_channel_dim: Whether to add channel dimension for CNN input
        validation_split: Optional fraction of training data to use as validation

    Returns:
        MNISTData containing preprocessed training and test splits

    Raises:
        ValueError: If validation_split is not in range [0, 1]
        RuntimeError: If dataset loading fails
    """
    try:
        logger.info("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

        # Normalize pixel values
        logger.info("Normalizing pixel values...")
        x_train = normalize_images(x_train)
        x_test = normalize_images(x_test)

        # Add channel dimension if requested
        if add_channel_dim:
            logger.info("Adding channel dimension...")
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)

        # Convert labels to categorical
        logger.info("Converting labels to one-hot encoding...")
        y_train = utils.to_categorical(y_train, num_classes)
        y_test = utils.to_categorical(y_test, num_classes)

        # Create validation split if requested
        if validation_split is not None:
            if not 0 < validation_split < 1:
                raise ValueError(
                    "validation_split must be between 0 and 1, "
                    f"got {validation_split}"
                )

            logger.info(f"Creating validation split: {validation_split}")
            split_idx = int(len(x_train) * (1 - validation_split))
            x_val = x_train[split_idx:]
            y_val = y_train[split_idx:]
            x_train = x_train[:split_idx]
            y_train = y_train[:split_idx]

            return MNISTData(x_train, y_train, x_val, y_val)

        return MNISTData(x_train, y_train, x_test, y_test)

    except Exception as e:
        logger.error(f"Error loading or preprocessing MNIST dataset: {str(e)}")
        raise RuntimeError("Failed to load or preprocess MNIST dataset") from e


def get_data_shape(data: MNISTData) -> Tuple[Tuple[int, ...], ...]:
    """Get shapes of all arrays in the dataset.

    Args:
        data: MNISTData instance

    Returns:
        Tuple of shapes for all arrays
    """
    return tuple(arr.shape for arr in data)