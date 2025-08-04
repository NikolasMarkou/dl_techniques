import numpy as np
from typing import Optional
from keras import datasets, utils


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .common import normalize_images, Dataset

# ---------------------------------------------------------------------

CIFAR10Data = Dataset

# ---------------------------------------------------------------------


def load_and_preprocess_cifar10(
        num_classes: int = 10,
        add_channel_dim: bool = False,
        validation_split: Optional[float] = None
) -> CIFAR10Data:
    """Load and preprocess CIFAR10 dataset.

    Args:
        num_classes: Number of output classes for one-hot encoding
        add_channel_dim: Whether to add channel dimension for CNN input
        validation_split: Optional fraction of training data to use as validation

    Returns:
        CIFAR10Data containing preprocessed training and test splits

    Raises:
        ValueError: If validation_split is not in range [0, 1]
        RuntimeError: If dataset loading fails
    """
    try:
        logger.info("Loading CIFAR10 dataset...")
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

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

            return CIFAR10Data(x_train, y_train, x_val, y_val)

        return CIFAR10Data(x_train, y_train, x_test, y_test)

    except Exception as e:
        logger.error(f"Error loading or preprocessing CIFAR10Data dataset: {str(e)}")
        raise RuntimeError("Failed to load or preprocess CIFAR10Data dataset") from e

# ---------------------------------------------------------------------

