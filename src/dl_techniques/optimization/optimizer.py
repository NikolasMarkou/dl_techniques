"""
Optimizer Builder Module for Deep Learning Techniques.

This module provides functionality to create and configure various optimizers
(Adam, AdamW, RMSprop, Adadelta) with support for gradient clipping and custom
learning rate schedules.

The optimizer builder supports:
- Multiple optimizer types with configurable hyperparameters
- Gradient clipping by value, local norm, and global norm
- Integration with learning rate schedules
- Default parameter fallbacks from constants module

Usage Example:
    >>> config = {
    ...     "type": "adam",
    ...     "beta_1": 0.9,
    ...     "beta_2": 0.999,
    ...     "gradient_clipping_by_norm": 1.0
    ... }
    >>> lr_schedule = 0.001  # or a LearningRateSchedule instance
    >>> optimizer = optimizer_builder(config, lr_schedule)
"""

import keras
from enum import Enum
from typing import Dict, Union, Optional, Any

from keras.api.optimizers import Optimizer
from keras.api.optimizers.schedules import LearningRateSchedule

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# enums
# ---------------------------------------------------------------------


class OptimizerType(str, Enum):
    """Enumeration of available optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    ADADELTA = "adadelta"


# ---------------------------------------------------------------------
# main functions
# ---------------------------------------------------------------------


def optimizer_builder(
        config: Dict[str, Union[str, Dict, float, int, bool]],
        lr_schedule: Union[float, LearningRateSchedule]
) -> Optimizer:
    """Build and configure a Keras optimizer from configuration dictionary.

    Creates an optimizer instance based on the specified type and configuration,
    with support for gradient clipping and custom hyperparameters. Falls back
    to default values from constants module when parameters are not specified.

    Args:
        config: Configuration dictionary containing optimizer settings.
            Required keys:
                - type: Optimizer type ('adam', 'adamw', 'rmsprop', 'adadelta')
            Optional keys:
                - Optimizer-specific hyperparameters (beta_1, beta_2, rho, etc.)
                - gradient_clipping_by_value: Clip gradients by absolute value
                - gradient_clipping_by_norm_local: Clip gradients by local norm
                - gradient_clipping_by_norm: Clip gradients by global norm
        lr_schedule: Learning rate as float or LearningRateSchedule instance.

    Returns:
        Configured Keras optimizer instance.

    Raises:
        ValueError: If config is not a dictionary or optimizer type is unknown.

    Example:
        >>> config = {
        ...     "type": "adam",
        ...     "beta_1": 0.9,
        ...     "beta_2": 0.999,
        ...     "epsilon": 1e-7,
        ...     "gradient_clipping_by_norm": 1.0
        ... }
        >>> optimizer = optimizer_builder(config, 0.001)
    """
    # Validate arguments
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")

    # Extract and validate optimizer type
    optimizer_type = config.get("type", DEFAULT_OPTIMIZER_TYPE)
    if not isinstance(optimizer_type, str):
        raise ValueError("optimizer type must be a string")

    optimizer_type = optimizer_type.strip().lower()

    # Extract gradient clipping configuration
    gradient_clipvalue = config.get("gradient_clipping_by_value")
    gradient_clipnorm = config.get("gradient_clipping_by_norm_local")
    gradient_global_clipnorm = config.get("gradient_clipping_by_norm")

    logger.info(f"Building optimizer: [{optimizer_type}] with lr_schedule type: [{type(lr_schedule).__name__}]")

    # Build base optimizer parameters common to all optimizers
    base_params = {
        "learning_rate": lr_schedule,
        "clipvalue": gradient_clipvalue,
        "clipnorm": gradient_clipnorm,
        "global_clipnorm": gradient_global_clipnorm
    }

    # Build the appropriate optimizer
    if optimizer_type == OptimizerType.RMSPROP:
        optimizer = _build_rmsprop_optimizer(config, base_params)

    elif optimizer_type == OptimizerType.ADAM:
        optimizer = _build_adam_optimizer(config, base_params)

    elif optimizer_type == OptimizerType.ADAMW:
        optimizer = _build_adamw_optimizer(config, base_params)

    elif optimizer_type == OptimizerType.ADADELTA:
        optimizer = _build_adadelta_optimizer(config, base_params)

    else:
        raise ValueError(
            f"Unknown optimizer_type: [{optimizer_type}]. "
            f"Supported types: {[t.value for t in OptimizerType]}"
        )

    logger.info(f"Successfully built {optimizer.__class__.__name__} optimizer")
    return optimizer


# ---------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------


def _build_rmsprop_optimizer(
        config: Dict[str, Any],
        base_params: Dict[str, Any]
) -> keras.optimizers.RMSprop:
    """Build RMSprop optimizer with configuration parameters.

    Args:
        config: Configuration dictionary with RMSprop-specific parameters.
        base_params: Base parameters common to all optimizers.

    Returns:
        Configured RMSprop optimizer instance.
    """
    optimizer_params = {
        "name": "RMSprop",
        "rho": config.get("rho", DEFAULT_RMSPROP_RHO),
        "momentum": config.get("momentum", DEFAULT_RMSPROP_MOMENTUM),
        "epsilon": config.get("epsilon", DEFAULT_RMSPROP_EPSILON),
        "centered": config.get("centered", DEFAULT_RMSPROP_CENTERED),
        **base_params
    }

    return keras.optimizers.RMSprop(**optimizer_params)


def _build_adam_optimizer(
        config: Dict[str, Any],
        base_params: Dict[str, Any]
) -> keras.optimizers.Adam:
    """Build Adam optimizer with configuration parameters.

    Args:
        config: Configuration dictionary with Adam-specific parameters.
        base_params: Base parameters common to all optimizers.

    Returns:
        Configured Adam optimizer instance.
    """
    optimizer_params = {
        "name": "Adam",
        "beta_1": config.get("beta_1", DEFAULT_ADAM_BETA_1),
        "beta_2": config.get("beta_2", DEFAULT_ADAM_BETA_2),
        "epsilon": config.get("epsilon", DEFAULT_ADAM_EPSILON),
        "amsgrad": config.get("amsgrad", DEFAULT_ADAM_AMSGRAD),
        **base_params
    }

    return keras.optimizers.Adam(**optimizer_params)


def _build_adamw_optimizer(
        config: Dict[str, Any],
        base_params: Dict[str, Any]
) -> keras.optimizers.AdamW:
    """Build AdamW optimizer with configuration parameters.

    Args:
        config: Configuration dictionary with AdamW-specific parameters.
        base_params: Base parameters common to all optimizers.

    Returns:
        Configured AdamW optimizer instance.
    """
    optimizer_params = {
        "name": "AdamW",
        "beta_1": config.get("beta_1", DEFAULT_ADAMW_BETA_1),
        "beta_2": config.get("beta_2", DEFAULT_ADAMW_BETA_2),
        "epsilon": config.get("epsilon", DEFAULT_ADAMW_EPSILON),
        "amsgrad": config.get("amsgrad", DEFAULT_ADAMW_AMSGRAD),
        **base_params
    }

    return keras.optimizers.AdamW(**optimizer_params)


def _build_adadelta_optimizer(
        config: Dict[str, Any],
        base_params: Dict[str, Any]
) -> keras.optimizers.Adadelta:
    """Build Adadelta optimizer with configuration parameters.

    Args:
        config: Configuration dictionary with Adadelta-specific parameters.
        base_params: Base parameters common to all optimizers.

    Returns:
        Configured Adadelta optimizer instance.
    """
    optimizer_params = {
        "name": "Adadelta",
        "rho": config.get("rho", DEFAULT_ADADELTA_RHO),
        "epsilon": config.get("epsilon", DEFAULT_ADADELTA_EPSILON),
        **base_params
    }

    return keras.optimizers.Adadelta(**optimizer_params)