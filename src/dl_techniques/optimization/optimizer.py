"""
Optimizer and Learning Rate Schedule Builder Module for Deep Learning Techniques.

This module provides utilities for building optimizers and learning rate schedules
for training neural networks in Keras. It offers a flexible configuration-based
approach to setting up common optimization algorithms with various learning rate
decay strategies.

The module consists of two main components:
1. learning_rate_schedule_builder: Creates learning rate schedules with optional warmup
2. optimizer_builder: Creates optimizers with configured learning rate schedules

Available learning rate schedules:
- exponential_decay: Gradual exponential reduction of learning rate
- cosine_decay: Cosine-based decay without restarts
- cosine_decay_restarts: Cosine-based decay with periodic restarts
- All schedules support warmup periods via WarmupSchedule wrapper

Supported optimizers:
- Adam: Adaptive moment estimation optimizer
- AdamW: Adam with decoupled weight decay
- RMSprop: Adaptive learning rate with momentum
- Adadelta: Adaptive learning rate method

Each optimizer supports gradient clipping options:
- By value (clipvalue): Clip each gradient to a specific range
- By local norm (clipnorm): Clip each gradient independently by its norm
- By global norm (global_clipnorm): Clip all gradients by their combined norm
"""

import keras
from enum import Enum
from typing import Dict, Union, Any

from keras.api.optimizers import Optimizer
from keras.api.optimizers.schedules import LearningRateSchedule

from dl_techniques.utils.logger import logger
from .constants import *
from .warmup_schedule import WarmupSchedule


# ---------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------


class ScheduleType(str, Enum):
    """Enumeration of available learning rate schedule types."""
    EXPONENTIAL_DECAY = "exponential_decay"
    COSINE_DECAY = "cosine_decay"
    COSINE_DECAY_RESTARTS = "cosine_decay_restarts"


class OptimizerType(str, Enum):
    """Enumeration of available optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    ADADELTA = "adadelta"


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Warmup defaults
DEFAULT_WARMUP_STEPS = 0
DEFAULT_WARMUP_START_LR = 1e-8

# RMSprop defaults
DEFAULT_RMSPROP_RHO = 0.9
DEFAULT_RMSPROP_MOMENTUM = 0.0
DEFAULT_RMSPROP_EPSILON = 1e-7
DEFAULT_RMSPROP_CENTERED = False

# Adam defaults
DEFAULT_ADAM_BETA_1 = 0.9
DEFAULT_ADAM_BETA_2 = 0.999
DEFAULT_ADAM_EPSILON = 1e-7
DEFAULT_ADAM_AMSGRAD = False

# AdamW defaults
DEFAULT_ADAMW_BETA_1 = 0.9
DEFAULT_ADAMW_BETA_2 = 0.999
DEFAULT_ADAMW_EPSILON = 1e-7
DEFAULT_ADAMW_AMSGRAD = False

# Adadelta defaults
DEFAULT_ADADELTA_RHO = 0.9
DEFAULT_ADADELTA_EPSILON = 1e-7

# Schedule defaults
DEFAULT_COSINE_ALPHA = 0.0001
DEFAULT_COSINE_RESTARTS_T_MUL = 2.0
DEFAULT_COSINE_RESTARTS_M_MUL = 0.9
DEFAULT_COSINE_RESTARTS_ALPHA = 0.001


# ---------------------------------------------------------------------
# Main Functions
# ---------------------------------------------------------------------


def learning_rate_schedule_builder(config: Dict[str, Any]) -> LearningRateSchedule:
    """Build a learning rate schedule from configuration.

    Creates a Keras learning rate schedule based on configuration options,
    with optional warmup period at the beginning of training.

    Args:
        config: Configuration dictionary containing schedule parameters.
            Required keys:
                - type: Schedule type ('exponential_decay', 'cosine_decay', 'cosine_decay_restarts')
                - learning_rate: Initial learning rate
                - decay_steps: Steps over which to decay
            Optional keys:
                - warmup_steps: Number of warmup steps (default: 0)
                - warmup_start_lr: Starting learning rate for warmup (default: 1e-8)
                - Other schedule-specific parameters

    Returns:
        A Keras LearningRateSchedule instance.

    Raises:
        ValueError: If config is invalid or schedule type is unknown.

    Example:
        >>> config = {
        ...     "type": "cosine_decay",
        ...     "warmup_steps": 1000,
        ...     "warmup_start_lr": 1e-8,
        ...     "learning_rate": 0.001,
        ...     "decay_steps": 10000,
        ...     "alpha": 0.0001
        ... }
        >>> lr_schedule = learning_rate_schedule_builder(config)
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")

    schedule_type = config.get("type")
    if not schedule_type:
        raise ValueError("schedule type must be specified in config")

    schedule_type = schedule_type.strip().lower()

    # Extract common parameters
    learning_rate = config.get("learning_rate")
    if learning_rate is None:
        raise ValueError("learning_rate must be specified in config")

    decay_steps = config.get("decay_steps")
    if decay_steps is None:
        raise ValueError("decay_steps must be specified in config")

    # Extract warmup parameters
    warmup_steps = config.get("warmup_steps", DEFAULT_WARMUP_STEPS)
    warmup_start_lr = config.get("warmup_start_lr", DEFAULT_WARMUP_START_LR)

    logger.info(f"Building schedule: [{schedule_type}] with warmup_steps: {warmup_steps}")

    # Create the base learning rate schedule
    if schedule_type == ScheduleType.EXPONENTIAL_DECAY:
        decay_rate = config.get("decay_rate")
        if decay_rate is None:
            raise ValueError("decay_rate must be specified for exponential_decay")

        schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )

    elif schedule_type == ScheduleType.COSINE_DECAY_RESTARTS:
        t_mul = config.get("t_mul", DEFAULT_COSINE_RESTARTS_T_MUL)
        m_mul = config.get("m_mul", DEFAULT_COSINE_RESTARTS_M_MUL)
        alpha = config.get("alpha", DEFAULT_COSINE_RESTARTS_ALPHA)

        schedule = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=learning_rate,
            first_decay_steps=decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha
        )

    elif schedule_type == ScheduleType.COSINE_DECAY:
        alpha = config.get("alpha", DEFAULT_COSINE_ALPHA)

        schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            alpha=alpha
        )

    else:
        raise ValueError(
            f"Unknown learning_rate schedule_type: [{schedule_type}]. "
            f"Supported types: {[t.value for t in ScheduleType]}"
        )

    # Apply warmup wrapper if warmup steps > 0
    if warmup_steps > 0:
        schedule = WarmupSchedule(
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            primary_schedule=schedule
        )

    return schedule

# ---------------------------------------------------------------------

def optimizer_builder(
        config: Dict[str, Any],
        lr_schedule: Union[float, LearningRateSchedule]
) -> Optimizer:
    """Build and configure a Keras optimizer from configuration dictionary.

    Creates an optimizer instance based on the specified type and configuration,
    with support for gradient clipping and custom hyperparameters. Falls back
    to default values from constants when parameters are not specified.

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
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")

    # Extract and validate optimizer type
    optimizer_type = config.get("type")
    if not optimizer_type:
        raise ValueError("optimizer type must be specified in config")

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
# Helper Functions
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