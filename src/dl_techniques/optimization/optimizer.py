"""
# ---------------------------------------------------------------------
# Optimizer and Learning Rate Schedule Builder
# ---------------------------------------------------------------------
#
# This module provides utilities for building optimizers and learning rate
# schedules for training neural networks in Keras. It offers a flexible
# configuration-based approach to setting up common optimization algorithms
# with various learning rate decay strategies.
#
# The module consists of two main components:
# 1. schedule_builder: Creates learning rate schedules with optional warmup
# 2. optimizer_builder: Creates optimizers with configured learning rate schedules
#
# Available learning rate schedules:
# - exponential_decay: Gradual exponential reduction of learning rate
# - cosine_decay: Cosine-based decay without restarts
# - cosine_decay_restarts: Cosine-based decay with periodic restarts
# - All schedules support warmup periods via WarmupSchedule wrapper
#
# Supported optimizers:
# - RMSprop: Adaptive learning rate with momentum
# - Adam: Adaptive moment estimation optimizer
# - Adadelta: Adaptive learning rate method
#
# Each optimizer supports gradient clipping options:
# - By value (clipvalue): Clip each gradient to a specific range
# - By local norm (clipnorm): Clip each gradient independently by its norm
# - By global norm (global_clipnorm): Clip all gradients by their combined norm
#
# Usage Example:
#   config = {
#       "type": "Adam",
#       "beta_1": 0.9,
#       "beta_2": 0.999,
#       "schedule": {
#           "type": "cosine_decay",
#           "warmup_steps": 1000,
#           "warmup_start_lr": 1e-6,
#           "config": {
#               "learning_rate": 1e-3,
#               "decay_steps": 10000,
#               "alpha": 0.0001
#           }
#       },
#       "gradient_clipping_by_norm": 1.0
#   }
#   optimizer, lr_schedule = optimizer_builder(config)
"""

import keras
from enum import Enum
from typing import Dict, Tuple, Union
from keras.api.optimizers import Optimizer
from keras.api.optimizers.schedules import LearningRateSchedule

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.constants import *
from dl_techniques.utils.logger import logger
from .warmup_schedule import WarmupSchedule

# ---------------------------------------------------------------------


class ScheduleType(str, Enum):
    """Enumeration of available learning rate schedule types."""
    EXPONENTIAL_DECAY = "exponential_decay"
    COSINE_DECAY = "cosine_decay"
    COSINE_DECAY_RESTARTS = "cosine_decay_restarts"


class OptimizerType(str, Enum):
    """Enumeration of available optimizer types."""
    RMSPROP = "rmsprop"
    ADAM = "adam"
    ADADELTA = "adadelta"


# Default parameter values
DEFAULT_WARMUP_STEPS = 0
DEFAULT_WARMUP_START_LR = 1e-8
DEFAULT_OPTIMIZER_TYPE = "RMSprop"

# RMSprop defaults
DEFAULT_RMSPROP_RHO = 0.9
DEFAULT_RMSPROP_MOMENTUM = 0.0
DEFAULT_RMSPROP_EPSILON = 1e-07
DEFAULT_RMSPROP_CENTERED = False

# Adam defaults
DEFAULT_ADAM_BETA_1 = 0.9
DEFAULT_ADAM_BETA_2 = 0.999
DEFAULT_ADAM_EPSILON = 1e-07
DEFAULT_ADAM_AMSGRAD = False

# Adadelta defaults
DEFAULT_ADADELTA_RHO = 0.9
DEFAULT_ADADELTA_EPSILON = 1e-07

# Cosine decay defaults
DEFAULT_COSINE_ALPHA = 0.0001

# Cosine decay restarts defaults
DEFAULT_COSINE_RESTARTS_T_MUL = 2.0
DEFAULT_COSINE_RESTARTS_M_MUL = 0.9
DEFAULT_COSINE_RESTARTS_ALPHA = 0.001


def schedule_builder(
        config: Dict[str, Union[str, Dict, int, float]]
) -> LearningRateSchedule:
    """Builds a learning rate schedule from configuration.

    Creates a Keras learning rate schedule based on configuration options,
    with optional warmup period at the beginning of training.

    Args:
        config: Configuration dictionary containing schedule parameters.
            Must include:
                - 'type': Schedule type (e.g., 'exponential_decay')
                - 'warmup_steps': Number of warmup steps (0 for no warmup)
                - 'config': Dictionary with schedule-specific parameters

    Returns:
        A Keras LearningRateSchedule instance.

    Raises:
        ValueError: If config is invalid or schedule_type is unknown.

    Example:
        >>> config = {
        ...     "type": "cosine_decay",
        ...     "warmup_steps": 1000,
        ...     "config": {
        ...         "learning_rate": 0.001,
        ...         "decay_steps": 10000,
        ...         "alpha": 0.0001
        ...     }
        ... }
        >>> lr_schedule = schedule_builder(config)
    """
    # Validate arguments
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")

    # Extract warmup parameters
    warmup_steps = config.get('warmup_steps', DEFAULT_WARMUP_STEPS)
    if warmup_steps is None:
        raise ValueError("warmup_steps must be specified in config")
    warmup_start_lr = config.get('warmup_start_lr', DEFAULT_WARMUP_START_LR)

    # Extract schedule type
    schedule_type = config.get(TYPE_STR)
    if schedule_type is None:
        raise ValueError("schedule_type cannot be None")
    if not isinstance(schedule_type, str):
        raise ValueError("schedule_type must be a string")

    # Get schedule parameters
    params = config.get(CONFIG_STR, {})
    schedule_type = schedule_type.strip().lower()

    logger.info(f"Building schedule: [{schedule_type}], with params: [{params}]")

    # Create the base learning rate schedule
    if schedule_type == ScheduleType.EXPONENTIAL_DECAY:
        decay_rate = params["decay_rate"]
        decay_steps = params["decay_steps"]
        learning_rate = params["learning_rate"]

        schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )

    elif schedule_type == ScheduleType.COSINE_DECAY_RESTARTS:
        decay_steps = params["decay_steps"]
        learning_rate = params["learning_rate"]
        t_mul = params.get("t_mul", DEFAULT_COSINE_RESTARTS_T_MUL)
        m_mul = params.get("m_mul", DEFAULT_COSINE_RESTARTS_M_MUL)
        alpha = params.get("alpha", DEFAULT_COSINE_RESTARTS_ALPHA)

        schedule = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=learning_rate,
            first_decay_steps=decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha
        )

    elif schedule_type == ScheduleType.COSINE_DECAY:
        decay_steps = params["decay_steps"]
        learning_rate = params["learning_rate"]
        alpha = params.get("alpha", DEFAULT_COSINE_ALPHA)

        schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            alpha=alpha
        )

    else:
        raise ValueError(
            f"Unknown learning_rate schedule_type: [{schedule_type}]"
        )

    # Apply warmup wrapper if warmup steps > 0
    return WarmupSchedule(
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        primary_schedule=schedule
    )


def optimizer_builder(
        config: Dict[str, Union[str, Dict, float]]
) -> Tuple[Optimizer, LearningRateSchedule]:
    """Builds an optimizer and learning rate schedule from configuration.

    Creates a Keras optimizer with the specified learning rate schedule
    and other hyperparameters based on the provided configuration.

    Args:
        config: Configuration dictionary containing optimizer parameters.
            Must include:
                - 'type': Optimizer type (e.g., 'Adam', 'RMSprop')
                - 'schedule': Dictionary with schedule configuration
            May include:
                - Optimizer-specific parameters (e.g., 'beta_1', 'beta_2')
                - Gradient clipping options

    Returns:
        A tuple containing:
            - The configured Keras optimizer
            - The learning rate schedule

    Raises:
        ValueError: If config is invalid or optimizer_type is unknown.

    Example:
        >>> config = {
        ...     "type": "Adam",
        ...     "beta_1": 0.9,
        ...     "schedule": {
        ...         "type": "cosine_decay",
        ...         "warmup_steps": 1000,
        ...         "config": {"learning_rate": 0.001, "decay_steps": 10000}
        ...     }
        ... }
        >>> optimizer, lr_schedule = optimizer_builder(config)
    """
    # Validate arguments
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")

    # Build learning rate schedule
    schedule_config = config["schedule"]
    lr_schedule = schedule_builder(config=schedule_config)

    # Extract gradient clipping configuration
    gradient_clipvalue = config.get("gradient_clipping_by_value")
    gradient_clipnorm = config.get("gradient_clipping_by_norm_local")
    gradient_global_clipnorm = config.get("gradient_clipping_by_norm")

    # Extract optimizer type and normalize
    optimizer_type = config.get("type", DEFAULT_OPTIMIZER_TYPE).strip().lower()

    # Build the appropriate optimizer
    if optimizer_type == OptimizerType.RMSPROP:
        # RMSprop optimizer configuration
        rho = config.get("rho", DEFAULT_RMSPROP_RHO)
        momentum = config.get("momentum", DEFAULT_RMSPROP_MOMENTUM)
        epsilon = config.get("epsilon", DEFAULT_RMSPROP_EPSILON)
        centered = config.get("centered", DEFAULT_RMSPROP_CENTERED)

        optimizer_parameters = dict(
            name="RMSprop",
            rho=rho,
            momentum=momentum,
            epsilon=epsilon,
            centered=centered,
            learning_rate=lr_schedule,
            clipvalue=gradient_clipvalue,
            clipnorm=gradient_clipnorm,
            global_clipnorm=gradient_global_clipnorm
        )

        optimizer = keras.optimizers.RMSprop(**optimizer_parameters)

    elif optimizer_type == OptimizerType.ADAM:
        # Adam optimizer configuration
        beta_1 = config.get("beta_1", DEFAULT_ADAM_BETA_1)
        beta_2 = config.get("beta_2", DEFAULT_ADAM_BETA_2)
        epsilon = config.get("epsilon", DEFAULT_ADAM_EPSILON)
        amsgrad = config.get("amsgrad", DEFAULT_ADAM_AMSGRAD)

        optimizer_parameters = dict(
            name="Adam",
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            learning_rate=lr_schedule,
            clipvalue=gradient_clipvalue,
            clipnorm=gradient_clipnorm,
            global_clipnorm=gradient_global_clipnorm
        )

        optimizer = keras.optimizers.Adam(**optimizer_parameters)

    elif optimizer_type == OptimizerType.ADADELTA:
        # Adadelta optimizer configuration
        rho = config.get("rho", DEFAULT_ADADELTA_RHO)
        epsilon = config.get("epsilon", DEFAULT_ADADELTA_EPSILON)
        
        optimizer_parameters = dict(
            name="Adadelta",
            rho=rho,
            epsilon=epsilon,
            learning_rate=lr_schedule,
            clipvalue=gradient_clipvalue,
            clipnorm=gradient_clipnorm,
            global_clipnorm=gradient_global_clipnorm
        )
        
        optimizer = keras.optimizers.Adadelta(**optimizer_parameters)
        
    else:
        raise ValueError(
            f"Unknown optimizer_type: [{optimizer_type}]"
        )

    return optimizer, lr_schedule