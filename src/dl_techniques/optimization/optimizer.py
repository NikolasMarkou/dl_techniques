"""
<<<<<<< HEAD
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
=======
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
# - AdamW: Adam with decoupled weight decay
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
>>>>>>> a543855 ([optimization] cleaning up and refactoring)
"""

import keras
from enum import Enum
<<<<<<< HEAD
from typing import Dict, Union, Optional, Any

=======
from typing import Dict, Tuple, Union, Any, Optional, Callable
>>>>>>> a543855 ([optimization] cleaning up and refactoring)
from keras.api.optimizers import Optimizer
from keras.api.optimizers.schedules import LearningRateSchedule

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from .constants import *
from dl_techniques.utils.logger import logger



# ---------------------------------------------------------------------
# Default Parameters
# ---------------------------------------------------------------------
<<<<<<< HEAD
# enums
# ---------------------------------------------------------------------
=======

# General defaults
DEFAULT_WARMUP_STEPS: int = 0
DEFAULT_WARMUP_START_LR: float = 1e-8
DEFAULT_OPTIMIZER_TYPE: str = "RMSprop"

# RMSprop defaults
DEFAULT_RMSPROP_RHO: float = 0.9
DEFAULT_RMSPROP_MOMENTUM: float = 0.0
DEFAULT_RMSPROP_EPSILON: float = 1e-07
DEFAULT_RMSPROP_CENTERED: bool = False

# Adam defaults
DEFAULT_ADAM_BETA_1: float = 0.9
DEFAULT_ADAM_BETA_2: float = 0.999
DEFAULT_ADAM_EPSILON: float = 1e-07
DEFAULT_ADAM_AMSGRAD: bool = False

# AdamW defaults
DEFAULT_ADAMW_BETA_1: float = 0.9
DEFAULT_ADAMW_BETA_2: float = 0.999
DEFAULT_ADAMW_EPSILON: float = 1e-07
DEFAULT_ADAMW_AMSGRAD: bool = False

# Adadelta defaults
DEFAULT_ADADELTA_RHO: float = 0.9
DEFAULT_ADADELTA_EPSILON: float = 1e-07

# Learning rate schedule defaults
DEFAULT_COSINE_ALPHA: float = 0.0001
DEFAULT_COSINE_RESTARTS_T_MUL: float = 2.0
DEFAULT_COSINE_RESTARTS_M_MUL: float = 0.9
DEFAULT_COSINE_RESTARTS_ALPHA: float = 0.001


# ---------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------

class ScheduleType(str, Enum):
    """Enumeration of available learning rate schedule types.

    Attributes:
        EXPONENTIAL_DECAY: Exponential decay schedule
        COSINE_DECAY: Cosine decay schedule without restarts
        COSINE_DECAY_RESTARTS: Cosine decay schedule with periodic restarts
    """
    EXPONENTIAL_DECAY = "exponential_decay"
    COSINE_DECAY = "cosine_decay"
    COSINE_DECAY_RESTARTS = "cosine_decay_restarts"
>>>>>>> a543855 ([optimization] cleaning up and refactoring)

# ---------------------------------------------------------------------

class OptimizerType(str, Enum):
<<<<<<< HEAD
    """Enumeration of available optimizer types."""
=======
    """Enumeration of available optimizer types.

    Attributes:
        ADAM: Adam optimizer
        ADAMW: AdamW optimizer with decoupled weight decay
        RMSPROP: RMSprop optimizer
        ADADELTA: Adadelta optimizer
    """
>>>>>>> a543855 ([optimization] cleaning up and refactoring)
    ADAM = "adam"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    ADADELTA = "adadelta"


# ---------------------------------------------------------------------
<<<<<<< HEAD
# main functions
# ---------------------------------------------------------------------


def optimizer_builder(
        config: Dict[str, Union[str, Dict, float, int, bool]],
        lr_schedule: Union[float, LearningRateSchedule]
) -> Optimizer:
    """Build and configure a Keras optimizer from configuration dictionary.
=======
# Helper Functions
# ---------------------------------------------------------------------

def _validate_schedule_config(config: Dict[str, Union[str, Dict, int, float]]) -> None:
    """Validate schedule configuration parameters.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValueError("Schedule config must be a dictionary")

    if TYPE_STR not in config:
        raise ValueError("Schedule config must include 'type' field")

    if not isinstance(config[TYPE_STR], str):
        raise ValueError("Schedule type must be a string")

    if CONFIG_STR not in config:
        raise ValueError("Schedule config must include 'config' field")

    if not isinstance(config[CONFIG_STR], dict):
        raise ValueError("Schedule 'config' field must be a dictionary")


def _validate_optimizer_config(config: Dict[str, Union[str, Dict, float]]) -> None:
    """Validate optimizer configuration parameters.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValueError("Optimizer config must be a dictionary")

    if "schedule" not in config:
        raise ValueError("Optimizer config must include 'schedule' field")


def _create_exponential_decay_schedule(params: Dict[str, Any]) -> keras.optimizers.schedules.ExponentialDecay:
    """Create exponential decay learning rate schedule.

    Args:
        params: Schedule parameters containing 'decay_rate', 'decay_steps', 'learning_rate'

    Returns:
        Exponential decay schedule

    Raises:
        KeyError: If required parameters are missing
    """
    required_params = ["decay_rate", "decay_steps", "learning_rate"]
    for param in required_params:
        if param not in params:
            raise KeyError(f"Missing required parameter for exponential decay: {param}")

    return keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=params["learning_rate"],
        decay_steps=params["decay_steps"],
        decay_rate=params["decay_rate"]
    )


def _create_cosine_decay_schedule(params: Dict[str, Any]) -> keras.optimizers.schedules.CosineDecay:
    """Create cosine decay learning rate schedule.

    Args:
        params: Schedule parameters containing 'decay_steps', 'learning_rate', optional 'alpha'

    Returns:
        Cosine decay schedule

    Raises:
        KeyError: If required parameters are missing
    """
    required_params = ["decay_steps", "learning_rate"]
    for param in required_params:
        if param not in params:
            raise KeyError(f"Missing required parameter for cosine decay: {param}")

    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=params["learning_rate"],
        decay_steps=params["decay_steps"],
        alpha=params.get("alpha", DEFAULT_COSINE_ALPHA)
    )


def _create_cosine_decay_restarts_schedule(params: Dict[str, Any]) -> keras.optimizers.schedules.CosineDecayRestarts:
    """Create cosine decay with restarts learning rate schedule.

    Args:
        params: Schedule parameters containing 'decay_steps', 'learning_rate',
                optional 't_mul', 'm_mul', 'alpha'

    Returns:
        Cosine decay restarts schedule

    Raises:
        KeyError: If required parameters are missing
    """
    required_params = ["decay_steps", "learning_rate"]
    for param in required_params:
        if param not in params:
            raise KeyError(f"Missing required parameter for cosine decay restarts: {param}")

    return keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=params["learning_rate"],
        first_decay_steps=params["decay_steps"],
        t_mul=params.get("t_mul", DEFAULT_COSINE_RESTARTS_T_MUL),
        m_mul=params.get("m_mul", DEFAULT_COSINE_RESTARTS_M_MUL),
        alpha=params.get("alpha", DEFAULT_COSINE_RESTARTS_ALPHA)
    )


def _extract_gradient_clipping_params(config: Dict[str, Union[str, Dict, float]]) -> Dict[str, Optional[float]]:
    """Extract gradient clipping parameters from config.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with gradient clipping parameters (clipvalue, clipnorm, global_clipnorm)
    """
    return {
        "clipvalue": config.get("gradient_clipping_by_value"),
        "clipnorm": config.get("gradient_clipping_by_norm_local"),
        "global_clipnorm": config.get("gradient_clipping_by_norm")
    }


def _create_rmsprop_optimizer(
    config: Dict[str, Union[str, Dict, float]],
    lr_schedule: LearningRateSchedule,
    clipping_params: Dict[str, Optional[float]]
) -> keras.optimizers.RMSprop:
    """Create RMSprop optimizer with configuration.

    Args:
        config: Optimizer configuration
        lr_schedule: Learning rate schedule
        clipping_params: Gradient clipping parameters

    Returns:
        Configured RMSprop optimizer
    """
    return keras.optimizers.RMSprop(
        name="RMSprop",
        rho=config.get("rho", DEFAULT_RMSPROP_RHO),
        momentum=config.get("momentum", DEFAULT_RMSPROP_MOMENTUM),
        epsilon=config.get("epsilon", DEFAULT_RMSPROP_EPSILON),
        centered=config.get("centered", DEFAULT_RMSPROP_CENTERED),
        learning_rate=lr_schedule,
        **{k: v for k, v in clipping_params.items() if v is not None}
    )


def _create_adam_optimizer(
    config: Dict[str, Union[str, Dict, float]],
    lr_schedule: LearningRateSchedule,
    clipping_params: Dict[str, Optional[float]]
) -> keras.optimizers.Adam:
    """Create Adam optimizer with configuration.

    Args:
        config: Optimizer configuration
        lr_schedule: Learning rate schedule
        clipping_params: Gradient clipping parameters

    Returns:
        Configured Adam optimizer
    """
    return keras.optimizers.Adam(
        name="Adam",
        beta_1=config.get("beta_1", DEFAULT_ADAM_BETA_1),
        beta_2=config.get("beta_2", DEFAULT_ADAM_BETA_2),
        epsilon=config.get("epsilon", DEFAULT_ADAM_EPSILON),
        amsgrad=config.get("amsgrad", DEFAULT_ADAM_AMSGRAD),
        learning_rate=lr_schedule,
        **{k: v for k, v in clipping_params.items() if v is not None}
    )


def _create_adamw_optimizer(
    config: Dict[str, Union[str, Dict, float]],
    lr_schedule: LearningRateSchedule,
    clipping_params: Dict[str, Optional[float]]
) -> keras.optimizers.AdamW:
    """Create AdamW optimizer with configuration.

    Args:
        config: Optimizer configuration
        lr_schedule: Learning rate schedule
        clipping_params: Gradient clipping parameters

    Returns:
        Configured AdamW optimizer
    """
    return keras.optimizers.AdamW(
        name="AdamW",
        beta_1=config.get("beta_1", DEFAULT_ADAMW_BETA_1),
        beta_2=config.get("beta_2", DEFAULT_ADAMW_BETA_2),
        epsilon=config.get("epsilon", DEFAULT_ADAMW_EPSILON),
        amsgrad=config.get("amsgrad", DEFAULT_ADAMW_AMSGRAD),
        learning_rate=lr_schedule,
        **{k: v for k, v in clipping_params.items() if v is not None}
    )


def _create_adadelta_optimizer(
    config: Dict[str, Union[str, Dict, float]],
    lr_schedule: LearningRateSchedule,
    clipping_params: Dict[str, Optional[float]]
) -> keras.optimizers.Adadelta:
    """Create Adadelta optimizer with configuration.

    Args:
        config: Optimizer configuration
        lr_schedule: Learning rate schedule
        clipping_params: Gradient clipping parameters

    Returns:
        Configured Adadelta optimizer
    """
    return keras.optimizers.Adadelta(
        name="Adadelta",
        rho=config.get("rho", DEFAULT_ADADELTA_RHO),
        epsilon=config.get("epsilon", DEFAULT_ADADELTA_EPSILON),
        learning_rate=lr_schedule,
        **{k: v for k, v in clipping_params.items() if v is not None}
    )


# ---------------------------------------------------------------------
# Main Functions
# ---------------------------------------------------------------------

def schedule_builder(
    config: Dict[str, Union[str, Dict, int, float]]
) -> LearningRateSchedule:
    """Build a learning rate schedule from configuration.

    Creates a Keras learning rate schedule based on configuration options,
    with optional warmup period at the beginning of training.

    Args:
        config: Configuration dictionary containing schedule parameters.
            Must include:
                - 'type': Schedule type ('exponential_decay', 'cosine_decay', 'cosine_decay_restarts')
                - 'warmup_steps': Number of warmup steps (0 for no warmup)
                - 'config': Dictionary with schedule-specific parameters
            Optional:
                - 'warmup_start_lr': Starting learning rate for warmup (default: 1e-8)

    Returns:
        A Keras LearningRateSchedule instance wrapped with warmup if specified.

    Raises:
        ValueError: If config is invalid or schedule_type is unknown.
        KeyError: If required parameters are missing from schedule config.

    Example:
        >>> config = {
        ...     "type": "cosine_decay",
        ...     "warmup_steps": 1000,
        ...     "warmup_start_lr": 1e-6,
        ...     "config": {
        ...         "learning_rate": 0.001,
        ...         "decay_steps": 10000,
        ...         "alpha": 0.0001
        ...     }
        ... }
        >>> lr_schedule = schedule_builder(config)
    """
    # Validate configuration
    _validate_schedule_config(config)

    # Extract parameters
    warmup_steps = config.get('warmup_steps', DEFAULT_WARMUP_STEPS)
    warmup_start_lr = config.get('warmup_start_lr', DEFAULT_WARMUP_START_LR)
    schedule_type = config[TYPE_STR].strip().lower()
    params = config[CONFIG_STR]

    logger.info(f"Building schedule: [{schedule_type}], with params: [{params}]")

    # Schedule factory mapping
    schedule_creators: Dict[str, Callable[[Dict[str, Any]], LearningRateSchedule]] = {
        ScheduleType.EXPONENTIAL_DECAY: _create_exponential_decay_schedule,
        ScheduleType.COSINE_DECAY: _create_cosine_decay_schedule,
        ScheduleType.COSINE_DECAY_RESTARTS: _create_cosine_decay_restarts_schedule,
    }

    if schedule_type not in schedule_creators:
        available_types = list(schedule_creators.keys())
        raise ValueError(f"Unknown learning rate schedule type: [{schedule_type}]. "
                        f"Available types: {available_types}")

    # Create the base learning rate schedule
    base_schedule = schedule_creators[schedule_type](params)

    # Apply warmup wrapper
    return WarmupSchedule(
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        primary_schedule=base_schedule
    )


def optimizer_builder(
    config: Dict[str, Union[str, Dict, float]]
) -> Tuple[Optimizer, LearningRateSchedule]:
    """Build an optimizer and learning rate schedule from configuration.
>>>>>>> a543855 ([optimization] cleaning up and refactoring)

    Creates an optimizer instance based on the specified type and configuration,
    with support for gradient clipping and custom hyperparameters. Falls back
    to default values from constants module when parameters are not specified.

    Args:
<<<<<<< HEAD
        config: Configuration dictionary containing optimizer settings.
            Required keys:
                - type: Optimizer type ('adam', 'adamw', 'rmsprop', 'adadelta')
            Optional keys:
                - Optimizer-specific hyperparameters (beta_1, beta_2, rho, etc.)
                - gradient_clipping_by_value: Clip gradients by absolute value
                - gradient_clipping_by_norm_local: Clip gradients by local norm
                - gradient_clipping_by_norm: Clip gradients by global norm
        lr_schedule: Learning rate as float or LearningRateSchedule instance.
=======
        config: Configuration dictionary containing optimizer parameters.
            Must include:
                - 'type': Optimizer type ('adam', 'adamw', 'rmsprop', 'adadelta')
                - 'schedule': Dictionary with schedule configuration
            Optional:
                - Optimizer-specific parameters (e.g., 'beta_1', 'beta_2' for Adam)
                - 'gradient_clipping_by_value': Clip gradients by value
                - 'gradient_clipping_by_norm_local': Clip gradients by local norm
                - 'gradient_clipping_by_norm': Clip gradients by global norm
>>>>>>> a543855 ([optimization] cleaning up and refactoring)

    Returns:
        Configured Keras optimizer instance.

    Raises:
        ValueError: If config is not a dictionary or optimizer type is unknown.

    Example:
        >>> config = {
        ...     "type": "adam",
        ...     "beta_1": 0.9,
        ...     "beta_2": 0.999,
<<<<<<< HEAD
        ...     "epsilon": 1e-7,
=======
        ...     "schedule": {
        ...         "type": "cosine_decay",
        ...         "warmup_steps": 1000,
        ...         "config": {"learning_rate": 0.001, "decay_steps": 10000}
        ...     },
>>>>>>> a543855 ([optimization] cleaning up and refactoring)
        ...     "gradient_clipping_by_norm": 1.0
        ... }
        >>> optimizer = optimizer_builder(config, 0.001)
    """
    # Validate configuration
    _validate_optimizer_config(config)

<<<<<<< HEAD
    # Extract and validate optimizer type
    optimizer_type = config.get("type", DEFAULT_OPTIMIZER_TYPE)
    if not isinstance(optimizer_type, str):
        raise ValueError("optimizer type must be a string")

    optimizer_type = optimizer_type.strip().lower()
=======
    # Build learning rate schedule
    lr_schedule = schedule_builder(config["schedule"])
>>>>>>> a543855 ([optimization] cleaning up and refactoring)

    # Extract gradient clipping parameters
    clipping_params = _extract_gradient_clipping_params(config)

<<<<<<< HEAD
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
=======
    # Extract and normalize optimizer type
    optimizer_type = config.get("type", DEFAULT_OPTIMIZER_TYPE).strip().lower()

    logger.info(f"Building optimizer: [{optimizer_type}] with gradient clipping: {clipping_params}")

    # Optimizer factory mapping
    optimizer_creators: Dict[str, Callable] = {
        OptimizerType.RMSPROP: _create_rmsprop_optimizer,
        OptimizerType.ADAM: _create_adam_optimizer,
        OptimizerType.ADAMW: _create_adamw_optimizer,
        OptimizerType.ADADELTA: _create_adadelta_optimizer,
    }

    if optimizer_type not in optimizer_creators:
        available_types = list(optimizer_creators.keys())
        raise ValueError(f"Unknown optimizer type: [{optimizer_type}]. "
                        f"Available types: {available_types}")

    # Create optimizer
    optimizer = optimizer_creators[optimizer_type](config, lr_schedule, clipping_params)

    return optimizer, lr_schedule


# ---------------------------------------------------------------------
>>>>>>> a543855 ([optimization] cleaning up and refactoring)
