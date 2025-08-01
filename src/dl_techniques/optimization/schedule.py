"""
Learning Rate Schedule Builder Module for Deep Learning Techniques.

This module provides functionality to create and configure various learning rate
schedules (Cosine Decay, Exponential Decay, Cosine Decay with Restarts) with
optional warmup periods for stable training initialization.

The schedule builder supports:
- Multiple schedule types with configurable parameters
- Linear warmup periods for training stability
- Default parameter fallbacks from constants module
- Integration with Keras optimizers
- Flattened configuration structure for simplicity

All schedules are automatically wrapped with a warmup schedule that linearly
increases the learning rate from a small initial value to the target rate
during the first N training steps.

Usage Example:
    >>> config = {
    ...     "type": "cosine_decay",
    ...     "warmup_steps": 1000,
    ...     "warmup_start_lr": 1e-8,
    ...     "learning_rate": 0.001,
    ...     "decay_steps": 10000,
    ...     "alpha": 0.0001
    ... }
    >>> lr_schedule = schedule_builder(config)
"""

import keras
from enum import Enum
from typing import Dict, Union, Any

from keras.api.optimizers.schedules import LearningRateSchedule

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .constants import *
from .warmup_schedule import WarmupSchedule
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# enums
# ---------------------------------------------------------------------


class ScheduleType(str, Enum):
    """Enumeration of available learning rate schedule types."""
    COSINE_DECAY = "cosine_decay"
    EXPONENTIAL_DECAY = "exponential_decay"
    COSINE_DECAY_RESTARTS = "cosine_decay_restarts"


# ---------------------------------------------------------------------
# main functions
# ---------------------------------------------------------------------


def schedule_builder(
        config: Dict[str, Union[str, int, float]]
) -> LearningRateSchedule:
    """Build a learning rate schedule with optional warmup from configuration.

    Creates a Keras learning rate schedule based on configuration options,
    automatically wrapped with a warmup period for training stability. The
    warmup phase linearly increases the learning rate from a small initial
    value to the target rate over the specified number of steps.

    Args:
        config: Flattened configuration dictionary containing all parameters.
            Required keys:
                - type: Schedule type ('cosine_decay', 'exponential_decay',
                       'cosine_decay_restarts')
                - learning_rate: Initial learning rate for the schedule
            Schedule-specific required keys:
                - decay_steps: Number of steps for decay (all schedules)
                - decay_rate: Decay rate (exponential_decay only)
            Optional keys:
                - warmup_steps: Number of warmup steps (default: 0)
                - warmup_start_lr: Starting learning rate for warmup (default: 1e-8)
                - alpha: Minimum learning rate fraction (cosine schedules)
                - t_mul: Period multiplier (cosine_decay_restarts)
                - m_mul: LR multiplier (cosine_decay_restarts)

    Returns:
        A WarmupSchedule instance wrapping the configured base schedule.

    Raises:
        ValueError: If config is invalid, schedule_type is unknown, or required
                   parameters are missing.

    Example:
        >>> config = {
        ...     "type": "cosine_decay",
        ...     "warmup_steps": 1000,
        ...     "warmup_start_lr": 1e-8,
        ...     "learning_rate": 0.001,
        ...     "decay_steps": 10000,
        ...     "alpha": 0.0001
        ... }
        >>> lr_schedule = schedule_builder(config)
    """
    # Validate arguments
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")

    # Extract and validate schedule type
    schedule_type = config.get("type")
    if schedule_type is None:
        raise ValueError("schedule_type cannot be None - must specify 'type' in config")
    if not isinstance(schedule_type, str):
        raise ValueError("schedule_type must be a string")

    schedule_type = schedule_type.strip().lower()

    # Extract warmup parameters with defaults
    warmup_steps = config.get('warmup_steps', DEFAULT_WARMUP_STEPS)
    if warmup_steps is None:
        raise ValueError("warmup_steps must be specified in config")
    warmup_start_lr = config.get('warmup_start_lr', DEFAULT_WARMUP_START_LR)

    # Filter out warmup parameters for schedule-specific parameters
    schedule_params = {k: v for k, v in config.items()
                      if k not in ['type', 'warmup_steps', 'warmup_start_lr']}

    logger.info(f"Building schedule: [{schedule_type}] with warmup_steps: [{warmup_steps}], schedule_params: [{schedule_params}]")

    # Create the base learning rate schedule
    if schedule_type == ScheduleType.EXPONENTIAL_DECAY:
        base_schedule = _build_exponential_decay_schedule(schedule_params)

    elif schedule_type == ScheduleType.COSINE_DECAY_RESTARTS:
        base_schedule = _build_cosine_decay_restarts_schedule(schedule_params)

    elif schedule_type == ScheduleType.COSINE_DECAY:
        base_schedule = _build_cosine_decay_schedule(schedule_params)

    else:
        raise ValueError(
            f"Unknown learning_rate schedule_type: [{schedule_type}]. "
            f"Supported types: {[t.value for t in ScheduleType]}"
        )

    # Wrap with warmup schedule
    warmup_schedule = WarmupSchedule(
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        primary_schedule=base_schedule
    )

    logger.info(f"Successfully built {base_schedule.__class__.__name__} schedule with {warmup_steps}-step warmup")
    return warmup_schedule


# ---------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------


def _build_exponential_decay_schedule(
        config: Dict[str, Any]
) -> keras.optimizers.schedules.ExponentialDecay:
    """Build ExponentialDecay schedule from flattened configuration.

    Args:
        config: Flattened configuration dictionary containing required parameters:
            - learning_rate: Initial learning rate
            - decay_steps: Number of steps between decay applications
            - decay_rate: Multiplicative factor for decay

    Returns:
        Configured ExponentialDecay schedule instance.

    Raises:
        KeyError: If required parameters are missing.
    """
    required_params = ["learning_rate", "decay_steps", "decay_rate"]
    _validate_required_params(config, required_params, "exponential_decay")

    return keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config["learning_rate"],
        decay_steps=config["decay_steps"],
        decay_rate=config["decay_rate"]
    )


def _build_cosine_decay_restarts_schedule(
        config: Dict[str, Any]
) -> keras.optimizers.schedules.CosineDecayRestarts:
    """Build CosineDecayRestarts schedule from flattened configuration.

    Args:
        config: Flattened configuration dictionary containing required and optional parameters:
            Required:
                - learning_rate: Initial learning rate
                - decay_steps: Steps in first decay period
            Optional:
                - t_mul: Factor to multiply period after each restart
                - m_mul: Factor to multiply initial LR after each restart
                - alpha: Minimum learning rate as fraction of initial

    Returns:
        Configured CosineDecayRestarts schedule instance.

    Raises:
        KeyError: If required parameters are missing.
    """
    required_params = ["learning_rate", "decay_steps"]
    _validate_required_params(config, required_params, "cosine_decay_restarts")

    return keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=config["learning_rate"],
        first_decay_steps=config["decay_steps"],
        t_mul=config.get("t_mul", DEFAULT_COSINE_RESTARTS_T_MUL),
        m_mul=config.get("m_mul", DEFAULT_COSINE_RESTARTS_M_MUL),
        alpha=config.get("alpha", DEFAULT_COSINE_RESTARTS_ALPHA)
    )


def _build_cosine_decay_schedule(
        config: Dict[str, Any]
) -> keras.optimizers.schedules.CosineDecay:
    """Build CosineDecay schedule from flattened configuration.

    Args:
        config: Flattened configuration dictionary containing required and optional parameters:
            Required:
                - learning_rate: Initial learning rate
                - decay_steps: Number of steps to decay over
            Optional:
                - alpha: Minimum learning rate as fraction of initial

    Returns:
        Configured CosineDecay schedule instance.

    Raises:
        KeyError: If required parameters are missing.
    """
    required_params = ["learning_rate", "decay_steps"]
    _validate_required_params(config, required_params, "cosine_decay")

    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config["learning_rate"],
        decay_steps=config["decay_steps"],
        alpha=config.get("alpha", DEFAULT_COSINE_ALPHA)
    )


def _validate_required_params(
        config: Dict[str, Any],
        required_params: list[str],
        schedule_name: str
) -> None:
    """Validate that all required parameters are present in config dictionary.

    Args:
        config: Configuration dictionary to validate.
        required_params: List of required parameter names.
        schedule_name: Name of the schedule (for error messages).

    Raises:
        KeyError: If any required parameter is missing.
    """
    missing_params = [param for param in required_params if param not in config]
    if missing_params:
        raise KeyError(
            f"Missing required parameters for {schedule_name}: {missing_params}. "
            f"Required parameters: {required_params}"
        )