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

All schedules built by `schedule_builder` are automatically wrapped with a
warmup schedule that linearly increases the learning rate from a small initial
value to the target rate during the first N training steps.

Alongside the config-driven `schedule_builder`, this module provides two
epoch-facing adapters used directly by the training scripts:
- create_learning_rate_schedule: epochs + absolute warmup steps
- create_warmup_lr_schedule: epochs + warmup expressed as a ratio
These have deliberately different defaults from `schedule_builder` — see the
comment above their definitions before changing either.

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
from typing import Any, Dict, Optional, Union

from keras.api.optimizers.schedules import LearningRateSchedule

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .warmup_schedule import WarmupSchedule
from .constants import *

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
# epoch-facing adapters
# ---------------------------------------------------------------------
#
# The two functions below are epoch/ratio-facing adapters over the same Keras
# primitives that `schedule_builder` assembles. They were moved here from
# `train.common.callbacks` and `train.common.nlp` (which now re-export them) so
# that all learning-rate schedule construction lives in one package.
#
# They are deliberately NOT reimplemented on top of `schedule_builder`: their
# observable behaviour differs from it in ways that dozens of callers depend on.
#   - `schedule_builder` ALWAYS wraps its result in a `WarmupSchedule`, even at
#     `warmup_steps=0`; `create_learning_rate_schedule` returns a BARE
#     `CosineDecay` on that path.
#   - They hard-code `alpha` (0.01 / 0.0) and `decay_rate` (0.9) rather than
#     taking them from config.
#   - `create_learning_rate_schedule('constant')` returns a bare float, which is
#     not a schedule type `schedule_builder` supports at all.
# Any "unification" that erases these differences is a silent behaviour change.


def create_learning_rate_schedule(
        initial_lr: float,
        schedule_type: str = 'cosine',
        total_epochs: int = 100,
        warmup_epochs: int = 5,
        steps_per_epoch: Optional[int] = None,
        warmup_steps: int = 0,
        warmup_start_lr: float = 1e-8,
) -> Union[float, LearningRateSchedule]:
    """Create an epoch-parameterised learning rate schedule.

    Args:
        initial_lr: Initial learning rate.
        schedule_type: Type of schedule ('cosine', 'exponential', 'constant').
        total_epochs: Total number of training epochs.
        warmup_epochs: RESERVED / no-op — kept only for backward positional
            compatibility. Use ``warmup_steps`` to activate warmup.
        steps_per_epoch: Steps per epoch (for step-based schedules like
            ImageNet). Required when ``warmup_steps > 0``.
        warmup_steps: Active warmup control. When ``> 0`` (cosine schedule
            only), the cosine decay is wrapped in a :class:`WarmupSchedule` that
            linearly ramps from ``warmup_start_lr`` to ``initial_lr`` over
            ``warmup_steps`` steps before decaying. ``0`` (default) means NO
            warmup — existing callers are unaffected.
        warmup_start_lr: Learning rate at the start of the warmup ramp (only
            used when ``warmup_steps > 0``).

    Returns:
        A ``LearningRateSchedule``, or a bare float for ``'constant'``.

    Raises:
        ValueError: If ``warmup_steps > 0`` but ``steps_per_epoch`` is None.
    """
    if schedule_type == 'cosine':
        # DECISION plan_2026-06-02_cc4d4e14/D-004: warmup is wired ONLY through the
        # explicit warmup_steps param (default 0 = no-op). Do NOT activate via
        # warmup_epochs — dozens of existing callers rely on the plain cosine path
        # and would silently gain warmup (behavior regression). warmup engages only
        # when warmup_steps>0, reproducing the inline CosineDecay+WarmupSchedule
        # block (alpha=0.01, max(1, total_steps-warmup_steps) guard) at the 11 C1
        # sites. See decisions.md D-004.
        if warmup_steps > 0:
            if steps_per_epoch is None:
                raise ValueError(
                    "create_learning_rate_schedule: warmup_steps>0 requires steps_per_epoch"
                )
            total_steps = total_epochs * steps_per_epoch
            primary = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=initial_lr,
                decay_steps=max(1, total_steps - warmup_steps),
                alpha=0.01,
            )
            return WarmupSchedule(
                warmup_steps=warmup_steps,
                warmup_start_lr=warmup_start_lr,
                primary_schedule=primary,
            )
        decay_steps = total_epochs if steps_per_epoch is None else total_epochs * steps_per_epoch
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            alpha=0.01
        )
    elif schedule_type == 'exponential':
        decay_steps = (total_epochs // 4) if steps_per_epoch is None else (total_epochs // 4) * steps_per_epoch
        return keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=0.9
        )
    else:  # constant
        return initial_lr


def create_warmup_lr_schedule(
        learning_rate: float,
        num_epochs: int,
        steps_per_epoch: int,
        warmup_ratio: float = 0.1,
) -> WarmupSchedule:
    """Create a warmup + cosine decay schedule from a warmup *ratio*.

    The NLP-training counterpart to :func:`create_learning_rate_schedule`:
    warmup length is expressed as a fraction of the total step budget rather
    than as an absolute step count, and warmup is always active.

    Args:
        learning_rate: Peak learning rate reached at the end of warmup.
        num_epochs: Total number of training epochs.
        steps_per_epoch: Steps per epoch.
        warmup_ratio: Fraction of total steps spent warming up.

    Returns:
        A :class:`WarmupSchedule` wrapping a ``CosineDecay`` (``alpha=0.0``).
    """
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(warmup_ratio * total_steps)
    primary = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=total_steps - warmup_steps, alpha=0.0,
    )
    return WarmupSchedule(
        warmup_steps=warmup_steps, primary_schedule=primary, warmup_start_lr=1e-7,
    )


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