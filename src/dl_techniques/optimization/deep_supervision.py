# ---------------------------------------------------------------------
# Deep Supervision Weight Scheduling
# ---------------------------------------------------------------------
#
# This module implements weight scheduling strategies for deep supervision
# in neural networks, particularly designed for U-Net architectures. Deep
# supervision applies loss functions at multiple depths/scales in a network,
# and these schedules determine how to weight each level's contribution.
#
# The schedules control the training focus during different stages:
# - Early training can focus on deeper, low-resolution outputs
# - Later training can shift to shallower, high-resolution outputs
# - Different applications may benefit from different weighting strategies
#
# Each schedule function takes a percentage of training completion (0.0 to 1.0)
# and returns a normalized weight array with one weight per supervision level.
#
# Available schedules:
# - constant_equal: Equal weights for all outputs
# - constant_low_to_high: Fixed weights favoring higher resolution
# - constant_high_to_low: Fixed weights favoring deeper layers
# - linear_low_to_high: Gradual linear transition from deep to shallow focus
# - non_linear_low_to_high: Quadratic transition for smoother focus shift
# - custom_sigmoid_low_to_high: Sigmoid-based transition with custom parameters
# - scale_by_scale_low_to_high: Progressive activation of outputs
# - cosine_annealing: Oscillating weights with overall trend
# - curriculum: Progressive activation based on training progress
#
# Usage Example:
#   config = {
#       "type": "linear_low_to_high",
#       "config": {}
#   }
#   weighter = schedule_builder(config, 5)  # For 5 outputs
#
#   # Get weights at 30% through training
#   weights = weighter(0.3)  # Returns numpy array of 5 weights
#
# ---------------------------------------------------------------------

import numpy as np
from enum import Enum, auto
from typing import Dict, Callable, Optional, Literal, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.constants import *
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


class ScheduleType(str, Enum):
    """Enumeration of available deep supervision schedule types."""
    CONSTANT_EQUAL = "constant_equal"
    CONSTANT_LOW_TO_HIGH = "constant_low_to_high"
    CONSTANT_HIGH_TO_LOW = "constant_high_to_low"
    LINEAR_LOW_TO_HIGH = "linear_low_to_high"
    NON_LINEAR_LOW_TO_HIGH = "non_linear_low_to_high"
    CUSTOM_SIGMOID_LOW_TO_HIGH = "custom_sigmoid_low_to_high"
    SCALE_BY_SCALE_LOW_TO_HIGH = "scale_by_scale_low_to_high"
    COSINE_ANNEALING = "cosine_annealing"
    CURRICULUM = "curriculum"


def schedule_builder(
        config: Dict[str, Union[str, Dict[str, Union[float, str]]]],
        no_outputs: int
) -> Callable[[float], np.ndarray]:
    """Builds a schedule function for deep supervision weight distribution.

    Creates a callable that generates weight arrays for multiple outputs at different
    training progress stages. The weights determine the contribution of each scale
    to the final loss during training.

    In the context of U-Net architecture:
    - Output 0: Final inference output (highest resolution)
    - Output (no_outputs-1): Deepest scale in the network (lowest resolution)
    - Outputs in between: Intermediate scales

    Args:
        config: Configuration dictionary containing the schedule type and parameters.
            Must include a 'type' key with string value specifying the schedule type.
            May include a 'config' key with parameters specific to the schedule type.
        no_outputs: Number of outputs (scales) for which weights must be generated.

    Returns:
        A function that takes a float percentage (0.0 to 1.0) and returns a numpy array
        of shape [no_outputs] representing the weights for each output. The weights
        always sum to 1.0.

    Raises:
        ValueError: If config is invalid, schedule_type is unknown, or no_outputs <= 0.

    Example:
        >>> config = {"type": "linear_low_to_high", "config": {}}
        >>> weight_scheduler = schedule_builder(config, 5)
        >>> weights_at_50_percent = weight_scheduler(0.5)  # Returns weights for 5 outputs
    """
    # Validate arguments
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    if no_outputs <= 0:
        raise ValueError("no_outputs must be a positive integer")

    # Extract schedule type
    schedule_type = config.get(TYPE_STR)
    if schedule_type is None:
        raise ValueError("schedule_type cannot be None")
    if not isinstance(schedule_type, str):
        raise ValueError("schedule_type must be a string")

    # Get schedule parameters
    params = config.get(CONFIG_STR, {})
    schedule_type = schedule_type.strip().lower()

    logger.info(
        f"Deep supervision schedule: [{schedule_type}], with params: [{params}]"
    )

    # Create and return the appropriate schedule function
    if schedule_type == ScheduleType.CONSTANT_EQUAL:
        return lambda percentage_done: constant_equal_schedule(percentage_done, no_outputs)

    elif schedule_type == ScheduleType.CONSTANT_LOW_TO_HIGH:
        return lambda percentage_done: constant_low_to_high_schedule(percentage_done, no_outputs)

    elif schedule_type == ScheduleType.CONSTANT_HIGH_TO_LOW:
        return lambda percentage_done: constant_high_to_low_schedule(percentage_done, no_outputs)

    elif schedule_type == ScheduleType.LINEAR_LOW_TO_HIGH:
        return lambda percentage_done: linear_low_to_high_schedule(percentage_done, no_outputs)

    elif schedule_type == ScheduleType.NON_LINEAR_LOW_TO_HIGH:
        return lambda percentage_done: non_linear_low_to_high_schedule(percentage_done, no_outputs)

    elif schedule_type == ScheduleType.CUSTOM_SIGMOID_LOW_TO_HIGH:
        # Extract parameters for custom sigmoid
        k: float = params.get('k', 10.0)
        x0: float = params.get('x0', 0.5)
        transition_point: float = params.get('transition_point', 0.25)

        return lambda percentage_done: custom_sigmoid_low_to_high_schedule(
            percentage_done, no_outputs, k, x0, transition_point
        )

    elif schedule_type == ScheduleType.SCALE_BY_SCALE_LOW_TO_HIGH:
        return lambda percentage_done: scale_by_scale_low_to_high_schedule(
            percentage_done, no_outputs
        )

    elif schedule_type == ScheduleType.COSINE_ANNEALING:
        # Extract parameters for cosine annealing schedule
        frequency: float = params.get('frequency', 3.0)
        final_ratio: float = params.get('final_ratio', 0.8)

        return lambda percentage_done: cosine_annealing_schedule(
            percentage_done, no_outputs, frequency, final_ratio
        )

    elif schedule_type == ScheduleType.CURRICULUM:
        # Extract parameters for curriculum schedule
        max_active_outputs: int = params.get('max_active_outputs', no_outputs)
        activation_strategy: str = params.get('activation_strategy', 'linear')

        return lambda percentage_done: curriculum_schedule(
            percentage_done, no_outputs, max_active_outputs, activation_strategy
        )

    else:
        raise ValueError(
            f"Unknown deep supervision schedule_type: [{schedule_type}]"
        )


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Safely normalize weights to sum to 1.0.

    Args:
        weights: Array of weights to normalize.

    Returns:
        Normalized weights summing to 1.0.
    """
    weights_sum = np.sum(weights)
    # Avoid division by zero
    if weights_sum == 0:
        return np.ones_like(weights) / len(weights)
    return weights / weights_sum


def constant_equal_schedule(percentage_done: float, no_outputs: int) -> np.ndarray:
    """Equal weighting for all outputs regardless of training progress.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of equal weights summing to 1.0.
    """
    return np.ones(no_outputs) / no_outputs


def constant_low_to_high_schedule(percentage_done: float, no_outputs: int) -> np.ndarray:
    """Constant weighting favoring higher resolution outputs.

    Weights increase linearly from the deepest layer to the final output.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of weights increasing from deeper to shallower layers.
    """
    weights = np.arange(1, no_outputs + 1, dtype=np.float64)
    return _normalize_weights(weights)


def constant_high_to_low_schedule(percentage_done: float, no_outputs: int) -> np.ndarray:
    """Constant weighting favoring deeper layer outputs.

    Weights decrease linearly from the deepest layer to the final output.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of weights decreasing from deeper to shallower layers.
    """
    weights = np.arange(no_outputs, 0, -1, dtype=np.float64)
    return _normalize_weights(weights)


def linear_low_to_high_schedule(percentage_done: float, no_outputs: int) -> np.ndarray:
    """Linear transition from focusing on deep layers to focusing on shallow layers.

    As training progresses, weight shifts from deeper to shallower layers linearly.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of weights that transition based on training progress.
    """
    # Initial weights favor deeper layers
    initial_weights = np.arange(no_outputs, 0, -1, dtype=np.float64)

    # Final weights favor shallower layers
    final_weights = np.arange(1, no_outputs + 1, dtype=np.float64)

    # Linear interpolation between initial and final weights
    weights = (1 - percentage_done) * initial_weights + percentage_done * final_weights

    return _normalize_weights(weights)


def non_linear_low_to_high_schedule(percentage_done: float, no_outputs: int) -> np.ndarray:
    """Non-linear transition from deep to shallow layer focus.

    Uses quadratic interpolation for smoother transition.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of weights with non-linear transition pattern.
    """
    # Initial weights favor deeper layers
    initial_weights = np.arange(no_outputs, 0, -1, dtype=np.float64)

    # Final weights favor shallower layers
    final_weights = np.arange(1, no_outputs + 1, dtype=np.float64)

    # Non-linear (quadratic) interpolation - smoother transition in middle of training
    factor = percentage_done ** 2
    weights = (1 - factor) * initial_weights + factor * final_weights

    return _normalize_weights(weights)


def custom_sigmoid_low_to_high_schedule(
        percentage_done: float,
        no_outputs: int,
        k: float = 10.0,
        x0: float = 0.5,
        transition_point: float = 0.25
) -> np.ndarray:
    """Sigmoid-based transition from deep to shallow layer focus.

    Uses a sigmoid function to create a smooth transition between layer importance.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.
        k: Sigmoid steepness parameter.
        x0: Sigmoid midpoint parameter.
        transition_point: Training percentage where transition begins.

    Returns:
        Array of weights with sigmoid-based transition pattern.
    """
    # Adjust percentage based on transition point
    adjusted_percentage = max(0, (percentage_done - transition_point) / (1 - transition_point))

    # Sigmoid function: 1 / (1 + exp(-k(x - x0)))
    sigmoid_factor = 1 / (1 + np.exp(-k * (adjusted_percentage - x0)))

    # Initial weights favor deeper layers
    initial_weights = np.arange(no_outputs, 0, -1, dtype=np.float64)

    # Final weights favor shallower layers
    final_weights = np.arange(1, no_outputs + 1, dtype=np.float64)

    # Sigmoid interpolation between initial and final weights
    weights = (1 - sigmoid_factor) * initial_weights + sigmoid_factor * final_weights

    return _normalize_weights(weights)


def scale_by_scale_low_to_high_schedule(percentage_done: float, no_outputs: int) -> np.ndarray:
    """Progressive activation of outputs from deep to shallow.

    Gradually activates outputs from deep to shallow as training progresses.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of weights with scale-by-scale progression.
    """
    # Determine which scale is currently active
    active_scale = min(int(np.floor(percentage_done * no_outputs)), no_outputs - 1)

    # Create weight array with active scale having weight 1.0
    weights = np.zeros(no_outputs, dtype=np.float64)
    weights[active_scale] = 1.0

    return weights


def cosine_annealing_schedule(
        percentage_done: float,
        no_outputs: int,
        frequency: float = 3.0,
        final_ratio: float = 0.8
) -> np.ndarray:
    """Cosine annealing schedule for weight distribution.

    Uses cosine function to create cyclical weight patterns during training.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.
        frequency: Number of complete cycles during training.
        final_ratio: Ratio between final and initial layer weights.

    Returns:
        Array of weights following cosine annealing pattern.
    """
    # Cosine factor oscillates between 0 and 1
    cosine_factor = 0.5 * (1 + np.cos(2 * np.pi * frequency * percentage_done))

    # Adjust factor with final ratio to control the range of oscillation
    adjusted_factor = (1 - final_ratio) * cosine_factor + final_ratio

    # Create base weights
    weights = np.ones(no_outputs, dtype=np.float64)

    # Apply increasing weight to deeper layers based on cosine factor
    for i in range(no_outputs):
        scale_factor = 1 + (no_outputs - i - 1) * adjusted_factor
        weights[i] = scale_factor

    return _normalize_weights(weights)


def curriculum_schedule(
        percentage_done: float,
        no_outputs: int,
        max_active_outputs: int = None,
        activation_strategy: str = 'linear'
) -> np.ndarray:
    """Curriculum learning schedule activating outputs progressively.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.
        max_active_outputs: Maximum number of simultaneously active outputs.
        activation_strategy: Strategy for activating outputs ('linear', 'exp', etc.).

    Returns:
        Array of weights implementing curriculum learning pattern.
    """
    if max_active_outputs is None or max_active_outputs <= 0:
        max_active_outputs = no_outputs
    else:
        max_active_outputs = min(max_active_outputs, no_outputs)

    # Start with all weights at zero
    weights = np.zeros(no_outputs, dtype=np.float64)

    # Calculate how many outputs should be active at current percentage
    if activation_strategy == 'exp':
        # Exponential activation (activates more outputs later in training)
        active_count = int(max_active_outputs * (percentage_done ** 2))
    else:
        # Linear activation (default)
        active_count = int(max_active_outputs * percentage_done)

    # Ensure at least one output is active
    active_count = max(1, active_count)

    # Activate outputs from deepest to shallowest
    active_indices = list(range(min(active_count, no_outputs)))

    # Set equal weights for active outputs
    weights[active_indices] = 1.0

    return _normalize_weights(weights)