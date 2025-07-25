"""
Deep Supervision Weight Scheduling Module for Deep Learning Techniques.

This module implements weight scheduling strategies for deep supervision in neural
networks, particularly designed for architectures like U-Net with multiple output
scales. Deep supervision applies loss functions at multiple depths/scales in a
network, and these schedules determine how to weight each level's contribution
during different stages of training.

The schedules control the training focus during different stages:
- Early training can focus on deeper, low-resolution outputs for coarse features
- Later training can shift to shallower, high-resolution outputs for fine details
- Different applications may benefit from different weighting strategies

Key Features:
- Nine different scheduling strategies for various training needs
- Normalized weight arrays that always sum to 1.0
- Progress-based weighting from 0.0 (start) to 1.0 (end) of training
- Configurable parameters for customizable behavior
- Robust error handling and validation

Mathematical Behavior:
Each schedule function takes a training progress percentage (0.0 to 1.0) and
returns a normalized weight array with one weight per supervision level.

In U-Net context:
- Output 0: Final inference output (highest resolution, shallowest)
- Output (n-1): Deepest scale in the network (lowest resolution, deepest)
- Intermediate outputs: Progressive scales between deep and shallow

Available Schedules:
- constant_equal: Equal weights for all outputs throughout training
- constant_low_to_high: Fixed weights favoring higher resolution outputs
- constant_high_to_low: Fixed weights favoring deeper layer outputs
- linear_low_to_high: Gradual linear transition from deep to shallow focus
- non_linear_low_to_high: Quadratic transition for smoother focus shift
- custom_sigmoid_low_to_high: Sigmoid-based transition with custom parameters
- scale_by_scale_low_to_high: Progressive activation of outputs
- cosine_annealing: Oscillating weights with overall trend
- curriculum: Progressive activation based on training progress

Usage Example:
    >>> # Basic linear transition
    >>> config = {
    ...     "type": "linear_low_to_high",
    ...     "config": {}
    ... }
    >>> weight_scheduler = schedule_builder(config, 5)  # For 5 outputs
    >>>
    >>> # Get weights at 30% through training
    >>> weights = weight_scheduler(0.3)  # Returns numpy array of 5 weights
    >>> print(f"Weights sum: {np.sum(weights)}")  # Always 1.0
    >>>
    >>> # Custom sigmoid transition
    >>> sigmoid_config = {
    ...     "type": "custom_sigmoid_low_to_high",
    ...     "config": {
    ...         "k": 10.0,
    ...         "x0": 0.5,
    ...         "transition_point": 0.25
    ...     }
    ... }
    >>> sigmoid_scheduler = schedule_builder(sigmoid_config, 3)
"""

import numpy as np
from enum import Enum
from typing import Dict, Callable, Optional, Union, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.constants import *
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# enums and constants
# ---------------------------------------------------------------------


class ScheduleType(str, Enum):
    """Enumeration of available deep supervision schedule types.

    Each schedule type implements a different strategy for weighting
    multiple outputs during training progression.
    """
    CONSTANT_EQUAL = "constant_equal"
    CONSTANT_LOW_TO_HIGH = "constant_low_to_high"
    CONSTANT_HIGH_TO_LOW = "constant_high_to_low"
    LINEAR_LOW_TO_HIGH = "linear_low_to_high"
    NON_LINEAR_LOW_TO_HIGH = "non_linear_low_to_high"
    CUSTOM_SIGMOID_LOW_TO_HIGH = "custom_sigmoid_low_to_high"
    SCALE_BY_SCALE_LOW_TO_HIGH = "scale_by_scale_low_to_high"
    COSINE_ANNEALING = "cosine_annealing"
    CURRICULUM = "curriculum"


# ---------------------------------------------------------------------
# main functions
# ---------------------------------------------------------------------


def schedule_builder(
        config: Dict[str, Union[str, Dict[str, Union[float, str]]]],
        no_outputs: int
) -> Callable[[float], np.ndarray]:
    """Build a deep supervision weight scheduler from configuration.

    Creates a callable that generates weight arrays for multiple outputs at
    different training progress stages. The weights determine the contribution
    of each scale to the final loss during training.

    In the context of U-Net architecture:
    - Output 0: Final inference output (highest resolution)
    - Output (no_outputs-1): Deepest scale in the network (lowest resolution)
    - Outputs in between: Intermediate scales

    Args:
        config: Configuration dictionary containing the schedule type and parameters.
            Required keys:
                - type: Schedule type string (see ScheduleType enum)
            Optional keys:
                - config: Dictionary with schedule-specific parameters
        no_outputs: Number of outputs (scales) for which weights must be generated.
            Must be a positive integer.

    Returns:
        A function that takes a float percentage (0.0 to 1.0) representing training
        progress and returns a numpy array of shape [no_outputs] with weights for
        each output. The weights always sum to 1.0.

    Raises:
        ValueError: If config is invalid, schedule_type is unknown, or no_outputs <= 0.
        TypeError: If config is not a dictionary or contains invalid types.

    Example:
        >>> # Linear transition from deep to shallow focus
        >>> config = {"type": "linear_low_to_high", "config": {}}
        >>> weight_scheduler = schedule_builder(config, 5)
        >>>
        >>> # Get weights at different training stages
        >>> early_weights = weight_scheduler(0.1)   # Favor deep outputs
        >>> mid_weights = weight_scheduler(0.5)     # Balanced
        >>> late_weights = weight_scheduler(0.9)    # Favor shallow outputs
        >>>
        >>> # All weight arrays sum to 1.0
        >>> assert np.allclose(np.sum(early_weights), 1.0)
    """
    # Validate input arguments
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")
    if not isinstance(no_outputs, int) or no_outputs <= 0:
        raise ValueError(f"no_outputs must be a positive integer, got {no_outputs}")

    # Extract and validate schedule type
    schedule_type = config.get(TYPE_STR)
    if schedule_type is None:
        raise ValueError("schedule_type cannot be None - must specify 'type' in config")
    if not isinstance(schedule_type, str):
        raise TypeError("schedule_type must be a string")

    schedule_type = schedule_type.strip().lower()

    # Extract schedule-specific parameters
    schedule_params = config.get(CONFIG_STR, {})
    if not isinstance(schedule_params, dict):
        raise TypeError("'config' must be a dictionary containing schedule parameters")

    logger.info(
        f"Building deep supervision schedule: [{schedule_type}] for {no_outputs} outputs, "
        f"params: [{schedule_params}]"
    )

    # Create and return the appropriate schedule function
    if schedule_type == ScheduleType.CONSTANT_EQUAL:
        return lambda progress: constant_equal_schedule(progress, no_outputs)

    elif schedule_type == ScheduleType.CONSTANT_LOW_TO_HIGH:
        return lambda progress: constant_low_to_high_schedule(progress, no_outputs)

    elif schedule_type == ScheduleType.CONSTANT_HIGH_TO_LOW:
        return lambda progress: constant_high_to_low_schedule(progress, no_outputs)

    elif schedule_type == ScheduleType.LINEAR_LOW_TO_HIGH:
        return lambda progress: linear_low_to_high_schedule(progress, no_outputs)

    elif schedule_type == ScheduleType.NON_LINEAR_LOW_TO_HIGH:
        return lambda progress: non_linear_low_to_high_schedule(progress, no_outputs)

    elif schedule_type == ScheduleType.CUSTOM_SIGMOID_LOW_TO_HIGH:
        # Extract and validate parameters for custom sigmoid
        k = schedule_params.get('k', 10.0)
        x0 = schedule_params.get('x0', 0.5)
        transition_point = schedule_params.get('transition_point', 0.25)

        return lambda progress: custom_sigmoid_low_to_high_schedule(
            progress, no_outputs, k, x0, transition_point
        )

    elif schedule_type == ScheduleType.SCALE_BY_SCALE_LOW_TO_HIGH:
        return lambda progress: scale_by_scale_low_to_high_schedule(
            progress, no_outputs
        )

    elif schedule_type == ScheduleType.COSINE_ANNEALING:
        # Extract and validate parameters for cosine annealing schedule
        frequency = schedule_params.get('frequency', 3.0)
        final_ratio = schedule_params.get('final_ratio', 0.8)

        return lambda progress: cosine_annealing_schedule(
            progress, no_outputs, frequency, final_ratio
        )

    elif schedule_type == ScheduleType.CURRICULUM:
        # Extract and validate parameters for curriculum schedule
        max_active_outputs = schedule_params.get('max_active_outputs', no_outputs)
        activation_strategy = schedule_params.get('activation_strategy', 'linear')

        return lambda progress: curriculum_schedule(
            progress, no_outputs, max_active_outputs, activation_strategy
        )

    else:
        raise ValueError(
            f"Unknown deep supervision schedule_type: [{schedule_type}]. "
            f"Supported types: {[t.value for t in ScheduleType]}"
        )


# ---------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Safely normalize weights to sum to 1.0.

    Args:
        weights: Array of weights to normalize. Must be non-negative.

    Returns:
        Normalized weights summing to 1.0. If input weights sum to zero,
        returns equal weights for all outputs.

    Example:
        >>> weights = np.array([1.0, 2.0, 3.0])
        >>> normalized = _normalize_weights(weights)
        >>> print(normalized)  # [0.166..., 0.333..., 0.5]
        >>> print(np.sum(normalized))  # 1.0
    """
    weights_sum = np.sum(weights)
    # Handle edge case where all weights are zero
    if weights_sum == 0:
        return np.ones_like(weights) / len(weights)
    return weights / weights_sum


# ---------------------------------------------------------------------
# constant schedules
# ---------------------------------------------------------------------


def constant_equal_schedule(percentage_done: float, no_outputs: int) -> np.ndarray:
    """Equal weighting for all outputs regardless of training progress.

    This schedule provides uniform attention to all output scales throughout
    training, which can be useful when all scales are equally important.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0 (unused).
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of equal weights, each equal to 1/no_outputs.
    """
    return np.ones(no_outputs, dtype=np.float64) / no_outputs


def constant_low_to_high_schedule(percentage_done: float, no_outputs: int) -> np.ndarray:
    """Constant weighting favoring higher resolution outputs.

    Weights increase linearly from the deepest layer to the final output,
    providing consistent emphasis on higher resolution features throughout training.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0 (unused).
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of weights increasing from deeper to shallower layers.
    """
    weights = np.arange(1, no_outputs + 1, dtype=np.float64)
    return _normalize_weights(weights)


def constant_high_to_low_schedule(percentage_done: float, no_outputs: int) -> np.ndarray:
    """Constant weighting favoring deeper layer outputs.

    Weights decrease linearly from the deepest layer to the final output,
    providing consistent emphasis on coarse, semantic features throughout training.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0 (unused).
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of weights decreasing from deeper to shallower layers.
    """
    weights = np.arange(no_outputs, 0, -1, dtype=np.float64)
    return _normalize_weights(weights)


# ---------------------------------------------------------------------
# progressive schedules
# ---------------------------------------------------------------------


def linear_low_to_high_schedule(percentage_done: float, no_outputs: int) -> np.ndarray:
    """Linear transition from focusing on deep layers to focusing on shallow layers.

    As training progresses, weight shifts from deeper to shallower layers linearly.
    This is often effective for segmentation tasks where coarse features should
    be learned first, followed by fine-grained details.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of weights that transition linearly based on training progress.
    """
    # Initial weights favor deeper layers (higher indices)
    initial_weights = np.arange(no_outputs, 0, -1, dtype=np.float64)

    # Final weights favor shallower layers (lower indices)
    final_weights = np.arange(1, no_outputs + 1, dtype=np.float64)

    # Linear interpolation between initial and final weights
    weights = (1 - percentage_done) * initial_weights + percentage_done * final_weights

    return _normalize_weights(weights)


def non_linear_low_to_high_schedule(percentage_done: float, no_outputs: int) -> np.ndarray:
    """Non-linear transition from deep to shallow layer focus.

    Uses quadratic interpolation for smoother transition, with more gradual
    change early in training and faster change later. This can be beneficial
    when fine-tuning requires more aggressive reweighting.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of weights with quadratic transition pattern.
    """
    # Initial weights favor deeper layers
    initial_weights = np.arange(no_outputs, 0, -1, dtype=np.float64)

    # Final weights favor shallower layers
    final_weights = np.arange(1, no_outputs + 1, dtype=np.float64)

    # Quadratic interpolation - smoother transition in middle of training
    quadratic_factor = percentage_done ** 2
    weights = (1 - quadratic_factor) * initial_weights + quadratic_factor * final_weights

    return _normalize_weights(weights)


def custom_sigmoid_low_to_high_schedule(
        percentage_done: float,
        no_outputs: int,
        k: float = 10.0,
        x0: float = 0.5,
        transition_point: float = 0.25
) -> np.ndarray:
    """Sigmoid-based transition from deep to shallow layer focus.

    Uses a sigmoid function to create a smooth, S-shaped transition between
    layer importance. Provides fine control over the transition timing and steepness.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.
        k: Sigmoid steepness parameter. Higher values create sharper transitions.
        x0: Sigmoid midpoint parameter (0.0 to 1.0). Where the transition is centered.
        transition_point: Training percentage where transition begins (0.0 to 1.0).

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

    Gradually activates outputs from deep to shallow as training progresses,
    with only one scale being active at a time. This can be useful for
    curriculum learning approaches.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of weights with only one active scale (weight = 1.0).
    """
    # Determine which scale is currently active based on progress
    active_scale_index = min(int(np.floor(percentage_done * no_outputs)), no_outputs - 1)

    # Create weight array with only the active scale having weight 1.0
    weights = np.zeros(no_outputs, dtype=np.float64)
    weights[active_scale_index] = 1.0

    return weights


# ---------------------------------------------------------------------
# advanced schedules
# ---------------------------------------------------------------------


def cosine_annealing_schedule(
        percentage_done: float,
        no_outputs: int,
        frequency: float = 3.0,
        final_ratio: float = 0.8
) -> np.ndarray:
    """Cosine annealing schedule for weight distribution.

    Uses cosine function to create cyclical weight patterns during training,
    which can help escape local minima and explore different feature combinations.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.
        frequency: Number of complete cosine cycles during training.
        final_ratio: Ratio between final and initial layer weight ranges.

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
        # Deeper layers (higher indices) get more modulation
        depth_factor = 1 + (no_outputs - i - 1) * adjusted_factor
        weights[i] = depth_factor

    return _normalize_weights(weights)


def curriculum_schedule(
        percentage_done: float,
        no_outputs: int,
        max_active_outputs: Optional[int] = None,
        activation_strategy: str = 'linear'
) -> np.ndarray:
    """Curriculum learning schedule activating outputs progressively.

    Implements curriculum learning by progressively activating more outputs
    as training advances. This can help models learn hierarchical features
    in a structured manner.

    Args:
        percentage_done: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.
        max_active_outputs: Maximum number of simultaneously active outputs.
            If None or <= 0, defaults to no_outputs.
        activation_strategy: Strategy for activating outputs:
            - 'linear': Linear activation progression
            - 'exp': Exponential activation (more outputs activated later)

    Returns:
        Array of weights implementing curriculum learning pattern.
    """
    # Validate and set max_active_outputs
    if max_active_outputs is None or max_active_outputs <= 0:
        max_active_outputs = no_outputs
    else:
        max_active_outputs = min(max_active_outputs, no_outputs)

    # Start with all weights at zero
    weights = np.zeros(no_outputs, dtype=np.float64)

    # Calculate how many outputs should be active at current percentage
    if activation_strategy.lower() == 'exp':
        # Exponential activation (activates more outputs later in training)
        active_count = int(max_active_outputs * (percentage_done ** 2))
    else:
        # Linear activation (default)
        active_count = int(max_active_outputs * percentage_done)

    # Ensure at least one output is active
    active_count = max(1, active_count)

    # Activate outputs from deepest to shallowest (curriculum progression)
    active_indices = list(range(min(active_count, no_outputs)))

    # Set equal weights for active outputs
    weights[active_indices] = 1.0

    return _normalize_weights(weights)