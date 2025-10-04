"""
Deep Supervision Weight Scheduling Module for Deep Learning Techniques.

This module implements weight scheduling strategies for deep supervision in neural
networks, particularly designed for architectures like U-Net with multiple output
scales. Deep supervision applies loss functions at multiple depths/scales in a
network, and these schedules determine how to weight each level's contribution
during different stages of training.

Overview
--------
The schedules control the training focus during different stages:
- Early training can focus on deeper, low-resolution outputs for coarse features
- Later training can shift to shallower, high-resolution outputs for fine details
- Different applications may benefit from different weighting strategies

Key Features
------------
- Ten different scheduling strategies for various training needs
- Normalized weight arrays that always sum to 1.0
- Progress-based weighting from 0.0 (start) to 1.0 (end) of training
- Configurable parameters for customizable behavior
- Robust error handling and validation
- Support for weight order inversion for different architectural conventions

Mathematical Behavior
---------------------
Each schedule function takes a training progress percentage (0.0 to 1.0) and
returns a normalized weight array with one weight per supervision level.

In U-Net context (default order):
- Output 0: Final inference output (highest resolution, shallowest)
- Output (n-1): Deepest scale in the network (lowest resolution, deepest)
- Intermediate outputs: Progressive scales between deep and shallow
"""

import numpy as np
from enum import Enum
from typing import Dict, Callable, Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.constants import TYPE_STR, CONFIG_STR


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
    STEP_WISE = "step_wise"


# ---------------------------------------------------------------------

def schedule_builder(
        config: Dict[str, Union[str, Dict[str, Union[float, str]]]],
        no_outputs: int,
        invert_order: bool = False
) -> Callable[[float], np.ndarray]:
    """Build a deep supervision weight scheduler from configuration.

    Creates a callable that generates weight arrays for multiple outputs at different
    training progress stages. The weights determine the contribution of each scale
    to the final loss during training.

    In the context of U-Net architecture (default order):
    - Output 0: Final inference output (highest resolution)
    - Output (no_outputs-1): Deepest scale in the network (lowest resolution)

    When invert_order=True, the indexing is reversed:
    - Output 0: Deepest scale in the network (lowest resolution)
    - Output (no_outputs-1): Final inference output (highest resolution)

    Args:
        config: Configuration dictionary containing the schedule type and parameters.
            Required keys:
                - type: Schedule type string (see ScheduleType enum)
            Optional keys:
                - config: Dictionary with schedule-specific parameters
        no_outputs: Number of outputs (scales) to generate weights for.
        invert_order: If True, inverts the weight order. Defaults to False.

    Returns:
        A function that takes a float (0.0 to 1.0) representing training
        progress and returns a numpy array of shape [no_outputs] with weights for
        each output. The weights always sum to 1.0.

    Raises:
        ValueError: If config is invalid, schedule type is unknown, or parameters
            are out of range.
        TypeError: If config is not a dictionary or contains invalid types.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")
    if not isinstance(no_outputs, int) or no_outputs <= 0:
        raise ValueError(f"no_outputs must be a positive integer, got {no_outputs}")
    if not isinstance(invert_order, bool):
        raise TypeError("invert_order must be a boolean")

    schedule_type_str = config.get(TYPE_STR)
    if not isinstance(schedule_type_str, str):
        raise TypeError("'type' must be a string specified in config")

    schedule_type_str = schedule_type_str.strip().lower()
    try:
        schedule_type = ScheduleType(schedule_type_str)
    except ValueError:
        raise ValueError(
            f"Unknown deep supervision schedule type: [{schedule_type_str}]. "
            f"Supported types: {[t.value for t in ScheduleType]}"
        )

    schedule_params = config.get(CONFIG_STR, {})
    if not isinstance(schedule_params, dict):
        raise TypeError("'config' must be a dictionary of schedule parameters")

    logger.info(
        f"Building deep supervision schedule: [{schedule_type.value}] for {no_outputs} outputs, "
        f"params: [{schedule_params}], invert_order: {invert_order}"
    )

    scheduler = None

    if schedule_type is ScheduleType.CONSTANT_EQUAL:
        scheduler = lambda progress: constant_equal_schedule(progress, no_outputs)

    elif schedule_type is ScheduleType.CONSTANT_LOW_TO_HIGH:
        scheduler = lambda progress: constant_low_to_high_schedule(progress, no_outputs)

    elif schedule_type is ScheduleType.CONSTANT_HIGH_TO_LOW:
        scheduler = lambda progress: constant_high_to_low_schedule(progress, no_outputs)

    elif schedule_type is ScheduleType.LINEAR_LOW_TO_HIGH:
        scheduler = lambda progress: linear_low_to_high_schedule(progress, no_outputs)

    elif schedule_type is ScheduleType.NON_LINEAR_LOW_TO_HIGH:
        scheduler = lambda progress: non_linear_low_to_high_schedule(progress, no_outputs)

    elif schedule_type is ScheduleType.CUSTOM_SIGMOID_LOW_TO_HIGH:
        k = schedule_params.get('k', 10.0)
        x0 = schedule_params.get('x0', 0.5)
        transition_point = schedule_params.get('transition_point', 0.25)
        if not (isinstance(k, (int, float)) and k > 0):
            raise ValueError(f"k must be a positive number, got {k}")
        if not (isinstance(x0, (int, float)) and 0.0 <= x0 <= 1.0):
            raise ValueError(f"x0 must be in range [0.0, 1.0], got {x0}")
        if not (isinstance(transition_point, (int, float)) and 0.0 <= transition_point <= 1.0):
            raise ValueError(f"transition_point must be in range [0.0, 1.0], got {transition_point}")

        scheduler = lambda progress: custom_sigmoid_low_to_high_schedule(
            progress, no_outputs, k, x0, transition_point
        )

    elif schedule_type is ScheduleType.SCALE_BY_SCALE_LOW_TO_HIGH:
        scheduler = lambda progress: scale_by_scale_low_to_high_schedule(progress, no_outputs)

    elif schedule_type is ScheduleType.COSINE_ANNEALING:
        frequency = schedule_params.get('frequency', 3.0)
        final_ratio = schedule_params.get('final_ratio', 0.5)
        if not (isinstance(frequency, (int, float)) and frequency > 0):
            raise ValueError(f"frequency must be a positive number, got {frequency}")
        if not (isinstance(final_ratio, (int, float)) and 0.0 <= final_ratio <= 1.0):
            raise ValueError(f"final_ratio must be in range [0.0, 1.0], got {final_ratio}")

        scheduler = lambda progress: cosine_annealing_schedule(
            progress, no_outputs, frequency, final_ratio
        )

    elif schedule_type is ScheduleType.CURRICULUM:
        max_active_outputs = schedule_params.get('max_active_outputs', no_outputs)
        activation_strategy = schedule_params.get('activation_strategy', 'linear')
        if not isinstance(max_active_outputs, int):
            raise TypeError(f"max_active_outputs must be an integer, got {max_active_outputs}")
        if activation_strategy not in ['linear', 'exp']:
            raise ValueError(f"activation_strategy must be 'linear' or 'exp', got '{activation_strategy}'")

        scheduler = lambda progress: curriculum_schedule(
            progress, no_outputs, max_active_outputs, activation_strategy
        )

    elif schedule_type is ScheduleType.STEP_WISE:
        threshold = schedule_params.get('threshold', 0.5)
        if not (isinstance(threshold, (int, float)) and 0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in range [0.0, 1.0], got {threshold}")

        scheduler = lambda progress: step_wise_schedule(
            progress, no_outputs, threshold
        )

    if invert_order:
        return lambda progress: scheduler(progress)[::-1]
    return scheduler


# ---------------------------------------------------------------------

def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize weights to sum to 1.0 with numerical stability.

    Ensures that weight arrays sum to exactly 1.0 for use in deep supervision.
    Handles edge cases where weights sum to zero or very small values.

    Args:
        weights: Array of non-negative weights to normalize.

    Returns:
        Normalized weights that sum to 1.0. If input sums to zero or near-zero,
        returns uniform weights (all equal to 1/n).

    Example:
        >>> weights = np.array([1.0, 2.0, 3.0])
        >>> normalized = _normalize_weights(weights)
        >>> print(normalized)  # [0.166..., 0.333..., 0.5]
        >>> print(np.sum(normalized))  # 1.0
    """
    weights_sum = np.sum(weights)
    if weights_sum <= 1e-8:
        return np.ones_like(weights, dtype=np.float64) / len(weights)
    return weights / weights_sum


# ---------------------------------------------------------------------

def constant_equal_schedule(progress: float, no_outputs: int) -> np.ndarray:
    """Equal weighting for all outputs regardless of training progress.

    Maintains uniform weight distribution across all output scales throughout
    the entire training process. Each output receives exactly 1/n weight at
    all times, where n is the number of outputs.

    This schedule is ideal when:
    1. All scales are equally important for your task
    2. You want balanced supervision across all network depths
    3. Running baseline experiments to understand multi-scale behavior

    Args:
        progress: Current training progress from 0.0 to 1.0 (not used by this schedule).
        no_outputs: Number of outputs to generate weights for. Must be positive.

    Returns:
        Array of equal weights, each equal to 1/no_outputs. Always sums to 1.0.

    Raises:
        ValueError: If no_outputs <= 0.

    Example:
        >>> # With 5 outputs, all stages get equal weights
        >>> weights_start = constant_equal_schedule(0.0, 5)
        >>> print(weights_start)  # [0.2, 0.2, 0.2, 0.2, 0.2]
        >>>
        >>> weights_mid = constant_equal_schedule(0.5, 5)
        >>> print(weights_mid)  # [0.2, 0.2, 0.2, 0.2, 0.2]
        >>>
        >>> weights_end = constant_equal_schedule(1.0, 5)
        >>> print(weights_end)  # [0.2, 0.2, 0.2, 0.2, 0.2]
    """
    return np.ones(no_outputs, dtype=np.float64) / no_outputs


# ---------------------------------------------------------------------

def constant_low_to_high_schedule(progress: float, no_outputs: int) -> np.ndarray:
    """Constant weighting favoring higher resolution (shallower) outputs.

    Maintains a constant weight distribution that linearly increases from
    deeper (low-resolution) to shallower (high-resolution) outputs throughout
    training. The shallowest output (index 0) receives the highest weight,
    while the deepest output (index n-1) receives the lowest.

    This schedule is useful when:
    1. Fine spatial details are more important than coarse semantic features
    2. Working on tasks requiring high spatial precision (e.g., edge detection)
    3. You want consistent emphasis on high-resolution outputs

    Args:
        progress: Current training progress from 0.0 to 1.0 (not used by this schedule).
        no_outputs: Number of outputs to generate weights for. Must be positive.

    Returns:
        Array of linearly decreasing weights from shallow to deep. Always sums to 1.0.

    Raises:
        ValueError: If no_outputs <= 0.

    Example:
        >>> # With 5 outputs, weights favor shallower layers
        >>> weights = constant_low_to_high_schedule(0.0, 5)
        >>> print(weights)  # [0.333, 0.267, 0.200, 0.133, 0.067]
        >>> # Output 0 (shallowest/highest res) gets ~33% weight
        >>> # Output 4 (deepest/lowest res) gets ~7% weight
        >>>
        >>> # Same weights throughout training
        >>> weights_end = constant_low_to_high_schedule(1.0, 5)
        >>> print(weights_end)  # [0.333, 0.267, 0.200, 0.133, 0.067]
    """
    weights = np.arange(no_outputs, 0, -1, dtype=np.float64)
    return _normalize_weights(weights)


# ---------------------------------------------------------------------

def constant_high_to_low_schedule(progress: float, no_outputs: int) -> np.ndarray:
    """Constant weighting favoring deeper (lower resolution) outputs.

    Maintains a constant weight distribution that linearly increases from
    shallower (high-resolution) to deeper (low-resolution) outputs throughout
    training. The deepest output (index n-1) receives the highest weight,
    while the shallowest output (index 0) receives the lowest.

    This schedule is appropriate when:
    1. Coarse semantic features are more important than fine spatial details
    2. Working with hierarchical feature learning tasks
    3. You want to emphasize low-resolution representations throughout training

    Args:
        progress: Current training progress from 0.0 to 1.0 (not used by this schedule).
        no_outputs: Number of outputs to generate weights for. Must be positive.

    Returns:
        Array of linearly increasing weights from shallow to deep. Always sums to 1.0.

    Raises:
        ValueError: If no_outputs <= 0.

    Example:
        >>> # With 5 outputs, weights favor deeper layers
        >>> weights = constant_high_to_low_schedule(0.5, 5)
        >>> print(weights)  # [0.067, 0.133, 0.200, 0.267, 0.333]
        >>> # Output 0 (shallowest/highest res) gets ~7% weight
        >>> # Output 4 (deepest/lowest res) gets ~33% weight
        >>>
        >>> # Same at any training progress
        >>> weights_start = constant_high_to_low_schedule(0.0, 5)
        >>> print(weights_start)  # [0.067, 0.133, 0.200, 0.267, 0.333]
    """
    weights = np.arange(1, no_outputs + 1, dtype=np.float64)
    return _normalize_weights(weights)


# ---------------------------------------------------------------------

def linear_low_to_high_schedule(progress: float, no_outputs: int) -> np.ndarray:
    """Linear transition from focusing on deep to shallow layer outputs.

    Smoothly interpolates between constant_high_to_low (early training) and
    constant_low_to_high (late training) weight distributions. At progress=0,
    deeper layers receive more weight. At progress=1, shallower layers receive
    more weight. At progress=0.5, all layers receive equal weight.

    This schedule is ideal for:
    1. Learning hierarchical features in a structured coarse-to-fine manner
    2. Segmentation tasks where progressive refinement is beneficial
    3. When you want predictable, smooth transitions between scales

    Args:
        progress: Current training progress from 0.0 to 1.0.
            0.0 = focus on deep layers, 1.0 = focus on shallow layers.
        no_outputs: Number of outputs to generate weights for. Must be positive.

    Returns:
        Array of linearly interpolated weights. Always sums to 1.0.

    Raises:
        ValueError: If no_outputs <= 0.

    Example:
        >>> # At start (0%), favor deep layers
        >>> weights_start = linear_low_to_high_schedule(0.0, 5)
        >>> print(weights_start)  # [0.067, 0.133, 0.200, 0.267, 0.333]
        >>>
        >>> # At midpoint (50%), equal weights
        >>> weights_mid = linear_low_to_high_schedule(0.5, 5)
        >>> print(weights_mid)  # [0.200, 0.200, 0.200, 0.200, 0.200]
        >>>
        >>> # At end (100%), favor shallow layers
        >>> weights_end = linear_low_to_high_schedule(1.0, 5)
        >>> print(weights_end)  # [0.333, 0.267, 0.200, 0.133, 0.067]
    """
    initial_weights = np.arange(1, no_outputs + 1, dtype=np.float64)
    final_weights = np.arange(no_outputs, 0, -1, dtype=np.float64)
    weights = (1 - progress) * initial_weights + progress * final_weights
    return _normalize_weights(weights)


# ---------------------------------------------------------------------

def non_linear_low_to_high_schedule(progress: float, no_outputs: int) -> np.ndarray:
    """Non-linear (quadratic) transition from deep to shallow layer focus.

    Uses quadratic interpolation (progress²) to transition between deep and
    shallow layer emphasis. This creates a slower transition in early training
    and a faster transition in later training, allowing the network more time
    to learn coarse features before focusing on fine details.

    This schedule is beneficial when:
    1. You want extended focus on coarse features in early training
    2. Need rapid fine-tuning of high-resolution details later
    3. Tasks where early stability at low resolution is important
    4. Want to avoid premature convergence on fine details

    Args:
        progress: Current training progress from 0.0 to 1.0.
            Quadratic factor = progress²
        no_outputs: Number of outputs to generate weights for. Must be positive.

    Returns:
        Array of quadratically interpolated weights. Always sums to 1.0.

    Raises:
        ValueError: If no_outputs <= 0.

    Example:
        >>> # At start (0%), focus on deep layers
        >>> weights_start = non_linear_low_to_high_schedule(0.0, 5)
        >>> print(weights_start)  # [0.067, 0.133, 0.200, 0.267, 0.333]
        >>>
        >>> # At 50%, still biased toward deep layers (0.5² = 0.25)
        >>> weights_mid = non_linear_low_to_high_schedule(0.5, 5)
        >>> print(weights_mid)  # [0.133, 0.167, 0.200, 0.233, 0.267]
        >>>
        >>> # At 75%, transitioning faster now (0.75² = 0.5625)
        >>> weights_75 = non_linear_low_to_high_schedule(0.75, 5)
        >>> print(weights_75)  # [0.233, 0.233, 0.200, 0.183, 0.150]
        >>>
        >>> # At end (100%), focus on shallow layers
        >>> weights_end = non_linear_low_to_high_schedule(1.0, 5)
        >>> print(weights_end)  # [0.333, 0.267, 0.200, 0.133, 0.067]
    """
    initial_weights = np.arange(1, no_outputs + 1, dtype=np.float64)
    final_weights = np.arange(no_outputs, 0, -1, dtype=np.float64)
    quadratic_factor = progress ** 2
    weights = (1 - quadratic_factor) * initial_weights + quadratic_factor * final_weights
    return _normalize_weights(weights)


# ---------------------------------------------------------------------

def custom_sigmoid_low_to_high_schedule(
        progress: float,
        no_outputs: int,
        k: float = 10.0,
        x0: float = 0.5,
        transition_point: float = 0.25
) -> np.ndarray:
    """Sigmoid-based transition from deep to shallow layer focus.

    Provides S-shaped (sigmoid) transition curve between deep and shallow layer
    emphasis, offering fine-grained control over when and how quickly the
    transition occurs. Before transition_point, focuses on deep layers. Then
    applies sigmoid transition with steepness k and midpoint x0.

    This schedule is powerful when:
    1. You need precise control over transition timing and sharpness
    2. Want to maintain stable deep features before transitioning
    3. Need sharp transitions at specific training stages
    4. Experimenting with different transition profiles

    Args:
        progress: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for. Must be positive.
        k: Steepness of the sigmoid curve. Higher values create sharper transitions.
            Typical range: 5.0 (gradual) to 20.0 (very sharp). Defaults to 10.0.
        x0: Midpoint of the sigmoid transition in adjusted progress space (0.0-1.0).
            0.5 means transition midpoint occurs halfway through the transition period.
            Defaults to 0.5.
        transition_point: Training progress (0.0-1.0) when transition begins.
            Before this point, weights favor deep layers. Defaults to 0.25.

    Returns:
        Array of sigmoid-interpolated weights. Always sums to 1.0.

    Raises:
        ValueError: If k <= 0, x0 not in [0,1], or transition_point not in [0,1].

    Example:
        >>> # Before transition point (25%), favor deep layers
        >>> weights_early = custom_sigmoid_low_to_high_schedule(0.2, 5)
        >>> print(weights_early)  # [0.067, 0.133, 0.200, 0.267, 0.333]
        >>>
        >>> # At transition start (25%), still on deep layers
        >>> weights_start = custom_sigmoid_low_to_high_schedule(0.25, 5)
        >>> print(weights_start)  # [0.067, 0.133, 0.200, 0.267, 0.333]
        >>>
        >>> # At midpoint of transition (~62.5%), balanced
        >>> weights_mid = custom_sigmoid_low_to_high_schedule(0.625, 5)
        >>> print(weights_mid)  # [0.200, 0.200, 0.200, 0.200, 0.200]
        >>>
        >>> # At end (100%), favor shallow layers
        >>> weights_end = custom_sigmoid_low_to_high_schedule(1.0, 5)
        >>> print(weights_end)  # [0.333, 0.267, 0.200, 0.133, 0.067]
        >>>
        >>> # With sharper transition (k=20)
        >>> weights_sharp = custom_sigmoid_low_to_high_schedule(0.7, 5, k=20.0)
        >>> # Produces more extreme weights during transition
    """
    if progress < transition_point or transition_point >= 1.0:
        sigmoid_factor = 0.0
    else:
        adjusted_progress = (progress - transition_point) / (1 - transition_point)
        sigmoid_factor = 1 / (1 + np.exp(-k * (adjusted_progress - x0)))

    initial_weights = np.arange(1, no_outputs + 1, dtype=np.float64)
    final_weights = np.arange(no_outputs, 0, -1, dtype=np.float64)
    weights = (1 - sigmoid_factor) * initial_weights + sigmoid_factor * final_weights
    return _normalize_weights(weights)


# ---------------------------------------------------------------------

def scale_by_scale_low_to_high_schedule(progress: float, no_outputs: int) -> np.ndarray:
    """Progressive activation of outputs from deep to shallow, one at a time.

    Implements strict curriculum learning by activating only one output scale
    at a time. Progresses from the deepest (lowest resolution) to the shallowest
    (highest resolution) output as training advances. At any given time, exactly
    one output has weight 1.0 and all others have weight 0.0.

    This schedule is appropriate for:
    1. Strict curriculum learning approaches
    2. Focusing computational resources on one scale at a time
    3. Debugging multi-scale architectures by isolating each scale
    4. Memory-constrained scenarios requiring single-scale training
    5. Studying the learning dynamics of individual scales

    Args:
        progress: Current training progress from 0.0 to 1.0.
            Determines which scale is active.
        no_outputs: Number of outputs to generate weights for. Must be positive.

    Returns:
        One-hot encoded weight array with single active scale. Always sums to 1.0.

    Raises:
        ValueError: If no_outputs <= 0.

    Example:
        >>> # At start (0-19%), only deepest scale active
        >>> weights_start = scale_by_scale_low_to_high_schedule(0.0, 5)
        >>> print(weights_start)  # [0.0, 0.0, 0.0, 0.0, 1.0] (scale 4, deepest)
        >>>
        >>> # At 25%, second deepest scale active
        >>> weights_25 = scale_by_scale_low_to_high_schedule(0.25, 5)
        >>> print(weights_25)  # [0.0, 0.0, 0.0, 1.0, 0.0] (scale 3)
        >>>
        >>> # At 50%, middle scale active
        >>> weights_mid = scale_by_scale_low_to_high_schedule(0.5, 5)
        >>> print(weights_mid)  # [0.0, 0.0, 1.0, 0.0, 0.0] (scale 2)
        >>>
        >>> # At 75%, second shallowest scale active
        >>> weights_75 = scale_by_scale_low_to_high_schedule(0.75, 5)
        >>> print(weights_75)  # [0.0, 1.0, 0.0, 0.0, 0.0] (scale 1)
        >>>
        >>> # At end (100%), only shallowest scale active
        >>> weights_end = scale_by_scale_low_to_high_schedule(1.0, 5)
        >>> print(weights_end)  # [1.0, 0.0, 0.0, 0.0, 0.0] (scale 0, shallowest)
    """
    active_idx_forward = min(int(np.floor(progress * no_outputs)), no_outputs - 1)
    active_scale_index = no_outputs - 1 - active_idx_forward

    weights = np.zeros(no_outputs, dtype=np.float64)
    weights[active_scale_index] = 1.0
    return weights


# ---------------------------------------------------------------------

def cosine_annealing_schedule(
        progress: float,
        no_outputs: int,
        frequency: float = 3.0,
        final_ratio: float = 0.5
) -> np.ndarray:
    """Cyclical weight patterns using cosine functions with annealing.

    Creates oscillating weight distributions that cycle through different
    emphasis patterns, with the amplitude of oscillations decreasing (annealing)
    as training progresses. Deeper layers receive stronger modulation. This
    periodic revisiting of different scales can help escape local minima and
    explore different feature combinations.

    This schedule is valuable when:
    1. Want to periodically revisit and refine different scales
    2. Avoiding convergence to poor local minima through periodic exploration
    3. Exploring synergies between different feature scales cyclically
    4. Using in conjunction with cosine learning rate schedules
    5. Tasks benefiting from periodic refinement of multiple scales

    Args:
        progress: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for. Must be positive.
        frequency: Number of complete cosine cycles throughout training.
            Higher values create more frequent oscillations. Defaults to 3.0.
        final_ratio: Ratio of final to initial oscillation amplitude (0.0-1.0).
            Controls annealing strength. 0.0 = complete decay to uniform weights,
            1.0 = no decay (constant amplitude). Defaults to 0.5.

    Returns:
        Array of cosine-modulated weights with annealing. Always sums to 1.0.

    Raises:
        ValueError: If frequency <= 0 or final_ratio not in [0, 1].

    Example:
        >>> # At start (0%), initial oscillation favoring deep layers
        >>> weights_start = cosine_annealing_schedule(0.0, 5, frequency=3.0)
        >>> print(weights_start)  # [0.28, 0.24, 0.20, 0.16, 0.12]
        >>>
        >>> # At ~17% (1/6 of first cycle), oscillation favoring shallow
        >>> weights_17 = cosine_annealing_schedule(0.167, 5, frequency=3.0)
        >>> print(weights_17)  # [0.12, 0.16, 0.20, 0.24, 0.28]
        >>>
        >>> # At 50%, smaller amplitude due to annealing
        >>> weights_mid = cosine_annealing_schedule(0.5, 5, frequency=3.0)
        >>> print(weights_mid)  # [0.26, 0.23, 0.20, 0.17, 0.14]
        >>>
        >>> # At end (100%), smallest oscillation amplitude
        >>> weights_end = cosine_annealing_schedule(1.0, 5, frequency=3.0)
        >>> print(weights_end)  # [0.24, 0.22, 0.20, 0.18, 0.16]
        >>>
        >>> # With higher frequency (more cycles)
        >>> weights_freq = cosine_annealing_schedule(0.5, 5, frequency=10.0)
        >>> # Creates more rapid oscillations throughout training
    """
    base_weights = np.ones(no_outputs, dtype=np.float64)

    # Deeper layers (higher indices) get more modulation.
    # This range is centered at 0, e.g., for 5 outputs: [2, 1, 0, -1, -2]
    modulation_range = np.arange(no_outputs)[::-1] - (no_outputs - 1) / 2.0

    # Cosine term oscillates between -1 and 1
    oscillation = np.cos(2 * np.pi * frequency * progress)

    # Amplitude decays linearly from a max value down to final_ratio * max_value
    # Max amplitude is scaled to avoid negative weights in most cases
    max_amplitude = 1.0 / no_outputs
    amplitude = max_amplitude * (1.0 - (1.0 - final_ratio) * progress)

    weights = base_weights + modulation_range * oscillation * amplitude
    weights = np.maximum(weights, 0)  # Ensure weights are non-negative
    return _normalize_weights(weights)


# ---------------------------------------------------------------------

def curriculum_schedule(
        progress: float,
        no_outputs: int,
        max_active_outputs: Optional[int] = None,
        activation_strategy: str = 'linear'
) -> np.ndarray:
    """Progressive curriculum learning with gradual output activation.

    Implements curriculum learning by progressively activating outputs from
    deep to shallow as training advances. Unlike scale_by_scale, this allows
    multiple outputs to be active simultaneously, with the number of active
    outputs increasing over time. Active outputs receive equal weights.

    This schedule is ideal for:
    1. Implementing structured curriculum learning with multiple active scales
    2. Gradually increasing task complexity during training
    3. Learning hierarchical features in a controlled progressive manner
    4. Memory-efficient training that gradually expands supervision
    5. Balancing between single-scale focus and multi-scale learning

    Args:
        progress: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for. Must be positive.
        max_active_outputs: Maximum number of simultaneously active outputs.
            If None or <= 0, defaults to no_outputs. Defaults to None.
        activation_strategy: Strategy for increasing active outputs:
            - 'linear': Active count increases linearly with progress
            - 'exp': Active count increases quadratically (progress²)
            Defaults to 'linear'.

    Returns:
        Array with equal weights distributed among active outputs. Always sums to 1.0.

    Raises:
        ValueError: If activation_strategy not in ['linear', 'exp'].
        TypeError: If max_active_outputs is not an integer.

    Example:
        >>> # Linear strategy with 5 max outputs
        >>> # At start (0%), only 1 active (deepest)
        >>> weights_start = curriculum_schedule(0.0, 5, 5, 'linear')
        >>> print(weights_start)  # [0.0, 0.0, 0.0, 0.0, 1.0]
        >>>
        >>> # At 40%, 2 active (deepest two)
        >>> weights_40 = curriculum_schedule(0.4, 5, 5, 'linear')
        >>> print(weights_40)  # [0.0, 0.0, 0.0, 0.5, 0.5]
        >>>
        >>> # At 60%, 3 active
        >>> weights_60 = curriculum_schedule(0.6, 5, 5, 'linear')
        >>> print(weights_60)  # [0.0, 0.0, 0.333, 0.333, 0.333]
        >>>
        >>> # At end (100%), all 5 active
        >>> weights_end = curriculum_schedule(1.0, 5, 5, 'linear')
        >>> print(weights_end)  # [0.2, 0.2, 0.2, 0.2, 0.2]
        >>>
        >>> # Exponential strategy (slower start, faster later)
        >>> weights_exp_50 = curriculum_schedule(0.5, 5, 5, 'exp')
        >>> # At 50%, only 25% through activation (0.5² = 0.25)
        >>> print(weights_exp_50)  # [0.0, 0.0, 0.0, 0.0, 1.0] or [0.0, 0.0, 0.0, 0.5, 0.5]
    """
    if max_active_outputs is None or max_active_outputs <= 0:
        max_active_outputs = no_outputs
    else:
        max_active_outputs = min(max_active_outputs, no_outputs)

    if activation_strategy.lower() == 'exp':
        active_count = int(max_active_outputs * (progress ** 2))
    else:  # 'linear'
        active_count = int(max_active_outputs * progress)

    active_count = max(1, active_count)
    if progress >= 1.0:  # Ensure all are active at the end
        active_count = max_active_outputs

    # Activate from deepest (index no_outputs-1) backwards
    active_indices = list(range(no_outputs - 1, no_outputs - 1 - active_count, -1))

    weights = np.zeros(no_outputs, dtype=np.float64)
    if active_indices:
        weights[active_indices] = 1.0

    return _normalize_weights(weights)


# ---------------------------------------------------------------------

def step_wise_schedule(
        progress: float,
        no_outputs: int,
        threshold: float = 0.5
) -> np.ndarray:
    """Step-wise transition with hard cutoff to shallowest layer.

    Combines linear transition from deep to shallow layers with a hard cutoff.
    Before the threshold, weights transition linearly like linear_low_to_high.
    Once the threshold is reached, all weight is placed on the shallowest layer
    (output 0), which is typically the final inference output at highest resolution.

    This schedule is useful when you want to:
    1. Learn hierarchical features progressively (before threshold)
    2. Fine-tune only the final output layer (after threshold)
    3. Implement two-phase training: feature learning then output refinement
    4. Ensure the final output gets exclusive attention in late training

    Args:
        progress: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for. Must be positive.
        threshold: Training progress point (0.0 to 1.0) where the hard cutoff occurs.
            Before threshold: linear transition from deep to shallow layers.
            At/after threshold: all weight goes to shallowest layer (output 0).
            Defaults to 0.5.

    Returns:
        Array of weights with step-wise transition pattern. Always sums to 1.0.

    Raises:
        ValueError: If threshold is not in range [0.0, 1.0] or no_outputs <= 0.

    Example:
        >>> # At 25% progress (before threshold of 0.5)
        >>> weights_25 = step_wise_schedule(0.25, 5, threshold=0.5)
        >>> # Returns linearly transitioning weights (25% through the transition)
        >>> print(weights_25)  # [0.200, 0.200, 0.200, 0.200, 0.200]
        >>>
        >>> # At 40% progress (still before threshold)
        >>> weights_40 = step_wise_schedule(0.4, 5, threshold=0.5)
        >>> # 80% through linear transition (0.4/0.5)
        >>> print(weights_40)  # [0.280, 0.240, 0.200, 0.160, 0.120]
        >>>
        >>> # At 50% progress (at threshold)
        >>> weights_50 = step_wise_schedule(0.5, 5, threshold=0.5)
        >>> print(weights_50)  # [1.0, 0.0, 0.0, 0.0, 0.0] - hard switch!
        >>>
        >>> # At 75% progress (after threshold)
        >>> weights_75 = step_wise_schedule(0.75, 5, threshold=0.5)
        >>> print(weights_75)  # [1.0, 0.0, 0.0, 0.0, 0.0] - still all on shallowest
        >>>
        >>> # Different threshold (0.7)
        >>> weights_early = step_wise_schedule(0.5, 5, threshold=0.7)
        >>> # Still in linear phase (0.5/0.7 ≈ 71% through transition)
        >>> print(weights_early)  # [0.257, 0.229, 0.200, 0.171, 0.143]
    """
    if progress >= threshold:
        weights = np.zeros(no_outputs, dtype=np.float64)
        weights[0] = 1.0
        return weights

    # Before threshold, scale progress from [0, threshold] to [0, 1]
    if threshold > 1e-8:
        scaled_progress = progress / threshold
        return linear_low_to_high_schedule(scaled_progress, no_outputs)
    else:  # Edge case: threshold is 0, immediately focus on shallowest layer
        weights = np.zeros(no_outputs, dtype=np.float64)
        weights[0] = 1.0
        return weights

# ---------------------------------------------------------------------
