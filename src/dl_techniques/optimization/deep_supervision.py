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

Available Schedules
-------------------

1. **constant_equal**
   Description:
     Equal weighting for all outputs throughout training. Provides uniform
     attention to all scales regardless of training progress.

   Use When:
     - All scales are equally important for your task
     - You want balanced supervision across all depths
     - Initial baseline experiments to understand multi-scale behavior

   Parameters: None

   Behavior:
     All outputs receive weight = 1/n at all training stages

   Example Output (5 outputs):
     Progress 0%:   [0.20, 0.20, 0.20, 0.20, 0.20]
     Progress 50%:  [0.20, 0.20, 0.20, 0.20, 0.20]
     Progress 100%: [0.20, 0.20, 0.20, 0.20, 0.20]

2. **constant_low_to_high**
   Description:
     Constant weighting that favors higher resolution (shallower) outputs
     throughout training. Weights increase linearly from deepest to shallowest.

   Use When:
     - Fine details are more important than coarse features
     - Working on tasks requiring high spatial precision
     - You want consistent emphasis on output resolution

   Parameters: None

   Behavior:
     Output i receives weight proportional to (i + 1)

   Example Output (5 outputs):
     Progress 0%:   [0.33, 0.27, 0.20, 0.13, 0.07]
     Progress 50%:  [0.33, 0.27, 0.20, 0.13, 0.07]
     Progress 100%: [0.33, 0.27, 0.20, 0.13, 0.07]

3. **constant_high_to_low**
   Description:
     Constant weighting that favors deeper layer outputs throughout training.
     Weights decrease linearly from deepest to shallowest.

   Use When:
     - Coarse semantic features are more important
     - Working with hierarchical feature learning
     - You want to emphasize low-resolution representations

   Parameters: None

   Behavior:
     Output i receives weight proportional to (n - i)

   Example Output (5 outputs):
     Progress 0%:   [0.07, 0.13, 0.20, 0.27, 0.33]
     Progress 50%:  [0.07, 0.13, 0.20, 0.27, 0.33]
     Progress 100%: [0.07, 0.13, 0.20, 0.27, 0.33]

4. **linear_low_to_high**
   Description:
     Linear transition from focusing on deep layers to focusing on shallow
     layers as training progresses. The weight shift is smooth and gradual.

   Use When:
     - Learning hierarchical features in a structured manner
     - Segmentation tasks where coarse-to-fine learning is beneficial
     - You want predictable, smooth transitions between scales

   Parameters: None

   Behavior:
     Linearly interpolates between constant_high_to_low and constant_low_to_high
     Weight_i(t) = (1-t) * W_deep[i] + t * W_shallow[i]

   Example Output (5 outputs):
     Progress 0%:   [0.07, 0.13, 0.20, 0.27, 0.33]  # Deep focus
     Progress 50%:  [0.20, 0.20, 0.20, 0.20, 0.20]  # Balanced
     Progress 100%: [0.33, 0.27, 0.20, 0.13, 0.07]  # Shallow focus

5. **non_linear_low_to_high**
   Description:
     Non-linear (quadratic) transition from deep to shallow layer focus.
     Slower transition early, faster transition later in training.

   Use When:
     - You want more time focusing on coarse features early
     - Rapid fine-tuning of details in later training stages
     - Tasks where early stability is important

   Parameters: None

   Behavior:
     Uses quadratic interpolation: factor = progress²
     Weight_i(t) = (1-t²) * W_deep[i] + t² * W_shallow[i]

   Example Output (5 outputs):
     Progress 0%:   [0.07, 0.13, 0.20, 0.27, 0.33]  # Deep focus
     Progress 50%:  [0.13, 0.17, 0.20, 0.23, 0.27]  # Still deeper bias
     Progress 100%: [0.33, 0.27, 0.20, 0.13, 0.07]  # Shallow focus

6. **custom_sigmoid_low_to_high**
   Description:
     Sigmoid-based transition providing S-shaped curve from deep to shallow
     focus. Offers fine control over transition timing and steepness.

   Use When:
     - You need precise control over when the transition occurs
     - Want sharp transitions at specific training stages
     - Experimenting with different transition profiles

   Parameters:
     - k (float, default=10.0): Steepness of sigmoid. Higher = sharper transition
     - x0 (float, default=0.5): Midpoint of transition (0.0-1.0)
     - transition_point (float, default=0.25): When transition begins (0.0-1.0)

   Behavior:
     Sigmoid factor: 1 / (1 + exp(-k * (adjusted_progress - x0)))
     Adjusted progress scales from transition_point to 1.0

   Example Output (5 outputs, k=10, x0=0.5, transition_point=0.25):
     Progress 0%:   [0.07, 0.13, 0.20, 0.27, 0.33]  # Before transition
     Progress 25%:  [0.07, 0.13, 0.20, 0.27, 0.33]  # At transition start
     Progress 50%:  [0.20, 0.20, 0.20, 0.20, 0.20]  # Midpoint
     Progress 75%:  [0.32, 0.26, 0.20, 0.14, 0.08]  # Transitioning
     Progress 100%: [0.33, 0.27, 0.20, 0.13, 0.07]  # Shallow focus

7. **scale_by_scale_low_to_high**
   Description:
     Progressive activation of outputs from deep to shallow, with only one
     scale active at a time. Implements strict curriculum learning.

   Use When:
     - Implementing curriculum learning approaches
     - You want to focus on one scale at a time
     - Debugging multi-scale architectures
     - Memory constraints require single-scale training

   Parameters: None

   Behavior:
     Active scale index = floor(progress * n_outputs)
     Only one output has weight = 1.0, others = 0.0

   Example Output (5 outputs):
     Progress 0%:   [1.00, 0.00, 0.00, 0.00, 0.00]  # Scale 0
     Progress 25%:  [0.00, 1.00, 0.00, 0.00, 0.00]  # Scale 1
     Progress 50%:  [0.00, 0.00, 1.00, 0.00, 0.00]  # Scale 2
     Progress 75%:  [0.00, 0.00, 0.00, 1.00, 0.00]  # Scale 3
     Progress 100%: [0.00, 0.00, 0.00, 0.00, 1.00]  # Scale 4

8. **cosine_annealing**
   Description:
     Cyclical weight patterns using cosine functions. Creates oscillations
     that can help escape local minima and explore feature combinations.

   Use When:
     - Want to periodically revisit different scales
     - Avoiding convergence to poor local minima
     - Exploring different feature combinations cyclically
     - Used in conjunction with cosine learning rate schedules

   Parameters:
     - frequency (float, default=3.0): Number of complete cosine cycles
     - final_ratio (float, default=0.8): Ratio between final/initial ranges

   Behavior:
     Cosine factor: 0.5 * (1 + cos(2π * frequency * progress))
     Deeper layers get more modulation

   Example Output (5 outputs, frequency=3.0, final_ratio=0.8):
     Progress 0%:   [0.15, 0.17, 0.19, 0.22, 0.27]  # Start of cycle
     Progress 17%:  [0.22, 0.21, 0.20, 0.19, 0.18]  # Mid cycle
     Progress 33%:  [0.16, 0.17, 0.19, 0.21, 0.26]  # Cycle restart
     Progress 100%: [0.17, 0.18, 0.19, 0.21, 0.25]  # End

9. **curriculum**
   Description:
     Progressive activation of multiple outputs implementing curriculum
     learning. Gradually activates more outputs as training advances.

   Use When:
     - Implementing structured curriculum learning
     - Want multiple active outputs (unlike scale_by_scale)
     - Learning hierarchical features progressively
     - Controlling complexity growth during training

   Parameters:
     - max_active_outputs (int, default=n_outputs): Max simultaneous active outputs
     - activation_strategy (str, default='linear'): 'linear' or 'exp'

   Behavior:
     Linear: active_count = max_active * progress
     Exponential: active_count = max_active * progress²
     Active outputs get equal weights, normalized

   Example Output (5 outputs, max_active=5, strategy='linear'):
     Progress 0%:   [1.00, 0.00, 0.00, 0.00, 0.00]  # 1 active
     Progress 25%:  [0.50, 0.50, 0.00, 0.00, 0.00]  # 2 active
     Progress 50%:  [0.33, 0.33, 0.33, 0.00, 0.00]  # 3 active
     Progress 75%:  [0.25, 0.25, 0.25, 0.25, 0.00]  # 4 active
     Progress 100%: [0.20, 0.20, 0.20, 0.20, 0.20]  # 5 active

10. **step_wise**
    Description:
      Combines linear transition with hard cutoff. Linearly transitions from
      deep to shallow until threshold, then focuses entirely on shallowest layer.

    Use When:
      - Want smooth transition followed by fine-tuning phase
      - Need to dedicate final training to output layer refinement
      - Two-phase training: feature learning then fine-tuning
      - Want to ensure final output gets exclusive attention late in training

    Parameters:
      - threshold (float, default=0.5): Progress point for hard cutoff (0.0-1.0)

    Behavior:
      Before threshold: linear_low_to_high scaled to [0, threshold]
      At/after threshold: [1.0, 0.0, 0.0, ..., 0.0]

    Example Output (5 outputs, threshold=0.5):
      Progress 0%:   [0.07, 0.13, 0.20, 0.27, 0.33]  # Deep focus
      Progress 25%:  [0.20, 0.20, 0.20, 0.20, 0.20]  # Balanced (50% of threshold)
      Progress 50%:  [1.00, 0.00, 0.00, 0.00, 0.00]  # Hard switch!
      Progress 75%:  [1.00, 0.00, 0.00, 0.00, 0.00]  # Only shallowest
      Progress 100%: [1.00, 0.00, 0.00, 0.00, 0.00]  # Only shallowest

Schedule Selection Guide
------------------------

Task Type Recommendations:

**Medical Image Segmentation:**
  - Start with: linear_low_to_high or step_wise (threshold=0.7)
  - Alternative: custom_sigmoid_low_to_high for controlled transitions

**Natural Image Segmentation:**
  - Start with: linear_low_to_high or non_linear_low_to_high
  - Alternative: cosine_annealing for robust feature learning

**Object Detection:**
  - Start with: constant_low_to_high or step_wise (threshold=0.6)
  - Alternative: curriculum with exponential strategy

**Fine-Grained Classification:**
  - Start with: step_wise (threshold=0.5) or custom_sigmoid_low_to_high
  - Alternative: scale_by_scale_low_to_high for debugging

**General Guidelines:**
  - Use constant_* schedules for baseline experiments
  - Use linear/non_linear for standard training
  - Use step_wise when final output quality is critical
  - Use curriculum for structured learning
  - Use cosine_annealing with other cyclical techniques

Usage Examples
--------------

Basic Linear Transition::

    >>> config = {
    ...     "type": "linear_low_to_high",
    ...     "config": {}
    ... }
    >>> weight_scheduler = schedule_builder(config, 5)  # For 5 outputs
    >>>
    >>> # Get weights at 30% through training
    >>> weights = weight_scheduler(0.3)  # Returns numpy array of 5 weights
    >>> print(f"Weights sum: {np.sum(weights)}")  # Always 1.0

Custom Sigmoid Transition::

    >>> sigmoid_config = {
    ...     "type": "custom_sigmoid_low_to_high",
    ...     "config": {
    ...         "k": 10.0,           # Sharp transition
    ...         "x0": 0.5,           # Centered at 50%
    ...         "transition_point": 0.25  # Start at 25%
    ...     }
    ... }
    >>> sigmoid_scheduler = schedule_builder(sigmoid_config, 3)
    >>> weights = sigmoid_scheduler(0.6)

Step-wise with Threshold::

    >>> stepwise_config = {
    ...     "type": "step_wise",
    ...     "config": {
    ...         "threshold": 0.7  # Switch to final output at 70%
    ...     }
    ... }
    >>> stepwise_scheduler = schedule_builder(stepwise_config, 4)
    >>>
    >>> # Before threshold: gradual transition
    >>> early_weights = stepwise_scheduler(0.35)  # Transitioning
    >>>
    >>> # After threshold: focus on output 0
    >>> late_weights = stepwise_scheduler(0.8)  # [1.0, 0.0, 0.0, 0.0]

Curriculum Learning::

    >>> curriculum_config = {
    ...     "type": "curriculum",
    ...     "config": {
    ...         "max_active_outputs": 4,
    ...         "activation_strategy": "exp"  # Exponential activation
    ...     }
    ... }
    >>> curriculum_scheduler = schedule_builder(curriculum_config, 5)

Inverted Weight Order::

    >>> # For architectures where output 0 is deepest layer
    >>> config = {"type": "linear_low_to_high", "config": {}}
    >>> inverted_scheduler = schedule_builder(config, 5, invert_order=True)
    >>> weights = inverted_scheduler(0.5)  # Order reversed

Integration with Training Loop
-------------------------------

Typical usage in Keras training::

    import keras

    # Initialize scheduler
    config = {"type": "step_wise", "config": {"threshold": 0.6}}
    weight_scheduler = schedule_builder(config, num_outputs=4)

    # Custom training loop
    @keras.function
    def train_step(x, y, epoch, total_epochs):
        # Calculate training progress
        progress = epoch / total_epochs

        # Get current weights
        ds_weights = weight_scheduler(progress)

        with tf.GradientTape() as tape:
            # Forward pass with multiple outputs
            outputs = model(x, training=True)

            # Weighted multi-scale loss
            total_loss = 0.0
            for i, (output, weight) in enumerate(zip(outputs, ds_weights)):
                scale_loss = loss_fn(y, output)
                total_loss += weight * scale_loss

        # Backward pass
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return total_loss

Notes
-----
- All weight arrays sum to exactly 1.0 after normalization
- Progress values should be in range [0.0, 1.0]
- For inverted architectures, use invert_order=True
- Schedules are stateless - same progress always returns same weights
- Thread-safe for parallel training scenarios

References
----------
- Deep Supervision: C.-Y. Lee et al., "Deeply-Supervised Nets" (2014)
- U-Net Architecture: O. Ronneberger et al., "U-Net: Convolutional Networks
  for Biomedical Image Segmentation" (2015)
- Curriculum Learning: Y. Bengio et al., "Curriculum Learning" (2009)

See Also
--------
dl_techniques.losses : Loss functions compatible with deep supervision
dl_techniques.models.unet : U-Net implementation with deep supervision support
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
    STEP_WISE = "step_wise"

# ---------------------------------------------------------------------

def schedule_builder(
        config: Dict[str, Union[str, Dict[str, Union[float, str]]]],
        no_outputs: int,
        invert_order: bool = False  # New parameter to invert weight order
) -> Callable[[float], np.ndarray]:
    """Build a deep supervision weight scheduler from configuration.

    Creates a callable that generates weight arrays for multiple outputs at different
    training progress stages. The weights determine the contribution of each scale
    to the final loss during training.

    In the context of U-Net architecture (default order):
    - Output 0: Final inference output (highest resolution)
    - Output (no_outputs-1): Deepest scale in the network (lowest resolution)
    - Outputs in between: Intermediate scales

    When invert_order=True, the indexing is reversed:
    - Output 0: Deepest scale in the network (lowest resolution)
    - Output (no_outputs-1): Final inference output (highest resolution)

    Args:
        config: Configuration dictionary containing the schedule type and parameters.
            Required keys:
                - type: Schedule type string (see ScheduleType enum)
            Optional keys:
                - config: Dictionary with schedule-specific parameters
        no_outputs: Number of outputs (scales) for which weights must be generated.
            Must be a positive integer.
        invert_order: If True, inverts the weight order so that output 0 corresponds
            to the deepest layer and output (no_outputs-1) to the shallowest.
            Defaults to False.

    Returns:
        A function that takes a float percentage (0.0 to 1.0) representing training
        progress and returns a numpy array of shape [no_outputs] with weights for
        each output. The weights always sum to 1.0.

    Raises:
        ValueError: If config is invalid, schedule_type is unknown, or no_outputs <= 0.
        TypeError: If config is not a dictionary or contains invalid types.

    Example:
        >>> # Linear transition with default order
        >>> config = {"type": "linear_low_to_high", "config": {}}
        >>> weight_scheduler = schedule_builder(config, 5, invert_order=False)
        >>> weights_at_50_percent = weight_scheduler(0.5)  # Returns weights for 5 outputs
        >>>
        >>> # Linear transition with inverted order
        >>> inverted_scheduler = schedule_builder(config, 5, invert_order=True)
        >>> inverted_weights = inverted_scheduler(0.5)  # Same logic, inverted array
        >>>
        >>> # Verify inversion relationship
        >>> assert np.allclose(weights_at_50_percent, inverted_weights[::-1])
    """
    # Validate input arguments
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")
    if not isinstance(no_outputs, int) or no_outputs <= 0:
        raise ValueError(f"no_outputs must be a positive integer, got {no_outputs}")
    if not isinstance(invert_order, bool):
        raise TypeError("invert_order must be a boolean")

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
        f"params: [{schedule_params}], invert_order: {invert_order}"
    )

    # Create the appropriate schedule function
    scheduler = None

    if schedule_type == ScheduleType.CONSTANT_EQUAL:
        scheduler = lambda progress: constant_equal_schedule(progress, no_outputs)

    elif schedule_type == ScheduleType.CONSTANT_LOW_TO_HIGH:
        scheduler = lambda progress: constant_low_to_high_schedule(progress, no_outputs)

    elif schedule_type == ScheduleType.CONSTANT_HIGH_TO_LOW:
        scheduler = lambda progress: constant_high_to_low_schedule(progress, no_outputs)

    elif schedule_type == ScheduleType.LINEAR_LOW_TO_HIGH:
        scheduler = lambda progress: linear_low_to_high_schedule(progress, no_outputs)

    elif schedule_type == ScheduleType.NON_LINEAR_LOW_TO_HIGH:
        scheduler = lambda progress: non_linear_low_to_high_schedule(progress, no_outputs)

    elif schedule_type == ScheduleType.CUSTOM_SIGMOID_LOW_TO_HIGH:
        # Extract and validate parameters for custom sigmoid
        k = schedule_params.get('k', 10.0)
        x0 = schedule_params.get('x0', 0.5)
        transition_point = schedule_params.get('transition_point', 0.25)

        scheduler = lambda progress: custom_sigmoid_low_to_high_schedule(
            progress, no_outputs, k, x0, transition_point
        )

    elif schedule_type == ScheduleType.SCALE_BY_SCALE_LOW_TO_HIGH:
        scheduler = lambda progress: scale_by_scale_low_to_high_schedule(
            progress, no_outputs
        )

    elif schedule_type == ScheduleType.COSINE_ANNEALING:
        # Extract and validate parameters for cosine annealing schedule
        frequency = schedule_params.get('frequency', 3.0)
        final_ratio = schedule_params.get('final_ratio', 0.8)

        scheduler = lambda progress: cosine_annealing_schedule(
            progress, no_outputs, frequency, final_ratio
        )

    elif schedule_type == ScheduleType.CURRICULUM:
        # Extract and validate parameters for curriculum schedule
        max_active_outputs = schedule_params.get('max_active_outputs', no_outputs)
        activation_strategy = schedule_params.get('activation_strategy', 'linear')

        scheduler = lambda progress: curriculum_schedule(
            progress, no_outputs, max_active_outputs, activation_strategy
        )

    elif schedule_type == ScheduleType.STEP_WISE:
        # Extract and validate parameters for step-wise schedule
        threshold = schedule_params.get('threshold', 0.5)

        # Validate threshold
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in range [0.0, 1.0], got {threshold}")

        scheduler = lambda progress: step_wise_schedule(
            progress, no_outputs, threshold
        )

    else:
        raise ValueError(
            f"Unknown deep supervision schedule_type: [{schedule_type}]. "
            f"Supported types: {[t.value for t in ScheduleType]}"
        )

    # Apply weight order inversion if requested
    if invert_order:
        return lambda progress: scheduler(progress)[::-1]
    else:
        return scheduler

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
    # Handle an edge case where all weights are zero
    if weights_sum == 0:
        return np.ones_like(weights) / len(weights)
    return weights / weights_sum


# ---------------------------------------------------------------------

def constant_equal_schedule(progress: float, no_outputs: int) -> np.ndarray:
    """Equal weighting for all outputs regardless of training progress.

    This schedule provides uniform attention to all output scales throughout
    training, which can be useful when all scales are equally important.

    Args:
        progress: Current training progress from 0.0 to 1.0 (unused).
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of equal weights, each equal to 1/no_outputs.
    """
    return np.ones(no_outputs, dtype=np.float64) / no_outputs

# ---------------------------------------------------------------------

def constant_low_to_high_schedule(progress: float, no_outputs: int) -> np.ndarray:
    """Constant weighting favoring higher resolution outputs.

    Weights increase linearly from the deepest layer to the final output,
    providing consistent emphasis on higher resolution features throughout training.

    Args:
        progress: Current training progress from 0.0 to 1.0 (unused).
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of weights increasing from deeper to shallower layers.
    """
    weights = np.arange(1, no_outputs + 1, dtype=np.float64)
    return _normalize_weights(weights)

# ---------------------------------------------------------------------

def constant_high_to_low_schedule(progress: float, no_outputs: int) -> np.ndarray:
    """Constant weighting favoring deeper layer outputs.

    Weights decrease linearly from the deepest layer to the final output,
    providing consistent emphasis on coarse, semantic features throughout training.

    Args:
        progress: Current training progress from 0.0 to 1.0 (unused).
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of weights decreasing from deeper to shallower layers.
    """
    weights = np.arange(no_outputs, 0, -1, dtype=np.float64)
    return _normalize_weights(weights)


# ---------------------------------------------------------------------

def linear_low_to_high_schedule(progress: float, no_outputs: int) -> np.ndarray:
    """Linear transition from focusing on deep layers to focusing on shallow layers.

    As training progresses, weight shifts from deeper to shallower layers linearly.
    This is often effective for segmentation tasks where coarse features should
    be learned first, followed by fine-grained details.

    Args:
        progress: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of weights that transition linearly based on training progress.
    """
    # Initial weights favor deeper layers (higher indices)
    initial_weights = np.arange(1, no_outputs + 1, dtype=np.float64)

    # Final weights favor shallower layers (lower indices)
    final_weights = np.arange(no_outputs, 0, -1, dtype=np.float64)

    # Linear interpolation between initial and final weights
    weights = (1 - progress) * initial_weights + progress * final_weights

    return _normalize_weights(weights)

# ---------------------------------------------------------------------

def non_linear_low_to_high_schedule(progress: float, no_outputs: int) -> np.ndarray:
    """Non-linear transition from deep to shallow layer focus.

    Uses quadratic interpolation for smoother transition, with more gradual
    change early in training and faster change later. This can be beneficial
    when fine-tuning requires more aggressive reweighting.

    Args:
        progress: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of weights with quadratic transition pattern.
    """
    # Initial weights favor deeper layers (higher indices)
    initial_weights = np.arange(1, no_outputs + 1, dtype=np.float64)

    # Final weights favor shallower layers (lower indices)
    final_weights = np.arange(no_outputs, 0, -1, dtype=np.float64)

    # Quadratic interpolation - smoother transition in middle of training
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

    Uses a sigmoid function to create a smooth, S-shaped transition between
    layer importance. Provides fine control over the transition timing and steepness.

    Args:
        progress: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.
        k: Sigmoid steepness parameter. Higher values create sharper transitions.
        x0: Sigmoid midpoint parameter (0.0 to 1.0). Where the transition is centered.
        transition_point: Training percentage where transition begins (0.0 to 1.0).

    Returns:
        Array of weights with sigmoid-based transition pattern.
    """
    # Adjust percentage based on transition point
    adjusted_percentage = max(0, (progress - transition_point) / (1 - transition_point))

    # Sigmoid function: 1 / (1 + exp(-k(x - x0)))
    sigmoid_factor = 1 / (1 + np.exp(-k * (adjusted_percentage - x0)))

    # Initial weights favor deeper layers
    initial_weights = np.arange(no_outputs, 0, -1, dtype=np.float64)

    # Final weights favor shallower layers
    final_weights = np.arange(1, no_outputs + 1, dtype=np.float64)

    # Sigmoid interpolation between initial and final weights
    weights = (1 - sigmoid_factor) * initial_weights + sigmoid_factor * final_weights

    return _normalize_weights(weights)

# ---------------------------------------------------------------------

def scale_by_scale_low_to_high_schedule(progress: float, no_outputs: int) -> np.ndarray:
    """Progressive activation of outputs from deep to shallow.

    Gradually activates outputs from deep to shallow as training progresses,
    with only one scale being active at a time. This can be useful for
    curriculum learning approaches.

    Args:
        progress: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.

    Returns:
        Array of weights with only one active scale (weight = 1.0).
    """
    # Determine which scale is currently active based on progress
    active_scale_index = min(int(np.floor(progress * no_outputs)), no_outputs - 1)

    # Create weight array with only the active scale having weight 1.0
    weights = np.zeros(no_outputs, dtype=np.float64)
    weights[active_scale_index] = 1.0

    return weights

# ---------------------------------------------------------------------

def cosine_annealing_schedule(
    progress: float,
    no_outputs: int,
    frequency: float = 3.0,
    final_ratio: float = 0.8
) -> np.ndarray:
    """Cosine annealing schedule for weight distribution.

    Uses cosine function to create cyclical weight patterns during training,
    which can help escape local minima and explore different feature combinations.

    Args:
        progress: Current training progress from 0.0 to 1.0.
        no_outputs: Number of outputs to generate weights for.
        frequency: Number of complete cosine cycles during training.
        final_ratio: Ratio between final and initial layer weight ranges.

    Returns:
        Array of weights following cosine annealing pattern.
    """
    # Cosine factor oscillates between 0 and 1
    cosine_factor = 0.5 * (1 + np.cos(2 * np.pi * frequency * progress))

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

# ---------------------------------------------------------------------

def curriculum_schedule(
    progress: float,
    no_outputs: int,
    max_active_outputs: Optional[int] = None,
    activation_strategy: str = 'linear'
) -> np.ndarray:
    """Curriculum learning schedule activating outputs progressively.

    Implements curriculum learning by progressively activating more outputs
    as training advances. This can help models learn hierarchical features
    in a structured manner.

    Args:
        progress: Current training progress from 0.0 to 1.0.
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
        active_count = int(max_active_outputs * (progress ** 2))
    else:
        # Linear activation (default)
        active_count = int(max_active_outputs * progress)

    # Ensure at least one output is active
    active_count = max(1, active_count)

    # Activate outputs from deepest to shallowest (curriculum progression)
    active_indices = list(range(min(active_count, no_outputs)))

    # Set equal weights for active outputs
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
        >>> weights = step_wise_schedule(0.25, 4, threshold=0.5)
        >>> # Returns linearly transitioning weights favoring deeper layers
        >>> print(weights)  # e.g., [0.1, 0.2, 0.3, 0.4] (normalized)
        >>>
        >>> # At 50% progress (at threshold)
        >>> weights = step_wise_schedule(0.5, 4, threshold=0.5)
        >>> print(weights)  # [1.0, 0.0, 0.0, 0.0] - all weight on shallowest layer
        >>>
        >>> # At 75% progress (after threshold)
        >>> weights = step_wise_schedule(0.75, 4, threshold=0.5)
        >>> print(weights)  # [1.0, 0.0, 0.0, 0.0] - still all on shallowest layer
    """
    # Validate inputs
    if not isinstance(no_outputs, int) or no_outputs <= 0:
        raise ValueError(f"no_outputs must be a positive integer, got {no_outputs}")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in range [0.0, 1.0], got {threshold}")
    if not (0.0 <= progress <= 1.0):
        raise ValueError(f"progress must be in range [0.0, 1.0], got {progress}")

    # After threshold: all weight goes to shallowest layer (output 0)
    if progress >= threshold:
        weights = np.zeros(no_outputs, dtype=np.float64)
        weights[0] = 1.0
        return weights

    # Before threshold: linear transition scaled to the threshold range
    # Scale progress from [0, threshold] to [0, 1] for linear interpolation
    if threshold > 0:
        scaled_progress = progress / threshold
        return linear_low_to_high_schedule(progress=scaled_progress, no_outputs=no_outputs)
    else:
        # Edge case: threshold is 0, immediately go to shallowest layer
        weights = np.zeros(no_outputs, dtype=np.float64)
        weights[0] = 1.0
        return weights
