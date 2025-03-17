import numpy as np
from typing import Dict, Tuple, Callable, List, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.constants import *
from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# Helper functions for custom sigmoid schedule
# ---------------------------------------------------------------------

def custom_sigmoid(x: float, k: float = 10, x0: float = 0.5) -> float:
    """
    Custom sigmoid function for smooth, adjustable transitions.

    Parameters
    ----------
    x : float
        Input value
    k : float
        Controls the steepness of the sigmoid curve
    x0 : float
        Controls the midpoint of the transition

    Returns
    -------
    float
        Sigmoid output in range [0,1]
    """
    return 1 / (1 + np.exp(-k * (x - x0)))


def normalize(x: float, x_min: float, x_max: float) -> float:
    """
    Min-max normalization to ensure output is in range [0,1]

    Parameters
    ----------
    x : float
        Value to normalize
    x_min : float
        Minimum expected value
    x_max : float
        Maximum expected value

    Returns
    -------
    float
        Normalized value in range [0,1]
    """
    return (x - x_min) / (x_max - x_min)


def custom_function(x: float, k: float = 10, x0: float = 0.5) -> float:
    """
    Applies sigmoid and normalizes output to ensure range [0,1].

    Parameters
    ----------
    x : float
        Input value in range [0,1]
    k : float
        Controls sigmoid steepness
    x0 : float
        Controls sigmoid midpoint

    Returns
    -------
    float
        Normalized sigmoid output
    """
    y = custom_sigmoid(x, k, x0)
    y_normalized = normalize(y, custom_sigmoid(0, k, x0), custom_sigmoid(1, k, x0))
    return y_normalized


# ---------------------------------------------------------------------
# Schedule functions
# ---------------------------------------------------------------------

def constant_equal_schedule(percentage_done: float, no_outputs: int) -> np.ndarray:
    """
    Equal weights for all outputs, constant throughout training.

    Creates a uniform distribution where all outputs receive the same weight
    (1/no_outputs). This is the simplest baseline approach.

    For U-Net: Treats all decoder stages with equal importance.

    Parameters
    ----------
    percentage_done : float
        Training progress from 0.0 to 1.0 (not used in this schedule as weights are constant)
    no_outputs : int
        Number of outputs (scales) to generate weights for

    Returns
    -------
    np.ndarray
        Array of equal weights summing to 1.0
    """
    d = np.array([1.0, ] * no_outputs)
    d = d / np.sum(d)
    return d


def constant_low_to_high_schedule(percentage_done: float, no_outputs: int) -> np.ndarray:
    """
    Weights increase linearly from output1 to outputN, constant throughout training.

    Output weights are proportional to their index (1,2,3,...), creating a 
    distribution that favors deeper layers.

    For U-Net: Emphasizes deeper features (semantic information) over
    the final inference output throughout training.

    Parameters
    ----------
    percentage_done : float
        Training progress from 0.0 to 1.0 (not used in this schedule as weights are constant)
    no_outputs : int
        Number of outputs (scales) to generate weights for

    Returns
    -------
    np.ndarray
        Array of increasing weights summing to 1.0
    """
    d = np.array(list(range(1, no_outputs + 1))).astype(np.float32)
    d = d / np.sum(d)
    return d


def constant_high_to_low_schedule(percentage_done: float, no_outputs: int) -> np.ndarray:
    """
    Weights decrease linearly from output1 to outputN, constant throughout training.

    Output weights are inverse proportional to their index, creating a
    distribution that favors shallower layers.

    For U-Net: Emphasizes the final inference output (fine details) over
    deeper features throughout training.

    Parameters
    ----------
    percentage_done : float
        Training progress from 0.0 to 1.0 (not used in this schedule as weights are constant)
    no_outputs : int
        Number of outputs (scales) to generate weights for

    Returns
    -------
    np.ndarray
        Array of decreasing weights summing to 1.0
    """
    d = np.array(list(range(1, no_outputs + 1))).astype(np.float32)
    d = d / np.sum(d)
    d = d[::-1]  # Reverse the array to make weights decrease
    return d


def linear_low_to_high_schedule(percentage_done: float, no_outputs: int) -> np.ndarray:
    """
    Linear transition from emphasizing deep features to emphasizing inference output.

    At start of training (percentage_done=0): Weights increase from output1 to outputN
    At end of training (percentage_done=1): Weights decrease from output1 to outputN
    In between: Linear interpolation between these two states

    For U-Net: Gradually shifts focus from semantic/contextual information to fine details
    as training progresses. Midway through training, all outputs have equal weight.

    Parameters
    ----------
    percentage_done : float
        Training progress from 0.0 to 1.0
    no_outputs : int
        Number of outputs (scales) to generate weights for

    Returns
    -------
    np.ndarray
        Array of weights that evolve linearly during training
    """
    d_start = np.array(list(range(1, no_outputs + 1))).astype(np.float32)
    d_start = d_start / np.sum(d_start)
    d_end = d_start[::-1]  # End state is the reverse of start state
    # Linear interpolation between start and end states
    return d_start * (1.0 - percentage_done) + d_end * percentage_done


def non_linear_low_to_high_schedule(percentage_done: float, no_outputs: int) -> np.ndarray:
    """
    Non-linear transition from emphasizing deep features to emphasizing inference output.

    Similar to linear_low_to_high but uses a tanh function for smoother transitions.
    The tanh function creates more gradual changes at the beginning and end of training,
    with faster transitions in the middle.

    For U-Net: Provides a more natural shift from contextual information to fine details,
    with stabilized periods at the beginning and end of training.

    Parameters
    ----------
    percentage_done : float
        Training progress from 0.0 to 1.0
    no_outputs : int
        Number of outputs (scales) to generate weights for

    Returns
    -------
    np.ndarray
        Array of weights that evolve non-linearly during training
    """
    d_start = np.array(list(range(1, no_outputs + 1))).astype(np.float32)
    d_start = d_start / np.sum(d_start)
    d_end = d_start[::-1]  # End state is the reverse of start state
    # Use tanh to create a smooth, non-linear transition rate
    x = np.clip(np.tanh(2.5 * percentage_done), a_min=0.0, a_max=1.0)
    return d_start * (1.0 - x) + d_end * x


def custom_sigmoid_low_to_high_schedule(
        percentage_done: float,
        no_outputs: int,
        k: float = 10,
        x0: float = 0.5,
        transition_point: float = 0.25
) -> np.ndarray:
    """
    Two-phase training approach with focus transition at a specific point.

    Before the transition_point: Smooth transition from emphasizing deep features
    to emphasizing the inference output, controlled by a custom sigmoid.

    After the transition_point: Fixed distribution with 90% weight on the inference
    output and 10% distributed among all other outputs.

    For U-Net: Ensures the model first learns contextual information, then strongly
    focuses on refining the final output. Good for complex segmentation tasks.

    Parameters
    ----------
    percentage_done : float
        Training progress from 0.0 to 1.0
    no_outputs : int
        Number of outputs (scales) to generate weights for
    k : float
        Steepness of sigmoid transition
    x0 : float
        Midpoint of sigmoid transition
    transition_point : float
        Point in training (0-1) where the final distribution is reached

    Returns
    -------
    np.ndarray
        Array of weights that follow the custom sigmoid pattern
    """
    if percentage_done >= transition_point:
        # After transition point: 90% weight on output1, 10% distributed among others
        w = np.array([0.9] + [0.1 / (no_outputs - 1)] * (no_outputs - 1))
    else:
        # Before transition point: Smooth transition using custom sigmoid
        d_start = np.array(range(1, no_outputs + 1)) / np.sum(range(1, no_outputs + 1))
        d_end = np.array([0.9] + [0.1 / (no_outputs - 1)] * (no_outputs - 1))
        x = custom_function(percentage_done / transition_point, k, x0)
        w = d_start * (1 - x) + d_end * x
    return w / np.sum(w)


def scale_by_scale_low_to_high_schedule(percentage_done: float, no_outputs: int) -> np.ndarray:
    """
    Progressive focused training that systematically shifts emphasis across scales.

    Divides training into N equal intervals (where N is the number of outputs).
    In each interval, weight is primarily (75%) concentrated on one specific output,
    with remaining weight (25%) distributed equally among earlier outputs.

    As training progresses:
    - First interval: Focus on deepest output (outputN)
    - Second interval: Focus on output(N-1), with outputN zeroed out
    - And so on until final interval: Focus entirely on output1 (inference)

    For U-Net: Creates a structured curriculum that builds features from deep to shallow,
    systematically focusing on each scale. Effective for learning hierarchical features.

    Parameters
    ----------
    percentage_done : float
        Training progress from 0.0 to 1.0
    no_outputs : int
        Number of outputs (scales) to generate weights for

    Returns
    -------
    np.ndarray
        Array of weights following the progressive focus pattern
    """
    # Handle edge cases first
    if no_outputs == 1:
        return np.array([1.0])

    # Initialize result array with zeros
    result = np.zeros(no_outputs, dtype=np.float64)  # Explicitly use float64 for precision

    # Calculate interval size and number of active elements
    interval_size = 1.0 / no_outputs

    # Calculate active elements, handling numerical precision issues
    active_elements = no_outputs - int(np.floor(percentage_done / interval_size))
    active_elements = np.clip(active_elements, 1, no_outputs)  # Ensure bounds

    # Special case: when exactly at an interval boundary
    if np.isclose(percentage_done % interval_size, 0.0):
        active_elements = no_outputs - int(percentage_done / interval_size)
        active_elements = max(1, active_elements)

    # Distribution logic
    if active_elements > 1:
        # Distribute 25% of weight equally among all outputs except the focal one
        equal_portion = 0.25 / (active_elements - 1)
        result[:active_elements - 1] = equal_portion
        # Assign 75% weight to the focal output
        result[active_elements - 1] = 0.75
    else:
        # In the final interval, all weight goes to output1 (inference)
        result[0] = 1.0

    # Final validation
    assert np.isclose(np.sum(result), 1.0), "Results must sum to 1"
    assert np.all(result >= 0), "All values must be non-negative"

    # Round very small values to exactly zero to prevent floating point artifacts
    result[np.isclose(result, 0, atol=1e-5)] = 0.0

    return result


def cosine_annealing_schedule(
        percentage_done: float,
        no_outputs: int,
        frequency: float = 3.0,
        final_ratio: float = 0.8
) -> np.ndarray:
    """
    Implements a cosine annealing schedule that oscillates between emphasizing deep and shallow features.

    The weights follow a cosine wave pattern with decreasing amplitude, gradually converging
    to a final distribution that emphasizes the inference output.

    For U-Net: Helps escape local minima by periodically shifting focus between semantic
    features and fine details. Similar to cyclic learning rates, this can improve
    generalization and exploration of the loss landscape.

    Parameters
    ----------
    percentage_done : float
        Training progress from 0.0 to 1.0
    no_outputs : int
        Number of outputs (scales) to generate weights for
    frequency : float
        Number of complete oscillations during training
    final_ratio : float
        Final weight ratio between inference output (output1) and deepest output

    Returns
    -------
    np.ndarray
        Array of weights following the cosine annealing pattern
    """
    # Target final distribution (emphasizes inference output)
    d_final = np.linspace(final_ratio, 1.0, no_outputs)
    d_final = d_final / np.sum(d_final)

    # Initial distribution (emphasizes deeper features)
    d_initial = d_final[::-1]

    # Calculate amplitude decay - reduces the oscillation over time
    amplitude = np.cos(np.pi * percentage_done) * 0.5 + 0.5

    # Calculate phase for the current training percentage
    phase = np.cos(2 * np.pi * frequency * percentage_done)

    # Interpolate between distributions based on phase and amplitude
    if phase > 0:
        # Moving toward initial distribution
        mix_ratio = phase * amplitude
        weights = d_final * (1 - mix_ratio) + d_initial * mix_ratio
    else:
        # Moving toward final distribution
        mix_ratio = -phase * amplitude
        weights = d_final * (1 - mix_ratio) + d_initial * mix_ratio

    # Ensure weights sum to 1.0
    weights = weights / np.sum(weights)
    return weights


def curriculum_schedule(
        percentage_done: float,
        no_outputs: int,
        max_active_outputs: int = None,
        activation_strategy: str = 'linear'
) -> np.ndarray:
    """
    Implements a curriculum learning strategy by gradually incorporating shallower outputs.

    Initially focuses entirely on the deepest output (outputN), then progressively
    activates and includes shallower outputs as training progresses, ultimately
    emphasizing the inference output (output1).

    For U-Net: Enforces a strict curriculum where the model first learns deep semantic
    features, then gradually incorporates finer details as training progresses.
    This helps establish a strong hierarchical representation.

    Parameters
    ----------
    percentage_done : float
        Training progress from 0.0 to 1.0
    no_outputs : int
        Number of outputs (scales) to generate weights for
    max_active_outputs : int, optional
        Maximum number of simultaneously active outputs (default: all outputs)
    activation_strategy : str
        How outputs become active: 'linear' or 'sqrt'

    Returns
    -------
    np.ndarray
        Array of weights following the curriculum pattern
    """
    if max_active_outputs is None:
        max_active_outputs = no_outputs

    # Initialize weights array
    weights = np.zeros(no_outputs)

    # Determine how many outputs are active at current percentage
    if activation_strategy == 'sqrt':
        # Square root activates outputs more rapidly at first, then slows down
        active_outputs = min(int(np.sqrt(percentage_done) * no_outputs) + 1, max_active_outputs)
    else:  # 'linear'
        # Linear activation strategy
        active_outputs = min(int(percentage_done * no_outputs) + 1, max_active_outputs)

    # Ensure at least one output is active
    active_outputs = max(1, active_outputs)

    # Determine which outputs to activate (starting from deepest)
    active_indices = list(range(no_outputs - active_outputs, no_outputs))

    # If inference output (output1) is active, give it increasing weight
    if 0 in active_indices:
        inference_weight = percentage_done ** 2  # Quadratic increase in importance
        remaining_weight = 1.0 - inference_weight

        # Distribute the remaining weight among other active outputs
        if len(active_indices) > 1:
            for i in active_indices:
                if i == 0:
                    weights[i] = inference_weight
                else:
                    # Weight deeper features more
                    weights[i] = remaining_weight * (i + 1) / sum(range(1, active_outputs))
        else:
            # Only inference output is active
            weights[0] = 1.0
    else:
        # Inference output not active yet, focus on deepest active feature
        active_weights = np.array(range(1, active_outputs + 1))
        active_weights = active_weights / np.sum(active_weights)

        # Assign weights to active outputs
        for idx, i in enumerate(active_indices):
            weights[i] = active_weights[idx]

    # Ensure weights sum to 1.0 (handle numerical precision issues)
    weights = weights / np.sum(weights)
    return weights

# ---------------------------------------------------------------------
# Main schedule builder function
# ---------------------------------------------------------------------

def schedule_builder(
        config: Dict,
        no_outputs: int) -> Callable[[float], np.ndarray]:
    """
    Builds a schedule function that computes weighting arrays for deep supervision
    outputs at different training progress stages (percentage_done).

    In the context of U-Net architecture:
    - Output 1: Final inference output (highest resolution)
    - Output 5 (or highest number): Deepest scale in the network
    - Outputs in between: Intermediate scales

    Parameters
    ----------
    config : Dict
        Configuration dictionary containing the type of schedule and optional parameters.
    no_outputs : int
        Number of outputs (scales) for which weights must be generated.

    Returns
    -------
    schedule : Callable[[float], np.ndarray]
        A function taking a float percentage (0.0 to 1.0) and returning a numpy array
        of shape [no_outputs] summing to 1.0, which represents the weights for each output.
    """
    # --- argument checking
    if not isinstance(config, Dict):
        raise ValueError("config must be a dictionary")
    if no_outputs <= 0:
        raise ValueError("no_outputs must be positive integer")

    # --- select type
    schedule_type = config.get(TYPE_STR, None)

    # --- sanity checks
    if schedule_type is None:
        raise ValueError("schedule_type cannot be None")
    if not isinstance(schedule_type, str):
        raise ValueError("schedule_type must be a string")

    # --- select schedule
    params = config.get(CONFIG_STR, {})
    schedule_type = schedule_type.strip().lower()
    logger.info(f"deep supervision schedule: "
                f"[{schedule_type}], "
                f"with params: [{params}]")

    # --- Create and return the appropriate schedule function
    if schedule_type == "constant_equal":
        return lambda percentage_done: constant_equal_schedule(percentage_done, no_outputs)

    elif schedule_type == "constant_low_to_high":
        return lambda percentage_done: constant_low_to_high_schedule(percentage_done, no_outputs)

    elif schedule_type == "constant_high_to_low":
        return lambda percentage_done: constant_high_to_low_schedule(percentage_done, no_outputs)

    elif schedule_type == "linear_low_to_high":
        return lambda percentage_done: linear_low_to_high_schedule(percentage_done, no_outputs)

    elif schedule_type == "non_linear_low_to_high":
        return lambda percentage_done: non_linear_low_to_high_schedule(percentage_done, no_outputs)

    elif schedule_type == "custom_sigmoid_low_to_high":
        # Extract parameters for custom sigmoid
        k = params.get('k', 10)
        x0 = params.get('x0', 0.5)
        transition_point = params.get('transition_point', 0.25)

        return lambda percentage_done: custom_sigmoid_low_to_high_schedule(
            percentage_done, no_outputs, k, x0, transition_point
        )

    elif schedule_type == "scale_by_scale_low_to_high":
        return lambda percentage_done: scale_by_scale_low_to_high_schedule(percentage_done, no_outputs)

    else:
        raise ValueError(f"don't know how to handle "
                         f"deep supervision schedule_type [{schedule_type}]")

# ---------------------------------------------------------------------