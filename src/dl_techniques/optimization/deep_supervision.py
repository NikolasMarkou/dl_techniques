# ---------------------------------------------------------------------

import numpy as np
from typing import Dict, Tuple, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.constants import *
from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------

def schedule_builder(
        config: Dict,
        no_outputs: int) -> Callable[[float], np.ndarray]:
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
    schedule = None
    params = config.get(CONFIG_STR, {})
    schedule_type = schedule_type.strip().lower()
    logger.info(f"deep supervision schedule: "
                f"[{schedule_type}], "
                f"with params: [{params}]")

    if schedule_type == "constant_equal":
        def schedule(percentage_done: float = 0.0):
            d = np.array([1.0, ] * no_outputs)
            d = d / np.sum(d)
            return d
    elif schedule_type == "constant_low_to_high":
        def schedule(percentage_done: float = 0.0):
            d = np.array(list(range(1, no_outputs + 1))).astype(np.float32)
            d = d / np.sum(d)
            return d
    elif schedule_type == "constant_high_to_low":
        def schedule(percentage_done: float = 0.0):
            d = np.array(list(range(1, no_outputs + 1))).astype(np.float32)
            d = d / np.sum(d)
            d = d[::-1]
            return d
    elif schedule_type == "linear_low_to_high":
        def schedule(percentage_done: float = 0.0):
            d_start = np.array(list(range(1, no_outputs + 1))).astype(np.float32)
            d_start = d_start / np.sum(d_start)
            d_end = d_start[::-1]
            return d_start * (1.0 - percentage_done) + d_end * percentage_done
    elif schedule_type == "non_linear_low_to_high":
        def schedule(percentage_done: float = 0.0):
            d_start = np.array(list(range(1, no_outputs + 1))).astype(np.float32)
            d_start = d_start / np.sum(d_start)
            d_end = d_start[::-1]
            x = np.clip(np.tanh(2.5 * percentage_done), a_min=0.0, a_max=1.0)
            return d_start * (1.0 - x) + d_end * x
    elif schedule_type == "custom_sigmoid_low_to_high":
        def custom_sigmoid(x: float, k: float = 10, x0: float = 0.5) -> float:
            return 1 / (1 + np.exp(-k * (x - x0)))

        def normalize(x: float, x_min: float, x_max: float) -> float:
            return (x - x_min) / (x_max - x_min)

        def custom_function(x: float, k: float = 10, x0: float = 0.5) -> float:
            y = custom_sigmoid(x, k, x0)
            y_normalized = normalize(y, custom_sigmoid(0, k, x0), custom_sigmoid(1, k, x0))
            return y_normalized

        k = params.get('k', 10)
        x0 = params.get('x0', 0.5)
        transition_point = params.get('transition_point', 0.25)

        def schedule(percentage_done: float = 0.0) -> np.ndarray:
            if percentage_done >= transition_point:
                w = np.array([0.9] + [0.1 / (no_outputs - 1)] * (no_outputs - 1))
            else:
                d_start = np.array(range(1, no_outputs + 1)) / np.sum(range(1, no_outputs + 1))
                d_end = np.array([0.9] + [0.1 / (no_outputs - 1)] * (no_outputs - 1))
                x = custom_function(percentage_done / transition_point, k, x0)
                w = d_start * (1 - x) + d_end * x
            return w / np.sum(w)
    elif schedule_type == "scale_by_scale_low_to_high":
        def schedule(percentage_done: float = 0.0):
            """
            Generate an array of positive floats that sum to 1, where elements are progressively
            zeroed out based on percentage_done and their values redistributed to earlier elements.

            Args:
                percentage_done (float): Progress value between 0 and 1

            Returns:
                numpy.ndarray: Array of floats that sum to 1
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
                # Calculate equal portions with high precision
                equal_portion = 0.25 / (active_elements - 1)
                # Assign small portions first
                result[:active_elements - 1] = equal_portion
                # Assign the larger portion to last active element
                result[active_elements - 1] = 0.75
            else:
                # Single active element gets everything
                result[0] = 1.0

            # Final validation
            assert np.isclose(np.sum(result), 1.0), "Results must sum to 1"
            assert np.all(result >= 0), "All values must be non-negative"

            # Round very small values to exactly zero to prevent floating point artifacts
            result[np.isclose(result, 0, atol=1e-5)] = 0.0

            return result
    else:
        raise ValueError(f"don't know how to handle "
                         f"deep supervision schedule_type [{schedule_type}]")

    return schedule

# ---------------------------------------------------------------------
