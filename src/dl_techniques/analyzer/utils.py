"""
Utility Functions for Model Analyzer

Common utility functions used throughout the analyzer module.
"""

import keras
import numpy as np
import matplotlib.colors as mcolors
from typing import List, Optional, Dict, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

def safe_set_xticklabels(ax, labels, rotation=0, max_labels=10):
    """Safely set x-tick labels with proper handling."""
    try:
        if len(labels) > max_labels:
            step = len(labels) // max_labels
            indices = range(0, len(labels), step)
            ax.set_xticks([i for i in indices])
            ax.set_xticklabels([labels[i] for i in indices], rotation=rotation, ha='right' if rotation > 0 else 'center')
        else:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=rotation, ha='right' if rotation > 0 else 'center')
    except Exception as e:
        logger.warning(f"Could not set x-tick labels: {e}")


def safe_tight_layout(fig, **kwargs):
    """Safely apply tight_layout with error handling."""
    try:
        fig.tight_layout(**kwargs)
    except Exception as e:
        logger.warning(f"Could not apply tight_layout: {e}")
        try:
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
        except Exception:
            pass


def smooth_curve(values: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply smoothing to a curve using a moving average."""
    if len(values) < window_size:
        return values

    # Pad the array to handle edges
    padded = np.pad(values, (window_size//2, window_size//2), mode='edge')

    # Apply moving average
    smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')

    return smoothed


def find_metric_in_history(history: Dict[str, List[float]], patterns: List[str],
                          exclude_prefixes: Optional[List[str]] = None) -> Optional[List[float]]:
    """
    Robustly find a metric in training history by checking multiple possible names.

    ENHANCED VERSION: This addresses the potential ambiguity identified in the code review
    by implementing a more systematic approach to metric matching with better fallback logic.

    This function uses a three-pass approach:
    1. First, try exact string matches for efficiency and clarity
    2. If no exact match, try pattern matching with word boundaries
    3. If still no match, try fuzzy matching with common variations

    Args:
        history: Training history dictionary
        patterns: List of possible metric names to check (in order of preference)
        exclude_prefixes: List of prefixes to exclude (e.g., ['val_'] when looking for training metrics)

    Returns:
        The metric values if found, None otherwise

    Example:
        >>> history = {'loss': [0.5, 0.3], 'val_loss': [0.6, 0.4], 'accuracy': [0.8, 0.9]}
        >>> find_metric_in_history(history, ['loss'], exclude_prefixes=['val_'])
        [0.5, 0.3]
        >>> find_metric_in_history(history, ['accuracy', 'acc'])
        [0.8, 0.9]
    """
    if exclude_prefixes is None:
        exclude_prefixes = []

    # Pass 1: Try exact matches (most reliable)
    for pattern in patterns:
        if pattern in history:
            # Check if this key should be excluded
            if not any(pattern.startswith(prefix) for prefix in exclude_prefixes):
                return history[pattern]

    # Pass 2: Try word-boundary pattern matching (safe substring matching)
    for key in history:
        # Skip if key starts with excluded prefix
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue

        # Split key into components (handle underscores, spaces, camelCase)
        key_components = _split_metric_name(key)

        for pattern in patterns:
            pattern_components = _split_metric_name(pattern)

            # Check if all pattern components are present in key components
            if all(p_comp in key_components for p_comp in pattern_components):
                logger.debug(f"Found metric '{key}' matching pattern '{pattern}' via component matching")
                return history[key]

    # Pass 3: Try fuzzy matching with common variations
    for key in history:
        # Skip if key starts with excluded prefix
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue

        for pattern in patterns:
            if _fuzzy_metric_match(key, pattern):
                logger.debug(f"Found metric '{key}' matching pattern '{pattern}' via fuzzy matching")
                return history[key]

    # Log available keys for debugging if no match found
    available_keys = [k for k in history.keys()
                     if not any(k.startswith(prefix) for prefix in exclude_prefixes)]
    logger.debug(f"No match found for patterns {patterns}. Available keys: {available_keys}")
    return None


def _split_metric_name(name: str) -> List[str]:
    """
    Split a metric name into its component parts for robust matching.

    Handles common naming conventions:
    - underscore_separated
    - camelCase
    - Mixed_camelCase

    Args:
        name: Metric name to split

    Returns:
        List of lowercase components
    """
    import re

    # First split on underscores and spaces
    parts = name.replace('_', ' ').replace('-', ' ').split()

    # Then split camelCase within each part
    expanded_parts = []
    for part in parts:
        # Split camelCase (e.g., "valAccuracy" -> ["val", "Accuracy"])
        camel_split = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', part)
        if camel_split:
            expanded_parts.extend(camel_split)
        else:
            expanded_parts.append(part)

    # Convert to lowercase and remove empty strings
    return [p.lower() for p in expanded_parts if p]


def _fuzzy_metric_match(key: str, pattern: str) -> bool:
    """
    Perform fuzzy matching for common metric name variations.

    This handles common abbreviations and variations:
    - acc/accuracy
    - val/validation
    - cat/categorical
    - etc.

    Args:
        key: The metric key from history
        pattern: The pattern we're looking for

    Returns:
        True if they likely refer to the same metric
    """
    # Define common equivalences
    equivalences = {
        'acc': ['accuracy', 'acc'],
        'accuracy': ['accuracy', 'acc'],
        'val': ['validation', 'val'],
        'validation': ['validation', 'val'],
        'cat': ['categorical', 'cat'],
        'categorical': ['categorical', 'cat'],
        'sparse': ['sparse_categorical', 'sparse'],
        'binary': ['binary_accuracy', 'binary'],
        'loss': ['loss', 'cost', 'error'],
        'lr': ['learning_rate', 'lr'],
        'learning_rate': ['learning_rate', 'lr'],
    }

    # Normalize both strings
    key_parts = _split_metric_name(key)
    pattern_parts = _split_metric_name(pattern)

    # Check if pattern parts can be found in key using equivalences
    for p_part in pattern_parts:
        found = False
        equivalent_terms = equivalences.get(p_part, [p_part])

        for k_part in key_parts:
            if k_part in equivalent_terms or p_part in equivalences.get(k_part, [k_part]):
                found = True
                break

        if not found:
            return False

    return True


def find_model_metric(model_metrics: Dict[str, Any],
                     metric_keys: List[str],
                     default: float = 0.0) -> float:
    """Helper function to find a metric value from model metrics with fallback chain.

    This reduces code duplication in summary tables where we need to check multiple
    possible metric names in order of preference.

    Args:
        model_metrics: Dictionary of model metrics
        metric_keys: List of metric keys to check in order of preference
        default: Default value if no metrics found

    Returns:
        The first found metric value or default

    Example:
        >>> metrics = {'categorical_accuracy': 0.85, 'loss': 0.3}
        >>> find_model_metric(metrics, ['accuracy', 'categorical_accuracy', 'compile_metrics'])
        0.85
    """
    for key in metric_keys:
        if key in model_metrics and model_metrics[key] is not None:
            try:
                return float(model_metrics[key])
            except (ValueError, TypeError):
                logger.warning(f"Could not convert metric '{key}' to float: {model_metrics[key]}")
                continue
    return default


def lighten_color(color: str, factor: float) -> Tuple[float, float, float]:
    """Lighten a color by interpolating towards white."""
    # Convert color to RGB
    rgb = mcolors.to_rgb(color)

    # Interpolate towards white
    lightened = tuple(rgb[i] + (1 - rgb[i]) * factor for i in range(3))

    return lightened


def find_pareto_front(costs1: np.ndarray, costs2: np.ndarray) -> List[int]:
    """Find indices of Pareto optimal points (maximizing both objectives).

    Time Complexity: O(N²) where N is the number of points.
    Space Complexity: O(N) for storing the result indices.

    This implementation is suitable for small-to-medium datasets (<100 points).
    For larger datasets, consider using more efficient algorithms like the
    non-dominated sorting approach.

    Args:
        costs1: Array of first objective values (to be maximized)
        costs2: Array of second objective values (to be maximized)

    Returns:
        List of indices of Pareto optimal points, sorted in ascending order

    Example:
        >>> costs1 = np.array([1, 2, 3])
        >>> costs2 = np.array([3, 2, 1])
        >>> find_pareto_front(costs1, costs2)
        [0, 2]  # Points (1,3) and (3,1) are Pareto optimal
    """
    population_size = len(costs1)
    pareto_indices = []

    for i in range(population_size):
        is_pareto = True
        for j in range(population_size):
            if i != j:
                # Check if j dominates i (better in both objectives)
                if costs1[j] >= costs1[i] and costs2[j] >= costs2[i]:
                    if costs1[j] > costs1[i] or costs2[j] > costs2[i]:
                        is_pareto = False
                        break
        if is_pareto:
            pareto_indices.append(i)

    return sorted(pareto_indices)


def normalize_metric(values: List[float], higher_better: bool = True) -> np.ndarray:
    """Normalize metric values to 0-1 range.

    Args:
        values: List of metric values to normalize
        higher_better: If True, higher values are considered better.
                      If False, lower values are considered better (e.g., for loss).

    Returns:
        Normalized array where higher values are always better (0-1 range)

    Example:
        >>> normalize_metric([1, 2, 3], higher_better=True)
        array([0. , 0.5, 1. ])
        >>> normalize_metric([1, 2, 3], higher_better=False)
        array([1. , 0.5, 0. ])
    """
    arr = np.array(values)
    if len(arr) == 0:
        return arr

    min_val, max_val = arr.min(), arr.max()
    if max_val == min_val:
        # All values are the same, return middle value
        return np.ones_like(arr) * 0.5

    # Normalize to 0-1 range
    normalized = (arr - min_val) / (max_val - min_val)

    # If lower is better, flip the normalization
    if not higher_better:
        normalized = 1 - normalized

    return normalized


def validate_training_history(history: Dict[str, List[float]]) -> Dict[str, List[str]]:
    """
    Validate training history and return a report of potential issues.

    This helps identify common problems with training history data that could
    cause analysis issues.

    Args:
        history: Training history dictionary

    Returns:
        Dictionary with 'warnings' and 'errors' keys containing lists of issues
    """
    report = {'warnings': [], 'errors': []}

    if not history:
        report['errors'].append("Training history is empty")
        return report

    # Check for empty metrics
    for key, values in history.items():
        if not values:
            report['warnings'].append(f"Metric '{key}' has no values")
        elif not isinstance(values, (list, np.ndarray)):
            report['errors'].append(f"Metric '{key}' is not a list or array")
        elif any(not isinstance(v, (int, float, np.number)) for v in values):
            report['warnings'].append(f"Metric '{key}' contains non-numeric values")

    # Check for length mismatches
    lengths = {key: len(values) for key, values in history.items() if values}
    if lengths:
        min_length = min(lengths.values())
        max_length = max(lengths.values())
        if min_length != max_length:
            report['warnings'].append(f"Metrics have different lengths: {lengths}")

    # Check for common metric patterns
    has_train_loss = any(find_metric_in_history(history, ['loss'], exclude_prefixes=['val_']))
    has_val_loss = any(find_metric_in_history(history, ['val_loss']))

    if not has_train_loss:
        report['warnings'].append("No training loss found - training analysis may be limited")
    if not has_val_loss:
        report['warnings'].append("No validation loss found - overfitting analysis not possible")

    return report

def truncate_model_name(name: str, max_len: int = 12, filler: str = "...") -> str:
    """Truncates a string by replacing middle characters with a filler."""
    if len(name) <= max_len:
        return name

    chars_to_keep = max_len - len(filler)
    if chars_to_keep < 2:
        return name[:max_len]

    start_len = (chars_to_keep + 1) // 2
    end_len = chars_to_keep // 2

    return f"{name[:start_len]}{filler}{name[-end_len:]}"

# ---------------------------------------------------------------------


def recursively_get_layers(layer_or_model: Any) -> List[keras.layers.Layer]:
    """
    Recursively traverses a Keras model or layer to get a flat list of all layers.

    This enhanced version correctly handles complex subclassed models where layers
    might be stored as attributes, in lists, or in dictionaries. It now uses object
    identity for visited checks to correctly handle layers with duplicate names.

    Args:
        layer_or_model: The Keras model or layer to traverse.

    Returns:
        A flat list of all Keras layers found in their order of discovery.
    """
    all_layers = []
    # Use a queue for traversal and a set to track visited layer *objects*.
    queue = [layer_or_model]
    # [-] OLD: visited_names = set()
    # [+] NEW: Track objects, not names.
    visited_layers = set()

    while queue:
        current_layer = queue.pop(0)

        # Keras can sometimes wrap layers; get the innermost object.
        if hasattr(current_layer, "_layer"):
            current_layer = getattr(current_layer, "_layer")

        # Avoid cycles and redundant processing using object identity.
        # [-] OLD check was based on name
        # if not hasattr(current_layer, "name") or current_layer.name in visited_names:
        #     continue
        # visited_names.add(current_layer.name)

        # [+] NEW check is based on the object itself.
        if current_layer in visited_layers:
            continue
        visited_layers.add(current_layer)

        if isinstance(current_layer, keras.layers.Layer):
            all_layers.append(current_layer)

        # --- Enhanced Discovery Logic ---
        # 1. Standard Keras containers
        if hasattr(current_layer, 'layers') and current_layer.layers:
            queue = current_layer.layers + queue
            continue

        # 2. Subclassed models: Check all attributes
        for attr_name in dir(current_layer):
            if attr_name.startswith("_"):
                continue
            try:
                attr = getattr(current_layer, attr_name)

                if isinstance(attr, (list, tuple)):
                    layer_items = [item for item in attr if isinstance(item, keras.layers.Layer)]
                    if layer_items:
                        queue = layer_items + queue
                elif isinstance(attr, dict):
                    layer_items = [v for v in attr.values() if isinstance(v, keras.layers.Layer)]
                    if layer_items:
                        queue = layer_items + queue
                elif isinstance(attr, keras.layers.Layer):
                    queue.insert(0, attr)
            except Exception:
                continue

    # Filter for primitive "leaf" layers that don't contain other layers.
    primitive_layers = [
        layer for layer in all_layers
        if not (hasattr(layer, 'layers') and layer.layers)
    ]

    return primitive_layers if primitive_layers else all_layers

# ---------------------------------------------------------------------
