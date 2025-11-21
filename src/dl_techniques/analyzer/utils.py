"""
Utility Functions for Model Analyzer

Common utility functions used throughout the analyzer module, including
robust data sampling and metric extraction.
"""

import keras
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.colors as mcolors
from typing import List, Optional, Dict, Tuple, Any, Iterator

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.analyzer.data_types import DataInput

# ---------------------------------------------------------------------

class DataSampler:
    """
    Helper class to handle robust data sampling from various input formats.

    Supports:
    - NumPy arrays
    - Dictionaries of NumPy arrays (for multi-input models)
    - TensorFlow Datasets (if available)
    - Python Iterators/Generators
    """

    @staticmethod
    def sample(data: DataInput, n_samples: int) -> DataInput:
        """
        Sample a subset of data from the input, handling various formats.

        Args:
            data: The DataInput object containing x_data and y_data.
            n_samples: The desired number of samples.

        Returns:
            A new DataInput object containing the sampled data as NumPy arrays.

        Raises:
            ValueError: If input data formats are inconsistent or unsupported.
        """
        x_data = data.x_data
        y_data = data.y_data

        # 1. Handle TensorFlow Datasets
        if isinstance(x_data, (tf.data.Dataset, tf.distribute.DistributedDataset)):
            return DataSampler._sample_tf_dataset(x_data, n_samples)

        # 2. Handle Dictionaries (Multi-input models)
        if isinstance(x_data, dict):
            return DataSampler._sample_dict_inputs(x_data, y_data, n_samples)

        # 3. Handle Standard NumPy Arrays / Lists
        if hasattr(x_data, '__len__') and hasattr(x_data, '__getitem__'):
            return DataSampler._sample_array_inputs(x_data, y_data, n_samples)

        # 4. Handle Generic Iterators
        if isinstance(x_data, Iterator):
            return DataSampler._sample_iterator(x_data, y_data, n_samples)

        # Fallback
        logger.warning("Unknown data type in DataSampler. Returning original data.")
        return data

    @staticmethod
    def _sample_array_inputs(x: Any, y: Any, n_samples: int) -> DataInput:
        """
        Sample from indexable array-like inputs (NumPy, Lists).

        Args:
            x: Input features (array-like).
            y: Target labels (array-like).
            n_samples: Number of samples to select.

        Returns:
            DataInput with sampled subsets.
        """
        total_samples = len(x)

        if total_samples <= n_samples:
            return DataInput(x_data=np.array(x), y_data=np.array(y))

        indices = np.random.choice(total_samples, n_samples, replace=False)

        # Handle x sampling
        if isinstance(x, np.ndarray):
            x_sampled = x[indices]
        else:
            x_sampled = np.array([x[i] for i in indices])

        # Handle y sampling
        if isinstance(y, np.ndarray):
            y_sampled = y[indices]
        else:
            y_sampled = np.array([y[i] for i in indices])

        return DataInput(x_data=x_sampled, y_data=y_sampled)

    @staticmethod
    def _sample_dict_inputs(x: Dict[str, Any], y: Any, n_samples: int) -> DataInput:
        """
        Sample from dictionary inputs (common in multi-input Keras models).

        Args:
            x: Dictionary of input features.
            y: Target labels.
            n_samples: Number of samples to select.

        Returns:
            DataInput with sampled subsets.
        """
        # Get length from the first key
        first_key = next(iter(x))
        total_samples = len(x[first_key])

        if total_samples <= n_samples:
            # Convert all values to numpy arrays if they aren't already
            x_out = {k: np.array(v) for k, v in x.items()}
            return DataInput(x_data=x_out, y_data=np.array(y))

        indices = np.random.choice(total_samples, n_samples, replace=False)

        x_sampled = {}
        for key, val in x.items():
            if isinstance(val, np.ndarray):
                x_sampled[key] = val[indices]
            else:
                x_sampled[key] = np.array([val[i] for i in indices])

        if isinstance(y, np.ndarray):
            y_sampled = y[indices]
        else:
            y_sampled = np.array([y[i] for i in indices])

        return DataInput(x_data=x_sampled, y_data=y_sampled)

    @staticmethod
    def _sample_tf_dataset(dataset: Any, n_samples: int) -> DataInput:
        """
        Sample from a TensorFlow Dataset.

        Assumes the dataset yields (x, y) tuples or just x.
        Note: This ignores the original `y_data` in DataInput if the dataset provides labels.

        Args:
            dataset: The tf.data.Dataset.
            n_samples: Number of samples to take.

        Returns:
            DataInput with numpy arrays extracted from the dataset.
        """
        logger.info(f"Sampling {n_samples} from TensorFlow Dataset...")

        # Unbatch to handle individual samples, then take n_samples
        # This is general but might be slow for huge datasets if not shuffled
        ds_iter = dataset.unbatch().take(n_samples).as_numpy_iterator()

        x_list = []
        y_list = []

        for item in ds_iter:
            if isinstance(item, tuple) and len(item) >= 2:
                x_list.append(item[0])
                y_list.append(item[1])
            else:
                # Dataset only yields features
                x_list.append(item)
                # We can't recover y if it's not in the dataset

        x_out = np.array(x_list)
        y_out = np.array(y_list) if y_list else np.zeros(len(x_list)) # Fallback placeholder

        return DataInput(x_data=x_out, y_data=y_out)

    @staticmethod
    def _sample_iterator(x_iter: Iterator, y_iter: Optional[Iterator], n_samples: int) -> DataInput:
        """
        Sample from generic Python iterators.

        Args:
            x_iter: Iterator for features.
            y_iter: Iterator for labels (optional).
            n_samples: Number of samples to take.

        Returns:
            DataInput with numpy arrays.
        """
        x_list = list(itertools.islice(x_iter, n_samples))
        x_out = np.array(x_list)

        if y_iter:
            y_list = list(itertools.islice(y_iter, n_samples))
            y_out = np.array(y_list)
        else:
            # Fallback if y provided as array but x as iterator?
            # Complex edge case, assume y matches length or isn't provided via iterator
            y_out = np.zeros(len(x_list))

        return DataInput(x_data=x_out, y_data=y_out)


# ---------------------------------------------------------------------
# Existing Utility Functions
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

    Args:
        history: Training history dictionary.
        patterns: List of possible metric names to check (in order of preference).
        exclude_prefixes: List of prefixes to exclude.

    Returns:
        The metric values if found, None otherwise.
    """
    if exclude_prefixes is None:
        exclude_prefixes = []

    # Pass 1: Try exact matches (most reliable)
    for pattern in patterns:
        if pattern in history:
            if not any(pattern.startswith(prefix) for prefix in exclude_prefixes):
                return history[pattern]

    # Pass 2: Try word-boundary pattern matching
    for key in history:
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue

        key_components = _split_metric_name(key)
        for pattern in patterns:
            pattern_components = _split_metric_name(pattern)
            if all(p_comp in key_components for p_comp in pattern_components):
                return history[key]

    # Pass 3: Try fuzzy matching
    for key in history:
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue

        for pattern in patterns:
            if _fuzzy_metric_match(key, pattern):
                return history[key]

    return None


def _split_metric_name(name: str) -> List[str]:
    """Split a metric name into its component parts for robust matching."""
    import re
    parts = name.replace('_', ' ').replace('-', ' ').split()
    expanded_parts = []
    for part in parts:
        camel_split = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', part)
        if camel_split:
            expanded_parts.extend(camel_split)
        else:
            expanded_parts.append(part)
    return [p.lower() for p in expanded_parts if p]


def _fuzzy_metric_match(key: str, pattern: str) -> bool:
    """
    Perform fuzzy matching for common metric name variations.

    Checks if parts of the pattern exist in the key using common abbreviations.
    """
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

    key_parts = _split_metric_name(key)
    pattern_parts = _split_metric_name(pattern)

    for p_part in pattern_parts:
        found = False
        # Get all equivalent terms for the current pattern part
        equivalent_terms = equivalences.get(p_part, [p_part])

        for k_part in key_parts:
            # Check if key part is in equivalents OR pattern part is in equivalents of key part
            if k_part in equivalent_terms or p_part in equivalences.get(k_part, [k_part]):
                found = True
                break

        if not found:
            return False

    return True


def find_model_metric(model_metrics: Dict[str, Any],
                     metric_keys: List[str],
                     default: float = 0.0) -> float:
    """
    Helper function to find a metric value from model metrics with fallback chain.

    Args:
        model_metrics: Dictionary of model metrics.
        metric_keys: List of metric keys to check in order of preference.
        default: Default value if no metrics found.

    Returns:
        The first found metric value or default.
    """
    for key in metric_keys:
        if key in model_metrics and model_metrics[key] is not None:
            try:
                return float(model_metrics[key])
            except (ValueError, TypeError):
                continue
    return default


def lighten_color(color: str, factor: float) -> Tuple[float, float, float]:
    """Lighten a color by interpolating towards white."""
    rgb = mcolors.to_rgb(color)
    lightened = tuple(rgb[i] + (1 - rgb[i]) * factor for i in range(3))
    return lightened


def find_pareto_front(costs1: np.ndarray, costs2: np.ndarray) -> List[int]:
    """
    Find indices of Pareto optimal points (maximizing both objectives).

    Args:
        costs1: Array of first objective values.
        costs2: Array of second objective values.

    Returns:
        List of indices of Pareto optimal points, sorted in ascending order.
    """
    population_size = len(costs1)
    pareto_indices = []

    for i in range(population_size):
        is_pareto = True
        for j in range(population_size):
            if i != j:
                if costs1[j] >= costs1[i] and costs2[j] >= costs2[i]:
                    if costs1[j] > costs1[i] or costs2[j] > costs2[i]:
                        is_pareto = False
                        break
        if is_pareto:
            pareto_indices.append(i)

    return sorted(pareto_indices)


def normalize_metric(values: List[float], higher_better: bool = True) -> np.ndarray:
    """
    Normalize metric values to 0-1 range.

    Args:
        values: List of metric values.
        higher_better: If True, higher is better (0 maps to min, 1 maps to max).

    Returns:
        Normalized array (0-1).
    """
    arr = np.array(values)
    if len(arr) == 0:
        return arr

    min_val, max_val = arr.min(), arr.max()
    if max_val == min_val:
        return np.ones_like(arr) * 0.5

    normalized = (arr - min_val) / (max_val - min_val)

    if not higher_better:
        normalized = 1 - normalized

    return normalized


def validate_training_history(history: Dict[str, List[float]]) -> Dict[str, List[str]]:
    """
    Validate training history and return a report of potential issues.

    Args:
        history: Training history dictionary.

    Returns:
        Dictionary with 'warnings' and 'errors'.
    """
    report = {'warnings': [], 'errors': []}

    if not history:
        report['errors'].append("Training history is empty")
        return report

    for key, values in history.items():
        if not values:
            report['warnings'].append(f"Metric '{key}' has no values")
        elif not isinstance(values, (list, np.ndarray)):
            report['errors'].append(f"Metric '{key}' is not a list or array")

    has_train_loss = any(find_metric_in_history(history, ['loss'], exclude_prefixes=['val_']))
    has_val_loss = any(find_metric_in_history(history, ['val_loss']))

    if not has_train_loss:
        report['warnings'].append("No training loss found")
    if not has_val_loss:
        report['warnings'].append("No validation loss found")

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


def recursively_get_layers(layer_or_model: Any) -> List[keras.layers.Layer]:
    """
    Recursively traverses a Keras model or layer to get a flat list of all layers.

    Args:
        layer_or_model: The Keras model or layer to traverse.

    Returns:
        A flat list of all Keras layers found.
    """
    all_layers = []
    queue = [layer_or_model]
    visited_layers = set()

    while queue:
        current_layer = queue.pop(0)

        if hasattr(current_layer, "_layer"):
            current_layer = getattr(current_layer, "_layer")

        if current_layer in visited_layers:
            continue
        visited_layers.add(current_layer)

        if isinstance(current_layer, keras.layers.Layer):
            all_layers.append(current_layer)

        # 1. Standard Keras containers
        if hasattr(current_layer, 'layers') and current_layer.layers:
            queue = current_layer.layers + queue
            continue

        # 2. Subclassed models: Check attributes
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

    # Filter for "leaf" layers
    primitive_layers = [
        layer for layer in all_layers
        if not (hasattr(layer, 'layers') and layer.layers)
    ]

    return primitive_layers if primitive_layers else all_layers