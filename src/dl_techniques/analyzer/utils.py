"""
Utility Functions for Model Analyzer

Common utility functions used throughout the analyzer module, including
robust data sampling and metric extraction.
"""

import re
import keras
import itertools
import numpy as np
import matplotlib.colors as mcolors
from typing import List, Optional, Dict, Tuple, Any, Iterator

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.analyzer.data_types import DataInput

# ---------------------------------------------------------------------

def make_rng(random_state: Optional[int] = None) -> np.random.Generator:
    """Build the analyzer's random generator.

    Single source of truth for how ``AnalysisConfig.random_state`` becomes a
    generator, so no call site reaches for the global ``np.random.*`` state.

    Args:
        random_state: Seed, or ``None`` for a nondeterministic generator.

    Returns:
        A ``numpy.random.Generator``. ``None`` yields an unseeded one, which is the
        historical (unreproducible) behaviour and remains the default.
    """
    return np.random.default_rng(random_state)


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
    def sample(
            data: DataInput,
            n_samples: int,
            rng: Optional[np.random.Generator] = None,
    ) -> DataInput:
        """
        Sample a subset of data from the input, handling various formats.

        Args:
            data: The DataInput object containing x_data and y_data.
            n_samples: The desired number of samples.
            rng: Generator used for the subsampling draw. ``None`` builds an
                unseeded one, so the draw is not reproducible - pass
                ``make_rng(config.random_state)`` to make the whole analysis
                reproducible.

        Returns:
            A new DataInput object containing the sampled data as NumPy arrays.

        Raises:
            ValueError: If input data formats are inconsistent or unsupported.
        """
        # DECISION plan-2026-09-01T225724-e79ad4bd/D-031
        # `DataSampler` is the single sampling chokepoint on the analysis default
        # path (`model_analyzer.py:451`). Do NOT reintroduce a bare
        # `np.random.choice` below: the draw selects WHICH samples every downstream
        # calibration and confidence number is computed from, so an unseeded draw
        # made the entire analysis unreproducible run to run.
        # See decisions.md D-031.
        if rng is None:
            rng = make_rng()

        x_data = data.x_data
        y_data = data.y_data

        # 1. Handle TensorFlow Datasets
        try:
            import tensorflow as tf
            if isinstance(x_data, (tf.data.Dataset, tf.distribute.DistributedDataset)):
                return DataSampler._sample_tf_dataset(x_data, n_samples)
        except ImportError:
            pass

        # 2. Handle Dictionaries (Multi-input models)
        if isinstance(x_data, dict):
            return DataSampler._sample_dict_inputs(x_data, y_data, n_samples, rng)

        # 3. Handle Standard NumPy Arrays / Lists
        if hasattr(x_data, '__len__') and hasattr(x_data, '__getitem__'):
            return DataSampler._sample_array_inputs(x_data, y_data, n_samples, rng)

        # 4. Handle Generic Iterators
        if isinstance(x_data, Iterator):
            return DataSampler._sample_iterator(x_data, y_data, n_samples)

        # Fallback
        logger.warning("Unknown data type in DataSampler. Returning original data.")
        return data

    @staticmethod
    def _sample_array_inputs(
            x: Any, y: Any, n_samples: int,
            rng: np.random.Generator,
    ) -> DataInput:
        """
        Sample from indexable array-like inputs (NumPy, Lists).

        Args:
            x: Input features (array-like).
            y: Target labels (array-like).
            n_samples: Number of samples to select.
            rng: Generator for the without-replacement draw.

        Returns:
            DataInput with sampled subsets.
        """
        total_samples = len(x)

        if total_samples <= n_samples:
            return DataInput(x_data=np.array(x), y_data=np.array(y))

        indices = rng.choice(total_samples, n_samples, replace=False)

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
    def _sample_dict_inputs(
            x: Dict[str, Any], y: Any, n_samples: int,
            rng: np.random.Generator,
    ) -> DataInput:
        """
        Sample from dictionary inputs (common in multi-input Keras models).

        Args:
            x: Dictionary of input features.
            y: Target labels.
            n_samples: Number of samples to select.
            rng: Generator for the without-replacement draw.

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

        indices = rng.choice(total_samples, n_samples, replace=False)

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
        if not y_list:
            raise ValueError(
                "TF Dataset yields only features (no labels). "
                "Labels are required for calibration and confidence analysis. "
                "Provide a dataset that yields (x, y) tuples."
            )
        y_out = np.array(y_list)

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

    has_train_loss = find_metric_in_history(history, ['loss'], exclude_prefixes=['val_']) is not None
    has_val_loss = find_metric_in_history(history, ['val_loss']) is not None

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


def _sublayers_in(attr: Any) -> List[keras.layers.Layer]:
    """Return the Keras layers directly held by one attribute value.

    Args:
        attr: An attribute value: a layer, or a list/tuple/dict that may hold
            layers. Anything else yields an empty list.

    Returns:
        The layers found, in the container's own iteration order.
    """
    if isinstance(attr, keras.layers.Layer):
        return [attr]
    if isinstance(attr, (list, tuple)):
        return [item for item in attr if isinstance(item, keras.layers.Layer)]
    if isinstance(attr, dict):
        return [v for v in attr.values() if isinstance(v, keras.layers.Layer)]
    return []


def _user_property_names(klass: type) -> List[str]:
    """Return the public `property` names ``klass`` declares outside Keras.

    A property Keras itself declares on any base of ``klass`` is excluded - by
    NAME, discovered from those bases, so the exclusion covers a subclass that
    OVERRIDES one and cannot go stale as Keras grows. Those properties are
    expensive (`weights`, `variables`, `trainable_weights`) and two of them
    (`input`, `output`) raise on a built layer.

    Args:
        klass: The class to inspect, usually ``type(some_layer)``.

    Returns:
        Property names in MRO order, de-duplicated, most-derived class first.
    """
    mro = getattr(klass, '__mro__', ())

    # Names Keras itself declares as properties anywhere in this class's bases.
    # Derived, never hand-listed, so it cannot go stale as Keras grows - and it
    # covers a user OVERRIDE of one of them too, which is still Keras' API
    # surface and still expensive.
    keras_owned = {
        attr_name
        for base in mro
        if getattr(base, '__module__', '').split('.')[0] == 'keras'
        for attr_name, descriptor in vars(base).items()
        if isinstance(descriptor, property)
    }

    names: List[str] = []
    seen = set()
    for base in mro:
        if getattr(base, '__module__', '').split('.')[0] == 'keras':
            continue
        for attr_name, descriptor in vars(base).items():
            if attr_name.startswith('_') or attr_name in seen:
                continue
            if attr_name in keras_owned:
                continue
            if isinstance(descriptor, property):
                seen.add(attr_name)
                names.append(attr_name)
    return names


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

        # 2. Subclassed layers: check the INSTANCE ATTRIBUTES.
        #
        # DECISION plan-2026-09-01T225724-e79ad4bd/D-023
        # Iterate `vars()`, never `dir()`. `dir()` returns 52 public names for a
        # subclassed Layer and 96 for a subclassed Model, most of them PROPERTIES
        # (`weights`, `variables`, `trainable_weights`, `losses`, `input`, `output`),
        # so a `getattr` sweep over it evaluates every one of them - and `input` /
        # `output` RAISE on a built layer, which the old blanket
        # `except Exception: continue` silently swallowed. `vars()` reads the
        # instance `__dict__` only: no property is touched, and the order is the
        # attribute assignment order from `__init__`, which is deterministic across
        # `keras.models.clone_model` (pinned by
        # `test_the_walk_order_is_stable_across_clone_model`). That determinism is
        # load-bearing: `model_analyzer.py` indexes `all_layers[layer_id]` on a
        # CLONED model in `create_smoothed_model`. See decisions.md D-023.
        found: List[keras.layers.Layer] = []
        for attr_name, attr in list(vars(current_layer).items()):
            if attr_name.startswith("_"):
                continue
            found.extend(_sublayers_in(attr))

        # 3. Subclassed layers: USER-DEFINED properties over private attributes.
        #
        # DECISION plan-2026-09-01T225724-e79ad4bd/D-038
        # `vars()` alone MISSES `self._inner = [...]` exposed as `@property def
        # blocks`, a common Keras idiom - measured: the `vars()` walk returned
        # ['input_layer', 'blk'] where the sublayers `prop_d1`/`prop_d2` exist,
        # i.e. silent coverage loss in weight, spectral AND information-flow
        # analysis. The properties are therefore evaluated too, but ONLY those
        # declared outside Keras itself: a Keras-owned property is skipped by
        # its DEFINING CLASS's module, never by a hand-written name list, which
        # would go stale the moment Keras adds one. That is what preserves
        # D-023's goal - `weights`, `variables`, `trainable_weights`, `losses`,
        # `input` and `output` are all Keras-owned, expensive, and the last two
        # RAISE on a built layer. See decisions.md D-038.
        for attr_name in _user_property_names(type(current_layer)):
            try:
                attr = getattr(current_layer, attr_name)
            except Exception as e:
                logger.debug(
                    f"Property '{attr_name}' of "
                    f"{type(current_layer).__name__} raised; skipped: {e}")
                continue
            found.extend(
                sub for sub in _sublayers_in(attr)
                if not any(sub is existing for existing in found))

        if found:
            queue = found + queue

    # Filter for "leaf" layers
    primitive_layers = [
        layer for layer in all_layers
        if not (hasattr(layer, 'layers') and layer.layers)
    ]

    return primitive_layers if primitive_layers else all_layers