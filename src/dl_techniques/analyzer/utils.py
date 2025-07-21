"""
Utility Functions for Model Analyzer
============================================================================

Common utility functions used throughout the analyzer module.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Optional, Dict, Tuple
from dl_techniques.utils.logger import logger


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
    """Flexibly find a metric in training history by checking multiple possible names.

    Args:
        history: Training history dictionary
        patterns: List of possible metric names to check
        exclude_prefixes: List of prefixes to exclude (e.g., ['val_'] when looking for training metrics)

    Returns:
        The metric values if found, None otherwise
    """
    # First try exact matches
    for pattern in patterns:
        if pattern in history:
            return history[pattern]

    # If no exact match and we need more flexible matching (only for special cases)
    # Be very careful with substring matching to avoid ambiguity
    if exclude_prefixes is None:
        exclude_prefixes = []

    # Try more specific patterns that won't cause ambiguity
    for key in history:
        # Skip if key starts with excluded prefix
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue

        for pattern in patterns:
            # Only match if pattern is a complete word/component in the key
            # This prevents 'acc' from matching 'val_acc'
            key_parts = key.replace('_', ' ').split()
            if pattern in key_parts:
                return history[key]

    return None


def lighten_color(color: str, factor: float) -> Tuple[float, float, float]:
    """Lighten a color by interpolating towards white."""
    # Convert color to RGB
    rgb = mcolors.to_rgb(color)

    # Interpolate towards white
    lightened = tuple(rgb[i] + (1 - rgb[i]) * factor for i in range(3))

    return lightened


def find_pareto_front(costs1: np.ndarray, costs2: np.ndarray) -> List[int]:
    """Find indices of Pareto optimal points (maximizing both objectives)."""
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
    """Normalize metric values to 0-1 range."""
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