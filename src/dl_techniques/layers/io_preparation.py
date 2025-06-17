"""Tensor normalization and clipping utilities.

This module provides functions for normalizing and denormalizing tensors,
as well as clipping tensor values to specified ranges. These operations
are commonly used in deep learning preprocessing and postprocessing steps.
"""

import numpy as np
import tensorflow as tf
from typing import Union
from numpy.typing import ArrayLike

# ---------------------------------------------------------------------

NumericTensor = Union[tf.Tensor, np.ndarray, ArrayLike]

# ---------------------------------------------------------------------

def clip_tensor(
        input_tensor: NumericTensor,
        clip_min: float,
        clip_max: float,
) -> tf.Tensor:
    """Clips tensor values to a specified range.

    Args:
        input_tensor: Input tensor to be clipped.
        clip_min: Minimum value for clipping.
        clip_max: Maximum value for clipping.

    Returns:
        tf.Tensor: Clipped tensor with values in [clip_min, clip_max].

    Raises:
        ValueError: If clip_min >= clip_max.
    """
    if clip_min >= clip_max:
        raise ValueError(
            f"clip_min ({clip_min}) must be less than clip_max ({clip_max})"
        )

    return tf.clip_by_value(
        input_tensor,
        clip_value_min=clip_min,
        clip_value_max=clip_max
    )

# ---------------------------------------------------------------------


def clip_unnormalized_tensor(
        input_tensor: NumericTensor,
        v_min: float = 0.0,
        v_max: float = 255.0,
) -> tf.Tensor:
    """Clips unnormalized tensor values to a specified range.

    Commonly used for image data in [0, 255] range.

    Args:
        input_tensor: Input tensor to be clipped.
        v_min: Minimum value for clipping. Defaults to 0.0.
        v_max: Maximum value for clipping. Defaults to 255.0.

    Returns:
        tf.Tensor: Clipped tensor with values in [v_min, v_max].
    """
    return clip_tensor(input_tensor, v_min, v_max)

# ---------------------------------------------------------------------


def clip_normalized_tensor(
        input_tensor: NumericTensor,
        v_min: float = -0.5,
        v_max: float = 0.5,
) -> tf.Tensor:
    """Clips normalized tensor values to a specified range.

    Commonly used for normalized data in [-0.5, 0.5] range.

    Args:
        input_tensor: Input tensor to be clipped.
        v_min: Minimum value for clipping. Defaults to -0.5.
        v_max: Maximum value for clipping. Defaults to 0.5.

    Returns:
        tf.Tensor: Clipped tensor with values in [v_min, v_max].
    """
    return clip_tensor(input_tensor, v_min, v_max)

# ---------------------------------------------------------------------


def normalize_tensor(
        input_tensor: NumericTensor,
        source_min: float = 0.0,
        source_max: float = 255.0,
        target_min: float = -0.5,
        target_max: float = 0.5,
) -> tf.Tensor:
    """Normalizes tensor from source range to target range.

    Args:
        input_tensor: Input tensor to be normalized.
        source_min: Minimum value of source range. Defaults to 0.0.
        source_max: Maximum value of source range. Defaults to 255.0.
        target_min: Minimum value of target range. Defaults to -0.5.
        target_max: Maximum value of target range. Defaults to 0.5.

    Returns:
        tf.Tensor: Normalized tensor with values in [target_min, target_max].

    Raises:
        ValueError: If source or target ranges are invalid.
    """
    if source_min >= source_max:
        raise ValueError(
            f"source_min ({source_min}) must be less than source_max ({source_max})"
        )
    if target_min >= target_max:
        raise ValueError(
            f"target_min ({target_min}) must be less than target_max ({target_max})"
        )

    # Clip to source range
    x_clipped = clip_tensor(input_tensor, source_min, source_max)

    # Normalize to [0, 1]
    x_normalized = (x_clipped - source_min) / (source_max - source_min)

    # Scale to target range
    return x_normalized * (target_max - target_min) + target_min

# ---------------------------------------------------------------------


def denormalize_tensor(
        input_tensor: NumericTensor,
        source_min: float = -0.5,
        source_max: float = 0.5,
        target_min: float = 0.0,
        target_max: float = 255.0,
) -> tf.Tensor:
    """Denormalizes tensor from source range to target range.

    Args:
        input_tensor: Input tensor to be denormalized.
        source_min: Minimum value of source range. Defaults to -0.5.
        source_max: Maximum value of source range. Defaults to 0.5.
        target_min: Minimum value of target range. Defaults to 0.0.
        target_max: Maximum value of target range. Defaults to 255.0.

    Returns:
        tf.Tensor: Denormalized tensor with values in [target_min, target_max].
    """
    return normalize_tensor(
        input_tensor,
        source_min=source_min,
        source_max=source_max,
        target_min=target_min,
        target_max=target_max,
    )

# ---------------------------------------------------------------------


# Aliases for backward compatibility
layer_normalize = normalize_tensor
layer_denormalize = denormalize_tensor

# ---------------------------------------------------------------------
