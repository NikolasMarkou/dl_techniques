"""Tensor normalization and clipping layers.

This module provides Keras layers for normalizing and denormalizing tensors,
as well as clipping tensor values to specified ranges. These operations
are commonly used in deep learning preprocessing and postprocessing steps.
"""

import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ClipLayer(keras.layers.Layer):
    """Layer that clips tensor values to a specified range.

    This layer applies element-wise clipping to input tensors, constraining
    all values to lie within ``[clip_min, clip_max]``. The operation is
    ``output = max(clip_min, min(clip_max, input))``, applied element-wise.
    The layer is stateless with no trainable parameters. Gradients are 1 for
    values within range and 0 outside.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────┐
        │   Input (any shape)  │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  clip(x, min, max)   │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Output (same shape) │
        └──────────────────────┘

    :param clip_min: Minimum value for clipping. Must be less than clip_max.
    :type clip_min: float
    :param clip_max: Maximum value for clipping. Must be greater than clip_min.
    :type clip_max: float
    :param kwargs: Additional arguments for Layer base class.

    :raises ValueError: If clip_min >= clip_max.
    """

    def __init__(
        self,
        clip_min: float,
        clip_max: float,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if clip_min >= clip_max:
            raise ValueError(
                f"clip_min ({clip_min}) must be less than clip_max ({clip_max})"
            )

        # Store configuration
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply clipping to input tensor.

        :param inputs: Input tensor of arbitrary shape.
        :type inputs: keras.KerasTensor
        :param training: Unused, present for API consistency.
        :type training: Optional[bool]
        :return: Clipped tensor with same shape as input.
        :rtype: keras.KerasTensor
        """
        return ops.clip(inputs, self.clip_min, self.clip_max)

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape (identical to input).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'clip_min': self.clip_min,
            'clip_max': self.clip_max,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NormalizationLayer(keras.layers.Layer):
    """Layer that normalizes tensor values from source range to target range.

    This layer performs linear scaling to transform tensor values from
    ``[source_min, source_max]`` to ``[target_min, target_max]``. The
    transformation first clips values to the source range, normalizes to
    ``[0, 1]``, then scales to the target range:
    ``x_clipped = clip(input, source_min, source_max)``,
    ``x_norm = (x_clipped - source_min) / (source_max - source_min)``,
    ``output = x_norm * (target_max - target_min) + target_min``.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────┐
        │      Input (any shape)       │
        └─────────────┬────────────────┘
                      │
                      ▼
        ┌──────────────────────────────┐
        │  Clip to [src_min, src_max]  │
        └─────────────┬────────────────┘
                      │
                      ▼
        ┌──────────────────────────────┐
        │  Normalize to [0, 1]         │
        └─────────────┬────────────────┘
                      │
                      ▼
        ┌──────────────────────────────┐
        │  Scale to [tgt_min, tgt_max] │
        └─────────────┬────────────────┘
                      │
                      ▼
        ┌──────────────────────────────┐
        │     Output (same shape)      │
        └──────────────────────────────┘

    :param source_min: Minimum value of source range. Defaults to 0.0.
    :type source_min: float
    :param source_max: Maximum value of source range. Must be greater than
        source_min. Defaults to 255.0.
    :type source_max: float
    :param target_min: Minimum value of target range. Defaults to -0.5.
    :type target_min: float
    :param target_max: Maximum value of target range. Must be greater than
        target_min. Defaults to 0.5.
    :type target_max: float
    :param kwargs: Additional arguments for Layer base class.

    :raises ValueError: If source_min >= source_max or target_min >= target_max.
    """

    def __init__(
        self,
        source_min: float = 0.0,
        source_max: float = 255.0,
        target_min: float = -0.5,
        target_max: float = 0.5,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate ranges
        if source_min >= source_max:
            raise ValueError(
                f"source_min ({source_min}) must be less than source_max ({source_max})"
            )
        if target_min >= target_max:
            raise ValueError(
                f"target_min ({target_min}) must be less than target_max ({target_max})"
            )

        # Store configuration
        self.source_min = float(source_min)
        self.source_max = float(source_max)
        self.target_min = float(target_min)
        self.target_max = float(target_max)

        # Pre-compute scaling factors for efficiency
        self.source_range = self.source_max - self.source_min
        self.target_range = self.target_max - self.target_min

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply normalization to input tensor.

        :param inputs: Input tensor of arbitrary shape.
        :type inputs: keras.KerasTensor
        :param training: Unused, present for API consistency.
        :type training: Optional[bool]
        :return: Normalized tensor with same shape as input.
        :rtype: keras.KerasTensor
        """
        # Clip to source range
        x_clipped = ops.clip(inputs, self.source_min, self.source_max)

        # Normalize to [0, 1]
        x_normalized = (x_clipped - self.source_min) / self.source_range

        # Scale to target range
        return x_normalized * self.target_range + self.target_min

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape (identical to input).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'source_min': self.source_min,
            'source_max': self.source_max,
            'target_min': self.target_min,
            'target_max': self.target_max,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DenormalizationLayer(keras.layers.Layer):
    """Layer that denormalizes tensor values from source range to target range.

    This layer performs the inverse operation of NormalizationLayer, transforming
    tensor values from a normalized source range back to their original target
    range. The mathematical operation is identical to NormalizationLayer but with
    swapped source/target semantics:
    ``x_clipped = clip(input, source_min, source_max)``,
    ``x_norm = (x_clipped - source_min) / (source_max - source_min)``,
    ``output = x_norm * (target_max - target_min) + target_min``.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────┐
        │      Input (any shape)       │
        └─────────────┬────────────────┘
                      │
                      ▼
        ┌──────────────────────────────┐
        │  Clip to [src_min, src_max]  │
        └─────────────┬────────────────┘
                      │
                      ▼
        ┌──────────────────────────────┐
        │  Normalize to [0, 1]         │
        └─────────────┬────────────────┘
                      │
                      ▼
        ┌──────────────────────────────┐
        │  Scale to [tgt_min, tgt_max] │
        └─────────────┬────────────────┘
                      │
                      ▼
        ┌──────────────────────────────┐
        │     Output (same shape)      │
        └──────────────────────────────┘

    :param source_min: Minimum value of source (normalized) range. Defaults to -0.5.
    :type source_min: float
    :param source_max: Maximum value of source (normalized) range. Defaults to 0.5.
    :type source_max: float
    :param target_min: Minimum value of target (denormalized) range. Defaults to 0.0.
    :type target_min: float
    :param target_max: Maximum value of target (denormalized) range. Defaults to 255.0.
    :type target_max: float
    :param kwargs: Additional arguments for Layer base class.

    :raises ValueError: If source_min >= source_max or target_min >= target_max.
    """

    def __init__(
        self,
        source_min: float = -0.5,
        source_max: float = 0.5,
        target_min: float = 0.0,
        target_max: float = 255.0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate ranges
        if source_min >= source_max:
            raise ValueError(
                f"source_min ({source_min}) must be less than source_max ({source_max})"
            )
        if target_min >= target_max:
            raise ValueError(
                f"target_min ({target_min}) must be less than target_max ({target_max})"
            )

        # Store configuration
        self.source_min = float(source_min)
        self.source_max = float(source_max)
        self.target_min = float(target_min)
        self.target_max = float(target_max)

        # Pre-compute scaling factors for efficiency
        self.source_range = self.source_max - self.source_min
        self.target_range = self.target_max - self.target_min

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply denormalization to input tensor.

        :param inputs: Input tensor of arbitrary shape.
        :type inputs: keras.KerasTensor
        :param training: Unused, present for API consistency.
        :type training: Optional[bool]
        :return: Denormalized tensor with same shape as input.
        :rtype: keras.KerasTensor
        """
        # Clip to source range
        x_clipped = ops.clip(inputs, self.source_min, self.source_max)

        # Normalize to [0, 1]
        x_normalized = (x_clipped - self.source_min) / self.source_range

        # Scale to target range
        return x_normalized * self.target_range + self.target_min

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape (identical to input).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'source_min': self.source_min,
            'source_max': self.source_max,
            'target_min': self.target_min,
            'target_max': self.target_max,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TensorPreprocessingLayer(keras.layers.Layer):
    """Composite preprocessing layer combining normalization and clipping.

    This layer provides a unified interface for common tensor preprocessing,
    combining normalization from source to target range with optional
    additional clipping to ensure outputs stay within specified bounds.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │          Input (any shape)            │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  NormalizationLayer                  │
        │  (source → target range)             │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  ClipLayer (optional)                │
        │  (final_clip_min, final_clip_max)    │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │         Output (same shape)          │
        └──────────────────────────────────────┘

    :param source_min: Minimum value of source range. Defaults to 0.0.
    :type source_min: float
    :param source_max: Maximum value of source range. Defaults to 255.0.
    :type source_max: float
    :param target_min: Minimum value of target range. Defaults to -0.5.
    :type target_min: float
    :param target_max: Maximum value of target range. Defaults to 0.5.
    :type target_max: float
    :param enable_final_clipping: Whether to apply additional clipping after
        normalization. Defaults to False.
    :type enable_final_clipping: bool
    :param final_clip_min: Minimum value for final clipping. Only used if
        enable_final_clipping is True. Defaults to -1.0.
    :type final_clip_min: float
    :param final_clip_max: Maximum value for final clipping. Only used if
        enable_final_clipping is True. Defaults to 1.0.
    :type final_clip_max: float
    :param kwargs: Additional arguments for Layer base class.
    """

    def __init__(
        self,
        source_min: float = 0.0,
        source_max: float = 255.0,
        target_min: float = -0.5,
        target_max: float = 0.5,
        enable_final_clipping: bool = False,
        final_clip_min: float = -1.0,
        final_clip_max: float = 1.0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.source_min = float(source_min)
        self.source_max = float(source_max)
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        self.enable_final_clipping = enable_final_clipping
        self.final_clip_min = float(final_clip_min)
        self.final_clip_max = float(final_clip_max)

        # CREATE sub-layers in __init__ (they are unbuilt)
        self.normalizer = NormalizationLayer(
            source_min=source_min,
            source_max=source_max,
            target_min=target_min,
            target_max=target_max,
            name='normalizer'
        )

        if enable_final_clipping:
            self.clipper = ClipLayer(
                clip_min=final_clip_min,
                clip_max=final_clip_max,
                name='final_clipper'
            )
        else:
            self.clipper = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all its sub-layers.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Build sub-layers in computational order
        self.normalizer.build(input_shape)

        if self.clipper is not None:
            # Normalization doesn't change shape
            self.clipper.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through sub-layers.

        :param inputs: Input tensor of arbitrary shape.
        :type inputs: keras.KerasTensor
        :param training: Unused, present for API consistency.
        :type training: Optional[bool]
        :return: Preprocessed tensor with same shape as input.
        :rtype: keras.KerasTensor
        """
        x = self.normalizer(inputs, training=training)

        if self.clipper is not None:
            x = self.clipper(x, training=training)

        return x

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape (identical to input).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'source_min': self.source_min,
            'source_max': self.source_max,
            'target_min': self.target_min,
            'target_max': self.target_max,
            'enable_final_clipping': self.enable_final_clipping,
            'final_clip_min': self.final_clip_min,
            'final_clip_max': self.final_clip_max,
        })
        return config

# ---------------------------------------------------------------------

