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
    """
    Layer that clips tensor values to a specified range.

    This layer applies element-wise clipping to input tensors, constraining all values
    to lie within the specified [clip_min, clip_max] range. Values below clip_min are
    set to clip_min, and values above clip_max are set to clip_max. The operation
    preserves tensor shape and is differentiable.

    **Intent**: Provide a reusable, serializable layer for value clipping that can be
    integrated into any Keras model pipeline. Useful for constraining outputs, preventing
    numerical instability, or enforcing data range constraints.

    **Mathematical Operation**:
        output = max(clip_min, min(clip_max, input))

    Applied element-wise to all tensor elements.

    Args:
        clip_min: Float, minimum value for clipping. Values below this threshold
            will be set to clip_min. Must be less than clip_max.
        clip_max: Float, maximum value for clipping. Values above this threshold
            will be set to clip_max. Must be greater than clip_min.
        **kwargs: Additional arguments for Layer base class (name, trainable, etc.).

    Input shape:
        Arbitrary N-D tensor with any shape.

    Output shape:
        Same as input shape. Tensor shape is preserved.

    Raises:
        ValueError: If clip_min >= clip_max.

    Example:
        ```python
        # Clip values to [0, 1] range
        clip_layer = ClipLayer(clip_min=0.0, clip_max=1.0)
        inputs = keras.Input(shape=(784,))
        clipped = clip_layer(inputs)

        # Use in a model pipeline
        model = keras.Sequential([
            keras.layers.Dense(128, activation='tanh'),
            ClipLayer(clip_min=-0.5, clip_max=0.5),  # Constrain tanh output
            keras.layers.Dense(10, activation='softmax')
        ])

        # Clip image data to valid pixel range
        image_clip = ClipLayer(clip_min=0.0, clip_max=255.0, name='pixel_clip')
        ```

    Note:
        This layer is stateless and introduces no trainable parameters. The clipping
        operation is differentiable with gradients of 1 for values within the range
        and 0 for values outside the range.
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
        """Apply clipping to input tensor."""
        return ops.clip(inputs, self.clip_min, self.clip_max)

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Output shape is identical to input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'clip_min': self.clip_min,
            'clip_max': self.clip_max,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NormalizationLayer(keras.layers.Layer):
    """
    Layer that normalizes tensor values from source range to target range.

    This layer performs linear scaling to transform tensor values from a source
    value range [source_min, source_max] to a target range [target_min, target_max].
    The transformation first clips values to the source range, normalizes to [0, 1],
    then scales to the target range. This is commonly used for data preprocessing,
    especially in computer vision_heads tasks.

    **Intent**: Provide a robust, configurable normalization layer that can handle
    different data ranges and scaling requirements. Particularly useful for converting
    between different data representations (e.g., pixel values 0-255 to model inputs -0.5 to 0.5).

    **Mathematical Operation**:
    1. **Clipping**: x_clipped = clip(input, source_min, source_max)
    2. **Normalize to [0, 1]**: x_norm = (x_clipped - source_min) / (source_max - source_min)
    3. **Scale to target**: output = x_norm * (target_max - target_min) + target_min

    Args:
        source_min: Float, minimum value of source range. Input values are expected
            to be in [source_min, source_max]. Defaults to 0.0 (typical for images).
        source_max: Float, maximum value of source range. Must be greater than
            source_min. Defaults to 255.0 (typical for uint8 images).
        target_min: Float, minimum value of target range. Output will be scaled to
            [target_min, target_max]. Defaults to -0.5.
        target_max: Float, maximum value of target range. Must be greater than
            target_min. Defaults to 0.5.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        Arbitrary N-D tensor with any shape.

    Output shape:
        Same as input shape. Values scaled to [target_min, target_max].

    Raises:
        ValueError: If source_min >= source_max or target_min >= target_max.

    Example:
        ```python
        # Standard image normalization (0-255 to -0.5, 0.5)
        img_norm = NormalizationLayer()

        # Custom normalization for different data ranges
        custom_norm = NormalizationLayer(
            source_min=-1.0, source_max=1.0,
            target_min=0.0, target_max=1.0
        )

        # Use in preprocessing pipeline
        model = keras.Sequential([
            NormalizationLayer(name='input_norm'),  # Normalize input images
            keras.layers.Conv2D(32, 3, activation='relu'),
            # ... rest of model
        ])

        # Normalize data to neural network friendly range
        data_norm = NormalizationLayer(
            source_min=0, source_max=1000,
            target_min=-1, target_max=1,
            name='data_preprocessor'
        )
        ```

    Note:
        This layer automatically clips input values to the source range before
        normalization to ensure robust behavior. The transformation is linear
        and fully differentiable.
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
        """Apply normalization to input tensor."""
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
        """Output shape is identical to input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
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
    """
    Layer that denormalizes tensor values from source range to target range.

    This layer performs the inverse operation of NormalizationLayer, transforming
    tensor values from a normalized source range back to their original target range.
    This is commonly used in the output processing of generative models where
    network outputs need to be converted back to interpretable data ranges.

    **Intent**: Provide the inverse transformation to NormalizationLayer, enabling
    seamless conversion from model outputs back to original data ranges. Essential
    for generative models, image synthesis, and any application requiring output
    post-processing.

    **Mathematical Operation**:
    Identical to NormalizationLayer but with swapped source/target semantics:
    1. **Clipping**: x_clipped = clip(input, source_min, source_max)
    2. **Normalize**: x_norm = (x_clipped - source_min) / (source_max - source_min)
    3. **Scale**: output = x_norm * (target_max - target_min) + target_min

    Args:
        source_min: Float, minimum value of source (normalized) range. Defaults to -0.5.
        source_max: Float, maximum value of source (normalized) range. Defaults to 0.5.
        target_min: Float, minimum value of target (denormalized) range. Defaults to 0.0.
        target_max: Float, maximum value of target (denormalized) range. Defaults to 255.0.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        Arbitrary N-D tensor with any shape.

    Output shape:
        Same as input shape. Values scaled to [target_min, target_max].

    Example:
        ```python
        # Standard denormalization (model output to image pixels)
        denorm = DenormalizationLayer()  # -0.5,0.5 -> 0,255

        # Custom denormalization
        custom_denorm = DenormalizationLayer(
            source_min=0.0, source_max=1.0,
            target_min=-1.0, target_max=1.0
        )

        # Use in generator model output
        generator = keras.Sequential([
            # ... generator layers ...
            keras.layers.Dense(784, activation='tanh'),  # Output in [-1, 1]
            DenormalizationLayer(
                source_min=-1.0, source_max=1.0,
                target_min=0.0, target_max=255.0,
                name='output_denorm'
            )
        ])

        # Paired with normalization layer
        norm_layer = NormalizationLayer(source_min=0, source_max=255,
                                       target_min=-0.5, target_max=0.5)
        denorm_layer = DenormalizationLayer(source_min=-0.5, source_max=0.5,
                                           target_min=0, target_max=255)
        ```

    Note:
        This layer is mathematically equivalent to NormalizationLayer with
        swapped source/target parameters. Use this for semantic clarity when
        performing the inverse operation.
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
        """Apply denormalization to input tensor."""
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
        """Output shape is identical to input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
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
    """
    Composite preprocessing layer combining normalization and clipping operations.

    This layer provides a unified interface for common tensor preprocessing operations,
    combining normalization and additional clipping in a single, serializable layer.
    It first normalizes the input from source to target range, then optionally applies
    additional clipping to ensure outputs stay within specified bounds.

    **Intent**: Provide a comprehensive preprocessing layer that handles the most
    common tensor transformation patterns in a single, configurable component.
    Reduces model complexity by combining multiple operations while maintaining
    full configurability and serialization support.

    **Architecture**:
    ```
    Input → Normalize(source→target) → [Optional Clip(final_min, final_max)] → Output
    ```

    Args:
        source_min: Float, minimum value of source range. Defaults to 0.0.
        source_max: Float, maximum value of source range. Defaults to 255.0.
        target_min: Float, minimum value of target range. Defaults to -0.5.
        target_max: Float, maximum value of target range. Defaults to 0.5.
        enable_final_clipping: Boolean, whether to apply additional clipping after
            normalization. Useful for ensuring strict bounds. Defaults to False.
        final_clip_min: Float, minimum value for final clipping. Only used if
            enable_final_clipping=True. Defaults to -1.0.
        final_clip_max: Float, maximum value for final clipping. Only used if
            enable_final_clipping=True. Defaults to 1.0.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        Arbitrary N-D tensor with any shape.

    Output shape:
        Same as input shape. Values in target range, optionally clipped.

    Attributes:
        normalizer: NormalizationLayer instance for primary transformation.
        clipper: ClipLayer instance for final clipping (if enabled).

    Example:
        ```python
        # Standard preprocessing with final clipping
        preprocessor = TensorPreprocessingLayer(
            source_min=0, source_max=255,
            target_min=-1, target_max=1,
            enable_final_clipping=True,
            final_clip_min=-0.9, final_clip_max=0.9
        )

        # Simple normalization without extra clipping
        simple_preprocessor = TensorPreprocessingLayer(
            enable_final_clipping=False
        )

        # Use as input preprocessing in model
        model = keras.Sequential([
            TensorPreprocessingLayer(name='input_preprocessing'),
            keras.layers.Conv2D(32, 3, activation='relu'),
            # ... rest of model
        ])
        ```

    Note:
        This layer demonstrates the composite pattern from the Keras guide,
        using sub-layers that are explicitly built in the build() method for
        proper serialization support.
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
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
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
        """Forward pass through sub-layers."""
        x = self.normalizer(inputs, training=training)

        if self.clipper is not None:
            x = self.clipper(x, training=training)

        return x

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Output shape is identical to input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
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

