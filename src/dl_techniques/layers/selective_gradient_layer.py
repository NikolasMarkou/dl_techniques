import keras
from typing import Tuple, List, Dict, Any, Optional

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SelectiveGradientMask(keras.layers.Layer):
    """
    A layer that selectively stops gradients based on a binary mask.

    This layer implements a dual-path mechanism during training that allows
    selective gradient flow control while preserving the forward pass:
    - Stopped gradient path: For masked regions (mask = 1)
    - Normal gradient path: For unmasked regions (mask = 0)

    The computation flow during training:
    ```
    input -> [stop_gradient(input) * mask + input * (1-mask)] -> output
    ```

    During inference, the input passes through unchanged for optimal performance.

    Key Features:
        - Selective gradient blocking based on binary masks
        - Preserves forward pass behavior in all modes
        - Efficient dual-path computation during training
        - Shape-preserving operation
        - Full serialization support

    Args:
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        List of two tensors with identical shapes:
        - signal: Primary input tensor of shape (batch_size, ...)
        - mask: Binary mask tensor of shape (batch_size, ...) where:
          * 1 indicates positions where gradients should be stopped
          * 0 indicates positions where gradients should flow normally

    Output shape:
        Tensor with same shape as input signal: (batch_size, ...)

    Example:
        ```python
        # Basic usage with explicit inputs
        signal_input = keras.Input(shape=(28, 28, 1), name='signal')
        mask_input = keras.Input(shape=(28, 28, 1), name='mask')
        masked_output = SelectiveGradientMask()([signal_input, mask_input])

        model = keras.Model(
            inputs=[signal_input, mask_input],
            outputs=masked_output
        )

        # Usage with dynamic mask generation
        features = keras.Input(shape=(64,))
        threshold_layer = keras.layers.Dense(64, activation='sigmoid')
        thresholded = threshold_layer(features)

        # Create binary mask based on threshold
        mask = keras.ops.cast(keras.ops.greater(thresholded, 0.5), "float32")
        selective_output = SelectiveGradientMask()([thresholded, mask])

        # In training scenarios for selective learning
        encoder = keras.layers.Dense(128, activation='relu')
        encoded = encoder(features)

        # Apply selective masking to control gradient flow
        attention_mask = keras.ops.cast(keras.ops.greater(encoded, 0), "float32")
        selectively_trained = SelectiveGradientMask()([encoded, attention_mask])
        ```

    Note:
        - Mask values should be binary (0 or 1) for optimal behavior
        - Both input tensors must have identical shapes
        - Gradients are completely blocked where mask = 1
        - Forward pass remains unchanged during inference for efficiency
        - The layer automatically handles training mode detection
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the SelectiveGradientMask layer.

        Args:
            **kwargs: Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass with selective gradient masking.

        During training, applies selective gradient stopping using a dual-path approach:
        1. Creates stopped gradient path for masked regions (mask = 1)
        2. Creates normal gradient path for unmasked regions (mask = 0)
        3. Combines both paths to preserve forward behavior

        During inference, passes the signal through unchanged for efficiency.

        Args:
            inputs: List containing [signal, mask] tensors with identical shapes.
            training: Boolean indicating training mode. If None, uses Keras' global training phase.

        Returns:
            keras.KerasTensor: Output tensor with same shape as signal input.
                             During training: selective gradient masking applied.
                             During inference: signal passed through unchanged.

        Raises:
            ValueError: If inputs is not a list of exactly 2 tensors or shapes don't match.
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                f"SelectiveGradientMask expects a list of exactly 2 tensors [signal, mask], "
                f"got {type(inputs)} with length {len(inputs) if hasattr(inputs, '__len__') else 'unknown'}"
            )

        signal, mask = inputs

        # Validate tensor shapes match
        if signal.shape != mask.shape:
            raise ValueError(
                f"Signal and mask tensors must have identical shapes. "
                f"Got signal: {signal.shape}, mask: {mask.shape}"
            )

        # During inference: pass signal through unchanged for efficiency
        if not training:
            return signal

        # During training: apply selective gradient masking

        # Validate mask values are reasonable (optional warning, not blocking)
        # This is a best-practice check but doesn't prevent execution
        mask_min = keras.ops.min(mask)
        mask_max = keras.ops.max(mask)

        # Check for values significantly outside [0, 1] range
        if keras.ops.convert_to_numpy(mask_min) < -0.1 or keras.ops.convert_to_numpy(mask_max) > 1.1:
            # Note: In production, you might want to log this, but we avoid external dependencies
            pass

        # Create dual-path selective masking:
        # Path 1: Stopped gradient for masked regions (mask = 1)
        stopped_gradient_path = keras.ops.multiply(
            keras.ops.stop_gradient(signal),
            mask
        )

        # Path 2: Normal gradient for unmasked regions (mask = 0)
        normal_gradient_path = keras.ops.multiply(
            signal,
            keras.ops.subtract(1.0, mask)
        )

        # Combine both paths to preserve forward pass behavior
        # while controlling gradient flow
        output = keras.ops.add(stopped_gradient_path, normal_gradient_path)

        return output

    def compute_output_shape(
        self,
        input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: List of input shapes [signal_shape, mask_shape].

        Returns:
            Tuple representing the output shape (same as signal shape).

        Raises:
            ValueError: If input_shape is not a list of exactly 2 shapes or shapes don't match.
        """
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                f"Expected list of 2 input shapes [signal_shape, mask_shape], "
                f"got {type(input_shape)} with length {len(input_shape) if hasattr(input_shape, '__len__') else 'unknown'}"
            )

        signal_shape, mask_shape = input_shape

        if signal_shape != mask_shape:
            raise ValueError(
                f"Signal and mask shapes must be identical. "
                f"Got signal: {signal_shape}, mask: {mask_shape}"
            )

        # Output shape is the same as the signal input shape
        return signal_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration. This layer has no
            additional parameters beyond the base Layer class.
        """
        config = super().get_config()
        # This layer has no additional parameters to serialize
        return config

# ---------------------------------------------------------------------
