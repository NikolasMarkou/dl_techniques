import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple


@keras.saving.register_keras_serializable()
class LogitNorm(keras.layers.Layer):
    """
    LogitNorm layer for classification tasks.

    This layer implements logit normalization by applying L2 normalization with
    a learned temperature parameter. This technique helps stabilize training and
    can improve model calibration by preventing the model from being overconfident
    in its predictions.

    The layer normalizes input logits by their L2 norm and scales by a temperature
    parameter. This is particularly effective for classification tasks where
    model calibration is important.

    Mathematical formulation:
        output = inputs / (||inputs||_2 * temperature)

    Where ||inputs||_2 is the L2 norm computed along the specified axis.

    Args:
        temperature: Float, temperature scaling parameter. Higher values produce
            more spread-out logits, lower values produce more concentrated logits.
            Must be positive. Defaults to 0.04 (optimal for CIFAR-10 according to paper).
        axis: Integer, axis along which to perform L2 normalization.
            Defaults to -1 (last dimension).
        epsilon: Float, small constant added for numerical stability when computing
            the L2 norm. Must be positive. Defaults to 1e-7.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., features)`.
        Most commonly used with 2D logits: `(batch_size, num_classes)`.

    Output shape:
        Same as input shape.

    Example:
        ```python
        # Basic usage for classification
        logits = keras.Input(shape=(10,))  # 10 classes
        normalized_logits = LogitNorm()(logits)

        # Custom temperature for different calibration
        layer = LogitNorm(temperature=0.1)

        # In a classification model
        inputs = keras.Input(shape=(784,))
        x = keras.layers.Dense(512, activation='relu')(inputs)
        x = keras.layers.Dense(10)(x)  # Raw logits
        outputs = LogitNorm(temperature=0.05)(x)  # Normalize before softmax
        model = keras.Model(inputs, outputs)
        ```

    References:
        - Mitigating Neural Network Overconfidence with Logit Normalization
        - https://arxiv.org/abs/2205.09310

    Raises:
        ValueError: If temperature or epsilon is not positive.

    Note:
        This layer does not contain trainable parameters. The temperature is
        a fixed hyperparameter that should be tuned during model development.
        For learnable temperature scaling, consider using a separate Dense layer.
    """

    def __init__(
            self,
            temperature: float = 0.04,
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Store configuration
        self.temperature = temperature
        self.axis = axis
        self.epsilon = epsilon

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply logit normalization to input tensor.

        Args:
            inputs: Input logits tensor of any shape.
            training: Boolean indicating training mode. Not used in this layer
                but included for consistency with Keras Layer interface.

        Returns:
            Normalized logits tensor with the same shape as inputs.
        """
        # Compute L2 norm along specified axis, ensuring numerical stability
        norm = ops.sqrt(
            ops.maximum(
                ops.sum(ops.square(inputs), axis=self.axis, keepdims=True),
                self.epsilon
            )
        )

        # Normalize logits and scale by temperature
        return inputs / (norm * self.temperature)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple, identical to input shape since this is
            an element-wise transformation.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration with all
            parameters needed to recreate the layer.
        """
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'axis': self.axis,
            'epsilon': self.epsilon,
        })
        return config