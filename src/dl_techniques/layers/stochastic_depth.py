import keras
import tensorflow as tf
from typing import Optional, Dict, Any, Union

# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class StochasticDepth(keras.layers.Layer):
    """Implements Stochastic Depth for deep networks.

    This layer implements batch-wise dropping of residual paths as described in the
    Stochastic Depth paper. Unlike sample-wise dropping methods (like DropPath in timm),
    this implementation drops the same paths for all samples in a batch.

    Args:
        drop_path_rate: Float between 0 and 1, probability of dropping the residual path.
            Default is 0.5.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Raises:
        ValueError: If drop_path_rate is not in the interval [0, 1).

    References:
        - Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)
        - timm library implementation (https://github.com/rwightman/pytorch-image-models)
    """

    def __init__(
            self,
            drop_path_rate: float = 0.5,
            **kwargs: Any
    ) -> None:
        """Initializes the StochasticDepth layer.

        Args:
            drop_path_rate: Probability of dropping the residual path.
            **kwargs: Additional keyword arguments passed to the parent Layer class.

        Raises:
            ValueError: If drop_path_rate is not in [0, 1).
        """
        super().__init__(**kwargs)

        # Validate drop_path_rate
        if not isinstance(drop_path_rate, (int, float)):
            raise TypeError("drop_path_rate must be a number")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError(
                f"drop_path_rate must be in [0, 1), got {drop_path_rate}"
            )

        self.drop_path_rate = float(drop_path_rate)
        self.dropout: Optional[keras.layers.Dropout] = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Builds the layer's internal state.

        Creates a Dropout layer with appropriate noise shape based on input dimensions.

        Args:
            input_shape: TensorShape of the input tensor.
        """
        if not isinstance(input_shape, (tf.TensorShape, list, tuple)):
            raise TypeError(f"Expected TensorShape, got {type(input_shape)}")

        dims = len(input_shape)
        if dims < 2:
            raise ValueError(
                f"Input must have at least 2 dimensions, got {dims}"
            )

        # Create noise shape: (batch_size, 1, 1, ..., 1)
        noise_shape = (input_shape[0],) + (1,) * (dims - 1)

        self.dropout = keras.layers.Dropout(
            rate=self.drop_path_rate,
            noise_shape=noise_shape
        )

        self.dropout.build(input_shape)

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether in training mode. If None,
                the layer's default behavior is used.

        Returns:
            Output tensor with same shape as input.

        Raises:
            ValueError: If dropout layer is not built.
        """
        return self.dropout(inputs, training=training)

    def compute_output_shape(
            self,
            input_shape: Union[tf.TensorShape, tuple]
    ) -> tf.TensorShape:
        """Computes the output shape of the layer.

        Args:
            input_shape: Shape of input tensor.

        Returns:
            TensorShape representing output shape.
        """
        return tf.TensorShape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Returns the config dictionary for layer serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        base_config = super().get_config()
        config = {"drop_path_rate": self.drop_path_rate}
        return {**base_config, **config}

# ---------------------------------------------------------------------
