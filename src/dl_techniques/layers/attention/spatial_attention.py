import keras
from typing import Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class SpatialAttention(keras.layers.Layer):
    """Spatial attention module of CBAM.

    This module applies spatial attention using channel-wise pooling
    followed by a convolution operation.

    Args:
        kernel_size: Size of the convolution kernel.
        kernel_initializer: Initializer for the convolution kernels.
        kernel_regularizer: Regularizer function for the convolution kernels.
        use_bias: Whether to include bias in convolution layer.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
        self,
        kernel_size: int = 7,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias

        # Sublayers will be initialized in build()
        self.conv = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's weights based on input shape.

        Args:
            input_shape: Shape tuple indicating the input shape of the layer.
        """
        self._build_input_shape = input_shape

        # Create convolution layer
        self.conv = keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            activation='sigmoid',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=self.use_bias,
            name='spatial_attention_conv'
        )

        # Build the convolution layer with concatenated pooling features (2 channels)
        conv_input_shape = list(input_shape)
        conv_input_shape[-1] = 2  # avg_pool + max_pool
        self.conv.build(tuple(conv_input_shape))

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Apply spatial attention to input tensor.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Spatial attention map of shape (batch_size, height, width, 1).
        """
        # Apply channel-wise pooling
        avg_pool = keras.ops.mean(inputs, axis=-1, keepdims=True)
        max_pool = keras.ops.max(inputs, axis=-1, keepdims=True)

        # Concatenate pooled features
        concat = keras.ops.concatenate([avg_pool, max_pool], axis=-1)

        # Apply convolution and sigmoid activation
        return self.conv(concat, training=training)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape.
        """
        input_shape_list = list(input_shape)
        return tuple(input_shape_list[:-1] + [1])

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

