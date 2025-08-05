import keras
from typing import Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class ChannelAttention(keras.layers.Layer):
    """Channel attention module of CBAM.

    This module applies channel-wise attention by using both max-pooling
    and average-pooling features, followed by a shared MLP network.

    Args:
        channels: Number of input channels.
        ratio: Reduction ratio for the shared MLP.
        kernel_initializer: Initializer for the dense layer kernels.
        kernel_regularizer: Regularizer function for the dense layer kernels.
        use_bias: Whether to include bias in dense layers.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
        self,
        channels: int,
        ratio: int = 8,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.ratio = ratio
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias

        # Sublayers will be initialized in build()
        self.shared_mlp = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's weights based on input shape.

        Args:
            input_shape: Shape tuple indicating the input shape of the layer.
        """
        self._build_input_shape = input_shape

        # Create shared MLP layers
        self.shared_mlp = keras.Sequential([
            keras.layers.Dense(
                self.channels // self.ratio,
                activation='relu',
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='channel_attention_dense_1'
            ),
            keras.layers.Dense(
                self.channels,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='channel_attention_dense_2'
            )
        ], name='shared_mlp')

        # Build the sequential model
        dummy_input_shape = (1, 1, self.channels)
        self.shared_mlp.build(dummy_input_shape)

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Apply channel attention to input tensor.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Channel attention map of shape (batch_size, 1, 1, channels).
        """
        # Apply global average pooling and global max pooling
        avg_pool = keras.ops.mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = keras.ops.max(inputs, axis=[1, 2], keepdims=True)

        # Pass through shared MLP
        avg_out = self.shared_mlp(avg_pool, training=training)
        max_out = self.shared_mlp(max_pool, training=training)

        # Combine and apply sigmoid activation
        return keras.ops.sigmoid(avg_out + max_out)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape.
        """
        input_shape_list = list(input_shape)
        return tuple([input_shape_list[0], 1, 1, input_shape_list[-1]])

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "ratio": self.ratio,
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

# ---------------------------------------------------------------------

