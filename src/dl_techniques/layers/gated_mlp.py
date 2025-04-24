"""
Implementation of the Gated MLP (gMLP) architecture from the paper:
"Pay Attention to MLPs" by Liu et al., 2021 (https://arxiv.org/abs/2105.08050)

This module implements a Gated Multi-Layer Perceptron (MLP) layer for use in
neural networks. The layer combines gating mechanisms with 1x1 convolutions
to create a powerful feature transformation block without self-attention.
"""

import keras
from typing import Optional, Union, Literal, Any

from dl_techniques.layers.conv2d_builder import activation_wrapper


@keras.saving.register_keras_serializable()
class GatedMLP(keras.layers.Layer):
    """A Gated MLP layer implementation using 1x1 convolutions.

    This layer implements a gated MLP architecture where the input is processed through
    three separate 1x1 convolution paths: gate, up, and down projections. The gating
    mechanism allows the network to selectively focus on relevant features.

    Architecture:
    ------------
                       +------------------+
                       |      Input       |
                       +--------+---------+
                                |
                     +----------+-----------+
          +----------+       Split         +-----------+
          |          +--------------------+            |
          |                                            |
    +-----v------+                             +-------v-----+
    | Gate Conv  |                             |   Up Conv   |
    | (1x1 Conv) |                             | (1x1 Conv)  |
    +-----+------+                             +-------+-----+
          |                                            |
    +-----v------+                             +-------v-----+
    | Activation |                             | Activation  |
    | (ReLU/GELU)|                             | (ReLU/GELU) |
    +-----+------+                             +-------+-----+
          |                                            |
          |          +--------------------+            |
          +----------> Element-wise       <------------+
                     |  Multiplication    |
                     +----------+---------+
                                |
                     +----------v---------+
                     |    Down Conv       |
                     |    (1x1 Conv)      |
                     +----------+---------+
                                |
                     +----------v---------+
                     |    Activation      |
                     | (Linear/ReLU/etc.) |
                     +----------+---------+
                                |
                     +----------v---------+
                     |      Output        |
                     +--------------------+

    Args:
        filters: Number of filters for the output convolution.
        use_bias: Whether to use bias in the convolution layers.
        kernel_initializer: Initializer for the kernel weights matrices.
        bias_initializer: Initializer for the bias vectors.
        kernel_regularizer: Regularizer function applied to kernel weights.
        bias_regularizer: Regularizer function applied to bias vectors.
        attention_activation: Activation function for the gate and up projections.
        output_activation: Activation function for the output.
        data_format: String, either "channels_last" or "channels_first".
        **kwargs: Additional arguments passed to the parent Layer class.

    Input shape:
        4D tensor with shape:
        - If data_format="channels_last": (batch_size, height, width, channels)
        - If data_format="channels_first": (batch_size, channels, height, width)

    Output shape:
        4D tensor with shape:
        - Same as input shape but with 'filters' output channels

    Example:
        >>> x = np.random.rand(4, 32, 32, 64)  # Input feature map
        >>> gmlp = GatedMLP(filters=128, attention_activation="gelu")
        >>> y = gmlp(x)
        >>> print(y.shape)
        (4, 32, 32, 128)
    """

    def __init__(
            self,
            filters: int,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            attention_activation: Literal["relu", "gelu", "swish", "linear"] = "relu",
            output_activation: Literal["relu", "gelu", "swish", "linear"] = "linear",
            data_format: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.filters = filters
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.attention_activation = attention_activation
        self.output_activation = output_activation
        self.data_format = data_format or keras.backend.image_data_format()

        if self.data_format not in {"channels_first", "channels_last"}:
            raise ValueError(f"data_format must be 'channels_first' or 'channels_last', got {data_format}")

        # These will be initialized in build()
        self.conv_gate = None
        self.conv_up = None
        self.conv_down = None

        # Store the build shape for serialization
        self._build_input_shape = None

    def build(self, input_shape):
        """Build the GatedMLP layer.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Create the convolution layers
        self.conv_gate = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            data_format=self.data_format,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )
        self.conv_gate.build(input_shape)

        self.conv_up = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            data_format=self.data_format,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )
        self.conv_up.build(input_shape)

        # Calculate the shape for the intermediate tensor after gate and up paths
        input_shape_list = list(input_shape)
        if self.data_format == "channels_last":
            intermediate_shape = tuple(input_shape_list[:-1] + [self.filters])
        else:  # channels_first
            intermediate_shape = tuple([input_shape_list[0], self.filters] + input_shape_list[2:])

        self.conv_down = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            data_format=self.data_format,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )
        self.conv_down.build(intermediate_shape)

        # Set up activation functions directly instead of using external function
        self.attention_activation_fn = activation_wrapper(self.attention_activation)
        self.output_activation_fn = activation_wrapper(self.output_activation)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass for the GatedMLP layer.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            The output tensor after applying the Gated MLP operations.
        """
        # Gate branch
        x_gate = self.conv_gate(inputs, training=training)
        x_gate = self.attention_activation_fn(x_gate)

        # Up branch
        x_up = self.conv_up(inputs, training=training)
        x_up = self.attention_activation_fn(x_up)

        # Combine gate and up paths with element-wise multiplication
        x_combined = x_gate * x_up

        # Down path
        x_gated_mlp = self.conv_down(x_combined, training=training)

        # Apply output activation
        output = self.output_activation_fn(x_gated_mlp)

        return output

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape.
        """
        # Convert input_shape to a list if it's a tuple or TensorShape
        input_shape_list = list(input_shape)

        if self.data_format == "channels_last":
            return tuple(input_shape_list[:-1] + [self.filters])
        else:  # channels_first
            return tuple([input_shape_list[0], self.filters] + input_shape_list[2:])

    def get_config(self):
        """Return the configuration of the layer for serialization.

        Returns:
            Dictionary containing the configuration of the layer.
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "attention_activation": self.attention_activation,
            "output_activation": self.output_activation,
            "data_format": self.data_format,
        })
        return config

    def get_build_config(self):
        """Get the config needed to build the layer from a config.

        This method is needed for proper model saving and loading.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config):
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])