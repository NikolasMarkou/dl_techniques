"""
Implementation of the Gated MLP (gMLP) architecture from the paper:
"Pay Attention to MLPs" by Liu et al., 2021 (https://arxiv.org/abs/2105.08050)

This module implements a Gated Multi-Layer Perceptron (MLP) layer for use in
neural networks. The layer combines gating mechanisms with 1x1 convolutions
to create a powerful feature transformation block without self-attention.

Architecture:

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
"""

import keras
from typing import Optional, Union, Literal, Any, Tuple, Callable


@keras.saving.register_keras_serializable()
class GatedMLP(keras.layers.Layer):
    """
    A Gated MLP layer implementation using 1x1 convolutions.

    This layer implements a gated MLP architecture where the input is processed through
    three separate 1x1 convolution paths: gate, up, and down projections. The gating
    mechanism allows the network to selectively focus on relevant features.

    The architecture follows the pattern from "Pay Attention to MLPs" where:
    1. Input is processed through parallel gate and up convolutions
    2. Both branches apply activation functions
    3. Element-wise multiplication combines the branches
    4. A down convolution processes the combined features
    5. Final activation produces the output

    Args:
        filters: Integer, number of filters for the output convolution. Must be positive.
        use_bias: Boolean, whether to use bias in the convolution layers. Defaults to True.
        kernel_initializer: String or Initializer, initializer for the kernel weights matrices.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer, initializer for the bias vectors.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for the kernel weights.
        bias_regularizer: Optional regularizer for the bias vectors.
        attention_activation: String or callable, activation function for the gate and up projections.
            Can be 'relu', 'gelu', 'swish', 'linear', or any valid Keras activation.
            Defaults to 'relu'.
        output_activation: String or callable, activation function for the output.
            Can be 'relu', 'gelu', 'swish', 'linear', or any valid Keras activation.
            Defaults to 'linear'.
        data_format: String, either 'channels_last' or 'channels_first'.
            Defaults to None (uses keras.backend.image_data_format()).
        **kwargs: Additional arguments passed to the parent Layer class.

    Input shape:
        4D tensor with shape:
        - If data_format='channels_last': (batch_size, height, width, channels)
        - If data_format='channels_first': (batch_size, channels, height, width)

    Output shape:
        4D tensor with shape:
        - If data_format='channels_last': (batch_size, height, width, filters)
        - If data_format='channels_first': (batch_size, filters, height, width)

    Raises:
        ValueError: If filters is not positive or data_format is invalid.

    Example:
        ```python
        # Basic usage
        x = np.random.rand(4, 32, 32, 64)  # Input feature map
        gmlp = GatedMLP(filters=128, attention_activation='gelu')
        y = gmlp(x)
        print(y.shape)  # (4, 32, 32, 128)

        # Advanced usage with regularization
        gmlp = GatedMLP(
            filters=256,
            attention_activation='gelu',
            output_activation='swish',
            kernel_regularizer=keras.regularizers.L2(1e-4),
            use_bias=False
        )

        # In a model
        inputs = keras.Input(shape=(224, 224, 3))
        x = keras.layers.Conv2D(64, 3, activation='relu')(inputs)
        x = GatedMLP(128, attention_activation='gelu')(x)
        outputs = keras.layers.GlobalAveragePooling2D()(x)
        model = keras.Model(inputs, outputs)
        ```

    Note:
        This implementation follows the modern Keras 3 pattern where all sub-layers
        are created in __init__ and Keras handles the building automatically. This
        ensures proper serialization and eliminates common build errors.
    """

    def __init__(
        self,
        filters: int,
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        attention_activation: Union[str, Callable] = 'relu',
        output_activation: Union[str, Callable] = 'linear',
        data_format: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")

        # Store configuration - ALL parameters needed for reconstruction
        self.filters = filters
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.attention_activation = attention_activation
        self.output_activation = output_activation
        self.data_format = data_format or keras.backend.image_data_format()

        # Validate data format
        if self.data_format not in {'channels_first', 'channels_last'}:
            raise ValueError(
                f"data_format must be 'channels_first' or 'channels_last', "
                f"got {self.data_format}"
            )

        # Get activation functions
        self.attention_activation_fn = keras.activations.get(attention_activation)
        self.output_activation_fn = keras.activations.get(output_activation)

        # CREATE all sub-layers in __init__ - this is the key change!
        # Gate convolution path
        self.conv_gate = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            data_format=self.data_format,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='gate_conv'
        )

        # Up convolution path
        self.conv_up = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            data_format=self.data_format,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='up_conv'
        )

        # Down convolution path
        self.conv_down = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            data_format=self.data_format,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='down_conv'
        )

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass for the GatedMLP layer.

        Args:
            inputs: Input tensor with shape (batch_size, height, width, channels)
                for channels_last format or (batch_size, channels, height, width)
                for channels_first format.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor after applying the Gated MLP operations.
        """
        # Gate branch: input -> conv -> activation
        x_gate = self.conv_gate(inputs, training=training)
        x_gate = self.attention_activation_fn(x_gate)

        # Up branch: input -> conv -> activation
        x_up = self.conv_up(inputs, training=training)
        x_up = self.attention_activation_fn(x_up)

        # Combine gate and up paths with element-wise multiplication (gating)
        x_combined = keras.ops.multiply(x_gate, x_up)

        # Down path: combined -> conv -> activation
        x_gated_mlp = self.conv_down(x_combined, training=training)
        output = self.output_activation_fn(x_gated_mlp)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple.
        """
        # Convert to list for manipulation
        input_shape_list = list(input_shape)

        if self.data_format == 'channels_last':
            # (..., height, width, channels) -> (..., height, width, filters)
            input_shape_list[-1] = self.filters
        else:  # channels_first
            # (..., channels, height, width) -> (..., filters, height, width)
            input_shape_list[-3] = self.filters

        return tuple(input_shape_list)

    def get_config(self) -> dict[str, Any]:
        """
        Return the configuration of the layer for serialization.

        This method must return ALL arguments needed to recreate the layer
        via __init__.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'attention_activation': keras.activations.serialize(self.attention_activation_fn),
            'output_activation': keras.activations.serialize(self.output_activation_fn),
            'data_format': self.data_format,
        })
        return config

    # Note: NO get_build_config() or build_from_config() methods needed!
    # Keras handles the build lifecycle automatically with the modern pattern.