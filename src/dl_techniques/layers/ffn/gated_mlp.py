"""
A spatially-gated MLP block as an alternative to self-attention.

This layer realizes the Gated MLP (gMLP) architecture, proposed as a
computationally efficient yet powerful alternative to the self-attention
mechanisms prevalent in Transformer models. It adapts the principles of Gated
Linear Units (GLUs) for spatial data (e.g., image feature maps) by using
1x1 convolutions, which act as position-wise linear transformations.

The core idea is to replace the explicit token-mixing of self-attention with
an implicit, spatially-aware gating mechanism. This allows the network to
dynamically control the flow of information at each spatial location based on
the local features, without incurring the quadratic complexity of attention.

Architectural Overview:
The gMLP operates through a dual-pathway gating structure, where all linear
projections are implemented as 1x1 convolutions:

1.  **Parallel Projections**: The input tensor is fed into two independent
    1x1 convolutional layers. These function as position-wise dense layers,
    projecting the feature vector at each spatial location into an
    intermediate representation. One pathway learns a "gate" representation,
    while the other learns a "value" (or "up") representation.

2.  **Non-linear Activation**: The outputs of both the gate and value pathways
    are passed through a non-linear activation function.

3.  **Gating Mechanism**: The activated gate tensor is element-wise multiplied
    with the activated value tensor. This is the central operation of the
    layer. The gate effectively learns a dynamic, content-aware spatial mask
    that modulates the information carried by the value pathway. Features at
    locations where the gate has high activation are preserved, while those
    at locations with low activation are suppressed.

4.  **Output Projection**: The resulting gated tensor is passed through a
    final 1x1 convolutional layer (the "down" projection) to produce the
    layer's output, consolidating the filtered information.

Foundational Mathematics:
Let `X` be an input tensor with feature vectors `x_{ij}` at each spatial
position `(i, j)`. The transformation at each position is:

1.  Gate and Value Computation:
    `g_{ij} = activation_{attn}(W_g @ x_{ij} + b_g)`
    `v_{ij} = activation_{attn}(W_v @ x_{ij} + b_v)`
    where `W_g` and `W_v` are the kernel weights of the gate and up 1x1
    convolutions, respectively.

2.  Gating Operation:
    `h_{ij} = g_{ij} * v_{ij}`
    where `*` denotes the Hadamard (element-wise) product.

3.  Output Projection:
    `y_{ij} = activation_{out}(W_d @ h_{ij} + b_d)`
    where `W_d` is the kernel weight of the down 1x1 convolution.

The use of 1x1 convolutions allows this entire sequence of operations to be
applied efficiently across all spatial positions in parallel, with shared
weights `{W_g, W_v, W_d}`.

References:
The gMLP architecture was introduced in:
-   Liu, H., Dai, Z., So, D. R., & Le, Q. V. (2021). Pay Attention to MLPs.
    In Advances in Neural Information Processing Systems (NeurIPS).

The gating mechanism itself is an instance of the Gated Linear Unit,
originally proposed in:
-   Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language
    Modeling with Gated Convolutional Networks. In Proceedings of the 34th
    International Conference on Machine Learning (ICML).

"""

import keras
from typing import Optional, Union, Tuple, Literal, Any, Callable

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GatedMLP(keras.layers.Layer):
    """
    A Gated MLP layer implementation using 1x1 convolutions.

    This layer implements a gated MLP architecture where the input is processed through
    three separate 1x1 convolution paths: gate, up, and down projections. The gating
    mechanism allows the network to selectively focus on relevant features by computing
    element-wise multiplication between gate and up projections before applying the
    down projection.

    The architecture follows the gMLP design from "Pay Attention to MLPs" which provides
    an alternative to self-attention mechanisms while maintaining competitive performance.

    Mathematical formulation:
        gate = activation(Conv1x1_gate(x))
        up = activation(Conv1x1_up(x))
        gated = gate ⊙ up  # Element-wise multiplication
        output = activation(Conv1x1_down(gated))

    Args:
        filters: Integer, number of filters for all convolution layers. Must be positive.
        use_bias: Boolean, whether to use bias in the convolution layers. Defaults to True.
        kernel_initializer: String or Initializer, initializer for the kernel weights matrices.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer, initializer for the bias vectors.
            Defaults to 'zeros'.
        kernel_regularizer: Optional Regularizer, regularizer function applied to kernel weights.
            Defaults to None.
        bias_regularizer: Optional Regularizer, regularizer function applied to bias vectors.
            Defaults to None.
        attention_activation: String, activation function for the gate and up projections.
            Available options: 'relu', 'gelu', 'swish', 'silu', 'linear'. Defaults to 'relu'.
        output_activation: String, activation function for the output projection.
            Available options: 'relu', 'gelu', 'swish', 'silu', 'linear'. Defaults to 'linear'.
        data_format: String, data format for convolutions. Either 'channels_last' or
            'channels_first'. Defaults to None (uses Keras default).
        **kwargs: Additional arguments passed to the parent Layer class.

    Input shape:
        4D tensor with shape:
        - If data_format='channels_last': (batch_size, height, width, channels)
        - If data_format='channels_first': (batch_size, channels, height, width)

    Output shape:
        4D tensor with shape:
        - If data_format='channels_last': (batch_size, height, width, filters)
        - If data_format='channels_first': (batch_size, filters, height, width)

    Attributes:
        conv_gate: Conv2D layer for gate projection.
        conv_up: Conv2D layer for up projection.
        conv_down: Conv2D layer for down projection.

    Example:
        ```python
        # Basic usage
        x = keras.Input(shape=(32, 32, 64))
        gmlp = GatedMLP(filters=128)
        y = gmlp(x)
        print(y.shape)  # (None, 32, 32, 128)

        # Advanced configuration with regularization
        gmlp = GatedMLP(
            filters=256,
            attention_activation='gelu',
            output_activation='swish',
            kernel_regularizer=keras.regularizers.L2(1e-4),
            bias_regularizer=keras.regularizers.L1(1e-5)
        )

        # In a model
        inputs = keras.Input(shape=(224, 224, 3))
        x = keras.layers.Conv2D(64, 3, padding='same')(inputs)
        x = GatedMLP(128, attention_activation='gelu')(x)
        outputs = keras.layers.GlobalAveragePooling2D()(x)
        model = keras.Model(inputs, outputs)
        ```

    References:
        - Pay Attention to MLPs: https://arxiv.org/abs/2105.08050

    Raises:
        ValueError: If filters is not positive.
        ValueError: If data_format is not 'channels_first' or 'channels_last'.
        ValueError: If activation function is not supported.

    Note:
        This implementation uses 1x1 convolutions which are equivalent to dense layers
        applied spatially. The gating mechanism provides a learnable way to control
        information flow without explicit attention mechanisms.
    """

    def __init__(
        self,
        filters: int,
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        attention_activation: Literal["relu", "gelu", "swish", "silu", "linear"] = "relu",
        output_activation: Literal["relu", "gelu", "swish", "silu", "linear"] = "linear",
        data_format: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")

        # Store ALL configuration parameters
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
        if self.data_format not in {"channels_first", "channels_last"}:
            raise ValueError(
                f"data_format must be 'channels_first' or 'channels_last', got {self.data_format}"
            )

        # Validate activation functions
        valid_activations = {"relu", "gelu", "swish", "silu", "linear"}
        if attention_activation not in valid_activations:
            raise ValueError(
                f"attention_activation must be one of {valid_activations}, got {attention_activation}"
            )
        if output_activation not in valid_activations:
            raise ValueError(
                f"output_activation must be one of {valid_activations}, got {output_activation}"
            )

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        self.conv_gate = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            data_format=self.data_format,
            activation=None,  # Apply activation separately
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="conv_gate"
        )

        self.conv_up = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            data_format=self.data_format,
            activation=None,  # Apply activation separately
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="conv_up"
        )

        self.conv_down = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            data_format=self.data_format,
            activation=None,  # Apply activation separately
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="conv_down"
        )

        # Get activation functions
        self.attention_activation_fn = self._get_activation(attention_activation)
        self.output_activation_fn = self._get_activation(output_activation)

    def _get_activation(self, activation: str) -> Callable[[keras.KerasTensor], keras.KerasTensor]:
        """
        Get activation function by name.

        Args:
            activation: String name of activation function.

        Returns:
            Callable activation function.

        Raises:
            ValueError: If activation is not supported.
        """
        if activation == "relu":
            return keras.activations.relu
        elif activation == "gelu":
            return keras.activations.gelu
        elif activation == "swish" or activation == "silu":
            return keras.activations.swish
        elif activation == "linear":
            return keras.activations.linear
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the GatedMLP layer and all its sub-layers.

        This explicitly builds all sub-layers to ensure robust serialization
        following modern Keras 3 patterns.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Build sub-layers in computational order for robust serialization
        self.conv_gate.build(input_shape)
        self.conv_up.build(input_shape)

        # Calculate intermediate shape after gate/up convolutions
        input_shape_list = list(input_shape)
        if self.data_format == "channels_last":
            intermediate_shape = tuple(input_shape_list[:-1] + [self.filters])
        else:  # channels_first
            intermediate_shape = tuple([input_shape_list[0], self.filters] + input_shape_list[2:])

        # Build down convolution with intermediate shape
        self.conv_down.build(intermediate_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass for the GatedMLP layer.

        Implements the gated MLP computation:
        1. Compute gate and up projections with activation
        2. Element-wise multiply gate and up outputs (gating mechanism)
        3. Apply down projection with output activation

        Args:
            inputs: Input tensor of shape determined by data_format.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor after applying the Gated MLP operations.
        """
        # Gate branch: Conv1x1 + Activation
        x_gate = self.conv_gate(inputs, training=training)
        x_gate = self.attention_activation_fn(x_gate)

        # Up branch: Conv1x1 + Activation
        x_up = self.conv_up(inputs, training=training)
        x_up = self.attention_activation_fn(x_up)

        # Gating mechanism: element-wise multiplication
        x_gated = keras.ops.multiply(x_gate, x_up)

        # Down projection: Conv1x1 + Activation
        x_output = self.conv_down(x_gated, training=training)
        output = self.output_activation_fn(x_output)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple.
        """
        input_shape_list = list(input_shape)

        if self.data_format == "channels_last":
            return tuple(input_shape_list[:-1] + [self.filters])
        else:  # channels_first
            return tuple([input_shape_list[0], self.filters] + input_shape_list[2:])

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
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