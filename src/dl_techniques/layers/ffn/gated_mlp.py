"""
A gated MLP block built from 1x1 convolutions.

This layer gates one projection of the input with another projection of the
same input. Both projections are 1x1 convolutions. A 1x1 convolution is a
dense layer applied at every spatial position, with the weights shared across
positions.

The forward pass has four stages:

1.  **Parallel projections**. The input feeds two independent 1x1
    convolutions. One produces the gate, the other the value.
2.  **Activation**. The gate and the value both go through
    ``attention_activation``.
3.  **Gating**. The activated gate multiplies the activated value element by
    element. Where the gate is large the value passes through. Where it is
    near zero the value is suppressed.
4.  **Output projection**. A third 1x1 convolution maps the gated tensor to
    ``filters`` channels, then ``output_activation`` is applied.

**Mathematics:**
For the feature vector ``x_ij`` at spatial position ``(i, j)``:

    g_ij = activation_attn(W_g @ x_ij + b_g)
    v_ij = activation_attn(W_v @ x_ij + b_v)
    h_ij = g_ij * v_ij
    y_ij = activation_out(W_d @ h_ij + b_d)

``*`` is the element-wise product. ``W_g``, ``W_v`` and ``W_d`` are the
kernels of the gate, up and down convolutions. All three are shared across
every spatial position, so the whole block is three convolutions and one
multiply.

**This is not the gMLP block from the paper.** Every convolution here is 1x1,
so nothing mixes information across spatial positions: output position
``(i, j)`` depends only on input position ``(i, j)``. The part of gMLP that
replaces attention is its Spatial Gating Unit, which projects along the token
axis. That projection is absent here. What this layer implements is a Gated
Linear Unit over channels, applied position-wise. If you need token mixing,
use a layer that has it. This one cannot learn any. MEASURED on a
``(1, 5, 5, 4)`` input with ``filters=8``: perturbing one pixel changes the
output at that pixel and at no other, with an off-pixel delta of exactly 0.0.

**Known defect: the gate and the value are the same function.** All three
convolutions receive the SAME initializer instance, and ``conv_gate`` and
``conv_up`` have the same shape, so they start with bit-identical kernels.
The product ``g * v`` is symmetric in the two, so their gradients are equal
too and they never diverge. MEASURED with the defaults: ``max|delta|`` is
0.0 at build time and still 0.0 after 5 epochs of SGD, for the kernels and
the biases. The layer therefore computes ``attn(conv(x))**2``, not a gate
times a value. This is documented, not fixed - changing it changes every
existing checkpoint.

References:
-   Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language
    Modeling with Gated Convolutional Networks. In Proceedings of the 34th
    International Conference on Machine Learning (ICML). This is the gating
    mechanism the layer actually implements.
-   Liu, H., Dai, Z., So, D. R., & Le, Q. V. (2021). Pay Attention to MLPs.
    In Advances in Neural Information Processing Systems (NeurIPS). This is
    the architecture the layer is named after; see the paragraph above for
    what is missing.

"""

import keras
from typing import Optional, Union, Tuple, Literal, Any, Callable
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GatedMLP(keras.layers.Layer):
    """
    Gated MLP layer built from three 1x1 convolutions.

    Two 1x1 convolutions read the input. ``conv_gate`` produces the gate and
    ``conv_up`` produces the value. Both go through ``attention_activation``,
    the gate multiplies the value element by element, and ``conv_down`` maps
    the product to ``filters`` channels:
    ``y = out_act(conv_down(attn_act(conv_gate(x)) * attn_act(conv_up(x))))``.

    All three convolutions use ``filters`` output channels, so the gate and
    the value have the same width and the output width equals ``filters``.
    The input width is independent of ``filters``.

    Every kernel is 1x1 with stride 1, so the spatial size never changes and
    no information moves between positions. See the module docstring for how
    this differs from the published gMLP block.

    **Architecture Overview:**

    .. code-block:: text

            Input  [B, H, W, C]
                     │
               ┌─────┴─────┐
               ▼           ▼
        ┌────────────┐ ┌────────────┐
        │ conv_gate  │ │  conv_up   │
        │  1x1, F    │ │   1x1, F   │
        └─────┬──────┘ └─────┬──────┘
              ▼              ▼
        ┌────────────┐ ┌────────────┐
        │ attn activ │ │ attn activ │
        └─────┬──────┘ └─────┬──────┘
              └──────┬───────┘
                     ▼
              multiply  [B, H, W, F]
                     │
                     ▼
            ┌────────────────┐
            │   conv_down    │
            │    1x1, F      │
            └───────┬────────┘
                    ▼
            ┌────────────────┐
            │  output activ  │
            └───────┬────────┘
                    ▼
            Output [B, H, W, F]

        F = filters. Shapes are for data_format='channels_last';
        the fork below gives the 'channels_first' layout.

    **Gate and value split (block internals):**

    .. code-block:: text

        x  [B, H, W, C]
        │
        ├──► conv_gate (1x1) ──► attn_act ──► g [B, H, W, F]
        │                                          │
        └──► conv_up   (1x1) ──► attn_act ──► v ───┤
                                                   ▼
                                        h = g * v  [B, H, W, F]

        The SAME activation runs on both branches, so neither
        branch is the linear one. This differs from SwiGLU-style
        gating, where only the gate is non-linear.

    **The data_format fork:**

    .. code-block:: text

                  data_format
                       │
              ┌────────┴────────┐
              ▼                 ▼
        'channels_last'   'channels_first'
              │                 │
              ▼                 ▼
        channel axis -1   channel axis 1
        in  [B, H, W, C]  in  [B, C, H, W]
        mid [B, H, W, F]  mid [B, F, H, W]
        out [B, H, W, F]  out [B, F, H, W]

        Both leaves run the same three convolutions. Only the
        axis holding the channels moves. call() does not branch:
        the layout is handed to each Conv2D, while build() and
        compute_output_shape() put `filters` on axis -1 or on
        axis 1. `mid` is the width conv_down is built for.
        data_format=None resolves at __init__ time through
        keras.backend.image_data_format().

    :param filters: Number of output channels for all three convolutions.
        Must be positive.
    :type filters: int
    :param use_bias: Whether the convolutions carry a bias. Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the convolution kernels. The
        same instance is passed to all three convolutions. Defaults to
        'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the biases. Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the kernels. Defaults to None.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for the biases. Defaults to None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param attention_activation: Activation for the gate and the value. One of
        'relu', 'gelu', 'swish', 'silu', 'linear'. 'swish' and 'silu' select
        the same function. Defaults to 'relu'.
    :type attention_activation: str
    :param output_activation: Activation applied after ``conv_down``. Same
        five choices. Defaults to 'linear', which is the identity.
    :type output_activation: str
    :param data_format: 'channels_last' or 'channels_first'. Defaults to None,
        which reads ``keras.backend.image_data_format()``.
    :type data_format: Optional[str]
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on).
    :type kwargs: Any

    :ivar filters: The stored channel count.
    :vartype filters: int
    :ivar use_bias: Whether the convolutions carry a bias.
    :vartype use_bias: bool
    :ivar kernel_initializer: The resolved kernel initializer.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar bias_initializer: The resolved bias initializer.
    :vartype bias_initializer: keras.initializers.Initializer
    :ivar kernel_regularizer: The resolved kernel regularizer, or ``None``.
    :vartype kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar bias_regularizer: The resolved bias regularizer, or ``None``.
    :vartype bias_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar attention_activation: The activation NAME, still a string. Only the
        five names above pass validation, and ``deserialize_activation``
        returns a string unchanged, so this attribute is never a callable.
        ``get_config()`` stores it.
    :vartype attention_activation: str
    :ivar output_activation: The output activation name, same rule.
    :vartype output_activation: str
    :ivar data_format: The RESOLVED layout. Never ``None``, even when ``None``
        was passed.
    :vartype data_format: str
    :ivar conv_gate: The 1x1 convolution producing the gate.
    :vartype conv_gate: keras.layers.Conv2D
    :ivar conv_up: The 1x1 convolution producing the value.
    :vartype conv_up: keras.layers.Conv2D
    :ivar conv_down: The 1x1 convolution producing the output.
    :vartype conv_down: keras.layers.Conv2D
    :ivar attention_activation_fn: The callable looked up from
        ``attention_activation``. This is what ``call()`` runs.
    :vartype attention_activation_fn: Callable
    :ivar output_activation_fn: The callable looked up from
        ``output_activation``.
    :vartype output_activation_fn: Callable

    :raises ValueError: If ``filters`` is not positive.
    :raises ValueError: If ``data_format`` is neither 'channels_first' nor
        'channels_last'.
    :raises ValueError: If ``attention_activation`` or ``output_activation``
        is not one of the five supported names.

    Input shape:
        4D tensor. ``(batch, height, width, channels)`` for 'channels_last',
        ``(batch, channels, height, width)`` for 'channels_first'.

    Output shape:
        Same rank and same spatial size as the input, with the channel axis
        set to ``filters``.

    Example:
        .. code-block:: python

            block = GatedMLP(filters=64, attention_activation='gelu')
            y = block(keras.random.normal((2, 32, 32, 16)))
            y.shape                 # (2, 32, 32, 64)

    Note:
        The three convolutions are created in ``__init__`` and built in
        ``build()``. ``conv_down`` has to be built by hand because it sees
        ``filters`` channels, not the input width.

    Warning:
        ``conv_gate`` and ``conv_up`` share one initializer instance and have
        the same shape, so they start bit-identical and stay bit-identical
        under training. See the module docstring for the measurement. Until
        that is fixed, treat this layer as a squared activation rather than
        as a gate.
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
        """
        Validate the configuration and create the three convolutions.

        Every argument is documented on the class. ``data_format`` is
        resolved before it is checked, so passing ``None`` picks up the Keras
        default and then that default is validated too.

        :raises ValueError: If ``filters`` is not positive, if the resolved
            ``data_format`` is neither 'channels_first' nor 'channels_last',
            or if either activation name is not supported.
        """
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
        self.attention_activation = deserialize_activation(attention_activation)
        self.output_activation = deserialize_activation(output_activation)
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
            # Activation is applied in call(), not inside the convolution.
            activation=None,
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
            # Activation is applied in call(), not inside the convolution.
            activation=None,
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
            # Activation is applied in call(), not inside the convolution.
            activation=None,
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

        :param activation: String name of activation function.
        :type activation: str
        :return: Callable activation function.
        :rtype: Callable
        :raises ValueError: If activation is not supported.
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

        Explicitly builds all sub-layers to ensure robust serialization
        following modern Keras 3 patterns.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # Build sub-layers in computational order for robust serialization
        self.conv_gate.build(input_shape)
        self.conv_up.build(input_shape)

        # Calculate intermediate shape after gate/up convolutions
        input_shape_list = list(input_shape)
        if self.data_format == "channels_last":
            intermediate_shape = tuple(input_shape_list[:-1] + [self.filters])
        else:
            # channels_first: the channel axis is 1, not -1.
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

        :param inputs: Input tensor of shape determined by data_format.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer should behave in training mode
            or inference mode.
        :type training: Optional[bool]
        :return: Output tensor after applying the Gated MLP operations.
        :rtype: keras.KerasTensor
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

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        input_shape_list = list(input_shape)

        if self.data_format == "channels_last":
            return tuple(input_shape_list[:-1] + [self.filters])
        else:
            # channels_first: the channel axis is 1, not -1.
            return tuple([input_shape_list[0], self.filters] + input_shape_list[2:])

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "attention_activation": serialize_activation(self.attention_activation),
            "output_activation": serialize_activation(self.output_activation),
            "data_format": self.data_format,
        })
        return config
