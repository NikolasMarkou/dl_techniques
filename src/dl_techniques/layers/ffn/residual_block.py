"""
A residual block: a two-layer MLP plus a learnable projection shortcut.

The block adds two paths. The main path is a two-layer MLP. The shortcut is a
single Dense layer applied to the same input. The two results are added
element-wise.

ResNet's idea is to have the main path learn `F(x) = H(x) - x` rather than
`H(x)` itself. Two things follow. The gradient gets a short route back through
the shortcut, so it does not have to survive every layer of the main path. And
a block that should do nothing is easy to reach: drive the main path's weights
to zero.

The shortcut here is a Dense layer, not a bare identity. That is a projection
shortcut. It costs `input_dim * output_dim` extra parameters, and in exchange
the block works when the input width and the output width differ. When they are
equal the shortcut still has to learn the identity; it does not get it for free.

Main path, with an optional dropout between the two Dense layers:

    F(x) = W_2 @ activation(W_1 @ x + b_1) + b_2

Shortcut path:

    S(x) = W_s @ x + b_s

Output:

    y = F(x) + S(x)

The bias terms exist only when `use_bias` is True. The dropout layer is created
only when `dropout_rate > 0`.

References:
    - He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for
      Image Recognition. CVPR.
"""

import keras
from typing import Optional, Union, Any, Tuple, Callable

from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.ffn.residual_block")
class ResidualBlock(keras.layers.Layer):
    """
    Residual block: a two-layer MLP added to a learnable projection shortcut.

    The main path computes ``F(x) = W_2 @ activation(W_1 @ x + b_1) + b_2``,
    with an optional dropout between the two Dense layers. The shortcut
    computes ``S(x) = W_s @ x + b_s``. The output is ``y = F(x) + S(x)``.
    Because the shortcut is a Dense layer rather than an identity, the input
    width and the output width may differ.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────┐
        │    Input (..., input_dim)    │
        └──────────────┬───────────────┘
                       │
                 ┌─────┴──────────┐
                 ▼                ▼
        ┌──────────────┐  ┌──────────────────┐
        │ hidden_layer │  │ residual_layer   │
        │Dense(hidden) │  │Dense(output_dim) │
        │+ activation  │  │ (linear shortcut)│
        └──────┬───────┘  └────────┬─────────┘
               ▼                   │
        ┌──────────────┐           │
        │   Dropout    │           │
        │  (optional)  │           │
        └──────┬───────┘           │
               ▼                   │
        ┌──────────────┐           │
        │ output_layer │           │
        │Dense(output) │           │
        │  (linear)    │           │
        └──────┬───────┘           │
               │                   │
               └────────┬──────────┘
                        ▼
        ┌──────────────────────────────┐
        │     Element-wise Addition    │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │   Output (..., output_dim)   │
        └──────────────────────────────┘

    **Width arithmetic (block internals):**

    .. code-block:: text

        D_in = input width, H = hidden_dim, D_out = output_dim

        main path   x [.., D_in]
             ─► W_1 (D_in x H) + b_1        ─► [.., H]
             ─► activation                  ─► [.., H]
             ─► dropout  (only if rate > 0) ─► [.., H]
             ─► W_2 (H x D_out) + b_2       ─► [.., D_out]

        shortcut    x [.., D_in]
             ─► W_s (D_in x D_out) + b_s    ─► [.., D_out]

        add         [.., D_out] + [.., D_out] ─► [.., D_out]

        D_in and D_out are free to differ. W_s is what changes
        the width, so the two addends always match. When
        D_in == D_out the shortcut is still a Dense layer and
        has to learn the identity. Biases exist only when
        use_bias is True. The add is a plain `+` in call(),
        not a keras.layers.Add.

    :param hidden_dim: Integer, dimensionality of the hidden layer. Must be positive.
    :type hidden_dim: int
    :param output_dim: Integer, dimensionality of the output space. Must be positive.
    :type output_dim: int
    :param dropout_rate: Dropout rate for regularization between 0 and 1.
        Defaults to 0.0 (no dropout).
    :type dropout_rate: float
    :param activation: Activation function for hidden layer.
        Accepts string names ('relu', 'gelu') or callable functions.
        Defaults to 'relu'.
    :type activation: Union[str, Callable]
    :param use_bias: Whether to use bias in all dense layers. Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for kernel weights.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias weights.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights. Accepts string
        names ('l1', 'l2') or Regularizer instances. Defaults to None.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Optional regularizer for bias weights. Accepts string
        names ('l1', 'l2') or Regularizer instances. Defaults to None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments for the Layer base class
        (``name``, ``dtype``, ``trainable``, and so on).
    :type kwargs: Any

    :ivar hidden_dim: The stored hidden width.
    :vartype hidden_dim: int
    :ivar output_dim: The stored output width.
    :vartype output_dim: int
    :ivar dropout_rate: The stored dropout rate.
    :vartype dropout_rate: float
    :ivar activation: The RESOLVED activation callable, not the name that was
        passed. ``get_config()`` serializes it back to a name.
    :vartype activation: Callable
    :ivar use_bias: Whether every Dense layer carries a bias.
    :vartype use_bias: bool
    :ivar kernel_initializer: The resolved kernel initializer. Each Dense layer
        gets its own clone of it, never this instance.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar bias_initializer: The resolved bias initializer, cloned the same way.
    :vartype bias_initializer: keras.initializers.Initializer
    :ivar kernel_regularizer: The resolved kernel regularizer, or ``None``.
    :vartype kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar bias_regularizer: The resolved bias regularizer, or ``None``.
    :vartype bias_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar hidden_layer: The first Dense layer of the main path, with the
        activation applied inside it.
    :vartype hidden_layer: keras.layers.Dense
    :ivar output_layer: The second Dense layer of the main path. Linear.
    :vartype output_layer: keras.layers.Dense
    :ivar residual_layer: The Dense layer on the shortcut. Linear.
    :vartype residual_layer: keras.layers.Dense
    :ivar dropout: The dropout layer, or ``None`` when ``dropout_rate`` is 0.
    :vartype dropout: Optional[keras.layers.Dropout]

    :raises ValueError: If ``hidden_dim`` is not positive.
    :raises ValueError: If ``output_dim`` is not positive.
    :raises ValueError: If ``dropout_rate`` is outside ``[0.0, 1.0]``.

    Input shape:
        Tensor of shape ``(batch_size, ..., input_dim)``. Any rank of 2 or
        more works; the Dense layers act on the last axis only.

    Output shape:
        Same shape as the input with the last axis set to ``output_dim``.

    Example:
        .. code-block:: python

            block = ResidualBlock(hidden_dim=64, output_dim=32)
            y = block(keras.random.normal((4, 16)))
            y.shape                 # (4, 32)

    Note:
        The shortcut is always a Dense layer, even when the input and output
        widths match. That costs parameters an identity shortcut would not,
        and it is what lets the block change width.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.0,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'relu',
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        """
        Validate the configuration and create the three Dense layers.

        Every argument is documented on the class. The dropout layer is
        created here only when ``dropout_rate > 0``; otherwise ``self.dropout``
        stays ``None`` and no dropout runs at all.

        :raises ValueError: If ``hidden_dim`` or ``output_dim`` is not
            positive, or if ``dropout_rate`` is outside ``[0.0, 1.0]``.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration parameters
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Create all sub-layers here; build() builds them.
        # Main path, first Dense: the activation runs inside it.

        # DECISION plan-2026-08-22T035419-a11304c8/D-200 -- clone_initializer per layer.
        # Do NOT restore the shared instance: the skip PROJECTION and the main-path
        # output layer had bit-identical kernels (MEASURED max|delta| = 0.0), which
        # makes the residual branch a copy of the transform it is supposed to bypass.
        self.hidden_layer = keras.layers.Dense(
            units=self.hidden_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="hidden_layer"
        )

        # Main path, second Dense. Linear.
        self.output_layer = keras.layers.Dense(
            units=self.output_dim,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output_layer"
        )

        # Shortcut path. Linear, and it does the width change.
        self.residual_layer = keras.layers.Dense(
            units=self.output_dim,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="residual_layer"
        )

        # Optional dropout layer
        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(
                rate=self.dropout_rate,
                name="dropout"
            )
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        Explicitly builds each sub-layer for robust serialization, ensuring all
        weight variables exist before weight restoration during model loading.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # Build sub-layers in computational order for robust serialization
        self.hidden_layer.build(input_shape)

        # Compute intermediate shape after hidden layer
        hidden_output_shape = list(input_shape)
        hidden_output_shape[-1] = self.hidden_dim
        hidden_output_shape = tuple(hidden_output_shape)

        # Dropout doesn't change shape, so use hidden_output_shape
        if self.dropout is not None:
            self.dropout.build(hidden_output_shape)

        self.output_layer.build(hidden_output_shape)

        # Residual layer takes original input shape
        self.residual_layer.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass with residual connection.

        :param inputs: Input tensor of shape (batch_size, ..., input_dim).
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode for dropout.
        :type training: Optional[bool]
        :return: Output tensor of shape (batch_size, ..., output_dim).
        :rtype: keras.KerasTensor
        """
        # Main path: input -> hidden -> dropout -> output
        hidden = self.hidden_layer(inputs, training=training)

        if self.dropout is not None:
            hidden = self.dropout(hidden, training=training)

        main_output = self.output_layer(hidden, training=training)

        # Residual path: input -> projection
        residual_output = self.residual_layer(inputs, training=training)

        # Combine main path and residual path
        return main_output + residual_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple with last dimension set to output_dim.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Dictionary containing all configuration parameters needed to
            reconstruct this layer.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout_rate": self.dropout_rate,
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
