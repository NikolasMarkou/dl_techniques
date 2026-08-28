"""
The MLP block from the Swin Transformer.

This is the position-wise feed-forward network that runs after windowed
attention in every Swin block. It transforms each token on its own, with the
same weights at every position.

The name is historical. This is the ordinary Transformer FFN, the same shape
almost every Transformer uses. What sets this implementation apart from
``MLPBlock`` in the same package is two dropouts instead of one, and an
``output_dim`` that may be left as ``None`` to match the input width.

The forward path:

1.  ``fc1`` projects each token from its input width up to ``hidden_dim``.
    The usual expansion is 4x. This is an inverted bottleneck: wide in the
    middle, narrow at both ends.
2.  An activation (GELU by default) runs element-wise. This is the only
    non-linearity. Without it the two Dense layers would collapse into a
    single linear map.
3.  ``drop1``.
4.  ``fc2`` projects back down to the output width.
5.  ``drop2``.

The maths, for one token vector ``x``:

    FFN(x) = activation(x @ W_1 + b_1) @ W_2 + b_2

``W_1`` is ``(input_dim, hidden_dim)`` and ``W_2`` is
``(hidden_dim, output_dim)``. GELU is ``x * Phi(x)``, where ``Phi`` is the
standard Gaussian CDF.

This layer holds no residual add and no normalization. The caller owns both.

References:
-   Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer
    using Shifted Windows. ICCV. (the block this MLP sits in)
-   Vaswani, A., et al. (2017). Attention Is All You Need. NIPS.
    (the same FFN, introduced first)

"""

import keras
from typing import Tuple, Optional, Dict, Any, Union, Callable

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SwinMLP(keras.layers.Layer):
    """
    The Swin Transformer MLP block.

    Two Dense layers with one activation between them:
    ``FFN(x) = activation(x @ W_1 + b_1) @ W_2 + b_2``. Each token is
    transformed on its own, with the same weights at every position.

    Two things separate this from ``MLPBlock``. There are TWO dropouts, one
    after the activation and one after ``fc2``. And ``output_dim`` may be
    ``None``, which means "match the input width", resolved at build time.

    Despite the name, this is the plain Transformer FFN. Swin uses it, but so
    does nearly every other Transformer.

    **Architecture Overview:**

    .. code-block:: text

            Input  [..., input_dim]
                        │
                        ▼
            ┌─────────────────────────┐
            │ fc1                     │
            │ Dense(hidden_dim)       │
            └────────────┬────────────┘
                         ▼
            ┌─────────────────────────┐
            │ act                     │
            │ (default gelu)          │
            └────────────┬────────────┘
                         ▼
            ┌─────────────────────────┐
            │ drop1                   │
            └────────────┬────────────┘
                         ▼
            ┌─────────────────────────┐
            │ fc2                     │
            │ Dense(output width)     │
            └────────────┬────────────┘
                         ▼
            ┌─────────────────────────┐
            │ drop2                   │
            └────────────┬────────────┘
                         ▼
            Output [..., output width]

        Both dropouts always exist and are always called. At
        dropout_rate=0.0 they are the identity. There is no
        residual add here; the caller owns it.

        The output width is output_dim, or the input width when
        output_dim is None. fc2 is created in build(), not in
        __init__, because that width is not known earlier.

    :param hidden_dim: Width of the expansion, the ``units`` of ``fc1``. Must
        be positive.
    :type hidden_dim: int
    :param use_bias: Whether ``fc1`` and ``fc2`` carry a bias. Defaults to
        True.
    :type use_bias: bool
    :param output_dim: Width of the output. ``None`` (the default) means take
        the input width, so the layer is shape-preserving. Must be positive
        when given.
    :type output_dim: Optional[int]
    :param activation: Activation applied after ``fc1``. A Keras name
        ('gelu', 'relu', 'swish') or a callable. Defaults to 'gelu'.
    :type activation: Union[str, Callable]
    :param dropout_rate: Rate for BOTH dropouts, in ``[0.0, 1.0]``. Defaults
        to 0.0. The two layers always exist; at 0.0 they pass their input
        through.
    :type dropout_rate: float
    :param kernel_initializer: Initializer for the kernels. Defaults to
        'glorot_uniform'. Both Dense layers get the same instance.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the biases. Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the kernels. A Keras name
        ('l2') or a Regularizer instance. Defaults to None.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for the biases. Defaults to None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param activity_regularizer: Regularizer on layer outputs. Defaults to
        None. Read the Note below before setting it.
    :type activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on).
    :type kwargs: Any

    :ivar hidden_dim: The stored expansion width.
    :vartype hidden_dim: int
    :ivar use_bias: Whether the projections carry a bias.
    :vartype use_bias: bool
    :ivar output_dim: The output width as REQUESTED, possibly ``None``. This
        is what ``get_config()`` stores. The resolved width is not kept as an
        attribute; it lives in ``fc2.units``.
    :vartype output_dim: Optional[int]
    :ivar activation: The resolved activation function, wrapped by ``act``.
    :vartype activation: Callable
    :ivar dropout_rate: The stored dropout rate, shared by both dropouts.
    :vartype dropout_rate: float
    :ivar kernel_initializer: The resolved kernel initializer.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar bias_initializer: The resolved bias initializer.
    :vartype bias_initializer: keras.initializers.Initializer
    :ivar kernel_regularizer: The resolved kernel regularizer, or ``None``.
    :vartype kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar bias_regularizer: The resolved bias regularizer, or ``None``.
    :vartype bias_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar activity_regularizer: The resolved activity regularizer, or
        ``None``. This shadows the attribute of the same name on
        ``keras.layers.Layer``.
    :vartype activity_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar fc1: ``Dense(hidden_dim)``, the expansion.
    :vartype fc1: keras.layers.Dense
    :ivar act: ``Activation(activation)``.
    :vartype act: keras.layers.Activation
    :ivar drop1: ``Dropout(dropout_rate)``, after the activation.
    :vartype drop1: keras.layers.Dropout
    :ivar drop2: ``Dropout(dropout_rate)``, after ``fc2``.
    :vartype drop2: keras.layers.Dropout
    :ivar fc2: ``Dense`` at the resolved output width. ``None`` until
        ``build()`` runs.
    :vartype fc2: Optional[keras.layers.Dense]

    :raises ValueError: If ``hidden_dim`` is not positive.
    :raises ValueError: If ``dropout_rate`` is outside ``[0.0, 1.0]``.
    :raises ValueError: If ``output_dim`` is given and not positive.
    :raises ValueError: At build time, if the input is rank 1 or its last
        axis is ``None``.

    Input shape:
        Tensor of rank >= 2, shape ``(..., input_dim)``. The last axis must
        be statically known.

    Output shape:
        Same rank and leading axes as the input. The last axis is
        ``output_dim``, or ``input_dim`` when ``output_dim`` is ``None``.

    Example:
        .. code-block:: python

            mlp = SwinMLP(hidden_dim=384)
            y = mlp(keras.random.normal((2, 49, 96)))
            y.shape                 # (2, 49, 96)

    Note:
        ``activity_regularizer`` is charged THREE times, not once. It is
        passed to ``fc1`` and to ``fc2``, and it is also assigned to
        ``self.activity_regularizer``, which is the attribute
        ``keras.layers.Layer.__call__`` reads when it adds an activity loss
        on this layer's own output. Measured on Keras 3 with
        ``activity_regularizer='l2'``: ``len(layer.losses) == 3``. If you
        want one penalty, put the regularizer on the surrounding layer
        instead.

    Note:
        Sub-layers are created in ``__init__`` and built explicitly in
        ``build()``, except ``fc2``, which is created in ``build()`` once the
        output width is known.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_bias: bool = True,
        output_dim: Optional[int] = None,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = "gelu",
        dropout_rate: float = 0.0,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        """Validate the configuration and create the sub-layers.

        Every argument is documented on the class. Validation runs before any
        attribute is stored, so a rejected configuration leaves no half-built
        layer behind. ``fc2`` is the exception to "create everything here":
        it needs the output width, which ``build()`` resolves.

        :param hidden_dim: Expansion width. Must be positive.
        :type hidden_dim: int
        :param use_bias: Whether ``fc1`` and ``fc2`` carry a bias.
        :type use_bias: bool
        :param output_dim: Output width, or ``None`` to take the input width.
            Must be positive when given.
        :type output_dim: Optional[int]
        :param activation: Activation name or callable, applied after ``fc1``.
        :type activation: Union[str, Callable]
        :param dropout_rate: Rate for both dropouts. Must be in
            ``[0.0, 1.0]``.
        :type dropout_rate: float
        :param kernel_initializer: Initializer for both kernels.
        :type kernel_initializer: Union[str, keras.initializers.Initializer]
        :param bias_initializer: Initializer for both biases.
        :type bias_initializer: Union[str, keras.initializers.Initializer]
        :param kernel_regularizer: Regularizer for both kernels, or ``None``.
        :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param bias_regularizer: Regularizer for both biases, or ``None``.
        :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param activity_regularizer: Activity regularizer, or ``None``. Read
            the class Note before using it: the penalty lands three times.
        :type activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param kwargs: Extra arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If ``hidden_dim`` is not positive, if
            ``dropout_rate`` is outside ``[0.0, 1.0]``, or if ``output_dim``
            is given and not positive.
        """
        super().__init__(**kwargs)

        # Validate parameters
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be between 0.0 and 1.0, got {dropout_rate}")
        if output_dim is not None and output_dim <= 0:
            raise ValueError(f"output_dim must be positive when specified, got {output_dim}")

        # Store configuration parameters
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.output_dim = output_dim
        self.activation = keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        # These are unbuilt at creation time
        self.fc1 = keras.layers.Dense(
            units=self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="fc1"
        )

        # Activation layer
        self.act = keras.layers.Activation(self.activation, name="act")

        # Dropout layers (created even if rate=0.0 for consistent serialization)
        self.drop1 = keras.layers.Dropout(self.dropout_rate, name="drop1")
        self.drop2 = keras.layers.Dropout(self.dropout_rate, name="drop2")

        # fc2's width is the input width whenever output_dim is None, so it
        # cannot be created here. build() creates it.
        self.fc2 = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create ``fc2`` at the resolved width, then build every sub-layer.

        ``fc2`` cannot be created in ``__init__``, because its width is the
        input width whenever ``output_dim`` is ``None``. It is created here
        instead. The ``self.built`` guard on the first line is what keeps
        that safe: without it, a second ``build()`` from functional reuse or
        from deserialization would replace ``fc2`` and drop weights that
        already exist.

        Sub-layers are then built in the order ``call()`` uses them, so every
        weight exists before a save or a restore.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is rank 1, or if its last axis is
            ``None``.
        """
        # Guard against re-build (functional reuse / deserialization). Without
        # this, fc2 is re-created below and the previously-built weights are
        # dropped. Must be the FIRST line of build().
        if self.built:
            return

        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input_shape}")

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        # Determine output dimension
        output_dim = self.output_dim if self.output_dim is not None else input_dim

        # Create second dense layer now that we know the output dimension
        self.fc2 = keras.layers.Dense(
            units=output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="fc2"
        )

        # CRITICAL: Explicitly build all sub-layers in computational order
        # This ensures weight variables exist before serialization/deserialization
        self.fc1.build(input_shape)

        # Compute intermediate shape for subsequent layers
        fc1_output_shape = self.fc1.compute_output_shape(input_shape)

        # Activation layer doesn't change shape, but we build it for consistency
        self.act.build(fc1_output_shape)

        # Dropout layers don't change shape
        self.drop1.build(fc1_output_shape)
        self.fc2.build(fc1_output_shape)
        self.drop2.build(self.fc2.compute_output_shape(fc1_output_shape))

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Run the block: ``fc1`` -> activation -> ``drop1`` -> ``fc2`` ->
        ``drop2``.

        Both dropouts always exist and are always called. At
        ``dropout_rate=0.0`` they are the identity.

        :param inputs: Input tensor of shape ``(..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether to run in training mode. Only affects the
            two dropouts.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(..., output_dim)``.
        :rtype: keras.KerasTensor
        """
        # First linear transformation (expansion)
        x = self.fc1(inputs)

        # Non-linear activation
        x = self.act(x)

        # First dropout (applied after activation)
        x = self.drop1(x, training=training)

        # Second linear transformation (contraction)
        x = self.fc2(x)

        # Second dropout (applied before output)
        x = self.drop2(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape.

        When ``output_dim`` is set, the last axis becomes it. When it is
        ``None``, the shape is returned unchanged, because the layer maps
        the input width back to itself.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The input shape with the last axis set to ``output_dim``, or
            unchanged when ``output_dim`` is ``None``.
        :rtype: Tuple[Optional[int], ...]
        """
        # Convert to list for manipulation
        output_shape = list(input_shape)

        # Set output dimension based on configuration
        if self.output_dim is not None:
            output_shape[-1] = self.output_dim
        # If output_dim is None, output shape matches input shape (identity transformation)

        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the config needed to rebuild this layer.

        Holds every ``__init__`` argument. ``output_dim`` is stored exactly
        as it was passed in, so a ``None`` stays ``None`` and a reloaded
        layer re-resolves the width from its own input.

        :return: The complete layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "use_bias": self.use_bias,
            "output_dim": self.output_dim,
            "activation": keras.activations.serialize(self.activation),
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

# ---------------------------------------------------------------------
