"""
The position-wise feed-forward network from the Transformer.

This is the second sub-block of a Transformer layer, the one that runs after
attention. It transforms each token on its own. No token reads another, so
the whole sequence goes through in parallel as two matmuls.

The shape is expand-then-contract:

1.  ``fc1`` projects each token from its input width up to ``hidden_dim``.
    The usual expansion is 4x. The wider space gives the activation more
    room to separate features.
2.  An activation (GELU by default) runs element-wise on the wide tensor.
    This is the only non-linearity. Without it the two Dense layers would
    collapse into a single linear map.
3.  ``fc2`` projects back down to ``output_dim``. Set ``output_dim`` to the
    model width so the result can be added to the residual stream.

The maths, for one token vector ``x``:

    FFN(x) = activation(x @ W_1 + b_1) @ W_2 + b_2

``W_1`` is ``(input_dim, hidden_dim)`` and ``W_2`` is
``(hidden_dim, output_dim)``. GELU is ``x * Phi(x)``, where ``Phi`` is the
standard Gaussian CDF. The same weights apply at every position.

This layer holds no residual add and no normalization. The caller owns both.

References:
-   Vaswani, A., et al. (2017). Attention Is All You Need. NIPS.
    (introduced this FFN as a Transformer sub-block)

"""

import keras
from typing import Optional, Union, Any, Dict, Tuple, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.initializers import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.ffn.mlp", legacy_alias=False)
class MLPBlock(keras.layers.Layer):
    """
    The standard Transformer MLP block.

    Two Dense layers with one activation between them:
    ``FFN(x) = activation(x @ W_1 + b_1) @ W_2 + b_2``. Each token is
    transformed on its own, with the same weights at every position.

    The input width does NOT have to equal ``output_dim``. ``fc1`` is built
    from whatever width arrives, and ``output_dim`` sets only the output.

    Dropout runs in ONE place, after the activation. There is no dropout on
    the output of ``fc2``.

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
            │ activation              │
            │ (default gelu)          │
            └────────────┬────────────┘
                         ▼
            ┌─────────────────────────┐
            │ dropout                 │
            │ (only if rate > 0.0)    │
            └────────────┬────────────┘
                         ▼
            ┌─────────────────────────┐
            │ fc2                     │
            │ Dense(output_dim)       │
            └────────────┬────────────┘
                         ▼
            Output [..., output_dim]

        `dropout` really is conditional: at dropout_rate=0.0
        the attribute is None and no Dropout layer exists in
        the graph. There is no second dropout after fc2.

        There is no residual add and no normalization inside
        this layer. The caller owns both.

    :param hidden_dim: Width of the expansion, the ``units`` of ``fc1``. Must
        be positive. Transformers usually set it to 4x the model width.
    :type hidden_dim: int
    :param output_dim: Width of the output, the ``units`` of ``fc2``. Must be
        positive. It does NOT constrain the input width.
    :type output_dim: int
    :param activation: Activation applied after ``fc1``. A Keras name
        ('gelu', 'relu', 'swish') or a callable. Defaults to 'gelu'.
    :type activation: Union[str, Callable]
    :param dropout_rate: Dropout rate applied after the activation, in
        ``[0.0, 1.0)``. At 0.0 no Dropout layer is created. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether ``fc1`` and ``fc2`` carry a bias. Defaults to
        True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the kernels. A Keras name
        ('glorot_uniform', 'he_normal') or an Initializer instance. Each
        Dense layer gets its own clone of it. Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param output_kernel_initializer: Initializer for the OUTPUT projection
        (``fc2``) only. ``None`` (the default) means ``fc2`` gets a clone of
        ``kernel_initializer``, which is the historical behaviour. Pass one
        to give the residual-path projection a different scale from the
        expansion -- see ``TransformerLayer``'s
        ``residual_output_kernel_initializer``.
    :type output_kernel_initializer: Optional[Union[str, keras.initializers.Initializer]]
    :param bias_initializer: Initializer for the biases. Each Dense layer
        gets its own clone of it, never the same instance. Defaults to
        'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the kernels. A Keras name
        ('l2') or a Regularizer instance. Defaults to None.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for the biases. A Keras name ('l1')
        or a Regularizer instance. Defaults to None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on).
    :type kwargs: Any

    :ivar hidden_dim: The stored expansion width.
    :vartype hidden_dim: int
    :ivar output_dim: The stored output width.
    :vartype output_dim: int
    :ivar activation_name: The activation exactly as passed in, before
        ``keras.activations.get``. Stored for reference; ``get_config()``
        serializes ``activation_fn`` instead.
    :vartype activation_name: Union[str, Callable]
    :ivar activation_fn: The resolved activation, called in ``call()``.
    :vartype activation_fn: Callable
    :ivar dropout_rate: The stored dropout rate.
    :vartype dropout_rate: float
    :ivar use_bias: Whether the projections carry a bias.
    :vartype use_bias: bool
    :ivar kernel_initializer: The resolved kernel initializer, cloned per
        Dense layer in the same way.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar output_kernel_initializer: The resolved ``fc2`` initializer, or
        ``None`` when ``fc2`` follows ``kernel_initializer``.
    :vartype output_kernel_initializer: Optional[keras.initializers.Initializer]
    :ivar bias_initializer: The resolved bias initializer. It is the source
        the per-layer clones are rebuilt from, and is not handed to either
        Dense layer itself.
    :vartype bias_initializer: keras.initializers.Initializer
    :ivar kernel_regularizer: The resolved kernel regularizer, or ``None``.
    :vartype kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar bias_regularizer: The resolved bias regularizer, or ``None``.
    :vartype bias_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar fc1: ``Dense(hidden_dim)``, the expansion.
    :vartype fc1: keras.layers.Dense
    :ivar fc2: ``Dense(output_dim)``, the contraction.
    :vartype fc2: keras.layers.Dense
    :ivar dropout: ``Dropout(dropout_rate)``, or ``None`` when the rate is
        0.0.
    :vartype dropout: Optional[keras.layers.Dropout]

    :raises ValueError: If ``hidden_dim`` or ``output_dim`` is not positive.
    :raises ValueError: If ``dropout_rate`` is outside ``[0.0, 1.0)``.

    Input shape:
        Tensor of rank >= 2, shape ``(..., input_dim)``. The input width is
        independent of ``output_dim``.

    Output shape:
        Same rank and leading axes as the input, with the last axis set to
        ``output_dim``.

    Example:
        .. code-block:: python

            mlp = MLPBlock(hidden_dim=2048, output_dim=512)
            y = mlp(keras.random.normal((2, 10, 512)))
            y.shape                 # (2, 10, 512)

    Note:
        Sub-layers are created in ``__init__`` and built explicitly in
        ``build()``. Keras does not build them on its own here, because
        ``fc2`` sees ``hidden_dim`` rather than the input width.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = "gelu",
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        output_kernel_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        """Validate the configuration and create the two projections.

        Every argument is documented on the class. Validation runs before any
        attribute is stored, so a rejected configuration leaves no half-built
        layer behind.

        :param hidden_dim: Hidden width of ``fc1``. Must be positive.
        :type hidden_dim: int
        :param output_dim: Output width of ``fc2``. Must be positive.
        :type output_dim: int
        :param activation: Activation name or callable, applied after ``fc1``.
        :type activation: Union[str, Callable]
        :param dropout_rate: Dropout rate applied after the activation only.
            Must be in ``[0.0, 1.0)``.
        :type dropout_rate: float
        :param use_bias: Whether ``fc1`` and ``fc2`` carry a bias.
        :type use_bias: bool
        :param kernel_initializer: Initializer for both kernels, cloned once
            per kernel.
        :type kernel_initializer: Union[str, keras.initializers.Initializer]
        :param output_kernel_initializer: Initializer for ``fc2`` only.
            ``None`` means ``fc2`` gets a clone of ``kernel_initializer``.
        :type output_kernel_initializer: Optional[Union[str, keras.initializers.Initializer]]
        :param bias_initializer: Initializer for both biases, cloned once per
            bias.
        :type bias_initializer: Union[str, keras.initializers.Initializer]
        :param kernel_regularizer: Regularizer for both kernels, or ``None``.
        :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param bias_regularizer: Regularizer for both biases, or ``None``.
        :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param kwargs: Extra arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If ``hidden_dim`` or ``output_dim`` is not
            positive, or if ``dropout_rate`` is outside ``[0.0, 1.0)``.
        """
        super().__init__(**kwargs)

        # Validate inputs immediately
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")

        # Store ALL configuration parameters
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.output_kernel_initializer = (
            keras.initializers.get(output_kernel_initializer)
            if output_kernel_initializer is not None else None
        )
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Get activation function once
        self.activation_fn = keras.activations.get(activation)

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern)
        # They will be unbuilt until build() is called
        # fc1 takes its own kernel clone for the same reason fc2 already
        # does, and BOTH biases are cloned too: the bias half of this defect
        # is invisible at the 'zeros' default, and with an unseeded
        # RandomNormal() instance fc1.bias == fc2.bias at
        # hidden_dim == output_dim (MEASURED max|delta| = 0.0 at 16/16).
        # The rule and the mechanism are written out at glu_ffn.py,
        # decisions.md D-008.
        self.fc1 = keras.layers.Dense(
            units=self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="fc1"
        )

        self.fc2 = keras.layers.Dense(
            units=self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(
                self.output_kernel_initializer
                if self.output_kernel_initializer is not None
                else self.kernel_initializer
            ),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="fc2"
        )

        # Create dropout layer if needed
        if self.dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(
                rate=self.dropout_rate,
                name="dropout"
            )
        else:
            self.dropout = None

        logger.info(
            f"Initialized MLPBlock with hidden_dim={hidden_dim}, "
            f"output_dim={output_dim}, activation={activation}, "
            f"dropout_rate={dropout_rate}"
        )

    def build(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> None:
        """
        Build the two projections and the dropout.

        Keras calls this the first time the layer sees input. Each sub-layer
        is built explicitly, in the order ``call()`` uses them, so every
        weight exists before a save or a restore. ``fc2`` and the dropout see
        the intermediate width ``hidden_dim``, not the input width.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # Build sub-layers in computational order for robust serialization
        self.fc1.build(input_shape)

        # Compute intermediate shape after first dense layer
        intermediate_shape = list(input_shape)
        intermediate_shape[-1] = self.hidden_dim
        intermediate_shape_tuple = tuple(intermediate_shape)

        # Dropout doesn't change shape
        if self.dropout is not None:
            self.dropout.build(intermediate_shape_tuple)

        # Build second dense layer
        self.fc2.build(intermediate_shape_tuple)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Run the block: ``fc1`` -> activation -> dropout -> ``fc2``.

        Dropout only exists when ``dropout_rate`` is above 0.0, and it sits
        between the activation and ``fc2``. Nothing runs after ``fc2``.

        :param inputs: Input tensor of shape ``(..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether to run in training mode. Only affects
            dropout.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(..., output_dim)``.
        :rtype: keras.KerasTensor
        """
        # First dense layer (expansion)
        x = self.fc1(inputs)

        # Activation function
        x = self.activation_fn(x)

        # Dropout after activation if enabled
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        # Second dense layer (projection)
        x = self.fc2(x)

        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape.

        Every axis is preserved except the last, which becomes
        ``output_dim``.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The input shape with the last axis set to ``output_dim``.
        :rtype: Tuple[Optional[int], ...]
        """
        # Convert to list for manipulation
        output_shape = list(input_shape)
        # Only the last dimension changes (to output_dim)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the config needed to rebuild this layer.

        Holds every ``__init__`` argument. ``activation`` is stored in its
        serialized form, taken from ``activation_fn``, so a callable
        round-trips as well as a name.

        :return: The complete layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "activation": keras.activations.serialize(self.activation_fn),
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "output_kernel_initializer": (
                keras.initializers.serialize(self.output_kernel_initializer)
                if self.output_kernel_initializer is not None else None
            ),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
