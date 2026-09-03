"""
The Squared-ReLU feed-forward network from the Primer architecture, built by
:class:`SquaredReLUFFN`.

This is the standard transformer MLP with one change: the intermediate
non-linearity is fixed to ``relu(x) ** 2`` instead of being configurable.
The Primer architecture search found this one of its most transferable
changes to the vanilla FFN. At a single sequence position:

    FFN(x) = W_2 @ relu(W_1 @ x + b_1)**2 + b_2

The squared ReLU rises faster than plain ReLU on the positive half-line and
keeps exact sparsity on the negative half-line, which the Primer search found
improves training efficiency. There is no configurable activation; use
``MLPBlock`` if the architecture needs one.

References:
    - So et al., 2021. Primer: Searching for Efficient Transformers for
      Language Modeling. (https://arxiv.org/abs/2109.08668)
    - Vaswani et al., 2017. Attention Is All You Need. (base FFN structure
      this layer specializes)
"""

import keras
from typing import Optional, Union, Any, Dict, Tuple

from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.ffn.squared_relu_ffn")
class SquaredReLUFFN(keras.layers.Layer):
    """
    Squared-ReLU (Primer) feed-forward network.

    Two Dense layers with ``relu(x) ** 2`` between them:
    ``FFN(x) = W_2 @ relu(W_1 @ x + b_1)**2 + b_2``. The same weights apply at
    every token position.

    The non-linearity is fixed. There is no ``activation`` argument, because
    the squared ReLU is the one thing that distinguishes this layer from a
    plain MLP block. Use ``MLPBlock`` if you need a configurable activation.

    Architecture:

    .. code-block:: text

               Input  [..., input_dim]
                          │
                          ▼
                  ┌───────────────┐
                  │      fc1      │
                  │   Dense(H)    │
                  └───────┬───────┘
                          │  [..., H]
                          ▼
                  ┌───────────────┐
                  │ relu(x) ** 2  │
                  │  (fixed)      │
                  └───────┬───────┘
                          ▼
                  ┌───────────────┐
                  │    dropout    │
                  └───────┬───────┘
                          ▼
                  ┌───────────────┐
                  │      fc2      │
                  │   Dense(O)    │
                  └───────┬───────┘
                          ▼
              Output [..., output_dim]

        H = hidden_dim, O = output_dim. `dropout` is always in the
        graph; at dropout_rate=0.0 it is a no-op, so it is not
        drawn as a conditional stage. There is no sub-block below
        this level: `call()` runs these four stages in order and
        has no branch.

    :param hidden_dim: Width of the expansion, ``fc1``'s output. Must be
        positive. A 4x expansion over the input width is typical.
    :type hidden_dim: int
    :param output_dim: Width of the final output, ``fc2``'s output. Must be
        positive. Usually equal to the input width so the block can sit inside
        a residual connection.
    :type output_dim: int
    :param dropout_rate: Dropout rate applied after the squared ReLU. Must be
        in ``[0.0, 1.0)`` -- the upper bound is exclusive here, unlike the
        other FFN layers in this package, so ``1.0`` raises. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether both Dense layers carry a bias. Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the kernels of both Dense
        layers. Each layer receives its own clone of it. Defaults to
        'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the biases of both Dense layers,
        cloned per layer in the same way. Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the kernels of both Dense
        layers. A string name ('l2') or a Regularizer. Defaults to None.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for the biases of both Dense layers.
        A string name ('l1') or a Regularizer. Defaults to None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on).
    :type kwargs: Any

    :ivar hidden_dim: Width of the expansion.
    :vartype hidden_dim: int
    :ivar output_dim: Width of the output.
    :vartype output_dim: int
    :ivar dropout_rate: The stored dropout rate.
    :vartype dropout_rate: float
    :ivar use_bias: Whether the Dense layers carry a bias.
    :vartype use_bias: bool
    :ivar kernel_initializer: The resolved kernel initializer. It is the
        source the per-layer clones are rebuilt from, and is not handed to
        either Dense layer itself.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar bias_initializer: The resolved bias initializer, cloned per layer
        in the same way.
    :vartype bias_initializer: keras.initializers.Initializer
    :ivar kernel_regularizer: The resolved kernel regularizer, or ``None``.
    :vartype kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar bias_regularizer: The resolved bias regularizer, or ``None``.
    :vartype bias_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar fc1: ``Dense(hidden_dim)``, the expansion.
    :vartype fc1: keras.layers.Dense
    :ivar fc2: ``Dense(output_dim)``, the contraction.
    :vartype fc2: keras.layers.Dense
    :ivar dropout: ``Dropout(dropout_rate)``, applied after the squared ReLU.
    :vartype dropout: keras.layers.Dropout

    :raises ValueError: If ``hidden_dim`` or ``output_dim`` is not positive.
    :raises ValueError: If ``dropout_rate`` is outside ``[0.0, 1.0)``.

    Input shape:
        Tensor of rank >= 2, shape ``(..., input_dim)``. ``build()`` does not
        require the last axis to be known; the wrapped ``Dense`` does.

    Output shape:
        Same rank and leading axes as the input, with the last axis set to
        ``output_dim``.

    Example:
        .. code-block:: python

            ffn = SquaredReLUFFN(hidden_dim=256, output_dim=64)
            y = ffn(keras.random.normal((2, 10, 64)))
            y.shape  # (2, 10, 64)

    Note:
        ``relu(x) ** 2`` grows quadratically on the positive half-line, so
        activations can get large. Keep an eye on the scale of ``hidden_dim``
        and on the initializer if you see the loss diverge early.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        """Validate the configuration and create the two Dense layers.

        Every argument is documented on the class. Validation runs before any
        attribute is stored, so a rejected configuration leaves no half-built
        layer behind.

        :raises ValueError: If ``hidden_dim`` or ``output_dim`` is not
            positive, or ``dropout_rate`` is outside ``[0.0, 1.0)``. Note the
            open upper bound: ``dropout_rate=1.0`` is rejected here, where the
            other FFN layers in this package accept it.
        """
        super().__init__(**kwargs)

        # Reject bad configuration before storing anything.
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")

        # Store every constructor argument; get_config() returns all of them.
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Each Dense takes its own clone of both initializers (see
        # glu_ffn.py, decisions.md D-008): a shared instance replays the
        # identical draw when shapes collide, measured max|delta|=0.0.
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
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="fc2"
        )

        self.dropout = keras.layers.Dropout(
            rate=self.dropout_rate,
            name="dropout"
        )

        logger.info(
            f"Initialized SquaredReLUFFN with hidden_dim={hidden_dim}, "
            f"output_dim={output_dim}, dropout_rate={dropout_rate}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the weights of every sub-layer.

        Each sub-layer is built explicitly so that all weight variables exist
        before Keras restores saved weights. A lazily-built sub-layer would be
        skipped on load and would silently keep its fresh initialization.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        self.fc1.build(input_shape)

        # fc1 emits (..., hidden_dim). Dropout does not change shape, so the
        # same shape builds both the dropout and fc2.
        intermediate_shape = list(input_shape)
        intermediate_shape[-1] = self.hidden_dim
        intermediate_shape_tuple = tuple(intermediate_shape)

        self.dropout.build(intermediate_shape_tuple)
        self.fc2.build(intermediate_shape_tuple)

        # Keras requires the parent build() call last.
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Run the squared-ReLU FFN forward pass.

        :param inputs: Input tensor of any rank, shape ``(..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Training-mode flag, passed to the dropout sub-layer.
        :type training: Optional[bool]
        :return: Tensor with the same rank as ``inputs`` and last axis
            ``output_dim``.
        :rtype: keras.KerasTensor
        """
        # Expand to (..., hidden_dim).
        x = self.fc1(inputs)

        # The fixed non-linearity. There is no activation argument.
        x = keras.ops.square(keras.ops.relu(x))

        # A no-op outside training and at dropout_rate=0.0.
        x = self.dropout(x, training=training)

        # Contract to (..., output_dim).
        x = self.fc2(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Return the input shape with its last axis set to ``output_dim``.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape with the last axis replaced by ``output_dim``.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to rebuild this layer.

        :return: The base layer config plus every ``__init__`` argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config
