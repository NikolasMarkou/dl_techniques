"""
A GELU-gated linear unit feed-forward network.

This is a replacement for the position-wise FFN of a transformer block. A
standard FFN applies one fixed non-linearity to every feature. A GeGLU
computes a gate from the same input and multiplies the features by it, so it
can suppress or amplify a feature per token rather than per weight.

The layer runs in three stages:

1. **Project and split.** One Dense maps the input to ``2 * hidden_dim``. The
   result is split in half along the last axis: the first half is the gate,
   the second is the value. This is equivalent to two separate Dense layers of
   width ``hidden_dim`` and is usually faster.
2. **Gate.** The gate half goes through the activation (GELU by default) and
   multiplies the value half element-wise.
3. **Project out.** A second Dense maps the product to ``output_dim``.

The maths, for an input vector ``x``:

    [g, v] = W @ x + b         (W is (input_dim, 2 * hidden_dim))
    h = GELU(g) * v            (element-wise product)
    y = W_out @ h + b_out

Compare a standard FFN, ``y = W_out @ GELU(W_in @ x + b_in) + b_out``. There
the non-linearity acts on the same tensor it filters. Here the filter comes
from a separate projection of the input.

References:
-   Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv preprint
    arXiv:2002.05202. (introduces GeGLU)
-   Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language
    Modeling with Gated Convolutional Networks. ICML. (the original GLU)

GeGLU FFNs are used in several large language models, including PaLM.

"""

import keras
from keras import ops, layers, initializers, regularizers, activations
from typing import Optional, Union, Any, Dict, Callable, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.ffn.geglu_ffn")
class GeGLUFFN(keras.layers.Layer):
    """
    GELU-gated linear unit feed-forward network (GeGLU).

    One Dense projects the input to ``2 * hidden_dim``. The result is split in
    half along the last axis into a gate and a value. The gate goes through
    ``activation`` and multiplies the value element-wise. A second Dense
    projects the product to ``output_dim``:
    ``output = W_out @ (activation(gate) * value) + b_out``, where
    ``[gate, value] = split(W_in @ x + b_in)``.

    One wide projection followed by a split is equivalent to two separate
    projections of width ``hidden_dim``, and is usually faster.

    ``activation`` defaults to ``'gelu'``, which is what makes this GeGLU, but
    it is a constructor argument and any Keras activation works.

    **Architecture Overview:**

    .. code-block:: text

               Input  [..., input_dim]
                          │
                          ▼
              ┌───────────────────────┐
              │      input_proj       │
              │      Dense(2H)        │
              └───────────┬───────────┘
                          │  [..., 2H]
                          ▼
                  split on last axis
                ┌─────────┴─────────┐
                ▼                   ▼
              gate                value
            [..., H]             [..., H]
                │                   │
                ▼                   │
          ┌───────────┐             │
          │activation │             │
          └─────┬─────┘             │
                └─────────┬─────────┘
                          ▼
                    multiply  [..., H]
                          │
                          ▼
                  ┌───────────────┐
                  │    dropout    │
                  └───────┬───────┘
                          ▼
                  ┌───────────────┐
                  │  output_proj  │
                  │   Dense(O)    │
                  └───────┬───────┘
                          ▼
              Output [..., output_dim]

        H = hidden_dim, O = output_dim. `dropout` is always in the
        graph; at dropout_rate=0.0 it is a no-op, so it is not
        drawn as a conditional stage.

    **The GeGLU gate path:**

    .. code-block:: text

        input_proj output  [..., 2H]
        │
        │  ops.split(x, 2, axis=-1)
        │
        ├── first half  ──► gate   [..., H] ──► activation
        │                                            │
        └── second half ──► value  [..., H] ─────────┤
                                                     ▼
                              h = activation(gate) * value

        The split is positional. Channels 0..H-1 are always the
        gate, channels H..2H-1 always the value. Nothing marks
        them apart in the weights, so the two halves differ only
        because the gate half gets the non-linearity.

        With the default 'gelu', activation(gate) is near zero for
        strongly negative gate values, which suppresses the
        matching value channel for that token.

    :param hidden_dim: Width of the gate and of the value, each. The input
        projection is twice this wide. Must be positive.
    :type hidden_dim: int
    :param output_dim: Width of the final output. Must be positive.
    :type output_dim: int
    :param activation: Activation applied to the gate half only. A string name
        ('gelu', 'relu', 'swish') or a callable. Defaults to 'gelu'.
    :type activation: Union[str, Callable]
    :param dropout_rate: Dropout rate applied to the gated tensor, in
        ``[0.0, 1.0]``. Active only when ``training=True``. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether both Dense layers carry a bias. Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the kernels of both Dense
        layers. Each layer receives its own clone of it. Defaults to
        'glorot_uniform'.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for the biases of both Dense layers,
        cloned per layer in the same way. Used only when ``use_bias=True``.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Regularizer for the kernels of both Dense
        layers. Defaults to None.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Regularizer for the biases of both Dense layers.
        Used only when ``use_bias=True``. Defaults to None.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on).
    :type kwargs: Any

    :ivar hidden_dim: Width of the gate and of the value.
    :vartype hidden_dim: int
    :ivar output_dim: Width of the output.
    :vartype output_dim: int
    :ivar activation: The resolved gate activation, from
        ``keras.activations.get``.
    :vartype activation: Callable
    :ivar dropout_rate: The stored dropout rate.
    :vartype dropout_rate: float
    :ivar use_bias: Whether the Dense layers carry a bias.
    :vartype use_bias: bool
    :ivar kernel_initializer: The resolved kernel initializer. It is the
        source the per-layer clones are rebuilt from, and is not handed to
        either Dense layer itself.
    :vartype kernel_initializer: initializers.Initializer
    :ivar bias_initializer: The resolved bias initializer, cloned per layer
        in the same way.
    :vartype bias_initializer: initializers.Initializer
    :ivar kernel_regularizer: The resolved kernel regularizer, or ``None``.
    :vartype kernel_regularizer: Optional[regularizers.Regularizer]
    :ivar bias_regularizer: The resolved bias regularizer, or ``None``.
    :vartype bias_regularizer: Optional[regularizers.Regularizer]
    :ivar input_proj: ``Dense(2 * hidden_dim)``, split in ``call()``.
    :vartype input_proj: layers.Dense
    :ivar output_proj: ``Dense(output_dim)``, the final projection.
    :vartype output_proj: layers.Dense
    :ivar dropout: ``Dropout(dropout_rate)``, applied to the gated tensor.
    :vartype dropout: layers.Dropout

    :raises ValueError: If ``hidden_dim`` or ``output_dim`` is not positive, or
        ``dropout_rate`` is outside ``[0.0, 1.0]``, or a sub-layer constructor
        fails. The constructor catches any exception from sub-layer creation
        and re-raises it as ``ValueError``.
    :raises ValueError: From ``build()``, if the last axis of the input shape
        is ``None``.

    Input shape:
        Tensor of rank >= 2, shape ``(..., input_dim)``. The last axis must be
        known at build time; it may be any width.

    Output shape:
        Same rank and leading axes as the input, with the last axis set to
        ``output_dim``.

    Example:
        .. code-block:: python

            ffn = GeGLUFFN(hidden_dim=256, output_dim=64)
            y = ffn(keras.random.normal((2, 10, 64)))
            y.shape  # (2, 10, 64)

    Note:
        All sub-layers are created in ``__init__`` and built explicitly in
        ``build()``, so a ``.keras`` round-trip restores every weight.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'gelu',
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and create the two Dense projections.

        Every argument is documented on the class. Validation runs before any
        attribute is stored, so a rejected configuration leaves no half-built
        layer behind.

        :raises ValueError: If ``hidden_dim`` or ``output_dim`` is not
            positive, or ``dropout_rate`` is outside ``[0.0, 1.0]``, or a
            sub-layer constructor fails.
        """
        super().__init__(**kwargs)

        # Reject bad configuration before storing anything.
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be between 0 and 1, got {dropout_rate}"
            )

        # Store every constructor argument; get_config() returns all of them.
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Create every sub-layer here, unbuilt. build() builds them.
        # A failure below is re-raised as ValueError so callers see one
        # exception type from this constructor.
        try:
            # Each Dense takes its OWN clone of both initializers; the rule
            # and the mechanism are written out at glu_ffn.py, decisions.md
            # D-008. The two kernels collide at hidden_dim=8, output_dim=16
            # over an 8-wide input (both (8, 16)) and the two biases collide
            # whenever output_dim == 2 * hidden_dim -- MEASURED max|delta| =
            # 0.0 in both cases. Do NOT move the initializers back into
            # dense_kwargs.
            dense_kwargs = {
                "use_bias": self.use_bias,
                "kernel_regularizer": self.kernel_regularizer,
                "bias_regularizer": self.bias_regularizer,
            }

            # input_dim -> hidden_dim * 2; call() splits this in half.
            self.input_proj = layers.Dense(
                units=hidden_dim * 2,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                name="input_proj",
                **dense_kwargs
            )

            # hidden_dim -> output_dim.
            self.output_proj = layers.Dense(
                units=output_dim,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                name="output_proj",
                **dense_kwargs
            )

            self.dropout = layers.Dropout(dropout_rate, name="dropout")

        except Exception as e:
            logger.error(f"Failed to create GeGLUFFN sub-layers: {e}")
            raise ValueError(f"Failed to create GeGLUFFN sub-layers: {e}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the weights of every sub-layer.

        Each sub-layer is built explicitly so that all weight variables exist
        before Keras restores saved weights. A lazily-built sub-layer would be
        skipped on load and would silently keep its fresh initialization.

        This method has no ``if self.built: return`` guard, so calling it twice
        rebuilds the sub-layers. Keras calls it once.

        :param input_shape: Shape tuple of the input tensor. The last axis must
            be known.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last axis of ``input_shape`` is ``None``.
        """
        if input_shape[-1] is None:
            raise ValueError("Last dimension of input must be defined")

        self.input_proj.build(input_shape)

        # After the split each half is (..., hidden_dim), which is what the
        # dropout and the output projection see.
        intermediate_shape = (*input_shape[:-1], self.hidden_dim)

        self.dropout.build(intermediate_shape)
        self.output_proj.build(intermediate_shape)

        # Keras requires the parent build() call last.
        super().build(input_shape)


    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Run the GeGLU forward pass.

        :param inputs: Input tensor of any rank, shape ``(..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Training-mode flag, passed to the dropout sub-layer.
        :type training: Optional[bool]
        :return: Tensor with the same rank as ``inputs`` and last axis
            ``output_dim``.
        :rtype: keras.KerasTensor
        """
        # One projection to 2 * hidden_dim, then split in half.
        gate_and_value = self.input_proj(inputs)
        gate, value = ops.split(gate_and_value, 2, axis=-1)

        # Only the gate half is passed through the activation.
        activated_gate = self.activation(gate)
        gated_value = activated_gate * value

        # A no-op outside training and at dropout_rate=0.0.
        gated_value = self.dropout(gated_value, training=training)

        # Project (..., hidden_dim) down to (..., output_dim).
        output = self.output_proj(gated_value)

        return output

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
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'activation': activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
