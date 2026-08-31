"""
A Gated Linear Unit feed-forward network.

This is a drop-in replacement for the position-wise FFN of a transformer
block. A standard FFN projects the input up, applies one fixed non-linearity,
and projects back down. A GLU splits the up-projection into two branches. One
branch is the value and carries the content. The other is the gate and, after
an activation, multiplies the value element-wise. The gate is computed from
the same input, so the layer can suppress or amplify a feature per token
rather than per weight.

The layer runs three projections:

1. ``gate_proj`` -- a Dense to ``hidden_dim``, followed by ``activation``.
2. ``value_proj`` -- a second, independent Dense to ``hidden_dim``.
3. ``output_proj`` -- a Dense from ``hidden_dim`` to ``output_dim``, applied
   to the product of the two branches.

The maths, for an input vector ``x``:

    g = W_g @ x + b_g          (gate)
    v = W_v @ x + b_v          (value)
    h = activation(g) * v      (element-wise product)
    y = W_out @ h + b_out

``W_g`` and ``W_v`` are separate matrices; nothing ties them. The choice of
``activation`` names the variant: ``'swish'`` is SwiGLU, ``'gelu'`` is GeGLU,
``'sigmoid'`` is the original GLU, ``'linear'`` is the bilinear variant.

``GLUFFN`` is registered in ``ffn/factory.py`` under three keys -- ``glu``
(``activation='swish'``), ``reglu`` (``'relu'``) and ``bilinear``
(``'linear'``). They are the same class with different defaults.

References:
-   Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv preprint
    arXiv:2002.05202. (the Transformer-FFN analysis of the GLU family)
-   Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language
    Modeling with Gated Convolutional Networks. ICML. (the original GLU)

"""

import keras
from typing import Callable, Optional, Union, Any, Dict, Tuple
from keras import layers, initializers, regularizers, activations

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.ffn.glu_ffn")
class GLUFFN(keras.layers.Layer):
    """
    Gated Linear Unit feed-forward network.

    Two Dense layers read the same input. One is the gate, one is the value.
    The gate goes through ``activation`` and multiplies the value element-wise.
    A third Dense projects the result to ``output_dim``:
    ``output = W_out @ (activation(W_gate @ x) * (W_value @ x))``.

    The ``activation`` argument picks the variant. ``'swish'`` (the default)
    gives SwiGLU-style gating, ``'gelu'`` gives GeGLU, ``'sigmoid'`` gives the
    original GLU of Dauphin et al., and ``'linear'`` gives the bilinear
    variant.

    This class backs three ``FFN_REGISTRY`` keys. See the block-internals
    diagram below for the activation each key supplies.

    **Architecture Overview:**

    .. code-block:: text

               Input  [..., input_dim]
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
          ┌───────────┐       ┌────────────┐
          │ gate_proj │       │ value_proj │
          │  Dense(H) │       │  Dense(H)  │
          └─────┬─────┘       └─────┬──────┘
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

    **Gate / value split and variant selection:**

    .. code-block:: text

        x  [..., input_dim]
        │
        ├──► gate_proj  ──► g  [..., H] ──► activation(g)
        │                                        │
        └──► value_proj ──► v  [..., H] ─────────┤
                                                 ▼
                              h = activation(g) * v  [..., H]

        Only the gate branch is non-linear. The `value_proj`
        output reaches the multiply untouched.

        `activation` selects the variant. The three factory keys
        that build this class:

          factory key   activation   variant
          -----------   ----------   ---------------------
          glu           'swish'      SwiGLU-style gate
          reglu         'relu'       ReGLU
          bilinear      'linear'     bilinear, no gate
                                     non-linearity

    :param hidden_dim: Width of the gate and value projections. Must be a
        positive ``int``. Usually 2-4x the input width.
    :type hidden_dim: int
    :param output_dim: Width of the final output. Must be a positive ``int``.
        In a transformer block it usually equals the input width so the block
        can sit inside a residual connection.
    :type output_dim: int
    :param activation: Activation applied to the gate branch only. A string
        name ('swish', 'gelu', 'sigmoid', 'relu', 'linear') or a callable.
        Defaults to 'swish'.
    :type activation: Union[str, Callable]
    :param dropout_rate: Dropout rate applied to the gated tensor, in
        ``[0.0, 1.0]``. Active only when ``training=True``. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether all three Dense projections carry a bias.
        Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the kernels of all three Dense
        layers. Each layer receives its OWN clone of it, never the resolved
        instance itself. Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for the biases of all three Dense
        layers. Cloned per layer in the same way as the kernel initializer.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Regularizer for the kernels of all three Dense
        layers. Defaults to None.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Regularizer for the biases of all three Dense
        layers. Defaults to None.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on).
    :type kwargs: Any

    :ivar hidden_dim: Width of the gate and value projections.
    :vartype hidden_dim: int
    :ivar output_dim: Width of the output.
    :vartype output_dim: int
    :ivar activation: The resolved gate activation, from
        ``keras.activations.get``.
    :vartype activation: Callable
    :ivar dropout_rate: The stored dropout rate, cast to ``float``.
    :vartype dropout_rate: float
    :ivar use_bias: Whether the Dense layers carry a bias, cast to ``bool``.
    :vartype use_bias: bool
    :ivar kernel_initializer: The resolved kernel initializer. It is the
        source the three per-layer clones are rebuilt from, and is not
        handed to any Dense layer itself.
    :vartype kernel_initializer: initializers.Initializer
    :ivar bias_initializer: The resolved bias initializer, cloned per Dense
        layer in the same way.
    :vartype bias_initializer: initializers.Initializer
    :ivar kernel_regularizer: The resolved kernel regularizer, or ``None``.
    :vartype kernel_regularizer: Optional[regularizers.Regularizer]
    :ivar bias_regularizer: The resolved bias regularizer, or ``None``.
    :vartype bias_regularizer: Optional[regularizers.Regularizer]
    :ivar gate_proj: ``Dense(hidden_dim)``, the gate branch.
    :vartype gate_proj: layers.Dense
    :ivar value_proj: ``Dense(hidden_dim)``, the value branch.
    :vartype value_proj: layers.Dense
    :ivar output_proj: ``Dense(output_dim)``, the final projection.
    :vartype output_proj: layers.Dense
    :ivar dropout: ``Dropout(dropout_rate)``, applied to the gated tensor.
    :vartype dropout: layers.Dropout

    :raises ValueError: If ``hidden_dim`` or ``output_dim`` is not a positive
        ``int``, or ``dropout_rate`` is not a number in ``[0.0, 1.0]``.
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

            ffn = GLUFFN(hidden_dim=256, output_dim=64,
                         activation='gelu')
            y = ffn(keras.random.normal((2, 10, 64)))
            y.shape  # (2, 10, 64)

    Note:
        The gate is the only non-linear branch, so gradients reach the value
        branch through a plain product. That is what makes gradient flow
        better than a ReLU-then-project FFN of the same width.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'swish',
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Validate the configuration and create the three Dense projections.

        Every argument is documented on the class. Validation runs before any
        attribute is stored, so a rejected configuration leaves no half-built
        layer behind.

        :raises ValueError: If ``hidden_dim`` or ``output_dim`` is not a
            positive ``int``, or ``dropout_rate`` is not a number in
            ``[0.0, 1.0]``.
        """
        super().__init__(**kwargs)

        # Reject bad configuration before storing anything.
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}")
        if not isinstance(dropout_rate, (int, float)) or not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store every constructor argument; get_config() returns all of them.
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.dropout_rate = float(dropout_rate)
        self.use_bias = bool(use_bias)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)


        # Create every sub-layer here, unbuilt. build() builds them.
        # DECISION plan-2026-08-29T043546-e97b34d8/D-008 -- clone_initializer
        # per Dense, and the reason the whole ffn package now does this.
        # MECHANISM: keras.initializers.get('glorot_uniform') returns an
        # INSTANCE that has already drawn a concrete seed (measured
        # seed=369497522). One such instance handed to two weights therefore
        # draws the SAME numbers twice; clone_initializer rebuilds it from
        # get_config(), which drops that seed, so each clone draws fresh.
        # A raw STRING is safe -- Keras resolves it once per sub-layer -- so
        # only the resolve-once-share-twice spelling is affected. Here it was
        # fatal: gate_proj and value_proj are both Dense(hidden_dim) off the
        # same input and call() computes activation(gate) * value, so with
        # one shared instance the two were the same function and the gating
        # did not exist (MEASURED max|delta| = 0.0 for kernels AND biases, at
        # every configuration -- no shape precondition). Do NOT put
        # self.kernel_initializer / self.bias_initializer back into
        # dense_kwargs. A SEEDED initializer defeats the clone by design and
        # is why the guard for this uses an unseeded one.
        # See decisions.md D-008.
        dense_kwargs = {
            "use_bias": self.use_bias,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
        }

        self.gate_proj = layers.Dense(
            self.hidden_dim,
            # The gate activation is applied in call(), not here.
            activation=None,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            name="gate_proj",
            **dense_kwargs
        )

        self.value_proj = layers.Dense(
            self.hidden_dim,
            activation=None,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            name="value_proj",
            **dense_kwargs
        )

        self.output_proj = layers.Dense(
            self.output_dim,
            activation=None,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            name="output_proj",
            **dense_kwargs
        )

        self.dropout = layers.Dropout(
            rate=self.dropout_rate,
            name="dropout"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the weights of every sub-layer.

        Each sub-layer is built explicitly so that all weight variables exist
        before Keras restores saved weights. A lazily-built sub-layer would be
        skipped on load and would silently keep its fresh initialization.

        :param input_shape: Shape tuple of the input tensor. The last axis must
            be known.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last axis of ``input_shape`` is ``None``.
        """
        if self.built:
            return

        if input_shape[-1] is None:
            raise ValueError("The last dimension of input_shape must be defined")

        # Both projections read the raw input, so both take input_shape.
        self.gate_proj.build(input_shape)
        self.value_proj.build(input_shape)

        # Both projections emit (..., hidden_dim), so one shape serves both
        # downstream sub-layers.
        intermediate_shape = self.gate_proj.compute_output_shape(input_shape)

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
        Run the GLU forward pass.

        :param inputs: Input tensor of any rank. The last axis is the feature
            axis and may be any width.
        :type inputs: keras.KerasTensor
        :param training: Training-mode flag, passed to the dropout sub-layer.
        :type training: Optional[bool]
        :return: Tensor with the same rank as ``inputs`` and last axis
            ``output_dim``.
        :rtype: keras.KerasTensor
        """
        # Two parallel projections of the same input.
        # Both produce shape (..., hidden_dim).
        gate = self.gate_proj(inputs)
        value = self.value_proj(inputs)

        # Gate the value. Only the gate branch gets the non-linearity.
        gated_value = self.activation(gate) * value

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
