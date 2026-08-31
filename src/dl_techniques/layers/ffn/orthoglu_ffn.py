"""
An orthogonally-regularized Gated Linear Unit FFN.

This layer is a GLU whose two Dense projections are replaced by
``OrthoBlock``, a Dense layer carrying a soft orthonormality penalty on its
kernel. The gating decides which features pass; the penalty keeps the space
they are gated in well conditioned.

The layer runs in three stages:

1. **Project.** An ``OrthoBlock`` maps the input to ``2 * hidden_dim``.
2. **Gate.** The projection is split in half. The first half goes through the
   activation and multiplies the second half element-wise. This is the plain
   GLU mechanism.
3. **Project out.** A second ``OrthoBlock`` maps the product to
   ``output_dim``.

The maths, for an input vector ``x``:

    [g, v] = O_in(x)
    h = activation(g) * v
    y = O_out(h)

The penalty pushes each block's kernel ``W`` towards ``W^T W = I``. An exactly
orthogonal ``W`` would give three things:

-   **Norm preservation.** ``||Wx||_2 == ||x||_2``, so activation magnitudes
    neither explode nor vanish through the layer.
-   **Gradient stability.** The backward pass preserves gradient norm too,
    since ``||W^T g|| == ||g||``.
-   **Feature decorrelation.** Orthonormal rows cannot be redundant, so the
    block is pushed to learn distinct filters.

The penalty is a loss term, not a hard constraint, so these hold only
approximately. ``ortho_reg_factor`` sets how hard it pulls.

References:
-   Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv preprint
    arXiv:2002.05202. (the GLU mechanism and its variants)
-   Bansal, N., et al. (2018). Can We Gain More from Orthogonality
    Regularizations in Training Deep Networks? NeurIPS.
-   Cisse, M., et al. (2017). Parseval Networks: Improving Robustness to
    Adversarial Examples. ICML.

"""

import keras
from keras import ops, layers, activations
from typing import Optional, Union, Any, Dict, Callable, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..orthoblock import OrthoBlock
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.ffn.orthoglu_ffn")
class OrthoGLUFFN(keras.layers.Layer):
    """
    Orthogonally-regularized gated linear unit feed-forward network.

    A GLU whose two Dense projections are replaced by ``OrthoBlock``. The
    input block projects to ``2 * hidden_dim``; the result is split into a
    gate and a value, the gate goes through ``activation`` and multiplies the
    value, and the output block projects the product to ``output_dim``:
    ``output = O_out(activation(gate) * value)`` with
    ``[gate, value] = split(O_in(x))``.

    Each ``OrthoBlock`` carries a soft orthonormality penalty on its kernel,
    which pulls ``W^T W`` towards the identity. That keeps activation and
    gradient norms from drifting and pushes the block to learn non-redundant
    filters.

    **Architecture Overview:**

    .. code-block:: text

               Input  [..., input_dim]
                          │
                          ▼
              ┌───────────────────────┐
              │   input_proj_ortho    │
              │   OrthoBlock(2H)      │
              │   use_bias always     │
              │   False here          │
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
              ┌───────────────────────┐
              │   output_proj_ortho   │
              │   OrthoBlock(O)       │
              │   use_bias applies    │
              └───────────┬───────────┘
                          ▼
              Output [..., output_dim]

        H = hidden_dim, O = output_dim. `dropout` is always in the
        graph; at dropout_rate=0.0 it is a no-op, so it is not
        drawn as a conditional stage.

    **Inside one OrthoBlock, and where the penalty enters:**

    .. code-block:: text

        x ──► ┌──────────────────┐
              │   ortho_dense    │──► z  [..., units]
              │   Dense(units)   │
              └────────┬─────────┘
                       │ kernel W
                       ▼
              SoftOrthonormalConstraintRegularizer
                lambda = ortho_reg_factor
                l1     = 1e-5   (fixed inside OrthoBlock)
                       │
                       ▼
              added to the layer's regularization losses,
              pulling W^T W towards I

        z ──► ZeroCenteredRMSNorm ──► LayerScale
                                      per channel, clipped
                                      to [0, 1]
                                             │
                                             ▼
                                      block output

        The regularizer is ALWAYS attached. `ortho_reg_factor`
        only sets its lambda. At 0.0 the orthonormal term
        vanishes and the fixed l1 term remains, so 0.0 does not
        turn the penalty off.

        Both OrthoBlocks are built with activation=None, so
        OrthoBlock's own activation stage is inert here and the
        only non-linearity in this layer is the gate activation.
        LayerScale also carries a BinaryPreferenceRegularizer
        (multiplier 1e-4), a second loss not drawn above.

    :param hidden_dim: Width of the gate and of the value, each. The input
        block projects to twice this. Must be positive.
    :type hidden_dim: int
    :param output_dim: Width of the final output. Must be positive.
    :type output_dim: int
    :param activation: Activation applied to the gate half only. A string name
        ('gelu', 'relu') or a callable. Defaults to 'gelu'.
    :type activation: Union[str, Callable]
    :param dropout_rate: Dropout rate applied to the gated tensor, in
        ``[0.0, 1.0]``. Active only when ``training=True``. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether the OUTPUT ``OrthoBlock`` carries a bias. The
        input block is constructed with ``use_bias=False`` whatever this says.
        Defaults to True.
    :type use_bias: bool
    :param ortho_reg_factor: Orthonormality penalty strength. One float is
        used for both blocks. A ``(input, output)`` pair sets them separately.
        A list is accepted as well as a tuple, because a ``.keras`` round-trip
        returns a tuple as a list. Defaults to 1.0.
    :type ortho_reg_factor: Union[float, Tuple[float, float]]
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
    :ivar use_bias: Whether the output block carries a bias.
    :vartype use_bias: bool
    :ivar ortho_reg_factor: The penalty strength exactly as given, before it is
        unpacked into per-block factors. This is what ``get_config()``
        returns.
    :vartype ortho_reg_factor: Union[float, Tuple[float, float]]
    :ivar input_proj_ortho: ``OrthoBlock(2 * hidden_dim)``, split in ``call()``.
    :vartype input_proj_ortho: OrthoBlock
    :ivar output_proj_ortho: ``OrthoBlock(output_dim)``, the final projection.
    :vartype output_proj_ortho: OrthoBlock
    :ivar dropout: ``Dropout(dropout_rate)``, applied to the gated tensor.
    :vartype dropout: layers.Dropout

    :raises ValueError: If ``hidden_dim`` or ``output_dim`` is not positive, or
        ``dropout_rate`` is outside ``[0.0, 1.0]``.
    :raises ValueError: From ``OrthoBlock``, if a resolved
        ``ortho_reg_factor`` is negative or not a number.
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

            ffn = OrthoGLUFFN(hidden_dim=128, output_dim=64,
                              ortho_reg_factor=(0.1, 0.01))
            y = ffn(keras.random.normal((2, 10, 64)))
            y.shape  # (2, 10, 64)

    Note:
        The orthonormality penalty is a training-time loss, not a hard
        constraint. ``W^T W`` is pulled towards ``I``; it is never forced to
        equal it, so the norm-preservation argument holds approximately.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = "gelu",
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        ortho_reg_factor: Union[float, Tuple[float, float]] = 1.0,
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and create the two OrthoBlocks.

        Every argument is documented on the class. Validation runs before any
        attribute is stored, so a rejected configuration leaves no half-built
        layer behind.

        :raises ValueError: If ``hidden_dim`` or ``output_dim`` is not
            positive, or ``dropout_rate`` is outside ``[0.0, 1.0]``.
        :raises ValueError: From ``OrthoBlock``, if a resolved
            ``ortho_reg_factor`` is negative or not a number.
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
        self.ortho_reg_factor = ortho_reg_factor

        # Unpack a pair, or duplicate a single value, into per-block factors.
        # A list is accepted as well as a tuple: a tuple ortho_reg_factor is
        # serialized to a JSON list and comes back as a list after a .keras
        # round-trip, so testing only for tuple would misroute the reloaded
        # value into OrthoBlock.
        ortho_factors = (
            tuple(ortho_reg_factor)
            if isinstance(ortho_reg_factor, (tuple, list))
            else (ortho_reg_factor, ortho_reg_factor)
        )

        # Create every sub-layer here, unbuilt. build() builds them.
        # This block is always bias-free, whatever use_bias says.
        self.input_proj_ortho = OrthoBlock(
            units=hidden_dim * 2,
            activation=None,
            use_bias=False,
            ortho_reg_factor=ortho_factors[0],
            name="input_proj_ortho",
        )

        # This is the only block use_bias reaches.
        self.output_proj_ortho = OrthoBlock(
            units=output_dim,
            activation=None,
            use_bias=self.use_bias,
            ortho_reg_factor=ortho_factors[1],
            name="output_proj_ortho",
        )

        self.dropout = layers.Dropout(dropout_rate, name="dropout")

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
            raise ValueError("Last dimension of input must be defined")

        self.input_proj_ortho.build(input_shape)

        # After the split each half is (..., hidden_dim), which is what the
        # dropout and the output block see.
        intermediate_shape = (*input_shape[:-1], self.hidden_dim)
        self.dropout.build(intermediate_shape)
        self.output_proj_ortho.build(intermediate_shape)

        # Keras requires the parent build() call last.
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Run the OrthoGLU forward pass.

        ``training`` is forwarded to both ``OrthoBlock`` sub-layers as well as
        to dropout, because an ``OrthoBlock`` contains a normalization stage
        that behaves differently in training mode.

        :param inputs: Input tensor of any rank, shape ``(..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Training-mode flag.
        :type training: Optional[bool]
        :return: Tensor with the same rank as ``inputs`` and last axis
            ``output_dim``.
        :rtype: keras.KerasTensor
        """
        # One orthogonally-regularized projection to 2 * hidden_dim.
        gate_and_value = self.input_proj_ortho(inputs, training=training)

        # Split in half; only the gate half gets the activation.
        gate, value = ops.split(gate_and_value, indices_or_sections=2, axis=-1)
        activated_gate = self.activation(gate)
        gated_value = activated_gate * value

        # A no-op outside training and at dropout_rate=0.0.
        gated_value = self.dropout(gated_value, training=training)

        # Second orthogonally-regularized projection, down to output_dim.
        output = self.output_proj_ortho(gated_value, training=training)

        return output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
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

        ``ortho_reg_factor`` is stored and returned as given. A tuple comes
        back from a ``.keras`` round-trip as a list, which ``__init__``
        accepts.

        :return: The base layer config plus every ``__init__`` argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "activation": activations.serialize(self.activation),
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "ortho_reg_factor": self.ortho_reg_factor
            }
        )
        return config

# ---------------------------------------------------------------------
