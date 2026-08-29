"""
A Monarch-structured Feed-Forward Network (order-2 Monarch matrices).

This layer is a drop-in replacement for the standard position-wise
Feed-Forward Network (FFN) of a Transformer. Each of the two dense projections
is replaced by an **order-2 Monarch matrix**: a structured (sub-quadratic)
parameterization written as the product of two block-diagonal matrices with a
fixed reshape/permute between them, introduced by Dao et al. (2022).

It cuts parameter count and FLOPs from ``O(n^2)`` to ``O(n^1.5)`` at
``nblocks = sqrt(n)``, and keeps much of a dense matrix's expressivity. The
same structure covers many fast transforms, including the FFT and the
Hadamard transform.

**Architecture Overview:**
The block keeps the familiar expand-then-contract FFN shape. Only the two
linear maps change: each is a Monarch map instead of a single dense kernel.
The flow diagram is on ``MonarchFFN`` itself, beside the code that runs it.

**Mathematics:**
An order-2 Monarch linear map sends a vector of dimension ``n_in`` to a vector
of dimension ``n_out``. Both dimensions are split into ``nblocks`` blocks
(``b_in = n_in / nblocks``, ``b_out = n_out / nblocks``). The map is computed in
five reshape/einsum steps, and no dense ``n x n`` kernel is ever materialized:

1.  reshape ``(..., n_in)`` -> ``(..., nblocks, b_in)``
2.  first block-diagonal multiply with ``L`` of shape ``(nblocks, b_in, b_out)``
    contracting the ``b_in`` axis -> ``(..., nblocks, b_out)``
3.  permutation: transpose the ``(nblocks, b_out)`` axes -> ``(..., b_out, nblocks)``
    (this transpose IS the Monarch interleaving permutation)
4.  second block-diagonal multiply with ``R`` of shape ``(b_out, nblocks, nblocks)``
    contracting the ``nblocks`` axis -> ``(..., b_out, nblocks)``
5.  reshape back -> ``(..., n_out)``

For the square case (``n_in == n_out``) this is exactly an order-2 Monarch
matrix. For the non-square case (``n_in != n_out``) the first block-diagonal
factor is rectangular per block (``b_in -> b_out``). That is the smallest
generalization that keeps the structure intact without falling back to a dense
projection. It requires ``nblocks`` to divide **all** of ``input_dim``,
``hidden_dim`` and ``output_dim`` so the block grids line up; that is checked
in ``__init__`` and in ``build``.

References:
-   Dao, T., Chen, B., Sohoni, N., Desai, A., Poli, M., Grogan, J., Liu, A.,
    Rao, A., Rudra, A., & Ré, C. (2022). Monarch: Expressive Structured
    Matrices for Efficient and Accurate Training. ICML.
    arXiv preprint arXiv:2204.00595.
"""

import keras
from typing import Callable, Optional, Union, Any, Dict, Tuple
from keras import initializers, regularizers, activations

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MonarchFFN(keras.layers.Layer):
    """
    Order-2 Monarch-structured Feed-Forward Network.

    Each of the two dense projections of a standard FFN becomes an order-2
    Monarch map: two block-diagonal factors with a reshape/permute between
    them. The computation is
    ``FFN(x) = monarch_contract(dropout(activation(monarch_expand(x))))``, with
    an optional bias after each Monarch map. The module docstring carries the
    five-step Monarch math and the reference (Dao et al. 2022).

    **Architecture Overview:**

    .. code-block:: text

        Input  [..., input_dim]
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ expand: Monarch -> hidden_dim   │
        │   factors L_e, R_e   [+ bias_e] │
        └────────────────┬────────────────┘
                         ▼  [..., hidden_dim]
        ┌─────────────────────────────────┐
        │ activation  (default GELU)      │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ dropout  (no-op at rate 0)      │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ contract: Monarch -> output_dim │
        │   factors L_c, R_c   [+ bias_c] │
        └────────────────┬────────────────┘
                         ▼
        Output [..., output_dim]

        The bias rows exist only when use_bias=True. Dropout
        is always in the graph; at rate 0.0 it is a no-op.

    **One Monarch map (the block arithmetic):**

    .. code-block:: text

        n_in -> n_out, with k = nblocks,
        b_in = n_in / k and b_out = n_out / k:

          x   [..., n_in]
          │
          ▼  reshape
          x   [..., k, b_in]
          │
          ▼  einsum '...ki,kio->...ko'  with L (k, b_in, b_out)
          x   [..., k, b_out]
          │
          ▼  transpose the last two axes
          x   [..., b_out, k]        <- the Monarch permutation
          │
          ▼  einsum '...ok,okj->...oj'  with R (b_out, k, k)
          x   [..., b_out, k]
          │
          ▼  reshape
          y   [..., n_out]

        L is rectangular per block (b_in -> b_out); R is square
        in the block axis and mixes across blocks. No dense
        (n_in, n_out) kernel is ever built. Kernel parameters
        are k*b_in*b_out + b_out*k*k, against n_in*n_out dense.

    :param hidden_dim: Intermediate (expansion) width. Must be a positive int
        and divisible by ``nblocks``.
    :type hidden_dim: int
    :param output_dim: Output width. Must be a positive int and divisible by
        ``nblocks``.
    :type output_dim: int
    :param nblocks: Number of Monarch blocks, the structure knob. The per-block
        size is ``dim / nblocks``. Must be a positive int that divides
        ``input_dim``, ``hidden_dim`` and ``output_dim``. Defaults to 4.
    :type nblocks: int
    :param activation: Activation applied after the expand map. A name or a
        callable. Defaults to 'gelu'.
    :type activation: Union[str, Callable]
    :param dropout_rate: Dropout rate applied after the activation. Must be in
        ``[0.0, 1.0)``. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether to add a bias vector after each Monarch map.
        Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the Monarch factor weights.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for the bias vectors. Defaults to 'zeros'.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Regularizer for the Monarch factor weights.
        Defaults to None.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Regularizer for the bias vectors.
        Defaults to None.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on).
    :type kwargs: Any

    :ivar hidden_dim: The stored expansion width.
    :vartype hidden_dim: int
    :ivar output_dim: The stored output width.
    :vartype output_dim: int
    :ivar nblocks: The stored block count.
    :vartype nblocks: int
    :ivar activation: The RESOLVED activation callable, not the name that was
        passed. ``get_config()`` serializes it back to a name.
    :vartype activation: Callable
    :ivar dropout_rate: The stored dropout rate, cast to ``float``.
    :vartype dropout_rate: float
    :ivar use_bias: Whether each Monarch map is followed by a bias.
    :vartype use_bias: bool
    :ivar kernel_initializer: The resolved factor initializer.
    :vartype kernel_initializer: initializers.Initializer
    :ivar bias_initializer: The resolved bias initializer.
    :vartype bias_initializer: initializers.Initializer
    :ivar kernel_regularizer: The resolved factor regularizer, or ``None``.
    :vartype kernel_regularizer: Optional[regularizers.Regularizer]
    :ivar bias_regularizer: The resolved bias regularizer, or ``None``.
    :vartype bias_regularizer: Optional[regularizers.Regularizer]
    :ivar dropout: ``Dropout(dropout_rate)``. Always present, even at rate 0.0.
    :vartype dropout: keras.layers.Dropout
    :ivar expand_l: Expand-map factor ``L``, shape
        ``(nblocks, input_dim // nblocks, hidden_dim // nblocks)``. ``None``
        until ``build()``.
    :vartype expand_l: Optional[keras.Variable]
    :ivar expand_r: Expand-map factor ``R``, shape
        ``(hidden_dim // nblocks, nblocks, nblocks)``. ``None`` until
        ``build()``.
    :vartype expand_r: Optional[keras.Variable]
    :ivar contract_l: Contract-map factor ``L``, shape
        ``(nblocks, hidden_dim // nblocks, output_dim // nblocks)``. ``None``
        until ``build()``.
    :vartype contract_l: Optional[keras.Variable]
    :ivar contract_r: Contract-map factor ``R``, shape
        ``(output_dim // nblocks, nblocks, nblocks)``. ``None`` until
        ``build()``.
    :vartype contract_r: Optional[keras.Variable]
    :ivar expand_bias: Bias of shape ``(hidden_dim,)``, or ``None`` when
        ``use_bias`` is False.
    :vartype expand_bias: Optional[keras.Variable]
    :ivar contract_bias: Bias of shape ``(output_dim,)``, or ``None`` when
        ``use_bias`` is False.
    :vartype contract_bias: Optional[keras.Variable]

    :raises ValueError: If ``hidden_dim`` or ``output_dim`` is not a positive
        integer.
    :raises ValueError: If ``nblocks`` is not a positive integer.
    :raises ValueError: If ``hidden_dim`` or ``output_dim`` is not divisible by
        ``nblocks``.
    :raises ValueError: If ``dropout_rate`` is outside ``[0.0, 1.0)``.
    :raises ValueError: From ``build()``, if the last input dimension is
        undefined or is not divisible by ``nblocks``.

    Input shape:
        Tensor of rank >= 2, shape ``(..., input_dim)``, with ``input_dim``
        divisible by ``nblocks``.

    Output shape:
        Same rank and leading axes as the input, with the last axis set to
        ``output_dim``.

    Example:
        .. code-block:: python

            ffn = MonarchFFN(
                hidden_dim=1024, output_dim=256, nblocks=4)
            y = ffn(keras.random.normal((2, 10, 512)))
            y.shape                 # (2, 10, 256)

    Note:
        At ``nblocks = 1`` the Monarch map degenerates to a dense matrix. At
        ``nblocks = sqrt(dim)`` the parameter count is ``O(dim^1.5)`` instead
        of ``O(dim^2)``. The structure is exact for square maps, and kept for
        non-square maps by making the first factor rectangular per block.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        nblocks: int = 4,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'gelu',
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Validate the configuration and create the dropout sub-layer.

        Every argument is documented on the class. Validation runs before any
        attribute is stored, so a rejected configuration leaves no half-built
        layer behind. The Monarch factor weights are not created here: their
        shapes need ``input_dim``, which only ``build()`` knows.

        :raises ValueError: If ``hidden_dim``, ``output_dim`` or ``nblocks`` is
            not a positive integer, if ``hidden_dim`` or ``output_dim`` is not
            divisible by ``nblocks``, or if ``dropout_rate`` is outside
            ``[0.0, 1.0)``.
        """
        super().__init__(**kwargs)

        # Reject bad configuration before storing anything.
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}")
        if not isinstance(nblocks, int) or nblocks <= 0:
            raise ValueError(f"nblocks must be a positive integer, got {nblocks}")
        if hidden_dim % nblocks != 0:
            raise ValueError(
                f"hidden_dim must be divisible by nblocks, "
                f"got hidden_dim={hidden_dim}, nblocks={nblocks}"
            )
        if output_dim % nblocks != 0:
            raise ValueError(
                f"output_dim must be divisible by nblocks, "
                f"got output_dim={output_dim}, nblocks={nblocks}"
            )
        if not isinstance(dropout_rate, (int, float)) or not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")

        # Store every constructor argument; get_config() returns all of them.
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nblocks = nblocks
        self.activation = activations.get(activation)
        self.dropout_rate = float(dropout_rate)
        self.use_bias = bool(use_bias)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Sub-layers are created here, per the Keras 3 pattern. The Monarch
        # factor weights are not: their shapes need input_dim, so build() makes
        # them.
        self.dropout = keras.layers.Dropout(rate=self.dropout_rate, name="dropout")

        # Weight handles, filled in by build().
        self.expand_l = None
        self.expand_r = None
        self.contract_l = None
        self.contract_r = None
        self.expand_bias = None
        self.contract_bias = None

        logger.info(
            f"Initialized MonarchFFN with hidden_dim={hidden_dim}, "
            f"output_dim={output_dim}, nblocks={nblocks}, "
            f"activation={activation}, dropout_rate={dropout_rate}"
        )

    def _add_monarch_weights(
        self, prefix: str, n_in: int, n_out: int
    ) -> Tuple[keras.Variable, keras.Variable]:
        """
        Create the two block-diagonal factors of one order-2 Monarch map.

        :param prefix: Name prefix for the created weights.
        :type prefix: str
        :param n_in: Input dimension of this Monarch map (divisible by nblocks).
        :type n_in: int
        :param n_out: Output dimension of this Monarch map (divisible by nblocks).
        :type n_out: int
        :return: Tuple ``(L, R)`` of the two block-diagonal factor weights.
        :rtype: Tuple[keras.Variable, keras.Variable]
        """
        b_in = n_in // self.nblocks
        b_out = n_out // self.nblocks
        # First block-diagonal factor: per-block (b_in -> b_out) map.
        l = self.add_weight(
            name=f"{prefix}_l",
            shape=(self.nblocks, b_in, b_out),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        # Second block-diagonal factor: mixes across blocks, per output-block row.
        r = self.add_weight(
            name=f"{prefix}_r",
            shape=(b_out, self.nblocks, self.nblocks),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        return l, r

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the Monarch factor weights and the dropout sub-layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last input dimension is undefined or not
            divisible by nblocks.
        """
        if self.built:
            return

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("The last dimension of input_shape must be defined")

        # DECISION plan_2026-06-19_2ea7a9a0/D-003: for the non-square expand and
        # contract maps, factor L is rectangular per block (nblocks, b_in, b_out)
        # and factor R stays square (b_out, nblocks, nblocks). Do NOT square them
        # up with a trailing Dense; that re-adds an unstructured O(n^2) kernel.
        # The price is this guard: nblocks must divide input_dim as well.
        # That plan directory is gone, so this comment is the record.
        if input_dim % self.nblocks != 0:
            raise ValueError(
                f"input_dim must be divisible by nblocks, "
                f"got input_dim={input_dim}, nblocks={self.nblocks}"
            )

        # Expand Monarch map: input_dim -> hidden_dim
        self.expand_l, self.expand_r = self._add_monarch_weights(
            "expand", input_dim, self.hidden_dim
        )
        # Contract Monarch map: hidden_dim -> output_dim
        self.contract_l, self.contract_r = self._add_monarch_weights(
            "contract", self.hidden_dim, self.output_dim
        )

        if self.use_bias:
            self.expand_bias = self.add_weight(
                name="expand_bias",
                shape=(self.hidden_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
            )
            self.contract_bias = self.add_weight(
                name="contract_bias",
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
            )

        # Dropout doesn't change shape; build on the (..., hidden_dim) intermediate.
        intermediate_shape = list(input_shape)
        intermediate_shape[-1] = self.hidden_dim
        self.dropout.build(tuple(intermediate_shape))

        # CRITICAL: Always call parent build() at the end.
        super().build(input_shape)

    def _monarch_map(
        self,
        x: keras.KerasTensor,
        l: keras.Variable,
        r: keras.Variable,
        n_in: int,
        n_out: int,
    ) -> keras.KerasTensor:
        """
        Apply one order-2 Monarch linear map to ``x`` of shape ``(..., n_in)``.

        The map is graph-safe. The last axis is split into ``(nblocks, b_in)``
        and merged back with reshape. The leading batch and sequence sizes are
        dynamic, so they are read with ``keras.ops.shape`` and stacked into an
        explicit 1-D integer target shape. There is no ``-1`` placeholder and
        no tuple/tensor concatenation. Block sizes are static ints from config,
        and only ``keras.ops`` reshape/transpose/einsum are used.

        :param x: Input tensor of shape ``(..., n_in)``.
        :type x: keras.KerasTensor
        :param l: First block-diagonal factor, shape ``(nblocks, b_in, b_out)``.
        :type l: keras.Variable
        :param r: Second block-diagonal factor, shape ``(b_out, nblocks, nblocks)``.
        :type r: keras.Variable
        :param n_in: Input dimension of this map.
        :type n_in: int
        :param n_out: Output dimension of this map.
        :type n_out: int
        :return: Output tensor of shape ``(..., n_out)``.
        :rtype: keras.KerasTensor
        """
        nblocks = self.nblocks
        b_in = n_in // nblocks
        b_out = n_out // nblocks

        # Per-axis dynamic leading sizes as a list of scalar int tensors.
        x_shape = keras.ops.shape(x)
        leading = [x_shape[i] for i in range(len(x.shape) - 1)]

        # 1. reshape (..., n_in) -> (..., nblocks, b_in)
        split_shape = keras.ops.stack(leading + [nblocks, b_in])
        x = keras.ops.reshape(x, split_shape)

        # 2. first block-diagonal multiply over b_in:
        #    (..., k, b_in) x (k, b_in, b_out) -> (..., k, b_out)
        x = keras.ops.einsum("...ki,kio->...ko", x, l)

        # 3. permutation: transpose the (k, b_out) trailing axes -> (..., b_out, k)
        ndim = len(x.shape)
        perm = list(range(ndim - 2)) + [ndim - 1, ndim - 2]
        x = keras.ops.transpose(x, axes=perm)

        # 4. second block-diagonal multiply over k:
        #    (..., b_out, k) x (b_out, k, k) -> (..., b_out, k)
        x = keras.ops.einsum("...ok,okj->...oj", x, r)

        # 5. reshape back (..., b_out, k) -> (..., n_out)
        merge_shape = keras.ops.stack(leading + [b_out * nblocks])
        x = keras.ops.reshape(x, merge_shape)
        return x

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass: expand Monarch -> activation -> dropout -> contract Monarch.

        :param inputs: Input tensor of shape ``(..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer is in training mode (affects dropout).
        :type training: Optional[bool]
        :return: Output tensor of shape ``(..., output_dim)``.
        :rtype: keras.KerasTensor
        """
        # Expand Monarch map (input_dim -> hidden_dim)
        input_dim = inputs.shape[-1]
        x = self._monarch_map(inputs, self.expand_l, self.expand_r, input_dim, self.hidden_dim)
        if self.use_bias:
            x = x + self.expand_bias

        # Activation
        x = self.activation(x)

        # Dropout (only active during training)
        x = self.dropout(x, training=training)

        # Contract Monarch map (hidden_dim -> output_dim)
        x = self._monarch_map(
            x, self.contract_l, self.contract_r, self.hidden_dim, self.output_dim
        )
        if self.use_bias:
            x = x + self.contract_bias

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape: last dimension becomes ``output_dim``.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple with last dimension = output_dim.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns ALL constructor parameters for perfect reconstruction.

        :return: Dictionary containing the complete layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'nblocks': self.nblocks,
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
