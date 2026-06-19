"""
A Monarch-structured Feed-Forward Network (order-2 Monarch matrices).

This layer is a drop-in replacement for the standard position-wise
Feed-Forward Network (FFN) of a Transformer, in which each of the two dense
projections is replaced by an **order-2 Monarch matrix**. A Monarch matrix is
a structured (sub-quadratic) parameterization expressed as the product of two
block-diagonal matrices interleaved with a fixed reshape/permute, introduced
by Dao et al. (2022). It recovers a large fraction of the expressivity of a
dense matrix while reducing parameter count and FLOPs from ``O(n^2)`` to
``O(n^1.5)`` (for ``nblocks = sqrt(n)``), and generalizes many fast-transform
structures (FFT, Hadamard, etc.).

Architectural Overview:
The block keeps the familiar "expand-then-contract" FFN shape, but each linear
map is realized as a Monarch map rather than a single dense kernel::

    Input (..., input_dim)
        |
        v   expand: Monarch(input_dim -> hidden_dim)   [+bias]
        |
        v   activation (e.g. GELU)
        |
        v   Dropout (optional)
        |
        v   contract: Monarch(hidden_dim -> output_dim) [+bias]
        |
        v
    Output (..., output_dim)

Foundational Mathematics:
An order-2 Monarch linear map sends a vector of dimension ``n_in`` to a vector
of dimension ``n_out``. Both dimensions are split into ``nblocks`` blocks
(``b_in = n_in / nblocks``, ``b_out = n_out / nblocks``). The map is computed in
five reshape/einsum steps (no dense ``n x n`` kernel is ever materialized):

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
factor is rectangular per block (``b_in -> b_out``); this is the natural,
minimal generalization that keeps the structure intact without falling back to
a dense projection. The only requirement is that ``nblocks`` divides **all** of
``input_dim``, ``hidden_dim`` and ``output_dim`` so that the block grids line up
(validated in ``__init__`` / ``build``).

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

    Replaces each of the two dense projections of a standard FFN with an order-2
    Monarch map (product of two block-diagonal factors interleaved with a
    reshape/permute). The computation is
    ``FFN(x) = monarch_contract(dropout(activation(monarch_expand(x))))``, with
    optional bias terms after each Monarch map. See the module docstring for the
    five-step Monarch math and references (Dao et al. 2022).

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────┐
        │   Input (..., input_dim)    │
        └──────────────┬──────────────┘
                       ▼
        ┌─────────────────────────────┐
        │ expand: Monarch -> hidden_dim│  (L_e, R_e)  [+ bias_e]
        └──────────────┬──────────────┘
                       ▼
        ┌─────────────────────────────┐
        │   Activation (e.g. GELU)    │
        └──────────────┬──────────────┘
                       ▼
        ┌─────────────────────────────┐
        │     Dropout (optional)      │
        └──────────────┬──────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ contract: Monarch -> out_dim │  (L_c, R_c)  [+ bias_c]
        └──────────────┬───────────────┘
                       ▼
        ┌─────────────────────────────┐
        │  Output (..., output_dim)   │
        └─────────────────────────────┘

    :param hidden_dim: Integer, intermediate (expansion) dimension. Must be
        positive and divisible by ``nblocks``.
    :type hidden_dim: int
    :param output_dim: Integer, output dimension. Must be positive and divisible
        by ``nblocks``.
    :type output_dim: int
    :param nblocks: Integer, number of Monarch blocks (the structure knob). The
        per-block size is ``dim / nblocks``. Must be a positive integer that
        divides ``input_dim``, ``hidden_dim`` and ``output_dim``. Defaults to 4.
    :type nblocks: int
    :param activation: Activation function name or callable applied after the
        expand Monarch map. Defaults to 'gelu'.
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
    :param kernel_regularizer: Optional regularizer for the Monarch factor weights.
        Defaults to None.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for the bias vectors.
        Defaults to None.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If hidden_dim or output_dim is not a positive integer.
    :raises ValueError: If nblocks is not a positive integer.
    :raises ValueError: If hidden_dim or output_dim is not divisible by nblocks.
    :raises ValueError: If dropout_rate is not in ``[0.0, 1.0)``.
    :raises ValueError: (in ``build``) If input_dim is not divisible by nblocks.

    Note:
        For ``nblocks = 1`` the Monarch map degenerates to a dense matrix. With
        ``nblocks = sqrt(dim)`` the parameter count is ``O(dim^1.5)`` rather than
        ``O(dim^2)``. The structure is preserved exactly for square maps and via
        rectangular per-block factors for non-square maps.
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
        """Initialize the Monarch FFN layer with comprehensive parameter validation."""
        super().__init__(**kwargs)

        # Comprehensive input validation with informative error messages
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

        # Store ALL configuration parameters for serialization
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

        # CREATE all sub-layers in __init__ (Modern Keras 3 pattern).
        # Monarch factor weights are created in build() (they need input_dim).
        self.dropout = keras.layers.Dropout(rate=self.dropout_rate, name="dropout")

        # Weight handles created in build()
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

        # DECISION plan_2026-06-19_2ea7a9a0/D-003
        # Non-square order-2 Monarch sizing: the expand map (input_dim -> hidden_dim)
        # and contract map (hidden_dim -> output_dim) are generally non-square. We keep
        # the structure exact by making the FIRST block-diagonal factor L rectangular
        # PER BLOCK (b_in -> b_out, shape (nblocks, b_in, b_out)) and the SECOND factor R
        # square in the block axis (shape (b_out, nblocks, nblocks)). Do NOT "fix" a
        # non-square map by appending a trailing keras.layers.Dense projection — that
        # injects an unstructured O(n^2) dense kernel and silently destroys the Monarch
        # structure this layer exists to provide (PM1 / LESSONS: fix-forward-to-non-crash
        # != correct). The unavoidable consequence of this block-grid alignment is that
        # nblocks MUST divide input_dim, hidden_dim AND output_dim; that is enforced for
        # hidden_dim/output_dim in __init__ and for input_dim here. See decisions.md D-003.
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

        Graph-safe: the last axis is split into ``(nblocks, b_in)`` / merged back
        via reshape, and the dynamic leading (batch/sequence) dims are carried
        through by stacking the per-axis sizes from ``keras.ops.shape`` into an
        explicit 1-D integer target shape (no ambiguous ``-1`` placeholder, no
        tuple/tensor concatenation pitfalls). All block sizes are static ints from
        config. Only ``keras.ops`` reshape/transpose/einsum are used.

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
