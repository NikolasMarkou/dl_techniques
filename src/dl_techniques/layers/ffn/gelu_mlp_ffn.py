"""
A GELU (tanh-approximation) position-wise Feed-Forward Network.

This layer ports the Stable Diffusion 3 (SD3) ``FeedForward`` block into the
dl_techniques FFN family. It is the standard Transformer "expand-then-contract"
MLP, but pinned to the *tanh* (approximate) form of the Gaussian Error Linear
Unit activation, which is the SD3-faithful choice and the single behavioral
difference from the repo's exact-erf primitives (``MLPBlock`` with ``gelu`` and
``GeGLUFFN``).

Architectural Overview:
The network applies, position-wise and with shared weights across the sequence:

1.  **Expansion** ``fc1``: a linear projection from ``input_dim`` to
    ``hidden_dim``.
2.  **GELU-tanh activation**: ``keras.ops.gelu(x, approximate=True)`` — the
    tanh approximation
    ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`` rather than the
    exact ``x * Phi(x)`` erf form. This is the SD3 convention.
3.  **Dropout** (optional, identity when ``dropout_rate == 0``).
4.  **Contraction** ``fc2``: a linear projection from ``hidden_dim`` to
    ``output_dim``.

Foundational Mathematics:
For an input vector ``x`` at a single sequence position::

    FFN(x) = W_2 @ gelu_tanh(W_1 @ x + b_1) + b_2

where ``gelu_tanh`` is the tanh-approximate GELU. ``output_dim`` defaults to the
input feature dimension (resolved in ``build``) so the block is residual-ready,
matching the SD3 ``FeedForward(dim, dim_out=dim)`` default.

References:
-   Esser, P., et al. (2024). Scaling Rectified Flow Transformers for
    High-Resolution Image Synthesis (Stable Diffusion 3). arXiv:2403.03206.
-   Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs).
    arXiv:1606.08415.
-   Vaswani, A., et al. (2017). Attention Is All You Need. NIPS.

"""

import keras
from keras import ops
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class GELUMLPFFN(keras.layers.Layer):
    """
    GELU (tanh-approximation) Feed-Forward Network (SD3 ``FeedForward``).

    Implements ``Dense(hidden_dim) -> gelu(approximate=True) -> Dropout ->
    Dense(output_dim)``. The activation is the *tanh* approximation of GELU
    (``keras.ops.gelu(x, approximate=True)``), which is the Stable Diffusion 3
    convention and the only behavioral difference from ``MLPBlock`` (exact-erf
    GELU) and ``GeGLUFFN`` (gated exact-erf GELU).

    When ``output_dim`` is ``None`` (the default), the output dimension is
    resolved at ``build`` time to the input feature dimension, so the block is
    residual-ready out of the box, matching SD3's ``FeedForward(dim,
    dim_out=dim)`` default.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────┐
        │   Input (..., input_dim)     │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │   fc1: Dense(hidden_dim)     │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  GELU (approximate=tanh)     │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │     Dropout (optional)       │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │   fc2: Dense(output_dim)     │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  Output (..., output_dim)    │
        └──────────────────────────────┘

    :param hidden_dim: Integer, intermediate (expansion) dimension. Must be
        positive.
    :type hidden_dim: int
    :param output_dim: Integer or None, final output dimension. When ``None``,
        resolved to the input feature dimension in ``build``. Must be positive
        when provided. Defaults to ``None``.
    :type output_dim: Optional[int]
    :param dropout_rate: Float in [0.0, 1.0), dropout applied after the GELU
        activation. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether the dense layers use bias vectors. Defaults to
        True.
    :type use_bias: bool
    :param kwargs: Additional keyword arguments for the ``keras.layers.Layer``
        base class.

    :raises ValueError: If ``hidden_dim`` is not positive.
    :raises ValueError: If ``output_dim`` is provided and not positive.
    :raises ValueError: If ``dropout_rate`` is not in [0.0, 1.0).

    Note:
        Follows modern Keras 3 patterns: all sub-layers are created in
        ``__init__`` and explicitly built in ``build`` for robust
        serialization. ``fc2`` is created in ``__init__`` with a placeholder
        unit count when ``output_dim`` is ``None`` and re-instantiated with the
        resolved dimension in ``build`` before any weights are created.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs immediately
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim is not None and output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")

        # Store ALL configuration parameters
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias

        # Resolved at build() when output_dim is None.
        self._resolved_output_dim: Optional[int] = output_dim

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern).
        self.fc1 = keras.layers.Dense(
            units=self.hidden_dim,
            use_bias=self.use_bias,
            name="fc1",
        )
        # When output_dim is None, fc2 is re-instantiated in build() with the
        # resolved unit count (before weights exist). The placeholder uses
        # hidden_dim purely so the attribute is always a Dense instance.
        self.fc2 = keras.layers.Dense(
            units=self.output_dim if self.output_dim is not None else self.hidden_dim,
            use_bias=self.use_bias,
            name="fc2",
        )
        self.dropout = keras.layers.Dropout(rate=self.dropout_rate, name="dropout")

        logger.info(
            f"Initialized GELUMLPFFN with hidden_dim={hidden_dim}, "
            f"output_dim={output_dim}, dropout_rate={dropout_rate}, "
            f"use_bias={use_bias}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        Resolves ``output_dim`` to the input feature dimension when it was left
        as ``None``, then explicitly builds each sub-layer in computational
        order for robust serialization.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last input dimension is undefined.
        """
        if self.built:
            return

        if input_shape[-1] is None:
            raise ValueError("Last dimension of input must be defined")

        # Resolve output_dim to input feature dim if not explicitly provided.
        if self._resolved_output_dim is None:
            self._resolved_output_dim = int(input_shape[-1])
            # Re-instantiate fc2 with the resolved unit count (no weights yet).
            self.fc2 = keras.layers.Dense(
                units=self._resolved_output_dim,
                use_bias=self.use_bias,
                name="fc2",
            )

        # Build sub-layers in computational order.
        self.fc1.build(input_shape)

        intermediate_shape = (*input_shape[:-1], self.hidden_dim)
        self.dropout.build(intermediate_shape)
        self.fc2.build(intermediate_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Apply the GELU-tanh FFN to the input.

        :param inputs: Input tensor of shape (..., input_dim).
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode (affects dropout).
        :type training: Optional[bool]
        :return: Output tensor of shape (..., output_dim).
        :rtype: keras.KerasTensor
        """
        x = self.fc1(inputs)
        # SD3-faithful: tanh approximation of GELU (NOT the exact-erf form).
        x = ops.gelu(x, approximate=True)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        return x

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Before ``build``, when ``output_dim`` was left as ``None``, the input
        feature dimension is preserved (the SD3 residual-ready default).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (last dim replaced by the output dim, or
            preserved when not yet resolved and unconfigured).
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        if self._resolved_output_dim is not None:
            output_shape[-1] = self._resolved_output_dim
        # else: output_dim is None and layer not yet built -> preserve last dim.
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer's configuration for serialization.

        Serializes the *original* ``output_dim`` (which may be ``None``) so a
        reloaded layer re-resolves it identically on build.

        :return: Dictionary containing the complete layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
        })
        return config

# ---------------------------------------------------------------------
