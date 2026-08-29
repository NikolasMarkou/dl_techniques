"""
The Stable Diffusion 3 feed-forward block, with tanh-approximate GELU.

This is the SD3 ``FeedForward`` block ported into the dl_techniques FFN
family. It is the ordinary Transformer expand-then-contract MLP with one
thing pinned: the activation is the *tanh* approximation of GELU, not the
exact erf form. That is the whole difference from ``MLPBlock`` with
``activation='gelu'`` and from ``GeGLUFFN``.

The forward path, applied to each position with shared weights:

1.  ``fc1`` projects from the input width up to ``hidden_dim``.
2.  ``keras.ops.gelu(x, approximate=True)``, which computes
    ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`` instead of
    the exact ``x * Phi(x)``.
3.  Dropout. The layer is always present; at rate 0.0 it is the identity.
4.  ``fc2`` projects from ``hidden_dim`` down to the output width.

The maths, for one token vector ``x``:

    FFN(x) = gelu_tanh(x @ W_1 + b_1) @ W_2 + b_2

``output_dim`` defaults to ``None``, which means "take the input width",
resolved in ``build``. The block is then residual-ready with no extra
argument, matching SD3's ``FeedForward(dim, dim_out=dim)``.

References:
-   Esser, P., et al. (2024). Scaling Rectified Flow Transformers for
    High-Resolution Image Synthesis (Stable Diffusion 3). arXiv:2403.03206.
-   Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs).
    arXiv:1606.08415. (both GELU forms, exact and tanh)
-   Vaswani, A., et al. (2017). Attention Is All You Need. NIPS.

"""

import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.ffn.gelu_mlp_ffn")
class GELUMLPFFN(keras.layers.Layer):
    """
    The Stable Diffusion 3 feed-forward block, with tanh-approximate GELU.

    Two Dense layers with one activation between them:
    ``FFN(x) = gelu_tanh(x @ W_1 + b_1) @ W_2 + b_2``. The activation is
    ``keras.ops.gelu(x, approximate=True)``, the tanh form. That is the only
    behavioural difference from ``MLPBlock`` with ``activation='gelu'``, which
    uses the exact erf form, and from ``GeGLUFFN``, which gates an exact-erf
    GELU.

    ``output_dim`` defaults to ``None``, which is unusual for this package.
    Every other FFN here wants an explicit output width. Here ``None`` means
    "match the input width", resolved at build time, so the block can be
    dropped straight onto a residual stream. This mirrors SD3's
    ``FeedForward(dim, dim_out=dim)``.

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
            │ gelu(approximate=True)  │
            │ the tanh form           │
            └────────────┬────────────┘
                         ▼
            ┌─────────────────────────┐
            │ dropout                 │
            │ (always present)        │
            └────────────┬────────────┘
                         ▼
            ┌─────────────────────────┐
            │ fc2                     │
            │ Dense(output width)     │
            └────────────┬────────────┘
                         ▼
            Output [..., output width]

        `dropout` is NOT conditional here. The Dropout layer is
        always created and always called; at dropout_rate=0.0 it
        is the identity. This differs from MLPBlock, where the
        attribute is None at 0.0.

    **How the output width is resolved:**

    .. code-block:: text

        output_dim = 512 (explicit)
            fc2 built with units=512 in __init__.
            _resolved_output_dim = 512 from the start.

        output_dim = None (the default)
            __init__ builds a PLACEHOLDER fc2 with hidden_dim
            units, purely so the attribute exists.
            build() sets _resolved_output_dim = input_shape[-1]
            and REPLACES fc2 with Dense(that width). No weights
            exist yet, so nothing is lost.

        get_config() stores output_dim, never the resolved
        value. A reloaded layer re-resolves from its own input.

    :param hidden_dim: Width of the expansion, the ``units`` of ``fc1``. Must
        be positive.
    :type hidden_dim: int
    :param output_dim: Width of the output. ``None`` (the default) means take
        the input feature width at build time. Must be positive when given.
    :type output_dim: Optional[int]
    :param dropout_rate: Dropout rate applied after the activation, in
        ``[0.0, 1.0)``. Defaults to 0.0, where the Dropout layer still exists
        and is the identity.
    :type dropout_rate: float
    :param use_bias: Whether ``fc1`` and ``fc2`` carry a bias. Defaults to
        True.
    :type use_bias: bool
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on).
    :type kwargs: Any

    :ivar hidden_dim: The stored expansion width.
    :vartype hidden_dim: int
    :ivar output_dim: The output width as REQUESTED, possibly ``None``. This
        is what ``get_config()`` stores.
    :vartype output_dim: Optional[int]
    :ivar dropout_rate: The stored dropout rate.
    :vartype dropout_rate: float
    :ivar use_bias: Whether the projections carry a bias.
    :vartype use_bias: bool
    :ivar _resolved_output_dim: The output width actually used by ``fc2``.
        Equals ``output_dim`` when that was given, otherwise the input
        feature width filled in by ``build()``.
    :vartype _resolved_output_dim: Optional[int]
    :ivar fc1: ``Dense(hidden_dim)``, the expansion.
    :vartype fc1: keras.layers.Dense
    :ivar fc2: ``Dense`` at the resolved output width, the contraction.
    :vartype fc2: keras.layers.Dense
    :ivar dropout: ``Dropout(dropout_rate)``. Always present.
    :vartype dropout: keras.layers.Dropout

    :raises ValueError: If ``hidden_dim`` is not positive.
    :raises ValueError: If ``output_dim`` is given and not positive.
    :raises ValueError: If ``dropout_rate`` is outside ``[0.0, 1.0)``.
    :raises ValueError: At build time, if the last input axis is ``None``.

    Input shape:
        Tensor of rank >= 2, shape ``(..., input_dim)``. The last axis must
        be statically known.

    Output shape:
        Same rank and leading axes as the input. The last axis is
        ``output_dim``, or ``input_dim`` when ``output_dim`` is ``None``.

    Example:
        .. code-block:: python

            ffn = GELUMLPFFN(hidden_dim=2048)
            y = ffn(keras.random.normal((2, 10, 512)))
            y.shape                 # (2, 10, 512)

            ffn = GELUMLPFFN(hidden_dim=2048, output_dim=256)
            y = ffn(keras.random.normal((2, 10, 512)))
            y.shape                 # (2, 10, 256)

    Note:
        Sub-layers are created in ``__init__`` and built explicitly in
        ``build()``. Keras does not build them on its own here, because
        ``fc2`` sees ``hidden_dim`` rather than the input width.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and create the sub-layers.

        Every argument is documented on the class. Validation runs before any
        attribute is stored, so a rejected configuration leaves no half-built
        layer behind.

        When ``output_dim`` is ``None``, ``fc2`` is still created here, with
        ``hidden_dim`` units as a placeholder, so the attribute is always a
        ``Dense``. ``build()`` throws that placeholder away and makes a new
        one at the resolved width, before any weight exists.

        :param hidden_dim: Expansion width. Must be positive.
        :type hidden_dim: int
        :param output_dim: Output width, or ``None`` to take the input width.
            Must be positive when given.
        :type output_dim: Optional[int]
        :param dropout_rate: Dropout rate applied after the activation. Must
            be in ``[0.0, 1.0)``.
        :type dropout_rate: float
        :param use_bias: Whether ``fc1`` and ``fc2`` carry a bias.
        :type use_bias: bool
        :param kwargs: Extra arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If ``hidden_dim`` is not positive, if
            ``output_dim`` is given and not positive, or if ``dropout_rate``
            is outside ``[0.0, 1.0)``.
        """
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
        Resolve the output width, then build the sub-layers.

        When ``output_dim`` was left as ``None``, it is set here to the input
        feature width and ``fc2`` is replaced by a new ``Dense`` with that
        many units. The replacement is safe because no weights exist yet: the
        early ``self.built`` return means this runs at most once.

        Sub-layers are then built in the order ``call()`` uses them, so every
        weight exists before a save or a restore.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last input axis is ``None``. The width is
            needed to size ``fc1``, and to size ``fc2`` when ``output_dim``
            is ``None``.
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
        Run the block: ``fc1`` -> GELU-tanh -> dropout -> ``fc2``.

        The activation is ``keras.ops.gelu(x, approximate=True)``, the tanh
        form. It is NOT the exact-erf GELU that ``MLPBlock`` uses.

        :param inputs: Input tensor of shape ``(..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether to run in training mode. Only affects
            dropout.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(..., output_dim)``.
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
        Compute the output shape.

        Once ``build`` has run, the last axis becomes the resolved output
        width. Before ``build``, and only when ``output_dim`` was left as
        ``None``, the last axis is passed through unchanged. That is the
        right answer for the residual-ready default, where the output width
        equals the input width.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The input shape with the last axis set to the output width,
            or unchanged when the width is not resolved yet.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        if self._resolved_output_dim is not None:
            output_shape[-1] = self._resolved_output_dim
        # else: output_dim is None and layer not yet built -> preserve last dim.
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the config needed to rebuild this layer.

        ``output_dim`` is stored exactly as it was passed in, so a ``None``
        stays ``None``. A reloaded layer therefore re-resolves it from its
        own input shape instead of being pinned to whatever the original
        input happened to be.

        :return: The complete layer configuration.
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
