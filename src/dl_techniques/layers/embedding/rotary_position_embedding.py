"""
Rotary Position Embedding (RoPE) for transformer attention.

RoPE injects position by ROTATING pairs of channels inside the query and key
vectors. It adds nothing to the token embedding. The rotation angle depends on
the token's absolute position, so the dot product of a rotated query and a
rotated key depends only on the distance between the two tokens.

Architecture:
    The head dimension is read as pairs of ADJACENT channels. Pair ``i`` is
    ``(x[2i], x[2i+1])``, treated as one 2D vector. Each pair is rotated by
    ``m * theta_i``, where ``m`` is the token position and ``theta_i`` is that
    pair's own frequency. Different pairs rotate at different rates, so one
    token carries a multi-scale positional signal.

    Only the leading ``rope_dim`` channels are rotated. ``rope_percentage``
    sets that fraction and the rest pass through untouched. The cos and sin
    values for every position up to ``max_seq_len`` are precomputed into two
    non-trainable tables.

Foundational Mathematics:
    Read a ``d``-dimensional vector as ``d/2`` complex numbers. The transform
    at position ``m`` multiplies each one by a complex exponential::

        f(x, m)_i = x_i * e^(j * m * theta_i)

    The inner product of a query at ``m`` and a key at ``n`` is then::

        <f(q, m), f(k, n)>
            = Re( sum_i q_i * conj(k_i) * e^(j * (m - n) * theta_i) )

    It depends on ``m - n``, not on ``m`` or ``n`` alone. That relative
    property is the reason RoPE exists. The frequencies form a geometric
    ladder::

        theta_i = 1 / (rope_theta^(2i / d))

    Low ``i`` rotates fast and resolves nearby tokens. High ``i`` rotates
    slowly and carries long-range position. The code uses the real-valued
    form of the same complex multiply: a 2x2 rotation applied to each
    channel pair.

References:
    - Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
      "RoFormer: Enhanced Transformer with Rotary Position Embedding".
"""

import keras
from typing import Optional, Any, Tuple, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RotaryPositionEmbedding(keras.layers.Layer):
    """Rotate adjacent channel pairs of ``q``/``k`` by a position angle.

    Takes a post-head-split tensor of shape
    ``(batch, heads, seq_len, head_dim)`` and rotates the leading
    ``rope_dim`` channels. Pair ``i`` is the ADJACENT pair
    ``(x[2i], x[2i+1])``, rotated by ``m * theta_i`` at position ``m`` with
    ``theta_i = 1 / (rope_theta^(2i/d))``. The remaining channels are copied
    through. Shape is unchanged. The layer learns nothing; both tables are
    non-trainable and precomputed in :meth:`build`.

    The pairing is INTERLEAVED, not split-half. A split-half implementation
    rotates ``(x[i], x[i + d/2])`` instead. Both conventions are valid
    rotations and both give the relative-position property, so a swap will
    train fine and load an externally converted checkpoint wrong.

    **Architecture Overview:**

    .. code-block:: text

        input x  (B, heads, seq_len, head_dim)
                               │
                   split on the channel axis
             ┌─────────────────┴──────────────────┐
             ▼                                    ▼
             x_rope                               x_pass
             [..., :rope_dim]                     [..., rope_dim:]
             │                                    │
             reshape to ADJACENT pairs            │
             (..., rope_dim/2, 2), so pair i      │
             is (x1, x2) = (x[2i], x[2i+1])       │
             │                                    │
             angle = m * theta_i at position m;   │
             cos and sin read from the tables     │
             │                                    │
             out[2i]   = x1*cos - x2*sin          │
             out[2i+1] = x1*sin + x2*cos          │
             │                                    │
             reshape back to (..., rope_dim)      │
             └─────────────────┬──────────────────┘
                               ▼
              concatenate on the channel axis, ONLY
              when rope_dim < head_dim. At
              rope_percentage=1.0 the two are equal,
              x_pass is empty, and the rotated
              tensor is returned as it is.
                               │
                               ▼
        output  (B, heads, seq_len, head_dim)

    :param head_dim: Dimensionality of each attention head. Must be positive.
        An odd value logs a warning; RoPE needs pairs.
    :type head_dim: int
    :param max_seq_len: Largest position the tables cover. Must be positive.
        A longer input raises at call time.
    :type max_seq_len: int
    :param rope_theta: Base of the frequency ladder. Larger values stretch the
        wavelengths and suit longer sequences. Defaults to ``10000.0``.
    :type rope_theta: float
    :param rope_percentage: Fraction of ``head_dim`` that is rotated. Defaults
        to ``0.5``. Must be in ``(0, 1]``. The derived ``rope_dim`` is rounded
        DOWN to an even number, so a small ``head_dim`` can round it to 0, in
        which case :meth:`call` returns the input unchanged.
    :type rope_percentage: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar rope_dim: Number of leading channels actually rotated. Even, and at
        most ``head_dim``. Derived in ``__init__``, not a config key.
    :vartype rope_dim: int
    :ivar cos_cached: Non-trainable table of shape
        ``(max_seq_len, rope_dim // 2)``. ``None`` until ``build()`` runs.
    :vartype cos_cached: keras.Variable or None
    :ivar sin_cached: Non-trainable table of the same shape as ``cos_cached``.
    :vartype sin_cached: keras.Variable or None

    Input shape:
        4D tensor with shape ``(batch_size, num_heads, seq_len, head_dim)``.

    Output shape:
        4D tensor with the same shape as the input.

    :raises ValueError: If ``head_dim``, ``max_seq_len`` or ``rope_theta`` is
        not positive, or if ``rope_percentage`` is outside ``(0, 1]``. Raised
        from ``__init__``.
    :raises ValueError: If the input is not 4D, or if its last dimension is
        not ``head_dim``. Raised from ``build()``.
    :raises ValueError: If the static sequence length exceeds ``max_seq_len``.
        Raised from ``call()``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.embedding import (
            create_embedding_layer,
        )

        rope = create_embedding_layer(
            "rope", head_dim=64, max_seq_len=128,
        )
        q = keras.random.normal((2, 8, 16, 64))
        rope(q).shape  # (2, 8, 16, 64)
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        rope_percentage: float = 0.5,
        **kwargs: Any
    ) -> None:
        """Validate the configuration and derive ``rope_dim``.

        No weight is created here; the tables are built in :meth:`build`.

        :param head_dim: Dimensionality of each attention head.
        :type head_dim: int
        :param max_seq_len: Largest position the tables will cover.
        :type max_seq_len: int
        :param rope_theta: Base of the frequency ladder.
        :type rope_theta: float
        :param rope_percentage: Fraction of ``head_dim`` to rotate.
        :type rope_percentage: float
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        :raises ValueError: If any of the first three arguments is not
            positive, or if ``rope_percentage`` is outside ``(0, 1]``.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {rope_theta}")
        if not 0.0 < rope_percentage <= 1.0:
            raise ValueError(f"rope_percentage must be in (0, 1], got {rope_percentage}")

        if head_dim % 2 != 0:
            logger.warning(f"head_dim ({head_dim}) is odd, RoPE works best with even dimensions")

        # Store ALL configuration parameters
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.rope_percentage = rope_percentage

        # Calculate RoPE dimensions (ensure even for proper complex pairs)
        self.rope_dim = int(head_dim * rope_percentage)
        if self.rope_dim % 2 != 0:
            self.rope_dim -= 1
            logger.info(f"Adjusted rope_dim to {self.rope_dim} to ensure even dimension")

        # Initialize weight attributes - created in build()
        self.cos_cached = None
        self.sin_cached = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the cos/sin lookup tables for RoPE computation.

        :param input_shape: Expected shape
            ``(batch_size, num_heads, seq_len, head_dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If input shape is not 4D or ``head_dim`` does not
            match.
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch, heads, seq_len, head_dim), "
                f"got shape with {len(input_shape)} dimensions: {input_shape}"
            )

        input_head_dim = input_shape[-1]
        if input_head_dim != self.head_dim:
            raise ValueError(
                f"Input head_dim ({input_head_dim}) doesn't match "
                f"layer head_dim ({self.head_dim})"
            )

        # Create layer's own weights - cos/sin cache tables
        self._create_rope_cache()

        # Always call parent build at the end
        super().build(input_shape)

    def _create_rope_cache(self) -> None:
        """Create the cos and sin lookup tables.

        Both tables have shape ``(max_seq_len, rope_dim // 2)`` and are
        non-trainable. They are built at ``self.variable_dtype``, so a
        ``float64`` policy gets float64 tables while ``float32`` and
        ``mixed_float16`` are byte-unchanged.

        When ``rope_dim`` rounds down to 0 the method logs a warning and
        creates two zero-filled ``(max_seq_len, 1)`` placeholders so the
        weight set stays the same shape across configurations. :meth:`call`
        returns early in that case and never reads them.

        :return: Nothing. Sets ``cos_cached`` and ``sin_cached``.
        :rtype: None
        """
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-016
        # Build the tables at `variable_dtype`. Do NOT restore `dtype='float32'`:
        # under a float64 policy `x1 * cos` raised `InvalidArgumentError: cannot
        # compute Mul ... expected to be a double tensor but is a float tensor` on
        # the first forward pass (measured 2026-07-28, TF 2.18 / CUDA), taking
        # GatedAttention, GroupedQueryAttention and MultiHeadLatentAttention with
        # it. Do NOT use `compute_dtype`: under mixed_float16 that stores the
        # tables in float16 and the high-frequency entries of a long table lose
        # most of their precision. See decisions.md D-016.
        cache_dtype = self.variable_dtype

        # Calculate frequency dimension (half of rope_dim for complex pairs)
        freq_dim = self.rope_dim // 2

        if freq_dim == 0:
            logger.warning("rope_dim is too small, no rotary embedding will be applied")
            # Create dummy weights to maintain consistency
            self.cos_cached = self.add_weight(
                name='cos_cached',
                shape=(self.max_seq_len, 1),
                initializer='zeros',
                trainable=False,
                dtype=cache_dtype
            )
            self.sin_cached = self.add_weight(
                name='sin_cached',
                shape=(self.max_seq_len, 1),
                initializer='zeros',
                trainable=False,
                dtype=cache_dtype
            )
            return

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-021
        # An INITIALIZER fills the tables. Do NOT go back to
        # `add_weight(initializer='zeros')` then `.assign(table)`: Keras 3 builds
        # a sublayer first reached from a parent's `call()` inside a
        # `StatelessScope`, which records the `.assign()` and then discards it,
        # so cos and sin stay 0 at every position and the rotated slice of q and
        # k is zeroed. Measured 2026-08-15, CPU: a direct `.build(...)` gives
        # `cos[0] == 1.0`, through a parent's `call()` it was `0.0`. Compute the
        # table INSIDE the initializer; a tensor built out here belongs to the
        # symbolic pass's FuncGraph and raises "out of scope". See decisions.md
        # D-021.
        #
        # DECISION plan-2026-08-17T183311-79c63e38/D-044
        # The fix does not repair an OLD checkpoint, and the failure is silent.
        # These are non-trainable weights, so Keras serializes them; a `.keras`
        # file saved before 2026-08-15 carries the all-zero tables and loading it
        # overwrites the initializer's output with no error and no log. Re-train,
        # or check `cos_cached[0]` is 1.0 after loading. Do NOT add an
        # `.assign()` repair on load. See decisions.md D-044.
        def _table_initializer(trig):
            """Make an initializer that fills a table with ``trig``.

            :param trig: ``keras.ops.cos`` or ``keras.ops.sin``.
            :type trig: Callable
            :return: A Keras initializer callable.
            :rtype: Callable
            """

            def initializer(shape, dtype=None):
                """Compute the whole table at variable-creation time.

                :param shape: ``(max_seq_len, freq_dim)``.
                :type shape: Tuple[int, int]
                :param dtype: Requested dtype, or ``None`` to use
                    ``cache_dtype``.
                :type dtype: Optional[str]
                :return: The filled table.
                :rtype: keras.KerasTensor
                """
                table_dtype = dtype or cache_dtype
                # 1 / (theta ^ (2i / rope_dim)) for i in [0, freq_dim)
                inv_freq = 1.0 / (
                    self.rope_theta ** (
                        keras.ops.arange(0, shape[1], dtype=table_dtype)
                        * 2.0 / self.rope_dim
                    )
                )
                positions = keras.ops.arange(shape[0], dtype=table_dtype)
                return trig(keras.ops.outer(positions, inv_freq))

            return initializer

        table_shape = (self.max_seq_len, freq_dim)
        self.cos_cached = self.add_weight(
            name='cos_cached',
            shape=table_shape,
            initializer=_table_initializer(keras.ops.cos),
            trainable=False,
            dtype=cache_dtype
        )
        self.sin_cached = self.add_weight(
            name='sin_cached',
            shape=table_shape,
            initializer=_table_initializer(keras.ops.sin),
            trainable=False,
            dtype=cache_dtype
        )

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply rotary position embedding to the input tensor.

        :param inputs: Input tensor with shape
            ``(batch_size, num_heads, seq_len, head_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode (unused by this layer).
        :type training: Optional[bool]
        :return: Output tensor with same shape, RoPE applied to the first
            ``rope_dim`` dimensions.
        :rtype: keras.KerasTensor
        :raises ValueError: If input sequence length exceeds ``max_seq_len``.
        """
        # Get sequence length from input
        seq_len = keras.ops.shape(inputs)[2]

        # Early return if no RoPE dimensions
        if self.rope_dim == 0:
            return inputs

        # Validate sequence length at build time if known
        seq_len_static = inputs.shape[2]
        if seq_len_static is not None and seq_len_static > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len_static}) exceeds max_seq_len ({self.max_seq_len}). "
                f"Please increase max_seq_len or truncate the input."
            )

        return self._apply_rope_rotation(inputs, seq_len)

    def _apply_rope_rotation(
        self,
        x: keras.KerasTensor,
        seq_len: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Apply rotary position embedding transformation.

        :param x: Input tensor with shape
            ``(batch_size, num_heads, seq_len, head_dim)``.
        :type x: keras.KerasTensor
        :param seq_len: Current sequence length tensor.
        :type seq_len: keras.KerasTensor
        :return: Tensor with RoPE applied to the first ``rope_dim`` dimensions.
        :rtype: keras.KerasTensor
        """
        # Split the channel axis: the leading rope_dim channels are rotated,
        # the rest are copied through unchanged.
        x_rope = x[..., :self.rope_dim]
        x_pass = x[..., self.rope_dim:]

        # Read the cos/sin rows for the current sequence length, each of shape
        # (seq_len, rope_dim // 2), and cast the TABLE to the input dtype.
        # Casting this direction makes the multiply below dtype-safe in the
        # code, instead of relying on Keras variable autocast to fire.
        # Under mixed_float16 autocast has already produced float16 and the cast
        # is an identity; where it does not fire, `x1 * cos` would otherwise
        # raise `InvalidArgumentError: cannot compute Mul ...`. Never cast the
        # input to the table.
        cos = keras.ops.cast(self.cos_cached[:seq_len], x.dtype)
        sin = keras.ops.cast(self.sin_cached[:seq_len], x.dtype)

        # Reshape x_rope to separate complex pairs
        # From: (batch, heads, seq_len, rope_dim)
        # To: (batch, heads, seq_len, rope_dim // 2, 2)
        rope_pairs = self.rope_dim // 2

        # Get dynamic shape and construct new shape for reshaping
        input_shape = keras.ops.shape(x_rope)
        batch_size = input_shape[0]
        num_heads = input_shape[1]
        seq_len_dynamic = input_shape[2]

        # Reshape to expose complex pairs
        new_shape = [batch_size, num_heads, seq_len_dynamic, rope_pairs, 2]
        x_rope_reshaped = keras.ops.reshape(x_rope, new_shape)

        # Take the two members of each adjacent pair. x1 is channel 2i, the
        # real-like part; x2 is channel 2i+1, the imaginary-like part.
        x1 = x_rope_reshaped[..., 0]
        x2 = x_rope_reshaped[..., 1]

        # Apply rotary transformation:
        # [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated_1 = x1 * cos - x2 * sin
        rotated_2 = x1 * sin + x2 * cos

        # Stack the rotated components back together
        x_rope_rotated = keras.ops.stack([rotated_1, rotated_2], axis=-1)

        # Reshape back to original rope dimensions
        # From: (batch, heads, seq_len, rope_pairs, 2)
        # To: (batch, heads, seq_len, rope_dim)
        x_rope_rotated = keras.ops.reshape(
            x_rope_rotated,
            [batch_size, num_heads, seq_len_dynamic, self.rope_dim]
        )

        # Concatenate rotated and pass-through dimensions
        if self.rope_dim < self.head_dim:
            return keras.ops.concatenate([x_rope_rotated, x_pass], axis=-1)
        else:
            return x_rope_rotated

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape (identical to input shape).

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (same as input shape).
        :rtype: Tuple[Optional[int], ...]
        """
        # RoPE preserves tensor shape while applying rotational transformations
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters for proper
            serialization.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'head_dim': self.head_dim,
            'max_seq_len': self.max_seq_len,
            'rope_theta': self.rope_theta,
            'rope_percentage': self.rope_percentage,
        })
        return config

# ---------------------------------------------------------------------
