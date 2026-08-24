"""
Applies rotary embeddings to inject relative positional information.

This layer implements Rotary Position Embedding (RoPE), a method for
encoding the relative positions of tokens in a sequence. Unlike traditional
additive positional embeddings, RoPE applies a rotational transformation
directly to the query and key vectors within the attention mechanism. This
approach has been shown to naturally incorporate relative positional
information and improve sequence length extrapolation.

Architecture:
    The core architectural insight of RoPE is to avoid altering the content
    of token embeddings with positional data. Instead, it modifies the
    attention calculation itself. The feature dimensions of the input query
    or key vectors are conceptually grouped into pairs. Each pair is treated
    as a 2D vector (or the real and imaginary components of a complex
    number) that is then rotated in-place.

    The angle of rotation for each pair is a function of two things: the
    token's absolute position in the sequence and the feature pair's index.
    This means that for a given token, different feature pairs are rotated
    by different angles, creating a rich positional signal.

    To maintain model stability, this rotation is often applied to only a
    fraction of the feature dimensions (controlled by `rope_percentage`),
    leaving the remaining dimensions unchanged. For computational efficiency,
    the sine and cosine values required for these rotations are pre-computed
    for all positions up to `max_seq_len` and stored in non-trainable
    lookup tables.

Foundational Mathematics:
    The primary objective of RoPE is to ensure that the dot product between a
    query vector `q` at position `m` and a key vector `k` at position `n`
    depends only on their relative displacement `m-n`. RoPE achieves this
    by defining a transformation `f(x, p)` that rotates a vector `x` based
    on its absolute position `p`.

    The transformation is elegantly defined using complex numbers. For a
    `d`-dimensional vector, we view it as `d/2` complex numbers. The
    transformation for a vector `x` at position `m` is equivalent to an
    element-wise multiplication with a complex exponential:

        f(x, m)_i = x_i * e^(j * m * θ_i)

    The dot product of two transformed vectors `f(q, m)` and `f(k, n)` then
    satisfies the desired relative property:

        <f(q, m), f(k, n)> = Re( Σ_i (q_i * e^(j*m*θ_i)) * (k_i * e^(-j*n*θ_i)) )
                          = Re( Σ_i (q_i * k_i*) * e^(j*(m-n)θ_i) )

    This shows the inner product is a function of the original vectors and
    their relative position `m-n`. The frequencies `θ_i` are fixed and form
    a geometric progression:

        θ_i = 1 / (rope_theta^(2i / d))

    This provides a multi-scale representation of position, where different
    frequencies capture positional relationships over different distances.
    The implementation uses the real-valued equivalent of this complex
    multiplication, which is a standard 2D rotation matrix applied to each
    pair of features.

References:
    - The original concept was introduced in:
      Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
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
    """Rotary Position Embedding (RoPE) layer for transformer attention.

    Applies rotary transformations to query and key vectors, encoding relative
    positional information through trigonometric rotation of feature-dimension
    pairs. For each pair ``[x_{2i}, x_{2i+1}]`` at position ``m``, RoPE
    applies a 2D rotation by angle ``m * theta_i`` where
    ``theta_i = 1 / (rope_theta^(2i/d))``. The resulting dot product
    ``<f(q,m), f(k,n)>`` depends only on the relative displacement ``m - n``,
    enabling natural relative position encoding without learned parameters.
    Partial application (controlled by ``rope_percentage``) leaves remaining
    dimensions unchanged for stability.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │  Input (batch, heads, seq_len, head_dim) │
        └───────────────────┬──────────────────────┘
                            ▼
        ┌───────────────────┬──────────────────────┐
        │  x_rope           │  x_pass              │
        │  [:rope_dim]      │  [rope_dim:]         │
        └────────┬──────────┘──────────┬───────────┘
                 ▼                     │
        ┌────────────────────┐         │
        │  Reshape to pairs  │         │
        │  (rope_dim//2, 2)  │         │
        └────────┬───────────┘         │
                 ▼                     │
        ┌────────────────────┐         │
        │  Rotate each pair  │         │
        │  [cos -sin] [x1]   │         │
        │  [sin  cos] [x2]   │         │
        └────────┬───────────┘         │
                 ▼                     │
        ┌────────────────────┐         │
        │  Reshape back      │         │
        └────────┬───────────┘         │
                 └──────────┬──────────┘
                            ▼
        ┌──────────────────────────────────────────┐
        │  Concatenate → Output (same shape)       │
        └──────────────────────────────────────────┘

    :param head_dim: Dimensionality of each attention head. Must be positive.
    :type head_dim: int
    :param max_seq_len: Maximum sequence length for precomputing rotary tables.
        Must be positive.
    :type max_seq_len: int
    :param rope_theta: Base frequency for rotary computation. Higher values
        work better for longer sequences. Defaults to ``10000.0``.
    :type rope_theta: float
    :param rope_percentage: Fraction of head dimensions to apply RoPE to.
        Defaults to ``0.5`` for stability. Must be in ``(0, 1]``.
    :type rope_percentage: float
    :param kwargs: Additional Layer base class arguments.

    :raises ValueError: If ``head_dim``, ``max_seq_len``, or ``rope_theta``
        are not positive.
    :raises ValueError: If ``rope_percentage`` is not in the range ``(0, 1]``.
    :raises ValueError: If input sequence length exceeds ``max_seq_len``.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        rope_percentage: float = 0.5,
        **kwargs: Any
    ) -> None:
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
        """Create and cache cos/sin lookup tables for rotary embeddings.

        .. note::
            **FIXED (plan-2026-07-27T183600-b4ef45f0, step 5b).** The tables used to
            be hard-coded to ``dtype='float32'``. Under a ``float64`` global policy
            the layer therefore raised on its very first forward pass, with no mask
            and no unusual input:
            ``InvalidArgumentError: cannot compute Mul as input #1(zero-based) was
            expected to be a double tensor but is a float tensor`` (from
            ``x1 * cos`` in :meth:`_apply_rope_rotation`), taking
            ``GatedAttention``, ``GroupedQueryAttention`` and
            ``MultiHeadLatentAttention`` down with it.

            The tables are now built in ``self.variable_dtype``, which is
            ``float32`` under both the ``float32`` and the ``mixed_float16``
            policies (so those two are byte-unchanged) and ``float64`` under a
            ``float64`` policy. Do NOT use ``compute_dtype`` here: under
            ``mixed_float16`` that would store the tables in ``float16``, where the
            high-frequency entries of a long ``max_seq_len`` table lose most of
            their precision — mixed precision deliberately keeps variables wide and
            autocasts them on read.
        """
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-016
        # Build the tables at the precision this layer stores VARIABLES at, not at
        # the compute dtype and not at a hard-coded 'float32'.
        #
        # WHAT NOT TO DO:
        #   * Do NOT restore `dtype='float32'`. That is the defect: under a float64
        #     policy the rotation `x1 * cos` raised `InvalidArgumentError: cannot
        #     compute Mul ... expected to be a double tensor but is a float tensor`
        #     on the FIRST forward pass, with no mask and nothing unusual, taking
        #     GatedAttention / GroupedQueryAttention / MultiHeadLatentAttention with
        #     it. Measured 2026-07-28, TF 2.18 / CUDA.
        #   * Do NOT use `self.compute_dtype`. Under `mixed_float16` that stores the
        #     tables in float16, where the high-frequency entries of a long
        #     `max_seq_len` table lose most of their precision. Mixed precision
        #     deliberately keeps variables wide and autocasts them on READ.
        # See decisions.md D-016.
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
        # The tables are materialized by an INITIALIZER, not by `add_weight`
        # followed by `.assign()`.
        #
        # WHAT NOT TO DO: do NOT go back to
        #     w = self.add_weight(..., initializer='zeros'); w.assign(table)
        # Keras 3 builds a sublayer that is first reached from a PARENT's
        # `call()` (i.e. every real model: `Model.__call__` runs a symbolic pass
        # first) inside a `StatelessScope`, which RECORDS `.assign()` and then
        # DISCARDS it. The variables therefore stayed at their `'zeros'`
        # initializer and `cos == sin == 0` for every position, which silently
        # ZEROES the rotated `rope_dim` slice of q and k — at the
        # `rope_percentage=1.0` used by `GroupedQueryAttention` that is the whole
        # head, making attention uniform and the block exactly
        # permutation-equivariant. Measured 2026-08-15, CPU: a bare
        # `RotaryPositionEmbedding.build(...)` gave `cos[0] == 1`, while the same
        # layer reached through a parent's `call()` gave `cos[0] == 0`.
        # Initializers are honoured at variable-CREATION time and so survive the
        # stateless scope. The whole table is computed INSIDE the initializer:
        # a tensor built here and closed over would belong to the symbolic
        # build pass's scratch `FuncGraph` and raise "cannot be accessed from
        # here ... out of scope" on the eager pass. See decisions.md D-021.
        #
        # DECISION plan-2026-08-17T183311-79c63e38/D-044
        # THE FIX DOES NOT PROTECT A PRE-FIX CHECKPOINT, and the failure is
        # SILENT. These tables are still `add_weight(..., trainable=False)`, and
        # Keras serializes non-trainable weights. A `.keras` file saved before
        # 2026-08-15 therefore carries the ALL-ZERO tables, and loading it
        # OVERWRITES this initializer's correct output in a load that succeeds
        # completely, logs nothing and raises nothing — reinstating exactly the
        # defect above (the whole rotated head zeroed at `rope_percentage=1.0`).
        # Nothing in the tree detects it. "Invalidated" understates it: such a
        # checkpoint loads FINE and is wrong. Re-train, or verify after loading
        # that `cos_cached[0]` is 1 rather than 0. Do NOT add a `.assign()`-based
        # repair on load — see WHAT NOT TO DO above. See decisions.md D-044.
        def _table_initializer(trig):
            def initializer(shape, dtype=None):
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
        # Split into RoPE and pass-through dimensions
        x_rope = x[..., :self.rope_dim]     # Apply RoPE to these dimensions
        x_pass = x[..., self.rope_dim:]     # Pass these through unchanged

        # Get cached cos/sin values for current sequence length.
        #
        # FIXED (plan-2026-07-27T183600-b4ef45f0, step 5b): the cast is what makes
        # the rotation below dtype-agreement-safe STRUCTURALLY rather than by
        # relying on Keras' variable autocasting to fire. `x` is the caller's
        # tensor; the tables are variables. Under `mixed_float16` autocast already
        # brought them to `float16`, so this cast is an identity; under any policy
        # or caller where it does not fire, `x1 * cos` would raise
        # `InvalidArgumentError: cannot compute Mul ...` instead of silently
        # promoting. Cast the TABLE to the input, never the input to the table.
        cos = keras.ops.cast(self.cos_cached[:seq_len], x.dtype)  # (seq_len, rope_dim // 2)
        sin = keras.ops.cast(self.sin_cached[:seq_len], x.dtype)  # (seq_len, rope_dim // 2)

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

        # Extract real and imaginary components of each complex pair
        x1 = x_rope_reshaped[..., 0]  # Real-like component
        x2 = x_rope_reshaped[..., 1]  # Imaginary-like component

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
