"""
Add a learned absolute position vector to every token in a sequence.

This module provides :class:`PositionalEmbedding`, the learned-table variant
of absolute positional encoding. Self-attention treats its input as an
unordered set, so a transformer stack has no notion of token order until
something injects one. This layer injects it by adding one trainable vector
per position.

Architecture:
    The layer owns a single weight, a table of shape
    ``(1, max_seq_len, dim)``. For an input sequence of length ``L`` the
    first ``L`` rows are sliced and added to the token embeddings. An
    optional dropout follows the addition.

    The table is LEARNED, not computed. The original Transformer used fixed
    sinusoids; BERT and GPT use a learned table, and so does this layer. The
    model is free to discover whatever representation of position its task
    needs, at the cost of a hard ceiling: positions beyond ``max_seq_len``
    have no row and cannot be represented.

Foundational Mathematics:
    Let ``X`` be the input of shape ``(L, D)`` and ``P`` the table of shape
    ``(M, D)``, with ``M = max_seq_len``. The output is::

        Y_i = X_i + P_i    for i = 0, 1, ..., L - 1

    Adding position into the same vector space as token identity is what
    lets attention score a pair on both counts at once. The dot product
    between positions ``i`` and ``j`` expands into terms in ``X_i``, ``X_j``,
    ``P_i`` and ``P_j``, so the model can learn to attend by relative
    offset even though the encoding itself is absolute.

References:
    - Vaswani, A., et al. (2017). "Attention Is All You Need". Introduces
      positional encodings, using a fixed sinusoidal form rather than the
      learned table implemented here.
    - Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional
      Transformers for Language Understanding". Uses the learned absolute
      table this layer implements.
"""

import keras
from keras import ops
from typing import Optional, Dict, Any, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PositionalEmbedding(keras.layers.Layer):
    """Add a learned absolute position vector to each token.

    Owns one table of shape ``(1, max_seq_len, dim)``. For an input of
    length ``L`` the first ``L`` rows are sliced and added element-wise,
    ``Y_i = X_i + P_i``, and a dropout follows. The operation is
    position-wise: the output at step ``i`` depends only on the input at
    step ``i``.

    ``max_seq_len`` is a hard ceiling, and it is NOT checked with a friendly
    error. An input longer than the table reaches the slice and the backend
    raises there, at call time. Size the table for the longest sequence the
    model will ever see.

    ``scale`` resolves the DEFAULT initializer only. It is the standard
    deviation used when ``pos_initializer`` is the bare string
    ``"truncated_normal"`` (the default). A caller-supplied initializer
    INSTANCE is used exactly as given, whatever its type, and ``scale`` is
    then ignored -- so
    ``pos_initializer=keras.initializers.TruncatedNormal(stddev=0.5)``
    keeps its ``0.5``.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────┐
        │  Input X   (batch, seq_len, dim)       │
        └────────────────────────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │  Slice the table to the input length   │
        │  P    (1, max_seq_len, dim)            │
        │  P[:, :seq_len, :]                     │
        │  seq_len > max_seq_len raises here     │
        └────────────────────────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │  Add:  Y = X + P[:, :seq_len, :]       │
        │  broadcast over the batch axis         │
        └────────────────────────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │  Dropout(dropout_rate)                 │
        └────────────────────────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │  Output Y  (batch, seq_len, dim)       │
        └────────────────────────────────────────┘

    :param max_seq_len: Number of rows in the table, and therefore the
        longest sequence this layer can encode. Must be positive.
    :type max_seq_len: int
    :param dim: Width of the table, which must equal the last dimension of
        the input. Must be positive.
    :type dim: int
    :param dropout_rate: Dropout applied after the addition. Must be in
        ``[0, 1]``. Defaults to ``0.0``.
    :type dropout_rate: float
    :param pos_initializer: Initializer for the table. Defaults to
        ``"truncated_normal"``. An instance is used exactly as supplied.
    :type pos_initializer: Union[str, keras.initializers.Initializer]
    :param scale: Standard deviation for the DEFAULT initializer, i.e. used
        only when ``pos_initializer`` is the string ``"truncated_normal"``.
        Ignored for a caller-supplied instance. Must be positive. Defaults
        to ``0.02``.
    :type scale: float
    :param kwargs: Additional keyword arguments for the Layer base class.

    :ivar pos_embedding: The position table, shape
        ``(1, max_seq_len, dim)``. ``None`` until :meth:`build` runs.
    :vartype pos_embedding: Optional[keras.Variable]

    Input shape:
        3D tensor ``(batch, seq_len, dim)``.

    Output shape:
        The same shape as the input.

    :raises ValueError: If ``max_seq_len``, ``dim`` or ``scale`` is not
        positive, or if ``dropout_rate`` is outside ``[0, 1]``. Raised from
        ``__init__``.
    :raises ValueError: If the input is not rank 3, or if its last dimension
        is not ``dim``. Raised from ``build()``.

    Example:

    .. code-block:: python

        import numpy as np
        from dl_techniques.layers.embedding import (
            positional_embedding as pe,
        )

        layer = pe.PositionalEmbedding(max_seq_len=16, dim=8)
        x = np.zeros((2, 5, 8), dtype="float32")
        layer(x).shape  # (2, 5, 8)
    """

    def __init__(
            self,
            max_seq_len: int,
            dim: int,
            dropout_rate: float = 0.0,
            pos_initializer: Union[str, keras.initializers.Initializer] = "truncated_normal",
            scale: float = 0.02,
            **kwargs: Any
    ) -> None:
        """Validate the configuration, resolve the initializer, build dropout.

        The table itself is created in :meth:`build`, once the input width is
        known and can be checked against ``dim``.

        :param max_seq_len: Number of rows in the position table.
        :type max_seq_len: int
        :param dim: Width of the position table.
        :type dim: int
        :param dropout_rate: Dropout applied after the addition.
        :type dropout_rate: float
        :param pos_initializer: Initializer for the table. An instance is
            honoured as given; only the string ``"truncated_normal"`` is
            resolved with ``scale``.
        :type pos_initializer: Union[str, keras.initializers.Initializer]
        :param scale: Standard deviation for the default
            ``"truncated_normal"`` initializer.
        :type scale: float
        :param kwargs: Additional keyword arguments for the Layer base class.
        :type kwargs: Any
        :raises ValueError: If ``max_seq_len``, ``dim`` or ``scale`` is not
            positive, or if ``dropout_rate`` is outside ``[0, 1]``.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {dropout_rate}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")

        # Store ALL configuration parameters
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.dropout_rate = dropout_rate
        self.scale = scale
        self.pos_initializer = keras.initializers.get(pos_initializer)

        # DECISION plan-2026-08-28T181715-3870472c/D-001
        # Branch on the ARGUMENT, never on the resolved object. An
        # `isinstance(self.pos_initializer, TruncatedNormal)` test here would
        # also catch a caller's own `TruncatedNormal(stddev=0.5)` and silently
        # rewrite its stddev to `scale` (measured: 0.5 -> 0.02), while letting
        # every other initializer type through untouched. Do NOT reintroduce
        # it. `keras.initializers.get("truncated_normal")` already returns a
        # TruncatedNormal, so the string test below is the only one that can
        # distinguish "the caller took the default" from "the caller built an
        # instance". See decisions.md D-001.
        if isinstance(pos_initializer, str) and pos_initializer == "truncated_normal":
            self.pos_initializer = keras.initializers.TruncatedNormal(stddev=self.scale)

        # DECISION plan-2026-08-22T035419-a11304c8/D-055
        # This flag is TRUE because the op is position-wise, so a mask passes
        # through unchanged. Do NOT read it as a masking repair. Measured
        # 2026-08-22 on a TextEncoder with padded input and no explicit
        # attention_mask, the padded-vs-unpadded gap was 1.290977e-02 both
        # with the flag and without it; the explicit `attention_mask=`
        # argument brought it to 2.384186e-07. See decisions.md D-055.
        self.supports_masking = True

        # CREATE sub-layer in __init__ (modern Keras 3 pattern)
        self.dropout = keras.layers.Dropout(self.dropout_rate, name="pos_dropout")

        # Weight will be initialized in build()
        self.pos_embedding = None

        logger.info(f"Initialized PositionalEmbedding with max_seq_len={self.max_seq_len}, "
                    f"dim={self.dim}, dropout={self.dropout_rate}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the position table and build the dropout sub-layer.

        :param input_shape: Shape of the input, ``(batch, seq_len, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 3, or if its last
            dimension is not ``dim``.
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"Input must be 3D (batch_size, seq_len, dim), got shape {input_shape}"
            )

        if input_shape[-1] != self.dim:
            raise ValueError(
                f"Input dimension {input_shape[-1]} does not match expected dim {self.dim}"
            )

        # Create positional embeddings weight
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.max_seq_len, self.dim),
            initializer=self.pos_initializer,
            trainable=True,
        )

        # Build the sub-layer explicitly so it carries a shape through
        # serialization. Dropout does not change shape, so the input shape
        # is the right argument.
        self.dropout.build(input_shape)

        logger.info(f"Built PositionalEmbedding with input_shape={input_shape}")

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Slice the table to the input length and add it.

        :param inputs: Input of shape ``(batch, seq_len, dim)``.
        :type inputs: keras.KerasTensor
        :param training: Training flag forwarded to the dropout sub-layer.
        :type training: Optional[bool]
        :return: The input with position vectors added, same shape.
        :rtype: keras.KerasTensor
        """
        # Read the length from ops.shape so this stays graph-safe under a
        # dynamic sequence axis.
        input_shape = ops.shape(inputs)
        seq_len = input_shape[1]

        # A seq_len above max_seq_len fails inside this slice, at call time.
        # There is no earlier check.
        positions = ops.slice(
            self.pos_embedding,
            start_indices=(0, 0, 0),
            shape=(1, seq_len, self.dim)
        )

        # The table has a leading axis of 1, so it broadcasts over batch.
        outputs = inputs + positions

        outputs = self.dropout(outputs, training=training)

        return outputs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Report the output shape, which equals the input shape.

        :param input_shape: Shape of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: Configuration dictionary carrying every ``__init__``
            argument. ``pos_initializer`` is serialized, so a table built
            with a caller-supplied initializer reloads with the same one.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "max_seq_len": self.max_seq_len,
            "dim": self.dim,
            "dropout_rate": self.dropout_rate,
            "pos_initializer": keras.initializers.serialize(self.pos_initializer),
            "scale": self.scale,
        })
        return config


# ---------------------------------------------------------------------
