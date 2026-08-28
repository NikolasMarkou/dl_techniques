"""
A dual-configuration Rotary Position Embedding (RoPE).

This layer holds TWO complete RoPE table pairs in one module, built with
different frequency bases. It exists for hybrid-attention architectures such
as Gemma3, which mix full attention with local sliding-window attention and
want a different positional scale for each.

Architecture:
    Two independent, non-trainable cos/sin table pairs are built once:

    1.  **Global RoPE.** A large ``theta_base``, 1,000,000 by default. The
        frequencies are low and the wavelengths long, so the signal changes
        slowly across the sequence. That suits full attention, where the
        point is long-range dependency.

    2.  **Local RoPE.** A smaller ``theta_base``, 10,000 by default. The
        frequencies are higher, so nearby tokens are told apart more
        sharply. That suits a sliding window, where only a small context
        matters.

    The caller picks ``'global'`` or ``'local'`` per call. Nothing else
    changes: the same rotation code runs against whichever pair was chosen.

Pairing convention:
    This layer uses SPLIT-HALF pairing, the GPT-NeoX form Gemma3 follows:
    channel ``j`` rotates with channel ``j + head_dim/2``. The table is built
    at half width and then duplicated, ``concat([freqs, freqs])``, which is
    what makes the ``[-x2, x1]`` rotation correct. Verified by execution.
    This differs from ``rotary_position_embedding.py``, which pairs ``x[2i]``
    with ``x[2i+1]``. The two are not interchangeable.

Foundational Mathematics:
    RoPE encodes an absolute position ``m`` by rotating a feature pair, and
    the inner product of two rotated vectors depends only on ``m - k``. The
    rotation angle for pair ``i`` is ``m * theta_i``, with::

        theta_i = 1 / (theta_base^(2i / d))

    ``theta_base`` sets the wavelength directly. A larger base gives smaller
    ``theta_i``, so the signal varies slowly. A smaller base gives larger
    ``theta_i`` and a signal that changes quickly.

    Keeping both ``{theta_i}_global`` and ``{theta_i}_local`` lets one model
    match the positional scale to the attention span it is currently using.

References:
    - Google (2024). "Gemma 3 Technical Report" (the dual-RoPE mechanism).
    - Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary
      Position Embedding".
"""

import keras
from typing import Optional, Any, Tuple, Literal, Dict

# ---------------------------------------------------------------------

RopeType = Literal['global', 'local']

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DualRotaryPositionEmbedding(keras.layers.Layer):
    """Two RoPE table pairs in one layer, selected per call.

    Rotates a post-head-split tensor of shape
    ``(batch, heads, seq_len, head_dim)``. Two independent cos/sin pairs are
    built once, one from ``global_theta_base`` and one from
    ``local_theta_base``. ``call()`` takes a ``rope_type`` argument and uses
    exactly one pair; shape is unchanged either way.

    **Pairing convention: SPLIT-HALF**, the Gemma 3 / GPT-NeoX form: channel
    ``j`` rotates with channel ``j + head_dim/2`` -- verified by impulse (at
    ``head_dim=8`` a one-hot at channel 0 leaks into channel 4 and nowhere
    else). It is NOT the INTERLEAVED pairing used by
    ``rotary_position_embedding.py`` and ``axial_rope_2d.py`` in this same
    package. Both are valid rotations and both give the relative-position
    property, so the wrong one TRAINS FINE: the difference is invisible in a
    config, in a shape and at load time, and shows up only as plausible,
    wrong numbers.

    *Which checkpoints this can consume.* Split-half is HF's ``rotate_half``,
    which is what ``modeling_gemma3.py``, ``modeling_gpt_neox.py``,
    ``modeling_llama.py`` and the Qwen family all use (each defines the
    identical ``rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])``), so a
    ``q_proj``/``k_proj`` from any HF checkpoint in that lineage drops in
    directly. It CANNOT consume weights from an INTERLEAVED implementation --
    GPT-J's ``rotate_every_two``, Meta's official LLaMA ``apply_rotary_emb``,
    or this package's own ``RotaryPositionEmbedding`` -- without permuting the
    projection rows first. HF ships exactly that permutation as
    ``convert_llama_weights_to_hf.py::permute``; the reason it has to exist is
    huggingface/transformers issue #25199, "[LLaMA] Rotary positional
    embedding differs with official implementation", which is the same weights
    serving two conventions.

    *References*: Su, J., et al. (2021). "RoFormer: Enhanced Transformer with
    Rotary Position Embedding". arXiv:2104.09864 (RoPE itself). Google (2025).
    "Gemma 3 Technical Report" (the dual global/local theta design this layer
    implements).

    **Architecture Overview:**

    .. code-block:: text

        Two independent table pairs, built once in build():

          global path                    local path
          theta = global_theta_base      theta = local_theta_base
          default 1e6, low frequency     default 1e4, high frequency
          cos_global_cached              cos_local_cached
          sin_global_cached              sin_local_cached
          each (max_seq_len, head_dim)   each (max_seq_len, head_dim)

        SHARED between the paths: head_dim, max_seq_len, the table
        builder `_create_rope_cache_tables` and the rotation code.
        NOT shared: theta_base and all four weights. The two paths
        have no weight and no table row in common.

        input x (B, heads, seq_len, head_dim)
                        │
             rope_type selects ONE pair at call time
                 ┌──────┴──────┐
                 ▼             ▼
             'global'       'local'
                 └──────┬──────┘
                        ▼
        cos, sin = pair[:seq_len], each expanded to
        (1, 1, seq_len, head_dim)
                        │
        split-half: x1 = x[..., :d/2], x2 = x[..., d/2:]
        rotated = concat([-x2, x1])
        out = x * cos + rotated * sin
                        │
                        ▼
        output (B, heads, seq_len, head_dim), cast back to x.dtype

    :param head_dim: Dimensionality of each attention head. Must be positive
        and even, because the rotation pairs the two halves.
    :type head_dim: int
    :param max_seq_len: Largest position both table pairs cover. Must be
        positive. A longer input raises at call time.
    :type max_seq_len: int
    :param global_theta_base: Frequency base for the global pair. Larger
        values stretch the wavelengths and suit long-range attention.
        Defaults to ``1_000_000.0``.
    :type global_theta_base: float
    :param local_theta_base: Frequency base for the local pair. Smaller
        values sharpen nearby-token discrimination. Defaults to
        ``10_000.0``.
    :type local_theta_base: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar cos_global_cached: Non-trainable float32 table of shape
        ``(max_seq_len, head_dim)``. ``None`` until ``build()`` runs.
    :vartype cos_global_cached: keras.Variable or None
    :ivar sin_global_cached: Sine partner of ``cos_global_cached``.
    :vartype sin_global_cached: keras.Variable or None
    :ivar cos_local_cached: Local-path cosine table, same shape.
    :vartype cos_local_cached: keras.Variable or None
    :ivar sin_local_cached: Local-path sine table, same shape.
    :vartype sin_local_cached: keras.Variable or None

    Input shape:
        4D tensor with shape ``(batch_size, num_heads, seq_len, head_dim)``.

    Output shape:
        4D tensor with the same shape as the input.

    :raises ValueError: If ``head_dim`` is not positive and even, if
        ``max_seq_len`` is not positive, or if either ``theta_base`` is not
        positive. Raised from ``__init__``.
    :raises ValueError: If the input is not 4D, or if its last dimension is
        not ``head_dim``. Raised from ``build()``.
    :raises ValueError: If ``rope_type`` is neither ``'global'`` nor
        ``'local'``, or if the static sequence length exceeds
        ``max_seq_len``. Raised from ``call()``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.embedding import (
            create_embedding_layer,
        )

        rope = create_embedding_layer(
            "dual_rope", head_dim=64, max_seq_len=128,
        )
        q = keras.random.normal((2, 8, 16, 64))
        rope(q, rope_type="local").shape  # (2, 8, 16, 64)
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        global_theta_base: float = 1_000_000.0,
        local_theta_base: float = 10_000.0,
        **kwargs: Any
    ) -> None:
        """Validate the configuration and store it.

        No weight is created here; both table pairs are built in
        :meth:`build`.

        :param head_dim: Dimensionality of each attention head.
        :type head_dim: int
        :param max_seq_len: Largest position the tables will cover.
        :type max_seq_len: int
        :param global_theta_base: Frequency base for the global pair.
        :type global_theta_base: float
        :param local_theta_base: Frequency base for the local pair.
        :type local_theta_base: float
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        :raises ValueError: If ``head_dim`` is not positive and even, if
            ``max_seq_len`` is not positive, or if either ``theta_base`` is
            not positive.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if global_theta_base <= 0:
            raise ValueError(f"global_theta_base must be positive, got {global_theta_base}")
        if local_theta_base <= 0:
            raise ValueError(f"local_theta_base must be positive, got {local_theta_base}")

        # Store ALL configuration parameters
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.global_theta_base = global_theta_base
        self.local_theta_base = local_theta_base

        # Initialize weight attributes - created in build()
        self.cos_global_cached = None
        self.sin_global_cached = None
        self.cos_local_cached = None
        self.sin_local_cached = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create both table pairs, four non-trainable weights in total.

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

        # Create both RoPE cache tables
        self._create_global_rope_cache()
        self._create_local_rope_cache()

        # Always call parent build at the end
        super().build(input_shape)

    def _create_rope_cache_tables(self, theta_base: float, cache_prefix: str) -> Tuple[Any, Any]:
        """Create one cos/sin table pair for a given ``theta_base``.

        Both weights have shape ``(max_seq_len, head_dim)`` and are
        non-trainable float32. The half-width frequency table is duplicated,
        ``concat([freqs, freqs])``, which is what makes the split-half
        rotation in :meth:`_apply_dual_rope_rotation` correct.

        Called twice, once per path. Each call creates its OWN weights; the
        two paths share this code, not the tables.

        :param theta_base: Frequency base for this path.
        :type theta_base: float
        :param cache_prefix: Name prefix for the two weights, ``'global'`` or
            ``'local'``.
        :type cache_prefix: str
        :return: The pair ``(cos_cache, sin_cache)``.
        :rtype: Tuple[Any, Any]
        """
        # Half of head_dim, one frequency per rotated channel pair.
        freq_dim = self.head_dim // 2

        # An INITIALIZER computes each table INSIDE itself. Do NOT compute
        # `cos_table`/`sin_table` out here and `.assign()` them. Keras 3 runs a
        # symbolic build pass inside a `StatelessScope` whenever this layer is
        # first reached from a parent's `call()`, which covers every real model,
        # and that scope records the `.assign()` and then discards it, leaving
        # both caches at their 'zeros' initializer. Measured on CPU 2026-08-15:
        # a direct `.build(...)` gives `cos_global_cached[0, 0] == 1.0`, while
        # the same layer reached through a parent's `call()` gave `0.0` with the
        # whole table zero, so `call()` multiplied q and k by cos=0 / sin=0 and
        # returned exactly zeros. Equally, do NOT close over a table computed
        # out here: that tensor belongs to the symbolic pass's scratch
        # `FuncGraph` and raises "cannot be accessed from here ... out of scope"
        # on the eager pass. Same defect and fix as
        # `rotary_position_embedding.py` (D-021). See decisions.md D-027
        # of plan-2026-08-14T233721-d4f9beb2.
        def _table_initializer(trig):
            """Make an initializer that fills a table with ``trig``.

            Closes over ``theta_base``, so each path gets its own.

            :param trig: ``keras.ops.cos`` or ``keras.ops.sin``.
            :type trig: Callable
            :return: A Keras initializer callable.
            :rtype: Callable
            """

            def initializer(shape, dtype=None):
                """Compute the whole table at variable-creation time.

                :param shape: ``(max_seq_len, head_dim)``.
                :type shape: Tuple[int, int]
                :param dtype: Requested dtype, or ``None`` for
                    ``'float32'``.
                :type dtype: Optional[str]
                :return: The filled table.
                :rtype: keras.KerasTensor
                """
                table_dtype = dtype or 'float32'
                # 1 / (theta_base ^ (2i / head_dim)) for i in [0, freq_dim).
                inv_freq = 1.0 / (
                    theta_base ** (
                        keras.ops.arange(0, shape[1] // 2, dtype=table_dtype)
                        * 2.0 / self.head_dim
                    )
                )
                positions = keras.ops.arange(shape[0], dtype=table_dtype)
                freqs = keras.ops.outer(positions, inv_freq)
                # Duplicate to the full head_dim. This is what makes the
                # split-half rotation correct: slot i serves channel i and
                # channel i + head_dim/2 (the Gemma3 form).
                freqs_full = keras.ops.concatenate([freqs, freqs], axis=1)
                return trig(freqs_full)

            return initializer

        table_shape = (self.max_seq_len, freq_dim * 2)
        cos_cache = self.add_weight(
            name=f'cos_{cache_prefix}_cached',
            shape=table_shape,
            initializer=_table_initializer(keras.ops.cos),
            trainable=False,
            dtype='float32'
        )
        sin_cache = self.add_weight(
            name=f'sin_{cache_prefix}_cached',
            shape=table_shape,
            initializer=_table_initializer(keras.ops.sin),
            trainable=False,
            dtype='float32'
        )

        return cos_cache, sin_cache

    def _create_global_rope_cache(self) -> None:
        """Create the global path's table pair from ``global_theta_base``.

        :return: Nothing. Sets ``cos_global_cached`` and
            ``sin_global_cached``.
        :rtype: None
        """
        self.cos_global_cached, self.sin_global_cached = self._create_rope_cache_tables(
            theta_base=self.global_theta_base,
            cache_prefix='global'
        )

    def _create_local_rope_cache(self) -> None:
        """Create the local path's table pair from ``local_theta_base``.

        :return: Nothing. Sets ``cos_local_cached`` and ``sin_local_cached``.
        :rtype: None
        """
        self.cos_local_cached, self.sin_local_cached = self._create_rope_cache_tables(
            theta_base=self.local_theta_base,
            cache_prefix='local'
        )

    def call(
        self,
        inputs: keras.KerasTensor,
        rope_type: RopeType = 'global',
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply dual rotary position embedding to input tensor.

        :param inputs: Input tensor with shape
            ``(batch_size, num_heads, seq_len, head_dim)``.
        :type inputs: keras.KerasTensor
        :param rope_type: Type of RoPE to apply. ``'global'`` for full
            attention with long-range modeling, ``'local'`` for sliding
            attention with local patterns. Defaults to ``'global'``.
        :type rope_type: RopeType
        :param training: Whether in training mode (unused by this layer).
        :type training: Optional[bool]
        :return: Output tensor with same shape, appropriate RoPE transformation
            applied.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``rope_type`` is invalid or sequence length
            exceeds ``max_seq_len``.
        """
        # Validate rope_type
        if rope_type not in ['global', 'local']:
            raise ValueError(f"rope_type must be 'global' or 'local', got '{rope_type}'")

        # Get sequence length from input
        seq_len = keras.ops.shape(inputs)[2]

        # Validate sequence length at build time if known
        seq_len_static = inputs.shape[2]
        if seq_len_static is not None and seq_len_static > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len_static}) exceeds max_seq_len ({self.max_seq_len}). "
                f"Please increase max_seq_len or truncate the input."
            )

        return self._apply_dual_rope_rotation(inputs, seq_len, rope_type)

    def _apply_dual_rope_rotation(
        self,
        x: keras.KerasTensor,
        seq_len: keras.KerasTensor,
        rope_type: RopeType
    ) -> keras.KerasTensor:
        """Apply rotary position embedding using the selected RoPE configuration.

        :param x: Input tensor with shape
            ``(batch_size, num_heads, seq_len, head_dim)``.
        :type x: keras.KerasTensor
        :param seq_len: Current sequence length tensor.
        :type seq_len: keras.KerasTensor
        :param rope_type: Which pair to use, ``'global'`` or ``'local'``.
        :type rope_type: RopeType
        :return: Tensor of the same shape with the selected rotation applied.
        :rtype: keras.KerasTensor
        """
        # Pick ONE table pair. The other pair is untouched on this call.
        # Each slice has shape (seq_len, head_dim).
        if rope_type == 'global':
            cos = self.cos_global_cached[:seq_len]
            sin = self.sin_global_cached[:seq_len]
        else:
            cos = self.cos_local_cached[:seq_len]
            sin = self.sin_local_cached[:seq_len]

        # Split-half pairing, the Gemma3 form: channel j rotates with channel
        # j + head_dim/2. x1 is the first half, x2 the second.
        half_dim = self.head_dim // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]

        # (seq_len, head_dim) -> (1, 1, seq_len, head_dim) so the tables
        # broadcast over batch and heads.
        cos = keras.ops.expand_dims(keras.ops.expand_dims(cos, 0), 0)
        sin = keras.ops.expand_dims(keras.ops.expand_dims(sin, 0), 0)

        # The 90-degree companion vector [-x2, x1].
        rotated = keras.ops.concatenate([-x2, x1], axis=-1)

        x_rotated = (x * cos) + (rotated * sin)

        # The tables are float32, so cast back to whatever the caller sent.
        return keras.ops.cast(x_rotated, x.dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape (identical to input shape).

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (same as input shape).
        :rtype: Tuple[Optional[int], ...]
        """
        # The rotation is shape-preserving.
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
            'global_theta_base': self.global_theta_base,
            'local_theta_base': self.local_theta_base,
        })
        return config

# ---------------------------------------------------------------------
