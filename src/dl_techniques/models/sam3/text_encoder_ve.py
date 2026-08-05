"""
SAM 3's CLIP text tower: a causal per-token encoder plus the ``d_model`` resizer.

This module provides the single public class :class:`Sam3TextEncoder`, the text
side of SAM 3's open-vocabulary prompt path. It is a thin wrapper: all of the
transformer arithmetic is the repo's existing
:class:`~dl_techniques.layers.transformers.text_encoder.TextEncoder`, and this
class supplies the two things that layer does not.

Architecture:
    Token ids ``(batch, seq)`` -> learned token + learned absolute positional
    embeddings -> ``depth`` pre-normalized self-attention blocks at ``width`` ->
    a final normalization -> a single ``Dense(width -> d_model)`` resizer,
    yielding the full **per-token** sequence ``(batch, seq, d_model)``.

    At the settled SAM 3 configuration that is ``width=1024``, ``depth=24``,
    ``num_heads=16``, ``mlp_ratio=4.0``, ``context_length=32``,
    ``vocab_size=49408`` and ``d_model=256``.

What this wrapper adds, and why each piece exists:
    1. **The causal mask.** The wrapped encoder is BIDIRECTIONAL by default --
       its docstring's "not causally masked" describes the default, not a
       structural limitation, and nothing in it builds a causal mask for you.
       MEASURED at the settled width: perturbing the LAST token moves the
       position-0 output by ``0.1404891`` with no mask and by exactly ``0.0``
       with the explicit lower-triangular keep-mask this class passes. Omitting
       it is a silent value defect with no shape symptom and no exception, which
       is why the guard for it asserts EXACT zero rather than a tolerance -- a
       tolerance loose enough to feel safe is loose enough to hide the leak.
    2. **The resizer**, applied to the whole sequence rather than to a pooled
       vector.

    The mask is built explicitly on every call, at the sequence length actually
    supplied, and passed as ``attention_mask=``. There is no cached buffer: the
    reference slices a pre-registered ``(ctx, ctx)`` buffer to the live sequence
    length, and rebuilding the same triangle from ``arange`` is the identical
    quantity with no state to keep consistent.

No pooling of any kind happens here:
    The reference computes a pooled sentence vector inside its text transformer
    and its own wrapper then **discards** it, keeping only the per-token
    sequence. Reproducing that pooling and throwing the result away would be
    wasted work, and a naive CLIP transliteration that wires the pooled vector
    downstream diverges from SAM 3's actual data flow with matching ranks at
    every intermediate step. Pooling in SAM 3 is a MASKED MEAN and it happens
    later, in the scorer -- not here, and never as an end-of-text argmax.

References:
    - Ravi, N. et al. (2025). "SAM 3: Segment Anything with Concepts."
    - Radford, A. et al. (2021). "Learning Transferable Visual Models From
      Natural Language Supervision" (CLIP; the text tower's causal masking and
      pre-normalized residual blocks).
"""

import keras
from keras import layers, ops
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformers.text_encoder import TextEncoder

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Sam3TextEncoder(keras.layers.Layer):
    """SAM 3's CLIP text tower, emitting the full per-token sequence.

    :param d_model: Width of the resizer's output, i.e. the detector's model
        width. Default: ``256``.
    :type d_model: int
    :param width: Transformer width of the text tower. Default: ``1024``.
    :type width: int
    :param depth: Number of self-attention blocks. Default: ``24``.
    :type depth: int
    :param num_heads: Attention heads per block. Default: ``16``.
    :type num_heads: int
    :param context_length: Maximum supported sequence length. Default: ``32``.
    :type context_length: int
    :param vocab_size: Token-embedding table size. Default: ``49408``.
    :type vocab_size: int
    :param mlp_ratio: Feed-forward expansion ratio; the hidden width is
        ``int(width * mlp_ratio)`` (TRUNCATION, never rounding). Default:
        ``4.0``.
    :type mlp_ratio: float
    :raises ValueError: If any dimension is non-positive, or ``width`` is not
        divisible by ``num_heads``.

    Example:
        >>> encoder = Sam3TextEncoder(d_model=8, width=32, depth=2, num_heads=4,
        ...                           context_length=8, vocab_size=100)
        >>> import numpy as np
        >>> out = encoder(np.zeros((2, 8), dtype="int32"))
        >>> out.shape
        (2, 8, 8)
    """

    def __init__(
            self,
            d_model: int = 256,
            width: int = 1024,
            depth: int = 24,
            num_heads: int = 16,
            context_length: int = 32,
            vocab_size: int = 49408,
            mlp_ratio: float = 4.0,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        for name, value in (
                ("d_model", d_model), ("width", width), ("depth", depth),
                ("num_heads", num_heads), ("context_length", context_length),
                ("vocab_size", vocab_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if width % num_heads != 0:
            raise ValueError(
                f"width ({width}) must be divisible by num_heads ({num_heads})"
            )
        if mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        # Store ALL configuration parameters.
        self.d_model = int(d_model)
        self.width = int(width)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.context_length = int(context_length)
        self.vocab_size = int(vocab_size)
        self.mlp_ratio = float(mlp_ratio)

        # DECISION plan-2026-08-04T044628-4c240b4c/D-101
        # `output_mode='none'` is a CONSTRUCTION-time choice, and it must stay
        # one. The wrapped layer also offers `get_sequence_features()`, which
        # returns the same tensor -- do NOT use it: it MUTATES
        # `self.pooling_layer.strategy`, re-invokes `self(...)`, and restores the
        # strategy in a `finally`. That is a side-effecting pattern inside what
        # would become a traced `call()`, and the mutation is not
        # trace-safe. Constructing with `output_mode='none'` reaches the same
        # per-token sequence with no state change at all. See decisions.md D-101.
        #
        # DECISION plan-2026-08-04T044628-4c240b4c/D-102
        # `normalization_position='pre'` and the three zeroed dropout rates are
        # NOT stylistic. The wrapped layer defaults to POST-normalization with
        # 0.1 dropout everywhere; the reference tower is pre-normalized
        # (`x = x + attn(norm(x))`) and carries a terminal normalization, which
        # this layer creates only in the 'pre' regime, and it has no dropout at
        # all. Do NOT "restore the defaults": post-norm changes every output
        # value with no shape symptom, the terminal normalization silently
        # disappears, and non-zero dropout makes `training=True` stochastic so
        # that any cross-call value oracle stops being reproducible. Pinned by
        # `test_text_encoder.py::TestUpstreamStructuralParity`. See D-102.
        self.encoder = TextEncoder(
            vocab_size=self.vocab_size, embed_dim=self.width, depth=self.depth,
            num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
            max_seq_len=self.context_length, output_mode="none",
            normalization_position="pre", dropout_rate=0.0,
            attention_dropout_rate=0.0, embed_dropout_rate=0.0,
            name="text_transformer",
        )
        self.resizer = layers.Dense(self.d_model, use_bias=True, name="resizer")

        logger.info(
            f"Sam3TextEncoder: width={self.width}, depth={self.depth}, "
            f"heads={self.num_heads}, ctx={self.context_length} -> "
            f"d_model={self.d_model}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the tower and the resizer.

        :param input_shape: Token-id shape ``(batch, seq)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the rank is not 2, or the static sequence length
            exceeds ``context_length``.
        """
        # DECISION plan-2026-08-04T044628-4c240b4c/D-136
        # Re-entry guard. D-126 recorded this class as the ONLY one in the
        # package missing it and resolved the symptom in the caller
        # (`Sam3Image._build_once`); that count was WRONG -- `Sam3DotProduct
        # Scoring.build` was missing it too, and both were masked by the same
        # caller. The guard is added at both sites so a composer that does not
        # copy `_build_once` is not surprised. Do NOT delete it.
        # See decisions.md D-136 (which corrects D-126).
        if self.built:
            return
        if len(input_shape) != 2:
            raise ValueError(
                f"Sam3TextEncoder expects token ids of shape (batch, seq), got "
                f"{input_shape}"
            )
        seq_len = input_shape[1]
        if seq_len is not None and seq_len > self.context_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds context_length "
                f"{self.context_length}; the positional table has only "
                f"{self.context_length} entries"
            )
        self.encoder.build(input_shape)
        self.resizer.build(tuple(input_shape) + (self.width,))
        super().build(input_shape)

    def causal_keep_mask(
            self, batch_size: Any, seq_len: Any
    ) -> keras.KerasTensor:
        """Build the lower-triangular boolean KEEP mask.

        ``mask[b, q, k]`` is ``True`` exactly when query position ``q`` may
        attend key position ``k``, i.e. when ``k <= q``. This is a KEEP
        predicate, not the additive ``-inf`` form the reference uses; the
        wrapped attention path consumes keep predicates.

        :param batch_size: Batch size (a tensor scalar is fine).
        :type batch_size: Any
        :param seq_len: Sequence length (a tensor scalar is fine).
        :type seq_len: Any
        :return: Boolean mask ``(batch, seq, seq)``.
        :rtype: keras.KerasTensor
        """
        positions = ops.arange(seq_len)
        keep = ops.expand_dims(positions, -1) >= ops.expand_dims(positions, 0)
        return ops.broadcast_to(
            ops.expand_dims(keep, axis=0), (batch_size, seq_len, seq_len)
        )

    def call(
            self, inputs: keras.KerasTensor, training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Encode token ids into a resized per-token sequence.

        :param inputs: Token ids ``(batch, seq)``, integer dtype.
        :type inputs: keras.KerasTensor
        :param training: Training-mode flag, forwarded verbatim.
        :type training: Optional[bool]
        :return: Per-token features ``(batch, seq, d_model)``.
        :rtype: keras.KerasTensor
        """
        shape = ops.shape(inputs)
        mask = self.causal_keep_mask(shape[0], shape[1])
        features = self.encoder(inputs, attention_mask=mask, training=training)
        return self.resizer(features, training=training)

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return ``(batch, seq, d_model)``.

        :param input_shape: Token-id shape ``(batch, seq)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return (input_shape[0], input_shape[1], self.d_model)

    def get_config(self) -> Dict[str, Any]:
        """Return every ``__init__`` parameter.

        :return: Serializable configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "width": self.width, "depth": self.depth,
            "num_heads": self.num_heads, "vocab_size": self.vocab_size,
            "context_length": self.context_length, "mlp_ratio": self.mlp_ratio,
        })
        return config
