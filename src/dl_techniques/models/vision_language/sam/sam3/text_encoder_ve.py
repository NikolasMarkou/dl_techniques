"""Sam3TextEncoder, the causal text tower for SAM 3's open-vocabulary prompt.

Wraps the repository's shared :class:`~dl_techniques.layers.transformers.text_encoder.TextEncoder`
and adds the two things that layer does not supply on its own: an explicit
lower-triangular causal keep-mask, rebuilt every call from the actual sequence
length, and a ``Dense(width -> d_model)`` resizer applied to the whole
per-token sequence. It emits every token's features, never a pooled summary
-- pooling happens later, in the scorer, as a masked mean.

The wrapped layer is bidirectional and builds no causal mask by itself, so
this class must always pass one; omitting it is a silent value defect with
no shape symptom.

This is not a bit-faithful port of the upstream tower. It applies the
wrapped layer's ``embed_norm``, which the reference does not have (measured
106.9% of output amplitude at the settled width), and it has no counterpart
to the reference's unused 524,288-parameter ``text_projection``. Both
divergences are inert while training starts from scratch and become binding
only at a future weight transfer.

References:
    - Ravi et al., 2025. SAM 3: Segment Anything with Concepts.
    - Radford et al., 2021. Learning Transferable Visual Models From Natural
      Language Supervision.
"""

import keras
from keras import layers, ops
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformers.text_encoder import TextEncoder
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.sam3.text_encoder_ve")
class Sam3TextEncoder(keras.layers.Layer):
    """SAM 3's CLIP text tower, emitting the full per-token sequence.

    See the module docstring for the two accepted divergences from the
    upstream reference tower (``embed_norm``, missing ``text_projection``).

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

        # DECISION plan-2026-08-04T044628-4c240b4c/D-101: use output_mode='none',
        # never get_sequence_features().
        # That method mutates self.pooling_layer.strategy and re-invokes self(...) -- not trace-safe inside call(). See decisions.md.
        #
        # DECISION plan-2026-08-04T044628-4c240b4c/D-102: normalization_position
        # ='pre' and zero dropout are required, not stylistic defaults.
        # The reference tower is pre-normalized with a terminal norm and no dropout; restoring post-norm/dropout defaults changes every output value. See decisions.md.
        #
        # DECISION plan-2026-08-04T044628-4c240b4c/D-142: this accepts an extra
        # embed_norm the reference lacks (measured 106.9% of output amplitude).
        # No kwarg removes it without also disabling every block/final norm; fix at the weight-transfer phase alongside the missing text_projection. See decisions.md.
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
        # DECISION plan-2026-08-04T044628-4c240b4c/D-136: keep this re-entry
        # guard (corrects D-126, which wrongly said only this class needed it).
        # Sam3DotProductScoring.build needed the same guard; both were masked by the same caller. See decisions.md.
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
