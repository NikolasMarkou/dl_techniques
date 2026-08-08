"""
SAM 3 CLIP Text Tower: a causal per-token encoder plus the ``d_model`` resizer.
===============================================================================

:class:`Sam3TextEncoder` is the text side of SAM 3's open-vocabulary prompt
path. It is a thin wrapper: the transformer arithmetic is the repository's
existing :class:`~dl_techniques.layers.transformers.text_encoder.TextEncoder`,
and this class supplies the two things that layer does not -- a causal mask and
a sequence-wide resizer.

Based on:
---------
- Ravi, N. et al. (2025). "SAM 3: Segment Anything with Concepts."
- Radford, A. et al. (2021). CLIP -- causal masking, pre-normalized blocks.

Key Features:
------------
- An explicit lower-triangular keep-mask, rebuilt from ``arange`` on every call
  at the sequence length actually supplied. No cached buffer.
- A single ``Dense(width -> d_model)`` resizer over the WHOLE sequence.
- No pooling of any kind happens here. Pooling in SAM 3 is a MASKED MEAN and it
  happens later, in the scorer -- never as an end-of-text argmax.

Architecture Overview:
---------------------
1. Token ids ``(batch, seq)`` -> learned token + absolute positional embeddings.
2. -> ``depth`` pre-normalized self-attention blocks at ``width``.
3. -> a final normalization -> ``Dense(width -> d_model)``, yielding the full
   per-token sequence ``(batch, seq, d_model)``.
Settled configuration: ``width=1024``, ``depth=24``, ``num_heads=16``,
``mlp_ratio=4.0``, ``context_length=32``, ``vocab_size=49408``, ``d_model=256``.

Usage Examples:
--------------
```python
from dl_techniques.models.SAM.SAM3.text_encoder_ve import Sam3TextEncoder
encoder = Sam3TextEncoder(d_model=256, width=1024, depth=24, num_heads=16)
prompt = encoder(token_ids)          # (batch, seq, 256)
```

Measured caveats:
----------------
- The wrapped encoder is BIDIRECTIONAL by default and builds no causal mask for
  you. MEASURED at the settled width: perturbing the LAST token moves the
  position-0 output by ``0.1404891`` with no mask and by exactly ``0.0`` with
  the keep-mask this class passes. Omitting it is a silent value defect with no
  shape symptom and no exception, which is why the guard asserts EXACT zero
  rather than a tolerance.
- **ACCEPTED divergence 1 (D-142), an extra ``embed_norm``.** The reference goes
  ``token_embedding -> + positional_embedding -> transformer`` with nothing
  between (``sam3/model/text_encoder_ve.py:238-245`` at the pinned SHA); the
  wrapped repo layer creates an ``embed_norm``, applies it unconditionally, and
  offers no constructor argument that removes it. MEASURED at the SETTLED
  ``width=1024`` by replacing it with a passthrough: max abs output delta
  ``5.917600`` against a max output amplitude of ``5.536552`` -- **106.9 %**,
  and 460 % of the output RMS. The same probe at this package's ``tiny`` width
  reads ``1.805329`` against ``3.722330`` (**48.5 %**), so a toy-width probe
  understates the divergence 2.2x. This is NOT a reparametrization: a
  normalization on the residual stream destroys the per-token mean and scale of
  the embedding irreversibly.
- **ACCEPTED divergence 2 (D-142), no ``text_projection``.** The reference's
  ``TextTransformer`` carries a ``(width, output_dim)`` parameter its own
  wrapper never consumes but which exists in any released checkpoint. Upstream
  does not pass ``output_dim``, so it is ``(1024, 512)`` = **524,288**
  parameters, not the ``1024 x 256`` a reader assuming ``output_dim == d_model``
  would expect.
- **Net, in signed named terms**: this tower is the reference's **plus 2,048**
  unmatched (``embed_norm``) and **minus 524,288** missing
  (``text_projection``) -- ``353,202,432`` here against ``353,724,672``
  upstream, a difference of exactly ``522,240``. Phase 1 loads no pretrained
  weights so nothing is wrong today; both are BINDING on any future weight
  transfer.
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

    **This is not a bit-faithful port of the upstream tower**, and the two
    accepted divergences are stated in the module docstring above rather than
    left to be rediscovered: the wrapped layer applies an ``embed_norm`` the
    reference does not have (MEASURED **106.9 %** of the output amplitude at
    the settled width), and the reference's unconsumed 524,288-parameter
    ``text_projection`` has no counterpart here. Both are inert while phase 1
    trains from scratch and both are BINDING on any weight transfer (D-142).

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
        #
        # DECISION plan-2026-08-04T044628-4c240b4c/D-142
        # This construction call ACCEPTS an extra normalization the reference
        # does not have. `TextEncoder` builds an `embed_norm` and applies it
        # between the embeddings and block 0 (`text_encoder.py:775`); the
        # reference goes token-embed -> +pos -> transformer with nothing
        # between. MEASURED at the settled width: max abs output delta
        # **5.917600** against an amplitude of 5.536552 -- 106.9 %, the largest
        # divergence in this package. There is NO kwarg that removes it:
        # `normalization_type` governs `embed_norm`, every block norm and
        # `final_norm` together. Do NOT "fix" it by editing
        # `layers/transformers/text_encoder.py` -- that file is byte-frozen
        # under I-1 and has other consumers. The ONE working remedy, measured
        # feasible and RED-proven in
        # `test_text_encoder.py::TestEmbeddingNormDivergence`, is a PRE-BUILD
        # per-instance substitution here:
        #     self.encoder.embed_norm = keras.layers.Identity()
        # It is deliberately NOT applied: it moves this tower's count to
        # 353,200,384 and the assembly's to 821,706,550, invalidating pinned
        # figures across an already-gated iteration for a divergence that is
        # inert while phase 1 loads no pretrained weights. Apply it -- together
        # with the missing 524,288-parameter `text_projection` -- at the
        # weight-transfer phase. See decisions.md D-142 and progress.md
        # § Deferred D-6.
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
