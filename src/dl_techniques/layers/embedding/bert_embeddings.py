"""
Composite input embeddings for BERT-style encoders.

This layer builds the initial vector for each token of a sequence by summing a
token embedding, a position embedding and, optionally, a segment (token type)
embedding, then normalizing the sum and applying dropout. It is the standard
BERT embedding stage, generalized so that several encoder families share one
implementation instead of each keeping a private copy.

Shared by three model packages:
    ``BertEmbeddings`` is not local to any one model. Three packages consume it,
    each verified against the tree:

    - ``models/language/bert/model.py`` imports the class directly.
    - ``models/language/fnet/model.py`` imports the class directly.
    - ``models/language/distilbert/model.py`` builds it through
      ``create_embedding_layer('bert_embeddings', ...)``, passing
      ``use_token_type_embeddings=False`` and ``mask_zero=False``.

    ``models/language/modern_bert/`` is NOT a consumer. It uses the separate
    ``ModernBertEmbeddings`` class in ``modern_bert_embeddings.py``, which has no
    position embedding at all; the two must not be conflated. A change to this
    file's behaviour, defaults or error messages therefore lands in three model
    families at once, and the sharing is documented here rather than only on the
    consumer side so an editor of this file sees it first.

Architecture:
    A token's representation is treated as a function of its identity, its
    position and the sentence it belongs to. One vector is produced per source
    and the three are summed element-wise:

    1.  Token embeddings. The ordinary word-embedding lookup, mapping a token id
        to a dense vector: the context-independent meaning of the token.

    2.  Position embeddings. Attention is permutation-invariant, so order has to
        be injected explicitly. Two modes exist. ``'learned'`` (the BERT
        convention) allocates a trainable table with one row per absolute
        position, letting the model choose how to represent order. Choosing
        ``'sinusoidal'`` instead computes a fixed sin/cos table on the fly: it
        allocates no weight and is not bounded by ``max_position_embeddings``.

    3.  Segment (token type) embeddings. BERT's Next Sentence Prediction task
        concatenates two sentences into a single sequence, and this term is the
        learnable signal for which sentence a token came from. It is optional:
        DistilBERT has no segment embedding and switches the term off.

Mathematics:
    For a token at position ``i``, the embedding before normalization is::

        E(token_i) = E_word(token_i) + E_position(i) + E_segment(A or B)

    with the third term present only when token type embeddings are enabled.
    The sum passes through the configured normalization layer, which keeps the
    distribution reaching the first encoder block stable, and then through
    dropout for regularization.

References:
    - Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). "BERT:
      Pre-training of Deep Bidirectional Transformers for Language
      Understanding".
"""

import math
import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..norms.rms_norm import RMSNorm
from ..norms.band_rms import BandRMS
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Accepted enum values -- the single source of truth.
#
# DECISION plan-2026-08-10T183739-b007f435/D-009
# factory.py's validate_embedding_config imports these two tuples. Do NOT
# inline a literal list into the constructor checks or into the factory: the
# normalization list was once two hand-maintained copies, and one rule kept in
# two places is a defect waiting for one side to be edited alone.
# See decisions.md D-009.
# ---------------------------------------------------------------------

VALID_NORMALIZATION_TYPES: Tuple[str, ...] = (
    'layer_norm', 'rms_norm', 'band_rms', 'batch_norm'
)
VALID_POSITION_EMBEDDING_TYPES: Tuple[str, ...] = ('learned', 'sinusoidal')

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BertEmbeddings(keras.layers.Layer):
    """BERT embedding layer combining word, position and token type embeddings.

    Builds a composite token representation by summing a word embedding
    ``E_word(token_i)``, a position embedding ``E_position(i)`` and, when token
    type embeddings are enabled, a segment embedding ``E_segment(A|B)``. The sum
    is normalized by the configured normalization layer and passed through
    dropout. Three model packages share this layer -- ``bert``, ``fnet`` and
    ``distilbert`` -- so its defaults and error messages are a contract across
    all three; see the module docstring.

    **Architecture Overview:**

    .. code-block:: text

        input_ids     position_ids         token_type_ids
        (batch, L)    (batch, L)           (batch, L), optional
             │                 │                     │
             ▼                 ▼                     ▼
        ┌─────────┐   ┌─────────────────┐  ┌──────────────────┐
        │Embedding│   │ 'learned':      │  │ Embedding        │
        │ (V, D)  │   │   Embedding     │  │ (type_vocab, D)  │
        │         │   │   (max_pos, D)  │  │                  │
        │         │   │ 'sinusoidal':   │  │ built only when  │
        │         │   │   fixed sin/cos │  │ token types are  │
        │         │   │   table, no     │  │ enabled          │
        │         │   │   weight        │  │                  │
        └────┬────┘   └────────┬────────┘  └─────────┬────────┘
             │                 │                     │
             └────────┬────────┘                     │
                      ▼                              │
        word_embeds + position_embeds                │
                      │                              │
                      └───────────────┬──────────────┘
                                      ▼
                       element-wise sum (batch, L, D)
                                      │
                                      ▼
              LayerNormalization / RMSNorm / BandRMS / BatchNorm
                                      │
                                      ▼
                                   Dropout
                                      │
                                      ▼
                       output (batch, L, hidden_size)

    :param vocab_size: Size of the vocabulary. Must be positive.
    :type vocab_size: int
    :param hidden_size: Hidden dimension for embeddings. Must be positive.
    :type hidden_size: int
    :param max_position_embeddings: Maximum sequence length for positional
        embeddings. Must be positive.
    :type max_position_embeddings: int
    :param type_vocab_size: Size of the token type vocabulary. Required (and
        must be positive) when ``use_token_type_embeddings`` is ``True``;
        must be ``None`` otherwise. A value supplied while token type
        embeddings are disabled is normalized to ``None`` with a warning, so
        it can never be serialized as an inert config key.
    :type type_vocab_size: Optional[int]
    :param initializer_range: Standard deviation for weight initialization.
        Must be positive.
    :type initializer_range: float
    :param layer_norm_eps: Epsilon value for normalization layers. Must be
        positive.
    :type layer_norm_eps: float
    :param dropout_rate: Dropout probability for embeddings. Must be between
        0 and 1.
    :type dropout_rate: float
    :param normalization_type: Type of normalization layer to use. Supported:
        ``'layer_norm'``, ``'rms_norm'``, ``'band_rms'``, ``'batch_norm'``.
    :type normalization_type: str
    :param use_token_type_embeddings: Whether to build and add the segment
        (token type) embedding term. ``True`` reproduces BERT. Set ``False``
        for DistilBERT-style models, which have no segment embedding: no
        ``token_type_embeddings`` weight is created and passing
        ``token_type_ids`` to ``call()`` raises.
    :type use_token_type_embeddings: bool
    :param position_embedding_type: How positional information is produced.
        ``'learned'`` (BERT, the default) allocates a trainable
        ``(max_position_embeddings, hidden_size)`` embedding table.
        ``'sinusoidal'`` computes a fixed, non-trainable sin/cos table on the
        fly — it allocates no weight and is not bounded by
        ``max_position_embeddings``, but requires an even ``hidden_size``.
        Any other value raises; there is no silent fallback to a default.
    :type position_embedding_type: str
    :param mask_zero: Whether the inner ``word_embeddings`` sub-layer treats
        token id ``0`` as a padding mask. ``True`` reproduces BERT.

        MEASURED CAVEAT — this layer does not PROPAGATE that mask at either
        setting. ``BertEmbeddings.supports_masking`` is ``False`` and it
        defines no ``compute_mask``. The inner ``Embedding``'s mask is
        dropped at the ``word_embeds + position_embeds`` sum, so no
        ``_keras_mask`` reaches a consumer eagerly or in a functional graph.
        The forward output is bit-identical either way, max abs diff ``0.0``.
        The flag is therefore observable only through ``get_config()`` and
        ``word_embeddings.mask_zero``. Set it ``False`` in models that thread
        an explicit ``attention_mask``. That keeps the declared intent correct
        if this layer ever gains mask propagation.
    :type mask_zero: bool
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: From the constructor, on any of five checks.

        - ``vocab_size``, ``hidden_size``, ``max_position_embeddings``,
          ``initializer_range`` or ``layer_norm_eps`` is not positive.
        - ``dropout_rate`` is outside ``[0, 1]``.
        - ``type_vocab_size`` is missing or non-positive while
          ``use_token_type_embeddings`` is ``True``.
        - ``normalization_type`` or ``position_embedding_type`` is not one of
          the accepted values.
        - ``hidden_size`` is odd while ``position_embedding_type`` is
          ``'sinusoidal'``.

    Input shape:
        Integer tensor ``input_ids`` of shape ``(batch_size, seq_length)``.
        ``token_type_ids`` and ``position_ids`` are separate call arguments and
        are not part of this shape; see :meth:`call`.

    Output shape:
        3D tensor with shape ``(batch_size, seq_length, hidden_size)``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.embedding.bert_embeddings import (
            BertEmbeddings,
        )

        embed = BertEmbeddings(
            vocab_size=30522,
            hidden_size=768,
            max_position_embeddings=512,
            type_vocab_size=2,
        )
        ids = keras.ops.zeros((2, 128), dtype="int32")
        embed(ids).shape  # (2, 128, 768)
    """

    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            max_position_embeddings: int,
            type_vocab_size: Optional[int] = None,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-8,
            dropout_rate: float = 0.0,
            normalization_type: str = "layer_norm",
            use_token_type_embeddings: bool = True,
            position_embedding_type: str = "learned",
            mask_zero: bool = True,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create every sub-layer.

        All sub-layers are constructed here and built in :meth:`build`. See the
        class docstring for what each parameter means and for the full list of
        conditions that raise.

        :param vocab_size: Size of the vocabulary.
        :type vocab_size: int
        :param hidden_size: Hidden dimension for embeddings.
        :type hidden_size: int
        :param max_position_embeddings: Maximum sequence length covered by the
            learned position table.
        :type max_position_embeddings: int
        :param type_vocab_size: Size of the token type vocabulary, or ``None``
            when token type embeddings are disabled.
        :type type_vocab_size: Optional[int]
        :param initializer_range: Standard deviation of the truncated normal
            initializer used for every embedding table.
        :type initializer_range: float
        :param layer_norm_eps: Epsilon for the normalization layer.
        :type layer_norm_eps: float
        :param dropout_rate: Dropout probability applied to the sum.
        :type dropout_rate: float
        :param normalization_type: Which normalization layer to build.
        :type normalization_type: str
        :param use_token_type_embeddings: Whether to build the segment term.
        :type use_token_type_embeddings: bool
        :param position_embedding_type: ``'learned'`` or ``'sinusoidal'``.
        :type position_embedding_type: str
        :param mask_zero: Passed to the inner word ``Embedding``; see the class
            docstring for what it does and does not do here.
        :type mask_zero: bool
        :param kwargs: Additional keyword arguments for the Layer base class.
        :type kwargs: Any
        :raises ValueError: If any parameter is invalid; see the class
            docstring for the full list.
        """
        super().__init__(**kwargs)

        # Validate parameters
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if max_position_embeddings <= 0:
            raise ValueError(f"max_position_embeddings must be positive, got {max_position_embeddings}")
        if use_token_type_embeddings:
            if type_vocab_size is None or type_vocab_size <= 0:
                raise ValueError(
                    f"type_vocab_size must be positive when use_token_type_embeddings is True, "
                    f"got {type_vocab_size}"
                )
        elif type_vocab_size is not None:
            # Never carry an inert value into get_config(): it would be serialized
            # into every checkpoint as a config key that shapes nothing.
            logger.warning(
                f"type_vocab_size={type_vocab_size} is ignored because "
                f"use_token_type_embeddings is False; storing None instead."
            )
            type_vocab_size = None
        if initializer_range <= 0:
            raise ValueError(f"initializer_range must be positive, got {initializer_range}")
        if layer_norm_eps <= 0:
            raise ValueError(f"layer_norm_eps must be positive, got {layer_norm_eps}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        valid_norm_types = list(VALID_NORMALIZATION_TYPES)
        if normalization_type not in valid_norm_types:
            raise ValueError(f"normalization_type must be one of {valid_norm_types}, got {normalization_type}")

        # DECISION plan-2026-08-10T183739-b007f435/D-006
        # An unrecognized position_embedding_type MUST raise. Do NOT "simplify"
        # this into an `else:` that falls back to 'learned': a silent fallback
        # of exactly that shape is the defect this rule exists to keep out, here
        # and in every caller that forwards a type string. See decisions.md D-006.
        valid_position_types = list(VALID_POSITION_EMBEDDING_TYPES)
        if position_embedding_type not in valid_position_types:
            raise ValueError(
                f"position_embedding_type must be one of {valid_position_types}, "
                f"got {position_embedding_type}"
            )
        if position_embedding_type == 'sinusoidal' and hidden_size % 2 != 0:
            raise ValueError(
                f"hidden_size must be even for sinusoidal position embeddings "
                f"(sin/cos pairs are interleaved), got {hidden_size}"
            )

        # Store parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.dropout_rate = dropout_rate
        self.normalization_type = normalization_type
        self.use_token_type_embeddings = use_token_type_embeddings
        self.position_embedding_type = position_embedding_type
        self.mask_zero = mask_zero

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        #
        # DECISION plan-2026-08-10T183739-b007f435/D-007
        # Statement ORDER is part of the contract: each TruncatedNormal seeds
        # from the process-global RNG at construction, so a reorder changes every
        # weight while moving the output only at the ~7th decimal (measured in
        # plan b007f435, mutation M2) -- below atol=1e-6 and invisible to every
        # structural check. Do NOT reorder; guard in place. See decisions.md D-007.
        self.word_embeddings = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=initializer_range
            ),
            mask_zero=mask_zero,
            name="word_embeddings"
        )

        if position_embedding_type == 'learned':
            self.position_embeddings = keras.layers.Embedding(
                input_dim=max_position_embeddings,
                output_dim=hidden_size,
                embeddings_initializer=keras.initializers.TruncatedNormal(
                    stddev=initializer_range
                ),
                name="position_embeddings"
            )
        else:
            # 'sinusoidal': a fixed table computed in call(); no weight is allocated.
            self.position_embeddings = None

        if use_token_type_embeddings:
            self.token_type_embeddings = keras.layers.Embedding(
                input_dim=type_vocab_size,
                output_dim=hidden_size,
                embeddings_initializer=keras.initializers.TruncatedNormal(
                    stddev=initializer_range
                ),
                name="token_type_embeddings"
            )
        else:
            self.token_type_embeddings = None

        # Create normalization layer based on type
        self.layer_norm = self._create_normalization_layer("layer_norm")

        self.dropout = keras.layers.Dropout(
            rate=dropout_rate,
            name="dropout"
        )

        logger.info(f"Created BertEmbeddings with hidden_size={hidden_size}, "
                    f"vocab_size={vocab_size}, normalization_type={normalization_type}")

    def _create_normalization_layer(self, name: str) -> keras.layers.Layer:
        """Create a normalization layer based on the configuration type.

        :param name: Name for the normalization layer.
        :type name: str
        :return: Configured normalization layer instance.
        :rtype: keras.layers.Layer
        :raises ValueError: If ``normalization_type`` is not supported.
        """
        if self.normalization_type == 'layer_norm':
            return keras.layers.LayerNormalization(
                epsilon=self.layer_norm_eps,
                name=name
            )
        elif self.normalization_type == 'rms_norm':
            return RMSNorm(
                epsilon=self.layer_norm_eps,
                name=name
            )
        elif self.normalization_type == 'band_rms':
            return BandRMS(
                epsilon=self.layer_norm_eps,
                name=name
            )
        elif self.normalization_type == 'batch_norm':
            return keras.layers.BatchNormalization(
                epsilon=self.layer_norm_eps,
                name=name
            )
        else:
            raise ValueError(
                f"Unknown normalization type: {self.normalization_type}. "
                f"Supported types: layer_norm, rms_norm, band_rms, batch_norm"
            )

    def _sinusoidal_position_embeddings(
            self,
            position_ids: keras.KerasTensor,
            target_dtype: str
    ) -> keras.KerasTensor:
        """Compute the fixed sin/cos positional table for the given positions.

        The table is ``PE(p, 2i) = sin(p / 10000^(2i/d))`` and
        ``PE(p, 2i+1) = cos(p / 10000^(2i/d))``, with the sin and cos halves
        interleaved along the feature axis. No weight is allocated and no
        position bound is enforced, so this branch accepts positions beyond
        ``max_position_embeddings``.

        :param position_ids: Integer positions of shape
            ``(batch_size, seq_length)``.
        :type position_ids: keras.KerasTensor
        :param target_dtype: dtype of the tensor this table will be summed
            with. The table is COMPUTED in the WIDER of ``float32`` and the
            layer's ``variable_dtype`` (so a ``float64`` policy gets float64
            precision, and every narrow policy gets float32), then cast to this
            dtype as the last step.
        :type target_dtype: str
        :return: Positional embeddings of shape
            ``(batch_size, seq_length, hidden_size)`` in ``target_dtype``.
        :rtype: keras.KerasTensor
        """
        # Two separate dtype rules follow. Neither may be collapsed into the
        # other, and neither may be re-keyed off self.compute_dtype.
        #
        # DECISION plan-2026-08-10T183739-b007f435/D-008
        # COMPUTE in the WIDER of float32 and self.variable_dtype. float32 is
        # a FLOOR, but not for the reason it is tempting to give: the ladder
        # 10000^(-2i/d) does NOT underflow float16. Its infimum is 1/10000,
        # above float16's smallest normal 6.103516e-05, and a sweep at
        # hidden_size 128, 768, 1024, 4096 and 65536 on 2026-08-28 found zero
        # underflows at any of them. What float16 cannot hold is the POSITION.
        # At hidden_size 768 over positions 0..511 the float16 table is wrong
        # by about 2.5e-01 against a float64 oracle, because the float16 ulp
        # at position 511 is already 0.25; the float32 table is wrong by
        # about 5e-05. Re-measured 2026-08-28 with a NumPy float64 oracle,
        # which is the instrument these two figures are against: 2.497240e-01
        # and 5.282748e-05. The float16 figure is device-independent; the
        # float32 one is NOT, so quote it to one digit. Running the same
        # arithmetic through `keras.ops` gives 3.731341e-05 on CPU and
        # 4.270122e-05 on GPU. float32 is not a ceiling either: a float64 policy
        # computing in float32 capped accuracy at ~1.5e-06 against ~1e-16
        # expected (measured at c6ab51084). Gate on variable_dtype, NOT
        # compute_dtype -- the narrow one under a mixed policy.
        # See decisions.md D-008.
        #
        # DECISION plan-2026-08-10T183739-b007f435/D-016
        # CAST to target_dtype -- the dtype of the tensor this table is summed
        # with -- NOT self.compute_dtype: under mixed_float16 a Keras sub-layer
        # autocasts its output, so the two disagree and the sum raises
        # `InvalidArgumentError: cannot compute AddV2 as input #1 was expected to
        # be a half tensor but is a float tensor`. See decisions.md D-016.
        variable_dtype = keras.backend.standardize_dtype(self.variable_dtype)
        compute_precision = "float64" if variable_dtype == "float64" else "float32"

        positions = ops.cast(position_ids, compute_precision)
        positions = ops.expand_dims(positions, axis=-1)

        div_term = ops.exp(
            ops.arange(0, self.hidden_size, 2, dtype=compute_precision)
            * -(math.log(10000.0) / self.hidden_size)
        )
        angles = positions * div_term

        interleaved = ops.stack([ops.sin(angles), ops.cos(angles)], axis=-1)
        shape = ops.shape(position_ids)
        table = ops.reshape(interleaved, (shape[0], shape[1], self.hidden_size))

        return ops.cast(table, target_dtype)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the embeddings layer by explicitly building all sub-layers.

        :param input_shape: Shape tuple for input_ids
            ``(batch_size, seq_length)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is invalid.
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2D input shape (batch_size, seq_length), "
                             f"got {len(input_shape)}D: {input_shape}")

        logger.info(f"Building Embeddings with input_shape: {input_shape}")

        # Build each sub-layer here rather than letting the first call do
        # it. A sub-layer built lazily inside `call()` has no weights when
        # `.keras` saving walks the tree, so its kernels reload as fresh
        # values.
        self.word_embeddings.build(input_shape)
        if self.position_embeddings is not None:
            self.position_embeddings.build(input_shape)
        if self.token_type_embeddings is not None:
            self.token_type_embeddings.build(input_shape)

        # Build normalization and dropout with embeddings output shape
        embeddings_output_shape = (*input_shape, self.hidden_size)
        self.layer_norm.build(embeddings_output_shape)
        self.dropout.build(embeddings_output_shape)

        logger.info("Embeddings built successfully")

        super().build(input_shape)

    def call(
            self,
            input_ids: keras.KerasTensor,
            token_type_ids: Optional[keras.KerasTensor] = None,
            position_ids: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Compute composite embeddings from token, position, and segment IDs.

        :param input_ids: Token IDs of shape ``(batch_size, seq_length)``.
        :type input_ids: keras.KerasTensor
        :param token_type_ids: Token type IDs of shape
            ``(batch_size, seq_length)``. If ``None``, defaults to all zeros.
            Must be ``None`` when ``use_token_type_embeddings`` is ``False``.
        :type token_type_ids: Optional[keras.KerasTensor]
        :param position_ids: Position IDs of shape
            ``(batch_size, seq_length)`` or ``(seq_length,)``; the rank-1 form
            is broadcast across the batch. Accepted as a tensor, a NumPy array
            or any nested Python sequence (list/tuple), which is converted
            before the rank is read. If ``None``, defaults to sequential
            positions.
        :type position_ids: Optional[keras.KerasTensor]
        :param training: Whether the layer is in training mode.
        :type training: Optional[bool]
        :return: Embedded and normalized tokens of shape
            ``(batch_size, seq_length, hidden_size)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``token_type_ids`` is supplied while
            ``use_token_type_embeddings`` is ``False``; or if ``position_ids``
            is neither rank 1 nor rank 2.
        """
        input_shape = ops.shape(input_ids)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = ops.arange(seq_length, dtype="int32")
            position_ids = ops.expand_dims(position_ids, axis=0)
            position_ids = ops.broadcast_to(position_ids, (batch_size, seq_length))
        else:
            # DECISION plan-2026-08-10T183739-b007f435/D-021
            # Materialize BEFORE reading the rank. Do NOT reduce this to a bare
            # `len(position_ids.shape)`: a list/tuple/int has no `.shape`, so that
            # form turns every non-array caller into an opaque AttributeError (a
            # regression measured at 7e65bdb43). The `hasattr` gate converts only
            # non-arrays, leaving the symbolic path intact. See decisions.md D-021.
            if not hasattr(position_ids, 'shape'):
                position_ids = ops.convert_to_tensor(position_ids)

            # DECISION plan-2026-08-10T183739-b007f435/D-015
            # Rank normalization happens HERE, once, for BOTH branches. Do NOT
            # delete it or push it into either branch: the same rank-1 input the
            # learned branch silently broadcasts crashes the sinusoidal branch on
            # its rank-2 reshape (`IndexError`, at c6ab51084).
            # See decisions.md D-015.
            position_rank = len(position_ids.shape)
            if position_rank == 1:
                position_ids = ops.broadcast_to(
                    ops.expand_dims(position_ids, axis=0), (batch_size, seq_length)
                )
            elif position_rank != 2:
                raise ValueError(
                    f"position_ids must be rank 1 (seq_length,) or rank 2 "
                    f"(batch_size, seq_length), got rank {position_rank} "
                    f"with shape {tuple(position_ids.shape)}"
                )

        # Apply word and position embeddings
        word_embeds = self.word_embeddings(input_ids)

        if self.position_embedding_type == 'learned':
            position_embeds = self.position_embeddings(position_ids)
        else:
            position_embeds = self._sinusoidal_position_embeddings(
                position_ids,
                keras.backend.standardize_dtype(word_embeds.dtype)
            )

        embeddings = word_embeds + position_embeds

        # Add the segment term only when this layer owns one
        if self.use_token_type_embeddings:
            if token_type_ids is None:
                token_type_ids = ops.zeros_like(input_ids, dtype="int32")
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        elif token_type_ids is not None:
            raise ValueError(
                "token_type_ids was provided but use_token_type_embeddings is False; "
                "this layer has no segment embedding to look them up in."
            )

        # Apply normalization and dropout
        embeddings = self.layer_norm(embeddings, training=training)
        embeddings = self.dropout(embeddings, training=training)

        return embeddings

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape given input shape.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape ``(*input_shape, hidden_size)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return (*input_shape, self.hidden_size)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'max_position_embeddings': self.max_position_embeddings,
            'type_vocab_size': self.type_vocab_size,
            'initializer_range': self.initializer_range,
            'layer_norm_eps': self.layer_norm_eps,
            'dropout_rate': self.dropout_rate,
            'normalization_type': self.normalization_type,
            'use_token_type_embeddings': self.use_token_type_embeddings,
            'position_embedding_type': self.position_embedding_type,
            'mask_zero': self.mask_zero,
        })
        return config

    # NOTE: no custom from_config — the default `cls(**config)` preserves the
    # layer `name` (all config values are primitives). The previous override
    # stripped `name`, losing it on `.keras` round-trip.

# ---------------------------------------------------------------------
