"""
Construct the composite input embeddings for BERT-style models.

This layer builds the initial vector representation for each token in an
input sequence by combining three distinct sources of information. This
composite structure is essential for enabling a non-recurrent,
attention-based model like BERT to understand the nuances of language,
including token identity, sequence order, and sentence relationships.

Architecture:
    The architecture is based on the principle that a token's meaning is a
    function of its identity, its position, and the sentence it belongs to.
    To capture this, the layer generates three separate embedding vectors
    which are then summed element-wise:

    1.  **Token Embeddings:** This is the standard word embedding lookup,
        mapping each token ID from the vocabulary to a high-dimensional
        vector. It provides the foundational, context-independent meaning
        of the token.

    2.  **Positional Embeddings:** Since the Transformer architecture is
        inherently permutation-invariant (it has no built-in sense of
        sequence order), positional information must be explicitly injected.
        Unlike the fixed sinusoidal embeddings used in the original
        Transformer, BERT utilizes *learnable* positional embeddings. A
        unique vector is learned for each absolute position in the
        sequence (up to a maximum length), allowing the model to flexibly
        learn the optimal way to represent token order for its pre-training
        tasks.

    3.  **Segment (Token Type) Embeddings:** This component is specifically
        designed to support BERT's pre-training objective of Next Sentence
        Prediction (NSP). When two sentences (A and B) are concatenated to
        form a single input sequence, this embedding provides a simple,
        learnable signal that allows the model to distinguish between tokens
        belonging to sentence A and those belonging to sentence B.

Foundational Mathematics:
    The final embedding for a token at position `i` in the input sequence is
    the element-wise sum of the three constituent embeddings:

        E_final(token_i) = E_word(token_i) + E_position(i) + E_segment(A or B)

    This summation projects the three distinct information sources into a
    single, unified vector space. The subsequent Transformer layers are then
    trained to process these rich, composite representations.

    Following the summation, two final steps are applied:
    -   **Layer Normalization:** The combined embedding vector is normalized.
        This stabilizes the learning process by ensuring that the inputs to
        the first Transformer layer have a consistent distribution, which is
        crucial for training deep networks.
    -   **Dropout:** A standard dropout layer is applied for regularization,
        preventing the model from becoming overly reliant on any single
        feature in the combined embedding.

References:
    - The embedding strategy is a core component of the BERT model,
      introduced in:
      Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). "BERT:
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
# Accepted enum values -- the SINGLE source of truth.
#
# DECISION plan-2026-08-10T183739-b007f435/D-009
# These two tuples are imported by layers/embedding/factory.py's
# validate_embedding_config. Do NOT inline a literal list back into either the
# constructor's checks or the factory's: before this step the normalization list
# existed as two hand-maintained copies (factory.py and this file), which is a
# lockstep invariant, i.e. a defect waiting for one side to be edited alone.
# The factory validation is defence-in-depth in front of the constructor's own
# raise, so the two MUST agree by construction, not by discipline.
# See decisions.md D-009.
# ---------------------------------------------------------------------

VALID_NORMALIZATION_TYPES: Tuple[str, ...] = (
    'layer_norm', 'rms_norm', 'band_rms', 'batch_norm'
)
VALID_POSITION_EMBEDDING_TYPES: Tuple[str, ...] = ('learned', 'sinusoidal')

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BertEmbeddings(keras.layers.Layer):
    """BERT embedding layer combining word, position, and token type embeddings.

    Constructs composite token representations by summing three learnable
    embedding lookups: word embeddings ``E_word(token_i)`` mapping token IDs to
    dense vectors, positional embeddings ``E_position(i)`` encoding absolute
    sequence position, and segment embeddings ``E_segment(A|B)`` distinguishing
    sentence membership. The combined embedding
    ``E = E_word + E_position + E_segment`` is then layer-normalized and passed
    through dropout for regularization.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
        │  input_ids   │  │ position_ids │  │ token_type_ids   │
        │  (batch, L)  │  │ (batch, L)   │  │ (batch, L)       │
        └──────┬───────┘  └──────┬───────┘  └──────┬───────────┘
               ▼                 ▼                  ▼
        ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐
        │ Word Embed   │ │ Pos Embed    │ │ Token Type Embed   │
        │ (vocab, D)   │ │ (max_pos, D) │ │ (type_vocab, D)    │
        └──────┬───────┘ └──────┬───────┘ └──────┬─────────────┘
               └────────┬───────┴────────┬───────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Element-wise Sum                    │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  LayerNorm / RMSNorm / BandRMS       │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Dropout                             │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Output (batch, L, hidden_size)      │
        └──────────────────────────────────────┘

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
        Any other value raises; there is deliberately no silent fallback.
    :type position_embedding_type: str
    :param mask_zero: Whether the inner ``word_embeddings`` sub-layer treats
        token id ``0`` as a padding mask. ``True`` reproduces BERT.

        MEASURED CAVEAT — this layer does not PROPAGATE that mask at either
        setting: ``BertEmbeddings.supports_masking`` is ``False``, it defines
        no ``compute_mask``, and the inner ``Embedding``'s mask is dropped at
        the ``word_embeds + position_embeds`` sum, so no ``_keras_mask``
        reaches a consumer eagerly or in a functional graph, and the forward
        output is bit-identical (max abs diff ``0.0``) either way. The flag is
        therefore observable only through ``get_config()`` and
        ``word_embeddings.mask_zero``. Set it ``False`` in models that thread
        an explicit ``attention_mask``, so the declared intent stays correct
        if this layer ever gains mask propagation.
    :type mask_zero: bool
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If any parameter is invalid or out of expected range.
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
        # An unrecognized position_embedding_type MUST raise. Do NOT "simplify" this
        # into an `else:` that falls back to 'learned' -- a silent normalization
        # fallback of exactly that shape is the defect this plan exists to delete
        # (models/distilbert/model.py:170-174, measured in findings/
        # step1-premise-rederivation.md (c)). See decisions.md D-006.
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
        # The ORDER of these constructor statements is load-bearing: each
        # TruncatedNormal instance draws its seed from the process-global RNG at
        # construction, so reordering them changes every initialized weight. A
        # measured probe (findings/step2-i1-reference.md, mutation M2) showed a
        # swap of position_embeddings <-> token_type_embeddings moves the forward
        # output only at the ~7th decimal -- invisible to an atol=1e-6 comparison
        # and to every structural check (paths, shapes and param total all still
        # matched). Do NOT reorder, and do NOT hoist the new branches above an
        # existing construction: guard in place. See decisions.md D-007.
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
        # DECISION plan-2026-08-10T183739-b007f435/D-008
        # DECISION plan-2026-08-10T183739-b007f435/D-016
        # Two separate dtype rules, neither of which may be collapsed into the
        # other or re-keyed off self.compute_dtype:
        #  (1) COMPUTE in the WIDER of float32 and self.variable_dtype. float32 is
        #      the FLOOR because 10000^(-2i/d) underflows fp16 across the feature
        #      axis; it is not the ceiling -- a float64 policy computing in float32
        #      capped the table's accuracy at ~1.5e-06 where a float64 user expects
        #      ~1e-16 (measured at c6ab51084). Gate on variable_dtype, NOT
        #      compute_dtype: under a mixed policy compute_dtype is the NARROW one
        #      (float16/bfloat16) while variables stay float32, so compute_dtype
        #      would silently re-introduce the underflow this floor exists to stop.
        #  (2) CAST to target_dtype -- the dtype of the tensor this table is
        #      actually summed with, NOT self.compute_dtype. Under mixed_float16 a
        #      Keras sub-layer autocasts its output, so compute_dtype and the real
        #      tensor dtype can disagree; summing a float32 table with a float16
        #      word-embedding raises `InvalidArgumentError: cannot compute AddV2 as
        #      input #1 was expected to be a half tensor but is a float tensor`
        #      (measured at HEAD in findings/step1-premise-rederivation.md (b)).
        # See decisions.md D-008 and D-016.
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

        # CRITICAL: Explicitly build all sub-layers for robust serialization
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
            # `len(position_ids.shape)`: a Python list/tuple/int has no `.shape`,
            # so that form turns every non-array caller into an opaque
            # `AttributeError: 'list' object has no attribute 'shape'` raised from
            # inside call() -- the exact failure shape D-015 exists to remove, and
            # a REGRESSION measured at 7e65bdb43 against c6ab51084 (where a
            # `[[0,1,2,3],[0,1,2,3]]` was accepted by the sinusoidal branch).
            # The `hasattr` gate is deliberate: convert_to_tensor is applied only
            # to objects that are not already arrays/tensors, so the symbolic
            # KerasTensor path through a functional graph is left untouched.
            # See decisions.md D-021.
            if not hasattr(position_ids, 'shape'):
                position_ids = ops.convert_to_tensor(position_ids)

            # DECISION plan-2026-08-10T183739-b007f435/D-015
            # Rank normalization happens HERE, once, for BOTH branches. Do NOT
            # delete it and do NOT push it down into either branch: the two
            # branches consume position_ids differently, so without this the SAME
            # rank-1 input that the learned branch silently broadcasts
            # (keras.layers.Embedding returns (seq, hidden), broadcast in the sum)
            # crashes the sinusoidal branch with an opaque
            # `IndexError: tuple index out of range` from the
            # (shape[0], shape[1], hidden_size) reshape, which assumes rank 2.
            # Measured at c6ab51084. See decisions.md D-015.
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
