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

import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from ..norms.rms_norm import RMSNorm
from ..norms.band_rms import BandRMS

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BertEmbeddings(keras.layers.Layer):
    """
    BERT embeddings layer combining word, position, and token type embeddings.

    This layer implements the embedding component of BERT, which combines:
    - Word embeddings: Map token IDs to dense vector representations
    - Position embeddings: Add positional information to sequence tokens
    - Token type embeddings: Distinguish between different sentence segments

    The embeddings are summed together, normalized, and passed through dropout
    for regularization.

    Args:
        vocab_size: Size of the vocabulary. Must be positive.
        hidden_size: Hidden dimension for embeddings. Must be positive.
        max_position_embeddings: Maximum sequence length for positional embeddings.
            Must be positive.
        type_vocab_size: Size of the token type vocabulary. Must be positive.
        initializer_range: Standard deviation for weight initialization. Must be positive.
        layer_norm_eps: Epsilon value for normalization layers. Must be positive.
        hidden_dropout_prob: Dropout probability for embeddings. Must be between 0 and 1.
        normalization_type: Type of normalization layer to use.
            Supported: 'layer_norm', 'rms_norm', 'band_rms', 'batch_norm'.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        - input_ids: (batch_size, sequence_length)
        - token_type_ids: (batch_size, sequence_length) - optional
        - position_ids: (batch_size, sequence_length) - optional

    Output shape:
        (batch_size, sequence_length, hidden_size)

    Raises:
        ValueError: If any parameter is invalid or out of expected range.

    Example:
        ```python
        embeddings = BertEmbeddings(
            vocab_size=30522,
            hidden_size=768,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            normalization_type='layer_norm'
        )

        input_ids = keras.ops.array([[101, 2023, 2003, 102]])  # [CLS] this is [SEP]
        embedded = embeddings(input_ids)
        ```
    """

    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            max_position_embeddings: int,
            type_vocab_size: int,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-8,
            hidden_dropout_prob: float = 0.0,
            normalization_type: str = "layer_norm",
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
        if type_vocab_size <= 0:
            raise ValueError(f"type_vocab_size must be positive, got {type_vocab_size}")
        if initializer_range <= 0:
            raise ValueError(f"initializer_range must be positive, got {initializer_range}")
        if layer_norm_eps <= 0:
            raise ValueError(f"layer_norm_eps must be positive, got {layer_norm_eps}")
        if not (0.0 <= hidden_dropout_prob <= 1.0):
            raise ValueError(f"hidden_dropout_prob must be between 0 and 1, got {hidden_dropout_prob}")

        valid_norm_types = ['layer_norm', 'rms_norm', 'band_rms', 'batch_norm']
        if normalization_type not in valid_norm_types:
            raise ValueError(f"normalization_type must be one of {valid_norm_types}, got {normalization_type}")

        # Store parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.normalization_type = normalization_type

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        self.word_embeddings = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=initializer_range
            ),
            mask_zero=True,
            name="word_embeddings"
        )

        self.position_embeddings = keras.layers.Embedding(
            input_dim=max_position_embeddings,
            output_dim=hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=initializer_range
            ),
            name="position_embeddings"
        )

        self.token_type_embeddings = keras.layers.Embedding(
            input_dim=type_vocab_size,
            output_dim=hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=initializer_range
            ),
            name="token_type_embeddings"
        )

        # Create normalization layer based on type
        self.layer_norm = self._create_normalization_layer("layer_norm")

        self.dropout = keras.layers.Dropout(
            rate=hidden_dropout_prob,
            name="dropout"
        )

        logger.info(f"Created BertEmbeddings with hidden_size={hidden_size}, "
                    f"vocab_size={vocab_size}, normalization_type={normalization_type}")

    def _create_normalization_layer(self, name: str) -> keras.layers.Layer:
        """
        Create a normalization layer based on the configuration type.

        Args:
            name: Name for the normalization layer.

        Returns:
            Configured normalization layer instance.

        Raises:
            ValueError: If normalization_type is not supported.
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

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the embeddings layer by explicitly building all sub-layers.

        This follows the modern Keras 3 pattern where the parent layer
        explicitly builds all child layers for robust serialization.

        Args:
            input_shape: Shape tuple for input_ids (batch_size, seq_length).

        Raises:
            ValueError: If input_shape is invalid.
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
        self.position_embeddings.build(input_shape)
        self.token_type_embeddings.build(input_shape)

        # Build normalization and dropout with embeddings output shape
        embeddings_output_shape = (*input_shape, self.hidden_size)
        self.layer_norm.build(embeddings_output_shape)
        self.dropout.build(embeddings_output_shape)

        super().build(input_shape)
        logger.info("Embeddings built successfully")

    def call(
            self,
            input_ids: keras.KerasTensor,
            token_type_ids: Optional[keras.KerasTensor] = None,
            position_ids: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply embeddings to input tokens.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_length).
            token_type_ids: Token type IDs of shape (batch_size, seq_length).
                If None, defaults to all zeros.
            position_ids: Position IDs of shape (batch_size, seq_length).
                If None, defaults to sequential positions.
            training: Whether the layer is in training mode.

        Returns:
            Embedded and normalized tokens of shape
            (batch_size, seq_length, hidden_size).
        """
        input_shape = ops.shape(input_ids)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = ops.arange(seq_length, dtype="int32")
            position_ids = ops.expand_dims(position_ids, axis=0)
            position_ids = ops.broadcast_to(position_ids, (batch_size, seq_length))

        # Create token type IDs if not provided
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids, dtype="int32")

        # Apply all embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        # Sum all embeddings
        embeddings = word_embeds + position_embeds + token_type_embeds

        # Apply normalization and dropout
        embeddings = self.layer_norm(embeddings, training=training)
        embeddings = self.dropout(embeddings, training=training)

        return embeddings

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape given input shape."""
        return (*input_shape, self.hidden_size)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'max_position_embeddings': self.max_position_embeddings,
            'type_vocab_size': self.type_vocab_size,
            'initializer_range': self.initializer_range,
            'layer_norm_eps': self.layer_norm_eps,
            'hidden_dropout_prob': self.hidden_dropout_prob,
            'normalization_type': self.normalization_type,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BertEmbeddings':
        """Create layer from configuration."""
        return cls(**{k: v for k, v in config.items() if k != 'name'})

# ---------------------------------------------------------------------
