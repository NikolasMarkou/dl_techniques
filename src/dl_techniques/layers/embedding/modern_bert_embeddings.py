import keras
from keras import ops, initializers, layers
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ModernBertEmbeddings(keras.layers.Layer):
    """ModernBERT embedding layer combining word and token type embeddings.

    Computes initial token representations by summing word embeddings and
    segment (token type) embeddings, then applying layer normalization and
    dropout. Unlike classical BERT, this layer omits absolute positional
    embeddings since positional information is handled by Rotary Position
    Embeddings (RoPE) within the attention layers. The combined embedding is
    ``E = E_word(token_i) + E_segment(type_i)``, followed by LayerNorm and
    Dropout.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────┐  ┌──────────────────┐
        │  input_ids   │  │ token_type_ids   │
        │  (batch, L)  │  │ (batch, L)       │
        └──────┬───────┘  └──────┬───────────┘
               ▼                 ▼
        ┌──────────────┐ ┌────────────────────┐
        │ Word Embed   │ │ Token Type Embed   │
        │ (vocab, D)   │ │ (type_vocab, D)    │
        └──────┬───────┘ └──────┬─────────────┘
               └────────┬───────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Element-wise Sum                    │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  LayerNormalization                  │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Dropout                             │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Output (batch, L, hidden_size)      │
        └──────────────────────────────────────┘

    :param vocab_size: Size of the vocabulary.
    :type vocab_size: int
    :param hidden_size: Dimensionality of the embedding vectors.
    :type hidden_size: int
    :param type_vocab_size: Number of segment types (e.g., 2 for sentence
        A/B).
    :type type_vocab_size: int
    :param initializer_range: Standard deviation for the truncated normal
        initializer used for embedding weights.
    :type initializer_range: float
    :param layer_norm_eps: Small epsilon value for numerical stability in
        layer normalization.
    :type layer_norm_eps: float
    :param dropout_rate: Dropout rate applied to the final embeddings.
    :type dropout_rate: float
    :param use_bias: Whether the layer normalization sub-layer should use a
        bias term.
    :type use_bias: bool
    :param kwargs: Additional arguments for the ``keras.layers.Layer`` base
        class.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        type_vocab_size: int,
        initializer_range: float,
        layer_norm_eps: float,
        dropout_rate: float,
        use_bias: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Store all configuration
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias

        # CREATE all sub-layers in __init__ (they remain unbuilt)
        self.word_embeddings = layers.Embedding(
            self.vocab_size,
            self.hidden_size,
            embeddings_initializer=initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="word_embeddings",
        )
        self.token_type_embeddings = layers.Embedding(
            self.type_vocab_size,
            self.hidden_size,
            embeddings_initializer=initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="token_type_embeddings",
        )
        self.layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, center=self.use_bias, name="layer_norm"
        )
        self.dropout = layers.Dropout(self.dropout_rate, name="dropout")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create weights for the embedding, norm, and dropout sub-layers.

        :param input_shape: Shape of the input tensor
            ``(batch_size, sequence_length)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Build sub-layers explicitly in computational order for robust serialization
        self.word_embeddings.build(input_shape)
        self.token_type_embeddings.build(input_shape)

        # The output shape of the embedding summation is needed to build subsequent layers
        embedding_shape = tuple(input_shape) + (self.hidden_size,)
        self.layer_norm.build(embedding_shape)
        self.dropout.build(embedding_shape)

        super().build(input_shape)

    def call(
        self,
        input_ids: keras.KerasTensor,
        token_type_ids: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Compute the final embedding vectors from token and segment IDs.

        :param input_ids: Tensor of token indices of shape
            ``(batch_size, seq_length)``.
        :type input_ids: keras.KerasTensor
        :param token_type_ids: Optional tensor of segment indices of shape
            ``(batch_size, seq_length)``. Defaults to zeros if ``None``.
        :type token_type_ids: Optional[keras.KerasTensor]
        :param training: Whether in training mode for dropout.
        :type training: Optional[bool]
        :return: Combined, normalized, and regularized embedding tensor of
            shape ``(batch_size, seq_length, hidden_size)``.
        :rtype: keras.KerasTensor
        """
        seq_length = ops.shape(input_ids)[1]
        # Default token_type_ids to zeros if not provided
        if token_type_ids is None:
            token_type_ids = ops.zeros(
                (ops.shape(input_ids)[0], seq_length), dtype="int32"
            )

        word_embeds = self.word_embeddings(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        embeddings = word_embeds + token_type_embeds

        embeddings = self.layer_norm(embeddings, training=training)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape: ``(batch_size, sequence_length, hidden_size)``.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape with ``hidden_size`` appended.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape + (self.hidden_size,)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer's configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "type_vocab_size": self.type_vocab_size,
                "initializer_range": self.initializer_range,
                "layer_norm_eps": self.layer_norm_eps,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
            }
        )
        return config


# ---------------------------------------------------------------------
