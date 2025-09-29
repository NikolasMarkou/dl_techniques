import keras
from keras import ops, initializers, layers
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ModernBertEmbeddings(keras.layers.Layer):
    """
    Computes embeddings for ModernBERT from token and type IDs.

    This layer handles the initial embedding lookup for input tokens and segment
    (token type) IDs. It combines these two embeddings and then applies layer
    normalization and dropout. Notably, it does not include absolute position
    embeddings, as positional information is handled by Rotary Position
    Embeddings (RoPE) within the attention layers.

    **Intent**:
    To provide the initial, fixed-dimensional vector representations for input
    tokens that serve as the input to the main transformer encoder stack,
    following modern Keras patterns for robust serialization.

    **Architecture**:
    ```
    Input (input_ids, token_type_ids)
           ↓
    Word Embeddings + Token Type Embeddings
           ↓
    Layer Normalization
           ↓
    Dropout
           ↓
    Output [batch, seq_len, hidden_size]
    ```

    Args:
        vocab_size: Integer, the size of the vocabulary.
        hidden_size: Integer, the dimensionality of the embedding vectors.
        type_vocab_size: Integer, the number of segment types (e.g., 2 for
            sentence A/B).
        initializer_range: Float, standard deviation for the truncated normal
            initializer used for embedding weights.
        layer_norm_eps: Float, a small epsilon value for numerical stability in
            the layer normalization.
        hidden_dropout_prob: Float, dropout rate applied to the final embeddings.
        use_bias: Boolean, whether the layer normalization sub-layer should use
            a bias term.
        **kwargs: Additional arguments for the `keras.layers.Layer` base class.

    Input shape:
        - `input_ids`: 2D tensor of shape `(batch_size, sequence_length)`.
        - `token_type_ids`: (Optional) 2D tensor of shape
          `(batch_size, sequence_length)`.

    Output shape:
        A 3D tensor of shape `(batch_size, sequence_length, hidden_size)`.

    Attributes:
        word_embeddings: `layers.Embedding` for token IDs.
        token_type_embeddings: `layers.Embedding` for segment type IDs.
        layer_norm: `layers.LayerNormalization` applied after embedding summation.
        dropout: `layers.Dropout` applied as the final step.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        type_vocab_size: int,
        initializer_range: float,
        layer_norm_eps: float,
        hidden_dropout_prob: float,
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
        self.hidden_dropout_prob = hidden_dropout_prob
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
        self.dropout = layers.Dropout(self.hidden_dropout_prob, name="dropout")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Creates the weights for the embedding, norm, and dropout layers."""
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
        """
        Computes the final embedding vectors.

        Args:
            input_ids: Tensor of token indices.
            token_type_ids: (Optional) Tensor of segment indices.
            training: (Optional) Boolean indicating training mode for dropout.

        Returns:
            The combined, normalized, and regularized embedding tensor.
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

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer's configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "type_vocab_size": self.type_vocab_size,
                "initializer_range": self.initializer_range,
                "layer_norm_eps": self.layer_norm_eps,
                "hidden_dropout_prob": self.hidden_dropout_prob,
                "use_bias": self.use_bias,
            }
        )
        return config


# ---------------------------------------------------------------------
