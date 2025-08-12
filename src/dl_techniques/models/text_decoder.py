import keras
from typing import Optional, Dict, Any, Tuple, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from .bert import BertEmbeddings, BertConfig
from ..layers.transformer import TransformerLayer


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TextDecoder(keras.layers.Layer):
    """
    Text decoder using BERT-style embeddings with causal attention for generation.

    This layer implements a transformer-based text decoder that uses BERT's
    embedding approach (word + position + token type embeddings) but with
    causal attention for autoregressive text generation. It's designed for
    vision-language models that need to generate text conditioned on visual input.

    The decoder consists of:
    - BERT-style embeddings (word, position, token type)
    - Stack of causal transformer decoder layers
    - Layer normalization and dropout for regularization

    Args:
        vocab_size: Size of the vocabulary for text embeddings.
        hidden_dim: Hidden dimension of the transformer layers.
        num_layers: Number of transformer decoder layers.
        num_heads: Number of attention heads in each transformer layer.
        mlp_dim: Dimension of the feed-forward network.
        max_position_embeddings: Maximum sequence length for positional embeddings.
        type_vocab_size: Size of the token type vocabulary.
        dropout: Dropout rate applied throughout the model.
        activation: Activation function for transformer layers.
        normalization_type: Type of normalization layer to use.
        normalization_position: Position of normalization ('pre' or 'post').
        attention_type: Type of attention mechanism.
        ffn_type: Type of feed-forward network architecture.
        use_bias: Whether to use bias in linear layers.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias vectors.
        kernel_regularizer: Regularizer for kernel weights.
        bias_regularizer: Regularizer for bias vectors.
        initializer_range: Standard deviation for weight initialization.
        layer_norm_eps: Epsilon value for normalization layers.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        2D tensor of shape (batch_size, sequence_length) containing token IDs

    Output shape:
        3D tensor of shape (batch_size, sequence_length, hidden_dim)

    Example:
        >>> decoder = TextDecoder(
        ...     vocab_size=32000,
        ...     hidden_dim=768,
        ...     num_layers=12,
        ...     num_heads=12,
        ...     mlp_dim=3072
        ... )
        >>>
        >>> text_tokens = keras.ops.random.uniform((2, 50), 0, 32000, dtype='int32')
        >>> features = decoder(text_tokens)  # Shape: (2, 50, 768)
    """

    def __init__(
            self,
            vocab_size: int = 32000,
            hidden_dim: int = 768,
            num_layers: int = 12,
            num_heads: int = 12,
            mlp_dim: int = 3072,
            max_position_embeddings: int = 512,
            type_vocab_size: int = 2,
            dropout: float = 0.1,
            activation: str = 'gelu',
            normalization_type: str = 'layer_norm',
            normalization_position: str = 'post',
            attention_type: str = 'multi_head_attention',
            ffn_type: str = 'mlp',
            use_bias: bool = True,
            kernel_initializer: str = 'glorot_uniform',
            bias_initializer: str = 'zeros',
            kernel_regularizer: Optional[str] = None,
            bias_regularizer: Optional[str] = None,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-12,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.dropout = dropout
        self.activation = activation
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        # Validate configuration
        self._validate_config()

        # FIX: Instantiate sublayers in __init__ for proper serialization
        bert_config = BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_dim,
            num_heads=self.num_heads,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            hidden_dropout_prob=self.dropout,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            normalization_type=self.normalization_type
        )
        self.embeddings = BertEmbeddings(
            config=bert_config,
            name='embeddings'
        )

        self.decoder_layers = [
            TransformerLayer(
                hidden_size=self.hidden_dim,
                num_heads=self.num_heads,
                intermediate_size=self.mlp_dim,
                normalization_type=self.normalization_type,
                normalization_position=self.normalization_position,
                attention_type=self.attention_type,
                ffn_type=self.ffn_type,
                dropout_rate=self.dropout,
                attention_dropout_rate=self.dropout,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                bias_initializer=self.bias_initializer,
                kernel_regularizer=keras.regularizers.get(self.kernel_regularizer),
                bias_regularizer=keras.regularizers.get(self.bias_regularizer),
                name=f'decoder_layer_{i}'
            ) for i in range(self.num_layers)
        ]

        self.final_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            name='final_norm'
        )

        # Store build input shape for serialization
        self._build_input_shape = None

        logger.info(f"Created TextDecoder with {num_layers} layers, "
                    f"hidden_dim={hidden_dim}, vocab_size={vocab_size}")

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the text decoder by creating embeddings and transformer layers.

        Args:
            input_shape: Shape tuple of input token IDs (batch_size, seq_length)

        Raises:
            ValueError: If input shape is invalid
        """
        if self.built:
            return

        self._build_input_shape = input_shape

        if len(input_shape) != 2:
            raise ValueError(
                f"Expected 2D input shape (batch_size, seq_length), "
                f"got {input_shape}"
            )

        logger.info(f"Building TextDecoder with input_shape: {input_shape}")
        super().build(input_shape)
        logger.info("TextDecoder built successfully")

    def call(
            self,
            inputs: keras.KerasTensor,
            token_type_ids: Optional[keras.KerasTensor] = None,
            position_ids: Optional[keras.KerasTensor] = None,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the text decoder.

        Args:
            inputs: Input token IDs of shape [batch_size, seq_length]
            token_type_ids: Token type IDs for segment embeddings (optional)
            position_ids: Position IDs for custom positional encoding (optional)
            attention_mask: Attention mask to avoid attention on padding tokens (optional)
            training: Whether in training mode

        Returns:
            Hidden states of shape [batch_size, seq_length, hidden_dim]
        """
        embeddings = self.embeddings(
            input_ids=inputs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            training=training
        )

        hidden_states = embeddings
        for decoder_layer in self.decoder_layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                training=training
            )

        hidden_states = self.final_norm(hidden_states, training=training)

        return hidden_states

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape given input shape.

        Args:
            input_shape: Input shape tuple

        Returns:
            Output shape tuple [batch_size, seq_length, hidden_dim]
        """
        return (*input_shape, self.hidden_dim)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing layer configuration
        """
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'mlp_dim': self.mlp_dim,
            'max_position_embeddings': self.max_position_embeddings,
            'type_vocab_size': self.type_vocab_size,
            'dropout': self.dropout,
            'activation': self.activation,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'attention_type': self.attention_type,
            'ffn_type': self.ffn_type,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'initializer_range': self.initializer_range,
            'layer_norm_eps': self.layer_norm_eps,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """
        Get build configuration for serialization.

        Returns:
            Dictionary containing build configuration
        """
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Build layer from configuration.

        Args:
            config: Build configuration dictionary
        """
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TextDecoder':
        """
        Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            TextDecoder instance
        """
        return cls(**config)


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_text_decoder_base(**kwargs) -> TextDecoder:
    """
    Create base text decoder configuration.

    Args:
        **kwargs: Additional arguments to override base configuration

    Returns:
        TextDecoder with base configuration
    """
    config = {
        'vocab_size': 32000,
        'hidden_dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'mlp_dim': 3072,
        'dropout': 0.1,
    }
    config.update(kwargs)

    logger.info("Creating base text decoder")
    return TextDecoder(**config)


def create_text_decoder_small(**kwargs) -> TextDecoder:
    """
    Create small text decoder configuration.

    Args:
        **kwargs: Additional arguments to override small configuration

    Returns:
        TextDecoder with small configuration
    """
    config = {
        'vocab_size': 32000,
        'hidden_dim': 384,
        'num_layers': 6,
        'num_heads': 6,
        'mlp_dim': 1536,
        'dropout': 0.1,
    }
    config.update(kwargs)

    logger.info("Creating small text decoder")
    return TextDecoder(**config)


def create_text_decoder_large(**kwargs) -> TextDecoder:
    """
    Create large text decoder configuration.

    Args:
        **kwargs: Additional arguments to override large configuration

    Returns:
        TextDecoder with large configuration
    """
    config = {
        'vocab_size': 32000,
        'hidden_dim': 1024,
        'num_layers': 24,
        'num_heads': 16,
        'mlp_dim': 4096,
        'dropout': 0.1,
    }
    config.update(kwargs)

    logger.info("Creating large text decoder")
    return TextDecoder(**config)